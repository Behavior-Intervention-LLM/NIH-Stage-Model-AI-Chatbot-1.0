"""
Storage layer for the implicit feedback system.

Lives in the same database as user accounts and chat history (see auth.py):
SQLite locally, hosted Postgres when DATABASE_URL is set.

Tables
------
feedback_turns       One row per assistant turn -- what was asked, what was
                     retrieved, which agents ran, how long it took.
feedback_signals     Behavioural signals derived from what the user did *next*.
                     This is the replacement for thumbs up/down: nobody is
                     asked to rate anything, the rating is inferred.
feedback_judgements  LLM-as-judge scores for a turn, graded against the
                     passages that same turn actually retrieved.
feedback_sources     Learned per-document weights, fed back into retrieval.
feedback_gaps        Recurring questions the system keeps answering badly.

Every table is keyed by a TEXT primary key, so no table needs the
SERIAL/AUTOINCREMENT split that auth.py and chat_history.py carry.

Nothing in this module may break the chat path; callers go through
app.feedback.safe().
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from auth import _IS_POSTGRES, _db, _normalize_username, _sql

# Bounds on stored text. Retrieved passages are kept because the judge grades
# groundedness against them, but they must not turn this table into a second
# copy of the corpus.
MAX_MESSAGE_CHARS = 4000
MAX_REPLY_CHARS = 8000
MAX_PASSAGE_CHARS = 600
MAX_STORED_SOURCES = 6

_initialised = False


def utcnow_text() -> str:
    """Microsecond precision so ORDER BY created_at is stable within a second."""
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    """Create the feedback tables if absent. Safe to call repeatedly."""
    global _initialised
    if _initialised:
        return

    turns_ddl = """
        CREATE TABLE IF NOT EXISTS feedback_turns (
            turn_uid TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            username TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            user_message TEXT NOT NULL,
            reply TEXT NOT NULL,
            workflow TEXT,
            intent_label TEXT,
            query_type TEXT,
            need_stage INTEGER NOT NULL DEFAULT 0,
            stage TEXT,
            stage_confidence REAL NOT NULL DEFAULT 0,
            intent_confidence REAL NOT NULL DEFAULT 0,
            agents_called TEXT NOT NULL DEFAULT '[]',
            sources TEXT NOT NULL DEFAULT '[]',
            retrieval_count INTEGER NOT NULL DEFAULT 0,
            tool_errors INTEGER NOT NULL DEFAULT 0,
            latency_ms INTEGER NOT NULL DEFAULT 0,
            reply_chars INTEGER NOT NULL DEFAULT 0
        )
    """
    signals_ddl = """
        CREATE TABLE IF NOT EXISTS feedback_signals (
            turn_uid TEXT PRIMARY KEY,
            behavioral_score REAL NOT NULL DEFAULT 0,
            confidence REAL NOT NULL DEFAULT 0,
            rephrased INTEGER NOT NULL DEFAULT 0,
            corrected INTEGER NOT NULL DEFAULT 0,
            accepted INTEGER NOT NULL DEFAULT 0,
            followed_up INTEGER NOT NULL DEFAULT 0,
            abandoned INTEGER NOT NULL DEFAULT 0,
            seconds_to_next REAL,
            detail TEXT NOT NULL DEFAULT '{}',
            computed_at TEXT NOT NULL
        )
    """
    judgements_ddl = """
        CREATE TABLE IF NOT EXISTS feedback_judgements (
            turn_uid TEXT PRIMARY KEY,
            groundedness REAL,
            relevance REAL,
            stage_consistency REAL,
            user_need_met REAL,
            overall REAL NOT NULL DEFAULT 0,
            inferred_user_need TEXT,
            unsupported_claims TEXT NOT NULL DEFAULT '[]',
            rationale TEXT,
            model TEXT,
            created_at TEXT NOT NULL
        )
    """
    sources_ddl = """
        CREATE TABLE IF NOT EXISTS feedback_sources (
            source TEXT PRIMARY KEY,
            weight REAL NOT NULL DEFAULT 1.0,
            observations INTEGER NOT NULL DEFAULT 0,
            good_turns INTEGER NOT NULL DEFAULT 0,
            bad_turns INTEGER NOT NULL DEFAULT 0,
            mean_credit REAL NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
    """
    gaps_ddl = """
        CREATE TABLE IF NOT EXISTS feedback_gaps (
            topic_key TEXT PRIMARY KEY,
            terms TEXT NOT NULL DEFAULT '[]',
            occurrences INTEGER NOT NULL DEFAULT 0,
            mean_quality REAL NOT NULL DEFAULT 0,
            example_query TEXT,
            status TEXT NOT NULL DEFAULT 'open',
            updated_at TEXT NOT NULL
        )
    """

    with _db() as conn:
        for ddl in (turns_ddl, signals_ddl, judgements_ddl, sources_ddl, gaps_ddl):
            conn.execute(ddl)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_turns_session "
            "ON feedback_turns(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_turns_username "
            "ON feedback_turns(username)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_turns_created "
            "ON feedback_turns(created_at)"
        )
    _initialised = True


# ==================== writes ====================


def record_turn(
    *,
    session_id: str,
    username: str,
    user_message: str,
    reply: str,
    workflow: Optional[str] = None,
    intent_label: Optional[str] = None,
    query_type: Optional[str] = None,
    need_stage: bool = False,
    stage: Optional[str] = None,
    stage_confidence: float = 0.0,
    intent_confidence: float = 0.0,
    agents_called: Optional[List[str]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    tool_errors: int = 0,
    latency_ms: int = 0,
) -> str:
    """Append one observed turn and return its turn_uid."""
    init_db()
    turn_uid = uuid.uuid4().hex
    username = _normalize_username(username or "anonymous")
    trimmed_sources = [
        {
            "source": str(s.get("source", "")),
            "score": float(s.get("score") or 0.0),
            "passage": str(s.get("passage", ""))[:MAX_PASSAGE_CHARS],
        }
        for s in (sources or [])[:MAX_STORED_SOURCES]
    ]

    with _db() as conn:
        row = conn.execute(
            _sql("SELECT COUNT(*) AS n FROM feedback_turns WHERE session_id = ?"),
            (session_id,),
        ).fetchone()
        turn_index = int(row["n"]) if row else 0
        conn.execute(
            _sql(
                "INSERT INTO feedback_turns ("
                "  turn_uid, session_id, username, turn_index, created_at,"
                "  user_message, reply, workflow, intent_label, query_type,"
                "  need_stage, stage, stage_confidence, intent_confidence,"
                "  agents_called, sources, retrieval_count, tool_errors,"
                "  latency_ms, reply_chars"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                turn_uid,
                session_id,
                username,
                turn_index,
                utcnow_text(),
                (user_message or "")[:MAX_MESSAGE_CHARS],
                (reply or "")[:MAX_REPLY_CHARS],
                workflow,
                intent_label,
                query_type,
                1 if need_stage else 0,
                stage,
                float(stage_confidence or 0.0),
                float(intent_confidence or 0.0),
                json.dumps(agents_called or []),
                json.dumps(trimmed_sources),
                len(trimmed_sources),
                int(tool_errors or 0),
                int(latency_ms or 0),
                len(reply or ""),
            ),
        )
    return turn_uid


def save_signals(turn_uid: str, signals: Dict[str, Any]) -> None:
    """Upsert derived behavioural signals for a turn."""
    init_db()
    with _db() as conn:
        conn.execute(
            _sql(
                "INSERT INTO feedback_signals ("
                "  turn_uid, behavioral_score, confidence, rephrased, corrected,"
                "  accepted, followed_up, abandoned, seconds_to_next, detail, computed_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (turn_uid) DO UPDATE SET "
                "  behavioral_score = excluded.behavioral_score,"
                "  confidence = excluded.confidence,"
                "  rephrased = excluded.rephrased,"
                "  corrected = excluded.corrected,"
                "  accepted = excluded.accepted,"
                "  followed_up = excluded.followed_up,"
                "  abandoned = excluded.abandoned,"
                "  seconds_to_next = excluded.seconds_to_next,"
                "  detail = excluded.detail,"
                "  computed_at = excluded.computed_at"
            ),
            (
                turn_uid,
                float(signals.get("behavioral_score", 0.0)),
                float(signals.get("confidence", 0.0)),
                1 if signals.get("rephrased") else 0,
                1 if signals.get("corrected") else 0,
                1 if signals.get("accepted") else 0,
                1 if signals.get("followed_up") else 0,
                1 if signals.get("abandoned") else 0,
                signals.get("seconds_to_next"),
                json.dumps(signals.get("detail", {})),
                utcnow_text(),
            ),
        )


def save_judgement(turn_uid: str, judgement: Dict[str, Any], model: str) -> None:
    """Upsert an LLM-as-judge evaluation for a turn."""
    init_db()
    with _db() as conn:
        conn.execute(
            _sql(
                "INSERT INTO feedback_judgements ("
                "  turn_uid, groundedness, relevance, stage_consistency, user_need_met,"
                "  overall, inferred_user_need, unsupported_claims, rationale, model, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (turn_uid) DO UPDATE SET "
                "  groundedness = excluded.groundedness,"
                "  relevance = excluded.relevance,"
                "  stage_consistency = excluded.stage_consistency,"
                "  user_need_met = excluded.user_need_met,"
                "  overall = excluded.overall,"
                "  inferred_user_need = excluded.inferred_user_need,"
                "  unsupported_claims = excluded.unsupported_claims,"
                "  rationale = excluded.rationale,"
                "  model = excluded.model,"
                "  created_at = excluded.created_at"
            ),
            (
                turn_uid,
                judgement.get("groundedness"),
                judgement.get("relevance"),
                judgement.get("stage_consistency"),
                judgement.get("user_need_met"),
                float(judgement.get("overall") or 0.0),
                (judgement.get("inferred_user_need") or "")[:500],
                json.dumps(judgement.get("unsupported_claims") or []),
                (judgement.get("rationale") or "")[:1000],
                model,
                utcnow_text(),
            ),
        )


def upsert_source_weight(
    source: str,
    *,
    weight: float,
    observations: int,
    good_turns: int,
    bad_turns: int,
    mean_credit: float,
) -> None:
    init_db()
    with _db() as conn:
        conn.execute(
            _sql(
                "INSERT INTO feedback_sources ("
                "  source, weight, observations, good_turns, bad_turns, mean_credit, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (source) DO UPDATE SET "
                "  weight = excluded.weight,"
                "  observations = excluded.observations,"
                "  good_turns = excluded.good_turns,"
                "  bad_turns = excluded.bad_turns,"
                "  mean_credit = excluded.mean_credit,"
                "  updated_at = excluded.updated_at"
            ),
            (source, float(weight), int(observations), int(good_turns),
             int(bad_turns), float(mean_credit), utcnow_text()),
        )


def upsert_gap(
    topic_key: str,
    *,
    terms: List[str],
    occurrences: int,
    mean_quality: float,
    example_query: str,
) -> None:
    init_db()
    with _db() as conn:
        conn.execute(
            _sql(
                "INSERT INTO feedback_gaps ("
                "  topic_key, terms, occurrences, mean_quality, example_query, status, updated_at"
                ") VALUES (?, ?, ?, ?, ?, 'open', ?) "
                "ON CONFLICT (topic_key) DO UPDATE SET "
                "  terms = excluded.terms,"
                "  occurrences = excluded.occurrences,"
                "  mean_quality = excluded.mean_quality,"
                "  example_query = excluded.example_query,"
                "  updated_at = excluded.updated_at"
            ),
            (topic_key, json.dumps(terms), int(occurrences), float(mean_quality),
             (example_query or "")[:300], utcnow_text()),
        )


# ==================== reads ====================


def get_turn(turn_uid: str) -> Optional[Dict[str, Any]]:
    init_db()
    with _db() as conn:
        row = conn.execute(
            _sql("SELECT * FROM feedback_turns WHERE turn_uid = ?"), (turn_uid,)
        ).fetchone()
    return _decode_turn(row) if row else None


def session_turns(session_id: str) -> List[Dict[str, Any]]:
    """All turns of a session in chronological order."""
    init_db()
    with _db() as conn:
        rows = conn.execute(
            _sql(
                "SELECT * FROM feedback_turns WHERE session_id = ? "
                "ORDER BY turn_index, created_at"
            ),
            (session_id,),
        ).fetchall()
    return [_decode_turn(r) for r in rows]


def turns_awaiting_judgement(limit: int = 25) -> List[Dict[str, Any]]:
    init_db()
    with _db() as conn:
        rows = conn.execute(
            _sql(
                "SELECT t.* FROM feedback_turns t "
                "LEFT JOIN feedback_judgements j ON j.turn_uid = t.turn_uid "
                "WHERE j.turn_uid IS NULL ORDER BY t.created_at DESC LIMIT ?"
            ),
            (int(limit),),
        ).fetchall()
    return [_decode_turn(r) for r in rows]


def scored_turns(limit: int = 5000) -> List[Dict[str, Any]]:
    """Turns joined with whichever signals and judgements exist for them.

    LEFT JOINs, so a turn with no evidence at all still comes back -- the
    scoring layer decides how much weight it carries.
    """
    init_db()
    with _db() as conn:
        rows = conn.execute(
            _sql(
                "SELECT t.*, "
                "  s.behavioral_score, s.confidence AS behavioral_confidence,"
                "  s.rephrased, s.corrected, s.accepted, s.abandoned,"
                "  j.overall AS judge_overall, j.groundedness, j.relevance,"
                "  j.user_need_met, j.inferred_user_need, j.rationale "
                "FROM feedback_turns t "
                "LEFT JOIN feedback_signals s ON s.turn_uid = t.turn_uid "
                "LEFT JOIN feedback_judgements j ON j.turn_uid = t.turn_uid "
                "ORDER BY t.created_at DESC LIMIT ?"
            ),
            (int(limit),),
        ).fetchall()
    return [_decode_turn(r) for r in rows]


def all_source_weights() -> Dict[str, float]:
    init_db()
    with _db() as conn:
        rows = conn.execute("SELECT source, weight FROM feedback_sources").fetchall()
    return {r["source"]: float(r["weight"]) for r in rows}


def source_rows() -> List[Dict[str, Any]]:
    init_db()
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM feedback_sources ORDER BY weight DESC, observations DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def gap_rows(limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    with _db() as conn:
        rows = conn.execute(
            _sql(
                "SELECT * FROM feedback_gaps WHERE status = 'open' "
                "ORDER BY occurrences DESC, mean_quality ASC LIMIT ?"
            ),
            (int(limit),),
        ).fetchall()
    out = []
    for r in rows:
        row = dict(r)
        row["terms"] = _loads(row.get("terms"), [])
        out.append(row)
    return out


def turn_count() -> int:
    init_db()
    with _db() as conn:
        row = conn.execute("SELECT COUNT(*) AS n FROM feedback_turns").fetchone()
    return int(row["n"]) if row else 0


def purge_user(username: str) -> None:
    """Delete every feedback row belonging to a user (GDPR-style erasure)."""
    init_db()
    username = _normalize_username(username)
    with _db() as conn:
        conn.execute(
            _sql(
                "DELETE FROM feedback_signals WHERE turn_uid IN "
                "(SELECT turn_uid FROM feedback_turns WHERE username = ?)"
            ),
            (username,),
        )
        conn.execute(
            _sql(
                "DELETE FROM feedback_judgements WHERE turn_uid IN "
                "(SELECT turn_uid FROM feedback_turns WHERE username = ?)"
            ),
            (username,),
        )
        conn.execute(
            _sql("DELETE FROM feedback_turns WHERE username = ?"), (username,)
        )


# ==================== helpers ====================


def _loads(raw: Any, default: Any) -> Any:
    if raw in (None, ""):
        return default
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return default


def _decode_turn(row: Any) -> Dict[str, Any]:
    turn = dict(row)
    turn["agents_called"] = _loads(turn.get("agents_called"), [])
    turn["sources"] = _loads(turn.get("sources"), [])
    turn["need_stage"] = bool(turn.get("need_stage"))
    if "unsupported_claims" in turn:
        turn["unsupported_claims"] = _loads(turn.get("unsupported_claims"), [])
    return turn


__all__ = [
    "init_db",
    "record_turn",
    "save_signals",
    "save_judgement",
    "upsert_source_weight",
    "upsert_gap",
    "get_turn",
    "session_turns",
    "turns_awaiting_judgement",
    "scored_turns",
    "all_source_weights",
    "source_rows",
    "gap_rows",
    "turn_count",
    "purge_user",
    "utcnow_text",
]
