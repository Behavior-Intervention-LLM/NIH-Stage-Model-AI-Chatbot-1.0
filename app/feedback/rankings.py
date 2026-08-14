"""
Aggregations over observed turns -- the four rankings.

Everything is computed in Python from a single annotated read of the turn
history rather than in SQL, because the quality number is a fusion of two
columns plus a confidence weighting (see scoring.py) and expressing that in
portable SQLite/Postgres SQL would be worse than the extra pass.

Volumes here are chat turns, not events -- a few thousand rows -- so a full
scan per request is fine. If that stops being true, materialise
`quality`/`evidence` onto feedback_turns at write time and move these to SQL.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from app.feedback import scoring, store


def _mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _load(limit: int = 5000) -> List[Dict[str, Any]]:
    return scoring.annotate(store.scored_turns(limit=limit))


def _bucket(
    turns: List[Dict[str, Any]], key_fn: Callable[[Dict[str, Any]], Optional[str]]
) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "turns": 0,
            "qualities": [],
            "latencies": [],
            "rephrased": 0,
            "corrected": 0,
            "retrieval_empty": 0,
            "tool_errors": 0,
        }
    )
    for turn in turns:
        key = key_fn(turn)
        if not key:
            continue
        b = buckets[key]
        b["turns"] += 1
        if float(turn.get("evidence") or 0.0) > 0.0:
            b["qualities"].append(float(turn.get("quality") or 0.0))
        b["latencies"].append(float(turn.get("latency_ms") or 0))
        b["rephrased"] += 1 if turn.get("rephrased") else 0
        b["corrected"] += 1 if turn.get("corrected") else 0
        b["retrieval_empty"] += 1 if int(turn.get("retrieval_count") or 0) == 0 else 0
        b["tool_errors"] += int(turn.get("tool_errors") or 0)
    return buckets


def _summarise(name_field: str, buckets: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for name, b in buckets.items():
        rows.append(
            {
                name_field: name,
                "turns": b["turns"],
                "mean_quality": _mean(b["qualities"]),
                "scored_turns": len(b["qualities"]),
                "mean_latency_ms": int(_mean(b["latencies"])),
                "rephrase_rate": round(b["rephrased"] / b["turns"], 3) if b["turns"] else 0.0,
                "correction_rate": round(b["corrected"] / b["turns"], 3) if b["turns"] else 0.0,
                "empty_retrieval_rate": round(b["retrieval_empty"] / b["turns"], 3) if b["turns"] else 0.0,
                "tool_errors": b["tool_errors"],
            }
        )
    rows.sort(key=lambda r: (-r["turns"], r[name_field]))
    return rows


def overview(limit: int = 5000) -> Dict[str, Any]:
    """Headline health numbers, plus how much of the data is actually scored."""
    turns = _load(limit)
    scored = [t for t in turns if float(t.get("evidence") or 0.0) > 0.0]
    judged = [t for t in turns if t.get("judge_overall") is not None]
    behavioural = [t for t in turns if float(t.get("behavioral_confidence") or 0.0) > 0.0]

    qualities = [float(t.get("quality") or 0.0) for t in scored]
    return {
        "total_turns": len(turns),
        "total_sessions": len({t["session_id"] for t in turns}),
        "total_users": len({t["username"] for t in turns}),
        "scored_turns": len(scored),
        "judge_coverage": round(len(judged) / len(turns), 3) if turns else 0.0,
        "behavioural_coverage": round(len(behavioural) / len(turns), 3) if turns else 0.0,
        "mean_quality": _mean(qualities),
        "good_turns": sum(1 for q in qualities if q >= scoring.GOOD_THRESHOLD),
        "bad_turns": sum(1 for q in qualities if q <= scoring.BAD_THRESHOLD),
        "mean_latency_ms": int(_mean([float(t.get("latency_ms") or 0) for t in turns])),
        "mean_groundedness": _mean(
            [float(t["groundedness"]) for t in turns if t.get("groundedness") is not None]
        ),
        "mean_user_need_met": _mean(
            [float(t["user_need_met"]) for t in turns if t.get("user_need_met") is not None]
        ),
    }


def feature_ranking(limit: int = 5000) -> Dict[str, List[Dict[str, Any]]]:
    """Ranking of use: which workflows and intents get exercised, and how well."""
    turns = _load(limit)
    return {
        "workflows": _summarise("workflow", _bucket(turns, lambda t: t.get("workflow") or "unknown")),
        "intents": _summarise("intent_label", _bucket(turns, lambda t: t.get("intent_label") or "unknown")),
        "query_types": _summarise("query_type", _bucket(turns, lambda t: t.get("query_type") or "unknown")),
        "stages": _summarise(
            "stage", _bucket(turns, lambda t: f"Stage {t['stage']}" if t.get("stage") else None)
        ),
    }


def user_ranking(limit: int = 5000) -> List[Dict[str, Any]]:
    """Ranking of user usage: volume, breadth, and how well each user is served."""
    turns = _load(limit)
    per_user: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "turns": 0,
            "sessions": set(),
            "qualities": [],
            "workflows": set(),
            "first_seen": None,
            "last_seen": None,
            "corrected": 0,
        }
    )

    for turn in turns:
        u = per_user[turn["username"]]
        u["turns"] += 1
        u["sessions"].add(turn["session_id"])
        if turn.get("workflow"):
            u["workflows"].add(turn["workflow"])
        if float(turn.get("evidence") or 0.0) > 0.0:
            u["qualities"].append(float(turn.get("quality") or 0.0))
        u["corrected"] += 1 if turn.get("corrected") else 0
        created = turn.get("created_at")
        if u["first_seen"] is None or str(created) < str(u["first_seen"]):
            u["first_seen"] = created
        if u["last_seen"] is None or str(created) > str(u["last_seen"]):
            u["last_seen"] = created

    rows = []
    for username, u in per_user.items():
        sessions = len(u["sessions"])
        rows.append(
            {
                "username": username,
                "turns": u["turns"],
                "sessions": sessions,
                "turns_per_session": round(u["turns"] / sessions, 2) if sessions else 0.0,
                "workflows_used": sorted(u["workflows"]),
                "mean_quality": _mean(u["qualities"]),
                "correction_rate": round(u["corrected"] / u["turns"], 3) if u["turns"] else 0.0,
                "first_seen": u["first_seen"],
                "last_seen": u["last_seen"],
            }
        )
    rows.sort(key=lambda r: (-r["turns"], r["username"]))
    return rows


def response_ranking(
    limit: int = 5000, top_n: int = 20, order: str = "worst"
) -> List[Dict[str, Any]]:
    """Ranking of API responses: the individual turns that went best or worst.

    `order="worst"` is the one worth reading -- it is the queue of answers to
    go fix, with the judge's reason attached.
    """
    turns = [t for t in _load(limit) if float(t.get("evidence") or 0.0) > 0.0]
    reverse = order != "worst"
    turns.sort(key=lambda t: (float(t.get("quality") or 0.0), t.get("created_at") or ""), reverse=reverse)

    rows = []
    for turn in turns[:top_n]:
        rows.append(
            {
                "turn_uid": turn["turn_uid"],
                "created_at": turn["created_at"],
                "username": turn["username"],
                "session_id": turn["session_id"],
                "question": (turn.get("user_message") or "")[:280],
                "reply_preview": (turn.get("reply") or "")[:280],
                "quality": round(float(turn.get("quality") or 0.0), 3),
                "evidence": round(float(turn.get("evidence") or 0.0), 3),
                "behavioral_score": turn.get("behavioral_score"),
                "judge_overall": turn.get("judge_overall"),
                "groundedness": turn.get("groundedness"),
                "user_need_met": turn.get("user_need_met"),
                "inferred_user_need": turn.get("inferred_user_need"),
                "rationale": turn.get("rationale"),
                "rephrased": bool(turn.get("rephrased")),
                "corrected": bool(turn.get("corrected")),
                "workflow": turn.get("workflow"),
                "stage": turn.get("stage"),
                "sources": [s.get("source") for s in (turn.get("sources") or [])],
                "latency_ms": turn.get("latency_ms"),
            }
        )
    return rows


def source_ranking() -> List[Dict[str, Any]]:
    """Ranking of documents by their learned contribution, with current weight."""
    return store.source_rows()


def knowledge_gaps(limit: int = 50) -> List[Dict[str, Any]]:
    """What users keep asking for that the corpus answers badly."""
    return store.gap_rows(limit=limit)


def inferred_user_needs(limit: int = 5000, top_n: int = 30) -> List[Dict[str, Any]]:
    """What the judge thinks users are actually trying to do, most common first.

    This is the system's own model of user intent, built without asking anyone.
    """
    turns = _load(limit)
    clusters: Dict[str, Dict[str, Any]] = {}
    from app.feedback.signals import topic_key

    for turn in turns:
        need = (turn.get("inferred_user_need") or "").strip()
        if not need:
            continue
        key, terms = topic_key(need, max_terms=4)
        if not key:
            continue
        bucket = clusters.setdefault(
            key, {"need": need, "count": 0, "qualities": [], "terms": terms}
        )
        bucket["count"] += 1
        if float(turn.get("evidence") or 0.0) > 0.0:
            bucket["qualities"].append(float(turn.get("quality") or 0.0))

    rows = [
        {
            "need": b["need"],
            "occurrences": b["count"],
            "mean_quality": _mean(b["qualities"]),
            "terms": b["terms"],
        }
        for b in clusters.values()
    ]
    rows.sort(key=lambda r: (-r["occurrences"], r["mean_quality"]))
    return rows[:top_n]
