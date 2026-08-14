"""
LLM-as-judge evaluation of completed turns.

Behavioural signals (see signals.py) are honest but sparse: the last turn of
every session has no follow-up, and plenty of users neither complain nor thank
anyone. The judge fills that gap by grading every turn against the passages
that turn actually retrieved -- so it is checking the answer against its own
evidence, not against the model's memory.

It runs *after* the response has been returned, on a daemon thread, so it adds
nothing to user-visible latency. It uses the cheap routing model by default.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from app.core.llm import llm_client
from app.feedback import store
from app.logging_config import logger

_PROMPT_FILE = Path(__file__).resolve().parents[1] / "prompts" / "judge.md"

_FALLBACK_SYSTEM = (
    "You are a strict evaluator of an NIH Stage Model research assistant. "
    "Grade one exchange against the retrieved passages shown. Return JSON only "
    "with keys: groundedness, relevance, stage_consistency, user_need_met, "
    "inferred_user_need, unsupported_claims, overall, rationale. All numeric "
    "fields are 0-1; stage_consistency may be null when no stage was claimed. "
    "Be calibrated, not generous."
)

# One judge call must never cost more than the answer it is grading.
MAX_EVIDENCE_CHARS = 700
MAX_ANSWER_CHARS = 4000


def _load_system_prompt() -> str:
    """Read the `## system` section of app/prompts/judge.md."""
    try:
        text = _PROMPT_FILE.read_text(encoding="utf-8")
    except Exception:
        return _FALLBACK_SYSTEM

    lines = text.splitlines()
    buf: List[str] = []
    capturing = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## ") and not stripped.startswith("###"):
            capturing = stripped[3:].strip().lower() == "system"
            continue
        if capturing:
            buf.append(line)
    body = "\n".join(buf).strip()
    return body or _FALLBACK_SYSTEM


class FeedbackJudge:
    """Grades one turn and persists the result."""

    def __init__(self):
        self._system_prompt: Optional[str] = None

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = _load_system_prompt()
        return self._system_prompt

    @property
    def model(self) -> str:
        """Prefer an explicit judge model, else the cheap routing model."""
        return (
            getattr(settings, "FEEDBACK_JUDGE_MODEL", None)
            or getattr(settings, "LLM_INTENT_MODEL", None)
            or settings.LLM_MODEL
        )

    @staticmethod
    def _build_user_prompt(turn: Dict[str, Any]) -> str:
        parts: List[str] = []
        parts.append("## User question\n" + (turn.get("user_message") or "").strip())
        parts.append(
            "## Answer given\n" + (turn.get("reply") or "").strip()[:MAX_ANSWER_CHARS]
        )

        sources = turn.get("sources") or []
        if sources:
            evidence = ["## Retrieved passages the answer had available"]
            for i, src in enumerate(sources, 1):
                passage = str(src.get("passage") or "").replace("\n", " ")
                evidence.append(
                    f"{i}. [{src.get('source', 'unknown')}] {passage[:MAX_EVIDENCE_CHARS]}"
                )
            parts.append("\n".join(evidence))
        else:
            parts.append(
                "## Retrieved passages the answer had available\n"
                "(none -- retrieval returned nothing for this turn)"
            )

        claimed_stage = turn.get("stage")
        if claimed_stage:
            parts.append(
                f"## Stage claimed by the system\nStage {claimed_stage} "
                f"(internal confidence {float(turn.get('stage_confidence') or 0.0):.2f})"
            )
        else:
            parts.append("## Stage claimed by the system\n(none)")

        parts.append("Return the JSON object now.")
        return "\n\n".join(parts)

    @staticmethod
    def _clamp01(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    def _normalise(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        grounded = self._clamp01(raw.get("groundedness"))
        relevance = self._clamp01(raw.get("relevance"))
        stage_consistency = self._clamp01(raw.get("stage_consistency"))
        need_met = self._clamp01(raw.get("user_need_met"))
        overall = self._clamp01(raw.get("overall"))

        if overall is None:
            # The model skipped its holistic verdict: weight need-met and
            # groundedness the way the prompt asks it to.
            weighted = [
                (need_met, 0.45),
                (grounded, 0.30),
                (relevance, 0.15),
                (stage_consistency, 0.10),
            ]
            available = [(v, w) for v, w in weighted if v is not None]
            total_weight = sum(w for _, w in available)
            overall = (
                sum(v * w for v, w in available) / total_weight if total_weight else 0.0
            )

        claims = raw.get("unsupported_claims")
        if not isinstance(claims, list):
            claims = []

        return {
            "groundedness": grounded,
            "relevance": relevance,
            "stage_consistency": stage_consistency,
            "user_need_met": need_met,
            "overall": round(float(overall), 4),
            "inferred_user_need": str(raw.get("inferred_user_need") or "").strip(),
            "unsupported_claims": [str(c)[:200] for c in claims[:8]],
            "rationale": str(raw.get("rationale") or "").strip(),
        }

    def evaluate(self, turn: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Grade a turn. Returns the normalised judgement, or None on failure."""
        if not llm_client.is_enabled():
            return None
        raw = llm_client.chat_json(
            system_prompt=self.system_prompt,
            user_prompt=self._build_user_prompt(turn),
            model=self.model,
        )
        if not isinstance(raw, dict):
            return None
        return self._normalise(raw)

    def judge_turn(self, turn_uid: str) -> Optional[Dict[str, Any]]:
        """Grade a stored turn and persist the verdict."""
        turn = store.get_turn(turn_uid)
        if turn is None:
            return None
        judgement = self.evaluate(turn)
        if judgement is None:
            return None
        store.save_judgement(turn_uid, judgement, model=self.model)
        return judgement

    def judge_pending(self, limit: int = 25) -> int:
        """Grade turns that have no judgement yet. Returns how many succeeded."""
        done = 0
        for turn in store.turns_awaiting_judgement(limit=limit):
            try:
                if self.judge_turn(turn["turn_uid"]):
                    done += 1
            except Exception:
                logger.warning(
                    "Feedback judge failed for turn %s", turn.get("turn_uid"), exc_info=True
                )
        return done


judge = FeedbackJudge()


def judge_in_background(turn_uid: str) -> None:
    """Fire-and-forget grading so the chat path never waits on the judge."""
    if not getattr(settings, "FEEDBACK_JUDGE_ENABLED", True):
        return

    def _run():
        try:
            judge.judge_turn(turn_uid)
        except Exception:
            logger.warning("Background judge failed for turn %s", turn_uid, exc_info=True)

    threading.Thread(target=_run, daemon=True, name=f"judge-{turn_uid[:8]}").start()
