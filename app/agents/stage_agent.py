"""
Stage Agent: classify NIH stage with optional tool-backed support.
"""
from pathlib import Path

from app.agents.base import BaseAgent
from app.core.llm import llm_client
from app.core.stage_model import (
    CLASSIFICATION_STAGES,
    COMPLETED_FEATURES_AT_STAGE,
    STAGE_CLARIFY_CONFIDENCE,
    STAGE_COMMIT_CONFIDENCE,
    STAGE_MODEL,
    find_stage_in_text,
    is_definition_query,
    normalize_stage,
)
from app.core.types import AgentOutput, SessionState


class StageAgent(BaseAgent):
    """Stage classification agent."""

    # Stage I is scored as its two sub-stages: IA (generation and refinement)
    # then IB (feasibility and pilot testing).
    STAGES = CLASSIFICATION_STAGES
    MISSING_INFO_HINTS = [
        "Is the intervention already manualized?",
        "What are the sample size and study design (e.g., pilot vs RCT)?",
        "Do you already have mechanism-testing outcomes?",
        "Do you already have efficacy or effectiveness outcomes?",
        "Is the study in controlled settings or real-world implementation settings?",
    ]

    def __init__(self):
        super().__init__("StageAgent")
        self._prompt_file = (
            Path(__file__).resolve().parents[1] / "prompts" / "stage.md"
        )
        self._fallback_system_prompt = (
            "You are an NIH Stage Model classifier. Output JSON only.\n"
            "Fields: stage(0/I/II/III/IV/V/null), confidence(0~1), feature_updates(object), "
            "reasoning_summary(string), missing_info(list[string]), clarifying_question(string|null)."
        )

    def _get_system_prompt(self) -> str:
        """Load stage system prompt from app/prompts/stage.md with safe fallback."""
        try:
            if self._prompt_file.exists():
                text = self._prompt_file.read_text(encoding="utf-8").strip()
                if text:
                    return text
        except Exception:
            pass
        return self._fallback_system_prompt

    def _normalize_missing_info(self, data: dict, confidence: float) -> list[str]:
        """Support both missing_info and miss_info from prompt outputs."""
        raw = data.get("missing_info")
        if raw is None:
            raw = data.get("miss_info", [])
        if not isinstance(raw, list):
            raw = []
        items = [str(x).strip() for x in raw if str(x).strip()]
        if confidence < STAGE_CLARIFY_CONFIDENCE and not items:
            items = self.MISSING_INFO_HINTS[:]
        return items[:5]

    @staticmethod
    def _build_clarifying_question(missing_info: list[str]) -> str | None:
        if not missing_info:
            return None
        key_items = "; ".join(missing_info[:3])
        return (
            "To improve stage confidence, please provide the following key details: "
            f"{key_items}."
        )

    def run(self, state: SessionState, user_message: str, context: str = "") -> AgentOutput:
        if is_definition_query(user_message):
            return AgentOutput(
                decision={
                    "stage": None,
                    "feature_updates": {},
                    "reasoning_summary": "Definition query detected; stage classification skipped.",
                    "missing_info": [],
                    "clarifying_question": None,
                },
                confidence=0.95,
                analysis="Skipped stage classification for definition query",
                actions=[],
            )

        llm_output = self._run_with_llm(user_message, context)
        if llm_output:
            return llm_output
        return self._run_with_rules(user_message)

    def _run_with_llm(self, user_message: str, context: str = "") -> AgentOutput | None:
        if not llm_client.is_enabled():
            return None

        system_prompt = self._get_system_prompt()
        user_prompt = (
            f"User message: {user_message}\n"
            f"Context: {context[:1200]}\n"
            "Classify most likely stage. If insufficient info, stage=null with lower confidence."
        )
        data = llm_client.chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
        if not data:
            return None

        # normalize_stage accepts "II", "2", "Stage 2", "IA", "1b". The old
        # `str(stage).upper() in STAGES` test dropped every one of those to
        # None, including the sub-stages stage.md asks the model to reason
        # about.
        stage = normalize_stage(data.get("stage"))

        confidence = float(data.get("confidence", data.get("stage_confidence", 0.5)))
        confidence = max(0.0, min(1.0, confidence))
        feature_updates = data.get("feature_updates", {}) or {}
        if not isinstance(feature_updates, dict):
            feature_updates = {}

        reasoning_summary = str(data.get("reasoning_summary", "")).strip()
        missing_info = self._normalize_missing_info(data, confidence)

        clarifying_question = data.get("clarifying_question")
        if clarifying_question is not None:
            clarifying_question = str(clarifying_question).strip() or None
        if confidence < STAGE_CLARIFY_CONFIDENCE and not clarifying_question:
            clarifying_question = self._build_clarifying_question(missing_info)

        # No tool call on low confidence. This used to emit a db_tool lookup
        # for a stage definition, which (a) the orchestrator never executed —
        # _stage_reason drops actions and _rag_plan overwrites the pending
        # list — and (b) would only have returned the definitions that are
        # already in this agent's own system prompt and in
        # app/core/stage_model.py. Low confidence is reported through
        # missing_info and the clarifying question instead.
        return AgentOutput(
            decision={
                "stage": stage,
                "feature_updates": feature_updates,
                "reasoning_summary": reasoning_summary,
                "missing_info": missing_info,
                "clarifying_question": clarifying_question,
            },
            confidence=confidence,
            analysis=f"LLM stage={stage}, confidence={confidence:.2f}",
            actions=[],
        )

    def _run_with_rules(self, user_message: str) -> AgentOutput:
        message_lower = user_message.lower()

        stage = None
        confidence = 0.0
        feature_updates = {}
        matched_signals = []

        explicit = find_stage_in_text(user_message)
        if explicit:
            stage = explicit
            confidence = 0.9
            matched_signals.append(f"Explicit stage mention: {explicit}")

        if not stage:
            stage_scores = {s: 0 for s in self.STAGES}

            for stage_key in self.STAGES:
                keywords = STAGE_MODEL[stage_key].keywords
                hits = [kw for kw in keywords if kw in message_lower]
                stage_scores[stage_key] = len(hits)
                if hits:
                    matched_signals.append(f"Stage {stage_key} signals: {', '.join(hits[:3])}")

            sorted_scores = sorted(stage_scores.items(), key=lambda x: x[1], reverse=True)
            best_stage, best_score = sorted_scores[0]
            second_score = sorted_scores[1][1]

            if best_score > 0:
                margin = best_score - second_score
                confidence = min(0.85, 0.55 + 0.1 * best_score + 0.05 * margin)
                stage = best_stage if confidence >= STAGE_COMMIT_CONFIDENCE else None

        if stage and confidence >= STAGE_COMMIT_CONFIDENCE:
            feature_updates.update(COMPLETED_FEATURES_AT_STAGE.get(stage, {}))

        reasoning_summary = (
            f"Matched signals: {'; '.join(matched_signals[:4])}. Current confidence={confidence:.2f}."
            if matched_signals
            else f"Insufficient stage signals. Current confidence={confidence:.2f}."
        )

        clarifying_question = None
        missing_info = self.MISSING_INFO_HINTS[:] if confidence < STAGE_CLARIFY_CONFIDENCE else []
        if confidence < STAGE_CLARIFY_CONFIDENCE:
            clarifying_question = self._build_clarifying_question(missing_info)

        return AgentOutput(
            decision={
                "stage": stage,
                "feature_updates": feature_updates,
                "reasoning_summary": reasoning_summary,
                "missing_info": missing_info,
                "miss_info": missing_info,
                "clarifying_question": clarifying_question,
            },
            confidence=confidence,
            analysis=f"Rule stage={stage}, confidence={confidence:.2f}",
            actions=[],
        )

    def update_state(self, state: SessionState, output: AgentOutput):
        if (
            "stage" in output.decision
            and output.decision["stage"]
            and output.confidence >= STAGE_COMMIT_CONFIDENCE
        ):
            state.slots.stage = output.decision["stage"]
            state.slots.stage_confidence = output.confidence

        if "feature_updates" in output.decision:
            for key, value in output.decision["feature_updates"].items():
                if hasattr(state.slots, key):
                    setattr(state.slots, key, value)
                else:
                    state.slots.extracted_features[key] = value

        for key in ["reasoning_summary", "missing_info", "clarifying_question"]:
            if key in output.decision and output.decision[key] is not None:
                state.slots.extracted_features[key] = output.decision[key]