"""
Stage Agent: classify NIH stage with optional tool-backed support.
"""
import re
from pathlib import Path

from app.agents.base import BaseAgent
from app.core.llm import llm_client
from app.core.types import AgentOutput, SessionState, ToolCall


class StageAgent(BaseAgent):
    """Stage classification agent."""

    STAGES = ["0", "I", "II", "III", "IV", "V"]
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
        if confidence < 0.75 and not items:
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
        message_lower = user_message.lower()
        is_stage_definition_query = (
            any(kw in message_lower for kw in ["what is", "what's", "define", "how many stages", "number of stages", "list stages", "explain"])
            and any(kw in message_lower for kw in ["nih stage model", "nih stage", "stage model"])
        )
        if is_stage_definition_query:
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

        stage = data.get("stage")
        if stage is not None:
            stage = str(stage).upper()
            if stage not in self.STAGES:
                stage = None

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
        if confidence < 0.75 and not clarifying_question:
            clarifying_question = self._build_clarifying_question(missing_info)

        tool_calls = []
        if confidence < 0.58:
            tool_calls.append(
                ToolCall(
                    tool_name="db_tool",
                    tool_args={"query": f"Classify research stage: {user_message}"},
                    success_criteria="Return matching stage definition",
                )
            )

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
            actions=tool_calls,
        )

    def _run_with_rules(self, user_message: str) -> AgentOutput:
        message_lower = user_message.lower()

        stage = None
        confidence = 0.0
        feature_updates = {}
        matched_signals = []

        explicit_match = re.search(r"stage\s*(0|i{1,3}|iv|v)\b", message_lower, flags=re.IGNORECASE)
        if explicit_match:
            token = explicit_match.group(1).upper()
            if token in self.STAGES:
                stage = token
                confidence = 0.9
                matched_signals.append(f"Explicit stage mention: {token}")

        if not stage:
            stage_scores = {s: 0 for s in self.STAGES}
            keyword_map = {
                "0": ["basic research", "mechanism", "hypothesis", "preliminary", "predictor"],
                "I": ["feasibility", "pilot", "small sample", "manualization", "usability", "acceptability"],
                "II": ["efficacy", "randomized", "rct", "control", "mechanism tested"],
                "III": ["effectiveness", "real world", "diverse", "pragmatic"],
                "IV": ["implementation", "dissemination", "scale", "adoption"],
                "V": ["sustainability", "maintenance", "long term"],
            }

            for stage_key, keywords in keyword_map.items():
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
                stage = best_stage if confidence >= 0.58 else None

        if stage and confidence >= 0.58:
            if stage in {"I", "II", "III", "IV", "V"}:
                feature_updates["intervention_defined"] = True
                feature_updates["manualized"] = True
            if stage in {"II", "III", "IV", "V"}:
                feature_updates["mechanism_tested"] = True
            if stage in {"III", "IV", "V"}:
                feature_updates["efficacy_tested"] = True
            if stage in {"IV", "V"}:
                feature_updates["effectiveness_tested"] = True
            if stage == "V":
                feature_updates["implementation_tested"] = True

        tool_calls = []
        if confidence < 0.58:
            tool_calls.append(
                ToolCall(
                    tool_name="db_tool",
                    tool_args={"query": f"Classify research stage: {user_message}"},
                    success_criteria="Return matching stage definition",
                )
            )

        reasoning_summary = (
            f"Matched signals: {'; '.join(matched_signals[:4])}. Current confidence={confidence:.2f}."
            if matched_signals
            else f"Insufficient stage signals. Current confidence={confidence:.2f}."
        )

        clarifying_question = None
        missing_info = self.MISSING_INFO_HINTS[:] if confidence < 0.75 else []
        if confidence < 0.75:
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
            actions=tool_calls,
        )

    def update_state(self, state: SessionState, output: AgentOutput):
        if (
            "stage" in output.decision
            and output.decision["stage"]
            and output.confidence >= 0.58
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