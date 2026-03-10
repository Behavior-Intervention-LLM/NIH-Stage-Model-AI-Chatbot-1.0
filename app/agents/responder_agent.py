"""
Responder Agent: execute plan and generate final user-facing response.
"""
import re
from typing import List

from app.agents.base import BaseAgent
from app.core.llm import llm_client
from app.core.types import AgentOutput, SessionState


class ResponderAgent(BaseAgent):
    """Response generation agent."""

    STAGE_INFO = {
        "0": "Stage 0 focuses on basic research, mechanism discovery, and hypothesis building.",
        "I": "Stage I focuses on feasibility testing, intervention refinement, and manualization.",
        "II": "Stage II focuses on efficacy testing and mechanism validation, often with randomized controlled designs.",
        "III": "Stage III focuses on effectiveness in real-world and diverse settings.",
        "IV": "Stage IV focuses on implementation, dissemination, and scale-up.",
        "V": "Stage V focuses on sustainability and long-term maintenance.",
    }

    def __init__(self):
        super().__init__("ResponderAgent")

    @staticmethod
    def _collect_evidence(state: SessionState) -> tuple[List[str], List[str], List]:
        evidence_lines: List[str] = []
        evidence_sources: List[str] = []
        citations = []

        for artifact in state.artifacts[-8:]:
            if artifact.citations:
                citations.extend(artifact.citations)
                for c in artifact.citations:
                    if c.source and c.source not in evidence_sources:
                        evidence_sources.append(c.source)
            if artifact.tool_name and artifact.tool_name not in evidence_sources:
                evidence_sources.append(artifact.tool_name)
            if isinstance(artifact.content, str):
                if "no matching" not in artifact.content.lower() and "not found" not in artifact.content.lower():
                    evidence_lines.append(artifact.content[:500])

        return evidence_lines, evidence_sources, citations

    def run(self, state: SessionState, user_message: str, context: str = "") -> AgentOutput:
        llm_output = self._run_with_llm(state, user_message, context)
        if llm_output:
            return llm_output
        return self._run_with_rules(state, user_message)

    def _run_with_llm(self, state: SessionState, user_message: str, context: str = "") -> AgentOutput | None:
        if not llm_client.is_enabled():
            return None

        evidence_lines, evidence_sources, citations = self._collect_evidence(state)
        rag_active = len(evidence_lines) > 0 or len(evidence_sources) > 0

        message_lower = user_message.lower().strip()
        intent_payload = state.slots.extracted_features.get("intent_payload", {}) or {}
        intent_query_type = str(intent_payload.get("query_type", "")).lower()

        is_stage_definition_query = (
            any(k in message_lower for k in ["what is", "what's", "define", "how many stages", "number of stages", "list stages"])
            and any(k in message_lower for k in ["nih stage model", "nih stage", "stage model"])
        ) or intent_query_type == "definition"

        planner_outline = state.slots.extracted_features.get("planner_outline")
        next_question = state.slots.extracted_features.get("next_question")
        stage_reasoning = state.slots.extracted_features.get("reasoning_summary")
        missing_info = state.slots.extracted_features.get("missing_info", [])
        clarifying_question = state.slots.extracted_features.get("clarifying_question")
        intent_missing = state.slots.extracted_features.get("intent_missing_info", []) or []
        intent_clarifying = state.slots.extracted_features.get("intent_clarifying_question")

        if is_stage_definition_query:
            system_prompt = (
                "You are an NIH Stage Model explainer. "
                "Answer only the current definition/list question. "
                "Do not include prior case details or stage-classification follow-ups. "
                "If retrieval evidence is available, prioritize the newest/revised source and mention it explicitly."
            )
            user_prompt = (
                f"Question: {user_message}\n"
                f"Knowledge sources: {evidence_sources}\n"
                f"Evidence snippets: {evidence_lines}\n"
                "Provide: (1) number of stages, (2) stage names, and (3) one-line description per stage."
            )
            text = llm_client.chat_text(system_prompt=system_prompt, user_prompt=user_prompt)
        else:
            system_prompt = (
                "You are an NIH Stage Model assistant.\n"
                "Rules:\n"
                "1) Answer the user's current question directly.\n"
                "2) If stage is known, explicitly write Stage 0 / Stage I / Stage II / Stage III / Stage IV / Stage V.\n"
                "3) Include a short reasoning summary.\n"
                "4) If information is missing, list missing items and ask a follow-up question.\n"
                "5) If evidence exists, explicitly include 'Based on knowledge sources: ...'.\n"
                "6) Provide a detailed but concise response in 2-4 short paragraphs."
            )
            user_prompt = (
                f"Question: {user_message}\n"
                f"Slots: {state.slots.model_dump()}\n"
                f"Intent payload: {intent_payload}\n"
                f"Planner outline: {planner_outline}\n"
                f"Next question: {next_question}\n"
                f"Stage reasoning: {stage_reasoning}\n"
                f"Missing info (stage): {missing_info}\n"
                f"Missing info (intent): {intent_missing}\n"
                f"Clarifying question (stage): {clarifying_question}\n"
                f"Clarifying question (intent): {intent_clarifying}\n"
                f"RAG active: {rag_active}\n"
                f"Knowledge sources: {evidence_sources}\n"
                f"Evidence snippets: {evidence_lines}\n"
                f"Recent context: {context[:1600]}"
            )
            text = llm_client.chat_text(system_prompt=system_prompt, user_prompt=user_prompt)

        if not text:
            return None

        return AgentOutput(
            decision={},
            confidence=0.9,
            analysis="LLM generated final response",
            user_facing=text.strip(),
            metadata={"citations": [c.model_dump() for c in citations]},
        )

    def _run_with_rules(self, state: SessionState, user_message: str) -> AgentOutput:
        evidence_texts, evidence_sources, citations = self._collect_evidence(state)
        rag_active = len(evidence_texts) > 0 or len(evidence_sources) > 0

        planner_output = state.slots.extracted_features.get("planner_outline", "")
        stage_reasoning = state.slots.extracted_features.get("reasoning_summary", "")
        missing_info = state.slots.extracted_features.get("missing_info", []) or []
        clarifying_question = state.slots.extracted_features.get("clarifying_question")
        intent_payload = state.slots.extracted_features.get("intent_payload", {}) or {}
        intent_missing = state.slots.extracted_features.get("intent_missing_info", []) or []
        intent_clarifying = state.slots.extracted_features.get("intent_clarifying_question")

        response_parts: List[str] = []
        message_lower = user_message.lower().strip()
        intent_query_type = str(intent_payload.get("query_type", "")).lower()

        if any(greet in message_lower for greet in ["hello", "hi", "hey"]):
            response_parts.append(
                "Hi! I am your NIH Stage Model assistant. You can ask what NIH Stage Model is, "
                "stage-specific requirements, or next-step suggestions."
            )

        asks_definition = (
            "nih stage model" in message_lower
            and any(k in message_lower for k in ["what is", "what's", "define", "how many stages", "number of stages", "list stages", "explain"])
        ) or intent_query_type == "definition"

        if asks_definition:
            response_parts.append(
                "The NIH Stage Model has 6 stages: Stage 0, Stage I, Stage II, Stage III, Stage IV, and Stage V."
            )
            response_parts.append(
                "Stage 0 (basic mechanisms), Stage I (feasibility/manualization), "
                "Stage II (efficacy + mechanism validation), Stage III (effectiveness in real-world settings), "
                "Stage IV (implementation/dissemination), and Stage V (sustainability)."
            )
            if rag_active and evidence_sources:
                response_parts.append(
                    f"Based on version-aware knowledge sources: {', '.join(evidence_sources[:3])}."
                )
                if evidence_texts:
                    response_parts.append(f"Latest evidence summary: {evidence_texts[0][:220]}...")
            return AgentOutput(
                decision={},
                confidence=0.95,
                analysis="Definition query answered without carryover",
                user_facing="\n".join(response_parts),
                metadata={"citations": [c.model_dump() for c in citations]},
            )

        stage_match = re.search(r"stage\s*(0|i{1,3}|iv|v)\b", message_lower, flags=re.IGNORECASE)
        if stage_match:
            stage_token = stage_match.group(1).upper()
            if stage_token in self.STAGE_INFO:
                if any(k in message_lower for k in ["requirement", "requirements", "criteria"]):
                    response_parts.append(self.STAGE_INFO[stage_token])
                elif not response_parts:
                    response_parts.append(self.STAGE_INFO[stage_token])

        if state.slots.stage:
            response_parts.append(f"Based on current information, your project is most likely at **Stage {state.slots.stage}**.")
            response_parts.append(
                "Reasoning basis: your study description aligns with common goals and designs of this stage "
                "(e.g., feasibility, RCT, real-world implementation)."
            )

        if stage_reasoning:
            response_parts.append(f"Reasoning summary: {stage_reasoning}")

        if planner_output:
            response_parts.append(planner_output)

        if intent_payload.get("query_type"):
            response_parts.append(f"Intent interpretation: `{intent_payload.get('query_type')}`.")

        if rag_active:
            response_parts.append(f"Based on knowledge sources: {', '.join(evidence_sources[:5])}.")
            response_parts.append("The response below is grounded in retrieved knowledge (RAG):")
            for i, evidence in enumerate(evidence_texts[:3], 1):
                response_parts.append(f"{i}. {evidence[:240]}...")

        next_question = state.slots.extracted_features.get("next_question")
        if next_question:
            response_parts.append(f"To improve answer precision, could you clarify: {next_question}")

        if missing_info:
            response_parts.append("The following key information is still missing (and affects stage confidence):")
            for i, item in enumerate(missing_info[:4], 1):
                response_parts.append(f"{i}. {item}")

        if intent_missing:
            response_parts.append("Based on intent extraction, please also provide:")
            for i, item in enumerate(intent_missing[:3], 1):
                response_parts.append(f"{i}. {item}")

        if clarifying_question:
            response_parts.append(f"Please provide: {clarifying_question}")
        if intent_clarifying and intent_clarifying != clarifying_question:
            response_parts.append(f"Additional clarifying question: {intent_clarifying}")

        if not response_parts:
            response_parts.append(
                "I can help with NIH Stage Model questions. You can ask: "
                "\"What is NIH Stage Model?\", \"What are Stage I requirements?\", "
                "or \"What should be my next step?\""
            )

        return AgentOutput(
            decision={},
            confidence=0.9,
            analysis="Generated final response",
            user_facing="\n".join(response_parts),
            metadata={"citations": [c.model_dump() for c in citations]},
        )
