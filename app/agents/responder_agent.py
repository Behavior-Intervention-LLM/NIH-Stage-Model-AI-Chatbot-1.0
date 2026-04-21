"""
Responder Agent: execute plan and generate final user-facing response.
"""
import json
import re
from pathlib import Path
from typing import Any, List

from app.agents.base import BaseAgent
from app.core.llm import llm_client
from app.core.types import AgentOutput, SessionState


class ResponderAgent(BaseAgent):
    """Response generation agent."""

    _FALLBACK_SECTIONS = {
        "system_definition": (
            "Use retrieval evidence to support or refine your answer, "
            "but do not restrict your answer only to the provided snippets. "
            "If the retrieved evidence is incomplete, provide the best complete answer from your general knowledge "
            "and note that the snippet is incomplete."
        ),

        "system_general": (
            "You are an NIH Stage Model assistant. Synthesize a single clear answer for the user.\n"
            "Use every relevant fact in the CONTEXT block below (stage, confidence, workflow outputs, RAG snippets, "
            "missing info, guardrails). Do not dump raw field names or imitate an internal execution trace.\n"
            "If stage confidence is low or stage is unknown, say so plainly and ask focused follow-ups.\n"
            "If retrieval evidence exists, ground claims briefly (sources or quotes as appropriate).\n"
            "Match the user's language when obvious. Output plain text only (no JSON)."
        ),
        "user_instruction_definition": (
            "Provide: (1) number of stages, (2) stage names, and (3) one-line description per stage."
        ),
    }

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
        self._prompt_file = Path(__file__).resolve().parents[1] / "prompts" / "responder.md"

    @staticmethod
    def _parse_responder_markdown(text: str) -> dict[str, str]:
        """Split app/prompts/responder.md on `## section_name` headings (level-2 only)."""
        sections: dict[str, str] = {}
        current: str | None = None
        buf: list[str] = []
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("## ") and not s.startswith("###"):
                if current is not None:
                    sections[current] = "\n".join(buf).strip()
                current = s[3:].strip()
                buf = []
            elif current is not None:
                buf.append(line)
        if current is not None:
            sections[current] = "\n".join(buf).strip()
        return sections

    def _get_responder_sections(self) -> dict[str, str]:
        merged = dict(self._FALLBACK_SECTIONS)
        try:
            if self._prompt_file.exists():
                raw = self._prompt_file.read_text(encoding="utf-8")
                parsed = self._parse_responder_markdown(raw)
                for key, val in parsed.items():
                    if val:
                        merged[key] = val
        except Exception:
            pass
        return merged

    @staticmethod
    def _collect_evidence(state: SessionState) -> tuple[List[str], List[str], List]:
        evidence_lines: List[str] = []
        evidence_sources: List[str] = []
        citations = []

        # 1. NEW: Check extracted_features for RAG data
        xf = state.slots.extracted_features
        rag_docs = xf.get("retrieved_context") or xf.get("context")
        
        if isinstance(rag_docs, list):
            for doc in rag_docs:
                # Use the 'text' and 'metadata' keys we cleaned up in RAGAgent
                text = doc.get("text", "")
                pmcid = doc.get("metadata", {}).get("pmcid", "unknown")
                
                if text:
                    evidence_lines.append(text)
                if pmcid not in evidence_sources:
                    evidence_sources.append(pmcid)

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

    def _workflow_structured_excerpt(self, structured: Any, max_chars: int = 4500) -> str:
        if not structured:
            return "{}"
        try:
            raw = json.dumps(structured, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            raw = str(structured)
        if len(raw) > max_chars:
            return raw[:max_chars] + "\n… [truncated]"
        return raw

    def _build_general_context(
        self,
        state: SessionState,
        user_message: str,
        context: str,
        evidence_lines: List[str],
        evidence_sources: List[str],
    ) -> str:
        rag_active = len(evidence_lines) > 0 or len(evidence_sources) > 0
        intent_payload = state.slots.extracted_features.get("intent_payload", {}) or {}
        xf = state.slots.extracted_features
        workflow = xf.get("workflow", "navigator")
        workflow_summary = xf.get("workflow_summary") or ""
        workflow_structured = xf.get("workflow_structured_output") or {}
        guardrail_warnings = xf.get("guardrail_warnings") or []

        lines = [
            f"User question: {user_message}",
            "",
            "--- CONTEXT (for synthesis; do not quote section headers to the user) ---",
            f"Inferred stage: {state.slots.stage!r} | stage_confidence: {state.slots.stage_confidence!r}",
            f"Workflow mode: {workflow}",
            f"Workflow agent summary: {workflow_summary}",
            "",
            "Workflow structured JSON:",
            self._workflow_structured_excerpt(workflow_structured),
            "",
            f"Intent payload: {intent_payload}",
            f"Planner outline: {xf.get('planner_outline')}",
            f"Next question (upstream): {xf.get('next_question')}",
            f"Stage reasoning summary: {xf.get('reasoning_summary')}",
            f"Missing info (stage): {xf.get('missing_info')}",
            f"Missing info (intent): {xf.get('intent_missing_info')}",
            f"Clarifying question (stage): {xf.get('clarifying_question')}",
            f"Clarifying question (intent): {xf.get('intent_clarifying_question')}",
            f"RAG active: {rag_active}",
            f"Knowledge sources: {evidence_sources}",
            f"Evidence snippets: {evidence_lines}",
            f"Guardrail warnings: {guardrail_warnings}",
            f"Stage uncertain (low confidence or unknown): {xf.get('stage_uncertain_hint', False)}",
            "",
            "Full slots (reference):",
            json.dumps(state.slots.model_dump(), ensure_ascii=False, default=str)[:6000],
            "",
            f"Recent conversation context:\n{(context or '')[:2000]}",
        ]
        return "\n".join(lines)

    def run(self, state: SessionState, user_message: str, context: str = "") -> AgentOutput:
        if llm_client.is_enabled():
            llm_output = self._run_with_llm(state, user_message, context)
            if llm_output and (llm_output.user_facing or "").strip():
                return llm_output
            return AgentOutput(
                decision={},
                confidence=0.25,
                analysis="LLM returned empty response",
                user_facing=(
                    "I could not generate a reply (the language model returned no text). "
                    "Please try again or check your LLM provider configuration."
                ),
                metadata={},
            )
        return self._run_with_rules(state, user_message)

    def _run_with_llm(self, state: SessionState, user_message: str, context: str = "") -> AgentOutput | None:
        if not llm_client.is_enabled():
            return None

        evidence_lines, evidence_sources, citations = self._collect_evidence(state)

        message_lower = user_message.lower().strip()
        intent_payload = state.slots.extracted_features.get("intent_payload", {}) or {}
        intent_query_type = str(intent_payload.get("query_type", "")).lower()

        # is_stage_definition_query = (intent_query_type == "definition")

        is_stage_definition_query = (
            any(k in message_lower for k in ["what is", "what's", "define", "how many stages", "number of stages", "list stages"])
            and any(k in message_lower for k in ["nih stage model", "nih stage", "stage model"])
        ) or intent_query_type == "definition"

        # stage_info_text = self._format_stage_info()

        sections = self._get_responder_sections()

        if is_stage_definition_query:
            system_prompt = sections["system_general"]

            # should be the part that concat responder.md
            user_tail = sections.get("user_instruction_definition") or self._FALLBACK_SECTIONS[
                "user_instruction_definition"
            ]

            base_context = self._build_general_context(
                state, user_message, context, evidence_lines, evidence_sources
            )
            
            user_prompt = (
                f"{base_context}\n\n"
                # f"{stage_info_text}\n"
                f"--- TASK INSTRUCTION ---\n"
                f"{user_tail}\n"
            )

            text = llm_client.chat_text(system_prompt=system_prompt, user_prompt=user_prompt)

        else:
            system_prompt = sections["system_general"]
            user_prompt = self._build_general_context(
                state, user_message, context, evidence_lines, evidence_sources
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

    # def _run_with_llm(self, state: SessionState, user_message: str, context: str = "") -> AgentOutput:
    #     evidence_lines, evidence_sources, citations = self._collect_evidence(state)
    #     sections = self._get_responder_sections()
    #     mode = self._detect_query_mode(user_message, state)

    #     system_prompt = sections["system_global"]
    #     user_prompt = self._build_user_prompt(
    #         state=state,
    #         user_message=user_message,
    #         context=context,
    #         evidence_lines=evidence_lines,
    #         evidence_sources=evidence_sources,
    #         mode=mode,
    #         sections=sections,
    #     )

    #     text = llm_client.chat_text(system_prompt=system_prompt, user_prompt=user_prompt)

    #     if not text or not text.strip():
    #         return AgentOutput(
    #             decision={},
    #             confidence=0.2,
    #             analysis="LLM returned empty response",
    #             user_facing=(
    #                 "I could not generate a reply because the language model returned no text. "
    #                 "Please try again or check the LLM provider configuration."
    #             ),
    #             metadata={"mode": mode},
    #         )

    #     return AgentOutput(
    #         decision={"mode": mode, "rag_active": bool(evidence_lines or evidence_sources)},
    #         confidence=0.92,
    #         analysis=f"LLM generated final response in {mode} mode",
    #         user_facing=text.strip(),
    #         metadata={
    #             "mode": mode,
    #             "citations": [c.model_dump() for c in citations],
    #             "evidence_sources": evidence_sources,
    #         },
    #     )

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