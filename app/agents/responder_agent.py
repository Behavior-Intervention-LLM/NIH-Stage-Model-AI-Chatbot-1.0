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
        # "system_definition": (
        #     "Use retrieval evidence to support or refine your answer, "
        #     "but do not restrict your answer only to the provided snippets. "
        #     "If the retrieved evidence is incomplete, provide the best complete answer from your general knowledge "
        #     "and note that the snippet is incomplete."
        # ),

        "system_general": (
            "You are a chatbot designed for users to help with any and all information pertaining to Behavioral Science Intervention Methodologies. Your responses should align well and closely with the philosophies and pillars of foundation of the National Institute of Health",
            "Use relevant facts in the CONTEXT block below (stage, confidence, workflow outputs, RAG snippets when answering"
            "If there is any missing info, do not dump raw field names or imitate an internal execution trace.\n"
            "If stage confidence is low or stage is unknown, say so plainly and ask focused follow-ups.\n"
            "If retrieval evidence exists, ground claims briefly (sources or quotes as appropriate).\n"
            "Match the user's language when obvious. Output plain text only (no JSON)."
        ),
        
        "user_instruction_definition": (
            "Provide: (1) number of stages, (2) stage names, and (3) one-line description per stage."
        ),
    }

    STAGE_INFO = {
            "0": "Stage 0 involves basic science that occurs prior to intervention development, but is relevant (ultimately translatable) to intervention development. Research on mechanisms of change is an integral part of all other stages of intervention development, involving basic science questions about behavior change within the context of intervention development studies.",
            "1": "Stage I encompasses all activities related to the creation and preliminary testing of a new behavioral intervention, including generation of new interventions as well as modification, adaptation, or refinement of existing interventions (Stage IA), culminating in feasibility and pilot testing (Stage IB). Stage I can also include modification of an intervention for implementability, development of training materials, and may be conducted in research or community settings.",
            "2": "Stage II (Pure Efficacy) consists of experimental testing of promising behavioral interventions in research settings, with research-based providers.",
            "3": "Stage III (Real World Efficacy) consists of experimental testing of promising behavioral interventions in community settings, with community-based providers or caregivers, while maintaining a high level of control necessary to establish internal validity. This is sometimes referred to as a hybrid efficacy-effectiveness stage.",
            "4": "Stage IV (Effectiveness) examines empirically supported behavioral interventions in community settings, with community-based providers or caregivers, while maximizing external validity.",
            "5": "Stage V (Implementation and Dissemination) examines strategies of implementation and adoption of empirically supported interventions in community settings.",
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
        """Gather this turn's retrieved passages.

        The orchestrator has already dropped artifacts whose passages fell
        below the relevance floor, so anything reaching here is usable — no
        string-sniffing for "not found" needed. Only real document names go
        into evidence_sources; the tool name is not a knowledge source.
        """
        evidence_lines: List[str] = []
        evidence_sources: List[str] = []
        citations = []

        for artifact in state.artifacts:
            for c in artifact.citations:
                citations.append(c)
                if c.source and c.source not in evidence_sources:
                    evidence_sources.append(c.source)
            if isinstance(artifact.content, str) and artifact.content.strip():
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
        assessment = xf.get("evidence_assessment") or {}
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
            f"Retrieval verdict: {assessment.get('reason', 'unknown')} "
            f"(attempts={assessment.get('attempts')}, "
            f"usable_passages={assessment.get('usable_count', 0)}, "
            f"best_similarity={assessment.get('best_score')})",
            (
                "Retrieval found no relevant passage in the document corpus. Answer "
                "from general knowledge and say plainly that the corpus does not "
                "cover this; do not imply the answer is document-grounded and do "
                "not cite sources."
                if not rag_active
                else "Ground claims in the evidence snippets below and name the sources you use."
            ),
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

    def _build_llm_prompts(
        self, state: SessionState, user_message: str, context: str = ""
    ) -> tuple[str, str, List]:
        """Build (system_prompt, user_prompt, citations) for the final response."""
        evidence_lines, evidence_sources, citations = self._collect_evidence(state)

        message_lower = user_message.lower().strip()
        intent_payload = state.slots.extracted_features.get("intent_payload", {}) or {}
        intent_query_type = str(intent_payload.get("query_type", "")).lower()

        is_stage_definition_query = (
            any(k in message_lower for k in ["what is", "what's", "define", "how many stages", "number of stages", "list stages"])
            and any(k in message_lower for k in ["nih stage model", "nih stage", "stage model"])
        ) or intent_query_type == "definition"

        sections = self._get_responder_sections()
        system_prompt = sections["system_general"]
        base_context = self._build_general_context(
            state, user_message, context, evidence_lines, evidence_sources
        )

        if is_stage_definition_query:
            user_tail = sections.get("user_instruction_definition") or self._FALLBACK_SECTIONS[
                "user_instruction_definition"
            ]
            user_prompt = (
                f"{base_context}\n\n"
                f"--- TASK INSTRUCTION ---\n"
                f"{user_tail}\n"
            )
        else:
            user_prompt = base_context

        return system_prompt, user_prompt, citations

    def _run_with_llm(self, state: SessionState, user_message: str, context: str = "") -> AgentOutput | None:
        if not llm_client.is_enabled():
            return None

        system_prompt, user_prompt, citations = self._build_llm_prompts(state, user_message, context)
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

    def run_stream(self, state: SessionState, user_message: str, context: str, on_chunk) -> AgentOutput:
        """Like run(), but emits response text incrementally via on_chunk(str).
        Returns the complete AgentOutput once the stream finishes."""
        if not llm_client.is_enabled():
            out = self._run_with_rules(state, user_message)
            if out.user_facing:
                on_chunk(out.user_facing)
            return out

        system_prompt, user_prompt, citations = self._build_llm_prompts(state, user_message, context)
        parts: List[str] = []
        for chunk in llm_client.chat_text_stream(system_prompt=system_prompt, user_prompt=user_prompt):
            parts.append(chunk)
            on_chunk(chunk)
        text = "".join(parts).strip()

        if not text:
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

        return AgentOutput(
            decision={},
            confidence=0.9,
            analysis="LLM generated final response (streamed)",
            user_facing=text,
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