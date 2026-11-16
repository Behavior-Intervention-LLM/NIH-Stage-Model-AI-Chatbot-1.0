"""Application partner using the shared responder's streaming/LLM execution."""
from pathlib import Path

from app.agents.rag_agent import RAGAgent
from app.agents.responder_agent import ResponderAgent
from app.config import settings
from app.core.stage_model import stage_overview
from app.core.types import AgentOutput, SessionState


class GrantPartnerAgent(ResponderAgent):
    def __init__(self):
        super().__init__()
        self.name = "GrantPartnerAgent"
        self._grant_prompt_file = Path(__file__).resolve().parents[1] / "prompts" / "grant.md"

    @staticmethod
    def _collect_evidence(state):
        # Send full usable passages: the general responder's 500-character
        # artifact excerpt is too short for conditional application requirements.
        citations = []
        seen = set()
        for artifact in state.artifacts:
            for citation in artifact.citations:
                key = (citation.source, citation.metadata.get("page"), citation.passage)
                if RAGAgent.is_usable(citation) and key not in seen:
                    seen.add(key)
                    citations.append(citation)
        lines = []
        for c in citations:
            m = c.metadata
            lines.append(
                f"[{c.source}, PDF p. {m.get('page', '?')}, {m.get('section', '')}; "
                f"FORMS {m.get('forms_version', '?')}, released {m.get('released', '?')}]\n"
                f"{c.passage}"
            )
        return lines, list(dict.fromkeys(c.source for c in citations)), citations

    def _build_llm_prompts(self, state, user_message, context=""):
        lines, sources, citations = self._collect_evidence(state)
        system_prompt = self._grant_prompt_file.read_text(encoding="utf-8")
        system_prompt += "\n\nCanonical NIH Stage Model (not application activity codes):\n" + stage_overview()
        user_prompt = self._build_general_context(state, user_message, context, lines, sources)
        accepted_offer = state.slots.extracted_features.get("accepted_offer")
        if accepted_offer:
            user_prompt += f"\nThe user accepted this task; complete it: {accepted_offer}"
        return system_prompt, user_prompt, citations, settings.LLM_COMPOSE_MAX_TOKENS

    def _run_with_rules(self, state: SessionState, user_message: str) -> AgentOutput:
        return AgentOutput(
            decision={}, confidence=0.0,
            analysis="Application partner requires a configured language model",
            user_facing="The application partner needs a configured language model to draft or review your application. Please check the LLM configuration.",
        )
