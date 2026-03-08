"""
Measure Finder Agent: shortlist measures and fit notes.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.core.types import AgentOutput, SessionState


class MeasureFinderAgent(BaseAgent):
    def __init__(self):
        super().__init__("MeasureFinderAgent")

    def run(self, state: SessionState, user_message: str, context: str = "") -> AgentOutput:
        msg = user_message.lower()
        construct = "sleep self-regulation" if "sleep" in msg else "medication adherence" if "adherence" in msg else "target construct"

        shortlist = [
            {
                "name": "PROMIS short-form measure",
                "construct_definition": f"Candidate measure for {construct}",
                "administration_mode": "self-report",
                "burden": "low",
                "psychometrics_notes": "Strong general psychometric support; verify fit for the target population.",
                "population_fit": "adult populations",
                "why_fit": "Brief format with broad psychometric support and easy longitudinal use.",
            },
            {
                "name": "Behavior-specific validated scale",
                "construct_definition": f"Alternative construct-specific measure for {construct}",
                "administration_mode": "self-report",
                "burden": "moderate",
                "psychometrics_notes": "Potentially stronger construct alignment but more variable by population.",
                "population_fit": "population-specific adaptation needed",
                "why_fit": "Higher construct specificity for mechanism-aligned endpoints.",
            },
        ]

        decision = {
            "workflow": "measure_finder",
            "structured_output": {
                "construct": construct,
                "measure_to_construct_mapping": f"Shortlisted measures are mapped to the construct '{construct}' using burden, fit, and likely mechanism relevance.",
                "measure_shortlist": shortlist,
                "gap_flags": ["Confirm population-specific validation before final selection."],
            },
            "summary": "Generated measure shortlist with burden, fit, and gap flags.",
        }
        return AgentOutput(
            decision=decision,
            confidence=0.85,
            analysis="Measure finder structured output generated",
        )

    def update_state(self, state: SessionState, output: AgentOutput):
        state.slots.extracted_features["workflow"] = "measure_finder"
        state.slots.extracted_features["workflow_structured_output"] = output.decision.get("structured_output", {})
        state.slots.extracted_features["workflow_summary"] = output.decision.get("summary", "")
