"""
Study Builder Agent: create stage-aware study design matrix output.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.core.types import AgentOutput, SessionState


class StudyBuilderAgent(BaseAgent):
    def __init__(self):
        super().__init__("StudyBuilderAgent")

    def run(self, state: SessionState, user_message: str, context: str = "") -> AgentOutput:
        stage = state.slots.stage or "I"
        design = "pilot feasibility trial" if stage in {"0", "I"} else "efficacy RCT" if stage == "II" else "pragmatic effectiveness design"

        decision = {
            "workflow": "study_builder",
            "structured_output": {
                "stage_anchor": f"Stage {stage}",
                "research_questions": [
                    "Does the intervention improve the target behavior?",
                    "Is mechanism change observed in the expected direction?",
                ],
                "design_options": [design, "two-arm randomized design with clear comparator"],
                "comparator_options": ["attention control", "usual care"],
                "fidelity_plan": [
                    "manual adherence checklist",
                    "session-level fidelity audit",
                ],
                "mechanism_measurement_plan": [
                    "baseline + mid + endpoint mechanism measurements",
                    "prespecified mediation model",
                ],
                "feasibility_and_sample_size_logic": "Use pilot precision logic for early-stage studies and power-based logic for efficacy/effectiveness without pretending to do full power analysis absent assumptions.",
            },
            "summary": "Generated stage-aware study design matrix with comparator and fidelity planning.",
        }
        return AgentOutput(
            decision=decision,
            confidence=0.84,
            analysis="Study builder structured output generated",
        )

    def update_state(self, state: SessionState, output: AgentOutput):
        state.slots.extracted_features["workflow"] = "study_builder"
        state.slots.extracted_features["workflow_structured_output"] = output.decision.get("structured_output", {})
        state.slots.extracted_features["workflow_summary"] = output.decision.get("summary", "")
