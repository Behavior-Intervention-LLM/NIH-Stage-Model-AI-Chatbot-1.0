"""
Grant Partner Agent: specific aims skeleton + mock reviewer critique + revision plan.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.core.types import AgentOutput, SessionState


class GrantPartnerAgent(BaseAgent):
    def __init__(self):
        super().__init__("GrantPartnerAgent")

    def run(self, state: SessionState, user_message: str, context: str = "") -> AgentOutput:
        stage = state.slots.stage or "I"
        mechanism_hint = "self-efficacy" if "adherence" in user_message.lower() else "target mechanism"

        structured_output = {
            "stage_anchor": f"Stage {stage}",
            "specific_aims_skeleton": [
                {
                    "aim": "Aim 1",
                    "text": f"Test whether the intervention improves the target behavior in a Stage {stage}-appropriate design.",
                },
                {
                    "aim": "Aim 2",
                    "text": f"Evaluate whether {mechanism_hint} changes in the expected direction and plausibly mediates the behavioral outcome.",
                },
            ],
            "mock_reviewer_critique": {
                "significance": "Strong if the target behavior is clinically meaningful and population burden is clearly stated.",
                "innovation": "Moderate; strengthen novelty by clarifying what is new in mechanism targeting or delivery approach.",
                "approach": "Needs tighter alignment between stage, design choice, comparator, and mechanism measurement.",
                "stage_fit": f"Current plan appears most consistent with Stage {stage}.",
                "mechanism_test_quality": "Adequate only if timing and construct measures are prespecified.",
                "scalability_fidelity": "Address implementation feasibility and fidelity monitoring earlier.",
            },
            "revision_plan_top5": [
                "Clarify the stage-specific research question and justify why this stage is appropriate.",
                "Specify the primary comparator and what decision it supports.",
                "Tighten the mechanism measurement schedule and mediation logic.",
                "Add a concise fidelity monitoring plan tied to intervention delivery.",
                "State explicit decision rules for progression to the next NIH stage.",
            ],
            "before_after_aim_language": {
                "before": "We will test whether our intervention works.",
                "after": f"We will conduct a Stage {stage}-appropriate study to test whether the intervention improves the target behavior and whether mechanism change supports progression to the next stage.",
            },
        }

        return AgentOutput(
            decision={
                "workflow": "grant_partner",
                "structured_output": structured_output,
                "summary": "Generated a stage-aligned Specific Aims skeleton, reviewer-aware critique, and revision plan.",
            },
            confidence=0.83,
            analysis="Grant partner structured output generated",
        )

    def update_state(self, state: SessionState, output: AgentOutput):
        state.slots.extracted_features["workflow"] = "grant_partner"
        state.slots.extracted_features["workflow_structured_output"] = output.decision.get("structured_output", {})
        state.slots.extracted_features["workflow_summary"] = output.decision.get("summary", "")
