"""
Mechanism Coach Agent: generate mechanism-focused structured guidance.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.core.types import AgentOutput, SessionState


class MechanismCoachAgent(BaseAgent):
    def __init__(self):
        super().__init__("MechanismCoachAgent")

    def run(self, state: SessionState, user_message: str, context: str = "") -> AgentOutput:
        msg = user_message.lower()
        candidates = []
        if "adherence" in msg:
            candidates.append("self-efficacy")
            candidates.append("habit strength")
        if "sleep" in msg:
            candidates.append("sleep self-regulation")
            candidates.append("sleep-related beliefs")
        if not candidates:
            candidates = ["self-efficacy", "motivation", "contextual cue response"]

        ranked = []
        for i, name in enumerate(candidates[:3]):
            ranked.append(
                {
                    "name": name,
                    "plausibility_score": round(0.86 - i * 0.08, 2),
                    "modifiability_score": round(0.82 - i * 0.07, 2),
                    "measurement_feasibility_score": round(0.9 - i * 0.06, 2),
                }
            )

        decision = {
            "workflow": "mechanism_coach",
            "structured_output": {
                "candidate_mechanisms_ranked": ranked,
                "measurement_options": [
                    {
                        "option": "PROMIS short forms for relevant constructs",
                        "pros": ["brief", "good psychometric support"],
                        "cons": ["may be less construct-specific"],
                        "respondent_burden": "low",
                    },
                    {
                        "option": "Validated domain-specific scale aligned to target behavior",
                        "pros": ["higher construct specificity"],
                        "cons": ["may require population fit review"],
                        "respondent_burden": "moderate",
                    },
                ],
                "manipulation_options": [
                    "coaching + feedback",
                    "goal-setting with implementation intentions",
                ],
                "verification_plan": {
                    "design_logic": "pair mechanism change with behavior change in a stage-appropriate design",
                    "mediation_plan": "collect repeated mechanism measures and test whether mechanism change aligns with outcome change",
                },
            },
            "summary": "Generated mechanism candidates with ranking, measurement links, and verification suggestions.",
        }
        return AgentOutput(
            decision=decision,
            confidence=0.86,
            analysis="Mechanism coaching structured output generated",
        )

    def update_state(self, state: SessionState, output: AgentOutput):
        state.slots.extracted_features["workflow"] = "mechanism_coach"
        state.slots.extracted_features["workflow_structured_output"] = output.decision.get("structured_output", {})
        state.slots.extracted_features["workflow_summary"] = output.decision.get("summary", "")
