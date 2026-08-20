"""
Planner Agent： stage +  + ，next step
"""
from app.agents.base import BaseAgent
from app.core.stage_model import NEXT_STAGE, STAGE_MODEL
from app.core.types import SessionState, AgentOutput, PlanStep, PlanStepType, ToolCall
from app.core.llm import llm_client


class PlannerAgent(BaseAgent):
    """Planning agent"""

    def __init__(self):
        super().__init__("PlannerAgent")

    @staticmethod
    def _unmet_prerequisite(state: SessionState, stage: str) -> tuple[str, str] | None:
        """The slot that still has to be filled before this stage's own work
        can be called finished. Returns (step description, question to ask)."""
        if stage == "0" and not state.slots.intervention_defined:
            return (
                "Clarify what the intervention will be",
                "What intervention are you developing, and which mechanism does it target?",
            )
        # Manualization is Stage IA's output and Stage IB's precondition.
        if stage in {"I", "IA"} and not state.slots.manualized:
            return (
                "Clarify whether the intervention is manualized",
                "Is the intervention manualized yet — is there a written protocol or manual?",
            )
        return None


    def run(self, state: SessionState, user_message: str, context: str = "") -> AgentOutput:
        """
        Generate execution plan
        
        output：
        - plan_steps: List[PlanStep]
        - next_question: Optional[str]
        - final_response_outline: Optional[str]
        """
        llm_output = self._run_with_llm(state, user_message, context)
        if llm_output:
            return llm_output
        return self._run_with_rules(state)

    def _run_with_llm(self, state: SessionState, user_message: str, context: str = "") -> AgentOutput | None:
        if not llm_client.is_enabled():
            return None

        system_prompt = (
            " Planner Agent。output JSON，explain。\n"
            "Fields: plan_steps(list), next_question(str|null), final_response_outline(str|null)\n"
            "Each plan_steps item fields: step_type(ask_user/call_tool/draft_output/verify), "
            "tool_name(optional), tool_args_schema(optional), success_criteria(optional), description"
        )
        user_prompt = (
            f"User message: {user_message}\n"
            f"Current stage: {state.slots.stage}\n"
            f"Known slots: {state.slots.model_dump()}\n"
            f"Context: {context[:1200]}"
        )
        data = llm_client.chat_json(system_prompt=system_prompt, user_prompt=user_prompt)
        if not data:
            return None

        raw_steps = data.get("plan_steps", []) or []
        plan_steps = []
        for item in raw_steps:
            if not isinstance(item, dict):
                continue
            step_type = str(item.get("step_type", "")).strip()
            if step_type not in {t.value for t in PlanStepType}:
                continue
            plan_steps.append(
                PlanStep(
                    step_type=PlanStepType(step_type),
                    tool_name=item.get("tool_name"),
                    tool_args_schema=item.get("tool_args_schema"),
                    success_criteria=item.get("success_criteria"),
                    description=str(item.get("description", "")),
                )
            )

        next_question = data.get("next_question")
        final_response_outline = data.get("final_response_outline")

        tool_calls = []
        for step in plan_steps:
            if step.step_type == PlanStepType.CALL_TOOL and step.tool_name:
                tool_calls.append(
                    ToolCall(
                        tool_name=step.tool_name,
                        tool_args=step.tool_args_schema or {},
                        success_criteria=step.success_criteria,
                    )
                )

        return AgentOutput(
            decision={
                "plan_steps": [step.model_dump() for step in plan_steps],
                "next_question": next_question,
                "final_response_outline": final_response_outline,
            },
            confidence=0.82,
            analysis=f"LLM Generated  {len(plan_steps)}  plan steps",
            actions=tool_calls,
        )

    def _run_with_rules(self, state: SessionState) -> AgentOutput:
        plan_steps = []
        next_question = None
        final_response_outline = None
        
        current_stage = state.slots.stage

        # If stage is unknown, collect information first
        if not current_stage:
            plan_steps.append(PlanStep(
                step_type=PlanStepType.ASK_USER,
                description="Ask user for study details to identify stage"
            ))
            next_question = (
                "Which stage is your project in? To place it, tell me what the "
                "intervention is and what has already been tested."
            )
        else:
            # The per-stage ladder this replaced was hardcoded and had drifted
            # off the real taxonomy (it advertised Stage III as effectiveness).
            # Progression now comes from app/core/stage_model.NEXT_STAGE.
            gate = self._unmet_prerequisite(state, current_stage)
            if gate:
                description, question = gate
                plan_steps.append(PlanStep(
                    step_type=PlanStepType.ASK_USER,
                    description=description,
                ))
                next_question = question
            else:
                # The stage definitions themselves come from STAGE_MODEL
                # below, so the tool call is only for corpus documents.
                # (This used to call db_tool, which has been deleted — it
                # returned the same constants STAGE_MODEL already holds.)
                following = NEXT_STAGE.get(current_stage)
                plan_steps.append(PlanStep(
                    step_type=PlanStepType.CALL_TOOL,
                    tool_name="versioned_rag_tool",
                    tool_args_schema={
                        "query": (
                            f"Stage {following} requirements" if following
                            else f"Stage {current_stage} next step"
                        )
                    },
                    description=(
                        f"Retrieve Stage {following} guidance" if following
                        else "Retrieve related documents and guidance"
                    ),
                ))
                if following:
                    final_response_outline = (
                        f"Summarize what remains in {STAGE_MODEL[current_stage].label}, then "
                        f"introduce {STAGE_MODEL[following].title}: "
                        f"{STAGE_MODEL[following].summary}"
                    )
                else:
                    final_response_outline = (
                        f"Provide next-step guidance within "
                        f"{STAGE_MODEL[current_stage].title}"
                    )


        # Generate tool calls
        tool_calls = []
        for step in plan_steps:
            if step.step_type == PlanStepType.CALL_TOOL and step.tool_name:
                tool_calls.append(ToolCall(
                    tool_name=step.tool_name,
                    tool_args=step.tool_args_schema or {},
                    success_criteria=step.success_criteria
                ))
        
        return AgentOutput(
            decision={
                "plan_steps": [step.model_dump() for step in plan_steps],
                "next_question": next_question,
                "final_response_outline": final_response_outline
            },
            confidence=0.8,
            analysis=f"Generated  {len(plan_steps)}  plan steps",
            actions=tool_calls
        )
