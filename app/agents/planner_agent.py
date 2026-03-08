"""
Planner Agent： stage +  + ，next step
"""
from app.agents.base import BaseAgent
from app.core.types import SessionState, AgentOutput, PlanStep, PlanStepType, ToolCall
from app.core.llm import llm_client


class PlannerAgent(BaseAgent):
    """Planning agent"""
    
    def __init__(self):
        super().__init__("PlannerAgent")
    
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
            next_question = "stage？：intervention？？"
        
        # If stage is known, generate stage-based plan
        elif current_stage == "0":
            # Stage 0: intervention
            if not state.slots.intervention_defined:
                plan_steps.append(PlanStep(
                    step_type=PlanStepType.ASK_USER,
                    description="intervention"
                ))
                next_question = "intervention？"
            else:
                plan_steps.append(PlanStep(
                    step_type=PlanStepType.CALL_TOOL,
                    tool_name="db_tool",
                    tool_args_schema={"query": "Stage I requirements"},
                    description="Retrieve Stage I guidance"
                ))
                final_response_outline = "introduce Stage I requirements：manualization"
        
        elif current_stage == "I":
            # Stage I: mechanism
            if not state.slots.manualized:
                plan_steps.append(PlanStep(
                    step_type=PlanStepType.ASK_USER,
                    description="manualization"
                ))
                next_question = "interventionmanualization（manualized）？"
            else:
                plan_steps.append(PlanStep(
                    step_type=PlanStepType.CALL_TOOL,
                    tool_name="db_tool",
                    tool_args_schema={"query": "Stage II requirements"},
                    description="Retrieve Stage II guidance"
                ))
                final_response_outline = "introduce Stage II requirements：mechanismefficacy"
        
        elif current_stage == "II":
            # Stage II: efficacy
            plan_steps.append(PlanStep(
                step_type=PlanStepType.CALL_TOOL,
                tool_name="db_tool",
                tool_args_schema={"query": "Stage III requirements"},
                description="Retrieve Stage III guidance"
            ))
            final_response_outline = "introduce Stage III requirements：effectivenessreal-world testing"
        
        elif current_stage in ["III", "IV", "V"]:
            #  stage，next steprecommendation
            plan_steps.append(PlanStep(
                step_type=PlanStepType.CALL_TOOL,
                tool_name="vector_tool",
                tool_args_schema={"query": f"Stage {current_stage} next step"},
                description="Retrieve related documents and guidance"
            ))
            final_response_outline = f"Provide Stage {current_stage} next steprecommendation"
        
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
