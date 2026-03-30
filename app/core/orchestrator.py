"""Simplified implicit-intent orchestrator (LangGraph) for /chat only."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.base import BaseAgent
from app.agents.grant_partner_agent import GrantPartnerAgent
from app.agents.intent_agent import IntentAgent
from app.agents.mechanism_coach_agent import MechanismCoachAgent
from app.agents.measure_finder_agent import MeasureFinderAgent
from app.agents.planner_agent import PlannerAgent
from app.agents.rag_agent import RAGAgent
from app.agents.responder_agent import ResponderAgent
from app.agents.stage_agent import StageAgent
from app.agents.study_builder_agent import StudyBuilderAgent
from app.core.memory import memory_manager
from app.core.state_store import state_store
from app.core.types import AgentOutput, MessageRole, SessionState, ToolCall
from app.tools.base import ToolRegistry



class ChatGraphState(TypedDict, total=False):
    session_id: str
    user_message: str
    workflow_override: Optional[str]
    uploaded_context_text: Optional[str]
    state: SessionState
    context: str
    pending_tool_calls: List[ToolCall]
    called_agents: List[str]
    last_output: AgentOutput
    reply: str
    debug_info: Dict[str, Any]
    # routing signals from current-turn intent
    intent_need_stage: bool
    intent_query_type: str
    intent_label: str
    intent_confidence: float
    intent_is_definition: bool
    intent_workflow: str
    stage_result: Optional[str]
    stage_confidence: float
    # structured ReAct loop controls
    react_step: int
    max_react_steps: int
    react_last_planned_tools: int
    tool_results_count: int


class Orchestrator:
    """LangGraph execution engine with implicit intent-driven routing and structured ReAct loop."""

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.agents: Dict[str, BaseAgent] = {
            "intent_agent": IntentAgent(),
            "rag_agent": RAGAgent(),
            "stage_agent": StageAgent(),
            "planner_agent": PlannerAgent(),
            "mechanism_coach_agent": MechanismCoachAgent(),
            "study_builder_agent": StudyBuilderAgent(),
            "measure_finder_agent": MeasureFinderAgent(),
            "grant_partner_agent": GrantPartnerAgent(),
            "responder_agent": ResponderAgent(),
        }
        self.tool_registry = tool_registry or ToolRegistry()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ChatGraphState)

        graph.add_node("load_state", self._load_state)
        graph.add_node("intent", self._intent)
        graph.add_node("stage_reason", self._stage_reason)
        graph.add_node("planner", self._planner)
        graph.add_node("mechanism_coach", self._mechanism_coach)
        graph.add_node("study_builder", self._study_builder)
        graph.add_node("measure_finder", self._measure_finder)
        graph.add_node("grant_partner", self._grant_partner)
        graph.add_node("guardrails", self._guardrails)
        graph.add_node("rag_plan", self._rag_plan)
        graph.add_node("run_tools", self._run_tools)
        graph.add_node("react_judge", self._react_judge)
        graph.add_node("responder", self._responder)
        graph.add_node("finalize", self._finalize)

        graph.add_edge(START, "load_state")
        graph.add_edge("load_state", "intent")

        graph.add_conditional_edges(
            "intent",
            self._route_after_intent,
            {
                "rag_plan": "rag_plan",      # definition or general QA
                "stage_reason": "stage_reason",  # stage flow
            },
        )

        graph.add_conditional_edges(
            "stage_reason",
            self._route_after_stage,
            {
                "planner": "planner",
                "rag_plan": "rag_plan",
                "mechanism_coach": "mechanism_coach",
                "study_builder": "study_builder",
                "measure_finder": "measure_finder",
                "grant_partner": "grant_partner",
                "responder": "responder",
            },
        )

        graph.add_edge("planner", "rag_plan")
        graph.add_edge("mechanism_coach", "guardrails")
        graph.add_edge("study_builder", "guardrails")
        graph.add_edge("measure_finder", "guardrails")
        graph.add_edge("grant_partner", "guardrails")
        graph.add_edge("guardrails", "rag_plan")
        graph.add_edge("rag_plan", "run_tools")
        graph.add_edge("run_tools", "react_judge")
        graph.add_conditional_edges(
            "react_judge",
            self._route_after_react_judge,
            {
                "rag_plan": "rag_plan",
                "responder": "responder",
            },
        )
        graph.add_edge("responder", "finalize")
        graph.add_edge("finalize", END)

        return graph.compile()

    def _trace(self, gstate: ChatGraphState, step: Dict[str, Any]):
        debug = gstate.setdefault("debug_info", {})
        debug.setdefault("execution_trace", []).append(step)

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        """Robust bool parsing for LLM outputs (bool/int/str)."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y"}:
                return True
            if normalized in {"false", "0", "no", "n", "null", "none", ""}:
                return False
        return default

    def _add_agent(self, gstate: ChatGraphState, name: str, output: AgentOutput):
        called = list(gstate.get("called_agents", []))
        called.append(name)
        gstate["called_agents"] = called
        self._trace(
            gstate,
            {
                "kind": "agent",
                "name": name,
                "confidence": round(output.confidence, 4),
                "analysis": output.analysis,
                "decision_preview": {
                    k: output.decision.get(k)
                    for k in ["workflow", "intent_label", "query_type", "need_stage", "stage", "rag_invoked", "rag_strategy"]
                    if k in output.decision
                },
                "tool_actions": [a.tool_name for a in (output.actions or [])],
            },
        )

    def _load_state(self, gstate: ChatGraphState) -> ChatGraphState:
        session_id = gstate["session_id"]
        user_message = gstate["user_message"]
        uploaded_context_text = (gstate.get("uploaded_context_text") or "").strip()

        state = state_store.get_state(session_id)
        if not state:
            state = state_store.create_state(session_id)
        state.add_message(MessageRole.USER, user_message)

        # Persist uploaded context in session memory so subsequent turns can reuse it.
        existing_uploaded = str(state.slots.extracted_features.get("session_uploaded_context", "") or "")
        if uploaded_context_text:
            if uploaded_context_text not in existing_uploaded:
                if existing_uploaded:
                    existing_uploaded = f"{existing_uploaded}\n\n{uploaded_context_text}"
                else:
                    existing_uploaded = uploaded_context_text
            # Keep bounded session attachment context.
            existing_uploaded = existing_uploaded[:15000]
            state.slots.extracted_features["session_uploaded_context"] = existing_uploaded
            state.slots.extracted_features["session_uploaded_context_chars"] = len(existing_uploaded)

        reusable_uploaded = str(state.slots.extracted_features.get("session_uploaded_context", "") or "")
        effective_user_message = user_message
        if reusable_uploaded:
            effective_user_message = (
                f"{user_message}\n\n"
                "[Session uploaded context]\n"
                f"{reusable_uploaded[:4500]}"
            )

        return {
            **gstate,
            "state": state,
            "user_message": effective_user_message,
            "context": memory_manager.get_context_for_agent(state),
            "pending_tool_calls": [],
            "called_agents": [],
            "react_step": 0,
            "max_react_steps": 3,
            "react_last_planned_tools": 0,
            "tool_results_count": 0,
            "debug_info": {
                "execution_trace": [],
                "orchestration_engine": "langgraph",
                "session_uploaded_context_bound": bool(reusable_uploaded),
                "session_uploaded_context_chars": len(reusable_uploaded),
                "uploaded_context_added_this_turn": bool(uploaded_context_text),
            },
        }

    def _intent(self, gstate: ChatGraphState) -> ChatGraphState:
        state = gstate["state"]
        user_message = gstate["user_message"]
        context = gstate["context"]

        out = self.agents["intent_agent"].run(state, user_message, context)
        decision = out.decision or {}
        out.decision = decision

        self.agents["intent_agent"].update_state(state, out)
        self._add_agent(gstate, "intent_agent", out)

        need_stage = self._as_bool(decision.get("need_stage", False), default=True)
        query_type = str(decision.get("query_type", "general_qa")).lower()
        intent_label = str(decision.get("intent_label", "unknown")).lower()
        is_definition = self._as_bool(
            decision.get("is_definition_query", query_type == "definition"),
            default=(query_type == "definition"),
        )
        gstate["debug_info"]["intent_prelude"] = "passed"
        gstate["debug_info"]["intent_raw_workflow"] = str(decision.get("workflow", "navigator")).lower()
        gstate["debug_info"]["workflow_override"] = str(gstate.get("workflow_override") or "").strip().lower() or None

        return {
            **gstate,
            "state": state,
            "last_output": out,
            "intent_need_stage": need_stage,
            "intent_query_type": query_type,
            "intent_label": intent_label,
            "intent_confidence": float(out.confidence),
            "intent_is_definition": is_definition,
            "intent_workflow": str(decision.get("workflow", "navigator")).lower(),
        }

    def _route_after_intent(self, gstate: ChatGraphState) -> str:
        workflow = str(gstate.get("intent_workflow", "navigator")).lower()
        workflow_override = str(gstate.get("workflow_override") or "").strip().lower()
        if workflow_override == "auto":
            workflow_override = ""
        intent_label = str(gstate.get("intent_label", "unknown")).lower()
        intent_confidence = float(gstate.get("intent_confidence", 0.0) or 0.0)
        query_type = str(gstate.get("intent_query_type", "general_qa")).lower()
        is_definition = self._as_bool(gstate.get("intent_is_definition"), default=False)

        # Workflow card has priority for user-facing behavior.
        # Intent is still always executed first for extraction and tracing.
        if workflow_override in {"navigator", "mechanism_coach", "study_builder", "measure_finder", "grant_partner"}:
            if intent_label != "admin":
                workflow = workflow_override
                gstate["debug_info"]["workflow_resolution"] = "override_applied_user_priority"
            else:
                gstate["debug_info"]["workflow_resolution"] = "override_ignored_admin_only"
        else:
            gstate["debug_info"]["workflow_resolution"] = "intent_only"

        # Auto mode policy (intent-driven, no keyword forcing):
        # 1) stage-centric queries -> navigator(stage flow)
        # 2) specialized workflow when intent LLM predicts it with enough confidence
        # 3) otherwise fallback to direct RAG + normal response
        if workflow_override not in {"navigator", "mechanism_coach", "study_builder", "measure_finder", "grant_partner"}:
            if query_type in {"stage_classification", "stage_requirements", "next_step"} and not is_definition:
                workflow = "navigator"
                gstate["debug_info"]["workflow_correction"] = "auto_mode_stage_to_navigator"
            elif workflow in {"mechanism_coach", "study_builder", "measure_finder", "grant_partner"}:
                if intent_confidence >= 0.65 and intent_label not in {"unknown", "chit_chat", "admin"}:
                    gstate["debug_info"]["workflow_correction"] = "auto_mode_intent_specialized_workflow"
                else:
                    workflow = "rag_fallback"
                    gstate["debug_info"]["workflow_correction"] = "auto_mode_low_confidence_rag_fallback"

        # Persist the effective workflow so downstream stage routing uses the same decision.
        gstate["intent_workflow"] = workflow

        if workflow == "rag_fallback":
            gstate["debug_info"].update(
                {
                    "route_mode": "direct_reply",
                    "route_notes": "auto mode: no explicit workflow signal -> rag direct flow",
                    "workflow": "auto",
                }
            )
            return "rag_plan"

        if workflow in {"mechanism_coach", "study_builder", "measure_finder", "grant_partner"}:
            gstate["debug_info"].update(
                {
                    "route_mode": "stage_prelude",
                    "route_notes": f"intent -> stage prelude -> {workflow}",
                    "workflow": workflow,
                }
            )
            return "stage_reason"
        # Use intent.need_stage as the primary gate as requested.
        # If need_stage=True, force stage flow even when other fields are noisy.
        if self._as_bool(gstate.get("intent_need_stage"), default=False):
            gstate["debug_info"].update({"route_mode": "stage_flow", "route_notes": "intent -> stage flow", "workflow": workflow})
            return "stage_reason"
        if self._as_bool(gstate.get("intent_is_definition"), default=False):
            gstate["debug_info"].update(
                {"route_mode": "direct_reply", "route_notes": "intent -> definition direct/rag flow", "workflow": workflow}
            )
            return "rag_plan"
        gstate["debug_info"].update({"route_mode": "direct_reply", "route_notes": "intent -> direct/rag flow", "workflow": workflow})
        return "rag_plan"

    def _stage_reason(self, gstate: ChatGraphState) -> ChatGraphState:
        state = gstate["state"]
        user_message = gstate["user_message"]
        context = gstate["context"]

        out = self.agents["stage_agent"].run(state, user_message, context)
        self.agents["stage_agent"].update_state(state, out)

        # preserve extracted fields for responder
        state.slots.extracted_features["reasoning_summary"] = out.decision.get("reasoning_summary")
        state.slots.extracted_features["missing_info"] = out.decision.get("missing_info")
        state.slots.extracted_features["clarifying_question"] = out.decision.get("clarifying_question")

        self._add_agent(gstate, "stage_agent", out)
        return {
            **gstate,
            "state": state,
            "last_output": out,
            "stage_result": out.decision.get("stage"),
            "stage_confidence": float(out.confidence),
        }

    def _route_after_stage(self, gstate: ChatGraphState) -> str:
        workflow = str(gstate.get("intent_workflow", "navigator")).lower()
        must_clarify = (
            (self._as_bool(gstate.get("intent_need_stage"), default=False) or workflow in {"mechanism_coach", "study_builder", "measure_finder", "grant_partner"})
            and not self._as_bool(gstate.get("intent_is_definition"), default=False)
            and (
                gstate.get("stage_result") is None
                or float(gstate.get("stage_confidence", 0.0)) < 0.75
            )
        )
        if must_clarify:
            gstate["debug_info"].update(
                {
                    "route_mode": "stage_clarify",
                    "route_notes": "stage prelude unresolved -> clarify before workflow execution",
                    "workflow": workflow,
                }
            )
            return "responder"

        if workflow == "mechanism_coach":
            gstate["debug_info"].update(
                {
                    "route_mode": "workflow_mechanism_coach",
                    "route_notes": "stage ready -> mechanism coach workflow",
                    "workflow": workflow,
                }
            )
            return "mechanism_coach"
        if workflow == "study_builder":
            gstate["debug_info"].update(
                {
                    "route_mode": "workflow_study_builder",
                    "route_notes": "stage ready -> study builder workflow",
                    "workflow": workflow,
                }
            )
            return "study_builder"
        if workflow == "measure_finder":
            gstate["debug_info"].update(
                {
                    "route_mode": "workflow_measure_finder",
                    "route_notes": "stage ready -> measure finder workflow",
                    "workflow": workflow,
                }
            )
            return "measure_finder"
        if workflow == "grant_partner":
            gstate["debug_info"].update(
                {
                    "route_mode": "workflow_grant_partner",
                    "route_notes": "stage ready -> grant partner workflow",
                    "workflow": workflow,
                }
            )
            return "grant_partner"
        if gstate.get("intent_query_type") == "next_step":
            return "planner"
        return "rag_plan"

    def _planner(self, gstate: ChatGraphState) -> ChatGraphState:
        state = gstate["state"]
        user_message = gstate["user_message"]
        context = gstate["context"]

        out = self.agents["planner_agent"].run(state, user_message, context)
        self.agents["planner_agent"].update_state(state, out)
        state.slots.extracted_features["planner_outline"] = out.decision.get("final_response_outline")
        state.slots.extracted_features["next_question"] = out.decision.get("next_question")

        self._add_agent(gstate, "planner_agent", out)
        return {**gstate, "state": state, "last_output": out}

    def _mechanism_coach(self, gstate: ChatGraphState) -> ChatGraphState:
        state = gstate["state"]
        out = self.agents["mechanism_coach_agent"].run(state, gstate["user_message"], gstate["context"])
        self.agents["mechanism_coach_agent"].update_state(state, out)
        self._add_agent(gstate, "mechanism_coach_agent", out)
        return {**gstate, "state": state, "last_output": out}

    def _study_builder(self, gstate: ChatGraphState) -> ChatGraphState:
        state = gstate["state"]
        out = self.agents["study_builder_agent"].run(state, gstate["user_message"], gstate["context"])
        self.agents["study_builder_agent"].update_state(state, out)
        self._add_agent(gstate, "study_builder_agent", out)
        return {**gstate, "state": state, "last_output": out}

    def _measure_finder(self, gstate: ChatGraphState) -> ChatGraphState:
        state = gstate["state"]
        out = self.agents["measure_finder_agent"].run(state, gstate["user_message"], gstate["context"])
        self.agents["measure_finder_agent"].update_state(state, out)
        self._add_agent(gstate, "measure_finder_agent", out)
        return {**gstate, "state": state, "last_output": out}

    def _grant_partner(self, gstate: ChatGraphState) -> ChatGraphState:
        state = gstate["state"]
        out = self.agents["grant_partner_agent"].run(state, gstate["user_message"], gstate["context"])
        self.agents["grant_partner_agent"].update_state(state, out)
        self._add_agent(gstate, "grant_partner_agent", out)
        return {**gstate, "state": state, "last_output": out}

    def _guardrails(self, gstate: ChatGraphState) -> ChatGraphState:
        """Workflow-level guardrails before retrieval and final response composition."""
        state = gstate["state"]
        warnings = state.slots.extracted_features.get("guardrail_warnings", []) or []

        if (
            (
                self._as_bool(gstate.get("intent_need_stage"), default=False)
                or str(gstate.get("intent_workflow", "navigator")).lower() in {"mechanism_coach", "study_builder", "measure_finder", "grant_partner"}
            )
            and (
                gstate.get("stage_result") is None
                or float(gstate.get("stage_confidence", 0.0)) < 0.75
            )
        ):
            warnings.append("Low-confidence stage result: clarification-first mode enforced.")

        if state.slots.extracted_features.get("workflow") in {"mechanism_coach", "study_builder", "measure_finder", "grant_partner"}:
            warnings.append("Workflow output is educational guidance and should be validated by domain experts.")

        state.slots.extracted_features["guardrail_warnings"] = list(dict.fromkeys(warnings))
        self._trace(
            gstate,
            {
                "kind": "guardrail",
                "name": "workflow_guardrails",
                "warnings_count": len(state.slots.extracted_features["guardrail_warnings"]),
            },
        )
        return {**gstate, "state": state}

    def _rag_plan(self, gstate: ChatGraphState) -> ChatGraphState:
        state = gstate["state"]
        user_message = gstate["user_message"]
        context = gstate["context"]

        out = self.agents["rag_agent"].run(state, user_message, context)
        self.agents["rag_agent"].update_state(state, out)

        pending = list(gstate.get("pending_tool_calls", []))
        planned_now = 0
        if out.actions:
            pending.extend(out.actions)
            planned_now = len(out.actions)

        self._add_agent(gstate, "rag_agent", out)
        self._trace(
            gstate,
            {
                "kind": "react",
                "name": "plan",
                "step": gstate.get("react_step", 0) + 1,
                "planned_tools": planned_now,
                "analysis": "RAG planning generated tool actions",
            },
        )
        return {
            **gstate,
            "state": state,
            "last_output": out,
            "pending_tool_calls": pending,
            "react_last_planned_tools": planned_now,
        }

    def _run_tools(self, gstate: ChatGraphState) -> ChatGraphState:
        state = gstate["state"]
        pending = gstate.get("pending_tool_calls", [])
        count = 0

        for tool_call in pending:
            try:
                artifact = self.tool_registry.run_tool(tool_call.tool_name, tool_call.tool_args)
                state.artifacts.append(artifact)
                count += 1
                self._trace(
                    gstate,
                    {
                        "kind": "tool",
                        "name": tool_call.tool_name,
                        "success": artifact.metadata.get("success", True),
                        "sources": [c.source for c in artifact.citations[:3]],
                    },
                )
            except Exception as exc:
                gstate["debug_info"][f"tool_error_{tool_call.tool_name}"] = str(exc)
                self._trace(
                    gstate,
                    {"kind": "tool", "name": tool_call.tool_name, "success": False, "error": str(exc)},
                )

        gstate["debug_info"]["tools_called"] = len(state.artifacts)
        self._trace(
            gstate,
            {
                "kind": "react",
                "name": "observe",
                "step": gstate.get("react_step", 0) + 1,
                "executed_tools": len(pending),
                "successful_results": count,
                "analysis": "Tool observations stored as artifacts",
            },
        )
        return {**gstate, "state": state, "pending_tool_calls": [], "tool_results_count": count}

    def _react_judge(self, gstate: ChatGraphState) -> ChatGraphState:
        """Decide whether to continue ReAct loop or finalize response."""
        current_step = int(gstate.get("react_step", 0))
        next_step = current_step + 1
        max_steps = int(gstate.get("max_react_steps", 3))
        planned_tools = int(gstate.get("react_last_planned_tools", 0))
        successful_results = int(gstate.get("tool_results_count", 0))

        must_clarify = (
            (
                self._as_bool(gstate.get("intent_need_stage"), default=False)
                or str(gstate.get("intent_workflow", "navigator")).lower() in {"mechanism_coach", "study_builder", "measure_finder", "grant_partner"}
            )
            and not self._as_bool(gstate.get("intent_is_definition"), default=False)
            and (
                gstate.get("stage_result") is None
                or float(gstate.get("stage_confidence", 0.0)) < 0.75
            )
        )

        continue_loop = False
        judge_reason = "response_ready"
        if must_clarify:
            continue_loop = False
            judge_reason = "stage_uncertain_clarify_only"
        elif successful_results > 0:
            continue_loop = False
            judge_reason = "evidence_collected"
        elif planned_tools == 0:
            continue_loop = False
            judge_reason = "no_tool_needed"
        elif next_step >= max_steps:
            continue_loop = False
            judge_reason = "react_step_budget_reached"
        else:
            continue_loop = True
            judge_reason = "insufficient_observation_retry"

        self._trace(
            gstate,
            {
                "kind": "react",
                "name": "judge",
                "step": next_step,
                "continue_loop": continue_loop,
                "reason": judge_reason,
                "planned_tools": planned_tools,
                "successful_results": successful_results,
            },
        )
        gstate["debug_info"]["react"] = {
            "step": next_step,
            "max_steps": max_steps,
            "continue_loop": continue_loop,
            "judge_reason": judge_reason,
        }
        return {
            **gstate,
            "react_step": next_step,
            "react_continue": continue_loop,
            "react_judge_reason": judge_reason,
        }

    def _route_after_react_judge(self, gstate: ChatGraphState) -> str:
        if self._as_bool(gstate.get("react_continue"), default=False):
            return "rag_plan"
        return "responder"

    def _responder(self, gstate: ChatGraphState) -> ChatGraphState:
        state = gstate["state"]
        user_message = gstate["user_message"]
        context = gstate["context"]

        # Low stage confidence is passed to ResponderAgent; the LLM decides tone and follow-ups.
        if (
            self._as_bool(gstate.get("intent_need_stage"), default=False)
            and not self._as_bool(gstate.get("intent_is_definition"), default=False)
            and (
                gstate.get("stage_result") is None
                or float(gstate.get("stage_confidence", 0.0)) < 0.75
            )
        ):
            gstate.setdefault("debug_info", {})["stage_uncertain_hint"] = True
            self._trace(
                gstate,
                {
                    "kind": "note",
                    "name": "stage_uncertain",
                    "stage": gstate.get("stage_result"),
                    "stage_confidence": gstate.get("stage_confidence", 0.0),
                },
            )

        if gstate.get("debug_info", {}).get("stage_uncertain_hint"):
            state.slots.extracted_features["stage_uncertain_hint"] = True
        else:
            state.slots.extracted_features.pop("stage_uncertain_hint", None)

        out = self.agents["responder_agent"].run(state, user_message, context)
        self.agents["responder_agent"].update_state(state, out)
        self._add_agent(gstate, "responder_agent", out)

        workflow = state.slots.extracted_features.get("workflow", "navigator")
        workflow_summary = state.slots.extracted_features.get("workflow_summary", "")
        workflow_structured = state.slots.extracted_features.get("workflow_structured_output", {})
        guardrail_warnings = state.slots.extracted_features.get("guardrail_warnings", []) or []

        if workflow in {"mechanism_coach", "study_builder", "measure_finder", "grant_partner"} and workflow_structured:
            responder_summary = (out.user_facing or "").strip()
            pretty_structured = json.dumps(workflow_structured, ensure_ascii=False, indent=2)

            reply_parts: list[str] = []
            if responder_summary:
                reply_parts.append(responder_summary)
            else:
                intent_payload = state.slots.extracted_features.get("intent_payload", {}) or {}
                stage_value = state.slots.stage or gstate.get("stage_result")
                stage_conf = float(state.slots.stage_confidence or gstate.get("stage_confidence", 0.0) or 0.0)
                reasoning_summary = state.slots.extracted_features.get("reasoning_summary") or ""
                reply_parts.append(
                    f"**{workflow}** — stage **{stage_value}** (confidence {stage_conf:.2f}). "
                    f"{reasoning_summary or workflow_summary or 'See structured output below.'}"
                )

            # reply_parts.extend(["", "---", f"**Structured output** (`{workflow}`):", "```json", pretty_structured, "```"])
            if guardrail_warnings:
                reply_parts.append("")
                reply_parts.append("**Note:** " + " ".join(str(w) for w in guardrail_warnings[:3]))
            reply = "\n".join(reply_parts)
            gstate["debug_info"].update({"route_mode": f"{workflow}_answer"})
            return {**gstate, "state": state, "last_output": out, "reply": reply}

        reply = out.user_facing if out and out.user_facing else "I understand your question. Let me help you with that."
        if self._as_bool(gstate.get("intent_need_stage"), default=False):
            gstate["debug_info"].update({"route_mode": "stage_answer"})
        return {**gstate, "state": state, "last_output": out, "reply": reply}

    def _finalize(self, gstate: ChatGraphState) -> ChatGraphState:
        state = gstate["state"]
        reply = gstate["reply"]
        state.add_message(MessageRole.ASSISTANT, reply)

        if memory_manager.should_summarize(state):
            summary = memory_manager.create_summary(state)
            memory_manager.update_summary(state, summary)

        state_store.save_state(state)

        gstate["debug_info"].update(
            {
                "stage": state.slots.stage,
                "stage_confidence": state.slots.stage_confidence,
                "need_stage": state.slots.need_stage,
                "workflow": state.slots.extracted_features.get("workflow", "navigator"),
                "workflow_structured_output": state.slots.extracted_features.get("workflow_structured_output", {}),
                "guardrail_warnings": state.slots.extracted_features.get("guardrail_warnings", []),
                "agents_called": gstate.get("called_agents", []),
            }
        )
        return {**gstate, "state": state}

    def process_message(
        self,
        session_id: str,
        user_message: str,
        workflow_override: Optional[str] = None,
        uploaded_context_text: Optional[str] = None,
    ) -> tuple[str, dict]:
        result = self._graph.invoke(
            {
                "session_id": session_id,
                "user_message": user_message,
                "workflow_override": workflow_override,
                "uploaded_context_text": uploaded_context_text,
            }
        )
        return result.get("reply", "I understand your question. Let me help you with that."), result.get("debug_info", {})