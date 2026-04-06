"""Simplified implicit-intent orchestrator (LangGraph) for /chat only."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.base import BaseAgent
from app.agents.intent_agent import IntentAgent
from app.agents.rag_agent import RAGAgent
from app.agents.responder_agent import ResponderAgent
from app.agents.stage_agent import StageAgent
from app.core.memory import memory_manager
from app.core.state_store import state_store
from app.core.types import AgentOutput, MessageRole, SessionState


class ChatGraphState(TypedDict, total=False):
    session_id: str
    user_message: str
    workflow_override: Optional[str]
    uploaded_context_text: Optional[str]
    state: SessionState
    context: str
    called_agents: List[str]
    last_output: AgentOutput
    reply: str
    debug_info: Dict[str, Any]
    intent_need_stage: bool
    intent_query_type: str
    intent_label: str
    intent_confidence: float
    intent_is_definition: bool
    intent_workflow: str
    stage_result: Optional[str]
    stage_confidence: float


class Orchestrator:
    """LangGraph: load_state → intent → stage → RAG agent (internal retrieval only) → responder → finalize."""

    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {
            "intent_agent": IntentAgent(),
            "rag_agent": RAGAgent(),
            "stage_agent": StageAgent(),
            "responder_agent": ResponderAgent(),
        }
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ChatGraphState)
        graph.add_node("load_state", self._load_state)
        graph.add_node("intent", self._intent)
        graph.add_node("stage_reason", self._stage_reason)
        graph.add_node("rag", self._rag)
        graph.add_node("responder", self._responder)
        graph.add_node("finalize", self._finalize)

        graph.add_edge(START, "load_state")
        graph.add_edge("load_state", "intent")
        graph.add_edge("intent", "stage_reason")
        graph.add_edge("stage_reason", "rag")
        graph.add_edge("rag", "responder")
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
                    for k in [
                        "workflow",
                        "intent_label",
                        "query_type",
                        "need_stage",
                        "stage",
                        "rag_invoked",
                        "rag_strategy",
                        "strategy",
                    ]
                    if k in output.decision
                },
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

        existing_uploaded = str(state.slots.extracted_features.get("session_uploaded_context", "") or "")
        if uploaded_context_text:
            if uploaded_context_text not in existing_uploaded:
                if existing_uploaded:
                    existing_uploaded = f"{existing_uploaded}\n\n{uploaded_context_text}"
                else:
                    existing_uploaded = uploaded_context_text
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
            "called_agents": [],
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

    def _stage_reason(self, gstate: ChatGraphState) -> ChatGraphState:
        state = gstate["state"]
        user_message = gstate["user_message"]
        context = gstate["context"]

        out = self.agents["stage_agent"].run(state, user_message, context)
        self.agents["stage_agent"].update_state(state, out)

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

    def _rag(self, gstate: ChatGraphState) -> ChatGraphState:
        state = gstate["state"]
        user_message = gstate["user_message"]

        out = self.agents["rag_agent"].run(state, user_message, gstate["context"])
        self.agents["rag_agent"].update_state(state, out)
        self._add_agent(gstate, "rag_agent", out)

        n_docs = len((out.metadata or {}).get("retrieved_docs") or [])
        self._trace(
            gstate,
            {
                "kind": "rag",
                "name": "rag_agent",
                "results_found": out.decision.get("results_found", n_docs),
                "analysis": "Retrieval handled inside RAGAgent (no ToolRegistry)",
            },
        )

        gstate["debug_info"]["rag_results_count"] = n_docs
        return {**gstate, "state": state, "last_output": out}

    def _responder(self, gstate: ChatGraphState) -> ChatGraphState:
        state = gstate["state"]
        user_message = gstate["user_message"]
        context = gstate["context"]

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
