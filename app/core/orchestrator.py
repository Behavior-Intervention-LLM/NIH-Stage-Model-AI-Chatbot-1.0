"""Simplified implicit-intent orchestrator (LangGraph) for /chat only."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app import feedback
from app.agents.base import BaseAgent
# from app.agents.grant_partner_agent import GrantPartnerAgent
from app.agents.intent_agent import IntentAgent
# from app.agents.mechanism_coach_agent import MechanismCoachAgent
# from app.agents.measure_finder_agent import MeasureFinderAgent
# from app.agents.planner_agent import PlannerAgent
from app.agents.rag_agent import RAGAgent
from app.agents.responder_agent import ResponderAgent
from app.agents.stage_agent import StageAgent
# from app.agents.study_builder_agent import StudyBuilderAgent
from app.core.memory import memory_manager
from app.core.state_store import state_store
from app.config import settings
from app.core.types import AgentOutput, Citation, MessageRole, SessionState, ToolCall
from app.tools.base import ToolRegistry



# Most important class
class ChatGraphState(TypedDict, total=False):
    session_id: str
    user_message: str
    workflow_override: Optional[str]
    # Files attached to this turn: [{"name": ..., "text": ...}]
    attachments: List[Dict[str, str]]
    # Optional callable(str) that receives responder text incrementally.
    stream_handler: Optional[Any]
    state: SessionState
    context: str
    pending_tool_calls: List[ToolCall] # At the moment maybe not necessary
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

    react_last_planned_tools: int
    tool_results_count: int

    # Retrieval loop: which attempt we are on, what has already been searched,
    # and the verdict on what came back. Drives the rag_plan retry edge.
    rag_attempt: int
    rag_queries_tried: List[str]
    rag_assessment: Dict[str, Any]

    # Retrieval observed on THIS turn, accumulated across retrieval attempts.
    turn_citations: List[Citation]
    turn_sources: List[Dict[str, Any]]
    turn_tool_errors: int



class Orchestrator:
    """LangGraph fixed path: load_state → intent → stage → RAG (plan + run tools) → responder → finalize."""

    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        self.agents: Dict[str, BaseAgent] = {
            "intent_agent": IntentAgent(), #identification of whether BIH related or not
            "stage_agent": StageAgent(),
            "rag_agent": RAGAgent(),
            # "planner_agent": PlannerAgent(),
            # "mechanism_coach_agent": MechanismCoachAgent(),
            # "study_builder_agent": StudyBuilderAgent(),
            # "measure_finder_agent": MeasureFinderAgent(),
            # "grant_partner_agent": GrantPartnerAgent(),
            "responder_agent": ResponderAgent(),
        }
        self.tool_registry = tool_registry or ToolRegistry()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ChatGraphState)

        graph.add_node("load_state", self._load_state)
        graph.add_node("intent", self._intent)
        graph.add_node("stage_reason", self._stage_reason)
        # graph.add_node("planner", self._planner)
        # graph.add_node("mechanism_coach", self._mechanism_coach)
        # graph.add_node("study_builder", self._study_builder)
        # graph.add_node("measure_finder", self._measure_finder)
        # graph.add_node("grant_partner", self._grant_partner)
        # graph.add_node("guardrails", self._guardrails)
        graph.add_node("rag_plan", self._rag_plan)
        graph.add_node("run_tools", self._run_tools)
        graph.add_node("assess_evidence", self._assess_evidence)
        graph.add_node("responder", self._responder)
        graph.add_node("finalize", self._finalize)

        graph.add_edge(START, "load_state")
        graph.add_edge("load_state", "intent")
        graph.add_conditional_edges(
            "intent",
            self._route_after_intent,
            {
                "stage_reason": "stage_reason",
                "rag_plan": "rag_plan",
            },
        )
        graph.add_edge("stage_reason", "rag_plan")

        # graph.add_edge("planner", "rag_plan")
        # graph.add_edge("mechanism_coach", "guardrails")
        # graph.add_edge("study_builder", "guardrails")
        # graph.add_edge("measure_finder", "guardrails")
        # graph.add_edge("grant_partner", "guardrails")
        # graph.add_edge("guardrails", "rag_plan")

        # Retrieval loop: plan → retrieve → judge the evidence → retry with a
        # reformulated query, or hand what we have to the responder.
        graph.add_edge("rag_plan", "run_tools")
        graph.add_edge("run_tools", "assess_evidence")
        graph.add_conditional_edges(
            "assess_evidence",
            self._route_after_assess,
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
        incoming_attachments = gstate.get("attachments") or []

        state = state_store.get_state(session_id)
        if not state:
            state = state_store.create_state(session_id)
        state.add_message(MessageRole.USER, user_message)

        # Artifacts are this turn's tool observations, not a session log. They
        # used to accumulate, so the responder was handed passages retrieved
        # for earlier questions as evidence for the current one. Conversation
        # continuity lives in messages/summary, which is where it belongs.
        state.artifacts.clear()

        # Attachments persist for the session so follow-up questions can refer
        # back to them. They are held on the state object, NOT appended to the
        # user message: prepending meant the document was re-sent to the intent
        # and stage classifiers on every later turn, and appeared twice in the
        # responder prompt.
        added_now = 0
        for item in incoming_attachments:
            if state.add_attachment(
                str(item.get("name") or "attached document"), str(item.get("text") or "")
            ):
                added_now += 1

        # Agents that route rather than answer get told an attachment exists,
        # not what is in it — enough for "summarise this" to classify
        # correctly, without the document riding along on every call.
        context = memory_manager.get_context_for_agent(state)
        if state.attachments:
            listing = ", ".join(f"{a.name} ({a.chars} chars)" for a in state.attachments)
            context = f"{context}\n\nAttached to this conversation: {listing}"

        return {
            **gstate,
            "state": state,
            "user_message": user_message,
            "context": context,
            "pending_tool_calls": [],
            "called_agents": [],
            "react_last_planned_tools": 0,
            "tool_results_count": 0,
            "rag_attempt": 0,
            "rag_queries_tried": [],
            "rag_assessment": {},
            "turn_citations": [],
            "turn_sources": [],
            "turn_tool_errors": 0,
            "debug_info": {
                "execution_trace": [],
                "orchestration_engine": "langgraph",
                "attachments": [
                    {"name": a.name, "chars": a.chars} for a in state.attachments
                ],
                "attachments_added_this_turn": added_now,
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
        """Skip stage agent when stage classification is not needed."""
        need_stage = self._as_bool(gstate.get("intent_need_stage"), default=False)
        is_definition = self._as_bool(gstate.get("intent_is_definition"), default=False)
        intent_label = str(gstate.get("intent_label", "unknown")).lower()

        skip_stage = (
            is_definition
            or intent_label in {"chit_chat", "admin"}
            or not need_stage
        )
        route = "rag_plan" if skip_stage else "stage_reason"
        gstate["debug_info"]["stage_skipped"] = skip_stage
        gstate["debug_info"]["route_after_intent"] = route
        return route

    def _stage_reason(self, gstate: ChatGraphState) -> ChatGraphState:
        state = gstate["state"]
        user_message = gstate["user_message"]
        # Stage agent only uses the first 1200 chars of context — no need to pass more.
        context = gstate["context"][:1200]

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

    # Legacy: conditional_edges(stage_reason); replaced with a direct stage_reason -> rag_plan edge.
    # def _route_after_stage(self, gstate: ChatGraphState) -> str:
    #     gstate["debug_info"].update(
    #         {
    #             "route_mode": "stage_to_rag",
    #             "route_notes": "stage_reason -> rag_plan (fixed)",
    #             "workflow": "navigator",
    #         }
    #     )
    #     return "rag_plan"

    # def _planner(self, gstate: ChatGraphState) -> ChatGraphState:
    #     state = gstate["state"]
    #     user_message = gstate["user_message"]
    #     context = gstate["context"]

    #     out = self.agents["planner_agent"].run(state, user_message, context)
    #     self.agents["planner_agent"].update_state(state, out)
    #     state.slots.extracted_features["planner_outline"] = out.decision.get("final_response_outline")
    #     state.slots.extracted_features["next_question"] = out.decision.get("next_question")

    #     self._add_agent(gstate, "planner_agent", out)
    #     return {**gstate, "state": state, "last_output": out}

    # def _mechanism_coach(self, gstate: ChatGraphState) -> ChatGraphState:
    #     state = gstate["state"]
    #     out = self.agents["mechanism_coach_agent"].run(state, gstate["user_message"], gstate["context"])
    #     self.agents["mechanism_coach_agent"].update_state(state, out)
    #     self._add_agent(gstate, "mechanism_coach_agent", out)
    #     return {**gstate, "state": state, "last_output": out}

    # def _study_builder(self, gstate: ChatGraphState) -> ChatGraphState:
    #     state = gstate["state"]
    #     out = self.agents["study_builder_agent"].run(state, gstate["user_message"], gstate["context"])
    #     self.agents["study_builder_agent"].update_state(state, out)
    #     self._add_agent(gstate, "study_builder_agent", out)
    #     return {**gstate, "state": state, "last_output": out}

    # def _measure_finder(self, gstate: ChatGraphState) -> ChatGraphState:
    #     state = gstate["state"]
    #     out = self.agents["measure_finder_agent"].run(state, gstate["user_message"], gstate["context"])
    #     self.agents["measure_finder_agent"].update_state(state, out)
    #     self._add_agent(gstate, "measure_finder_agent", out)
    #     return {**gstate, "state": state, "last_output": out}

    # def _grant_partner(self, gstate: ChatGraphState) -> ChatGraphState:
    #     state = gstate["state"]
    #     out = self.agents["grant_partner_agent"].run(state, gstate["user_message"], gstate["context"])
    #     self.agents["grant_partner_agent"].update_state(state, out)
    #     self._add_agent(gstate, "grant_partner_agent", out)
    #     return {**gstate, "state": state, "last_output": out}

    # Not wired into the main graph; full implementation kept in comments below.
    # def _guardrails(self, gstate: ChatGraphState) -> ChatGraphState:
    #     """Workflow-level guardrails before retrieval and final response composition."""
    #     state = gstate["state"]
    #     warnings = state.slots.extracted_features.get("guardrail_warnings", []) or []
    #
    #     if (
    #         (
    #             self._as_bool(gstate.get("intent_need_stage"), default=False)
    #             or str(gstate.get("intent_workflow", "navigator")).lower()
    #             in {"mechanism_coach", "study_builder", "measure_finder", "grant_partner"}
    #         )
    #         and (
    #             gstate.get("stage_result") is None
    #             or float(gstate.get("stage_confidence", 0.0)) < 0.75
    #         )
    #     ):
    #         warnings.append("Low-confidence stage result: clarification-first mode enforced.")
    #
    #     if state.slots.extracted_features.get("workflow") in {
    #         "mechanism_coach",
    #         "study_builder",
    #         "measure_finder",
    #         "grant_partner",
    #     }:
    #         warnings.append(
    #             "Workflow output is educational guidance and should be validated by domain experts."
    #         )
    #
    #     state.slots.extracted_features["guardrail_warnings"] = list(dict.fromkeys(warnings))
    #     self._trace(
    #         gstate,
    #         {
    #             "kind": "guardrail",
    #             "name": "workflow_guardrails",
    #             "warnings_count": len(state.slots.extracted_features["guardrail_warnings"]),
    #         },
    #     )
    #     return {**gstate, "state": state}

    def _rag_plan(self, gstate: ChatGraphState) -> ChatGraphState:
        """Plan this attempt's retrieval. Re-entered on a retry, with the
        previous attempt's queries and verdict available to the agent."""
        state = gstate["state"]
        user_message = gstate["user_message"]

        attempt = int(gstate.get("rag_attempt", 0))
        queries_tried = list(gstate.get("rag_queries_tried", []))

        out = self.agents["rag_agent"].plan(
            state,
            user_message,
            attempt=attempt,
            previous_queries=queries_tried,
            assessment=gstate.get("rag_assessment") or {},
        )
        self.agents["rag_agent"].update_state(state, out)

        planned = list(out.actions or [])
        for call in planned:
            query = str(call.tool_args.get("query", "")).strip()
            if query:
                queries_tried.append(query)

        self._add_agent(gstate, "rag_agent", out)
        self._trace(
            gstate,
            {
                "kind": "react",
                "name": "plan",
                "step": attempt + 1,
                "planned_tools": len(planned),
                "strategy": out.decision.get("rag_strategy"),
                "queries": out.decision.get("queries", []),
                "analysis": out.analysis,
            },
        )

        return {
            **gstate,
            "state": state,
            "last_output": out,
            "pending_tool_calls": planned,
            "rag_queries_tried": queries_tried,
            "react_last_planned_tools": len(planned),
        }

    def _run_tools(self, gstate: ChatGraphState) -> ChatGraphState:
        """Execute the planned tool calls, accumulating this turn's evidence
        across retrieval attempts."""
        state = gstate["state"]
        pending = list(gstate.get("pending_tool_calls", []))

        citations: List[Citation] = list(gstate.get("turn_citations", []))
        turn_sources: List[Dict[str, Any]] = list(gstate.get("turn_sources", []))
        tool_errors = int(gstate.get("turn_tool_errors", 0))
        count = 0

        for tool_call in pending:
            try:
                artifact = self.tool_registry.run_tool(tool_call.tool_name, tool_call.tool_args)
                state.artifacts.append(artifact)
                count += 1
                citations.extend(artifact.citations)
                for citation in artifact.citations:
                    turn_sources.append(
                        {
                            "source": citation.source,
                            "score": citation.relevance_score,
                            "passage": citation.passage,
                        }
                    )
                self._trace(
                    gstate,
                    {
                        "kind": "tool",
                        "name": tool_call.tool_name,
                        "query": tool_call.tool_args.get("query"),
                        "success": artifact.metadata.get("success", True),
                        "citations": len(artifact.citations),
                        "sources": [c.source for c in artifact.citations[:3]],
                    },
                )
            except Exception as exc:
                tool_errors += 1
                gstate["debug_info"][f"tool_error_{tool_call.tool_name}"] = str(exc)
                self._trace(
                    gstate,
                    {"kind": "tool", "name": tool_call.tool_name, "success": False, "error": str(exc)},
                )

        # A retry usually re-surfaces some of the same passages. Deduplicate so
        # per-source credit in the feedback loop is not inflated by the number
        # of attempts it took to find the passage.
        deduped: Dict[tuple, Dict[str, Any]] = {}
        for entry in turn_sources:
            key = (entry.get("source"), str(entry.get("passage", ""))[:120])
            best = deduped.get(key)
            if best is None or float(entry.get("score") or 0.0) > float(best.get("score") or 0.0):
                deduped[key] = entry
        turn_sources = sorted(
            deduped.values(), key=lambda s: float(s.get("score") or 0.0), reverse=True
        )

        gstate["debug_info"]["tools_called"] = len(state.artifacts)
        self._trace(
            gstate,
            {
                "kind": "react",
                "name": "observe",
                "step": int(gstate.get("rag_attempt", 0)) + 1,
                "executed_tools": len(pending),
                "successful_results": count,
                "analysis": "Tool observations stored as artifacts",
            },
        )
        return {
            **gstate,
            "state": state,
            "pending_tool_calls": [],
            "tool_results_count": count,
            "turn_citations": citations,
            "turn_sources": turn_sources,
            "turn_tool_errors": tool_errors,
        }

    def _assess_evidence(self, gstate: ChatGraphState) -> ChatGraphState:
        """Judge whether retrieval produced anything worth answering from.

        This is the turn's only self-correction point: weak evidence sends the
        graph back to rag_plan with a rewritten query instead of letting the
        responder improvise over near-random passages.
        """
        state = gstate["state"]
        citations: List[Citation] = list(gstate.get("turn_citations", []))
        attempts_done = int(gstate.get("rag_attempt", 0)) + 1
        planned_this_attempt = int(gstate.get("react_last_planned_tools", 0))
        max_attempts = max(1, settings.RAG_MAX_ATTEMPTS)

        assessment = RAGAgent.assess_evidence(citations)
        assessment["attempts"] = attempts_done
        assessment["queries_tried"] = list(gstate.get("rag_queries_tried", []))
        assessment["has_attachments"] = bool(state.attachments)

        # Nothing was planned (intent skips retrieval, or reformulation had
        # nothing new to try), so there is no failure here to retry.
        retrieval_ran = planned_this_attempt > 0
        retry = (
            retrieval_ran
            and not assessment["sufficient"]
            and attempts_done < max_attempts
            # A question about an attached document ("summarise my protocol")
            # will legitimately miss the corpus. Reformulating the query cannot
            # fix that, and the answer already has grounding to work from, so
            # spending another LLM rewrite plus retrieval on it is pure waste.
            and not state.attachments
        )
        assessment["retrying"] = retry
        if not retrieval_ran and not citations:
            assessment["reason"] = "retrieval_not_attempted"

        self._trace(
            gstate,
            {
                "kind": "react",
                "name": "assess",
                "step": attempts_done,
                "sufficient": assessment["sufficient"],
                "reason": assessment["reason"],
                "best_score": assessment["best_score"],
                "threshold": assessment["threshold"],
                "retrying": retry,
            },
        )

        if retry:
            return {**gstate, "state": state, "rag_assessment": assessment, "rag_attempt": attempts_done}

        # Final verdict for this turn: keep only artifacts that carry at least
        # one passage above the relevance floor. Everything else — including a
        # tool's "no matches found" text — would otherwise reach the responder
        # as if it were evidence.
        threshold = assessment["threshold"]
        if assessment["sufficient"]:
            state.artifacts = [
                artifact
                for artifact in state.artifacts
                if any(RAGAgent.semantic_score(c) >= threshold for c in artifact.citations)
            ]
        else:
            state.artifacts = []

        state.slots.extracted_features["evidence_assessment"] = assessment
        gstate["debug_info"]["retrieval"] = assessment

        return {
            **gstate,
            "state": state,
            "rag_assessment": assessment,
            "rag_attempt": attempts_done,
        }

    def _route_after_assess(self, gstate: ChatGraphState) -> str:
        assessment = gstate.get("rag_assessment") or {}
        return "rag_plan" if assessment.get("retrying") else "responder"

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

        stream_handler = gstate.get("stream_handler")
        responder = self.agents["responder_agent"]
        if stream_handler is not None and hasattr(responder, "run_stream"):
            out = responder.run_stream(state, user_message, context, stream_handler)
        else:
            out = responder.run(state, user_message, context)
        responder.update_state(state, out)
        self._add_agent(gstate, "responder_agent", out)

        # Legacy four-workflow reply assembly (needs json and workflow_structured_output; off in simplified pipeline).
        # workflow = state.slots.extracted_features.get("workflow", "navigator")
        # workflow_summary = state.slots.extracted_features.get("workflow_summary", "")
        # workflow_structured = state.slots.extracted_features.get("workflow_structured_output", {})
        # guardrail_warnings = state.slots.extracted_features.get("guardrail_warnings", []) or []
        # if workflow in {"mechanism_coach", "study_builder", "measure_finder", "grant_partner"} and workflow_structured:
        #     import json
        #
        #     responder_summary = (out.user_facing or "").strip()
        #     pretty_structured = json.dumps(workflow_structured, ensure_ascii=False, indent=2)
        #     reply_parts: list[str] = []
        #     if responder_summary:
        #         reply_parts.append(responder_summary)
        #     else:
        #         intent_payload = state.slots.extracted_features.get("intent_payload", {}) or {}
        #         stage_value = state.slots.stage or gstate.get("stage_result")
        #         stage_conf = float(state.slots.stage_confidence or gstate.get("stage_confidence", 0.0) or 0.0)
        #         reasoning_summary = state.slots.extracted_features.get("reasoning_summary") or ""
        #         reply_parts.append(
        #             f"**{workflow}** — stage **{stage_value}** (confidence {stage_conf:.2f}). "
        #             f"{reasoning_summary or workflow_summary or 'See structured output below.'}"
        #         )
        #     if guardrail_warnings:
        #         reply_parts.append("")
        #         reply_parts.append("**Note:** " + " ".join(str(w) for w in guardrail_warnings[:3]))
        #     reply = "\n".join(reply_parts)
        #     gstate["debug_info"].update({"route_mode": f"{workflow}_answer"})
        #     return {**gstate, "state": state, "last_output": out, "reply": reply}

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
                "intent_label": gstate.get("intent_label"),
                "intent_query_type": gstate.get("intent_query_type"),
                "intent_confidence": gstate.get("intent_confidence", 0.0),
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
        uploaded_context_name: Optional[str] = None,
        attachments: Optional[List[Dict[str, str]]] = None,
        stream_handler: Optional[Any] = None,
        username: str = "anonymous",
    ) -> tuple[str, dict]:
        """Run the graph. If stream_handler is given, it receives the final
        response text incrementally (callable taking a str chunk)."""
        started = time.perf_counter()

        # Two ways in: a list of named files (the chat UI), or a single blob of
        # pre-extracted text (ChatRequest.document_text, for API callers who
        # did their own extraction). Normalise to one list.
        incoming = [dict(a) for a in (attachments or []) if (a.get("text") or "").strip()]
        if uploaded_context_text and uploaded_context_text.strip():
            incoming.append(
                {
                    "name": uploaded_context_name or "attached document",
                    "text": uploaded_context_text,
                }
            )

        result = self._graph.invoke(
            {
                "session_id": session_id,
                "user_message": user_message,
                "workflow_override": workflow_override,
                "attachments": incoming,
                "stream_handler": stream_handler,
            }
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        reply = result.get("reply", "I understand your question. Let me help you with that.")
        debug_info = result.get("debug_info", {})

        # Implicit feedback observation. Recorded here rather than in _finalize
        # because this is the only place with the caller's identity, the wall
        # clock, and the user's message before _load_state appends uploaded
        # context to it. Returns immediately; all work happens off-thread.
        turn_uid = feedback.observe_turn(
            session_id=session_id,
            username=username,
            user_message=user_message,
            reply=reply,
            debug_info=debug_info,
            sources=result.get("turn_sources") or [],
            tool_errors=int(result.get("turn_tool_errors") or 0),
            latency_ms=latency_ms,
        )

        debug_info["latency_ms"] = latency_ms
        # Identifies this turn for an explicit rating. None when feedback is
        # disabled, in which case the client shows no rating control.
        debug_info["turn_uid"] = turn_uid
        return reply, debug_info