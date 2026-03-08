"""
RAG Agent:
- decides when to invoke retrieval
- requests version-aware retrieval for downstream responder grounding
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.core.types import AgentOutput, SessionState, ToolCall


class RAGAgent(BaseAgent):
    """Version-aware retrieval planning agent."""

    def __init__(self):
        super().__init__("RAGAgent")

    def run(self, state: SessionState, user_message: str, context: str = "") -> AgentOutput:
        msg = user_message.lower()
        intent = state.slots.extracted_features.get("intent_payload", {}) or {}
        intent_label = str(intent.get("intent_label", "general_qa"))
        query_type = str(intent.get("query_type", "general_qa"))
        should_retrieve = True

        retrieval_query = user_message
        is_nih_definition = (
            any(k in msg for k in ["nih stage model", "nih stage", "stage model"])
            and any(k in msg for k in ["what is", "define", "how many stages", "number of stages", "list stages"])
        ) or query_type == "definition"
        if is_nih_definition:
            # Query expansion to bias retrieval toward canonical NIH stage definitions.
            retrieval_query = (
                f"{user_message} "
                "NIH Stage Model Stage 0 Stage I Stage II Stage III Stage IV Stage V "
                "Onken behavioral therapy development stage definitions"
            )

        call = ToolCall(
            tool_name="versioned_rag_tool",
            tool_args={"query": retrieval_query, "top_k": 6, "newest_k": 3},
            success_criteria="Return reranked evidence with newer/revised source preference",
        )
        return AgentOutput(
            decision={
                "rag_invoked": should_retrieve,
                "rag_strategy": "version_aware_recency_rerank",
                "rag_query": retrieval_query,
            },
            confidence=0.9,
            analysis="RAG requested with recency/version-aware ranking and query expansion",
            actions=[call],
        )

    def update_state(self, state: SessionState, output: AgentOutput):
        state.slots.extracted_features["rag_invoked"] = output.decision.get("rag_invoked", False)
        if output.decision.get("rag_strategy"):
            state.slots.extracted_features["rag_strategy"] = output.decision["rag_strategy"]

