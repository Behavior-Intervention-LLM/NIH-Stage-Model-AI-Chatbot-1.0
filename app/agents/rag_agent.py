"""
RAG Agent: plans retrieval, judges whether what came back is usable, and
reformulates the query when it is not.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent
from app.config import settings
from app.core.llm import llm_client
from app.core.types import AgentOutput, Citation, SessionState, ToolCall

# Retrieval is TF-IDF over a small corpus, so the dominant failure mode is
# vocabulary mismatch between how a researcher phrases a question and how the
# source documents word the same idea. These are stripped on reformulation so
# the retry carries only content-bearing terms.
_QUESTION_STOPWORDS = {
    "a", "about", "am", "an", "and", "any", "are", "as", "at", "be", "been",
    "but", "by", "can", "could", "did", "do", "does", "for", "from", "get",
    "give", "had", "has", "have", "how", "i", "if", "in", "is", "it", "its",
    "just", "know", "like", "many", "me", "much", "my", "need", "of", "on",
    "or", "please", "should", "so", "some", "tell", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "to", "us",
    "was", "we", "were", "what", "whats", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",
}


class RAGAgent(BaseAgent):
    """Plans and evaluates retrieval for a single turn."""

    # Intents where retrieval adds nothing.
    _SKIP_RETRIEVAL_INTENTS = {"chit_chat", "admin", "debug"}

    _RETRIEVAL_TOOL = "versioned_rag_tool"

    def __init__(self):
        super().__init__("RAGAgent")

    # -------------------------
    # QUERY CONSTRUCTION
    # -------------------------

    @staticmethod
    def _base_query(state: SessionState, user_message: str) -> str:
        # The orchestrator may append uploaded-document text to the message;
        # keep the query to the actual question so TF-IDF matching stays sharp.
        query = user_message.split("\n\n[Session uploaded context]")[0].strip()
        stage = state.slots.stage
        if stage:
            query += f" NIH Stage {stage}"
        return query

    @staticmethod
    def _keyword_query(question: str, state: SessionState) -> str:
        """Deterministic fallback rewrite: content words only, plus whatever
        the intent agent already extracted about what the user wants."""
        tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+|\d+", question.lower())
        content = [t for t in tokens if t not in _QUESTION_STOPWORDS and len(t) > 2]

        intent = state.slots.extracted_features.get("intent_payload", {}) or {}
        for signal in (intent.get("extracted_signals") or [])[:3]:
            content.extend(
                t for t in re.findall(r"[a-zA-Z]+", str(signal).lower())
                if t not in _QUESTION_STOPWORDS and len(t) > 2
            )
        goal = intent.get("user_goal")
        if goal:
            content.extend(
                t for t in re.findall(r"[a-zA-Z]+", str(goal).lower())
                if t not in _QUESTION_STOPWORDS and len(t) > 2
            )

        # Preserve order, drop repeats.
        seen: set[str] = set()
        deduped = [t for t in content if not (t in seen or seen.add(t))]
        return " ".join(deduped) or question

    def _llm_reformulations(
        self, question: str, failed_queries: List[str], best_score: float
    ) -> List[str]:
        """Ask the cheap model for alternative phrasings. Returns [] on any
        failure so the caller can fall back to the deterministic rewrite."""
        if not settings.RAG_LLM_REFORMULATION or not llm_client.is_enabled():
            return []

        system_prompt = (
            "You rewrite search queries for a keyword (TF-IDF) index of NIH "
            "Stage Model and behavioral-intervention research documents. "
            "The previous query retrieved nothing relevant, most likely "
            "because the user's wording does not match the documents' wording. "
            "Return JSON only: {\"queries\": [\"...\", \"...\"]} with at most 2 "
            "alternative queries. Use terminology a research paper or NIH "
            "policy document would use. Prefer nouns and technical terms; "
            "drop question words entirely. Do not repeat a failed query."
        )
        user_prompt = (
            f"User question: {question}\n"
            f"Failed queries: {failed_queries}\n"
            f"Best similarity achieved: {best_score:.3f} "
            f"(threshold {settings.RAG_MIN_RELEVANCE})\n"
            "Produce alternative queries."
        )
        data = llm_client.chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=settings.LLM_INTENT_MODEL or None,
        )
        if not isinstance(data, dict):
            return []

        raw = data.get("queries")
        if not isinstance(raw, list):
            return []

        out: List[str] = []
        lowered_failed = {q.strip().lower() for q in failed_queries}
        for item in raw[:2]:
            q = str(item).strip()
            if q and q.lower() not in lowered_failed:
                out.append(q[:300])
        return out

    # -------------------------
    # EVIDENCE ASSESSMENT
    # -------------------------

    @staticmethod
    def semantic_score(citation: Citation) -> float:
        """Raw retrieval similarity, ignoring recency and feedback weighting.

        versioned_rag_tool blends recency and learned weights into
        relevance_score, so sufficiency must read the unblended value —
        otherwise a boost could disguise a passage that matches nothing.
        """
        raw = (citation.metadata or {}).get("semantic_score")
        if raw is None:
            raw = citation.relevance_score
        try:
            return float(raw or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def term_coverage(citation: Citation) -> float:
        """Share of the query's distinctive terms present in the passage.

        Defaults to 1.0 (pass) when the retrieving tool does not report it,
        so tools other than versioned_rag_tool are gated on similarity alone
        rather than being silently rejected.
        """
        raw = (citation.metadata or {}).get("term_coverage")
        if raw is None:
            return 1.0
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 1.0

    @classmethod
    def is_usable(cls, citation: Citation) -> bool:
        return (
            cls.semantic_score(citation) >= settings.RAG_MIN_RELEVANCE
            and cls.term_coverage(citation) >= settings.RAG_MIN_COVERAGE
        )

    @classmethod
    def assess_evidence(cls, citations: List[Citation]) -> Dict[str, Any]:
        """Decide whether retrieval produced anything worth grounding on.

        Two signals, because on a corpus this small neither separates alone:
        cosine similarity is inflated by incidental common-word overlap, and
        coverage is high for any passage sharing the query's one rare word.
        Requiring both catches the case this gate exists for — the corpus has
        nothing on this topic — without rejecting real questions.

        It is a retrieval-miss detector, not a topic filter. A question whose
        wording genuinely overlaps the corpus will pass even if off-topic;
        Guardrails handles topic policy separately.
        """
        scores = [cls.semantic_score(c) for c in citations]
        covers = [cls.term_coverage(c) for c in citations]
        usable = [c for c in citations if cls.is_usable(c)]
        best = max(scores) if scores else 0.0

        if not citations:
            reason = "no_results"
        elif not usable:
            reason = (
                "below_relevance_threshold"
                if max(covers) >= settings.RAG_MIN_COVERAGE
                else "query_terms_absent_from_corpus"
            )
        else:
            reason = "sufficient"

        return {
            "sufficient": bool(usable),
            "reason": reason,
            "best_score": round(best, 4),
            "best_coverage": round(max(covers), 3) if covers else 0.0,
            "threshold": settings.RAG_MIN_RELEVANCE,
            "coverage_threshold": settings.RAG_MIN_COVERAGE,
            "usable_count": len(usable),
            "retrieved_count": len(citations),
        }

    # -------------------------
    # PLANNING
    # -------------------------

    def run(self, state: SessionState, user_message: str, context: str = "") -> AgentOutput:
        """First retrieval attempt of the turn."""
        return self.plan(state, user_message, attempt=0)

    def plan(
        self,
        state: SessionState,
        user_message: str,
        attempt: int = 0,
        previous_queries: Optional[List[str]] = None,
        assessment: Optional[Dict[str, Any]] = None,
    ) -> AgentOutput:
        """Plan retrieval for `attempt`. Attempt 0 searches the question as
        asked; later attempts rewrite it using what the failure looked like."""
        intent = state.slots.extracted_features.get("intent_payload", {}) or {}
        intent_label = str(intent.get("intent_label", "general_qa")).lower()

        if intent_label in self._SKIP_RETRIEVAL_INTENTS:
            return AgentOutput(
                decision={
                    "rag_invoked": False,
                    "rag_strategy": "skipped_by_intent",
                    "attempt": attempt,
                },
                confidence=0.9,
                analysis=f"Retrieval skipped for intent '{intent_label}'",
                actions=[],
            )

        previous_queries = list(previous_queries or [])
        question = user_message.split("\n\n[Session uploaded context]")[0].strip()
        top_k = settings.RAG_TOP_K

        if attempt == 0:
            queries = [self._base_query(state, user_message)]
            strategy = "direct_query"
        else:
            best_score = float((assessment or {}).get("best_score", 0.0))
            queries = self._llm_reformulations(question, previous_queries, best_score)
            strategy = "llm_reformulation"
            if not queries:
                queries = [self._keyword_query(question, state)]
                strategy = "keyword_reformulation"
            # A rewrite identical to something already tried is not a retry.
            tried = {q.strip().lower() for q in previous_queries}
            queries = [q for q in queries if q.strip().lower() not in tried]
            if not queries:
                return AgentOutput(
                    decision={
                        "rag_invoked": False,
                        "rag_strategy": "reformulation_exhausted",
                        "attempt": attempt,
                    },
                    confidence=0.4,
                    analysis="Reformulation produced no query distinct from those already tried",
                    actions=[],
                )

        return AgentOutput(
            decision={
                "rag_invoked": True,
                "rag_strategy": strategy,
                "attempt": attempt,
                "queries": queries,
            },
            confidence=0.9 if attempt == 0 else 0.6,
            analysis=(
                f"Planned retrieval attempt {attempt + 1} via {strategy}: {queries}"
            ),
            actions=[
                ToolCall(
                    tool_name=self._RETRIEVAL_TOOL,
                    tool_args={"query": q, "top_k": top_k},
                    success_criteria=f"passage with similarity >= {settings.RAG_MIN_RELEVANCE}",
                )
                for q in queries
            ],
        )

    # -------------------------
    # STATE UPDATE
    # -------------------------

    def update_state(self, state: SessionState, output: AgentOutput):
        # Retrieval results land in state.artifacts (via the orchestrator's
        # tool execution); the responder collects evidence from there.
        pass
