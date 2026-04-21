from __future__ import annotations
import os
import numpy as np
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from pathlib import Path

from app.agents.base import BaseAgent
from app.core.types import AgentOutput, SessionState, ToolCall

current_file = Path(__file__).resolve()
app_dir = current_file.parent.parent 
env_path = app_dir / "core" / "qdrant.env"
load_dotenv(dotenv_path=env_path)

class RAGAgent(BaseAgent):
    """Agentic RAG Engine: retrieval + reranking + structuring"""

    def __init__(self):
        super().__init__("RAGAgent")

        # --- Hardware ---
        self.device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        # --- Models (Initialize as None, set in try block) ---
        self.bi_encoder = None
        self.reranker = None
        self.embedding_available = False
        
        try:
            self.bi_encoder = SentenceTransformer(
                "mixedbread-ai/mxbai-embed-large-v1",
                device=self.device
            )

            self.reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                device=self.device
            )

            self.embedding_available = True

        except Exception as e:
            print(f"[RAG] Embedding models disabled: {e}")
            self.embedding_available = False

        # --- Vector DB (Initialize as None, set in try block) ---
        self.qdrant = None
        self.collection_name = "nih_stage_model_test"
        self.qdrant_available = False
        
        try:
            self.qdrant = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY")
            )
            self.qdrant_available = True

        except Exception as e:
            print(f"[RAG] Qdrant disabled due to error: {e}")
            self.qdrant = None
            self.qdrant_available = False

    # -------------------------
    # QUERY EXPANSION
    # -------------------------
    def _expand_query(self, query: str, intent_label: str, stage_result: str = None) -> str:
        parts = [query]
        
        # Only add stage model anchor if not already in query
        if "stage model" not in query.lower() and "NIH" not in query:
            parts.append("NIH Stage Model behavioral intervention")
        
        # Add stage-specific terms if we know the stage
        stage_expansions = {
            "IA": "intervention development manual fidelity",
            "IB": "pilot feasibility acceptability",
            "II": "efficacy trial mechanism mediation",
            "III": "effectiveness community practitioners",
            "IV": "pragmatic large scale implementation",
            "V": "dissemination implementation strategy"
        }
        if stage_result and stage_result in stage_expansions:
            parts.append(stage_expansions[stage_result])
        
        return " ".join(parts)

    # -------------------------
    # RETRIEVAL + RERANK
    # -------------------------
    def _retrieve(self, query: str, top_k: int = 6, stage_result: str = None):
        
        # Safety check
        if not self.embedding_available or not self.qdrant_available:
            print("[RAG] Models not available, skipping retrieval")
            return []

        # Step 1: Embed the expanded query
        prompt_query = f"Represent this sentence for searching relevant passages: {query}"
        query_vector = self.bi_encoder.encode(prompt_query).tolist()

        # Step 2: Vector search (NO payload filter)
        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=40,
            with_payload=True
            # NOTICE: query_filter is REMOVED
        ).points
        
        clean_docs = []
        for point in results:
            clean_docs.append({
                "text": point.payload.get("text", ""),
                "metadata": {
                    "pmcid": point.payload.get("pmcid", "unknown"),
                    "score": point.score
                }
            })
        return clean_docs

    # -------------------------
    # MAIN RUN
    # -------------------------
    def run(self, state: SessionState, user_message: str, context: str = "", stage_result: str = None, stage_confidence: float = 0.0) -> AgentOutput:
        
        if not self.qdrant_available or not self.embedding_available:
            return AgentOutput(
                decision={"rag_invoked": False, "strategy": "disabled", "results_found": 0},
                confidence=0.0,
                analysis="RAG models not available",
                actions=[]
            )

        intent = state.slots.extracted_features.get("intent_payload", {}) or {}
        intent_label = str(intent.get("intent_label", "general_qa"))

        # Now passes stage_result into expansion
        search_query = self._expand_query(user_message, intent_label, stage_result=stage_result)

        print(f"[*] RAGAgent retrieving for: {user_message[:50]}...")

        # Now passes stage_result into retrieval for payload filtering
        docs = self._retrieve(search_query, stage_result=stage_result)

        found = len(docs) > 0

        return AgentOutput(
            decision={
                "rag_invoked": True,
                "strategy": "agentic_hybrid_rerank",
                "results_found": len(docs)
            },
            confidence=0.95 if found else 0.3,
            analysis="Hybrid semantic retrieval with cross-encoder reranking and structured outputs",
            actions=[
                ToolCall(
                    tool_name="rag_retrieval",
                    tool_args={"query": search_query},
                    output=docs
                )
            ]
        )

    # -------------------------
    # STATE UPDATE
    # -------------------------
    def update_state(self, state: SessionState, output: AgentOutput):
        """Pass retrieved context to state"""
        if output.actions and len(output.actions) > 0:
            retrieved_docs = output.actions[0].output
            state.slots.extracted_features["retrieved_context"] = retrieved_docs
        else:
            state.slots.extracted_features["retrieved_context"] = []