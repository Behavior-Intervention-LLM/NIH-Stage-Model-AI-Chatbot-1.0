from __future__ import annotations

from app.agents.base import BaseAgent
from app.core.types import AgentOutput, SessionState, ToolCall

class RAGAgent(BaseAgent):
    """Agentic RAG Engine: retrieval + reranking + structuring"""

    def __init__(self):
        super().__init__("RAGAgent")

    #     # --- Hardware ---
    #     self.device = (
    #         "mps" if torch.backends.mps.is_available()
    #         else "cuda" if torch.cuda.is_available()
    #         else "cpu"
    #     )

    #     # --- Models ---
    #     try:
    #         from sentence_transformers import SentenceTransformer, CrossEncoder

    #         self.bi_encoder = SentenceTransformer(
    #             "mixedbread-ai/mxbai-embed-large-v1",
    #             device=self.device
    #         )

    #         self.reranker = CrossEncoder(
    #             "cross-encoder/ms-marco-MiniLM-L-6-v2",
    #             device=self.device
    #         )

    #         self.embedding_available = True

    #     except Exception as e:
    #         print(f"[RAG] Embedding models disabled: {e}")

    #         self.bi_encoder = None
    #         self.reranker = None
    #         self.embedding_available = False

    #     # --- Vector DB ---
    #     # self.qdrant = QdrantClient(
    #     #     url=os.getenv("QDRANT_URL"),
    #     #     api_key=os.getenv("QDRANT_API_KEY")
    #     # )
    #     # self.collection_name = "nih_stage_model"

    #     try:
    #         from qdrant_client import QdrantClient

    #         self.qdrant = QdrantClient(
    #             url=os.getenv("QDRANT_URL"),
    #             api_key=os.getenv("QDRANT_API_KEY")
    #         )
    #         self.collection_name = "nih_stage_model"
    #         self.qdrant_available = True

    #     except Exception as e:
    #         # print(f"[RAG] Qdrant disabled due to error: {e}")
    #         self.qdrant = None
    #         self.collection_name = None
    #         self.qdrant_available = False

    # # -------------------------
    # # QUERY EXPANSION
    # # -------------------------
    # def _expand_query(self, query: str, intent_label: str) -> str:
    #     base = query

    #     expansions = [
    #         "NIH Stage Model",
    #         "behavioral intervention development",
    #         "mechanism of action",
    #         "randomized controlled trial",
    #         "pilot feasibility study"
    #     ]

    #     if intent_label == "definition":
    #         expansions += ["Stage 0 Stage I Stage II Stage III Stage IV Stage V Onken"]

    #     return base + " " + " ".join(expansions)

    # # -------------------------
    # # RETRIEVAL + RERANK
    # # -------------------------
    # def _retrieve(self, query: str, top_k: int = 6):

    #     # Step 1: Embed query
    #     prompt_query = f"Represent this sentence for searching relevant passages: {query}"
    #     query_vector = self.bi_encoder.encode(prompt_query).tolist()

    #     # Step 2: Vector search (broad recall)
    #     # In v1.17.0+, .query() is the preferred high-level method

    #     # modified here (xinai)
    #     results = self.qdrant.search(
    #         collection_name=self.collection_name,
    #         query_vector=query_vector,
    #         limit=40,
    #         with_payload=True,
    #     )

    #     # response = self.qdrant.query(
    #     #     collection_name=self.collection_name,
    #     #     query_vector=query_vector,  # Use 'query_vector' for the embedding
    #     #     limit=40,
    #     #     with_payload=True
    #     # )
    #     # results = response

    #     # Step 3: Build passages (TEXT + KEYWORDS)
    #     passages = [
    #         (r.payload.get("text", "") + " " + r.payload.get("keyword_text", ""))
    #         for r in results
    #     ]

    #     # Step 4: Cross-encoder reranking
    #     pairs = [[query, p] for p in passages]
    #     scores = self.reranker.predict(pairs)

    #     # Step 5: Combine scores
    #     final_docs = []
    #     for i, r in enumerate(results):
    #         p = r.payload

    #         combined_score = 0.7 * float(scores[i]) + 0.3 * float(r.score)

    #         final_docs.append({
    #             "content": p.get("text"),
    #             "score": combined_score,

    #             # STRUCTURED SIGNALS
    #             "study_type": p.get("publication_type"),
    #             "year": p.get("year"),
    #             "mesh_terms": p.get("mesh_terms"),
    #             "keywords": p.get("keywords"),

    #             "metadata": {
    #                 "pmcid": p.get("pmcid"),
    #                 "title": p.get("title"),
    #                 "source": p.get("source")
    #             }
    #         })

    #     # Step 6: Sort
    #     final_docs.sort(key=lambda x: x["score"], reverse=True)

    #     # Step 7: Deduplicate (by paper)
    #     seen = set()
    #     deduped = []
    #     for doc in final_docs:
    #         pmcid = doc["metadata"]["pmcid"]
    #         if pmcid not in seen:
    #             deduped.append(doc)
    #             seen.add(pmcid)

    #     return deduped[:top_k]

    # Intents where retrieval adds nothing.
    _SKIP_RETRIEVAL_INTENTS = {"chit_chat", "admin", "debug"}

    @staticmethod
    def _search_query(state: SessionState, user_message: str) -> str:
        # The orchestrator may append uploaded-document text to the message;
        # keep the query to the actual question so TF-IDF matching stays sharp.
        query = user_message.split("\n\n[Session uploaded context]")[0].strip()
        stage = state.slots.stage
        if stage:
            query += f" NIH Stage {stage}"
        return query

    # -------------------------
    # MAIN RUN
    # -------------------------
    def run(self, state: SessionState, user_message: str, context: str = "") -> AgentOutput:
        """Plan retrieval deterministically (no LLM call): one vector_tool
        search against the local TF-IDF store, executed by the orchestrator."""
        intent = state.slots.extracted_features.get("intent_payload", {}) or {}
        intent_label = str(intent.get("intent_label", "general_qa")).lower()

        if intent_label in self._SKIP_RETRIEVAL_INTENTS:
            return AgentOutput(
                decision={"rag_invoked": False, "strategy": "skipped_by_intent", "results_found": 0},
                confidence=0.9,
                analysis=f"Retrieval skipped for intent '{intent_label}'",
                actions=[],
            )

        search_query = self._search_query(state, user_message)
        return AgentOutput(
            decision={"rag_invoked": True, "rag_strategy": "tfidf_vector_search"},
            confidence=0.9,
            analysis="Planned local vector store retrieval",
            actions=[
                ToolCall(
                    tool_name="vector_tool",
                    tool_args={"query": search_query, "top_k": 4},
                )
            ],
        )

    # -------------------------
    # STATE UPDATE
    # -------------------------
    def update_state(self, state: SessionState, output: AgentOutput):
        # Retrieval results land in state.artifacts (via the orchestrator's
        # tool execution); the responder collects evidence from there.
        pass
