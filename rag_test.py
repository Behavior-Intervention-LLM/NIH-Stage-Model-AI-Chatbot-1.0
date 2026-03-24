import os
import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# --- Config ---
QDRANT_HOST      = "localhost"
QDRANT_PORT      = 6333
COLLECTION_NAME  = "nih_definition"
EMBEDDING_MODEL  = "BAAI/bge-base-en-v1.5"
OLLAMA_URL       = "http://127.0.0.1:11434/api/chat"
LLM_MODEL        = "qwen2.5:3b-instruct"
TOP_K            = 5  # number of chunks to retrieve

# --- Clients ---
embedder = SentenceTransformer(EMBEDDING_MODEL)
qdrant   = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def retrieve(query: str, collection: str = COLLECTION_NAME) -> list[dict]:
    """Embed query and retrieve top K chunks from Qdrant."""
    query_vector = embedder.encode(query).tolist()
    
    # NEW QUERY API SYNTAX
    # 'query' replaces 'query_vector'
    results = qdrant.query_points(
        collection_name=collection,
        query=query_vector,
        limit=TOP_K
    ).points  # .points is required to get the list of results
    
    return [
        {
            "text":     r.payload.get("text", ""),
            "pmcid":    r.payload.get("pmcid", r.payload.get("filename", "")),
            "title":    r.payload.get("title", ""),
            "score":    round(r.score, 3),
        }
        for r in results
    ]

def build_prompt(query: str, chunks: list[dict]) -> str:
    """Build RAG prompt from retrieved chunks."""
    context = "\n\n".join([
        f"[Source: {c['pmcid']} | Score: {c['score']}]\n{c['text']}"
        for c in chunks
    ])
    return f"""You are a helpful research assistant specializing in the NIH Stage Model for behavioral intervention development.

Use the following retrieved context to answer the question. If the context doesn't contain enough information, say so clearly.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {query}

Answer:"""


def ask_ollama(prompt: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if r.status_code == 404:
            return f"Error 404: Ollama couldn't find the endpoint or the model '{LLM_MODEL}'."
        r.raise_for_status()
        return r.json()["message"]["content"]
    except Exception as e:
        return f"Connection Error: {e}"


def rag_query(query: str):
    """Full RAG pipeline: retrieve → prompt → generate."""
    print(f"\nQuery: {query}")
    print("-" * 50)

    # Retrieve
    chunks = retrieve(query)
    print(f"Retrieved {len(chunks)} chunks:")
    for c in chunks:
        print(f"  [{c['score']}] {c['pmcid']} — {c['title'][:60]}")

    # Generate
    print("\nGenerating response...")
    prompt   = build_prompt(query, chunks)
    response = ask_ollama(prompt)

    print(f"\nResponse:\n{response}")
    print("=" * 50)


if __name__ == "__main__":
    # Test queries
    rag_query("Only identify which nih stage nih sstage i am in. I am studying whether reading confidence-building books reduces anxiety during cancer treatment and trying to understand the psychological mechanisms behind it.")
    rag_query("Only identify which nih stage nih sstage i am in.We designed a 6-week program where cancer patients read selected confidence-building books and complete reflection exercises. We are running a small pilot study with 20 patients to see if the program is feasible.")
    rag_query("Only identify which nih stage nih sstage i am in.We conducted a randomized controlled trial with 300 cancer patients comparing our reading-based confidence intervention to standard care to evaluate its effectiveness in reducing treatment anxiety.")
    rag_query("Only identify which nih stage nih sstage i am in.Our intervention has shown positive results in clinical trials, and we are now testing how well it works when implemented in multiple oncology clinics with regular healthcare staff delivering the program.")
    rag_query("Only identify which nih stage nih sstage i am in.Our confidence-building reading program is now being implemented across hospital systems nationwide, and we are studying strategies to help clinics adopt and sustain the program long-term.")

