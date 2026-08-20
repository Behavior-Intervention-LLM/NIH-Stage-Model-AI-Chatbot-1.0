import requests
import time
import os
import uuid
import configparser
import xml.etree.ElementTree as ET
import nltk
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import settings


# --- Configuration ---
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), "config.ini"))

NCBI_SEARCH_URL        = config["NCBI"]["search_url"]
NCBI_FETCH_URL         = config["NCBI"]["fetch_url"]
NCBI_SLEEP             = float(config["NCBI"]["sleep"])
SEARCH_LIMIT_PER_QUERY = int(config["NCBI"]["search_limit_per_query"])
EMAIL                  = os.getenv("NCBI_EMAIL")

PMC_S3_BASE_URL        = config["S3"]["base_url"]
S3_SLEEP               = float(config["S3"]["sleep"])

CHUNK_SIZE             = int(config["CHUNKING"]["chunk_size"])
CHUNK_OVERLAP          = int(config["CHUNKING"]["chunk_overlap"])

EMBEDDING_MODEL        = config["EMBEDDING"]["model"]
EMBEDDING_DIM          = int(config["EMBEDDING"]["dim"])
EMBEDDING_BATCH_SIZE   = int(config["EMBEDDING"]["batch_size"])

Q_URL = settings.QDRANT_URL
Q_KEY = settings.QDRANT_API_KEY

COLLECTION_NAME = "nih_stage_model"
GPU_DEVICE      = os.getenv("EMBED_DEVICE", "cpu")  # "mps" for Apple Silicon, "cuda" for NVIDIA

SEARCH_QUERIES = [
    config["QUERIES"]["primary"],
    config["QUERIES"]["case_study"],
]

CHECKPOINT_FILE = os.path.join("data", "checkpoints", "processed_pmcids.txt")

print(f"Running: {os.path.basename}")
# --- Checkpoint helpers ---

def load_processed_ids() -> set:
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())

def mark_as_processed(pmcid: str):
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(f"{pmcid}\n")


# --- Lazy client initialisation (deferred to first pipeline run, not at import time) ---

_embedder = None
_qdrant   = None

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device=GPU_DEVICE)
    return _embedder

def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        if not Q_URL:
            raise ValueError("QDRANT_URL is not configured. Set it in your .env file.")
        print(f"Connecting to Qdrant Cloud at {Q_URL}...")
        _qdrant = QdrantClient(url=Q_URL, api_key=Q_KEY, timeout=60)
    return _qdrant


# --- Pipeline steps ---

def create_collection(force_recreate: bool = False):
    qdrant = _get_qdrant()
    collections = qdrant.get_collections().collections
    existing = [c.name for c in collections]

    if COLLECTION_NAME in existing:
        if force_recreate:
            qdrant.delete_collection(collection_name=COLLECTION_NAME)
            print(f"Deleted existing collection: '{COLLECTION_NAME}'")
        else:
            print(f"Collection already exists: '{COLLECTION_NAME}' — appending")
            return

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"Created collection: '{COLLECTION_NAME}'")


def search_pmc(query: str, limit: int) -> list[str]:
    params = {
        "db": "pmc",
        "term": query,
        "retmode": "json",
        "retmax": limit,
        "email": EMAIL,
    }
    r = requests.get(NCBI_SEARCH_URL, params=params)
    r.raise_for_status()
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    return [f"PMC{id}" for id in ids]


def collect_all_pmcids_with_source() -> dict[str, list[str]]:
    """Returns {"primary": [...], "case_study": [...]} with per-query deduplication."""
    query_sources = {
        SEARCH_QUERIES[0]: "primary",
        SEARCH_QUERIES[1]: "case_study",
    }
    results: dict[str, list[str]] = {"primary": [], "case_study": []}

    for query in SEARCH_QUERIES:
        source = query_sources[query]
        print(f"\n[{source.upper()}] Searching PMC for: {query[:60]}...")
        try:
            pmcids = search_pmc(query, SEARCH_LIMIT_PER_QUERY)
            results[source] = list(set(pmcids))
            print(f"  Retrieved: {len(pmcids)} | Unique: {len(results[source])}")
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(NCBI_SLEEP)

    print(f"\nTotal: {len(results['primary'])} primary + {len(results['case_study'])} case_study")
    return results


def fetch_full_text(pmcid: str) -> str | None:
    for version in [1, 2, 3]:
        url = f"{PMC_S3_BASE_URL}/{pmcid}.{version}/{pmcid}.{version}.txt"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                return r.text
        except requests.RequestException:
            continue
    print(f"  Could not fetch {pmcid} (tried versions 1–3)")
    return None


def fetch_metadata(pmcid: str) -> dict:
    url = f"{PMC_S3_BASE_URL}/{pmcid}.1/{pmcid}.1.json"
    metadata: dict = {}
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            metadata = r.json()
    except requests.RequestException:
        pass

    metadata["mesh_terms"] = []
    metadata["pub_types"]  = []
    metadata["keywords"]   = metadata.get("keywords", [])

    pmid = metadata.get("pmid")
    if pmid:
        try:
            time.sleep(NCBI_SLEEP)
            pm_req = requests.get(
                NCBI_FETCH_URL,
                params={"db": "pubmed", "id": pmid, "retmode": "xml", "email": EMAIL},
                timeout=10,
            )
            if pm_req.status_code == 200:
                root = ET.fromstring(pm_req.content)
                metadata["mesh_terms"] = [
                    m.text for m in root.findall(".//MeshHeading/DescriptorName") if m.text
                ]
                metadata["pub_types"] = [
                    p.text for p in root.findall(".//PublicationType") if p.text
                ]
        except Exception as e:
            print(f"    [!] Could not fetch PubMed MeSH for PMID {pmid}: {e}")

    return metadata


def remove_references(text: str) -> str:
    search_cutoff = int(len(text) * 0.8)
    tail_text = text[search_cutoff:].lower()
    targets = ["\nreferences", "\nbibliography", "\nworks cited", "\nliterature cited"]
    best_index = max((tail_text.rfind(t) for t in targets), default=-1)
    if best_index != -1:
        return text[:search_cutoff + best_index]
    return text


def is_meaningful(text: str, threshold: float = 0.6) -> bool:
    if not text:
        return False
    alpha_chars = sum(c.isalpha() for c in text)
    return (alpha_chars / len(text)) >= threshold


def semantic_chunk_text(
    text: str,
    embedder: SentenceTransformer,
    percentile_threshold: int = 15,
    max_words: int = 350,
) -> list[str]:
    sentences = nltk.sent_tokenize(text)
    sentences = [s for s in sentences if is_meaningful(s, threshold=0.5) and len(s.split()) > 3]

    if len(sentences) <= 2:
        return [" ".join(sentences)] if sentences else []

    sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
    similarities = torch.diagonal(
        util.cos_sim(sentence_embeddings[:-1], sentence_embeddings[1:])
    ).cpu().tolist()

    threshold = np.percentile(similarities, percentile_threshold)

    chunks: list[str] = []
    current_chunk = [sentences[0]]
    current_word_count = len(sentences[0].split())

    for i, sim in enumerate(similarities):
        next_sentence = sentences[i + 1]
        next_word_count = len(next_sentence.split())
        if sim < threshold or (current_word_count + next_word_count) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [next_sentence]
            current_word_count = next_word_count
        else:
            current_chunk.append(next_sentence)
            current_word_count += next_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c for c in chunks if len(c.strip()) > 100]


def upsert_paper(pmcid: str, chunks: list[str], metadata: dict, source: str = "case_study"):
    embeddings = _get_embedder().encode(chunks, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=False)

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{pmcid}_{i}")),
            vector=embedding.tolist(),
            payload={
                "pmcid":            pmcid,
                "pmid":             str(metadata.get("pmid", "")),
                "doi":              metadata.get("doi", ""),
                "text":             chunk,
                "chunk_index":      i,
                "total_chunks":     len(chunks),
                "section_title":    metadata.get("section_title", "General"),
                "title":            metadata.get("title", ""),
                "journal":          metadata.get("journal", ""),
                "year":             int(metadata.get("year", 0)) if metadata.get("year") else None,
                "authors":          metadata.get("authors", []),
                "publication_type": metadata.get("pub_types", []),
                "mesh_terms":       metadata.get("mesh_terms", []),
                "mesh_qualifiers":  metadata.get("mesh_qualifiers", []),
                "keywords":         metadata.get("keywords", []),
                "grants":           metadata.get("grants", []),
                "registry_ids":     metadata.get("registry_ids", []),
                "source":           source,
                "is_open_access":   metadata.get("is_oa", True),
                "language":         metadata.get("lang", "eng"),
                "keyword_text": " ".join(
                    (metadata.get("keywords") or [])
                    + (metadata.get("mesh_terms") or [])
                    + (metadata.get("pub_types") or [])
                ),
            },
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    qdrant = _get_qdrant()
    UPSERT_BATCH_SIZE = 100
    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points[i : i + UPSERT_BATCH_SIZE])
    print(f"  Upserted {len(points)} chunks for {pmcid} (source: {source})")


# --- Main pipeline ---

def run_pipeline(force_recreate: bool = False):
    create_collection(force_recreate=force_recreate)

    pmcid_dict    = collect_all_pmcids_with_source()
    processed_ids = load_processed_ids()
    print(f"Resuming pipeline. Already processed: {len(processed_ids)} papers.")

    embedder         = _get_embedder()
    success: list    = []
    failed:  list    = []

    for source, pmcids in pmcid_dict.items():
        for i, pmcid in enumerate(pmcids):
            if pmcid in processed_ids:
                print(f"Skipping [{source.upper()} {i+1}/{len(pmcids)}] {pmcid} (already processed)")
                continue

            print(f"\n[{source.upper()} {i+1}/{len(pmcids)}] {pmcid}")
            text = fetch_full_text(pmcid)
            if not text:
                failed.append((source, pmcid))
                continue

            try:
                metadata = fetch_metadata(pmcid)
                text     = remove_references(text)
                chunks   = semantic_chunk_text(text, embedder, percentile_threshold=15)
                upsert_paper(pmcid, chunks, metadata, source=source)
                mark_as_processed(pmcid)
                success.append((source, pmcid))
            except Exception as e:
                print(f"  FAILED on {pmcid}: {e}")
                failed.append((source, pmcid))

            time.sleep(S3_SLEEP)

    print("\n" + "=" * 60)
    print(f"Success: {len(success)}")
    print(f"  Primary:    {len([s for s, _ in success if s == 'primary'])}")
    print(f"  Case Study: {len([s for s, _ in success if s == 'case_study'])}")
    print(f"Failed: {len(failed)}")
    count_res = _get_qdrant().count(collection_name=COLLECTION_NAME)
    print(f"Total vectors: {count_res.count}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline(force_recreate=False)
