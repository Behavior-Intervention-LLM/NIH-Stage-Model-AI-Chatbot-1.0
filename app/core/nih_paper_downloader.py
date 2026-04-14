import requests
import time
import os
import uuid
import configparser
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import xml.etree.ElementTree as ET
import nltk
import numpy as np
from sentence_transformers import util

# Download the sentence tokenizer model (only needs to run once)
nltk.download('punkt', quiet=True)

# --- 1. Load Configurations ---
# Load secrets from your custom named env file
load_dotenv(os.path.join(os.path.dirname(__file__), "qdrant.env"))

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), "config.ini"))

# NCBI
NCBI_SEARCH_URL         = config["NCBI"]["search_url"]
NCBI_FETCH_URL          = config["NCBI"]["fetch_url"]
NCBI_SLEEP              = float(config["NCBI"]["sleep"])
SEARCH_LIMIT_PER_QUERY  = int(config["NCBI"]["search_limit_per_query"])
# Get email from qdrant.env (best practice for privacy)
EMAIL = os.getenv("NCBI_EMAIL") 

# AWS S3
PMC_S3_BASE_URL         = config["S3"]["base_url"]
S3_SLEEP                = float(config["S3"]["sleep"])

# Chunking
CHUNK_SIZE              = int(config["CHUNKING"]["chunk_size"])
CHUNK_OVERLAP           = int(config["CHUNKING"]["chunk_overlap"])

# Embedding
EMBEDDING_MODEL         = config["EMBEDDING"]["model"]
EMBEDDING_DIM           = int(config["EMBEDDING"]["dim"])
EMBEDDING_BATCH_SIZE    = int(config["EMBEDDING"]["batch_size"])

# Qdrant Cloud Credentials (from qdrant.env)
Q_URL = os.getenv("QDRANT_URL")
Q_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "nih_stage_model"

GPU_DEVICE = "mps" #Can CUDA for RTX 3080, A100, etc.

# Queries
SEARCH_QUERIES = [
    config["QUERIES"]["primary"],
    config["QUERIES"]["case_study"],
]

CHECKPOINT_FILE = "processed_pmcids.txt"

def load_processed_ids() -> set:
    """Reads the checkpoint file and returns a set of already processed PMCIDs."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set() # Return an empty set if we are starting fresh
    
    with open(CHECKPOINT_FILE, "r") as f:
        # Read lines, strip whitespace, and ignore empty lines
        return set(line.strip() for line in f if line.strip())

def mark_as_processed(pmcid: str):
    """Appends a successfully processed PMCID to the checkpoint file."""
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(f"{pmcid}\n")

# --- 2. Initialize Clients ---
print(f"Loading embedding model: {EMBEDDING_MODEL}...")
embedder = SentenceTransformer(EMBEDDING_MODEL, device=GPU_DEVICE)

# Updated for Qdrant Cloud
print(f"Connecting to Qdrant Cloud at {Q_URL}...")
qdrant = QdrantClient(
    url=Q_URL, 
    api_key=Q_KEY,
    timeout=60.0
)

# --- Step 1: Create collection (once) ---
def create_collection(force_recreate: bool = False):
    # Check if collection exists
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
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )
    print(f"Created collection: '{COLLECTION_NAME}'")

# --- Step 2: Search PMC for a single query ---
def search_pmc(query: str, limit: int) -> list[str]:
    params = {
        "db": "pmc",
        "term": query,
        "retmode": "json",
        "retmax": limit,
        "email": EMAIL
    }
    r = requests.get(NCBI_SEARCH_URL, params=params)
    r.raise_for_status() # Ensure the request worked
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    pmcids = [f"PMC{id}" for id in ids]
    return pmcids

# --- Step 3: Run all queries and deduplicate ---
def collect_all_pmcids() -> list[str]:
    seen = set()
    all_pmcids = []

    for i, query in enumerate(SEARCH_QUERIES):
        print(f"\n[Query {i+1}/{len(SEARCH_QUERIES)}] Searching PMC for: {query}")
        try:
            pmcids = search_pmc(query, SEARCH_LIMIT_PER_QUERY)
            new = [p for p in pmcids if p not in seen]
            seen.update(new)
            all_pmcids.extend(new)
            print(f"  Retrieved: {len(pmcids)} | New after dedup: {len(new)} | Total so far: {len(all_pmcids)}")
        except Exception as e:
            print(f"  Error searching query: {e}")
        
        time.sleep(NCBI_SLEEP)

    print(f"\nTotal unique PMCIDs across all queries: {len(all_pmcids)}")
    return all_pmcids

# --- Step 4: Fetch full text from S3 over HTTPS ---
def fetch_full_text(pmcid: str) -> str | None:
    for version in [1, 2, 3]:
        url = f"{PMC_S3_BASE_URL}/{pmcid}.{version}/{pmcid}.{version}.txt"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                return r.text
        except requests.RequestException:
            continue
    print(f"  Could not fetch {pmcid} (tried versions 1-3)")
    return None

# --- Step 5: Fetch metadata (Hybrid S3 + NCBI) ---
def fetch_metadata(pmcid: str) -> dict:
    # 1. Get basic structural metadata from S3 JSON
    url = f"{PMC_S3_BASE_URL}/{pmcid}.1/{pmcid}.1.json"
    metadata = {}
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            metadata = r.json()
    except requests.RequestException:
        pass

    # Initialize our Ground Truth lists so they are never None
    metadata["mesh_terms"] = []
    metadata["pub_types"] = []
    metadata["keywords"] = metadata.get("keywords", [])

    # 2. Enrich with high-accuracy MeSH data from PubMed
    pmid = metadata.get("pmid")
    if pmid:
        try:
            # Respect NCBI rate limits
            time.sleep(NCBI_SLEEP)
            
            pm_params = {
                "db": "pubmed", 
                "id": pmid, 
                "retmode": "xml",
                "email": EMAIL
            }
            # We use the fetch_url you defined in config.ini earlier
            pm_req = requests.get(NCBI_FETCH_URL, params=pm_params, timeout=10)
            
            if pm_req.status_code == 200:
                root = ET.fromstring(pm_req.content)
                
                # Extract MeSH Terms (The Accuracy Engine)
                mesh_headings = root.findall(".//MeshHeading/DescriptorName")
                metadata["mesh_terms"] = [m.text for m in mesh_headings if m.text]
                
                # Extract Publication Types (Study Design)
                pub_types = root.findall(".//PublicationType")
                metadata["pub_types"] = [p.text for p in pub_types if p.text]
                
        except Exception as e:
            # If it fails, we just silently continue with the basic S3 data
            print(f"    [!] Could not fetch PubMed MeSH for PMID {pmid}: {e}")

    return metadata

# --- Step 6: Chunk text ---
def remove_references(text: str) -> str:
    # Only search for the references section in the last 20% of the document
    search_cutoff = int(len(text) * 0.8)
    tail_text = text[search_cutoff:].lower()
    
    # Look for common structural patterns rather than just the word
    targets = ["\nreferences", "\nbibliography", "\nworks cited", "\nliterature cited"]
    
    best_index = -1
    for target in targets:
        idx = tail_text.rfind(target)
        if idx > best_index:
            best_index = idx
            
    if best_index != -1:
        # Cut the text at the global index
        global_cut_index = search_cutoff + best_index
        return text[:global_cut_index]
        
    return text

# alpha filtering for relevance + quality control
def is_meaningful(text: str, threshold: float = 0.6) -> bool:
    if not text:
        return False
    
    alpha_chars = sum(c.isalpha() for c in text)
    ratio = alpha_chars / len(text)
    
    return ratio >= threshold

import torch # Add this to your imports at the top

def semantic_chunk_text(text: str, embedder, percentile_threshold: int = 15, max_words: int = 350) -> list[str]:
    # 1. Split and clean sentences
    sentences = nltk.sent_tokenize(text)
    sentences = [s for s in sentences if is_meaningful(s, threshold=0.5) and len(s.split()) > 3]

    if len(sentences) <= 2:
        return [" ".join(sentences)] if sentences else []

    # 2. Embed sentences
    sentence_embeddings = embedder.encode(sentences, convert_to_tensor=True, show_progress_bar=False)

    # 3. VECTORIZED Cosine Similarity (Lightning fast on CUDA)
    # Computes similarities for all adjacent pairs in one operation
    similarities = torch.diagonal(util.cos_sim(sentence_embeddings[:-1], sentence_embeddings[1:])).cpu().tolist()

    # 4. Determine semantic breaks
    threshold = np.percentile(similarities, percentile_threshold)

    # 5. Group sentences into chunks WITH a max-word limit
    chunks = []
    current_chunk = [sentences[0]]
    current_word_count = len(sentences[0].split())

    for i, sim in enumerate(similarities):
        next_sentence = sentences[i + 1]
        next_word_count = len(next_sentence.split())

        # Break the chunk if the topic changes OR if adding the next sentence exceeds our max limit
        if sim < threshold or (current_word_count + next_word_count) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [next_sentence]
            current_word_count = next_word_count
        else:
            current_chunk.append(next_sentence)
            current_word_count += next_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # 6. Final filter for minimum length
    return [c for c in chunks if len(c.strip()) > 100]

# --- Step 7: Embed + upsert to Qdrant ---
def upsert_paper(pmcid: str, chunks: list[str], metadata: dict, source: str = "case_study"):
    """
    Args:
        source: "primary" or "case_study"
    """
    embeddings = embedder.encode(chunks, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=False)

    points = [
        PointStruct(
            # Generate a valid, deterministic UUID for Qdrant
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{pmcid}_{i}")),
            vector=embedding.tolist(),
            # Properly assign the dictionary to the payload argument
            payload={
                # --- Identifiers (Immutable Ground Truth) ---
                "pmcid": pmcid,
                "pmid": str(metadata.get("pmid", "")),
                "doi": metadata.get("doi", ""),
                
                # --- Content & Navigation ---
                "text": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "section_title": metadata.get("section_title", "General"), 
                
                # --- Bibliographic Data ---
                "title": metadata.get("title", ""),
                "journal": metadata.get("journal", ""),
                "year": int(metadata.get("year", 0)) if metadata.get("year") else None,
                "authors": metadata.get("authors", []), 
                "publication_type": metadata.get("pub_types", []), 
                
                # --- NLM Human-Indexed Metadata (The Accuracy Engine) ---
                "mesh_terms": metadata.get("mesh_terms", []),    
                "mesh_qualifiers": metadata.get("mesh_qualifiers", []), 
                "keywords": metadata.get("keywords", []),        
                
                # --- Study Signal Data ---
                "grants": metadata.get("grants", []),            
                "registry_ids": metadata.get("registry_ids", []), 
                
                # --- Retrieval & Filtering Helpers ---
                "source": source,                                
                "is_open_access": metadata.get("is_oa", True),
                "language": metadata.get("lang", "eng"),
                
                # --- Vector Optimization ---
                "keyword_text": " ".join(
                    (metadata.get("keywords") or []) + 
                    (metadata.get("mesh_terms") or []) + 
                    (metadata.get("pub_types") or [])
                )
            }
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    UPSERT_BATCH_SIZE = 100
    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i:i+UPSERT_BATCH_SIZE]
        )
    print(f"  Upserted {len(points)} chunks for {pmcid} (source: {source})")

def collect_all_pmcids_with_source() -> dict[str, list[str]]:
    """Returns: {"primary": [...], "case_study": [...]}"""
    results = {
        "primary": [],
        "case_study": []
    }
    
    query_sources = {
        SEARCH_QUERIES[0]: "primary",
        SEARCH_QUERIES[1]: "case_study"
    }
    
    for query in SEARCH_QUERIES:
        source = query_sources[query]
        print(f"\n[{source.upper()}] Searching PMC for: {query}")
        try:
            pmcids = search_pmc(query, SEARCH_LIMIT_PER_QUERY)
            results[source].extend(pmcids)
            print(f"  Retrieved: {len(pmcids)} papers for {source}")
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(NCBI_SLEEP)
    
    # Deduplicate within sources (shouldn't happen, but just in case)
    for source in results:
        results[source] = list(set(results[source]))
    
    print(f"\nTotal: {len(results['primary'])} primary + {len(results['case_study'])} case_study")
    return results

# --- Main pipeline ---
def run_pipeline(force_recreate: bool = False):
    create_collection(force_recreate=force_recreate)
    
    # Get papers with source tracking
    pmcid_dict = collect_all_pmcids_with_source()
    
    success, failed = [], []
    
    # Process with source
    for source, pmcids in pmcid_dict.items():
        for i, pmcid in enumerate(pmcids):
            print(f"\n[{source.upper()} {i+1}/{len(pmcids)}] {pmcid}")
            text = fetch_full_text(pmcid)
            if not text:
                failed.append((source, pmcid))
                continue
            metadata = fetch_metadata(pmcid)
            text = remove_references(text)
            # You can tweak the percentile_threshold to make chunks larger (lower %) or smaller (higher %)
            chunks = semantic_chunk_text(text, embedder, percentile_threshold=15)
            upsert_paper(pmcid, chunks, metadata, source=source)  # ← SOURCE TRACKED
            success.append((source, pmcid))
            time.sleep(S3_SLEEP)

    all_pmcids = collect_all_pmcids()
    # 1. Load the checkpoint data
    processed_ids = load_processed_ids()
    print(f"Resuming pipeline. Found {len(processed_ids)} already processed papers.")

    # Assume `all_pmcids` is your list of 1000 IDs to process
    for i, pmcid in enumerate(all_pmcids):
        
        # 2. The Checkpoint Skip Logic
        if pmcid in processed_ids:
            print(f"Skipping {i+1}/{len(all_pmcids)}: {pmcid} (Already processed)")
            continue
            
        print(f"Processing {i+1}/{len(all_pmcids)}: {pmcid}")
        
    try:
    
        # The Qdrant upload (preferably with the try/except retry block!)
        upsert_paper(pmcid, chunks, metadata, source=source)
        
        # 3. Mark as success! (ONLY happens if upsert_paper doesn't crash)
        mark_as_processed(pmcid)
        
    except Exception as e:
        print(f"FAILED on {pmcid}: {e}")
        # The script will print the error but keep going to the next paper!

    print("\n" + "=" * 60)
    print(f"Success: {len(success)}")
    print(f"  Primary: {len([s for s,_ in success if s=='primary'])}")
    print(f"  Case Study: {len([s for s,_ in success if s=='case_study'])}")
    print(f"Failed: {len(failed)}")
    count_res = qdrant.count(collection_name=COLLECTION_NAME)
    print(f"Total vectors: {count_res.count}")
    print("=" * 60)

if __name__ == "__main__":
    # Note: Set force_recreate=True if you want to wipe the cloud database and start over
    run_pipeline(force_recreate=True)