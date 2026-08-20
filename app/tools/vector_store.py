"""
：
"""
import os
import json
import re
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from app.tools.document_loader import DocumentChunk


# Function words are dropped at BOTH index and query time.
#
# Without this, IDF works against us: question words are rare in academic
# prose but common in user questions, so they score as highly distinctive.
# Measured on this corpus, "what" had idf 3.80 — above "stage" (2.00) and
# "efficacy" (3.35) — so "How do I move from efficacy to effectiveness?"
# was matched primarily on the word "move". Retrieval tracked phrasing
# rather than subject matter.
_STOPWORDS = {
    "a", "about", "after", "an", "and", "any", "are", "as", "at", "be", "been",
    "but", "by", "can", "could", "did", "do", "does", "for", "from", "get",
    "had", "has", "have", "how", "i", "if", "in", "is", "it", "its", "just",
    "like", "many", "me", "much", "my", "need", "of", "on", "or", "please",
    "should", "so", "some", "tell", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "us", "was", "we", "were",
    "what", "whats", "when", "where", "which", "who", "why", "will", "with",
    "would", "you", "your",
}

# "Stage II" tokenizes to ["stage", "ii"], which loses the pairing and leaves
# the bare numeral to match any other stage. Worse, the pronoun "I" and
# "Stage I" are the same token — so "I" cannot be stopworded without
# destroying stage-one queries. Collapsing the phrase into one token
# ("stage_ii") fixes both: the stage becomes a single high-signal term and
# the stray pronoun becomes safe to drop. Alternatives are ordered
# longest-first so "IV" is not consumed as "I".
_STAGE_PHRASE_RE = re.compile(r"\bstages?\s+(0|iv|iii|ii|ib|ia|i|v)\b", re.IGNORECASE)

_TOKEN_RE = re.compile(r'[一-鿿]+|[a-z]+_[a-z0-9]+|[a-zA-Z]+|\d+')


def _normalize(text: str) -> str:
    return _STAGE_PHRASE_RE.sub(lambda m: f" stage_{m.group(1).lower()} ", text.lower())


def _tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    tokens = _TOKEN_RE.findall(_normalize(text))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    return tokens


class SimpleVectorStore:
    """
    （ TF-IDF）
    recommendation（ Pinecone、Weaviate、Chroma）
    """
    
    def __init__(self, storage_path: str = "data/vector_store"):
        """
        Args:
            storage_path: 
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.chunks: List[DocumentChunk] = []
        self.vectors: Optional[np.ndarray] = None
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        
        # 
        self._load()
    
    def add_chunks(self, chunks: List[DocumentChunk], replace: bool = False):
        """Index chunks, skipping any already present.

        `replace=True` rebuilds the store from scratch — use it for a full
        re-ingest. The previous behaviour was an unconditional extend, so
        re-running the ingestion script appended a second complete copy of
        every document. That is how the shipped index reached 641 chunks
        holding only 186 unique passages: top-k came back as the same passage
        repeated, cutting effective retrieval depth to one.
        """
        if replace:
            self.chunks = []
            self.vocabulary = {}
            self.idf = {}
            self.vectors = None

        seen = {(c.source, c.content) for c in self.chunks}
        added = 0
        for chunk in chunks:
            key = (chunk.source, chunk.content)
            if key in seen:
                continue
            seen.add(key)
            self.chunks.append(chunk)
            added += 1

        self._rebuild_index()
        self._save()
        return added

    def _rebuild_index(self):
        """Rebuild the TF-IDF matrix over self.chunks."""
        if not self.chunks:
            return

        # Vocabulary and IDF derive wholly from the current chunks and must
        # not carry over terms from a previous corpus.
        self.vocabulary = {}
        self.idf = {}

        all_tokens = []
        for chunk in self.chunks:
            tokens = _tokenize(chunk.content)
            all_tokens.append(tokens)
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
        
        #  IDF
        doc_count = len(self.chunks)
        token_sets = [set(tokens) for tokens in all_tokens]
        df_counts: Dict[str, int] = {}
        for token_set in token_sets:
            for token in token_set:
                df_counts[token] = df_counts.get(token, 0) + 1
        for token in self.vocabulary:
            df = df_counts.get(token, 0)
            self.idf[token] = np.log((doc_count + 1) / (df + 1)) + 1
        
        #  TF-IDF 
        vectors = []
        for tokens in all_tokens:
            vector = np.zeros(len(self.vocabulary))
            token_count = len(tokens)
            
            #  TF
            token_freq = {}
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
            
            #  TF-IDF
            for token, freq in token_freq.items():
                if token in self.vocabulary:
                    tf = freq / token_count
                    idf = self.idf[token]
                    vector[self.vocabulary[token]] = tf * idf
            
            # 
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            vectors.append(vector)
        
        self.vectors = np.array(vectors)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """"""
        if not self.chunks or self.vectors is None:
            return []
        
        # 
        query_tokens = _tokenize(query)
        query_vector = np.zeros(len(self.vocabulary))
        
        token_count = len(query_tokens)
        if token_count == 0:
            return []
        
        token_freq = {}
        for token in query_tokens:
            token_freq[token] = token_freq.get(token, 0) + 1
        
        for token, freq in token_freq.items():
            if token in self.vocabulary:
                tf = freq / token_count
                idf = self.idf.get(token, 0)
                query_vector[self.vocabulary[token]] = tf * idf
        
        # 
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        
        # （）
        similarities = np.dot(self.vectors, query_vector)
        
        #  top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 
                chunk = self.chunks[idx]
                results.append({
                    "doc": chunk.to_dict(),
                    "score": float(similarities[idx]),
                    "content": chunk.content,
                    "source": chunk.source
                })
        
        return results
    
    def term_coverage(self, query: str, passage: str) -> float:
        """How much of the query's IDF mass the passage actually contains.

        Cosine similarity alone cannot tell a real hit from an accidental one
        on a small corpus: a single shared common word can lift an unrelated
        passage above a genuinely relevant one. Coverage asks the direct
        question instead — are the query's *distinctive* terms in this text?

        Terms outside the corpus vocabulary count fully against coverage:
        they are maximally rare and certainly absent from the passage.
        Treating them as weightless let a query like "capital of France"
        score a perfect 1.0 off the single word "capital".
        """
        # _tokenize already drops stopwords. No length filter: "stage_i" is
        # normalized to one token, but bare numerals and short technical terms
        # still matter, so cutting tokens by length would discard signal.
        query_terms = set(_tokenize(query))
        if not query_terms:
            return 0.0

        max_idf = max(self.idf.values()) if self.idf else 1.0
        passage_terms = set(_tokenize(passage))

        total = sum(self.idf.get(t, max_idf) for t in query_terms)
        if total <= 0:
            return 0.0
        hit = sum(self.idf.get(t, max_idf) for t in query_terms if t in passage_terms)
        return hit / total

    def _save(self):
        """"""
        data_file = self.storage_path / "chunks.json"
        metadata_file = self.storage_path / "metadata.json"
        vectors_file = self.storage_path / "vectors.npy"

        if self.vectors is not None:
            np.save(vectors_file, self.vectors)
        
        #  chunks
        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        # 
        metadata = {
            "vocabulary": self.vocabulary,
            "idf": self.idf,
            "chunk_count": len(self.chunks)
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def _load(self):
        """"""
        data_file = self.storage_path / "chunks.json"
        metadata_file = self.storage_path / "metadata.json"
        
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            self.chunks = [
                DocumentChunk(
                    content=chunk['content'],
                    source=chunk['source'],
                    chunk_index=chunk['chunk_index'],
                    metadata=chunk.get('metadata', {})
                )
                for chunk in chunks_data
            ]
            
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.vocabulary = metadata.get('vocabulary', {})
                    self.idf = metadata.get('idf', {})

            # Reuse the cached TF-IDF matrix when it matches the corpus;
            # only rebuild if the cache is missing or stale.
            vectors_file = self.storage_path / "vectors.npy"
            if (
                vectors_file.exists()
                and self.vocabulary
                and self.idf
            ):
                try:
                    vectors = np.load(vectors_file)
                    if vectors.shape == (len(self.chunks), len(self.vocabulary)):
                        self.vectors = vectors
                        return
                except Exception:
                    pass

            if self.chunks:
                self._rebuild_index()
                np.save(vectors_file, self.vectors)
    
    def get_stats(self) -> Dict:
        """"""
        sources = {}
        for chunk in self.chunks:
            sources[chunk.source] = sources.get(chunk.source, 0) + 1
        
        return {
            "total_chunks": len(self.chunks),
            "sources": sources,
            "vocabulary_size": len(self.vocabulary)
        }
