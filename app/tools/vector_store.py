"""
：
"""
import os
import json
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from app.tools.document_loader import DocumentChunk


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
    
    def add_chunks(self, chunks: List[DocumentChunk]):
        """"""
        self.chunks.extend(chunks)
        self._rebuild_index()
        self._save()
    
    def _rebuild_index(self):
        """（TF-IDF）"""
        if not self.chunks:
            return
        
        # 
        def tokenize(text: str) -> List[str]:
            import re
            # （）
            tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', text.lower())
            return tokens
        
        # 
        all_tokens = []
        for chunk in self.chunks:
            tokens = tokenize(chunk.content)
            all_tokens.append(tokens)
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
        
        #  IDF
        doc_count = len(self.chunks)
        for token, token_id in self.vocabulary.items():
            df = sum(1 for tokens in all_tokens if token in tokens)
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
        import re
        def tokenize(text: str) -> List[str]:
            tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', text.lower())
            return tokens
        
        query_tokens = tokenize(query)
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
    
    def _save(self):
        """"""
        data_file = self.storage_path / "chunks.json"
        metadata_file = self.storage_path / "metadata.json"
        
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
            
            # 
            if self.chunks:
                self._rebuild_index()
    
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
