"""
Vector Tool： / RAG（ SOBC //）
"""
from app.tools.base import BaseTool
from app.core.types import ToolResult, Citation
from app.tools.vector_store import SimpleVectorStore
from app.config import settings
import os


class VectorTool(BaseTool):
    """（RAG）"""
    
    def __init__(self, vector_store: SimpleVectorStore = None):
        super().__init__("vector_tool", " RAG")
        # 
        self.vector_store = vector_store or SimpleVectorStore(
            storage_path=getattr(settings, 'VECTOR_STORE_PATH', 'data/vector_store')
        )
    
    def run(self, query: str, top_k: int = 3, **kwargs) -> ToolResult:
        """"""
        try:
            # 
            results = self.vector_store.search(query, top_k=top_k)
            
            if results:
                citations = []
                passages = []
                
                for result in results:
                    doc = result["doc"]
                    citations.append(Citation(
                        source=doc["source"],
                        passage=doc["content"][:500] + "..." if len(doc["content"]) > 500 else doc["content"],
                        relevance_score=result["score"]
                    ))
                    passages.append(f"[source: {doc['source']}]\n{doc['content']}")
                
                return ToolResult(
                    text="\n\n---\n\n".join(passages),
                    structured={
                        "matches": [r["doc"] for r in results],
                        "scores": [r["score"] for r in results]
                    },
                    citations=citations,
                    success=True
                )
            else:
                return ToolResult(
                    text="",
                    structured={"matches": []},
                    success=True
                )
        
        except Exception as e:
            return ToolResult(
                text=f": {str(e)}",
                success=False,
                error=str(e)
            )
