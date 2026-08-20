"""
Tools 
"""
from app.tools.base import ToolRegistry
from app.tools.vector_tool import VectorTool
from app.tools.versioned_rag_tool import VersionedRAGTool
from app.tools.vector_store import SimpleVectorStore
from app.config import settings

# Shared across both retrieval tools so the index is loaded once.
vector_store = SimpleVectorStore(storage_path=settings.VECTOR_STORE_PATH)

# DBTool was removed: it held a hardcoded stage table with no database behind
# it, and app/core/stage_model.py now serves that data in-process.
tool_registry = ToolRegistry()
tool_registry.register(VectorTool(vector_store=vector_store))
tool_registry.register(VersionedRAGTool(vector_store=vector_store))