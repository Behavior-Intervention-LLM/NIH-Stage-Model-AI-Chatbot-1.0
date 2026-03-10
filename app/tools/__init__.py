"""
Tools 
"""
from app.tools.base import ToolRegistry
from app.tools.db_tool import DBTool
from app.tools.vector_tool import VectorTool
from app.tools.versioned_rag_tool import VersionedRAGTool
from app.tools.vector_store import SimpleVectorStore
from app.config import settings

# （）
vector_store = SimpleVectorStore(storage_path=settings.VECTOR_STORE_PATH)

# 
tool_registry = ToolRegistry()
tool_registry.register(DBTool())
tool_registry.register(VectorTool(vector_store=vector_store))
tool_registry.register(VersionedRAGTool(vector_store=vector_store))