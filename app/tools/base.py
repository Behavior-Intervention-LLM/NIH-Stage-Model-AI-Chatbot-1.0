"""
Tool 
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from app.core.types import ToolResult, Artifact


class BaseTool(ABC):
    """"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """"""
        pass
    
    def to_artifact(self, result: ToolResult) -> Artifact:
        """ Artifact"""
        return Artifact(
            tool_name=self.name,
            result_type="text" if result.text else "structured" if result.structured else "raw",
            content=result.text or result.structured or result.raw,
            citations=result.citations,
            metadata={"success": result.success, "error": result.error}
        )


class ToolRegistry:
    """"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        """"""
        self._tools[tool.name] = tool
    
    def run_tool(self, tool_name: str, args: Dict[str, Any]) -> Artifact:
        """"""
        if tool_name not in self._tools:
            raise ValueError(f" {tool_name} ")
        
        tool = self._tools[tool_name]
        result = tool.run(**args)
        return tool.to_artifact(result)
    
    def list_tools(self) -> list[str]:
        """"""
        return list(self._tools.keys())
