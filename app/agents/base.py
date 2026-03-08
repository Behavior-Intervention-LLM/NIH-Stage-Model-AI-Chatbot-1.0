"""
Base Agent 
"""
from abc import ABC, abstractmethod
from app.core.types import SessionState, AgentOutput


class BaseAgent(ABC):
    """Agent """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def run(self, state: SessionState, user_message: str, context: str = "") -> AgentOutput:
        """
         Agent
        
        Args:
            state: 
            user_message: 
            context: （ MemoryManager ）
        
        Returns:
            AgentOutput: Agent 
        """
        pass
    
    def update_state(self, state: SessionState, output: AgentOutput):
        """ Agent （，）"""
        pass
