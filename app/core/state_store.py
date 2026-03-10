"""
（， redis）
"""
from typing import Dict, Optional
from app.core.types import SessionState
import uuid


class StateStore:
    """（）"""
    
    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}
    
    def get_state(self, session_id: str) -> Optional[SessionState]:
        """"""
        return self._sessions.get(session_id)
    
    def create_state(self, session_id: Optional[str] = None) -> SessionState:
        """"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        state = SessionState(session_id=session_id)
        self._sessions[session_id] = state
        return state
    
    def save_state(self, state: SessionState):
        """"""
        self._sessions[state.session_id] = state
    
    def delete_state(self, session_id: str):
        """"""
        if session_id in self._sessions:
            del self._sessions[session_id]
    
    def list_sessions(self) -> list[str]:
        """ID"""
        return list(self._sessions.keys())


# 
state_store = StateStore()
