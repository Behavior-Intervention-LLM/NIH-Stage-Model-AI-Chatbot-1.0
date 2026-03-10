"""
Memory ：///slots
"""
from typing import List, Optional
from app.core.types import SessionState, Message, MessageRole


class MemoryManager:
    """Memory """
    
    def __init__(self, short_term_limit: int = 20, summary_threshold: int = 10):
        self.short_term_limit = short_term_limit
        self.summary_threshold = summary_threshold
    
    def get_short_term_memory(self, state: SessionState) -> List[Message]:
        """（ N ）"""
        return state.get_recent_messages(self.short_term_limit)
    
    def get_summary(self, state: SessionState) -> Optional[str]:
        """"""
        return state.summary
    
    def should_summarize(self, state: SessionState) -> bool:
        """"""
        return len(state.messages) >= self.summary_threshold and state.summary is None
    
    def create_summary(self, state: SessionState) -> str:
        """（， LLM）"""
        # TODO:  LLM 
        recent_messages = self.get_short_term_memory(state)
        summary_parts = []
        
        for msg in recent_messages[:10]:  # 10
            if msg.role == MessageRole.USER:
                summary_parts.append(f": {msg.content[:100]}")
            elif msg.role == MessageRole.ASSISTANT:
                summary_parts.append(f"assistant: {msg.content[:100]}")
        
        return "\n".join(summary_parts)
    
    def update_summary(self, state: SessionState, summary: str):
        """"""
        state.summary = summary
    
    def get_context_for_agent(self, state: SessionState, include_summary: bool = True) -> str:
        """ Agent """
        context_parts = []
        
        # 
        if include_summary and state.summary:
            context_parts.append(f":\n{state.summary}\n")
        
        #  slots 
        slots_info = []
        if state.slots.need_stage is not None:
            slots_info.append(f" Stage: {state.slots.need_stage}")
        if state.slots.stage:
            slots_info.append(f" Stage: {state.slots.stage} (confidence: {state.slots.stage_confidence:.2f})")
        if state.slots.user_goal:
            slots_info.append(f": {state.slots.user_goal}")
        
        if slots_info:
            context_parts.append(f":\n" + "\n".join(slots_info) + "\n")
        
        # 
        recent = self.get_short_term_memory(state)
        if recent:
            messages_text = "\n".join([
                f"{msg.role.value}: {msg.content}" 
                for msg in recent[-5:]  # 5
            ])
            context_parts.append(f":\n{messages_text}")
        
        return "\n".join(context_parts)


# 
memory_manager = MemoryManager()
