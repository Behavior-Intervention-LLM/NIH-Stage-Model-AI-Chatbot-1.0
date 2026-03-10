"""
//
"""
from typing import Optional
from app.core.types import Message


class Guardrails:
    """"""
    
    MAX_MESSAGE_LENGTH = 5000
    MAX_RESPONSE_LENGTH = 2000
    FORBIDDEN_PATTERNS = [
        r"<script",
        r"javascript:",
        r"onerror\s*=",
    ]
    
    @classmethod
    def validate_message(cls, message: str) -> tuple[bool, Optional[str]]:
        """Validate message"""
        # 
        if len(message) > cls.MAX_MESSAGE_LENGTH:
            return False, f"（ {cls.MAX_MESSAGE_LENGTH} ）"
        
        # 
        import re
        for pattern in cls.FORBIDDEN_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return False, ""
        
        return True, None
    
    @classmethod
    def sanitize_response(cls, response: str) -> str:
        """Sanitize response"""
        # 
        if len(response) > cls.MAX_RESPONSE_LENGTH:
            response = response[:cls.MAX_RESPONSE_LENGTH] + "..."
        
        return response
    
    @classmethod
    def check_content_policy(cls, content: str) -> bool:
        """"""
        # TODO: 
        return True
