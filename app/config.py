"""
Configuration: env vars and model settings
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App settings"""
    
    # API configuration
    API_TITLE: str = "NIH Stage Model AI Chatbot"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # LLM configuration (vLLM only)
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct")
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2000
    LLM_TIMEOUT_SECONDS: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
    
    # Memory configuration
    SHORT_TERM_LIMIT: int = 20
    SUMMARY_THRESHOLD: int = 10
    SUMMARY_REFRESH_EVERY_TURNS: int = int(os.getenv("SUMMARY_REFRESH_EVERY_TURNS", "6"))
    LONG_TERM_MEMORY_WINDOW: int = int(os.getenv("LONG_TERM_MEMORY_WINDOW", "50"))
    LONG_TERM_MEMORY_MAX_LINES: int = int(os.getenv("LONG_TERM_MEMORY_MAX_LINES", "8"))
    MEMORY_CONTEXT_MAX_CHARS: int = int(os.getenv("MEMORY_CONTEXT_MAX_CHARS", "6000"))
    
    # Optional storage configuration
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    STATE_TTL_SECONDS: int = int(os.getenv("STATE_TTL_SECONDS", "604800"))  # 7 days
    REDIS_KEY_PREFIX: str = os.getenv("REDIS_KEY_PREFIX", "nih_chatbot")
    
    # Optional external vector DB configuration
    VECTOR_DB_URL: Optional[str] = os.getenv("VECTOR_DB_URL")
    VECTOR_DB_API_KEY: Optional[str] = os.getenv("VECTOR_DB_API_KEY")
    
    # Local paths
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
    DOCUMENTS_DIR: str = os.getenv("DOCUMENTS_DIR", "data/documents")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings singleton
settings = Settings()
