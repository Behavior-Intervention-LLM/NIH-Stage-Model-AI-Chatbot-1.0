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
    
    # LLM configuration（ Ollama + ）
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")  # ollama
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct")
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2000
    LLM_TIMEOUT_SECONDS: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    
    # Memory configuration
    SHORT_TERM_LIMIT: int = 20
    SUMMARY_THRESHOLD: int = 10
    
    # configuration（）
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    
    # configuration（）
    VECTOR_DB_URL: Optional[str] = os.getenv("VECTOR_DB_URL")
    VECTOR_DB_API_KEY: Optional[str] = os.getenv("VECTOR_DB_API_KEY")
    
    # configuration
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
    DOCUMENTS_DIR: str = os.getenv("DOCUMENTS_DIR", "data/documents")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# configuration
settings = Settings()
