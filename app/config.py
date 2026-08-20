"""
Application settings loaded from environment variables.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # API
    API_TITLE: str = "NIH Stage Model AI Chatbot"
    API_VERSION: str = "1.0.1"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Auth: set AUTH_DISABLED=true to skip login checks (local development only)
    AUTH_DISABLED: bool = os.getenv("AUTH_DISABLED", "False").lower() == "true"

    # LLM provider: "openai" | "anthropic" | "ollama" | "groq"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-5.5")
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2000
    # Output cap for composition turns (essays, summaries, reports);
    # LLM_MAX_TOKENS is sized for conversational answers and truncates them.
    LLM_COMPOSE_MAX_TOKENS: int = int(os.getenv("LLM_COMPOSE_MAX_TOKENS", "4000"))
    # Fast/cheap model for routing-style calls (intent classification).
    # Falls back to LLM_MODEL when unset. OpenAI provider only.
    # Must be a non-reasoning model — reasoning models (gpt-5* family) spend
    # seconds thinking, which defeats the purpose for a routing classifier.
    LLM_INTENT_MODEL: Optional[str] = os.getenv("LLM_INTENT_MODEL", "gpt-4.1-mini")
    LLM_TIMEOUT_SECONDS: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

    # Ollama (local)
    # OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

    # Anthropic
    # ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    # ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    # OpenAI
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL")

    # Groq (OpenAI-compatible, free tier)
    # GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    # GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # QDRANT
    QDRANT_API_KEY: Optional[str] = os.getenv('QDRANT_API_KEY')
    QDRANT_URL: Optional[str] = os.getenv('QDRANT_URL')
    
    # Memory
    SHORT_TERM_LIMIT: int = 20
    SUMMARY_THRESHOLD: int = 10

    # Database
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")

    # Vector store (local TF-IDF)
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
    DOCUMENTS_DIR: str = os.getenv("DOCUMENTS_DIR", "data/documents")

    # External vector DB (optional)
    VECTOR_DB_URL: Optional[str] = os.getenv("VECTOR_DB_URL")
    VECTOR_DB_API_KEY: Optional[str] = os.getenv("VECTOR_DB_API_KEY")

    # Attached files (chat upload). These are conversation working context —
    # never indexed into the vector store, never persisted, dropped with the
    # session. One budget governs how much reaches the model, replacing four
    # inconsistent caps (3500/file, 12000 merged, 15000 stored, 4500 sent)
    # whose tightest limit was applied last and silently discarded most of a
    # multi-file upload.
    ATTACHMENT_MAX_CHARS: int = int(os.getenv("ATTACHMENT_MAX_CHARS", "24000"))

    # Retrieval loop (orchestrator: rag_plan → assess_evidence → retry?)
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
    # Total retrieval attempts per turn. 1 disables reformulation entirely.
    RAG_MAX_ATTEMPTS: int = int(os.getenv("RAG_MAX_ATTEMPTS", "2"))
    # A passage counts as evidence only if it clears BOTH floors.
    #
    # Calibrated over 20 on-topic and 10 off-topic queries against the rebuilt
    # index (179 chunks): on-topic bottoms out at cosine 0.138 / coverage
    # 0.290, so these floors reject no on-topic query measured while catching
    # 8 of 10 off-topic ones. The two that pass ("capital of France",
    # "who won the world cup") share real corpus vocabulary; raising coverage
    # to catch them costs on-topic recall, which is the worse error — the
    # responder still sees the passages and can judge them.
    #
    # Neither signal works alone: cosine alone ranks "who won the world cup"
    # (0.278) above most real Stage questions.
    #
    # Recalibrate if the corpus changes substantially.
    RAG_MIN_RELEVANCE: float = float(os.getenv("RAG_MIN_RELEVANCE", "0.10"))
    RAG_MIN_COVERAGE: float = float(os.getenv("RAG_MIN_COVERAGE", "0.25"))
    # Use the cheap LLM to rewrite a failed query. Falls back to the
    # deterministic keyword ladder when disabled or when the call fails.
    RAG_LLM_REFORMULATION: bool = os.getenv("RAG_LLM_REFORMULATION", "True").lower() == "true"

    # Implicit feedback system (app/feedback/) — no human ratings are collected.
    FEEDBACK_ENABLED: bool = os.getenv("FEEDBACK_ENABLED", "True").lower() == "true"
    # LLM-as-judge pass over each completed turn. Runs off the request path.
    FEEDBACK_JUDGE_ENABLED: bool = os.getenv("FEEDBACK_JUDGE_ENABLED", "True").lower() == "true"
    # Falls back to LLM_INTENT_MODEL (cheap, non-reasoning) when unset.
    FEEDBACK_JUDGE_MODEL: Optional[str] = os.getenv("FEEDBACK_JUDGE_MODEL")
    # Let learned document weights influence retrieval ranking.
    FEEDBACK_ADAPTIVE_RETRIEVAL: bool = os.getenv("FEEDBACK_ADAPTIVE_RETRIEVAL", "True").lower() == "true"
    # A document needs this many scored turns before its weight leaves 1.0.
    FEEDBACK_MIN_OBSERVATIONS: int = int(os.getenv("FEEDBACK_MIN_OBSERVATIONS", "3"))
    # Weights are clamped to [1 - span, 1 + span]: learned preference breaks
    # near-ties, it never overrides a strong semantic match.
    FEEDBACK_WEIGHT_SPAN: float = float(os.getenv("FEEDBACK_WEIGHT_SPAN", "0.4"))
    FEEDBACK_WEIGHT_GAIN: float = float(os.getenv("FEEDBACK_WEIGHT_GAIN", "1.5"))
    # Full relearn cadence, in turns. 0 disables automatic recomputation.
    FEEDBACK_RECOMPUTE_EVERY_TURNS: int = int(os.getenv("FEEDBACK_RECOMPUTE_EVERY_TURNS", "25"))
    # Comma-separated usernames allowed to read /analytics/*. When unset,
    # access requires AUTH_DISABLED (local dev only).
    ANALYTICS_ADMIN_USERS: str = os.getenv("ANALYTICS_ADMIN_USERS", "")

    # AWS S3 storage (optional — used to sync data/ on container startup)
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: Optional[str] = os.getenv("S3_BUCKET_NAME")
    S3_DATA_PREFIX: str = os.getenv("S3_DATA_PREFIX", "data/")



    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
