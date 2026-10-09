from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # LLM
    GROQ_API_KEY: str
    OPENAI_API_KEY: str = ""
    LLM_PROVIDER: str = "groq"
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
    LLM_MODEL: str = "llama-3.1-8b-instant"
    LLM_TIMEOUT: float = 90.0
    LLM_TEMPERATURE_GATES: float = 0.0
    LLM_TEMPERATURE_GENERATOR: float = 0.3
    LLM_MAX_TOKENS_GATES: int = 512
    LLM_MAX_TOKENS_GENERATOR: int = 1024
    LLM_SEED: int = 42
    LLM_MAX_RETRIES: int = 3
    LLM_RETRY_BASE_DELAY: float = 1.0

    # Database
    POSTGRES_PASSWORD: str
    DATABASE_URL: str = "postgresql+asyncpg://medbridge_app:password@127.0.0.1:5432/medbridge"

    # Qdrant
    QDRANT_URL: str = "http://127.0.0.1:6333"
    QDRANT_COLLECTION: str = "clinical_guidelines"

    # Retrieval
    RETRIEVAL_TOP_K_CANDIDATES: int = 20
    RETRIEVAL_TOP_K_RERANKED: int = 5
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    SPARSE_MODEL: str = "Qdrant/bm25"

    # Safety
    SOFT_ASK_MAX_COUNT: int = 2

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["http://localhost:8501"]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

@lru_cache()
def get_settings() -> Settings:
    return Settings()
