"""
Application configuration (Module M-01).

All settings are loaded from environment variables / ``.env`` via
pydantic-settings. Defaults are chosen so that a fresh checkout runs fully
locally (Ollama + Docker) with no paid API and no data leaving the machine
(``LOCAL_ONLY_MODE=true``).
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # LLM inference
    # ------------------------------------------------------------------
    LLM_PROVIDER: Literal["ollama", "groq", "openai"] = "ollama"
    # When true, remote providers (Groq/OpenAI) are never contacted even if
    # keys are configured: no patient text leaves the machine.
    LOCAL_ONLY_MODE: bool = True
    # Native Ollama endpoint (``/api/chat``). A trailing ``/v1`` is tolerated.
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    LLM_MODEL: str = "qwen3.5:4b"
    # Optional per-role overrides (empty string = use LLM_MODEL).
    LLM_MODEL_GATES: str = ""
    LLM_MODEL_GENERATOR: str = ""
    LLM_KEEP_ALIVE: str = "30m"
    LLM_NUM_CTX: int = 8192
    # Disable "thinking" traces on hybrid reasoning models (qwen3/qwen3.5).
    LLM_THINK: bool = False
    LLM_TIMEOUT: float = 120.0
    LLM_TEMPERATURE_GATES: float = 0.0
    LLM_TEMPERATURE_GENERATOR: float = 0.3
    LLM_MAX_TOKENS_GATES: int = 512
    LLM_MAX_TOKENS_GENERATOR: int = 900
    LLM_SEED: int = 42
    LLM_MAX_RETRIES: int = 3
    LLM_RETRY_BASE_DELAY: float = 1.0
    # Optional remote providers (ignored when LOCAL_ONLY_MODE=true).
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    POSTGRES_PASSWORD: str = ""
    DATABASE_URL: str = "postgresql+asyncpg://medbridge_app:change-me@127.0.0.1:5433/medbridge"

    # ------------------------------------------------------------------
    # Vector store & retrieval
    # ------------------------------------------------------------------
    QDRANT_URL: str = "http://127.0.0.1:6333"
    QDRANT_COLLECTION: str = "clinical_guidelines"
    RETRIEVAL_TOP_K_CANDIDATES: int = 20
    RETRIEVAL_TOP_K_RERANKED: int = 5
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    SPARSE_MODEL: str = "Qdrant/bm25"
    RETRIEVAL_CACHE_SIZE: int = 512
    RETRIEVAL_CACHE_TTL_SECONDS: int = 900
    # Launch retrieval concurrently with Gate 1; discarded on SOFT-ASK (ADL-027).
    SPECULATIVE_RETRIEVAL: bool = True
    # Evidence text per chunk passed to LLM prompts (characters).
    EVIDENCE_MAX_CHARS_PER_CHUNK: int = 2400

    # ------------------------------------------------------------------
    # Pipeline safety behaviour
    # ------------------------------------------------------------------
    SOFT_ASK_MAX_COUNT: int = 2
    EXTRACTOR_HISTORY_TURNS: int = 4
    CITATION_VALIDATION_ENABLED: bool = True
    SMALL_TALK_ENABLED: bool = True
    # V1 — calibrated Evidence Gate (quantitative sufficiency signal + votes).
    EVIDENCE_GATE_CALIBRATION_ENABLED: bool = False
    EVIDENCE_GATE_VOTES: int = 3
    EVIDENCE_MIN_RERANK_ANSWER: float = 0.0
    EVIDENCE_MIN_RERANK_ANY: float = -6.0
    EVIDENCE_MIN_ENTITY_COVERAGE: float = 0.0
    # V2 — post-generation Grounding Verifier ("Gate 3").
    GROUNDING_VERIFIER_ENABLED: bool = False
    GROUNDING_NLI_MODEL: str = "cross-encoder/nli-MiniLM2-L6-H768"
    GROUNDING_ENTAILMENT_THRESHOLD: float = 0.5
    GROUNDING_MAX_UNSUPPORTED_FRACTION: float = 0.34
    WARMUP_ON_STARTUP: bool = True

    # ------------------------------------------------------------------
    # Privacy & security
    # ------------------------------------------------------------------
    PII_REDACTION_ENABLED: bool = True
    ENCRYPTION_ENABLED: bool = True
    # Fernet key (urlsafe base64, 32 bytes). Generate with scripts/setup_env.py.
    ENCRYPTION_MASTER_KEY: str = ""
    SESSION_TOKEN_SECRET: str = ""
    SESSION_TOKEN_TTL_SECONDS: int = 7 * 24 * 3600
    AUTH_REQUIRED: bool = True
    ADMIN_USERNAME: str = "admin"
    # Empty password disables the admin dashboard login.
    ADMIN_PASSWORD: str = ""
    RATE_LIMIT_PER_MINUTE: int = 30
    MAX_REQUEST_BYTES: int = 32_768
    DATA_RETENTION_DAYS: int = 30

    # ------------------------------------------------------------------
    # Server & UI
    # ------------------------------------------------------------------
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = [
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://localhost:8501",
    ]
    APP_NAME: str = "MedBridge"
    EMERGENCY_NUMBER: str = "112"
    AMBULANCE_NUMBER: str = "108"
    EMERGENCY_REGION_LABEL: str = "India"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def gates_model(self) -> str:
        """Model used for the extractor and both gates."""
        return self.LLM_MODEL_GATES or self.LLM_MODEL

    @property
    def generator_model(self) -> str:
        """Model used for the response generator."""
        return self.LLM_MODEL_GENERATOR or self.LLM_MODEL


@lru_cache
def get_settings() -> Settings:
    return Settings()
