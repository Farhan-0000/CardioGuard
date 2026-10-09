"""
API Response Pydantic Models.

Defines validated response schemas for all outbound API responses.
All schemas follow Technical Specification Part II §5.1. Fields added for the
final release are optional so existing clients (Streamlit, tests) keep working.
"""
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from medbridge.api.schemas.enums import ActionEnum


class CitationResponse(BaseModel):
    """A single inline citation referencing a clinical guideline chunk."""

    marker: str = Field(..., description="Citation marker, e.g. '[1]'")
    chunk_id: str = Field(..., description="Unique chunk identifier, e.g. 'aha_2025_s8_c3'")
    source: str = Field(..., description="Guideline source, e.g. 'AHA/ACC 2025'")
    section: str = Field(..., description="Section reference, e.g. 'Section 8.2'")
    excerpt: str = Field(..., description="Truncated evidence text excerpt")
    guideline_id: str | None = Field(default=None, description="Corpus document id")
    title: str | None = Field(default=None, description="Full guideline title")
    org: str | None = Field(default=None, description="Issuing organisation")
    year: int | None = Field(default=None, description="Publication year")
    page: int | None = Field(default=None, description="Source page number")
    url: str | None = Field(default=None, description="Source URL")


# ----------------------------------------------------------------------
# "Why this answer?" explanation (the Dual Answerability Gate made visible)
# ----------------------------------------------------------------------

GateSource = Literal[
    "llm",            # decided by the gate LLM
    "fallback",       # gate LLM failed permanently → spec fallback
    "loop_breaker",   # soft_ask_count >= SOFT_ASK_MAX_COUNT → forced GENERALIZE
    "fastpath",       # deterministic emergency classifier
    "rule",           # deterministic rule (small talk, citation downgrade, grounding verifier)
    "calibrated",     # V1 calibrated evidence gate
    "ablation",       # gate disabled by an evaluation ablation preset
]


class GateDecision(BaseModel):
    """One gate decision with its rationale."""

    action: str
    rationale: str = ""
    missing_fields: list[str] = Field(default_factory=list)
    source: GateSource = "llm"
    votes: list[str] = Field(default_factory=list)
    features: dict[str, Any] = Field(default_factory=dict)


class RetrievedEvidence(BaseModel):
    """A reranked evidence chunk shown in the explanation panel."""

    rank: int
    chunk_id: str
    guideline_id: str = ""
    title: str = ""
    org: str = ""
    year: int | None = None
    section: str = ""
    page: int | None = None
    score_rrf: float = 0.0
    score_rerank: float = 0.0
    excerpt: str = ""
    cited: bool = False


class StageTiming(BaseModel):
    """Wall-clock duration of one pipeline stage."""

    stage: str
    ms: float
    status: Literal["ok", "skipped", "failed", "discarded"] = "ok"


class Explanation(BaseModel):
    """Structured trace of how the answer was produced."""

    gate1: GateDecision | None = None
    gate2: GateDecision | None = None
    gate3: dict[str, Any] | None = None
    emergency: dict[str, Any] | None = None
    retrieval: list[RetrievedEvidence] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
    total_ms: float = 0.0
    speculative_retrieval: dict[str, Any] | None = None
    citation_validation: dict[str, Any] | None = None
    redactions: int = 0
    redaction_categories: dict[str, int] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)
    ablation: str | None = None
    regenerated: bool = False
    soft_ask_count_before: int | None = Field(
        default=None,
        ge=0,
        description="Loop-breaker counter before this turn's Gate 1 (a regenerate restarts "
        "from the replaced turn's value)",
    )
    retrieval_status: Literal["ok", "empty", "failed", "skipped", "discarded"] | None = Field(
        default=None,
        description="'failed' = the retriever could not run (degraded turn; Gate 2 decided on "
        "no evidence), 'empty' = it ran and nothing matched",
    )
    effective_question: dict[str, Any] | None = Field(
        default=None,
        description="Set when the gates and generator saw the earlier SOFT-ASK question plus "
        "this follow-up (counts only, never text)",
    )


class MessageResponse(BaseModel):
    """Response body for POST /api/sessions/{session_id}/messages."""

    session_id: UUID
    response_text: str
    action: ActionEnum
    citations: list[CitationResponse] = Field(default_factory=list)
    soft_ask_count: int = Field(default=0, ge=0)
    timestamp: datetime
    message_id: UUID | None = Field(default=None, description="Assistant message id")
    user_message_id: UUID | None = Field(
        default=None, description="Stored (redacted) user message id"
    )
    redaction_count: int = Field(default=0, ge=0)
    explanation: Explanation | None = None


class SessionResponse(BaseModel):
    """Response body for POST /api/sessions (session creation)."""

    session_id: UUID
    created_at: datetime
    session_token: str | None = Field(
        default=None, description="Signed bearer token for this session"
    )
    expires_at: datetime | None = None


class HistoryMessage(BaseModel):
    """A single message entry in the conversation history."""

    role: str = Field(..., description="Message sender: 'user' or 'assistant'")
    content: str
    action: ActionEnum | None = None
    citations: list[CitationResponse] = Field(default_factory=list)
    timestamp: datetime
    message_id: UUID | None = None
    explanation: Explanation | None = None


class HistoryResponse(BaseModel):
    """Response body for GET /api/sessions/{session_id}/history."""

    session_id: UUID
    messages: list[HistoryMessage] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(..., description="'healthy', 'degraded' or 'unhealthy'")
    postgres_connected: bool
    qdrant_connected: bool
    timestamp: datetime
    ollama_reachable: bool | None = None
    llm_provider: str | None = None
    models: dict[str, Any] = Field(default_factory=dict)
    models_loaded: dict[str, bool] = Field(default_factory=dict)
    qdrant_points: int | None = None
    local_only_mode: bool | None = None
    version: str | None = None


class ErrorResponse(BaseModel):
    """
    Standardised error response body.

    Constitution §9 (NON-NEGOTIABLE #7): The backend must never return
    raw 500 errors. This schema guarantees a structured, safe response.
    """

    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error description")
    safe_fallback: str = Field(
        ...,
        description="Pre-vetted clinically safe fallback message for the patient",
    )
    details: Any | None = Field(
        default=None,
        description="Optional field-level or contextual error details",
    )
