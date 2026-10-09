"""
API Response Pydantic Models.

Defines validated response schemas for all outbound API responses.
All schemas follow Technical Specification Part II §5.1. Fields added for the
final release are optional so existing clients (Streamlit, tests) keep working.
"""
from datetime import datetime
from typing import Any, Literal, Optional
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
    guideline_id: Optional[str] = Field(default=None, description="Corpus document id")
    title: Optional[str] = Field(default=None, description="Full guideline title")
    org: Optional[str] = Field(default=None, description="Issuing organisation")
    year: Optional[int] = Field(default=None, description="Publication year")
    page: Optional[int] = Field(default=None, description="Source page number")
    url: Optional[str] = Field(default=None, description="Source URL")


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
    year: Optional[int] = None
    section: str = ""
    page: Optional[int] = None
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

    gate1: Optional[GateDecision] = None
    gate2: Optional[GateDecision] = None
    gate3: Optional[dict[str, Any]] = None
    emergency: Optional[dict[str, Any]] = None
    retrieval: list[RetrievedEvidence] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
    total_ms: float = 0.0
    speculative_retrieval: Optional[dict[str, Any]] = None
    citation_validation: Optional[dict[str, Any]] = None
    redactions: int = 0
    redaction_categories: dict[str, int] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)
    llm_calls: list[dict[str, Any]] = Field(default_factory=list)
    ablation: Optional[str] = None
    regenerated: bool = False


class MessageResponse(BaseModel):
    """Response body for POST /api/sessions/{session_id}/messages."""

    session_id: UUID
    response_text: str
    action: ActionEnum
    citations: list[CitationResponse] = Field(default_factory=list)
    soft_ask_count: int = Field(default=0, ge=0)
    timestamp: datetime
    message_id: Optional[UUID] = Field(default=None, description="Assistant message id")
    user_message_id: Optional[UUID] = Field(default=None, description="Stored (redacted) user message id")
    redaction_count: int = Field(default=0, ge=0)
    explanation: Optional[Explanation] = None


class SessionResponse(BaseModel):
    """Response body for POST /api/sessions (session creation)."""

    session_id: UUID
    created_at: datetime
    session_token: Optional[str] = Field(default=None, description="Signed bearer token for this session")
    expires_at: Optional[datetime] = None


class HistoryMessage(BaseModel):
    """A single message entry in the conversation history."""

    role: str = Field(..., description="Message sender: 'user' or 'assistant'")
    content: str
    action: Optional[ActionEnum] = None
    citations: list[CitationResponse] = Field(default_factory=list)
    timestamp: datetime
    message_id: Optional[UUID] = None
    explanation: Optional[Explanation] = None


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
    ollama_reachable: Optional[bool] = None
    llm_provider: Optional[str] = None
    models: dict[str, Any] = Field(default_factory=dict)
    models_loaded: dict[str, bool] = Field(default_factory=dict)
    qdrant_points: Optional[int] = None
    local_only_mode: Optional[bool] = None
    version: Optional[str] = None


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
    details: Optional[Any] = Field(
        default=None,
        description="Optional field-level or contextual error details",
    )
