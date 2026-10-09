"""
Admin / evaluation dashboard routes (ADL-034).

- POST /api/admin/login  ``{username, password}`` → ``{token, expires_at}``
  (constant-time comparison; rate limited to 5/min per IP by the security middleware)
- GET  /api/admin/audit?limit=50 → recent audit rows. Request/response previews are
  decrypted with the session key and passed through the PII redactor; rows of
  erased sessions show ``[erased]``.

The dashboard is disabled (403) while ``ADMIN_PASSWORD`` is empty.
"""
import hashlib
import hmac
import uuid
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.config import get_settings
from medbridge.db.connection import get_session
from medbridge.privacy.deps import require_admin
from medbridge.privacy.redaction import redact
from medbridge.privacy.tokens import issue_admin_token_with_expiry
from medbridge.state import turn_store

logger = structlog.get_logger(__name__)

admin_router = APIRouter(prefix="/admin", tags=["admin"])

PREVIEW_CHARS = 160
ERASED_MARKER = "[erased]"


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1, max_length=256)


class LoginResponse(BaseModel):
    token: str
    expires_at: datetime


class AuditItem(BaseModel):
    audit_id: uuid.UUID
    session_ref: str = Field(..., description="First 8 characters of the session id")
    created_at: datetime
    final_action: str
    gate_1_action: str | None = None
    gate_2_action: str | None = None
    latency_ms: int | None = None
    evidence_chunks: int = 0
    request_preview: str
    response_preview: str
    erased: bool = False


class AuditListResponse(BaseModel):
    items: list[AuditItem]
    count: int


def _digest_equal(supplied: str, expected: str) -> bool:
    """Constant-time comparison that also hides the expected length."""
    return hmac.compare_digest(
        hashlib.sha256(supplied.encode("utf-8")).digest(),
        hashlib.sha256(expected.encode("utf-8")).digest(),
    )


def preview(text: str | None) -> str:
    """Redacted single-line preview of decrypted text (``[erased]`` when shredded)."""
    if text is None:
        return ERASED_MARKER
    flat = " ".join(redact(text).text.split())
    return flat if len(flat) <= PREVIEW_CHARS else flat[: PREVIEW_CHARS - 1] + "…"


@admin_router.post("/login", response_model=LoginResponse, summary="Dashboard login")
async def handle_admin_login(payload: LoginRequest) -> LoginResponse:
    settings = get_settings()
    if not settings.ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin dashboard disabled (ADMIN_PASSWORD is not set).",
        )
    username_ok = _digest_equal(payload.username, settings.ADMIN_USERNAME)
    password_ok = _digest_equal(payload.password, settings.ADMIN_PASSWORD)
    if not (username_ok and password_ok):
        logger.warning("admin_login_failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )
    token, expires_at = issue_admin_token_with_expiry(settings.ADMIN_USERNAME)
    logger.info("admin_login_succeeded")
    return LoginResponse(token=token, expires_at=expires_at)


@admin_router.get(
    "/audit",
    response_model=AuditListResponse,
    summary="Recent audit-log entries (decrypted + redacted previews)",
    dependencies=[Depends(require_admin)],
)
async def handle_admin_audit(
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_session),
) -> AuditListResponse:
    entries = await turn_store.list_audit_entries(db, limit)
    items = [
        AuditItem(
            audit_id=e.audit_id,
            session_ref=str(e.session_id)[:8],
            created_at=e.created_at,
            final_action=e.final_action,
            gate_1_action=e.gate_1_action,
            gate_2_action=e.gate_2_action,
            latency_ms=e.latency_ms,
            evidence_chunks=e.evidence_chunk_count,
            request_preview=preview(e.request_text),
            response_preview=preview(e.response_text),
            erased=e.erased,
        )
        for e in entries
    ]
    return AuditListResponse(items=items, count=len(items))
