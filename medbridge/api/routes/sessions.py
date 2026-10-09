"""
Session Management REST Routes (TASK-29, Module M-04; ADL-032/033).

Endpoints:
- POST   /api/sessions                      → 201 ``SessionResponse`` incl. signed ``session_token``
- GET    /api/sessions/{session_id}/history → 200 ``HistoryResponse`` (message ids + explanations)
- DELETE /api/sessions/{session_id}         → 204, right to erasure (crypto-shredding)

Every route below ``/{session_id}`` requires the session token when
``AUTH_REQUIRED`` (``Authorization: Bearer <token>`` or ``X-Session-Token``).

Technical Specification Part II §4.2, §4.3.
"""
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Response, status
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.api.schemas.enums import ActionEnum
from medbridge.api.schemas.responses import (
    CitationResponse,
    Explanation,
    HistoryMessage,
    HistoryResponse,
    SessionResponse,
)
from medbridge.db.connection import get_session
from medbridge.privacy.deps import require_session_access
from medbridge.privacy.tokens import issue_session_token
from medbridge.state import turn_store
from medbridge.state.session_manager import (
    SessionErasedError,
    SessionNotFoundError,
    create_session,
)

sessions_router = APIRouter(prefix="/sessions", tags=["sessions"])

_CITATION_REQUIRED = ("marker", "chunk_id", "source", "section", "excerpt")


def citation_from_dict(data: dict[str, Any]) -> CitationResponse | None:
    """Build a ``CitationResponse`` from a stored dict (legacy rows may lack fields)."""
    try:
        return CitationResponse.model_validate(
            {**{k: "" for k in _CITATION_REQUIRED}, **{k: v for k, v in data.items() if v is not None}}
        )
    except ValidationError:
        return None


def explanation_from_trace(trace: dict[str, Any] | None) -> Explanation | None:
    """Parse a stored trace into an ``Explanation`` (``None`` if absent or malformed)."""
    if not trace:
        return None
    try:
        return Explanation.model_validate(trace)
    except ValidationError:
        return None


async def ensure_session_available(
    db: AsyncSession, session_id: uuid.UUID
) -> turn_store.SessionState:
    """Load the session state; raise 404 for unknown sessions and 410 for erased ones."""
    state = await turn_store.load_session_state(db, session_id)
    if not state.exists:
        raise SessionNotFoundError(session_id)
    if state.erased:
        raise SessionErasedError(session_id)
    return state


@sessions_router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new clinical guidance session",
)
@sessions_router.post(
    "/",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def handle_create_session(
    db: AsyncSession = Depends(get_session),
) -> SessionResponse:
    """Create a session (+ its data-encryption key) and return a signed session token."""
    session_id = await create_session(db)
    token, expires_at = issue_session_token(session_id)
    return SessionResponse(
        session_id=session_id,
        created_at=datetime.now(UTC),
        session_token=token,
        expires_at=expires_at,
    )


@sessions_router.get(
    "/{session_id}/history",
    response_model=HistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get conversation message history for a session",
    dependencies=[Depends(require_session_access)],
)
async def handle_get_session_history(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_session),
) -> HistoryResponse:
    """Return the decrypted, chronological history with per-answer explanations."""
    await ensure_session_available(db, session_id)
    rows = await turn_store.get_history(db, session_id)
    messages = [
        HistoryMessage(
            role=r.role,
            content=r.content,
            action=ActionEnum(r.action) if r.action else None,
            citations=[
                c for c in (citation_from_dict(d) for d in r.citations if isinstance(d, dict))
                if c is not None
            ],
            timestamp=r.created_at,
            message_id=r.message_id,
            explanation=explanation_from_trace(r.trace),
        )
        for r in rows
    ]
    return HistoryResponse(session_id=session_id, messages=messages)


@sessions_router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Erase a session (crypto-shredding, right to erasure)",
    dependencies=[Depends(require_session_access)],
)
async def handle_delete_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_session),
) -> Response:
    """Destroy the session key and mutable rows; audit/event rows become unreadable."""
    await turn_store.erase_session(db, session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
