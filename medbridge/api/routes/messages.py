"""
Message routes (Module M-04): JSON, Server-Sent Events and regeneration.

* ``POST /api/sessions/{id}/messages`` — run one turn, return ``MessageResponse``.
* ``POST /api/sessions/{id}/messages/stream`` — the same turn as an SSE stream:
  ``stage`` events while the pipeline runs, then ``meta`` (action, message id,
  redaction count), ``token`` events (``{"text": …}``, a few words every
  ~12 ms — a typewriter effect, since the generator returns validated JSON)
  and finally ``done`` with the full ``MessageResponse``. On failure an
  ``error`` event is followed by a safe ``done``.
* ``POST /api/sessions/{id}/regenerate`` — re-run the last user message
  (JSON, or SSE with ``?stream=true``).

Every route first checks the session (404 unknown, 410 ``SESSION_ERASED`` for a
crypto-shredded session) before any pipeline or LLM work starts; the SSE routes
do this in a pre-check so the error is a normal JSON response, not a stream.

The SSE routes run the orchestrator in its own task with its own database
session (committed when the turn finishes), so a client that disconnects
mid-stream does not abort the turn: it still completes and is persisted.

Concurrency: the orchestrator runs the turns of one session one at a time
(a second message, or a regenerate, sent while a stopped turn is still being
finished waits for it), except that a message hitting the emergency fast-path
is answered immediately. The turn store row-locks the snapshot when writing,
so the BP form and other workers never lose events either.
Wire format: ``event: <type>\\ndata: <json>\\n\\n``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.api.schemas.enums import ActionEnum
from medbridge.api.schemas.requests import MessageRequest
from medbridge.api.schemas.responses import MessageResponse
from medbridge.core.orchestrator import PipelineOrchestrator, get_orchestrator
from medbridge.core.pipeline_types import EventCallback, PipelineEvent
from medbridge.core.templates import PIPELINE_ERROR_TEMPLATE
from medbridge.db.connection import get_session, get_sessionmaker
from medbridge.db.models import Session as SessionModel
from medbridge.privacy.deps import require_session_access
from medbridge.state import turn_store
from medbridge.state.session_manager import SessionErasedError, SessionNotFoundError

logger = structlog.get_logger(__name__)

messages_router = APIRouter(prefix="/sessions", tags=["messages"])

TOKEN_WORDS = 3
TOKEN_DELAY_SECONDS = 0.012
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
STREAM_ERROR_MESSAGE = "Something went wrong while preparing your answer."

# Strong references to in-flight turns so a disconnected client cannot let them be GC'd.
_running_turns: set[asyncio.Task[MessageResponse]] = set()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


async def _ensure_session(db: AsyncSession, session_id: uuid.UUID) -> None:
    """404 for an unknown session, 410 for an erased one — checked before any LLM work."""
    session = await db.get(SessionModel, session_id)
    if session is None:
        raise SessionNotFoundError(session_id)
    if getattr(session, "erased_at", None) is not None:
        raise SessionErasedError(session_id)


async def _ensure_regenerable(db: AsyncSession, session_id: uuid.UUID) -> None:
    if await turn_store.get_last_user_message(db, session_id) is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="There is no previous message to regenerate in this session.",
        )


async def _precheck_stream(session_id: uuid.UUID, *, regenerate: bool) -> None:
    """Validate the session in a short-lived DB session before the stream starts."""
    async with get_sessionmaker()() as db:
        await _ensure_session(db, session_id)
        if regenerate:
            await _ensure_regenerable(db, session_id)


def sse_event(event: str, data: Any) -> str:
    """Encode one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def chunk_tokens(text: str, words: int = TOKEN_WORDS) -> list[str]:
    """Split ``text`` into small word groups; ``"".join(result) == text``."""
    tokens: list[str] = []
    current, count = "", 0
    for piece in re.findall(r"\s+|\S+\s*", text):
        current += piece
        if piece.strip():
            count += 1
            if count >= words:
                tokens.append(current)
                current, count = "", 0
    if current:
        tokens.append(current)
    return tokens


def _safe_response(session_id: uuid.UUID) -> MessageResponse:
    return MessageResponse(
        session_id=session_id,
        response_text=PIPELINE_ERROR_TEMPLATE,
        action=ActionEnum.GENERALIZE,
        citations=[],
        soft_ask_count=0,
        timestamp=datetime.now(UTC),
    )


async def _run_turn_with_own_session(
    orchestrator: PipelineOrchestrator,
    session_id: uuid.UUID,
    message: str,
    on_event: EventCallback,
    regenerate: bool,
) -> MessageResponse:
    async with get_sessionmaker()() as db:
        response = await orchestrator.process_message(
            session_id=session_id,
            message=message,
            db=db,
            on_event=on_event,
            regenerate=regenerate,
        )
        try:
            await db.commit()
        except Exception as exc:  # the answer is still delivered, just not stored
            logger.error(
                "stream_commit_failed", session_id=str(session_id), error=type(exc).__name__
            )
            with contextlib.suppress(Exception):
                await db.rollback()
            response = response.model_copy(update={"message_id": None, "user_message_id": None})
        return response


async def stream_turn(
    orchestrator: PipelineOrchestrator,
    session_id: uuid.UUID,
    message: str,
    *,
    regenerate: bool = False,
) -> AsyncIterator[str]:
    """Run one turn in a background task and yield its SSE events."""
    queue: asyncio.Queue[PipelineEvent | None] = asyncio.Queue()

    async def on_event(event: PipelineEvent) -> None:
        queue.put_nowait(event)

    async def run() -> MessageResponse:
        try:
            return await _run_turn_with_own_session(
                orchestrator, session_id, message, on_event, regenerate
            )
        finally:
            queue.put_nowait(None)

    task = asyncio.create_task(run())
    _running_turns.add(task)
    task.add_done_callback(_running_turns.discard)

    try:
        while (event := await queue.get()) is not None:
            yield sse_event(event.type, event.to_sse_payload())
        # shield: if the client disconnects now, the turn still finishes and persists
        response = await asyncio.shield(task)
    except Exception as exc:
        logger.error("stream_turn_failed", session_id=str(session_id), error=type(exc).__name__)
        yield sse_event("error", {"message": STREAM_ERROR_MESSAGE})
        yield sse_event("done", _safe_response(session_id).model_dump(mode="json"))
        return

    yield sse_event(
        "meta",
        {
            "action": response.action.value,
            "message_id": str(response.message_id) if response.message_id else None,
            "redaction_count": response.redaction_count,
        },
    )
    for token in chunk_tokens(response.response_text):
        yield sse_event("token", {"text": token})
        await asyncio.sleep(TOKEN_DELAY_SECONDS)
    yield sse_event("done", response.model_dump(mode="json"))


def _sse_response(
    orchestrator: PipelineOrchestrator, session_id: uuid.UUID, message: str, *, regenerate: bool
) -> StreamingResponse:
    return StreamingResponse(
        stream_turn(orchestrator, session_id, message, regenerate=regenerate),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------


@messages_router.post(
    "/{session_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_session_access)],
    summary="Submit a patient message to the guarded clinical pipeline",
)
async def handle_send_message(
    session_id: uuid.UUID,
    payload: MessageRequest,
    db: AsyncSession = Depends(get_session),
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
) -> MessageResponse:
    """Run one turn (emergency fast-path, both gates, retrieval, generation) and persist it."""
    await _ensure_session(db, session_id)
    return await orchestrator.process_message(session_id=session_id, message=payload.message, db=db)


@messages_router.post(
    "/{session_id}/messages/stream",
    dependencies=[Depends(require_session_access)],
    summary="Submit a patient message and stream progress + answer as Server-Sent Events",
    response_class=StreamingResponse,
)
async def handle_stream_message(
    session_id: uuid.UUID,
    payload: MessageRequest,
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
) -> StreamingResponse:
    """SSE variant of the message route (``stage``/``meta``/``token``/``done``/``error``)."""
    await _precheck_stream(session_id, regenerate=False)
    return _sse_response(orchestrator, session_id, payload.message, regenerate=False)


@messages_router.post(
    "/{session_id}/regenerate",
    response_model=MessageResponse,
    dependencies=[Depends(require_session_access)],
    summary="Re-run the last user message (JSON, or SSE with ?stream=true)",
)
async def handle_regenerate(
    session_id: uuid.UUID,
    stream: bool = Query(default=False, description="Stream the result as SSE"),
    db: AsyncSession = Depends(get_session),
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
) -> Any:
    """Regenerate the answer to the last user message.

    No new user message and no new clinical events are stored; a new assistant
    message and audit row are written, attached to the re-run user message, and
    the turn restarts from the loop-breaker counter recorded before it. Returns
    409 when there is nothing to regenerate.
    """
    if stream:
        await _precheck_stream(session_id, regenerate=True)
        return _sse_response(orchestrator, session_id, "", regenerate=True)
    await _ensure_session(db, session_id)
    await _ensure_regenerable(db, session_id)
    return await orchestrator.process_message(
        session_id=session_id, message="", db=db, regenerate=True
    )
