"""
Turn Store — consolidated, encrypted persistence for one pipeline turn (Module M-05).

CONTRACT (implemented by the persistence/privacy work-stream):

* The orchestrator never writes rows directly. At the end of each turn it calls
  ``persist_turn()`` exactly once; all inserts (clinical events, snapshot upsert,
  soft-ask counter, user + assistant messages, audit log) are added to the
  session and flushed ONCE. The caller (FastAPI dependency / orchestrator)
  commits.
* All free text and clinical payloads are encrypted with the per-session data
  key (``medbridge.privacy.crypto``) before they reach PostgreSQL. ``trace``
  is stored in plaintext JSONB and therefore must never contain message text.
* Readers (history, events, snapshot) transparently decrypt. If the session key
  was shredded (right to erasure), readers return ``erased=True`` / empty data
  and never raise.
* ``clinical_events`` and ``audit_logs`` remain append-only: nothing in this
  module issues UPDATE/DELETE against them.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.ai.schemas.extractor import DeltaEvent
from medbridge.api.schemas.enums import ActionEnum


@dataclass
class SessionState:
    """State loaded at the start of a turn."""

    snapshot: dict[str, Any] = field(default_factory=dict)
    soft_ask_count: int = 0
    exists: bool = True
    erased: bool = False


@dataclass
class ChatTurn:
    """A previous message used for extractor coreference (decrypted)."""

    role: str  # "user" | "assistant"
    content: str
    action: Optional[str] = None


@dataclass
class TurnRecord:
    """Everything produced by one pipeline turn, persisted atomically."""

    session_id: uuid.UUID
    user_message: str                      # already PII-redacted
    assistant_message: str
    final_action: ActionEnum
    citations: list[dict[str, Any]] = field(default_factory=list)
    gate_1_action: Optional[str] = None     # "SOFT-ASK" | "PROCEED" | None
    gate_1_rationale: Optional[str] = None
    gate_2_action: Optional[str] = None     # "ANSWER" | "GENERALIZE" | "ABSTAIN" | "ESCALATE" | None
    gate_2_rationale: Optional[str] = None
    evidence_chunk_ids: list[str] = field(default_factory=list)
    latency_ms: int = 0
    trace: dict[str, Any] = field(default_factory=dict)   # Explanation.model_dump(mode="json")
    delta_events: list[DeltaEvent] = field(default_factory=list)
    new_snapshot: Optional[dict[str, Any]] = None          # None = unchanged
    soft_ask_count: int = 0


@dataclass
class HistoryRow:
    message_id: uuid.UUID
    role: str
    content: str
    action: Optional[str]
    citations: list[dict[str, Any]]
    created_at: datetime
    trace: Optional[dict[str, Any]] = None   # assistant rows: explanation from the audit log


@dataclass
class EventRow:
    event_id: uuid.UUID
    event_type: str
    payload: dict[str, Any]
    created_at: datetime


async def persist_turn(db: AsyncSession, record: TurnRecord) -> tuple[uuid.UUID, uuid.UUID]:
    """Persist a full turn with a single flush. Returns (user_message_id, assistant_message_id)."""
    raise NotImplementedError


async def load_session_state(db: AsyncSession, session_id: uuid.UUID) -> SessionState:
    """Load (decrypted) snapshot + soft_ask_count. Missing session → ``exists=False``."""
    raise NotImplementedError


async def get_recent_turns(db: AsyncSession, session_id: uuid.UUID, n: int) -> list[ChatTurn]:
    """Return the last ``n`` messages (chronological, decrypted) for coreference."""
    raise NotImplementedError


async def get_history(db: AsyncSession, session_id: uuid.UUID) -> list[HistoryRow]:
    """Return the full decrypted history with per-assistant-message explanations."""
    raise NotImplementedError


async def get_last_user_message(db: AsyncSession, session_id: uuid.UUID) -> Optional[tuple[uuid.UUID, str]]:
    """Return (message_id, decrypted text) of the most recent user message, if any."""
    raise NotImplementedError


async def list_events(
    db: AsyncSession, session_id: uuid.UUID, event_types: Optional[list[str]] = None
) -> list[EventRow]:
    """Return decrypted clinical events (chronological), optionally filtered by type."""
    raise NotImplementedError


async def append_manual_events(
    db: AsyncSession, session_id: uuid.UUID, events: list[DeltaEvent]
) -> dict[str, Any]:
    """Append user-entered events (e.g. BP log form) through the same projector. Returns new snapshot."""
    raise NotImplementedError


async def save_feedback(
    db: AsyncSession,
    session_id: uuid.UUID,
    message_id: uuid.UUID,
    rating: int,
    comment: Optional[str] = None,
) -> uuid.UUID:
    """Store 👍(+1)/👎(-1) feedback with an optional (encrypted, redacted) comment."""
    raise NotImplementedError


async def erase_session(db: AsyncSession, session_id: uuid.UUID) -> None:
    """Right to erasure via crypto-shredding (destroy the session data key)."""
    raise NotImplementedError
