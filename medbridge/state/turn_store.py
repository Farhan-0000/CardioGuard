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

Trace storage: ``audit_logs.trace`` keeps the PHI-free parts of the
``Explanation`` (timings, actions, sources, models, retrieved guideline
chunks) in plaintext for metrics. Gate rationales, the emergency match, the
grounding-verifier sentences and any unknown key are moved into
``trace["private"]`` as an encrypted envelope, and ``trace["assistant_message_id"]``
links the row to its assistant message for the history "Why this answer?" view.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.ai.schemas.extractor import DeltaEvent
from medbridge.api.schemas.enums import ActionEnum, EventTypeEnum
from medbridge.config import get_settings
from medbridge.db.models import (
    AuditLog,
    ClinicalEvent,
    ContextSnapshot,
    Feedback,
    FinalActionEnum,
    Gate1ActionEnum,
    Gate2ActionEnum,
    MessageHistory,
    Session,
)
from medbridge.privacy.crypto import SessionCipher, get_session_cipher, shred_session_key
from medbridge.privacy.redaction import redact
from medbridge.state.projector import StateProjector, event_type_value
from medbridge.state.session_manager import (
    ROLE_ORDER,
    MessageNotFoundError,
    SessionErasedError,
    SessionNotFoundError,
    require_cipher,
    safe_decrypt,
    safe_decrypt_json,
    utcnow,
)

logger = structlog.get_logger(__name__)


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


@dataclass
class AuditEntry:
    """One audit row for the admin dashboard (``None`` text = erased/unreadable)."""

    audit_id: uuid.UUID
    session_id: uuid.UUID
    created_at: datetime
    final_action: str
    gate_1_action: str | None
    gate_2_action: str | None
    latency_ms: int | None
    evidence_chunk_count: int
    request_text: str | None
    response_text: str | None
    erased: bool


# ---------------------------------------------------------------------------
# Trace splitting (plaintext metrics vs encrypted PHI-bearing fields)
# ---------------------------------------------------------------------------

TRACE_PRIVATE_KEY = "private"
TRACE_MESSAGE_ID_KEY = "assistant_message_id"
# Explanation keys that never contain patient text (metrics read these in plaintext).
_PUBLIC_TRACE_KEYS = frozenset({
    "timings", "total_ms", "models", "llm_calls", "redactions", "redaction_categories",
    "ablation", "regenerated", "retrieval",
})
# Gate decisions are public except their free-text rationale.
_GATE_TRACE_KEYS = frozenset({"gate1", "gate2"})
# Only numeric/boolean members and enum-like status strings are public.
_SCALAR_TRACE_KEYS = frozenset({
    "citation_validation", "speculative_retrieval", "gate3", "emergency",
})
_PUBLIC_STRING_FIELDS = frozenset({"status", "action", "action_change", "error"})


def _is_public_member(key: str, value: Any) -> bool:
    if value is None or isinstance(value, bool | int | float):
        return True
    return key in _PUBLIC_STRING_FIELDS and isinstance(value, str) and len(value) <= 64


def split_trace(trace: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split an explanation dict into (public, private) parts."""
    public: dict[str, Any] = {}
    private: dict[str, Any] = {}
    for key, value in trace.items():
        if key in (TRACE_PRIVATE_KEY, TRACE_MESSAGE_ID_KEY):
            continue
        if key in _PUBLIC_TRACE_KEYS or value is None:
            public[key] = value
        elif key in _GATE_TRACE_KEYS and isinstance(value, dict):
            public[key] = {k: v for k, v in value.items() if k != "rationale"}
            if "rationale" in value:
                private[key] = {"rationale": value["rationale"]}
        elif key in _SCALAR_TRACE_KEYS and isinstance(value, dict):
            public[key] = {k: v for k, v in value.items() if _is_public_member(k, v)}
            rest = {k: v for k, v in value.items() if not _is_public_member(k, v)}
            if rest:
                private[key] = rest
        else:
            private[key] = value
    return public, private


def merge_trace(public: dict[str, Any], private: dict[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`split_trace` (internal keys removed)."""
    merged = {
        k: v for k, v in public.items() if k not in (TRACE_PRIVATE_KEY, TRACE_MESSAGE_ID_KEY)
    }
    for key, value in private.items():
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = {**current, **value}
        else:
            merged[key] = value
    return merged


def _stored_trace(stored: dict[str, Any], cipher: SessionCipher) -> dict[str, Any]:
    """Rebuild the full explanation from an ``audit_logs.trace`` value."""
    private = safe_decrypt_json(cipher, stored.get(TRACE_PRIVATE_KEY) or {}, {})
    return merge_trace(stored, private if isinstance(private, dict) else {})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gate_enum(
    enum_cls: type[Gate1ActionEnum] | type[Gate2ActionEnum], value: str | None
) -> Any:
    if not value:
        return None
    try:
        return enum_cls(str(value).strip().upper().replace("_", "-"))
    except ValueError:
        logger.warning("unknown_gate_action", gate=enum_cls.__name__, value=str(value)[:32])
        return None


def _encrypt_optional(cipher: SessionCipher, value: str | None) -> str | None:
    return cipher.encrypt(value) if value is not None else None


def _known_events(events: list[DeltaEvent]) -> list[tuple[EventTypeEnum, DeltaEvent]]:
    known: list[tuple[EventTypeEnum, DeltaEvent]] = []
    for event in events:
        try:
            known.append((EventTypeEnum(event_type_value(event)), event))
        except ValueError:
            logger.warning("unknown_event_type_skipped", event_type=event_type_value(event)[:32])
    return known


def _add_events(
    db: AsyncSession,
    session_id: uuid.UUID,
    cipher: SessionCipher,
    events: list[tuple[EventTypeEnum, DeltaEvent]],
    now: datetime,
) -> None:
    # Strictly increasing timestamps keep the replay order of one turn deterministic.
    for offset, (event_type, event) in enumerate(events):
        db.add(ClinicalEvent(
            event_id=uuid.uuid4(),
            session_id=session_id,
            event_type=event_type,
            payload=cipher.encrypt_json(event.payload),
            created_at=now + timedelta(microseconds=offset),
        ))


async def _last_user_message_row(
    db: AsyncSession, session_id: uuid.UUID
) -> MessageHistory | None:
    stmt = (
        select(MessageHistory)
        .where(MessageHistory.session_id == session_id, MessageHistory.role == "user")
        .order_by(MessageHistory.created_at.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalars().first()


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

async def persist_turn(db: AsyncSession, record: TurnRecord) -> tuple[uuid.UUID, uuid.UUID]:
    """Persist a full turn with a single flush. Returns (user_message_id, assistant_message_id).

    For a regenerated turn (``record.trace["regenerated"]``) no user message and no
    clinical events are stored and the snapshot is left unchanged (only the
    soft-ask counter is written), so the snapshot stays a fold over the events.

    Raises:
        SessionNotFoundError: the session does not exist.
        SessionErasedError: the session was erased (nothing is written).
    """
    session_id = record.session_id
    session = await db.get(Session, session_id)
    if session is None:
        raise SessionNotFoundError(session_id)
    if session.erased_at is not None:
        raise SessionErasedError(session_id)
    snap_row = await db.get(ContextSnapshot, session_id)
    regenerated = bool(record.trace.get("regenerated"))
    previous_user = await _last_user_message_row(db, session_id) if regenerated else None
    cipher = await require_cipher(db, session_id)
    now = utcnow()

    # 1. Clinical events (append-only) — never re-recorded on regenerate.
    if regenerated and record.delta_events:
        logger.info("regenerate_events_ignored", count=len(record.delta_events))
    if not regenerated:
        _add_events(db, session_id, cipher, _known_events(record.delta_events), now)

    # 2. Snapshot upsert + loop-breaker counter.
    new_snapshot = None if regenerated else record.new_snapshot
    if snap_row is None:
        db.add(ContextSnapshot(
            session_id=session_id,
            snapshot=cipher.encrypt_json(new_snapshot or {}),
            soft_ask_count=record.soft_ask_count,
            updated_at=now,
        ))
    else:
        if new_snapshot is not None:
            snap_row.snapshot = cipher.encrypt_json(new_snapshot)
        snap_row.soft_ask_count = record.soft_ask_count
        snap_row.updated_at = now

    # 3. Messages (user first, assistant 1 µs later for a stable order).
    if previous_user is not None:
        user_message_id = previous_user.message_id
    else:
        user_message_id = uuid.uuid4()
        db.add(MessageHistory(
            message_id=user_message_id,
            session_id=session_id,
            role="user",
            content=cipher.encrypt(record.user_message),
            action=None,
            citations=[],
            created_at=now,
        ))
    assistant_message_id = uuid.uuid4()
    final_action = FinalActionEnum(str(record.final_action))
    db.add(MessageHistory(
        message_id=assistant_message_id,
        session_id=session_id,
        role="assistant",
        content=cipher.encrypt(record.assistant_message),
        action=final_action,
        citations=list(record.citations),
        created_at=now + timedelta(microseconds=1),
    ))

    # 4. Immutable audit row (text + rationales encrypted, trace PHI-free).
    public_trace, private_trace = split_trace(record.trace)
    public_trace[TRACE_MESSAGE_ID_KEY] = str(assistant_message_id)
    if private_trace:
        public_trace[TRACE_PRIVATE_KEY] = cipher.encrypt_json(private_trace)
    db.add(AuditLog(
        audit_id=uuid.uuid4(),
        session_id=session_id,
        request_message=cipher.encrypt(record.user_message),
        gate_1_action=_gate_enum(Gate1ActionEnum, record.gate_1_action),
        gate_1_rationale=_encrypt_optional(cipher, record.gate_1_rationale),
        gate_2_action=_gate_enum(Gate2ActionEnum, record.gate_2_action),
        gate_2_rationale=_encrypt_optional(cipher, record.gate_2_rationale),
        final_action=final_action,
        evidence_chunk_ids=list(record.evidence_chunk_ids),
        response_text=cipher.encrypt(record.assistant_message),
        latency_ms=record.latency_ms,
        trace=public_trace,
        created_at=now,
    ))

    session.last_active_at = now
    await db.flush()
    return user_message_id, assistant_message_id


async def append_manual_events(
    db: AsyncSession, session_id: uuid.UUID, events: list[DeltaEvent]
) -> dict[str, Any]:
    """Append user-entered events (e.g. BP log form) through the same projector. Returns new snapshot.

    Raises:
        SessionNotFoundError / SessionErasedError.
    """
    state = await load_session_state(db, session_id)
    if not state.exists:
        raise SessionNotFoundError(session_id)
    if state.erased:
        raise SessionErasedError(session_id)
    cipher = await require_cipher(db, session_id)
    known = _known_events(events)
    new_snapshot = StateProjector().apply_events(state.snapshot, [e for _, e in known])
    now = utcnow()
    _add_events(db, session_id, cipher, known, now)

    snap_row = await db.get(ContextSnapshot, session_id)
    if snap_row is None:
        db.add(ContextSnapshot(
            session_id=session_id,
            snapshot=cipher.encrypt_json(new_snapshot),
            soft_ask_count=state.soft_ask_count,
            updated_at=now,
        ))
    else:
        snap_row.snapshot = cipher.encrypt_json(new_snapshot)
        snap_row.updated_at = now
    session = await db.get(Session, session_id)
    if session is not None:
        session.last_active_at = now
    await db.flush()
    return new_snapshot


async def save_feedback(
    db: AsyncSession,
    session_id: uuid.UUID,
    message_id: uuid.UUID,
    rating: int,
    comment: Optional[str] = None,
) -> uuid.UUID:
    """Store 👍(+1)/👎(-1) feedback with an optional (encrypted, redacted) comment.

    A new rating for the same message replaces the previous one.

    Raises:
        ValueError: rating not in {-1, 1}.
        SessionNotFoundError / SessionErasedError.
        MessageNotFoundError: the message is not an assistant message of this session.
    """
    if rating not in (-1, 1):
        raise ValueError("rating must be 1 or -1")
    cipher = await require_cipher(db, session_id)
    message = await db.get(MessageHistory, message_id)
    if message is None or message.session_id != session_id or message.role != "assistant":
        raise MessageNotFoundError(message_id)

    text = (comment or "").strip()
    if text and get_settings().PII_REDACTION_ENABLED:
        text = redact(text).text
    await db.execute(
        delete(Feedback).where(Feedback.session_id == session_id, Feedback.message_id == message_id)
    )
    feedback = Feedback(
        feedback_id=uuid.uuid4(),
        session_id=session_id,
        message_id=message_id,
        rating=rating,
        comment=cipher.encrypt(text) if text else None,
        created_at=utcnow(),
    )
    db.add(feedback)
    await db.flush()
    return feedback.feedback_id


async def erase_session(db: AsyncSession, session_id: uuid.UUID) -> None:
    """Right to erasure via crypto-shredding (destroy the session data key).

    Deletes the wrapped data key and the mutable rows (messages, snapshot,
    feedback) and stamps ``sessions.erased_at``. Encrypted rows in the
    append-only ``clinical_events`` / ``audit_logs`` tables stay in place but
    can never be decrypted again. Idempotent.

    Raises:
        SessionNotFoundError: the session does not exist.
    """
    session = await db.get(Session, session_id)
    if session is None:
        raise SessionNotFoundError(session_id)
    await shred_session_key(db, session_id)
    for model in (MessageHistory, ContextSnapshot, Feedback):
        await db.execute(delete(model).where(model.session_id == session_id))
    if session.erased_at is None:
        session.erased_at = utcnow()
    await db.flush()
    logger.info("session_erased", session_id=str(session_id))


# ---------------------------------------------------------------------------
# Readers (never raise for erased sessions)
# ---------------------------------------------------------------------------

async def load_session_state(db: AsyncSession, session_id: uuid.UUID) -> SessionState:
    """Load (decrypted) snapshot + soft_ask_count. Missing session → ``exists=False``."""
    session = await db.get(Session, session_id)
    if session is None:
        return SessionState(exists=False)
    if session.erased_at is not None:
        return SessionState(erased=True)
    cipher = await get_session_cipher(db, session_id)
    if cipher is None:
        return SessionState(erased=True)
    snap = await db.get(ContextSnapshot, session_id)
    if snap is None:
        return SessionState()
    snapshot = safe_decrypt_json(cipher, snap.snapshot, {})
    return SessionState(
        snapshot=snapshot if isinstance(snapshot, dict) else {},
        soft_ask_count=snap.soft_ask_count,
    )


async def get_recent_turns(db: AsyncSession, session_id: uuid.UUID, n: int) -> list[ChatTurn]:
    """Return the last ``n`` messages (chronological, decrypted) for coreference."""
    if n <= 0:
        return []
    cipher = await get_session_cipher(db, session_id)
    if cipher is None:
        return []
    stmt = (
        select(MessageHistory)
        .where(MessageHistory.session_id == session_id)
        .order_by(MessageHistory.created_at.desc(), ROLE_ORDER.desc())
        .limit(n)
    )
    rows = list((await db.execute(stmt)).scalars().all())
    return [
        ChatTurn(
            role=row.role,
            content=safe_decrypt(cipher, row.content),
            action=row.action.value if row.action else None,
        )
        for row in reversed(rows)
    ]


async def get_history(db: AsyncSession, session_id: uuid.UUID) -> list[HistoryRow]:
    """Return the full decrypted history with per-assistant-message explanations."""
    cipher = await get_session_cipher(db, session_id)
    if cipher is None:
        return []
    messages = (await db.execute(
        select(MessageHistory)
        .where(MessageHistory.session_id == session_id)
        .order_by(MessageHistory.created_at.asc(), ROLE_ORDER)
    )).scalars().all()
    traces = (await db.execute(
        select(AuditLog.trace)
        .where(AuditLog.session_id == session_id, AuditLog.trace.is_not(None))
    )).scalars().all()
    by_message: dict[str, dict[str, Any]] = {
        str(t[TRACE_MESSAGE_ID_KEY]): t
        for t in traces
        if isinstance(t, dict) and t.get(TRACE_MESSAGE_ID_KEY)
    }
    rows: list[HistoryRow] = []
    for m in messages:
        stored = by_message.get(str(m.message_id)) if m.role == "assistant" else None
        rows.append(HistoryRow(
            message_id=m.message_id,
            role=m.role,
            content=safe_decrypt(cipher, m.content),
            action=m.action.value if m.action else None,
            citations=list(m.citations or []),
            created_at=m.created_at,
            trace=_stored_trace(stored, cipher) if stored is not None else None,
        ))
    return rows


async def get_last_user_message(db: AsyncSession, session_id: uuid.UUID) -> Optional[tuple[uuid.UUID, str]]:
    """Return (message_id, decrypted text) of the most recent user message, if any."""
    cipher = await get_session_cipher(db, session_id)
    if cipher is None:
        return None
    row = await _last_user_message_row(db, session_id)
    if row is None:
        return None
    return row.message_id, safe_decrypt(cipher, row.content)


async def list_events(
    db: AsyncSession, session_id: uuid.UUID, event_types: Optional[list[str]] = None
) -> list[EventRow]:
    """Return decrypted clinical events (chronological), optionally filtered by type."""
    cipher = await get_session_cipher(db, session_id)
    if cipher is None:
        return []
    stmt = select(ClinicalEvent).where(ClinicalEvent.session_id == session_id)
    if event_types is not None:
        wanted = [EventTypeEnum(t) for t in event_types if t in EventTypeEnum._value2member_map_]
        if not wanted:
            return []
        stmt = stmt.where(ClinicalEvent.event_type.in_(wanted))
    rows = (await db.execute(stmt.order_by(ClinicalEvent.created_at.asc()))).scalars().all()
    events: list[EventRow] = []
    for row in rows:
        payload = safe_decrypt_json(cipher, row.payload, {})
        events.append(EventRow(
            event_id=row.event_id,
            event_type=row.event_type.value,
            payload=payload if isinstance(payload, dict) else {},
            created_at=row.created_at,
        ))
    return events


async def list_audit_entries(db: AsyncSession, limit: int = 50) -> list[AuditEntry]:
    """Most recent audit rows across sessions, decrypted where the key still exists."""
    rows = (await db.execute(
        select(AuditLog).order_by(AuditLog.created_at.desc()).limit(max(1, limit))
    )).scalars().all()
    ciphers: dict[uuid.UUID, SessionCipher | None] = {}
    entries: list[AuditEntry] = []
    for row in rows:
        if row.session_id not in ciphers:
            ciphers[row.session_id] = await get_session_cipher(db, row.session_id)
        cipher = ciphers[row.session_id]
        entries.append(AuditEntry(
            audit_id=row.audit_id,
            session_id=row.session_id,
            created_at=row.created_at,
            final_action=row.final_action.value,
            gate_1_action=row.gate_1_action.value if row.gate_1_action else None,
            gate_2_action=row.gate_2_action.value if row.gate_2_action else None,
            latency_ms=row.latency_ms,
            evidence_chunk_count=len(row.evidence_chunk_ids or []),
            request_text=safe_decrypt(cipher, row.request_message) if cipher else None,
            response_text=safe_decrypt(cipher, row.response_text) if cipher else None,
            erased=cipher is None,
        ))
    return entries
