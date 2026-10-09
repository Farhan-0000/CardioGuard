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
* Concurrency: every writer of a session's snapshot (``persist_turn``,
  ``append_manual_events``) first takes a row lock on its ``context_snapshots``
  row (``SELECT … FOR UPDATE``) and re-projects the snapshot by folding only
  *its own* new events onto the row it just locked — never by writing back a
  snapshot computed from state loaded earlier. Overlapping writers (a second
  turn, the BP form, another worker) are serialised and none of their events
  is lost, so ``snapshot == fold(clinical_events)`` always holds. The lock is
  held only for the single flush and the caller's commit, never across LLM
  calls. A regenerated answer is stored only while the re-run user message is
  still the session's latest (otherwise :class:`RegenerateConflictError`).
* Legacy plaintext: rows written before encryption existed (or with
  ``ENCRYPTION_ENABLED=false``) are plaintext. Crypto-shredding cannot cover
  the ones in the append-only tables (they can never be rewritten), so
  :func:`erase_session` counts them into ``sessions.legacy_plaintext_rows`` and
  logs a warning; every reader already hides an erased session's rows.

Trace storage: ``audit_logs.trace`` keeps the PHI-free parts of the
``Explanation`` in plaintext for metrics and threshold calibration: timings,
actions, sources, models, retrieved guideline chunks, the numeric Gate 2
evidence ``features`` (rerank scores, entity coverage, chunk count) and the
numeric/status members of ``gate3`` (claims, unsupported fraction,
``available``, action change), the loop-breaker counter before the turn
(``soft_ask_count_before``, used by regenerate), ``retrieval_status`` and the
counts of ``effective_question``. Gate rationales, the emergency rule lists and
BP reading, any non-numeric member and any unknown key are moved into
``trace["private"]`` as an encrypted envelope. ``trace["assistant_message_id"]``
links the row to its assistant message for the history "Why this answer?" view
and ``trace["user_message_id"]`` to the user message the turn answered.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import delete, func, not_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
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
from medbridge.privacy.crypto import (
    ENC_PREFIX,
    ENC_VERSION,
    SessionCipher,
    get_session_cipher,
    is_encrypted_text,
    shred_session_key,
)
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
    action: str | None = None


@dataclass
class TurnRecord:
    """Everything produced by one pipeline turn, persisted atomically."""

    session_id: uuid.UUID
    user_message: str                      # already PII-redacted
    assistant_message: str
    final_action: ActionEnum
    citations: list[dict[str, Any]] = field(default_factory=list)
    gate_1_action: str | None = None     # "SOFT-ASK" | "PROCEED" | None
    gate_1_rationale: str | None = None
    gate_2_action: str | None = None     # "ANSWER" | "GENERALIZE" | "ABSTAIN" | "ESCALATE" | None
    gate_2_rationale: str | None = None
    evidence_chunk_ids: list[str] = field(default_factory=list)
    latency_ms: int = 0
    trace: dict[str, Any] = field(default_factory=dict)   # Explanation.model_dump(mode="json")
    delta_events: list[DeltaEvent] = field(default_factory=list)
    # In-memory projection the gates used. The stored snapshot is re-projected from
    # ``delta_events`` onto the locked current row (see persist_turn), not copied from here.
    new_snapshot: dict[str, Any] | None = None
    soft_ask_count: int = 0
    # The re-run user message of a regenerate (the reply is attached to it).
    regenerated_user_message_id: uuid.UUID | None = None
    # False when Gate 1 did not run (emergency fast-path, small talk): the stored
    # loop-breaker counter is then left exactly as it is.
    update_soft_ask_count: bool = True


class RegenerateConflictError(RuntimeError):
    """The re-run user message is no longer the session's latest; nothing was stored."""

    def __init__(self, session_id: uuid.UUID) -> None:
        self.session_id = session_id
        super().__init__(f"Regenerated message of session {session_id} is no longer the latest")


@dataclass
class HistoryRow:
    message_id: uuid.UUID
    role: str
    content: str
    action: str | None
    citations: list[dict[str, Any]]
    created_at: datetime
    trace: dict[str, Any] | None = None   # assistant rows: explanation from the audit log


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
    # Plaintext at rest (pre-encryption row or ENCRYPTION_ENABLED=false); never shown once erased.
    legacy_plaintext: bool = False


# ---------------------------------------------------------------------------
# Trace splitting (plaintext metrics vs encrypted PHI-bearing fields)
# ---------------------------------------------------------------------------

TRACE_PRIVATE_KEY = "private"
TRACE_MESSAGE_ID_KEY = "assistant_message_id"
TRACE_USER_MESSAGE_ID_KEY = "user_message_id"
TRACE_COUNTER_BEFORE_KEY = "soft_ask_count_before"
_INTERNAL_TRACE_KEYS = frozenset({
    TRACE_PRIVATE_KEY, TRACE_MESSAGE_ID_KEY, TRACE_USER_MESSAGE_ID_KEY,
})
# Explanation keys that never contain patient text (metrics read these in plaintext).
_PUBLIC_TRACE_KEYS = frozenset({
    "timings", "total_ms", "models", "llm_calls", "redactions", "redaction_categories",
    "ablation", "regenerated", "retrieval", TRACE_COUNTER_BEFORE_KEY, "retrieval_status",
})
# Gate decisions are public except their free-text rationale.
_GATE_TRACE_KEYS = frozenset({"gate1", "gate2"})
# Only numeric/boolean members and enum-like status strings are public.
_SCALAR_TRACE_KEYS = frozenset({
    "citation_validation", "speculative_retrieval", "gate3", "emergency", "effective_question",
})
_PUBLIC_STRING_FIELDS = frozenset({"status", "action", "action_change", "error"})


def _is_public_member(key: str, value: Any) -> bool:
    if value is None or isinstance(value, bool | int | float):
        return True
    return key in _PUBLIC_STRING_FIELDS and isinstance(value, str) and len(value) <= 64


def _is_number(value: Any) -> bool:
    return isinstance(value, bool | int | float)


def _split_gate(value: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Gate decision: public except the free-text rationale and non-numeric features.

    ``features`` (the V1 evidence features: rerank scores, entity coverage,
    chunk count, override flag) are numbers and stay public so metrics and
    threshold calibration can read them without the session key.
    """
    public: dict[str, Any] = {}
    private: dict[str, Any] = {}
    for key, member in value.items():
        if key == "rationale":
            private[key] = member
        elif key == "features" and isinstance(member, dict):
            public[key] = {k: v for k, v in member.items() if _is_number(v)}
            rest = {k: v for k, v in member.items() if not _is_number(v)}
            if rest:
                private[key] = rest
        else:
            public[key] = member
    return public, private


def split_trace(trace: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split an explanation dict into (public, private) parts."""
    public: dict[str, Any] = {}
    private: dict[str, Any] = {}
    for key, value in trace.items():
        if key in _INTERNAL_TRACE_KEYS:
            continue
        if key in _PUBLIC_TRACE_KEYS or value is None:
            public[key] = value
        elif key in _GATE_TRACE_KEYS and isinstance(value, dict):
            public[key], gate_private = _split_gate(value)
            if gate_private:
                private[key] = gate_private
        elif key in _SCALAR_TRACE_KEYS and isinstance(value, dict):
            public[key] = {k: v for k, v in value.items() if _is_public_member(k, v)}
            rest = {k: v for k, v in value.items() if not _is_public_member(k, v)}
            if rest:
                private[key] = rest
        else:
            private[key] = value
    return public, private


def _merge_dicts(current: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for key, value in extra.items():
        existing = merged.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            merged[key] = {**existing, **value}
        else:
            merged[key] = value
    return merged


def merge_trace(public: dict[str, Any], private: dict[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`split_trace` (internal keys removed)."""
    merged = {k: v for k, v in public.items() if k not in _INTERNAL_TRACE_KEYS}
    for key, value in private.items():
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = _merge_dicts(current, value)
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

async def _lock_session_snapshot(db: AsyncSession, session_id: uuid.UUID) -> ContextSnapshot:
    """Row-lock the session's snapshot row (``SELECT … FOR UPDATE``) and return it fresh.

    Every snapshot writer calls this first, so writers of one session are
    serialised by PostgreSQL until the caller commits. ``populate_existing``
    replaces any copy the session's identity map still holds from the turn's
    ``load_session_state`` with the committed row. A missing row (legacy
    session) is created race-free first.

    Raises:
        SessionErasedError: the session was erased while this writer waited.
    """
    def _locked() -> Any:
        return (
            select(ContextSnapshot)
            .where(ContextSnapshot.session_id == session_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )

    row = (await db.execute(_locked())).scalars().first()
    if row is None:
        await db.execute(
            pg_insert(ContextSnapshot)
            .values(session_id=session_id, snapshot={}, soft_ask_count=0, updated_at=utcnow())
            .on_conflict_do_nothing(index_elements=[ContextSnapshot.session_id])
        )
        row = (await db.execute(_locked())).scalars().first()
    erased_at = (await db.execute(
        select(Session.erased_at).where(Session.session_id == session_id)
    )).scalar_one_or_none()
    if row is None or erased_at is not None:
        raise SessionErasedError(session_id)
    return row


def _fold_onto(
    cipher: SessionCipher,
    row: ContextSnapshot,
    events: list[tuple[EventTypeEnum, DeltaEvent]],
) -> dict[str, Any]:
    """Apply ``events`` to the (locked) current snapshot row and store the result."""
    current = safe_decrypt_json(cipher, row.snapshot, {})
    snapshot = StateProjector().apply_events(
        current if isinstance(current, dict) else {}, [event for _, event in events]
    )
    row.snapshot = cipher.encrypt_json(snapshot)
    return snapshot


async def persist_turn(db: AsyncSession, record: TurnRecord) -> tuple[uuid.UUID, uuid.UUID]:
    """Persist a full turn with a single flush. Returns (user_message_id, assistant_message_id).

    The snapshot row is locked first and the turn's (known) delta events are
    folded onto its *current* value, so events written meanwhile by another
    turn or the BP form are never overwritten.

    For a regenerated turn (``record.trace["regenerated"]`` or
    ``record.regenerated_user_message_id``) no user message and no clinical
    events are stored, the snapshot is left unchanged and the new answer is
    attached to the re-run user message.

    Raises:
        SessionNotFoundError: the session does not exist.
        SessionErasedError: the session was erased (nothing is written).
        RegenerateConflictError: the re-run user message is no longer the
            session's latest (another turn was stored meanwhile; nothing is written).
    """
    session_id = record.session_id
    session = await db.get(Session, session_id)
    if session is None:
        raise SessionNotFoundError(session_id)
    if session.erased_at is not None:
        raise SessionErasedError(session_id)
    snap_row = await _lock_session_snapshot(db, session_id)
    regenerated = (
        bool(record.trace.get("regenerated")) or record.regenerated_user_message_id is not None
    )
    previous_user = await _last_user_message_row(db, session_id) if regenerated else None
    expected_user = record.regenerated_user_message_id
    if expected_user is not None and (
        previous_user is None or previous_user.message_id != expected_user
    ):
        logger.warning("regenerate_conflict", session_id=str(session_id))
        raise RegenerateConflictError(session_id)
    cipher = await require_cipher(db, session_id)
    now = utcnow()  # taken under the lock: created_at order == commit order

    # 1. Clinical events (append-only) — never re-recorded on regenerate.
    if regenerated and record.delta_events:
        logger.info("regenerate_events_ignored", count=len(record.delta_events))
    known = [] if regenerated else _known_events(record.delta_events)
    _add_events(db, session_id, cipher, known, now)

    # 2. Snapshot = fold of the events (re-projected on the locked row) + counter.
    if known:
        _fold_onto(cipher, snap_row, known)
    if record.update_soft_ask_count:
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
    public_trace[TRACE_USER_MESSAGE_ID_KEY] = str(user_message_id)
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
    """Append user-entered events (e.g. the BP log form) through the same projector.

    Takes the same snapshot row lock as :func:`persist_turn` and folds the
    events onto the current row. Returns the new snapshot.

    Raises:
        SessionNotFoundError / SessionErasedError.
    """
    session = await db.get(Session, session_id)
    if session is None:
        raise SessionNotFoundError(session_id)
    if session.erased_at is not None:
        raise SessionErasedError(session_id)
    snap_row = await _lock_session_snapshot(db, session_id)
    cipher = await require_cipher(db, session_id)
    known = _known_events(events)
    now = utcnow()
    _add_events(db, session_id, cipher, known, now)
    new_snapshot = _fold_onto(cipher, snap_row, known)
    snap_row.updated_at = now
    session.last_active_at = now
    await db.flush()
    return new_snapshot


async def save_feedback(
    db: AsyncSession,
    session_id: uuid.UUID,
    message_id: uuid.UUID,
    rating: int,
    comment: str | None = None,
) -> uuid.UUID:
    """Store 👍(+1)/👎(-1) feedback with an optional (encrypted, redacted) comment.

    One rating per message: a new rating replaces the previous one atomically
    (``INSERT … ON CONFLICT (session_id, message_id) DO UPDATE`` on the unique
    constraint), so concurrent ratings can never leave two rows.

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
    insert = pg_insert(Feedback).values(
        feedback_id=uuid.uuid4(),
        session_id=session_id,
        message_id=message_id,
        rating=rating,
        comment=cipher.encrypt(text) if text else None,
        created_at=utcnow(),
    )
    upsert = insert.on_conflict_do_update(
        index_elements=[Feedback.session_id, Feedback.message_id],
        set_={
            "feedback_id": insert.excluded.feedback_id,
            "rating": insert.excluded.rating,
            "comment": insert.excluded.comment,
            "created_at": insert.excluded.created_at,
        },
    ).returning(Feedback.feedback_id)
    feedback_id: uuid.UUID = (await db.execute(upsert)).scalar_one()
    return feedback_id


async def count_legacy_plaintext_rows(db: AsyncSession, session_id: uuid.UUID) -> int:
    """Append-only rows of a session that are plaintext (written before encryption existed).

    Crypto-shredding cannot cover them and they can never be rewritten
    (immutability rules), so erasure records this count.
    """
    audits = (await db.execute(
        select(func.count()).select_from(AuditLog).where(
            AuditLog.session_id == session_id,
            not_(AuditLog.request_message.startswith(ENC_PREFIX, autoescape=True)),
        )
    )).scalar_one()
    events = (await db.execute(
        select(func.count()).select_from(ClinicalEvent).where(
            ClinicalEvent.session_id == session_id,
            func.coalesce(ClinicalEvent.payload.op("->>")("enc"), "") != ENC_VERSION,
        )
    )).scalar_one()
    return int(audits) + int(events)


async def erase_session(db: AsyncSession, session_id: uuid.UUID) -> None:
    """Right to erasure via crypto-shredding (destroy the session data key).

    Deletes the wrapped data key and the mutable rows (messages, snapshot,
    feedback) and stamps ``sessions.erased_at``. Encrypted rows in the
    append-only ``clinical_events`` / ``audit_logs`` tables stay in place but
    can never be decrypted again. Legacy plaintext rows in those tables cannot
    be shredded or rewritten: their number is recorded in
    ``sessions.legacy_plaintext_rows`` (and logged); like every other row of
    the session they are never shown again (readers return nothing for an
    erased session). Idempotent (also used by the retention purge).

    Raises:
        SessionNotFoundError: the session does not exist.
    """
    session = await db.get(Session, session_id)
    if session is None:
        raise SessionNotFoundError(session_id)
    legacy = await count_legacy_plaintext_rows(db, session_id)
    await shred_session_key(db, session_id)
    for model in (MessageHistory, ContextSnapshot, Feedback):
        await db.execute(delete(model).where(model.session_id == session_id))
    if session.erased_at is None:
        session.erased_at = utcnow()
    session.legacy_plaintext_rows = legacy
    await db.flush()
    if legacy:
        logger.warning(
            "erasure_legacy_plaintext_retained",
            session_id=str(session_id),
            rows=legacy,
            hint="Append-only rows written before encryption stay plaintext at rest; they are "
            "hidden from every view. Reset pre-encryption databases (see privacy/README.md).",
        )
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
    """Return the last ``n`` messages (chronological, decrypted) for coreference.

    A regenerated answer supersedes the earlier answers to the same user
    message (the UI shows only the latest), so of consecutive assistant
    messages only the most recent is returned: the window always holds real
    question/answer pairs, however often an answer was regenerated.
    """
    if n <= 0:
        return []
    cipher = await get_session_cipher(db, session_id)
    if cipher is None:
        return []
    next_role = func.lead(MessageHistory.role).over(
        order_by=(MessageHistory.created_at.asc(), ROLE_ORDER.asc())
    )
    ordered = (
        select(MessageHistory.message_id, next_role.label("next_role"))
        .where(MessageHistory.session_id == session_id)
        .subquery()
    )
    stmt = (
        select(MessageHistory)
        .join(ordered, ordered.c.message_id == MessageHistory.message_id)
        .where(
            not_(
                (MessageHistory.role == "assistant")
                & (func.coalesce(ordered.c.next_role, "") == "assistant")
            )
        )
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


async def get_last_user_message(
    db: AsyncSession, session_id: uuid.UUID
) -> tuple[uuid.UUID, str] | None:
    """Return (message_id, decrypted text) of the most recent user message, if any."""
    cipher = await get_session_cipher(db, session_id)
    if cipher is None:
        return None
    row = await _last_user_message_row(db, session_id)
    if row is None:
        return None
    return row.message_id, safe_decrypt(cipher, row.content)


def _counter_before(trace: Any) -> int | None:
    value = trace.get(TRACE_COUNTER_BEFORE_KEY) if isinstance(trace, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


async def get_soft_ask_count_before(
    db: AsyncSession, session_id: uuid.UUID, user_message_id: uuid.UUID
) -> int | None:
    """Loop-breaker counter as it was *before* the turn that answered ``user_message_id``.

    A regenerate re-runs that turn, so Gate 1 must start from this value — not
    from the stored counter, which already includes the replaced turn. It is
    read from the PHI-free trace of the FIRST audit row of that user message
    (every regenerate carries the same value forward, so repeated regenerates
    are idempotent). Rows written before the value was recorded fall back to
    the replaced turn: SOFT-ASK → stored counter − 1, loop-breaker →
    ``SOFT_ASK_MAX_COUNT``. ``None`` when it cannot be determined.
    """
    traces = (await db.execute(
        select(AuditLog.trace)
        .where(
            AuditLog.session_id == session_id,
            AuditLog.trace.op("->>")(TRACE_USER_MESSAGE_ID_KEY) == str(user_message_id),
        )
        .order_by(AuditLog.created_at.asc())
    )).scalars().all()
    for trace in traces:
        if (value := _counter_before(trace)) is not None:
            return value
    latest = (await db.execute(
        select(AuditLog.final_action, AuditLog.trace)
        .where(AuditLog.session_id == session_id)
        .order_by(AuditLog.created_at.desc())
        .limit(1)
    )).first()
    if latest is None:
        return None
    final_action, trace = latest
    if (value := _counter_before(trace)) is not None:
        return value
    gate2 = trace.get("gate2") if isinstance(trace, dict) else None
    if isinstance(gate2, dict) and gate2.get("source") == "loop_breaker":
        return get_settings().SOFT_ASK_MAX_COUNT
    if final_action == FinalActionEnum.SOFT_ASK:
        snap = await db.get(ContextSnapshot, session_id)
        return max(0, (snap.soft_ask_count if snap is not None else 0) - 1)
    return None


async def list_events(
    db: AsyncSession, session_id: uuid.UUID, event_types: list[str] | None = None
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
    """Most recent audit rows across sessions, decrypted where the key still exists.

    Rows of an erased session never carry text — including legacy plaintext
    rows, which are only flagged (``legacy_plaintext``).
    """
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
            legacy_plaintext=not is_encrypted_text(row.request_message),
        ))
    return entries
