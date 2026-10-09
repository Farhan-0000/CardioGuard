"""
Database Session Manager (ADL-014, ADL-031).

Session creation (incl. the per-session data key), legacy per-call
persistence helpers (used by the Streamlit path, the adversarial suites and
older tests), and the loop-breaker counter.

Message content and the snapshot are encrypted with the session data key
(``medbridge.privacy.crypto``); readers decrypt transparently and accept
legacy plaintext. New pipeline code uses ``medbridge.state.turn_store`` which
persists a whole turn with a single flush.

All functions accept an ``AsyncSession`` and do **not** commit — callers (or
the FastAPI dependency) are responsible for committing.
"""
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import case, select
from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.api.schemas.enums import ActionEnum
from medbridge.db.models import (
    ContextSnapshot,
    FinalActionEnum,
    MessageHistory,
    Session,
)
from medbridge.privacy.crypto import (
    DecryptionError,
    SessionCipher,
    create_session_key,
    get_session_cipher,
)

logger = structlog.get_logger(__name__)

UNREADABLE = "[unreadable]"


class SessionNotFoundError(Exception):
    """Raised when a session does not exist in the database."""

    def __init__(self, session_id: uuid.UUID) -> None:
        self.session_id = session_id
        super().__init__(f"Session {session_id} not found")


class SessionErasedError(SessionNotFoundError):
    """Raised when a session's data was erased (crypto-shredded)."""

    def __init__(self, session_id: uuid.UUID) -> None:
        super().__init__(session_id)
        self.args = (f"Session {session_id} was erased",)


class MessageNotFoundError(LookupError):
    """Raised when a message id does not belong to the session (or is not a reply)."""

    def __init__(self, message_id: uuid.UUID) -> None:
        self.message_id = message_id
        super().__init__(f"Message {message_id} not found")


# Ordering helper: within the same timestamp the user message precedes the reply.
ROLE_ORDER = case((MessageHistory.role == "user", 0), else_=1)


def utcnow() -> datetime:
    return datetime.now(UTC)


def safe_decrypt(cipher: SessionCipher, value: str) -> str:
    """Decrypt text, returning a marker instead of raising on a key mismatch."""
    try:
        return cipher.decrypt(value)
    except DecryptionError:
        logger.warning("decrypt_failed", kind="text")
        return UNREADABLE


def safe_decrypt_json(cipher: SessionCipher, value: Any, default: Any) -> Any:
    """Decrypt a JSON envelope, returning ``default`` on a key mismatch."""
    try:
        return cipher.decrypt_json(value)
    except DecryptionError:
        logger.warning("decrypt_failed", kind="json")
        return default


async def require_cipher(db: AsyncSession, session_id: uuid.UUID) -> SessionCipher:
    """Return the session cipher or raise ``SessionNotFoundError`` / ``SessionErasedError``."""
    cipher = await get_session_cipher(db, session_id)
    if cipher is not None:
        return cipher
    session = await db.get(Session, session_id)
    if session is None:
        raise SessionNotFoundError(session_id)
    raise SessionErasedError(session_id)


# ------------------------------------------------------------------
# Session CRUD
# ------------------------------------------------------------------

async def create_session(db: AsyncSession) -> uuid.UUID:
    """Create a new session (+ data key + empty snapshot) and return its UUID."""
    session_id = uuid.uuid4()
    db.add(Session(session_id=session_id))
    await create_session_key(db, session_id)
    # The empty initial snapshot carries no data; later writes are encrypted.
    db.add(ContextSnapshot(session_id=session_id, snapshot={}, soft_ask_count=0))
    await db.flush()
    return session_id


async def load_session_state(
    db: AsyncSession,
    session_id: uuid.UUID,
) -> dict[str, Any]:
    """Load the decrypted snapshot and soft_ask_count for a session (ADL-014).

    Returns a dict with keys ``"snapshot"`` (dict) and ``"soft_ask_count"`` (int).

    Raises:
        SessionNotFoundError: if the session does not exist.
        SessionErasedError: if the session was erased.
    """
    session = await db.get(Session, session_id)
    if session is None:
        raise SessionNotFoundError(session_id)
    cipher = await require_cipher(db, session_id)

    snap = await db.get(ContextSnapshot, session_id)
    if snap is None:
        return {"snapshot": {}, "soft_ask_count": 0}
    return {
        "snapshot": safe_decrypt_json(cipher, snap.snapshot, {}),
        "soft_ask_count": snap.soft_ask_count,
    }


# ------------------------------------------------------------------
# Message History
# ------------------------------------------------------------------

async def save_message(
    db: AsyncSession,
    session_id: uuid.UUID,
    role: str,
    content: str,
    action: ActionEnum | None = None,
    citations: list[dict[str, Any]] | None = None,
) -> uuid.UUID:
    """Append an (encrypted) message to message_history and return its UUID.

    Args:
        db: Active async database session.
        session_id: The session to append the message to.
        role: ``"user"`` or ``"assistant"``.
        content: Message text content (already PII-redacted by the caller).
        action: The final action taken (assistant messages only).
        citations: List of citation dicts (assistant messages only).
    """
    cipher = await require_cipher(db, session_id)
    message = MessageHistory(
        message_id=uuid.uuid4(),
        session_id=session_id,
        role=role,
        content=cipher.encrypt(content),
        action=FinalActionEnum(action.value) if action is not None else None,
        citations=citations or [],
        created_at=utcnow(),
    )
    db.add(message)
    await db.flush()
    return message.message_id


async def get_message_history(
    db: AsyncSession,
    session_id: uuid.UUID,
) -> list[dict[str, Any]]:
    """Return the chronological, decrypted message list for a session.

    Each message is a dict with keys: ``message_id``, ``role``, ``content``,
    ``action``, ``citations``, ``created_at``. Erased sessions return ``[]``.
    """
    cipher = await get_session_cipher(db, session_id)
    if cipher is None:
        return []
    stmt = (
        select(MessageHistory)
        .where(MessageHistory.session_id == session_id)
        .order_by(MessageHistory.created_at.asc(), ROLE_ORDER)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [
        {
            "message_id": row.message_id,
            "role": row.role,
            "content": safe_decrypt(cipher, row.content),
            "action": row.action.value if row.action else None,
            "citations": row.citations or [],
            "created_at": row.created_at,
        }
        for row in rows
    ]


# ------------------------------------------------------------------
# Soft-Ask Counter Management (Loop-Breaker — NON-NEGOTIABLE #4)
# ------------------------------------------------------------------

async def increment_soft_ask_count(
    db: AsyncSession,
    session_id: uuid.UUID,
) -> int:
    """Increment the soft_ask_count by 1 and return the new value.

    Raises:
        SessionNotFoundError: if the snapshot row does not exist.
    """
    snap = await db.get(ContextSnapshot, session_id)
    if snap is None:
        raise SessionNotFoundError(session_id)

    snap.soft_ask_count += 1
    await db.flush()
    return snap.soft_ask_count


async def reset_soft_ask_count(
    db: AsyncSession,
    session_id: uuid.UUID,
) -> None:
    """Reset the soft_ask_count to 0 (called when Context Gate returns PROCEED).

    Raises:
        SessionNotFoundError: if the snapshot row does not exist.
    """
    snap = await db.get(ContextSnapshot, session_id)
    if snap is None:
        raise SessionNotFoundError(session_id)

    snap.soft_ask_count = 0
    await db.flush()
