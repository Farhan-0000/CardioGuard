"""
Envelope encryption at rest + crypto-shredding (ADL-031, ADL-032).

Key hierarchy::

    ENCRYPTION_MASTER_KEY (KEK, from the environment, never stored in the DB)
        └── wraps one random data key (DEK) per session  → session_keys.wrapped_key
                └── encrypts message text, audit text, event payloads, snapshot,
                    feedback comments and the PHI part of the audit trace

Formats (Fernet = AES-128-CBC + HMAC-SHA256, from ``cryptography``):

* text columns   → ``"enc:v1:<fernet token>"``
* JSONB columns  → ``{"enc": "v1", "ct": "<fernet token>"}``

Readers accept legacy plaintext (rows written before encryption existed).

**Right to erasure** = deleting the ``session_keys`` row ("crypto-shredding").
Every ciphertext of that session — including rows in the append-only
``clinical_events`` and ``audit_logs`` tables, which can never be updated or
deleted — becomes permanently unreadable, so erasure does not violate the
immutability invariant.

Limitation: only *ciphertext* is shredded. Rows written before encryption
existed (or with ``ENCRYPTION_ENABLED=false``) are plaintext; in the append-only
tables they can never be rewritten, so erasure counts them into
``sessions.legacy_plaintext_rows`` and every reader hides the erased session
(see ``medbridge/privacy/README.md`` — reset such databases).

Unwrapped DEKs are kept in a small in-process LRU cache that is invalidated on
shred. Keys created inside a not-yet-committed transaction live in
``AsyncSession.info`` (never in the shared cache), so a rolled-back session
creation can never leave a usable key behind.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import uuid
from collections import OrderedDict
from typing import Any

import structlog
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.config import get_settings
from medbridge.db.models import Session, SessionKey

logger = structlog.get_logger(__name__)

ENC_PREFIX = "enc:v1:"
ENC_VERSION = "v1"
_DEK_CACHE_SIZE = 1024
_PENDING_INFO_KEY = "medbridge_pending_deks"


class DecryptionError(ValueError):
    """Ciphertext could not be decrypted with the session key."""


class SessionCipher:
    """Encrypts/decrypts one session's data with its data key.

    ``fernet=None`` means "no key": values pass through unchanged. With
    ``encrypt_enabled=False`` (``ENCRYPTION_ENABLED=false``) new values are
    written in plaintext but existing ciphertext can still be read.
    """

    def __init__(self, fernet: Fernet | None, *, encrypt_enabled: bool = True) -> None:
        self._fernet = fernet
        self._encrypt = encrypt_enabled and fernet is not None

    @classmethod
    def passthrough(cls) -> SessionCipher:
        """A cipher that stores plaintext (encryption disabled)."""
        return cls(None, encrypt_enabled=False)

    @property
    def encrypts(self) -> bool:
        return self._encrypt

    # -- text ---------------------------------------------------------------

    def encrypt(self, plaintext: str) -> str:
        if not self._encrypt or self._fernet is None:
            return plaintext
        return ENC_PREFIX + self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, value: str) -> str:
        """Decrypt ``enc:v1:`` text; legacy plaintext is returned unchanged."""
        if not isinstance(value, str) or not value.startswith(ENC_PREFIX):
            return value
        if self._fernet is None:
            raise DecryptionError("no data key available for encrypted value")
        try:
            return self._fernet.decrypt(value[len(ENC_PREFIX):].encode("ascii")).decode("utf-8")
        except (InvalidToken, ValueError) as exc:
            raise DecryptionError("ciphertext does not match the session key") from exc

    # -- JSON ---------------------------------------------------------------

    def encrypt_json(self, value: Any) -> Any:
        """Encrypt a JSON-serialisable value into ``{"enc": "v1", "ct": ...}``.

        When encryption is disabled the value is returned unchanged.
        """
        if not self._encrypt or self._fernet is None:
            return value
        raw = json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)
        token = self._fernet.encrypt(raw.encode("utf-8")).decode("ascii")
        return {"enc": ENC_VERSION, "ct": token}

    def decrypt_json(self, value: Any) -> Any:
        """Decrypt an envelope produced by :meth:`encrypt_json` (legacy values pass through)."""
        if not is_encrypted_json(value):
            return value
        if self._fernet is None:
            raise DecryptionError("no data key available for encrypted value")
        try:
            raw = self._fernet.decrypt(str(value["ct"]).encode("ascii"))
        except (InvalidToken, ValueError) as exc:
            raise DecryptionError("ciphertext does not match the session key") from exc
        return json.loads(raw.decode("utf-8"))


def is_encrypted_text(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(ENC_PREFIX)


def is_encrypted_json(value: Any) -> bool:
    return isinstance(value, dict) and value.get("enc") == ENC_VERSION and "ct" in value


# ---------------------------------------------------------------------------
# Master key (KEK)
# ---------------------------------------------------------------------------

_kek_cache: tuple[str, Fernet] | None = None
_ephemeral_master_key: str | None = None


def _derive_fernet_key(secret: str) -> bytes:
    """Return a Fernet key: the configured key itself, or a SHA-256 derivation of it."""
    try:
        if len(base64.urlsafe_b64decode(secret.encode("ascii"))) == 32:
            return secret.encode("ascii")
    except (binascii.Error, ValueError, UnicodeEncodeError):
        pass
    logger.warning(
        "encryption_master_key_not_fernet",
        hint="ENCRYPTION_MASTER_KEY is not a 32-byte urlsafe-base64 key; deriving one with "
        "SHA-256. Generate a proper key with scripts/setup_env.py.",
    )
    return base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())


def _get_kek() -> Fernet:
    """The key-encryption key derived from ``ENCRYPTION_MASTER_KEY``."""
    global _kek_cache, _ephemeral_master_key
    configured = get_settings().ENCRYPTION_MASTER_KEY.strip()
    if not configured:
        if _ephemeral_master_key is None:
            _ephemeral_master_key = Fernet.generate_key().decode("ascii")
            logger.warning(
                "encryption_master_key_missing",
                message="!!! ENCRYPTION_MASTER_KEY is empty: using an EPHEMERAL in-memory key. "
                "Everything encrypted by this process becomes unreadable after a restart. "
                "Run `python scripts/setup_env.py` to generate a persistent key. !!!",
            )
        configured = _ephemeral_master_key
    if _kek_cache is None or _kek_cache[0] != configured:
        _kek_cache = (configured, Fernet(_derive_fernet_key(configured)))
    return _kek_cache[1]


# ---------------------------------------------------------------------------
# DEK cache
# ---------------------------------------------------------------------------

_dek_cache: OrderedDict[uuid.UUID, bytes] = OrderedDict()


def _cache_get(session_id: uuid.UUID) -> bytes | None:
    dek = _dek_cache.get(session_id)
    if dek is not None:
        _dek_cache.move_to_end(session_id)
    return dek


def _cache_put(session_id: uuid.UUID, dek: bytes) -> None:
    _dek_cache[session_id] = dek
    _dek_cache.move_to_end(session_id)
    while len(_dek_cache) > _DEK_CACHE_SIZE:
        _dek_cache.popitem(last=False)


def clear_key_cache() -> None:
    """Drop all cached data keys (tests, key rotation)."""
    _dek_cache.clear()


def _pending(db: AsyncSession) -> dict[uuid.UUID, bytes]:
    """Data keys created in this (uncommitted) DB session."""
    info: dict[str, Any] = db.info
    pending: dict[uuid.UUID, bytes] = info.setdefault(_PENDING_INFO_KEY, {})
    return pending


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _cipher(dek: bytes) -> SessionCipher:
    return SessionCipher(Fernet(dek), encrypt_enabled=get_settings().ENCRYPTION_ENABLED)


async def create_session_key(db: AsyncSession, session_id: uuid.UUID) -> None:
    """Generate a data key for ``session_id`` and add its wrapped form to ``db``.

    Does not flush or commit. No-op when ``ENCRYPTION_ENABLED`` is false.
    """
    if not get_settings().ENCRYPTION_ENABLED:
        return
    dek = Fernet.generate_key()
    wrapped = _get_kek().encrypt(dek).decode("ascii")
    db.add(SessionKey(session_id=session_id, wrapped_key=wrapped))
    _pending(db)[session_id] = dek


async def get_session_cipher(db: AsyncSession, session_id: uuid.UUID) -> SessionCipher | None:
    """Return the cipher for a session, or ``None`` if it was erased (or does not exist).

    Sessions created before encryption existed get a key lazily (the row is
    added to ``db`` and persisted by the caller's flush/commit).
    """
    pending = _pending(db).get(session_id)
    if pending is not None:
        return _cipher(pending)

    session = await db.get(Session, session_id)
    if session is None or session.erased_at is not None:
        return None

    cached = _cache_get(session_id)
    if cached is not None:
        return _cipher(cached)

    row = await db.get(SessionKey, session_id)
    if row is None:
        if not get_settings().ENCRYPTION_ENABLED:
            return SessionCipher.passthrough()
        await create_session_key(db, session_id)
        return _cipher(_pending(db)[session_id])
    try:
        dek = _get_kek().decrypt(row.wrapped_key.encode("ascii"))
    except InvalidToken:
        logger.error(
            "session_key_unwrap_failed",
            session_id=str(session_id),
            hint="ENCRYPTION_MASTER_KEY changed since this session was created; its data is "
            "unreadable (equivalent to crypto-shredded).",
        )
        return None
    _cache_put(session_id, dek)
    return _cipher(dek)


async def shred_session_key(db: AsyncSession, session_id: uuid.UUID) -> None:
    """Destroy a session's data key (crypto-shredding). Does not commit."""
    await db.execute(delete(SessionKey).where(SessionKey.session_id == session_id))
    _dek_cache.pop(session_id, None)
    _pending(db).pop(session_id, None)
