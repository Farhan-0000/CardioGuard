"""Envelope encryption: SessionCipher formats, master key handling, key lookup/shredding."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.fernet import Fernet
from structlog.testing import capture_logs

from medbridge.db.models import Session, SessionKey
from medbridge.privacy import crypto
from medbridge.privacy.crypto import (
    ENC_PREFIX,
    DecryptionError,
    SessionCipher,
    create_session_key,
    get_session_cipher,
    is_encrypted_json,
    shred_session_key,
)


@pytest.fixture
def cipher() -> SessionCipher:
    return SessionCipher(Fernet(Fernet.generate_key()))


# ---------------------------------------------------------------------------
# SessionCipher
# ---------------------------------------------------------------------------

def test_text_round_trip_uses_prefix(cipher: SessionCipher) -> None:
    token = cipher.encrypt("My BP is 150/95 — Rahul")
    assert token.startswith(ENC_PREFIX)
    assert "150/95" not in token and "Rahul" not in token
    assert cipher.decrypt(token) == "My BP is 150/95 — Rahul"


def test_json_round_trip_uses_envelope(cipher: SessionCipher) -> None:
    payload = {"systolic": 150, "diastolic": 95, "note": "Telma 40"}
    envelope = cipher.encrypt_json(payload)
    assert set(envelope) == {"enc", "ct"} and envelope["enc"] == "v1"
    assert is_encrypted_json(envelope)
    assert "Telma" not in str(envelope)
    assert cipher.decrypt_json(envelope) == payload


def test_legacy_plaintext_is_accepted(cipher: SessionCipher) -> None:
    assert cipher.decrypt("plain legacy text") == "plain legacy text"
    assert cipher.decrypt_json({"systolic": 140}) == {"systolic": 140}


def test_wrong_key_raises_decryption_error(cipher: SessionCipher) -> None:
    other = SessionCipher(Fernet(Fernet.generate_key()))
    with pytest.raises(DecryptionError):
        other.decrypt(cipher.encrypt("secret"))
    with pytest.raises(DecryptionError):
        other.decrypt_json(cipher.encrypt_json({"a": 1}))


def test_passthrough_cipher_stores_plaintext() -> None:
    passthrough = SessionCipher.passthrough()
    assert not passthrough.encrypts
    assert passthrough.encrypt("hello") == "hello"
    assert passthrough.encrypt_json({"a": 1}) == {"a": 1}
    with pytest.raises(DecryptionError):
        passthrough.decrypt(ENC_PREFIX + "abc")


def test_disabled_encryption_still_reads_old_ciphertext() -> None:
    key = Fernet.generate_key()
    writer = SessionCipher(Fernet(key))
    reader = SessionCipher(Fernet(key), encrypt_enabled=False)
    assert reader.encrypt("new") == "new"
    assert reader.decrypt(writer.encrypt("old")) == "old"


# ---------------------------------------------------------------------------
# Master key (KEK)
# ---------------------------------------------------------------------------

def test_missing_master_key_uses_ephemeral_key_with_warning(
    override_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    override_settings(ENCRYPTION_MASTER_KEY="")
    monkeypatch.setattr(crypto, "_ephemeral_master_key", None)
    monkeypatch.setattr(crypto, "_kek_cache", None)
    with capture_logs() as logs:
        first = crypto._get_kek()
        second = crypto._get_kek()
    assert first is second
    warnings = [e for e in logs if e["event"] == "encryption_master_key_missing"]
    assert len(warnings) == 1 and warnings[0]["log_level"] == "warning"


def test_non_fernet_master_key_is_derived(
    override_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    override_settings(ENCRYPTION_MASTER_KEY="correct horse battery staple")
    monkeypatch.setattr(crypto, "_kek_cache", None)
    with capture_logs() as logs:
        kek = crypto._get_kek()
    assert kek.decrypt(kek.encrypt(b"dek")) == b"dek"
    assert any(e["event"] == "encryption_master_key_not_fernet" for e in logs)


# ---------------------------------------------------------------------------
# Key lookup with a mocked database
# ---------------------------------------------------------------------------

def _mock_db(rows: dict[tuple[type, uuid.UUID], Any]) -> MagicMock:
    db = MagicMock()
    db.info = {}
    db.get = AsyncMock(side_effect=lambda model, key: rows.get((model, key)))
    db.execute = AsyncMock()
    return db


async def test_create_session_key_wraps_and_registers_pending() -> None:
    sid = uuid.uuid4()
    db = _mock_db({})
    await create_session_key(db, sid)
    added = db.add.call_args.args[0]
    assert isinstance(added, SessionKey) and added.session_id == sid
    dek = crypto._get_kek().decrypt(added.wrapped_key.encode())
    assert db.info[crypto._PENDING_INFO_KEY][sid] == dek
    cipher = await get_session_cipher(db, sid)
    assert cipher is not None and cipher.encrypts
    db.get.assert_not_awaited()  # served from the pending key


async def test_get_session_cipher_unwraps_and_caches() -> None:
    sid = uuid.uuid4()
    dek = Fernet.generate_key()
    rows = {
        (Session, sid): Session(session_id=sid),
        (SessionKey, sid): SessionKey(
            session_id=sid, wrapped_key=crypto._get_kek().encrypt(dek).decode()
        ),
    }
    db = _mock_db(rows)
    first = await get_session_cipher(db, sid)
    assert first is not None
    assert SessionCipher(Fernet(dek)).decrypt(first.encrypt("x")) == "x"
    calls = db.get.await_count
    await get_session_cipher(db, sid)
    assert db.get.await_count == calls + 1  # session row only; key served from cache


async def test_erased_or_missing_session_has_no_cipher() -> None:
    sid = uuid.uuid4()
    erased = Session(session_id=sid, erased_at=datetime.now(UTC))
    assert await get_session_cipher(_mock_db({(Session, sid): erased}), sid) is None
    assert await get_session_cipher(_mock_db({}), uuid.uuid4()) is None


async def test_legacy_session_gets_key_lazily() -> None:
    sid = uuid.uuid4()
    db = _mock_db({(Session, sid): Session(session_id=sid)})
    cipher = await get_session_cipher(db, sid)
    assert cipher is not None and cipher.encrypts
    assert isinstance(db.add.call_args.args[0], SessionKey)


async def test_encryption_disabled_returns_passthrough(override_settings: Any) -> None:
    override_settings(ENCRYPTION_ENABLED="false")
    sid = uuid.uuid4()
    db = _mock_db({(Session, sid): Session(session_id=sid)})
    await create_session_key(db, sid)
    db.add.assert_not_called()
    cipher = await get_session_cipher(db, sid)
    assert cipher is not None and not cipher.encrypts


async def test_key_wrapped_with_another_master_key_is_unreadable() -> None:
    sid = uuid.uuid4()
    foreign = Fernet(Fernet.generate_key()).encrypt(Fernet.generate_key()).decode()
    db = _mock_db({
        (Session, sid): Session(session_id=sid),
        (SessionKey, sid): SessionKey(session_id=sid, wrapped_key=foreign),
    })
    with capture_logs() as logs:
        assert await get_session_cipher(db, sid) is None
    assert any(e["event"] == "session_key_unwrap_failed" for e in logs)


async def test_shred_deletes_key_and_invalidates_cache() -> None:
    sid = uuid.uuid4()
    crypto._cache_put(sid, Fernet.generate_key())
    db = _mock_db({})
    db.info[crypto._PENDING_INFO_KEY] = {sid: b"pending"}
    await shred_session_key(db, sid)
    db.execute.assert_awaited_once()
    assert "DELETE FROM session_keys" in str(db.execute.await_args.args[0])
    assert crypto._cache_get(sid) is None
    assert sid not in db.info[crypto._PENDING_INFO_KEY]
