"""
Token edge cases and the right-to-erasure dependency (privacy-security-2/-4).

* A non-ASCII ``X-Session-Token`` / bearer value is an invalid token (403),
  never an HTTP 500, on session and admin routes.
* ``DELETE /api/sessions/{id}`` accepts a correctly signed token for that
  session after it expired (bounded grace); every other route keeps strict
  expiry; other sessions, bad signatures and tokens beyond the grace are refused.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from unittest.mock import AsyncMock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from medbridge.api.middleware.error_handler import register_error_handlers
from medbridge.api.routes.sessions import sessions_router
from medbridge.config import get_settings
from medbridge.db.connection import get_session
from medbridge.privacy import tokens
from medbridge.privacy.deps import (
    ERASURE_TOKEN_GRACE_DAYS,
    erasure_token_grace_seconds,
    require_admin,
    require_session_access,
)
from medbridge.privacy.tokens import decode_token, issue_session_token, verify_session_token
from medbridge.state import turn_store
from medbridge.state.turn_store import SessionState

DAY = 86_400
NON_ASCII = "v1.\xe9.abc"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(sessions_router, prefix="/api")

    @app.get("/api/admin/probe")
    async def admin_probe(user: str = Depends(require_admin)) -> dict[str, str]:
        return {"user": user}

    @app.get("/api/sessions/{session_id}/probe", dependencies=[Depends(require_session_access)])
    async def probe(session_id: uuid.UUID) -> dict[str, str]:
        return {"ok": str(session_id)}

    async def no_db() -> object:
        yield object()

    app.dependency_overrides[get_session] = no_db
    monkeypatch.setattr(turn_store, "erase_session", AsyncMock())
    monkeypatch.setattr(turn_store, "load_session_state", AsyncMock(return_value=SessionState()))
    monkeypatch.setattr(turn_store, "get_history", AsyncMock(return_value=[]))
    yield TestClient(app, raise_server_exceptions=False)


def later(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    real_time = time.time
    monkeypatch.setattr(tokens.time, "time", lambda: real_time() + seconds)


# --- privacy-security-4 -----------------------------------------------------------


def test_non_ascii_token_is_invalid_not_an_error() -> None:
    assert decode_token(NON_ASCII) is None
    assert decode_token("v1.abc.€") is None
    assert not verify_session_token(NON_ASCII, uuid.uuid4())


@pytest.mark.parametrize("header", ["x-session-token", "authorization"])
def test_non_ascii_header_is_403_on_session_and_admin_routes(
    client: TestClient, header: str
) -> None:
    value = NON_ASCII if header == "x-session-token" else f"Bearer {NON_ASCII}"
    raw = {header: value.encode("latin-1")}  # Starlette decodes header bytes as latin-1
    sid = uuid.uuid4()
    for method, path in (
        ("get", f"/api/sessions/{sid}/history"),
        ("get", f"/api/sessions/{sid}/probe"),
        ("delete", f"/api/sessions/{sid}"),
        ("get", "/api/admin/probe"),
    ):
        response = getattr(client, method)(path, headers=raw)
        assert response.status_code == 403, (path, response.status_code)


# --- privacy-security-2 -----------------------------------------------------------


def test_grace_covers_the_retention_window() -> None:
    days = max(ERASURE_TOKEN_GRACE_DAYS, get_settings().DATA_RETENTION_DAYS)
    assert erasure_token_grace_seconds() == days * DAY
    assert erasure_token_grace_seconds() >= get_settings().DATA_RETENTION_DAYS * DAY


def test_erasure_accepts_an_expired_token_for_its_own_session(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = uuid.uuid4()
    token, _ = issue_session_token(sid)
    later(monkeypatch, get_settings().SESSION_TOKEN_TTL_SECONDS + 20 * DAY)  # day 27
    auth = {"Authorization": f"Bearer {token}"}

    assert client.get(f"/api/sessions/{sid}/history", headers=auth).status_code == 403
    response = client.delete(f"/api/sessions/{sid}", headers=auth)
    assert response.status_code == 204
    assert turn_store.erase_session.await_args.args[1] == sid  # type: ignore[attr-defined]


def test_erasure_still_rejects_foreign_forged_and_too_old_tokens(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid, other = uuid.uuid4(), uuid.uuid4()
    token, _ = issue_session_token(sid)
    foreign, _ = issue_session_token(other)
    version, body, signature = token.split(".")
    forged = f"{version}.{body}.{signature[:-1]}{'A' if signature[-1] != 'A' else 'B'}"
    assert client.delete(f"/api/sessions/{sid}").status_code == 401
    for bad in (foreign, forged):
        response = client.delete(f"/api/sessions/{sid}", headers={"X-Session-Token": bad})
        assert response.status_code == 403

    later(monkeypatch, get_settings().SESSION_TOKEN_TTL_SECONDS + erasure_token_grace_seconds()
          + 60)
    response = client.delete(f"/api/sessions/{sid}", headers={"X-Session-Token": token})
    assert response.status_code == 403  # the grace is bounded
    turn_store.erase_session.assert_not_awaited()  # type: ignore[attr-defined]


def test_grace_is_opt_in_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    sid = uuid.uuid4()
    token, _ = issue_session_token(sid)
    later(monkeypatch, get_settings().SESSION_TOKEN_TTL_SECONDS + DAY)
    assert not verify_session_token(token, sid)
    assert verify_session_token(token, sid, expired_grace_seconds=2 * DAY)
    assert not verify_session_token(token, sid, expired_grace_seconds=DAY // 2)
