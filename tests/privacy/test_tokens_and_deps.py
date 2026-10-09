"""Signed session/admin tokens and the FastAPI dependencies that enforce them."""
from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Any

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from medbridge.api.middleware.error_handler import register_error_handlers
from medbridge.privacy import tokens
from medbridge.privacy.deps import require_admin, require_session_access
from medbridge.privacy.tokens import (
    decode_token,
    issue_admin_token,
    issue_admin_token_with_expiry,
    issue_session_token,
    verify_admin_token,
    verify_session_token,
)

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def test_session_token_round_trip() -> None:
    sid = uuid.uuid4()
    token, expires_at = issue_session_token(sid)
    assert token.startswith("v1.") and token.count(".") == 2
    assert verify_session_token(token, sid)
    assert verify_session_token(token, str(sid))
    assert expires_at.timestamp() > time.time()


def test_session_token_is_bound_to_one_session() -> None:
    token, _ = issue_session_token(uuid.uuid4())
    assert not verify_session_token(token, uuid.uuid4())


def test_tampered_payload_or_signature_is_rejected() -> None:
    sid = uuid.uuid4()
    token, _ = issue_session_token(sid)
    version, body, sig = token.split(".")
    forged_body = base64.urlsafe_b64encode(
        json.dumps({"typ": "session", "sid": str(uuid.uuid4()), "exp": 9999999999}).encode()
    ).rstrip(b"=").decode()
    assert not verify_session_token(f"{version}.{forged_body}.{sig}", sid)
    flipped = sig[:-1] + ("A" if sig[-1] != "A" else "B")
    assert not verify_session_token(f"{version}.{body}.{flipped}", sid)
    assert not verify_session_token(f"v2.{body}.{sig}", sid)
    for garbage in ("", "abc", "v1.x", "v1.!!.??", "v1.a.b.c"):
        assert not verify_session_token(garbage, sid)


def test_expired_token_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    sid = uuid.uuid4()
    token, _ = issue_session_token(sid)
    real_time = time.time
    monkeypatch.setattr(tokens.time, "time", lambda: real_time() + 8 * 24 * 3600)
    assert not verify_session_token(token, sid)


def test_token_signed_with_another_secret_is_rejected(override_settings: Any) -> None:
    sid = uuid.uuid4()
    token, _ = issue_session_token(sid)
    override_settings(SESSION_TOKEN_SECRET="a-different-secret")
    assert not verify_session_token(token, sid)


def test_admin_token(override_settings: Any) -> None:
    settings = override_settings(ADMIN_USERNAME="admin", ADMIN_PASSWORD="pw")
    token, expires_at = issue_admin_token_with_expiry(settings.ADMIN_USERNAME)
    assert verify_admin_token(token)
    assert expires_at.timestamp() > time.time()
    assert verify_admin_token(issue_admin_token("admin"))
    assert not verify_admin_token(issue_admin_token("someone-else"))
    # Session and admin tokens are not interchangeable.
    session_token, _ = issue_session_token(uuid.uuid4())
    assert not verify_admin_token(session_token)
    assert not verify_session_token(token, uuid.uuid4())
    assert decode_token(token)["typ"] == "admin"
    # Disabling the dashboard invalidates admin tokens.
    override_settings(ADMIN_PASSWORD="")
    assert not verify_admin_token(token)


def test_missing_secret_uses_ephemeral_secret_with_warning(
    override_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    override_settings(SESSION_TOKEN_SECRET="")
    monkeypatch.setattr(tokens, "_ephemeral_secret", None)
    with capture_logs() as logs:
        token, _ = issue_session_token("s1")
        assert verify_session_token(token, "s1")
    assert [e["event"] for e in logs].count("session_token_secret_missing") == 1


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/api/sessions/{session_id}/probe", dependencies=[Depends(require_session_access)])
    async def probe(session_id: uuid.UUID) -> dict[str, str]:
        return {"ok": str(session_id)}

    @app.get("/api/admin/probe")
    async def admin_probe(user: str = Depends(require_admin)) -> dict[str, str]:
        return {"user": user}

    return TestClient(app, raise_server_exceptions=False)


def test_session_access_requires_token(client: TestClient) -> None:
    resp = client.get(f"/api/sessions/{uuid.uuid4()}/probe")
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "UNAUTHORIZED"
    assert resp.headers["www-authenticate"] == "Bearer"


def test_session_access_rejects_invalid_or_foreign_token(client: TestClient) -> None:
    sid = uuid.uuid4()
    other_token, _ = issue_session_token(uuid.uuid4())
    for headers in ({"Authorization": "Bearer nope"}, {"Authorization": f"Bearer {other_token}"},
                    {"X-Session-Token": other_token}):
        resp = client.get(f"/api/sessions/{sid}/probe", headers=headers)
        assert resp.status_code == 403
        assert resp.json()["error_code"] == "FORBIDDEN"


def test_session_access_accepts_bearer_or_header(client: TestClient) -> None:
    sid = uuid.uuid4()
    token, _ = issue_session_token(sid)
    assert client.get(f"/api/sessions/{sid}/probe",
                      headers={"Authorization": f"Bearer {token}"}).status_code == 200
    assert client.get(f"/api/sessions/{sid}/probe",
                      headers={"X-Session-Token": token}).status_code == 200


def test_session_access_can_be_disabled(client: TestClient, override_settings: Any) -> None:
    override_settings(AUTH_REQUIRED="false")
    assert client.get(f"/api/sessions/{uuid.uuid4()}/probe").status_code == 200


def test_admin_dependency(client: TestClient, override_settings: Any) -> None:
    override_settings(ADMIN_USERNAME="admin", ADMIN_PASSWORD="pw")
    assert client.get("/api/admin/probe").status_code == 401
    bad = client.get("/api/admin/probe", headers={"Authorization": "Bearer x.y.z"})
    assert bad.status_code == 403
    session_token, _ = issue_session_token(uuid.uuid4())
    assert client.get("/api/admin/probe",
                      headers={"Authorization": f"Bearer {session_token}"}).status_code == 403
    ok = client.get("/api/admin/probe",
                    headers={"Authorization": f"Bearer {issue_admin_token('admin')}"})
    assert ok.status_code == 200 and ok.json() == {"user": "admin"}


def test_admin_dashboard_disabled_without_password(
    client: TestClient, override_settings: Any
) -> None:
    override_settings(ADMIN_PASSWORD="")
    resp = client.get("/api/admin/probe",
                      headers={"Authorization": f"Bearer {issue_admin_token('admin')}"})
    assert resp.status_code == 403
    assert "disabled" in resp.json()["message"]
