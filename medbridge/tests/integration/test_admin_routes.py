"""Admin routes: login (constant-time, rate limited) and the redacted audit feed."""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.api.middleware.error_handler import register_error_handlers
from medbridge.api.routes.admin import ERASED_MARKER, PREVIEW_CHARS, admin_router, preview
from medbridge.api.schemas.enums import ActionEnum
from medbridge.config import get_settings
from medbridge.db.connection import get_session
from medbridge.privacy.middleware import register_security_middleware
from medbridge.state import turn_store
from medbridge.state.session_manager import create_session
from medbridge.state.turn_store import AuditEntry, TurnRecord


@pytest.fixture(autouse=True)
def admin_credentials(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "correct-horse")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _app(db: Any) -> FastAPI:
    app = FastAPI()
    register_error_handlers(app)
    register_security_middleware(app, get_settings())
    app.include_router(admin_router, prefix="/api")

    async def override() -> AsyncIterator[Any]:
        yield db

    app.dependency_overrides[get_session] = override
    return app


async def _client(db: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=_app(db)), base_url="http://test")


async def _login(http: httpx.AsyncClient) -> dict[str, str]:
    resp = await http.post("/api/admin/login",
                           json={"username": "admin", "password": "correct-horse"})
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['token']}"}


async def test_login_success_and_failure() -> None:
    async with await _client(MagicMock()) as http:
        ok = await http.post("/api/admin/login",
                             json={"username": "admin", "password": "correct-horse"})
        assert ok.status_code == 200
        body = ok.json()
        assert body["token"].startswith("v1.")
        assert datetime.fromisoformat(body["expires_at"]) > datetime.now(UTC)
        for creds in ({"username": "admin", "password": "wrong"},
                      {"username": "root", "password": "correct-horse"}):
            bad = await http.post("/api/admin/login", json=creds)
            assert bad.status_code == 401 and bad.json()["error_code"] == "UNAUTHORIZED"


async def test_login_is_rate_limited() -> None:
    async with await _client(MagicMock()) as http:
        statuses = [(await http.post("/api/admin/login",
                                     json={"username": "admin", "password": "x"})).status_code
                    for _ in range(6)]
    assert statuses == [401] * 5 + [429]


async def test_dashboard_disabled_without_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_PASSWORD", "")
    get_settings.cache_clear()
    async with await _client(MagicMock()) as http:
        resp = await http.post("/api/admin/login", json={"username": "admin", "password": "x"})
        assert resp.status_code == 403
        assert (await http.get("/api/admin/audit")).status_code == 403


async def test_audit_requires_admin_token(monkeypatch: pytest.MonkeyPatch) -> None:
    sid = uuid.uuid4()
    entries = [
        AuditEntry(uuid.uuid4(), sid, datetime.now(UTC), "ANSWER", "PROCEED", "ANSWER", 812, 3,
                   "My name is Rahul Sharma and my BP is 150/95", "Stage 2 [1].", False),
        AuditEntry(uuid.uuid4(), sid, datetime.now(UTC), "ESCALATE", None, None, 4, 0,
                   None, None, True, legacy_plaintext=True),
    ]
    listing = AsyncMock(return_value=entries)
    monkeypatch.setattr(turn_store, "list_audit_entries", listing)
    async with await _client(MagicMock()) as http:
        assert (await http.get("/api/admin/audit")).status_code == 401
        assert (await http.get("/api/admin/audit",
                               headers={"Authorization": "Bearer v1.a.b"})).status_code == 403
        resp = await http.get("/api/admin/audit?limit=2", headers=await _login(http))
        assert (await http.get("/api/admin/audit?limit=0",
                               headers=await _login(http))).status_code == 422
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-store"
    body = resp.json()
    assert body["count"] == 2
    first, second = body["items"]
    assert first["request_preview"] == "My name is [NAME] and my BP is 150/95"
    assert first["session_ref"] == str(sid)[:8] and first["evidence_chunks"] == 3
    assert second["request_preview"] == ERASED_MARKER and second["erased"] is True
    # a legacy plaintext row of an erased session is flagged, never shown
    assert second["legacy_plaintext"] is True and first["legacy_plaintext"] is False
    assert listing.await_args.args[1] == 2


def test_preview_truncates() -> None:
    text = "word " * 100
    out = preview(text)
    assert len(out) == PREVIEW_CHARS and out.endswith("…")
    assert preview(None) == ERASED_MARKER


@pytest.fixture
async def db() -> AsyncIterator[AsyncSession]:
    from medbridge.db.connection import get_sessionmaker

    async with get_sessionmaker()() as session:
        yield session
        await session.rollback()


@pytest.mark.postgres
async def test_audit_feed_decrypts_redacts_and_marks_erased(db: AsyncSession) -> None:
    live, erased = await create_session(db), await create_session(db)
    for sid, text in ((live, "I'm Priya Nair, BP 160/100"), (erased, "call me on 9876543210")):
        await turn_store.persist_turn(db, TurnRecord(
            session_id=sid, user_message=text, assistant_message="Please re-measure.",
            final_action=ActionEnum.GENERALIZE, evidence_chunk_ids=["a", "b"], latency_ms=10,
        ))
    await turn_store.erase_session(db, erased)
    async with await _client(db) as http:
        resp = await http.get("/api/admin/audit?limit=500", headers=await _login(http))
    items = {i["session_ref"]: i for i in resp.json()["items"]}
    assert items[str(live)[:8]]["request_preview"] == "I'm [NAME], BP 160/100"
    assert items[str(live)[:8]]["evidence_chunks"] == 2
    assert items[str(erased)[:8]]["request_preview"] == ERASED_MARKER
    assert items[str(erased)[:8]]["response_preview"] == ERASED_MARKER
