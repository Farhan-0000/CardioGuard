"""
Integration tests for application assembly and lifespan (medbridge/main.py, Module M-19).

Verifies the background warm-up and readiness flags, the retention task,
shutdown clean-up, CORS/security middleware (with the plain-CORS fallback),
router mounting and the web UI hook. Database, LLM and Qdrant are patched.
"""

from __future__ import annotations

import asyncio
import builtins
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from medbridge import main as main_module
from medbridge.api.routes.health import APP_VERSION
from medbridge.config import Settings, get_settings
from medbridge.main import app, create_app, run_warmup


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "CORS_ORIGINS": ["http://localhost:8501", "https://app.medbridge.internal"],
        "WARMUP_ON_STARTUP": False,
    }
    base.update(overrides)
    return get_settings().model_copy(update=base)


@pytest.fixture
def lifecycle(monkeypatch):
    """Patch every external resource touched by the lifespan."""
    mocks = MagicMock()
    mocks.init_db = AsyncMock()
    mocks.close_db = AsyncMock()
    mocks.llm = MagicMock(close=AsyncMock())
    mocks.retriever = MagicMock(close=AsyncMock())
    mocks.orchestrator = MagicMock(
        warmup=AsyncMock(return_value={"embedder_dense": True, "llm": True})
    )
    monkeypatch.setattr(main_module, "init_db", mocks.init_db)
    monkeypatch.setattr(main_module, "close_db", mocks.close_db)
    monkeypatch.setattr(main_module, "get_llm_wrapper", lambda: mocks.llm)
    monkeypatch.setattr(main_module, "get_hybrid_retriever", lambda: mocks.retriever)
    monkeypatch.setattr(main_module, "get_orchestrator", lambda: mocks.orchestrator)
    return mocks


class TestLifespan:
    def test_startup_and_shutdown_without_warmup(self, lifecycle) -> None:
        test_app = create_app(make_settings())
        with patch("medbridge.privacy.retention.retention_loop", new=AsyncMock()):
            with TestClient(test_app):
                lifecycle.init_db.assert_awaited_once()
                assert test_app.state.readiness == {"warmup": "disabled"}
        lifecycle.orchestrator.warmup.assert_not_called()
        lifecycle.llm.close.assert_awaited_once()
        lifecycle.retriever.close.assert_awaited_once()
        lifecycle.close_db.assert_awaited_once()

    def test_background_warmup_records_readiness(self, lifecycle) -> None:
        test_app = create_app(make_settings(WARMUP_ON_STARTUP=True))
        with TestClient(test_app) as client:
            for _ in range(100):
                if test_app.state.readiness.get("warmup") == "complete":
                    break
                client.get("/openapi.json")
            assert test_app.state.readiness["warmup"] == "complete"
            assert test_app.state.readiness["embedder_dense"] is True
        lifecycle.orchestrator.warmup.assert_awaited_once()

    def test_startup_survives_db_failure(self, lifecycle) -> None:
        lifecycle.init_db.side_effect = ConnectionError("db down")
        lifecycle.close_db.side_effect = RuntimeError("already closed")
        with TestClient(create_app(make_settings())):
            pass
        lifecycle.close_db.assert_awaited_once()

    def test_retention_loop_started_and_stopped(self, lifecycle) -> None:
        seen: dict[str, Any] = {}

        async def fake_loop(stop_event: asyncio.Event) -> None:
            seen["started"] = True
            try:
                await stop_event.wait()
            finally:  # stopped by the event or cancelled on shutdown
                seen["stopped"] = True

        with patch("medbridge.privacy.retention.retention_loop", new=fake_loop):
            with TestClient(create_app(make_settings())):
                pass
        assert seen == {"started": True, "stopped": True}


async def test_run_warmup_failure_is_recorded(lifecycle) -> None:
    lifecycle.orchestrator.warmup.side_effect = RuntimeError("ollama down")
    fake_app = MagicMock()
    fake_app.state.readiness = {}
    await run_warmup(fake_app)
    assert fake_app.state.readiness["warmup"] == "failed"


class TestMiddleware:
    def test_cors_allowed_origin_preflight(self) -> None:
        client = TestClient(create_app(make_settings()))
        response = client.options(
            "/api/sessions",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:8501"

    def test_cors_allowed_origin_get_request(self, lifecycle) -> None:
        client = TestClient(create_app(make_settings()))
        response = client.get(
            "/api/config/public", headers={"Origin": "https://app.medbridge.internal"}
        )
        assert (
            response.headers.get("access-control-allow-origin") == "https://app.medbridge.internal"
        )

    def test_cors_disallowed_origin_rejected(self) -> None:
        client = TestClient(create_app(make_settings()))
        response = client.options(
            "/api/sessions",
            headers={
                "Origin": "http://unauthorized-attacker.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" not in response.headers

    def test_security_middleware_is_registered(self) -> None:
        with patch("medbridge.privacy.middleware.register_security_middleware") as register:
            test_app = create_app(make_settings())
        register.assert_called_once()
        assert register.call_args.args[0] is test_app

    def test_fallback_to_plain_cors_when_privacy_middleware_missing(self, monkeypatch) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "medbridge.privacy.middleware":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.delitem(sys.modules, "medbridge.privacy.middleware", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)
        test_app = create_app(make_settings())
        assert any(m.cls is CORSMiddleware for m in test_app.user_middleware)


class TestAssembly:
    def test_metadata(self) -> None:
        assert app.title == "MedBridge API"
        assert app.version == APP_VERSION == "4.0.0"

    def test_routes_mounted(self) -> None:
        paths = set(app.openapi()["paths"])  # routers are included lazily in FastAPI >= 0.140
        for expected in (
            "/health",
            "/api/health",
            "/api/sessions",
            "/api/sessions/{session_id}/history",
            "/api/sessions/{session_id}/messages",
            "/api/sessions/{session_id}/messages/stream",
            "/api/sessions/{session_id}/regenerate",
            "/api/metrics",
            "/api/config/public",
        ):
            assert expected in paths, expected

    def test_public_config_reachable(self, lifecycle) -> None:
        response = TestClient(app).get("/api/config/public")
        assert response.status_code == 200
        assert response.json()["app_name"]

    def test_session_routes_require_token(self) -> None:
        client = TestClient(app, raise_server_exceptions=False)
        for path in ("messages", "messages/stream", "regenerate"):
            response = client.post(
                f"/api/sessions/00000000-0000-0000-0000-000000000000/{path}", json={"message": "hi"}
            )
            assert response.status_code == 401, path

    def test_metrics_requires_admin(self) -> None:
        response = TestClient(app, raise_server_exceptions=False).get("/api/metrics")
        assert response.status_code in (401, 403)

    def test_invalid_uuid_is_422(self) -> None:
        response = TestClient(app).get("/api/sessions/invalid-uuid/history")
        assert response.status_code == 422

    def test_bare_paths_not_mounted(self) -> None:
        assert TestClient(app).post("/sessions").status_code in (404, 405)

    def test_web_ui_mounted_when_available(self) -> None:
        with patch("medbridge.web.mount_web") as mount:
            test_app = create_app(make_settings())
        mount.assert_called_once_with(test_app)
