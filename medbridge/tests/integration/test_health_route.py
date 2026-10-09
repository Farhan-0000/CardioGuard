"""
Tests for ``GET /health`` and ``GET /api/health`` (medbridge/api/routes/health.py).

All probes (PostgreSQL, Qdrant, LLM, collection size, in-process model state)
are patched, so the status logic is tested hermetically.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from medbridge.api.routes import health as health_module
from medbridge.api.routes.health import (
    APP_VERSION,
    check_llm_health,
    collect_models_loaded,
    derive_status,
    health_router,
)
from medbridge.api.schemas.responses import HealthResponse
from medbridge.config import get_settings

LLM_OK = {
    "provider": "ollama",
    "reachable": True,
    "models": {"gemma3:4b": True},
    "loaded": ["gemma3:4b"],
}


def settings(**overrides: Any) -> Any:
    base = {
        "WARMUP_ON_STARTUP": False,
        "LLM_MODEL": "gemma3:4b",
        "LLM_MODEL_GATES": "",
        "LLM_MODEL_GENERATOR": "",
        "GROUNDING_VERIFIER_ENABLED": False,
    }
    base.update(overrides)
    return get_settings().model_copy(update=base)


@pytest.fixture
def probes(monkeypatch):
    state = SimpleNamespace(
        postgres=AsyncMock(return_value=True),
        qdrant=AsyncMock(return_value=True),
        llm=AsyncMock(return_value=dict(LLM_OK)),
        points=AsyncMock(return_value=905),
        settings=settings(),
        embedder=SimpleNamespace(loaded={"dense": True, "sparse": True}),
        reranker=SimpleNamespace(loaded=True),
    )
    monkeypatch.setattr(health_module, "check_postgres_health", state.postgres)
    monkeypatch.setattr(health_module, "check_qdrant_health", state.qdrant)
    monkeypatch.setattr(health_module, "check_llm_health", state.llm)
    monkeypatch.setattr(health_module, "get_qdrant_points", state.points)
    monkeypatch.setattr(health_module, "get_settings", lambda: state.settings)
    monkeypatch.setattr(health_module, "get_embedding_client", lambda: state.embedder)
    monkeypatch.setattr(health_module, "get_reranker", lambda: state.reranker)
    return state


def make_client(readiness: dict[str, Any] | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(health_router)
    app.include_router(health_router, prefix="/api")
    if readiness is not None:
        app.state.readiness = readiness
    return TestClient(app)


@pytest.mark.parametrize("path", ["/health", "/api/health"])
def test_healthy_with_all_fields(probes, path):
    response = make_client().get(path)
    assert response.status_code == 200
    body = HealthResponse.model_validate(response.json())
    assert body.status == "healthy"
    assert body.postgres_connected and body.qdrant_connected
    assert body.ollama_reachable is True
    assert body.llm_provider == "ollama"
    assert body.models == {"gemma3:4b": True}
    assert body.models_loaded == {
        "embedder_dense": True,
        "embedder_sparse": True,
        "reranker": True,
        "llm": True,
        "nli_verifier": False,
    }
    assert body.qdrant_points == 905
    assert body.local_only_mode == probes.settings.LOCAL_ONLY_MODE
    assert body.version == APP_VERSION


@pytest.mark.parametrize("failing", ["postgres", "qdrant"])
def test_storage_down_is_unhealthy_503(probes, failing):
    getattr(probes, failing).return_value = False
    response = make_client().get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"


def test_llm_unreachable_is_degraded_200(probes):
    probes.llm.return_value = {"provider": "ollama", "reachable": False, "models": {}}
    response = make_client().get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["ollama_reachable"] is False


def test_missing_model_is_degraded(probes):
    probes.llm.return_value = {**LLM_OK, "models": {"gemma3:4b": False}}
    assert make_client().get("/health").json()["status"] == "degraded"


def test_models_not_warm_is_degraded_when_warmup_enabled(probes):
    probes.settings = settings(WARMUP_ON_STARTUP=True)
    probes.embedder = SimpleNamespace(loaded={"dense": False, "sparse": False})
    probes.reranker = SimpleNamespace(loaded=False)
    probes.llm.return_value = {**LLM_OK, "loaded": []}
    body = make_client({"warmup": "running"}).get("/health").json()
    assert body["status"] == "degraded"
    assert body["models_loaded"]["embedder_dense"] is False


def test_readiness_flags_make_it_healthy_when_warm(probes):
    probes.settings = settings(WARMUP_ON_STARTUP=True, GROUNDING_VERIFIER_ENABLED=True)
    probes.llm.return_value = {**LLM_OK, "loaded": []}
    readiness = {
        "embedder_dense": True,
        "embedder_sparse": True,
        "reranker": True,
        "llm": True,
        "nli_verifier": True,
        "warmup": "complete",
    }
    body = make_client(readiness).get("/health").json()
    assert body["status"] == "healthy"
    assert body["models_loaded"]["nli_verifier"] is True


def test_remote_provider_has_no_ollama_flag(probes):
    probes.llm.return_value = {"provider": "groq", "reachable": True, "models": {"m": True}}
    body = make_client().get("/health").json()
    assert body["ollama_reachable"] is None
    assert body["llm_provider"] == "groq"


async def test_check_llm_health_never_raises():
    broken = MagicMock()
    broken.health = AsyncMock(side_effect=RuntimeError("boom"))
    with patch.object(health_module, "get_llm_wrapper", return_value=broken):
        result = await check_llm_health()
    assert result["reachable"] is False and result["models"] == {}


async def test_check_llm_health_passes_through():
    wrapper = MagicMock()
    wrapper.health = AsyncMock(return_value=dict(LLM_OK))
    with patch.object(health_module, "get_llm_wrapper", return_value=wrapper):
        assert await check_llm_health() == LLM_OK


def test_llm_residency_matches_tag_variants(probes):
    loaded = collect_models_loaded({}, {"loaded": ["qwen3:8b"]}, settings(LLM_MODEL="qwen3:8b"))
    assert loaded["llm"] is True
    partial = collect_models_loaded(
        {},
        {"loaded": ["qwen3:8b"]},
        settings(LLM_MODEL="qwen3:8b", LLM_MODEL_GENERATOR="gemma3:12b"),
    )
    assert partial["llm"] is False


def test_derive_status_table():
    warm = {"embedder_dense": True, "embedder_sparse": True, "reranker": True, "llm": True}
    s = settings(WARMUP_ON_STARTUP=True)
    assert derive_status(True, True, LLM_OK, warm, s) == "healthy"
    assert derive_status(False, True, LLM_OK, warm, s) == "unhealthy"
    assert derive_status(True, True, {"reachable": False}, warm, s) == "degraded"
    assert derive_status(True, True, LLM_OK, {**warm, "llm": False}, s) == "degraded"
    assert derive_status(True, True, LLM_OK, {}, settings()) == "healthy"


def test_nli_verifier_live_state_when_enabled(probes):
    enabled = settings(GROUNDING_VERIFIER_ENABLED=True)
    loaded_verifier = SimpleNamespace(loaded=True)
    with patch(
        "medbridge.ai.grounding_verifier.get_grounding_verifier", return_value=loaded_verifier
    ):
        assert collect_models_loaded({}, {}, enabled)["nli_verifier"] is True
        # disabled verifier: never reported as loaded, the model is not consulted
        assert collect_models_loaded({}, {}, settings())["nli_verifier"] is False
    with patch(
        "medbridge.ai.grounding_verifier.get_grounding_verifier", side_effect=RuntimeError("x")
    ):
        assert collect_models_loaded({}, {}, enabled)["nli_verifier"] is False
    # the warm-up flag alone is enough
    assert collect_models_loaded({"nli_verifier": True}, {}, enabled)["nli_verifier"] is True
