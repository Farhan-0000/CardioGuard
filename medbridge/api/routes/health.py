"""
Health & readiness routes (Module M-04) — ``GET /health`` and ``GET /api/health``.

Status semantics:

* ``unhealthy`` (HTTP 503) — PostgreSQL or Qdrant is unreachable;
* ``degraded``  (HTTP 200) — storage is up but the LLM server is unreachable,
  a configured model is missing, or (when ``WARMUP_ON_STARTUP``) the models
  have not finished warming up yet;
* ``healthy``   (HTTP 200) — everything is ready.

``models_loaded`` combines the start-up warm-up flags stored on
``app.state.readiness`` (see ``medbridge.main``) with the live in-process
state of the embedders, the reranker and the models resident in Ollama.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, Request, Response, status
from qdrant_client import AsyncQdrantClient
from sqlalchemy import text

from medbridge.ai.llm_wrapper import get_llm_wrapper
from medbridge.api.schemas.responses import HealthResponse
from medbridge.config import Settings, get_settings
from medbridge.db.connection import get_engine
from medbridge.retrieval.embedder import get_embedding_client
from medbridge.retrieval.hybrid_retriever import get_hybrid_retriever
from medbridge.retrieval.reranker import get_reranker

logger = structlog.get_logger(__name__)

APP_VERSION = "4.0.0"
_LLM_HEALTH_TIMEOUT_S = 5.0

health_router = APIRouter(tags=["health"])


async def check_postgres_health() -> bool:
    """Probe PostgreSQL with ``SELECT 1``."""
    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("health_check_postgres_failed", error=type(exc).__name__)
        return False


async def check_qdrant_health() -> bool:
    """Probe Qdrant with ``get_collections``."""
    client: AsyncQdrantClient | None = None
    try:
        client = AsyncQdrantClient(url=get_settings().QDRANT_URL, timeout=2)
        await client.get_collections()
        return True
    except Exception as exc:
        logger.warning("health_check_qdrant_failed", error=type(exc).__name__)
        return False
    finally:
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()


async def get_qdrant_points() -> int | None:
    """Number of indexed chunks in the guideline collection (``None`` if unknown)."""
    return await get_hybrid_retriever().collection_points()


async def check_llm_health() -> dict[str, Any]:
    """LLM provider readiness from ``ResilientLLMWrapper.health()`` (never raises)."""
    try:
        return await asyncio.wait_for(get_llm_wrapper().health(), timeout=_LLM_HEALTH_TIMEOUT_S)
    except Exception as exc:
        logger.warning("health_check_llm_failed", error=type(exc).__name__)
        return {"provider": get_settings().LLM_PROVIDER, "reachable": False, "models": {}}


def collect_models_loaded(
    readiness: dict[str, Any], llm: dict[str, Any], settings: Settings
) -> dict[str, bool]:
    """Merge warm-up flags with the live state of every model the pipeline uses."""
    embedder = get_embedding_client().loaded
    resident = {str(name) for name in llm.get("loaded", []) or []}
    configured = list(dict.fromkeys([settings.gates_model, settings.generator_model]))
    llm_resident = bool(resident) and all(
        any(r == m or r.startswith(f"{m}:") or m.startswith(f"{r}:") for r in resident)
        for m in configured
    )
    return {
        "embedder_dense": bool(readiness.get("embedder_dense")) or embedder["dense"],
        "embedder_sparse": bool(readiness.get("embedder_sparse")) or embedder["sparse"],
        "reranker": bool(readiness.get("reranker")) or get_reranker().loaded,
        "llm": bool(readiness.get("llm")) or llm_resident,
        "nli_verifier": bool(readiness.get("nli_verifier")) or _verifier_loaded(settings),
    }


def _verifier_loaded(settings: Settings) -> bool:
    """Live state of the Gate 3 NLI model (only consulted when the verifier is enabled)."""
    if not settings.GROUNDING_VERIFIER_ENABLED:
        return False
    try:
        from medbridge.ai.grounding_verifier import get_grounding_verifier

        return bool(getattr(get_grounding_verifier(), "loaded", False))
    except Exception as exc:  # health must never fail because of an optional model
        logger.warning("health_check_verifier_failed", error=type(exc).__name__)
        return False


def derive_status(
    postgres_ok: bool,
    qdrant_ok: bool,
    llm: dict[str, Any],
    models_loaded: dict[str, bool],
    settings: Settings,
) -> str:
    """``unhealthy`` / ``degraded`` / ``healthy`` (see module docstring)."""
    if not (postgres_ok and qdrant_ok):
        return "unhealthy"
    if not llm.get("reachable") or not all(bool(v) for v in (llm.get("models") or {}).values()):
        return "degraded"
    if settings.WARMUP_ON_STARTUP:
        required = ["embedder_dense", "embedder_sparse", "reranker", "llm"]
        if settings.GROUNDING_VERIFIER_ENABLED:
            required.append("nli_verifier")
        if not all(models_loaded.get(name) for name in required):
            return "degraded"
    return "healthy"


@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness/readiness: storage connectivity, LLM reachability and model warm-up",
)
async def health_check(request: Request, response: Response) -> HealthResponse:
    """Return 200 when storage is reachable (healthy or degraded), 503 otherwise."""
    settings = get_settings()
    postgres_ok, qdrant_ok, llm, points = await asyncio.gather(
        check_postgres_health(), check_qdrant_health(), check_llm_health(), get_qdrant_points()
    )
    readiness: dict[str, Any] = getattr(request.app.state, "readiness", None) or {}
    models_loaded = collect_models_loaded(readiness, llm, settings)
    status_str = derive_status(postgres_ok, qdrant_ok, llm, models_loaded, settings)
    response.status_code = (
        status.HTTP_503_SERVICE_UNAVAILABLE if status_str == "unhealthy" else status.HTTP_200_OK
    )
    reachable = llm.get("reachable")
    return HealthResponse(
        status=status_str,
        postgres_connected=postgres_ok,
        qdrant_connected=qdrant_ok,
        timestamp=datetime.now(UTC),
        ollama_reachable=bool(reachable) if llm.get("provider") == "ollama" else None,
        llm_provider=str(llm.get("provider") or settings.LLM_PROVIDER),
        models=dict(llm.get("models") or {}),
        models_loaded=models_loaded,
        qdrant_points=points,
        local_only_mode=settings.LOCAL_ONLY_MODE,
        version=APP_VERSION,
    )
