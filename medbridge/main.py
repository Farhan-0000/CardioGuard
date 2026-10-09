"""
MedBridge application entry point and lifespan (Module M-19).

``create_app`` assembles:

1. structured logging;
2. security middleware (CORS, rate limiting, request-size limit, security
   headers/CSP) from ``medbridge.privacy.middleware`` — falling back to plain
   CORS with a warning if that module is unavailable;
3. global error handlers (Non-Negotiable #7);
4. the REST/SSE API under ``/api`` and ``/health`` at the root;
5. the single-page web UI (``medbridge.web.mount_web``) when installed.

The lifespan initialises the database pool, starts a *background* model
warm-up (embedders, reranker, Qdrant connection, Ollama models and — when
enabled — the NLI grounding verifier) whose progress is exposed through
``app.state.readiness`` and ``/health``, and starts the data-retention job.
On shutdown every background task is cancelled and the LLM client, Qdrant
client and database pool are closed.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from medbridge.ai.llm_wrapper import get_llm_wrapper
from medbridge.api.middleware.error_handler import register_error_handlers
from medbridge.api.routes import api_router, health_router
from medbridge.api.routes.health import APP_VERSION
from medbridge.config import Settings, get_settings
from medbridge.core.orchestrator import get_orchestrator
from medbridge.db.connection import close_db, init_db
from medbridge.logging_config import configure_logging
from medbridge.retrieval.hybrid_retriever import get_hybrid_retriever

logger = structlog.get_logger(__name__)


async def run_warmup(app: FastAPI) -> None:
    """Warm every pipeline model and record readiness flags on ``app.state.readiness``."""
    readiness: dict[str, Any] = app.state.readiness
    readiness["warmup"] = "running"
    try:
        readiness.update(await get_orchestrator().warmup())
        readiness["warmup"] = "complete"
    except Exception as exc:  # warm-up is an optimisation; the app still serves lazily
        logger.error("warmup_failed", error=type(exc).__name__)
        readiness["warmup"] = "failed"


def _retention_task(stop_event: asyncio.Event) -> asyncio.Task[None] | None:
    """Start the data-retention purge loop (privacy work-stream) if available."""
    try:
        from medbridge.privacy.retention import retention_loop
    except ImportError as exc:
        logger.warning("retention_loop_unavailable", error=str(exc))
        return None
    return asyncio.create_task(retention_loop(stop_event), name="medbridge-retention")


async def _close(name: str, closer: Callable[[], Awaitable[None]]) -> None:
    try:
        await closer()
        logger.info("shutdown_closed", component=name)
    except Exception as exc:
        logger.warning("shutdown_close_failed", component=name, error=type(exc).__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: DB pool, background warm-up, retention job. Shutdown: cancel and close."""
    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    logger.info("app_startup_begin", host=settings.HOST, port=settings.PORT)
    app.state.readiness = {"warmup": "disabled"}

    try:
        await init_db()
        logger.info("db_initialized_successfully")
    except Exception as exc:
        logger.error("db_init_failed", error=type(exc).__name__)

    tasks: list[asyncio.Task[Any]] = []
    stop_event = asyncio.Event()
    if settings.WARMUP_ON_STARTUP:
        app.state.readiness["warmup"] = "pending"
        tasks.append(asyncio.create_task(run_warmup(app), name="medbridge-warmup"))
    retention = _retention_task(stop_event)
    if retention is not None:
        tasks.append(retention)

    logger.info("app_startup_complete", warmup=settings.WARMUP_ON_STARTUP)
    yield

    logger.info("app_shutdown_begin")
    stop_event.set()
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    await _close("llm_client", get_llm_wrapper().close)
    await _close("qdrant_client", get_hybrid_retriever().close)
    await _close("database", close_db)
    logger.info("app_shutdown_complete")


def _register_middleware(app: FastAPI, settings: Settings) -> None:
    try:
        from medbridge.privacy.middleware import register_security_middleware
    except ImportError as exc:
        logger.warning("security_middleware_unavailable_fallback_cors", error=str(exc))
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        return
    register_security_middleware(app, settings)


def _mount_web(app: FastAPI) -> None:
    try:
        from medbridge.web import mount_web
    except ImportError as exc:
        logger.warning("web_ui_unavailable", error=str(exc))
        return
    mount_web(app)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app_settings = settings or get_settings()
    configure_logging()

    app = FastAPI(
        title="MedBridge API",
        description=(
            "Evidence-grounded heart and blood-pressure guidance with a Dual "
            "Answerability Gate (Context Gate + Evidence Gate)."
        ),
        version=APP_VERSION,
        lifespan=lifespan,
    )
    app.state.settings = app_settings

    _register_middleware(app, app_settings)
    register_error_handlers(app)
    app.include_router(api_router, prefix="/api")
    app.include_router(health_router)
    _mount_web(app)
    return app


app = create_app()
