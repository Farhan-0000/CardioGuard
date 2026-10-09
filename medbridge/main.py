"""
MedBridge Application Entry Point and Lifespan Assembly (TASK-30, Module M-19).

Assembles:
1. Lifespan startup/shutdown: DB pool initialization, embedding model preloading, DB pool disposal.
2. CORS middleware using Settings.CORS_ORIGINS.
3. Global error handling middleware (Non-Negotiable #7).
4. REST API routers mounted under /api and root /health.

Technical Specification Part II §3.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from medbridge.api.middleware.error_handler import register_error_handlers
from medbridge.api.routes import api_router, health_router
from medbridge.config import Settings, get_settings
from medbridge.db.connection import close_db, init_db
from medbridge.logging_config import configure_logging
from medbridge.ai.llm_wrapper import get_llm_wrapper
from medbridge.retrieval.embedder import EmbeddingClient

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle."""
    settings = get_settings()
    logger.info("app_startup_begin", host=settings.HOST, port=settings.PORT)

    # 1. Initialize DB connection pool
    try:
        await init_db()
        logger.info("db_initialized_successfully")
    except Exception as exc:
        logger.error("db_init_failed", error=str(exc))

    # 2. Preload embedding models (dense BGE + sparse BM25)
    try:
        embedder = EmbeddingClient()
        _ = embedder._get_dense_model()
        _ = embedder._get_sparse_model()
        logger.info("embedding_models_preloaded")
    except Exception as exc:
        logger.warning("embedding_preload_skipped", error=str(exc))

    logger.info("app_startup_complete")
    yield

    # Shutdown
    logger.info("app_shutdown_begin")

    # Close persistent LLM HTTP client
    try:
        llm_wrapper = get_llm_wrapper()
        await llm_wrapper.close()
        logger.info("llm_client_closed_successfully")
    except Exception as exc:
        logger.warning("llm_client_close_failed", error=str(exc))

    try:
        await close_db()
        logger.info("db_closed_successfully")
    except Exception as exc:
        logger.error("db_close_failed", error=str(exc))

    logger.info("app_shutdown_complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app_settings = settings or get_settings()

    # Configure structured logging
    configure_logging()

    app = FastAPI(
        title="MedBridge API",
        description="Clinical Guidance Assistant for Hypertension Management",
        version="3.0.0",
        lifespan=lifespan,
    )

    # Attach CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register global error handlers (Non-Negotiable #7)
    register_error_handlers(app)

    # Mount API routes under /api
    app.include_router(api_router, prefix="/api")

    # Mount health check at root /health for health probes
    app.include_router(health_router)

    return app


app = create_app()
