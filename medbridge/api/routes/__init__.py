"""
API route assembly (Module M-04).

``api_router`` is mounted under ``/api`` by :func:`medbridge.main.create_app`;
``health_router`` is additionally mounted at the root so ``/health`` and
``/api/health`` both work. The feature and admin routers (privacy/persistence
work-stream) are optional at import time: if one cannot be imported the error
is logged and the rest of the API still starts.
"""

from __future__ import annotations

import importlib

import structlog
from fastapi import APIRouter

from medbridge.api.routes.config import config_router
from medbridge.api.routes.health import (
    check_postgres_health,
    check_qdrant_health,
    health_router,
)
from medbridge.api.routes.messages import messages_router
from medbridge.api.routes.metrics import metrics_router
from medbridge.api.routes.sessions import sessions_router

logger = structlog.get_logger(__name__)


def _optional_router(module_name: str, attribute: str) -> APIRouter | None:
    """Import ``module_name.attribute``; log and return ``None`` if unavailable."""
    try:
        router = getattr(importlib.import_module(module_name), attribute)
    except Exception as exc:  # a broken optional module must not take the API down
        logger.warning("optional_router_unavailable", module=module_name, error=repr(exc))
        return None
    return router if isinstance(router, APIRouter) else None


features_router = _optional_router("medbridge.api.routes.features", "features_router")
admin_router = _optional_router("medbridge.api.routes.admin", "admin_router")

# Combined API router mounted under /api
api_router = APIRouter()
api_router.include_router(sessions_router)
api_router.include_router(messages_router)
if features_router is not None:
    api_router.include_router(features_router)
if admin_router is not None:
    api_router.include_router(admin_router)
api_router.include_router(metrics_router)
api_router.include_router(config_router)
api_router.include_router(health_router)

__all__ = [
    "admin_router",
    "api_router",
    "check_postgres_health",
    "check_qdrant_health",
    "config_router",
    "features_router",
    "health_router",
    "messages_router",
    "metrics_router",
    "sessions_router",
]
