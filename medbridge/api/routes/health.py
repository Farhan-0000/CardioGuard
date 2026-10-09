"""
Health Check REST Routes (TASK-29, Module M-04).

Endpoints:
- GET /health: Verifies PostgreSQL and Qdrant connectivity (HTTP 200/503).

Technical Specification Part II §4.4.
"""
from datetime import datetime, timezone
import structlog
from fastapi import APIRouter, Response, status
from qdrant_client import AsyncQdrantClient
from sqlalchemy import text

from medbridge.api.schemas.responses import HealthResponse
from medbridge.config import get_settings
from medbridge.db.connection import get_engine

logger = structlog.get_logger(__name__)

health_router = APIRouter(tags=["health"])


async def check_postgres_health() -> bool:
    """Probe PostgreSQL database with a simple SELECT 1 query."""
    try:
        engine = get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("health_check_postgres_failed", error=str(exc))
        return False


async def check_qdrant_health() -> bool:
    """Probe Qdrant vector store with a get_collections query."""
    client = None
    try:
        settings = get_settings()
        client = AsyncQdrantClient(url=settings.QDRANT_URL, timeout=2.0)
        await client.get_collections()
        return True
    except Exception as exc:
        logger.warning("health_check_qdrant_failed", error=str(exc))
        return False
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass



@health_router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check verifying database and vector store connectivity",
)
async def health_check(response: Response) -> HealthResponse:
    """
    Verify PostgreSQL and Qdrant connectivity.
    Returns HTTP 200 if both services are reachable; returns HTTP 503 otherwise.
    """
    postgres_ok = await check_postgres_health()
    qdrant_ok = await check_qdrant_health()

    is_healthy = postgres_ok and qdrant_ok
    status_str = "healthy" if is_healthy else "unhealthy"

    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    else:
        response.status_code = status.HTTP_200_OK

    return HealthResponse(
        status=status_str,
        postgres_connected=postgres_ok,
        qdrant_connected=qdrant_ok,
        timestamp=datetime.now(timezone.utc),
    )
