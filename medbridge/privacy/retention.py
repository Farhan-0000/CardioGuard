"""
Data-retention TTL job (ADL-035).

Sessions inactive for longer than ``DATA_RETENTION_DAYS`` are crypto-shredded
exactly like a user-initiated ``DELETE /api/sessions/{id}``: the data key and
the mutable rows are destroyed, append-only rows stay but become unreadable.

``retention_loop`` is started from the FastAPI lifespan with an
``asyncio.Event`` used to stop it on shutdown. It runs shortly after start and
then hourly, and it never raises (a database outage is logged and retried at
the next tick).
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.config import get_settings
from medbridge.db.connection import get_sessionmaker
from medbridge.db.models import Session
from medbridge.state.turn_store import erase_session

logger = structlog.get_logger(__name__)

RETENTION_INTERVAL_SECONDS = 3600.0
RETENTION_INITIAL_DELAY_SECONDS = 30.0
RETENTION_BATCH_SIZE = 500


async def purge_expired_sessions(db: AsyncSession, older_than_days: int) -> int:
    """Crypto-shred sessions idle for more than ``older_than_days``. Returns the count.

    Does not commit. ``older_than_days <= 0`` disables the purge.
    """
    if older_than_days <= 0:
        return 0
    cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
    session_ids = (await db.execute(
        select(Session.session_id)
        .where(Session.last_active_at < cutoff, Session.erased_at.is_(None))
        .order_by(Session.last_active_at.asc())
        .limit(RETENTION_BATCH_SIZE)
    )).scalars().all()
    for session_id in session_ids:
        await erase_session(db, session_id)
    if session_ids:
        logger.info("retention_purge", sessions=len(session_ids), older_than_days=older_than_days)
    return len(session_ids)


async def _wait(stop_event: asyncio.Event, seconds: float) -> bool:
    """Sleep up to ``seconds``; return True if ``stop_event`` was set meanwhile."""
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except TimeoutError:
        return False
    return True


async def retention_loop(
    stop_event: asyncio.Event,
    *,
    interval_seconds: float = RETENTION_INTERVAL_SECONDS,
    initial_delay_seconds: float = RETENTION_INITIAL_DELAY_SECONDS,
) -> None:
    """Run :func:`purge_expired_sessions` periodically until ``stop_event`` is set."""
    if await _wait(stop_event, initial_delay_seconds):
        return
    while True:
        try:
            async with get_sessionmaker()() as db:
                await purge_expired_sessions(db, get_settings().DATA_RETENTION_DAYS)
                await db.commit()
        except Exception as exc:  # a background job must survive DB outages
            logger.warning("retention_purge_failed", error_type=type(exc).__name__)
        if await _wait(stop_event, interval_seconds):
            return
