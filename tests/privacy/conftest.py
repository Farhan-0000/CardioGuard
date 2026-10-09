"""Shared fixtures for the privacy test-suite."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from medbridge.config import Settings, get_settings
from medbridge.privacy import crypto


@pytest.fixture
def override_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Settings]]:
    """Set environment overrides and rebuild the cached Settings."""

    def apply(**values: Any) -> Settings:
        for key, value in values.items():
            raw = json.dumps(value) if isinstance(value, list | dict) else str(value)
            monkeypatch.setenv(key, raw)
        get_settings.cache_clear()
        return get_settings()

    yield apply
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_key_cache() -> Iterator[None]:
    """Isolate the in-process data-key cache between tests."""
    crypto.clear_key_cache()
    yield
    crypto.clear_key_cache()


@pytest.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """A test-database session rolled back after the test (needs the postgres marker)."""
    from medbridge.db.connection import get_sessionmaker

    async with get_sessionmaker()() as session:
        yield session
        await session.rollback()
