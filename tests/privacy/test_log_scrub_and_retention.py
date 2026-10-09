"""Log scrubbing processor and the data-retention job."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog
from sqlalchemy.exc import OperationalError

from medbridge.logging_config import configure_logging
from medbridge.privacy import retention
from medbridge.privacy.log_scrub import MAX_SCRUB_CHARS, scrub_pii_processor, scrub_value

# ---------------------------------------------------------------------------
# Log scrubbing
# ---------------------------------------------------------------------------


def test_processor_redacts_strings_recursively() -> None:
    event = {
        "event": "My name is Rahul Sharma, call 9876543210",
        "detail": {"email": "rahul@example.com", "items": ["Aadhaar 2345 6789 0123", 5]},
        "bp": "150/95",
        "session_id": str(uuid.uuid4()),
        "count": 3,
        "timestamp": "2026-10-09T10:00:00Z",
    }
    out = scrub_pii_processor(None, "info", dict(event))
    assert out["event"] == "My name is [NAME], call [PHONE]"
    assert out["detail"] == {"email": "[EMAIL]", "items": ["Aadhaar [AADHAAR]", 5]}
    assert out["bp"] == "150/95"
    assert out["session_id"] == event["session_id"]
    assert out["count"] == 3
    assert out["timestamp"] == event["timestamp"]


def test_scrub_is_bounded() -> None:
    long = "x" * (MAX_SCRUB_CHARS + 500)
    assert scrub_value(long).endswith("…[truncated]")
    deep: dict[str, Any] = {"a": {"b": {"c": {"d": {"e": "my name is Rahul"}}}}}
    assert scrub_value(deep)["a"]["b"]["c"]["d"] == "[nested]"
    assert len(scrub_value(list(range(1000)))) == 200


@pytest.fixture
def json_log_output() -> Any:
    stream = io.StringIO()
    configure_logging(json_format=True, log_level=logging.INFO)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.getLogger().handlers[0].formatter)
    logging.getLogger().handlers = [handler]
    yield stream
    logging.getLogger().handlers.clear()
    structlog.reset_defaults()


def test_configured_logging_scrubs_pii_and_masks_secrets(json_log_output: io.StringIO) -> None:
    log = structlog.get_logger("privacy-test")
    log.info("message_received", text="I'm Priya Nair, email priya@example.com, BP 150/95",
             session_token="v1.abc.def", prompt_tokens=42)
    try:
        raise ValueError("lookup failed for patient Ramesh Kumar phone +91 98765 43210")
    except ValueError:
        log.exception("failure")
    lines = [json.loads(x) for x in json_log_output.getvalue().splitlines()]
    first, second = lines
    assert first["text"] == "I'm [NAME], email [EMAIL], BP 150/95"
    assert first["session_token"] == "***MASKED***"
    assert first["prompt_tokens"] == 42
    raw = json_log_output.getvalue()
    for secret in ("Priya", "priya@example.com", "Ramesh Kumar", "98765 43210", "v1.abc.def"):
        assert secret not in raw
    assert "[PHONE]" in second["exception"]


# ---------------------------------------------------------------------------
# Retention (hermetic)
# ---------------------------------------------------------------------------


async def test_purge_disabled_for_non_positive_days() -> None:
    db = MagicMock()
    db.execute = AsyncMock()
    assert await retention.purge_expired_sessions(db, 0) == 0
    db.execute.assert_not_called()


async def test_purge_erases_each_expired_session(monkeypatch: pytest.MonkeyPatch) -> None:
    ids = [uuid.uuid4(), uuid.uuid4()]
    result = MagicMock()
    result.scalars.return_value.all.return_value = ids
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    erase = AsyncMock()
    monkeypatch.setattr(retention, "erase_session", erase)
    assert await retention.purge_expired_sessions(db, 30) == 2
    assert [c.args[1] for c in erase.await_args_list] == ids
    sql = str(db.execute.await_args.args[0])
    assert "last_active_at <" in sql and "erased_at IS NULL" in sql


async def test_retention_loop_survives_db_outage_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def broken_maker() -> Any:
        calls.append(1)
        raise OperationalError("SELECT 1", {}, ConnectionRefusedError("db down"))

    monkeypatch.setattr(retention, "get_sessionmaker", broken_maker)
    stop = asyncio.Event()
    task = asyncio.create_task(
        retention.retention_loop(stop, interval_seconds=0.01, initial_delay_seconds=0)
    )
    await asyncio.sleep(0.08)
    stop.set()
    await asyncio.wait_for(task, timeout=1)
    assert len(calls) >= 2  # kept retrying after failures


async def test_retention_loop_stops_during_initial_delay() -> None:
    stop = asyncio.Event()
    stop.set()
    await asyncio.wait_for(retention.retention_loop(stop, initial_delay_seconds=60), timeout=1)


# ---------------------------------------------------------------------------
# Retention (PostgreSQL)
# ---------------------------------------------------------------------------


@pytest.mark.postgres
async def test_purge_shreds_only_inactive_sessions(db_session: Any) -> None:
    from medbridge.db.models import Session
    from medbridge.state import turn_store
    from medbridge.state.session_manager import create_session

    old_id = await create_session(db_session)
    fresh_id = await create_session(db_session)
    old = await db_session.get(Session, old_id)
    old.last_active_at = datetime.now(UTC) - timedelta(days=45)
    await db_session.flush()

    purged = await retention.purge_expired_sessions(db_session, 30)
    assert purged >= 1
    assert (await turn_store.load_session_state(db_session, old_id)).erased
    assert not (await turn_store.load_session_state(db_session, fresh_id)).erased
