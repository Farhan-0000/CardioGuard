"""
Integration tests for TASK-13: State Projector DB persistence.

Tests transactional atomicity of persist_and_project() against the isolated
PostgreSQL test database; payloads and snapshots are stored encrypted.
"""
import os
import uuid

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

os.environ.setdefault("GROQ_API_KEY", "dummy")
os.environ.setdefault("POSTGRES_PASSWORD", "password")

from medbridge.ai.schemas.extractor import DeltaEvent
from medbridge.api.schemas.enums import EventTypeEnum
from medbridge.db.connection import get_engine, get_sessionmaker
from medbridge.db.models import ClinicalEvent, ContextSnapshot, Session as SessionModel
from medbridge.privacy.crypto import get_session_cipher
from medbridge.state.projector import StateProjector

pytestmark = pytest.mark.postgres


async def _plain(db: AsyncSession, session_id: uuid.UUID, value):
    cipher = await get_session_cipher(db, session_id)
    assert cipher is not None
    return cipher.decrypt_json(value)


@pytest.fixture
def projector() -> StateProjector:
    return StateProjector()


@pytest.fixture
async def db_session():
    """Provide an async session and clean up the test data afterwards."""
    maker = get_sessionmaker()
    async with maker() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def test_session_id(db_session: AsyncSession) -> uuid.UUID:
    """Create a test session row and return its UUID."""
    sid = uuid.uuid4()
    db_session.add(SessionModel(session_id=sid))
    await db_session.flush()
    return sid


@pytest.mark.asyncio
async def test_persist_creates_events_and_snapshot(
    projector: StateProjector,
    db_session: AsyncSession,
    test_session_id: uuid.UUID,
):
    """persist_and_project() should insert events and create a snapshot."""
    events = [
        DeltaEvent(
            event_type=EventTypeEnum.DEMOGRAPHIC,
            payload={"age": 55, "sex": "male"},
        ),
        DeltaEvent(
            event_type=EventTypeEnum.BP_READING,
            payload={"systolic": 140, "diastolic": 90},
        ),
    ]

    result = await projector.persist_and_project(
        session=db_session,
        session_id=test_session_id,
        events=events,
        current_snapshot={},
    )
    await db_session.flush()

    # Verify returned snapshot
    assert result["demographics"]["age"] == 55
    assert len(result["recent_bp_readings"]) == 1

    # Verify events persisted
    rows = (await db_session.execute(
        select(ClinicalEvent).where(ClinicalEvent.session_id == test_session_id)
    )).scalars().all()
    assert len(rows) == 2
    assert all(row.payload.get("enc") == "v1" for row in rows)  # encrypted at rest
    assert sorted([(await _plain(db_session, test_session_id, r.payload)).get("age", 0)
                   for r in rows]) == [0, 55]

    # Verify snapshot persisted (encrypted)
    snap = await db_session.get(ContextSnapshot, test_session_id)
    assert snap is not None
    assert snap.snapshot.get("enc") == "v1"
    stored = await _plain(db_session, test_session_id, snap.snapshot)
    assert stored["demographics"]["age"] == 55


@pytest.mark.asyncio
async def test_persist_upserts_existing_snapshot(
    projector: StateProjector,
    db_session: AsyncSession,
    test_session_id: uuid.UUID,
):
    """Calling persist_and_project() twice should update (not duplicate) the snapshot."""
    # First batch
    events1 = [
        DeltaEvent(
            event_type=EventTypeEnum.DEMOGRAPHIC,
            payload={"age": 55},
        ),
    ]
    snap1 = await projector.persist_and_project(
        session=db_session,
        session_id=test_session_id,
        events=events1,
        current_snapshot={},
    )
    await db_session.flush()

    # Second batch
    events2 = [
        DeltaEvent(
            event_type=EventTypeEnum.BP_READING,
            payload={"systolic": 130, "diastolic": 85},
        ),
    ]
    snap2 = await projector.persist_and_project(
        session=db_session,
        session_id=test_session_id,
        events=events2,
        current_snapshot=snap1,
    )
    await db_session.flush()

    # Snapshot should have both demographic and BP data
    stored_row = await db_session.get(ContextSnapshot, test_session_id)
    assert stored_row is not None
    stored = await _plain(db_session, test_session_id, stored_row.snapshot)
    assert stored["demographics"]["age"] == 55
    assert len(stored["recent_bp_readings"]) == 1

    # Total events should be 2 (1 + 1)
    rows = (await db_session.execute(
        select(ClinicalEvent).where(ClinicalEvent.session_id == test_session_id)
    )).scalars().all()
    assert len(rows) == 2
