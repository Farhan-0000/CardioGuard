"""
Integration tests for TASK-14: Database Session Manager.

Tests session creation (incl. the per-session data key), state loading,
counter increment/reset, encrypted message history and ordering against the
isolated PostgreSQL test database.
"""
import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

os.environ.setdefault("GROQ_API_KEY", "dummy")
os.environ.setdefault("POSTGRES_PASSWORD", "password")

from medbridge.api.schemas.enums import ActionEnum
from medbridge.db.connection import get_sessionmaker
from medbridge.db.models import SessionKey
from medbridge.state.session_manager import (
    SessionErasedError,
    SessionNotFoundError,
    create_session,
    get_message_history,
    increment_soft_ask_count,
    load_session_state,
    reset_soft_ask_count,
    save_message,
)
from medbridge.state.turn_store import erase_session

pytestmark = pytest.mark.postgres


@pytest.fixture
async def db_session():
    """Provide an async session and roll back after the test."""
    maker = get_sessionmaker()
    async with maker() as session:
        yield session
        await session.rollback()


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_session_returns_uuid(db_session: AsyncSession):
    session_id = await create_session(db_session)
    assert isinstance(session_id, uuid.UUID)


@pytest.mark.asyncio
async def test_create_session_initialises_snapshot(db_session: AsyncSession):
    """create_session should also create a ContextSnapshot row."""
    session_id = await create_session(db_session)
    state = await load_session_state(db_session, session_id)
    assert state["snapshot"] == {}
    assert state["soft_ask_count"] == 0


# ---------------------------------------------------------------------------
# load_session_state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_load_session_state_returns_snapshot_and_count(db_session: AsyncSession):
    session_id = await create_session(db_session)
    state = await load_session_state(db_session, session_id)
    assert "snapshot" in state
    assert "soft_ask_count" in state
    assert isinstance(state["snapshot"], dict)
    assert isinstance(state["soft_ask_count"], int)


@pytest.mark.asyncio
async def test_load_session_state_raises_for_missing_session(db_session: AsyncSession):
    fake_id = uuid.uuid4()
    with pytest.raises(SessionNotFoundError) as exc_info:
        await load_session_state(db_session, fake_id)
    assert str(fake_id) in str(exc_info.value)


# ---------------------------------------------------------------------------
# soft_ask_count increment / reset
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_increment_soft_ask_count(db_session: AsyncSession):
    session_id = await create_session(db_session)

    count = await increment_soft_ask_count(db_session, session_id)
    assert count == 1

    count = await increment_soft_ask_count(db_session, session_id)
    assert count == 2

    count = await increment_soft_ask_count(db_session, session_id)
    assert count == 3


@pytest.mark.asyncio
async def test_reset_soft_ask_count(db_session: AsyncSession):
    session_id = await create_session(db_session)

    # Increment twice, then reset
    await increment_soft_ask_count(db_session, session_id)
    await increment_soft_ask_count(db_session, session_id)
    await reset_soft_ask_count(db_session, session_id)

    state = await load_session_state(db_session, session_id)
    assert state["soft_ask_count"] == 0


@pytest.mark.asyncio
async def test_increment_raises_for_missing_session(db_session: AsyncSession):
    with pytest.raises(SessionNotFoundError):
        await increment_soft_ask_count(db_session, uuid.uuid4())


@pytest.mark.asyncio
async def test_reset_raises_for_missing_session(db_session: AsyncSession):
    with pytest.raises(SessionNotFoundError):
        await reset_soft_ask_count(db_session, uuid.uuid4())


# ---------------------------------------------------------------------------
# save_message / get_message_history
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_and_retrieve_messages(db_session: AsyncSession):
    session_id = await create_session(db_session)

    # Save user message
    msg1_id = await save_message(
        db_session,
        session_id=session_id,
        role="user",
        content="What is my BP target?",
    )
    assert isinstance(msg1_id, uuid.UUID)

    # Save assistant response
    msg2_id = await save_message(
        db_session,
        session_id=session_id,
        role="assistant",
        content="Based on guidelines, your target is...",
        action=ActionEnum.ANSWER,
        citations=[{"marker": "[1]", "source": "AHA/ACC 2025"}],
    )
    assert isinstance(msg2_id, uuid.UUID)

    # Retrieve history
    history = await get_message_history(db_session, session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "What is my BP target?"
    assert history[0]["action"] is None
    assert history[1]["role"] == "assistant"
    assert history[1]["action"] == "ANSWER"
    assert len(history[1]["citations"]) == 1


@pytest.mark.asyncio
async def test_message_history_chronological_ordering(db_session: AsyncSession):
    session_id = await create_session(db_session)

    # Save multiple messages
    for i in range(5):
        role = "user" if i % 2 == 0 else "assistant"
        action = ActionEnum.ANSWER if role == "assistant" else None
        await save_message(
            db_session,
            session_id=session_id,
            role=role,
            content=f"Message {i}",
            action=action,
        )

    history = await get_message_history(db_session, session_id)
    assert len(history) == 5

    # Verify chronological order
    for i in range(len(history) - 1):
        assert history[i]["created_at"] <= history[i + 1]["created_at"]


@pytest.mark.asyncio
async def test_empty_message_history(db_session: AsyncSession):
    session_id = await create_session(db_session)
    history = await get_message_history(db_session, session_id)
    assert history == []


# ---------------------------------------------------------------------------
# Encryption (ADL-031) and erasure (ADL-032)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_session_creates_wrapped_data_key(db_session: AsyncSession):
    session_id = await create_session(db_session)
    key = await db_session.get(SessionKey, session_id)
    assert key is not None
    assert key.wrapped_key.startswith("gAAAA")  # Fernet token of the wrapped key


@pytest.mark.asyncio
async def test_saved_messages_are_encrypted_at_rest(db_session: AsyncSession):
    session_id = await create_session(db_session)
    await save_message(db_session, session_id, role="user", content="My BP is 160/100")
    raw = (await db_session.execute(
        text("SELECT content FROM message_history WHERE session_id = :sid"),
        {"sid": session_id},
    )).scalar_one()
    assert raw.startswith("enc:v1:") and "160/100" not in raw
    history = await get_message_history(db_session, session_id)
    assert history[0]["content"] == "My BP is 160/100"


@pytest.mark.asyncio
async def test_erased_session_cannot_be_loaded_or_written(db_session: AsyncSession):
    session_id = await create_session(db_session)
    await save_message(db_session, session_id, role="user", content="hello")
    await erase_session(db_session, session_id)
    with pytest.raises(SessionErasedError):
        await load_session_state(db_session, session_id)
    with pytest.raises(SessionErasedError):
        await save_message(db_session, session_id, role="user", content="again")
    assert await get_message_history(db_session, session_id) == []
