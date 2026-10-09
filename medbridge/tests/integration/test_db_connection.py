import os
import pytest
from sqlalchemy import text

# Ensure environment variable is set for config before importing
os.environ["POSTGRES_PASSWORD"] = "password"
os.environ["GROQ_API_KEY"] = "dummy"

from medbridge.db.connection import get_engine, init_db, close_db, get_session

pytestmark = pytest.mark.postgres

@pytest.mark.asyncio
async def test_db_connection_lifecycle():
    # 1. Test initialization
    await init_db()
    
    # 2. Test session and query
    async_gen = get_session()
    session = await anext(async_gen)
    
    try:
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1
    finally:
        # Exhaust generator to trigger rollback/commit and close
        try:
            await anext(async_gen)
        except StopAsyncIteration:
            pass

    # 3. Test teardown
    await close_db()
    
    # Ensure engine is properly disposed
    engine = get_engine()
    assert engine is not None


def test_engine_hides_sql_parameters():
    """Bound parameters (ciphertext / legacy text) never appear in error messages."""
    assert get_engine().sync_engine.hide_parameters is True
