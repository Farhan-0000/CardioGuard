import pytest

from medbridge.db.connection import close_db


@pytest.fixture(autouse=True)
async def cleanup_db():
    yield
    await close_db()
