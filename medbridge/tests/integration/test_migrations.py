"""
Alembic migrations against the isolated ``medbridge_test`` database.

Runs ``downgrade base`` / ``upgrade head`` (destructive, hence the test DB),
verifies the immutability rules on the append-only tables and the privacy
migration (``g1a7c3e9d2b4``: trace, erased_at, session_keys, feedback) and the
hardening migration (``h2b8d4f6a1c3``: one feedback row per message with
de-duplication, ``sessions.legacy_plaintext_rows``) in both directions.
"""
import os
from collections.abc import Iterator

import pytest
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from alembic import command
from medbridge.config import get_settings
from medbridge.db.connection import get_engine

# Ensure required environment variables for Alembic/Settings
os.environ.setdefault("POSTGRES_PASSWORD", "password")
os.environ.setdefault("GROQ_API_KEY", "dummy")

pytestmark = pytest.mark.postgres

PRIVACY_REVISION = "g1a7c3e9d2b4"
PREVIOUS_REVISION = "f7737fab1c53"
HEAD_REVISION = "h2b8d4f6a1c3"


def run_alembic(cmd, *args, **kwargs):
    alembic_cfg = Config("alembic.ini")
    getattr(command, cmd)(alembic_cfg, *args, **kwargs)


@pytest.fixture(scope="module")
def apply_migrations():
    # Attempt to downgrade just in case DB is already populated
    try:
        run_alembic("downgrade", "base")
    except Exception:
        pass

    # Run migrations up
    run_alembic("upgrade", "head")

    yield

    # Run migrations down to test rollback, then re-apply to head for subsequent tests
    try:
        run_alembic("downgrade", "base")
    finally:
        run_alembic("upgrade", "head")


@pytest.fixture
def sync_engine(apply_migrations) -> Iterator[Engine]:
    url = get_settings().DATABASE_URL.replace("+asyncpg", "+psycopg2")
    engine = create_engine(url)
    yield engine
    engine.dispose()


@pytest.mark.asyncio
async def test_migrations_and_immutability(apply_migrations):
    engine = get_engine()

    async with engine.begin() as conn:
        # 1. Insert dummy session
        res = await conn.execute(
            text("INSERT INTO sessions (session_id) VALUES (gen_random_uuid()) RETURNING session_id")
        )
        session_id = res.scalar()

        # 2. Insert clinical event
        res = await conn.execute(
            text("INSERT INTO clinical_events (event_id, session_id, event_type, payload) VALUES (gen_random_uuid(), :sid, 'BP_READING', '{}') RETURNING event_id"),
            {"sid": session_id}
        )
        event_id = res.scalar()

        # 3. Test UPDATE on clinical_events is a no-op
        await conn.execute(
            text("UPDATE clinical_events SET event_type = 'LAB_RESULT' WHERE event_id = :eid"),
            {"eid": event_id}
        )

        # Verify it didn't change
        res = await conn.execute(
            text("SELECT event_type FROM clinical_events WHERE event_id = :eid"),
            {"eid": event_id}
        )
        assert res.scalar() == "BP_READING"

        # 4. Test DELETE on clinical_events is a no-op
        await conn.execute(
            text("DELETE FROM clinical_events WHERE event_id = :eid"),
            {"eid": event_id}
        )

        # Verify it still exists
        res = await conn.execute(
            text("SELECT COUNT(*) FROM clinical_events WHERE event_id = :eid"),
            {"eid": event_id}
        )
        assert res.scalar() == 1

        # 5. audit_logs (incl. the new trace column) is append-only as well
        res = await conn.execute(
            text("INSERT INTO audit_logs (audit_id, session_id, request_message, final_action, "
                 "response_text, trace) VALUES (gen_random_uuid(), :sid, 'enc:v1:x', 'ANSWER', "
                 "'enc:v1:y', '{\"total_ms\": 1}') RETURNING audit_id"),
            {"sid": session_id},
        )
        audit_id = res.scalar()
        await conn.execute(text("UPDATE audit_logs SET trace = NULL WHERE audit_id = :a"),
                           {"a": audit_id})
        await conn.execute(text("DELETE FROM audit_logs WHERE audit_id = :a"), {"a": audit_id})
        res = await conn.execute(text("SELECT trace->>'total_ms' FROM audit_logs "
                                      "WHERE audit_id = :a"), {"a": audit_id})
        assert res.scalar() == "1"

        # 6. The new tables are ordinary (deletable) tables
        await conn.execute(text("INSERT INTO session_keys (session_id, wrapped_key) "
                                "VALUES (:sid, 'k')"), {"sid": session_id})
        await conn.execute(text("DELETE FROM session_keys WHERE session_id = :sid"),
                           {"sid": session_id})
        res = await conn.execute(text("SELECT COUNT(*) FROM session_keys WHERE session_id = :sid"),
                                 {"sid": session_id})
        assert res.scalar() == 0


def test_privacy_schema_present_at_head(sync_engine: Engine):
    insp = inspect(sync_engine)
    assert "trace" in {c["name"] for c in insp.get_columns("audit_logs")}
    assert "erased_at" in {c["name"] for c in insp.get_columns("sessions")}
    assert {"session_keys", "feedback"} <= set(insp.get_table_names())
    assert "ix_audit_logs_created_at" in {i["name"] for i in insp.get_indexes("audit_logs")}
    fk = insp.get_foreign_keys("session_keys")[0]
    assert fk["referred_table"] == "sessions" and fk["options"].get("ondelete") == "CASCADE"
    checks = {c["name"] for c in insp.get_check_constraints("feedback")}
    assert "ck_feedback_rating" in checks
    with sync_engine.connect() as conn:
        rules = {r[0] for r in conn.execute(text(
            "SELECT rulename FROM pg_rules WHERE tablename IN ('session_keys', 'feedback')"
        ))}
    assert rules == set()  # no immutability rules on the new tables


def test_feedback_rating_check_constraint(sync_engine: Engine):
    with sync_engine.begin() as conn:
        sid = conn.execute(text("INSERT INTO sessions (session_id) VALUES (gen_random_uuid()) "
                                "RETURNING session_id")).scalar()
        conn.execute(text("INSERT INTO feedback (feedback_id, session_id, message_id, rating) "
                          "VALUES (gen_random_uuid(), :sid, gen_random_uuid(), -1)"), {"sid": sid})
    with pytest.raises(IntegrityError), sync_engine.begin() as conn:
        conn.execute(text("INSERT INTO feedback (feedback_id, session_id, message_id, rating) "
                          "VALUES (gen_random_uuid(), :sid, gen_random_uuid(), 0)"), {"sid": sid})


def test_privacy_migration_downgrade_and_upgrade(sync_engine: Engine):
    run_alembic("downgrade", PREVIOUS_REVISION)
    try:
        insp = inspect(sync_engine)
        assert "trace" not in {c["name"] for c in insp.get_columns("audit_logs")}
        assert "erased_at" not in {c["name"] for c in insp.get_columns("sessions")}
        assert not {"session_keys", "feedback"} & set(insp.get_table_names())
        with sync_engine.connect() as conn:
            rules = {r[0] for r in conn.execute(text("SELECT rulename FROM pg_rules"))}
        assert {"no_update_audit", "no_delete_audit", "no_update_events",
                "no_delete_events"} <= rules
    finally:
        run_alembic("upgrade", "head")
    with sync_engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    assert version == HEAD_REVISION


def test_feedback_unique_migration_dedupes_and_enforces(sync_engine: Engine):
    """h2b8d4f6a1c3: duplicates collapse to the newest row, then one row per message."""
    run_alembic("downgrade", PRIVACY_REVISION)
    try:
        insp = inspect(sync_engine)
        assert "legacy_plaintext_rows" not in {c["name"] for c in insp.get_columns("sessions")}
        with sync_engine.begin() as conn:
            sid = conn.execute(text("INSERT INTO sessions (session_id) VALUES "
                                    "(gen_random_uuid()) RETURNING session_id")).scalar()
            mid = conn.execute(text("SELECT gen_random_uuid()")).scalar()
            for rating, age in ((1, "2 minutes"), (-1, "1 minute")):  # a raced duplicate
                conn.execute(text(
                    "INSERT INTO feedback (feedback_id, session_id, message_id, rating, created_at)"
                    " VALUES (gen_random_uuid(), :sid, :mid, :rating, now() - CAST(:age AS interval))"
                ), {"sid": sid, "mid": mid, "rating": rating, "age": age})
    finally:
        run_alembic("upgrade", "head")
    with sync_engine.connect() as conn:
        ratings = conn.execute(text("SELECT rating FROM feedback WHERE message_id = :mid"),
                               {"mid": mid}).scalars().all()
    assert ratings == [-1]  # the newest rating survives
    insp = inspect(sync_engine)
    uniques = {u["name"]: u["column_names"] for u in insp.get_unique_constraints("feedback")}
    assert uniques["uq_feedback_session_message"] == ["session_id", "message_id"]
    assert "legacy_plaintext_rows" in {c["name"] for c in insp.get_columns("sessions")}
    with pytest.raises(IntegrityError), sync_engine.begin() as conn:
        conn.execute(text("INSERT INTO feedback (feedback_id, session_id, message_id, rating) "
                          "VALUES (gen_random_uuid(), :sid, :mid, 1)"), {"sid": sid, "mid": mid})
