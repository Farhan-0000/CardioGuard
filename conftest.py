"""
Repository-wide pytest configuration.

* Provides deterministic test secrets so privacy features (encryption, signed
  session tokens) work without a developer ``.env``.
* Service markers — ``postgres``, ``qdrant``, ``ollama`` — skip a test
  automatically when that service is not reachable, so the default suite is
  green on a laptop without Docker/Ollama running.
* ``live`` tests (full stack, real LLM) are excluded by default
  (``addopts = -m 'not live'``); run them with ``pytest -m live``.
* Database isolation: tests never touch the development/evaluation database.
  ``DATABASE_URL`` is rewritten to a sibling ``medbridge_test`` database
  (override with ``TEST_DATABASE_URL``), created and migrated on demand.
  Destructive tests (e.g. ``alembic downgrade base``) therefore cannot wipe
  live data.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
from functools import cache
from pathlib import Path
from urllib.parse import urlparse

import pytest

REPO_ROOT = Path(__file__).resolve().parent

# Deterministic, test-only secrets (never used outside the test process).
os.environ.setdefault("ENCRYPTION_MASTER_KEY", "dGVzdC1vbmx5LWtleS1kby1ub3QtdXNlLWluLXByb2Q=")
os.environ.setdefault("SESSION_TOKEN_SECRET", "test-only-session-token-secret")
os.environ.setdefault("ADMIN_PASSWORD", "test-admin-password")
os.environ.setdefault("WARMUP_ON_STARTUP", "false")


def _derive_test_database_url() -> str:
    explicit = os.environ.get("TEST_DATABASE_URL")
    if explicit:
        return explicit
    from medbridge.config import Settings

    base = Settings().DATABASE_URL
    head, _, _dbname = base.rpartition("/")
    return f"{head}/medbridge_test"


os.environ["DATABASE_URL"] = _derive_test_database_url()


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@cache
def _service_available(name: str) -> bool:
    from medbridge.config import get_settings

    settings = get_settings()
    if name == "postgres":
        url = urlparse(settings.DATABASE_URL.replace("+asyncpg", ""))
        return _port_open(url.hostname or "127.0.0.1", url.port or 5432)
    if name == "qdrant":
        url = urlparse(settings.QDRANT_URL)
        return _port_open(url.hostname or "127.0.0.1", url.port or 6333)
    if name == "ollama":
        url = urlparse(settings.OLLAMA_BASE_URL)
        return _port_open(url.hostname or "127.0.0.1", url.port or 11434)
    return True


def _ensure_test_database() -> None:
    """Create ``medbridge_test`` if missing and migrate it to head (best effort)."""
    url = urlparse(os.environ["DATABASE_URL"].replace("+asyncpg", ""))
    dbname = url.path.lstrip("/")
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=url.hostname, port=url.port or 5432, user=url.username,
            password=url.password, dbname="postgres", connect_timeout=3,
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
            if cur.fetchone() is None:
                cur.execute(f'CREATE DATABASE "{dbname}"')
        conn.close()
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[conftest] could not prepare test database: {exc}", file=sys.stderr)
        return
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=REPO_ROOT, env=dict(os.environ), check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    if _service_available("postgres"):
        _ensure_test_database()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        for service in ("postgres", "qdrant", "ollama"):
            if item.get_closest_marker(service) and not _service_available(service):
                item.add_marker(pytest.mark.skip(reason=f"{service} not reachable"))
