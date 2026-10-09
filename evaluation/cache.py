"""
Disk-backed request cache for evaluation memoisation (finalization plan §1, §3.3).

``DiskRequestCache`` structurally implements
``medbridge.ai.llm_wrapper.RequestCache`` (``get(key) -> str | None``,
``set(key, value) -> None``) and is installed by the runner and the judge with
``set_request_cache(cache)``. The LLM wrapper builds a content-addressed key
from the full request (model, prompts, schema, temperature, seed, ...), so
identical requests issued by different conditions — e.g. the extractor call
shared by C3 and C5 — are computed once, and switching model automatically
switches namespace.

Storage is a single SQLite file in WAL mode, safe across threads (one
connection guarded by a lock) and across processes (SQLite file locking with a
busy timeout).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from evaluation.paths import CACHE_PATH

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS llm_cache ("
    " key TEXT PRIMARY KEY,"
    " value TEXT NOT NULL,"
    " created_at REAL NOT NULL)"
)


class DiskRequestCache:
    """Thread-safe SQLite key/value cache for LLM responses."""

    def __init__(self, path: Path = CACHE_PATH, *, timeout_s: float = 30.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), timeout=timeout_s, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(_SCHEMA)
            self._conn.commit()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def get(self, key: str) -> str | None:
        """Return the cached value for ``key`` or ``None``."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM llm_cache WHERE key = ?", (key,)).fetchone()
            if row is None:
                self.misses += 1
                return None
            self.hits += 1
            return str(row[0])

    def set(self, key: str, value: str) -> None:
        """Store ``value`` under ``key`` (last write wins)."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, value, created_at) VALUES (?, ?, ?)",
                (key, value, time.time()),
            )
            self._conn.commit()
            self.writes += 1

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()
        return int(row[0]) if row else 0

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM llm_cache WHERE key = ?", (key,)).fetchone()
        return row is not None

    def stats(self) -> dict[str, int | str]:
        """Hit/miss/write counters for the run manifest."""
        return {
            "path": str(self.path),
            "entries": len(self),
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
        }

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()


def install_request_cache(path: Path | None) -> DiskRequestCache | None:
    """Create a ``DiskRequestCache`` and register it with the LLM wrapper.

    ``path=None`` disables memoisation (and clears any installed cache).
    Returns the cache, or ``None`` when disabled.
    """
    from medbridge.ai.llm_wrapper import set_request_cache

    if path is None:
        set_request_cache(None)
        return None
    cache = DiskRequestCache(path)
    set_request_cache(cache)
    return cache
