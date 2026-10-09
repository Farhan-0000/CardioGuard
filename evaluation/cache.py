"""
Disk-backed request cache for evaluation memoisation (finalization plan §1, §3.3).

``DiskRequestCache`` structurally implements
``medbridge.ai.llm_wrapper.RequestCache`` (``get(key) -> str | None``,
``set(key, value) -> None``) and is installed by the runner and the judge with
``set_request_cache(cache)``. The LLM wrapper builds a content-addressed key
from the full request (model tag, prompts, schema, temperature, seed, ...), so
identical requests issued by different conditions — e.g. the extractor call
shared by C3 and C5 — are computed once.

Model weights (review finding ablation-study-validity-5). The wrapper's key
names the model by its *tag*; ``ollama pull`` can move a tag to new weights.
The cache file is therefore **namespaced by the model digests**: the runner
and the judge resolve ``{tag: digest}`` from Ollama ``/api/tags`` for every
model they will call and open ``<stem>.<sha256(map)[:12]>.sqlite``. A re-pulled
model has a new digest, hence a new namespace, so outputs of older weights are
never replayed. The map is stored in the file's ``cache_meta`` table and in the
run manifest. When a digest cannot be resolved the caller runs uncached.

Only complete JSON documents are memoised (finding statistics-judges-5): a
response cut off at ``num_predict`` is not valid JSON, so it is neither stored
nor served — a re-run recomputes it (and the caller sees the truncation)
instead of replaying a silently "repaired" partial output.

Targeted bypass for re-runs of degraded turns: inside
``bypass_cache_reads(call_names)`` reads for those LLM calls miss (the request
is recomputed and the fresh output overwrites the entry) while every other call
is still memoised. The call name is published by ``instrument_llm`` (a thin
wrapper around ``llm.call`` that sets a context variable); when the current call
is unknown every read in the bypass scope misses.

Storage is a single SQLite file in WAL mode, safe across threads (one
connection guarded by a lock), concurrent asyncio tasks (every operation is
synchronous and atomic) and processes (SQLite file locking with a busy timeout).
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator, Mapping
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from evaluation.paths import CACHE_PATH

ALL_CALLS = "*"

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS llm_cache ("
    " key TEXT PRIMARY KEY,"
    " value TEXT NOT NULL,"
    " created_at REAL NOT NULL)"
)
_META_SCHEMA = "CREATE TABLE IF NOT EXISTS cache_meta (name TEXT PRIMARY KEY, value TEXT NOT NULL)"

#: Logical LLM call currently executing (set by ``instrument_llm``).
_CURRENT_CALL: ContextVar[str | None] = ContextVar("evaluation_llm_call", default=None)
#: Call names whose cached reads are bypassed in the current context.
_BYPASS: ContextVar[frozenset[str] | None] = ContextVar("evaluation_cache_bypass", default=None)


class CacheNamespaceError(RuntimeError):
    """The cache file belongs to a different set of model digests."""


def is_complete_json(value: str) -> bool:
    """True when ``value`` is one complete JSON document (a truncated output is not)."""
    try:
        json.loads(value)
    except (TypeError, ValueError):
        return False
    return True


def digest_namespace(model_digests: Mapping[str, str]) -> str:
    """Short, order-independent hash of a ``{model tag: digest}`` map."""
    blob = json.dumps(dict(sorted(model_digests.items())), separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def namespaced_path(base: Path, model_digests: Mapping[str, str]) -> Path:
    """``<dir>/<stem>.<namespace><suffix>`` for the given digest map."""
    base = Path(base)
    return base.with_name(f"{base.stem}.{digest_namespace(model_digests)}{base.suffix}")


def model_digests(ollama_info: Mapping[str, Any], models: Iterable[str]) -> dict[str, str]:
    """``{tag: digest}`` for ``models`` from ``manifest.ollama_models`` output.

    An untagged name also matches ``<name>:latest`` (Ollama's convention).
    Raises ``LookupError`` naming every model whose digest is unknown.
    """
    if "error" in ollama_info:
        raise LookupError(f"Ollama model list unavailable: {ollama_info['error']}")
    out: dict[str, str] = {}
    missing: list[str] = []
    for model in sorted(set(models)):
        entry = ollama_info.get(model) or ollama_info.get(f"{model}:latest")
        digest = entry.get("digest") if isinstance(entry, Mapping) else None
        if digest:
            out[model] = str(digest)
        else:
            missing.append(model)
    if missing:
        raise LookupError(f"no Ollama digest for {', '.join(missing)}")
    return out


@contextlib.contextmanager
def bypass_cache_reads(call_names: Iterable[str]) -> Iterator[frozenset[str]]:
    """Make cached reads miss for ``call_names`` (``ALL_CALLS`` = every call) in this context.

    Context variables propagate into tasks created inside the block, so every
    LLM call of a re-run pipeline turn is covered.
    """
    names = frozenset(call_names)
    token = _BYPASS.set(names or None)
    try:
        yield names
    finally:
        _BYPASS.reset(token)


def _read_bypassed() -> bool:
    names = _BYPASS.get()
    if not names:
        return False
    if ALL_CALLS in names:
        return True
    current = _CURRENT_CALL.get()
    return current is None or current in names


def instrument_llm(llm: Any) -> bool:
    """Publish the logical call name of every ``llm.call(...)`` to the cache (idempotent).

    Wraps the instance's ``call`` coroutine method; arguments and results pass
    through unchanged. Returns False when ``llm`` has no ``call`` method.
    """
    original = getattr(llm, "call", None)
    if original is None:
        return False
    if getattr(original, "_evaluation_instrumented", False):
        return True

    @functools.wraps(original)
    async def call(*args: Any, **kwargs: Any) -> Any:
        name = kwargs.get("call_name", args[0] if args else None)
        token = _CURRENT_CALL.set(None if name is None else str(name))
        try:
            return await original(*args, **kwargs)
        finally:
            _CURRENT_CALL.reset(token)

    call._evaluation_instrumented = True  # type: ignore[attr-defined]
    llm.call = call
    return True


class DiskRequestCache:
    """Thread-safe SQLite key/value cache for LLM responses."""

    def __init__(
        self,
        path: Path = CACHE_PATH,
        *,
        timeout_s: float = 30.0,
        model_digests: Mapping[str, str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_digests = dict(sorted((model_digests or {}).items()))
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.bypassed = 0
        self.rejected = 0
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), timeout=timeout_s, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(_SCHEMA)
            self._conn.execute(_META_SCHEMA)
            self._conn.commit()
        if self.model_digests:
            try:
                self._bind_namespace()
            except CacheNamespaceError:
                self.close()
                raise

    def _bind_namespace(self) -> None:
        """Record the digest map in the file, or fail if the file holds another map."""
        stored = self.metadata().get("model_digests")
        wanted = json.dumps(self.model_digests, sort_keys=True)
        if stored is None:
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO cache_meta (name, value) VALUES (?, ?)",
                    ("model_digests", wanted),
                )
                self._conn.commit()
        elif stored != wanted:
            raise CacheNamespaceError(
                f"{self.path} was written for model digests {stored}, not {wanted}; "
                "use the namespaced path (namespaced_path) or delete the file"
            )

    def metadata(self) -> dict[str, str]:
        """The ``cache_meta`` table (e.g. ``model_digests``)."""
        with self._lock:
            rows = self._conn.execute("SELECT name, value FROM cache_meta").fetchall()
        return {str(name): str(value) for name, value in rows}

    def get(self, key: str) -> str | None:
        """Return the cached value for ``key`` or ``None`` (bypassed reads always miss)."""
        bypassed = _read_bypassed()
        with self._lock:
            if bypassed:
                self.bypassed += 1
                return None
            row = self._conn.execute("SELECT value FROM llm_cache WHERE key = ?", (key,)).fetchone()
            if row is None or not is_complete_json(str(row[0])):
                self.misses += 1
                return None
            self.hits += 1
            return str(row[0])

    def set(self, key: str, value: str) -> None:
        """Store ``value`` under ``key`` (last write wins); incomplete JSON is not stored."""
        if not is_complete_json(value):
            with self._lock:
                self.rejected += 1
            return
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

    def stats(self) -> dict[str, Any]:
        """Hit/miss/write counters and provenance for the run manifest."""
        return {
            "path": str(self.path),
            "entries": len(self),
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "bypassed_reads": self.bypassed,
            "rejected_incomplete": self.rejected,
            "model_digests": dict(self.model_digests),
            "key_includes_digest": bool(self.model_digests),
        }

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()


def install_request_cache(
    path: Path | None,
    *,
    model_digests: Mapping[str, str] | None = None,
    instrument: bool = True,
) -> DiskRequestCache | None:
    """Create a ``DiskRequestCache`` and register it with the LLM wrapper.

    ``path=None`` disables memoisation (and clears any installed cache). With
    ``model_digests`` the file is namespaced by them (``namespaced_path``).
    ``instrument`` publishes call names for ``bypass_cache_reads`` (on the shared
    wrapper singleton used by every pipeline component and the judge).
    Returns the cache, or ``None`` when disabled.
    """
    llm_wrapper = importlib.import_module("medbridge.ai.llm_wrapper")
    if path is None:
        llm_wrapper.set_request_cache(None)
        return None
    target = namespaced_path(path, model_digests) if model_digests else Path(path)
    cache = DiskRequestCache(target, model_digests=model_digests)
    llm_wrapper.set_request_cache(cache)
    get_wrapper = getattr(llm_wrapper, "get_llm_wrapper", None)
    if instrument and callable(get_wrapper):
        instrument_llm(get_wrapper())
    return cache
