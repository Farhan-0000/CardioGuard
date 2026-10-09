"""
In-process LRU / TTL cache for the retrieval layer (ADL-027 performance work).

Used for two things:

* query embeddings (``EmbeddingClient``) — LRU only, embeddings are deterministic;
* hybrid-search results (``HybridRetriever``) — LRU + TTL, so a re-indexed
  collection is picked up after ``RETRIEVAL_CACHE_TTL_SECONDS``.

The cache is thread-safe (embeddings are computed in worker threads) and
bounded; ``maxsize <= 0`` disables it entirely.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Generic, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    """A small thread-safe LRU cache with an optional per-entry time-to-live."""

    def __init__(
        self,
        maxsize: int,
        ttl_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._maxsize = maxsize
        self._ttl = ttl_seconds if ttl_seconds and ttl_seconds > 0 else None
        self._clock = clock
        self._data: OrderedDict[K, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        """Whether the cache stores anything at all."""
        return self._maxsize > 0

    def get(self, key: K) -> V | None:
        """Return the cached value (refreshing its LRU position) or ``None``."""
        if not self.enabled:
            return None
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, value = entry
            if self._ttl is not None and self._clock() - stored_at > self._ttl:
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: K, value: V) -> None:
        """Insert or replace ``key``, evicting the least recently used entry when full."""
        if not self.enabled:
            return
        with self._lock:
            self._data[key] = (self._clock(), value)
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        """Drop every entry and reset the hit/miss counters."""
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
