"""
Tests for the dense + sparse embedding client (medbridge/retrieval/embedder.py).

The first group loads the real fastembed models (cached locally after the
first download); the async/caching group uses stubbed encoders.
"""

import asyncio
import threading
import time
from unittest.mock import patch

import pytest

from medbridge.retrieval.cache import TTLCache
from medbridge.retrieval.embedder import EmbeddingClient, QueryEmbedding, get_embedding_client


@pytest.fixture(scope="module")
def embedder() -> EmbeddingClient:
    return get_embedding_client()


def test_singleton(embedder):
    assert get_embedding_client() is embedder
    assert EmbeddingClient() is embedder


def test_dense_dimension_384(embedder):
    dense = embedder.embed_dense("Hyperkalemia management requires calcium gluconate.")
    assert isinstance(dense, list)
    assert len(dense) == 384
    assert all(isinstance(x, float) for x in dense)


def test_sparse_indices_values(embedder):
    sparse = embedder.embed_sparse("Hyperkalemia management requires calcium gluconate.")
    assert set(sparse) == {"indices", "values"}
    assert len(sparse["indices"]) == len(sparse["values"]) > 0


def test_batch_consistency(embedder):
    texts = ["Patient presents with headache.", "Severe migraine and photophobia."]
    dense_batch = embedder.embed_dense_batch(texts)
    assert len(dense_batch) == 2 and len(dense_batch[0]) == 384
    sparse_batch = embedder.embed_sparse_batch(texts)
    assert len(sparse_batch) == 2 and "indices" in sparse_batch[0]
    assert abs(dense_batch[0][0] - embedder.embed_dense(texts[0])[0]) < 1e-5


def test_warmup_loads_both_models(embedder):
    assert embedder.warmup() == {"dense": True, "sparse": True}
    assert embedder.loaded == {"dense": True, "sparse": True}


async def test_embed_query_matches_sync_encoders(embedder):
    embedder.clear_cache()
    text = "blood pressure target for diabetes"
    result = await embedder.embed_query(text)
    assert isinstance(result, QueryEmbedding)
    assert result.dense == embedder.embed_dense(text)
    assert result.sparse == embedder.embed_sparse(text)


# ---------------------------------------------------------------------------
# Async path with stubbed encoders (concurrency + LRU cache)
# ---------------------------------------------------------------------------


async def test_embed_query_runs_encoders_concurrently_and_caches(embedder):
    embedder.clear_cache()
    running: list[str] = []
    overlap = threading.Event()
    lock = threading.Lock()

    def slow(kind: str):
        def encode(text: str):
            with lock:
                running.append(kind)
                if len(running) == 2:
                    overlap.set()
            overlap.wait(timeout=1.0)
            time.sleep(0.01)
            return [1.0] if kind == "dense" else {"indices": [1], "values": [1.0]}

        return encode

    with (
        patch.object(embedder, "embed_dense", side_effect=slow("dense")) as dense,
        patch.object(embedder, "embed_sparse", side_effect=slow("sparse")) as sparse,
    ):
        first = await embedder.embed_query("unique query for cache test")
        second = await embedder.embed_query("unique query for cache test")

    assert overlap.is_set()  # both encoders were in flight at the same time
    assert first is second
    assert dense.call_count == 1 and sparse.call_count == 1
    embedder.clear_cache()


async def test_concurrent_queries_do_not_block_event_loop(embedder):
    embedder.clear_cache()

    def blocking(text: str):
        time.sleep(0.05)
        return [0.0]

    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0.005)
            ticks += 1

    with (
        patch.object(embedder, "embed_dense", side_effect=blocking),
        patch.object(embedder, "embed_sparse", side_effect=lambda t: {"indices": [], "values": []}),
    ):
        await asyncio.gather(embedder.embed_query("loop check"), ticker())
    assert ticks == 5
    embedder.clear_cache()


# ---------------------------------------------------------------------------
# TTLCache
# ---------------------------------------------------------------------------


def test_cache_lru_eviction():
    cache: TTLCache[str, int] = TTLCache(maxsize=2)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1  # refresh "a"
    cache.set("c", 3)  # evicts "b"
    assert cache.get("b") is None
    assert cache.get("a") == 1 and cache.get("c") == 3
    assert len(cache) == 2
    assert cache.hits == 3 and cache.misses == 1


def test_cache_ttl_expiry():
    now = [100.0]
    cache: TTLCache[str, int] = TTLCache(maxsize=4, ttl_seconds=10, clock=lambda: now[0])
    cache.set("k", 1)
    now[0] += 9
    assert cache.get("k") == 1
    now[0] += 2
    assert cache.get("k") is None
    assert len(cache) == 0


def test_cache_disabled_when_size_zero():
    cache: TTLCache[str, int] = TTLCache(maxsize=0)
    cache.set("k", 1)
    assert cache.get("k") is None
    assert not cache.enabled


def test_cache_clear_resets_counters():
    cache: TTLCache[str, int] = TTLCache(maxsize=2)
    cache.set("k", 1)
    cache.get("k")
    cache.clear()
    assert len(cache) == 0 and cache.hits == 0 and cache.misses == 0
