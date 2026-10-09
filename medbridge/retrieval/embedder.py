"""
Dense + sparse embedding client (Module M-16 support, ADL-022).

* Dense: ``BAAI/bge-small-en-v1.5`` (384-d) via fastembed.
* Sparse: ``Qdrant/bm25`` via fastembed.

Thread-safe process-wide singleton with lazy model loading. For query-time
use the async :meth:`EmbeddingClient.embed_query` runs the dense and sparse
encoders concurrently in worker threads (``asyncio.gather`` +
``asyncio.to_thread``) and memoises the result in a bounded LRU cache, so a
repeated query (e.g. a regenerated answer or a repeated evaluation item) costs
nothing. :meth:`warmup` loads both models at application start-up so the first
patient query does not pay the model-loading cost.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding

from medbridge.config import get_settings
from medbridge.retrieval.cache import TTLCache

_WARMUP_TEXT = "blood pressure treatment guideline"


@dataclass(frozen=True)
class QueryEmbedding:
    """Dense and sparse vectors for one query (treat as read-only)."""

    dense: list[float]
    sparse: dict[str, list[Any]]  # {"indices": [...], "values": [...]}


class EmbeddingClient:
    """Unified, thread-safe client for dense and sparse embeddings (lazy singleton)."""

    _instance: EmbeddingClient | None = None
    _lock = threading.Lock()
    _initialized: bool

    def __new__(cls) -> EmbeddingClient:
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        settings = get_settings()
        self.dense_model_name: str = settings.EMBEDDING_MODEL
        self.sparse_model_name: str = settings.SPARSE_MODEL
        self._dense_model: TextEmbedding | None = None
        self._sparse_model: SparseTextEmbedding | None = None
        self._init_lock = threading.Lock()
        self._query_cache: TTLCache[str, QueryEmbedding] = TTLCache(
            maxsize=settings.RETRIEVAL_CACHE_SIZE
        )
        self._initialized = True

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _get_dense_model(self) -> TextEmbedding:
        if self._dense_model is None:
            with self._init_lock:
                if self._dense_model is None:
                    self._dense_model = TextEmbedding(model_name=self.dense_model_name)
        return self._dense_model

    def _get_sparse_model(self) -> SparseTextEmbedding:
        if self._sparse_model is None:
            with self._init_lock:
                if self._sparse_model is None:
                    self._sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name)
        return self._sparse_model

    @property
    def loaded(self) -> dict[str, bool]:
        """Which models are resident in memory (used by ``/health``)."""
        return {"dense": self._dense_model is not None, "sparse": self._sparse_model is not None}

    def warmup(self) -> dict[str, bool]:
        """Load both models and run one probe embedding each (blocking; call in a thread).

        Returns ``{"dense": bool, "sparse": bool}`` — ``False`` for a model that
        failed to load (the error propagates only if *both* fail).
        """
        status = {"dense": False, "sparse": False}
        errors: list[Exception] = []
        for name, embed in (("dense", self.embed_dense), ("sparse", self.embed_sparse)):
            try:
                embed(_WARMUP_TEXT)
                status[name] = True
            except Exception as exc:  # model download / ONNX runtime failure
                errors.append(exc)
        if len(errors) == 2:
            raise errors[0]
        return status

    # ------------------------------------------------------------------
    # Synchronous encoders (also used by ingestion)
    # ------------------------------------------------------------------

    def embed_dense(self, text: str) -> list[float]:
        """Generate a dense embedding for a single text string."""
        model = self._get_dense_model()
        result = next(iter(model.embed([text])))
        dense: list[float] = result.tolist()
        return dense

    def embed_sparse(self, text: str) -> dict[str, list[Any]]:
        """Generate a sparse BM25 embedding (``{"indices", "values"}``) for one text."""
        model = self._get_sparse_model()
        result = next(iter(model.embed([text])))
        return {"indices": result.indices.tolist(), "values": result.values.tolist()}

    def embed_dense_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embeddings for a batch of text strings."""
        model = self._get_dense_model()
        return [res.tolist() for res in model.embed(texts)]

    def embed_sparse_batch(self, texts: list[str]) -> list[dict[str, list[Any]]]:
        """Generate sparse embeddings for a batch of text strings."""
        model = self._get_sparse_model()
        return [
            {"indices": res.indices.tolist(), "values": res.values.tolist()}
            for res in model.embed(texts)
        ]

    # ------------------------------------------------------------------
    # Async query path
    # ------------------------------------------------------------------

    async def embed_query(self, text: str) -> QueryEmbedding:
        """Embed a search query with both encoders concurrently (LRU-cached).

        The CPU-bound ONNX inference runs in the default thread pool so the
        event loop is never blocked.
        """
        cached = self._query_cache.get(text)
        if cached is not None:
            return cached
        dense, sparse = await asyncio.gather(
            asyncio.to_thread(self.embed_dense, text),
            asyncio.to_thread(self.embed_sparse, text),
        )
        embedding = QueryEmbedding(dense=dense, sparse=sparse)
        self._query_cache.set(text, embedding)
        return embedding

    def clear_cache(self) -> None:
        """Drop all cached query embeddings."""
        self._query_cache.clear()


def get_embedding_client() -> EmbeddingClient:
    """Return the process-wide embedding client singleton."""
    return EmbeddingClient()
