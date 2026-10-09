"""
Qdrant Hybrid Retriever (ADL-022, TASK-20, Module M-16).

Dense (bge-small-en-v1.5) and sparse (BM25) candidate lists are fused inside
Qdrant with Reciprocal Rank Fusion using an explicit ``k = 60``
(``RrfQuery(rrf=Rrf(k=60))`` — Qdrant's un-parameterised ``FusionQuery(RRF)``
uses a much smaller default constant, so the parameter is set explicitly to
match the architecture specification). The top-20 fused candidates are handed
to the cross-encoder reranker by the orchestrator.

Performance (ADL-027): the two query encoders run concurrently in worker
threads, query embeddings are LRU-cached by the embedding client and fused
results are cached here (LRU + TTL keyed by ``(query, top_k, mode)``).

Constitution §6.4: the retrieval module has strictly read-only access to Qdrant.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models

from medbridge.config import Settings, get_settings
from medbridge.retrieval.cache import TTLCache
from medbridge.retrieval.embedder import EmbeddingClient, get_embedding_client

logger = structlog.get_logger(__name__)

SearchMode = Literal["hybrid", "dense", "sparse"]

RRF_K: int = 60
_WARMUP_QUERY = "blood pressure target"


class RetrievedChunk(BaseModel):
    """A retrieved guideline chunk with its provenance metadata and scores."""

    chunk_id: str = Field(..., description="Unique chunk identifier, e.g. 'aha_2025_s8_c3'")
    chunk_text: str = Field(..., description="Full text content of the chunk")
    guideline_id: str = Field(..., description="Guideline source identifier, e.g. 'AHA_ACC_2025'")
    section_title: str = Field(..., description="Section or chapter title")
    page_number: int = Field(default=0, description="Source PDF page number")
    source_url: str = Field(default="", description="Reference document or guideline URL")
    score: float = Field(default=0.0, description="Current relevance score (RRF, then rerank)")
    doc_title: str | None = Field(default=None, description="Full document title")
    org: str | None = Field(default=None, description="Issuing organisation, e.g. 'AHA/ACC'")
    year: int | None = Field(default=None, description="Publication year")
    page_end: int | None = Field(default=None, description="Last source page of the chunk")
    topics: list[str] = Field(default_factory=list, description="Topic tags")
    score_rrf: float | None = Field(default=None, description="Reciprocal Rank Fusion score")
    score_rerank: float | None = Field(default=None, description="Cross-encoder score")


# ----------------------------------------------------------------------
# Payload parsing (contract §3.5 keys, tolerant of the pre-v4 key names)
# ----------------------------------------------------------------------


def _first(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _as_topics(value: Any) -> list[str]:
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    if isinstance(value, (list, tuple)):
        return [str(t) for t in value if str(t).strip()]
    return []


def chunk_from_payload(
    point_id: Any, payload: dict[str, Any] | None, score: float | None, mode: SearchMode
) -> RetrievedChunk:
    """Map a Qdrant point payload onto a :class:`RetrievedChunk`."""
    data = payload or {}
    fused = float(score) if score is not None else 0.0
    return RetrievedChunk(
        chunk_id=str(_first(data, "chunk_id") or point_id),
        chunk_text=str(_first(data, "chunk_text", "text") or ""),
        guideline_id=str(_first(data, "guideline_id", "doc_id") or ""),
        section_title=str(_first(data, "section_title", "section") or ""),
        page_number=_as_int(_first(data, "page_number", "page")) or 0,
        source_url=str(_first(data, "source_url", "url") or ""),
        score=fused,
        doc_title=_as_str(_first(data, "doc_title", "title")),
        org=_as_str(_first(data, "org", "organisation", "organization")),
        year=_as_int(_first(data, "year")),
        page_end=_as_int(_first(data, "page_end")),
        topics=_as_topics(data.get("topics")),
        score_rrf=fused if mode == "hybrid" else None,
    )


class HybridRetriever:
    """Qdrant hybrid (dense + BM25) retriever fused with RRF (k = 60)."""

    def __init__(
        self,
        client: AsyncQdrantClient | None = None,
        embedder: EmbeddingClient | None = None,
        collection_name: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.collection_name = collection_name or self._settings.QDRANT_COLLECTION
        self.top_k = self._settings.RETRIEVAL_TOP_K_CANDIDATES
        self._embedder = embedder or get_embedding_client()
        self._client = client
        self._cache: TTLCache[tuple[str, int, str], list[RetrievedChunk]] = TTLCache(
            maxsize=self._settings.RETRIEVAL_CACHE_SIZE,
            ttl_seconds=self._settings.RETRIEVAL_CACHE_TTL_SECONDS,
        )

    @property
    def embedder(self) -> EmbeddingClient:
        """The embedding client used for query vectors."""
        return self._embedder

    def _get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(url=self._settings.QDRANT_URL, timeout=10)
        return self._client

    async def hybrid_search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Hybrid (dense + sparse, RRF k=60) search — the production retrieval path.

        Returns an empty list on an empty query, an empty/missing collection or
        any connection/embedding failure (the Evidence Gate then decides).
        """
        return await self.search(query, top_k=top_k, mode="hybrid")

    async def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Alias for :meth:`hybrid_search`."""
        return await self.hybrid_search(query=query, top_k=top_k)

    async def search(
        self, query: str, top_k: int | None = None, mode: SearchMode = "hybrid"
    ) -> list[RetrievedChunk]:
        """Search the collection with one retrieval variant.

        ``mode="dense"`` and ``mode="sparse"`` exist for the retrieval
        evaluation (dense-only vs sparse-only vs hybrid ablation).
        """
        if not query or not query.strip():
            return []
        limit = top_k or self.top_k
        cache_key = (query, limit, mode)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return list(cached)

        try:
            embedding = await self._embedder.embed_query(query)
            sparse_vector = models.SparseVector(
                indices=embedding.sparse["indices"], values=embedding.sparse["values"]
            )
            client = self._get_client()
            if mode == "dense":
                result = await client.query_points(
                    collection_name=self.collection_name,
                    query=embedding.dense,
                    using="dense",
                    limit=limit,
                    with_payload=True,
                )
            elif mode == "sparse":
                result = await client.query_points(
                    collection_name=self.collection_name,
                    query=sparse_vector,
                    using="sparse",
                    limit=limit,
                    with_payload=True,
                )
            else:
                result = await client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        models.Prefetch(query=embedding.dense, using="dense", limit=limit),
                        models.Prefetch(query=sparse_vector, using="sparse", limit=limit),
                    ],
                    query=models.RrfQuery(rrf=models.Rrf(k=RRF_K)),
                    limit=limit,
                    with_payload=True,
                )
            chunks = [
                chunk_from_payload(point.id, point.payload, point.score, mode)
                for point in result.points
            ]
        except Exception as exc:  # Qdrant down, collection missing, embedder crash
            logger.warning(
                "retrieval_failed",
                mode=mode,
                collection=self.collection_name,
                error=type(exc).__name__,
                detail=str(exc)[:200],
            )
            return []

        logger.info(
            "retrieval_complete", mode=mode, count=len(chunks), collection=self.collection_name
        )
        if chunks:
            self._cache.set(cache_key, chunks)
        return list(chunks)

    async def collection_points(self) -> int | None:
        """Number of points in the collection (``None`` when unreachable)."""
        try:
            info = await self._get_client().get_collection(self.collection_name)
        except Exception as exc:
            logger.warning("collection_info_failed", error=type(exc).__name__)
            return None
        return int(info.points_count) if info.points_count is not None else None

    async def warmup(self) -> None:
        """Load both query encoders and run one probe search (opens the Qdrant connection).

        Raises if the probe cannot reach Qdrant so callers can record readiness.
        """
        await asyncio.to_thread(self._embedder.warmup)
        embedding = await self._embedder.embed_query(_WARMUP_QUERY)
        await self._get_client().query_points(
            collection_name=self.collection_name,
            query=embedding.dense,
            using="dense",
            limit=1,
            with_payload=False,
        )

    def clear_cache(self) -> None:
        """Drop all cached search results."""
        self._cache.clear()

    async def close(self) -> None:
        """Close the Qdrant client (a new one is created lazily on next use)."""
        if self._client is not None:
            await self._client.close()
            self._client = None


_retriever_instance: HybridRetriever | None = None
_retriever_lock = threading.Lock()


def get_hybrid_retriever() -> HybridRetriever:
    """Return the process-wide :class:`HybridRetriever` singleton (thread-safe)."""
    global _retriever_instance
    if _retriever_instance is None:
        with _retriever_lock:
            if _retriever_instance is None:
                _retriever_instance = HybridRetriever()
    return _retriever_instance
