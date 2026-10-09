"""
Qdrant Hybrid Retriever (ADL-022, TASK-20).

Combines dense embeddings (bge-small-en-v1.5) and sparse BM25 vectors
using Reciprocal Rank Fusion (RRF, k=60) to retrieve Top-20 candidates from Qdrant.

Constitution §6.4: The retrieval module has strictly read-only search access to Qdrant.
Technical Specification Part III §3.5.
"""
import structlog
from typing import Optional

from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models

from medbridge.config import get_settings
from medbridge.retrieval.embedder import EmbeddingClient, get_embedding_client

logger = structlog.get_logger(__name__)


class RetrievedChunk(BaseModel):
    """Pydantic data model representing a retrieved guideline chunk with metadata."""

    chunk_id: str = Field(..., description="Unique chunk identifier, e.g. 'aha_2025_s8_c3'")
    chunk_text: str = Field(..., description="Full text content of the chunk")
    guideline_id: str = Field(..., description="Guideline source identifier, e.g. 'AHA_ACC_2025'")
    section_title: str = Field(..., description="Section or chapter title")
    page_number: int = Field(default=0, description="Source PDF page number")
    source_url: str = Field(default="", description="Reference document or guideline URL")
    score: float = Field(default=0.0, description="RRF or fusion relevance score")


class HybridRetriever:
    """
    Qdrant Hybrid Retriever executing dual-vector search (Dense + BM25)
    fused via Reciprocal Rank Fusion (RRF, k=60).
    """

    def __init__(
        self,
        client: Optional[AsyncQdrantClient] = None,
        embedder: Optional[EmbeddingClient] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self._settings = get_settings()
        self.collection_name = collection_name or self._settings.QDRANT_COLLECTION
        self.top_k = self._settings.RETRIEVAL_TOP_K_CANDIDATES
        self._embedder = embedder or get_embedding_client()
        self._client = client or AsyncQdrantClient(url=self._settings.QDRANT_URL)

    async def hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """
        Execute hybrid search (Dense + BM25 with RRF fusion).

        Args:
            query: Clinical search query string.
            top_k: Number of candidate chunks to retrieve (defaults to RETRIEVAL_TOP_K_CANDIDATES).

        Returns:
            List of RetrievedChunk instances ordered by RRF relevance score.
            Returns an empty list on empty query, empty collection, or connection failure.
        """
        if not query or not query.strip():
            return []

        limit = top_k or self.top_k

        try:
            # 1. Generate query dense embedding (384-dim)
            dense_vector = self._embedder.embed_dense(query)

            # 2. Generate query sparse BM25 indices & values
            sparse_dict = self._embedder.embed_sparse(query)
            sparse_vector = models.SparseVector(
                indices=sparse_dict["indices"],
                values=sparse_dict["values"],
            )

            # 3. Execute Qdrant hybrid query with RRF fusion
            results = await self._client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(query=dense_vector, using="dense", limit=limit),
                    models.Prefetch(query=sparse_vector, using="sparse", limit=limit),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
            )

            # 4. Map ScoredPoint results to typed RetrievedChunk models
            chunks: list[RetrievedChunk] = []
            for point in results.points:
                payload = point.payload or {}
                chunk = RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id") or point.id),
                    chunk_text=str(payload.get("chunk_text") or ""),
                    guideline_id=str(payload.get("guideline_id") or ""),
                    section_title=str(payload.get("section_title") or ""),
                    page_number=int(payload.get("page_number", 0)),
                    source_url=str(payload.get("source_url") or ""),
                    score=float(point.score) if point.score is not None else 0.0,
                )
                chunks.append(chunk)

            logger.info(
                "Hybrid retrieval complete",
                extra={
                    "query": query[:60],
                    "count": len(chunks),
                    "collection": self.collection_name,
                },
            )
            return chunks

        except Exception as e:
            # Acceptance criteria: Returns empty list gracefully on connection failure or empty collection
            logger.warning(
                f"Qdrant hybrid search failed or collection unavailable: {e}",
                extra={"query": query[:60], "error": str(e)},
            )
            return []

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Convenience alias for hybrid_search matching orchestrator protocol."""
        return await self.hybrid_search(query=query, top_k=top_k)

    async def close(self) -> None:
        """Close client connection if initialized."""
        if self._client is not None:
            await self._client.close()


def get_hybrid_retriever() -> HybridRetriever:
    """Convenience factory function for dependency injection."""
    return HybridRetriever()
