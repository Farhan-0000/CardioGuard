"""
Qdrant Vector Indexer & Schema Validator (Module M-18, Ingestion Pipeline).

Validates chunk payloads, computes dense (``bge-small-en-v1.5``, 384-d) and
sparse (``Qdrant/bm25``) vectors in batches via :class:`EmbeddingClient`, and
upserts them into Qdrant. The ingestion module is the only writer of the
collection (Constitution §6.4); retrieval is read-only.

Payload keys per point follow interface contract §3.5: ``chunk_id, chunk_text,
guideline_id, doc_title, org, year, section_title, page_number, page_end,
source_url, topics, doc_sha256, chunk_index, content_type`` -- plus the additive
``source_format`` ("pdf" | "html"; for HTML, pages are section indices).

Technical Specification Part V §3.1, §4.2, §5.
"""
from __future__ import annotations

import time
import uuid
from collections import Counter
from collections.abc import Iterator
from typing import Any, Protocol

import structlog
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from medbridge.ingestion.chunker import RawChunk

logger = structlog.get_logger(__name__)

DEFAULT_COLLECTION = "clinical_guidelines"
DENSE_VECTOR_SIZE = 384
DEFAULT_BATCH_SIZE = 64
# Keyword payload indexes used by incremental ingestion, statistics and filters.
KEYWORD_INDEXES = ("guideline_id", "doc_sha256", "content_type", "topics")

# Qdrant Collection Configuration (Technical Specification Part V §3.1)
DENSE_VECTORS_CONFIG = {
    "dense": models.VectorParams(size=DENSE_VECTOR_SIZE, distance=models.Distance.COSINE),
}
SPARSE_VECTORS_CONFIG = {
    "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
}


class Embedder(Protocol):
    """Subset of :class:`medbridge.retrieval.embedder.EmbeddingClient` used here."""

    def embed_dense_batch(self, texts: list[str]) -> list[list[float]]: ...

    def embed_sparse_batch(self, texts: list[str]) -> list[dict[str, Any]]: ...


class ChunkValidationError(Exception):
    """Raised when a chunk payload fails schema validation."""


def validate_chunk_payload(chunk: RawChunk) -> bool:
    """Validate that a chunk meets all required payload constraints.

    Constraints:
        - chunk_id, guideline_id and section_title are non-empty strings
        - 1 <= page_number <= page_end
        - chunk_text has at least 5 non-blank characters
        - content_type is "text" or "table"

    Returns:
        True if valid.

    Raises:
        ChunkValidationError: If any constraint is violated.
    """
    if not chunk.chunk_id or not str(chunk.chunk_id).strip():
        raise ChunkValidationError(f"Invalid chunk_id: '{chunk.chunk_id}'")
    if not chunk.guideline_id or not str(chunk.guideline_id).strip():
        raise ChunkValidationError(
            f"Invalid guideline_id: '{chunk.guideline_id}' for chunk {chunk.chunk_id}"
        )
    if not chunk.section_title or not str(chunk.section_title).strip():
        raise ChunkValidationError(f"Invalid section_title for chunk {chunk.chunk_id}")
    if not isinstance(chunk.page_number, int) or chunk.page_number < 1:
        raise ChunkValidationError(
            f"Invalid page_number '{chunk.page_number}' for chunk {chunk.chunk_id}"
        )
    if chunk.page_end < chunk.page_number:
        raise ChunkValidationError(f"page_end < page_number for chunk {chunk.chunk_id}")
    if not chunk.chunk_text or len(chunk.chunk_text.strip()) < 5:
        raise ChunkValidationError(f"chunk_text is too short or empty for chunk {chunk.chunk_id}")
    if chunk.content_type not in ("text", "table"):
        raise ChunkValidationError(f"Invalid content_type for chunk {chunk.chunk_id}")
    return True


def point_id_for(chunk_id: str) -> str:
    """Deterministic Qdrant point id for a chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def _match(key: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


class QdrantIndexer:
    """Collection provisioning, batch embedding and indexing to Qdrant."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        embedder: Embedder | None = None,
        qdrant_url: str = "http://127.0.0.1:6333",
    ) -> None:
        self.client = client or QdrantClient(url=qdrant_url)
        if embedder is None:
            from medbridge.retrieval.embedder import get_embedding_client

            embedder = get_embedding_client()
        self.embedder = embedder

    # ------------------------------------------------------------- collection
    def collection_exists(self, collection_name: str = DEFAULT_COLLECTION) -> bool:
        """True when the collection exists on the server."""
        names = [c.name for c in self.client.get_collections().collections]
        return collection_name in names

    def ensure_collection(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        recreate: bool = False,
    ) -> None:
        """Create the collection (dense + sparse vectors) and payload indexes if missing.

        Args:
            collection_name: Name of the Qdrant collection.
            recreate: If True, drops the existing collection first (full rebuild).
        """
        if recreate and self.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
            logger.info("Deleted existing collection", collection=collection_name)

        if self.collection_exists(collection_name):
            logger.info("Collection already exists", collection=collection_name)
            return

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=DENSE_VECTORS_CONFIG,
            sparse_vectors_config=SPARSE_VECTORS_CONFIG,
        )
        for field_name in KEYWORD_INDEXES:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        logger.info("Created Qdrant collection with dense + sparse vectors",
                    collection=collection_name)

    # ---------------------------------------------------------------- queries
    def count(self, collection_name: str = DEFAULT_COLLECTION, **conditions: str) -> int:
        """Exact number of points matching all ``key=value`` payload conditions."""
        flt = (
            models.Filter(must=[_match(k, v) for k, v in conditions.items()])
            if conditions else None
        )
        try:
            return int(self.client.count(collection_name, count_filter=flt, exact=True).count)
        except UnexpectedResponse as exc:
            logger.warning("Qdrant count failed", collection=collection_name, error=str(exc))
            return 0

    def scroll_payloads(
        self, collection_name: str = DEFAULT_COLLECTION, fields: list[str] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield the payload (optionally only ``fields``) of every point."""
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                collection_name=collection_name,
                limit=1024,
                offset=offset,
                with_payload=fields if fields is not None else True,
                with_vectors=False,
            )
            for point in points:
                yield point.payload or {}
            if offset is None:
                return

    def payload_counts(
        self, collection_name: str = DEFAULT_COLLECTION
    ) -> dict[str, Counter[str]]:
        """Per-guideline counts of points by ``content_type``."""
        counts: dict[str, Counter[str]] = {}
        for payload in self.scroll_payloads(collection_name, ["guideline_id", "content_type"]):
            gid = str(payload.get("guideline_id", ""))
            counts.setdefault(gid, Counter())[str(payload.get("content_type", "text"))] += 1
        return counts

    def delete_document(
        self, guideline_id: str, collection_name: str = DEFAULT_COLLECTION
    ) -> None:
        """Delete every point belonging to ``guideline_id``."""
        self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[_match("guideline_id", guideline_id)])
            ),
            wait=True,
        )
        logger.info("Deleted document points", guideline_id=guideline_id)

    # --------------------------------------------------------------- indexing
    def index_chunks(
        self,
        chunks: list[RawChunk],
        collection_name: str = DEFAULT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
        recreate: bool = False,
        dry_run: bool = False,
    ) -> int:
        """Validate, embed (dense + sparse, batched) and upsert chunks into Qdrant.

        Args:
            chunks: Chunks to index.
            collection_name: Target Qdrant collection.
            batch_size: Chunks per embedding/upsert batch.
            recreate: If True, recreates the collection before indexing.
            dry_run: If True, validates only (no embeddings, no writes).

        Returns:
            Number of validated (dry run) or indexed chunks.
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return 0

        valid_chunks: list[RawChunk] = []
        for chunk in chunks:
            try:
                validate_chunk_payload(chunk)
                valid_chunks.append(chunk)
            except ChunkValidationError as err:
                logger.error("Skipping invalid chunk", error=str(err))
        logger.info("Schema validation complete", valid=len(valid_chunks), total=len(chunks))

        if dry_run:
            logger.info("DRY-RUN: skipping Qdrant collection creation and upsert")
            return len(valid_chunks)

        self.ensure_collection(collection_name=collection_name, recreate=recreate)

        total_indexed = 0
        started = time.perf_counter()
        for start in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[start:start + batch_size]
            texts = [c.chunk_text for c in batch]
            dense = self.embedder.embed_dense_batch(texts)
            sparse = self.embedder.embed_sparse_batch(texts)
            points = [
                models.PointStruct(
                    id=point_id_for(chunk.chunk_id),
                    vector={
                        "dense": list(dense_vec),
                        "sparse": models.SparseVector(
                            indices=list(sparse_vec["indices"]),
                            values=list(sparse_vec["values"]),
                        ),
                    },
                    payload=chunk.to_payload(),
                )
                for chunk, dense_vec, sparse_vec in zip(batch, dense, sparse, strict=True)
            ]
            self.client.upsert(collection_name=collection_name, points=points, wait=True)
            total_indexed += len(points)
            elapsed = time.perf_counter() - started
            logger.info(
                "Upserted batch",
                progress=f"{total_indexed}/{len(valid_chunks)}",
                chunks_per_s=round(total_indexed / elapsed, 1) if elapsed else None,
                collection=collection_name,
            )

        logger.info("Indexed chunks", count=total_indexed, collection=collection_name)
        return total_indexed
