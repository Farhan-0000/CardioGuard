"""
Qdrant Vector Indexer & Schema Validator (Module M-18, Ingestion Pipeline).

Validates chunk payloads, computes dense embeddings (bge-small-en-v1.5) and
sparse BM25 vectors via EmbeddingClient, and performs batch upsert to Qdrant.

Constitution §6.4: Dual vector search (Dense + BM25).
Technical Specification Part V §3.1, §4.2, §5.
"""
import structlog
import uuid
from typing import Optional

from qdrant_client import QdrantClient, models

from medbridge.ingestion.chunker import RawChunk
from medbridge.retrieval.embedder import EmbeddingClient, get_embedding_client

logger = structlog.get_logger(__name__)

# Qdrant Collection Configuration (Technical Specification Part V §3.1)
COLLECTION_CONFIG = {
    "collection_name": "clinical_guidelines",
    "vectors_config": {
        "dense": models.VectorParams(
            size=384,
            distance=models.Distance.COSINE,
        ),
    },
    "sparse_vectors_config": {
        "sparse": models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        ),
    },
}


class ChunkValidationError(Exception):
    """Raised when a chunk payload fails schema validation."""
    pass


def validate_chunk_payload(chunk: RawChunk) -> bool:
    """Validate that a chunk meets all required payload constraints.

    Constraints:
        - chunk_id must be non-empty string
        - guideline_id must be non-empty string
        - section_title must be non-empty string
        - page_number must be >= 1
        - chunk_text must be non-empty string with at least 5 characters

    Returns:
        True if valid.

    Raises:
        ChunkValidationError: If any constraint is violated.
    """
    if not chunk.chunk_id or not str(chunk.chunk_id).strip():
        raise ChunkValidationError(f"Invalid chunk_id: '{chunk.chunk_id}'")

    if not chunk.guideline_id or not str(chunk.guideline_id).strip():
        raise ChunkValidationError(f"Invalid guideline_id: '{chunk.guideline_id}' for chunk {chunk.chunk_id}")

    if not chunk.section_title or not str(chunk.section_title).strip():
        raise ChunkValidationError(f"Invalid section_title for chunk {chunk.chunk_id}")

    if not isinstance(chunk.page_number, int) or chunk.page_number < 1:
        raise ChunkValidationError(f"Invalid page_number '{chunk.page_number}' for chunk {chunk.chunk_id}")

    if not chunk.chunk_text or len(chunk.chunk_text.strip()) < 5:
        raise ChunkValidationError(f"chunk_text is too short or empty for chunk {chunk.chunk_id}")

    return True


class QdrantIndexer:
    """Handles collection provisioning, embedding computation, and batch indexing to Qdrant."""

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        embedder: Optional[EmbeddingClient] = None,
        qdrant_url: str = "http://127.0.0.1:6333",
    ) -> None:
        self.client = client or QdrantClient(url=qdrant_url)
        self.embedder = embedder or get_embedding_client()

    def ensure_collection(
        self,
        collection_name: str = "clinical_guidelines",
        recreate: bool = False,
    ) -> None:
        """Create collection with dense and sparse vector configurations if not existing.

        Args:
            collection_name: Name of the Qdrant collection.
            recreate: If True, drops existing collection before recreating.
        """
        if recreate:
            try:
                self.client.delete_collection(collection_name)
                logger.info("Deleted existing collection '%s'", collection_name)
            except Exception as e:
                logger.debug("Collection deletion skipped or failed: %s", e)

        try:
            collections = [c.name for c in self.client.get_collections().collections]
        except Exception:
            collections = []

        if collection_name not in collections:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=384,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                },
            )
            logger.info("Created Qdrant collection '%s' with dense + sparse vector params", collection_name)
        else:
            logger.info("Collection '%s' already exists", collection_name)

    def index_chunks(
        self,
        chunks: list[RawChunk],
        collection_name: str = "clinical_guidelines",
        batch_size: int = 50,
        recreate: bool = False,
        dry_run: bool = False,
    ) -> int:
        """Validate, vectorize, and batch upsert guideline chunks into Qdrant.

        Args:
            chunks: List of RawChunk objects to index.
            collection_name: Target Qdrant collection.
            batch_size: Number of chunks per upsert batch.
            recreate: If True, recreates collection before indexing.
            dry_run: If True, validates chunks without performing vector creation or DB writes.

        Returns:
            Number of successfully validated / indexed chunks.
        """
        if not chunks:
            logger.warning("No chunks provided for indexing")
            return 0

        # 1. Validate all chunk schemas
        valid_chunks: list[RawChunk] = []
        for chunk in chunks:
            try:
                validate_chunk_payload(chunk)
                valid_chunks.append(chunk)
            except ChunkValidationError as err:
                logger.error("Skipping invalid chunk: %s", err)

        logger.info(
            "Schema validation complete: %d/%d chunks valid",
            len(valid_chunks),
            len(chunks),
        )

        if dry_run:
            logger.info("DRY-RUN mode active: skipping Qdrant collection creation and upsert")
            return len(valid_chunks)

        # 2. Ensure collection is configured
        self.ensure_collection(collection_name=collection_name, recreate=recreate)

        # 3. Batch compute embeddings and upsert
        total_indexed = 0

        for i in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[i : i + batch_size]
            points: list[models.PointStruct] = []

            for chunk in batch:
                # Compute dense embedding (384-dim)
                dense_vector = self.embedder.embed_dense(chunk.chunk_text)

                # Compute sparse embedding (BM25)
                sparse_dict = self.embedder.embed_sparse(chunk.chunk_text)
                sparse_vector = models.SparseVector(
                    indices=sparse_dict["indices"],
                    values=sparse_dict["values"],
                )

                # Deterministic UUID from chunk_id
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

                point = models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vector,
                        "sparse": sparse_vector,
                    },
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "guideline_id": chunk.guideline_id,
                        "section_title": chunk.section_title,
                        "page_number": chunk.page_number,
                        "chunk_text": chunk.chunk_text,
                        "source_url": chunk.source_url,
                    },
                )
                points.append(point)

            self.client.upsert(
                collection_name=collection_name,
                points=points,
            )
            total_indexed += len(points)
            logger.info("Upserted batch %d-%d to '%s'", i + 1, i + len(points), collection_name)

        logger.info(
            "Successfully indexed %d chunks into Qdrant collection '%s'",
            total_indexed,
            collection_name,
        )
        return total_indexed
