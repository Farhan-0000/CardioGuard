"""
Integration tests for the hybrid retriever against an in-memory Qdrant.

Real fastembed dense (bge-small) + sparse (BM25) vectors are indexed into a
``:memory:`` Qdrant collection with the production schema; the retriever then
runs real hybrid (RRF k=60), dense-only and sparse-only searches. No Qdrant
server is needed (models are cached locally after the first download).
"""

from typing import Any

import pytest
from qdrant_client import AsyncQdrantClient, models

from medbridge.retrieval.embedder import EmbeddingClient, get_embedding_client
from medbridge.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk

COLLECTION_NAME = "test_clinical_guidelines"
EMPTY_COLLECTION_NAME = "test_empty_guidelines"

DOCUMENTS: list[dict[str, Any]] = [
    {
        "id": 1,
        "chunk_id": "aha_acc_2025_htn_c1",
        "chunk_text": (
            "First-line pharmacologic therapy for primary hypertension includes thiazide "
            "diuretics, CCBs, and ACE inhibitors or ARBs."
        ),
        "guideline_id": "AHA_ACC_2025",
        "doc_title": "2025 AHA/ACC High Blood Pressure Guideline",
        "org": "AHA/ACC",
        "year": 2025,
        "section_title": "First-Line Pharmacologic Therapy",
        "page_number": 24,
        "page_end": 25,
        "source_url": "https://guidelines.acc.org/htn2025",
        "topics": ["hypertension", "medication"],
    },
    {
        "id": 2,
        "chunk_id": "ada_2024_dm2_c1",
        "chunk_text": (
            "First-line therapy for type 2 diabetes consists of metformin and comprehensive "
            "lifestyle modification."
        ),
        "guideline_id": "ADA_2024",
        "doc_title": "ADA Standards of Care 2024",
        "org": "ADA",
        "year": 2024,
        "section_title": "Pharmacologic Approaches to Glycemic Treatment",
        "page_number": 105,
        "source_url": "https://diabetesjournals.org/care/2024",
        "topics": ["diabetes"],
    },
    {
        "id": 3,
        "chunk_id": "acc_aha_2023_hf_c1",
        "chunk_text": (
            "Guideline-directed medical therapy (GDMT) for heart failure with reduced ejection "
            "fraction (HFrEF) includes ARNi/ACEi/ARB, beta-blocker, MRA, and SGLT2 inhibitor."
        ),
        "guideline_id": "ACC_AHA_2023",
        "doc_title": "2022 AHA/ACC/HFSA Heart Failure Guideline",
        "org": "AHA/ACC/HFSA",
        "year": 2022,
        "section_title": "Guideline-Directed Medical Therapy for HFrEF",
        "page_number": 42,
        "source_url": "https://guidelines.acc.org/hf2023",
        "topics": ["heart_failure"],
    },
]


@pytest.fixture(scope="module")
def real_embedder() -> EmbeddingClient:
    return get_embedding_client()


@pytest.fixture(scope="module")
def indexed_points(real_embedder: EmbeddingClient) -> list[models.PointStruct]:
    """Embed the sample chunks once per module (pure CPU work, no event loop)."""
    points = []
    for doc in DOCUMENTS:
        sparse = real_embedder.embed_sparse(doc["chunk_text"])
        points.append(
            models.PointStruct(
                id=doc["id"],
                vector={
                    "dense": real_embedder.embed_dense(doc["chunk_text"]),
                    "sparse": models.SparseVector(
                        indices=sparse["indices"], values=sparse["values"]
                    ),
                },
                payload={k: v for k, v in doc.items() if k != "id"},
            )
        )
    return points


async def _create_collection(client: AsyncQdrantClient, name: str) -> None:
    await client.create_collection(
        collection_name=name,
        vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )


@pytest.fixture
async def qdrant_client(indexed_points):
    client = AsyncQdrantClient(location=":memory:")
    await _create_collection(client, COLLECTION_NAME)
    await _create_collection(client, EMPTY_COLLECTION_NAME)
    await client.upsert(collection_name=COLLECTION_NAME, points=indexed_points)
    yield client
    await client.close()


@pytest.fixture
def real_retriever(qdrant_client, real_embedder):
    return HybridRetriever(
        client=qdrant_client, embedder=real_embedder, collection_name=COLLECTION_NAME
    )


async def test_real_hybrid_retrieval_hypertension_query(real_retriever):
    chunks = await real_retriever.hybrid_search(
        "What is the first-line medication for primary hypertension?", top_k=2
    )
    assert chunks
    top = chunks[0]
    assert isinstance(top, RetrievedChunk)
    assert top.chunk_id == "aha_acc_2025_htn_c1"
    assert top.guideline_id == "AHA_ACC_2025"
    assert "thiazide" in top.chunk_text.lower()
    assert top.page_number == 24 and top.page_end == 25
    assert top.org == "AHA/ACC" and top.year == 2025
    assert top.doc_title == "2025 AHA/ACC High Blood Pressure Guideline"
    assert top.topics == ["hypertension", "medication"]
    # Qdrant RRF score = sum 1/(k + rank), rank from 0. With k=60 a chunk ranked first by
    # both retrievers scores 2/60; Qdrant's un-parameterised default (k=2) would give 1.0.
    assert 0.0 < top.score_rrf <= 2 / 60 + 1e-9


async def test_real_hybrid_retrieval_diabetes_query(real_retriever):
    chunks = await real_retriever.hybrid_search(
        "Metformin initial therapy for type 2 diabetes glycemic control", top_k=2
    )
    assert chunks[0].chunk_id == "ada_2024_dm2_c1"
    assert "metformin" in chunks[0].chunk_text.lower()


async def test_real_hybrid_retrieval_heart_failure_query(real_retriever):
    chunks = await real_retriever.retrieve(
        "GDMT for heart failure with reduced ejection fraction SGLT2 beta-blocker", top_k=2
    )
    assert chunks[0].chunk_id == "acc_aha_2023_hf_c1"
    assert "hfref" in chunks[0].chunk_text.lower()


@pytest.mark.parametrize("mode", ["dense", "sparse"])
async def test_single_vector_modes(real_retriever, mode):
    chunks = await real_retriever.search(
        "heart failure SGLT2 inhibitor therapy", top_k=3, mode=mode
    )
    assert chunks[0].chunk_id == "acc_aha_2023_hf_c1"
    assert chunks[0].score_rrf is None


async def test_real_hybrid_retrieval_empty_query(real_retriever):
    assert await real_retriever.hybrid_search("") == []
    assert await real_retriever.hybrid_search("   \n\t  ") == []


async def test_real_hybrid_retrieval_empty_collection(qdrant_client, real_embedder):
    empty = HybridRetriever(
        client=qdrant_client, embedder=real_embedder, collection_name=EMPTY_COLLECTION_NAME
    )
    assert await empty.hybrid_search("hypertension medication") == []


async def test_real_hybrid_retrieval_nonexistent_collection(qdrant_client, real_embedder):
    missing = HybridRetriever(
        client=qdrant_client, embedder=real_embedder, collection_name="nonexistent_collection_404"
    )
    assert await missing.hybrid_search("hypertension medication") == []
    assert await missing.collection_points() is None


async def test_collection_points_and_warmup(real_retriever):
    assert await real_retriever.collection_points() == len(DOCUMENTS)
    await real_retriever.warmup()
