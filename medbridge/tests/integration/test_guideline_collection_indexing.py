"""
Integration verification of the production guideline collection (TASK-34).

Requires a running Qdrant with the corpus indexed
(``python -m medbridge.ingestion --recreate``); skipped automatically when Qdrant
is unreachable (``qdrant`` marker).

Verifies:
1. Collection ``clinical_guidelines`` has 384-d COSINE dense + IDF-weighted sparse vectors.
2. Every curated document in ``data/corpus_manifest.json`` (>= 15) is indexed, with
   per-document point counts matching the manifest.
3. Payloads carry every key of interface contract §3.5 with sane values.
4. Hybrid retrieval returns relevant chunks across key clinical topics.
"""
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient, models

from medbridge.config import get_settings
from medbridge.ingestion.manifest import load_manifest
from medbridge.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk

pytestmark = pytest.mark.qdrant

COLLECTION_NAME = "clinical_guidelines"
MANIFEST = load_manifest(Path(__file__).resolve().parents[3] / "data" / "corpus_manifest.json")
EXPECTED_IDS = set(MANIFEST.by_id())
CONTRACT_PAYLOAD_KEYS = {
    "chunk_id", "chunk_text", "guideline_id", "doc_title", "org", "year", "section_title",
    "page_number", "page_end", "source_url", "topics", "doc_sha256", "chunk_index",
    "content_type",
}


@pytest.fixture
async def live_qdrant_client():
    """Connect to the live Qdrant instance."""
    client = AsyncQdrantClient(url=get_settings().QDRANT_URL)
    yield client
    await client.close()


async def _count(client: AsyncQdrantClient, **conditions: str) -> int:
    flt = models.Filter(must=[
        models.FieldCondition(key=k, match=models.MatchValue(value=v))
        for k, v in conditions.items()
    ])
    result = await client.count(COLLECTION_NAME, count_filter=flt, exact=True)
    return result.count


async def test_collection_configuration(live_qdrant_client: AsyncQdrantClient):
    """Collection exists with 384-dim dense and BM25 (IDF) sparse configurations."""
    names = [c.name for c in (await live_qdrant_client.get_collections()).collections]
    assert COLLECTION_NAME in names, f"Collection '{COLLECTION_NAME}' not found in Qdrant!"

    info = await live_qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    vectors_config = info.config.params.vectors
    assert vectors_config["dense"].size == 384
    assert vectors_config["dense"].distance == models.Distance.COSINE
    sparse_config = info.config.params.sparse_vectors
    assert sparse_config is not None and "sparse" in sparse_config
    assert sparse_config["sparse"].modifier == models.Modifier.IDF


async def test_every_manifest_document_is_indexed(live_qdrant_client: AsyncQdrantClient):
    """>= 15 curated documents, each indexed with the manifest's chunk count and hash."""
    assert len(EXPECTED_IDS) >= 15
    info = await live_qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    assert info.points_count == sum(d.chunk_count for d in MANIFEST.documents)
    for doc in MANIFEST.documents:
        indexed = await _count(live_qdrant_client, guideline_id=doc.guideline_id)
        assert indexed == doc.chunk_count > 0, doc.guideline_id
        same_hash = await _count(live_qdrant_client, guideline_id=doc.guideline_id,
                                 doc_sha256=doc.sha256)
        assert same_hash == indexed, f"{doc.guideline_id} indexed from a different file version"
    tables = await _count(live_qdrant_client, content_type="table")
    assert tables == sum(d.table_chunk_count for d in MANIFEST.documents) > 0


async def test_payload_integrity(live_qdrant_client: AsyncQdrantClient):
    """Payloads carry every contract key with valid values."""
    points, _ = await live_qdrant_client.scroll(
        collection_name=COLLECTION_NAME, limit=300, with_payload=True, with_vectors=False,
    )
    assert len(points) >= 15
    by_id = MANIFEST.by_id()
    for point in points:
        payload = point.payload or {}
        assert CONTRACT_PAYLOAD_KEYS <= set(payload), set(CONTRACT_PAYLOAD_KEYS) - set(payload)
        doc = by_id[payload["guideline_id"]]
        assert payload["doc_title"] == doc.title and payload["org"] == doc.org
        assert payload["year"] == doc.year and payload["topics"] == doc.topics
        assert payload["source_url"] == doc.url
        assert 1 <= payload["page_number"] <= payload["page_end"]
        assert payload["content_type"] in {"text", "table"}
        assert payload["source_format"] == doc.source_format
        assert isinstance(payload["chunk_index"], int) and payload["chunk_index"] >= 0
        assert payload["section_title"].strip()
        assert len(payload["chunk_text"].strip()) >= 10


@pytest.mark.parametrize("query,expected_keyword,expected_ids", [
    ("hypertension diagnostic criteria and blood pressure thresholds", "blood pressure", None),
    ("first line pharmacotherapy calcium channel blocker amlodipine", "calcium", None),
    ("lifestyle modifications sodium restriction DASH diet", "sodium", None),
    ("resistant hypertension aldosterone antagonist spironolactone", "spironolactone", None),
    ("anticoagulation to prevent stroke in atrial fibrillation", "anticoagula",
     {"NICE_NG196", "ACC_AHA_AF_2023", "MEDLINEPLUS_AFIB"}),
    ("statin therapy to lower LDL cholesterol", "statin",
     {"NICE_NG238", "AHA_ACC_CHOLESTEROL_2018", "ACC_AHA_PREVENTION_2019",
      "CCD_GUIDELINE_REVIEW_2024", "MEDLINEPLUS_CHOLESTEROL"}),
])
async def test_hybrid_search_across_clinical_topics(
    live_qdrant_client: AsyncQdrantClient,
    query: str,
    expected_keyword: str,
    expected_ids: set[str] | None,
):
    """Hybrid search returns relevant top-k chunks from curated documents."""
    retriever = HybridRetriever(client=live_qdrant_client, collection_name=COLLECTION_NAME)
    chunks = await retriever.hybrid_search(query=query, top_k=5)
    assert len(chunks) > 0, f"No chunks returned for clinical query: '{query}'"
    for c in chunks:
        assert isinstance(c, RetrievedChunk)
        assert c.chunk_id and c.guideline_id in EXPECTED_IDS
        assert len(c.chunk_text) > 20 and c.score > 0.0
    combined_text = " ".join(c.chunk_text.lower() for c in chunks)
    assert expected_keyword in combined_text, f"'{expected_keyword}' missing for '{query}'"
    if expected_ids:
        assert {c.guideline_id for c in chunks} & expected_ids
