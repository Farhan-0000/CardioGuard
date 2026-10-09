"""
Unit tests for the Qdrant hybrid retriever (medbridge/retrieval/hybrid_retriever.py).

The Qdrant client and the embedding client are mocked: query construction
(RRF with k = 60), payload mapping (contract §3.5 + legacy keys), search modes,
the result cache and graceful failure handling.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import models
from qdrant_client.http import models as http_models
from qdrant_client.http.exceptions import UnexpectedResponse

from medbridge.config import get_settings
from medbridge.retrieval.embedder import QueryEmbedding
from medbridge.retrieval.hybrid_retriever import (
    RRF_K,
    HybridRetriever,
    RetrievedChunk,
    chunk_from_payload,
    get_hybrid_retriever,
)

FULL_PAYLOAD = {
    "chunk_id": "aha_2025_s1_c1",
    "chunk_text": "Hypertension first line treatment includes ACE inhibitors.",
    "guideline_id": "AHA_ACC_2025",
    "doc_title": "2025 AHA/ACC Guideline for High Blood Pressure",
    "org": "AHA/ACC",
    "year": 2025,
    "section_title": "Pharmacologic Treatment",
    "page_number": 15,
    "page_end": 16,
    "source_url": "https://guidelines.example.org/htn2025",
    "topics": ["hypertension", "medication"],
    "doc_sha256": "0" * 64,
    "chunk_index": 3,
    "content_type": "text",
}


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_query = AsyncMock(
        return_value=QueryEmbedding(
            dense=[0.1] * 384, sparse={"indices": [10, 42, 99], "values": [0.5, 0.8, 1.2]}
        )
    )
    return embedder


@pytest.fixture
def mock_qdrant_client():
    return AsyncMock()


@pytest.fixture
def retriever(mock_qdrant_client, mock_embedder):
    return HybridRetriever(
        client=mock_qdrant_client, embedder=mock_embedder, collection_name="clinical_guidelines"
    )


def respond(client, *payloads, score=0.0333):
    points = [
        models.ScoredPoint(id=f"pt_{i}", version=1, score=score, payload=p)
        for i, p in enumerate(payloads)
    ]
    client.query_points.return_value = http_models.QueryResponse(points=points)


async def test_empty_or_whitespace_query_returns_empty_list(retriever, mock_qdrant_client):
    for query in ("", "   "):
        assert await retriever.hybrid_search(query) == []
        assert await retriever.retrieve(query) == []
        assert await retriever.search(query, mode="dense") == []
    mock_qdrant_client.query_points.assert_not_called()


async def test_hybrid_query_uses_rrf_k60(retriever, mock_qdrant_client, mock_embedder):
    respond(mock_qdrant_client, FULL_PAYLOAD)
    results = await retriever.hybrid_search("hypertension treatment", top_k=10)

    mock_embedder.embed_query.assert_awaited_once_with("hypertension treatment")
    kwargs = mock_qdrant_client.query_points.call_args.kwargs
    assert kwargs["collection_name"] == "clinical_guidelines"
    assert kwargs["limit"] == 10
    assert kwargs["with_payload"] is True
    dense, sparse = kwargs["prefetch"]
    assert dense.using == "dense" and dense.query == [0.1] * 384 and dense.limit == 10
    assert sparse.using == "sparse" and sparse.limit == 10
    assert sparse.query.indices == [10, 42, 99] and sparse.query.values == [0.5, 0.8, 1.2]
    assert isinstance(kwargs["query"], models.RrfQuery)
    assert kwargs["query"].rrf.k == RRF_K == 60

    assert len(results) == 1
    chunk = results[0]
    assert isinstance(chunk, RetrievedChunk)
    assert chunk.chunk_id == "aha_2025_s1_c1"
    assert chunk.doc_title == "2025 AHA/ACC Guideline for High Blood Pressure"
    assert chunk.org == "AHA/ACC" and chunk.year == 2025
    assert chunk.page_number == 15 and chunk.page_end == 16
    assert chunk.topics == ["hypertension", "medication"]
    assert chunk.source_url == "https://guidelines.example.org/htn2025"
    assert chunk.score == pytest.approx(0.0333) and chunk.score_rrf == pytest.approx(0.0333)
    assert chunk.score_rerank is None


async def test_default_top_k_from_settings(retriever, mock_qdrant_client):
    respond(mock_qdrant_client, FULL_PAYLOAD)
    await retriever.hybrid_search("bp")
    assert mock_qdrant_client.query_points.call_args.kwargs["limit"] == (
        get_settings().RETRIEVAL_TOP_K_CANDIDATES
    )


@pytest.mark.parametrize("mode,using", [("dense", "dense"), ("sparse", "sparse")])
async def test_single_vector_modes(retriever, mock_qdrant_client, mode, using):
    respond(mock_qdrant_client, FULL_PAYLOAD, score=0.8)
    results = await retriever.search("heart failure", top_k=7, mode=mode)
    kwargs = mock_qdrant_client.query_points.call_args.kwargs
    assert kwargs["using"] == using and kwargs["limit"] == 7
    assert "prefetch" not in kwargs
    if mode == "sparse":
        assert isinstance(kwargs["query"], models.SparseVector)
    else:
        assert kwargs["query"] == [0.1] * 384
    assert results[0].score == pytest.approx(0.8)
    assert results[0].score_rrf is None


async def test_legacy_and_missing_payload_fallbacks(retriever, mock_qdrant_client):
    legacy = {"chunk_id": "old_c1", "text": "Legacy text", "section": "Old section", "page": "7"}
    point_none_score = MagicMock(id="pt_888", score=None, payload={"chunk_id": "chunk_custom"})
    point_no_payload = http_models.ScoredPoint(id="pt_999", version=1, score=0.0166, payload=None)
    mock_qdrant_client.query_points.return_value = MagicMock(
        points=[point_no_payload, point_none_score, MagicMock(id="pt_1", score=0.5, payload=legacy)]
    )
    results = await retriever.hybrid_search("heart failure")

    assert results[0].chunk_id == "pt_999"
    assert results[0].chunk_text == "" and results[0].guideline_id == ""
    assert results[0].page_number == 0 and results[0].source_url == ""
    assert results[0].org is None and results[0].year is None and results[0].topics == []
    assert results[1].chunk_id == "chunk_custom" and results[1].score == 0.0
    assert results[2].chunk_text == "Legacy text"
    assert results[2].section_title == "Old section" and results[2].page_number == 7


def test_payload_parsing_is_tolerant():
    chunk = chunk_from_payload(
        "id-1",
        {"chunk_id": "c", "year": "not a year", "topics": "a, b,", "page_number": None, "org": 5},
        None,
        "hybrid",
    )
    assert chunk.year is None
    assert chunk.topics == ["a", "b"]
    assert chunk.page_number == 0
    assert chunk.org == "5"
    assert chunk.score == 0.0 and chunk.score_rrf == 0.0


async def test_results_are_cached_per_query_topk_mode(retriever, mock_qdrant_client):
    respond(mock_qdrant_client, FULL_PAYLOAD)
    first = await retriever.hybrid_search("cached query", top_k=5)
    second = await retriever.hybrid_search("cached query", top_k=5)
    assert first == second and first is not second
    assert mock_qdrant_client.query_points.await_count == 1
    await retriever.hybrid_search("cached query", top_k=6)
    await retriever.search("cached query", top_k=5, mode="dense")
    assert mock_qdrant_client.query_points.await_count == 3
    retriever.clear_cache()
    await retriever.hybrid_search("cached query", top_k=5)
    assert mock_qdrant_client.query_points.await_count == 4


async def test_empty_results_are_not_cached(retriever, mock_qdrant_client):
    mock_qdrant_client.query_points.return_value = http_models.QueryResponse(points=[])
    assert await retriever.hybrid_search("nothing") == []
    assert await retriever.hybrid_search("nothing") == []
    assert mock_qdrant_client.query_points.await_count == 2


async def test_graceful_handling_on_qdrant_connection_error(retriever, mock_qdrant_client):
    mock_qdrant_client.query_points.side_effect = ConnectionError("Qdrant connection refused")
    assert await retriever.hybrid_search("query during network outage") == []


async def test_graceful_handling_on_unexpected_response(retriever, mock_qdrant_client):
    mock_qdrant_client.query_points.side_effect = UnexpectedResponse(
        status_code=404,
        reason_phrase="Not Found",
        content=b'{"status":{"error":"Collection not found"}}',
        headers={},
    )
    assert await retriever.hybrid_search("query on missing collection") == []


async def test_graceful_handling_on_embedder_failure(retriever, mock_embedder):
    mock_embedder.embed_query.side_effect = RuntimeError("Embedding model runtime failure")
    assert await retriever.hybrid_search("query when model crashes") == []


async def test_collection_points(retriever, mock_qdrant_client):
    mock_qdrant_client.get_collection.return_value = MagicMock(points_count=905)
    assert await retriever.collection_points() == 905
    mock_qdrant_client.get_collection.side_effect = ConnectionError("down")
    assert await retriever.collection_points() is None


async def test_warmup_loads_embedder_and_probes_qdrant(
    retriever, mock_qdrant_client, mock_embedder
):
    mock_embedder.warmup = MagicMock(return_value={"dense": True, "sparse": True})
    await retriever.warmup()
    mock_embedder.warmup.assert_called_once()
    kwargs = mock_qdrant_client.query_points.call_args.kwargs
    assert kwargs["limit"] == 1 and kwargs["using"] == "dense"


async def test_warmup_propagates_qdrant_failure(retriever, mock_qdrant_client, mock_embedder):
    mock_embedder.warmup = MagicMock(return_value={"dense": True, "sparse": True})
    mock_qdrant_client.query_points.side_effect = ConnectionError("down")
    with pytest.raises(ConnectionError):
        await retriever.warmup()


async def test_close_client_connection(retriever, mock_qdrant_client):
    await retriever.close()
    mock_qdrant_client.close.assert_awaited_once()
    await retriever.close()  # idempotent: client already released
    mock_qdrant_client.close.assert_awaited_once()


def test_embedder_property(retriever, mock_embedder):
    assert retriever.embedder is mock_embedder


def test_factory_returns_singleton():
    first = get_hybrid_retriever()
    assert isinstance(first, HybridRetriever)
    assert get_hybrid_retriever() is first


def test_payload_source_format_is_read():
    html = chunk_from_payload(
        "id-2",
        {"chunk_id": "h", "chunk_text": "t", "guideline_id": "MEDLINEPLUS_HIGH_BP",
         "section_title": "Treatment", "page_number": 3, "source_format": "HTML"},
        0.5,
        "hybrid",
    )
    assert html.source_format == "html" and html.page_number == 3
    legacy = chunk_from_payload("id-3", {"chunk_id": "p", "chunk_text": "t"}, 0.5, "hybrid")
    assert legacy.source_format is None
