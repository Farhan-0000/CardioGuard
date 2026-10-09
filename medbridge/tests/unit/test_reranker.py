"""
Tests for the cross-encoder reranker (medbridge/retrieval/reranker.py).

Uses the real ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (cached locally) for the
ranking-quality checks and stubs for device selection.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from medbridge.retrieval.hybrid_retriever import RetrievedChunk
from medbridge.retrieval.reranker import CrossEncoderReranker, detect_device, get_reranker


@pytest.fixture(scope="module")
def reranker() -> CrossEncoderReranker:
    return get_reranker()


def make_chunk(i: int, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"c{i}",
        chunk_text=text,
        guideline_id="G",
        section_title="S",
        score=1.0 / (60 + i),
        score_rrf=1.0 / (60 + i),
    )


async def test_empty_chunks_handling(reranker):
    assert await reranker.rerank("Query", []) == []
    assert await reranker.rerank_chunks("Query", []) == []


async def test_rerank_output_count(reranker):
    chunks = [
        "Patient ate an apple.",
        "Treatment for hypertension includes ACE inhibitors.",
        "The sky is blue today.",
        "Amlodipine is a calcium channel blocker used for high blood pressure.",
        "Blood pressure reading was 120/80.",
        "Lisinopril is often prescribed.",
    ]
    result = await reranker.rerank("What is the treatment for hypertension?", chunks, top_k=3)
    assert len(result) == 3
    assert isinstance(result[0], tuple)
    assert isinstance(result[0][0], str) and isinstance(result[0][1], float)


async def test_score_sorting_descending(reranker):
    chunks = ["I like cats and dogs.", "Amlodipine 5mg daily for hypertension.", "Weather text."]
    result = await reranker.rerank("hypertension medication", chunks, top_k=3)
    assert result[0][1] >= result[1][1] >= result[2][1]
    assert "Amlodipine" in result[0][0]


async def test_rerank_chunks_preserves_rrf_and_sets_rerank(reranker):
    chunks = [
        make_chunk(1, "The weather is sunny."),
        make_chunk(2, "Thiazide diuretics are first-line therapy for hypertension."),
        make_chunk(3, "Cats sleep a lot."),
    ]
    result = await reranker.rerank_chunks("first-line hypertension drug", chunks, top_k=2)
    assert [c.chunk_id for c in result][0] == "c2"
    assert len(result) == 2
    for item in result:
        original = next(c for c in chunks if c.chunk_id == item.chunk_id)
        assert item.score_rrf == original.score_rrf
        assert item.score_rerank is not None and item.score == item.score_rerank
        assert item is not original
    assert result[0].score_rerank >= result[1].score_rerank
    assert all(c.score_rerank is None for c in chunks)  # inputs not mutated


async def test_warmup_loads_model(reranker):
    await reranker.warmup()
    assert reranker.loaded is True


def test_singleton(reranker):
    assert get_reranker() is reranker
    assert reranker.device in {"cuda", "mps", "cpu"}


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_detect_device_priority(cuda, mps, expected):
    fake_torch = types.SimpleNamespace(
        cuda=MagicMock(is_available=MagicMock(return_value=cuda)),
        backends=types.SimpleNamespace(mps=MagicMock(is_available=MagicMock(return_value=mps))),
    )
    with patch.dict(sys.modules, {"torch": fake_torch}):
        assert detect_device() == expected


def test_detect_device_without_torch():
    with patch.dict(sys.modules, {"torch": None}):
        assert detect_device() == "cpu"
