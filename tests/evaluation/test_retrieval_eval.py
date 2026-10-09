"""Retrieval metrics on toy data with a fake retriever/reranker."""

import json
import math
from types import SimpleNamespace

import pytest

from evaluation import retrieval_eval as re_


def chunk(gid, page, page_end=None, cid=None, text=None):
    return SimpleNamespace(
        chunk_id=cid or f"{gid}-{page}",
        guideline_id=gid,
        page_number=page,
        page_end=page_end,
        chunk_text=text or f"{gid} page {page}",
    )


def test_chunk_matches_page_ranges():
    span = re_.RelevantSpan(guideline_id="ESC_2024", pages=[10, 11])
    assert re_.chunk_matches(chunk("esc_2024", 11), span)
    assert re_.chunk_matches(chunk("ESC_2024", 8, 10), span)
    assert not re_.chunk_matches(chunk("ESC_2024", 12, 14), span)
    assert not re_.chunk_matches(chunk("WHO", 10), span)
    assert re_.chunk_matches(chunk("WHO", 99), re_.RelevantSpan(guideline_id="who"))


def test_query_metrics_known_values():
    spans = [
        re_.RelevantSpan(guideline_id="A", pages=[1]),
        re_.RelevantSpan(guideline_id="B", pages=[2]),
    ]
    ranked = [chunk("X", 1), chunk("A", 1), chunk("A", 1, cid="dup"), chunk("Y", 1)] + [
        chunk("Z", i) for i in range(10)
    ]
    ranked += [chunk("B", 2)]  # rank 15
    m = re_.query_metrics(ranked, spans)
    assert m["recall@5"] == 0.5 and m["recall@20"] == 1.0
    assert m["mrr"] == pytest.approx(0.5)
    idcg = 1 + 1 / math.log2(3)
    assert m["ndcg@5"] == pytest.approx(
        (1 / math.log2(3)) / idcg
    )  # duplicate hit is not credited twice


def test_aliases_in_loader(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "id": "R1",
                        "query": "q",
                        "gold": [{"guideline_id": "A", "page_start": 3, "page_end": 5}],
                    },
                    {"id": "R2", "query": "q2", "relevant": [{"guideline_id": "B", "page": 7}]},
                ]
            }
        )
    )
    queries = re_.load_queries(path)
    assert queries[0].relevant[0].pages == [3, 4, 5] and queries[1].relevant[0].pages == [7]
    path.write_text(json.dumps([{"id": "R3", "query": "q", "relevant": [{"guideline_id": "C"}]}]))
    assert re_.load_queries(path)[0].relevant[0].pages == []


class FakeModesRetriever:
    async def search(self, query, top_k=20, mode="hybrid"):
        good = chunk("A", 1, text="relevant long passage text")
        bad = [chunk("Z", i, text="z") for i in range(top_k - 1)]
        if mode == "dense":
            return [good, *bad]
        if mode == "sparse":
            return bad
        return [*bad[:3], good, *bad[3:]]


class LengthReranker:
    async def rerank(self, query, texts, top_k=None):
        return sorted(((t, float(len(t))) for t in texts), key=lambda x: -x[1])[:top_k]


async def test_evaluate_and_write_outputs(tmp_path):
    queries = [
        re_.RetrievalQuery(
            id="R1", query="q", relevant=[re_.RelevantSpan(guideline_id="A", pages=[1])]
        )
    ]
    result = await re_.evaluate_retrieval(
        queries, FakeModesRetriever(), LengthReranker(), n_boot=50
    )
    modes = result["modes"]
    assert modes["dense"]["mrr"]["estimate"] == 1.0
    assert modes["sparse"]["recall@20"]["estimate"] == 0.0
    assert modes["hybrid"]["mrr"]["estimate"] == 0.25
    assert modes["hybrid+rerank"]["mrr"]["estimate"] == 1.0
    assert (
        modes["hybrid+rerank"]["recall@20"]["estimate"] == modes["hybrid"]["recall@20"]["estimate"]
    )
    paths = re_.write_outputs({**result, "dataset": "toy"}, tmp_path / "res", tmp_path / "fig")
    names = {p.name for p in paths}
    assert {
        "metrics.json",
        "retrieval_metrics.md",
        "retrieval_metrics.csv",
        "retrieval_metrics.png",
        "retrieval_metrics.svg",
    } <= names


async def test_requires_search_api():
    class Old:
        async def hybrid_search(self, query, top_k=20):
            return []

    with pytest.raises(RuntimeError):
        await re_.ranked_lists(Old(), LengthReranker(), "q")
