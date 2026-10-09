"""Evidence fetching / reranking adapters and run-manifest helpers (all mocked)."""

from types import SimpleNamespace

import httpx

from evaluation import manifest as mf
from evaluation.evidence import (
    CorpusRetriever,
    EvidenceChunk,
    EvidenceStore,
    evidence_from_explanation,
    point_id_for,
    rerank_chunks,
)
from medbridge.retrieval.hybrid_retriever import RetrievedChunk


class FakeQdrant:
    def __init__(self, payloads):
        self.by_id = {point_id_for(p["chunk_id"]): p for p in payloads if not p.get("odd_id")}
        self.odd = [p for p in payloads if p.get("odd_id")]
        self.retrieve_calls = 0
        self.scroll_calls = 0

    async def retrieve(self, collection_name, ids, with_payload, with_vectors):
        self.retrieve_calls += 1
        return [SimpleNamespace(payload=self.by_id[i]) for i in ids if i in self.by_id]

    async def scroll(self, collection_name, scroll_filter, limit, with_payload, with_vectors):
        self.scroll_calls += 1
        wanted = set(scroll_filter.must[0].match.any)
        return [SimpleNamespace(payload=p) for p in self.odd if p["chunk_id"] in wanted], None

    async def close(self):
        pass


async def test_evidence_store_fetches_by_point_id_and_filter():
    client = FakeQdrant(
        [
            {"chunk_id": "a", "chunk_text": "A text", "page_number": 3},
            {"chunk_id": "b", "chunk_text": "B text", "odd_id": True},
        ]
    )
    store = EvidenceStore(client=client, collection="test")
    got = await store.get_many(["a", "b", "zz", "a"])
    assert set(got) == {"a", "b"} and got["b"]["chunk_text"] == "B text"
    await store.get_many(["a"])  # cached → no new request
    assert client.retrieve_calls == 1 and client.scroll_calls == 1


def test_evidence_from_explanation_prefers_full_text():
    explanation = {
        "retrieval": [
            {
                "rank": 1,
                "chunk_id": "a",
                "guideline_id": "ESC",
                "page": 3,
                "score_rerank": 2.0,
                "excerpt": "A…",
            },
            {"rank": 2, "chunk_id": "b", "excerpt": "B…"},
        ]
    }
    out = evidence_from_explanation(
        explanation, {"a": {"chunk_text": "A full", "page_end": 4, "doc_title": "T"}}
    )
    assert (
        out[0]["text"] == "A full" and out[0]["text_source"] == "qdrant" and out[0]["page_end"] == 4
    )
    assert out[0]["title"] == "T"
    assert out[1]["text"] == "B…" and out[1]["text_source"] == "excerpt"


class FakeReranker:
    async def rerank(self, query, chunks, top_k=None):
        scored = sorted(((c, float(len(c))) for c in chunks), key=lambda t: -t[1])
        return scored[:top_k]


class FakeHybrid:
    async def hybrid_search(self, query, top_k=20):
        return [
            RetrievedChunk(
                chunk_id=f"c{i}",
                chunk_text="x" * (i + 1),
                guideline_id="G",
                section_title="s",
                page_number=i,
            )
            for i in range(5)
        ]


async def test_rerank_chunks_maps_texts_back():
    chunks = await FakeHybrid().hybrid_search("q")
    ranked = await rerank_chunks(FakeReranker(), "q", chunks, 3)
    assert [c.chunk_id for c, _ in ranked] == ["c4", "c3", "c2"]
    assert ranked[0][1] == 5.0
    assert await rerank_chunks(FakeReranker(), "q", [], 3) == []


async def test_rerank_chunks_prefers_production_method():
    class ChunkReranker(FakeReranker):
        async def rerank_chunks(self, query, chunks, top_k=None):
            ranked = sorted(chunks, key=lambda c: c.page_number)[:top_k]
            return [c.model_copy(update={"score_rerank": float(c.page_number)}) for c in ranked]

    chunks = await FakeHybrid().hybrid_search("q")
    ranked = await rerank_chunks(ChunkReranker(), "q", chunks, 2)
    assert [(c.chunk_id, s) for c, s in ranked] == [("c0", 0.0), ("c1", 1.0)]


async def test_corpus_retriever_search_and_score():
    retriever = CorpusRetriever(retriever=FakeHybrid(), reranker=FakeReranker())
    top = await retriever.search("claim", top_k=2)
    assert [c.chunk_id for c in top] == ["c4", "c3"] and all(c.source == "judge" for c in top)
    scored = await retriever.score(
        "claim", [EvidenceChunk(chunk_id="s", text="ab"), EvidenceChunk(chunk_id="l", text="abcd")]
    )
    assert [c.chunk_id for c, _ in scored] == ["l", "s"]


async def test_corpus_retriever_does_not_memoise_an_outage():
    """ablation-study-validity-2 (judge side): an empty result is retried, not replayed."""

    class Flaky(FakeHybrid):
        def __init__(self):
            self.calls = 0

        async def hybrid_search(self, query, top_k=20):
            self.calls += 1
            return [] if self.calls == 1 else await super().hybrid_search(query, top_k)

    hybrid = Flaky()
    retriever = CorpusRetriever(retriever=hybrid, reranker=FakeReranker())
    assert await retriever.search("claim", top_k=2) == []
    assert [c.chunk_id for c in await retriever.search("claim", top_k=2)] == ["c4", "c3"]
    await retriever.search("claim", top_k=2)
    assert hybrid.calls == 2  # the non-empty result is memoised


def test_ollama_models_matches_latest_tags(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "models": [
                    {"name": "llama3:latest", "digest": "d1", "details": {}},
                    {"name": "qwen3:8b", "digest": "d2", "details": {}},
                ]
            }

    monkeypatch.setattr(mf.httpx, "get", lambda *a, **k: Response())
    info = mf.ollama_models("http://127.0.0.1:11434/v1", ["llama3", "qwen3:8b"])
    assert set(info) == {"llama3:latest", "qwen3:8b"} and info["llama3:latest"]["digest"] == "d1"


def test_evidence_chunk_roundtrip():
    chunk = EvidenceChunk(chunk_id="x", text="t", page=2)
    assert EvidenceChunk.from_dict({**chunk.to_dict(), "unknown": 1}) == chunk


def test_redact_secrets():
    out = mf.redact_secrets(
        {
            "DATABASE_URL": "postgresql+asyncpg://user:s3cret@127.0.0.1:5433/db",
            "ENCRYPTION_MASTER_KEY": "k",
            "SESSION_TOKEN_SECRET": "s",
            "ADMIN_PASSWORD": "p",
            "LLM_MODEL": "qwen3:8b",
        }
    )
    assert out == {
        "DATABASE_URL": "postgresql+asyncpg://user:***@127.0.0.1:5433/db",
        "LLM_MODEL": "qwen3:8b",
    }


def test_mask_url_credentials():
    cmd = (
        "python -m evaluation.run --database-url postgresql+asyncpg://u:p4ss@h:5433/medbridge_eval"
    )
    assert mf.mask_url_credentials(cmd).endswith("postgresql+asyncpg://u:***@h:5433/medbridge_eval")


def test_hardware_and_packages():
    hw = mf.hardware_info()
    assert {"platform", "machine", "cpu", "cpu_count", "ram_gb", "python"} <= set(hw)
    assert (
        mf.package_versions(("numpy", "definitely-not-installed"))["definitely-not-installed"]
        is None
    )


def test_ollama_models(monkeypatch):
    def fake_get(url, timeout):
        assert url == "http://h:11434/api/tags"
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={
                "models": [
                    {
                        "name": "gemma3:12b",
                        "digest": "d1",
                        "size": 8,
                        "details": {"parameter_size": "11.8B", "quantization_level": "Q4_K_M"},
                    },
                    {"name": "other", "digest": "d2"},
                ]
            },
        )

    monkeypatch.setattr(mf.httpx, "get", fake_get)
    out = mf.ollama_models("http://h:11434/v1", ["gemma3:12b"])
    assert out == {
        "gemma3:12b": {
            "digest": "d1",
            "size_bytes": 8,
            "family": None,
            "parameter_size": "11.8B",
            "quantization_level": "Q4_K_M",
            "modified_at": None,
        }
    }

    def down(url, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(mf.httpx, "get", down)
    assert "error" in mf.ollama_models("http://h:11434")
