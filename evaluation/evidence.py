"""
Evidence access for the evaluation harness.

* ``EvidenceStore`` — full chunk text by ``chunk_id`` from the Qdrant payload
  (the explanation only carries truncated excerpts, the judges need the whole
  chunk that Gate 2 and the generator saw).
* ``CorpusRetriever`` — judge-side retrieval, independent of the condition
  under test: hybrid search (dense ‖ sparse → RRF) → cross-encoder rerank →
  top-k, using the production ``medbridge.retrieval`` components.
* ``EvidenceChunk`` — the serialisable evidence unit stored in run records and
  judgment side files.
"""

from __future__ import annotations

import uuid
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient

    from medbridge.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
    from medbridge.retrieval.reranker import CrossEncoderReranker

_T = TypeVar("_T")


@dataclass
class EvidenceChunk:
    """One evidence passage with provenance."""

    chunk_id: str
    text: str
    guideline_id: str = ""
    title: str = ""
    section: str = ""
    page: int | None = None
    page_end: int | None = None
    score: float | None = None
    source: str = (
        "condition"  # "condition" (retrieved by the pipeline) | "judge" (judge-side retrieval)
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceChunk:
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in fields})

    @classmethod
    def from_retrieved(cls, chunk: Any, score: float | None, source: str) -> EvidenceChunk:
        """Build from a ``RetrievedChunk`` (tolerates older models without optional fields)."""
        return cls(
            chunk_id=str(chunk.chunk_id),
            text=str(chunk.chunk_text),
            guideline_id=str(getattr(chunk, "guideline_id", "") or ""),
            title=str(getattr(chunk, "doc_title", "") or ""),
            section=str(getattr(chunk, "section_title", "") or ""),
            page=getattr(chunk, "page_number", None),
            page_end=getattr(chunk, "page_end", None),
            score=score,
            source=source,
        )


def point_id_for(chunk_id: str) -> str:
    """Qdrant point id convention of the indexer: uuid5(NAMESPACE_DNS, chunk_id)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class EvidenceStore:
    """Fetch full chunk payloads from Qdrant by ``chunk_id`` (read-only, cached in memory)."""

    def __init__(
        self, client: AsyncQdrantClient | None = None, collection: str | None = None
    ) -> None:
        from medbridge.config import get_settings

        settings = get_settings()
        if client is None:
            from qdrant_client import AsyncQdrantClient

            client = AsyncQdrantClient(url=settings.QDRANT_URL)
        self._client = client
        self._collection = collection or settings.QDRANT_COLLECTION
        self._cache: dict[str, dict[str, Any]] = {}

    async def get_many(self, chunk_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """Return ``{chunk_id: payload}`` for the ids found (missing ids are omitted)."""
        from qdrant_client import models

        wanted = list(dict.fromkeys(str(c) for c in chunk_ids))
        missing = [c for c in wanted if c not in self._cache]
        if missing:
            points = await self._client.retrieve(
                collection_name=self._collection,
                ids=[point_id_for(c) for c in missing],
                with_payload=True,
                with_vectors=False,
            )
            self._remember(p.payload for p in points)
            still = [c for c in missing if c not in self._cache]
            if still:
                flt = models.Filter(
                    must=[models.FieldCondition(key="chunk_id", match=models.MatchAny(any=still))]
                )
                scrolled, _ = await self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=flt,
                    limit=max(len(still) * 2, 10),
                    with_payload=True,
                    with_vectors=False,
                )
                self._remember(p.payload for p in scrolled)
        return {c: self._cache[c] for c in wanted if c in self._cache}

    def _remember(self, payloads: Iterable[dict[str, Any] | None]) -> None:
        for payload in payloads:
            if payload and payload.get("chunk_id"):
                self._cache[str(payload["chunk_id"])] = dict(payload)

    async def close(self) -> None:
        await self._client.close()


def evidence_from_explanation(
    explanation: dict[str, Any], payloads: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Evidence passed to Gate 2 / the generator, with full text when the payload is known.

    Falls back to the explanation excerpt (``text_source = "excerpt"``) when
    the chunk could not be fetched.
    """
    out: list[dict[str, Any]] = []
    for item in explanation.get("retrieval") or []:
        chunk_id = str(item.get("chunk_id", ""))
        payload = payloads.get(chunk_id)
        text = str(payload.get("chunk_text", "")) if payload else ""
        out.append(
            {
                "rank": item.get("rank"),
                "chunk_id": chunk_id,
                "guideline_id": item.get("guideline_id") or (payload or {}).get("guideline_id", ""),
                "title": item.get("title") or (payload or {}).get("doc_title", ""),
                "section": item.get("section") or (payload or {}).get("section_title", ""),
                "page": item.get("page")
                if item.get("page") is not None
                else (payload or {}).get("page_number"),
                "page_end": (payload or {}).get("page_end"),
                "score_rrf": item.get("score_rrf"),
                "score_rerank": item.get("score_rerank"),
                "cited": bool(item.get("cited", False)),
                "text": text or str(item.get("excerpt", "")),
                "text_source": "qdrant" if text else "excerpt",
            }
        )
    return out


def _map_ranked(
    ranked: Sequence[Any], items: Sequence[_T], text_of: Callable[[_T], str]
) -> list[tuple[_T, float]]:
    """Map reranker output back to the input items.

    ``CrossEncoderReranker.rerank`` returns ``(text, score)`` pairs; result
    objects exposing ``chunk_text`` (and ``score_rerank``/``score``) are accepted
    too. Duplicate texts are consumed in order.
    """
    by_text: dict[str, deque[_T]] = defaultdict(deque)
    for item in items:
        by_text[text_of(item)].append(item)
    out: list[tuple[_T, float]] = []
    for entry in ranked:
        if hasattr(entry, "chunk_text"):
            text = str(entry.chunk_text)
            score = getattr(entry, "score_rerank", None)
            value = float(score if score is not None else getattr(entry, "score", 0.0))
        else:
            text, value = str(entry[0]), float(entry[1])
        if by_text[text]:
            out.append((by_text[text].popleft(), value))
    return out


async def rerank_chunks(
    reranker: CrossEncoderReranker, query: str, chunks: Sequence[RetrievedChunk], top_k: int
) -> list[tuple[RetrievedChunk, float]]:
    """Rerank retrieved chunks with the production cross-encoder (descending score).

    Uses ``CrossEncoderReranker.rerank_chunks`` (the pipeline's own path) when
    the reranker provides it, else the text-level ``rerank`` API.
    """
    if not chunks:
        return []
    by_chunk = getattr(reranker, "rerank_chunks", None)
    if by_chunk is not None:
        reranked = await by_chunk(query, list(chunks), top_k=top_k)
        return [
            (c, float(c.score_rerank if c.score_rerank is not None else c.score)) for c in reranked
        ]
    ranked = await reranker.rerank(query, [c.chunk_text for c in chunks], top_k=top_k)
    return _map_ranked(ranked, chunks, lambda c: c.chunk_text)


class CorpusRetriever:
    """Judge-side retrieval: hybrid top-``candidates`` → cross-encoder → top-k (cached per query)."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        candidates: int = 20,
    ) -> None:
        if retriever is None:
            from medbridge.retrieval.hybrid_retriever import HybridRetriever

            retriever = HybridRetriever()
        if reranker is None:
            from medbridge.retrieval.reranker import get_reranker

            reranker = get_reranker()
        self._retriever = retriever
        self._reranker = reranker
        self._candidates = candidates
        self._cache: dict[tuple[str, int], list[EvidenceChunk]] = {}

    async def search(self, query: str, top_k: int = 3) -> list[EvidenceChunk]:
        """Top-``top_k`` reranked corpus chunks for ``query``."""
        key = (query, top_k)
        if key not in self._cache:
            chunks = await self._retriever.hybrid_search(query, top_k=self._candidates)
            ranked = await rerank_chunks(self._reranker, query, chunks, top_k)
            self._cache[key] = [EvidenceChunk.from_retrieved(c, s, "judge") for c, s in ranked]
        return list(self._cache[key])

    async def score(
        self, query: str, chunks: Sequence[EvidenceChunk]
    ) -> list[tuple[EvidenceChunk, float]]:
        """Cross-encoder scores of ``chunks`` against ``query`` (descending)."""
        if not chunks:
            return []
        ranked = await self._reranker.rerank(query, [c.text for c in chunks], top_k=len(chunks))
        return _map_ranked(ranked, chunks, lambda c: c.text)
