# Retrieval (`medbridge.retrieval`)

Read-only hybrid search over the Qdrant collection written by
`medbridge.ingestion` (default `clinical_guidelines`).

```
query ─▶ dense (BAAI/bge-small-en-v1.5) ‖ sparse (Qdrant/bm25)      [worker threads, LRU-cached]
      ─▶ Qdrant prefetch + RrfQuery(k = 60) → top-20                 [result cache: LRU + TTL]
      ─▶ cross-encoder/ms-marco-MiniLM-L-6-v2 → top-5                [asyncio.to_thread]
```

## Modules

| Module | Responsibility |
|---|---|
| `hybrid_retriever.py` | `HybridRetriever.hybrid_search(query, top_k)` (production path), `search(query, top_k, mode="hybrid" \| "dense" \| "sparse")` for the retrieval evaluation, `collection_points()`, `warmup()`, `clear_cache()`, `close()`. `RetrievedChunk` model and `chunk_from_payload()`. Fusion uses an explicit `Rrf(k=60)` (Qdrant's un-parameterised RRF uses a different constant). Any failure returns `[]`; the Evidence Gate then decides. |
| `reranker.py` | `CrossEncoderReranker.rerank_chunks(query, chunks, top_k)` keeps the RRF score and adds `score_rerank`; inference always runs in a worker thread, serialised by a lock; device CUDA > MPS > CPU. |
| `embedder.py` | `EmbeddingClient`: lazy, thread-safe dense + sparse encoders; `embed_query()` runs both concurrently and memoises results; `warmup()` loads both models. |
| `cache.py` | Thread-safe bounded LRU/TTL cache (`RETRIEVAL_CACHE_SIZE`, `RETRIEVAL_CACHE_TTL_SECONDS`; size ≤ 0 disables). |

## Payload → `RetrievedChunk`

Contract keys (§3.5): `chunk_id, chunk_text, guideline_id, doc_title, org,
year, section_title, page_number, page_end, source_url, topics, doc_sha256,
chunk_index, content_type`, plus `source_format` (`"pdf"` | `"html"`). Pre-v4
key names (`text`, `doc_id`, `section`, `page`, `url`, `title`) are still
accepted.

`RetrievedChunk` fields: `chunk_id, chunk_text, guideline_id, section_title,
page_number, source_url, score` and the optional `doc_title, org, year,
page_end, topics, score_rrf, score_rerank, source_format`.

For HTML sources `page_number` is the index of the `h1`–`h3` section (a
virtual page), not a physical page. Consumers therefore never show it as a
page: `core.citations` sets the citation `page` to `None` and the section to
`"<section title> (section N)"`, the orchestrator's evidence panel does the
same, and the evidence prompt omits the `page` attribute.

## Tests

`medbridge/tests/unit/test_hybrid_retriever.py`, `test_reranker.py`,
`test_embedder.py`; `medbridge/tests/integration/test_hybrid_retriever_qdrant.py`
runs against the live collection (auto-skipped when Qdrant is down).
