# Ingestion pipeline (`medbridge.ingestion`)

Offline CLI that turns the curated corpus in `data/corpus_manifest.json` into
the hybrid dense + sparse Qdrant collection `clinical_guidelines`. The
ingestion module is the only writer of the collection; retrieval is read-only.

```
manifest ─▶ parse (PDF | HTML) ─▶ chunk (400 tok, 60 overlap) ─▶ dedupe/filter ─▶
batch embed (bge-small dense + BM25 sparse) ─▶ upsert (payload §3.5) ─▶ state + manifest counts
```

## Commands

```bash
python -m medbridge.ingestion                     # incremental: skip unchanged, re-index changed
python -m medbridge.ingestion --recreate          # full rebuild (drop + create collection)
python -m medbridge.ingestion --dry-run           # parse + chunk + validate only
python -m medbridge.ingestion --only NICE_NG133 WHO_HTN_PHARM_2021
python -m medbridge.ingestion --prune             # delete indexed docs no longer in the manifest
python -m medbridge.ingestion --stats             # reports/corpus_stats.{md,json} + figure
python -m medbridge.ingestion --pdf-dir data/guidelines --dry-run   # legacy folder mode
```

Options: `--manifest`, `--qdrant-url`, `--collection`, `--chunk-size`,
`--chunk-overlap`, `--batch-size`, `--state-file` (default
`data/.ingest_state.json`), `--reports-dir`, `--no-manifest-update`.

## Modules

| Module | Responsibility |
|---|---|
| `manifest.py` | Pydantic schema + validation of the curated manifest (ids, topics, licences, https URLs, SHA-256). |
| `parser.py` | PDF parsing with PyMuPDF: running header/footer removal (margin lines repeated on > 30 % of pages, rotated margin text, page numbers), references/bibliography tail cut, font-based heading detection, `find_tables()` tables kept as structured rows. `parse_document()` dispatches PDF/HTML. |
| `html_parser.py` | Standard-library `html.parser` extraction of the main content (`#id`/`.class`/tag root from the manifest), dropping nav/header/footer/scripts/modals/link lists/reference sections and citation superscripts; `h1`–`h3` sections become *virtual pages*. |
| `tokenizer.py` | Token counting with the embedder's own WordPiece tokenizer (`BAAI/bge-small-en-v1.5`, identical vocabulary to the `ms-marco-MiniLM-L-6-v2` reranker). |
| `chunker.py` | Section-, page- and table-aware packing under a model-token budget; overlap only within a section; generic headings ("Synopsis") qualified with their parent; table chunks with repeated header rows; bibliography/TOC filtering; normalised-text de-duplication. |
| `indexer.py` | Collection provisioning (384-d COSINE dense + IDF sparse, keyword payload indexes), batch embedding, upsert, per-document delete and counts. |
| `state.py` | Incremental plan (`INDEX` / `REINDEX` / `SKIP`) from Qdrant `doc_sha256` counts plus the chunking-config fingerprint in `data/.ingest_state.json`. |
| `pipeline.py` | Orchestrates one run, updates manifest counts and the state file. |
| `stats.py` | Corpus tables (markdown/JSON) and the composition figure (PNG 300 dpi + SVG). |

## Payload (interface contract §3.5)

`chunk_id, chunk_text, guideline_id, doc_title, org, year, section_title,
page_number, page_end, source_url, topics, doc_sha256, chunk_index,
content_type ("text" | "table")`, plus the additive `source_format`
(`"pdf"` | `"html"`).

`page_number`/`page_end` are 1-based physical PDF pages. HTML has no pages:
for HTML sources (`source_format == "html"`) they are the index of the
`h1`–`h3` section (virtual page), so a UI should cite "section N" rather than
"p. N" for them.

## Design notes

* **Chunk size 400 / overlap 60, measured in model tokens.** Both the dense
  embedder and the cross-encoder truncate at 512 word pieces. The previous
  index used 512 *whitespace words*; measured with the real tokenizer, 39 % of
  its 905 chunks exceeded 512 tokens and 33 % of all indexed tokens were never
  seen by the models. 400 tokens leaves room for the query in the reranker.
* **Incremental by hash.** A document is skipped when Qdrant already holds its
  points with the current file's `doc_sha256` and the chunking configuration
  is unchanged; a changed file is deleted and re-indexed.
* **De-duplication** uses a SHA-1 of the lower-cased alphanumeric text, shared
  across the documents processed in one run.
