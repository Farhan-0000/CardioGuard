"""
Corpus Ingestion Pipeline (Module M-18, Ingestion Pipeline).

Manifest-driven, incremental ingestion: for every curated document the file
hash is compared with the ``doc_sha256`` already stored in Qdrant; unchanged
documents are skipped, changed documents are deleted and re-indexed, new ones
are indexed. ``recreate=True`` rebuilds the collection from scratch.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from medbridge.ingestion.chunker import ChunkingStats, DocumentMetadata, SectionAwareChunker
from medbridge.ingestion.indexer import DEFAULT_BATCH_SIZE, DEFAULT_COLLECTION, QdrantIndexer
from medbridge.ingestion.manifest import (
    CorpusDocument,
    CorpusManifest,
    save_manifest,
    sha256_file,
)
from medbridge.ingestion.parser import PDFParseError, deduce_guideline_id, parse_document
from medbridge.ingestion.state import (
    Action,
    DocumentState,
    IngestState,
    RunSummary,
    load_state,
    now_iso,
    plan_action,
    save_state,
)

logger = structlog.get_logger(__name__)

SUPPORTED_SUFFIXES = (".pdf", ".html", ".htm")


@dataclass
class DocumentResult:
    """Outcome of ingesting one document."""

    guideline_id: str
    action: Action
    pages: int = 0
    chunks: int = 0
    table_chunks: int = 0
    seconds: float = 0.0
    error: str = ""


@dataclass
class IngestReport:
    """Outcome of one ingestion run."""

    results: list[DocumentResult] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    seconds: float = 0.0
    chunk_stats: ChunkingStats = field(default_factory=ChunkingStats)

    @property
    def failed(self) -> list[DocumentResult]:
        return [r for r in self.results if r.error]

    @property
    def chunks_indexed(self) -> int:
        return sum(r.chunks for r in self.results if r.action != Action.SKIP and not r.error)


def adhoc_document(path: Path, root: Path) -> CorpusDocument:
    """Minimal manifest entry for a file that is not in the manifest (legacy mode)."""
    suffix = path.suffix.lower()
    try:
        relative = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        relative = str(path.resolve())
    return CorpusDocument.model_construct(
        guideline_id=deduce_guideline_id(path.name), title=path.stem, short_title=path.stem,
        org="", year=0, doc_type="guideline", url="", download_url="", licence="unknown",
        file=relative, sha256="", pages=0, chunk_count=0, table_chunk_count=0, topics=[],
        language="en", source_format="pdf" if suffix == ".pdf" else "html", html_root="",
        volatile=False, doi="", notes="",
    )


def documents_from_directory(
    directory: Path, manifest: CorpusManifest | None, root: Path
) -> list[CorpusDocument]:
    """Legacy ``--pdf-dir`` mode: every PDF/HTML file in ``directory``.

    Files listed in the manifest keep their curated metadata; others get an
    identifier deduced from the filename.
    """
    known = manifest.by_filename() if manifest else {}
    docs: list[CorpusDocument] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            docs.append(known.get(path.name) or adhoc_document(path, root))
    return docs


def _metadata(doc: CorpusDocument, sha: str) -> DocumentMetadata:
    return DocumentMetadata(
        doc_title=doc.title, org=doc.org, year=doc.year, topics=tuple(doc.topics), doc_sha256=sha
    )


def ingest_documents(
    documents: list[CorpusDocument],
    *,
    root: Path,
    chunker: SectionAwareChunker,
    indexer: QdrantIndexer | None,
    config: dict[str, object],
    collection: str = DEFAULT_COLLECTION,
    recreate: bool = False,
    dry_run: bool = False,
    prune_to: set[str] | None = None,
    state_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> IngestReport:
    """Parse, chunk and (incrementally) index documents.

    Args:
        documents: Documents to ingest (manifest entries).
        root: Repository root that manifest ``file`` paths are relative to.
        chunker: Configured chunker.
        indexer: Qdrant indexer (may be ``None`` for a dry run).
        config: Chunking configuration fingerprint (see ``state.chunking_config``).
        collection: Target collection.
        recreate: Drop and rebuild the collection.
        dry_run: Parse/chunk/validate only.
        prune_to: When given, delete indexed guideline ids not in this set.
        state_path: Incremental state file (``None`` disables state tracking).
        batch_size: Embedding/upsert batch size.

    Documents are updated in place with ``sha256``, ``pages``, ``chunk_count``
    and ``table_chunk_count`` after successful indexing.
    """
    started = time.perf_counter()
    report = IngestReport(chunk_stats=chunker.stats)
    state = load_state(state_path) if state_path else IngestState()
    config_changed = bool(state.config) and state.config != config
    if config_changed and not recreate:
        logger.info("Chunking configuration changed; documents will be re-indexed",
                    old=state.config, new=config)
    live = indexer is not None and not dry_run
    if live and indexer is not None:
        indexer.ensure_collection(collection_name=collection, recreate=recreate)
    if recreate:
        state.documents = {}
    seen_hashes: set[str] = set()

    for doc in documents:
        t0 = time.perf_counter()
        gid = doc.guideline_id
        path = doc.resolve_path(root)
        if not path.exists():
            report.results.append(DocumentResult(gid, Action.INDEX, error=f"missing file {path}"))
            logger.error("Document file missing", guideline_id=gid, file=str(path))
            continue
        sha = sha256_file(path)
        if doc.sha256 and sha != doc.sha256:
            logger.warning("File hash differs from manifest", guideline_id=gid,
                           volatile=doc.volatile)

        action = Action.INDEX
        if live and indexer is not None:
            action = plan_action(
                points_with_hash=indexer.count(collection, guideline_id=gid, doc_sha256=sha),
                points_for_document=indexer.count(collection, guideline_id=gid),
                config_changed=config_changed,
                recreate=recreate,
            )
        if action == Action.SKIP:
            previous = state.documents.get(gid)
            report.results.append(DocumentResult(
                gid, Action.SKIP, pages=previous.pages if previous else doc.pages,
                chunks=previous.chunk_count if previous else doc.chunk_count,
                table_chunks=previous.table_chunk_count if previous else doc.table_chunk_count,
            ))
            logger.info("Skipping unchanged document", guideline_id=gid)
            continue

        try:
            parsed = parse_document(path, gid, doc.url, html_root=doc.html_root)
        except PDFParseError as exc:
            report.results.append(DocumentResult(gid, action, error=str(exc)))
            logger.error("Failed to parse document", guideline_id=gid, error=str(exc))
            continue
        chunks = chunker.chunk_document(parsed, _metadata(doc, sha), seen_hashes=seen_hashes)
        tables = sum(1 for c in chunks if c.content_type == "table")

        if live and indexer is not None:
            if action == Action.REINDEX:
                indexer.delete_document(gid, collection_name=collection)
            indexed = indexer.index_chunks(chunks, collection_name=collection,
                                           batch_size=batch_size)
        else:
            indexed = indexer.index_chunks(chunks, dry_run=True) if indexer else len(chunks)

        report.results.append(DocumentResult(
            gid, action, pages=parsed.total_pages, chunks=indexed, table_chunks=tables,
            seconds=round(time.perf_counter() - t0, 2),
        ))
        if not dry_run:
            doc.sha256 = sha
            doc.pages = parsed.total_pages
            doc.chunk_count = indexed
            doc.table_chunk_count = tables
            state.documents[gid] = DocumentState(
                sha256=sha, file=doc.file, chunk_count=indexed, table_chunk_count=tables,
                pages=parsed.total_pages, indexed_at=now_iso(),
            )
        logger.info("Ingested document", guideline_id=gid, action=str(action),
                    pages=parsed.total_pages, chunks=indexed, tables=tables,
                    seconds=round(time.perf_counter() - t0, 2))

    if live and indexer is not None and prune_to is not None:
        for gid in sorted(set(indexer.payload_counts(collection)) - prune_to):
            indexer.delete_document(gid, collection_name=collection)
            state.documents.pop(gid, None)
            report.pruned.append(gid)

    report.seconds = round(time.perf_counter() - started, 2)
    if not dry_run and state_path:
        state.config = config
        state.collection = collection
        state.last_run = RunSummary(
            mode="recreate" if recreate else "incremental",
            documents_indexed=sum(1 for r in report.results
                                  if r.action != Action.SKIP and not r.error),
            documents_skipped=sum(1 for r in report.results if r.action == Action.SKIP),
            chunks_indexed=report.chunks_indexed,
            seconds=report.seconds,
            finished_at=now_iso(),
        )
        if recreate:
            state.last_full_build = state.last_run
        save_state(state, state_path)
    return report


def update_manifest_file(manifest: CorpusManifest, path: Path, config: dict[str, object]) -> None:
    """Write refreshed per-document counts and the chunking config to the manifest."""
    manifest.chunking = {k: v for k, v in config.items() if isinstance(v, int | str)}
    save_manifest(manifest, path)
