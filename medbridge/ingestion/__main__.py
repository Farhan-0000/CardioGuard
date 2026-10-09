"""
Knowledge Ingestion Pipeline CLI (Module M-18, Ingestion Pipeline).

Parses the curated guideline corpus (PDF + HTML), chunks it with the embedder's
tokenizer, computes dense + sparse BM25 vectors in batches and indexes them
into Qdrant -- incrementally by default.

Usage:
    python -m medbridge.ingestion                      # incremental (manifest-driven)
    python -m medbridge.ingestion --recreate           # full rebuild of the collection
    python -m medbridge.ingestion --dry-run            # parse, chunk and validate only
    python -m medbridge.ingestion --only WHO_HTN_PHARM_2021 NICE_NG133
    python -m medbridge.ingestion --stats              # corpus tables + figure for the report
    python -m medbridge.ingestion --pdf-dir ./data/guidelines/ --dry-run   # legacy folder mode

Technical Specification Part V §3.4, §4.3.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import structlog

from medbridge.ingestion.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    SectionAwareChunker,
)
from medbridge.ingestion.indexer import DEFAULT_BATCH_SIZE, QdrantIndexer
from medbridge.ingestion.manifest import CorpusManifest, ManifestError, load_manifest
from medbridge.ingestion.pipeline import (
    IngestReport,
    documents_from_directory,
    ingest_documents,
    update_manifest_file,
)
from medbridge.ingestion.state import chunking_config, load_state
from medbridge.ingestion.stats import build_corpus_stats, write_reports
from medbridge.ingestion.tokenizer import get_token_counter, token_counter_source

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "corpus_manifest.json"
DEFAULT_STATE_FILE = REPO_ROOT / "data" / ".ingest_state.json"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"

logger = structlog.get_logger("medbridge.ingestion")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ingestion CLI."""
    from medbridge.config import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m medbridge.ingestion",
        description="MedBridge clinical guideline ingestion & indexing pipeline.",
    )
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST),
                        help="Corpus manifest (default: data/corpus_manifest.json)")
    parser.add_argument("--pdf-dir", type=str, default=None,
                        help="Legacy mode: ingest every PDF/HTML file in this directory "
                             "(manifest metadata is used for files listed in it)")
    parser.add_argument("--qdrant-url", type=str, default=settings.QDRANT_URL,
                        help=f"Qdrant server URL (default: {settings.QDRANT_URL})")
    parser.add_argument("--collection", type=str, default=settings.QDRANT_COLLECTION,
                        help=f"Target collection (default: {settings.QDRANT_COLLECTION})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Model tokens per chunk (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help=f"Overlap tokens between chunks (default: {DEFAULT_CHUNK_OVERLAP})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Embedding/upsert batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--only", nargs="+", default=None, metavar="GUIDELINE_ID",
                        help="Ingest only these guideline ids")
    parser.add_argument("--recreate", action="store_true", default=False,
                        help="Drop and recreate the collection before indexing (full rebuild)")
    parser.add_argument("--prune", action="store_true", default=False,
                        help="Delete indexed documents that are no longer in the manifest")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Parse, chunk and validate only; do not write to Qdrant")
    parser.add_argument("--stats", action="store_true", default=False,
                        help="Write reports/corpus_stats.{md,json} and the composition figure")
    parser.add_argument("--state-file", type=str, default=str(DEFAULT_STATE_FILE),
                        help="Incremental-ingestion state file (default: data/.ingest_state.json)")
    parser.add_argument("--reports-dir", type=str, default=str(DEFAULT_REPORTS_DIR),
                        help="Output directory for --stats (default: reports/)")
    parser.add_argument("--no-manifest-update", action="store_true", default=False,
                        help="Do not write refreshed hashes/page/chunk counts to the manifest")
    return parser


def _load_optional_manifest(path: Path, required: bool) -> CorpusManifest | None:
    try:
        return load_manifest(path)
    except ManifestError as exc:
        if required:
            logger.error("Cannot load corpus manifest", error=str(exc))
        return None


def run_stats(args: argparse.Namespace) -> int:
    """Write corpus statistics (from Qdrant when reachable, else the manifest)."""
    manifest = _load_optional_manifest(Path(args.manifest), required=True)
    if manifest is None:
        return 1
    qdrant_counts = None
    token_counts: list[int] | None = None
    try:
        indexer = QdrantIndexer(qdrant_url=args.qdrant_url, embedder=_NoEmbedder())
        if indexer.collection_exists(args.collection):
            qdrant_counts = indexer.payload_counts(args.collection)
            count = get_token_counter()
            token_counts = [count(str(p.get("chunk_text", "")))
                            for p in indexer.scroll_payloads(args.collection, ["chunk_text"])]
    except (OSError, RuntimeError) as exc:
        logger.warning("Qdrant unavailable; statistics from the manifest", error=str(exc))
    stats = build_corpus_stats(
        manifest, qdrant_counts=qdrant_counts, token_counts=token_counts,
        state=load_state(Path(args.state_file)), collection=args.collection,
    )
    for path in write_reports(stats, Path(args.reports_dir)):
        print(f"wrote {path}")
    totals = stats["totals"]
    print(f"{totals['documents']} documents, {totals['chunks']} chunks "
          f"({totals['table_chunks']} table chunks), source={stats['source']}")
    return 0


class _NoEmbedder:
    """Placeholder embedder for read-only indexer use (statistics)."""

    def embed_dense_batch(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding is not available in statistics mode")

    def embed_sparse_batch(self, texts: list[str]) -> list[dict[str, object]]:
        raise RuntimeError("embedding is not available in statistics mode")


def _print_report(report: IngestReport) -> None:
    for r in report.results:
        status = f"FAILED: {r.error}" if r.error else str(r.action)
        print(f"  {r.guideline_id:32s} {status:10s} pages={r.pages:4d} chunks={r.chunks:5d} "
              f"tables={r.table_chunks:4d} {r.seconds:6.1f}s")
    stats = report.chunk_stats
    print(f"chunks indexed: {report.chunks_indexed} | duplicates dropped: "
          f"{stats.duplicates_dropped} | reference-list chunks dropped: "
          f"{stats.reference_chunks_dropped} | TOC chunks dropped: {stats.toc_chunks_dropped} "
          f"| reference/TOC paragraphs dropped: {stats.boilerplate_paragraphs_dropped} "
          f"| pruned: {report.pruned or '-'} | {report.seconds:.1f}s")


def run_ingestion(args: argparse.Namespace) -> int:
    """Execute the ingestion workflow given parsed CLI arguments.

    Returns:
        0 on success, 1 on critical error (or when any document failed).
    """
    if args.stats:
        return run_stats(args)

    manifest_path = Path(args.manifest)
    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        if not pdf_dir.is_dir():
            logger.error("Document directory does not exist", path=str(pdf_dir))
            return 1
        manifest = _load_optional_manifest(manifest_path, required=False)
        documents = documents_from_directory(pdf_dir, manifest, REPO_ROOT)
        if not documents:
            logger.warning("No PDF/HTML files found", path=str(pdf_dir))
            return 0
    else:
        manifest = _load_optional_manifest(manifest_path, required=True)
        if manifest is None:
            return 1
        documents = list(manifest.documents)

    if args.only:
        wanted = set(args.only)
        documents = [d for d in documents if d.guideline_id in wanted]
        missing = wanted - {d.guideline_id for d in documents}
        if missing:
            logger.error("Unknown guideline ids", ids=sorted(missing))
            return 1

    chunker = SectionAwareChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    config = chunking_config(args.chunk_size, args.chunk_overlap,
                             token_counter_source() or "unknown")
    indexer = None if args.dry_run else QdrantIndexer(qdrant_url=args.qdrant_url)
    prune_to = {d.guideline_id for d in manifest.documents} if (
        args.prune and manifest is not None and not args.pdf_dir) else None

    report = ingest_documents(
        documents,
        root=REPO_ROOT,
        chunker=chunker,
        indexer=indexer,
        config=config,
        collection=args.collection,
        recreate=args.recreate,
        dry_run=args.dry_run,
        prune_to=prune_to,
        state_path=None if args.dry_run else Path(args.state_file),
        batch_size=args.batch_size,
    )
    _print_report(report)

    if (manifest is not None and not args.dry_run and not args.no_manifest_update
            and not args.pdf_dir):
        update_manifest_file(manifest, manifest_path, config)
    mode = "VALIDATED (DRY-RUN)" if args.dry_run else "INDEXED"
    logger.info("Ingestion completed", documents=len(report.results),
                chunks=sum(r.chunks for r in report.results), mode=mode,
                collection=args.collection, failed=len(report.failed))
    return 1 if report.failed else 0


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()
    sys.exit(run_ingestion(args))


if __name__ == "__main__":
    main()
