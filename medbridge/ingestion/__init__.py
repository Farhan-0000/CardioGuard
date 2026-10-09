"""
Knowledge Ingestion Pipeline (Module M-18).

Manifest-driven, incremental ingestion of the curated guideline corpus
(PDF + HTML) into the hybrid dense/sparse Qdrant collection.
"""
from medbridge.ingestion.chunker import RawChunk, SectionAwareChunker
from medbridge.ingestion.indexer import QdrantIndexer, validate_chunk_payload
from medbridge.ingestion.manifest import CorpusDocument, CorpusManifest, load_manifest
from medbridge.ingestion.parser import (
    ParsedDocument,
    ParsedPage,
    extract_pages_from_pdf,
    parse_document,
)

__all__ = [
    "CorpusDocument",
    "CorpusManifest",
    "ParsedDocument",
    "ParsedPage",
    "QdrantIndexer",
    "RawChunk",
    "SectionAwareChunker",
    "extract_pages_from_pdf",
    "load_manifest",
    "parse_document",
    "validate_chunk_payload",
]
