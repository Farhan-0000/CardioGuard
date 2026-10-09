"""
Incremental-ingestion state (Module M-18, Ingestion Pipeline).

Qdrant is the source of truth for *which document versions* are indexed: every
point carries ``doc_sha256``, so a document whose file hash already has points
is skipped and a document whose hash changed is deleted and re-indexed.

``data/.ingest_state.json`` records *how* the index was built (chunk size,
overlap, tokenizer, ingest version) plus per-document counts. When that
configuration changes, every document is re-chunked and re-indexed even if its
file hash is unchanged.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

# Bump when parsing/chunking logic changes in a way that alters chunks.
INGEST_VERSION = 3


class Action(StrEnum):
    """What incremental ingestion does with one manifest document."""

    INDEX = "index"
    REINDEX = "reindex"
    SKIP = "skip"


class DocumentState(BaseModel):
    """Per-document record of the last successful indexing."""

    sha256: str
    file: str
    chunk_count: int = 0
    table_chunk_count: int = 0
    pages: int = 0
    indexed_at: str = ""


class RunSummary(BaseModel):
    """Summary of the most recent ingestion run."""

    mode: str = ""
    documents_indexed: int = 0
    documents_skipped: int = 0
    chunks_indexed: int = 0
    seconds: float = 0.0
    finished_at: str = ""


class IngestState(BaseModel):
    """Contents of ``data/.ingest_state.json``."""

    config: dict[str, Any] = Field(default_factory=dict)
    collection: str = ""
    documents: dict[str, DocumentState] = Field(default_factory=dict)
    last_run: RunSummary = Field(default_factory=RunSummary)
    last_full_build: RunSummary = Field(default_factory=RunSummary)
    updated_at: str = ""


def chunking_config(chunk_size: int, chunk_overlap: int, tokenizer: str) -> dict[str, Any]:
    """The configuration fingerprint stored with the index."""
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "tokenizer": tokenizer,
        "ingest_version": INGEST_VERSION,
    }


def load_state(path: Path) -> IngestState:
    """Load the state file; a missing or unreadable file yields an empty state."""
    if not path.exists():
        return IngestState()
    try:
        return IngestState.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, ValidationError, OSError):
        return IngestState()


def save_state(state: IngestState, path: Path) -> None:
    """Persist the state file (timestamped)."""
    state.updated_at = now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")


def now_iso() -> str:
    """Current UTC time in ISO-8601 (seconds precision)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def plan_action(
    *,
    points_with_hash: int,
    points_for_document: int,
    config_changed: bool,
    recreate: bool,
) -> Action:
    """Decide what to do with one document during incremental ingestion.

    Args:
        points_with_hash: Indexed points whose ``guideline_id`` and ``doc_sha256``
            both match the current file.
        points_for_document: Indexed points for the ``guideline_id`` (any hash).
        config_changed: Chunking configuration differs from the one in the state file.
        recreate: The collection is being rebuilt from scratch.

    Returns:
        INDEX for new documents, REINDEX when the stored version is stale, SKIP
        when the same file version is already indexed with the same configuration.
    """
    if recreate or points_for_document == 0:
        return Action.INDEX
    if config_changed or points_with_hash != points_for_document:
        return Action.REINDEX
    return Action.SKIP
