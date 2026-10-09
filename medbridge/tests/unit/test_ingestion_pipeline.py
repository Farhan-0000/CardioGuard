"""
Unit tests for incremental ingestion (``pipeline.py`` + ``state.py``).

An in-memory fake indexer stands in for Qdrant so that the skip / re-index /
recreate / prune decisions can be verified hermetically.
"""
from collections import Counter
from pathlib import Path

import pytest

from medbridge.ingestion.chunker import RawChunk, SectionAwareChunker
from medbridge.ingestion.manifest import CorpusDocument, CorpusManifest, load_manifest
from medbridge.ingestion.pipeline import (
    documents_from_directory,
    ingest_documents,
    update_manifest_file,
)
from medbridge.ingestion.state import (
    Action,
    IngestState,
    chunking_config,
    load_state,
    plan_action,
    save_state,
)
from medbridge.ingestion.tokenizer import whitespace_token_count

CONFIG = chunking_config(400, 60, "whitespace")


class FakeIndexer:
    """Minimal in-memory stand-in for :class:`QdrantIndexer`."""

    def __init__(self) -> None:
        self.points: dict[str, dict] = {}
        self.deleted: list[str] = []
        self.recreated = 0

    def ensure_collection(self, collection_name: str = "", recreate: bool = False) -> None:
        if recreate:
            self.points.clear()
            self.recreated += 1

    def count(self, collection_name: str = "", **conditions: str) -> int:
        return sum(
            1 for p in self.points.values()
            if all(p.get(k) == v for k, v in conditions.items())
        )

    def delete_document(self, guideline_id: str, collection_name: str = "") -> None:
        self.deleted.append(guideline_id)
        self.points = {k: p for k, p in self.points.items() if p["guideline_id"] != guideline_id}

    def index_chunks(self, chunks: list[RawChunk], collection_name: str = "",
                     batch_size: int = 64, recreate: bool = False, dry_run: bool = False) -> int:
        if not dry_run:
            for chunk in chunks:
                self.points[chunk.chunk_id] = chunk.to_payload()
        return len(chunks)

    def payload_counts(self, collection_name: str = "") -> dict[str, Counter[str]]:
        counts: dict[str, Counter[str]] = {}
        for p in self.points.values():
            counts.setdefault(p["guideline_id"], Counter())[p["content_type"]] += 1
        return counts


def _write_page(path: Path, heading: str, body: str) -> None:
    path.write_text(f"<html><body><main><h1>{heading}</h1><p>{body}</p></main></body></html>",
                    encoding="utf-8")


def _doc(gid: str, file: str) -> CorpusDocument:
    return CorpusDocument(
        guideline_id=gid, title=f"{gid} title", short_title=gid, org="NLM MedlinePlus",
        year=2025, doc_type="patient_info", url=f"https://example.gov/{gid.lower()}",
        licence="Public domain (US Government)", file=file, topics=["hypertension"],
        source_format="html",
    )


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, list[CorpusDocument]]:
    (tmp_path / "docs").mkdir()
    _write_page(tmp_path / "docs" / "a.html", "High blood pressure",
                "High blood pressure means the force of blood on artery walls is too high.")
    _write_page(tmp_path / "docs" / "b.html", "Heart failure",
                "Heart failure means the heart cannot pump enough blood for the body.")
    return tmp_path, [_doc("DOC_A", "docs/a.html"), _doc("DOC_B", "docs/b.html")]


def _run(root: Path, docs: list[CorpusDocument], indexer: FakeIndexer, state: Path,
         **kwargs) -> dict[str, Action]:
    report = ingest_documents(
        docs, root=root, indexer=indexer,  # type: ignore[arg-type]
        chunker=SectionAwareChunker(token_counter=whitespace_token_count),
        config=kwargs.pop("config", CONFIG), state_path=state, **kwargs,
    )
    assert not report.failed
    return {r.guideline_id: r.action for r in report.results}


@pytest.mark.parametrize(
    ("with_hash", "for_doc", "config_changed", "recreate", "expected"),
    [
        (0, 0, False, False, Action.INDEX),     # new document
        (5, 5, False, False, Action.SKIP),      # same file, same config
        (0, 5, False, False, Action.REINDEX),   # file hash changed
        (3, 5, False, False, Action.REINDEX),   # mixed versions in the index
        (5, 5, True, False, Action.REINDEX),    # chunking configuration changed
        (5, 5, False, True, Action.INDEX),      # full rebuild
    ],
)
def test_plan_action(with_hash: int, for_doc: int, config_changed: bool, recreate: bool,
                     expected: Action) -> None:
    assert plan_action(points_with_hash=with_hash, points_for_document=for_doc,
                       config_changed=config_changed, recreate=recreate) is expected


def test_incremental_skip_reindex_and_recreate(corpus, tmp_path: Path) -> None:
    root, docs = corpus
    state = tmp_path / "state.json"
    indexer = FakeIndexer()

    assert _run(root, docs, indexer, state) == {"DOC_A": Action.INDEX, "DOC_B": Action.INDEX}
    assert {p["guideline_id"] for p in indexer.points.values()} == {"DOC_A", "DOC_B"}
    first_sha = docs[0].sha256
    assert len(first_sha) == 64 and docs[0].chunk_count >= 1 and docs[0].pages == 1

    # Unchanged files -> skipped.
    assert _run(root, docs, indexer, state) == {"DOC_A": Action.SKIP, "DOC_B": Action.SKIP}

    # Changed file -> its old points are deleted and it is re-indexed.
    _write_page(root / "docs" / "a.html", "High blood pressure",
                "Updated text: high blood pressure often has no symptoms at all.")
    assert _run(root, docs, indexer, state) == {"DOC_A": Action.REINDEX, "DOC_B": Action.SKIP}
    assert indexer.deleted == ["DOC_A"]
    assert docs[0].sha256 != first_sha
    assert all(p["doc_sha256"] == docs[0].sha256
               for p in indexer.points.values() if p["guideline_id"] == "DOC_A")

    # Chunking configuration change -> everything re-indexed.
    new_config = chunking_config(300, 40, "whitespace")
    actions = _run(root, docs, indexer, state, config=new_config)
    assert actions == {"DOC_A": Action.REINDEX, "DOC_B": Action.REINDEX}

    # Full rebuild.
    actions = _run(root, docs, indexer, state, config=new_config, recreate=True)
    assert actions == {"DOC_A": Action.INDEX, "DOC_B": Action.INDEX}
    assert indexer.recreated == 1

    saved = load_state(state)
    assert saved.config == new_config
    assert set(saved.documents) == {"DOC_A", "DOC_B"}
    assert saved.last_run.mode == "recreate" and saved.last_run.documents_indexed == 2
    assert saved.last_full_build == saved.last_run
    # A later incremental run keeps the full-build record.
    _run(root, docs, indexer, state, config=new_config)
    saved = load_state(state)
    assert saved.last_run.mode == "incremental" and saved.last_run.documents_skipped == 2
    assert saved.last_full_build.mode == "recreate"


def test_prune_removes_documents_not_in_manifest(corpus, tmp_path: Path) -> None:
    root, docs = corpus
    indexer = FakeIndexer()
    _run(root, docs, indexer, tmp_path / "s.json")
    report = ingest_documents(
        docs[:1], root=root, indexer=indexer,  # type: ignore[arg-type]
        chunker=SectionAwareChunker(token_counter=whitespace_token_count), config=CONFIG,
        prune_to={"DOC_A"}, state_path=tmp_path / "s.json",
    )
    assert report.pruned == ["DOC_B"]
    assert {p["guideline_id"] for p in indexer.points.values()} == {"DOC_A"}


def test_dry_run_writes_nothing(corpus, tmp_path: Path) -> None:
    root, docs = corpus
    indexer = FakeIndexer()
    state = tmp_path / "state.json"
    report = ingest_documents(
        docs, root=root, indexer=indexer,  # type: ignore[arg-type]
        chunker=SectionAwareChunker(token_counter=whitespace_token_count), config=CONFIG,
        dry_run=True, state_path=state,
    )
    assert [r.action for r in report.results] == [Action.INDEX, Action.INDEX]
    assert all(r.chunks >= 1 for r in report.results)
    assert indexer.points == {} and not state.exists()
    assert docs[0].sha256 == ""  # manifest untouched in dry-run mode


def test_missing_and_unparseable_files_are_reported(tmp_path: Path) -> None:
    (tmp_path / "empty.html").write_text("<html><body></body></html>", encoding="utf-8")
    docs = [_doc("MISSING_DOC", "nope.html"), _doc("EMPTY_DOC", "empty.html")]
    report = ingest_documents(
        docs, root=tmp_path, indexer=None,
        chunker=SectionAwareChunker(token_counter=whitespace_token_count), config=CONFIG,
        dry_run=True,
    )
    assert [r.guideline_id for r in report.failed] == ["MISSING_DOC", "EMPTY_DOC"]


def test_state_roundtrip_and_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "state.json"
    state = IngestState(config=CONFIG, collection="c")
    save_state(state, path)
    loaded = load_state(path)
    assert loaded.config == CONFIG and loaded.updated_at
    path.write_text("{not json", encoding="utf-8")
    assert load_state(path) == IngestState()
    assert load_state(tmp_path / "absent.json") == IngestState()


def test_documents_from_directory_uses_manifest_metadata(corpus, tmp_path: Path) -> None:
    root, docs = corpus
    (root / "docs" / "notes.txt").write_text("ignored", encoding="utf-8")
    (root / "docs" / "nice_extra.html").write_text("<html></html>", encoding="utf-8")
    manifest = CorpusManifest(documents=docs)
    found = documents_from_directory(root / "docs", manifest, root)
    assert [d.guideline_id for d in found] == ["DOC_A", "DOC_B", "NICE_NG136"]


def test_update_manifest_file_records_chunking(corpus, tmp_path: Path) -> None:
    _, docs = corpus
    manifest = CorpusManifest(documents=docs)
    path = tmp_path / "manifest.json"
    update_manifest_file(manifest, path, CONFIG)
    assert load_manifest(path).chunking["chunk_size"] == 400
