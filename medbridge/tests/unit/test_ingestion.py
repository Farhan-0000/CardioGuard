"""
Unit tests for the Knowledge Ingestion CLI Pipeline (Module M-18).

Tests:
1. PDF parser page and text extraction using PyMuPDF.
2. Section-aware chunking, token limits, and overlap windows.
3. Payload schema validation and the contract §3.5 payload.
4. Qdrant indexer (batch embedding, dry-run write bypass).
5. CLI argument parsing and legacy ``--pdf-dir`` mode.

All tests are hermetic: no Qdrant, no embedding model, no network.
"""
from pathlib import Path
from unittest.mock import MagicMock

import pymupdf
import pytest

from medbridge.ingestion.__main__ import build_arg_parser, run_ingestion
from medbridge.ingestion.chunker import RawChunk, SectionAwareChunker, count_tokens
from medbridge.ingestion.indexer import (
    ChunkValidationError,
    QdrantIndexer,
    validate_chunk_payload,
)
from medbridge.ingestion.parser import (
    ParsedDocument,
    ParsedPage,
    PDFParseError,
    deduce_guideline_id,
    extract_pages_from_pdf,
)
from medbridge.ingestion.tokenizer import whitespace_token_count

CONTRACT_PAYLOAD_KEYS = {
    "chunk_id", "chunk_text", "guideline_id", "doc_title", "org", "year", "section_title",
    "page_number", "page_end", "source_url", "topics", "doc_sha256", "chunk_index",
    "content_type",
}


def _chunker(**kwargs) -> SectionAwareChunker:
    kwargs.setdefault("token_counter", whitespace_token_count)
    return SectionAwareChunker(**kwargs)


# ---------------------------------------------------------------------------
# 1. PDF Parser Tests
# ---------------------------------------------------------------------------

class TestPDFParser:
    """Verify PyMuPDF text and page extraction from guideline PDFs."""

    def test_deduce_guideline_id(self) -> None:
        assert deduce_guideline_id("aha_acc_2025_hypertension.pdf") == "AHA_ACC_2025"
        assert deduce_guideline_id("ESC_ESH_2024.pdf") == "ESC_ESH_2024"
        assert deduce_guideline_id("medlineplus_blood_pressure.pdf") == "MEDLINEPLUS"
        assert deduce_guideline_id("custom_guideline_v2.pdf") == "CUSTOM_GUIDELINE_V2"

    def test_extract_pages_from_real_pdf(self, tmp_path: Path) -> None:
        """Create a real PDF and verify page-aware extraction."""
        pdf_file = tmp_path / "aha_acc_2025_test.pdf"
        doc = pymupdf.open()
        p1 = doc.new_page()
        p1.insert_text((50, 80), "Section 1: Diagnostic Criteria for Hypertension\nBlood pressure >= 130/80.")
        p2 = doc.new_page()
        p2.insert_text((50, 80), "Section 2: Pharmacologic Treatment\nFirst-line therapy includes CCBs.")
        doc.save(str(pdf_file))
        doc.close()

        parsed = extract_pages_from_pdf(pdf_file)

        assert isinstance(parsed, ParsedDocument)
        assert parsed.guideline_id == "AHA_ACC_2025"
        assert parsed.source_format == "pdf"
        assert parsed.total_pages == 2
        assert parsed.pages[0].page_number == 1
        assert "Diagnostic Criteria" in parsed.pages[0].text
        assert parsed.pages[1].page_number == 2
        assert "Pharmacologic Treatment" in parsed.pages[1].text
        headings = [b.text for b in parsed.pages[0].blocks if b.kind == "heading"]
        assert headings == ["Section 1: Diagnostic Criteria for Hypertension"]

    def test_nonexistent_pdf_raises_error(self, tmp_path: Path) -> None:
        missing_file = tmp_path / "does_not_exist.pdf"
        with pytest.raises(PDFParseError) as exc_info:
            extract_pages_from_pdf(missing_file)
        assert "File does not exist" in str(exc_info.value)

    def test_corrupt_pdf_raises_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "broken.pdf"
        bad.write_bytes(b"this is not a pdf")
        with pytest.raises(PDFParseError):
            extract_pages_from_pdf(bad)


# ---------------------------------------------------------------------------
# 2. Section-Aware Chunker Tests (Token Limits & Overlap)
# ---------------------------------------------------------------------------

class TestSectionAwareChunker:
    """Verify section-aware splitting, token limit enforcement, and overlap."""

    def test_section_header_detection(self) -> None:
        chunker = _chunker(chunk_size=100, chunk_overlap=20)
        assert chunker.is_section_header("Section 8: Pharmacological Treatment") is True
        assert chunker.is_section_header("Chapter 3: Diagnostics") is True
        assert chunker.is_section_header("8.2 First-Line Therapy") is True
        assert chunker.is_section_header("RECOMMENDATIONS FOR PATIENTS") is True
        assert chunker.is_section_header("This is ordinary body paragraph text.") is False
        assert chunker.is_section_header("00505-6") is False

    def test_chunking_token_limits_and_overlap(self) -> None:
        """Chunks respect chunk_size and consecutive chunks share chunk_overlap tokens."""
        chunk_size, chunk_overlap = 50, 10
        chunker = _chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        text = " ".join(f"word{i}" for i in range(1, 121))

        chunks = chunker.chunk_text(text=text, guideline_id="AHA_2025", page_number=1)

        assert len(chunks) > 1
        for c in chunks:
            assert count_tokens(c.chunk_text) <= chunk_size
            assert c.token_count <= chunk_size
        first, second = chunks[0].chunk_text.split(), chunks[1].chunk_text.split()
        shared = [w for w in first[-chunk_overlap:] if w in second[:chunk_overlap]]
        assert len(shared) == chunk_overlap
        # Every word is covered.
        covered = {w for c in chunks for w in c.chunk_text.split()}
        assert covered == set(text.split())

    def test_section_headers_preserved_in_chunks(self) -> None:
        chunker = _chunker(chunk_size=100, chunk_overlap=10, min_section_tokens=5)
        content = (
            "Section 1: Initial Diagnosis\n"
            "Patients with elevated blood pressure should be evaluated with ambulatory monitoring.\n\n"
            "Section 2: Lifestyle Modifications\n"
            "Sodium restriction and regular physical exercise are recommended for all patients."
        )
        doc = ParsedDocument(
            guideline_id="AHA_2025",
            file_path="test.pdf",
            pages=[ParsedPage(page_number=1, text=content)],
        )
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 2
        assert chunks[0].section_title == "Section 1: Initial Diagnosis"
        assert chunks[0].chunk_text.startswith("Section 1: Initial Diagnosis")
        assert "ambulatory monitoring" in chunks[0].chunk_text
        assert chunks[1].section_title == "Section 2: Lifestyle Modifications"
        assert "Sodium restriction" in chunks[1].chunk_text
        assert [c.chunk_index for c in chunks] == [0, 1]

    def test_invalid_chunk_parameters_raise_value_error(self) -> None:
        with pytest.raises(ValueError):
            _chunker(chunk_size=0)
        with pytest.raises(ValueError):
            _chunker(chunk_size=50, chunk_overlap=50)
        with pytest.raises(ValueError):
            _chunker(min_section_tokens=-1)


# ---------------------------------------------------------------------------
# 3. Payload Schema Validation Tests
# ---------------------------------------------------------------------------

class TestPayloadSchemaValidation:
    """Verify schema constraints for Qdrant payload integrity."""

    def test_valid_payload_accepted(self) -> None:
        valid_chunk = RawChunk(
            chunk_id="aha_2025_s1_c1",
            guideline_id="AHA_ACC_HTN_2025",
            section_title="Diagnosis",
            page_number=3,
            chunk_text="Hypertension is defined as systolic >= 130 mm Hg.",
            source_url="https://guidelines.acc.org/htn2025",
        )
        assert validate_chunk_payload(valid_chunk) is True
        assert valid_chunk.page_end == 3  # defaults to page_number

    def test_payload_has_contract_keys(self) -> None:
        chunk = RawChunk(
            chunk_id="c1", guideline_id="WHO_HTN_PHARM_2021", section_title="Targets",
            page_number=4, page_end=5, chunk_text="Target BP below 140/90 mm Hg.",
            topics=["hypertension"], doc_sha256="a" * 64, content_type="table",
        )
        payload = chunk.to_payload()
        assert set(payload) == CONTRACT_PAYLOAD_KEYS | {"source_format"}
        assert payload["source_format"] == "pdf"
        assert payload["page_end"] == 5
        assert payload["content_type"] == "table"
        assert payload["topics"] == ["hypertension"]

    def test_missing_or_empty_chunk_id_rejected(self) -> None:
        with pytest.raises(ChunkValidationError):
            validate_chunk_payload(RawChunk(chunk_id="   ", guideline_id="AHA",
                                            section_title="Sec", page_number=1,
                                            chunk_text="Valid chunk text exceeding five chars"))

    def test_invalid_page_number_rejected(self) -> None:
        with pytest.raises(ChunkValidationError):
            validate_chunk_payload(RawChunk(chunk_id="c1", guideline_id="AHA",
                                            section_title="Sec", page_number=0,
                                            chunk_text="Valid chunk text"))

    def test_empty_or_too_short_text_rejected(self) -> None:
        with pytest.raises(ChunkValidationError):
            validate_chunk_payload(RawChunk(chunk_id="c1", guideline_id="AHA",
                                            section_title="Sec", page_number=1,
                                            chunk_text="Hi"))


# ---------------------------------------------------------------------------
# 4. Qdrant Indexer & Dry-Run Tests
# ---------------------------------------------------------------------------

class TestQdrantIndexer:
    """Verify batch vectorisation, collection configuration and dry-run write bypass."""

    @pytest.fixture
    def mock_embedder(self):
        embedder = MagicMock()
        embedder.embed_dense_batch.side_effect = lambda texts: [[0.1] * 384 for _ in texts]
        embedder.embed_sparse_batch.side_effect = lambda texts: [
            {"indices": [1, 2], "values": [0.5, 0.9]} for _ in texts
        ]
        return embedder

    @pytest.fixture
    def mock_qdrant_client(self):
        client = MagicMock()
        client.get_collections.return_value = MagicMock(collections=[])
        return client

    @staticmethod
    def _chunks(n: int) -> list[RawChunk]:
        return [
            RawChunk(chunk_id=f"c{i}", guideline_id="AHA_2025", section_title="Sec",
                     page_number=1, chunk_text=f"Chunk number {i} text content.")
            for i in range(n)
        ]

    def test_dry_run_bypasses_qdrant_writes(self, mock_qdrant_client, mock_embedder) -> None:
        indexer = QdrantIndexer(client=mock_qdrant_client, embedder=mock_embedder)
        count = indexer.index_chunks(chunks=self._chunks(1), dry_run=True)

        assert count == 1
        mock_qdrant_client.create_collection.assert_not_called()
        mock_qdrant_client.upsert.assert_not_called()
        mock_embedder.embed_dense_batch.assert_not_called()

    def test_live_indexing_creates_collection_and_upserts_in_batches(
        self, mock_qdrant_client, mock_embedder
    ) -> None:
        indexer = QdrantIndexer(client=mock_qdrant_client, embedder=mock_embedder)
        count = indexer.index_chunks(chunks=self._chunks(5), batch_size=2, recreate=True)

        assert count == 5
        mock_qdrant_client.create_collection.assert_called_once()
        create_kwargs = mock_qdrant_client.create_collection.call_args.kwargs
        assert create_kwargs["vectors_config"]["dense"].size == 384
        assert "sparse" in create_kwargs["sparse_vectors_config"]
        assert mock_qdrant_client.upsert.call_count == 3  # 2 + 2 + 1
        assert mock_embedder.embed_dense_batch.call_count == 3
        points = mock_qdrant_client.upsert.call_args_list[0].kwargs["points"]
        assert CONTRACT_PAYLOAD_KEYS <= set(points[0].payload)
        assert set(points[0].vector) == {"dense", "sparse"}

    def test_recreate_deletes_existing_collection(self, mock_qdrant_client, mock_embedder) -> None:
        existing = MagicMock()
        existing.name = "clinical_guidelines"
        mock_qdrant_client.get_collections.side_effect = [
            MagicMock(collections=[existing]), MagicMock(collections=[]),
        ]
        indexer = QdrantIndexer(client=mock_qdrant_client, embedder=mock_embedder)
        indexer.ensure_collection("clinical_guidelines", recreate=True)

        mock_qdrant_client.delete_collection.assert_called_once_with("clinical_guidelines")
        mock_qdrant_client.create_collection.assert_called_once()
        assert mock_qdrant_client.create_payload_index.call_count == 4

    def test_invalid_chunks_are_skipped(self, mock_qdrant_client, mock_embedder) -> None:
        indexer = QdrantIndexer(client=mock_qdrant_client, embedder=mock_embedder)
        chunks = self._chunks(2) + [RawChunk(chunk_id="bad", guideline_id="AHA",
                                             section_title="S", page_number=1, chunk_text="x")]
        assert indexer.index_chunks(chunks=chunks, dry_run=True) == 2


# ---------------------------------------------------------------------------
# 5. CLI Parser & Execution Tests
# ---------------------------------------------------------------------------

class TestCLI:
    """Verify argument parsing and CLI coordination."""

    def test_build_arg_parser(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args([
            "--pdf-dir", "./data/guidelines",
            "--qdrant-url", "http://127.0.0.1:6333",
            "--collection", "clinical_guidelines",
            "--chunk-size", "512",
            "--chunk-overlap", "64",
            "--recreate",
            "--dry-run",
        ])
        assert args.pdf_dir == "./data/guidelines"
        assert args.qdrant_url == "http://127.0.0.1:6333"
        assert args.collection == "clinical_guidelines"
        assert args.chunk_size == 512
        assert args.chunk_overlap == 64
        assert args.recreate is True
        assert args.dry_run is True

    def test_defaults_are_manifest_driven(self) -> None:
        args = build_arg_parser().parse_args([])
        assert args.pdf_dir is None
        assert args.manifest.endswith("corpus_manifest.json")
        assert args.chunk_size == 400
        assert args.chunk_overlap == 60
        assert args.stats is False

    def test_run_ingestion_empty_dir(self, tmp_path: Path) -> None:
        args = build_arg_parser().parse_args(["--pdf-dir", str(tmp_path), "--dry-run"])
        assert run_ingestion(args) == 0

    def test_run_ingestion_missing_dir(self, tmp_path: Path) -> None:
        args = build_arg_parser().parse_args(["--pdf-dir", str(tmp_path / "nope"), "--dry-run"])
        assert run_ingestion(args) == 1

    def test_run_ingestion_legacy_dir_dry_run(self, tmp_path: Path, monkeypatch) -> None:
        pdf = tmp_path / "custom_guideline.pdf"
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 80), "1. Measuring blood pressure\nUse a validated upper-arm cuff.")
        doc.save(str(pdf))
        doc.close()
        monkeypatch.setattr(
            "medbridge.ingestion.__main__.SectionAwareChunker",
            lambda **kw: SectionAwareChunker(token_counter=whitespace_token_count, **kw),
        )
        args = build_arg_parser().parse_args(["--pdf-dir", str(tmp_path), "--dry-run",
                                              "--manifest", str(tmp_path / "none.json")])
        assert run_ingestion(args) == 0

    def test_unknown_only_id_fails(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(
            "medbridge.ingestion.__main__.SectionAwareChunker",
            lambda **kw: SectionAwareChunker(token_counter=whitespace_token_count, **kw),
        )
        args = build_arg_parser().parse_args(["--dry-run", "--only", "NOT_A_REAL_ID"])
        assert run_ingestion(args) == 1
