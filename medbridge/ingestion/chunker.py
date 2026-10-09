"""
Section-Aware Document Chunker (Module M-18, Ingestion Pipeline).

Splits parsed clinical guideline text into section-aware chunks adhering to
token limits (512 tokens) and overlap windows (64 tokens) while preserving
guideline section headers, page numbers, and citation metadata.

Technical Specification Part V §3.3, ADL-003.
"""
import structlog
import re
from typing import Optional

from pydantic import BaseModel, Field

from medbridge.ingestion.parser import ParsedDocument, ParsedPage

logger = structlog.get_logger(__name__)

# Common clinical section header regex patterns
_SECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(?:Section|SECTION)\s+\d+(?:\.\d+)*[:\s].*", re.MULTILINE),
    re.compile(r"^(?:Chapter|CHAPTER)\s+\d+[:\s].*", re.MULTILINE),
    re.compile(r"^\d+\.\d+(?:\.\d+)*\s+[A-Z].*", re.MULTILINE),
    re.compile(r"^[A-Z0-9\s,\-:]{4,80}$", re.MULTILINE),
]


class RawChunk(BaseModel):
    """Pydantic model representing an unindexed guideline chunk with metadata."""

    chunk_id: str = Field(..., description="Unique chunk identifier, e.g. 'aha_2025_s8_c1'")
    guideline_id: str = Field(..., description="Guideline source identifier, e.g. 'AHA_ACC_2025'")
    section_title: str = Field(default="General", description="Section or chapter title")
    page_number: int = Field(default=1, description="Source PDF page number (1-indexed)")
    chunk_text: str = Field(..., description="Full text content of the chunk")
    source_url: str = Field(default="", description="Reference document or guideline URL")


def count_tokens(text: str) -> int:
    """Approximate token count for a text string based on whitespace words."""
    if not text or not text.strip():
        return 0
    return len(text.split())


def _slugify(text: str, max_length: int = 20) -> str:
    """Generate a clean URL/identifier-friendly slug from text."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:max_length] or "sec"


class SectionAwareChunker:
    """Performs section-aware chunking over parsed clinical documents.

    Splits at section boundaries, then uses token windowing with overlap
    to enforce chunk size limits without severing clinical context units.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        """Initialize the chunker.

        Args:
            chunk_size: Maximum token count per chunk (default: 512).
            chunk_overlap: Number of tokens to overlap between adjacent chunks (default: 64).
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def is_section_header(self, line: str) -> bool:
        """Check if a line matches any clinical guideline section pattern."""
        stripped = line.strip()
        if not stripped or len(stripped) < 4:
            return False
        return any(pattern.match(stripped) is not None for pattern in _SECTION_PATTERNS)

    def chunk_document(self, document: ParsedDocument) -> list[RawChunk]:
        """Chunk an entire parsed document into section-aware RawChunk items.

        Args:
            document: ParsedDocument containing ordered pages.

        Returns:
            List of validated RawChunk objects ready for vector indexing.
        """
        all_chunks: list[RawChunk] = []
        chunk_counter = 1
        current_section = "Introduction"

        for page in document.pages:
            page_chunks = self.chunk_text(
                text=page.text,
                guideline_id=document.guideline_id,
                page_number=page.page_number,
                source_url=document.source_url,
                start_index=chunk_counter,
                initial_section=current_section,
            )
            if page_chunks:
                all_chunks.extend(page_chunks)
                chunk_counter += len(page_chunks)
                # Update ongoing section from the last chunk on the page
                current_section = page_chunks[-1].section_title

        logger.info(
            "Chunked document '%s' into %d chunks (size=%d, overlap=%d)",
            document.guideline_id,
            len(all_chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return all_chunks

    def chunk_text(
        self,
        text: str,
        guideline_id: str = "GUIDELINE",
        page_number: int = 1,
        source_url: str = "",
        start_index: int = 1,
        initial_section: str = "General",
    ) -> list[RawChunk]:
        """Split a string of text into section-aware chunks.

        Args:
            text: Raw text to split.
            guideline_id: Associated guideline identifier.
            page_number: PDF page number.
            source_url: URL reference.
            start_index: Starting integer for chunk IDs.
            initial_section: Active section title.

        Returns:
            List of RawChunk objects.
        """
        if not text or not text.strip():
            return []

        # 1. Split text into section blocks
        lines = text.split("\n")
        sections: list[tuple[str, list[str]]] = []
        current_title = initial_section
        current_lines: list[str] = []

        for line in lines:
            if self.is_section_header(line):
                if current_lines:
                    sections.append((current_title, current_lines))
                    current_lines = []
                current_title = line.strip()
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_title, current_lines))

        chunks: list[RawChunk] = []
        chunk_idx = start_index

        # 2. Tokenize and window each section
        step = self.chunk_size - self.chunk_overlap

        for sec_title, sec_lines in sections:
            sec_text = "\n".join(sec_lines).strip()
            if not sec_text:
                continue

            words = sec_text.split()
            if not words:
                continue

            if len(words) <= self.chunk_size:
                # Single chunk fits entire section
                sec_slug = _slugify(sec_title)
                chunk_id = f"{guideline_id.lower()}_{sec_slug}_c{chunk_idx}"
                chunks.append(
                    RawChunk(
                        chunk_id=chunk_id,
                        guideline_id=guideline_id,
                        section_title=sec_title,
                        page_number=page_number,
                        chunk_text=f"{sec_title}\n\n{sec_text}" if sec_title != "General" else sec_text,
                        source_url=source_url,
                    )
                )
                chunk_idx += 1
            else:
                # Sliding window over words
                for i in range(0, len(words), step):
                    window_words = words[i : i + self.chunk_size]
                    if not window_words:
                        break

                    window_text = " ".join(window_words)
                    sec_slug = _slugify(sec_title)
                    chunk_id = f"{guideline_id.lower()}_{sec_slug}_c{chunk_idx}"

                    full_chunk_text = (
                        f"{sec_title}\n\n{window_text}"
                        if sec_title != "General"
                        else window_text
                    )

                    chunks.append(
                        RawChunk(
                            chunk_id=chunk_id,
                            guideline_id=guideline_id,
                            section_title=sec_title,
                            page_number=page_number,
                            chunk_text=full_chunk_text,
                            source_url=source_url,
                        )
                    )
                    chunk_idx += 1

                    # If this window reached the end of words, stop
                    if i + self.chunk_size >= len(words):
                        break

        return chunks
