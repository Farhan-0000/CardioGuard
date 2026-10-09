"""
PDF Document Parser (Module M-18, Ingestion Pipeline).

Extracts text and page numbers from clinical guideline PDFs using PyMuPDF.
Preserves page-level metadata required for citation generation.

Technical Specification Part V §3.2, §8.
"""
import structlog
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pymupdf

logger = structlog.get_logger(__name__)


class PDFParseError(Exception):
    """Raised when a PDF file cannot be opened, read, or parsed."""

    def __init__(self, file_path: str, message: str) -> None:
        self.file_path = file_path
        super().__init__(f"Failed to parse PDF '{file_path}': {message}")


@dataclass
class ParsedPage:
    """Represents a single parsed PDF page with 1-indexed page number."""

    page_number: int
    text: str


@dataclass
class ParsedDocument:
    """Represents a complete parsed guideline document."""

    guideline_id: str
    file_path: str
    pages: list[ParsedPage] = field(default_factory=list)
    source_url: str = ""

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


def deduce_guideline_id(filename: str) -> str:
    """Infer a standardized guideline identifier from a file path or filename.

    Examples:
        'aha_acc_2025.pdf' -> 'AHA_ACC_2025'
        'ehae178.pdf' -> 'ESC_2024'
        'hypertension-in-adults-diagnosis-and-management...pdf' -> 'NICE_NG136'
        'esc-esh-2024.pdf' -> 'ESC_2024'
        'medlineplus_hypertension.pdf' -> 'MEDLINEPLUS'
    """
    stem = Path(filename).stem
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").upper()

    if "AHA" in clean or "JONES" in clean:
        return "AHA_ACC_2025"
    if "ESC_ESH" in clean:
        return "ESC_ESH_2024"
    if "ESC" in clean or "EHAE178" in clean or "EHAE" in clean:
        return "ESC_2024"
    if "NICE" in clean or "DIAGNOSIS_AND_MANAGEMENT" in clean or "NG136" in clean:
        return "NICE_NG136"
    if "MEDLINE" in clean:
        return "MEDLINEPLUS"

    return clean or "CLINICAL_GUIDELINE"


def extract_pages_from_pdf(
    pdf_path: Union[Path, str],
    guideline_id: Optional[str] = None,
    source_url: str = "",
) -> ParsedDocument:
    """Extract text and page numbers from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the target PDF document.
        guideline_id: Optional guideline identifier. If omitted, deduced from filename.
        source_url: Optional source reference or document URL.

    Returns:
        ParsedDocument containing the list of ParsedPage objects.

    Raises:
        PDFParseError: If the file does not exist, is corrupted, or fails to parse.
    """
    path_obj = Path(pdf_path)
    if not path_obj.exists():
        raise PDFParseError(str(pdf_path), "File does not exist")

    gid = guideline_id or deduce_guideline_id(path_obj.name)

    try:
        doc = pymupdf.open(str(path_obj))
    except Exception as e:
        logger.error("PyMuPDF failed to open %s: %s", path_obj, e)
        raise PDFParseError(str(pdf_path), str(e)) from e

    parsed_pages: list[ParsedPage] = []

    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_text = page.get_text("text") or ""
            # Normalize whitespace
            clean_text = page_text.replace("\r\n", "\n").strip()
            parsed_pages.append(
                ParsedPage(
                    page_number=page_idx + 1,  # 1-indexed
                    text=clean_text,
                )
            )
    except Exception as e:
        logger.error("Error extracting text from page in %s: %s", path_obj, e)
        raise PDFParseError(str(pdf_path), str(e)) from e
    finally:
        doc.close()

    logger.info(
        "Parsed PDF successfully: %s (%d pages, guideline_id=%s)",
        path_obj.name,
        len(parsed_pages),
        gid,
    )

    return ParsedDocument(
        guideline_id=gid,
        file_path=str(path_obj.resolve()),
        pages=parsed_pages,
        source_url=source_url,
    )
