"""
Prompt Construction Utilities (Module M-11/M-12).

Provides reusable functions for formatting clinical evidence chunks,
context snapshots, and XML-isolated prompt boundaries (ADL-018, ADL-025).
"""
from typing import Any

DEFAULT_EMPTY_CHUNKS_PLACEHOLDER = (
    '<chunk id="0" source="None" section="None">\n'
    "No clinical evidence provided.\n"
    "</chunk>"
)


def format_chunks(
    chunks: list[Any],
    empty_placeholder: str = DEFAULT_EMPTY_CHUNKS_PLACEHOLDER,
    max_chunks: int = 5,
) -> str:
    """
    Format up to max_chunks (default top 5) candidate chunks into XML tags (ADL-025).

    Supports RetrievedChunk objects, dictionaries, (chunk_text, score) tuples,
    and raw string chunks.

    Args:
        chunks: List of retrieved chunks in any supported format.
        empty_placeholder: Text to return when chunks list is empty.
        max_chunks: Maximum number of chunks to include in the output.

    Returns:
        Formatted XML string containing <chunk> elements.
    """
    if not chunks:
        return empty_placeholder

    formatted_pieces: list[str] = []
    for i, chunk in enumerate(chunks[:max_chunks], start=1):
        if hasattr(chunk, "chunk_text"):
            # RetrievedChunk model
            chunk_id = str(getattr(chunk, "chunk_id", i))
            source = str(getattr(chunk, "guideline_id", "Unknown"))
            section = str(getattr(chunk, "section_title", "General"))
            text = str(chunk.chunk_text)
        elif isinstance(chunk, dict):
            chunk_id = str(chunk.get("chunk_id", i))
            source = str(chunk.get("guideline_id", "Unknown"))
            section = str(chunk.get("section_title", "General"))
            text = str(chunk.get("chunk_text", ""))
        elif isinstance(chunk, tuple) and len(chunk) >= 2:
            # (chunk_text, score) tuple from reranker
            chunk_id = str(i)
            source = "RetrievedGuideline"
            section = "General"
            text = str(chunk[0])
        else:
            chunk_id = str(i)
            source = "RetrievedGuideline"
            section = "General"
            text = str(chunk)

        formatted_pieces.append(
            f'<chunk id="{chunk_id}" source="{source}" section="{section}">\n'
            f"{text.strip()}\n"
            f"</chunk>"
        )

    return "\n".join(formatted_pieces)
