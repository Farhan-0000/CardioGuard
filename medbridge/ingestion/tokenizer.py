"""
Token counting for chunk sizing (Module M-18, Ingestion Pipeline).

Chunks are sized with the *same* WordPiece tokenizer family used by the dense
embedder (``BAAI/bge-small-en-v1.5``) and the cross-encoder reranker
(``cross-encoder/ms-marco-MiniLM-L-6-v2``): both are BERT-uncased models with a
512-token input limit. Counting whitespace words (the previous approach)
badly under-estimates WordPiece length on clinical text ("amlodipine" -> 4
pieces, "130/80" -> 3). Measured on the previous index (905 chunks of <= 512
*words*): 1.95 word pieces per word on average, 39 % of chunks longer than 512
tokens, and 33 % of all indexed tokens beyond the 510-token window -- silently
truncated by both models. Measuring with the real tokenizer guarantees that a
400-token chunk plus a query still fits in the reranker's 512-token window.

The tokenizer file is resolved through fastembed's own model cache (no ONNX
session is created). When it cannot be loaded (offline first run, tests), a
deterministic word-piece approximation is used and a warning is logged once.
"""
from __future__ import annotations

import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_PUNCT_RE = re.compile(r"[^\w\s]")


def whitespace_token_count(text: str) -> int:
    """Count whitespace-delimited words (deterministic counter used in unit tests)."""
    if not text or not text.strip():
        return 0
    return len(text.split())


def approximate_wordpiece_count(text: str) -> int:
    """Approximate the BERT word-piece length of ``text`` without a tokenizer.

    Each word costs one piece, plus one per punctuation character (BERT splits
    punctuation into separate tokens) and one extra for long words, which
    WordPiece usually breaks up. Additive over words, like the real tokenizer.
    """
    total = 0
    for word in (text or "").split():
        total += 1 + len(_PUNCT_RE.findall(word)) + (1 if len(word) > 8 else 0)
    return total


class _TokenizerHolder:
    """Lazily loaded, process-wide ``tokenizers.Tokenizer`` for the embedder model."""

    _lock = threading.Lock()
    _counter: Callable[[str], int] | None = None
    _source: str = ""

    @classmethod
    def get(cls, model_name: str) -> Callable[[str], int]:
        """Return a token counter for ``model_name`` (cached after the first call)."""
        if cls._counter is not None:
            return cls._counter
        with cls._lock:
            if cls._counter is None:
                cls._counter, cls._source = cls._load(model_name)
        return cls._counter

    @staticmethod
    def _load(model_name: str) -> tuple[Callable[[str], int], str]:
        try:
            from fastembed import TextEmbedding
            from tokenizers import Tokenizer

            # lazy_load downloads/caches the model files without creating an ONNX session.
            backend: Any = TextEmbedding(model_name=model_name, lazy_load=True).model
            model_dir = Path(backend._model_dir)
            tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
            tokenizer.no_truncation()
            tokenizer.no_padding()

            def count(text: str) -> int:
                if not text:
                    return 0
                return len(tokenizer.encode(text, add_special_tokens=False).ids)

            logger.info("Loaded chunking tokenizer", model=model_name, path=str(model_dir))
            return count, model_name
        except (ImportError, OSError, ValueError, RuntimeError, AttributeError) as exc:
            logger.warning(
                "Tokenizer unavailable; using word-piece approximation",
                model=model_name,
                error=str(exc),
            )
            return approximate_wordpiece_count, "approximation"


def get_token_counter(model_name: str | None = None) -> Callable[[str], int]:
    """Return the token counter matching the configured dense embedding model.

    Args:
        model_name: fastembed model name; defaults to ``Settings.EMBEDDING_MODEL``.

    Returns:
        A callable ``count(text) -> int`` measuring model tokens without special tokens.
    """
    if model_name is None:
        from medbridge.config import get_settings

        model_name = get_settings().EMBEDDING_MODEL
    return _TokenizerHolder.get(model_name)


def token_counter_source() -> str:
    """Name of the tokenizer backing :func:`get_token_counter` ("" before first use)."""
    return _TokenizerHolder._source
