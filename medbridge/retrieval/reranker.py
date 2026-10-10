"""
Cross-Encoder Reranker (Module M-17, ADL-022).

Scores (query, chunk) pairs with ``cross-encoder/ms-marco-MiniLM-L-6-v2`` and
keeps the top-5 of the 20 fused candidates. Inference is CPU/GPU bound and is
always offloaded with ``asyncio.to_thread`` (Constitution §7.7) so the event
loop is never blocked.

* Device is chosen automatically: CUDA > Apple MPS > CPU.
* Thread-safe lazy singleton; ``predict`` is serialised with a lock because a
  single model instance is shared by concurrent requests.
* :meth:`rerank_chunks` works on :class:`RetrievedChunk` objects (no deep
  copies — scored chunks are shallow ``model_copy`` updates that keep the RRF
  score and add ``score_rerank``); :meth:`rerank` keeps the legacy
  ``(text, score)`` API.
"""

from __future__ import annotations

import asyncio
import threading

import structlog
from sentence_transformers import CrossEncoder

from medbridge.config import get_settings
from medbridge.core.inference_lock import TORCH_INFERENCE_LOCK
from medbridge.retrieval.hybrid_retriever import RetrievedChunk

logger = structlog.get_logger(__name__)

_WARMUP_PAIR = ("blood pressure target", "The recommended blood pressure target is discussed.")


def detect_device() -> str:
    """Return the best available torch device: ``cuda`` > ``mps`` > ``cpu``."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class CrossEncoderReranker:
    """Cross-encoder reranking service (lazy, thread-safe singleton)."""

    _instance: CrossEncoderReranker | None = None
    _lock = threading.Lock()
    _initialized: bool

    def __new__(cls) -> CrossEncoderReranker:
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        settings = get_settings()
        self.model_name: str = settings.RERANKER_MODEL
        self.top_k: int = settings.RETRIEVAL_TOP_K_RERANKED
        self.device: str = detect_device()
        self._model: CrossEncoder | None = None
        self._init_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._initialized = True

    @property
    def loaded(self) -> bool:
        """Whether the model is resident in memory."""
        return self._model is not None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            with self._init_lock, TORCH_INFERENCE_LOCK:
                if self._model is None:
                    self._model = CrossEncoder(self.model_name, device=self.device)
                    logger.info("reranker_loaded", model=self.model_name, device=self.device)
        return self._model

    def _score(self, query: str, texts: list[str]) -> list[float]:
        model = self._get_model()
        with self._predict_lock, TORCH_INFERENCE_LOCK:
            scores = model.predict([(query, text) for text in texts])
        return [float(s) for s in scores]

    def _sync_rerank(self, query: str, chunks: list[str], top_k: int) -> list[tuple[str, float]]:
        if not chunks:
            return []
        scored = list(zip(chunks, self._score(query, chunks), strict=True))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    async def rerank(
        self, query: str, chunks: list[str], top_k: int | None = None
    ) -> list[tuple[str, float]]:
        """Rerank raw texts; returns ``(text, score)`` pairs, best first (legacy API)."""
        if not chunks:
            return []
        return await asyncio.to_thread(self._sync_rerank, query, chunks, top_k or self.top_k)

    def _sync_rerank_chunks(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        scores = self._score(query, [c.chunk_text for c in chunks])
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            chunks[i].model_copy(update={"score_rerank": scores[i], "score": scores[i]})
            for i in order
        ]

    async def rerank_chunks(
        self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Rerank retrieved chunks; returns new chunk objects, best first.

        Each returned chunk has ``score_rerank`` (and ``score``) set to the
        cross-encoder score while ``score_rrf`` is preserved. Inputs are not
        mutated.
        """
        if not chunks:
            return []
        return await asyncio.to_thread(self._sync_rerank_chunks, query, chunks, top_k or self.top_k)

    async def warmup(self) -> None:
        """Load the model and score one probe pair (first-query latency → 0)."""
        await asyncio.to_thread(self._score, _WARMUP_PAIR[0], [_WARMUP_PAIR[1]])


def get_reranker() -> CrossEncoderReranker:
    """Return the process-wide reranker singleton."""
    return CrossEncoderReranker()
