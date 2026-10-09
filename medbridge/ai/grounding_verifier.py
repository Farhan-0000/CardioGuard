"""
V2 — post-generation Grounding Verifier, "Gate 3" (CLAUDE_AGENT_PROMPT §6.7, ADL-031).

After the generator has written an ANSWER/GENERALIZE draft (and before the
patient sees it), every clinical claim sentence is checked by a local
cross-encoder NLI model against the evidence chunks it cites (``[n]`` markers;
all chunks when uncited). A claim is *supported* when its best entailment
probability over ~3-sentence evidence windows reaches
``GROUNDING_ENTAILMENT_THRESHOLD``. Unsupported claims are removed from
``cleaned_text`` (markdown structure preserved); the orchestrator uses
``unsupported_fraction`` to keep, downgrade or abstain.

The verifier only *adds* a check after the Dual Answerability Gate — it never
replaces Gate 1 or Gate 2. The production NLI model
(``cross-encoder/nli-MiniLM2-L6-H768`` by default) deliberately differs from the
NLI judge used in the evaluation, so the verifier is not graded by itself.

Inference runs in a worker thread (``asyncio.to_thread``) on cuda > mps > cpu.
"""
from __future__ import annotations

import asyncio
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import structlog

from medbridge.ai.prompt_utils import chunk_text
from medbridge.config import get_settings
from medbridge.core.templates import get_loop_breaker_prefix

if TYPE_CHECKING:
    from medbridge.retrieval.hybrid_retriever import RetrievedChunk

logger = structlog.get_logger(__name__)

# sentence-transformers NLI cross-encoders use this order when labels are unnamed.
_DEFAULT_LABELS: dict[int, str] = {0: "contradiction", 1: "entailment", 2: "neutral"}
WINDOW_SENTENCES = 3
WINDOW_STRIDE = 2
WINDOW_MAX_CHARS = 1400  # ≈ 350 tokens
_MAX_SEQ_LENGTH = 512

_ABBREVIATIONS: frozenset[str] = frozenset({
    "e.g", "i.e", "etc", "vs", "dr", "mr", "mrs", "ms", "prof", "approx", "mg", "mcg", "no",
    "fig", "st", "al", "cf", "min", "max", "incl", "esp", "resp",
})
_BOUNDARY = re.compile(r"([.!?]+[\"')\]]*)((?:\s*\[\d+(?:\s*[,–-]\s*\d+)*\])*)\s+(?=\S)")
_MARKER = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
_LIST_ITEM = re.compile(r"^(\s*(?:[-*•+]|\d+[.)])\s+)(.*)$")
_HEADING = re.compile(r"^\s*(?:#{1,6}\s+.*|\*\*[^*]+\*\*:?|__[^_]+__:?)\s*$")
_NON_CLAIM_START = re.compile(
    r"^(?:hi|hello|hey|thanks|thank you|you're welcome|i hope (?:this|that) helps|"
    r"(?:i'm |i am )?sorry to hear|great question|good question|i understand|i know|"
    r"it's understandable|it is understandable|it can (?:feel|be) (?:worrying|stressful|"
    r"confusing|overwhelming)|it's (?:great|good) that you|here (?:is|are) )",
    re.IGNORECASE,
)
_BOILERPLATE = re.compile(
    r"^(?:please\s+)?(?:always\s+)?(?:consult|talk to|talk with|speak to|speak with|discuss|"
    r"check with|ask|contact|see|reach out to)\b.*\b(?:doctor|clinician|healthcare|"
    r"health care|gp|physician|pharmacist|care team|nurse|provider)\b"
    r"|\bnot (?:a substitute for|medical advice|personalised|personalized)\b"
    r"|\b(?:general|educational) information\b"
    r"|\binformational purposes\b"
    r"|\bi(?: am|'m) not a doctor\b"
    r"|\b(?:emergency services|call (?:911|112|108|999)|emergency room|emergency department)\b"
    r"|\bseek (?:immediate|urgent|emergency) (?:medical )?(?:care|help|attention)\b",
    re.IGNORECASE,
)


class NLIModel(Protocol):
    """The subset of ``sentence_transformers.CrossEncoder`` used here."""

    config: Any

    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> Any: ...


@dataclass
class GroundingResult:
    """Per-sentence verdicts plus the response with unsupported claims removed."""

    sentences: list[dict[str, Any]]
    n_claims: int
    n_unsupported: int
    unsupported_fraction: float
    cleaned_text: str
    model: str = ""
    available: bool = True
    latency_ms: float = 0.0
    unsupported_sentences: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Text segmentation
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping trailing ``[n]`` markers with their sentence.

    Does not split after common abbreviations (e.g., i.e., Dr., mg.), inside
    decimals (2.5) or before a lower-case continuation.
    """
    text = text.strip()
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        if match.group(1).startswith("."):
            last_word = re.search(r"([A-Za-z.]+)$", text[start:match.start()])
            if last_word and last_word.group(1).lower().strip(".") in _ABBREVIATIONS:
                continue
        if text[match.end()].islower():
            continue
        sentences.append(text[start:match.end()].strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def parse_markers(sentence: str) -> list[int]:
    """Citation indices in a sentence: ``[1]``, ``[1, 3]``, ``[2-4]`` → [1], [1, 3], [2, 3, 4]."""
    found: list[int] = []
    for group in _MARKER.findall(sentence):
        for part in re.split(r"\s*,\s*", group):
            bounds = [int(b) for b in re.split(r"\s*[–-]\s*", part) if b.strip()]
            if len(bounds) == 2 and bounds[0] <= bounds[1] <= bounds[0] + 10:
                found.extend(range(bounds[0], bounds[1] + 1))
            else:
                found.extend(bounds)
    return list(dict.fromkeys(found))


def plain_text(sentence: str) -> str:
    """Sentence without citation markers and markdown emphasis."""
    text = _MARKER.sub("", sentence)
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def is_claim(sentence: str) -> bool:
    """Whether a sentence makes a checkable clinical statement.

    Questions, greetings, empathy, lead-ins ending with ':', "talk to your
    doctor"/disclaimer boilerplate and emergency-safety advice are not claims.
    """
    text = plain_text(sentence)
    if len(text.split()) < 3 or text.endswith("?") or text.endswith(":"):
        return False
    return not (_NON_CLAIM_START.search(text) or _BOILERPLATE.search(text))


def evidence_windows(text: str) -> list[str]:
    """Overlapping windows of ~3 sentences (≤ ~350 tokens) used as NLI premises."""
    sentences = split_sentences(re.sub(r"\s+", " ", text))
    if not sentences:
        return []
    windows: list[str] = []
    last_start = max(0, len(sentences) - WINDOW_SENTENCES)
    for begin in range(0, last_start + 1, WINDOW_STRIDE):
        windows.append(" ".join(sentences[begin:begin + WINDOW_SENTENCES])[:WINDOW_MAX_CHARS])
    if last_start % WINDOW_STRIDE:
        windows.append(" ".join(sentences[last_start:])[:WINDOW_MAX_CHARS])
    return windows


@dataclass
class _Line:
    prefix: str
    sentences: list[str]
    structural: bool  # blank line or heading/lead-in — kept as is


def _segment(text: str) -> list[_Line]:
    lines: list[_Line] = []
    for raw in text.split("\n"):
        if not raw.strip() or _HEADING.match(raw):
            lines.append(_Line(raw, [], True))
            continue
        item = _LIST_ITEM.match(raw)
        prefix, content = (item.group(1), item.group(2)) if item else ("", raw)
        lines.append(_Line(prefix, split_sentences(content), False))
    return lines


def _is_lead_in(line: str) -> bool:
    return bool(_HEADING.match(line)) or line.rstrip().endswith(":")


def _rebuild(lines: list[_Line], drop: set[tuple[int, int]]) -> str:
    """Reassemble markdown without dropped sentences.

    Lines whose sentences were all dropped disappear (no empty bullets), and a
    heading or ``...:`` lead-in is removed when every line of the block it
    introduces was removed.
    """
    texts: list[str | None] = []  # None = line removed
    for li, line in enumerate(lines):
        if line.structural:
            texts.append(line.prefix)
            continue
        kept = [s for si, s in enumerate(line.sentences) if (li, si) not in drop]
        texts.append(line.prefix + " ".join(kept) if kept else None)
    emptied = [text is None for text in texts]

    def is_blank(i: int) -> bool:
        return not emptied[i] and not (texts[i] or "").strip()

    def is_lead_in(i: int) -> bool:
        text = texts[i]
        return bool(text and text.strip()) and _is_lead_in(text or "")

    out: list[str] = []
    for i, text in enumerate(texts):
        if text is None:
            continue
        if is_lead_in(i):
            j = i + 1
            while j < len(texts) and is_blank(j):
                j += 1
            block: list[int] = []
            while j < len(texts) and not is_blank(j) and not is_lead_in(j):
                block.append(j)
                j += 1
            if block and all(emptied[k] for k in block):
                continue
        out.append(text)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def _select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GroundingVerifier:
    """NLI-based claim verifier (Gate 3). The model is loaded lazily and thread-safely."""

    def __init__(
        self,
        model_name: str | None = None,
        threshold: float | None = None,
        *,
        model: NLIModel | None = None,
        device: str | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.GROUNDING_NLI_MODEL
        self.threshold = (
            settings.GROUNDING_ENTAILMENT_THRESHOLD if threshold is None else threshold
        )
        self._device = device
        self._model: NLIModel | None = model
        self._labels: tuple[int, int] | None = None
        self._load_error: str | None = None
        self._lock = threading.Lock()

    # -- model ------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        """True once the NLI model is resident in memory."""
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        """Why the model could not be loaded (``None`` if it loaded or was never tried)."""
        return self._load_error

    def _load(self) -> NLIModel | None:
        if self._model is not None or self._load_error is not None:
            return self._model
        with self._lock:
            if self._model is None and self._load_error is None:
                try:
                    from sentence_transformers import CrossEncoder

                    device = self._device or _select_device()
                    self._model = CrossEncoder(
                        self.model_name, device=device, max_length=_MAX_SEQ_LENGTH
                    )
                    logger.info("grounding_model_loaded", model=self.model_name, device=device)
                except (ImportError, OSError, RuntimeError, ValueError) as exc:
                    self._load_error = f"{type(exc).__name__}: {exc}"
                    logger.error("grounding_model_unavailable", model=self.model_name,
                                 error=type(exc).__name__)
        return self._model

    def _label_indices(self, model: NLIModel) -> tuple[int, int]:
        """(entailment, contradiction) column indices read from ``config.id2label``."""
        if self._labels is None:
            raw = getattr(getattr(model, "config", None), "id2label", None) or {}
            id2label = {int(k): str(v).lower() for k, v in dict(raw).items()}
            if not any("entail" in v for v in id2label.values()):
                logger.warning("grounding_model_unnamed_labels", model=self.model_name)
                id2label = dict(_DEFAULT_LABELS)
            entail = next(i for i, v in id2label.items() if "entail" in v)
            contra = next((i for i, v in id2label.items() if "contradict" in v), -1)
            self._labels = (entail, contra)
        return self._labels

    def _predict(self, pairs: list[tuple[str, str]]) -> list[tuple[float, float]]:
        model = self._load()
        if model is None or not pairs:
            return []
        entail, contra = self._label_indices(model)
        probabilities = model.predict(pairs, apply_softmax=True, show_progress_bar=False)
        rows: list[tuple[float, float]] = []
        for row in probabilities:
            values = [float(v) for v in row]
            rows.append((values[entail], values[contra] if contra >= 0 else 0.0))
        return rows

    async def warmup(self) -> bool:
        """Load the NLI model and run one inference; True when it is usable."""
        rows = await asyncio.to_thread(
            self._predict, [("Blood pressure is high.", "Blood pressure is high.")]
        )
        return bool(rows)

    # -- verification -----------------------------------------------------

    async def verify(
        self,
        response_text: str,
        chunks: Sequence[RetrievedChunk] | Sequence[Any],
    ) -> GroundingResult:
        """Check every claim sentence of ``response_text`` against its cited chunks.

        Never raises for model problems: if the NLI model cannot be loaded the
        result has ``available=False`` and the text is returned unchanged.
        """
        started = time.perf_counter()
        text = response_text or ""
        prefix = get_loop_breaker_prefix()
        lead = ""
        if prefix and text.startswith(prefix):
            lead, text = prefix, text[len(prefix):]

        lines = _segment(text)
        chunk_windows = [evidence_windows(chunk_text(c)) for c in chunks]
        all_windows = [w for windows in chunk_windows for w in windows]

        records: list[dict[str, Any]] = []
        positions: list[tuple[int, int]] = []
        pairs: list[tuple[str, str]] = []
        spans: list[tuple[int, int]] = []  # pair range per claim record
        for li, line in enumerate(lines):
            for si, sentence in enumerate(line.sentences):
                markers = parse_markers(sentence)
                valid = [m for m in markers if 1 <= m <= len(chunk_windows)]
                claim = is_claim(sentence)
                records.append({
                    "text": sentence, "markers": markers, "cited": bool(valid),
                    "is_claim": claim, "supported": True,
                    "entailment": 0.0, "contradiction": 0.0,
                })
                positions.append((li, si))
                start = len(pairs)
                if claim:
                    premises = (
                        [w for m in valid for w in chunk_windows[m - 1]] if valid else all_windows
                    )
                    hypothesis = plain_text(sentence)
                    pairs.extend((premise, hypothesis) for premise in premises)
                spans.append((start, len(pairs)))

        scores = await asyncio.to_thread(self._predict, pairs) if pairs else []
        available = bool(scores) or not pairs
        drop: set[tuple[int, int]] = set()
        unsupported: list[str] = []
        for record, (begin, end), position in zip(records, spans, positions, strict=True):
            if not record["is_claim"] or not available:
                continue
            window_scores = scores[begin:end]
            record["entailment"] = round(max((e for e, _ in window_scores), default=0.0), 4)
            record["contradiction"] = round(max((c for _, c in window_scores), default=0.0), 4)
            record["supported"] = record["entailment"] >= self.threshold
            if not record["supported"]:
                drop.add(position)
                unsupported.append(record["text"])

        n_claims = sum(1 for r in records if r["is_claim"])
        cleaned = _rebuild(lines, drop)
        if lead:
            cleaned = f"{lead}\n\n{cleaned}".strip() if cleaned else lead
        result = GroundingResult(
            sentences=records,
            n_claims=n_claims,
            n_unsupported=len(unsupported),
            unsupported_fraction=round(len(unsupported) / n_claims, 4) if n_claims else 0.0,
            cleaned_text=cleaned if available else response_text,
            model=self.model_name,
            available=available,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 1),
            unsupported_sentences=unsupported,
        )
        logger.info(
            "grounding_verified",
            n_claims=result.n_claims,
            n_unsupported=result.n_unsupported,
            available=available,
            latency_ms=result.latency_ms,
        )
        return result


_verifier_instance: GroundingVerifier | None = None


def get_grounding_verifier() -> GroundingVerifier:
    """Return the process-wide GroundingVerifier singleton."""
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = GroundingVerifier()
    return _verifier_instance
