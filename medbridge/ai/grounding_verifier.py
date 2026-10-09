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
_HEADING_MARK = re.compile(r"^(\s*#{1,6}\s+)(.*)$")
# A line whose content is wholly bold/underlined: "**...**", "__...__:".
_WRAPPED = re.compile(r"^(\*\*|__)(.+?)\1(:?)\s*$")
# A heading of this many words or more (or one with a [n] marker or a full stop)
# is a sentence written as a heading, and is verified like any other sentence.
HEADING_MAX_WORDS = 6
# Greeting / empathy / signposting lead-ins. Word boundaries matter: "hi" must not
# match "High blood pressure ...". The rest of the sentence is still checked.
_LEAD_IN = re.compile(
    r"^(?:(?:hi|hello|hey|thanks)\b|thank you\b|you're welcome\b|i hope (?:this|that) helps\b|"
    r"(?:i'm |i am )?sorry to hear\b|(?:great|good) question\b|i understand\b|i know\b|"
    r"it's understandable\b|it is understandable\b|it can (?:feel|be) (?:worrying|stressful|"
    r"confusing|overwhelming)\b|it's (?:great|good) that you\b|here (?:is|are)\b)",
    re.IGNORECASE,
)
# Disclaimer / referral clauses. A clause that matches (and carries no number with a
# unit) is boilerplate; the rest of its sentence is still checked.
_BOILERPLATE = re.compile(
    r"^(?:please\s+)?(?:always\s+)?(?:consult|talk to|talk with|speak to|speak with|discuss|"
    r"check with|ask|contact|see|reach out to)\b.*\b(?:doctor|clinician|healthcare|"
    r"health care|gp|physician|pharmacist|care team|nurse|provider)\b"
    r"|\bnot (?:a substitute for|medical advice|personalised|personalized)\b"
    r"|\b(?:general|educational) information\b"
    r"|\binformational purposes\b"
    r"|\bi(?: am|'m) not a doctor\b",
    re.IGNORECASE,
)
# Advice to get emergency care. Such sentences are safety advice: never checked and
# never removed, whatever else they contain.
_EMERGENCY_ADVICE = re.compile(
    r"\b(?:call|dial|phone|ring|contact)\s+(?:\d{3}\b|an ambulance|(?:your local )?emergency)"
    r"|\b(?:go|get|head)\s+(?:straight\s+|immediately\s+|right away\s+)?to\s+(?:the\s+|an\s+|a\s+"
    r"|your nearest\s+|the nearest\s+)?(?:emergency|er\b|a&e\b|hospital)"
    r"|\b(?:seek|get) (?:immediate|urgent|emergency) (?:medical )?(?:care|help|attention|treatment)\b"
    r"|\bemergency services\b",
    re.IGNORECASE,
)
# A number with a clinical unit ("20 mg", "130/80 mmHg", "5 %", "2 weeks").
_NUMBER_UNIT = re.compile(
    r"\b\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?\s*(?:mg|mcg|µg|g|kg|ml|l|mmhg|mmol|mg/dl|%|percent|"
    r"bpm|units?|iu|tablets?|times|hours?|days?|weeks?|months?|years?|cups?|drinks?|kcal|"
    r"mins?|minutes?)\b|\b\d{2,3}\s*/\s*\d{2,3}\b",
    re.IGNORECASE,
)
_CLAUSE_SPLIT = re.compile(r"(?<=[,:;])\s+|\s+[–—-]\s+")


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


def _strip_boilerplate(text: str) -> tuple[str, bool]:
    """``text`` without lead-ins and boilerplate clauses, and whether anything was removed."""
    exempt = False
    lead = _LEAD_IN.match(text)
    if lead:
        exempt = True
        text = re.sub(r"^[\s,:;!.–—-]*(?:that\s+)?", "", text[lead.end():], flags=re.IGNORECASE)
    kept: list[str] = []
    for clause in _CLAUSE_SPLIT.split(text):
        body = clause.strip(" ,:;–—-")
        if body and _BOILERPLATE.search(body) and not _NUMBER_UNIT.search(body):
            exempt = True
            continue
        kept.append(clause)
    return " ".join(kept).strip(" ,:;–—-"), exempt


def claim_text(sentence: str) -> str | None:
    """The checkable clinical statement in a sentence, or ``None`` if it makes none.

    Questions, lead-ins ending with ':' and emergency-care advice are never
    claims. Greetings, empathy, "talk to your doctor" referrals and disclaimers
    are removed from the sentence first; when that leaves text *and* the sentence
    carries a citation marker or a number with a unit, the remainder is the claim
    ("This is general information: X [1]" → "X"). Sentences without such
    boilerplate are claims as written.
    """
    text = plain_text(sentence)
    if len(text.split()) < 3 or text.endswith("?") or text.endswith(":"):
        return None
    if _EMERGENCY_ADVICE.search(text):
        return None
    remainder, exempt = _strip_boilerplate(text)
    if not exempt:
        return text
    has_signal = bool(_MARKER.search(sentence)) or bool(_NUMBER_UNIT.search(text))
    if not has_signal or len(remainder.split()) < 3:
        return None
    return remainder


def is_claim(sentence: str) -> bool:
    """Whether a sentence makes a checkable clinical statement (see :func:`claim_text`)."""
    return claim_text(sentence) is not None


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
    structural: bool  # blank line or short heading/lead-in — kept as is
    suffix: str = ""  # closing emphasis of a wholly bold line ("**")


def _is_short_heading(content: str) -> bool:
    text = plain_text(content)
    if _MARKER.search(content) or text.endswith((".", "!")):
        return False
    return len(text.split()) < HEADING_MAX_WORDS or text.endswith(":")


def _segment(text: str) -> list[_Line]:
    """Split markdown into lines of sentences.

    Short headings and bold lead-ins stay structural; a heading or wholly bold
    line that is really a sentence (long, cited or ending with a full stop) is
    split and verified like any other line, its markers kept around the text.
    """
    lines: list[_Line] = []
    for raw in text.split("\n"):
        if not raw.strip():
            lines.append(_Line(raw, [], True))
            continue
        heading = _HEADING_MARK.match(raw)
        item = None if heading else _LIST_ITEM.match(raw)
        match = heading or item
        prefix, content = (match.group(1), match.group(2)) if match else ("", raw)
        suffix = ""
        wrapped = _WRAPPED.match(content.strip())
        if wrapped:
            prefix += content[:len(content) - len(content.lstrip())] + wrapped.group(1)
            content, suffix = wrapped.group(2), wrapped.group(1) + wrapped.group(3)
        if (heading or wrapped) and _is_short_heading(content):
            lines.append(_Line(raw, [], True))
            continue
        lines.append(_Line(prefix, split_sentences(content), False, suffix))
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
        texts.append(line.prefix + " ".join(kept) + line.suffix if kept else None)
    emptied = [text is None for text in texts]

    def is_blank(i: int) -> bool:
        return not emptied[i] and not (texts[i] or "").strip()

    def is_lead_in(i: int) -> bool:
        text = texts[i]
        if not (text and text.strip()):
            return False
        return (lines[i].structural and _is_lead_in(text)) or text.rstrip().endswith(":")

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
                hypothesis = claim_text(sentence)
                claim = hypothesis is not None
                records.append({
                    "text": sentence, "markers": markers, "cited": bool(valid),
                    "is_claim": claim, "supported": True,
                    "entailment": 0.0, "contradiction": 0.0,
                })
                positions.append((li, si))
                start = len(pairs)
                if hypothesis is not None:
                    premises = (
                        [w for m in valid for w in chunk_windows[m - 1]] if valid else all_windows
                    )
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
