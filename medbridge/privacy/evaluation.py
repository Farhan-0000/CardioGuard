"""
Span-level evaluation of the PII redactor on a labelled test set.

Used by ``scripts/pii_eval.py`` (report generation) and by
``tests/privacy/test_redaction.py`` (quality gate). Gold spans are given as
substrings (``{"category": ..., "text": ..., "occurrence": 0}``) and resolved
to character offsets here, which keeps the JSON test set human-editable.

Two matching criteria are reported:

* **strict** — predicted span has exactly the gold offsets and category;
* **overlap** — predicted span overlaps a gold span of the same category.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from medbridge.privacy.redaction import CATEGORIES, PiiSpan, detect, redact

#: Fixed reference date so DOB → age replacements are reproducible.
EVAL_TODAY = date(2026, 10, 9)


@dataclass(frozen=True)
class GoldSpan:
    start: int
    end: int
    category: str


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "support": self.tp + self.fn,
        }


@dataclass
class EvalReport:
    """Aggregate metrics for one run over the test set."""

    n_examples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    strict: dict[str, Counts] = field(default_factory=dict)
    overlap: dict[str, Counts] = field(default_factory=dict)
    negatives_altered: list[str] = field(default_factory=list)
    protected_corrupted: list[dict[str, str]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    mean_latency_ms: float = 0.0

    def overall(self, mode: str = "strict") -> Counts:
        table = self.strict if mode == "strict" else self.overlap
        total = Counts()
        for c in table.values():
            total.tp += c.tp
            total.fp += c.fp
            total.fn += c.fn
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_examples": self.n_examples,
            "n_positive_examples": self.n_positive,
            "n_negative_examples": self.n_negative,
            "overall_strict": self.overall("strict").as_dict(),
            "overall_overlap": self.overall("overlap").as_dict(),
            "per_category_strict": {k: v.as_dict() for k, v in sorted(self.strict.items())},
            "per_category_overlap": {k: v.as_dict() for k, v in sorted(self.overlap.items())},
            "negatives_altered": self.negatives_altered,
            "protected_values_corrupted": self.protected_corrupted,
            "mean_latency_ms": round(self.mean_latency_ms, 4),
            "errors": self.errors,
        }


def load_testset(path: Path) -> list[dict[str, Any]]:
    """Load the ``examples`` list of a PII test-set JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    examples: list[dict[str, Any]] = data["examples"]
    return examples


def resolve_gold(example: dict[str, Any]) -> list[GoldSpan]:
    """Convert substring gold spans into character offsets."""
    text: str = example["text"]
    spans: list[GoldSpan] = []
    for gold in example.get("spans", []):
        needle: str = gold["text"]
        occurrence = int(gold.get("occurrence", 0))
        idx = -1
        for _ in range(occurrence + 1):
            idx = text.find(needle, idx + 1)
            if idx < 0:
                raise ValueError(f"{example['id']}: gold span {needle!r} not found in text")
        spans.append(GoldSpan(idx, idx + len(needle), gold["category"]))
    return spans


def _score(
    gold: list[GoldSpan], pred: list[PiiSpan], table: dict[str, Counts], strict: bool
) -> list[tuple[str, str]]:
    """Update ``table`` with TP/FP/FN; return (kind, description) error tuples."""
    errors: list[tuple[str, str]] = []
    matched_pred: set[int] = set()
    for g in gold:
        hit = None
        for i, p in enumerate(pred):
            if i in matched_pred or p.category != g.category:
                continue
            same = (p.start, p.end) == (g.start, g.end)
            if same or (not strict and p.start < g.end and g.start < p.end):
                hit = i
                break
        if hit is None:
            table.setdefault(g.category, Counts()).fn += 1
            errors.append(("FN", f"{g.category}@{g.start}:{g.end}"))
        else:
            matched_pred.add(hit)
            table.setdefault(g.category, Counts()).tp += 1
    for i, p in enumerate(pred):
        if i not in matched_pred:
            table.setdefault(p.category, Counts()).fp += 1
            errors.append(("FP", f"{p.category}@{p.start}:{p.end}"))
    return errors


def evaluate(examples: list[dict[str, Any]], today: date = EVAL_TODAY) -> EvalReport:
    """Run the redactor over ``examples`` and compute span-level metrics."""
    report = EvalReport(
        strict={c: Counts() for c in CATEGORIES},
        overlap={c: Counts() for c in CATEGORIES},
    )
    elapsed = 0.0
    for ex in examples:
        text: str = ex["text"]
        gold = resolve_gold(ex)
        t0 = time.perf_counter()
        pred = detect(text, today=today)
        elapsed += time.perf_counter() - t0
        redacted = redact(text, today=today).text

        report.n_examples += 1
        if ex.get("kind") == "negative":
            report.n_negative += 1
            if redacted != text:
                report.negatives_altered.append(ex["id"])
        else:
            report.n_positive += 1
        for value in ex.get("protected", []):
            if value in text and value not in redacted:
                report.protected_corrupted.append({"id": ex["id"], "value": value})

        strict_errors = _score(gold, pred, report.strict, strict=True)
        _score(gold, pred, report.overlap, strict=False)
        if strict_errors:
            report.errors.append({
                "id": ex["id"],
                "text": text,
                "errors": [
                    {"kind": kind, "span": desc, "value": _span_text(text, desc)}
                    for kind, desc in strict_errors
                ],
                "redacted": redacted,
            })
    report.mean_latency_ms = 1000.0 * elapsed / max(report.n_examples, 1)
    return report


def _span_text(text: str, desc: str) -> str:
    start, end = desc.split("@", 1)[1].split(":")
    return text[int(start):int(end)]
