"""
Results context: every real results / measurement file of the repository in one dict.

The report never hand-copies a number. Inline values (``{{path|filter}}``) and the
generated tables read from this context, which is built from files written by the
evaluation scripts. A missing or unreadable file simply leaves its key absent (the
report then shows a highlighted ``[PENDING: ...]`` marker).

Top-level keys (source files are repo-relative):

=====================  ===========================================================
study                  evaluation/results/summary.json (hallucination study, analyze.py)
retrieval              evaluation/results/retrieval/metrics.json
reranker               evaluation/results/retrieval/reranker_ablation/<model>/metrics.json
model_selection        evaluation/results/model_selection/summary.json
model_selection_r1     evaluation/results/model_selection_round1_prefix/summary.json
corpus                 reports/corpus_stats.json
pii                    reports/privacy_redaction_metrics.json
perf                   reports/performance/<stem>.json (every JSON file, by stem)
perf_latest            {"before": ..., "after": ...} newest latency-probe file per label
aq                     reports/medbridge_aq_benchmark_report.json (LIVE runs only)
rabbits                reports/rabbits_benchmark_report.json (LIVE runs only)
adversarial            reports/adversarial_live_report.json (LIVE runs only)
load                   reports/performance/load_test.json
tests                  reports/test_summary.json (or JUnit XML in reports/tests/)
calibration            evaluation/results/calibration.json
human_agreement        evaluation/results/human_agreement.json
smoke                  reports/smoke_test_latest.json
hallubench             statistics computed from evaluation/datasets/hallubench_v1.json
manifest               data/corpus_manifest.json
manifest_stats         statistics computed from the manifest
=====================  ===========================================================

Mock-mode benchmark reports (routing simulator, no LLM call) are deliberately NOT
loaded: they are a harness self-test, not evidence of model behaviour.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

#: Single-file sources: key -> repo-relative path.
SINGLE_SOURCES: dict[str, str] = {
    "study": "evaluation/results/summary.json",
    "retrieval": "evaluation/results/retrieval/metrics.json",
    "model_selection": "evaluation/results/model_selection/summary.json",
    "model_selection_r1": "evaluation/results/model_selection_round1_prefix/summary.json",
    "corpus": "reports/corpus_stats.json",
    "pii": "reports/privacy_redaction_metrics.json",
    "load": "reports/performance/load_test.json",
    "tests": "reports/test_summary.json",
    "calibration": "evaluation/results/calibration.json",
    "human_agreement": "evaluation/results/human_agreement.json",
    "smoke": "reports/smoke_test_latest.json",
    "manifest": "data/corpus_manifest.json",
}

#: Benchmark reports that are only meaningful when produced in live mode.
LIVE_ONLY_SOURCES: dict[str, str] = {
    "aq": "reports/medbridge_aq_benchmark_report.json",
    "rabbits": "reports/rabbits_benchmark_report.json",
    "adversarial": "reports/adversarial_live_report.json",
}

RERANKER_DIR = "evaluation/results/retrieval/reranker_ablation"
PERF_DIR = "reports/performance"
HALLUBENCH_PATH = "evaluation/datasets/hallubench_v1.json"
JUNIT_DIR = "reports/tests"


class Missing(KeyError):
    """Raised by :meth:`ResultsContext.get` when a path does not resolve."""


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _mode_of(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    mode = data.get("mode")
    if mode is None and isinstance(data.get("metrics"), dict):
        mode = data["metrics"].get("mode")
    if mode is None and isinstance(data.get("metadata"), dict):
        mode = data["metadata"].get("mode")
    return str(mode).lower() if mode is not None else None


# ----------------------------------------------------------------------
# Derived statistics
# ----------------------------------------------------------------------


def hallubench_stats(data: Any) -> dict[str, Any] | None:
    """Composition of MedBridge-HalluBench (items, turns, splits, categories, gold actions)."""
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return None
    items = [i for i in data["items"] if isinstance(i, dict)]
    by_category: dict[str, dict[str, int]] = {}
    gold: Counter[str] = Counter()
    styles: Counter[str] = Counter()
    subcats: dict[str, Counter[str]] = {}
    tags: Counter[str] = Counter()
    n_turns = 0
    multi = 0
    turn_hist: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    for item in items:
        cat = str(item.get("category", "unknown"))
        split = str(item.get("split", "test"))
        turns = [t for t in item.get("turns") or [] if isinstance(t, dict)]
        row = by_category.setdefault(
            cat, {"items": 0, "turns": 0, "dev": 0, "test": 0, "multi_turn": 0}
        )
        row["items"] += 1
        row["turns"] += len(turns)
        row[split] = row.get(split, 0) + 1
        if len(turns) > 1:
            row["multi_turn"] += 1
            multi += 1
        n_turns += len(turns)
        turn_hist[len(turns)] += 1
        split_counts[split] += 1
        for turn in turns:
            if turn.get("gold_action"):
                gold[str(turn["gold_action"])] += 1
        if item.get("style"):
            styles[str(item["style"])] += 1
        if item.get("subcategory"):
            subcats.setdefault(cat, Counter())[str(item["subcategory"])] += 1
        for tag in item.get("tags") or []:
            tags[str(tag)] += 1
    return {
        "name": data.get("name"),
        "version": data.get("version"),
        "n_items": len(items),
        "n_turns": n_turns,
        "n_multi_turn_items": multi,
        "n_single_turn_items": len(items) - multi,
        "turns_per_item": {str(k): v for k, v in sorted(turn_hist.items())},
        "by_split": dict(split_counts),
        "by_category": by_category,
        "by_gold_action": dict(gold),
        "by_style": dict(styles),
        "by_subcategory": {k: dict(v) for k, v in subcats.items()},
        "by_tag": dict(tags),
    }


def manifest_stats(data: Any) -> dict[str, Any] | None:
    """Counts over ``data/corpus_manifest.json`` (documents per type/org/format, year range)."""
    if not isinstance(data, dict) or not isinstance(data.get("documents"), list):
        return None
    docs = [d for d in data["documents"] if isinstance(d, dict)]
    years = [int(d["year"]) for d in docs if str(d.get("year", "")).isdigit()]
    topics = Counter(t for d in docs for t in d.get("topics") or [])
    return {
        "documents": len(docs),
        "by_doc_type": dict(Counter(str(d.get("doc_type")) for d in docs)),
        "by_org": dict(Counter(str(d.get("org")) for d in docs)),
        "by_format": dict(Counter(str(d.get("source_format")) for d in docs)),
        "organisations": len({str(d.get("org")) for d in docs}),
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
        "pages": sum(int(d.get("pages") or 0) for d in docs),
        "chunks": sum(int(d.get("chunk_count") or 0) for d in docs),
        "table_chunks": sum(int(d.get("table_chunk_count") or 0) for d in docs),
        "topics": dict(topics),
    }


def junit_summary(paths: Iterable[Path]) -> dict[str, Any] | None:
    """Test summary from JUnit XML files (one suite per file stem)."""
    suites: list[dict[str, Any]] = []
    for path in sorted(paths):
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError):
            continue
        nodes = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        agg = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
        for node in nodes:
            for key in ("tests", "failures", "errors", "skipped"):
                agg[key] += int(float(node.get(key, 0) or 0))
            agg["time"] += float(node.get("time", 0) or 0)
        total = agg["tests"]
        suites.append(
            {
                "name": path.stem,
                "total": total,
                "passed": total - agg["failures"] - agg["errors"] - agg["skipped"],
                "failed": agg["failures"],
                "errors": agg["errors"],
                "skipped": agg["skipped"],
                "duration_s": round(agg["time"], 1),
                "source": path.as_posix(),
            }
        )
    if not suites:
        return None
    totals = {
        k: sum(int(s[k]) for s in suites)
        for k in ("total", "passed", "failed", "errors", "skipped")
    }
    return {"suites": suites, "totals": totals, "source": "junit"}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def load_context(repo_root: Path) -> dict[str, Any]:
    """Load every source defensively; returns ``{key: data}`` plus ``_sources``/``_skipped``."""
    ctx: dict[str, Any] = {}
    sources: dict[str, str] = {}
    skipped: dict[str, str] = {}

    for key, rel in SINGLE_SOURCES.items():
        data = _read_json(repo_root / rel)
        if data is not None:
            ctx[key] = data
            sources[key] = rel

    for key, rel in LIVE_ONLY_SOURCES.items():
        data = _read_json(repo_root / rel)
        if data is None:
            continue
        mode = _mode_of(data)
        if mode == "live":
            ctx[key] = data
            sources[key] = rel
        else:
            skipped[key] = f"{rel} is a '{mode}' run (not live) - not used as a result"

    reranker: dict[str, Any] = {}
    for path in sorted((repo_root / RERANKER_DIR).glob("*/metrics.json")):
        data = _read_json(path)
        if data is not None:
            reranker[path.parent.name] = data
    if reranker:
        ctx["reranker"] = reranker
        sources["reranker"] = f"{RERANKER_DIR}/*/metrics.json"

    perf: dict[str, Any] = {}
    latest: dict[str, tuple[float, Any, str]] = {}
    for path in sorted((repo_root / PERF_DIR).glob("*.json")):
        data = _read_json(path)
        if data is None:
            continue
        perf[path.stem] = data
        if isinstance(data, dict) and "warm_summary" in data and data.get("label"):
            label = str(data["label"])
            mtime = path.stat().st_mtime
            if label not in latest or mtime > latest[label][0]:
                latest[label] = (mtime, data, path.relative_to(repo_root).as_posix())
    if perf:
        ctx["perf"] = perf
        sources["perf"] = f"{PERF_DIR}/*.json"
    if latest:
        ctx["perf_latest"] = {label: v[1] for label, v in latest.items()}
        ctx["perf_latest_sources"] = {label: v[2] for label, v in latest.items()}
        sources["perf_latest"] = ", ".join(v[2] for v in latest.values())

    if "tests" not in ctx:
        junit = junit_summary((repo_root / JUNIT_DIR).glob("*.xml"))
        if junit:
            ctx["tests"] = junit
            sources["tests"] = f"{JUNIT_DIR}/*.xml"

    bench = hallubench_stats(_read_json(repo_root / HALLUBENCH_PATH))
    if bench:
        ctx["hallubench"] = bench
        sources["hallubench"] = HALLUBENCH_PATH

    mstats = manifest_stats(ctx.get("manifest"))
    if mstats:
        ctx["manifest_stats"] = mstats
        sources["manifest_stats"] = SINGLE_SOURCES["manifest"]

    ctx["_sources"] = sources
    ctx["_skipped"] = skipped
    return ctx


# ----------------------------------------------------------------------
# Path resolution and formatting
# ----------------------------------------------------------------------

_BRACKET = re.compile(r"\[(?:\"([^\"]*)\"|'([^']*)'|([^\]]*))\]")


def _split_path(path: str) -> list[str]:
    """Split ``a.b["x.y"].c`` into raw segments; dotted keys are re-joined greedily later."""
    out: list[str] = []
    buf = ""
    i = 0
    while i < len(path):
        ch = path[i]
        if ch == "[":
            m = _BRACKET.match(path, i)
            if m:
                if buf:
                    out.extend(buf.split("."))
                    buf = ""
                out.append("\x00" + next(g for g in m.groups() if g is not None))
                i = m.end()
                if i < len(path) and path[i] == ".":
                    i += 1
                continue
        buf += ch
        i += 1
    if buf:
        out.extend(buf.split("."))
    return [s for s in out if s != ""]


def resolve(data: Any, path: str) -> Any:
    """Resolve a dotted path; keys that themselves contain dots (``qwen3.5:4b``) match greedily."""
    segments = _split_path(path.strip())
    node = data
    i = 0
    while i < len(segments):
        seg = segments[i]
        if seg.startswith("\x00"):  # bracketed literal key
            key = seg[1:]
            if isinstance(node, dict) and key in node:
                node = node[key]
            elif isinstance(node, list) and key.lstrip("-").isdigit():
                try:
                    node = node[int(key)]
                except IndexError as exc:
                    raise Missing(path) from exc
            else:
                raise Missing(path)
            i += 1
            continue
        if isinstance(node, dict):
            for j in range(len(segments), i, -1):
                parts = segments[i:j]
                if any(p.startswith("\x00") for p in parts):
                    continue
                key = ".".join(parts)
                if key in node:
                    node = node[key]
                    i = j
                    break
            else:
                raise Missing(path)
        elif isinstance(node, list) and seg.lstrip("-").isdigit():
            try:
                node = node[int(seg)]
            except IndexError as exc:
                raise Missing(path) from exc
            i += 1
        else:
            raise Missing(path)
    return node


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and (math.isnan(value) or math.isinf(value))


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, dict):
        for key in ("rate", "estimate", "kappa", "value"):
            if key in value and value[key] is not None:
                return float(value[key])
    return float(str(value).strip().rstrip("%"))


def _ci(value: Any, digits: int = 1, percent: bool = True) -> str:
    """``{rate, low, high}`` -> ``74.3% [65.0, 82.6]``; ``{estimate, low, high}`` -> ``0.743 [...]``."""
    if not isinstance(value, dict):
        return format_value(value, ["pct" if percent else "f3"])
    est = value.get("rate", value.get("estimate", value.get("kappa")))
    low, high = value.get("low"), value.get("high")
    if est is None or _is_nan(est):
        return "n/a"
    is_rate = "rate" in value
    if is_rate and percent:
        text = f"{100 * float(est):.{digits}f}%"
        if low is not None and high is not None and not (_is_nan(low) or _is_nan(high)):
            text += f" [{100 * float(low):.{digits}f}, {100 * float(high):.{digits}f}]"
        return text
    d = max(digits, 3) if digits <= 1 else digits
    text = f"{float(est):.{d}f}"
    if low is not None and high is not None and not (_is_nan(low) or _is_nan(high)):
        text += f" [{float(low):.{d}f}, {float(high):.{d}f}]"
    return text


def format_value(value: Any, filters: list[str] | None = None) -> str:
    """Apply formatting filters (see README "Inline values") to a resolved value."""
    filters = [f.strip() for f in (filters or []) if f.strip()]
    if value is None or _is_nan(value):
        return "n/a"
    if not filters:
        if isinstance(value, dict) and "rate" in value:
            return _ci(value)
        if isinstance(value, float):
            return f"{value:.3f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:.0f}"
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        if isinstance(value, dict):
            return ", ".join(f"{k}: {v}" for k, v in value.items())
        return str(value)
    out: Any = value
    for flt in filters:
        name, _, arg = flt.partition(":")
        name = name.lower()
        if name in {"pct", "pct0", "pct1", "pct2"}:
            digits = {"pct": 1, "pct0": 0, "pct1": 1, "pct2": 2}[name]
            out = f"{100 * _as_float(out):.{digits}f}%"
        elif name in {"perc", "percent"}:  # already a percentage (0-100)
            out = f"{_as_float(out):.1f}%"
        elif re.fullmatch(r"f\d", name):
            out = f"{_as_float(out):.{int(name[1])}f}"
        elif name == "int":
            out = f"{round(_as_float(out)):d}"
        elif name == "comma":
            out = f"{round(_as_float(out)):,d}"
        elif name == "s" or name == "ms2s":  # milliseconds -> seconds
            out = f"{_as_float(out) / 1000:.1f}"
        elif name == "ci":
            out = _ci(out, int(arg) if arg.isdigit() else 1)
        elif name == "cif":  # interval as plain numbers
            out = _ci(out, int(arg) if arg.isdigit() else 3, percent=False)
        elif name == "len":
            out = str(len(out)) if hasattr(out, "__len__") else "n/a"
        elif name == "upper":
            out = str(out).upper()
        elif name == "lower":
            out = str(out).lower()
        elif name == "date":
            out = str(out)[:10]
        elif name == "keys":
            out = ", ".join(str(k) for k in out) if isinstance(out, dict) else str(out)
        else:
            raise ValueError(f"unknown filter '{flt}'")
    return str(out)


class ResultsContext:
    """Thin wrapper with ``get(path)`` and ``render(expr)`` over :func:`load_context`."""

    def __init__(self, data: dict[str, Any], repo_root: Path | None = None) -> None:
        self.data = data
        self.repo_root = repo_root
        self.used: list[str] = []
        self.missing: list[str] = []

    @classmethod
    def load(cls, repo_root: Path) -> ResultsContext:
        return cls(load_context(repo_root), repo_root)

    def has(self, path: str) -> bool:
        try:
            resolve(self.data, path)
            return True
        except Missing:
            return False

    def get(self, path: str, default: Any = None, *, raise_missing: bool = False) -> Any:
        try:
            return resolve(self.data, path)
        except Missing:
            if raise_missing:
                raise
            return default

    def source(self, key: str) -> str | None:
        return (self.data.get("_sources") or {}).get(key.split(".")[0])

    def render(self, expr: str) -> tuple[str, bool]:
        """Render ``path|filter|filter`` -> (text, ok); ok=False means pending/missing."""
        path, *filters = [p.strip() for p in expr.split("|")]
        try:
            value = resolve(self.data, path)
        except Missing:
            self.missing.append(path)
            return f"[PENDING: {path}]", False
        try:
            text = format_value(value, filters)
        except (ValueError, TypeError) as exc:
            self.missing.append(f"{expr} ({exc})")
            return f"[PENDING: {expr}]", False
        self.used.append(path)
        return text, True
