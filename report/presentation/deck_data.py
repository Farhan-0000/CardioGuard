"""
Facts for the review deck that are computed from the repository when the deck is built.

The results context (``report/lib/results_context.py``) covers the evaluation and
measurement files. This module adds the facts that live in code or test data instead,
so that no slide types them in by hand:

* the emergency classifier run over its two labelled case files (deterministic, no LLM;
  these files were written alongside the rules, so the counts are *in-sample*);
* the example messages shown on the emergency slide, classified for real;
* harness sizes: brand names in the drug lexicon, AQ vignettes, adversarial payloads;
* static test inventory (test files and test functions per suite);
* the received prototype's pytest baseline and the code-review findings;
* configuration defaults (application model, retrieval depths, loop-breaker threshold);
* the LLM-judge and NLI-judge model names and which UI screenshots exist.

Every fact is optional: a failure leaves its key absent, and the slide then shows a
highlighted ``[PENDING: ...]`` marker. Nothing here calls a language model or a server.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: Labelled emergency case files (repo-relative).
EMERGENCY_FILES = {
    "english": "tests/data/emergency_cases.json",
    "hinglish": "tests/data/emergency_cases_hinglish.json",
}

#: Canonical screenshot names (report/screenshots/<name>.png) and the file names the
#: capture script (medbridge/web/screenshots.py) writes for the same scene.
SCREENSHOTS: dict[str, tuple[str, ...]] = {
    "welcome": ("welcome-desktop-light",),
    "answer_citations": ("answer-sources-desktop-light",),
    "why_panel": ("why-expanded-desktop-light",),
    "soft_ask": ("softask-desktop-light",),
    "escalate": ("escalate-desktop-light",),
    "abstain": ("abstain-desktop-light",),
    "bp_trend": ("context-desktop-light",),
    "dashboard": ("dashboard-desktop-light",),
    "dark_mode": ("answer-sources-desktop-dark", "welcome-desktop-dark"),
}
SCREENSHOT_DIR = "report/screenshots"

#: Test suites for the static inventory: (label, directory, glob).
TEST_SUITES: tuple[tuple[str, str, str], ...] = (
    ("Unit (medbridge/tests/unit)", "medbridge/tests/unit", "test_*.py"),
    ("Integration (medbridge/tests/integration)", "medbridge/tests/integration", "test_*.py"),
    ("Privacy (tests/privacy)", "tests/privacy", "test_*.py"),
    ("Evaluation package (tests/evaluation)", "tests/evaluation", "test_*.py"),
    ("Adversarial and RABBITS (tests/adversarial)", "tests/adversarial", "test_*.py"),
    ("Scripts (tests/scripts)", "tests/scripts", "test_*.py"),
    ("Report tooling (tests/report)", "tests/report", "test_*.py"),
    ("Web UI, JavaScript (medbridge/web/tests)", "medbridge/web/tests", "*.test.mjs"),
)

_PY_TEST_RE = re.compile(r"^\s*(?:async\s+)?def\s+test_", re.MULTILINE)
_JS_TEST_RE = re.compile(r"^\s*(?:test|it)\(", re.MULTILINE)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _ensure_import_path(root: Path) -> None:
    for p in (root, Path(__file__).resolve().parents[2]):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _load_classifier(root: Path) -> Callable[[str], Any] | None:
    _ensure_import_path(root)
    try:
        from medbridge.core.emergency_classifier import classify
    except Exception:
        return None
    return classify


def _is_escalate(action: Any) -> bool:
    if action is None:
        return False
    return str(getattr(action, "value", action)).upper() == "ESCALATE"


# ----------------------------------------------------------------------
# Individual facts
# ----------------------------------------------------------------------


def emergency_case_stats(root: Path) -> dict[str, Any] | None:
    """Confusion counts of the deterministic classifier on each labelled case file."""
    classify = _load_classifier(root)
    if classify is None:
        return None
    out: dict[str, Any] = {}
    for key, rel in EMERGENCY_FILES.items():
        data = _read_json(root / rel)
        cases = data.get("cases") if isinstance(data, dict) else None
        if not isinstance(cases, list) or not cases:
            continue
        tp = fp = fn = tn = 0
        timings: list[float] = []
        for case in cases:
            text = str(case.get("text", ""))
            expected = str(case.get("expected") or "").upper() == "ESCALATE"
            start = time.perf_counter()
            got = _is_escalate(classify(text))
            timings.append((time.perf_counter() - start) * 1000.0)
            if expected and got:
                tp += 1
            elif expected:
                fn += 1
            elif got:
                fp += 1
            else:
                tn += 1
        timings.sort()
        out[key] = {
            "source": rel,
            "cases": len(cases),
            "positives": tp + fn,
            "negatives": tn + fp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "recall": tp / (tp + fn) if (tp + fn) else None,
            "max_ms_this_build": round(timings[-1], 2) if timings else None,
        }
    return out or None


def classify_examples(root: Path, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the classifier on the slide's example messages (the slide shows the real output)."""
    classify = _load_classifier(root)
    out: list[dict[str, Any]] = []
    for ex in examples:
        text = str(ex.get("text", ""))
        expected = str(ex.get("expected", "")).upper() or None
        if classify is None:
            out.append({"text": text, "expected": expected, "actual": None, "match": None})
            continue
        actual = "ESCALATE" if _is_escalate(classify(text)) else "PASS-THROUGH"
        out.append(
            {"text": text, "expected": expected, "actual": actual, "match": actual == expected}
        )
    return out


def drug_lexicon(root: Path) -> dict[str, Any] | None:
    _ensure_import_path(root)
    try:
        from medbridge.nlp import drugs
    except Exception:
        return None
    brands = getattr(drugs, "BRAND_TO_GENERIC", None)
    if not isinstance(brands, dict):
        return None
    indian: Any = getattr(drugs, "INDIAN_BRANDS", frozenset())
    generics = getattr(drugs, "GENERIC_CLASS", {})
    return {
        "brands": len(brands),
        "indian_brands": len(indian),
        "generics": len(generics) if isinstance(generics, dict) else None,
        "source": "medbridge/nlp/drugs.py",
    }


def adversarial_payloads(root: Path) -> dict[str, Any] | None:
    _ensure_import_path(root)
    try:
        from tests.adversarial.payloads import PAYLOADS
    except Exception:
        return None
    cats = Counter(str(getattr(p, "category", "other")) for p in PAYLOADS)
    return {
        "payloads": len(PAYLOADS),
        "categories": len(cats),
        "emergency_payloads": sum(1 for p in PAYLOADS if getattr(p, "expect_escalate", False)),
        "source": "tests/adversarial/payloads.py",
    }


def aq_vignettes(root: Path) -> dict[str, Any] | None:
    files = sorted((root / "tests" / "benchmark" / "vignettes").glob("*.json"))
    subsets: dict[str, int] = {}
    for path in files:
        data = _read_json(path)
        if isinstance(data, list):
            subsets[path.stem] = len(data)
        elif isinstance(data, dict):
            for key in ("vignettes", "items", "cases"):
                if isinstance(data.get(key), list):
                    subsets[path.stem] = len(data[key])
                    break
    if not subsets:
        return None
    return {
        "vignettes": sum(subsets.values()),
        "subsets": len(subsets),
        "by_subset": subsets,
        "source": "tests/benchmark/vignettes/*.json",
    }


def test_inventory(root: Path) -> dict[str, Any] | None:
    suites: list[dict[str, Any]] = []
    for label, rel, pattern in TEST_SUITES:
        folder = root / rel
        files = sorted(folder.glob(pattern)) if folder.is_dir() else []
        if not files:
            continue
        regex = _JS_TEST_RE if pattern.endswith(".mjs") else _PY_TEST_RE
        count = 0
        for f in files:
            try:
                count += len(regex.findall(f.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                continue
        suites.append({"name": label, "path": rel, "files": len(files), "tests_defined": count})
    if not suites:
        return None
    py = [s for s in suites if not s["path"].endswith("web/tests")]
    return {
        "suites": suites,
        "python_files": sum(s["files"] for s in py),
        "python_tests_defined": sum(s["tests_defined"] for s in py),
        "js_tests_defined": sum(s["tests_defined"] for s in suites if s not in py),
        "note": "static count of test functions in the source; parametrised cases expand further",
    }


def pytest_baseline(root: Path) -> dict[str, Any] | None:
    """Last pytest summary line of the received prototype (reports/baseline/pytest_baseline.txt)."""
    path = root / "reports" / "baseline" / "pytest_baseline.txt"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    summary = None
    for line in text.splitlines():
        if re.search(r"\b\d+ passed\b", line) and " in " in line:
            summary = line.strip()
    if summary is None:
        return None
    out: dict[str, Any] = {"source": _rel(path, root), "line": summary.strip("= ")}
    for key in ("passed", "failed", "errors", "skipped"):
        m = re.search(rf"(\d+) {key[:-1] if key == 'errors' else key}", summary)
        if m:
            out[key] = int(m.group(1))
    cov = re.search(r"^TOTAL\s+\d+\s+\d+\s+(\d+)%", text, re.MULTILINE)
    if cov:
        out["coverage_pct"] = int(cov.group(1))
    return out


def code_review(root: Path) -> dict[str, Any] | None:
    """Counts from docs/notes/code_review_findings.md (confirmed findings and severities)."""
    path = root / "docs" / "notes" / "code_review_findings.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    confirmed_part, _, rejected_part = text.partition("\nRejected")
    severities = Counter(
        m.group(1).lower()
        for m in re.finditer(
            r"^- \*\*[^*]+\*\* \((high|medium|low)\b", confirmed_part, re.MULTILINE
        )
    )
    confirmed = sum(severities.values())
    if not confirmed:
        return None
    commit = re.search(r"fixed in ([0-9a-f]{7,40})", text)
    rejected = len(re.findall(r"^- \S", rejected_part, re.MULTILINE))
    return {
        "confirmed": confirmed,
        "high": severities.get("high", 0),
        "medium": severities.get("medium", 0),
        "low": severities.get("low", 0),
        "rejected": rejected,
        "commit": commit.group(1) if commit else None,
        "source": _rel(path, root),
    }


_CONFIG_KEYS = (
    "LLM_MODEL",
    "RETRIEVAL_TOP_K_CANDIDATES",
    "RETRIEVAL_TOP_K_RERANKED",
    "RERANKER_MODEL",
    "EMBEDDING_MODEL",
    "SOFT_ASK_MAX_COUNT",
    "SPECULATIVE_RETRIEVAL",
    "LOCAL_ONLY_MODE",
    "GROUNDING_VERIFIER_ENABLED",
    "RATE_LIMIT_PER_MINUTE",
    "DATA_RETENTION_DAYS",
)


def config_defaults(root: Path) -> dict[str, Any] | None:
    """Defaults declared in medbridge/config.py (read as text; no settings object is built)."""
    path = root / "medbridge" / "config.py"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    out: dict[str, Any] = {}
    for key in _CONFIG_KEYS:
        m = re.search(rf"^\s+{key}\s*:\s*[\w\[\]| ]+=\s*(.+?)\s*(?:#.*)?$", text, re.MULTILINE)
        if not m:
            continue
        raw = m.group(1).strip()
        if raw[:1] in "\"'":
            out[key] = raw.strip("\"'")
        elif raw in ("True", "False"):
            out[key] = raw == "True"
        else:
            try:
                out[key] = int(raw)
            except ValueError:
                try:
                    out[key] = float(raw)
                except ValueError:
                    out[key] = raw
    if not out:
        return None
    out["source"] = "medbridge/config.py"
    return out


def judges(root: Path) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    manifests = sorted(
        (root / "evaluation" / "results" / "judgments").glob("manifest_judge_*.json")
    )
    for path in reversed(manifests):
        data = _read_json(path)
        if isinstance(data, dict) and data.get("judge_model"):
            out["llm_judge_model"] = str(data["judge_model"])
            info = (data.get("ollama") or {}).get(data["judge_model"]) or {}
            if info.get("parameter_size"):
                out["llm_judge_size"] = (
                    f"{info.get('parameter_size')} {info.get('quantization_level', '')}".strip()
                )
            out["llm_judge_source"] = _rel(path, root)
            break
    nli_src = root / "evaluation" / "judges" / "nli_judge.py"
    try:
        m = re.search(
            r'^DEFAULT_NLI_MODEL\s*=\s*"([^"]+)"', nli_src.read_text(encoding="utf-8"), re.MULTILINE
        )
        if m:
            out["nli_model"] = m.group(1)
    except OSError:
        pass
    return out or None


def screenshots(root: Path) -> dict[str, str]:
    """Canonical scene name -> repo-relative path of an existing screenshot."""
    folder = root / SCREENSHOT_DIR
    found: dict[str, str] = {}
    for scene, aliases in SCREENSHOTS.items():
        for stem in (scene, *aliases):
            path = folder / f"{stem}.png"
            if path.is_file():
                found[scene] = _rel(path, root)
                break
    return found


def model_selection_decision(root: Path) -> dict[str, Any] | None:
    path = root / "evaluation" / "results" / "model_selection" / "DECISION.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"\*\*Decision:\s*([^*]+?)\*\*", text)
    return {
        "decision": m.group(1).strip() if m else None,
        "no_significant_difference": "no significant difference" in text.lower(),
        "source": _rel(path, root),
    }


# ----------------------------------------------------------------------
# Aggregate
# ----------------------------------------------------------------------


def collect(root: Path, examples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """All deck facts as one dict (stored under the ``deck`` key of the results context)."""
    facts: dict[str, Any] = {}
    errors: dict[str, str] = {}
    steps: dict[str, Callable[[Path], Any]] = {
        "emergency": emergency_case_stats,
        "drugs": drug_lexicon,
        "adversarial": adversarial_payloads,
        "aq": aq_vignettes,
        "tests": test_inventory,
        "baseline": pytest_baseline,
        "code_review": code_review,
        "config": config_defaults,
        "judges": judges,
        "decision": model_selection_decision,
    }
    for key, fn in steps.items():
        try:
            value = fn(root)
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"
            continue
        if value is not None:
            facts[key] = value
    facts["screenshots"] = screenshots(root)
    if examples:
        try:
            facts["emergency_examples"] = classify_examples(root, examples)
        except Exception as exc:
            errors["emergency_examples"] = f"{type(exc).__name__}: {exc}"
    if errors:
        facts["_errors"] = errors
    return facts
