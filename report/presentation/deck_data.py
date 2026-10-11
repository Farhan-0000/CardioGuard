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
* the LLM-judge and NLI-judge model names and which UI screenshots exist;
* the number of distinct issuing organisations in the corpus, with joint labels such as
  "AHA/ACC" split into their members (``corpus_stats.json`` counts distinct label strings);
* the study's paired tests indexed by comparison, the brief's §6.7 decision rule and the
  conditions the NLI judge covered (``evaluation/results/summary.json``);
* how often the post-study Gate 3 withdrawal rule would have applied in the V2 test run
  (run records joined with the LLM judge's verdicts);
* provenance of the live robustness runs and the pre-fix RABBITS and adversarial runs;
* the before/after latency probes compared on the same model, with server stage timings.

Every fact is optional: a failure leaves its key absent, and the slide then shows a
highlighted ``[PENDING: ...]`` marker. Nothing here calls a language model or a server.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeGuard

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
#: Raw captures of medbridge/web/screenshots.py (one folder per viewport and theme).
SCREENSHOT_RAW_DIRS = (
    "report/screenshots/raw/desktop_light",
    "report/screenshots/raw/desktop_dark",
)

#: Hallucination-study summary written by evaluation/analyze.py.
STUDY_SUMMARY = "evaluation/results/summary.json"
#: Live robustness runs recorded before the brand-name fix (commit 02d9774). The pre-fix
#: adversarial report is also in git history; it is read from there when the copy is missing.
RABBITS_BEFORE = "reports/rabbits_before_fix/rabbits_benchmark_report.json"
ADVERSARIAL_BEFORE = "reports/adversarial_before_fix/adversarial_live_report.json"
ADVERSARIAL_BEFORE_GIT = ("2053216", "reports/adversarial_live_report.json")
#: Live robustness runs on the current code (same files the results context reads).
ROBUSTNESS_AFTER = {
    "aq": "reports/medbridge_aq_benchmark_report.json",
    "rabbits": "reports/rabbits_benchmark_report.json",
    "adversarial": "reports/adversarial_live_report.json",
}
AQ_EVALUATOR = "tests/benchmark/evaluator.py"
PERF_DIR = "reports/performance"

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
    "GROUNDING_NLI_MODEL",
    "GROUNDING_ENTAILMENT_THRESHOLD",
    "GROUNDING_MAX_UNSUPPORTED_FRACTION",
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
    """Canonical scene name -> repo-relative path of an existing screenshot.

    Preference order: ``report/screenshots/<scene>.png``, then the capture script's name in
    the same folder, then the raw captures in ``report/screenshots/raw/<variant>/``. A raw
    capture that is newer than the curated copy wins, so a stale copy is never shown after
    a fresh capture against the real backend.
    """
    folder = root / SCREENSHOT_DIR
    found: dict[str, str] = {}
    for scene, aliases in SCREENSHOTS.items():
        curated = [folder / f"{stem}.png" for stem in (scene, *aliases)]
        raw = [root / d / f"{stem}.png" for stem in aliases for d in SCREENSHOT_RAW_DIRS]
        existing = [p for p in (*curated, *raw) if p.is_file()]
        if not existing:
            continue
        chosen = existing[0]
        newest = max(existing, key=lambda p: p.stat().st_mtime)
        if newest.stat().st_mtime > chosen.stat().st_mtime + 1.0:
            chosen = newest
        found[scene] = _rel(chosen, root)
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


def corpus_organisations(root: Path) -> dict[str, Any] | None:
    """Distinct issuing organisations of the corpus documents.

    ``reports/corpus_stats.json`` ``totals.organisations`` counts distinct ``org`` strings,
    so "AHA/ACC" and "ACC/AHA" (the same joint issuers) count twice. Here every joint label
    is split on "/" into its members. Without a document list the stored total is used.
    """
    path = root / "reports" / "corpus_stats.json"
    data = _read_json(path)
    if not isinstance(data, dict):
        return None
    orgs = {
        part.strip()
        for doc in data.get("documents") or []
        if isinstance(doc, dict) and isinstance(doc.get("org"), str)
        for part in doc["org"].split("/")
        if part.strip()
    }
    if orgs:
        return {"organisations": len(orgs), "names": sorted(orgs), "source": _rel(path, root)}
    total = (data.get("totals") or {}).get("organisations")
    return {"organisations": total, "source": _rel(path, root)} if isinstance(total, int) else None


# ----------------------------------------------------------------------
# Study statistics (evaluation/results/summary.json)
# ----------------------------------------------------------------------


def verdict_label(test: dict[str, Any] | None) -> str:
    """Three-state verdict of one paired test: Holm p < 0.05 and the sign of the risk difference."""
    if not isinstance(test, dict):
        return "not significant"
    p = test.get("pvalue_holm", test.get("pvalue"))
    rd = (test.get("risk_difference") or {}).get("estimate")
    if not isinstance(p, (int, float)) or not isinstance(rd, (int, float)):
        return "not significant"
    if not (math.isfinite(p) and math.isfinite(rd)) or p >= 0.05 or rd == 0:
        return "not significant"
    return "significant reduction" if rd < 0 else "significant increase"


def _index_tests(items: Any) -> dict[str, dict[str, Any]]:
    """``{"C5_vs_C1": {"hallucination": test, "unsafe": test}, ...}`` for the LLM judge."""
    out: dict[str, dict[str, Any]] = {}
    for t in items or []:
        if not isinstance(t, dict) or str(t.get("judge", "llm")) != "llm":
            continue
        key = f"{t.get('treatment')}_vs_{t.get('control')}"
        out.setdefault(key, {})[str(t.get("outcome"))] = {**t, "verdict": verdict_label(t)}
    return out


def study_stats(root: Path) -> dict[str, Any] | None:
    """Paired tests, Cochran's Q, the brief's §6.7 decision rule and NLI-judge coverage.

    The rule counts as met when C5 significantly reduces the same outcome (hallucination or
    unsafe answers, LLM judge, Holm-corrected) against both C1 and C2.
    """
    data = _read_json(root / STUDY_SUMMARY)
    if not isinstance(data, dict):
        return None
    tests = data.get("tests")
    if not isinstance(tests, dict):
        return None
    mcnemar = _index_tests(tests.get("mcnemar"))
    primary = [
        (outcome, control, mcnemar.get(f"C5_vs_{control}", {}).get(outcome))
        for outcome in ("hallucination", "unsafe")
        for control in ("C1", "C2")
    ]
    present = [(o, c, t) for o, c, t in primary if t]
    met = [
        o
        for o in ("hallucination", "unsafe")
        if all(
            (t or {}).get("verdict") == "significant reduction" for oo, _, t in primary if oo == o
        )
        and sum(1 for oo, _, t in primary if oo == o and t) == 2
    ]
    per = data.get("per_condition") or {}
    nli_reported = [
        c
        for c, m in per.items()
        if isinstance(m, dict)
        and ((m.get("judge_coverage") or {}).get("nli") or {}).get("reported")
    ]
    return {
        "mcnemar": mcnemar,
        "item": _index_tests(tests.get("sensitivity_item_level")),
        "cochran": {
            str(q.get("outcome")): q
            for q in tests.get("cochran_q") or []
            if isinstance(q, dict) and str(q.get("judge", "llm")) == "llm"
        },
        "rule": {
            "met": bool(met),
            "outcomes_met": met,
            "primary_total": len(present),
            "primary_significant": sum(
                1 for _, _, t in present if t.get("verdict") == "significant reduction"
            ),
        },
        "nli_reported": nli_reported,
        "n_conditions": len(per),
        "source": STUDY_SUMMARY,
    }


def gate3_withdrawal(
    root: Path, condition: str = "V2", split: str = "test"
) -> dict[str, Any] | None:
    """How often the post-study withdrawal rule (commit 16e594b) would have applied in the study.

    Counts the ``condition`` test turns (latest complete attempt per item, as in the analysis)
    whose Gate 3 record has every checked claim unsupported, and how many of those turns the
    LLM judge rated hallucinated (>= 1 UNSUPPORTED or CONTRADICTED claim; judgment joined on
    item, turn and attempt). The study code delivered those drafts unchanged when stripping
    would have left nothing, and otherwise reduced them to their unchecked non-claim sentences.

    When the paired C5 run exists, two more counts are added: ``unchanged``, the fully
    unsupported turns delivered with exactly C5's text (the draft went out as written), and
    ``citation_downgrades``, checked turns in which Gate 3 removed sentences, the citation
    rule then downgraded ANSWER to GENERALIZE (no valid citation left) and the C5 turn was
    not downgraded.
    """
    runs = root / "evaluation" / "results" / "runs" / f"{condition}.jsonl"
    judgments = root / "evaluation" / "results" / "judgments" / "llm" / f"{condition}.jsonl"
    if not runs.is_file() or not judgments.is_file():
        return None
    _ensure_import_path(root)
    try:
        from evaluation.io import iter_jsonl, latest_complete_attempts
    except Exception:
        return None
    records = latest_complete_attempts(
        r
        for r in iter_jsonl(runs)
        if str(r.get("split")) == split and "item_id" in r and "turn_index" in r
    )
    judged = {
        (str(j.get("item_id")), int(j.get("turn_index", 0)), str(j.get("attempt_id", ""))): j
        for j in iter_jsonl(judgments)
        if str(j.get("split")) == split
    }
    c5_path = runs.with_name("C5.jsonl")
    c5 = (
        {
            (str(r["item_id"]), int(r["turn_index"])): r
            for r in latest_complete_attempts(
                r
                for r in iter_jsonl(c5_path)
                if str(r.get("split")) == split and "item_id" in r and "turn_index" in r
            )
        }
        if c5_path.is_file()
        else {}
    )
    checked = withdrawn = hallucinated = unchanged = downgrades = 0
    for rec in records:
        if rec.get("error") or rec.get("action") is None:
            continue
        g3 = (rec.get("explanation") or {}).get("gate3")
        if not isinstance(g3, dict) or not g3.get("available", True):
            continue
        checked += 1
        pair = c5.get((str(rec["item_id"]), int(rec["turn_index"])))
        cv = (rec.get("explanation") or {}).get("citation_validation") or {}
        pair_cv = ((pair or {}).get("explanation") or {}).get("citation_validation") or {}
        if (
            pair is not None
            and cv.get("downgraded")
            and int(g3.get("removed_sentences_count") or 0) > 0
            and not pair_cv.get("downgraded")
        ):
            downgrades += 1
        n_claims = int(g3.get("n_claims") or 0)
        if n_claims <= 0 or int(g3.get("n_unsupported") or 0) < n_claims:
            continue
        withdrawn += 1
        if pair is not None and pair.get("response_text") == rec.get("response_text"):
            unchanged += 1
        key = (str(rec["item_id"]), int(rec["turn_index"]), str(rec.get("attempt_id", "")))
        claims = (judged.get(key) or {}).get("claims") or []
        if any(c.get("label") in ("UNSUPPORTED", "CONTRADICTED") for c in claims):
            hallucinated += 1
    if not checked:
        return None
    out: dict[str, Any] = {
        "condition": condition,
        "checked": checked,
        "turns": withdrawn,
        "hallucinated": hallucinated,
        "source": f"{_rel(runs, root)} + {_rel(judgments, root)}",
    }
    if c5:
        out["unchanged"] = unchanged
        out["citation_downgrades"] = downgrades
        out["source"] += f" + {_rel(c5_path, root)}"
    return out


# ----------------------------------------------------------------------
# Robustness runs before and after the brand-name fix
# ----------------------------------------------------------------------


def _is_live(data: Any) -> TypeGuard[dict[str, Any]]:
    if not isinstance(data, dict):
        return False
    mode = data.get("mode")
    for key in ("metrics", "metadata"):
        if mode is None and isinstance(data.get(key), dict):
            mode = data[key].get("mode")
    return str(mode).lower() == "live"


def _run_info(data: dict[str, Any]) -> dict[str, Any]:
    raw = data.get("metadata")
    meta: dict[str, Any] = raw if isinstance(raw, dict) else {}
    commit = str(meta.get("git_commit") or "")
    gate3 = meta.get("grounding_verifier")
    return {
        "commit": commit.split("+")[0] or None,
        "dirty": "+dirty" in commit,
        "gate3": gate3,
        "gate3_label": {True: "Gate 3 on", False: "Gate 3 off"}.get(gate3)
        if isinstance(gate3, bool)
        else None,
        "generated": str(data.get("generated_at") or data.get("timestamp") or "")[:10] or None,
    }


def _git_json(root: Path, rev: str, path: str) -> Any | None:
    """A JSON file from git history (only inside a git checkout; never raises)."""
    if not (root / ".git").exists():
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "show", f"{rev}:{path}"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except ValueError:
        return None


def aq_targets(root: Path) -> dict[str, float] | None:
    """Acceptance targets of the MedBridge-AQ harness (constants in tests/benchmark/evaluator.py)."""
    try:
        text = (root / AQ_EVALUATOR).read_text(encoding="utf-8")
    except OSError:
        return None
    out = {
        m.group(1).lower(): float(m.group(2))
        for m in re.finditer(r"^ACCEPTANCE_([A-Z_]+)\s*=\s*([0-9.]+)", text, re.MULTILINE)
    }
    return out or None


def robustness_runs(root: Path) -> dict[str, Any] | None:
    """Run provenance (commit, Gate 3 flag) of the live robustness reports, plus the pre-fix runs.

    ``*_after`` hold only provenance; their metrics come from the results context, which
    loads the same files and ignores mock-mode output. ``*_before`` hold the pre-fix metrics.
    """
    out: dict[str, Any] = {}
    for key, rel in ROBUSTNESS_AFTER.items():
        data = _read_json(root / rel)
        if _is_live(data):
            out[f"{key}_after"] = {**_run_info(data), "source": rel}
    before = _read_json(root / RABBITS_BEFORE)
    if _is_live(before):
        slim = {k: v for k, v in before.items() if k not in ("items_detail", "results")}
        out["rabbits_before"] = {**slim, **_run_info(before), "source": RABBITS_BEFORE}
    adv, src = _read_json(root / ADVERSARIAL_BEFORE), ADVERSARIAL_BEFORE
    if adv is None:
        rev, path = ADVERSARIAL_BEFORE_GIT
        adv, src = _git_json(root, rev, path), f"git show {rev}:{path}"
    if _is_live(adv):
        slim = {k: v for k, v in adv.items() if k not in ("results",)}
        out["adversarial_before"] = {**slim, **_run_info(adv), "source": src}
    targets = aq_targets(root)
    if targets:
        out["aq_targets"] = targets
    return out or None


# ----------------------------------------------------------------------
# Latency probes before and after the optimisation
# ----------------------------------------------------------------------


def _model_of(tag: str) -> str:
    """``gemma3-4b`` -> ``gemma3:4b`` (probe file names replace the tag's colon)."""
    head, sep, tail = tag.rpartition("-")
    return f"{head}:{tail}" if sep and head else tag


def _probe_turns(probe: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per scenario turn: actions, warm p50/p95 (s), cold latency (s), server stage p50s (ms)."""
    out: dict[str, dict[str, Any]] = {}
    for key, stats in (probe.get("warm_summary") or {}).items():
        if not isinstance(stats, dict):
            continue
        turn, _, action = str(key).partition(":")
        row = out.setdefault(turn, {"actions": [], "p50s": [], "p95s": [], "stages_ms": {}})
        row["actions"].append(action)
        for name, src in (("p50s", "p50"), ("p95s", "p95")):
            value = stats.get(src)
            if isinstance(value, (int, float)) and math.isfinite(value):
                row[name].append(float(value))
        if not row["stages_ms"] and isinstance(stats.get("server_stage_ms_p50"), dict):
            row["stages_ms"] = dict(stats["server_stage_ms_p50"])
    for row in out.values():
        p50s = sorted(row.pop("p50s"))
        p95s = row.pop("p95s")
        row["p50"] = p50s[len(p50s) // 2] if p50s else None
        row["p95"] = max(p95s) if p95s else None
    for cold in probe.get("cold") or []:
        if not isinstance(cold, dict):
            continue
        turn = f"{cold.get('scenario')}-t{cold.get('turn')}"
        value = cold.get("latency_s")
        if isinstance(value, (int, float)) and math.isfinite(value):
            row = out.setdefault(turn, {"actions": [], "p50": None, "p95": None, "stages_ms": {}})
            row["cold"] = float(value)
            if not row["actions"] and cold.get("action"):
                row["actions"].append(str(cold["action"]))
    return out


def perf_comparison(root: Path) -> dict[str, Any] | None:
    """Before/after latency per scenario turn, on the same model, from the probe files.

    ``before`` is the probe labelled ``before`` (``before_<model>.json``, the received code);
    ``after`` is ``after_<same model>.json`` (final code, Gate 3 included), so the comparison
    holds the model fixed. ``deployed`` is the final code on the configured application
    model and ``warmup`` the cold pass with start-up warm-up on, when those files exist.
    """
    folder = root / PERF_DIR
    probes: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.glob("*.json")):
        data = _read_json(path)
        if isinstance(data, dict) and ("warm_summary" in data or "cold_summary" in data):
            probes[path.stem] = data
    before_stem = next((s for s, d in probes.items() if str(d.get("label")) == "before"), None)
    if before_stem is None:
        return None
    tag = before_stem.removeprefix("before_") if before_stem.startswith("before_") else ""
    after_stem = f"after_{tag}" if tag and f"after_{tag}" in probes else None
    if after_stem is None:
        after_stem = next((s for s, d in probes.items() if str(d.get("label")) == "after"), None)
    warmup_stem = f"after_warmup_{tag}" if tag and f"after_warmup_{tag}" in probes else None
    deployed_model = (config_defaults(root) or {}).get("LLM_MODEL")
    deployed_stem = None
    if isinstance(deployed_model, str):
        cand = "after_" + deployed_model.replace(":", "-")
        if cand in probes and cand != after_stem:
            deployed_stem = cand
    files = {
        "before": before_stem,
        "after": after_stem,
        "deployed": deployed_stem,
        "warmup": warmup_stem,
    }
    parsed = {k: _probe_turns(probes[s]) for k, s in files.items() if s}
    turns: dict[str, dict[str, Any]] = {}
    for turn in sorted(set().union(*(set(v) for v in parsed.values()))):
        row: dict[str, Any] = {}
        for k, by_turn in parsed.items():
            t = by_turn.get(turn)
            if not t:
                continue
            row[f"{k}_actions"] = "/".join(t["actions"]) or None
            for field_name in ("p50", "p95", "cold"):
                if t.get(field_name) is not None:
                    row[f"{k}_{field_name}"] = t[field_name]
            if t.get("stages_ms"):
                row[f"{k}_stages_ms"] = t["stages_ms"]
        b, a = row.get("before_p50"), row.get("after_p50")
        if isinstance(b, float) and isinstance(a, float):
            row["change_s"] = a - b
            row["speedup"] = b / a if a > 0 else None
        turns[turn] = row
    gate3 = sorted(
        float(r["after_stages_ms"]["verification"])
        for r in turns.values()
        if isinstance(r.get("after_stages_ms"), dict)
        and isinstance(r["after_stages_ms"].get("verification"), (int, float))
    )
    out: dict[str, Any] = {
        "model": _model_of(tag) if tag else None,
        "deployed_model": deployed_model if deployed_stem else None,
        "turns": turns,
        "files": {k: f"{PERF_DIR}/{s}.json" for k, s in files.items() if s},
        "source": ", ".join(f"{PERF_DIR}/{s}.json" for s in files.values() if s),
    }
    if gate3:
        out["gate3_ms_p50"] = gate3[len(gate3) // 2]
    return out


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
        "corpus": corpus_organisations,
        "stats": study_stats,
        "gate3_withdrawal": gate3_withdrawal,
        "robust": robustness_runs,
        "perf": perf_comparison,
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
