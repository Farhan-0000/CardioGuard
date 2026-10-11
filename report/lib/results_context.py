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
perf_latest            {"before": ..., "after": ...} newest latency-probe file per label; when
                       no probe is labelled plain "after", "after" aliases the after probe of the
                       baseline's model (before_<model>.json -> after_<model>.json)
perf_derived           per probe and turn: time in LLM stages and its share of the server total
aq                     reports/medbridge_aq_benchmark_report.json (LIVE runs only)
rabbits                reports/rabbits_benchmark_report.json (LIVE runs only)
rabbits_before         reports/rabbits_before_fix/rabbits_benchmark_report.json (LIVE, pre-fix)
rabbits_counts         {"before", "after"}: flips and abstentions counted from items_detail
adversarial            reports/adversarial_live_report.json (LIVE runs only)
adversarial_before     reports/adversarial_before_fix/adversarial_live_report.json (LIVE, pre-fix)
load                   reports/performance/load_test.json
annotation             counts computed from evaluation/annotation/human_annotation_sheet.csv
smoke_inputs           request texts parsed (AST) from scripts/smoke_test.py
study_gate3            Gate 3 outcomes of the V2 / V12 test runs joined with the LLM judge
routing_errors         misroutes of C5 (HalluBench) and of the live AQ run, from confusion matrices
corpus_orgs            issuing organisations of the corpus with joint labels ("AHA/ACC") split
tests                  reports/test_summary.json (or JUnit XML in reports/tests/)
calibration            evaluation/results/calibration.json
human_agreement        evaluation/results/human_agreement.json
smoke                  reports/smoke_test_latest.json
hallubench             statistics computed from evaluation/datasets/hallubench_v1.json
manifest               data/corpus_manifest.json
manifest_stats         statistics computed from the manifest
study_manifests        evaluation/results/runs/manifest_*.json (full run manifests, by run id)
judge_manifests        evaluation/results/judgments/manifest_judge_*.json (by file stem)
study_records          counts computed from evaluation/results/runs/<condition>.jsonl
study_judgments        counts computed from evaluation/results/judgments/{llm,nli}/*.jsonl
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
    # live runs of the code before the brand-name robustness fix (commit 02d9774)
    "rabbits_before": "reports/rabbits_before_fix/rabbits_benchmark_report.json",
    "adversarial_before": "reports/adversarial_before_fix/adversarial_live_report.json",
}

ANNOTATION_SHEET = "evaluation/annotation/human_annotation_sheet.csv"
SMOKE_SCRIPT = "scripts/smoke_test.py"
#: Server stages that are language-model calls (see the orchestrator's stage names).
LLM_STAGES: tuple[str, ...] = ("extraction", "context_gate", "evidence_gate", "generation")

RERANKER_DIR = "evaluation/results/retrieval/reranker_ablation"
PERF_DIR = "reports/performance"
HALLUBENCH_PATH = "evaluation/datasets/hallubench_v1.json"
JUNIT_DIR = "reports/tests"
RUNS_DIR = "evaluation/results/runs"
JUDGMENTS_DIR = "evaluation/results/judgments"


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        return []
    return out


def _latest_complete(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per item, the most recent attempt that recorded every turn (rule of evaluation/io.py)."""
    attempts: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    order: dict[tuple[str, str], int] = {}
    for position, rec in enumerate(records):
        if "item_id" not in rec or "turn_index" not in rec:
            continue
        key = (str(rec["item_id"]), str(rec.get("attempt_id", "")))
        attempts.setdefault(key, {})[int(rec["turn_index"])] = rec
        order[key] = position
    best: dict[str, tuple[int, dict[int, dict[str, Any]]]] = {}
    for (item_id, attempt_id), turns in attempts.items():
        n_turns = int(next(iter(turns.values())).get("n_turns", len(turns)) or len(turns))
        if set(turns) != set(range(n_turns)):
            continue
        position = order[(item_id, attempt_id)]
        if item_id not in best or position > best[item_id][0]:
            best[item_id] = (position, turns)
    return [best[i][1][t] for i in sorted(best) for t in sorted(best[i][1])]


def study_record_stats(runs_dir: Path, split: str = "test") -> dict[str, Any] | None:
    """Provenance of the study outputs in ``evaluation/results/runs/<condition>.jsonl``.

    ``by_condition[C]``: lines and final turns of ``split`` and which run produced the final
    records; ``by_run[run_id]``: records written (any split) with first and last timestamps,
    so that interrupted runs whose manifest never got a ``finished_at`` still have a span.
    """
    files = sorted(p for p in runs_dir.glob("*.jsonl") if not p.name.startswith("manifest"))
    if not files:
        return None
    by_condition: dict[str, dict[str, Any]] = {}
    by_run: dict[str, dict[str, Any]] = {}
    for path in files:
        records = _read_jsonl(path)
        if not records:
            continue
        cond = path.stem
        for rec in records:
            run = by_run.setdefault(
                str(rec.get("run_id")), {"records": 0, "first": None, "last": None, "by_split": {}}
            )
            run["records"] += 1
            sp = str(rec.get("split"))
            run["by_split"].setdefault(sp, {}).setdefault(cond, 0)
            run["by_split"][sp][cond] += 1
            ts = rec.get("timestamp")
            if isinstance(ts, str):
                run["first"] = ts if run["first"] is None or ts < run["first"] else run["first"]
                run["last"] = ts if run["last"] is None or ts > run["last"] else run["last"]
        selected = [r for r in records if str(r.get("split")) == split]
        final = _latest_complete(selected)
        by_condition[cond] = {
            "lines": len(selected),
            "final_turns": len(final),
            "final_items": len({r["item_id"] for r in final}),
            "final_by_run": dict(Counter(str(r.get("run_id")) for r in final)),
            "degraded": sum(1 for r in final if r.get("degraded")),
        }
    if not by_condition:
        return None
    return {"split": split, "by_condition": by_condition, "by_run": by_run}


def study_judgment_stats(judgments_dir: Path, split: str = "test") -> dict[str, Any] | None:
    """Judge coverage per condition from ``judgments/{llm,nli}/<condition>.jsonl``.

    For each judge and condition (latest record per item-turn of ``split``): records, records
    with an error (failed judgment, with item ids), template responses (zero claims) and the
    item categories covered, which shows whether a judge covered a condition completely.
    """
    out: dict[str, Any] = {}
    for judge in ("llm", "nli"):
        per: dict[str, Any] = {}
        for path in sorted((judgments_dir / judge).glob("*.jsonl")):
            latest: dict[tuple[str, int], dict[str, Any]] = {}
            splits: Counter[str] = Counter()
            for rec in _read_jsonl(path):
                splits[str(rec.get("split"))] += 1
                if str(rec.get("split")) != split or "item_id" not in rec:
                    continue
                latest[(str(rec["item_id"]), int(rec.get("turn_index", 0)))] = rec
            rows = list(latest.values())
            errors = [r for r in rows if r.get("error")]
            per[path.stem] = {
                "records": len(rows),
                "records_by_split": dict(splits),
                "errors": len(errors),
                "error_items": [
                    {
                        "item_id": r["item_id"],
                        "turn": int(r.get("turn_index", 0)),
                        "category": r.get("category"),
                        "subcategory": r.get("subcategory"),
                        "error": str(r.get("error"))[:120],
                    }
                    for r in errors
                ],
                "templates": sum(1 for r in rows if r.get("template")),
                "templates_by_kind": dict(
                    Counter(str(r.get("template")) for r in rows if r.get("template"))
                ),
                "by_category": dict(Counter(str(r.get("category")) for r in rows)),
            }
        if per:
            out[judge] = per
    return out or None


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


def annotation_stats(path: Path) -> dict[str, Any] | None:
    """Counts over the exported human annotation sheet (``evaluation.human_annotation``).

    Items, blinded responses and claims, claims per category and how many claims each
    annotator has labelled so far (``A1_label`` / ``A2_label`` non-empty).
    """
    import csv

    try:
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, csv.Error, UnicodeDecodeError):
        return None
    if not rows or "claim" not in rows[0]:
        return None

    def filled(key: str) -> int:
        return sum(1 for r in rows if str(r.get(key) or "").strip())

    return {
        "n_items": len({r.get("item_id") for r in rows}),
        "n_responses": len({r.get("response_code") for r in rows}),
        "n_claims": len(rows),
        "by_category": dict(Counter(str(r.get("category")) for r in rows)),
        "labelled_a1": filled("A1_label"),
        "labelled_a2": filled("A2_label"),
        "labelled_both": sum(
            1
            for r in rows
            if str(r.get("A1_label") or "").strip() and str(r.get("A2_label") or "").strip()
        ),
    }


def smoke_inputs(path: Path) -> dict[str, str] | None:
    """Request texts of the smoke test, read from ``scripts/smoke_test.py`` without importing it.

    ``MessageCase("<check name>", "<message>", ...)`` definitions give the message of each
    message check; ``PII_MESSAGE`` and ``STREAM_MESSAGE`` give those of the ``pii_redaction``
    and ``sse_stream`` checks.
    """
    import ast

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None

    def literal(node: ast.AST) -> str | None:
        try:
            value = ast.literal_eval(node)
        except (ValueError, SyntaxError):
            return None
        return value if isinstance(value, str) else None

    out: dict[str, str] = {}
    constants = {"PII_MESSAGE": "pii_redaction", "STREAM_MESSAGE": "sse_stream"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and getattr(value.func, "id", None) == "MessageCase"
            and len(value.args) >= 2
        ):
            name, message = literal(value.args[0]), literal(value.args[1])
            if name and message:
                out[name] = message
        elif target.id in constants:
            text = literal(value)
            if text:
                out[constants[target.id]] = text
    return out or None


def perf_derived(perf: dict[str, Any]) -> dict[str, Any] | None:
    """Per latency probe and turn: milliseconds spent in LLM stages and their share.

    Uses the per-stage server medians (``server_stage_ms_p50``) of the ``cold_summary`` and
    ``warm_summary`` of the after-optimisation probes; the received prototype logged no
    stage timings, so its probe has none.
    """
    out: dict[str, Any] = {}
    for stem, probe in perf.items():
        if not isinstance(probe, dict):
            continue
        entry: dict[str, Any] = {}
        for phase in ("cold", "warm"):
            summary = probe.get(f"{phase}_summary")
            if not isinstance(summary, dict):
                continue
            turns: dict[str, Any] = {}
            for key, stats in summary.items():
                if not isinstance(stats, dict):
                    continue
                stages = stats.get("server_stage_ms_p50")
                total = stats.get("server_total_ms_p50")
                if not isinstance(stages, dict) or not total:
                    continue
                llm_ms = sum(float(stages.get(s) or 0.0) for s in LLM_STAGES)
                turns[key] = {
                    "llm_ms": round(llm_ms, 1),
                    "llm_share": llm_ms / float(total),
                    "other_ms": round(float(total) - llm_ms, 1),
                }
            if turns:
                entry[phase] = turns
        if entry:
            out[stem] = entry
    return out or None


#: Order used to say which variant of a RABBITS pair got the more cautious action.
CAUTION_ORDER: dict[str, int] = {
    "ANSWER": 0,
    "GENERALIZE": 1,
    "SOFT-ASK": 1,
    "ABSTAIN": 2,
    "ESCALATE": 3,
}


def rabbits_counts(report: Any) -> dict[str, Any] | None:
    """Counts over the per-item records (``items_detail``) of a RABBITS report.

    ``flips``: items whose brand and generic messages were routed differently;
    ``brand_more_cautious``: flips in which the brand message got the more cautious action
    (ANSWER < GENERALIZE, SOFT-ASK < ABSTAIN < ESCALATE); ``brand_abstain`` /
    ``generic_abstain``: items whose brand / generic message was abstained.
    """
    if not isinstance(report, dict):
        return None
    items = [i for i in report.get("items_detail") or [] if isinstance(i, dict)]
    if not items:
        return None
    flips = [i for i in items if not i.get("action_match")]
    pairs = Counter(f"{i.get('brand_action')} vs {i.get('generic_action')}" for i in flips)
    return {
        "items": len(items),
        "flips": len(flips),
        "brand_more_cautious": sum(
            1
            for i in flips
            if CAUTION_ORDER.get(str(i.get("brand_action")), -1)
            > CAUTION_ORDER.get(str(i.get("generic_action")), -1)
        ),
        "brand_abstain": sum(1 for i in items if i.get("brand_action") == "ABSTAIN"),
        "generic_abstain": sum(1 for i in items if i.get("generic_action") == "ABSTAIN"),
        "flip_pairs": dict(pairs.most_common()),
        "flips_by_template": dict(Counter(str(i.get("template")) for i in flips)),
    }


def study_gate3_stats(
    runs_dir: Path, judgments_dir: Path, conditions: tuple[str, ...] = ("V2", "V12")
) -> dict[str, Any] | None:
    """Gate 3 outcomes of the final test-split records, joined with the LLM judge.

    ``all_unsupported``: turns in which the verifier checked at least one claim and found
    every one unsupported. With tau = 1.0 the studied code then delivered the draft
    unchanged (removing every claim leaves nothing); commit 16e594b withdraws such drafts.
    ``all_unsupported_hallucinated``: those of them the LLM judge rated hallucinated (at
    least one UNSUPPORTED or CONTRADICTED claim). ``all_unsupported_unchanged``: those of
    them delivered with exactly C5's text (stripping would have left nothing, so the draft
    went out unchanged). ``citation_downgrades``: checked turns in which Gate 3 removed
    sentences, the citation rule then downgraded ANSWER to GENERALIZE (no valid citation
    left) and the paired C5 turn was not downgraded. The last two need ``C5.jsonl``.
    """
    c5 = {
        (str(r["item_id"]), int(r.get("turn_index", 0))): r
        for r in _latest_complete(
            [r for r in _read_jsonl(runs_dir / "C5.jsonl") if str(r.get("split")) == "test"]
        )
    }
    out: dict[str, Any] = {}
    for cond in conditions:
        records = _read_jsonl(runs_dir / f"{cond}.jsonl")
        final = _latest_complete([r for r in records if str(r.get("split")) == "test"])
        if not final:
            continue
        judged: dict[tuple[str, int], dict[str, Any]] = {}
        for rec in _read_jsonl(judgments_dir / "llm" / f"{cond}.jsonl"):
            if str(rec.get("split")) == "test" and "item_id" in rec:
                judged[(str(rec["item_id"]), int(rec.get("turn_index", 0)))] = rec
        checked = downgrades = 0
        withdrawn: list[dict[str, Any]] = []
        for rec in final:
            if rec.get("degraded"):
                continue
            g3 = (rec.get("explanation") or {}).get("gate3")
            if not isinstance(g3, dict) or not g3.get("available", True):
                continue
            n = int(g3.get("n_claims") or 0)
            if n <= 0:
                continue
            checked += 1
            if int(g3.get("n_unsupported") or 0) >= n:
                withdrawn.append(rec)
            pair = c5.get((str(rec["item_id"]), int(rec.get("turn_index", 0))))
            cv = (rec.get("explanation") or {}).get("citation_validation") or {}
            pair_cv = ((pair or {}).get("explanation") or {}).get("citation_validation") or {}
            if (
                pair is not None
                and cv.get("downgraded")
                and int(g3.get("removed_sentences_count") or 0) > 0
                and not pair_cv.get("downgraded")
            ):
                downgrades += 1
        hallucinated = 0
        for rec in withdrawn:
            j = judged.get((str(rec["item_id"]), int(rec.get("turn_index", 0))))
            labels = [str(c.get("label")) for c in (j or {}).get("claims") or []]
            if any(lb in ("UNSUPPORTED", "CONTRADICTED") for lb in labels):
                hallucinated += 1
        unchanged = sum(
            1
            for rec in withdrawn
            if (pair := c5.get((str(rec["item_id"]), int(rec.get("turn_index", 0))))) is not None
            and pair.get("response_text") == rec.get("response_text")
        )
        out[cond] = {
            "turns_checked": checked,
            "all_unsupported": len(withdrawn),
            "all_unsupported_hallucinated": hallucinated,
            "all_unsupported_by_action": dict(
                Counter(str(r.get("action") or r.get("final_action")) for r in withdrawn)
            ),
        }
        if c5:
            out[cond]["all_unsupported_unchanged"] = unchanged
            out[cond]["citation_downgrades"] = downgrades
    return out or None


def corpus_org_stats(corpus: Any) -> dict[str, Any] | None:
    """Distinct issuing organisations, joint labels split on "/".

    ``totals.organisations`` of ``reports/corpus_stats.json`` counts distinct label strings,
    so "AHA/ACC", "ACC/AHA" and "AHA" count three times; the deck and the paper count the
    members of joint labels once each (``report/presentation/deck_data.py``).
    """
    docs = (corpus or {}).get("documents") if isinstance(corpus, dict) else None
    names = sorted(
        {
            part.strip()
            for doc in docs or []
            if isinstance(doc, dict) and isinstance(doc.get("org"), str)
            for part in doc["org"].split("/")
            if part.strip()
        }
    )
    return {"split": len(names), "names": names} if names else None


def routing_error_stats(study: Any, aq: Any) -> dict[str, Any] | None:
    """Misroutes of C5 on HalluBench and of the live AQ run, from their confusion matrices.

    ``misroutes``: turns (vignettes) whose action differs from the gold action;
    ``answer_generalize``: ANSWER and GENERALIZE confused in either direction;
    ``answer_abstain``: expected ANSWER, routed to ABSTAIN.
    """

    def summarise(matrix: dict[str, dict[str, Any]]) -> dict[str, int]:
        def cell(gold: str, pred: str) -> int:
            return int((matrix.get(gold) or {}).get(pred) or 0)

        total = sum(int(v or 0) for row in matrix.values() for v in row.values())
        correct = sum(cell(g, g) for g in matrix)
        return {
            "total": total,
            "misroutes": total - correct,
            "answer_generalize": cell("ANSWER", "GENERALIZE") + cell("GENERALIZE", "ANSWER"),
            "answer_abstain": cell("ANSWER", "ABSTAIN"),
        }

    out: dict[str, Any] = {}
    routing = ((study or {}).get("per_condition") or {}).get("C5", {}).get("routing_system")
    if isinstance(routing, dict) and routing.get("labels") and routing.get("confusion"):
        labels = [str(x) for x in routing["labels"]]
        out["C5"] = summarise(
            {
                g: dict(zip(labels, row, strict=False))
                for g, row in zip(labels, routing["confusion"], strict=False)
            }
        )
    matrix = ((aq or {}).get("metrics") or {}).get("confusion_matrix")
    if isinstance(matrix, dict) and matrix:
        out["aq"] = summarise(
            {str(g): dict(row) for g, row in matrix.items() if isinstance(row, dict)}
        )
    return out or None


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

    counts = {
        label: rabbits_counts(ctx.get(key))
        for label, key in (("before", "rabbits_before"), ("after", "rabbits"))
    }
    if any(counts.values()):
        ctx["rabbits_counts"] = {k: v for k, v in counts.items() if v}
        sources["rabbits_counts"] = ", ".join(
            sources[key] for key in ("rabbits_before", "rabbits") if key in sources
        )

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
    if latest and "after" not in latest and "before" in latest:
        # before_<model>.json is compared with after_<model>.json (same model, same scenarios)
        model = Path(latest["before"][2]).stem.removeprefix("before_").removeprefix("before")
        same_model = f"after_{model}" if model else ""
        if same_model in latest:
            latest["after"] = latest[same_model]
    if latest:
        ctx["perf_latest"] = {label: v[1] for label, v in latest.items()}
        ctx["perf_latest_sources"] = {label: v[2] for label, v in latest.items()}
        sources["perf_latest"] = ", ".join(sorted({v[2] for v in latest.values()}))
    derived = perf_derived(perf)
    if derived:
        ctx["perf_derived"] = derived
        sources["perf_derived"] = f"{PERF_DIR}/after_*.json (server stage medians)"

    annotation = annotation_stats(repo_root / ANNOTATION_SHEET)
    if annotation:
        ctx["annotation"] = annotation
        sources["annotation"] = ANNOTATION_SHEET

    inputs = smoke_inputs(repo_root / SMOKE_SCRIPT)
    if inputs:
        ctx["smoke_inputs"] = inputs
        sources["smoke_inputs"] = SMOKE_SCRIPT

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

    run_manifests: dict[str, Any] = {}
    for path in sorted((repo_root / RUNS_DIR).glob("manifest_*.json")):
        data = _read_json(path)
        if isinstance(data, dict):
            run_manifests[str(data.get("run_id") or path.stem.removeprefix("manifest_"))] = data
    if run_manifests:
        ctx["study_manifests"] = run_manifests
        sources["study_manifests"] = f"{RUNS_DIR}/manifest_*.json"

    judge_manifests: dict[str, Any] = {}
    for path in sorted((repo_root / JUDGMENTS_DIR).glob("manifest_judge_*.json")):
        data = _read_json(path)
        if isinstance(data, dict):
            judge_manifests[path.stem] = data
    if judge_manifests:
        ctx["judge_manifests"] = judge_manifests
        sources["judge_manifests"] = f"{JUDGMENTS_DIR}/manifest_judge_*.json"

    records = study_record_stats(repo_root / RUNS_DIR)
    if records:
        ctx["study_records"] = records
        sources["study_records"] = f"{RUNS_DIR}/<condition>.jsonl"

    judgments = study_judgment_stats(repo_root / JUDGMENTS_DIR)
    if judgments:
        ctx["study_judgments"] = judgments
        sources["study_judgments"] = f"{JUDGMENTS_DIR}/{{llm,nli}}/<condition>.jsonl"

    orgs = corpus_org_stats(ctx.get("corpus"))
    if orgs:
        ctx["corpus_orgs"] = orgs
        sources["corpus_orgs"] = SINGLE_SOURCES["corpus"]

    errors = routing_error_stats(ctx.get("study"), ctx.get("aq"))
    if errors:
        ctx["routing_errors"] = errors
        sources["routing_errors"] = ", ".join(
            sources[key] for key in ("study", "aq") if key in sources
        )

    gate3 = study_gate3_stats(repo_root / RUNS_DIR, repo_root / JUDGMENTS_DIR)
    if gate3:
        ctx["study_gate3"] = gate3
        sources["study_gate3"] = f"{RUNS_DIR}/V2.jsonl, {RUNS_DIR}/V12.jsonl + LLM judgments"

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


_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def format_sci(value: float, digits: int = 2) -> str:
    """``5.41e-16`` -> ``5.41 × 10⁻¹⁶`` (Unicode superscripts, readable in Word and PDF)."""
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa} × 10{str(int(exponent)).translate(_SUPERSCRIPT)}"


def format_p(value: Any) -> str:
    """p-value: 3 decimals from 0.01, 4 decimals from 0.001, scientific notation below."""
    p = _as_float(value)
    if _is_nan(p):
        return "n/a"
    if p >= 0.01:
        return f"{p:.3f}"
    if p >= 0.001:
        return f"{p:.4f}"
    return format_sci(p)


def _pp(value: Any) -> str:
    """Risk difference (fraction) in percentage points, with its interval if present."""
    if isinstance(value, dict):
        est = value.get("estimate", value.get("rate"))
        text = f"{100 * float(est):.1f} pp"
        low, high = value.get("low"), value.get("high")
        if low is not None and high is not None and not (_is_nan(low) or _is_nan(high)):
            text += f" [{100 * float(low):.1f}, {100 * float(high):.1f}]"
        return text
    return f"{100 * _as_float(value):.1f} pp"


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
        elif name == "p":  # p-value (scientific notation below 0.001)
            out = format_p(out)
        elif name == "sci":
            out = format_sci(_as_float(out), int(arg) if arg.isdigit() else 2)
        elif name == "pp":  # fraction difference -> percentage points (with CI if a dict)
            out = _pp(out)
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
