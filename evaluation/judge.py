"""
Judge CLI — labels the run records with the LLM judge and/or the NLI judge.

Outputs ``<judgments>/{llm,nli}/<condition>.jsonl`` keyed by
``(condition, item_id, turn_index)``; every judgment stores the run record's
``attempt_id`` so judgments of a re-run item are never mixed with stale ones.
Judge-side retrieved chunk texts are kept in ``<judgments>/evidence_chunks.jsonl``.

The NLI judge reuses the claims of the LLM judge, so run ``--judge llm`` first
(``--judge both`` does both in order).

Usage::

    python -m evaluation.judge --judge both --judge-model gemma3:12b
    python -m evaluation.judge --judge nli --split dev --resume
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from evaluation.datasets.schema import load_dataset
from evaluation.io import (
    OutputExistsError,
    append_jsonl,
    guard_output,
    judgment_file,
    load_judgments,
    load_run_records,
)
from evaluation.judges.common import EvidenceSidecar
from evaluation.judges.llm_judge import LLMJudge, LLMJudgeConfig
from evaluation.judges.nli_judge import (
    DEFAULT_NLI_MODEL,
    NLIJudge,
    claims_fingerprint,
    load_thresholds,
)
from evaluation.manifest import utc_now
from evaluation.paths import CACHE_PATH, CALIBRATION_PATH, DEFAULT_DATASET, JUDGMENTS_DIR, RUNS_DIR

Log = Callable[[str], None]


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def select_records(
    runs: dict[str, list[dict[str, Any]]], conditions: Sequence[str] | None, split: str
) -> dict[str, list[dict[str, Any]]]:
    """Filter run records by condition list (``None`` = all) and split; drop errored turns."""
    wanted = None if conditions is None else {c.upper() for c in conditions}
    out: dict[str, list[dict[str, Any]]] = {}
    for condition, records in runs.items():
        if wanted is not None and condition.upper() not in wanted:
            continue
        out[condition] = [
            r for r in records if not r.get("error") and (split == "all" or r.get("split") == split)
        ]
    return out


def needs_judging(existing: dict[tuple[str, int], dict[str, Any]], record: dict[str, Any]) -> bool:
    """True unless a judgment without error exists for this exact run attempt."""
    prev = existing.get((str(record["item_id"]), int(record["turn_index"])))
    return (
        prev is None
        or bool(prev.get("error"))
        or prev.get("attempt_id") != record.get("attempt_id")
    )


def histories(records: Sequence[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, str]]]:
    """Earlier turns (user text + assistant reply) of the same item, for the behaviour judge."""
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_item[str(rec["item_id"])].append(rec)
    out: dict[tuple[str, int], list[dict[str, str]]] = {}
    for item_id, recs in by_item.items():
        recs.sort(key=lambda r: int(r["turn_index"]))
        for rec in recs:
            index = int(rec["turn_index"])
            out[(item_id, index)] = [
                {"user": str(r.get("user_text", "")), "assistant": str(r.get("response_text", ""))}
                for r in recs
                if int(r["turn_index"]) < index
            ]
    return out


async def judge_with_llm(
    judge: LLMJudge,
    records_by_condition: dict[str, list[dict[str, Any]]],
    gold_notes: dict[str, str],
    judgments_dir: Path,
    *,
    resume: bool,
    fresh: bool,
    log: Log = _log,
) -> dict[str, dict[str, int]]:
    """Run the LLM judge over every selected record; appends judgments as it goes.

    Each judgment also stores ``judge_calls`` (number of judge LLM calls, how
    many were memoised, and their total latency) for the runtime report.
    """
    from medbridge.ai.llm_wrapper import llm_call_log

    tag = f"llm-{time.strftime('%Y%m%dT%H%M%S')}"
    counts: dict[str, dict[str, int]] = {}
    for condition, records in records_by_condition.items():
        path = judgment_file(judgments_dir, "llm", condition)
        guard_output(path, resume=resume, fresh=fresh, tag=tag)
        existing = load_judgments(path)
        todo = [r for r in records if needs_judging(existing, r)]
        stats = {"records": len(records), "judged": 0, "errors": 0, "templates": 0}
        history = histories(records)
        started = time.perf_counter()
        for i, record in enumerate(todo, start=1):
            key = (str(record["item_id"]), int(record["turn_index"]))
            with llm_call_log() as calls:
                try:
                    judgment = await judge.judge_turn(
                        record, gold_notes.get(key[0], ""), history.get(key, [])
                    )
                except Exception as exc:  # retrieval / infrastructure failure: record, continue
                    judgment = judge.empty_judgment(record)
                    judgment["error"] = f"{type(exc).__name__}: {exc}"
            judgment["judge_calls"] = {
                "n": len(calls),
                "cached": sum(1 for c in calls if c.cached),
                "latency_ms": round(sum(c.latency_ms for c in calls), 1),
            }
            append_jsonl(path, judgment)
            stats["judged"] += 1
            stats["errors"] += 1 if judgment.get("error") else 0
            stats["templates"] += 1 if judgment.get("template") else 0
            if i % 10 == 0 or i == len(todo):
                rate = (time.perf_counter() - started) / i
                log(
                    f"[judge llm {condition}] {i}/{len(todo)} | errors {stats['errors']} | "
                    f"ETA {rate * (len(todo) - i) / 60:.1f} min"
                )
        counts[condition] = stats
    return counts


def judge_with_nli(
    judge: NLIJudge,
    records_by_condition: dict[str, list[dict[str, Any]]],
    judgments_dir: Path,
    sidecar: EvidenceSidecar,
    *,
    resume: bool,
    fresh: bool,
    log: Log = _log,
) -> dict[str, dict[str, int]]:
    """Score the LLM judge's claims with the NLI model for every selected record."""
    tag = f"nli-{time.strftime('%Y%m%dT%H%M%S')}"
    counts: dict[str, dict[str, int]] = {}
    for condition, records in records_by_condition.items():
        llm = load_judgments(judgment_file(judgments_dir, "llm", condition))
        path = judgment_file(judgments_dir, "nli", condition)
        guard_output(path, resume=resume, fresh=fresh, tag=tag)
        existing = load_judgments(path)
        stats = {"records": len(records), "judged": 0, "missing_llm": 0}
        for record in records:
            key = (str(record["item_id"]), int(record["turn_index"]))
            source = llm.get(key)
            if (
                source is None
                or source.get("error")
                or source.get("attempt_id") != record.get("attempt_id")
            ):
                stats["missing_llm"] += 1
                continue
            prev = existing.get(key)
            if prev is not None and prev.get("claims_fingerprint") == claims_fingerprint(source):
                continue
            texts = {
                str(e["chunk_id"]): str(e.get("text", ""))
                for e in record.get("evidence_texts") or []
            }

            def resolve(chunk_id: str, _texts: dict[str, str] = texts) -> str | None:
                if chunk_id in _texts:
                    return _texts[chunk_id]
                chunk = sidecar.get(chunk_id)
                return chunk.text if chunk is not None else None

            append_jsonl(path, judge.judge_turn(source, resolve))
            stats["judged"] += 1
        log(
            f"[judge nli {condition}] judged {stats['judged']} | "
            f"missing LLM judgments {stats['missing_llm']}"
        )
        counts[condition] = stats
    return counts


def model_family(model: str) -> str:
    """Model family of an Ollama tag (``gemma3:12b`` → ``gemma3``, ``qwen3.5:4b`` → ``qwen3``)."""
    name = model.split(":", 1)[0].lower()
    return name.split(".", 1)[0]


def same_family_models(judge_model: str, runs: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Application models (from the run records) sharing the judge model's family."""
    family = model_family(judge_model)
    used = {
        str(m) for recs in runs.values() for r in recs for m in (r.get("models") or {}).values()
    }
    return sorted(m for m in used if model_family(m) == family)


def load_gold_notes(dataset: Path) -> dict[str, str]:
    """``item_id → gold_notes`` (empty when the dataset file is unavailable)."""
    if not dataset.exists():
        _log(
            f"[judge] WARNING dataset {dataset} not found; behaviour judge runs without gold notes"
        )
        return {}
    return {item.id: item.gold_notes for item in load_dataset(dataset).items}


async def _main_async(args: argparse.Namespace) -> int:
    conditions = (
        None
        if args.conditions.lower() == "all"
        else [c.strip() for c in args.conditions.split(",") if c.strip()]
    )
    runs = select_records(load_run_records(args.runs_dir), conditions, args.split)
    if not runs:
        _log(f"[judge] no run records found in {args.runs_dir}")
        return 2
    args.judgments_dir.mkdir(parents=True, exist_ok=True)
    sidecar = EvidenceSidecar(args.judgments_dir)
    summary: dict[str, Any] = {"started_at": utc_now()}
    if args.judge in ("llm", "both"):
        from evaluation.cache import install_request_cache
        from evaluation.evidence import CorpusRetriever
        from medbridge.ai.llm_wrapper import get_llm_wrapper

        clash = same_family_models(args.judge_model, runs)
        if clash:
            _log(
                f"[judge] WARNING judge model {args.judge_model} is from the same family as the "
                f"application model(s) {clash}: risk of self-preference bias (use another family)"
            )
        cache = install_request_cache(None if args.no_cache else args.cache)
        llm_judge = LLMJudge(
            get_llm_wrapper(),
            CorpusRetriever(),
            LLMJudgeConfig(model=args.judge_model, judge_top_k=args.judge_top_k),
            on_evidence=sidecar.add,
        )
        summary["llm"] = await judge_with_llm(
            llm_judge,
            runs,
            load_gold_notes(args.dataset),
            args.judgments_dir,
            resume=args.resume,
            fresh=args.fresh,
        )
        if cache is not None:
            summary["cache"] = cache.stats()
            cache.close()
        await get_llm_wrapper().close()
    if args.judge in ("nli", "both"):
        from evaluation.judges.nli_judge import CrossEncoderNLI

        theta, theta_c = load_thresholds(args.calibration)
        nli = NLIJudge(CrossEncoderNLI(args.nli_model, device=args.nli_device), theta, theta_c)
        summary["nli"] = judge_with_nli(
            nli, runs, args.judgments_dir, sidecar, resume=args.resume, fresh=args.fresh
        )
    summary["finished_at"] = utc_now()
    _log(f"[judge] done: {summary}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Judge HalluBench run records")
    parser.add_argument("--judge", choices=["llm", "nli", "both"], default="both")
    parser.add_argument("--judge-model", default="gemma3:12b")
    parser.add_argument("--judge-top-k", type=int, default=3, help="judge-side passages per claim")
    parser.add_argument("--conditions", default="all", help="'all' or a comma list")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--judgments-dir", type=Path, default=JUDGMENTS_DIR)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="for gold_notes")
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--nli-device", default=None, help="mps | cuda | cpu (auto by default)")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume and args.fresh:
        _log("[judge] --resume and --fresh are mutually exclusive")
        return 2
    try:
        return asyncio.run(_main_async(args))
    except OutputExistsError as exc:
        _log(f"[judge] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
