"""
Judge CLI — labels the run records with the LLM judge and/or the NLI judge.

Outputs ``<judgments>/{llm,nli}/<condition>.jsonl`` keyed by
``(condition, item_id, turn_index)``; every judgment stores the run record's
``attempt_id`` so judgments of a re-run item are never mixed with stale ones.
Judge-side retrieved chunk texts are kept in ``<judgments>/evidence_chunks.jsonl``
and every invocation writes ``<judgments>/manifest_judge_<timestamp>.json``
(judge model + Ollama digest, cache provenance, workers, counts).

The NLI judge reuses the claims of the LLM judge, so run ``--judge llm`` first
(``--judge both`` does both in order).

Throughput: ``--workers N`` judges up to N responses concurrently (asyncio;
point ``--ollama-url`` at a dedicated Ollama server started with
``OLLAMA_NUM_PARALLEL >= N``). Judgments are still appended in the input order,
so the output files are identical whatever N is. The request cache is safe
under this concurrency (every cache operation is atomic) and is namespaced by
the judge model's Ollama digest.

Usage::

    python -m evaluation.judge --judge both --judge-model gemma3:12b --workers 4
    python -m evaluation.judge --judge nli --split dev --resume
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

from evaluation import manifest as mf
from evaluation.datasets.schema import load_dataset
from evaluation.io import (
    OutputExistsError,
    append_jsonl,
    guard_output,
    judgment_file,
    load_judgments,
    load_run_records,
    write_json,
)
from evaluation.judges.common import EvidenceSidecar, cited_chunks_by_claim
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
_In = TypeVar("_In")
_Out = TypeVar("_Out")


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


async def map_ordered(
    items: Sequence[_In],
    worker: Callable[[_In], Awaitable[_Out]],
    workers: int,
    sink: Callable[[int, _Out], None],
) -> None:
    """Run ``worker`` over ``items`` with at most ``workers`` in flight.

    ``sink(position, result)`` is called in INPUT order as soon as every earlier
    item has finished (results that complete early are buffered), so the
    output is deterministic for any degree of concurrency and progress is still
    persisted incrementally.
    """
    semaphore = asyncio.Semaphore(max(1, workers))
    done: dict[int, _Out] = {}
    next_position = 0

    async def run(position: int, item: _In) -> None:
        nonlocal next_position
        async with semaphore:
            result = await worker(item)
        done[position] = result
        while next_position in done:
            sink(next_position, done.pop(next_position))
            next_position += 1

    await asyncio.gather(*(run(i, item) for i, item in enumerate(items)))


async def judge_with_llm(
    judge: LLMJudge,
    records_by_condition: dict[str, list[dict[str, Any]]],
    gold_notes: dict[str, str],
    judgments_dir: Path,
    *,
    resume: bool,
    fresh: bool,
    workers: int = 1,
    log: Log = _log,
) -> dict[str, dict[str, int]]:
    """Run the LLM judge over every selected record; appends judgments as it goes.

    Each judgment carries ``judge_calls`` (number of judge LLM calls, how many
    were memoised or truncated, total latency) recorded by the judge itself, so
    the statistics stay per judgment under ``workers > 1``.
    """
    tag = f"llm-{time.strftime('%Y%m%dT%H%M%S')}"
    counts: dict[str, dict[str, int]] = {}
    for condition, records in records_by_condition.items():
        path = judgment_file(judgments_dir, "llm", condition)
        guard_output(path, resume=resume, fresh=fresh, tag=tag)
        counts[condition] = await _judge_condition_llm(
            judge, condition, records, gold_notes, path, workers, log
        )
    return counts


async def _judge_condition_llm(
    judge: LLMJudge,
    condition: str,
    records: list[dict[str, Any]],
    gold_notes: dict[str, str],
    path: Path,
    workers: int,
    log: Log,
) -> dict[str, int]:
    existing = load_judgments(path)
    todo = [r for r in records if needs_judging(existing, r)]
    stats = {"records": len(records), "judged": 0, "errors": 0, "templates": 0, "truncated": 0}
    history = histories(records)
    started = time.perf_counter()

    async def one(record: dict[str, Any]) -> dict[str, Any]:
        key = (str(record["item_id"]), int(record["turn_index"]))
        try:
            return await judge.judge_turn(record, gold_notes.get(key[0], ""), history.get(key, []))
        except Exception as exc:  # retrieval / infrastructure failure: record, continue
            judgment = judge.empty_judgment(record)
            judgment["error"] = f"{type(exc).__name__}: {exc}"
            return judgment

    def sink(position: int, judgment: dict[str, Any]) -> None:
        append_jsonl(path, judgment)
        stats["judged"] += 1
        stats["errors"] += 1 if judgment.get("error") else 0
        stats["templates"] += 1 if judgment.get("template") else 0
        stats["truncated"] += int((judgment.get("judge_calls") or {}).get("truncated") or 0)
        done = position + 1
        if done % 10 == 0 or done == len(todo):
            rate = (time.perf_counter() - started) / done
            log(
                f"[judge llm {condition}] {done}/{len(todo)} | errors {stats['errors']} | "
                f"ETA {rate * (len(todo) - done) / 60:.1f} min"
            )

    await map_ordered(todo, one, workers, sink)
    return stats


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
    """Score the LLM judge's claims with the NLI model for every selected record.

    A stored NLI judgment is reused only when it belongs to the same run
    attempt, the same LLM-judge claims (fingerprint) and the same NLI model;
    a re-run item (new ``attempt_id``) is always re-scored, even when its claims
    are unchanged, so the analysis can join it to the current run record.
    """
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
            if (
                prev is not None
                and not prev.get("error")
                and prev.get("attempt_id") == source.get("attempt_id")
                and prev.get("judge_model") == judge.scorer.name
                and prev.get("claims_fingerprint") == claims_fingerprint(source)
            ):
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

            cited = cited_chunks_by_claim(record, source.get("claims") or [])
            append_jsonl(path, judge.judge_turn(source, resolve, cited))
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


def install_judge_cache(
    path: Path | None, judge_model: str, ollama_url: str
) -> tuple[Any, dict[str, Any]]:
    """Request cache namespaced by the judge model's Ollama digest (or disabled, with why)."""
    from evaluation.cache import install_request_cache, model_digests

    if path is None:
        install_request_cache(None)
        return None, {"status": "disabled"}
    try:
        digests = model_digests(mf.ollama_models(ollama_url), [judge_model])
    except LookupError as exc:
        _log(f"[judge] WARNING request cache disabled: {exc}")
        install_request_cache(None)
        return None, {"status": f"disabled: {exc}"}
    cache = install_request_cache(path, model_digests=digests)
    return cache, {"status": "enabled", **(cache.stats() if cache else {})}


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
    stamp = time.strftime("%Y%m%dT%H%M%S")
    manifest_path = args.judgments_dir / f"manifest_judge_{stamp}.json"
    summary: dict[str, Any] = {
        "started_at": utc_now(),
        "command": mf.mask_url_credentials(" ".join(sys.argv)),
        "judge": args.judge,
        "judge_model": args.judge_model,
        "workers": args.workers,
        "split": args.split,
        "conditions": conditions,
        "git": mf.git_info(),
    }
    if args.judge in ("llm", "both"):
        from evaluation.evidence import CorpusRetriever
        from medbridge.ai.llm_wrapper import get_llm_wrapper
        from medbridge.config import get_settings

        clash = same_family_models(args.judge_model, runs)
        if clash:
            _log(
                f"[judge] WARNING judge model {args.judge_model} is from the same family as the "
                f"application model(s) {clash}: risk of self-preference bias (use another family)"
            )
        ollama_url = get_settings().OLLAMA_BASE_URL
        summary["ollama"] = mf.ollama_models(ollama_url, [args.judge_model])
        cache, summary["cache"] = install_judge_cache(
            None if args.no_cache else args.cache, args.judge_model, ollama_url
        )
        write_json(manifest_path, summary)
        llm_judge = LLMJudge(
            get_llm_wrapper(),
            CorpusRetriever(),
            LLMJudgeConfig(model=args.judge_model, judge_top_k=args.judge_top_k),
            on_evidence=sidecar.add,
        )
        summary["llm_config"] = dict(vars(llm_judge.config))
        summary["llm"] = await judge_with_llm(
            llm_judge,
            runs,
            load_gold_notes(args.dataset),
            args.judgments_dir,
            resume=args.resume,
            fresh=args.fresh,
            workers=args.workers,
        )
        if cache is not None:
            summary["cache"] = {"status": "enabled", **cache.stats()}
            cache.close()
        await get_llm_wrapper().close()
    if args.judge in ("nli", "both"):
        from evaluation.judges.nli_judge import CrossEncoderNLI

        theta, theta_c = load_thresholds(args.calibration)
        nli = NLIJudge(CrossEncoderNLI(args.nli_model, device=args.nli_device), theta, theta_c)
        summary["nli_model"] = args.nli_model
        summary["nli"] = judge_with_nli(
            nli, runs, args.judgments_dir, sidecar, resume=args.resume, fresh=args.fresh
        )
    summary["finished_at"] = utc_now()
    write_json(manifest_path, summary)
    _log(f"[judge] done: {summary.get('llm')} {summary.get('nli')} → {manifest_path}")
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="responses judged concurrently (needs an Ollama server with parallel slots)",
    )
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama base URL for the judge (sets OLLAMA_BASE_URL), e.g. a dedicated server",
    )
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
    if args.workers < 1:
        _log("[judge] --workers must be >= 1")
        return 2
    mf.apply_settings_env({"OLLAMA_BASE_URL": args.ollama_url})
    try:
        return asyncio.run(_main_async(args))
    except OutputExistsError as exc:
        _log(f"[judge] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
