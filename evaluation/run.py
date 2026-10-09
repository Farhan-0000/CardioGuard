"""
HalluBench runner — executes every benchmark item through the real pipeline
under each study condition (CLAUDE_AGENT_PROMPT §6.2).

Each condition is an ``AblationConfig`` preset passed to
``PipelineOrchestrator.process_message(..., ablation=...)``; the runner never
re-implements pipeline logic. Every item gets a fresh session; its turns are
sent in order. One JSON line is written per turn to
``<out>/<condition>.jsonl``::

    {run_id, attempt_id, condition, item_id, category, subcategory, split, style,
     turn_index, n_turns, user_text, gold_action, acceptable_actions, session_id,
     action, response_text, citations, explanation, evidence_texts, latency_ms,
     llm_calls, models, soft_ask_count, error, evidence_error, timestamp}

``evidence_texts`` holds the FULL text of the chunks passed to Gate 2 / the
generator (fetched from Qdrant by ``chunk_id``; the explanation only carries
excerpts) so the judges see exactly what the pipeline saw.

Resumable (``--resume`` skips items whose latest attempt completed without
error), robust (a failing item is recorded with ``error`` and the run goes
on), memoised (``evaluation.cache.DiskRequestCache``) and documented by a run
manifest ``<out>/manifest_<run_id>.json``.

Usage::

    python -m evaluation.run --conditions all --split all
    python -m evaluation.run --conditions C1,C5 --split dev --limit 12   # dry run
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import functools
import hashlib
import os
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from evaluation import manifest as mf
from evaluation.datasets.schema import (
    CATEGORIES,
    HalluItem,
    SplitFilter,
    filter_items,
    load_dataset,
)
from evaluation.io import (
    OutputExistsError,
    append_jsonl,
    guard_output,
    iter_jsonl,
    latest_complete_attempts,
    run_file,
    write_json,
)
from evaluation.paths import CACHE_PATH, DEFAULT_DATASET, RUNS_DIR
from medbridge.core.pipeline_types import AblationConfig

if TYPE_CHECKING:
    from medbridge.api.schemas.responses import MessageResponse

#: ``--conditions all`` — the six study conditions plus the two single variants.
DEFAULT_CONDITIONS: tuple[str, ...] = ("C0", "C1", "C2", "C3", "C4", "C5", "V1", "V2")


class RunnerError(RuntimeError):
    """Configuration problem that should stop the run before any item is executed."""


@dataclass
class TurnOutput:
    """What the executor returns for one turn."""

    response: MessageResponse
    latency_ms: float
    llm_calls: list[dict[str, Any]] = field(default_factory=list)


class TurnExecutor(Protocol):
    """Runs turns through the pipeline (the real one, or a fake in tests)."""

    async def new_session(self) -> uuid.UUID: ...

    async def run_turn(
        self, session_id: uuid.UUID, text: str, ablation: AblationConfig
    ) -> TurnOutput: ...

    async def evidence(self, explanation: dict[str, Any]) -> list[dict[str, Any]]: ...

    async def close(self) -> None: ...


@dataclass
class RunConfig:
    """Runner options (mirrors the CLI)."""

    dataset: Path = DEFAULT_DATASET
    conditions: list[str] = field(default_factory=lambda: list(DEFAULT_CONDITIONS))
    split: SplitFilter = "all"
    limit: int | None = None
    resume: bool = False
    fresh: bool = False
    workers: int = 1
    model: str | None = None
    out: Path = RUNS_DIR
    cache_path: Path | None = CACHE_PATH
    run_id: str = ""

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = new_run_id()
        if self.workers < 1:
            raise RunnerError("--workers must be >= 1")
        if self.resume and self.fresh:
            raise RunnerError("--resume and --fresh are mutually exclusive")


def new_run_id() -> str:
    """Sortable run id: UTC timestamp + short random suffix."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def parse_conditions(spec: str) -> list[str]:
    """Parse ``all`` / ``C0,C5,V12`` / ``all,V12`` into preset names (order kept, deduplicated)."""
    known = AblationConfig.all_names()
    out: list[str] = []
    for token in (t.strip() for t in spec.split(",")):
        if not token:
            continue
        if token.lower() == "all":
            out.extend(DEFAULT_CONDITIONS)
            continue
        name = token.upper()
        if name not in known:
            raise RunnerError(f"unknown condition {token!r}; known: {', '.join(known)}")
        out.append(name)
    if not out:
        raise RunnerError("no conditions selected")
    return list(dict.fromkeys(out))


def select_items(items: Sequence[HalluItem], limit: int | None) -> list[HalluItem]:
    """First ``limit`` items taken round-robin across categories (dataset order kept).

    Balanced subsets make ``--limit`` dry runs and the model-selection
    experiment representative of every category.
    """
    if limit is None or limit >= len(items):
        return list(items)
    buckets: dict[str, list[HalluItem]] = {c: [] for c in CATEGORIES}
    for item in items:
        buckets[item.category].append(item)
    chosen: list[HalluItem] = []
    while len(chosen) < limit and any(buckets.values()):
        for category in CATEGORIES:
            if buckets[category] and len(chosen) < limit:
                chosen.append(buckets[category].pop(0))
    order = {item.id: i for i, item in enumerate(items)}
    return sorted(chosen, key=lambda it: order[it.id])


def completed_items(path: Path) -> set[str]:
    """Items whose latest complete attempt has no turn with an error."""
    done: dict[str, bool] = {}
    for rec in latest_complete_attempts(iter_jsonl(path)):
        item_id = str(rec["item_id"])
        done[item_id] = done.get(item_id, True) and not rec.get("error")
    return {item_id for item_id, ok in done.items() if ok}


def make_record(
    *,
    run_id: str,
    attempt_id: str,
    condition: str,
    item: HalluItem,
    turn_index: int,
    output: TurnOutput | None = None,
    evidence_texts: list[dict[str, Any]] | None = None,
    error: str | None = None,
    evidence_error: str | None = None,
    session_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    """Build the JSONL record for one turn (``output=None`` for failed turns)."""
    turn = item.turns[turn_index]
    record: dict[str, Any] = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "condition": condition,
        "item_id": item.id,
        "category": item.category,
        "subcategory": item.subcategory,
        "split": item.split,
        "style": item.style,
        "turn_index": turn_index,
        "n_turns": len(item.turns),
        "user_text": turn.text,
        "gold_action": turn.gold_action,
        "acceptable_actions": list(turn.acceptable_actions),
        "session_id": str(session_id) if session_id else None,
        "action": None,
        "response_text": "",
        "citations": [],
        "explanation": {},
        "evidence_texts": [],
        "latency_ms": None,
        "llm_calls": [],
        "models": {},
        "soft_ask_count": None,
        "error": error,
        "evidence_error": evidence_error,
        "timestamp": mf.utc_now(),
    }
    if output is None:
        return record
    response = output.response
    explanation = (
        response.explanation.model_dump(mode="json") if response.explanation is not None else {}
    )
    record.update(
        action=str(getattr(response.action, "value", response.action)),
        response_text=response.response_text,
        citations=[c.model_dump(mode="json") for c in response.citations],
        explanation=explanation,
        evidence_texts=evidence_texts or [],
        latency_ms=round(output.latency_ms, 1),
        llm_calls=explanation.get("llm_calls") or output.llm_calls,
        models=explanation.get("models") or {},
        soft_ask_count=response.soft_ask_count,
    )
    return record


def retrieval_discarded(explanation: dict[str, Any]) -> bool:
    """True when speculative retrieval ran but was discarded (Gate 1 said SOFT-ASK)."""
    return any(
        t.get("stage") in {"retrieval", "rerank"} and t.get("status") == "discarded"
        for t in explanation.get("timings") or []
    )


def prepare_outputs(config: RunConfig, log: Callable[[str], None]) -> None:
    """Refuse to mix runs unless ``--resume``; ``--fresh`` moves old result files aside."""
    config.out.mkdir(parents=True, exist_ok=True)
    for condition in config.conditions:
        try:
            backup = guard_output(
                run_file(config.out, condition),
                resume=config.resume,
                fresh=config.fresh,
                tag=config.run_id,
            )
        except OutputExistsError as exc:
            raise RunnerError(str(exc)) from exc
        if backup is not None:
            log(f"[run] moved previous results to {backup}")


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes:d}m{secs:02d}s"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class EvaluationRunner:
    """Runs (condition × item) through an executor and appends per-turn records."""

    def __init__(
        self, config: RunConfig, executor: TurnExecutor, log: Callable[[str], None] = _log
    ) -> None:
        self.config = config
        self.executor = executor
        self.log = log

    def items(self) -> list[HalluItem]:
        """Dataset items selected by split and ``--limit``."""
        bench = load_dataset(self.config.dataset)
        return select_items(filter_items(bench.items, self.config.split), self.config.limit)

    async def run(self) -> dict[str, dict[str, int]]:
        """Prepare the output files and execute every selected (condition, item) pair."""
        prepare_outputs(self.config, self.log)
        return await self.execute(self.items())

    async def execute(self, items: list[HalluItem]) -> dict[str, dict[str, int]]:
        """Execute ``items`` under every configured condition; returns per-condition counts."""
        counts: dict[str, dict[str, int]] = {}
        for condition in self.config.conditions:
            counts[condition] = await self._run_condition(condition, items)
        return counts

    async def _run_condition(self, condition: str, items: list[HalluItem]) -> dict[str, int]:
        path = run_file(self.config.out, condition)
        ablation = AblationConfig.preset(condition)
        done = completed_items(path) if self.config.resume else set()
        todo = [it for it in items if it.id not in done]
        stats = {"items": len(items), "skipped": len(items) - len(todo), "run": 0, "errors": 0}
        self.log(f"[{condition}] {len(todo)} items to run ({stats['skipped']} already complete)")
        semaphore = asyncio.Semaphore(self.config.workers)
        started = time.perf_counter()

        async def worker(item: HalluItem) -> None:
            async with semaphore:
                ok = await self._run_item(condition, ablation, item, path)
            stats["run"] += 1
            stats["errors"] += 0 if ok else 1
            elapsed = time.perf_counter() - started
            remaining = len(todo) - stats["run"]
            eta = elapsed / stats["run"] * remaining
            self.log(
                f"[{condition}] {stats['skipped'] + stats['run']}/{len(items)} items "
                f"({item.id}{'' if ok else ' ERROR'}) | errors {stats['errors']} | "
                f"elapsed {_fmt_duration(elapsed)} | ETA {_fmt_duration(eta)}"
            )

        await asyncio.gather(*(worker(item) for item in todo))
        return stats

    async def _run_item(
        self, condition: str, ablation: AblationConfig, item: HalluItem, path: Path
    ) -> bool:
        """Run all turns of one item in a fresh session. Never raises; returns success."""
        record = functools.partial(
            make_record,
            run_id=self.config.run_id,
            attempt_id=uuid.uuid4().hex,
            condition=condition,
            item=item,
        )
        try:
            session_id = await self.executor.new_session()
        except Exception as exc:  # infrastructure failure (DB down…): record and move on
            message = f"session creation failed: {type(exc).__name__}: {exc}"
            for index in range(len(item.turns)):
                append_jsonl(path, record(turn_index=index, error=message))
            return False
        for index, turn in enumerate(item.turns):
            try:
                output = await self.executor.run_turn(session_id, turn.text, ablation)
            except Exception as exc:  # the orchestrator never throws; this is DB/infra level
                message = f"{type(exc).__name__}: {exc}"
                append_jsonl(
                    path,
                    record(turn_index=index, error=message, session_id=session_id),
                )
                for rest in range(index + 1, len(item.turns)):
                    append_jsonl(
                        path,
                        record(
                            turn_index=rest,
                            session_id=session_id,
                            error=f"skipped: turn {index} failed",
                        ),
                    )
                return False
            explanation = (
                output.response.explanation.model_dump(mode="json")
                if output.response.explanation is not None
                else {}
            )
            evidence: list[dict[str, Any]] = []
            evidence_error = None
            if not retrieval_discarded(explanation):
                try:
                    evidence = await self.executor.evidence(explanation)
                except Exception as exc:  # Qdrant hiccup: keep the turn, flag the evidence
                    evidence_error = f"{type(exc).__name__}: {exc}"
            append_jsonl(
                path,
                record(
                    turn_index=index,
                    output=output,
                    evidence_texts=evidence,
                    evidence_error=evidence_error,
                    session_id=session_id,
                ),
            )
        return True


# ----------------------------------------------------------------------
# Real executor (live pipeline)
# ----------------------------------------------------------------------


def _call_stat_to_dict(stat: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(stat) and not isinstance(stat, type):
        return dataclasses.asdict(stat)
    if isinstance(stat, dict):
        return dict(stat)
    return dict(vars(stat))


class PipelineExecutor:
    """Executes turns in-process through ``PipelineOrchestrator`` with a real DB session."""

    def __init__(self) -> None:
        from evaluation.evidence import EvidenceStore
        from medbridge.core.orchestrator import PipelineOrchestrator
        from medbridge.db.connection import get_sessionmaker

        self._orchestrator = PipelineOrchestrator()
        self._maker = get_sessionmaker()
        self._store = EvidenceStore()

    async def new_session(self) -> uuid.UUID:
        """Create a session (``create_session`` also creates its data key) and commit."""
        from medbridge.state.session_manager import create_session

        async with self._maker() as db:
            session_id = await create_session(db)
            await db.commit()
        return session_id

    async def run_turn(
        self, session_id: uuid.UUID, text: str, ablation: AblationConfig
    ) -> TurnOutput:
        """One ``process_message`` call in its own transaction, with per-turn LLM call stats."""
        from medbridge.ai.llm_wrapper import llm_call_log

        async with self._maker() as db:
            with llm_call_log() as calls:
                started = time.perf_counter()
                response = await self._orchestrator.process_message(
                    session_id, text, db, ablation=ablation
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
            await db.commit()
        return TurnOutput(
            response=response,
            latency_ms=latency_ms,
            llm_calls=[_call_stat_to_dict(c) for c in calls],
        )

    async def evidence(self, explanation: dict[str, Any]) -> list[dict[str, Any]]:
        """Full text of the explanation's retrieved chunks."""
        from evaluation.evidence import evidence_from_explanation

        ids = [str(r.get("chunk_id")) for r in explanation.get("retrieval") or []]
        payloads = await self._store.get_many(ids) if ids else {}
        return evidence_from_explanation(explanation, payloads)

    async def close(self) -> None:
        from medbridge.ai.llm_wrapper import get_llm_wrapper
        from medbridge.db.connection import close_db

        await self._store.close()
        await get_llm_wrapper().close()
        await close_db()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def apply_env_overrides(model: str | None, database_url: str | None) -> None:
    """Override settings for this process only (before any pipeline object is built)."""
    if model:
        os.environ["LLM_MODEL"] = model
        os.environ["LLM_MODEL_GATES"] = ""
        os.environ["LLM_MODEL_GENERATOR"] = ""
    if database_url:
        os.environ["DATABASE_URL"] = database_url
    from medbridge.config import get_settings

    get_settings.cache_clear()


def build_manifest(config: RunConfig, items: list[HalluItem]) -> dict[str, Any]:
    """Run manifest: code version, models, hardware, seeds, settings (no secrets)."""
    from medbridge.config import get_settings

    settings = get_settings()
    roles = {"gates": settings.gates_model, "generator": settings.generator_model}
    dataset_bytes = config.dataset.read_bytes()
    return {
        "run_id": config.run_id,
        "status": "running",
        "started_at": mf.utc_now(),
        "finished_at": None,
        "command": mf.mask_url_credentials(" ".join(sys.argv)),
        "git": mf.git_info(),
        "dataset": {
            "path": str(config.dataset),
            "sha256": hashlib.sha256(dataset_bytes).hexdigest(),
            "n_items_selected": len(items),
            "item_ids": [it.id for it in items],
        },
        "conditions": config.conditions,
        "split": config.split,
        "limit": config.limit,
        "workers": config.workers,
        "model_override": config.model,
        "models": roles,
        "ollama": mf.ollama_models(settings.OLLAMA_BASE_URL, sorted(set(roles.values()))),
        "seeds": {
            "llm_seed": settings.LLM_SEED,
            "temperature_gates": settings.LLM_TEMPERATURE_GATES,
            "temperature_generator": settings.LLM_TEMPERATURE_GENERATOR,
        },
        "hardware": mf.hardware_info(),
        "packages": mf.package_versions(),
        "settings": mf.settings_snapshot(),
        "cache": None,
        "counts": None,
    }


def _install_cache(path: Path | None) -> tuple[Any, dict[str, Any]]:
    if path is None:
        return None, {"status": "disabled"}
    try:
        from evaluation.cache import install_request_cache

        cache = install_request_cache(path)
    except ImportError as exc:
        _log(f"[run] WARNING request cache unavailable ({exc}); running without memoisation")
        return None, {"status": f"unavailable: {exc}"}
    return cache, {"status": "enabled", **(cache.stats() if cache else {})}


async def run_cli(
    config: RunConfig, executor_factory: Callable[[], TurnExecutor] = PipelineExecutor
) -> int:
    """Run the benchmark and write the manifest. Returns a process exit code."""
    items = select_items(
        filter_items(load_dataset(config.dataset).items, config.split), config.limit
    )
    if not items:
        _log("[run] no items selected (check --split / --dataset)")
        return 2
    try:
        prepare_outputs(config, _log)
    except RunnerError as exc:
        _log(f"[run] {exc}")
        return 2
    manifest = build_manifest(config, items)
    manifest_path = config.out / f"manifest_{config.run_id}.json"
    cache, manifest["cache"] = _install_cache(config.cache_path)
    write_json(manifest_path, manifest)
    executor = executor_factory()
    status = "interrupted"
    counts: dict[str, dict[str, int]] = {}
    try:
        counts = await EvaluationRunner(config, executor).execute(items)
        status = "finished"
    finally:
        await executor.close()
        if cache is not None:
            manifest["cache"] = {"status": "enabled", **cache.stats()}
            cache.close()
        manifest.update(status=status, finished_at=mf.utc_now(), counts=counts)
        write_json(manifest_path, manifest)
    total_errors = sum(c.get("errors", 0) for c in counts.values())
    _log(f"[run] {status}; manifest {manifest_path}; items with errors: {total_errors}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MedBridge-HalluBench through the pipeline")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--conditions", default="all", help="'all' (C0..C5,V1,V2) or a comma list, e.g. C0,C5,V12"
    )
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument(
        "--limit", type=int, default=None, help="first N items (category round-robin)"
    )
    parser.add_argument("--resume", action="store_true", help="skip items already completed")
    parser.add_argument(
        "--fresh", action="store_true", help="move existing results aside and restart"
    )
    parser.add_argument("--workers", type=int, default=1, help="concurrent items (GPU is shared)")
    parser.add_argument("--model", default=None, help="override LLM_MODEL for every role")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("EVAL_DATABASE_URL"),
        help="database for evaluation sessions (default: DATABASE_URL)",
    )
    parser.add_argument("--out", type=Path, default=RUNS_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH, help="request cache file")
    parser.add_argument("--no-cache", action="store_true", help="disable request memoisation")
    parser.add_argument("--run-id", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        conditions = parse_conditions(args.conditions)
        config = RunConfig(
            dataset=args.dataset,
            conditions=conditions,
            split=args.split,
            limit=args.limit,
            resume=args.resume,
            fresh=args.fresh,
            workers=args.workers,
            model=args.model,
            out=args.out,
            cache_path=None if args.no_cache else args.cache,
            run_id=args.run_id,
        )
    except RunnerError as exc:
        _log(f"[run] {exc}")
        return 2
    apply_env_overrides(args.model, args.database_url)
    return asyncio.run(run_cli(config))


if __name__ == "__main__":
    raise SystemExit(main())
