"""
HalluBench runner — executes every benchmark item through the real pipeline
under each study condition (CLAUDE_AGENT_PROMPT §6.2).

Each condition is an ``AblationConfig`` preset passed to
``PipelineOrchestrator.process_message(..., ablation=...)``; the runner never
re-implements pipeline logic. Every item gets a fresh session; its turns are
sent in order. One JSON line is written per turn to
``<out>/<condition>.jsonl``::

    {run_id, attempt_id, condition, item_id, category, subcategory, split, style, tags,
     turn_index, n_turns, user_text, gold_action, acceptable_actions, session_id,
     action, response_text, citations, explanation, evidence_texts, latency_ms,
     llm_calls, models, soft_ask_count, error, evidence_error,
     degraded, degraded_reasons, calibration, run_signature, retry_of, cache_bypass,
     timestamp}

``evidence_texts`` holds the FULL text of the chunks passed to Gate 2 / the
generator (fetched from Qdrant by ``chunk_id``; the explanation only carries
excerpts) so the judges see exactly what the pipeline saw.

Degraded turns (``evaluation.degraded``): a turn whose explanation shows any
fallback or failure (Gate 1/2 spec fallback, generator fallback template,
extraction failure, failed or empty retrieval, Gate 3 unavailable, failed
persistence / state load, failed LLM call, runner error) is recorded with
``degraded = true`` and its reasons; later turns of the same item inherit
``earlier_turn_degraded``. A degraded attempt is not complete: ``--resume``
re-runs it, ``--retry-degraded`` re-runs ONLY degraded items. When an item is
re-run after a degraded attempt, cached responses of the LLM calls that failed
are bypassed (``evaluation.cache.bypass_cache_reads``) so they are recomputed.
A circuit breaker stops the run after ``--max-consecutive-degraded`` degraded
items in a row (an Ollama / Qdrant outage must not fill the result files), and
a pre-flight check refuses to start when Ollama, the models, Qdrant or (for
V2/V12) the grounding verifier are unavailable.

Calibration gate (protocol in ``evaluation/README.md``): V1 / V2 / V12 on any
split other than ``dev`` need ``evaluation/results/calibration.json`` with the
dev-calibrated ``v1`` / ``v2`` thresholds. The runner injects them into the
settings of this process before any pipeline object is built, verifies they
are the effective values, and records the values + file sha256 in the run
manifest and in every record. Without calibration it refuses to run unless
``--allow-uncalibrated`` (debugging only; marked in the manifest and records).

Resumable (``--resume``; refused when the existing records were produced with a
different model, dataset or calibration), robust (a failing item is recorded
with ``error`` and the run goes on), memoised (``evaluation.cache``, namespaced
by the Ollama model digests) and documented by a run manifest
``<out>/manifest_<run_id>.json``.

Usage::

    python -m evaluation.run --conditions C5 --split dev                 # protocol step 1
    python -m evaluation.run --conditions all --split test --resume      # protocol step 4
    python -m evaluation.run --conditions all --split test --retry-degraded
    python -m evaluation.run --conditions C1,C5 --split dev --limit 12   # dry run
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
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
from evaluation.degraded import annotate_attempt, bypass_calls, degradation_reasons
from evaluation.io import (
    OutputExistsError,
    append_jsonl,
    guard_output,
    iter_jsonl,
    latest_complete_attempts,
    read_json,
    run_file,
    write_json,
)
from evaluation.paths import CACHE_PATH, CALIBRATION_PATH, DEFAULT_DATASET, RUNS_DIR
from medbridge.core.pipeline_types import AblationConfig

if TYPE_CHECKING:
    from medbridge.api.schemas.responses import MessageResponse

#: ``--conditions all`` — the six study conditions plus the two single variants.
DEFAULT_CONDITIONS: tuple[str, ...] = ("C0", "C1", "C2", "C3", "C4", "C5", "V1", "V2")

#: Settings calibrated on the dev split (``calibration.json[block]["thresholds"]``).
CALIBRATED_SETTINGS: dict[str, tuple[str, ...]] = {
    "v1": (
        "EVIDENCE_MIN_RERANK_ANSWER",
        "EVIDENCE_MIN_RERANK_ANY",
        "EVIDENCE_MIN_ENTITY_COVERAGE",
    ),
    "v2": ("GROUNDING_ENTAILMENT_THRESHOLD", "GROUNDING_MAX_UNSUPPORTED_FRACTION"),
}


class RunnerError(RuntimeError):
    """Configuration problem that should stop the run before any item is executed."""


class CalibrationError(RunnerError):
    """A calibrated variant was requested on a held-out split without calibration."""


@dataclass
class TurnOutput:
    """What the executor returns for one turn."""

    response: MessageResponse
    latency_ms: float
    llm_calls: list[dict[str, Any]] = field(default_factory=list)


class TurnExecutor(Protocol):
    """Runs turns through the pipeline (the real one, or a fake in tests).

    An executor may also provide ``async def preflight(conditions) -> list[str]``
    (problems that must stop the run); it is called when present.
    """

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
    retry_degraded: bool = False
    workers: int = 1
    model: str | None = None
    out: Path = RUNS_DIR
    cache_path: Path | None = CACHE_PATH
    calibration_path: Path = CALIBRATION_PATH
    allow_uncalibrated: bool = False
    max_consecutive_degraded: int = 10
    preflight: bool = True
    run_id: str = ""

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = new_run_id()
        if self.workers < 1:
            raise RunnerError("--workers must be >= 1")
        if self.resume and self.fresh:
            raise RunnerError("--resume and --fresh are mutually exclusive")
        if self.retry_degraded and self.fresh:
            raise RunnerError("--retry-degraded and --fresh are mutually exclusive")
        if self.max_consecutive_degraded < 0:
            raise RunnerError("--max-consecutive-degraded must be >= 0")

    @property
    def appends(self) -> bool:
        """True when new attempts are appended to existing result files."""
        return self.resume or self.retry_degraded


def new_run_id() -> str:
    """Sortable run id: UTC timestamp + short random suffix."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def parse_conditions(spec: str) -> list[str]:
    """Parse ``all`` / ``C0,C5,V12`` / ``all,V12`` into preset names (order kept, deduplicated).

    Any preset ``AblationConfig.preset`` knows is accepted (e.g. the optional
    ``C1G`` reference = retrieval + production prompt, no gates).
    """
    known = AblationConfig.all_names()
    out: list[str] = []
    for token in (t.strip() for t in spec.split(",")):
        if not token:
            continue
        if token.lower() == "all":
            out.extend(DEFAULT_CONDITIONS)
            continue
        name = token.upper()
        try:
            AblationConfig.preset(name)
        except ValueError:
            raise RunnerError(f"unknown condition {token!r}; known: {', '.join(known)}") from None
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


# ----------------------------------------------------------------------
# Attempt bookkeeping (resume / retry-degraded)
# ----------------------------------------------------------------------


def latest_attempts(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Latest complete attempt of every item, as ``{item_id: records sorted by turn}``."""
    out: dict[str, list[dict[str, Any]]] = {}
    for rec in latest_complete_attempts(iter_jsonl(path)):
        out.setdefault(str(rec["item_id"]), []).append(rec)
    return out


def attempt_reasons(records: Sequence[dict[str, Any]]) -> list[str]:
    """All degradation reasons of one attempt (stored or recomputed, propagated over turns)."""
    reasons: list[str] = []
    for turn_reasons in annotate_attempt(records):
        reasons.extend(r for r in turn_reasons if r not in reasons)
    return reasons


def completed_items(path: Path) -> set[str]:
    """Items whose latest complete attempt has no error and no degraded turn."""
    return {
        item_id
        for item_id, records in latest_attempts(path).items()
        if not attempt_reasons(records)
    }


def degraded_items(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Items whose latest complete attempt has an error or a degraded turn (with its records)."""
    return {
        item_id: records
        for item_id, records in latest_attempts(path).items()
        if attempt_reasons(records)
    }


# ----------------------------------------------------------------------
# Calibration gate
# ----------------------------------------------------------------------


def calibration_blocks(condition: str) -> tuple[str, ...]:
    """Calibration blocks (``v1``, ``v2``) a condition's preset depends on."""
    preset = AblationConfig.preset(condition)
    blocks: list[str] = []
    if preset.calibrated_evidence_gate:
        blocks.append("v1")
    if preset.grounding_verifier:
        blocks.append("v2")
    return tuple(blocks)


@dataclass
class CalibrationPlan:
    """Which calibrated settings this run injects, from which file (sha256)."""

    path: str
    sha256: str | None
    status: str  # not_required | calibrated | uncalibrated_dev | uncalibrated_allowed
    required: list[str]
    values: dict[str, float]
    missing: list[str]
    complete_blocks: list[str]
    effective: dict[str, float] = field(default_factory=dict)

    def env(self) -> dict[str, str | None]:
        """Environment overrides for the injected values."""
        return {key: repr(float(value)) for key, value in self.values.items()}

    def for_condition(self, condition: str) -> dict[str, Any]:
        """The calibration record stored with every turn of ``condition``."""
        blocks = calibration_blocks(condition)
        if not blocks:
            return {"required": [], "status": "not_required"}
        keys = [key for block in blocks for key in CALIBRATED_SETTINGS[block]]
        calibrated = all(block in self.complete_blocks for block in blocks)
        return {
            "required": list(blocks),
            "status": "calibrated" if calibrated else self.status,
            "sha256": self.sha256,
            "path": self.path,
            "values": {key: self.effective.get(key, self.values.get(key)) for key in keys},
        }

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan_calibration(
    conditions: Sequence[str], split: str, path: Path, allow_uncalibrated: bool
) -> CalibrationPlan:
    """Resolve the calibrated settings for the selected conditions.

    Raises ``CalibrationError`` when a calibrated variant would run on a split
    other than ``dev`` without the dev-calibrated values (and no override).
    """
    required = sorted({block for c in conditions for block in calibration_blocks(c)})
    exists = path.exists()
    plan = CalibrationPlan(
        path=str(path),
        sha256=file_sha256(path) if exists and required else None,
        status="not_required",
        required=required,
        values={},
        missing=[],
        complete_blocks=[],
    )
    if not required:
        return plan
    data = read_json(path) if exists else {}
    for block in required:
        section = data.get(block) if isinstance(data, dict) else None
        thresholds = (section or {}).get("thresholds") if isinstance(section, dict) else None
        thresholds = thresholds if isinstance(thresholds, dict) else {}
        block_missing = [k for k in CALIBRATED_SETTINGS[block] if thresholds.get(k) is None]
        plan.missing.extend(f"{block}.{k}" for k in block_missing)
        if not block_missing:
            plan.complete_blocks.append(block)
            plan.values.update({k: float(thresholds[k]) for k in CALIBRATED_SETTINGS[block]})
    if not plan.missing:
        plan.status = "calibrated"
    elif split == "dev":
        plan.status = "uncalibrated_dev"
    elif allow_uncalibrated:
        plan.status = "uncalibrated_allowed"
    else:
        where = f"{path}" if exists else f"{path} (missing)"
        raise CalibrationError(
            f"conditions {', '.join(c for c in conditions if calibration_blocks(c))} on split "
            f"{split!r} need dev-calibrated thresholds, but {where} lacks "
            f"{', '.join(plan.missing)}. Follow the protocol (README §1): run C5 on dev, judge "
            "dev, run `python -m evaluation.calibrate`, then run the variants on test. "
            "--allow-uncalibrated exists for debugging only."
        )
    return plan


def apply_calibration(plan: CalibrationPlan) -> None:
    """Inject the calibrated values into this process's settings and verify them."""
    if not plan.values:
        return
    mf.apply_settings_env(plan.env())
    from medbridge.config import get_settings

    settings = get_settings()
    effective = {key: float(getattr(settings, key)) for key in plan.values}
    wrong = {k: v for k, v in effective.items() if abs(v - plan.values[k]) > 1e-12}
    if wrong:
        raise CalibrationError(f"calibrated settings not effective after injection: {wrong}")
    plan.effective = effective


def effective_calibrated_settings(plan: CalibrationPlan) -> None:
    """Record the values actually in force for every required setting (defaults included)."""
    from medbridge.config import get_settings

    settings = get_settings()
    for block in plan.required:
        for key in CALIBRATED_SETTINGS[block]:
            plan.effective.setdefault(key, float(getattr(settings, key)))


# ----------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------


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
    calibration: dict[str, Any] | None = None,
    run_signature: dict[str, Any] | None = None,
    retry_of: str | None = None,
    cache_bypass: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the JSONL record for one turn (``output=None`` for failed turns).

    ``degraded`` / ``degraded_reasons`` hold this turn's own reasons; the runner
    adds ``earlier_turn_degraded`` for later turns of a degraded attempt.
    """
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
        "tags": list(item.tags),
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
        "degraded": False,
        "degraded_reasons": [],
        "calibration": calibration or {"required": [], "status": "not_required"},
        "run_signature": run_signature,
        "retry_of": retry_of,
        "cache_bypass": sorted(cache_bypass),
        "timestamp": mf.utc_now(),
    }
    if output is not None:
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
    reasons = degradation_reasons(record)
    record.update(degraded=bool(reasons), degraded_reasons=reasons)
    return record


def retrieval_discarded(explanation: dict[str, Any]) -> bool:
    """True when speculative retrieval ran but was discarded (Gate 1 said SOFT-ASK)."""
    return any(
        t.get("stage") in {"retrieval", "rerank"} and t.get("status") == "discarded"
        for t in explanation.get("timings") or []
    )


def signature_mismatches(path: Path, signature: dict[str, Any]) -> list[str]:
    """Differences between ``signature`` and the run signatures of the existing records."""
    problems: list[str] = []
    for records in latest_attempts(path).values():
        for rec in records:
            stored = rec.get("run_signature")
            if not isinstance(stored, dict):
                continue  # written before signatures existed
            for key, value in signature.items():
                if key in stored and stored[key] != value:
                    problem = f"{key}: existing {stored[key]!r} vs now {value!r}"
                    if problem not in problems:
                        problems.append(problem)
    return problems


def prepare_outputs(
    config: RunConfig,
    log: Callable[[str], None],
    signatures: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Refuse to mix runs unless appending; ``--fresh`` moves old result files aside.

    When appending (``--resume`` / ``--retry-degraded``) the existing records
    must have been produced with the same run signature (models, dataset,
    calibrated settings) — otherwise ``--fresh`` is required.
    """
    config.out.mkdir(parents=True, exist_ok=True)
    for condition in config.conditions:
        path = run_file(config.out, condition)
        try:
            backup = guard_output(
                path,
                resume=config.appends,
                fresh=config.fresh,
                tag=config.run_id,
            )
        except OutputExistsError as exc:
            raise RunnerError(str(exc)) from exc
        if backup is not None:
            log(f"[run] moved previous results to {backup}")
        if config.appends and signatures and condition in signatures:
            problems = signature_mismatches(path, signatures[condition])
            if problems:
                raise RunnerError(
                    f"{path} holds records produced under a different setup "
                    f"({'; '.join(problems)}); use --fresh to start this condition over"
                )


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}h{minutes:02d}m{secs:02d}s" if hours else f"{minutes:d}m{secs:02d}s"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@dataclass
class _Retry:
    """Previous degraded attempt of an item that is being re-run."""

    attempt_id: str
    reasons: list[str]
    bypass: frozenset[str]


class EvaluationRunner:
    """Runs (condition × item) through an executor and appends per-turn records."""

    def __init__(
        self,
        config: RunConfig,
        executor: TurnExecutor,
        log: Callable[[str], None] = _log,
        *,
        calibration: CalibrationPlan | None = None,
        signatures: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.executor = executor
        self.log = log
        self.calibration = calibration
        self.signatures = signatures or {}
        self.aborted: str | None = None
        self._consecutive_degraded = 0

    def items(self) -> list[HalluItem]:
        """Dataset items selected by split and ``--limit``."""
        bench = load_dataset(self.config.dataset)
        return select_items(filter_items(bench.items, self.config.split), self.config.limit)

    async def run(self) -> dict[str, dict[str, int]]:
        """Prepare the output files and execute every selected (condition, item) pair."""
        prepare_outputs(self.config, self.log, self.signatures)
        return await self.execute(self.items())

    async def execute(self, items: list[HalluItem]) -> dict[str, dict[str, int]]:
        """Execute ``items`` under every configured condition; returns per-condition counts."""
        counts: dict[str, dict[str, int]] = {}
        for condition in self.config.conditions:
            if self.aborted:
                break
            counts[condition] = await self._run_condition(condition, items)
        return counts

    def _plan(
        self, condition: str, items: list[HalluItem]
    ) -> tuple[list[HalluItem], dict[str, _Retry]]:
        """Items to run for ``condition`` and the degraded attempts they replace."""
        path = run_file(self.config.out, condition)
        if not self.config.appends:
            return list(items), {}
        previous = latest_attempts(path)
        retries: dict[str, _Retry] = {}
        for item_id, records in previous.items():
            reasons = attempt_reasons(records)
            if reasons:
                retries[item_id] = _Retry(
                    attempt_id=str(records[0].get("attempt_id", "")),
                    reasons=reasons,
                    bypass=bypass_calls(reasons),
                )
        if self.config.resume:
            todo = [it for it in items if it.id not in previous or it.id in retries]
        else:  # --retry-degraded alone: only the degraded items
            todo = [it for it in items if it.id in retries]
        return todo, retries

    async def _run_condition(self, condition: str, items: list[HalluItem]) -> dict[str, int]:
        path = run_file(self.config.out, condition)
        ablation = AblationConfig.preset(condition)
        todo, retries = self._plan(condition, items)
        stats = {
            "items": len(items),
            "skipped": len(items) - len(todo),
            "run": 0,
            "errors": 0,
            "degraded": 0,
            "retried": sum(1 for it in todo if it.id in retries),
            "not_started": 0,
        }
        self.log(
            f"[{condition}] {len(todo)} items to run ({stats['skipped']} skipped, "
            f"{stats['retried']} re-runs of degraded attempts)"
        )
        semaphore = asyncio.Semaphore(self.config.workers)
        started = time.perf_counter()

        async def worker(item: HalluItem) -> None:
            async with semaphore:
                if self.aborted:
                    stats["not_started"] += 1
                    return
                outcome = await self._run_item(
                    condition, ablation, item, path, retries.get(item.id)
                )
            stats["run"] += 1
            stats["errors"] += int(outcome == "error")
            stats["degraded"] += int(outcome != "ok")
            self._circuit_breaker(outcome)
            elapsed = time.perf_counter() - started
            remaining = len(todo) - stats["run"] - stats["not_started"]
            eta = elapsed / stats["run"] * remaining
            flag = "" if outcome == "ok" else f" {outcome.upper()}"
            self.log(
                f"[{condition}] {stats['skipped'] + stats['run']}/{len(items)} items "
                f"({item.id}{flag}) | errors {stats['errors']} | degraded {stats['degraded']} | "
                f"elapsed {_fmt_duration(elapsed)} | ETA {_fmt_duration(eta)}"
            )

        await asyncio.gather(*(worker(item) for item in todo))
        return stats

    def _circuit_breaker(self, outcome: str) -> None:
        self._consecutive_degraded = 0 if outcome == "ok" else self._consecutive_degraded + 1
        limit = self.config.max_consecutive_degraded
        if limit and self._consecutive_degraded >= limit and not self.aborted:
            self.aborted = (
                f"circuit breaker: {self._consecutive_degraded} consecutive degraded items "
                "(check Ollama / Qdrant / database); re-run with --resume or --retry-degraded"
            )
            self.log(f"[run] ABORT {self.aborted}")

    async def _run_item(
        self,
        condition: str,
        ablation: AblationConfig,
        item: HalluItem,
        path: Path,
        retry: _Retry | None = None,
    ) -> str:
        """Run all turns of one item in a fresh session. Never raises.

        Returns ``"ok"``, ``"degraded"`` or ``"error"``. A re-run of a degraded
        attempt bypasses the cached responses of the calls that failed.
        """
        bypass = retry.bypass if retry is not None else frozenset()
        scope: contextlib.AbstractContextManager[Any] = contextlib.nullcontext()
        if bypass:
            from evaluation.cache import bypass_cache_reads

            scope = bypass_cache_reads(bypass)
        with scope:
            return await self._run_turns(condition, ablation, item, path, retry, bypass)

    async def _run_turns(
        self,
        condition: str,
        ablation: AblationConfig,
        item: HalluItem,
        path: Path,
        retry: _Retry | None,
        bypass: frozenset[str],
    ) -> str:
        record = functools.partial(
            make_record,
            run_id=self.config.run_id,
            attempt_id=uuid.uuid4().hex,
            condition=condition,
            item=item,
            calibration=(
                self.calibration.for_condition(condition) if self.calibration is not None else None
            ),
            run_signature=self.signatures.get(condition),
            retry_of=retry.attempt_id if retry is not None else None,
            cache_bypass=sorted(bypass),
        )
        try:
            session_id = await self.executor.new_session()
        except Exception as exc:  # infrastructure failure (DB down…): record and move on
            message = f"session creation failed: {type(exc).__name__}: {exc}"
            for index in range(len(item.turns)):
                append_jsonl(path, record(turn_index=index, error=message))
            return "error"
        earlier_degraded = False
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
                return "error"
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
            rec = record(
                turn_index=index,
                output=output,
                evidence_texts=evidence,
                evidence_error=evidence_error,
                session_id=session_id,
            )
            if earlier_degraded and "earlier_turn_degraded" not in rec["degraded_reasons"]:
                rec["degraded_reasons"].append("earlier_turn_degraded")
                rec["degraded"] = True
            earlier_degraded = earlier_degraded or rec["degraded"]
            append_jsonl(path, rec)
        return "degraded" if earlier_degraded else "ok"


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

    async def preflight(self, conditions: Sequence[str]) -> list[str]:
        """Problems that would make every turn degraded (empty list = ready).

        Ollama reachable with the configured models, the Qdrant collection
        reachable and non-empty (conditions with retrieval), and the grounding
        verifier's NLI model loadable (conditions with Gate 3).
        """
        from medbridge.config import get_settings

        settings = get_settings()
        problems: list[str] = []
        models = sorted({settings.gates_model, settings.generator_model})
        info = mf.ollama_models(settings.OLLAMA_BASE_URL, models)
        if "error" in info:
            problems.append(f"Ollama unreachable at {settings.OLLAMA_BASE_URL}: {info['error']}")
        else:
            present = {name.removesuffix(":latest") for name in info} | set(info)
            missing = [m for m in models if m not in present]
            if missing:
                problems.append(f"Ollama models not pulled: {', '.join(missing)}")
        presets = [AblationConfig.preset(c) for c in conditions]
        if any(p.use_retrieval for p in presets):
            from medbridge.retrieval.hybrid_retriever import get_hybrid_retriever

            points = await get_hybrid_retriever().collection_points()
            if not points:
                problems.append(
                    f"Qdrant collection {settings.QDRANT_COLLECTION!r} unreachable or empty "
                    f"(points: {points})"
                )
        if any(p.grounding_verifier for p in presets):
            from medbridge.ai.grounding_verifier import get_grounding_verifier

            try:
                ready = await get_grounding_verifier().warmup()
            except Exception as exc:
                problems.append(f"grounding verifier failed to load: {type(exc).__name__}: {exc}")
            else:
                if not ready:
                    problems.append("grounding verifier (Gate 3) NLI model unavailable")
        return problems

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


def apply_env_overrides(
    model: str | None, database_url: str | None, ollama_url: str | None = None
) -> None:
    """Override settings for this process only (before any pipeline object is built)."""
    values: dict[str, str | None] = {"DATABASE_URL": database_url, "OLLAMA_BASE_URL": ollama_url}
    if model:
        values.update(LLM_MODEL=model, LLM_MODEL_GATES="", LLM_MODEL_GENERATOR="")
    mf.apply_settings_env(values)


def role_models() -> dict[str, str]:
    """Application models per role from the current settings."""
    from medbridge.config import get_settings

    settings = get_settings()
    return {"gates": settings.gates_model, "generator": settings.generator_model}


def run_signatures(
    config: RunConfig, plan: CalibrationPlan, dataset_sha256: str
) -> dict[str, dict[str, Any]]:
    """What must not change between attempts appended to the same condition file."""
    models = role_models()
    out: dict[str, dict[str, Any]] = {}
    for condition in config.conditions:
        calibration = plan.for_condition(condition)
        out[condition] = {
            "models": models,
            "dataset_sha256": dataset_sha256,
            "calibration_values": calibration.get("values"),
        }
    return out


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
        "mode": {
            "resume": config.resume,
            "fresh": config.fresh,
            "retry_degraded": config.retry_degraded,
            "max_consecutive_degraded": config.max_consecutive_degraded,
            "preflight": config.preflight,
        },
        "model_override": config.model,
        "models": roles,
        "ollama_base_url": mf.mask_url_credentials(settings.OLLAMA_BASE_URL),
        "ollama": mf.ollama_models(settings.OLLAMA_BASE_URL, sorted(set(roles.values()))),
        "seeds": {
            "llm_seed": settings.LLM_SEED,
            "temperature_gates": settings.LLM_TEMPERATURE_GATES,
            "temperature_generator": settings.LLM_TEMPERATURE_GENERATOR,
        },
        "hardware": mf.hardware_info(),
        "packages": mf.package_versions(),
        "settings": mf.settings_snapshot(),
        "calibration": None,
        "preflight": None,
        "cache": None,
        "counts": None,
        "warnings": [],
    }


def _install_cache(path: Path | None, ollama_info: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Request cache namespaced by the digests of the application models (or disabled)."""
    if path is None:
        return None, {"status": "disabled"}
    try:
        from evaluation.cache import install_request_cache, model_digests

        digests = model_digests(ollama_info, set(role_models().values()))
        cache = install_request_cache(path, model_digests=digests)
    except LookupError as exc:
        _log(f"[run] WARNING request cache disabled: {exc}")
        return None, {"status": f"disabled: {exc}"}
    except ImportError as exc:
        _log(f"[run] WARNING request cache unavailable ({exc}); running without memoisation")
        return None, {"status": f"unavailable: {exc}"}
    return cache, {"status": "enabled", **(cache.stats() if cache else {})}


def _digest_drift(before: dict[str, Any], base_url: str) -> list[str]:
    """Models whose Ollama digest changed while the run was in progress."""
    if "error" in before:
        return []
    after = mf.ollama_models(base_url, sorted(before))
    if "error" in after:
        return []
    drift = []
    for name, info in before.items():
        new = (after.get(name) or {}).get("digest")
        if isinstance(info, dict) and new and new != info.get("digest"):
            drift.append(f"{name}: {info.get('digest')} -> {new}")
    return drift


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
        plan = plan_calibration(
            config.conditions, config.split, config.calibration_path, config.allow_uncalibrated
        )
        apply_calibration(plan)
        effective_calibrated_settings(plan)
    except RunnerError as exc:
        _log(f"[run] {exc}")
        return 2
    if plan.status == "uncalibrated_allowed":
        _log(f"[run] WARNING running uncalibrated variants (debug only): missing {plan.missing}")
    dataset_sha = hashlib.sha256(config.dataset.read_bytes()).hexdigest()
    signatures = run_signatures(config, plan, dataset_sha)
    try:
        prepare_outputs(config, _log, signatures)
    except RunnerError as exc:
        _log(f"[run] {exc}")
        return 2
    manifest = build_manifest(config, items)
    manifest["calibration"] = {
        **plan.as_dict(),
        "per_condition": {c: plan.for_condition(c) for c in config.conditions},
        "allow_uncalibrated": config.allow_uncalibrated,
    }
    manifest_path = config.out / f"manifest_{config.run_id}.json"
    cache, manifest["cache"] = _install_cache(config.cache_path, manifest["ollama"])
    write_json(manifest_path, manifest)
    executor = executor_factory()
    status = "interrupted"
    counts: dict[str, dict[str, int]] = {}
    runner = EvaluationRunner(config, executor, calibration=plan, signatures=signatures)
    try:
        preflight = getattr(executor, "preflight", None)
        problems: list[str] = list(await preflight(config.conditions)) if preflight else []
        manifest["preflight"] = {"ran": preflight is not None, "problems": problems}
        if problems and config.preflight:
            for problem in problems:
                _log(f"[run] pre-flight: {problem}")
            status = "preflight_failed"
        else:
            counts = await runner.execute(items)
            status = "aborted" if runner.aborted else "finished"
    finally:
        await executor.close()
        if cache is not None:
            manifest["cache"] = {"status": "enabled", **cache.stats()}
            cache.close()
        drift = _digest_drift(manifest["ollama"], _ollama_url())
        if drift:
            manifest["warnings"].append(f"model digest changed during the run: {drift}")
            _log(f"[run] WARNING model digest changed during the run: {drift}")
        if runner.aborted:
            manifest["warnings"].append(runner.aborted)
        manifest.update(status=status, finished_at=mf.utc_now(), counts=counts)
        write_json(manifest_path, manifest)
    total_errors = sum(c.get("errors", 0) for c in counts.values())
    total_degraded = sum(c.get("degraded", 0) for c in counts.values())
    _log(
        f"[run] {status}; manifest {manifest_path}; items with errors: {total_errors}; "
        f"degraded items: {total_degraded}"
    )
    return 0 if status == "finished" else 2


def _ollama_url() -> str:
    from medbridge.config import get_settings

    return get_settings().OLLAMA_BASE_URL


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="run missing items and re-run errored or degraded ones",
    )
    parser.add_argument(
        "--retry-degraded",
        action="store_true",
        help="re-run only items whose latest attempt is degraded (cached failed calls bypassed)",
    )
    parser.add_argument(
        "--fresh", action="store_true", help="move existing results aside and restart"
    )
    parser.add_argument("--workers", type=int, default=1, help="concurrent items (GPU is shared)")
    parser.add_argument("--model", default=None, help="override LLM_MODEL for every role")
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama base URL for this run (sets OLLAMA_BASE_URL), e.g. a dedicated server",
    )
    parser.add_argument(
        "--database-url",
        default=os.environ.get("EVAL_DATABASE_URL"),
        help="database for evaluation sessions (default: DATABASE_URL)",
    )
    parser.add_argument("--out", type=Path, default=RUNS_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH, help="request cache file")
    parser.add_argument("--no-cache", action="store_true", help="disable request memoisation")
    parser.add_argument(
        "--calibration", type=Path, default=CALIBRATION_PATH, help="dev calibration file"
    )
    parser.add_argument(
        "--allow-uncalibrated",
        action="store_true",
        help="DEBUG ONLY: run V1/V2/V12 on test without calibration.json (marked in the outputs)",
    )
    parser.add_argument(
        "--max-consecutive-degraded",
        type=int,
        default=10,
        help="abort after this many degraded items in a row (0 = never)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="start even if the pre-flight check finds problems",
    )
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
            retry_degraded=args.retry_degraded,
            workers=args.workers,
            model=args.model,
            out=args.out,
            cache_path=None if args.no_cache else args.cache,
            calibration_path=args.calibration,
            allow_uncalibrated=args.allow_uncalibrated,
            max_consecutive_degraded=args.max_consecutive_degraded,
            preflight=not args.skip_preflight,
            run_id=args.run_id,
        )
    except RunnerError as exc:
        _log(f"[run] {exc}")
        return 2
    apply_env_overrides(args.model, args.database_url, args.ollama_url)
    return asyncio.run(run_cli(config))


if __name__ == "__main__":
    raise SystemExit(main())
