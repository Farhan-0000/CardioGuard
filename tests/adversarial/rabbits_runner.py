"""
RABBITS — brand ↔ generic drug-name robustness benchmark.

The dataset is generated from ``medbridge.nlp.drugs.all_brand_generic_pairs()``
(every cardiovascular brand the normaliser knows, US/EU and Indian). Each
brand → generic pair is asked with two natural templates; the brand and the
generic message differ **only** in the drug name:

* ``personal`` — "I take {drug} at my usual dose every morning. Is it okay to
  have it with grapefruit juice?" (no dose numbers, so no dose is wrong for any drug)
* ``general``  — "What is {drug} used for?"

Combination generics are written "x and y" (e.g. "telmisartan and amlodipine").

Metrics per brand/generic item:

* **action consistency** — same routing action for brand and generic;
* **extracted-generic consistency** (personal template) — the medicine stored
  in the session snapshot has the same generic ingredient set for both
  variants (i.e. the brand was normalised to its generic);
* **citation-set Jaccard** — overlap of cited chunk ids (and guideline ids);
* **response length ratio** — characters of the brand answer / generic answer.

Modes:

* ``mock`` (hermetic) — the real orchestrator, redactor, emergency classifier,
  Context Extractor post-processing, drug normaliser, query expansion and
  citation validation, with stubbed LLM calls (the extractor stub echoes the
  drug name as written; Gate 1, retrieval, Gate 2 and the generator follow
  fixed rules, see ``tests/adversarial/stubs.py``) and an in-memory store.
  It tests the deterministic normalisation path, **not model behaviour**.
* ``live`` — the full production pipeline with the real LLM, one new session
  of the evaluation database per message; ``--limit`` / ``--resume`` supported.
  Generic messages shared by several brands are run once and reused.

Usage::

    python -m tests.adversarial.rabbits_runner --mode mock
    python -m tests.adversarial.rabbits_runner --mode live --limit 20 --resume
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import statistics
import sys
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medbridge.api.schemas.enums import ActionEnum
from medbridge.nlp.drugs import INDIAN_BRANDS, all_brand_generic_pairs

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "rabbits_benchmark_report.md"
RUNS_DIR = REPO_ROOT / "reports" / "benchmark_runs"

TEMPLATES: dict[str, str] = {
    "personal": (
        "I take {drug} at my usual dose every morning. Is it okay to have it with grapefruit juice?"
    ),
    "general": "What is {drug} used for?",
}
MOCK_NOTE = (
    "MOCK MODE — LLM calls stubbed (the extractor echoes the drug name as written; Gate 1, "
    "retrieval, Gate 2 and the generator follow fixed rules). Real components: redaction, "
    "emergency classifier, extractor post-processing, drug normaliser, query expansion, citation "
    "validation. This checks the deterministic normalisation path and is not evidence of model "
    "behaviour."
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def generic_surface(generic: str) -> str:
    """How a patient writes the generic: combinations as "x and y"."""
    return generic.replace(" + ", " and ")


@dataclass(frozen=True)
class RabbitsItem:
    """One brand/generic comparison (a pair asked with one template)."""

    item_id: str
    brand: str
    generic: str
    drug_class: str
    indian_brand: bool
    template: str
    brand_message: str
    generic_message: str

    @property
    def ingredients(self) -> frozenset[str]:
        return frozenset(p.strip() for p in self.generic.split("+"))


def build_dataset(templates: dict[str, str] | None = None) -> list[RabbitsItem]:
    """All brand/generic pairs × templates (sorted by brand, then template)."""
    items: list[RabbitsItem] = []
    for info in all_brand_generic_pairs():
        brand = info.brand or ""
        for name, template in (templates or TEMPLATES).items():
            items.append(
                RabbitsItem(
                    item_id=f"{_slug(brand)}::{name}",
                    brand=brand,
                    generic=info.generic,
                    drug_class=info.drug_class,
                    indian_brand=brand in INDIAN_BRANDS,
                    template=name,
                    brand_message=template.format(drug=brand),
                    generic_message=template.format(drug=generic_surface(info.generic)),
                )
            )
    return items


def select_items(
    items: list[RabbitsItem], limit: int | None, brands: Iterable[str] | None = None
) -> list[RabbitsItem]:
    """Filter by brand names (case-insensitive) and keep the first ``limit`` items."""
    wanted = {b.strip().lower() for b in brands or [] if b.strip()}
    chosen = [i for i in items if not wanted or i.brand.lower() in wanted]
    return chosen if limit is None else chosen[:limit]


# ---------------------------------------------------------------------------
# Turn results and item scoring
# ---------------------------------------------------------------------------


@dataclass
class TurnResult:
    """One message processed by the pipeline."""

    key: str
    message: str
    action: str
    response_text: str = ""
    citation_chunks: list[str] = field(default_factory=list)
    citation_guidelines: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)  # stored snapshot drug names
    latency_ms: float = 0.0
    gate1: str | None = None
    gate2_source: str | None = None
    session_id: str | None = None
    error: str | None = None


def message_key(message: str) -> str:
    return hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]


def ingredient_set(medications: Iterable[str]) -> frozenset[str]:
    """Generic ingredients named by stored medication names (``"a + b"`` → {a, b})."""
    out: set[str] = set()
    for name in medications:
        out.update(p.strip().lower() for p in str(name).split("+") if p.strip())
    return frozenset(out)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class ItemResult:
    item: RabbitsItem
    brand: TurnResult
    generic: TurnResult

    @property
    def action_match(self) -> bool:
        return self.brand.action == self.generic.action and not (
            self.brand.error or self.generic.error
        )

    @property
    def extraction_applicable(self) -> bool:
        return self.item.template == "personal"

    @property
    def extraction_match(self) -> bool | None:
        """Same non-empty ingredient set stored for both variants (personal template only)."""
        if not self.extraction_applicable:
            return None
        brand, generic = (
            ingredient_set(self.brand.medications),
            ingredient_set(self.generic.medications),
        )
        return bool(brand) and brand == generic

    @property
    def brand_normalised(self) -> bool | None:
        """The brand variant stored exactly the pair's generic ingredients."""
        if not self.extraction_applicable:
            return None
        return ingredient_set(self.brand.medications) == self.item.ingredients

    @property
    def citation_jaccard(self) -> float:
        return jaccard(self.brand.citation_chunks, self.generic.citation_chunks)

    @property
    def guideline_jaccard(self) -> float:
        return jaccard(self.brand.citation_guidelines, self.generic.citation_guidelines)

    @property
    def length_ratio(self) -> float | None:
        if not self.generic.response_text:
            return None
        return len(self.brand.response_text) / len(self.generic.response_text)


@dataclass
class RabbitsSummary:
    mode: str
    timestamp: str
    pairs: int
    indian_pairs: int
    items: int
    action_consistency_pct: float
    extraction_items: int
    extraction_consistency_pct: float | None
    brand_normalised_pct: float | None
    citation_jaccard_mean: float
    guideline_jaccard_mean: float
    length_ratio_mean: float | None
    length_ratio_median: float | None
    length_ratio_within_20pct: float | None
    turns_run: int
    errors: int
    latency_p50_ms: float | None
    by_template: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_origin: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: list[ItemResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def action_flips(self) -> list[ItemResult]:
        return [r for r in self.results if not r.action_match]

    @property
    def extraction_flips(self) -> list[ItemResult]:
        return [r for r in self.results if r.extraction_match is False]


def _pct(n: int, d: int) -> float | None:
    return round(n / d * 100.0, 2) if d else None


def summarise(
    results: list[ItemResult],
    mode: str,
    turns: list[TurnResult],
    metadata: dict[str, Any] | None = None,
) -> RabbitsSummary:
    def block(rows: list[ItemResult]) -> dict[str, Any]:
        ext = [r for r in rows if r.extraction_applicable]
        return {
            "items": len(rows),
            "action_consistency_pct": _pct(sum(r.action_match for r in rows), len(rows)),
            "extraction_consistency_pct": _pct(
                sum(bool(r.extraction_match) for r in ext), len(ext)
            ),
            "citation_jaccard_mean": round(statistics.mean(r.citation_jaccard for r in rows), 4)
            if rows
            else None,
        }

    ratios = [r.length_ratio for r in results if r.length_ratio is not None]
    ext = [r for r in results if r.extraction_applicable]
    latencies = sorted(t.latency_ms for t in turns if not t.error)
    pairs = {r.item.brand for r in results}
    return RabbitsSummary(
        mode=mode,
        timestamp=datetime.now(UTC).isoformat(),
        pairs=len(pairs),
        indian_pairs=sum(1 for b in pairs if b in INDIAN_BRANDS),
        items=len(results),
        action_consistency_pct=_pct(sum(r.action_match for r in results), len(results)) or 0.0,
        extraction_items=len(ext),
        extraction_consistency_pct=_pct(sum(bool(r.extraction_match) for r in ext), len(ext)),
        brand_normalised_pct=_pct(sum(bool(r.brand_normalised) for r in ext), len(ext)),
        citation_jaccard_mean=round(statistics.mean(r.citation_jaccard for r in results), 4)
        if results
        else 0.0,
        guideline_jaccard_mean=round(statistics.mean(r.guideline_jaccard for r in results), 4)
        if results
        else 0.0,
        length_ratio_mean=round(statistics.mean(ratios), 4) if ratios else None,
        length_ratio_median=round(statistics.median(ratios), 4) if ratios else None,
        length_ratio_within_20pct=_pct(sum(0.8 <= x <= 1.25 for x in ratios), len(ratios)),
        turns_run=len(turns),
        errors=sum(1 for t in turns if t.error),
        latency_p50_ms=round(statistics.median(latencies), 1) if latencies else None,
        by_template={
            name: block([r for r in results if r.item.template == name])
            for name in sorted({r.item.template for r in results})
        },
        by_origin={
            origin: block([r for r in results if r.item.indian_brand == (origin == "indian")])
            for origin in ("indian", "international")
            if any(r.item.indian_brand == (origin == "indian") for r in results)
        },
        results=results,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------


def _turn_from_response(
    key: str,
    message: str,
    response: Any,
    latency_ms: float,
    medications: list[str],
    session_id: Any,
) -> TurnResult:
    explanation = getattr(response, "explanation", None)
    gate1 = getattr(getattr(explanation, "gate1", None), "action", None)
    gate2 = getattr(getattr(explanation, "gate2", None), "source", None)
    return TurnResult(
        key=key,
        message=message,
        action=ActionEnum(response.action).value,
        response_text=response.response_text,
        citation_chunks=[c.chunk_id for c in response.citations],
        citation_guidelines=sorted({c.guideline_id or c.source for c in response.citations}),
        medications=medications,
        latency_ms=round(latency_ms, 1),
        gate1=gate1,
        gate2_source=gate2,
        session_id=str(session_id),
    )


def _medications(snapshot: dict[str, Any]) -> list[str]:
    return [
        str(m.get("drug") or "") for m in snapshot.get("current_medications") or [] if m.get("drug")
    ]


class MockRabbitsExecutor:
    """Hermetic executor: real deterministic pipeline parts, stubbed LLM calls."""

    def __init__(self) -> None:
        from medbridge.ai.context_extractor import ContextExtractor
        from medbridge.core.orchestrator import PipelineOrchestrator
        from tests.adversarial.stubs import (
            EchoExtractorLLM,
            InMemoryTurnStore,
            PassthroughReranker,
            StubContextGate,
            StubEvidenceGate,
            StubGenerator,
            StubRetriever,
        )

        self.store = InMemoryTurnStore()
        self.orchestrator = PipelineOrchestrator(
            context_extractor=ContextExtractor(llm=EchoExtractorLLM()),  # type: ignore[arg-type]
            context_gate=StubContextGate(),  # type: ignore[arg-type]
            hybrid_retriever=StubRetriever(),  # type: ignore[arg-type]
            reranker=PassthroughReranker(),  # type: ignore[arg-type]
            evidence_gate=StubEvidenceGate(),  # type: ignore[arg-type]
            response_generator=StubGenerator(),  # type: ignore[arg-type]
            store=self.store,
        )

    async def run(self, message: str) -> TurnResult:
        from tests.adversarial.stubs import NullDB

        session_id = uuid.uuid4()
        self.store.seed(session_id)
        started = time.perf_counter()
        response = await self.orchestrator.process_message(session_id, message, NullDB())  # type: ignore[arg-type]
        latency = (time.perf_counter() - started) * 1000.0
        return _turn_from_response(
            message_key(message),
            message,
            response,
            latency,
            _medications(self.store.snapshot(session_id)),
            session_id,
        )

    async def close(self) -> None:
        return None


class LiveRabbitsExecutor:
    """Full production pipeline, one new evaluation-DB session per message."""

    def __init__(self) -> None:
        from tests.benchmark.live import LivePipelineExecutor

        self._live = LivePipelineExecutor()

    async def run(self, message: str) -> TurnResult:
        from medbridge.db.connection import get_sessionmaker
        from medbridge.state import turn_store

        turn = await self._live.run(message)
        async with get_sessionmaker()() as db:
            state = await turn_store.load_session_state(db, turn.session_id)
        return _turn_from_response(
            message_key(message),
            message,
            turn.response,
            turn.latency_ms,
            _medications(state.snapshot),
            turn.session_id,
        )

    async def close(self) -> None:
        await self._live.close()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def load_turn_cache(path: Path) -> dict[str, TurnResult]:
    cache: dict[str, TurnResult] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            turn = TurnResult(**row)
        except (ValueError, TypeError):
            continue
        if not turn.error:
            cache[turn.key] = turn
    return cache


async def run_rabbits(
    items: list[RabbitsItem],
    executor: Any,
    *,
    mode: str = "mock",
    checkpoint: Path | None = None,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
    metadata: dict[str, Any] | None = None,
) -> RabbitsSummary:
    """Run every item (brand + generic message) and summarise. Generic turns are reused."""
    cache = load_turn_cache(checkpoint) if (checkpoint is not None and resume) else {}

    async def get(message: str) -> TurnResult:
        key = message_key(message)
        if key in cache:
            return cache[key]
        try:
            turn = await executor.run(message)
        except Exception as exc:  # keep going; recorded as an error turn
            turn = TurnResult(
                key=key,
                message=message,
                action=ActionEnum.GENERALIZE.value,
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
            )
        cache[key] = turn
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")
        return turn

    results: list[ItemResult] = []
    for index, item in enumerate(items, 1):
        brand = await get(item.brand_message)
        generic = await get(item.generic_message)
        result = ItemResult(item=item, brand=brand, generic=generic)
        results.append(result)
        if progress is not None:
            flag = "ok  " if result.action_match else "FLIP"
            progress(
                f"  [{index:>3}/{len(items)}] {flag} {item.brand:<14} {item.template:<8} "
                f"{brand.action:<10} vs {generic.action:<10}"
            )
    used = {r.brand.key for r in results} | {r.generic.key for r in results}
    return summarise(results, mode, [t for k, t in cache.items() if k in used], metadata)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _fmt(value: float | None, suffix: str = "%") -> str:
    return "n/a" if value is None else f"{value}{suffix}"


def summary_to_json(summary: RabbitsSummary) -> dict[str, Any]:
    data = {k: v for k, v in asdict(summary).items() if k != "results"}
    data["note"] = MOCK_NOTE if summary.mode == "mock" else "production pipeline (live LLM)"
    data["items_detail"] = [
        {
            "item_id": r.item.item_id,
            "brand": r.item.brand,
            "generic": r.item.generic,
            "drug_class": r.item.drug_class,
            "indian_brand": r.item.indian_brand,
            "template": r.item.template,
            "brand_message": r.item.brand_message,
            "generic_message": r.item.generic_message,
            "brand_action": r.brand.action,
            "generic_action": r.generic.action,
            "action_match": r.action_match,
            "brand_medications": r.brand.medications,
            "generic_medications": r.generic.medications,
            "extraction_match": r.extraction_match,
            "brand_normalised": r.brand_normalised,
            "citation_jaccard": round(r.citation_jaccard, 4),
            "guideline_jaccard": round(r.guideline_jaccard, 4),
            "length_ratio": round(r.length_ratio, 4) if r.length_ratio is not None else None,
            "brand_error": r.brand.error,
            "generic_error": r.generic.error,
        }
        for r in summary.results
    ]
    return data


def generate_markdown(summary: RabbitsSummary) -> str:
    s = summary
    lines = ["# RABBITS Report — Brand vs Generic Drug-Name Robustness", ""]
    if s.mode == "mock":
        lines += [f"> **{MOCK_NOTE}**", ""]
    lines += [
        f"- **Mode:** `{s.mode}`"
        + (" (stubbed LLM, hermetic)" if s.mode == "mock" else " (production pipeline, real LLM)"),
        f"- **Generated:** {s.timestamp}",
        f"- **Pairs:** {s.pairs} brand → generic pairs ({s.indian_pairs} Indian brands), "
        f"{s.items} items ({', '.join(TEMPLATES)} templates); brand and generic messages differ "
        "only in the drug name.",
        f"- **Pipeline turns:** {s.turns_run} (generic messages shared by several brands run "
        f"once); errors: {s.errors}",
    ]
    for key, value in s.metadata.items():
        lines.append(f"- **{key}:** `{value}`")
    lines += [
        "",
        "## Summary",
        "",
        "| Metric | Value | Definition |",
        "| :--- | :---: | :--- |",
        f"| Action consistency | {_fmt(s.action_consistency_pct)} | same routing action for brand "
        "and generic message |",
        f"| Extracted-generic consistency | {_fmt(s.extraction_consistency_pct)} "
        f"(n={s.extraction_items}) | personal template: same non-empty generic ingredient set "
        "stored in the snapshot |",
        f"| Brand normalised to its generic | {_fmt(s.brand_normalised_pct)} | personal template: "
        "brand variant stored exactly the pair's generic ingredients |",
        f"| Citation-set Jaccard (chunks) | {s.citation_jaccard_mean} | mean over items; 1.0 when "
        "neither variant cites |",
        f"| Citation-set Jaccard (guidelines) | {s.guideline_jaccard_mean} | "
        "same, by guideline id |",
        f"| Response length ratio | mean {_fmt(s.length_ratio_mean, '')}, median "
        f"{_fmt(s.length_ratio_median, '')} | brand chars / generic chars; "
        f"{_fmt(s.length_ratio_within_20pct)} of items within 0.8–1.25 |",
        f"| Median turn latency | {_fmt(s.latency_p50_ms, ' ms')} | "
        "client wall-clock per message |",
        "",
        "## By template",
        "",
        "| Template | Items | Action consistency | Extraction consistency | Citation Jaccard |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for name, b in s.by_template.items():
        lines.append(
            f"| {name} | {b['items']} | {_fmt(b['action_consistency_pct'])} | "
            f"{_fmt(b['extraction_consistency_pct'])} | {b['citation_jaccard_mean']} |"
        )
    lines += [
        "",
        "## By brand origin",
        "",
        "| Origin | Items | Action consistency | Extraction consistency | Citation Jaccard |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for name, b in s.by_origin.items():
        lines.append(
            f"| {name} | {b['items']} | {_fmt(b['action_consistency_pct'])} | "
            f"{_fmt(b['extraction_consistency_pct'])} | {b['citation_jaccard_mean']} |"
        )

    lines += ["", f"## Action flips ({len(s.action_flips)})", ""]
    if s.action_flips:
        lines += [
            "| Brand | Generic | Template | Brand action | Generic action | Error |",
            "| :--- | :--- | :--- | :---: | :---: | :--- |",
        ]
        for r in s.action_flips:
            lines.append(
                f"| {r.item.brand} | {r.item.generic} | {r.item.template} | "
                f"{r.brand.action} | {r.generic.action} | "
                f"{r.brand.error or r.generic.error or ''} |"
            )
    else:
        lines.append("None.")
    lines += ["", f"## Extraction flips ({len(s.extraction_flips)})", ""]
    if s.extraction_flips:
        lines += [
            "| Brand | Expected generic | Stored (brand msg) | Stored (generic msg) |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for r in s.extraction_flips:
            lines.append(
                f"| {r.item.brand} | {r.item.generic} | "
                f"{', '.join(r.brand.medications) or '—'} | "
                f"{', '.join(r.generic.medications) or '—'} |"
            )
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)


def write_reports(summary: RabbitsSummary, output: Path) -> tuple[Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generate_markdown(summary), encoding="utf-8")
    json_path = output.with_suffix(".json")
    json_path.write_text(
        json.dumps(summary_to_json(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output, json_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RABBITS brand vs generic robustness benchmark")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="first N items (each item = brand message + generic message)",
    )
    parser.add_argument("--brands", default="", help="comma-separated brand names to include")
    parser.add_argument("--resume", action="store_true", help="reuse turns from the checkpoint")
    parser.add_argument("--database-url", default=None, help="evaluation DB (live)")
    parser.add_argument("--run-name", default=None, help="checkpoint file stem")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    return parser


async def main_async(args: argparse.Namespace) -> int:
    items = select_items(build_dataset(), args.limit, args.brands.split(","))
    if not items:
        print("[ERROR] no items selected", file=sys.stderr)
        return 2
    metadata: dict[str, Any] = {"templates": json.dumps(TEMPLATES)}
    executor: Any
    checkpoint: Path | None = None
    if args.mode == "live":
        from tests.benchmark.live import (
            LiveConfigError,
            configure_eval_environment,
            resolve_eval_database_url,
            run_metadata,
        )

        try:
            url = resolve_eval_database_url(args.database_url)
        except LiveConfigError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2
        configure_eval_environment(url)
        metadata.update(run_metadata(url))
        checkpoint = RUNS_DIR / f"{args.run_name or 'rabbits_live'}.jsonl"
        if checkpoint.exists() and not args.resume:
            checkpoint.rename(
                checkpoint.with_name(
                    f"{checkpoint.stem}.{time.strftime('%Y%m%d-%H%M%S')}.bak.jsonl"
                )
            )
        metadata["checkpoint"] = str(checkpoint.relative_to(REPO_ROOT))
        executor = LiveRabbitsExecutor()
    else:
        executor = MockRabbitsExecutor()
    try:
        summary = await run_rabbits(
            items,
            executor,
            mode=args.mode,
            checkpoint=checkpoint,
            resume=args.resume,
            progress=None if args.quiet else print,
            metadata=metadata,
        )
    finally:
        await executor.close()
    md, js = write_reports(summary, args.output)
    print(
        f"RABBITS ({summary.mode}): items={summary.items} pairs={summary.pairs} "
        f"action consistency={summary.action_consistency_pct}% "
        f"extraction consistency={summary.extraction_consistency_pct}% "
        f"flips={len(summary.action_flips)} errors={summary.errors}"
    )
    print(f"Report: {md} (+ {js.name})")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(build_parser().parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
