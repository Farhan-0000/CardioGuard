"""
Unit tests for the MedBridge-AQ benchmark harness (hermetic by default).

Covers the dataset (size, numbering, corpus-valid citation labels, consistency
with the deterministic emergency rule), citation matching, snapshot → event
seeding, the mock routing simulator, the live path through a fake executor,
checkpoint/resume, metrics, the report (computed statements, mock banner) and
the CLI. One ``postgres`` test seeds a real session in the test database.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from medbridge.api.schemas.enums import ActionEnum
from medbridge.api.schemas.responses import (
    CitationResponse,
    Explanation,
    GateDecision,
    MessageResponse,
)
from medbridge.core.emergency_classifier import classify as classify_emergency
from medbridge.state.projector import StateProjector
from tests.benchmark.benchmark_runner import run_benchmark_cli
from tests.benchmark.citations import (
    citation_precision,
    corpus_documents_for,
    label_matches,
    load_manifest_documents,
    manifest_citation,
)
from tests.benchmark.evaluator import (
    SUBSET_FILES,
    VIGNETTES_DIR,
    BenchmarkEvaluator,
    calculate_metrics,
    load_checkpoint,
    load_vignettes,
    select_round_robin,
)
from tests.benchmark.live import (
    LiveConfigError,
    LiveTurn,
    mask_url,
    resolve_eval_database_url,
    snapshot_to_events,
)
from tests.benchmark.report_generator import MOCK_BANNER, ReportGenerator
from tests.benchmark.schemas import CitationRecord, ClinicalVignette, VignetteEvaluationResult

# Highest id of each subset before the natural-phrasing extension.
ORIGINAL_MAX_ID = {"ANS": 46, "CONTRA": 30, "EMERG": 38, "SOFT": 42, "GEN": 35, "ABST": 34}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def test_dataset_size_subsets_and_unique_ids() -> None:
    vignettes = load_vignettes(None)
    assert len(vignettes) >= 270
    assert {v.subset for v in vignettes} == set(SUBSET_FILES)
    ids = [v.id for v in vignettes]
    assert len(ids) == len(set(ids))
    for v in vignettes:
        assert v.query.strip()
        assert isinstance(v.expected_action, ActionEnum)


def test_new_vignettes_continue_numbering_and_cover_every_subset() -> None:
    new_per_prefix: dict[str, int] = {}
    for subset, filename in SUBSET_FILES.items():
        items = json.loads((VIGNETTES_DIR / filename).read_text())
        prefix = items[0]["id"].split("-")[0]
        numbers = [int(i["id"].split("-")[1]) for i in items]
        assert numbers == list(range(1, len(items) + 1)), f"{filename} ids not contiguous"
        assert all(i["subset"] == subset for i in items)
        new_per_prefix[prefix] = sum(1 for n in numbers if n > ORIGINAL_MAX_ID[prefix])
    assert sum(new_per_prefix.values()) >= 45
    assert all(count > 0 for count in new_per_prefix.values())


def test_sample_subset_has_two_per_subset() -> None:
    sample = load_vignettes("sample")
    assert len(sample) == 12
    assert len({v.subset for v in sample}) == 6


def test_unknown_subset_is_rejected() -> None:
    with pytest.raises(ValueError):
        load_vignettes("no_such_subset")


def test_every_expected_citation_label_names_a_corpus_document() -> None:
    unmatched = {
        (v.id, label)
        for v in load_vignettes(None)
        for label in v.expected_citations
        if not corpus_documents_for(label)
    }
    assert unmatched == set()


def test_vignettes_agree_with_the_deterministic_emergency_rule() -> None:
    """Any query the fast-path escalates must be expected ESCALATE (cf. CONTRA-025)."""
    conflicts = [
        v.id
        for v in load_vignettes(None)
        if classify_emergency(v.query) == ActionEnum.ESCALATE
        and v.expected_action != ActionEnum.ESCALATE
    ]
    assert conflicts == []
    contra_025 = next(v for v in load_vignettes(None) if v.id == "CONTRA-025")
    assert contra_025.expected_action == ActionEnum.ESCALATE


def test_soft_ask_and_loop_breaker_counters_are_consistent() -> None:
    for v in load_vignettes(None):
        if v.soft_ask_count >= 2:
            assert v.expected_action == ActionEnum.GENERALIZE, v.id
        if v.subset == "incomplete_context_soft_ask":
            assert v.soft_ask_count < 2, v.id


def test_round_robin_selection() -> None:
    vignettes = load_vignettes(None)
    picked = select_round_robin(vignettes, 7)
    assert len(picked) == 7
    assert [v.subset for v in picked[:6]] == list(SUBSET_FILES)
    assert select_round_robin(vignettes, None) == vignettes


# ---------------------------------------------------------------------------
# Citation matching
# ---------------------------------------------------------------------------


def _doc(gid: str) -> CitationRecord:
    return next(manifest_citation(d) for d in load_manifest_documents() if d["guideline_id"] == gid)


def test_label_matching_rules() -> None:
    esc_2024, esc_2019 = _doc("ESC_HTN_2024"), _doc("ESC_HTN_EMERGENCIES_2019")
    assert label_matches("ESC 2024", esc_2024)
    assert not label_matches("ESC 2024", esc_2019)
    assert not label_matches("ESC/ESH 2024", esc_2024)  # not a corpus name
    assert label_matches("NICE_NG133", _doc("NICE_NG133"))
    assert not label_matches("NICE_NG133", _doc("NICE_NG238"))
    assert label_matches("MedlinePlus", _doc("MEDLINEPLUS_HIGH_BP"))
    assert label_matches("KDIGO 2021", _doc("KDIGO_BP_CKD_2021"))
    assert not label_matches("KDIGO 2024", _doc("KDIGO_BP_CKD_2021"))
    assert label_matches("AHA/ACC 2025", CitationRecord(source="AHA/ACC 2025"))  # source fallback
    assert not label_matches("", esc_2024)


def test_citation_precision_only_when_scorable() -> None:
    aha = _doc("AHA_ACC_HTN_2025")
    nice = _doc("NICE_NG136")
    assert citation_precision(["AHA/ACC 2025"], []) == (None, 0)
    assert citation_precision([], [aha]) == (None, 0)
    assert citation_precision(["AHA/ACC 2025"], [aha, nice]) == (0.5, 1)


def test_citation_record_from_citation_response() -> None:
    cit = CitationResponse(
        marker="[1]",
        chunk_id="c1",
        source="ESC 2024",
        section="s",
        excerpt="e",
        guideline_id="ESC_HTN_2024",
        title="2024 ESC Guidelines",
        org="ESC",
        year=2024,
    )
    record = CitationRecord.from_response(cit)
    assert record.guideline_id == "ESC_HTN_2024" and record.year == 2024
    assert label_matches("ESC 2024", record)


# ---------------------------------------------------------------------------
# Snapshot seeding
# ---------------------------------------------------------------------------


def test_snapshot_to_events_maps_every_field() -> None:
    snapshot = {
        "demographics": {"age": 28, "sex": "female", "pregnancy_weeks": 14},
        "recent_bp_readings": [{"sbp": 146, "dbp": 92}, {"sbp": 150, "dbp": 95, "setting": "home"}],
        "current_medications": [{"drug": "Norvasc", "dosage": "5mg"}, {"drug": "Lisinopril"}],
        "symptoms": ["dry cough"],
        "lab_results": [{"lab_type": "potassium", "value": "4.6 mEq/L"}],
    }
    events = snapshot_to_events(snapshot)
    by_type: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        by_type.setdefault(str(e.event_type), []).append(e.payload)
    assert by_type["DEMOGRAPHIC"][0]["pregnancy_status"] == "pregnant"
    assert by_type["BP_READING"] == [
        {"systolic": 146, "diastolic": 92},
        {"systolic": 150, "diastolic": 95, "setting": "home"},
    ]
    norvasc = by_type["MEDICATION_ADDED"][0]
    assert norvasc["drug_name"] == "amlodipine" and norvasc["brand_name"] == "Norvasc"
    assert norvasc["dosage"] == "5mg"
    assert by_type["MEDICATION_ADDED"][1]["drug_name"] == "lisinopril"
    assert by_type["SYMPTOM_REPORTED"] == [{"symptom": "dry cough"}]
    assert by_type["LAB_RESULT"] == [{"lab_type": "potassium", "value": 4.6, "unit": "mEq/L"}]

    projected = StateProjector().apply_events({}, events)
    assert projected["demographics"]["age"] == 28
    assert len(projected["recent_bp_readings"]) == 2
    assert [m["drug"] for m in projected["current_medications"]] == ["amlodipine", "lisinopril"]


def test_snapshot_to_events_empty_snapshot() -> None:
    assert snapshot_to_events({}) == []


def test_every_vignette_snapshot_converts() -> None:
    for v in load_vignettes(None):
        events = snapshot_to_events(v.patient_snapshot)
        StateProjector().apply_events({}, events)  # must not raise


# ---------------------------------------------------------------------------
# Live configuration helpers
# ---------------------------------------------------------------------------


def test_resolve_eval_database_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "DATABASE_URL=postgresql+asyncpg://u:p@h:5433/medbridge\n"
        "EVAL_DATABASE_URL=postgresql+asyncpg://u:p@h:5433/medbridge_eval\n"
    )
    monkeypatch.delenv("EVAL_DATABASE_URL", raising=False)
    monkeypatch.delenv("MEDBRIDGE_DEV_DATABASE_URL", raising=False)
    assert resolve_eval_database_url(None, env).endswith("/medbridge_eval")
    assert resolve_eval_database_url("postgresql+asyncpg://u:p@h/other", env).endswith("/other")
    with pytest.raises(LiveConfigError):
        resolve_eval_database_url("postgresql+asyncpg://u:p@h:5433/medbridge", env)
    with pytest.raises(LiveConfigError):
        resolve_eval_database_url(None, tmp_path / "missing.env")
    assert (
        mask_url("postgresql+asyncpg://u:secret@h:5433/db")
        == "postgresql+asyncpg://u:***@h:5433/db"
    )


# ---------------------------------------------------------------------------
# Evaluator — mock simulator
# ---------------------------------------------------------------------------


async def test_mock_simulator_routes_by_pipeline_rules() -> None:
    evaluator = BenchmarkEvaluator(mode="mock")
    cases = [
        (
            "emergency_escalate",
            "My BP is 220/130 with crushing chest pain.",
            0,
            ActionEnum.ESCALATE,
        ),
        ("out_of_domain_abstain", "Best cream for eczema?", 0, ActionEnum.ABSTAIN),
        (
            "incomplete_context_soft_ask",
            "My BP was 150/90. What should I do?",
            0,
            ActionEnum.SOFT_ASK,
        ),
        ("lifestyle_and_generalize", "Just give me general advice.", 2, ActionEnum.GENERALIZE),
        ("complete_context_answer", "52yo diabetic on lisinopril, target?", 0, ActionEnum.ANSWER),
    ]
    for subset, query, count, action in cases:
        v = ClinicalVignette(
            id=f"T-{subset}",
            subset=subset,
            query=query,
            soft_ask_count=count,
            expected_action=action,
            expected_citations=["AHA/ACC 2025"],
        )
        result = await evaluator.evaluate_vignette(v)
        assert result.actual_action == action and result.action_match, subset
    answer = await evaluator.evaluate_vignette(
        ClinicalVignette(
            id="T-ANS",
            subset="complete_context_answer",
            query="target?",
            expected_action=ActionEnum.ANSWER,
            expected_citations=["ESC 2024", "NICE_NG133"],
        )
    )
    assert {c.guideline_id for c in answer.actual_citations} == {"ESC_HTN_2024", "NICE_NG133"}
    assert answer.citation_precision == 1.0
    loop = await evaluator.evaluate_vignette(
        ClinicalVignette(
            id="T-LOOP",
            subset="lifestyle_and_generalize",
            query="no idea",
            soft_ask_count=2,
            expected_action=ActionEnum.GENERALIZE,
        )
    )
    assert loop.decision_source == "loop_breaker"


async def test_mock_benchmark_full_dataset_passes() -> None:
    evaluator = BenchmarkEvaluator(mode="mock")
    results, metrics = await evaluator.run_benchmark(load_vignettes(None))
    assert metrics.total_vignettes == len(results) >= 270
    assert metrics.overall_routing_accuracy == 100.0
    assert metrics.emergency_detection_recall == 100.0
    assert metrics.loop_breaker_count == metrics.loop_breaker_forced_count > 0
    assert metrics.passed_acceptance_criteria


# ---------------------------------------------------------------------------
# Evaluator — live path through a fake executor (hermetic)
# ---------------------------------------------------------------------------


class FakeExecutor:
    """Stands in for LivePipelineExecutor: returns canned MessageResponses."""

    def __init__(self, action: ActionEnum = ActionEnum.ANSWER, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any], int]] = []
        self.action = action
        self.fail_on = fail_on
        self.closed = False

    async def run(self, message: str, snapshot: dict[str, Any], soft_ask_count: int) -> LiveTurn:
        self.calls.append((message, snapshot, soft_ask_count))
        if self.fail_on and self.fail_on in message:
            raise RuntimeError("boom")
        sid = uuid.uuid4()
        explanation = Explanation(
            gate1=GateDecision(action="PROCEED", source="llm"),
            gate2=GateDecision(action=self.action.value, source="llm"),
            total_ms=1234.5,
        )
        response = MessageResponse(
            session_id=sid,
            response_text="answer [1]",
            action=self.action,
            citations=[
                CitationResponse(
                    marker="[1]",
                    chunk_id="c1",
                    source="AHA/ACC 2025",
                    section="8",
                    excerpt="x",
                    guideline_id="AHA_ACC_HTN_2025",
                    org="AHA/ACC",
                    year=2025,
                    title="t",
                )
            ],
            timestamp=datetime.now(UTC),
            explanation=explanation,
        )
        return LiveTurn(session_id=sid, response=response, latency_ms=42.0)

    async def close(self) -> None:
        self.closed = True


async def test_live_path_scores_response_and_records_explanation() -> None:
    executor = FakeExecutor()
    evaluator = BenchmarkEvaluator(mode="live", executor=executor)
    vignette = load_vignettes("complete_context_answer")[0]
    result = await evaluator.evaluate_vignette(vignette)
    assert executor.calls == [(vignette.query, vignette.patient_snapshot, vignette.soft_ask_count)]
    assert result.mode == "live" and result.action_match
    assert result.actual_citations[0].guideline_id == "AHA_ACC_HTN_2025"
    assert result.citation_precision == 1.0
    assert result.explanation is not None and result.explanation.server_total_ms == 1234.5
    assert result.decision_source == "llm" and result.session_id


async def test_live_path_records_errors_without_raising() -> None:
    evaluator = BenchmarkEvaluator(mode="live", executor=FakeExecutor(fail_on="diabetes"))
    vignette = load_vignettes("complete_context_answer")[0]  # mentions diabetes
    result = await evaluator.evaluate_vignette(vignette)
    assert result.error and "boom" in result.error
    assert not result.action_match


async def test_cli_live_checkpoint_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tests.benchmark.benchmark_runner as runner

    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path / "runs")
    out = tmp_path / "live.md"
    first = FakeExecutor(action=ActionEnum.ESCALATE)
    code = await run_benchmark_cli(
        subset="emergency_escalate",
        mode="live",
        output_path=out,
        quiet=True,
        limit=3,
        executor=first,
        run_name="t",
    )
    assert code == 0 and len(first.calls) == 3 and first.closed
    ckpt = tmp_path / "runs" / "t.jsonl"
    assert len(load_checkpoint(ckpt)) == 3

    second = FakeExecutor(action=ActionEnum.ESCALATE)
    await run_benchmark_cli(
        subset="emergency_escalate",
        mode="live",
        output_path=out,
        quiet=True,
        limit=5,
        executor=second,
        run_name="t",
        resume=True,
    )
    assert len(second.calls) == 2  # 3 resumed from the checkpoint
    data = json.loads(out.with_suffix(".json").read_text())
    assert data["mode"] == "live" and len(data["results"]) == 5
    assert "MOCK MODE" not in out.read_text()


# ---------------------------------------------------------------------------
# Metrics and report
# ---------------------------------------------------------------------------


def _row(**kw: Any) -> VignetteEvaluationResult:
    base: dict[str, Any] = {
        "vignette_id": "V",
        "subset": "complete_context_answer",
        "query": "q",
        "expected_action": ActionEnum.ANSWER,
        "actual_action": ActionEnum.ANSWER,
        "action_match": True,
    }
    base.update(kw)
    return VignetteEvaluationResult(**base)


def test_metric_calculations_and_acceptance_logic() -> None:
    rows = [
        _row(
            vignette_id="V1",
            subset="emergency_escalate",
            expected_action=ActionEnum.ESCALATE,
            actual_action=ActionEnum.ESCALATE,
            is_emergency=True,
            emergency_detected=True,
        ),
        _row(vignette_id="V2", actual_citations=["AHA/ACC 2025"], citation_precision=1.0),
        _row(
            vignette_id="V3",
            subset="out_of_domain_abstain",
            expected_action=ActionEnum.ABSTAIN,
            actual_action=ActionEnum.ABSTAIN,
            is_out_of_domain=True,
        ),
        _row(
            vignette_id="V4",
            subset="lifestyle_and_generalize",
            soft_ask_count=2,
            expected_action=ActionEnum.GENERALIZE,
            actual_action=ActionEnum.GENERALIZE,
        ),
    ]
    metrics = calculate_metrics(rows, mode="mock")
    assert metrics.total_vignettes == 4 and metrics.overall_routing_accuracy == 100.0
    assert metrics.overall_citation_precision == 100.0 and metrics.citation_evaluated == 1
    assert metrics.emergency_detection_recall == 100.0 and metrics.hallucination_rate == 0.0
    assert metrics.loop_breaker_count == 1 and metrics.loop_breaker_generalize_count == 1
    assert metrics.passed_acceptance_criteria

    worse = calculate_metrics(
        rows
        + [
            _row(
                vignette_id="V5",
                subset="out_of_domain_abstain",
                expected_action=ActionEnum.ABSTAIN,
                actual_action=ActionEnum.ANSWER,
                action_match=False,
                is_out_of_domain=True,
                hallucinated=True,
            )
        ],
        mode="live",
    )
    assert worse.out_of_domain_answer_count == 1 and worse.hallucination_rate == 50.0
    assert not worse.passed_acceptance_criteria
    assert worse.confusion_matrix["ABSTAIN"]["ANSWER"] == 1


def test_metrics_without_citations_report_not_applicable() -> None:
    metrics = calculate_metrics([_row(expected_citations=["AHA/ACC 2025"])], mode="live")
    assert metrics.overall_citation_precision is None
    assert not metrics.acceptance["citation_precision"]  # applicable but nothing cited
    only_escalate = calculate_metrics(
        [
            _row(
                expected_action=ActionEnum.ESCALATE,
                actual_action=ActionEnum.ESCALATE,
                is_emergency=True,
                emergency_detected=True,
            )
        ],
        mode="live",
    )
    assert set(only_escalate.acceptance) == {"routing_accuracy", "emergency_recall"}
    assert calculate_metrics([], mode="live").total_vignettes == 0


def test_report_statements_are_computed_from_data(tmp_path: Path) -> None:
    rows = [
        _row(vignette_id="V1"),
        _row(
            vignette_id="V2",
            expected_action=ActionEnum.SOFT_ASK,
            actual_action=ActionEnum.GENERALIZE,
            action_match=False,
        ),
    ]
    mock_report = ReportGenerator.generate_markdown_report(calculate_metrics(rows, "mock"), rows)
    assert MOCK_BANNER in mock_report
    assert "no vignette in this run starts at the threshold" in mock_report
    assert "| V2 |" in mock_report  # misrouted list
    live_report = ReportGenerator.generate_markdown_report(calculate_metrics(rows, "live"), rows)
    assert "MOCK MODE" not in live_report
    assert "1/2 expected actions matched" in live_report
    # No numeric claim may appear that is not derived from the metrics object.
    assert not re.search(r"successfully forced", live_report)
    path = ReportGenerator.save_report(live_report, tmp_path / "r.md")
    assert path.read_text() == live_report


async def test_cli_mock_sample_writes_markdown_and_json(tmp_path: Path) -> None:
    out = tmp_path / "cli_report.md"
    code = await run_benchmark_cli(subset="sample", mode="mock", output_path=out, quiet=True)
    assert code == 0 and out.exists()
    data = json.loads(out.with_suffix(".json").read_text())
    assert data["mode"] == "mock" and "not evidence" in data["note"]
    assert len(data["results"]) == 12


async def test_cli_rejects_unknown_subset(tmp_path: Path) -> None:
    code = await run_benchmark_cli(
        subset="nope", mode="mock", output_path=tmp_path / "x.md", quiet=True
    )
    assert code == 2


# ---------------------------------------------------------------------------
# Live seeding against the (isolated) test database
# ---------------------------------------------------------------------------


@pytest.mark.postgres
async def test_live_executor_seeds_snapshot_and_counter_in_new_session() -> None:
    from medbridge.db.connection import get_sessionmaker
    from medbridge.state import turn_store
    from tests.benchmark.live import LivePipelineExecutor

    vignette = next(v for v in load_vignettes(None) if v.id == "CONTRA-001")
    session_id, seeded = await LivePipelineExecutor.new_session(vignette.patient_snapshot, 2)
    assert seeded == len(snapshot_to_events(vignette.patient_snapshot)) > 0
    async with get_sessionmaker()() as db:
        state = await turn_store.load_session_state(db, session_id)
        events = await turn_store.list_events(db, session_id)
    assert state.soft_ask_count == 2
    assert state.snapshot["demographics"]["pregnancy_status"] == "pregnant"
    assert state.snapshot["current_medications"][0]["drug"] == "lisinopril"
    assert {e.event_type for e in events} >= {"DEMOGRAPHIC", "BP_READING", "MEDICATION_ADDED"}
