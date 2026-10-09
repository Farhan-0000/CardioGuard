"""Runner: records, error handling, resume/fresh semantics and the CLI path — mocked orchestrator."""

import json
import sys
import types
import uuid
from datetime import UTC, datetime

import pytest

from evaluation import run as rn
from evaluation.datasets.schema import HalluBench
from evaluation.io import iter_jsonl, load_run_records, run_file
from medbridge.api.schemas.enums import ActionEnum
from medbridge.api.schemas.responses import (
    CitationResponse,
    Explanation,
    GateDecision,
    MessageResponse,
    RetrievedEvidence,
    StageTiming,
)
from tests.evaluation.synthetic import make_dataset


class FakeExecutor:
    """Deterministic stand-in for the pipeline; optionally fails for chosen items/turns."""

    def __init__(self, fail_texts=(), fail_sessions=False, discard=False):
        self.fail_texts = set(fail_texts)
        self.fail_sessions = fail_sessions
        self.discard = discard
        self.turns: list[tuple[str, str]] = []
        self.closed = False

    async def new_session(self):
        if self.fail_sessions:
            raise ConnectionError("db down")
        return uuid.uuid4()

    async def run_turn(self, session_id, text, ablation):
        if text in self.fail_texts:
            raise RuntimeError("commit failed")
        self.turns.append((ablation.name, text))
        explanation = Explanation(
            gate1=GateDecision(action="PROCEED", rationale="ok"),
            gate2=GateDecision(action="ANSWER", rationale="good evidence"),
            retrieval=[
                RetrievedEvidence(
                    rank=1,
                    chunk_id="esc_c1",
                    guideline_id="ESC_2024",
                    excerpt="short",
                    score_rerank=3.2,
                )
            ],
            timings=[
                StageTiming(
                    stage="retrieval", ms=12.0, status="discarded" if self.discard else "ok"
                )
            ],
            llm_calls=[
                {
                    "call_name": "response_generator",
                    "model": "m",
                    "latency_ms": 10.0,
                    "cached": False,
                    "success": True,
                    "attempts": 1,
                }
            ],
            models={"generator": "m"},
            ablation=ablation.name,
        )
        response = MessageResponse(
            session_id=session_id,
            response_text=f"Answer to {text} [1].",
            action=ActionEnum.ANSWER,
            citations=[
                CitationResponse(
                    marker="[1]", chunk_id="esc_c1", source="ESC", section="s", excerpt="e"
                )
            ],
            timestamp=datetime.now(UTC),
            explanation=explanation,
        )
        return rn.TurnOutput(response=response, latency_ms=123.456, llm_calls=[])

    async def evidence(self, explanation):
        return [
            {"chunk_id": r["chunk_id"], "text": "FULL chunk text", "text_source": "qdrant"}
            for r in explanation["retrieval"]
        ]

    async def close(self):
        self.closed = True


@pytest.fixture()
def dataset(tmp_path):
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(make_dataset(2)))
    return path


def _config(tmp_path, dataset, **kw):
    base = dict(dataset=dataset, conditions=["C1", "C5"], out=tmp_path / "runs", cache_path=None)
    base.update(kw)
    return rn.RunConfig(**base)


async def test_records_written_for_every_turn(tmp_path, dataset):
    executor = FakeExecutor()
    counts = await rn.EvaluationRunner(
        _config(tmp_path, dataset), executor, log=lambda m: None
    ).run()
    assert counts["C1"] == {"items": 12, "skipped": 0, "run": 12, "errors": 0}
    runs = load_run_records(tmp_path / "runs")
    assert set(runs) == {"C1", "C5"}
    assert len(runs["C5"]) == 14  # 12 items, 2 of them two-turn
    rec = runs["C5"][0]
    for key in (
        "run_id",
        "condition",
        "item_id",
        "category",
        "subcategory",
        "split",
        "turn_index",
        "user_text",
        "gold_action",
        "acceptable_actions",
        "action",
        "response_text",
        "citations",
        "explanation",
        "evidence_texts",
        "latency_ms",
        "llm_calls",
        "models",
        "error",
        "timestamp",
    ):
        assert key in rec
    assert rec["action"] == "ANSWER" and rec["latency_ms"] == 123.5
    assert rec["evidence_texts"][0]["text"] == "FULL chunk text"
    assert rec["llm_calls"][0]["call_name"] == "response_generator"  # from the explanation
    assert rec["explanation"]["ablation"] == "C5"
    assert {name for name, _ in executor.turns} == {"C1", "C5"}


async def test_failures_recorded_and_resume_retries_them(tmp_path, dataset):
    bench = HalluBench.model_validate_json(dataset.read_text())
    two_turn = next(i for i in bench.items if len(i.turns) == 2)
    failing = FakeExecutor(fail_texts={two_turn.turns[0].text})
    counts = await rn.EvaluationRunner(
        _config(tmp_path, dataset, conditions=["C5"]), failing, log=lambda m: None
    ).run()
    assert counts["C5"]["errors"] == 1
    lines = list(iter_jsonl(run_file(tmp_path / "runs", "C5")))
    failed = [r for r in lines if r["item_id"] == two_turn.id]
    assert [r["error"] for r in failed] == ["RuntimeError: commit failed", "skipped: turn 0 failed"]
    assert rn.completed_items(run_file(tmp_path / "runs", "C5")) == {i.id for i in bench.items} - {
        two_turn.id
    }

    healthy = FakeExecutor()
    counts = await rn.EvaluationRunner(
        _config(tmp_path, dataset, conditions=["C5"], resume=True), healthy, log=lambda m: None
    ).run()
    assert counts["C5"] == {"items": 12, "skipped": 11, "run": 1, "errors": 0}
    assert [t for _, t in healthy.turns] == [t.text for t in two_turn.turns]
    latest = [r for r in load_run_records(tmp_path / "runs")["C5"] if r["item_id"] == two_turn.id]
    assert [r["error"] for r in latest] == [None, None]


async def test_partial_attempt_is_ignored_and_rerun(tmp_path, dataset):
    config = _config(tmp_path, dataset, conditions=["C5"], limit=1)
    await rn.EvaluationRunner(config, FakeExecutor(), log=lambda m: None).run()
    path = run_file(tmp_path / "runs", "C5")
    bench = HalluBench.model_validate_json(dataset.read_text())
    two_turn = next(i for i in bench.items if len(i.turns) == 2)
    partial = rn.make_record(
        run_id="r", attempt_id="partial", condition="C5", item=two_turn, turn_index=0
    )
    with path.open("a") as fh:
        fh.write(json.dumps(partial) + "\n")
        fh.write('{"truncated": ')  # crash mid-write
    assert two_turn.id not in rn.completed_items(path)
    assert all(r["attempt_id"] != "partial" for r in load_run_records(tmp_path / "runs")["C5"])


async def test_session_failure_marks_all_turns(tmp_path, dataset):
    counts = await rn.EvaluationRunner(
        _config(tmp_path, dataset, conditions=["C1"], limit=2),
        FakeExecutor(fail_sessions=True),
        log=lambda m: None,
    ).run()
    assert counts["C1"]["errors"] == 2
    records = list(iter_jsonl(run_file(tmp_path / "runs", "C1")))
    assert all(r["error"].startswith("session creation failed") for r in records)


async def test_discarded_retrieval_has_no_evidence(tmp_path, dataset):
    await rn.EvaluationRunner(
        _config(tmp_path, dataset, conditions=["C3"], limit=1),
        FakeExecutor(discard=True),
        log=lambda m: None,
    ).run()
    rec = load_run_records(tmp_path / "runs")["C3"][0]
    assert rec["evidence_texts"] == []


def test_existing_results_need_resume_or_fresh(tmp_path, dataset):
    path = run_file(tmp_path / "runs", "C5")
    path.parent.mkdir(parents=True)
    path.write_text('{"item_id": "x"}\n')
    with pytest.raises(rn.RunnerError):
        rn.prepare_outputs(_config(tmp_path, dataset, conditions=["C5"]), lambda m: None)
    rn.prepare_outputs(_config(tmp_path, dataset, conditions=["C5"], resume=True), lambda m: None)
    assert path.exists()
    config = _config(tmp_path, dataset, conditions=["C5"], fresh=True)
    rn.prepare_outputs(config, lambda m: None)
    assert not path.exists()
    assert (tmp_path / "runs" / f"C5.jsonl.{config.run_id}.bak").exists()
    with pytest.raises(rn.RunnerError):
        _config(tmp_path, dataset, resume=True, fresh=True)


def test_parse_conditions():
    assert rn.parse_conditions("all") == list(rn.DEFAULT_CONDITIONS)
    assert rn.parse_conditions("c0, C5,v12") == ["C0", "C5", "V12"]
    assert rn.parse_conditions("all,V12")[-1] == "V12"
    with pytest.raises(rn.RunnerError):
        rn.parse_conditions("C9")
    with pytest.raises(rn.RunnerError):
        rn.parse_conditions(" , ")


def test_select_items_round_robin():
    bench = HalluBench.model_validate(make_dataset(3))
    chosen = rn.select_items(bench.items, 7)
    categories = [i.category for i in chosen]
    assert len(chosen) == 7 and len(set(categories)) == 6
    assert rn.select_items(bench.items, None) == bench.items
    order = [i.id for i in bench.items]
    assert [i.id for i in chosen] == sorted((i.id for i in chosen), key=order.index)


async def test_run_cli_writes_manifest(tmp_path, dataset, monkeypatch):
    monkeypatch.setattr(rn.mf, "ollama_models", lambda *a, **k: {"m": {"digest": "abc"}})
    installed = []
    fake = types.ModuleType("medbridge.ai.llm_wrapper")
    fake.set_request_cache = installed.append
    monkeypatch.setitem(sys.modules, "medbridge.ai.llm_wrapper", fake)
    executor = FakeExecutor()
    config = _config(
        tmp_path, dataset, conditions=["C5"], limit=3, cache_path=tmp_path / "cache.sqlite"
    )
    assert await rn.run_cli(config, executor_factory=lambda: executor) == 0
    manifest = json.loads((tmp_path / "runs" / f"manifest_{config.run_id}.json").read_text())
    assert manifest["status"] == "finished"
    assert manifest["counts"]["C5"]["run"] == 3
    assert manifest["cache"]["status"] == "enabled"
    assert manifest["ollama"] == {"m": {"digest": "abc"}}
    assert len(manifest["dataset"]["sha256"]) == 64
    assert "ENCRYPTION_MASTER_KEY" not in manifest["settings"]
    assert (
        "***" in manifest["settings"]["DATABASE_URL"]
        or "@" not in manifest["settings"]["DATABASE_URL"]
    )
    assert executor.closed and installed and installed[0] is not None


async def test_run_cli_refuses_existing_results(tmp_path, dataset, monkeypatch):
    path = run_file(tmp_path / "runs", "C5")
    path.parent.mkdir(parents=True)
    path.write_text('{"item_id": "x"}\n')
    config = _config(tmp_path, dataset, conditions=["C5"])
    assert await rn.run_cli(config, executor_factory=FakeExecutor) == 2


def test_model_override_env(monkeypatch):
    import os

    from medbridge.config import get_settings

    for key in ("LLM_MODEL", "LLM_MODEL_GATES", "LLM_MODEL_GENERATOR", "DATABASE_URL"):
        monkeypatch.setenv(key, os.environ.get(key, "x"))  # monkeypatch restores the originals
    try:
        rn.apply_env_overrides(
            "gemma3:4b", "postgresql+asyncpg://u:p@127.0.0.1:5433/medbridge_eval"
        )
        settings = get_settings()
        assert settings.LLM_MODEL == "gemma3:4b" and settings.gates_model == "gemma3:4b"
        assert settings.generator_model == "gemma3:4b"
        assert settings.DATABASE_URL.endswith("/medbridge_eval")
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
