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
from medbridge.core.templates import GENERALIZE_FALLBACK_TEMPLATE
from tests.evaluation.synthetic import make_dataset


class FakeExecutor:
    """Deterministic stand-in for the pipeline; optionally fails or degrades chosen turns.

    ``degrade`` maps a user text to a kind of fallback the orchestrator would take
    (``gate2_fallback``, ``generator_fallback``, ``retrieval_outage``,
    ``persistence``, ``gate3_unavailable``, ``llm_failure``).
    """

    def __init__(self, fail_texts=(), fail_sessions=False, discard=False, degrade=None, llm=None):
        self.fail_texts = set(fail_texts)
        self.fail_sessions = fail_sessions
        self.discard = discard
        self.degrade = dict(degrade or {})
        self.llm = llm
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
        if self.llm is not None:  # exercises the request cache like the real pipeline
            await self.llm.call("context_extractor")
            await self.llm.call(call_name="response_generator")
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
        response_text = f"Answer to {text} [1]."
        kind = self.degrade.get(text)
        if kind == "gate2_fallback":
            explanation.gate2 = GateDecision(action="GENERALIZE", source="fallback")
        elif kind == "generator_fallback":
            response_text = GENERALIZE_FALLBACK_TEMPLATE
            explanation.timings.append(StageTiming(stage="generation", ms=1.0, status="failed"))
        elif kind == "retrieval_outage":
            explanation.retrieval = []
        elif kind == "persistence":
            explanation.timings.append(StageTiming(stage="persistence", ms=1.0, status="failed"))
        elif kind == "gate3_unavailable":
            explanation.gate3 = {"status": "unavailable", "available": False}
        elif kind == "llm_failure":
            explanation.llm_calls.append(
                {"call_name": "context_gate", "success": False, "attempts": 3}
            )
        response = MessageResponse(
            session_id=session_id,
            response_text=response_text,
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
    assert counts["C1"] == {
        "items": 12,
        "skipped": 0,
        "run": 12,
        "errors": 0,
        "degraded": 0,
        "retried": 0,
        "not_started": 0,
    }
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
        "degraded",
        "degraded_reasons",
        "tags",
        "calibration",
        "timestamp",
    ):
        assert key in rec
    assert rec["degraded"] is False and rec["degraded_reasons"] == []
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
    assert counts["C5"]["errors"] == 1 and counts["C5"]["degraded"] == 1
    lines = list(iter_jsonl(run_file(tmp_path / "runs", "C5")))
    failed = [r for r in lines if r["item_id"] == two_turn.id]
    assert [r["error"] for r in failed] == ["RuntimeError: commit failed", "skipped: turn 0 failed"]
    assert all(r["degraded"] and r["degraded_reasons"] == ["error"] for r in failed)
    assert rn.completed_items(run_file(tmp_path / "runs", "C5")) == {i.id for i in bench.items} - {
        two_turn.id
    }

    healthy = FakeExecutor()
    counts = await rn.EvaluationRunner(
        _config(tmp_path, dataset, conditions=["C5"], resume=True), healthy, log=lambda m: None
    ).run()
    assert counts["C5"]["skipped"] == 11 and counts["C5"]["run"] == 1
    assert counts["C5"]["errors"] == 0 and counts["C5"]["retried"] == 1
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
    assert rn.parse_conditions("all,c1g")[-1] == "C1G"  # optional reference preset
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


def _fake_ollama(base_url, names=None, timeout=5.0):
    """Every requested model is pulled, with a digest derived from its tag."""
    return {name: {"digest": f"sha256:{name}"} for name in (names or ["m"])}


async def test_run_cli_writes_manifest(tmp_path, dataset, monkeypatch):
    monkeypatch.setattr(rn.mf, "ollama_models", _fake_ollama)
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
    assert manifest["cache"]["key_includes_digest"] is True
    models = sorted(set(rn.role_models().values()))
    assert manifest["cache"]["model_digests"] == {m: f"sha256:{m}" for m in models}
    assert manifest["cache"]["path"] != str(tmp_path / "cache.sqlite")  # namespaced by digest
    assert manifest["ollama"] == {m: {"digest": f"sha256:{m}"} for m in models}
    assert manifest["calibration"]["status"] == "not_required"
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


# ----------------------------------------------------------------------
# Degraded turns (finding ablation-study-validity-1 / -2, runner side)
# ----------------------------------------------------------------------


def _bench(dataset):
    return HalluBench.model_validate_json(dataset.read_text())


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("gate2_fallback", "gate2_fallback"),
        ("generator_fallback", "generator_fallback_template"),
        ("retrieval_outage", "retrieval_empty"),
        ("persistence", "stage_failed:persistence"),
        ("gate3_unavailable", "gate3_unavailable"),
        ("llm_failure", "llm_call_failed:context_gate"),
    ],
)
async def test_degraded_turns_are_flagged_and_resumed(tmp_path, dataset, kind, reason):
    bench = _bench(dataset)
    single = next(i for i in bench.items if len(i.turns) == 1)
    executor = FakeExecutor(degrade={single.turns[0].text: kind})
    counts = await rn.EvaluationRunner(
        _config(tmp_path, dataset, conditions=["C5"]), executor, log=lambda m: None
    ).run()
    assert counts["C5"]["degraded"] == 1 and counts["C5"]["errors"] == 0
    path = run_file(tmp_path / "runs", "C5")
    rec = next(r for r in iter_jsonl(path) if r["item_id"] == single.id)
    assert rec["degraded"] is True and reason in rec["degraded_reasons"] and rec["error"] is None
    # a degraded attempt is not complete: --resume re-runs exactly that item
    assert single.id not in rn.completed_items(path)
    assert set(rn.degraded_items(path)) == {single.id}
    healthy = FakeExecutor()
    counts = await rn.EvaluationRunner(
        _config(tmp_path, dataset, conditions=["C5"], resume=True), healthy, log=lambda m: None
    ).run()
    assert counts["C5"]["run"] == 1 and counts["C5"]["retried"] == 1
    assert [t for _, t in healthy.turns] == [single.turns[0].text]
    latest = next(r for r in load_run_records(tmp_path / "runs")["C5"] if r["item_id"] == single.id)
    assert latest["degraded"] is False and latest["retry_of"] == rec["attempt_id"]


async def test_degraded_first_turn_taints_the_rest_of_the_item(tmp_path, dataset):
    bench = _bench(dataset)
    two_turn = next(i for i in bench.items if len(i.turns) == 2)
    executor = FakeExecutor(degrade={two_turn.turns[0].text: "persistence"})
    await rn.EvaluationRunner(
        _config(tmp_path, dataset, conditions=["C5"]), executor, log=lambda m: None
    ).run()
    recs = [r for r in iter_jsonl(run_file(tmp_path / "runs", "C5")) if r["item_id"] == two_turn.id]
    assert recs[0]["degraded_reasons"] == ["stage_failed:persistence"]
    assert recs[1]["degraded_reasons"] == ["earlier_turn_degraded"]


async def test_legacy_records_without_flags_are_classified_on_resume(tmp_path, dataset):
    bench = _bench(dataset)
    item = next(i for i in bench.items if len(i.turns) == 1)
    await rn.EvaluationRunner(
        _config(tmp_path, dataset, conditions=["C5"], limit=1),
        FakeExecutor(degrade={item.turns[0].text: "gate2_fallback"}),
        log=lambda m: None,
    ).run()
    path = run_file(tmp_path / "runs", "C5")
    legacy = []
    for rec in iter_jsonl(path):
        rec.pop("degraded"), rec.pop("degraded_reasons")  # written before the flags existed
        legacy.append(json.dumps(rec))
    path.write_text("\n".join(legacy) + "\n")
    assert item.id not in rn.completed_items(path)


class _CacheLLM:
    """Pipeline LLM stand-in that reads/writes the installed request cache."""

    def __init__(self, cache):
        self.cache = cache
        self.computed: list[str] = []

    async def call(self, call_name, *args, **kwargs):
        if self.cache.get(f"k-{call_name}") is None:
            self.computed.append(call_name)
            self.cache.set(f"k-{call_name}", '{"ok": true}')


async def test_retry_degraded_reruns_only_degraded_items_bypassing_failed_calls(tmp_path, dataset):
    from evaluation.cache import DiskRequestCache, instrument_llm

    bench = _bench(dataset)
    single = [i for i in bench.items if len(i.turns) == 1]
    bad = single[0]
    cache = DiskRequestCache(tmp_path / "cache.sqlite")
    llm = _CacheLLM(cache)
    instrument_llm(llm)
    first = FakeExecutor(degrade={bad.turns[0].text: "generator_fallback"}, llm=llm)
    await rn.EvaluationRunner(
        _config(tmp_path, dataset, conditions=["C5"], limit=3), first, log=lambda m: None
    ).run()
    llm.computed.clear()
    retry = FakeExecutor(llm=llm)
    counts = await rn.EvaluationRunner(
        _config(tmp_path, dataset, conditions=["C5"], retry_degraded=True),
        retry,
        log=lambda m: None,
    ).run()
    # only the degraded item is re-run (not the 9 items never started, not the healthy ones)
    assert [t for _, t in retry.turns] == [bad.turns[0].text]
    assert counts["C5"]["run"] == 1 and counts["C5"]["retried"] == 1
    # the failed generator call is recomputed, the extractor call is still memoised
    assert llm.computed == ["response_generator"]
    latest = {r["item_id"]: r for r in load_run_records(tmp_path / "runs")["C5"]}
    assert latest[bad.id]["cache_bypass"] == ["response_generator"]
    assert latest[bad.id]["degraded"] is False and len(latest) == 3
    assert all(r["retry_of"] is None for item_id, r in latest.items() if item_id != bad.id)
    cache.close()


async def test_circuit_breaker_stops_an_outage(tmp_path, dataset):
    bench = _bench(dataset)
    executor = FakeExecutor(
        degrade={t.text: "retrieval_outage" for i in bench.items for t in i.turns}
    )
    runner = rn.EvaluationRunner(
        _config(tmp_path, dataset, conditions=["C1", "C5"], max_consecutive_degraded=3),
        executor,
        log=lambda m: None,
    )
    counts = await runner.run()
    assert runner.aborted and "circuit breaker" in runner.aborted
    assert counts["C1"]["run"] == 3 and counts["C1"]["not_started"] == 9
    assert "C5" not in counts  # later conditions are not started


async def test_run_cli_aborts_on_failed_preflight(tmp_path, dataset, monkeypatch):
    monkeypatch.setattr(rn.mf, "ollama_models", _fake_ollama)

    class Unready(FakeExecutor):
        async def preflight(self, conditions):
            return ["Qdrant collection 'x' unreachable or empty (points: None)"]

    executor = Unready()
    config = _config(tmp_path, dataset, conditions=["C5"], limit=2)
    assert await rn.run_cli(config, executor_factory=lambda: executor) == 2
    manifest = json.loads((tmp_path / "runs" / f"manifest_{config.run_id}.json").read_text())
    assert manifest["status"] == "preflight_failed" and manifest["preflight"]["problems"]
    assert executor.turns == [] and executor.closed


# ----------------------------------------------------------------------
# Calibration gate (findings ablation-study-validity-3 / statistics-judges-3)
# ----------------------------------------------------------------------

CALIBRATED_KEYS = [k for keys in rn.CALIBRATED_SETTINGS.values() for k in keys]


@pytest.fixture()
def restore_settings(monkeypatch):
    """Let the runner inject calibrated settings; restore env + settings afterwards."""
    import os

    from medbridge.config import get_settings

    for key in CALIBRATED_KEYS:
        if key in os.environ:
            monkeypatch.setenv(key, os.environ[key])
        else:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(rn.mf, "ollama_models", _fake_ollama)
    yield
    monkeypatch.undo()
    get_settings.cache_clear()


def _write_calibration(path, v1=True, v2=True):
    data = {"split": "dev"}
    if v1:
        data["v1"] = {
            "thresholds": {
                "EVIDENCE_MIN_RERANK_ANSWER": 1.25,
                "EVIDENCE_MIN_RERANK_ANY": -3.5,
                "EVIDENCE_MIN_ENTITY_COVERAGE": 0.3,
            }
        }
    if v2:
        data["v2"] = {
            "thresholds": {
                "GROUNDING_ENTAILMENT_THRESHOLD": 0.35,
                "GROUNDING_MAX_UNSUPPORTED_FRACTION": 0.2,
            }
        }
    path.write_text(json.dumps(data))
    return path


async def test_variants_on_test_split_refuse_without_calibration(
    tmp_path, dataset, restore_settings
):
    config = _config(
        tmp_path,
        dataset,
        conditions=["C5", "V1", "V2"],
        split="test",
        calibration_path=tmp_path / "calibration.json",
    )
    executor = FakeExecutor()
    assert await rn.run_cli(config, executor_factory=lambda: executor) == 2
    assert executor.turns == [] and not (tmp_path / "runs" / "V1.jsonl").exists()
    # a calibration file with only the V1 block still refuses V2
    _write_calibration(tmp_path / "calibration.json", v2=False)
    assert await rn.run_cli(config, executor_factory=lambda: executor) == 2


async def test_calibrated_values_are_injected_and_recorded(tmp_path, dataset, restore_settings):
    from medbridge.config import get_settings

    cal = _write_calibration(tmp_path / "calibration.json")
    config = _config(
        tmp_path,
        dataset,
        conditions=["C5", "V1", "V12"],
        split="test",
        limit=3,
        calibration_path=cal,
    )
    assert await rn.run_cli(config, executor_factory=FakeExecutor) == 0
    settings = get_settings()
    assert settings.EVIDENCE_MIN_RERANK_ANSWER == 1.25
    assert settings.GROUNDING_MAX_UNSUPPORTED_FRACTION == 0.2
    manifest = json.loads((tmp_path / "runs" / f"manifest_{config.run_id}.json").read_text())
    sha = rn.file_sha256(cal)
    assert manifest["calibration"]["status"] == "calibrated"
    assert manifest["calibration"]["sha256"] == sha
    assert manifest["calibration"]["effective"]["EVIDENCE_MIN_RERANK_ANY"] == -3.5
    runs = load_run_records(tmp_path / "runs")
    for rec in runs["V1"]:
        assert rec["calibration"]["status"] == "calibrated" and rec["calibration"]["sha256"] == sha
        assert rec["calibration"]["values"] == {
            "EVIDENCE_MIN_RERANK_ANSWER": 1.25,
            "EVIDENCE_MIN_RERANK_ANY": -3.5,
            "EVIDENCE_MIN_ENTITY_COVERAGE": 0.3,
        }
    assert set(runs["V12"][0]["calibration"]["values"]) == set(CALIBRATED_KEYS)
    assert runs["C5"][0]["calibration"]["status"] == "not_required"


async def test_uncalibrated_override_and_dev_split_are_marked(tmp_path, dataset, restore_settings):
    config = _config(
        tmp_path,
        dataset,
        conditions=["V1"],
        split="test",
        limit=2,
        allow_uncalibrated=True,
        calibration_path=tmp_path / "missing.json",
    )
    assert await rn.run_cli(config, executor_factory=FakeExecutor) == 0
    manifest = json.loads((tmp_path / "runs" / f"manifest_{config.run_id}.json").read_text())
    assert manifest["calibration"]["status"] == "uncalibrated_allowed"
    assert manifest["calibration"]["allow_uncalibrated"] is True
    rec = load_run_records(tmp_path / "runs")["V1"][0]
    assert rec["calibration"]["status"] == "uncalibrated_allowed"
    assert rec["calibration"]["values"]["EVIDENCE_MIN_RERANK_ANY"] is not None  # defaults used
    dev = _config(
        tmp_path / "dev",
        dataset,
        conditions=["V2"],
        split="dev",
        limit=1,
        calibration_path=tmp_path / "missing.json",
    )
    assert await rn.run_cli(dev, executor_factory=FakeExecutor) == 0
    rec = load_run_records(tmp_path / "dev" / "runs")["V2"][0]
    assert rec["calibration"]["status"] == "uncalibrated_dev"


async def test_resume_refuses_records_from_other_thresholds(tmp_path, dataset, restore_settings):
    cal = _write_calibration(tmp_path / "calibration.json")
    config = _config(
        tmp_path, dataset, conditions=["V1"], split="test", limit=2, calibration_path=cal
    )
    assert await rn.run_cli(config, executor_factory=FakeExecutor) == 0
    data = json.loads(cal.read_text())
    data["v1"]["thresholds"]["EVIDENCE_MIN_RERANK_ANSWER"] = 2.0  # re-calibrated since
    cal.write_text(json.dumps(data))
    resume = _config(
        tmp_path, dataset, conditions=["V1"], split="test", calibration_path=cal, resume=True
    )
    executor = FakeExecutor()
    assert await rn.run_cli(resume, executor_factory=lambda: executor) == 2
    assert executor.turns == []
    fresh = _config(
        tmp_path,
        dataset,
        conditions=["V1"],
        split="test",
        limit=2,
        calibration_path=cal,
        fresh=True,
    )
    assert await rn.run_cli(fresh, executor_factory=FakeExecutor) == 0


def test_ollama_url_override(monkeypatch):
    import os

    from medbridge.config import get_settings

    monkeypatch.setenv("OLLAMA_BASE_URL", os.environ.get("OLLAMA_BASE_URL", "x"))
    try:
        rn.apply_env_overrides(None, None, "http://127.0.0.1:11555")
        assert get_settings().OLLAMA_BASE_URL == "http://127.0.0.1:11555"
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def test_retry_degraded_and_fresh_are_exclusive(tmp_path, dataset):
    with pytest.raises(rn.RunnerError):
        _config(tmp_path, dataset, retry_degraded=True, fresh=True)
