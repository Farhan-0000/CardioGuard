"""Dev-split calibration: NLI thresholds and V1 evidence-gate thresholds."""

import json
import sys

import pytest

from evaluation import calibrate as cb
from tests.evaluation.synthetic import write_study


@pytest.fixture(autouse=True)
def no_production_calibration(monkeypatch):
    """Force the documented local V1 rule/features regardless of the AI layer's progress."""
    monkeypatch.setitem(sys.modules, "medbridge.ai.evidence_calibration", None)


def test_calibrate_nli_recovers_separating_threshold():
    pairs = (
        [("SUPPORTED", 0.8, 0.0)] * 30
        + [("UNSUPPORTED", 0.3, 0.1)] * 30
        + [("CONTRADICTED", 0.1, 0.9)] * 10
    )
    res = cb.calibrate_nli(pairs)
    assert 0.3 < res["theta"] <= 0.8
    assert res["kappa_binary"] == pytest.approx(1.0)
    assert 0.1 < res["theta_c"] <= 0.9 and res["kappa_3class"] == pytest.approx(1.0)
    assert cb.calibrate_nli([]) is None


def _trace(category, gold, gate2, rerank, coverage=1.0, final=None):
    return cb.GateTrace(
        item_id=f"{category}-{rerank}-{gate2}",
        turn_index=0,
        category=category,
        gold_action=gold,
        acceptable=(gold,),
        final_action=final or gate2,
        gate2_action=gate2,
        gate2_ran=True,
        max_rerank=rerank,
        entity_coverage=coverage,
    )


def test_threshold_rule():
    thr = cb.Thresholds(min_rerank_answer=2.0, min_rerank_any=-3.0, min_entity_coverage=0.5)
    assert cb.threshold_rule("ESCALATE", -10, 0, thr) == "ESCALATE"
    assert cb.threshold_rule("ANSWER", -5, 1, thr) == "ABSTAIN"
    assert cb.threshold_rule("ANSWER", 1.0, 1, thr) == "GENERALIZE"
    assert cb.threshold_rule("ANSWER", 3.0, 0.2, thr) == "GENERALIZE"
    assert cb.threshold_rule("ANSWER", 3.0, 0.9, thr) == "ANSWER"
    assert cb.threshold_rule("GENERALIZE", 0.0, 0.0, thr) == "GENERALIZE"
    assert cb.Thresholds(1.5, -100.0, 0.0).env_lines()[1] == "EVIDENCE_MIN_RERANK_ANY=-100"


def test_simulated_action_keeps_downstream_downgrade():
    thr = cb.Thresholds(-100.0, -100.0, 0.0)
    t = _trace("answerable_complete", "ANSWER", "ANSWER", 3.0, final="GENERALIZE")
    assert cb.simulated_action(t, thr) == "GENERALIZE"
    t2 = cb.GateTrace(
        "x", 0, "emergency", "ESCALATE", ("ESCALATE",), "ESCALATE", None, False, -100.0, 0.0
    )
    assert cb.simulated_action(t2, cb.Thresholds(50, 50, 1.0)) == "ESCALATE"


def test_calibrate_v1_removes_false_answers_without_losing_coverage():
    traces = (
        [_trace("answerable_complete", "ANSWER", "ANSWER", r) for r in (2.0, 3.0, 4.0, 5.0, 6.0)]
        + [_trace("unanswerable_in_domain", "ABSTAIN", "ANSWER", r) for r in (-8.0, -7.0, -6.5)]
        + [_trace("unanswerable_in_domain", "ABSTAIN", "ABSTAIN", -9.0)]
    )
    res = cb.calibrate_v1(traces)
    assert res["c5_dev_baseline"]["false_answer_rate"] == pytest.approx(0.75)
    assert res["dev_result"]["false_answer_rate"] == 0.0
    assert res["dev_result"]["coverage"] == 1.0
    any_thr = res["thresholds"]["EVIDENCE_MIN_RERANK_ANY"]
    assert -6.5 < any_thr <= 2.0
    assert res["env_lines"][1].startswith("EVIDENCE_MIN_RERANK_ANY=")
    assert cb.calibrate_v1([]) is None


def test_local_features():
    feats = cb.local_features(
        "Is amlodipine safe with grapefruit juice?",
        ["Amlodipine is a calcium channel blocker."],
        [1.5, -2.0],
    )
    assert feats["max_rerank"] == 1.5 and feats["mean_rerank"] == pytest.approx(-0.25)
    assert feats["entity_coverage"] == pytest.approx(1 / 4)  # amlodipine, safe, grapefruit, juice
    assert cb.local_features("", [], [])["max_rerank"] == cb.DISABLED


class FakeVerifier:
    """Production-verifier stand-in: entailment 0.9 for 'Claim one' sentences, 0.1 otherwise."""

    model_name = "fake-nli-minilm"

    def __init__(self, available=True):
        self.available = available
        self.calls = []

    async def verify(self, response_text, chunks):
        import re
        from types import SimpleNamespace

        self.calls.append((response_text, [c["chunk_id"] for c in chunks]))
        sentences = []
        for text in re.split(r"(?<=[.!?])\s+", response_text.strip()):
            claim = len(text.split()) >= 3
            entail = (0.9 if "one" in text else 0.1) if claim else 0.0
            sentences.append({"text": text, "is_claim": claim, "entailment": entail})
        return SimpleNamespace(sentences=sentences, available=self.available)


def test_calibrate_end_to_end(tmp_path, capsys):
    paths = write_study(tmp_path, conditions=("C1", "C5"))
    out = tmp_path / "calibration.json"
    assert (
        cb.main(
            [
                "--runs-dir",
                str(paths["runs"]),
                "--judgments-dir",
                str(paths["judgments"]),
                "--output",
                str(out),
                "--part",
                "nli,v1",
            ]
        )
        == 0
    )
    data = json.loads(out.read_text())
    assert data["split"] == "dev"
    assert data["nli"]["n_claims"] > 0 and 0 < data["nli"]["theta"] < 1
    assert data["v1"]["n_gate2_decisions"] > 0
    assert set(data["v1"]["thresholds"]) == {
        "EVIDENCE_MIN_RERANK_ANSWER",
        "EVIDENCE_MIN_RERANK_ANY",
        "EVIDENCE_MIN_ENTITY_COVERAGE",
    }
    printed = capsys.readouterr().out
    assert "EVIDENCE_MIN_RERANK_ANY=" in printed
    assert "c5_dev_degraded_excluded" in data["v1"]["inputs"]
    # only the selected part is recomputed; the other is kept
    cb.calibrate(paths["runs"], paths["judgments"], out, part="nli")
    assert json.loads(out.read_text())["v1"] == data["v1"]
    # V2 with the (fake) production verifier
    verifier = FakeVerifier()
    result = cb.calibrate(paths["runs"], paths["judgments"], out, part="v2", verifier=verifier)
    v2 = result["v2"]
    assert set(v2["thresholds"]) == {
        "GROUNDING_ENTAILMENT_THRESHOLD",
        "GROUNDING_MAX_UNSUPPORTED_FRACTION",
    }
    assert 0.1 < v2["thresholds"]["GROUNDING_ENTAILMENT_THRESHOLD"] <= 0.9
    assert v2["verifier_model"] == "fake-nli-minilm" and v2["n_sentences"] > 0
    assert v2["dev_result"]["coverage"] >= 0.85 * v2["c5_dev_baseline"]["coverage"] - 1e-9
    assert verifier.calls and json.loads(out.read_text())["v1"] == data["v1"]
    assert "GROUNDING_MAX_UNSUPPORTED_FRACTION=" in v2["env_lines"][1]


def test_parts_without_data_keep_previous_blocks(tmp_path, capsys):
    """statistics-judges-7: a part with nothing to calibrate never overwrites with null."""
    out = tmp_path / "calibration.json"
    previous = {"nli": {"theta": 0.35, "theta_c": 0.6}, "v1": {"thresholds": {"x": 1}}}
    out.write_text(json.dumps(previous))
    empty = tmp_path / "empty"
    result = cb.calibrate(empty / "runs", empty / "judgments", out, part="all")
    assert result["nli"] == previous["nli"] and result["v1"] == previous["v1"]
    assert result["v2"] is None
    assert "keeping the previous block" in capsys.readouterr().err
    with pytest.raises(ValueError):
        cb.calibrate(empty / "runs", empty / "judgments", out, part="v3")


def _v2(
    candidate=True, answerable=True, covered=True, entail=(0.9, 0.2), claims=((0, "SUPPORTED"),)
):
    labels = [lab for _, lab in claims]
    return cb.V2Response(
        item_id="x",
        turn_index=0,
        candidate=candidate,
        answerable=answerable,
        covered=covered,
        hallucinated=any(lab != "SUPPORTED" for lab in labels),
        entailments=tuple(entail),
        judge_claims=tuple(claims),
    )


def test_simulate_v2_rule():
    resp = _v2(entail=(0.9, 0.2, None), claims=((0, "SUPPORTED"), (1, "UNSUPPORTED")))
    # θ 0.5: sentence 1 unsupported → fraction 1/2
    assert cb.simulate_v2(resp, 0.5, 0.5) == (False, True)  # kept, bad sentence removed
    assert cb.simulate_v2(resp, 0.5, 0.4) == (False, False)  # withdrawn → ABSTAIN
    assert cb.simulate_v2(resp, 0.1, 0.0) == (True, True)  # nothing removed
    unlinked = _v2(entail=(0.9,), claims=((None, "UNSUPPORTED"),))
    assert cb.simulate_v2(unlinked, 0.5, 1.0) == (True, True)  # cannot be removed
    template = _v2(candidate=False, entail=(), claims=())
    assert cb.simulate_v2(template, 0.5, 0.0) == (False, True)


def test_calibrate_v2_objective():
    good = [_v2(entail=(0.92, 0.88), claims=((0, "SUPPORTED"), (1, "SUPPORTED")))] * 6
    mixed = [_v2(entail=(0.9, 0.15), claims=((0, "SUPPORTED"), (1, "UNSUPPORTED")))] * 3
    bad = [_v2(entail=(0.2, 0.1), claims=((0, "UNSUPPORTED"), (1, "CONTRADICTED")))] * 1
    res = cb.calibrate_v2(good + mixed + bad)
    theta = res["thresholds"]["GROUNDING_ENTAILMENT_THRESHOLD"]
    assert 0.2 < theta <= 0.88 and res["kappa_sentence"] == pytest.approx(1.0)
    # τ removes every hallucination while keeping >= 85 % of coverage (9/10 ≥ 0.85)
    assert res["dev_result"]["hallucination_rate"] == 0.0
    assert res["dev_result"]["coverage"] >= 0.85 * res["c5_dev_baseline"]["coverage"]
    assert res["thresholds"]["GROUNDING_MAX_UNSUPPORTED_FRACTION"] >= 0.5  # keeps the mixed ones
    assert cb.calibrate_v2([_v2(candidate=False, entail=(), claims=())]) is None


def test_verifier_chunks_follow_marker_order():
    record = {
        "citations": [{"marker": "[1]", "chunk_id": "c3"}, {"marker": "[2]", "chunk_id": "c1"}],
        "evidence_texts": [
            {"chunk_id": "c1", "text": "one"},
            {"chunk_id": "c2", "text": "two"},
            {"chunk_id": "c3", "text": "three"},
        ],
    }
    assert [c["chunk_id"] for c in cb.verifier_chunks(record)] == ["c3", "c1", "c2"]
    assert cb.verifier_chunks(record)[0]["chunk_text"] == "three"


async def test_unavailable_verifier_is_an_error(tmp_path):
    paths = write_study(tmp_path, conditions=("C5",), n_per_category=3)
    from evaluation.io import judgment_file, load_judgments, load_run_records

    records = [r for r in load_run_records(paths["runs"])["C5"] if r["split"] == "dev"]
    judgments = load_judgments(judgment_file(paths["judgments"], "llm", "C5"))
    with pytest.raises(RuntimeError):
        await cb.score_v2_responses(records, judgments, FakeVerifier(available=False))


def test_degraded_c5_turns_are_not_used(tmp_path):
    paths = write_study(tmp_path, conditions=("C5",), n_per_category=3)
    from evaluation.io import load_run_records

    runs = load_run_records(paths["runs"])
    clean, dropped = cb._clean_c5_dev(runs)
    assert dropped == 0
    first_dev = next(r for r in runs["C5"] if r["split"] == "dev")
    first_dev["explanation"]["gate2"]["source"] = "fallback"
    clean, dropped = cb._clean_c5_dev(runs)
    assert dropped >= 1 and first_dev not in clean


def test_traces_skip_errors_and_use_logged_scores(tmp_path):
    paths = write_study(tmp_path, conditions=("C5",), n_per_category=3)
    from evaluation.io import load_run_records

    records = load_run_records(paths["runs"])["C5"]
    records[0]["error"] = "x"
    traces = cb.build_traces(records)
    assert len(traces) == len(records) - 1
    ran = [t for t in traces if t.gate2_ran]
    assert ran and all(t.max_rerank > cb.DISABLED for t in ran if t.category != "emergency")


def test_logged_gate_features_are_preferred():
    record = {
        "user_text": "unrelated words",
        "explanation": {
            "gate2": {"features": {"max_rerank": 4.0, "entity_coverage": 0.25, "overridden": False}}
        },
        "evidence_texts": [],
    }
    feats = cb.record_features(record)
    assert feats["max_rerank"] == 4.0 and feats["entity_coverage"] == 0.25
