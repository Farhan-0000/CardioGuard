"""NLI judge with a fake scorer: windows, threshold rule, relabelling, staleness, CLI loop."""

import json

import pytest

from evaluation import judge as jg
from evaluation.io import append_jsonl, load_judgments
from evaluation.judges.common import EvidenceSidecar
from evaluation.judges.nli_judge import (
    NLIJudge,
    _label_index,
    claims_fingerprint,
    load_thresholds,
    nli_label,
    premise_windows,
    relabel,
)


class OverlapScorer:
    """Entailment = token overlap of hypothesis with premise; contradiction when 'never' is in it."""

    name = "fake-nli"

    def __init__(self):
        self.pairs = []

    def score(self, pairs):
        self.pairs.extend(pairs)
        out = []
        for premise, hypothesis in pairs:
            p, h = set(premise.lower().split()), set(hypothesis.lower().split())
            entail = len(p & h) / len(h)
            out.append((entail, 0.9 if "never" in p else 0.0))
        return out


def _llm_judgment(claims):
    return {
        "condition": "C5",
        "item_id": "I1",
        "turn_index": 0,
        "attempt_id": "a",
        "split": "test",
        "category": "answerable_complete",
        "subcategory": "x",
        "action": "ANSWER",
        "template": None,
        "claims": claims,
        "error": None,
    }


def test_premise_windows():
    text = "One. Two! Three? Four; five.\nSix."
    windows = premise_windows(text, 3)
    assert windows[0] == "One. Two! Three?" and len(windows) == 4
    assert premise_windows("Only one sentence.", 3) == ["Only one sentence."]
    assert premise_windows("", 3) == []


@pytest.mark.parametrize(
    "entail,contra,label",
    [
        (0.7, 0.9, "SUPPORTED"),
        (0.2, 0.6, "CONTRADICTED"),
        (0.4, 0.3, "UNSUPPORTED"),
        (0.5, 0.5, "SUPPORTED"),
        (0.6, 0.55, "SUPPORTED"),
    ],
)
def test_label_rule(entail, contra, label):
    assert nli_label(entail, contra, 0.5, 0.5) == label


def test_judge_turn_uses_llm_evidence_ids():
    claims = [
        {
            "index": 1,
            "claim": "salt raises blood pressure",
            "evidence_ids": ["c1"],
            "pool_ids": ["c1", "c2"],
        },
        {
            "index": 2,
            "claim": "aspirin cures hypertension",
            "evidence_ids": ["c2"],
            "pool_ids": ["c2"],
        },
        {"index": 3, "claim": "orphan claim", "evidence_ids": ["missing"], "pool_ids": []},
    ]
    texts = {
        "c1": "High salt intake raises blood pressure. Other text here.",
        "c2": "Aspirin should never be used to treat hypertension.",
    }
    scorer = OverlapScorer()
    out = NLIJudge(scorer, 0.75, 0.5).judge_turn(_llm_judgment(claims), texts.get)
    labels = [c["label"] for c in out["claims"]]
    assert labels == ["SUPPORTED", "CONTRADICTED", "UNSUPPORTED"]
    assert out["claims"][0]["best_chunk_id"] == "c1"
    assert out["claims"][2]["n_windows"] == 0 and out["claims"][2]["entailment"] == 0.0
    assert out["n_supported"] == 1 and out["n_contradicted"] == 1 and out["n_unsupported"] == 1
    assert out["claims_fingerprint"] == claims_fingerprint(_llm_judgment(claims))
    assert all(h in {c["claim"] for c in claims} for _, h in scorer.pairs)


def test_relabel_and_thresholds(tmp_path):
    judgment = {
        "claims": [
            {"entailment": 0.6, "contradiction": 0.1},
            {"entailment": 0.3, "contradiction": 0.8},
        ]
    }
    strict = relabel(judgment, 0.7, 0.5)
    assert [c["label"] for c in strict["claims"]] == ["UNSUPPORTED", "CONTRADICTED"]
    assert strict["n_unsupported"] == 1 and strict["theta"] == 0.7
    assert load_thresholds(tmp_path / "missing.json") == (0.5, 0.5)
    (tmp_path / "cal.json").write_text(json.dumps({"nli": {"theta": 0.35, "theta_c": 0.8}}))
    assert load_thresholds(tmp_path / "cal.json") == (0.35, 0.8)


def test_label_index_mapping():
    id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}
    assert _label_index(id2label, "entail") == 1 and _label_index(id2label, "contradict") == 0
    with pytest.raises(ValueError):
        _label_index({0: "yes"}, "entail")


def test_judge_with_nli_cli_loop(tmp_path):
    claims = [
        {
            "index": 1,
            "claim": "salt raises blood pressure",
            "evidence_ids": ["c1", "judge_1"],
            "pool_ids": ["c1", "judge_1"],
        }
    ]
    llm = _llm_judgment(claims)
    append_jsonl(tmp_path / "llm" / "C5.jsonl", llm)
    append_jsonl(
        tmp_path / "evidence_chunks.jsonl",
        {"chunk_id": "judge_1", "text": "Salt raises blood pressure."},
    )
    records = {
        "C5": [
            {
                "item_id": "I1",
                "turn_index": 0,
                "attempt_id": "a",
                "evidence_texts": [{"chunk_id": "c1", "text": "unrelated"}],
            },
            {"item_id": "I2", "turn_index": 0, "attempt_id": "z", "evidence_texts": []},
        ]
    }
    sidecar = EvidenceSidecar(tmp_path)
    judge = NLIJudge(OverlapScorer())
    counts = jg.judge_with_nli(
        judge, records, tmp_path, sidecar, resume=False, fresh=False, log=lambda m: None
    )
    assert counts["C5"] == {"records": 2, "judged": 1, "missing_llm": 1}
    saved = load_judgments(tmp_path / "nli" / "C5.jsonl")[("I1", 0)]
    assert (
        saved["claims"][0]["label"] == "SUPPORTED"
        and saved["claims"][0]["best_chunk_id"] == "judge_1"
    )
    counts = jg.judge_with_nli(
        judge, records, tmp_path, sidecar, resume=True, fresh=False, log=lambda m: None
    )
    assert counts["C5"]["judged"] == 0
    llm["claims"][0]["claim"] = "salt is fine"  # LLM judgment changed → NLI judgment is stale
    append_jsonl(tmp_path / "llm" / "C5.jsonl", llm)
    counts = jg.judge_with_nli(
        judge, records, tmp_path, sidecar, resume=True, fresh=False, log=lambda m: None
    )
    assert counts["C5"]["judged"] == 1


def test_nli_is_refreshed_for_a_rerun_attempt_with_identical_claims(tmp_path):
    """statistics-judges-2: a re-run item keeps its NLI outcome (incl. zero-claim templates)."""
    from evaluation.io import load_all_judgments
    from evaluation.outcomes import build_outcomes

    claims = [{"index": 1, "claim": "salt raises blood pressure", "evidence_ids": ["c1"]}]
    record = {
        "condition": "C5",
        "item_id": "I1",
        "turn_index": 0,
        "n_turns": 1,
        "attempt_id": "a",
        "action": "ANSWER",
        "evidence_texts": [{"chunk_id": "c1", "text": "Salt raises blood pressure."}],
    }
    template = {**record, "item_id": "I2", "action": "ABSTAIN", "evidence_texts": []}
    llm = _llm_judgment(claims)
    append_jsonl(tmp_path / "llm" / "C5.jsonl", llm)
    append_jsonl(
        tmp_path / "llm" / "C5.jsonl",
        {**_llm_judgment([]), "item_id": "I2", "template": "abstain_template"},
    )
    judge, sidecar = NLIJudge(OverlapScorer()), EvidenceSidecar(tmp_path)
    counts = jg.judge_with_nli(
        judge,
        {"C5": [record, template]},
        tmp_path,
        sidecar,
        resume=False,
        fresh=False,
        log=lambda m: None,
    )
    assert counts["C5"]["judged"] == 2
    # both items re-run (new attempt "b"), deterministic pipeline → identical claims
    rerun = [{**record, "attempt_id": "b"}, {**template, "attempt_id": "b"}]
    append_jsonl(tmp_path / "llm" / "C5.jsonl", {**llm, "attempt_id": "b"})
    append_jsonl(
        tmp_path / "llm" / "C5.jsonl",
        {**_llm_judgment([]), "item_id": "I2", "template": "abstain_template", "attempt_id": "b"},
    )
    counts = jg.judge_with_nli(
        judge, {"C5": rerun}, tmp_path, sidecar, resume=True, fresh=False, log=lambda m: None
    )
    assert counts["C5"]["judged"] == 2
    saved = load_judgments(tmp_path / "nli" / "C5.jsonl")
    assert saved[("I1", 0)]["attempt_id"] == "b" and saved[("I2", 0)]["attempt_id"] == "b"
    judgments = {j: load_all_judgments(tmp_path, j) for j in ("llm", "nli")}
    outcomes = build_outcomes({"C5": rerun}, judgments)["C5"]
    assert all(o.claims["nli"] is not None for o in outcomes)
    # a second pass with nothing new is a no-op
    counts = jg.judge_with_nli(
        judge, {"C5": rerun}, tmp_path, sidecar, resume=True, fresh=False, log=lambda m: None
    )
    assert counts["C5"]["judged"] == 0


def test_cited_chunks_are_scored_separately():
    """statistics-judges-8: the NLI judge scores the chunk each sentence actually cites."""
    claims = [
        {"index": 1, "claim": "salt raises blood pressure", "evidence_ids": ["c1"]},
        {"index": 2, "claim": "statins lower cholesterol", "evidence_ids": ["c1"]},
    ]
    texts = {
        "c1": "High salt intake raises blood pressure.",
        "c2": "Statins lower cholesterol in most adults.",
    }
    out = NLIJudge(OverlapScorer(), 0.75, 0.5).judge_turn(
        _llm_judgment(claims), texts.get, {1: ["c2"], 2: ["c2"]}
    )
    first, second = out["claims"]
    assert first["label"] == "SUPPORTED" and first["best_chunk_id"] == "c1"
    assert first["n_windows"] == 1  # evidence windows only
    assert first["cited_scores"]["c2"]["entailment"] < 0.75  # cited chunk does not support it
    assert second["cited_scores"]["c2"]["entailment"] == 1.0
    assert (
        "cited_scores"
        not in NLIJudge(OverlapScorer()).judge_turn(_llm_judgment(claims), texts.get)["claims"][0]
    )


def test_cited_chunks_by_claim_follow_the_sentence_markers():
    from evaluation.judges.common import cited_chunks_by_claim

    record = {
        "response_text": "Salt raises blood pressure [1]. Statins lower cholesterol [2][1].",
        "citations": [{"marker": "[1]", "chunk_id": "c9"}, {"marker": "[2]", "chunk_id": "c4"}],
    }
    claims = [
        {"index": 1, "claim": "Salt raises blood pressure", "source_sentence": "Salt raises"},
        {"index": 2, "claim": "Statins lower cholesterol"},
        {"index": 3, "claim": "unrelated words entirely"},
    ]
    assert cited_chunks_by_claim(record, claims) == {1: ["c9"], 2: ["c4", "c9"], 3: []}


def test_nli_calibrated_flag_requires_an_nli_block(tmp_path):
    """statistics-judges-7: the file existing is not enough."""
    from evaluation.judges.nli_judge import nli_calibration

    assert nli_calibration(tmp_path / "missing.json") == (0.5, 0.5, False)
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"nli": None, "v1": {"thresholds": {}}}))
    assert nli_calibration(path) == (0.5, 0.5, False)
    path.write_text(json.dumps({"nli": {"theta": 0.4, "theta_c": 0.7}}))
    assert nli_calibration(path) == (0.4, 0.7, True)
