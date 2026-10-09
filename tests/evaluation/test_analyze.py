"""End-to-end analysis on a synthetic study + unit checks of outcome definitions."""

import json

import pytest

from evaluation import analyze as az
from evaluation.io import load_all_judgments, load_run_records
from evaluation.outcomes import TurnOutcome, build_outcomes, impute_latency, uncached_call_medians
from tests.evaluation.synthetic import write_study


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("study")
    paths = write_study(root)
    out = root / "out"
    summary = az.analyze(
        runs_dir=paths["runs"],
        judgments_dir=paths["judgments"],
        calibration_path=root / "none.json",
        split="test",
        summary_path=out / "summary.json",
        tables_dir=out / "tables",
        figures_dir=out / "figures",
        error_analysis_path=out / "error_analysis.md",
        n_boot=300,
    )
    return paths, out, summary


def test_summary_structure_and_split(study):
    _, out, summary = study
    assert summary["split"] == "test"
    assert summary["conditions"] == ["C0", "C1", "C2", "C3", "C4", "C5", "V1"]
    c5 = summary["per_condition"]["C5"]
    # 4 test items per category (i % 3 != 0 of 6), missing-context items have 2 turns
    assert c5["n_items"] == 24
    assert c5["n_turns"] == 28
    on_disk = json.loads((out / "summary.json").read_text())
    assert on_disk["per_condition"]["C1"]["llm"]["hallucination"]["n"] > 0


def test_gating_effects_visible(study):
    per = study[2]["per_condition"]
    assert per["C5"]["llm"]["hallucination"]["rate"] < per["C1"]["llm"]["hallucination"]["rate"]
    assert per["C5"]["llm"]["unsafe_answer"]["rate"] < per["C3"]["llm"]["unsafe_answer"]["rate"]
    assert per["C1"]["routing_system"] is None and per["C1"]["routing_behaviour"] is not None
    assert per["C5"]["routing_system"]["n"] == 28
    assert per["C5"]["emergency"]["recall_system"]["rate"] == 1.0
    assert per["C5"]["emergency"]["recall_fastpath"]["rate"] == 1.0
    assert per["C0"]["emergency"]["recall_system"]["rate"] == 0.0
    ratio = per["C1"]["llm"]["unsupported_claim_ratio"]
    assert ratio["low"] <= ratio["estimate"] <= ratio["high"]


def test_statistics_present(study):
    tests = study[2]["tests"]
    families = {r["family"] for r in tests["mcnemar"]}
    assert families == {"c5_vs_baselines", "variants_vs_c5"}
    c5_vs_c1 = next(
        r
        for r in tests["mcnemar"]
        if r["control"] == "C1" and r["outcome"] == "hallucination" and r["judge"] == "llm"
    )
    assert c5_vs_c1["pvalue_holm"] >= c5_vs_c1["pvalue"]
    assert c5_vs_c1["risk_difference"]["estimate"] < 0
    assert c5_vs_c1["odds_ratio"]["estimate"] < 1
    assert any(r["outcome"] == "unsafe" for r in tests["mcnemar"])
    assert tests["cochran_q"][0]["conditions"] == ["C0", "C1", "C2", "C3", "C4", "C5"]
    assert tests["sensitivity_item_level"]
    agreement = study[2]["judge_agreement"]
    assert agreement["n_claims"] > 0 and -1 <= agreement["kappa_3class"]["kappa"] <= 1


def test_outputs_written(study):
    _, out, summary = study
    figures = out / "figures"
    for name in (
        "hallucination_rate_by_condition",
        "unsafe_answer_rate_unanswerable",
        "risk_coverage",
        "ablation_waterfall",
        "confusion_matrix_C5",
        "confusion_matrix_V1",
        "claim_support_stacked",
        "latency_by_condition",
        "judge_agreement",
    ):
        assert (figures / f"{name}.png").stat().st_size > 1000, name
        assert (figures / f"{name}.svg").exists(), name
    for table in (
        "main_results",
        "statistical_tests",
        "cochran_q",
        "routing_per_class",
        "category_breakdown",
        "latency",
        "judge_agreement",
    ):
        assert (out / "tables" / f"{table}.md").exists()
        assert (out / "tables" / f"{table}.csv").exists()
    report = (out / "error_analysis.md").read_text()
    assert "C5 false ANSWER" in report and "Judge disagreements" in report
    assert summary["figures"] and summary["tables"]


def _outcome(**kw) -> TurnOutcome:
    base = dict(
        condition="C5",
        item_id="X",
        turn_index=0,
        category="unanswerable_in_domain",
        subcategory="fabricated_drug",
        split="test",
        gold_action="ABSTAIN",
        acceptable=("ABSTAIN",),
        user_text="q",
        response_text="r",
        action="ANSWER",
        error=None,
    )
    base.update(kw)
    return TurnOutcome(**base)


def test_unsafe_definition():
    o = _outcome(behaviour="GENERAL_INFO")
    o.claims["llm"] = [{"label": "SUPPORTED"}]
    assert o.unsafe("llm") == 0 and o.hallucinated("llm") == 0
    o.flags = {"endorsed_fabricated_entity": True}
    assert o.unsafe("llm") == 1
    o2 = _outcome(behaviour="ANSWERED_SPECIFIC")
    o2.claims["llm"] = []
    assert o2.unsafe("llm") == 1
    o3 = _outcome(behaviour="GENERAL_INFO", category="answerable_complete")
    o3.claims["llm"] = [{"label": "UNSUPPORTED"}]
    assert o3.unsafe("llm") is None and o3.hallucinated("llm") == 1
    assert _outcome().unsafe("llm") is None  # not judged


def test_citation_precision_counts():
    o = _outcome(
        response_text="BP target is below 130/80 [1]. Salt raises BP [2]. Hello [1].",
        citations=[{"marker": "[1]"}],
    )
    o.claims["llm"] = [
        {
            "index": 1,
            "claim": "BP target is below 130/80",
            "source_sentence": "BP target is below 130/80 [1].",
            "label": "SUPPORTED",
        },
        {
            "index": 2,
            "claim": "Salt raises BP",
            "source_sentence": "Salt raises BP [2].",
            "label": "SUPPORTED",
        },
    ]
    denominator, numerator, unverifiable = az.citation_counts(o)
    assert (denominator, numerator, unverifiable) == (2, 1, 1)  # [2] is not a valid marker


def test_attempt_mismatch_ignored(tmp_path):
    paths = write_study(tmp_path, conditions=("C5",), n_per_category=1)
    runs = load_run_records(paths["runs"])
    runs["C5"][0]["attempt_id"] = "different"
    judgments = {j: load_all_judgments(paths["judgments"], j) for j in ("llm", "nli")}
    outcomes = build_outcomes(runs, judgments)
    assert outcomes["C5"][0].claims["llm"] is None
    assert outcomes["C5"][1].claims["llm"] is not None


def test_latency_imputation():
    records = [
        {
            "latency_ms": 100.0,
            "llm_calls": [{"call_name": "g", "model": "m", "latency_ms": 900.0, "cached": False}],
        },
        {
            "latency_ms": 50.0,
            "llm_calls": [{"call_name": "g", "model": "m", "latency_ms": 1.0, "cached": True}],
        },
    ]
    medians = uncached_call_medians(records)
    assert medians == {("g", "m"): 900.0}
    assert impute_latency(records[1], medians) == (950.0, 1)
    assert impute_latency({"latency_ms": None}, medians) == (None, 0)
