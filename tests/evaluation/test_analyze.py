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
        "sensitivity_item_level",
        "degraded_turns",
        "sensitivity_metrics",
        "sensitivity_tests",
        "sensitivity_claim_filter",
        "stratified_by_category",
        "stratified_tests",
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


# ----------------------------------------------------------------------
# Degraded turns: headline excludes them, sensitivity includes them, > 5 % refuses
# ----------------------------------------------------------------------


def _analyze(paths, out, **kw):
    kw.setdefault("calibration_path", paths["runs"].parent / "none.json")
    return az.analyze(
        runs_dir=paths["runs"],
        judgments_dir=paths["judgments"],
        split="test",
        summary_path=out / "summary.json",
        tables_dir=out / "tables",
        figures_dir=out / "figures",
        error_analysis_path=out / "error_analysis.md",
        n_boot=200,
        **kw,
    )


def _degrade(runs_dir, condition, n, how="gate2"):
    """Rewrite the first ``n`` test records of a condition as orchestrator fallbacks."""
    path = runs_dir / f"{condition}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    done = 0
    for row in rows:
        if done < n and row["split"] == "test" and row["n_turns"] == 1:
            if how == "gate2":
                row["explanation"]["gate2"] = {"action": "GENERALIZE", "source": "fallback"}
            else:
                row["llm_calls"].append({"call_name": "context_gate", "success": False})
            done += 1
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_synthetic_study_has_no_degraded_turns(study):
    summary = study[2]
    assert summary["degraded"]["exceeded"] == [] and summary["degraded"]["override"] is False
    assert all(d["n_degraded"] == 0 for d in summary["degraded"]["per_condition"].values())
    assert summary["nli_thresholds"]["calibrated"] is False  # no calibration file


def test_degraded_rate_above_threshold_refuses_headline(tmp_path):
    paths = write_study(tmp_path, conditions=("C1", "C5"))
    _degrade(paths["runs"], "C1", 3)  # 3 / 28 test turns > 5 %
    with pytest.raises(az.AnalysisRefused):
        _analyze(paths, tmp_path / "out")
    assert not (tmp_path / "out" / "summary.json").exists()
    assert (
        az.main(
            [
                "--runs-dir",
                str(paths["runs"]),
                "--judgments-dir",
                str(paths["judgments"]),
                "--summary",
                str(tmp_path / "out" / "s.json"),
                "--tables-dir",
                str(tmp_path / "t"),
                "--figures-dir",
                str(tmp_path / "f"),
                "--error-analysis",
                str(tmp_path / "e.md"),
                "--n-boot",
                "50",
            ]
        )
        == 2
    )
    summary = _analyze(paths, tmp_path / "out", allow_degraded=True)
    deg = summary["degraded"]
    assert deg["exceeded"] == ["C1"] and deg["override"] is True
    assert deg["per_condition"]["C1"]["by_reason"] == {"gate2_fallback": 3}
    c1 = summary["per_condition"]["C1"]
    assert c1["n_degraded_excluded"] == 3 and c1["n_valid"] == c1["n_turns"] - 3
    including = summary["sensitivity"]["including_degraded"]["per_condition"]["C1"]
    assert including["n_valid"] == c1["n_turns"]
    # degraded turns never enter the paired headline tests
    c5_vs_c1 = next(
        r
        for r in summary["tests"]["mcnemar"]
        if r["control"] == "C1" and r["outcome"] == "hallucination" and r["judge"] == "llm"
    )
    assert c5_vs_c1["n_pairs"] == c1["n_turns"] - 3
    report = (tmp_path / "out" / "error_analysis.md").read_text()
    assert "Degraded turns" in report and "gate2_fallback" in report


def test_small_degraded_share_is_reported_but_allowed(tmp_path):
    paths = write_study(tmp_path, conditions=("C1", "C5"))
    _degrade(paths["runs"], "C5", 1, how="llm")  # 1 / 28 < 5 %
    summary = _analyze(paths, tmp_path / "out")
    assert summary["degraded"]["per_condition"]["C5"]["by_reason"] == {
        "llm_call_failed:context_gate": 1
    }
    assert summary["per_condition"]["C5"]["n_degraded_excluded"] == 1


# ----------------------------------------------------------------------
# Item-level McNemar uses identical turn sets (statistics-judges-4)
# ----------------------------------------------------------------------


def test_item_level_pairs_only_complete_items():
    def turn(condition, index, hallucinated):
        o = _outcome(
            condition=condition,
            item_id="X",
            turn_index=index,
            category="answerable_missing_context",
            n_turns=2,
            action="ANSWER",
        )
        o.claims["llm"] = (
            None
            if hallucinated is None
            else [{"label": "UNSUPPORTED" if hallucinated else "SUPPORTED"}]
        )
        return o

    outcomes = {
        "C5": [turn("C5", 0, False), turn("C5", 1, None)],  # judge failed on turn 1
        "C1": [turn("C1", 0, False), turn("C1", 1, True)],
    }
    turn_level = az.compare(outcomes, "C5", "C1", "hallucination", "llm", "turn")
    assert turn_level["n_pairs"] == 1 and turn_level["c_control_only"] == 0
    assert az.compare(outcomes, "C5", "C1", "hallucination", "llm", "item") is None
    outcomes["C5"][1] = turn("C5", 1, False)
    item = az.compare(outcomes, "C5", "C1", "hallucination", "llm", "item")
    assert item["n_pairs"] == 1 and item["c_control_only"] == 1 and item["n_items_excluded"] == 0


# ----------------------------------------------------------------------
# Citation metrics (statistics-judges-6 / -8)
# ----------------------------------------------------------------------


def test_marker_validity_counts_occurrences():
    # validate_citations output for 'Salt raises BP [1]. DASH lowers BP [1]. Exercise helps [1].
    # Walk daily [2]. Drug X is new [7].' with 3 chunks: valid=2 (distinct), stripped=1
    o = _outcome(
        response_text="Salt raises BP [1]. DASH lowers BP [1]. Exercise helps [1]. Walk daily [2]. "
        "Drug X is new.",
        citation_validation={"valid": 2, "stripped": 1},
    )
    assert az.marker_reference_counts(o) == (4, 1)  # 80 %, not 2/3
    o.citation_validation = {"valid": 2, "valid_refs": 4, "stripped": 1}
    assert az.marker_reference_counts(o) == (4, 1)
    assert az.marker_reference_counts(_outcome()) == (0, 0)


def test_supported_citation_precision_checks_the_cited_chunk():
    o = _outcome(
        response_text="Reducing salt lowers blood pressure [1].",
        citations=[{"marker": "[1]", "chunk_id": "statins_chunk"}],
    )
    claim = {
        "index": 1,
        "claim": "Reducing salt lowers blood pressure",
        "source_sentence": "Reducing salt lowers blood pressure [1].",
        "label": "SUPPORTED",
        "evidence_ids": ["statins_chunk", "judge_sodium"],
        "cited_evidence": ["judge_sodium"],  # supported by the judge's own retrieval
    }
    o.claims["llm"] = [claim]
    o.claims["nli"] = [
        {"index": 1, "cited_scores": {"statins_chunk": {"entailment": 0.05, "contradiction": 0.0}}}
    ]
    counts = az.supported_citation_counts(o, theta=0.5)
    assert counts["markers"] == 1 and counts["evaluable"] == 1 and counts["precise"] == 0
    assert counts["nli_evaluable"] == 1 and counts["nli_precise"] == 0
    # the chunk-blind metric (kept as "markers in fully supported sentences") counts it
    assert az.citation_counts(o) == (1, 1, 0)
    claim["cited_evidence"] = ["statins_chunk"]
    o.claims["nli"][0]["cited_scores"]["statins_chunk"]["entailment"] = 0.8
    counts = az.supported_citation_counts(o, theta=0.5)
    assert counts["precise"] == 1 and counts["nli_precise"] == 1
    claim["evidence_ids"] = ["judge_sodium"]  # the cited chunk was never shown to the judge
    counts = az.supported_citation_counts(o, theta=0.5)
    assert counts["evaluable"] == 0 and counts["markers"] == 1


def test_citation_summary_fields(study):
    citations = study[2]["per_condition"]["C1"]["citations"]
    for key in (
        "supported_precision",
        "supported_precision_nli",
        "marker_validity",
        "markers_in_fully_supported_sentences",
        "markers_unverifiable_llm",
    ):
        assert key in citations
    assert "precision" not in citations  # the chunk-blind metric is no longer called precision


# ----------------------------------------------------------------------
# Sensitivity: prompt_overlap items, claim filter; stratification by category
# ----------------------------------------------------------------------


def test_prompt_overlap_sensitivity_and_dataset_fallback(study, tmp_path):
    paths, _, summary = study
    view = summary["sensitivity"]["without_prompt_overlap"]
    assert view["n_items_removed"] == 6  # one test item per category is tagged
    assert all(i.endswith("-001") for i in view["items_removed"])
    full = summary["per_condition"]["C5"]["n_valid"]
    assert view["per_condition"]["C5"]["n_valid"] == full - 7  # the MC item has two turns
    assert view["tests"] and all(r["unit"] == "turn" for r in view["tests"])
    # records written before the runner stored tags get them from the dataset
    for path in paths["runs"].glob("*.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text("\n".join(json.dumps({**r, "tags": []}) for r in rows) + "\n")
    no_tags = _analyze(paths, tmp_path / "a", dataset=None)
    assert no_tags["sensitivity"]["without_prompt_overlap"]["n_items_removed"] == 0
    with_dataset = _analyze(paths, tmp_path / "b", dataset=paths["dataset"])
    assert with_dataset["sensitivity"]["without_prompt_overlap"]["n_items_removed"] == 6


def test_claim_filter_sensitivity(tmp_path):
    paths = write_study(tmp_path, conditions=("C1", "C5"))
    path = paths["judgments"] / "llm" / "C1.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    modified = 0
    for row in rows:
        if row["claims"] and row["split"] == "test":
            row["excluded_claims"] = [
                {
                    "claim": "Talk to your doctor",
                    "reason": "generic_referral",
                    "label": "UNSUPPORTED",
                }
            ]
            modified += 1
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary = _analyze(paths, tmp_path / "out")
    filt = summary["sensitivity"]["filter_disabled"]["per_condition"]
    assert filt["C1"]["hallucination_unfiltered"]["k"] == modified  # every judged answer now
    assert filt["C1"]["excluded_claims"] == modified
    assert (
        filt["C1"]["hallucination_unfiltered"]["rate"]
        > filt["C1"]["hallucination_filtered"]["rate"]
    )
    assert filt["C1"]["excluded_by_reason"] == {"generic_referral": filt["C1"]["excluded_claims"]}
    assert filt["C5"]["excluded_claims"] == 0
    assert summary["sensitivity"]["filter_disabled"]["tests"]


def test_stratified_by_category(study):
    strata = study[2]["by_category"]
    assert set(strata) == {
        "answerable_complete",
        "answerable_missing_context",
        "general_topic",
        "unanswerable_in_domain",
        "out_of_domain",
        "emergency",
    }
    ui = strata["unanswerable_in_domain"]
    assert ui["per_condition"]["C5"]["n_valid"] == 4
    assert {r["outcome"] for r in ui["tests"]} == {"hallucination", "unsafe"}
    assert {r["outcome"] for r in strata["general_topic"]["tests"]} == {"hallucination"}


# ----------------------------------------------------------------------
# Calibration check of the variants; NLI calibrated flag (statistics-judges-3 / -7)
# ----------------------------------------------------------------------


def test_variants_are_flagged_unless_produced_with_current_calibration(study, tmp_path):
    paths, out, summary = study
    check = summary["calibration_check"]["V1"]
    assert check["calibrated"] is False  # synthetic V1 records carry no calibration
    assert "(UNCALIBRATED)" in (out / "tables" / "main_results.md").read_text()
    thresholds = {
        "EVIDENCE_MIN_RERANK_ANSWER": 1.0,
        "EVIDENCE_MIN_RERANK_ANY": -4.0,
        "EVIDENCE_MIN_ENTITY_COVERAGE": 0.2,
    }
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps({"nli": None, "v1": {"thresholds": thresholds}}))
    root = tmp_path / "copy"
    root.mkdir()
    paths2 = write_study(root, conditions=("C5", "V1"))
    path = paths2["runs"] / "V1.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text(
        "\n".join(
            json.dumps({**r, "calibration": {"status": "calibrated", "values": thresholds}})
            for r in rows
        )
        + "\n"
    )
    summary = _analyze(paths2, root / "out", calibration_path=cal)
    assert summary["calibration_check"]["V1"]["calibrated"] is True
    assert summary["nli_thresholds"]["calibrated"] is False  # nli: null → defaults
    thresholds_changed = {**thresholds, "EVIDENCE_MIN_RERANK_ANY": -2.0}
    cal.write_text(json.dumps({"v1": {"thresholds": thresholds_changed}}))
    summary = _analyze(paths2, root / "out2", calibration_path=cal)
    assert summary["calibration_check"]["V1"]["calibrated"] is False


def test_unfiltered_claims_need_labelled_exclusions():
    o = _outcome()
    o.claims["llm"] = [{"label": "SUPPORTED"}]
    o.excluded_claims = [{"claim": "Talk to your doctor", "reason": "generic_referral"}]
    assert o.claims_unfiltered() is None and o.hallucinated_unfiltered() is None  # old judgment
    o.excluded_claims[0]["label"] = "UNSUPPORTED"
    assert o.hallucinated_unfiltered() == 1 and o.hallucinated("llm") == 0
    o.excluded_claims = []
    assert o.hallucinated_unfiltered() == 0
    assert _outcome().claims_unfiltered() is None  # not judged


def test_c1g_reference_anchors_the_gate_waterfall(tmp_path):
    """C1 is now vanilla RAG; gate effects are measured from C1G (same prompt as C3-C5)."""
    paths = write_study(tmp_path, conditions=("C1", "C1G", "C3", "C4", "C5"))
    summary = _analyze(paths, tmp_path / "out")
    assert summary["conditions"][:2] == ["C1", "C1G"]
    reference = [r for r in summary["tests"]["mcnemar"] if r["family"] == "c5_vs_reference"]
    assert reference and all(r["control"] == "C1G" for r in reference)
    assert all(r["pvalue_holm"] == r["pvalue"] for r in reference)  # its own Holm family
    assert (tmp_path / "out" / "figures" / "ablation_waterfall.png").exists()
    baselines = {
        r["control"] for r in summary["tests"]["mcnemar"] if r["family"] == "c5_vs_baselines"
    }
    assert "C1G" not in baselines  # the pre-registered family is unchanged
