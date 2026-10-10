"""
Tests for the review-presentation builder (report/build_presentation.py + report/presentation/**).

A deck is built from a fixture repository root in ``tmp_path`` (small synthetic results files
with the real schemas) and read back with python-pptx: slide count and size, speaker notes on
every slide, team placeholders, pending slides while the hallucination study is missing,
real numbers once a ``summary.json`` exists, screenshots versus placeholders, ignored
mock-mode reports, citations and the layout checks of the drawing kit.

The study summary is produced with the real ``evaluation/analyze.py`` functions from
synthetic turn outcomes (test data only, written to ``tmp_path``, never to the repository).
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import pytest

pptx = pytest.importorskip("pptx")

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "report"
if str(REPORT_DIR) not in sys.path:
    sys.path.insert(0, str(REPORT_DIR))

import build_presentation as bp  # noqa: E402
from lib import tables  # noqa: E402
from presentation import deck_data  # noqa: E402
from presentation import pptx_kit as K  # noqa: E402

N_SLIDES = 18


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _interval(est: float, half: float = 0.05) -> dict[str, float]:
    return {"estimate": est, "low": est - half, "high": est + half}


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_png(path: Path, size: tuple[int, int] = (800, 500)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (210, 225, 240)).save(path)


def _synthetic_summary() -> dict:
    """summary.json with the exact schema of evaluation/analyze.py (synthetic outcomes)."""
    analyze = pytest.importorskip("evaluation.analyze")
    from evaluation.io import sanitize_json
    from evaluation.outcomes import TurnOutcome

    rng = random.Random(11)
    categories = [
        ("answerable_complete", "ANSWER", ("ANSWER", "GENERALIZE")),
        ("unanswerable_in_domain", "ABSTAIN", ("ABSTAIN", "GENERALIZE")),
        ("out_of_domain", "ABSTAIN", ("ABSTAIN",)),
        ("emergency", "ESCALATE", ("ESCALATE",)),
    ]
    behaviour = {"ANSWER": "ANSWERED_SPECIFIC", "ABSTAIN": "ABSTAINED", "ESCALATE": "ESCALATED"}
    outcomes: dict[str, list] = {}
    for cond, p_bad in (
        ("C0", 0.5),
        ("C1", 0.4),
        ("C2", 0.35),
        ("C3", 0.3),
        ("C4", 0.2),
        ("C5", 0.1),
    ):
        rows = []
        for i in range(40):
            cat, gold, acceptable = categories[i % len(categories)]
            action = gold if (cond in analyze.GATED and rng.random() > 0.2) else "ANSWER"
            bad = rng.random() < p_bad
            claims = [
                {
                    "index": 0,
                    "label": "UNSUPPORTED" if bad else "SUPPORTED",
                    "entailment": 0.2 if bad else 0.9,
                    "contradiction": 0.1,
                }
            ]
            rows.append(
                TurnOutcome(
                    condition=cond,
                    item_id=f"HB-{i:04d}",
                    turn_index=0,
                    category=cat,
                    subcategory="x",
                    split="test",
                    gold_action=gold,
                    acceptable=acceptable,
                    user_text="q",
                    response_text="a",
                    action=action,
                    error=None,
                    latency_ms=1000.0 + i,
                    latency_imputed_ms=1200.0 + i,
                    behaviour=behaviour.get(action, "GENERAL_INFO"),
                    claims={"llm": claims, "nli": [dict(c) for c in claims]},
                    fastpath=action == "ESCALATE",
                )
            )
        outcomes[cond] = rows
    per = {
        c: analyze.condition_metrics(rows, c, n_boot=200, seed=1) for c, rows in outcomes.items()
    }
    summary = {
        "generated_at": "test",
        "split": "test",
        "unit": "item-turn",
        "conditions": list(per),
        "degraded": {"per_condition": analyze.degraded_report(outcomes)},
        "calibration_check": {},
        "per_condition": per,
        "tests": analyze.run_tests(outcomes),
        "judge_agreement": analyze.judge_agreement(outcomes),
    }
    return sanitize_json(summary)


@pytest.fixture()
def fixture_root(tmp_path: Path) -> Path:
    """Repository root with small synthetic results files (no hallucination study yet)."""
    root = tmp_path / "repo"
    refs = root / "report" / "references"
    refs.mkdir(parents=True)
    for name in ("references.json", "patents.json"):
        shutil.copy(REPORT_DIR / "references" / name, refs / name)
    _write_json(
        root / "reports" / "corpus_stats.json",
        {
            "documents": [],
            "totals": {
                "documents": 5,
                "organisations": 3,
                "chunks": 1234,
                "pdf_documents": 2,
                "html_documents": 3,
                "pages_or_sections": 99,
                "chunk_tokens": {"max": 400},
            },
        },
    )
    modes = {
        "dense": {
            "recall@5": _interval(0.71),
            "recall@20": _interval(0.81),
            "mrr": _interval(0.61),
            "ndcg@5": _interval(0.6),
            "n": 12,
        },
        "hybrid": {
            "recall@5": _interval(0.72),
            "recall@20": _interval(0.82),
            "mrr": _interval(0.642),
            "ndcg@5": _interval(0.62),
            "n": 12,
        },
        "hybrid+rerank": {
            "recall@5": _interval(0.73),
            "recall@20": _interval(0.82),
            "mrr": _interval(0.65),
            "ndcg@5": _interval(0.63),
            "n": 12,
        },
    }
    _write_json(
        root / "evaluation" / "results" / "retrieval" / "metrics.json",
        {"modes": modes, "candidates": 20},
    )
    role = {"valid_first_attempt": {"rate": 1.0, "k": 10, "n": 10}, "tokens_per_s_median": 20.0}
    _write_json(
        root / "evaluation" / "results" / "model_selection" / "summary.json",
        {
            "split": "dev",
            "models": ["model-a:4b", "model-b:8b"],
            "results": {
                m: {
                    "n_valid": 10,
                    "routing": {
                        "accuracy_strict": {"rate": acc, "low": acc - 0.1, "high": acc + 0.1},
                        "accuracy_lenient": {"rate": acc + 0.05},
                        "macro_f1": acc - 0.02,
                    },
                    "roles": {"context_gate": role, "response_generator": role},
                    "latency_turn_p50_ms": lat,
                    "latency_turn_p95_ms": lat * 2,
                    "ollama": {
                        "parameter_size": size,
                        "quantization_level": "Q4_K_M",
                        "size_bytes": 3.3e9,
                    },
                }
                for m, acc, lat, size in (
                    ("model-a:4b", 0.6, 9000.0, "4B"),
                    ("model-b:8b", 0.7, 15000.0, "8B"),
                )
            },
        },
    )
    _write_json(
        root / "reports" / "performance" / "before_test.json",
        {
            "label": "before",
            "cold": [{"scenario": "S1", "turn": 1, "latency_s": 0.03}],
            "warm_summary": {"S1-t1:ESCALATE": {"n": 3, "p50": 0.02, "p95": 0.03}},
        },
    )
    (root / "medbridge").mkdir()
    (root / "medbridge" / "config.py").write_text(
        "class Settings:\n"
        '    LLM_MODEL: str = "model-a:4b"\n'
        "    RETRIEVAL_TOP_K_CANDIDATES: int = 20\n"
        "    RETRIEVAL_TOP_K_RERANKED: int = 5\n"
        "    SOFT_ASK_MAX_COUNT: int = 2\n",
        encoding="utf-8",
    )
    # mock-mode benchmark reports must never be shown as results
    _write_json(
        root / "reports" / "medbridge_aq_benchmark_report.json",
        {"mode": "mock", "metrics": {"mode": "mock", "overall_routing_accuracy": 100.0}},
    )
    _write_json(
        root / "reports" / "rabbits_benchmark_report.json",
        {"mode": "mock", "action_consistency_pct": 100.0},
    )
    return root


def _add_study(root: Path) -> dict:
    summary = _synthetic_summary()
    _write_json(root / "evaluation" / "results" / "summary.json", summary)
    for name in (
        "hallucination_rate_by_condition",
        "unsafe_answer_rate_unanswerable",
        "ablation_waterfall",
    ):
        _write_png(root / "reports" / "figures" / f"{name}.png")
    return summary


def _build(root: Path, tmp_path: Path):
    out = tmp_path / "out" / "deck.pptx"
    out_path, log = bp.build(out, bp.DEFAULT_CONTENT, root)
    return pptx.Presentation(str(out_path)), log


def _texts(slide) -> str:
    parts = []
    for shape in slide.shapes:
        if shape.has_text_frame:
            parts.append(shape.text_frame.text)
        if getattr(shape, "has_table", False) and shape.has_table:
            for row in shape.table.rows:
                parts.extend(cell.text for cell in row.cells)
    return "\n".join(parts)


def _pictures(slide) -> int:
    return sum(1 for s in slide.shapes if s.shape_type == 13)  # MSO_SHAPE_TYPE.PICTURE


def _slide(prs, log, slide_id: str):
    index = next(s["index"] for s in log["slides"] if s["id"] == slide_id)
    return prs.slides[index - 1]


# ----------------------------------------------------------------------
# Deck without study results
# ----------------------------------------------------------------------


def test_deck_structure_notes_and_placeholders(fixture_root: Path, tmp_path: Path):
    prs, log = _build(fixture_root, tmp_path)
    assert len(prs.slides) == N_SLIDES == log["slide_count"]
    assert round(prs.slide_width / prs.slide_height, 3) == round(16 / 9, 3)
    for slide in prs.slides:
        notes = slide.notes_slide.notes_text_frame.text.strip()
        assert len(notes) > 200, "every slide needs real speaker notes"
    title = _texts(prs.slides[0])
    assert "[TO BE FILLED: university]" in title
    assert "Vijay Aravindh S. (23BAI1160)" in title and "Dr. Maheswari S." in title
    assert "CardioGuard" in title and "MedBridge" in title
    assert log["layout_issues"] == [], log["layout_issues"]
    assert log["citation_keys_missing"] == []


def test_results_slides_are_pending_without_study(fixture_root: Path, tmp_path: Path):
    prs, log = _build(fixture_root, tmp_path)
    assert {"results_hallucination", "results_unsafe"} <= set(log["pending_slides"])
    for sid in ("results_hallucination", "results_unsafe"):
        slide = _slide(prs, log, sid)
        text = _texts(slide)
        assert "pending" in text.lower()
        assert "[PENDING" in text
        assert _pictures(slide) == 0
        assert "%" not in text.replace("95 %", "").replace("95\u00a0%", ""), "no rate may be quoted"
    # the emergency slide's independent estimate is pending, not invented
    assert "[PENDING: hallucination study]" in _texts(_slide(prs, log, "emergency"))


def test_real_numbers_come_from_results_files(fixture_root: Path, tmp_path: Path):
    prs, log = _build(fixture_root, tmp_path)
    retrieval = _texts(_slide(prs, log, "retrieval"))
    assert "0.642 [0.592, 0.692]" in retrieval  # hybrid MRR from the fixture metrics.json
    assert "1,234" in retrieval and "Configuration (12 queries)" in retrieval
    models = _texts(_slide(prs, log, "models"))
    assert "model-a:4b" in models and "model-b:8b" in models
    perf = _texts(_slide(prs, log, "performance"))
    assert "0.02" in perf and "[PENDING: after-optimisation" in perf
    assert any("model_selection/summary.json" in s for s in _slide_record(log, "models")["sources"])


def _slide_record(log: dict, slide_id: str) -> dict:
    return next(s for s in log["slides"] if s["id"] == slide_id)


def test_mock_mode_benchmark_reports_are_ignored(fixture_root: Path, tmp_path: Path):
    prs, log = _build(fixture_root, tmp_path)
    assert {"aq", "rabbits"} <= set(log["results_skipped"])
    text = _texts(_slide(prs, log, "robustness"))
    assert text.count("[PENDING: live run") == 3
    assert "100.0" not in text


def test_screenshots_replace_placeholders(fixture_root: Path, tmp_path: Path):
    prs, log = _build(fixture_root, tmp_path)
    assert _pictures(_slide(prs, log, "ui")) == 0
    assert "[PENDING: screenshot answer_citations.png]" in _texts(_slide(prs, log, "ui"))
    # the capture script's file names are recognised too
    _write_png(fixture_root / "report" / "screenshots" / "answer_citations.png", (1440, 900))
    _write_png(
        fixture_root / "report" / "screenshots" / "why-expanded-desktop-light.png", (1440, 900)
    )
    prs, log = _build(fixture_root, tmp_path)
    ui = _slide(prs, log, "ui")
    assert _pictures(ui) == 2
    assert _slide_record(log, "ui")["state"] == "partial"


# ----------------------------------------------------------------------
# Deck with study results
# ----------------------------------------------------------------------


def test_results_slides_fill_from_summary(fixture_root: Path, tmp_path: Path):
    summary = _add_study(fixture_root)
    prs, log = _build(fixture_root, tmp_path)
    assert "results_hallucination" not in log["pending_slides"]
    halluc = _slide(prs, log, "results_hallucination")
    text = _texts(halluc)
    c5 = summary["per_condition"]["C5"]["llm"]["hallucination"]
    assert tables.pct(c5) in text
    assert "Hallucination rate by condition" in text and "pending" not in text.lower()
    assert _pictures(halluc) == 1
    unsafe = _slide(prs, log, "results_unsafe")
    utext = _texts(unsafe)
    assert _pictures(unsafe) == 2
    assert "C5 vs C1" in utext and "C5 vs C2" in utext
    t = next(
        m
        for m in summary["tests"]["mcnemar"]
        if m["control"] == "C1" and m["outcome"] == "hallucination" and m["judge"] == "llm"
    )
    assert tables.pvalue(t["pvalue_holm"]) in utext
    assert "Decision rule:" in utext
    notes = halluc.notes_slide.notes_text_frame.text
    assert tables.pct(c5) in notes
    assert log["layout_issues"] == [], log["layout_issues"]


def test_verdict_is_mechanical(fixture_root: Path, tmp_path: Path):
    builder = bp.DeckBuilder(bp.load_content(bp.DEFAULT_CONTENT), fixture_root)
    assert builder._significant(None) is None
    assert (
        builder._significant({"pvalue_holm": 0.01, "risk_difference": {"estimate": -0.2}}) is True
    )
    assert (
        builder._significant({"pvalue_holm": 0.01, "risk_difference": {"estimate": 0.2}}) is False
    )
    assert (
        builder._significant({"pvalue_holm": 0.2, "risk_difference": {"estimate": -0.2}}) is False
    )


# ----------------------------------------------------------------------
# Content rendering and citations
# ----------------------------------------------------------------------


def test_citations_values_and_no_break_spaces(fixture_root: Path):
    builder = bp.DeckBuilder(bp.load_content(bp.DEFAULT_CONTENT), fixture_root)
    builder.cur = bp.SlideRecord(1, "x")
    out = builder.r("Burden [@who2023globalreport; @gallifant2024rabbits] and [@no_such_key]")
    assert "(WHO 2023; Gallifant et al. 2024)" in out
    assert "[CITATION NOT FOUND: no_such_key]" in out
    assert builder.cur.citations_missing == ["no_such_key"]
    assert builder.r("{{corpus.totals.chunks|comma}} chunks, 5.8 %") == "1,234 chunks, 5.8\u00a0%"
    assert (
        builder.r("{{study.per_condition.C5.llm.hallucination|ci}}")
        == "[PENDING: hallucination study]"
    )
    assert "study.per_condition.C5.llm.hallucination" in builder.cur.pending


def test_short_labels():
    assert bp.short_label({"authors": ["World Health Organization"], "year": 2023}) == "WHO 2023"
    assert bp.short_label(
        {"authors": ["NCD Risk Factor Collaboration (NCD-RisC)"], "year": 2021}
    ) == ("NCD-RisC 2021")
    assert bp.short_label({"authors": ["Krzysztof Wołk"], "year": 2025}) == "Wołk 2025"
    assert bp.short_label({"authors": ["A One", "B Two"], "year": 2020}) == "One & Two 2020"
    assert (
        bp.short_label({"authors": ["A One", "B Two", "C Three"], "year": 2020})
        == "One et al. 2020"
    )


def _strings(node: object) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [s for v in node.values() for s in _strings(v)]
    if isinstance(node, list):
        return [s for v in node for s in _strings(v)]
    return []


def test_content_citations_all_resolve():
    content = "\n".join(_strings(bp.load_content(bp.DEFAULT_CONTENT)))
    refs = bp.refs_lib.load_references(REPO_ROOT)
    keys = {k for body in bp.CITE_RE.findall(content) for k in bp.refs_lib.parse_cite_keys(body)}
    assert keys, "the deck should cite the literature it quotes"
    assert keys <= set(refs), sorted(keys - set(refs))


# ----------------------------------------------------------------------
# Drawing kit
# ----------------------------------------------------------------------


def test_text_fit_shrinks_then_reports_overflow():
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    kit = K.Kit()
    kit.begin("t")
    fit = kit.text(slide, 0, 0, 4, 1, "short words that fit easily", size=18, min_size=10)
    assert not fit.overflow and fit.size == 18
    long = " ".join(["overflowing"] * 120)
    fit = kit.text(slide, 0, 1.2, 4, 0.6, long, size=18, min_size=10, name="long")
    assert fit.overflow and fit.size == 10
    assert any(i.element == "long" and i.kind == "overflow" for i in kit.issues)


def test_table_rebalances_columns_for_long_words():
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    kit = K.Kit()
    kit.begin("t")
    fit = kit.table(
        slide,
        0,
        0,
        6,
        ["Measurement", "Notes"],
        [["x", "a b c"]],
        widths=[0.5, 5.5],
        size=14,
        min_size=10,
    )
    assert not kit.issues
    assert fit.shape.table.columns[0].width > K.emu(0.5)


def test_overlap_and_off_slide_detection():
    kit = K.Kit()
    kit.begin("t")
    kit.claim("a", 1, 1, 3, 2)
    kit.claim("b", 2, 2, 3, 2)
    kit.claim("c", 12, 7, 3, 2)
    kinds = {(i.element, i.kind) for i in kit.issues}
    assert ("b", "overlap") in kinds and ("c", "off-slide") in kinds


def test_line_counting_keeps_no_break_space_together():
    m = K.Measurer()
    runs = K.parse_runs("value 5.8\u00a0%")
    assert m.longest_token(runs, 12) == pytest.approx(m.width("5.8\u00a0%", 12))
    assert m.count_lines(K.parse_runs("a " * 200), 12, 100) > m.count_lines(
        K.parse_runs("a"), 12, 100
    )


def test_placeholder_runs_are_highlighted():
    runs = K.parse_runs("Uni: [TO BE FILLED: university] and **bold**")
    assert [r.highlight for r in runs] == [False, True, False, False]
    assert runs[-1].bold


# ----------------------------------------------------------------------
# Deck facts and the real repository
# ----------------------------------------------------------------------


def test_deck_fact_parsers(tmp_path: Path):
    (tmp_path / "docs" / "notes").mkdir(parents=True)
    (tmp_path / "docs" / "notes" / "code_review_findings.md").write_text(
        "# Review: 3 findings, all fixed in abc1234\n\n"
        "- **x-1** (high, 3/3): a\n- **x-2** (low, 3/3): b\n- **x-3** (medium, 3/3): c\n\n"
        "Rejected (1):\n- y-1: d\n",
        encoding="utf-8",
    )
    cr = deck_data.code_review(tmp_path)
    assert cr == {
        "confirmed": 3,
        "high": 1,
        "medium": 1,
        "low": 1,
        "rejected": 1,
        "commit": "abc1234",
        "source": "docs/notes/code_review_findings.md",
    }
    (tmp_path / "reports" / "baseline").mkdir(parents=True)
    (tmp_path / "reports" / "baseline" / "pytest_baseline.txt").write_text(
        "TOTAL   100   4   96%\n===== 2 failed, 40 passed, 1 skipped in 9.0s =====\n",
        encoding="utf-8",
    )
    base = deck_data.pytest_baseline(tmp_path)
    assert base["passed"] == 40 and base["failed"] == 2 and base["skipped"] == 1
    assert base["coverage_pct"] == 96
    (tmp_path / "medbridge").mkdir()
    (tmp_path / "medbridge" / "config.py").write_text(
        'class S:\n    LLM_MODEL: str = "m:4b"\n    SOFT_ASK_MAX_COUNT: int = 2  # loop\n'
        "    LOCAL_ONLY_MODE: bool = True\n",
        encoding="utf-8",
    )
    cfg = deck_data.config_defaults(tmp_path)
    assert (
        cfg["LLM_MODEL"] == "m:4b"
        and cfg["SOFT_ASK_MAX_COUNT"] == 2
        and cfg["LOCAL_ONLY_MODE"] is True
    )


def test_emergency_facts_on_real_case_files():
    stats = deck_data.emergency_case_stats(REPO_ROOT)
    if stats is None:
        pytest.skip("emergency classifier or case files not available")
    for c in stats.values():
        assert c["tp"] + c["fn"] == c["positives"] and c["fp"] + c["tn"] == c["negatives"]
        assert c["cases"] == c["positives"] + c["negatives"]
    examples = deck_data.classify_examples(
        REPO_ROOT,
        [{"text": "my chest feels really tight and my left arm went numb", "expected": "ESCALATE"}],
    )
    assert examples[0]["actual"] == "ESCALATE"


def test_real_repository_deck_builds_cleanly(tmp_path: Path):
    out = tmp_path / "deck.pptx"
    log_path = tmp_path / "log.json"
    code = bp.main(["--out", str(out), "--log", str(log_path), "--strict", "--quiet"])
    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert code == 0, log["layout_issues"] or log["citation_keys_missing"]
    assert log["slide_count"] == N_SLIDES
    assert log["emergency_example_mismatches"] == []
    prs = pptx.Presentation(str(out))
    assert len(prs.slides) == N_SLIDES
    for key in ("aq", "rabbits", "adversarial"):
        assert key not in log["results_sources"] or "live" in json.dumps(
            log["results_sources"][key]
        )


@pytest.mark.slow
def test_pdf_export_with_libreoffice(fixture_root: Path, tmp_path: Path):
    if bp.find_soffice() is None:
        pytest.skip("LibreOffice not installed")
    out = tmp_path / "out" / "deck.pptx"
    bp.build(out, bp.DEFAULT_CONTENT, fixture_root)
    pdf, detail = bp.convert_to_pdf(out, tmp_path / "out" / "deck.pdf", timeout=240)
    assert pdf is not None, detail
    pages = bp.pdf_page_count(pdf)
    assert pages is None or pages == N_SLIDES
