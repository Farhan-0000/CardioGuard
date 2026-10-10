"""
Tests for the report builder (report/build_report.py + report/lib/**).

A small report is built from a fixture content directory and a fixture repository
root in ``tmp_path`` and read back with python-docx: headings and numbering,
captions and SEQ fields, TOC / LoF / LoT fields, section page numbering,
highlighted placeholders and pending markers, generated tables, IEEE references.

The hallucination-study tables are checked against a ``summary.json`` that is
produced with the real ``evaluation/analyze.py`` functions from synthetic turn
outcomes (test data only; it is written to ``tmp_path``, never to the repository).
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import pytest

docx = pytest.importorskip("docx")

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "report"
if str(REPORT_DIR) not in sys.path:
    sys.path.insert(0, str(REPORT_DIR))

import build_report  # noqa: E402
from lib import content as mk  # noqa: E402
from lib import references as refs  # noqa: E402
from lib import results_context as rc  # noqa: E402
from lib import tables  # noqa: E402

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _synthetic_summary() -> dict:
    """summary.json with the exact schema of evaluation/analyze.py (synthetic outcomes)."""
    analyze = pytest.importorskip("evaluation.analyze")
    from evaluation.io import sanitize_json
    from evaluation.outcomes import TurnOutcome

    rng = random.Random(7)
    categories = [
        ("answerable_complete", "ANSWER", ("ANSWER", "GENERALIZE")),
        ("unanswerable_in_domain", "ABSTAIN", ("ABSTAIN", "GENERALIZE")),
        ("out_of_domain", "ABSTAIN", ("ABSTAIN",)),
        ("emergency", "ESCALATE", ("ESCALATE",)),
    ]
    behaviour_for = {"ANSWER": "ANSWERED_SPECIFIC", "ABSTAIN": "ABSTAINED", "ESCALATE": "ESCALATED"}
    outcomes: dict[str, list] = {}
    for cond, p_bad in (
        ("C0", 0.5),
        ("C1", 0.35),
        ("C2", 0.3),
        ("C3", 0.25),
        ("C4", 0.2),
        ("C5", 0.1),
    ):
        rows = []
        for i in range(40):
            cat, gold, acceptable = categories[i % len(categories)]
            gated = cond in analyze.GATED
            action = gold if (gated and rng.random() > 0.2) else "ANSWER"
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
                    behaviour=behaviour_for.get(action, "GENERAL_INFO"),
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


def _write_png(path: Path, size: tuple[int, int] = (400, 200)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (200, 220, 240)).save(path)


CH1 = """# Introduction

Opening paragraph citing retrieval-augmented generation [@lewis2020rag] and a survey
[@ji2023survey; @lewis2020rag]. The corpus has {{corpus.totals.documents|int}} documents and
a missing value {{retrieval.modes.hybrid.mrr|cif}}. An unknown key [@no_such_key].

## Background

### Burden

- first bullet with **bold** and *italic* and `code`
  - nested bullet
- second bullet

1. step one
2. step two

Table: Hand-written table with a citation [@cormack2009rrf]
| Name | Value |
|---|---:|
| alpha | 1 |
| beta | 2 |

[[figure:figs/arch.png|Architecture overview|width=4.0]]

[[figure:figs/missing.png|A figure that does not exist yet]]

[[table:corpus]]

[[table:aq_benchmark]]

[[table:hallucination_by_condition]]

[[table:mcnemar]]

```algorithm
Algorithm: Dual gate routing
Input: message m
1  if EMERGENCY(m) then return ESCALATE
2  return ANSWER
```

\\pagebreak

## Objectives

[TO BE FILLED: objective owner]
"""

CH2 = """# Literature

## Prior work

[[figure:figs/arch.png|Second chapter figure]]

[[table:action_taken]]

[[include:results_later.md]]
"""

APPENDIX = """# Screenshots

[[figure:figs/arch.png|Welcome screen]]
"""


@pytest.fixture()
def fixture_repo(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    (root / "report" / "references").mkdir(parents=True)
    for name in ("references.json", "patents.json"):
        shutil.copy(REPORT_DIR / "references" / name, root / "report" / "references" / name)
    (root / "reports").mkdir()
    (root / "reports" / "corpus_stats.json").write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "short_title": "Doc A",
                        "org": "WHO",
                        "year": 2021,
                        "doc_type": "guideline",
                        "source_format": "pdf",
                        "pages": 10,
                        "chunks": 20,
                    },
                    {
                        "short_title": "Doc B",
                        "org": "NICE",
                        "year": 2019,
                        "doc_type": "guideline",
                        "source_format": "html",
                        "pages": 5,
                        "chunks": 7,
                    },
                ],
                "totals": {
                    "documents": 2,
                    "pages_or_sections": 15,
                    "chunks": 27,
                    "organisations": 2,
                },
            }
        )
    )
    (root / "evaluation" / "results").mkdir(parents=True)
    (root / "evaluation" / "results" / "summary.json").write_text(json.dumps(_synthetic_summary()))
    # a mock-mode benchmark report must NOT be used as a result
    (root / "reports" / "medbridge_aq_benchmark_report.json").write_text(
        json.dumps({"mode": "mock", "metrics": {"mode": "mock", "total_vignettes": 3}})
    )
    _write_png(root / "figs" / "arch.png")

    content = tmp_path / "content"
    content.mkdir()
    (content / "00_title.md").write_text(
        "**MY REPORT TITLE**\n\nSubmitted by\n\n**[TO BE FILLED: university name]**\n"
    )
    (content / "01_bonafide.md").write_text("#* Bonafide Certificate\n\nCertified text.\n")
    (content / "04_abstract.md").write_text(
        "#* Abstract\n\nAbstract text with {{corpus.totals.chunks|int}} chunks.\n\n"
        "[[include:abstract_results.md]]\n"
    )
    (content / "ch1_introduction.md").write_text(CH1)
    (content / "ch2_literature.md").write_text(CH2)
    (content / "appendix_a_screens.md").write_text(APPENDIX)
    (content / "action_taken.md").write_text(
        "Table: Action taken on previous review comments\n"
        "| Comment | Action Taken | Evidence |\n|---|---|---|\n"
        "| Mock benchmark | Live study [@lewis2020rag] | Section 4.5 |\n"
    )
    (content / "abstract_results.md").write_text("Results paragraph (included).\n")
    # a placeholder file for chapter 3 must be used only because no real ch3 file exists
    (content / "ch3_placeholder.md").write_text(
        "<!-- placeholder -->\n# Methodology\n\nPlaceholder text.\n"
    )
    return root, content


@pytest.fixture()
def built(fixture_repo: tuple[Path, Path], tmp_path: Path):
    root, content = fixture_repo
    out = tmp_path / "out" / "report.docx"
    log = tmp_path / "out" / "log.json"
    result = build_report.build(out, content, repo_root=root, log_path=log, quiet=True)
    return result, docx.Document(str(out)), out


def _paras(document, style: str | None = None) -> list:
    return [p for p in document.paragraphs if style is None or p.style.name == style]


def _xml(path: Path, member: str = "word/document.xml") -> str:
    import zipfile

    with zipfile.ZipFile(path) as zf:
        return zf.read(member).decode("utf-8")


# ----------------------------------------------------------------------
# Built document
# ----------------------------------------------------------------------


def test_headings_numbered_and_chapter_labels(built):
    _, document, _ = built
    h1 = [p.text for p in _paras(document, "Heading 1")]
    assert any(t.startswith("CHAPTER 1") and "INTRODUCTION" in t for t in h1)
    assert any(t.startswith("CHAPTER 2") and "LITERATURE" in t for t in h1)
    assert any(t.startswith("CHAPTER 3") and "METHODOLOGY" in t for t in h1)
    assert any(t.startswith("CHAPTER 5") for t in h1)  # placeholder chapters 4 and 5
    assert any(t.startswith("APPENDIX A") and "SCREENSHOTS" in t for t in h1)
    assert "REFERENCES" in h1
    assert {"ABSTRACT", "LIST OF FIGURES", "LIST OF TABLES", "LIST OF ABBREVIATIONS"} <= set(h1)
    h2 = [p.text for p in _paras(document, "Heading 2")]
    h3 = [p.text for p in _paras(document, "Heading 3")]
    assert "1.1 Background" in h2 and "1.2 Objectives" in h2 and "2.1 Prior work" in h2
    assert "1.1.1 Burden" in h3
    front = [p.text for p in _paras(document, "Front Title")]
    assert "BONAFIDE CERTIFICATE" in front and "TABLE OF CONTENTS" in front


def test_captions_numbered_per_chapter_with_seq_fields(built):
    result, document, out = built
    captions = [p.text for p in _paras(document, "Caption")]
    assert "Table 1.1: Hand-written table with a citation [3]" in captions
    assert "Figure 1.1: Architecture overview" in captions
    assert "Figure 1.2: A figure that does not exist yet" in captions
    assert "Figure 2.1: Second chapter figure" in captions
    assert "Figure A.1: Welcome screen" in captions
    assert any(c.startswith("Algorithm 1.1: Dual gate routing") for c in captions)
    assert any(
        c.startswith("Table 2.1: Action taken on previous review comments") for c in captions
    )
    xml = _xml(out)
    assert "SEQ Figure \\* ARABIC \\s 1 \\r 1" in xml  # first caption of a chapter
    assert "SEQ Figure \\* ARABIC \\s 1 " in xml
    assert "SEQ Table \\* ARABIC" in xml and "SEQ Algorithm \\* ARABIC" in xml


def test_index_fields_update_on_open_and_page_numbering(built):
    _, document, out = built
    xml = _xml(out)
    assert 'TOC \\o "1-3" \\h \\z \\u' in xml
    assert 'TOC \\h \\z \\c "Figure"' in xml
    assert 'TOC \\h \\z \\c "Table"' in xml
    assert 'w:updateFields w:val="true"' in _xml(out, "word/settings.xml")
    assert len(document.sections) == 2
    front, main = document.sections
    pg_front = front._sectPr.find(f"{W}pgNumType")
    pg_main = main._sectPr.find(f"{W}pgNumType")
    assert pg_front.get(f"{W}fmt") == "lowerRoman" and pg_front.get(f"{W}start") == "1"
    assert pg_main.get(f"{W}fmt") == "decimal" and pg_main.get(f"{W}start") == "1"
    assert front.different_first_page_header_footer  # no number on the title page
    for section in (front, main):
        assert not section.footer.is_linked_to_previous
        footer_xml = section.footer._element.xml
        assert "PAGE" in footer_xml
        assert abs(section.page_width.inches - 8.27) < 0.01
        assert abs(section.left_margin.inches - 1.5) < 0.01
        assert abs(section.right_margin.inches - 1.0) < 0.01


def test_highlights_placeholders_and_pending_markers(built):
    result, document, out = built
    highlighted = []
    for para in document.paragraphs:
        for run in para.runs:
            if run.font.highlight_color is not None:
                highlighted.append(run.text)
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    highlighted += [r.text for r in para.runs if r.font.highlight_color is not None]
    joined = " | ".join(highlighted)
    assert "[TO BE FILLED: university name]" in joined
    assert "[TO BE FILLED: objective owner]" in joined
    assert "[FIGURE PENDING: figs/missing.png]" in joined
    assert "[TABLE PENDING: aq_benchmark]" in joined  # mock AQ report is not a result
    assert "[PENDING: retrieval.modes.hybrid.mrr]" in joined
    assert "[PENDING: results_later.md]" in joined
    assert "[CITATION NOT FOUND: no_such_key]" in joined
    log = result.log
    assert "no_such_key" in log["citation_keys_missing"]
    assert "aq" in log["results_skipped"]
    assert log["front_matter_sources"]["title"] == "00_title.md"
    assert log["front_matter_sources"]["declaration"] == "built-in default"
    assert "ch3_placeholder.md" in log["chapters"]
    assert any(name.startswith("abstract_results.md") for name in log["include_only_files"])


def test_inline_values_and_includes(built):
    _, document, _ = built
    text = "\n".join(p.text for p in document.paragraphs)
    assert "The corpus has 2 documents" in text
    assert "Abstract text with 27 chunks." in text
    assert "Results paragraph (included)." in text
    assert "MY REPORT TITLE" in text


def test_references_ieee_in_order_of_first_citation(built):
    result, document, _ = built
    entries = [p.text for p in _paras(document, "Reference Entry")]
    assert len(entries) == 3  # lewis, ji, cormack (unknown key is not invented)
    assert entries[0].startswith('[1]\tP. Lewis et al., "Retrieval-Augmented Generation')
    assert entries[1].startswith("[2]\tZ. Ji et al.") and "doi: 10.1145/3571730" in entries[1]
    assert entries[2].startswith("[3]\tG. V. Cormack, C. L. A. Clarke, and S. Buettcher")
    body = "\n".join(p.text for p in document.paragraphs)
    assert "generation [1] and a survey [1], [2]." in body


def test_generated_tables_render_real_and_study_data(built):
    _, document, _ = built
    tables_text = []
    for table in document.tables:
        rows = [[c.text for c in r.cells] for r in table.rows]
        tables_text.append(rows)
    corpus = next(t for t in tables_text if t[0][:2] == ["#", "Document"])
    assert corpus[1][1] == "Doc A" and corpus[-1][1] == "Total" and corpus[-1][-1] == "27"
    halluc = next(t for t in tables_text if t[0][0] == "Condition" and "Turns" in t[0])
    labels = [r[0] for r in halluc[1:]]
    assert labels[0] == "C0 Direct LLM" and "C5 MedBridge (dual gate)" in labels
    assert all("[" in r[2] for r in halluc[1:])  # rate with Wilson CI
    mc = next(t for t in tables_text if t[0][0] == "Comparison")
    assert any(r[0] == "C5 vs C0" for r in mc[1:])
    action = next(
        t
        for t in tables_text
        if t[0] == ["Comparison", "Action Taken", "Evidence"]
        or t[0] == ["Comment", "Action Taken", "Evidence"]
    )
    assert "Live study [1]" in action[1][1]
    # header shading and repeated header row on generated tables
    first = document.tables[1]
    tc_pr = first.rows[0].cells[0]._tc.tcPr
    assert tc_pr.find(f"{W}shd").get(f"{W}fill") == "D9E2F3"
    assert first.rows[0]._tr.trPr.find(f"{W}tblHeader") is not None


def test_numbered_lists_restart_and_styles(built):
    _, document, _ = built
    numbered = _paras(document, "List Number")
    assert [p.text for p in numbered][:2] == ["step one", "step two"]
    num_ids = {p._p.pPr.find(f"{W}numPr").find(f"{W}numId").get(f"{W}val") for p in numbered}
    assert len(num_ids) == 1
    assert _paras(document, "List Bullet 2")[0].text == "nested bullet"
    normal = document.styles["Normal"]
    assert normal.font.name == "Times New Roman" and normal.font.size.pt == 12
    assert normal.paragraph_format.line_spacing == 1.5
    assert document.styles["Heading 1"].font.size.pt == 16
    assert document.styles["Heading 2"].font.size.pt == 14
    assert document.styles["Heading 3"].font.size.pt == 12


def test_auto_action_taken_page_and_explicit_appendix_letters(fixture_repo, tmp_path: Path):
    root, content = fixture_repo
    # action_taken.md no longer referenced -> placed after the list of abbreviations
    (content / "ch2_literature.md").write_text("# Chapter 2: Literature\n\nText.\n")
    (content / "appendix_a_screens.md").write_text(
        "#* Appendix C: Test Cases\n\n[[figure:figs/arch.png|Case figure]]\n"
    )
    out = tmp_path / "o2" / "r.docx"
    result = build_report.build(out, content, repo_root=root, quiet=True)
    document = docx.Document(str(out))
    h1 = [p.text for p in _paras(document, "Heading 1")]
    assert "ACTION TAKEN ON PREVIOUS REVIEW COMMENTS" in h1
    assert h1.index("LIST OF ABBREVIATIONS") < h1.index("ACTION TAKEN ON PREVIOUS REVIEW COMMENTS")
    assert any(t.startswith("CHAPTER 2") and t.endswith("LITERATURE") for t in h1)
    assert any(t.startswith("APPENDIX C") and t.endswith("TEST CASES") for t in h1)
    assert "Figure C.1: Case figure" in [p.text for p in _paras(document, "Caption")]
    assert "auto" in result.log["front_matter_sources"]["action_taken"]


# ----------------------------------------------------------------------
# Units
# ----------------------------------------------------------------------


def test_markup_parser_blocks(tmp_path: Path):
    ctx = mk.ParseContext(content_dir=tmp_path)
    (tmp_path / "a.md").write_text("[[include:b.md]]\n")
    (tmp_path / "b.md").write_text("[[include:a.md]]\nfrom b\n")
    text = (
        "#* Unnumbered\n\n## Sec\n\nPara one\ncontinues.  \nhard break\n\n"
        "Table: Cap with {{a|pct}}\n| x | y |\n|:-:|--:|\n| 1 | 2 |\n\n"
        "[[figure:f.png|Cap with {{v|f2}} value|width=3.5]]\n\n"
        "[[table:corpus|Custom caption]]\n\nTable: Above generated\n[[table:pii_metrics]]\n\n"
        "```algorithm\nAlgorithm 1: Title here\nline 1\n```\n\n> quoted\n\n"
        "Published in\n2019. Not a list.\n\n[[include:a.md]]\n"
    )
    blocks = mk.parse_markup(text, ctx, source="t.md")
    kinds = [type(b).__name__ for b in blocks]
    assert kinds[:3] == ["Heading", "Heading", "Paragraph"]
    assert blocks[0].numbered is False and blocks[1].level == 2
    assert blocks[2].text == "Para one continues.\nhard break"
    table = next(b for b in blocks if isinstance(b, mk.Table))
    assert table.caption == "Cap with {{a|pct}}" and table.align == ["c", "r"]
    fig = next(b for b in blocks if isinstance(b, mk.Figure))
    assert fig.caption == "Cap with {{v|f2}} value" and fig.width == 3.5
    gens = [b for b in blocks if isinstance(b, mk.GeneratedTable)]
    assert (gens[0].key, gens[0].caption) == ("corpus", "Custom caption")
    assert (gens[1].key, gens[1].caption) == ("pii_metrics", "Above generated")
    alg = next(b for b in blocks if isinstance(b, mk.Algorithm))
    assert alg.title == "Title here" and alg.lines == ["line 1"]
    assert any(isinstance(b, mk.Paragraph) and b.style == "quote" for b in blocks)
    assert any(
        isinstance(b, mk.Paragraph) and b.text == "Published in 2019. Not a list." for b in blocks
    )
    assert any("recursive include" in w for w in ctx.warnings)


def test_table_caption_options_set_widths_and_size(tmp_path: Path):
    ctx = mk.ParseContext(content_dir=tmp_path)
    text = (
        "Table: Weighted {{a|pct}} {widths=1, 3,2; size=9}\n| x | y | z |\n|---|---|---|\n"
        "| 1 | 2 | 3 |\n\n"
        "Table: Wrong count {widths=1,2}\n| x | y | z |\n|---|---|---|\n| 1 | 2 | 3 |\n\n"
        "Table: Bad values {widths=a,b; size=99}\n| x |\n|---|\n| 1 |\n\n"
        "Table: Plain braces {not options}\n| x |\n|---|\n| 1 |\n\n"
        "[[table:corpus|Generated {size=8}]]\n"
    )
    blocks = mk.parse_markup(text, ctx, source="t.md")
    t1, t2, t3, t4 = [b for b in blocks if isinstance(b, mk.Table)]
    assert t1.caption == "Weighted {{a|pct}}"
    assert t1.col_widths == [1.0, 3.0, 2.0] and t1.font_size == 9.0
    assert t2.caption == "Wrong count" and t2.col_widths is None  # must name every column
    assert t3.caption == "Bad values" and t3.col_widths is None and t3.font_size is None
    assert t4.caption == "Plain braces {not options}" and t4.col_widths is None
    gen = next(b for b in blocks if isinstance(b, mk.GeneratedTable))
    assert gen.caption == "Generated"  # options are never printed


def test_column_widths_respect_weights_and_capitals():
    from lib import docx_helpers as dh

    widths = dh.compute_column_widths(
        ["No.", "Comment", "Action"], [["1", "short", "long " * 20]], weights=[0.5, 1.5, 3.0]
    )
    assert abs(sum(widths) - dh.TEXT_WIDTH_IN) < 1e-6
    assert widths[0] < widths[1] < widths[2]
    caps = dh.compute_column_widths(["Action", "Text"], [["GENERALIZE", "x " * 200]])
    lower = dh.compute_column_widths(["Action", "Text"], [["generalize", "x " * 200]])
    assert caps[0] > lower[0]  # an all-caps label gets room so it does not split


def test_inline_tokenizer():
    toks = mk.tokenize(
        "A **bold** *it* `c*d` [@a; @b] {{x.y|pct}} [TO BE FILLED: n [x]] "
        "[link](https://e.org) 5 * 3 \\*lit\\*"
    )
    kinds = [(t.kind, t.text) for t in toks]
    assert ("text", "bold") in kinds and any(t.bold for t in toks if t.text == "bold")
    assert any(t.italic for t in toks if t.text == "it")
    assert ("code", "c*d") in kinds
    assert next(t for t in toks if t.kind == "cite").keys == ["a", "b"]
    assert ("value", "x.y|pct") in kinds
    assert ("fill", "[TO BE FILLED: n [x]]") in kinds
    assert next(t for t in toks if t.kind == "link").url == "https://e.org"
    assert "5 * 3 *lit*" in "".join(t.text for t in toks if t.kind == "text")


def test_reference_formatting_and_labels():
    assert refs.format_person("Daniel W. Jones") == "D. W. Jones"
    assert refs.format_person("Shi-Qi Yan") == "S.-Q. Yan"
    assert refs.format_person("World Health Organization") == "World Health Organization"
    assert refs.format_person("John P. Bufe, III") == "J. P. Bufe III"
    assert refs.format_authors(["A B", "C D"]) == "A. B and C. D"
    assert refs.format_authors(["A B", "C D", "E F"]) == "A. B, C. D, and E. F"
    assert refs.format_authors(["A B", "C D", "et al."]) == "A. B et al."
    reg = refs.CitationRegistry({k: {"key": k} for k in "abcdef"})
    reg.register(["c", "a", "b", "e"])
    assert reg.label(["a", "b", "c"])[0] == "[1]–[3]"
    assert reg.label(["c", "e", "zz"]) == ("[1], [4]", ["zz"])


def test_every_reference_formats():
    entries = refs.load_references(REPO_ROOT)
    assert len(entries) >= 90
    for entry in entries.values():
        text = refs.format_entry_text(entry)
        assert entry["title"].rstrip(".")[:30] in text
        assert str(entry["year"]) in text


def test_results_context_resolution_and_filters():
    data = {
        "results": {"qwen3.5:4b": {"routing": {"macro_f1": 0.6321}}},
        "modes": {"hybrid+rerank": {"recall@5": {"estimate": 0.7333, "low": 0.638, "high": 0.821}}},
        "p": {"k": 3, "n": 4, "rate": 0.75, "low": 0.3, "high": 0.95},
        "lst": [5, 6],
    }
    ctx = rc.ResultsContext(data)
    assert ctx.render("results.qwen3.5:4b.routing.macro_f1|f2") == ("0.63", True)
    assert ctx.render('modes["hybrid+rerank"].recall@5|cif') == ("0.733 [0.638, 0.821]", True)
    assert ctx.render("modes.hybrid+rerank.recall@5.estimate|pct") == ("73.3%", True)
    assert ctx.render("p|ci") == ("75.0% [30.0, 95.0]", True)
    assert ctx.render("p.rate|pct0") == ("75%", True)
    assert ctx.render("lst.1") == ("6", True)
    assert ctx.render("missing.path") == ("[PENDING: missing.path]", False)
    assert ctx.render("p.k|bogus")[1] is False
    assert rc.format_value(1234.4, ["comma"]) == "1,234"
    assert rc.format_value(9354.4, ["s"]) == "9.4"


def test_classify_content_files():
    assert build_report.classify("00_title") == ("front", "title")
    assert build_report.classify("01_bonafide") == ("front", "bonafide")
    assert build_report.classify("05_abbreviations") == ("front", "abbreviations")
    assert build_report.classify("abstract") == ("front", "abstract")
    assert build_report.classify("abstract_results")[0] == "include"
    assert build_report.classify("ch4_study_results") == ("chapter", (4, "ch4_study_results"))
    assert build_report.classify("appendix_b_tests")[0] == "appendix"
    assert build_report.classify("action_taken")[0] == "include"
    assert build_report.classify("references")[0] == "skip"


def test_real_context_tables_never_use_mock_reports():
    ctx = rc.ResultsContext.load(REPO_ROOT)
    for key in ("aq", "rabbits", "adversarial"):
        data = ctx.data.get(key)
        assert data is None or rc._mode_of(data) == "live"
    for key in tables.GENERATORS:
        errors: list[str] = []
        tables.generate(key, ctx, errors)
        assert not errors, errors


@pytest.mark.slow
def test_pdf_export_with_libreoffice(built, tmp_path: Path):
    from lib import libreoffice

    if libreoffice.find_soffice() is None:
        pytest.skip("LibreOffice not installed")
    _, _, out = built
    conv = libreoffice.convert_to_pdf(out, tmp_path / "out" / "report.pdf", timeout=240)
    assert conv.pdf is not None and conv.pdf.exists()
    pages = libreoffice.pdf_page_count(conv.pdf)
    assert pages is None or pages > 5
