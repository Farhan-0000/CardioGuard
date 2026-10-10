"""
Tests for the IEEE paper builders (report/paper/build_paper.py, build_latex.py,
paper_common.py) and the paper figure script.

A small paper is built from a fixture content directory and a fixture repository root in
``tmp_path`` (the verified reference lists are copied in; every other data file is
synthetic test data written to ``tmp_path``, never to the repository). The DOCX is read
back with python-docx; the LaTeX output is checked for structure, escaping and ASCII.
The repository's own paper content is checked for unresolved citations/cross-references
without writing anything into the repository.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

docx = pytest.importorskip("docx")

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "report" / "paper"
for path in (PAPER_DIR, REPO_ROOT / "report"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_latex  # noqa: E402
import build_paper  # noqa: E402
import paper_common as pc  # noqa: E402
from lib import tables as report_tables  # noqa: E402

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

TITLE = """# A Test Paper on Gates

| Author | Affiliation | Email |
|---|---|---|
| Alice Example | Dept. of Testing; Test University; Chennai, India | alice@example.org |
| Bob Example | [TO BE FILLED: department]; Test University | [TO BE FILLED: email] |
| Carol Example | Dept. of Testing; Test University | carol@example.org |
"""

ABSTRACT = """#* Abstract

This abstract mentions {{hallubench.n_items}} items and nothing else.

#* Index Terms

testing, gates
"""

INTRO = """# Introduction {#sec:intro}

Retrieval helps [@lewis2020rag; @ji2023survey] and a patent exists [@paul2025glasshealthrag].
An unknown key [@no_such_key]. See [@fig:pipe], [@tab:demo], [@alg:route] and [@sec:method];
a dangling reference [@fig:nope]. The corpus has {{corpus.totals.documents|int}} documents and
a missing value {{retrieval.modes.hybrid.mrr|cif}}. Specials: 50% & more_text #1 {braces}
~tilde^ $5 a\\b. Unicode: ≥ 180 mmHg, κ, τ, café, ✓, “quoted” and "straight".
[TO BE FILLED: a placeholder]

- first item with **bold**, *italic* and `code/path_name.py`
  - nested item
- second item

1. one
2. two
"""

METHOD = """# Method {#sec:method}

## Overview {#sec:overview}

[[figure:reports/figures/test_fig.png|A column figure {#fig:pipe}|width=3.3]]

[[figure:reports/figures/test_fig.png|A wide figure {#fig:wide}|width=7.0]]

[[figure:reports/figures/missing.png|A missing figure]]

Table: A demo table {#tab:demo}

| Condition | Value | Notes |
|---|:-:|---|
| C0 | ✓ | text with [@cormack2009rrf] |
| C1 | – | 10% |

[[table:hallubench_composition]]

[[table:mcnemar|Paired tests {#tab:mcnemar}]]

```algorithm
Routing of one turn {#alg:route}
1  if a >= b: return X      # comment {braces}
2  a <- b ≥ c
```

### A third level

Text after the algorithm.

```
plain code block
```
"""

RESULTS = """# Results {#sec:results}

[[include:results.md]]
"""

ACK = """#* Acknowledgment

Thanks to everyone.
"""


def _write_png(path: Path, size: tuple[int, int] = (600, 400)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (200, 220, 240)).save(path)


@pytest.fixture()
def fixture_paper(tmp_path: Path) -> tuple[Path, Path]:
    """(content dir, repo root) with synthetic data and the real verified reference lists."""
    root = tmp_path / "repo"
    refs = root / "report" / "references"
    refs.mkdir(parents=True)
    for name in ("references.json", "patents.json"):
        shutil.copy(REPO_ROOT / "report" / "references" / name, refs / name)
    (root / "reports").mkdir()
    (root / "reports" / "corpus_stats.json").write_text(
        json.dumps({"totals": {"documents": 27, "chunks": 2747}}), encoding="utf-8"
    )
    items = []
    for i, cat in enumerate(["answerable_complete", "emergency", "out_of_domain"] * 4):
        turns = [{"text": "q", "gold_action": "ANSWER"}]
        if i == 0:
            turns.append({"text": "q2", "gold_action": "ANSWER"})
        items.append(
            {"id": f"T-{i}", "category": cat, "split": "dev" if i < 3 else "test", "turns": turns}
        )
    ds = root / "evaluation" / "datasets"
    ds.mkdir(parents=True)
    (ds / "hallubench_v1.json").write_text(
        json.dumps({"name": "fixture", "version": "0", "items": items}), encoding="utf-8"
    )
    _write_png(root / "reports" / "figures" / "test_fig.png")
    content = tmp_path / "content"
    content.mkdir()
    for name, text in (
        ("00_title.md", TITLE),
        ("01_abstract.md", ABSTRACT),
        ("10_intro.md", INTRO),
        ("20_method.md", METHOD),
        ("50_results.md", RESULTS),
        ("90_ack.md", ACK),
        ("notes_include_only.md", "# Not a section\n\nNever rendered."),
    ):
        (content / name).write_text(text, encoding="utf-8")
    return content, root


def _texts(document) -> list[str]:
    out = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                out += [p.text for p in cell.paragraphs]
    return out


def _highlighted(document) -> list[str]:
    runs = [r for p in document.paragraphs for r in p.runs]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                runs += [r for p in cell.paragraphs for r in p.runs]
    return [r.text for r in runs if r.font.highlight_color is not None]


def _cols(section) -> int:
    cols = section._sectPr.find(f"{W}cols")
    return int(cols.get(f"{W}num", "1")) if cols is not None else 1


# ----------------------------------------------------------------------
# Shared loader
# ----------------------------------------------------------------------


def test_helpers():
    assert [pc.to_roman(n) for n in (1, 4, 9, 14, 40)] == ["I", "IV", "IX", "XIV", "XL"]
    assert pc.split_label("Caption {#fig:x .wide}") == ("Caption", "fig:x", {"wide"})
    assert pc.split_label("No label") == ("No label", None, set())
    assert pc.is_xref("fig:a") and pc.is_xref("sec:b") and not pc.is_xref("lewis2020rag")


def test_load_paper_numbering_and_citations(fixture_paper):
    content, root = fixture_paper
    paper = pc.load_paper(content, root)
    assert paper.title == "A Test Paper on Gates"
    assert [a.name for a in paper.authors] == ["Alice Example", "Bob Example", "Carol Example"]
    assert paper.authors[0].affiliation == ["Dept. of Testing", "Test University", "Chennai, India"]
    assert paper.index_terms == "testing, gates"
    assert "notes_include_only.md" not in paper.files
    assert paper.xref("sec:intro") == "Section I"
    assert paper.xref("sec:method") == "Section II"
    assert paper.xref("sec:overview") == "Section II-A"
    assert paper.xref("fig:pipe") == "Fig. 1"
    assert paper.xref("fig:wide") == "Fig. 2"
    assert paper.xref("tab:demo") == "Table I"
    assert (
        paper.xref("tab:hallubench_composition") == "Table II"
    )  # default label of a generated table
    assert paper.xref("tab:mcnemar") == "Table III"  # pending generated table is numbered too
    assert paper.xref("alg:route") == "Algorithm 1"
    assert paper.labels["fig:wide"].wide and not paper.labels["fig:pipe"].wide
    # IEEE order of first citation; xrefs are not citations; unknown keys are reported
    assert paper.registry.order[:3] == ["lewis2020rag", "ji2023survey", "paul2025glasshealthrag"]
    assert "cormack2009rrf" in paper.registry.order
    assert paper.registry.missing == ["no_such_key"]
    assert paper.unresolved_xrefs == ["fig:nope"]
    assert any("results.md" in w for w in paper.warnings)


def test_anonymous_mode_drops_names_and_acknowledgment(fixture_paper):
    content, root = fixture_paper
    paper = pc.load_paper(content, root, anonymous=True)
    assert [a.name for a in paper.authors] == ["Anonymous Author(s)"]
    headings = [b.text for b in paper.blocks if getattr(b, "level", 0) == 1]
    assert "Acknowledgment" not in headings and "Results" in headings


def test_table_width_detection():
    narrow = pc.Table(caption="x", header=["A", "B"], rows=[["1", "2"]])
    wide = pc.Table(caption="x", header=[f"C{i}" for i in range(8)], rows=[["1"] * 8])
    assert not pc.table_is_wide(narrow)
    assert pc.table_is_wide(wide)
    assert pc.table_is_wide(narrow, {"wide"})


# ----------------------------------------------------------------------
# DOCX
# ----------------------------------------------------------------------


def test_docx_layout_and_content(fixture_paper, tmp_path):
    content, root = fixture_paper
    out = tmp_path / "paper.docx"
    result = build_paper.build(content, out, repo_root=root, log_path=tmp_path / "log.json")
    assert out.is_file()
    document = docx.Document(str(out))

    # sections: title (1 col) -> body (2 cols) -> wide figure (1) -> body (2) -> closing (1)
    cols = [_cols(s) for s in document.sections]
    assert cols[0] == 1 and cols[1] == 2 and cols[-1] == 1
    assert 1 in cols[2:-1] and cols.count(2) >= 2
    sect = document.sections[1]
    assert abs(sect.page_width.inches - 8.5) < 0.01
    assert abs(sect.left_margin.inches - 0.625) < 0.01
    space = sect._sectPr.find(f"{W}cols").get(f"{W}space")
    assert space == "360"  # 0.25 in between the columns

    normal = document.styles["Normal"]
    assert normal.font.name == "Times New Roman" and normal.font.size.pt == 10
    assert document.styles["IEEE Title"].font.size.pt == 24
    assert document.styles["Heading 1"].font.small_caps

    texts = _texts(document)
    joined = "\n".join(texts)
    assert document.paragraphs[0].text == "A Test Paper on Gates"
    assert "Alice Example" in joined and "alice@example.org" in joined
    assert "I. Introduction" in texts and "II. Method" in texts and "III. Results" in texts
    assert "A. Overview" in texts and "1) A third level" in texts
    assert "Acknowledgment" in texts and "References" in texts
    abstract = next(t for t in texts if t.startswith("Abstract"))
    assert abstract.startswith("Abstract—This abstract mentions 12 items")
    assert any(t.startswith("Index Terms—testing") for t in texts)

    # captions and cross-references
    assert any(t.startswith("Fig. 1. A column figure") for t in texts)
    assert any(t.startswith("Fig. 2. A wide figure") for t in texts)
    assert "TABLE I" in texts and "TABLE II" in texts and "TABLE III" in texts
    assert any(t.startswith("Algorithm 1: Routing of one turn") for t in texts)
    intro = next(t for t in texts if t.startswith("Retrieval helps"))
    assert "Retrieval helps [1], [2]" in intro and "a patent exists [3]" in intro
    assert "See Fig. 1, Table I, Algorithm 1 and Section II" in intro
    assert "has 27 documents" in intro
    assert "[CITATION NOT FOUND: no_such_key]" in intro
    assert "[REF NOT FOUND: fig:nope]" in intro

    # references in order of first citation, IEEE format
    refs = [t for t in texts if re.match(r"^\[\d+\]\t", t)]
    assert len(refs) == len(result.log["citation_keys"]) == 4
    assert refs[0].startswith("[1]\tP. Lewis et al.,")
    assert "U.S. Patent 12,243,646 B1" in refs[2]

    # highlighted placeholders and pending markers
    hl = _highlighted(document)
    assert "[TO BE FILLED: a placeholder]" in hl
    assert "[PENDING: retrieval.modes.hybrid.mrr]" in hl
    assert "[PENDING: results.md]" in hl
    assert "[FIGURE PENDING: reports/figures/missing.png]" in hl
    assert "[TABLE PENDING: mcnemar]" in hl
    assert "[CITATION NOT FOUND: no_such_key]" in hl

    # pictures: two real figures embedded
    blips = out.read_bytes().count(b"word/media/")
    assert blips >= 1
    assert len(document.inline_shapes) == 2

    log = json.loads((tmp_path / "log.json").read_text(encoding="utf-8"))
    assert log["citation_keys_missing"] == ["no_such_key"]
    assert log["unresolved_xrefs"] == ["fig:nope"]
    assert log["counts"] == {"section": 3, "figure": 3, "table": 3, "algorithm": 1}
    assert log["stats"]["figures"] == 2 and log["stats"]["figures_pending"] == 1
    assert log["stats"]["tables_pending"] == 1


def test_docx_generated_table_and_a4(fixture_paper, tmp_path):
    content, root = fixture_paper
    out = tmp_path / "a4.docx"
    build_paper.build(content, out, repo_root=root, paper_size="a4", log_path=None)
    document = docx.Document(str(out))
    assert abs(document.sections[0].page_width.inches - 8.27) < 0.01
    # the generated HalluBench table carries its fixture counts and a source note
    table = next(t for t in document.tables if t.rows[0].cells[0].text == "Category")
    last = [c.text for c in table.rows[-1].cells]
    assert last[:3] == ["Total", "12", "13"]
    texts = _texts(document)
    assert any(t.startswith("Source: evaluation/datasets/hallubench_v1.json") for t in texts)


def test_docx_cli_strict(fixture_paper, tmp_path):
    content, _ = fixture_paper
    code = build_paper.main(
        [
            "--content-dir",
            str(content),
            "--out",
            str(tmp_path / "x.docx"),
            "--no-log",
            "--quiet",
            "--strict",
        ]
    )
    assert code == 1  # unknown citation + unresolved cross-reference
    assert build_paper.main(["--content-dir", str(tmp_path / "missing"), "--quiet"]) == 2


# ----------------------------------------------------------------------
# LaTeX
# ----------------------------------------------------------------------


def test_latex_output(fixture_paper, tmp_path):
    content, root = fixture_paper
    out_dir = tmp_path / "latex"
    result = build_latex.build(content, out_dir, repo_root=root, log_path=None)
    tex = result.main_tex.read_text(encoding="utf-8")
    bib = result.bib.read_text(encoding="utf-8")

    assert r"\documentclass[conference]{IEEEtran}" in tex
    assert r"\title{A Test Paper on Gates}" in tex
    assert r"\IEEEauthorblockN{Alice Example}" in tex and r"\and" in tex
    assert r"\begin{abstract}" in tex and r"\begin{IEEEkeywords}" in tex
    assert r"\section{Introduction}\label{sec:intro}" in tex
    assert r"\subsection{Overview}\label{sec:overview}" in tex
    assert r"\subsubsection{A third level}" in tex
    assert r"\section*{Acknowledgment}" in tex
    assert r"\cite{lewis2020rag,ji2023survey}" in tex
    assert r"Fig.~\ref{fig:pipe}, Table~\ref{tab:demo}, Algorithm~\ref{alg:route}" in tex
    assert r"Section~\ref{sec:method}" in tex
    assert "no_such_key}" not in tex.split(r"\pending{[CITATION NOT FOUND")[0]
    assert r"\pending{[CITATION NOT FOUND: no\_such\_key]}" in tex
    assert r"\pending{[REF NOT FOUND: fig:nope]}" in tex
    assert "has 27 documents" in tex
    assert r"\pending{[PENDING: retrieval.modes.hybrid.mrr]}" in tex
    assert r"\pending{[TO BE FILLED: a placeholder]}" in tex
    # escaping and unicode
    assert r"50\% \& more\_text \#1 \{braces\}" in tex
    assert r"\textasciitilde{}tilde\textasciicircum{} \$5 a\textbackslash{}b" in tex
    assert r"$\geq$ 180 mmHg, $\kappa$, $\tau$, caf\'{e}, $\checkmark$" in tex
    assert "``quoted'' and ``straight''" in tex
    assert r"\texttt{code/\allowbreak{}path\_\allowbreak{}name.\allowbreak{}py}" in tex
    # lists
    assert tex.count(r"\begin{itemize}") == 2 and r"\begin{enumerate}" in tex
    # floats
    assert r"\begin{figure}[t]" in tex and r"\begin{figure*}[t]" in tex
    assert r"\includegraphics[width=0.94\columnwidth]{figures/test_fig.png}" in tex
    assert r"\includegraphics[width=\textwidth]{figures/test_fig.png}" in tex
    assert r"\pending{[FIGURE PENDING: reports/figures/missing.png]}" in tex
    assert r"\caption{A demo table}\label{tab:demo}" in tex
    assert r"\label{tab:hallubench_composition}" in tex
    assert r"\pending{[TABLE PENDING: mcnemar]}" in tex
    assert (
        r"\begin{algorithm}[t]" in tex and r"\caption{Routing of one turn}\label{alg:route}" in tex
    )
    assert "2  a <- b >= c" in tex  # listings content is mapped to ASCII
    assert r"\pending{[PENDING: results.md]}" in tex
    assert (out_dir / "figures" / "test_fig.png").is_file()
    # structure and ASCII
    assert tex.isascii() and bib.isascii()
    assert build_latex.validate_tex(tex) == []
    assert result.problems == []
    # bibliography: only known cited keys, IEEEtran-friendly fields
    assert "@inproceedings{lewis2020rag," in bib
    assert "@patent{paul2025glasshealthrag," in bib
    assert "Paul, Jr., Dereck William and Graham Ramsey" in bib
    assert "and others" in bib
    assert "no_such_key" not in bib
    keys = re.findall(r"^@\w+\{([^,]+),", bib, re.MULTILINE)
    assert keys == result.log["citation_keys"]


def test_latex_cli_strict_and_stale_figures(fixture_paper, tmp_path):
    content, _ = fixture_paper
    out_dir = tmp_path / "latex"
    (out_dir / "figures").mkdir(parents=True)
    stale = out_dir / "figures" / "old.png"
    stale.write_bytes(b"x")
    code = build_latex.main(
        [
            "--content-dir",
            str(content),
            "--out-dir",
            str(out_dir),
            "--no-log",
            "--quiet",
            "--strict",
        ]
    )
    assert code == 1
    assert not stale.exists()


def test_validate_tex_detects_problems():
    assert build_latex.validate_tex(r"\begin{a}x{y}\end{a}") == []
    assert build_latex.validate_tex(r"\begin{lstlisting}{ unbalanced\end{lstlisting}") == []
    assert build_latex.validate_tex(r"50\% {ok} % comment {") == []
    assert any("unclosed brace" in p for p in build_latex.validate_tex("{open"))
    assert any("unbalanced" in p for p in build_latex.validate_tex("close}"))
    assert any("does not match" in p for p in build_latex.validate_tex(r"\begin{a}\end{b}"))
    assert any("non-ASCII" in p for p in build_latex.validate_tex("café"))


def test_escaper():
    esc = build_latex.Escaper()
    assert esc.text("a_b & c%") == r"a\_b \& c\%"
    assert esc.text("üñ") == r"\"{u}\~{n}"
    assert esc.text("中") == "?" and "中" in esc.unmapped
    assert esc.ascii_code("x ≤ y → z") == "x <= y -> z"
    assert build_latex.url_arg("https://a.b/c%20d#e") == r"https://a.b/c\%20d\#e"


@pytest.mark.skipif(
    shutil.which("pdflatex") is None or shutil.which("bibtex") is None, reason="no TeX installation"
)
def test_latex_compiles(fixture_paper, tmp_path):  # pragma: no cover - needs TeX
    content, root = fixture_paper
    out_dir = tmp_path / "latex"
    build_latex.build(content, out_dir, repo_root=root, log_path=None)
    for cmd in (
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
    ):
        subprocess.run(cmd, cwd=out_dir, capture_output=True, timeout=180, check=False)
    assert (out_dir / "main.pdf").is_file()


# ----------------------------------------------------------------------
# The repository's own paper content (read-only checks)
# ----------------------------------------------------------------------


def test_repository_paper_content_resolves():
    paper = pc.load_paper(pc.DEFAULT_CONTENT, REPO_ROOT)
    assert paper.registry.missing == [], "paper cites keys missing from references.json"
    assert paper.unresolved_xrefs == []
    assert len(paper.registry.order) >= 20
    sections = [el.display for el in paper.elements.values() if el.kind == "section"]
    assert sections == ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
    assert paper.title.startswith("MedBridge:")
    for path in sorted(pc.DEFAULT_CONTENT.glob("*.md")):
        text = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.DOTALL)
        assert "$" not in text, f"{path.name}: TeX math is not supported by the markup"
        assert "CardioGuard" not in text, f"{path.name}: use the system name MedBridge"
        assert not re.search(r"/Users/|[A-Za-z]:\\\\", text), f"{path.name}: absolute path"


def test_results_template_uses_known_generators():
    text = (pc.DEFAULT_CONTENT / "results_template.md").read_text(encoding="utf-8")
    keys = re.findall(r"\[\[table:([a-z_]+)", text)
    assert keys and all(k in report_tables.GENERATORS for k in keys)
    assert not (pc.DEFAULT_CONTENT / "results_template.md").name.startswith(tuple("0123456789"))


def test_paper_figure_script(tmp_path):
    pytest.importorskip("matplotlib")
    sys.path.insert(0, str(PAPER_DIR / "figures"))
    try:
        import make_figures
    finally:
        sys.path.pop(0)
    paths = make_figures.gate_pipeline(tmp_path)
    assert [p.name for p in paths] == ["gate_pipeline.png", "gate_pipeline.svg"]
    assert all(p.stat().st_size > 1000 for p in paths)
