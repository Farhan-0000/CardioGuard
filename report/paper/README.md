# IEEE paper builders (MedBridge)

This folder holds the conference paper about MedBridge's Dual Answerability Gate, the
scripts that build it as **DOCX** and **LaTeX (IEEEtran)**, a draft **provisional patent
disclosure**, and the venue research.

Working title: *MedBridge: A Dual Answerability Gate for Hallucination-Resistant Local
Retrieval-Augmented Clinical Guidance*. Authors: Vijay Aravindh S., Mohammad Farhan Rashid,
Maheswari S. (affiliations and e-mails are `[TO BE FILLED]`).

| Path | What it is |
|---|---|
| `content/*.md` | the paper text, in the report markup (see below) |
| `content/results_template.md` | template for Section V, to copy to `content/results.md` after the study |
| `build_paper.py` | builds `MedBridge_IEEE_Paper.docx` (two-column IEEE layout) |
| `build_latex.py` | builds `latex/main.tex`, `latex/references.bib` and `latex/figures/` |
| `paper_common.py` | shared loader: content discovery, labels, numbering, citations |
| `figures/make_figures.py` | draws the single-column pipeline figure `figures/gate_pipeline.{png,svg}` |
| `latex/README.md` | how to compile the LaTeX version (Overleaf or local TeX) |
| `patent_disclosure_draft.md` | Indian provisional patent application outline (draft for the TTO / patent agent) |
| `VENUES.md` | verified venue recommendations and deadlines |
| `build/*.json` | build logs (pending markers, unknown citations, counts) |

## Quick start

```bash
# macOS / Linux, from the repository root
.venv-mac/bin/python report/paper/figures/make_figures.py   # only if the figure changed
.venv-mac/bin/python report/paper/build_paper.py            # -> report/paper/MedBridge_IEEE_Paper.docx
.venv-mac/bin/python report/paper/build_latex.py            # -> report/paper/latex/
.venv-mac/bin/python report/paper/build_paper.py --pdf      # + PDF preview (LibreOffice)

# Windows (PowerShell), from the repository root
.venv\Scripts\python report\paper\build_paper.py
.venv\Scripts\python report\paper\build_latex.py
```

| Option | Builder | Meaning |
|---|---|---|
| `--paper a4` | DOCX | A4 page geometry of the IEEE A4 template (default: US Letter) |
| `--anonymous` | both | double-blind review copy: authors replaced, Acknowledgment removed |
| `--pdf` | DOCX | also writes a PDF preview next to the DOCX (needs LibreOffice) |
| `--strict` | both | exit code 1 on unknown citation keys or unresolved cross-references (LaTeX: also on structural problems) |
| `--content-dir`, `--out`, `--out-dir`, `--log`, `--no-log`, `--quiet` | both | paths and logging |

Exit codes: 0 success, 1 strict check failed, 2 content directory missing, 3 (DOCX) `--pdf`
requested but LibreOffice produced no PDF.

## Numbers, citations and placeholders

The same rules as the report apply (`report/README.md`):

- Every number comes from a results or measurement file through `{{path|filter}}` values or
  `[[table:KEY]]` generated tables (`report/lib/results_context.py`, `report/lib/tables.py`).
  A missing file shows a highlighted `[PENDING: ...]` (DOCX: yellow; LaTeX: red bold).
  The two numbers typed in the text (the McNemar counts of the model-selection experiment)
  are copied from `evaluation/results/model_selection/DECISION.md` and were re-checked
  against the per-turn runs; the source is noted in a comment in
  `content/40_experimental_setup.md`.
- Citations (`[@key]`) resolve only against `report/references/references.json` and
  `patents.json`; unknown keys are listed in the build log and shown as a highlighted marker.
- `[TO BE FILLED: ...]` marks text the team must write (affiliations, e-mails, ethics
  statement, acknowledgment, and every sentence that depends on the study results).
- **Study results do not exist yet.** Section V includes `content/results.md`, which is
  written after `python -m evaluation.analyze`; until then the paper shows
  `[PENDING: results.md]`.

## Content files and markup

The paper uses the report markup (headings, lists, tables, `[[figure:...]]`,
`[[table:KEY]]`, `[[include:...]]`, algorithm fences, values, citations; see
`report/README.md`, "Markup"). File roles:

| File | Role |
|---|---|
| `00_title.md` | first `#` heading = title; first pipe table = authors (`Author`, `Affiliation` with lines separated by `;`, `Email`) |
| `01_abstract.md` | `#* Abstract` paragraph(s) and `#* Index Terms` |
| `NN_*.md` (two-digit prefix) | body, assembled in name order; `#` = section (I, II, ...), `##` = subsection (A, B, ...), `###` = 1), 2), ...; `#*` = unnumbered (Acknowledgment) |
| other `*.md` | include-only (`results.md`, `results_template.md`) |

Additions for the paper:

| Markup | Result |
|---|---|
| `... {#fig:name}` at the end of a figure caption (also `#tab:`, `#alg:`, `#sec:` on table captions, algorithm titles and headings) | label |
| `[@fig:name]`, `[@tab:name]`, `[@alg:name]`, `[@sec:name]` | "Fig. 1", "Table II", "Algorithm 1", "Section III" (LaTeX: `Fig.~\ref{fig:name}` ...) |
| `{#tab:name .wide}` | sets a table or figure across both columns |
| `[[figure:...\|...\|width=7.0]]` | figures wider than 3.6 in span both columns automatically |
| generated tables | label `tab:<KEY>` unless the caption sets one; tables that cannot fit a 3.5 in column span both columns |

TeX math (`$...$`) is not part of the markup: write symbols as Unicode (τ, κ, ≥); the LaTeX
builder maps them to commands.

## Writing Section V after the study

1. Run the study and `python -m evaluation.analyze` (writes `evaluation/results/summary.json`
   and `reports/figures/*`).
2. Copy `content/results_template.md` to `content/results.md`. Its generated tables and
   figures fill themselves; replace every `[TO BE FILLED ...]` with interpretation.
3. Fill the results-dependent sentences in the abstract, Section VI and Section VIII.
4. Rebuild both versions; check the build logs for `pending_markers` and
   `citation_keys_missing`.
5. Report non-significant results as such (see the decision rule in the project brief).

## Page budget

Without Section V the paper renders at about six pages in the LibreOffice preview (US
Letter), most venues allow six pages (ICITIIT 2027: 6, up to 8 with a fee; EMBC 2027: 4 + 1
page of references; see `VENUES.md`). Section V will add roughly one page. Ways to trim,
in order: keep only two results tables and one figure; drop Table I (the HalluBench
composition, which the text summarises); drop Algorithm 1 (Fig. 1 shows the same flow);
shorten Section II. Always check the final page count in Word or in the compiled LaTeX PDF,
not only in the LibreOffice preview.

## Notes on the DOCX layout

- Page geometry follows the IEEE conference template: US Letter, margins 0.75 in (top),
  1 in (bottom), 0.625 in (sides), two columns of 3.5 in with 0.25 in between; A4 uses the
  A4 template margins. Body: Times New Roman 10 pt. Title 24 pt, authors 11 pt, abstract
  9 pt bold, captions 8 pt, references 8 pt.
- The title block is a one-column section; the body is a continuous two-column section
  (`w:cols w:num="2"`). Full-width floats are placed between continuous section breaks, and
  a closing continuous break balances the columns of the last page.
- python-docx cannot float figures to the top of a column as LaTeX does; place them by
  moving the directive in the content if a column breaks awkwardly.
- Before submission, many IEEE venues require checking the PDF with IEEE PDF eXpress and
  adding the copyright notice they specify. [TO BE FILLED: venue-specific instructions.]

## Tests

```bash
.venv-mac/bin/python -m pytest tests/report/test_paper_builder.py -q
```

The tests build a small paper from fixture content in a temporary folder, read the DOCX
back (sections and columns, styles, numbering, captions, cross-references, IEEE references,
highlights) and check the LaTeX output (structure, escaping, ASCII only, floats,
bibliography). They also check that the repository's own paper content cites only verified
keys and has no unresolved cross-reference. A LaTeX compile test runs only where `pdflatex`
and `bibtex` are installed.

## Patent disclosure draft

`patent_disclosure_draft.md` outlines an Indian provisional application (Form 1/Form 2
structure, title, field, background with the patents in `report/references/patents.json`,
summary, detailed description, drawings list and draft claims). It is a draft for the team,
the institution's TTO and a registered patent agent. Read its Section 1 before submitting
the paper or making the repository public: a public disclosure before filing can destroy
novelty.
