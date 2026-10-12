# Report builder

`report/build_report.py` generates the final project report
(`report/MedBridge_Final_Report.docx`, and optionally a PDF) from the content files in
`report/content/` and the real results files of the repository, in the layout of the VIT
Chennai BCSE497J project report template (`report/templates/BCSE497J_report_template.docx`).
The report is titled *CardioGuard*; the system is called MedBridge throughout.

Nothing in the report is typed in by hand if a script measured it. Inline values and data
tables are read from `evaluation/results/` and `reports/` when the report is built. When a
results file does not exist yet, the report shows a yellow `[PENDING: ...]` or
`[TABLE PENDING: ...]` marker. Citations can only use keys from
`report/references/references.json` and `report/references/patents.json`.

## Quick start

```bash
# macOS / Linux (from the repository root)
.venv-mac/bin/python report/build_report.py          # DOCX only
.venv-mac/bin/python report/build_report.py --pdf    # DOCX + PDF (needs LibreOffice)
.venv-mac/bin/python report/build_report.py --all-members --pdf   # + one copy per student

# Windows (PowerShell, from the repository root)
.venv\Scripts\python report\build_report.py --all-members --pdf
```

The template asks **every student to submit their own copy**: the cover and title pages list
the whole team, while the declaration, the certificate and the acknowledgement carry only that
student's name. Fill in `report/team.json` once (programme, month and year, guide's
designation and area, Head of the Department; names and register numbers are already there),
then build with `--all-members`. You get the team copy `MedBridge_Final_Report.docx`, in which
those three pages keep a highlighted placeholder for the name, and one copy per student,
`MedBridge_Final_Report_<register number>.docx` (and `.pdf`). Any value in `team.json` still
reading `[TO BE FILLED: ...]` is highlighted in the document.

| Option | Meaning |
|---|---|
| `--out PATH` | output `.docx` (default `report/MedBridge_Final_Report.docx`) |
| `--pdf` | also write a PDF next to the DOCX with LibreOffice; TOC, List of Figures and List of Tables are refreshed first |
| `--content-dir DIR` | content directory (default `report/content`) |
| `--log PATH` / `--no-log` | JSON build log (default `report/build/report_build_log.json`) |
| `--strict` | exit code 1 if an unknown citation key is used |
| `--quiet` | no console summary |
| `--member REG_NO` | build only that student's copy (`..._REG_NO.docx`) |
| `--all-members` | build the team copy and one copy per student in `report/team.json` |
| `--team-file PATH` | another team file (default `report/team.json`) |

Exit codes: 0 = success, 1 = unknown citation keys with `--strict`, 2 = content directory
missing or a problem with the team file (for example an unknown `--member`), 3 = `--pdf`
requested but no PDF could be produced.

Requirements: `python-docx` (pinned in `requirements.txt`). `Pillow` is optional: it caps very
tall figures and comes installed with matplotlib. `PyMuPDF` reports the PDF page count.
LibreOffice is needed for `--pdf`. Install LibreOffice with
`brew install --cask libreoffice` (macOS), `winget install TheDocumentFoundation.LibreOffice`
(Windows) or `sudo apt install libreoffice-writer` (Linux). The builder finds `soffice` on
the `PATH` or in the usual install folders.

## What gets built

Assembly order, as the template prescribes:

1. Template pages without a page number, written from `report/team.json` by
   `report/lib/vit_front.py` with the template's own wording: cover page ("A project report
   on"), title page, declaration by the candidate and the certificate (guide, two examiners,
   Head of the Department, seal), with the VIT Chennai logo
   (`report/templates/vit_chennai_logo.png`). The template's "Certificate by External Guide"
   applies only to PAT-discontinued students and is not included.
2. Preliminary pages numbered i, ii, iii (italic, bottom centre): abstract, acknowledgement
   (the template text, worded without assuming the guide's pronouns), contents, list of
   tables, list of figures and list of symbols and abbreviations.
3. Chapters 1 to 5, the last being "Conclusion and Future Work". Page numbering restarts at 1
   with Chapter 1, and every chapter starts on a new page.
4. References: numbered `[n].` in the order of first citation, each entry in APA 7 style
   (Author, A. A. (Year). Title. *Source*, *volume*(issue), pages. DOI), as in the template's
   sample list. The IEEE paper keeps IEEE references.
5. Appendices numbered Appendix 1, 2, ... (captions Figure A1.1, Table A1.1). The *Action
   Taken on Previous Review Comments* page is Appendix 7.

Formatting follows the template's page specifications: A4, margins 3.81 cm left and 2.54 cm
right, top and bottom; Times New Roman 12 pt with 1.5 line spacing, justified, and a 0.5 inch
first-line indent; chapter heading "Chapter n" (12 pt bold) above the chapter title (16 pt
bold); section headings 14 pt and subsection headings 12 pt, both in capitals; captions 11 pt
(the template's minimum is 10 pt); page numbers at the bottom centre. The contents and the
lists list two heading levels and are single spaced, as the template allows. Tables have a
shaded header row that repeats on every page. Placeholders and pending markers have a yellow
highlight. The template limits the report to 60 to 100 pages in all; the current build has
100, so keep additions short (the build log records the page count).

### Fields: table of contents, lists, captions, page numbers

The table of contents (`TOC \o "1-3"`), List of Figures (`TOC \c "Figure"`) and List of Tables
(`TOC \c "Table"`) are real Word fields. Each caption number is a `SEQ` field: `\s 1` restarts
it at every chapter in Word, and the first caption of a chapter also carries `\r 1` so that
LibreOffice restarts it too. Page numbers are `PAGE` fields in two sections with separate
footers: lower-case roman for the preliminary pages, arabic from Chapter 1. The cached
caption numbers are written by the builder, so they are right before any field update.

python-docx cannot lay out pages, so the TOC and the two lists hold placeholder text until
they are updated:

* **Microsoft Word** updates them when the file is opened, because the document sets
  `updateFields`. Answer *Yes* to the prompt. To update by hand: select all, then F9.
* **LibreOffice Writer**: Tools, then Update, then Update All.
* **`--pdf`** runs LibreOffice headless with a small Basic macro. The macro sits in a
  temporary user profile, so it never changes your own LibreOffice settings. It loads the
  DOCX, updates every index twice (so page numbers settle) and exports the PDF. If the macro
  fails, a plain `soffice --convert-to pdf` is used and the build log says the indexes were
  not updated.

## Content files (`report/content/*.md`)

The builder works out each file's role from its name:

| Role | File name pattern (examples) |
|---|---|
| Abstract / list of symbols and abbreviations | `04_abstract.md`, `05_abbreviations.md`. Any `0N_` or `front_` prefix works, and so do the plain names. |
| (superseded) title, bonafide, declaration, acknowledgement | files such as `00_title.md` are skipped: those pages come from the template and `report/team.json`. |
| Chapter *n* | `ch1_introduction.md`, `chapter3_design.md`, ... Several files with the same number are joined in name order. |
| Appendices | `appendices.md`, `appendix_a_screens.md`, ... |
| Include-only | any other file, for example `abstract_results.md`, `ch4_study_results.md` or `action_taken.md`. These are only used through `[[include:...]]`. |

A front-matter or chapter slot with no file gets a built-in default
(`report/lib/frontmatter.py`) or a highlighted placeholder chapter, so the document
structure is always complete. A file whose first line is `<!-- placeholder -->` is used
only when no real file fills the same slot.

`action_taken.md` is included at the end of `appendices.md` as Appendix 7. It would be shown
as its own preliminary page, after the list of abbreviations, only if no content file
referenced it through `[[include:action_taken.md]]` or `[[table:action_taken]]`.

### Markup

| Markup | Result |
|---|---|
| `# Title` | chapter: "Chapter n" plus the title on a new page. In the appendices it becomes "Appendix n". A leading "Chapter 3:" is dropped. |
| `## Title` / `### Title` | numbered section n.m / subsection n.m.k |
| `#* Title` (also `##*`) | unnumbered heading, used for front matter. In the appendices, `#* Appendix 3: Test Cases` keeps the number 3 (a letter is converted: C becomes 3). |
| `#### Title` | unnumbered minor heading |
| `**bold**`, `*italic*`, `` `code` `` | inline styles. Long code paths are allowed to wrap. |
| `- item`, `1. item` | bullet / numbered list. Indent by 2 spaces to nest. Every numbered list starts again at 1. |
| `> text` | indented quotation |
| `[text](https://...)` | hyperlink |
| `[@key]`, `[@a; @b]` | numbered citation such as `[3]` or `[1], [4]–[6]`. Keys must exist in `report/references/*.json`; an unknown key shows a highlighted `[CITATION NOT FOUND: key]` and is listed in the build log. |
| `{{path}}`, `{{path\|filter}}` | value from a results file (see below). A missing value shows `[PENDING: path]`. |
| `[TO BE FILLED: ...]` | yellow placeholder for the team. `[PENDING ...]`, `[TODO ...]` and `[TBD ...]` are highlighted too. |
| `Table: Caption` followed by a pipe table | "Table c.n: Caption" above a formatted table. An alignment row such as `\|:--:\|--:\|` sets column alignment. |
| `[[figure:path.png\|Caption\|width=6.0]]` | centred image with "Figure c.n: Caption" below it. Paths are relative to the repository root. The width is in inches, capped at the 5.77 in text width. For `.svg`, a `.png` with the same name is used. A missing file shows `[FIGURE PENDING: path]`. |
| `[[table:KEY]]`, `[[table:KEY\|Caption]]` | generated data table (see below). A `Table: Caption` line just before it also sets the caption. |
| `[[include:file.md]]` | inlines another content file. A missing file shows `[PENDING: file.md]`. |
| ` ```algorithm ` fence | bordered pseudocode box with the caption "Algorithm c.n: first line". The font shrinks (9 to 6.5 pt) so that long lines fit. |
| other ` ``` ` fences | shaded monospace box with no caption |
| `\pagebreak` | page break |
| `<!-- ... -->` | comment, not printed |

In the preliminary pages, table captions are left out on purpose: those tables are not
numbered and do not appear in the List of Tables.

### Inline values

`{{path|filter}}` reads from the results context (`report/lib/results_context.py`). Path
segments are separated by dots. A key that itself contains dots, such as `qwen3.5:4b`, is
matched greedily. Brackets also work: `reranker["BAAI_bge-reranker-base"]`.

| Key | Source file |
|---|---|
| `study` | `evaluation/results/summary.json` (written by `evaluation/analyze.py`) |
| `retrieval` | `evaluation/results/retrieval/metrics.json` |
| `reranker.<dir>` | `evaluation/results/retrieval/reranker_ablation/<dir>/metrics.json` |
| `model_selection`, `model_selection_r1` | `evaluation/results/model_selection/summary.json`, `.../model_selection_round1_prefix/summary.json` |
| `corpus` | `reports/corpus_stats.json` |
| `pii` | `reports/privacy_redaction_metrics.json` |
| `perf.<stem>`, `perf_latest.before/after` | `reports/performance/*.json` (latency probes, newest file for each label) |
| `aq`, `rabbits`, `adversarial` | `reports/medbridge_aq_benchmark_report.json`, `reports/rabbits_benchmark_report.json`, `reports/adversarial_live_report.json`, **live runs only**. Mock-mode reports are ignored (they come from a routing simulator, not the model). |
| `load` | `reports/performance/load_test.json` |
| `tests` | `reports/test_summary.json`, or else JUnit XML files in `reports/tests/` |
| `calibration`, `human_agreement` | `evaluation/results/calibration.json`, `evaluation/results/human_agreement.json` |
| `smoke` | `reports/smoke_test_latest.json` |
| `hallubench` | computed from `evaluation/datasets/hallubench_v1.json` (`n_items`, `n_turns`, `by_category.<cat>.items/turns/dev/test/multi_turn`, `by_gold_action`, `by_style`, ...) |
| `manifest`, `manifest_stats` | `data/corpus_manifest.json` and counts computed from it |

Filters can be chained, for example `{{x|pct}}`:

| Filter | Example result |
|---|---|
| `pct` / `pct0` / `pct2` | fraction to percent: `0.743` gives `74.3%` / `74%` / `74.30%` |
| `perc` | value already in percent: `100.0` gives `100.0%` |
| `f0` … `f3` | fixed decimals |
| `int`, `comma` | `2747`, `2,747` |
| `s` | milliseconds to seconds: `9354.4` gives `9.4` |
| `ci` | Wilson proportion `{rate, low, high}` gives `52.5% [40.0, 64.7]` |
| `cif` | interval `{estimate, low, high}` gives `0.740 [0.650, 0.825]` |
| `len`, `keys`, `upper`, `lower`, `date` | utilities |

Examples: `{{corpus.totals.documents|int}}`, `{{retrieval.modes.hybrid.mrr|cif}}`,
`{{model_selection_r1.results.qwen3.5:4b.routing.accuracy_strict|ci}}`,
`{{pii.datasets.blind_v2.results.all.overall_strict.recall|f3}}`,
`{{study.per_condition.C5.llm.hallucination|ci}}`.

### Generated tables (`[[table:KEY]]`)

Each generator returns nothing while its source file is missing, and the report then shows
`[TABLE PENDING: KEY]` under the default caption. Every generated table has a small "Source:"
note under it that names the results file.

| Key | Content | Source |
|---|---|---|
| `corpus`, `corpus_by_type`, `corpus_topics` | knowledge-base documents / per type / per topic | corpus stats |
| `retrieval_metrics` | Recall@5/20, MRR, nDCG@5 with 95% CI per retrieval mode | retrieval metrics |
| `reranker_ablation` | no reranker, the default reranker and the ablation rerankers | retrieval and reranker metrics |
| `model_selection`, `model_selection_round1`, `model_selection_schema` | routing accuracy, macro-F1, JSON validity, latency, speed per model | model-selection summaries |
| `pii_metrics`, `pii_per_category` | strict and overlap P/R/F1 (development and blind sets), recall per category | PII metrics |
| `perf_before_after`, `perf_baseline` | latency per scenario turn, before and after optimisation / baseline only | latency probes |
| `test_summary` | tests, passed, failed, skipped per suite | test summary |
| `aq_benchmark`, `aq_subsets` | MedBridge-AQ (live) overall and per subset | AQ report |
| `rabbits`, `adversarial`, `load_test`, `smoke_test` | live robustness, security, load and smoke results | the respective reports |
| `hallucination_by_condition` | hallucination rates (LLM and NLI judges, Wilson CI), unsupported-claim ratio, coverage, over-abstention | study summary |
| `unsafe_answer` | unsafe-answer rate and appropriate abstention on unanswerable items | study summary |
| `routing_by_condition` | strict and lenient routing accuracy, macro-F1, emergency recall, false escalation | study summary |
| `mcnemar`, `cochran_q` | paired exact McNemar tests with Holm correction, risk difference, odds ratio; Cochran's Q | study summary |
| `judge_agreement` | Cohen's kappa between the LLM and NLI judges (plus human agreement if present) | study summary |
| `calibration` | NLI, V1 and V2 thresholds calibrated on the dev split | calibration |
| `degraded_turns`, `latency_by_condition`, `category_breakdown` | study validity and stratified views | study summary |
| `hallubench_composition`, `hallubench_actions` | benchmark items per category and split, gold actions | HalluBench dataset |
| `action_taken` | first pipe table of `report/content/action_taken.md` | content |

The study generators follow the `summary.json` schema written by `evaluation/analyze.py`
(`per_condition.<C>.llm.hallucination`, `tests.mcnemar[]`, `judge_agreement`, ...). A V1, V2
or V12 condition that `calibration_check` flags as not calibrated is labelled UNCALIBRATED.

## Build log

Each build prints a summary and writes `report/build/report_build_log.json` with:

* the file used for each front-matter slot;
* the chapters and appendices assembled, and the include-only files;
* every pending marker (missing figures, tables, values, includes) and every unknown
  citation key;
* the counts of figures, tables, algorithms and references;
* the results files that were loaded and the ones that were skipped (for example a
  mock-mode benchmark report);
* the PDF page count and the conversion method.

Before handing in, check the log for `pending_markers` and `citation_keys_missing`. Search
the document for the yellow `[TO BE FILLED: ...]` placeholders, which the team completes by
hand.

## Tests

```bash
.venv-mac/bin/python -m pytest tests/report -q
```

The tests build a small report from a fixture content directory in a temporary folder. They
read it back with python-docx and check the headings and numbering, the captions and SEQ
fields, the TOC/LoF/LoT fields and `updateFields`, the section page numbering, the
highlights, the generated tables, the template pages, the per-student copies, the numbered
APA references and the citation order. The study
tables are checked against a `summary.json` built in the temporary folder from synthetic
outcomes with the real `evaluation/analyze.py` functions. One slow test converts the
fixture to PDF when LibreOffice is installed.

## Layout of the code

| File | Role |
|---|---|
| `report/build_report.py` | CLI, content discovery, rendering, references, sections, PDF |
| `report/lib/content.py` | markup parser (blocks and inline tokens) |
| `report/lib/results_context.py` | loads results files, path resolution, value filters |
| `report/lib/tables.py` | `[[table:KEY]]` generators |
| `report/lib/references.py` | APA (report) and IEEE (paper) formatting and citation numbering |
| `report/lib/vit_front.py` | VIT template pages from `report/team.json` |
| `report/team.json` | names, register numbers, guide, Head of the Department, programme, date |
| `report/lib/docx_helpers.py` | styles, fields, footers, table formatting, lists |
| `report/lib/frontmatter.py` | fallback abstract and abbreviations pages, placeholder chapters |
| `report/lib/libreoffice.py` | headless LibreOffice PDF export with index refresh |
