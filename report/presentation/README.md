# Review presentation builder

`report/build_presentation.py` generates the review deck
`report/MedBridge_Review_Presentation.pptx` (18 slides, 16:9, speaker notes on every slide).
The deck is titled *CardioGuard*; the system is called MedBridge throughout.

As with the report, nobody types a measured number into the slides. Every value is read
when the deck is built, from the results and measurement files (`evaluation/results/`,
`reports/`, `data/corpus_manifest.json`), through the report's results context
(`report/lib/results_context.py`), or it is computed from the code base
(`report/presentation/deck_data.py`). If a result does not exist yet, the slide shows a
yellow `[PENDING: ...]` marker instead. Re-run the builder once new results land; the whole
deck regenerates.

## Quick start

```bash
# macOS / Linux, from the repository root
.venv-mac/bin/python report/build_presentation.py              # PPTX only
.venv-mac/bin/python report/build_presentation.py --pdf        # PPTX + PDF (LibreOffice)
.venv-mac/bin/python report/build_presentation.py --png /tmp/slides --strict

# Windows (PowerShell), from the repository root
.venv\Scripts\python report\build_presentation.py --pdf
```

| Option | Meaning |
|---|---|
| `--out PATH` | output `.pptx` (default `report/MedBridge_Review_Presentation.pptx`) |
| `--content PATH` | slide text and notes (default `report/presentation/slides.toml`) |
| `--pdf` | also write a PDF next to the PPTX with headless LibreOffice |
| `--png DIR` | render every slide to `DIR/slide-NN.png`. Uses LibreOffice and PyMuPDF and is meant for checking the layout |
| `--strict` | exit code 1 on a layout issue, an unknown citation key, or an emergency example whose classifier output differs from `slides.toml` |
| `--log PATH` / `--no-log` | JSON build log (default `report/presentation/build/presentation_build_log.json`) |
| `--quiet` | no console summary |

Exit codes: 0 means success. 1 means `--strict` found a problem. 2 means the content file is
missing or invalid. 3 means a PDF or PNG was requested but LibreOffice could not convert the deck.

Requirements: `python-pptx` (in `requirements.txt`; it pulls in `Pillow`, which the text
measurer uses). LibreOffice is needed only for `--pdf` and `--png`, and PyMuPDF only for
`--png`. The builder never calls a language model or a server, so it is safe to run while an
experiment is in progress.

## Slides and their data

| # | Slide | Data read at build time |
|---|---|---|
| 1 | Title | team, guide, `[TO BE FILLED: university / department / review date]` |
| 2 | Problem and motivation | burden figures quoted from verified references (`report/references/references.json`) |
| 3 | Gap and objectives | literature citations |
| 4 | Architecture | `report/figures/architecture_pipeline.png`; defaults from `medbridge/config.py` |
| 5 | Dual Answerability Gate | the five actions; loop-breaker threshold from the config |
| 6 | Emergency fast-path | the classifier is run on `tests/data/emergency_cases*.json` (in-sample) and on the example messages; HalluBench emergency recall from the study |
| 7 | Knowledge base and retrieval | `reports/corpus_stats.json` (organisations counted with joint labels such as AHA/ACC split), `evaluation/results/retrieval/metrics.json`, `.../reranker_ablation/*/metrics.json`; the "CIs overlap" sentence is computed |
| 8 | Privacy and security | `reports/privacy_redaction_metrics.json` |
| 9 | UI screenshots | `report/screenshots/*.png` (placeholders until they are captured) |
| 10 | Evaluation design | `evaluation/datasets/hallubench_v1.json`; judge models from the judge manifest and `evaluation/judges/nli_judge.py` |
| 11 | Model selection | `evaluation/results/model_selection/summary.json`, `DECISION.md`, `reports/figures/model_selection.png` |
| 12 | Hallucination results | `evaluation/results/summary.json`, `reports/figures/hallucination_rate_by_condition.png` |
| 13 | Unsafe answers and ablation | `summary.json` (McNemar with Holm correction), `unsafe_answer_rate_unanswerable.png`, `ablation_waterfall.png` |
| 14 | Robustness | live AQ, RABBITS and adversarial reports only. Mock-mode reports are ignored. |
| 15 | Performance | `reports/performance/*.json` (newest `before` and `after` latency probes) |
| 16 | Testing and quality | `reports/test_summary.json` or JUnit XML, `reports/baseline/pytest_baseline.txt`, `reports/smoke_test_latest.json`, `docs/notes/code_review_findings.md`, static test inventory |
| 17 | Limitations and future work | model-selection and PII values |
| 18 | Thank you and Q&A | likely viva questions in the notes |

Slides 12 and 13 become "pending" slides while `evaluation/results/summary.json` is missing.
They show placeholders and the decision rule, and they quote no rate. Once the file exists
they show the figures, a per-condition table, and a verdict line. The verdict is worked out
mechanically from the paired McNemar tests: a comparison is a significant reduction when the
Holm-corrected p is below 0.05 and the risk difference is negative, a significant increase when
p is below 0.05 and the risk difference is positive, and not significant otherwise. The
decision rule counts as met when C5 significantly reduces the same outcome (hallucination or
unsafe answers) against both C1 and C2.

## Editing the text

All slide text and speaker notes are in `slides.toml`. The markup is the same as in the report:

| Markup | Result |
|---|---|
| `**bold**` | bold run |
| `{{path\|filter}}` | value from a results file, for example `{{corpus.totals.documents\|int}}`. `deck.*` paths are code facts, for example `{{deck.drugs.brands\|int}}` |
| `[@key]`, `[@a; @b]` | citation. The slide shows an author-year label and the notes list the full IEEE entry. Unknown keys are flagged |
| `[TO BE FILLED: ...]` | yellow placeholder for the team |

Slide ids are fixed because the layout code looks them up, but you can change the wording,
the bullets and the notes freely. A slide with an unknown id is drawn as a plain title plus
`bullets` list. Never type a measured number: use `{{...}}`. A number quoted from the
literature must carry its citation.

## Screenshots

Capture the screenshots against the real backend, never the development mock:

```bash
python -m medbridge.web.screenshots --base-url http://127.0.0.1:8000 --out report/screenshots
```

The builder accepts both the report's names (`answer_citations.png`, `why_panel.png`, ...)
and the names the capture script writes (`answer-sources-desktop-light.png`, ...).

## Layout checks

python-pptx cannot lay text out, so `pptx_kit.py` measures every text box and table. It uses
the real Arial widths (read with Pillow from the installed TrueType font) and shrinks the font
within a set range until the text fits. The build log lists an *issue* when text still
overflows at the minimum size, when a word does not fit its table column, when a shape leaves
the slide, or when two layout regions overlap. `--strict` fails the build on any of these.
Use `--png` to look at the rendered slides.

## Tests

```bash
.venv-mac/bin/python -m pytest tests/report/test_presentation_builder.py -q
```

The tests build decks from a fixture repository in a temporary folder. The study summary in
that folder is produced with the real `evaluation/analyze.py` functions from synthetic outcomes.
The tests check the slide count, the notes, the placeholders, that the pending slides quote no
number, that numbers come from the fixture files, that mock reports are ignored, that
screenshots replace placeholders, that citations resolve, and that the layout checker works.
One slow test converts the deck to PDF when LibreOffice is installed.

## Layout of the code

| File | Role |
|---|---|
| `report/build_presentation.py` | CLI, slide layouts, notes, build log, PDF and PNG export |
| `report/presentation/slides.toml` | slide text and speaker notes |
| `report/presentation/pptx_kit.py` | drawing primitives, palette, text measurement and fitting, layout issues |
| `report/presentation/deck_data.py` | facts computed from the code base (classifier runs, harness sizes, test inventory, config) |
