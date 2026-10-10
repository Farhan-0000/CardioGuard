# Compiling the LaTeX version (IEEEtran)

`main.tex`, `references.bib` and `figures/` are **generated** by
`report/paper/build_latex.py` from `report/paper/content/*.md`. Rebuild them after every
content change:

```bash
.venv-mac/bin/python report/paper/build_latex.py          # macOS / Linux
.venv\Scripts\python report\paper\build_latex.py          # Windows
```

Nothing is compiled by the build script. The source uses only standard packages that ship
with TeX Live, MiKTeX and Overleaf: `IEEEtran` (conference mode), `cite`, `amsmath`,
`amssymb`, `graphicx`, `xcolor`, `booktabs`, `array`, `tabularx`, `listings`, `algorithm`
and `url`. The file is plain ASCII; special characters are escaped by the builder.

## Overleaf (no installation needed)

1. Zip the contents of this folder: `main.tex`, `references.bib` and the `figures/` folder
   (this README is not needed).
2. In Overleaf: **New Project, Upload Project**, and choose the zip.
3. **Menu, Settings**: Compiler **pdfLaTeX**, Main document **main.tex**.
4. Click **Recompile**. Overleaf runs BibTeX automatically. If references show as `[?]`,
   recompile once more.
5. To share with co-authors, use **Share** in Overleaf; to submit, download the PDF.

If the team starts editing in Overleaf, either stop regenerating `main.tex` (and keep the
Overleaf copy as the master), or port each edit back to `report/paper/content/*.md` and
re-upload. Do not do both, or edits will be overwritten.

## Local compile

```bash
cd report/paper/latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
# or: latexmk -pdf main.tex
```

Install TeX with `brew install --cask mactex-no-gui` (macOS), MiKTeX or TeX Live (Windows),
or `sudo apt install texlive-publishers texlive-latex-extra texlive-science` (Linux).

## What to check in the PDF

- Red bold text is `\pending{...}`: placeholders (`[TO BE FILLED: ...]`) and results that
  do not exist yet. None may remain in a submitted version.
- Page count against the venue limit (see `report/paper/VENUES.md` and the page-budget
  section of `report/paper/README.md`).
- Floats: LaTeX places figures and tables at the top of a column (`[t]`); full-width floats
  (`figure*`, `table*`) appear at the top of the next page, as IEEE requires.
- For the double-blind review copy, rebuild with `--anonymous`.
- Camera-ready: follow the venue's instructions for the IEEE copyright notice (for
  conferences usually a footer added with `\IEEEoverridecommandlockouts` and
  `\IEEEpubid{...}`) and check the PDF with IEEE PDF eXpress if the venue requires it.
  [TO BE FILLED: venue-specific instructions.]
