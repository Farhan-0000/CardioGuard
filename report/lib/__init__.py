"""Library behind ``report/build_report.py`` (MedBridge final report generator).

Modules
-------
results_context  loads every real results / measurement file into one dict
content          parser for the report content markup (``report/content/*.md``)
references       IEEE reference list and citation numbering
tables           generators for ``[[table:KEY]]`` data tables
docx_helpers     low-level python-docx helpers (styles, fields, sections, tables)
frontmatter      default title page / bonafide / declaration / abbreviations
libreoffice      DOCX -> PDF conversion with index (TOC / LoF / LoT) refresh
"""
