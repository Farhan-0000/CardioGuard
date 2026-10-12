"""Library behind ``report/build_report.py`` (MedBridge final report generator).

Modules
-------
results_context  loads every real results / measurement file into one dict
content          parser for the report content markup (``report/content/*.md``)
references       APA (report) and IEEE (paper) entries and citation numbering
tables           generators for ``[[table:KEY]]`` data tables
docx_helpers     low-level python-docx helpers (styles, fields, sections, tables)
frontmatter      fallback abstract and abbreviations pages, placeholder chapters
vit_front        VIT BCSE497J template pages (cover, title, declaration, certificate,
                 acknowledgement) from report/team.json
libreoffice      DOCX -> PDF conversion with index (TOC / LoF / LoT) refresh
"""
