"""
Preliminary pages of the VIT Chennai BCSE497J report template.

``report/templates/BCSE497J_report_template.docx`` fixes the order and wording of the
pages before the abstract: cover page, title page, declaration by the candidate and
the certificate. The template asks every student to submit a copy in which the
declaration, the certificate and the acknowledgement carry only that student's name,
while the cover and title pages list the whole team. The details come from
``report/team.json``; any value still reading ``[TO BE FILLED: ...]`` is highlighted.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.shared import Inches, Pt

from lib import docx_helpers as dh

TEAM_FILE = "report/team.json"
LOGO_FILE = "report/templates/vit_chennai_logo.png"
FILL = re.compile(r"\[TO BE FILLED:[^\]]*\]")


@dataclass
class Member:
    name: str
    reg_no: str

    @property
    def label(self) -> str:
        return f"{self.name} ({self.reg_no})"


@dataclass
class Team:
    title: str
    degree: str = "Bachelor of Technology"
    programme: str = "[TO BE FILLED: programme name with specialisation]"
    school: str = "School of Computer Science and Engineering"
    institute: str = "Vellore Institute of Technology, Chennai"
    place: str = "Chennai"
    month_year: str = "[TO BE FILLED: month, year of submission]"
    members: list[Member] = field(default_factory=list)
    guide_name: str = "[TO BE FILLED: guide's name]"
    guide_designation: str = "[TO BE FILLED: guide's designation]"
    guide_area: str = "[TO BE FILLED: guide's area of expertise]"
    hod_name: str = "[TO BE FILLED: name of the Head of the Department]"

    def member(self, reg_no: str) -> Member:
        for m in self.members:
            if m.reg_no.lower() == reg_no.lower():
                return m
        known = ", ".join(m.reg_no for m in self.members) or "none"
        raise KeyError(f"no team member with register number {reg_no} (known: {known})")


def load_team(repo_root: Path, path: Path | None = None) -> Team:
    """Team details from ``report/team.json``; without the file every field is a placeholder."""
    file = path or repo_root / TEAM_FILE
    if path is None and not file.is_file():
        return Team(title="[TO BE FILLED: project title]")
    data: dict[str, Any] = json.loads(file.read_text(encoding="utf-8"))
    guide = data.get("guide") or {}
    hod = data.get("hod") or {}
    team = Team(title=str(data.get("title", "")).strip())
    for key in ("degree", "programme", "school", "institute", "place", "month_year"):
        if data.get(key):
            setattr(team, key, str(data[key]).strip())
    team.members = [
        Member(str(m["name"]).strip(), str(m["reg_no"]).strip()) for m in data.get("members") or []
    ]
    if guide.get("name"):
        team.guide_name = str(guide["name"]).strip()
    if guide.get("designation"):
        team.guide_designation = str(guide["designation"]).strip()
    if guide.get("area"):
        team.guide_area = str(guide["area"]).strip()
    if hod.get("name"):
        team.hod_name = str(hod["name"]).strip()
    return team


class FrontWriter:
    """Writes the fixed-format template pages straight into the document."""

    def __init__(self, doc: Any, repo_root: Path, on_fill: Callable[[str], None]) -> None:
        self.doc = doc
        self.logo = repo_root / LOGO_FILE
        self.on_fill = on_fill

    # -- primitives ------------------------------------------------------------

    def para(
        self,
        text: str = "",
        *,
        size: float = 12,
        bold: bool = False,
        italic: bool = False,
        underline: bool = False,
        upper: bool = False,
        align: Any = WD_ALIGN_PARAGRAPH.CENTER,
        before: float = 0,
        after: float = 6,
        spacing: float = 1.15,
        indent: float = 0.0,
    ) -> Any:
        p = self.doc.add_paragraph()
        p.alignment = align
        pf = p.paragraph_format
        pf.space_before = Pt(before)
        pf.space_after = Pt(after)
        pf.line_spacing = spacing
        pf.first_line_indent = Inches(indent)
        pf.page_break_before = False
        self.runs(p, text, size=size, bold=bold, italic=italic, underline=underline, upper=upper)
        return p

    def runs(
        self,
        p: Any,
        text: str,
        *,
        size: float = 12,
        bold: bool = False,
        italic: bool = False,
        underline: bool = False,
        upper: bool = False,
    ) -> None:
        pos = 0
        for m in [*FILL.finditer(text), None]:
            end = m.start() if m else len(text)
            if end > pos:
                chunk = text[pos:end]
                r = p.add_run(chunk.upper() if upper else chunk)
                self._style(r, size, bold, italic, underline)
            if m:
                r = p.add_run(m.group(0))
                self._style(r, size, bold, italic, underline)
                dh.highlight(r)
                self.on_fill(m.group(0))
                pos = m.end()

    @staticmethod
    def _style(run: Any, size: float, bold: bool, italic: bool, underline: bool) -> None:
        run.font.size = Pt(size)
        run.bold = bold or None
        run.italic = italic or None
        run.underline = underline or None

    def logo_para(self, width: float = 2.6, *, before: float = 0, after: float = 6) -> None:
        p = self.para(before=before, after=after, spacing=1.0)
        if self.logo.is_file():
            p.add_run().add_picture(str(self.logo), width=Inches(width))
        else:
            self.runs(p, "[TO BE FILLED: VIT Chennai logo]", bold=True)

    def page_break(self) -> None:
        dh.add_page_break(self.doc.add_paragraph())

    def signature_line(self, left: str, right: str, *, bold_right: bool = True) -> None:
        p = self.para(align=WD_ALIGN_PARAGRAPH.LEFT, after=0, spacing=1.5)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(dh.TEXT_WIDTH_IN), WD_TAB_ALIGNMENT.RIGHT)
        self.runs(p, left)
        p.add_run("\t")
        self.runs(p, right, bold=bold_right)

    # -- pages -----------------------------------------------------------------

    def _title_block(self, team: Team, *, project_report_on: bool) -> None:
        if project_report_on:
            self.para("A project report on", size=12, italic=True, before=24, after=12)
        self.para(
            team.title,
            size=20,
            bold=True,
            upper=True,
            before=0 if project_report_on else 36,
            after=18,
            spacing=1.15,
        )
        self.para(
            "Submitted in partial fulfillment for the award of the degree of",
            size=14,
            italic=True,
            after=12,
        )
        self.para(f"{team.degree} in {team.programme}", size=22, bold=True, after=24)
        self.para("by", size=14, italic=True, after=12)
        for m in team.members:
            self.para(m.label, size=16, bold=True, upper=True, after=4)
        self.logo_para(2.8, before=36, after=12)
        self.para(team.school, size=16, bold=True, upper=True, after=8)
        self.para(team.month_year, size=12, after=0)

    def cover_page(self, team: Team) -> None:
        self._title_block(team, project_report_on=True)
        self.page_break()

    def title_page(self, team: Team) -> None:
        self._title_block(team, project_report_on=False)
        self.page_break()

    def declaration(self, team: Team, member: Member | None) -> None:
        who = member.label if member else "[TO BE FILLED: your name (register number)]"
        self.logo_para(2.6, after=24)
        self.para("DECLARATION", size=14, bold=True, underline=True, after=24)
        body = dict(align=WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.5, after=12, indent=0.5)
        self.para(
            f"I hereby declare that the thesis entitled “{team.title}” submitted by "
            f"{who}, for the award of the degree of {team.degree} in {team.programme}, "
            f"{team.institute} is a record of bonafide work carried out by me under the "
            f"supervision of {team.guide_name}.",
            **body,
        )
        self.para(
            "I further declare that the work reported in this thesis has not been submitted "
            "and will not be submitted, either in part or in full, for the award of any other "
            "degree or diploma in this institute or any other institute or university.",
            **body,
        )
        self.para(
            f"Place: {team.place}", align=WD_ALIGN_PARAGRAPH.LEFT, before=72, after=0, spacing=1.5
        )
        self.signature_line("Date:", "Signature of the Candidate")
        self.page_break()

    def certificate(self, team: Team, member: Member | None) -> None:
        who = member.label if member else "[TO BE FILLED: your name (register number)]"
        self.logo_para(2.6, after=12)
        self.para(team.school, size=14, bold=True, after=12)
        self.para("CERTIFICATE", size=16, after=18)
        self.para(
            f"This is to certify that the report entitled “{team.title}” is prepared "
            f"and submitted by {who} to {team.institute}, in partial fulfillment of the "
            f"requirement for the award of the degree of {team.degree} in {team.programme} is "
            "a bonafide record carried out under my guidance. The project fulfills the "
            "requirements as per the regulations of this University and in my opinion meets "
            "the necessary standards for submission. The contents of this report have not been "
            "submitted and will not be submitted either in part or in full, for the award of "
            "any other degree or diploma and the same is certified.",
            align=WD_ALIGN_PARAGRAPH.JUSTIFY,
            spacing=1.5,
            after=24,
            indent=0.5,
        )
        left = dict(align=WD_ALIGN_PARAGRAPH.LEFT, after=2, spacing=1.15)
        self.para("Signature of the Guide:", **left)
        self.para(f"Name: {team.guide_name}", **left)
        self.para("Date:", **left)
        table = self.doc.add_table(rows=3, cols=2)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        dh.set_column_widths(table, [dh.TEXT_WIDTH_IN / 2] * 2)
        for col in range(2):
            for row, text in enumerate(("Signature of the Examiner", "Name:", "Date:")):
                cell_p = table.rows[row].cells[col].paragraphs[0]
                cell_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                cell_p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                cell_p.paragraph_format.space_before = Pt(18 if row == 0 else 2)
                self.runs(cell_p, text)
        self.para("Approved by the Head of Department,", before=30, after=0)
        self.para(f"{team.degree} in {team.programme}", bold=True, after=8)
        self.para(f"Name: {team.hod_name}", after=2)
        self.para("Date:", after=18)
        self.para(f"(Seal of the {team.school})", size=10, bold=True, after=0)

    def acknowledgement(self, team: Team, member: Member | None) -> None:
        """The template's acknowledgement, worded without assuming the guide's pronouns."""
        body = dict(align=WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.15, after=8)
        self.para(
            f"It is my pleasure to express with deep sense of gratitude to {team.guide_name}, "
            f"{team.guide_designation}, {team.school}, {team.institute}, for the constant "
            "guidance, continual encouragement and understanding, and more than all for "
            "teaching me patience in my endeavor. My association with "
            f"{team.guide_name} is not confined to academics only, but it is a great "
            "opportunity on my part of work with an intellectual and expert in the field of "
            f"{team.guide_area}.",
            **body,
        )
        self.para(
            "It is with gratitude that I would like to extend my thanks to the visionary leader "
            "Dr. G. Viswanathan our Honorable Chancellor, Mr. Sankar Viswanathan, Dr. Sekar "
            "Viswanathan, Dr. G V Selvam Vice Presidents, Dr. Sandhya Pentareddy, Executive "
            "Director, Ms. Kadhambari S. Viswanathan, Assistant Vice-President, Dr. V. S. "
            "Kanchana Bhaaskaran Vice-Chancellor, Dr. T. Thyagarajan Pro-Vice Chancellor, VIT "
            "Chennai and Dr. P. K. Manoharan, Additional Registrar for providing an exceptional "
            "working environment and inspiring all of us during the tenure of the course.",
            **body,
        )
        self.para(
            "Special mention to Dr. Viswanathan V, Dean, Dr. Nithyanandam P, Associate Dean, "
            "Dr. Suganya G, Associate Dean, Dr. Sweetlin Hemalatha C, Associate Dean, School of "
            "Computer Science and Engineering, Vellore Institute of Technology, Chennai for "
            "spending their valuable time and efforts in sharing their knowledge and for "
            "helping us in every aspect.",
            **body,
        )
        self.para(
            "In jubilant state, I express ingeniously my whole-hearted thanks to "
            f"{team.hod_name}, Head of the Department, B.Tech. {team.programme} and the Project "
            "Coordinators for their valuable support and encouragement to take up and complete "
            "the thesis.",
            **body,
        )
        self.para(
            "My sincere thanks to all the faculties and staffs at Vellore Institute of "
            "Technology, Chennai who helped me acquire the requisite knowledge. I would like to "
            "thank my parents for their support. It is indeed a pleasure to thank my friends "
            "who encouraged me to take up and complete this task.",
            **body,
        )
        who = member.name if member else "[TO BE FILLED: name of the student]"
        self.para(f"Place: {team.place}", align=WD_ALIGN_PARAGRAPH.LEFT, before=18, after=0)
        self.signature_line("Date:", who)
