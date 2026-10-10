"""
IEEE references and citation numbering.

Only the verified entries in ``report/references/references.json`` and
``report/references/patents.json`` can be cited. ``[@key]`` / ``[@a; @b]`` markers
are numbered in order of first appearance across the whole report; the reference
list is printed in that order. Unknown keys are never invented: they are reported
and rendered as a highlighted ``[CITATION NOT FOUND: key]`` marker.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REFERENCE_FILES: tuple[str, ...] = (
    "report/references/references.json",
    "report/references/patents.json",
)

#: Words that mark an author string as an organisation (kept verbatim, no initials).
_ORG_WORDS = re.compile(
    r"\b(organi[sz]ation|collaboration|committee|association|work group|group|institute|"
    r"department|council|parliament|project|foundation|excellence|team|office|agency|"
    r"society|commission|ministry|university|consortium|initiative|program|programme|"
    r"services|inc|ltd|corp|contributors|developers|authors)\b",
    re.IGNORECASE,
)
_SUFFIX = re.compile(r",?\s+(Jr\.?|Sr\.?|II|III|IV)$")
_PARTICLES = {"van", "von", "der", "den", "de", "da", "del", "di", "la", "le", "du", "dos"}
_MONTHS = (
    "Jan.",
    "Feb.",
    "Mar.",
    "Apr.",
    "May",
    "Jun.",
    "Jul.",
    "Aug.",
    "Sep.",
    "Oct.",
    "Nov.",
    "Dec.",
)


@dataclass
class Segment:
    """A run of reference text; ``italic`` for titles of journals/books/venues."""

    text: str
    italic: bool = False


def load_references(repo_root: Path) -> dict[str, dict[str, Any]]:
    """All citable entries by key (references first, then patents)."""
    out: dict[str, dict[str, Any]] = {}
    for rel in REFERENCE_FILES:
        path = repo_root / rel
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entries = data if isinstance(data, list) else data.get("references", [])
        for entry in entries:
            if isinstance(entry, dict) and entry.get("key"):
                out.setdefault(str(entry["key"]), entry)
    return out


# ----------------------------------------------------------------------
# Author names
# ----------------------------------------------------------------------


def is_organisation(name: str) -> bool:
    name = name.strip()
    if " " not in name:
        return True
    if "(" in name or ":" in name or "," in _SUFFIX.sub("", name):
        return True
    return bool(_ORG_WORDS.search(name))


def _initial(token: str) -> str:
    if token.endswith(".") and len(token) <= 3:
        return token
    parts = [p for p in token.split("-") if p]
    return "-".join(f"{p[0]}." for p in parts)


def format_person(name: str) -> str:
    """``Daniel W. Jones`` -> ``D. W. Jones``; ``Shi-Qi Yan`` -> ``S.-Q. Yan``."""
    name = name.strip()
    if is_organisation(name):
        return name
    suffix = ""
    m = _SUFFIX.search(name)
    if m:
        suffix = " " + m.group(1).rstrip(".") + ("." if m.group(1).startswith(("Jr", "Sr")) else "")
        name = name[: m.start()].strip()
    tokens = name.split()
    surname = [tokens.pop()]
    while tokens and tokens[-1].lower() in _PARTICLES:
        surname.insert(0, tokens.pop())
    initials = " ".join(_initial(t) for t in tokens)
    return f"{initials} {' '.join(surname)}{suffix}".strip()


def format_authors(authors: list[str]) -> str:
    """IEEE author list: up to six names; with ``et al.`` in the data, first author + et al."""
    names = [a for a in (authors or []) if a and a.strip()]
    et_al = any(a.strip().lower().rstrip(".") == "et al" for a in names)
    names = [a for a in names if a.strip().lower().rstrip(".") != "et al"]
    if not names:
        return ""
    formatted = [format_person(a) for a in names]
    if et_al or len(formatted) > 6:
        return f"{formatted[0]} et al."
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return ", ".join(formatted[:-1]) + ", and " + formatted[-1]


# ----------------------------------------------------------------------
# Entry formatting
# ----------------------------------------------------------------------


def _clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _pages(pages: str) -> str:
    p = _clean(pages)
    if not p:
        return ""
    if re.search(r"\w\s*[-–]+\s*\w", p):
        return "pp. " + re.sub(r"\s*[-–]+\s*", "–", p)
    if re.fullmatch(r"[A-Za-z]?\d+", p):
        return f"p. {p}"
    return p


def _date(iso: str) -> str:
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", iso)
    if not m:
        return iso
    return f"{_MONTHS[int(m.group(2)) - 1]} {int(m.group(3))}, {m.group(1)}"


def _accessed(identifier: str) -> str:
    m = re.search(r"accessed\s+(\d{4}-\d{2}-\d{2})", identifier, re.IGNORECASE)
    return f" Accessed: {_date(m.group(1))}." if m else ""


def _venue_short(venue: str) -> str:
    """Drop a trailing URL that some entries embed in the venue field."""
    venue = _clean(venue)
    venue = re.sub(r",?\s*https?://\S+$", "", venue)
    return venue.rstrip(",")


def _locator(entry: dict[str, Any]) -> list[str]:
    parts = []
    if _clean(entry.get("volume")):
        parts.append(f"vol. {_clean(entry['volume'])}")
    if _clean(entry.get("issue")):
        parts.append(f"no. {_clean(entry['issue'])}")
    if _clean(entry.get("pages")):
        parts.append(_pages(entry["pages"]))
    return parts


def _doi_or_id(entry: dict[str, Any]) -> str:
    doi = _clean(entry.get("doi"))
    if doi:
        return f"doi: {doi}."
    ident = _clean(entry.get("identifier"))
    m = re.search(r"arXiv:\s*\d{4}\.\d{4,5}(v\d+)?", ident)
    if m:
        return f"{m.group(0).replace(' ', '')}."
    return ""


def _online(entry: dict[str, Any]) -> str:
    url = _clean(entry.get("url"))
    return f"[Online]. Available: {url}" if url else ""


def format_entry(entry: dict[str, Any]) -> list[Segment]:
    """One IEEE reference (without the ``[n]`` label) as text segments."""
    kind = str(entry.get("type", "misc")).lower()
    authors = format_authors(entry.get("authors") or [])
    title = _clean(entry.get("title")).rstrip(".")
    venue = _venue_short(entry.get("venue", ""))
    year = _clean(entry.get("year"))
    ident = _clean(entry.get("identifier"))
    segs: list[Segment] = []

    def add(text: str, italic: bool = False) -> None:
        if text:
            segs.append(Segment(text, italic))

    lead = f"{authors}, " if authors else ""
    if kind == "book":
        add(lead)
        add(title, italic=True)
        rest = [venue, year]
        add(". " + ", ".join(r for r in rest if r) + ".")
        add(" " + _doi_or_id(entry))
    elif kind == "patent":
        add(f'{lead}"{title}," ')
        num = re.search(r"(US)\s*([\d,]+)\s*(B\d|A\d)", ident)
        granted = re.search(r"granted\s+(\d{4}-\d{2}-\d{2})", ident)
        if num:
            add(f"U.S. Patent {num.group(2)} {num.group(3)}")
        else:
            add(ident.split("(")[0].strip())
        add(f", {_date(granted.group(1))}." if granted else f", {year}.")
        assignee = re.search(r"assignee:\s*([^)]+)\)", entry.get("venue", ""))
        if assignee:
            add(f" Assignee: {assignee.group(1).strip()}.")
        add(" " + _online(entry))
    elif kind in {"inproceedings", "conference"}:
        add(f'{lead}"{title}," in ')
        add(venue if venue.lower().startswith(("proc", "advances")) else f"Proc. {venue}", True)
        loc = [p for p in _locator(entry) if not p.startswith("vol.")]
        add(", " + ", ".join([year, *loc]) + ".")
        tail = _doi_or_id(entry)
        add(" " + tail if tail else "")
    elif kind in {"article", "guideline"} and (entry.get("volume") or entry.get("pages")):
        add(f'{lead}"{title}," ')
        add(venue, True)
        add(", " + ", ".join([*_locator(entry), year]) + ".")
        tail = _doi_or_id(entry)
        add(" " + tail if tail else "")
    elif kind == "preprint":
        add(f'{lead}"{title}," ')
        arx = re.search(r"arXiv:\s*\d{4}\.\d{4,5}", ident)
        add(f"arXiv preprint {arx.group(0).replace(' ', '')}" if arx else venue, not arx)
        add(f", {year}.")
        doi = _clean(entry.get("doi"))
        add(f" doi: {doi}." if doi else "")
    elif kind in {"web", "software"}:
        add(f'{lead}"{title}," ')
        add(venue, True)
        add(f", {year}.")
        add(_accessed(ident))
        add(" " + _online(entry))
    else:  # report, guideline without volume, law, misc
        add(f'{lead}"{title}," ')
        add(venue, True)
        extra = ""
        if kind in {"report", "guideline"} and ident.upper().startswith("ISBN"):
            first = re.split(r"[;(]", ident)[0].strip().rstrip(",")
            if first and len(first) <= 80:
                extra = f", {first}"
        add(f"{extra}, {year}.")
        tail = _doi_or_id(entry)
        add(" " + tail if tail else "")
        if not _clean(entry.get("doi")):
            add(" " + _online(entry))
    # tidy whitespace / punctuation between segments
    out: list[Segment] = []
    for seg in segs:
        text = re.sub(r"\s+", " ", seg.text)
        if out and out[-1].text.endswith(" ") and text.startswith(" "):
            text = text.lstrip()
        out.append(Segment(text, seg.italic))
    if out:
        out[-1].text = out[-1].text.rstrip()
        out[0].text = out[0].text.lstrip()
    return [s for s in out if s.text]


def format_entry_text(entry: dict[str, Any]) -> str:
    return "".join(s.text for s in format_entry(entry))


# ----------------------------------------------------------------------
# Citation registry
# ----------------------------------------------------------------------

CITE_RE = re.compile(r"\[(@[^\[\]]+)\]")


def parse_cite_keys(body: str) -> list[str]:
    """``@a; @b, @c`` -> ``[a, b, c]``."""
    return [k.strip().lstrip("@").strip() for k in re.split(r"[;,]", body) if k.strip()]


@dataclass
class CitationRegistry:
    """Numbers citation keys by first appearance and formats ``[n]`` labels."""

    entries: dict[str, dict[str, Any]]
    order: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def register(self, keys: list[str]) -> None:
        for key in keys:
            if key in self.entries:
                if key not in self.order:
                    self.order.append(key)
            elif key not in self.missing:
                self.missing.append(key)

    def register_text(self, text: str) -> None:
        for m in CITE_RE.finditer(text or ""):
            self.register(parse_cite_keys(m.group(1)))

    def number(self, key: str) -> int | None:
        if key not in self.entries:
            return None
        if key not in self.order:
            self.order.append(key)
        return self.order.index(key) + 1

    def label(self, keys: list[str]) -> tuple[str, list[str]]:
        """IEEE label for a group, e.g. ``[1], [3]–[5]``; returns (label, unknown keys)."""
        nums = sorted({n for n in (self.number(k) for k in keys) if n is not None})
        unknown = [k for k in keys if k not in self.entries]
        parts: list[str] = []
        i = 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            if j - i >= 2:
                parts.append(f"[{nums[i]}]–[{nums[j]}]")
            else:
                parts.extend(f"[{n}]" for n in nums[i : j + 1])
            i = j + 1
        return ", ".join(parts), unknown

    def bibliography(self) -> list[tuple[int, dict[str, Any]]]:
        return [(i + 1, self.entries[k]) for i, k in enumerate(self.order)]
