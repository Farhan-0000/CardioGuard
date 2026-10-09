"""
Deterministic PII/PHI redaction (privacy layer, ADL-030).

``redact()`` runs on every patient message *before* any LLM call, before
persistence and inside the log scrubber. It replaces direct identifiers with
typed placeholders and converts exact dates of birth to an age:

==================  =====================================================
Placeholder          Detected from
==================  =====================================================
``[NAME]``           person names, **only** after strong cues ("my name is",
                     "I'm X Y", "this is X calling", Mr./Mrs./Dr. X,
                     "patient X", "name: X", "my wife X", "regards, X")
``[PHONE]``          Indian mobiles/landlines (+91/0091/0), US numbers,
                     labelled numbers ("phone:", "WhatsApp")
``[EMAIL]``          e-mail addresses
``[ADDRESS]``        house/flat/plot/door numbers + street words + locality
``[PINCODE]``        PIN/ZIP codes with an address or "pincode" context
``[AADHAAR]``        12-digit Aadhaar (4-4-4 grouped or labelled)
``[PAN]``            Indian PAN ``AAAAA9999A``
``[SSN]``            US social security numbers
``[MRN]``            MRN / patient id / UHID / ABHA / hospital / IP numbers
``[INSURANCE_ID]``   insurance / policy / member / claim / TPA numbers
``[URL]`` ``[IP]``   URLs, IPv4 and IPv6 addresses
``[AGE: N]``         exact dates of birth ("DOB 12/03/1970") → age today
==================  =====================================================

Clinical content is deliberately preserved: BP readings (``145/92``), lab
values, doses (``10 mg``, ``2.5mg``), drug names (incl. brands like
``Telma 40``), ages stated as numbers (``I'm 54``), dates of readings and
emergency numbers (``108``, ``112``, ``911``) never match a detector.

The detector is regex/rule based (no ML model, no network, <1 ms per
message) so it is deterministic, explainable and cross-platform. Precision
and recall on a synthetic test set are reported by ``scripts/pii_eval.py``.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import date

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

CATEGORIES: tuple[str, ...] = (
    "NAME", "PHONE", "EMAIL", "ADDRESS", "PINCODE", "AADHAAR", "PAN", "SSN",
    "MRN", "INSURANCE_ID", "URL", "IP", "AGE",
)


@dataclass
class RedactionResult:
    """Outcome of :func:`redact`."""

    text: str
    count: int = 0
    categories: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PiiSpan:
    """A detected identifier: ``text[start:end]`` will be replaced by ``replacement``."""

    start: int
    end: int
    category: str
    replacement: str


# ---------------------------------------------------------------------------
# Shared fragments
# ---------------------------------------------------------------------------

_MONTHS: dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_RE = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
# Separator between a label ("MRN", "phone") and its value.
_SEP = r"\s*(?:(?:is|was|:|#|=|-|–)\s*)*"
# "id" / "number" / "no." suffix of a label.
_NO = r"(?:id|number|no\b\.?|#)"
# Identifier value: alphanumeric with / or - separators, containing a digit.
_ID_VALUE = r"(?=[A-Za-z0-9/-]*\d)[A-Za-z0-9][A-Za-z0-9/-]{3,}(?<![/-])"

# ---------------------------------------------------------------------------
# Detector patterns
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>\"'`]+", re.IGNORECASE)
_URL_TRAILING = ".,;:!?)]}'\""

_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IPV4_RE = re.compile(rf"(?<![\w.]){_OCTET}(?:\.{_OCTET}){{3}}(?![\w]|\.\d)")
_HEX = r"[0-9A-Fa-f]{1,4}"
_IPV6_RE = re.compile(
    rf"(?<![\w:])(?:(?:{_HEX}:){{7}}{_HEX}|(?:{_HEX}:){{1,6}}:(?:{_HEX}:){{0,5}}{_HEX})(?![\w:])"
)

_DOB_CUE = (
    r"(?:\bDOB\b|\bD\.O\.B\.?|\bdate\s+of\s+birth\b|\bbirth\s*date\b|\bbirthday\b|"
    r"\bborn(?:\s+on)?\b)"
)
_DATE_NUMERIC = r"\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})(?!\d)"
_DATE_ISO = r"\d{4}-\d{1,2}-\d{1,2}(?!\d)"
_DATE_DMY_TEXT = rf"\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH_RE},?\s+\d{{4}}(?!\d)"
_DATE_MDY_TEXT = rf"{_MONTH_RE}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}(?!\d)"
_DOB_RE = re.compile(
    rf"{_DOB_CUE}\s*(?:(?:is|was|on|:|-|–)\s*)*"
    rf"(?P<date>{_DATE_ISO}|{_DATE_NUMERIC}|{_DATE_DMY_TEXT}|{_DATE_MDY_TEXT})",
    re.IGNORECASE,
)

_MRN_RE = re.compile(
    r"\b(?:MRN|UHID|CR\s+no\.?|"
    r"ABHA(?:\s+(?:id|number|no\b\.?|address|card))?|"
    r"medical\s+record\s+(?:number|no\b\.?|#)|"
    rf"(?:patient|hospital|registration|case|admission)\s+{_NO}|"
    r"reg\.?\s+no\b\.?|(?:IPD?|OPD?)\s+(?:no\b\.?|number))"
    rf"{_SEP}(?P<value>[A-Za-z0-9._-]+@[A-Za-z][A-Za-z.]*[A-Za-z]|{_ID_VALUE})",
    re.IGNORECASE,
)
_ABHA_NUMBER_RE = re.compile(r"(?<![\w-])\d{2}-\d{4}-\d{4}-\d{4}(?![\w-])")

_INSURANCE_RE = re.compile(
    r"\b(?:insurance(?:\s+policy)?|policy|member(?:ship)?|subscriber|claim|group|TPA|"
    r"health\s+card|PMJAY|PM-JAY|Ayushman(?:\s+Bharat)?|CGHS|ECHS|ESIC|Medicare|Medicaid)"
    rf"\b(?:\s+card)?(?:\s+{_NO})?"
    rf"{_SEP}(?P<value>{_ID_VALUE})",
    re.IGNORECASE,
)

_SSN_LABELLED_RE = re.compile(
    rf"\b(?:SSN|social\s+security(?:\s+{_NO})?)\b"
    rf"{_SEP}(?P<value>\d{{3}}[\s-]?\d{{2}}[\s-]?\d{{4}})(?!\d)",
    re.IGNORECASE,
)
_SSN_RE = re.compile(r"(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-])")

_PAN_LABELLED_RE = re.compile(
    rf"\bPAN\b(?:\s+card)?(?:\s+{_NO})?{_SEP}(?P<value>[A-Za-z]{{5}}\d{{4}}[A-Za-z])\b",
    re.IGNORECASE,
)
# Unlabelled PAN: 4th letter encodes the holder type (P = person, C = company, ...).
_PAN_RE = re.compile(r"\b[A-Z]{3}[ABCFGHJLPT][A-Z]\d{4}[A-Z]\b")

_AADHAAR_LABELLED_RE = re.compile(
    r"\b(?:aadhaa?r|adhaar|UIDAI|UID)\b(?:\s+(?:card|number|no\b\.?))*"
    rf"{_SEP}(?P<value>\d{{4}}[\s-]?\d{{4}}[\s-]?\d{{4}})(?![\w-])",
    re.IGNORECASE,
)
_AADHAAR_RE = re.compile(
    r"(?<![\w-])(?:[2-9]\d{3}(?P<sep>[\s-])\d{4}(?P=sep)\d{4}|[2-9]\d{11})(?![\w-])"
)

_PHONE_LABELLED_RE = re.compile(
    r"\b(?:phone|ph|mobile|mob|cell|tel|telephone|whatsapp|landline|"
    r"contact(?:\s+(?:no\b\.?|number))?|(?:my|his|her|their)\s+number)\b"
    r"(?:\s+(?:no\b\.?|number))?"
    rf"{_SEP}(?P<value>\+?\(?\d[\d\s().-]{{6,}}\d)(?!\d)",
    re.IGNORECASE,
)
_PHONE_IN_RE = re.compile(r"(?<![\w+])(?:(?:\+|00)91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
_PHONE_IN_LANDLINE_RE = re.compile(r"(?<![\w-])0\d{2,4}[\s-]\d{6,8}(?![\w-])")
_PHONE_US_RE = re.compile(
    r"(?<![\w-])(?:\+1[\s.-]?)?(?:\([2-9]\d{2}\)\s?|[2-9]\d{2}[.-])\d{3}[.-]\d{4}(?![\w-])"
    r"|(?<![\w-])\+1[\s.-]?[2-9]\d{2}[\s.-]\d{3}[\s.-]\d{4}(?![\w-])"
)

_STREET_WORDS = (
    r"(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Marg|Nagar|Terrace|Drive|Boulevard|Blvd|Way|"
    r"Place|Circle|Court|Ct|Highway|Hwy|Main|Cross|Layout|Colony|Enclave|Extension|Chowk|"
    r"Bazaar|Gali|Salai|Parkway|Square|Close)"
)
_ADDRESS_CUE_RE = re.compile(
    r"(?:\b(?i:flat|house|h\.?\s?no|plot|door|apt|apartment|unit|suite|villa|bungalow|shop)"
    r"\.?(?:\s*(?i:no|number)\.?)?\s*[:#]?\s*|#\s?|\bNo\.\s?)"
    r"\d+[A-Za-z]?(?:[/-]\d+[A-Za-z]?)*(?=\s*,|\s+[A-Z])"
)
_ADDRESS_STREET_RE = re.compile(
    rf"\b\d{{1,5}}[A-Za-z]?,?\s+(?:(?:[A-Z][a-z]+|\d{{1,3}}(?:st|nd|rd|th))\s+){{0,3}}"
    rf"{_STREET_WORDS}\.?(?![\w])"
)
# One comma-separated address continuation (locality, city, state).
_ADDRESS_SEGMENT_RE = re.compile(
    r",\s*(?P<seg>(?:#?\d{1,5}[A-Za-z]?(?:st|nd|rd|th)?\s+)?[A-Z][A-Za-z.'&-]*"
    r"(?:\s+(?:[A-Z][A-Za-z.'&-]*|No\.?\s?\d{1,4}[A-Za-z]?|\d{1,3}(?:st|nd|rd|th)?(?![\d/.])))*)"
)
_ADDRESS_STOP_WORDS = frozenset({
    "bp", "blood", "pulse", "hr", "heart", "age", "aged", "dob", "my", "i", "and", "but",
    "is", "was", "the", "call", "email", "phone", "mobile", "ph", "contact", "pin", "pincode",
    "mrn", "patient", "please", "thanks", "regards", "on", "with", "since", "taking",
})

# A PIN/ZIP must not continue as a decimal, date or longer number ("560001." is fine).
_NOT_MORE_DIGITS = r"(?!\d|[/.,-]\d)"
_PIN_LABELLED_RE = re.compile(
    r"\b(?:pin\s*code|pincode|pin|postal\s*code|post\s*code|zip(?:\s*code)?)"
    rf"{_SEP}(?P<value>[1-9]\d{{2}}\s?\d{{3}}|\d{{5}}(?:-\d{{4}})?){_NOT_MORE_DIGITS}",
    re.IGNORECASE,
)
_PIN_AFTER_ADDRESS_RE = re.compile(
    rf"\s*[,-]?\s*(?P<value>[1-9]\d{{5}}|\d{{5}}(?:-\d{{4}})?){_NOT_MORE_DIGITS}"
)
_PIN_IN_CITY_RE = re.compile(
    r"\b(?:in|at|near|from)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s*[,-]?\s*"
    rf"(?P<value>[1-9]\d{{2}}\s?\d{{3}}){_NOT_MORE_DIGITS}"
)

# --- names --------------------------------------------------------------------

_NAME_CUES_ANY_CASE = re.compile(
    r"\b(?:my\s+name\s+is|my\s+name's|patient(?:'s)?\s+name(?:\s+is)?|full\s+name(?:\s+is)?|"
    r"name\s*[:\-–])\s*",
    re.IGNORECASE,
)
_NAME_CUES_CAPITALISED = re.compile(
    r"(?:\b(?:Mr|Mrs|Ms|Mx|Dr|Prof|Smt)\.?\s+|\b(?:Miss|Shri|Sri|Kumari)\s+|"
    r"\b(?i:i\s+am|i'm|i’m|im)\s+|"
    r"\b(?i:patient)\s+|"
    r"\b(?i:call\s+me)\s+|"
    r"\b(?i:regards|sincerely|warm\s+regards)\s*,?\s*|"
    r"\b(?i:my|our)\s+(?i:wife|husband|son|daughter|mother|father|mom|mum|dad|brother|sister|"
    r"grandmother|grandfather|grandma|grandpa|uncle|aunt|friend|neighbou?r|doctor|"
    r"cardiologist|physician|gp|nurse|caregiver)\s*,?\s+)"
)
_NAME_CUE_THIS_IS = re.compile(r"\b(?i:this\s+is)\s+")
_THIS_IS_TRAILER = re.compile(r"\s*,?\s+(?:calling|speaking|here|from|writing)\b", re.IGNORECASE)

# A token may end in a possessive ("Sharma's") but not continue as a word.
_TOKEN_END = r"(?!\w)(?!['’][A-Za-z]{2})"
_NAME_TOKEN_CAP = re.compile(
    rf"[A-Z]\.(?=\s)|[A-Z](?:[a-z]+|'[A-Z][a-z]+)(?:-[A-Z][a-z]+)?{_TOKEN_END}"
)
_NAME_TOKEN_ANY = re.compile(
    rf"[A-Z]\.(?=\s)|[A-Za-z](?:[A-Za-z]*|'[A-Za-z]+)(?:-[A-Za-z]+)?{_TOKEN_END}"
)

_NAME_STOPWORDS = frozenset("""
i me my mine myself the a an and or but so if is am are was were be been being have has had do
does did not no yes ok okay fine good well better worse sorry here there now then today tonight
yesterday tomorrow just also still very really currently recently already again always never
in on at of for to from with by about as into after before since during over under near this
that these those it its he she they we you your his her their our what which who whom how why
when where can could should would will shall may might must please hi hello hey thanks thank
dear sir madam doctor doc nurse patient mr mrs ms dr prof
diabetic hypertensive pregnant type stage male female man woman boy girl indian american british
asian african hindu muslim christian sikh jain vegetarian vegan allergic worried concerned
confused scared afraid tired dizzy sick ill unwell healthy overweight obese old young new taking
using feeling getting going trying looking having asking writing calling wondering retired single
married active smoker nonsmoker alcoholic not
january february march april may june july august september october november december
monday tuesday wednesday thursday friday saturday sunday
bp blood pressure pulse heart sugar age aged years year education history report reports name
""".split())

_MAX_NAME_TOKENS = 4


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

Span = tuple[int, int, str, str]


def _simple(pattern: re.Pattern[str], category: str, group: str | int = 0) -> Callable[
    [str, date], Iterator[Span]
]:
    """Build a detector yielding every match (or a named group) of ``pattern``."""
    placeholder = f"[{category}]"

    def detect(text: str, _today: date) -> Iterator[Span]:
        for m in pattern.finditer(text):
            if m.group(group):
                yield m.start(group), m.end(group), category, placeholder

    return detect


def _detect_urls(text: str, _today: date) -> Iterator[Span]:
    for m in _URL_RE.finditer(text):
        value = m.group(0).rstrip(_URL_TRAILING)
        yield m.start(), m.start() + len(value), "URL", "[URL]"


def _detect_ips(text: str, today: date) -> Iterator[Span]:
    yield from _simple(_IPV4_RE, "IP")(text, today)
    yield from _simple(_IPV6_RE, "IP")(text, today)


def _parse_date(raw: str, today: date) -> date | None:
    """Parse a DOB expression (day-first when ambiguous, as used in India/UK)."""
    s = raw.strip().lower().replace(",", " ")
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})", s)
        if m:
            a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            d, mo = (a, b) if b <= 12 else (b, a)
        else:
            parts = re.findall(r"[a-z]+|\d+", s)
            nums = [int(p) for p in parts if p.isdigit()]
            month_names = [p for p in parts if p in _MONTHS]
            if len(nums) < 2 or not month_names:
                return None
            mo = _MONTHS[month_names[0]]
            d, y = nums[0], nums[-1]
        if y < 100:
            y += 1900 if y > today.year % 100 else 2000
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def _age_on(dob: date, today: date) -> int:
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


def _detect_dob(text: str, today: date) -> Iterator[Span]:
    for m in _DOB_RE.finditer(text):
        dob = _parse_date(m.group("date"), today)
        age = _age_on(dob, today) if dob else -1
        replacement = f"[AGE: {age}]" if 0 <= age <= 130 else "[AGE]"
        yield m.start("date"), m.end("date"), "AGE", replacement


def _detect_aadhaar(text: str, today: date) -> Iterator[Span]:
    yield from _simple(_AADHAAR_LABELLED_RE, "AADHAAR", "value")(text, today)
    yield from _simple(_AADHAAR_RE, "AADHAAR")(text, today)


def _detect_phones(text: str, today: date) -> Iterator[Span]:
    for m in _PHONE_LABELLED_RE.finditer(text):
        value = m.group("value")
        digits = sum(ch.isdigit() for ch in value)
        if 8 <= digits <= 13:
            yield m.start("value"), m.end("value"), "PHONE", "[PHONE]"
    for pattern in (_PHONE_IN_RE, _PHONE_US_RE, _PHONE_IN_LANDLINE_RE):
        yield from _simple(pattern, "PHONE")(text, today)


def _extend_address(text: str, end: int) -> int:
    """Extend an address core over comma-separated locality/city/state segments."""
    for _ in range(6):
        m = _ADDRESS_SEGMENT_RE.match(text, end)
        if not m:
            break
        first_word = re.split(r"[\s.]", m.group("seg"), maxsplit=1)[0].lower()
        if first_word in _ADDRESS_STOP_WORDS:
            break
        end = m.end("seg")
    while end > 0 and text[end - 1] in ".,":
        end -= 1
    return end


def _detect_addresses(text: str, _today: date) -> Iterator[Span]:
    for m in _ADDRESS_CUE_RE.finditer(text):
        end = _extend_address(text, m.end())
        if end > m.end():  # a house-number cue needs at least one locality segment
            yield m.start(), end, "ADDRESS", "[ADDRESS]"
    for m in _ADDRESS_STREET_RE.finditer(text):
        yield m.start(), _extend_address(text, m.end()), "ADDRESS", "[ADDRESS]"


def _detect_pincodes(text: str, today: date) -> Iterator[Span]:
    yield from _simple(_PIN_LABELLED_RE, "PINCODE", "value")(text, today)
    yield from _simple(_PIN_IN_CITY_RE, "PINCODE", "value")(text, today)


def _is_initial(token: str) -> bool:
    return len(token) == 2 and token.endswith(".")


def _consume_name(text: str, pos: int, any_case: bool) -> tuple[int, int] | None:
    """Consume up to four space-separated name tokens at ``pos``; return the span or None."""
    token_re = _NAME_TOKEN_ANY if any_case else _NAME_TOKEN_CAP
    tokens: list[tuple[str, int, int]] = []
    cursor = pos
    while len(tokens) < _MAX_NAME_TOKENS:
        m = token_re.match(text, cursor)
        if not m:
            break
        token = m.group(0)
        if not _is_initial(token) and token.lower() in _NAME_STOPWORDS:
            break
        tokens.append((token, m.start(), m.end()))
        if text[m.end():m.end() + 1] != " ":
            break
        cursor = m.end() + 1
    while tokens and _is_initial(tokens[-1][0]):
        tokens.pop()
    if not tokens:
        return None
    return tokens[0][1], tokens[-1][2]


def _detect_names(text: str, _today: date) -> Iterator[Span]:
    for cue_re, any_case in ((_NAME_CUES_ANY_CASE, True), (_NAME_CUES_CAPITALISED, False)):
        for m in cue_re.finditer(text):
            span = _consume_name(text, m.end(), any_case)
            if span:
                yield span[0], span[1], "NAME", "[NAME]"
    for m in _NAME_CUE_THIS_IS.finditer(text):
        span = _consume_name(text, m.end(), any_case=False)
        if span and _THIS_IS_TRAILER.match(text, span[1]):
            yield span[0], span[1], "NAME", "[NAME]"


# Priority order: earlier detectors win when spans overlap.
_DETECTORS: tuple[Callable[[str, date], Iterator[Span]], ...] = (
    _detect_urls,
    _simple(_EMAIL_RE, "EMAIL"),
    _detect_ips,
    _detect_dob,
    _simple(_MRN_RE, "MRN", "value"),
    _simple(_INSURANCE_RE, "INSURANCE_ID", "value"),
    _simple(_SSN_LABELLED_RE, "SSN", "value"),
    _simple(_PAN_LABELLED_RE, "PAN", "value"),
    _simple(_ABHA_NUMBER_RE, "MRN"),
    _detect_aadhaar,
    _simple(_SSN_RE, "SSN"),
    _simple(_PAN_RE, "PAN"),
    _detect_phones,
    _detect_addresses,
    _detect_pincodes,
    _detect_names,
)


def detect(text: str, *, today: date | None = None) -> list[PiiSpan]:
    """Return non-overlapping PII spans in ``text`` (sorted by position)."""
    if not text:
        return []
    ref = today or date.today()
    accepted: list[PiiSpan] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < s.end and s.start < end for s in accepted)

    for detector in _DETECTORS:
        for start, end, category, replacement in detector(text, ref):
            if end > start and not overlaps(start, end):
                accepted.append(PiiSpan(start, end, category, replacement))
        if detector is _detect_addresses:
            accepted.extend(_pins_after_addresses(text, accepted))
    return sorted(accepted, key=lambda s: s.start)


def _pins_after_addresses(text: str, accepted: list[PiiSpan]) -> list[PiiSpan]:
    """PIN/ZIP codes that directly follow a detected address."""
    found: list[PiiSpan] = []
    for span in accepted:
        if span.category != "ADDRESS":
            continue
        m = _PIN_AFTER_ADDRESS_RE.match(text, span.end)
        if m and not any(m.start("value") < s.end and s.start < m.end("value") for s in accepted):
            found.append(PiiSpan(m.start("value"), m.end("value"), "PINCODE", "[PINCODE]"))
    return found


def redact(text: str, *, today: date | None = None) -> RedactionResult:
    """Replace PII/PHI identifiers in ``text`` with typed placeholders.

    Args:
        text: Untrusted free text (patient message, feedback comment, log value).
        today: Reference date for DOB → age conversion (defaults to today).

    Returns:
        ``RedactionResult`` with the redacted text, the number of replaced
        spans and a per-category count. Text without PII is returned unchanged.
    """
    spans = detect(text, today=today)
    if not spans:
        return RedactionResult(text=text)
    parts: list[str] = []
    cursor = 0
    for span in spans:
        parts.append(text[cursor:span.start])
        parts.append(span.replacement)
        cursor = span.end
    parts.append(text[cursor:])
    counts = Counter(s.category for s in spans)
    return RedactionResult(text="".join(parts), count=len(spans), categories=dict(counts))
