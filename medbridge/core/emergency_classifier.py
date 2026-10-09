"""
Deterministic natural-language emergency classifier (ADL-002 + finalization NL extension).

The fast-path triage stage of the pipeline: it runs on the **raw** patient
message **before any LLM call** and returns ``ActionEnum.ESCALATE`` for
hypertensive crises, red-flag symptoms and acute danger. It is pure Python —
regular expressions compiled at import time plus a few dictionary look-ups — so
it is deterministic, needs no network and no model, and runs in well under the
10 ms budget (typically < 2 ms for a 2,000-character message).

Processing steps
----------------
1. **Normalisation** (:func:`normalize_text`): NFKC, unified apostrophes/dashes,
   lower case, letter runs longer than two collapsed (``"dizzzzy"`` →
   ``"dizzy"``), ``b.p.`` → ``bp``, a fixed misspelling map (``chset`` →
   ``chest``, ``breathe`` → ``breath``, ``headace`` → ``headache``, ``presure``
   → ``pressure``, ``cant`` → ``can't`` …) and a deterministic edit-distance-1
   matcher (Damerau/OSA: one insertion, deletion, substitution or adjacent
   transposition) for red-flag keywords of length ≥ 6. The matcher is a
   pre-computed variant → keyword table, so a look-up is O(1); variants that
   are ordinary English words (``painting``, ``blurred``, ``lightness`` …)
   are excluded so they are never "corrected" into a symptom.
2. **Blood pressure** extraction: ``210/125``, ``190 over 125``, ``185 by
   110``, ``systolic 190``, ``top number 200``, ``bp of 185``, ``my bp is over
   180`` (strictly greater → 181), ``bp 200+``, ``pressure is like 200
   something``, ``in the 190s``, ``it hit 200`` … with sanity checks
   (systolic 50–300, diastolic 20–200, systolic > diastolic; no decimals, no
   dates such as ``12/10/2024``, no fractions/doses such as ``1/2 tablet`` or
   ``180 mg``, no lab values, heart rates, targets such as ``below 130/80`` or
   conditional thresholds such as ``call if it is over 180/110``). A pair typed
   in the wrong order next to BP wording (``bp 120/190``) is read swapped.
   Crisis: systolic ≥ 180 **or** diastolic ≥ 120 when the message mentions
   blood pressure; a bare number pair with no BP wording must meet **both**
   thresholds (avoids order numbers, room numbers …). A systolic < 80 with
   dizziness/faintness/confusion is also escalated (``bp_low_symptomatic``).
3. **Red-flag rules**: one compiled regex per rule id (see :data:`RULE_IDS`).
4. **Context filters** applied per clause (clause boundaries: ``. ; ? !``,
   newline, *but*, *however*, *although*, *though*, *whereas*):

   * **Negation** — an explicit cue (*no, not, never, without, denies, no
     longer, free of, haven't had, don't feel* …) before the symptom in the
     same clause, with no affirmation in between (*I have*, *my …*, *now*,
     *suddenly* …). Lists are followed (``no chest pain, shortness of breath or
     fainting``) only when the first negated item is itself a red flag, so
     ``no fever, chest pain`` still escalates. Answer-style post-negation
     (``chest pain? no``, ``chest pain: none``) is supported. Pseudo-negations
     (*not sure*, *no relief*, *not going away*, *never felt this bad* …) do
     not negate.
   * **Informational** questions (*what are the warning signs of …*, *how do I
     recognise …*, *what should I do if someone faints*, *if I ever get …*) —
     unless the message also carries a first-person present marker (*I'm
     having*, *right now*, *my chest is …*, *I have … pain*).
   * **Historical** mentions with an explicit time anchor (*last year*, *in
     2015*, *years ago*, *as a child*, *history of*, *used to*, *it turned out
     to be reflux* …) and no recency marker in the same clause (*now*,
     *today*, *again*, *still*, *since*, *an hour ago* …).
   * **Heavy exertion** (breathlessness only): *short of breath when I run /
     at the gym* is not a red flag; walking, stairs, rest or lying down are.

   A negated, informational or historical symptom never suppresses a BP crisis
   or another affirmed symptom (``no chest pain but my bp is 190/125`` →
   ESCALATE). BP readings are only filtered by historical anchors, targets and
   conditional thresholds.

5. **Hinglish / Indian-English lexicon** (ADL: Hinglish red-flag lexicon). Roman-
   script Hindi spelling variants are canonicalised token by token
   (:data:`_HINGLISH_VARIANTS`: ``sine/seeney`` → ``seene``, ``nhi/nai/nahin`` →
   ``nahi``, ``sans/swas`` → ``saans``, ``hogayi`` → ``ho gayi`` …), then 14
   rules with ``_hi`` ids (:data:`HINGLISH_RULE_IDS`) match chest pain
   (``seene mein dard``, ``chhati mein jalan``), heart attack (``dil ka daura
   pada``), breathlessness (``saans nahi aa rahi``, ``dam ghut raha``), syncope
   (``behosh ho gayi``, ``chakkar aake gir gaya``), stroke signs (``lakwa maar
   gaya``, ``muh tedha``, ``haath nahi hil raha``, ``zubaan ladkhada rahi``),
   severe headache (``bahut tez sar dard``, ``sar phat raha``), blood in cough
   or vomit, seizure and suicidal ideation (``marna chahta hoon``). Hindi
   negation follows the symptom (``seene mein dard nahi hai``), so a
   post-position scope is added: ``nahi`` within a short allow-listed gap after
   the match, closing the statement (``nahi ja raha`` / ``kam nahi ho raha`` —
   not going away / not easing — do not negate), extended over ``aur`` / ``ya``
   lists of red flags, voided by ``pehle kabhi nahi`` (never before) and
   cancelled by a later ``aaj ho raha hai``. ``na to X na Y``, ``lekin`` /
   ``magar`` clause boundaries, Hindi informational frames (``... ke lakshan kya
   hain``, ``agar kisi ko ...``), historical anchors (``pichle saal``, ``2 saal
   pehle``, ``... hua tha``), Hindi BP wording (``bp bahut high hai 200``,
   ``upar wala``, ``180 se upar`` thresholds) and exertion words are handled by
   the same filters. Every Hinglish pattern needs a Hindi token or an
   Indian-English idiom (``chest is paining``), so English messages behave
   exactly as before.

Context-handling decisions (safety first: when ambiguous, keep recall)
----------------------------------------------------------------------
=======================================  =========================================  ========
Mention                                  Example                                    Decision
=======================================  =========================================  ========
First person, present symptom            "my chest feels tight"                     ESCALATE
First person, recent past (no anchor)    "I passed out twice"                       ESCALATE
Third person, present (bystander)        "my mother is unconscious"                 ESCALATE
Side-effect / can-X-cause-Y question     "is chest pain a side effect of ...?"      ESCALATE
Personal conditional question            "how do I know if I'm having a stroke"     ESCALATE
BP crisis value, question or statement   "is 190/125 dangerous?"                    ESCALATE
Negated symptom                          "no chest pain", "denies dizziness"        no
Negated symptom + BP crisis              "no chest pain but my bp is 190/125"       ESCALATE
Educational / generic-person question    "what are the warning signs of a stroke?"  no
Own hypothetical future                  "if I ever get chest pain ..."             no
Historical, time-anchored (any person)   "my father had a stroke in 2015"           no
Historical + recency in the same clause  "chest pain last year, back now"           ESCALATE
Breathlessness with heavy exertion only  "out of breath when I go jogging"          no
Bare number pair below dual threshold    "order number 180/90"                      no
Conditional threshold (if ... over N)    "call if my BP is above 180"               no
=======================================  =========================================  ========

Technical Specification Part III §3.2; constitution §12 NON-NEGOTIABLE 1.
"""
from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass, field
from functools import cached_property

from medbridge.api.schemas.enums import ActionEnum

__all__ = [
    "DIASTOLIC_CRISIS",
    "HINGLISH_RULE_IDS",
    "RULE_IDS",
    "SYSTOLIC_CRISIS",
    "SYSTOLIC_LOW",
    "EmergencyResult",
    "classify",
    "classify_detailed",
    "normalize_text",
]

SYSTOLIC_CRISIS = 180
DIASTOLIC_CRISIS = 120
SYSTOLIC_LOW = 80


@dataclass(frozen=True)
class EmergencyResult:
    """Detailed outcome of :func:`classify_detailed`.

    Attributes:
        action: ``ActionEnum.ESCALATE`` when at least one rule fired, else ``None``.
        matched_rules: Stable rule ids that fired, in :data:`RULE_IDS` order.
        negated: Rule ids whose only matches were explicitly negated.
        bp: The decisive blood-pressure reading as ``(systolic, diastolic)``;
            a component that was not stated is ``0``. ``None`` when no
            plausible reading was found.
        evidence_spans: Normalised text spans that triggered the escalation.
        suppressed: Rule ids matched only in informational, historical or
            heavy-exertion context (kept for the audit trace).
    """

    action: ActionEnum | None
    matched_rules: list[str] = field(default_factory=list)
    negated: list[str] = field(default_factory=list)
    bp: tuple[int, int] | None = None
    evidence_spans: list[str] = field(default_factory=list)
    suppressed: list[str] = field(default_factory=list)


#: Generic one-word filler slots and their negation-guarded form: a filler never
#: absorbs "not"/"no"/"never", so "my speech is not slurred" or "I'm not having a
#: stroke" cannot match a symptom pattern that skips one or more words.
_GUARDED_FILLERS: tuple[tuple[str, str], ...] = (
    (r"(?:\w+\s+)", r"(?:(?!(?:not|no|never|nahi)\b)\w+\s+)"),
    (r"(?:\s+\w+)", r"(?:\s+(?!(?:not|no|never|nahi)\b)\w+)"),
    (r"(?:[\s,]+\w+)", r"(?:[\s,]+(?!(?:not|no|never|nahi)\b)\w+)"),
)


def _rx(*alternatives: str) -> re.Pattern[str]:
    """Compile alternatives into one pattern (text is already lower case).

    Generic filler slots are rewritten to their negation-guarded form (see
    :data:`_GUARDED_FILLERS`).
    """
    pattern = "|".join(f"(?:{alt})" for alt in alternatives)
    for generic, guarded in _GUARDED_FILLERS:
        pattern = pattern.replace(generic, guarded)
    return re.compile(pattern)


# ======================================================================
# 1. Normalisation
# ======================================================================

_CHAR_MAP = str.maketrans(
    {
        "\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u2032": "'", "`": "'", "\u00b4": "'",
        "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2212": "-",
    }
)
_REPEAT_RE = re.compile(r"([a-z])\1{2,}")
_BP_ABBREV_RE = re.compile(r"\bb\s?\.\s?p\b\.?")
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
_SPACE_RE = re.compile(r"[^\S\n]+")

#: Deterministic misspelling / contraction map applied token by token.
_TYPO_MAP: dict[str, str] = {
    # chest
    "chset": "chest", "chets": "chest", "chesst": "chest", "chect": "chest", "chst": "chest",
    "cheast": "chest", "chestt": "chest", "chesr": "chest", "chestpain": "chest pain",
    "chestpains": "chest pains",
    # breath / breathing
    "breathe": "breath", "breth": "breath", "brath": "breath", "breaht": "breath",
    "braeth": "breath", "breat": "breath", "breeth": "breath", "brethe": "breath",
    "breathin": "breathing", "brething": "breathing", "breating": "breathing",
    "breahting": "breathing", "breathng": "breathing", "shortnes": "shortness",
    # numbness / dizziness
    "numbnes": "numbness", "numbess": "numbness", "numness": "numbness",
    "numbneess": "numbness", "dizy": "dizzy", "dizzi": "dizzy", "dizzey": "dizzy",
    "dizi": "dizzy", "dizzie": "dizzy", "dizzyy": "dizzy", "diziness": "dizziness",
    "dizzyness": "dizziness", "dizzines": "dizziness", "dizzness": "dizziness",
    "dizyness": "dizziness", "dizzynes": "dizziness",
    # fainting / stroke / seizure
    "fanted": "fainted", "fainterd": "fainted", "fainetd": "fainted", "fiainted": "fainted",
    "faintd": "fainted", "fainded": "fainted", "feinted": "fainted", "fanting": "fainting",
    "fant": "faint", "fiant": "faint", "stroek": "stroke", "strok": "stroke",
    "storke": "stroke", "stoke": "stroke", "siezure": "seizure", "seazure": "seizure",
    "sezure": "seizure", "unconcious": "unconscious",
    # headache / pain / tight
    "headace": "headache", "hedache": "headache", "headach": "headache",
    "hedake": "headache", "headche": "headache", "haedache": "headache",
    "headake": "headache", "headahce": "headache", "headachee": "headache",
    "hedach": "headache", "pian": "pain", "painn": "pain", "tite": "tight",
    # pressure / blood pressure / bp
    "presure": "pressure", "pressur": "pressure", "preasure": "pressure",
    "pressue": "pressure", "pressre": "pressure", "prssure": "pressure",
    "preassure": "pressure", "pressuer": "pressure", "bloodpressure": "blood pressure",
    "sistolic": "systolic", "systolik": "systolic", "diastolik": "diastolic",
    "distolic": "diastolic", "diastollic": "diastolic",
    # contractions written without an apostrophe
    "cant": "can't", "couldnt": "couldn't", "dont": "don't", "doesnt": "doesn't",
    "didnt": "didn't", "havent": "haven't", "hasnt": "hasn't", "isnt": "isn't",
    "wasnt": "wasn't", "wont": "won't", "im": "i'm", "ive": "i've",
}

#: Red-flag keywords (length >= 6) eligible for edit-distance-1 correction.
_FUZZY_KEYWORDS: tuple[str, ...] = (
    "breath", "breathing", "breathless", "breathlessness", "shortness", "headache",
    "headaches", "numbness", "dizziness", "pressure", "seizure", "seizures", "unconscious",
    "unresponsive", "consciousness", "collapse", "collapsed", "thunderclap", "convulsion",
    "convulsions", "convulsing", "tightness", "squeezing", "heaviness", "discomfort",
    "palpitation", "palpitations", "suicide", "suicidal", "disoriented", "slurred",
    "slurring", "drooping", "fainting", "paralysed", "paralyzed", "paralysis", "weakness",
    "vision", "speech", "pregnant", "coughing", "sputum", "frothy", "blindness",
    "epigastric", "systolic", "diastolic",
)

#: Real English words one edit away from a keyword — never "corrected" (checked
#: against a 236k-word English dictionary at development time; every entry here
#: is an actual edit-distance-1 variant of a keyword above).
_FUZZY_STOPWORDS: frozenset[str] = frozenset(
    {
        "asystolic", "blandness", "blurred", "blurring", "breach", "breadth", "breadthless",
        "couching", "drooling", "dropping", "dumbness", "fizziness", "headiness", "lightness",
        "painting", "pressured", "pressures", "regnant", "rightness", "roughing", "scutum",
        "slurped", "slurping", "soughing", "spurred", "spurring", "tainting", "toughing",
        "trooping", "wreath", "wreathless",
    }
)


def _edit1_variants(word: str) -> set[str]:
    """Return every string at OSA distance exactly 1 from ``word`` (a–z alphabet)."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = {left + right[1:] for left, right in splits if right}
    transposes = {
        left + right[1] + right[0] + right[2:] for left, right in splits if len(right) > 1
    }
    replaces = {left + c + right[1:] for left, right in splits if right for c in letters}
    inserts = {left + c + right for left, right in splits for c in letters}
    return (deletes | transposes | replaces | inserts) - {word}


def _build_fuzzy_index(keywords: tuple[str, ...], stopwords: frozenset[str]) -> dict[str, str]:
    """Pre-compute the variant → keyword table (first keyword wins on a tie)."""
    protected = set(keywords) | stopwords
    index: dict[str, str] = {}
    for keyword in keywords:
        for variant in _edit1_variants(keyword):
            if variant not in protected:
                index.setdefault(variant, keyword)
    return index


_FUZZY_INDEX: dict[str, str] = _build_fuzzy_index(_FUZZY_KEYWORDS, _FUZZY_STOPWORDS)

#: Roman-script Hindi (Hinglish) spelling variants → one canonical spelling, so the
#: Hinglish rules below match a single form. Canonical forms map to themselves,
#: which also protects them from the English edit-distance corrector. Variants
#: that are also English dictionary entries were reviewed at development time and
#: kept only when they are rare in patient English (``sine``, ``sans``, ``hath``,
#: ``dora``, ``khasi`` …). Ambiguous short words (``me``, ``main``, ``sir``, ``fat``,
#: ``pair``) are never rewritten; the patterns list them as alternatives instead.
_HINGLISH_VARIANTS: dict[str, tuple[str, ...]] = {
    "nahi": ("nahin", "nahee", "nahii", "nahiin", "nhi", "nhin", "nai", "ni", "nehi", "nahe",
             "nhai"),
    "seene": ("sine", "seenay", "seeney", "siney", "sinay", "seenai", "sene"),
    "seena": ("sina", "seenaa", "sinaa"),
    "chhati": ("chati", "chhaati", "chaati", "chhathi", "chathi", "chhatee", "chatee", "chhatii",
               "chhatti", "chatti", "chhaatee", "chaatee"),
    "dard": ("dardh", "darad", "durd", "drd"),
    "jalan": ("jalun", "jalaan", "jlan"),
    "bharipan": ("bhaaripan", "bharipun", "bharippan", "bhaaripun"),
    "bhaari": ("bhari", "bharee", "bhaaree", "bhary", "bhaary"),
    "dabav": ("dabaav", "dabaw", "dabaaw", "dabao", "dabaao", "dbav"),
    "takleef": ("taklif", "takleeph", "taqleef", "takleif", "taklef", "takliif", "taklip"),
    "dikkat": ("dikat", "diqqat", "dikkt", "dikkath", "diqat", "dikkatt"),
    "pareshani": ("preshani", "pareshaani", "pershani", "pareshanee", "prashani", "pareshanii"),
    "saans": ("sans", "saas", "saanse", "sanse", "saanso", "sanso", "swas", "swaas", "shwas",
              "shwaas", "shvas", "svas", "saanss"),
    "dam": ("dum", "dumm"),
    "ghutan": ("ghuttan", "ghootan", "ghutaan"),
    "behosh": ("behos", "behoosh", "beyhosh", "behoss", "behush", "beohosh"),
    "behoshi": ("behosi", "behoshee", "beyhoshi", "behooshi", "behushi"),
    "chakkar": ("chakar", "chakker", "chakkr", "chakr", "chhakar", "chakkur"),
    "hosh": ("hos", "hoss"),
    "lakwa": ("lakva", "lakhwa", "lakhva", "laqwa", "lakwaa", "lakvaa", "lakuva", "lakawa",
              "lakuwa", "lakhwaa"),
    "lakwe": ("lakve", "lakhwe", "lakhve", "laqwe"),
    "tedha": ("terha", "teda", "tedhaa", "terdha", "tedhi", "terhi", "tedi", "tedhe", "terhe",
              "teddha", "tedda"),
    "haath": ("hath", "haat", "haathh", "hathh"),
    "baazu": ("bazu", "baju", "baaju", "baazoo", "bajoo", "bazoo", "baanh", "banh", "baahu"),
    "kamzori": ("kamjori", "kamzoori", "kamjoori", "kamzory", "kamzri"),
    "kamzor": ("kamjor", "kamzoor", "kamjoor"),
    "chehra": ("chehera", "chera", "cehra", "chahra", "chehraa", "chehara"),
    "chehre": ("chere", "chehray", "cehre", "chehere", "chahre"),
    "muh": ("munh", "muhh", "moonh", "mooh", "muhn"),
    "honth": ("hont", "hoth", "hoont"),
    "aawaz": ("awaaz", "awaz", "aawaaz", "avaaz", "awaj", "aawaj", "avaz", "aavaz", "awaaj"),
    "zubaan": ("zuban", "jubaan", "juban", "zaban", "zabaan", "jabaan", "jaban"),
    "jeebh": ("jeeb", "jibh"),
    "tez": ("tej", "tezz", "taiz", "tejj", "teez"),
    "bahut": ("bohot", "bahot", "bhot", "bht", "bohat", "bhut", "bhaut", "boht", "bahuth", "bohut",
              "bahoot"),
    "khoon": ("khun", "khuun", "khoonn"),
    "ulti": ("ulty", "ultee", "ulati", "oolti", "ultii"),
    "ultiyan": ("ultiyaan", "ultiya", "ultiyon", "ultian"),
    "khansi": ("khaansi", "khasi", "khaasi", "khaansee", "khansee", "khanshi", "khaasee"),
    "balgam": ("balgham", "balgum", "bulgum", "balgaam"),
    "jeena": ("jina", "jeenaa", "jiina", "jinaa"),
    "jeene": ("jine", "jiine", "jeeney"),
    "mujhe": ("muje", "mujhay", "mjhe", "mujhey", "muze"),
    "abhi": ("abi", "abhee", "abhie", "abhii"),
    "achanak": ("achaanak", "acchanak", "achnak", "achank", "achaank"),
    "dhadkan": ("dhadkun", "dharkan", "dhadhkan", "dhadkane"),
    "mirgi": ("mirgee", "mirgii"),
    "jhatke": ("jhatka", "jhatkey", "jatke", "jhatkay", "jhatkaa"),
    "daura": ("dauraa", "dora", "daure"),
    "lakshan": ("laxan", "lakshn", "lachhan", "laxshan", "lakshanon", "lakshano", "lakshaan"),
    "pichle": ("pichhle", "pichley", "pichhley", "pichli", "pichhli"),
    "pehle": ("pahle", "pehele", "phle", "pehley", "pahley", "pehlay", "pahele", "pehale"),
    "lekin": ("lekn", "lkn", "laikin", "lakin"),
    "kya": ("kyaa", "kia"),
    "kaise": ("kese", "kaisey", "kaese", "kaisay"),
    "theek": ("thik", "teek", "thiik"),
    "bilkul": ("bilkol", "bikul"),
    "koi": ("koii", "koee"),
    "raha": ("rha", "rahaa", "rhaa"),
    "rahi": ("rhi", "rahee", "rhee", "rahii"),
    "rahe": ("rhe", "rahey"),
    "gaya": ("gya", "gayaa", "gyaa"),
    "gayi": ("gyi", "gai", "gayee", "gayii", "gaee"),
    "gaye": ("gye", "gayey"),
    "hoon": ("hu", "hun", "huun"),
    "chahta": ("chahata", "chahtaa", "chahtha"),
    "chahti": ("chahati", "chahtii"),
    "khudkushi": ("khudkhushi", "khudkhusi", "khudkusi"),
    "aatmhatya": ("atmhatya", "aatmahatya", "atmahatya"),
    "khatam": ("khatm", "khtm", "khattam"),
    "aankh": ("ankh", "aakh"),
    "aankhon": ("ankhon", "aankho", "ankho"),
    "andhera": ("andhra", "andheraa", "andhere"),
    # chat-style fused auxiliaries ("hogaya", "horahi") are split into two words
    "ho gaya": ("hogaya", "hogya", "hogayaa"),
    "ho gayi": ("hogayi", "hogyi", "hogai", "hogayee"),
    "ho gaye": ("hogaye", "hogye"),
    "ho raha": ("horaha", "horha"),
    "ho rahi": ("horahi", "horhi"),
    "ho rahe": ("horahe", "horhe"),
    "aa raha": ("araha", "aaraha", "aarha"),
    "aa rahi": ("arahi", "aarahi", "aarhi"),
}
_HINGLISH_MAP: dict[str, str] = {
    variant: canonical
    for canonical, variants in _HINGLISH_VARIANTS.items()
    for variant in (canonical, *variants)
    if " " not in variant
}


def _correct_token(match: re.Match[str]) -> str:
    token = match.group(0)
    mapped = _TYPO_MAP.get(token) or _HINGLISH_MAP.get(token)
    if mapped is not None:
        return mapped
    if len(token) < 5:
        return token
    return _FUZZY_INDEX.get(token, token)


def normalize_text(text: str) -> str:
    """Normalise a raw message for deterministic rule matching.

    Args:
        text: Raw patient message.

    Returns:
        Lower-cased text with unified punctuation, collapsed letter runs,
        corrected misspellings and single spaces (newlines are kept because
        they are clause boundaries).
    """
    text = unicodedata.normalize("NFKC", text).translate(_CHAR_MAP).lower()
    text = _REPEAT_RE.sub(r"\1\1", text)
    text = _BP_ABBREV_RE.sub("bp", text)
    text = _WORD_RE.sub(_correct_token, text)
    return _SPACE_RE.sub(" ", text).strip()


# ======================================================================
# 2. Clause analysis (negation, informational, historical, exertion)
# ======================================================================

_CLAUSE_BOUNDARY_RE = re.compile(
    r"[.;?!\n]+(?!\d)|\b(?:but|however|although|though|whereas|lekin|magar|parantu|kintu"
    r"|balki)\b"
)

_NEG_CUE_RE = _rx(
    r"\b(?:no longer|no|not|never|without|w/o|denies|deny|denied|denying|none|nor|neither"
    r"|free of|absence of|negative for|ruled out|rules out|rule out)\b",
    r"\bna\s+(?:to|toh|hi)\b",
    r"\b(?:do|does|did|have|has|had|is|are|was|were|could|would|should|ca|wo|ai)n't\s+"
    r"(?:(?:really|ever|even|actually|currently|now)\s+)?(?:have|has|had|having|get|got|getting"
    r"|feel|feeling|felt|experienc\w*|notic\w*|suffer\w*|want|think|faint\w*|pass\w*|black\w*"
    r"|collaps\w*|lose|lost|cough\w*|see|seem|hurt|hurts|ache|aches|been)\b",
)
_PSEUDO_NEG_RE = _rx(
    r"\b(?:no doubt|no idea|no clue|not sure|not certain|not quite sure|not only|not just"
    r"|no matter|no relief|no improvement|not improving|not better|no better|not going away"
    r"|not getting better|not stopping|no change|not helping|no help|not normal|not know"
    r"|not clear|no one|nobody)\b",
    r"n't know\b",
)
_NEG_VOID_RE = _rx(
    r"\b(?:like this|this bad|this severe|this strong|this intense|this painful|as bad as this"
    r"|before now|until now|until today|first time)\b",
    r"\b(?:pehle\s+kabhi|kabhi\s+aisa|aisa\s+(?:\w+\s+)?(?:pehle|kabhi)|pehli\s+baar"
    r"|is\s+tarah\s+ka)\b",
)
_NEG_HARD_TERM_RE = _rx(
    r"\b(?:but|however|although|though|yet|except|apart from|other than|aside from|besides"
    r"|now|still|suddenly|today|tonight|currently|instead|which|who|whereas)\b",
    r"\b(?:i|we|you|my|his|her|their|it|it's|its|there's|he|she|he's|she's|they|they're|this"
    r"|that)\b",
    r"\b(?:hai|hain|tha|thi|raha|rahi|rahe|lekin|magar|mujhe|mera|meri|mere|abhi|achanak"
    r"|aaj)\b",
)
_NEG_SOFT_TERM_RE = _rx(r",|&|\b(?:and|with|plus|also|then|aur|ya|tatha)\b")
_POST_NEG_INLINE_RE = re.compile(
    r"\s*[:=\-]+\s*(?:no|none|nope|negative|denied|absent|nil|nahi)\b"
)
_POST_NEG_CLAUSE_RE = re.compile(
    r"^\W*(?:no|nope|none|nah|negative|not really|not at all|never|denied|denies|absent|nil"
    r"|nahi|nahi ji|bilkul nahi)\W*$"
)
_NEG_WINDOW_CHARS = 80

# --- Hinglish negation: the cue ("nahi") FOLLOWS the symptom ---------------------
#: "na to X na Y": after a "na to / na hi" cue a bare "na" is a cue too.
_HI_NEITHER_RE = re.compile(r"\bna\s+(?:to|toh|hi)\b")
_HI_NA_RE = re.compile(r"\bna\b")
_HI_POST_GAP = (
    r"(?:(?:koi|bilkul|bhi|to|toh|hi|ab|abhi|kabhi|ekdam|zara|jara|kuch|waisa|aisa|wala|wali"
    r"|ka|ki|ke|filhaal|aajkal|shikayat|samasya|problem|pareshani|takleef|dikkat|issue"
    r"|complaint)\s+){0,4}"
)
#: "nahi" must close the statement ("dard nahi hai", "dard nahi.") — "dard nahi ja
#: raha" (not going away) and "kam nahi ho raha" (not easing) are not negations.
_HI_POST_CUE = (
    r"nahi\b(?=\s*(?:$|[^\w\s]|(?:hai|hain|h|he|tha|thi|the|hua|hui|hue|hota|hoti|hote|ho|hoon"
    r"|raha|rahi|rahe|lag\w*|mehsoos|feel|aa|aaya|aayi|aata|aati|aate|pada|padi|pade|ab|abhi"
    r"|aaj|bilkul|ji|sir|doctor|lekin|magar|aur|or|and|to|toh|kuch|bas|yaar|na|mujhe|par|pe)\b))"
)
_HI_POST_NEG_RE = re.compile(r"\s+" + _HI_POST_GAP + _HI_POST_CUE)
#: List separators for "X aur/ya Y nahi" (the negation scopes over the list).
_HI_LIST_SEP_RE = re.compile(
    r"\s*(?:,|&|/|\b(?:aur|ya|or|and|tatha|na)\b)\s*(?:(?:na|koi|ki|ka|ke|bhi|phir)\s+)*"
)
#: The last list item may be a non-red-flag word or two ("... ya chakkar nahi").
_HI_TAIL_ITEM_NEG_RE = re.compile(
    r"(?:(?!(?:hai|hain|h|tha|thi|ho|hota|hoti|hua|hui|raha|rahi|rahe|aa|gaya|gayi|lekin"
    r"|magar|mujhe|mera|meri|mere|abhi)\b)[a-z']+\s+){1,3}" + _HI_POST_GAP + _HI_POST_CUE
)
#: "... kabhi nahi hua tha, aaj ho raha hai": a later present affirmation cancels it.
_HI_REAFFIRM_RE = re.compile(
    r"\b(?:ab|abhi|aaj)\b(?:\s+\w+){0,2}?\s+(?:ho\s+(?:raha|rahi|rahe)|aa\s+(?:raha|rahi|rahe)"
    r"|shuru)\b"
)
_HI_LIST_MAX = 4

#: Educational question frames only — broad topical words ("exam", "in women")
#: are deliberately not cues, so they can never hide a personal statement.
_INFO_CUE_RE = _rx(
    r"\b(?:what|which)(?:'s|\s+(?:is|are|were|was))\s+(?:the\s+|some\s+)?(?:\w+\s+){0,2}?"
    r"(?:signs?|symptoms?|indicators?|red flags?|types?|causes?)\b",
    r"\b(?:what|which)\s+(?:\w+\s+){0,2}?(?:signs?|symptoms?)\s+(?:of|for|should|to|indicate"
    r"|mean|are)\b",
    r"\bhow\s+(?:do|can|would|could|should|will|does|did)\s+(?:i|you|we|one|someone|somebody"
    r"|people|a person|anyone)\s+(?:\w+\s+)?(?:recogni[sz]e|identify|spot|tell|detect|know"
    r"|notice)\b",
    r"\bhow to\s+(?:recogni[sz]e|identify|spot|tell|detect|know)\b",
    r"\b(?:tell|teach|explain|show|list|give)\s+(?:me\s+|us\s+)?(?:about\s+)?(?:the\s+|some\s+)?"
    r"(?:\w+\s+){0,2}?(?:signs?|symptoms?)\b",
    r"\b(?:signs|symptoms)\s+(?:and|&)\s+(?:signs|symptoms)\b",
    r"\b(?:if|when|in case)\s+my\s+(?:husband|wife|partner|mother|mom|mum|father|dad|child|son"
    r"|daughter|friend|parents?|grandmother|grandfather|grandma|grandpa|baby|brother|sister"
    r"|relative)\s+(?:ever\s+)?(?:gets|has|starts|develops|faints|collapse|collapses|passes"
    r"|goes|experiences|shows)\b",
    r"\b(?:if|when|in case)\s+(?:someone|somebody|a person|people|anyone|a patient|patients"
    r"|you)\s+(?:\w+\s+)?(?:is|are|'s|has|have|gets?|starts?|suddenly|experiences?|develops?"
    r"|shows?|faints?|collapses?|passes|goes)\b",
    r"^\s*what(?:'s| is| are| does)\s+(?:(?:a|an|the)\s+)?(?:\w+\s+){0,3}?\w+(?:\s+mean)?\s*$",
    r"\bif\s+i\s+(?:ever|were to)\b|\bin case i\b|\bin the future\b",
    # Hinglish: "... ke lakshan kya hain", "kaise pehchane", "ke baare mein",
    # "agar kisi ko ...", "... kya hota hai", "behosh hone par kya karein"
    r"\b(?:kya|kaun\s+se|kaunse|konse|kon\s+se|kaun\s+si|kaunsi|kis|kin)\s+(?:\w+\s+){0,2}?"
    r"(?:lakshan|symptoms?|signs?|nishaan|nishan|sanket)\b",
    r"\b(?:lakshan|symptoms?|signs?|sanket)\s+(?:\w+\s+){0,2}?(?:kya|kaun|kaunse|konse|batao"
    r"|bataiye|bataye|bataen|btao|samjhao|samjhaiye|hote|hoti)\b",
    r"\bkaise\s+(?:\w+\s+){0,2}?(?:pata|pehchan|pehchaan|pahchan|pahchaan|pehchane|jaane"
    r"|jaanein|jaanen|samjhe|samjhein|maloom|malum|bachein|bachaye|bachayein|roke|rokein)\w*",
    r"\b(?:ke|ki|ka)\s+(?:baare|bare|barey|baarey)\s+(?:mein|me|main|mai)\b",
    r"\bagar\s+(?:kabhi|kisi|kisiko|koi|ghar\s+mein\s+kisi|mujhe\s+kabhi)\b",
    r"\b(?:kya|kyun|kyon)\s+(?:hota|hoti|hote)\s+(?:hai|hain|h)\b",
    r"\b(?:hone|ho\s+jaane|ho\s+jane|aane|padne)\s+(?:par|pe|pr)\s+(?:kya|kaise)\b",
)
#: First-person / present-tense experience markers; any of them in the message
#: cancels informational suppression (the question may be about a live event).
_PERSONAL_RE = _rx(
    r"\bi(?:'m| am)\s+(?:\w+\s+)?(?:having|feeling|experiencing|getting)\b",
    r"\b(?:now|right now|currently|at the moment|as we speak|just now|today|tonight"
    r"|this (?:morning|afternoon|evening))\b",
    r"\b(?:my|his|her|their)\s+(?:[\w']+\s+)?(?:chest|head|face|arm|leg|heart|lips?|speech"
    r"|vision|eyes?|smile)\s+(?:is|are|was|feels?|felt|hurts?|went|keeps?|has|have|started"
    r"|looks?|seems?)\b",
    r"\b(?:i have|i had|i've got|i've|i feel|i felt|i got|i get)\s+(?:(?:a|an|some|this|that"
    r"|these|the|bad|severe|sudden)\s+)?(?:\w+\s+)?(?:pain|ache|aching|tight\w*|pressure"
    r"|numb\w*|weak\w*|droop\w*|slurr\w*|dizz\w*|faint\w*|headache|symptoms?|signs?)\b",
    r"\bi\s+(?:(?:just|suddenly|also|then|again|really|nearly|almost)\s+)?(?:fainted"
    r"|passed out|blacked out|collapsed|can't|cannot|couldn't|keep|woke|started|noticed)\b",
    # Hinglish first-person / present markers
    r"\b(?:abhi|is\s+waqt|isi\s+waqt|is\s+samay|filhaal|aaj\s+subah|aaj\s+raat|subah\s+se"
    r"|raat\s+se|ghante\s+se|ghanton\s+se|der\s+se)\b",
    r"\bmujhe\s+(?:\w+\s+){0,4}?(?:ho\s+(?:raha|rahi|rahe|gaya|gayi)|lag\s+(?:raha|rahi)"
    r"|mehsoos|feel\s+ho|aa\s+(?:raha|rahi|rahe))\b",
    r"\bmujhe\s+(?:\w+\s+){0,3}?(?:dard|pain|takleef|dikkat|pareshani|jalan|chakkar|behoshi"
    r"|ghabrahat|bharipan|dabav)\s+(?:\w+\s+)?(?:hai|h)\b",
    r"\b(?:mera|meri|mere)\s+(?:\w+\s+)?(?:seena|seene|chhati|chehra|chehre|muh|haath|baazu"
    r"|pair|sar|dil|aawaz|zubaan|saans)\b",
    r"\b(?:ye|yeh|yah|yahi)\s+(?:sab\s+)?(?:\w+\s+){0,3}?(?:lakshan|symptoms?|signs?"
    r"|heart\s+attack|lakwa|stroke)\b",
)
#: Present progressive with no subject ("dard ho raha hai") is a live report unless
#: the clause is about somebody generic ("agar kisi ko ... ho raha hai").
_HI_PROGRESSIVE_RE = re.compile(
    r"\b(?:ho|aa|uth|lag)\s+(?:raha|rahi|rahe)\s+(?:hai|hain|h|hoon)\b"
)
_HI_GENERIC_PERSON_RE = re.compile(
    r"\b(?:kisi|kisiko|koi|someone|somebody|vyakti|insaan|logon|log|mareez|aadmi)\b"
)
_HISTORICAL_RE = _rx(
    r"\b(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|several"
    r"|few|many|couple of|some)\s+(?:years?|yrs?|months?|decades?)\s+(?:ago|back|before"
    r"|earlier)\b",
    r"\blast\s+(?:year|month|summer|winter|spring|autumn|fall|decade)\b|\bin\s+(?:19|20)\d\d\b",
    r"\bback in (?:(?:19|20)\d\d|the day|the (?:\d0s|eighties|nineties)|my|school|college"
    r"|high school)\b",
    r"\b(?:years ago|long (?:time )?ago|used to|previous(?:ly)?|prior|hx of|h/o)\b",
    r"\b(?:family |medical |past |previous |prior )?history of\b",
    r"\bas a (?:child|kid|teen(?:ager)?|baby|young (?:man|woman|adult))\b",
    r"\bwhen i was (?:younger|little|young|pregnant|a (?:child|kid|teen(?:ager)?|baby|student)"
    r"|in (?:school|college))\b",
    r"\bruns in (?:my|the|our) family\b|\bin my (?:last|previous|first) pregnancy\b",
    r"\bturned out (?:to be)?\b|\bit was (?:just |only |probably |actually )?(?:acid reflux"
    r"|reflux|heartburn|gas|indigestion|anxiety|a panic attack|panic|muscular|a pulled muscle"
    r"|muscle strain)\b",
    r"\b(?:well[- ]controlled|under control|seizure[- ]free)\b",
    # Hinglish anchors: "2 saal pehle", "pichle saal", "2019 mein", "bachpan mein",
    # "... ki history", "pehle 200 rehta tha", "gas nikla", "control mein hai"
    r"\b(?:\d+|ek|do|teen|char|chaar|paanch|panch|chhe|saat|aath|nau|das|kai|kuch|bahut)\s+"
    r"(?:saal|saalon|baras|varsh|mahine|mahino|mahinon)\s+(?:pehle|se\s+pehle|purani|purana)\b",
    r"\bpichle\s+(?:saal|baras|varsh|mahine|dashak)\b|\b(?:saalon|bahut|kaafi)\s+pehle\b",
    r"\b(?:19|20)\d\d\s+(?:mein|main|mai)\b|\bbachpan\s+(?:mein|me|main|se)\b|\bpurani\s+baat\b",
    r"\b(?:ki|ka|ke)\s+history\b|\bhistory\s+(?:hai|rahi|thi|h)\b",
    r"\b(?:rehta|rehti|rahta|rahti|hota|hoti)\s+(?:tha|thi|the)\b",
    r"\b(?:gas|acidity|badhazmi)\s+(?:ki\s+wajah\s+se\s+)?(?:tha|thi|nikla|nikli|nikle)\b",
    r"\bcontrol\s+(?:mein|me|main)\s+(?:hai|h|rehta|rahta)\b",
)
_RECENT_RE = _rx(
    r"\b(?:now|right now|currently|today|tonight|just|again|still|ongoing|at the moment|since"
    r"|keeps?|kept|started|starting|yesterday|last night|is back|came back|returned)\b",
    r"\bthis (?:morning|afternoon|evening|week)\b",
    r"\b(?:\d+|a|an|one|two|few|couple of|a few|several)\s+(?:minutes?|mins?|hours?|hrs?"
    r"|days?)\s+ago\b",
    r"\b(?:abhi|aaj|kal|subah\s+se|raat\s+se|phir\s+se|fir\s+se|dobara|dubara|ab\s+bhi"
    r"|ab\s+tak|ab\s+phir|ab\s+fir|wapas|vapas|ghante|ghanton)\b",
)
_RESOLVED_RE = _rx(
    r"\bnow\s+(?:it's\s+|it is\s+|i'm\s+|i am\s+|everything(?:'s| is)\s+|all\s+|things are\s+"
    r"|they're\s+|they are\s+)?(?:\w+\s+)?(?:fine|normal|ok|okay|better|good|controlled|resolved"
    r"|gone|stable)\b",
    r"\b(?:is|are|it's|i'm|am|all|everything's)\s+(?:\w+\s+)?(?:fine|normal|ok|okay|better|good"
    r"|controlled|resolved|gone|stable)\s+now\b",
    r"\b(?:ab|abhi)\s+(?:sab\s+|bilkul\s+|ekdam\s+|to\s+|toh\s+)?(?:theek|normal|sahi|control)\b",
)
_HEAVY_EXERTION_RE = _rx(
    r"\b(?:when|while|after|during|if|whenever|with)\s+(?:(?:i|i'm|doing|heavy|intense|hard"
    r"|vigorous|strenuous|go|going|play|playing)\s+)*(?:run|running|lift\w*|training)\b",
    r"\b(?:jogging|jog|(?<!breathing )exercis\w*|working out|work out|workouts?|gym|sports?"
    r"|sprint\w*|weightlifting|lifting weights|swimming|hiking|cycling|biking|marathons?"
    r"|cardio|hiit|football|soccer|basketball|tennis|badminton|cricket|dancing)\b",
    r"\b(?:daud\w*|bhaag(?:ne|te|na|kar)|bhag(?:ne|te|na|kar)|kasrat|vyayam|khel(?:ne|te|ta"
    r"|ti|kar))\b",
)
_MINIMAL_EXERTION_RE = _rx(
    r"\b(?:at rest|resting|rest|sitting|lying|laying|lie down|lay down|sleep\w*|talking"
    r"|walking|walk|bathroom|toilet|getting dressed|dressing|shower\w*|even|minimal|light"
    r"|few steps|across the room|stairs|doing nothing|in bed|bed)\b",
    r"\b(?:seedhi\w*|sidhi\w*|chalne|chalte|baithe|baith\w*|lete|leta|leti|letne|aaram|sote"
    r"|soye|bistar)\b",
)


class _Clause:
    """One clause of the normalised message; context flags are computed once, lazily."""

    def __init__(self, start: int, text: str, next_text: str) -> None:
        self.start = start
        self.text = text
        self.next_text = next_text

    @cached_property
    def voided(self) -> bool:
        """``never had chest pain like this`` — the negation affirms a new symptom."""
        return bool(_NEG_VOID_RE.search(self.text))

    @cached_property
    def negation_cues(self) -> list[tuple[int, int]]:
        """Valid negation cue spans (clause-relative); empty when voided."""
        if self.voided:
            return []
        pseudo = [m.span() for m in _PSEUDO_NEG_RE.finditer(self.text)]
        cues = [
            m.span()
            for m in _NEG_CUE_RE.finditer(self.text)
            if not any(ps <= m.start() < pe for ps, pe in pseudo)
        ]
        neither = _HI_NEITHER_RE.search(self.text)
        if neither is not None:  # "na to X na Y": the later bare "na" is a cue too
            cues += [m.span() for m in _HI_NA_RE.finditer(self.text, neither.end())]
        return cues

    @cached_property
    def post_negated(self) -> bool:
        """``chest pain? no`` — the following clause is a bare negative answer."""
        return bool(_POST_NEG_CLAUSE_RE.match(self.next_text))

    @cached_property
    def informational(self) -> bool:
        return bool(_INFO_CUE_RE.search(self.text))

    @cached_property
    def historical(self) -> bool:
        if not _HISTORICAL_RE.search(self.text):
            return False
        # "... in 2020, now it's well controlled": a resolution is not recency.
        return not _RECENT_RE.search(_RESOLVED_RE.sub(" ", self.text))

    @cached_property
    def heavy_exertion(self) -> bool:
        return bool(_HEAVY_EXERTION_RE.search(self.text)) and not _MINIMAL_EXERTION_RE.search(
            self.text
        )


class _Analysis:
    """Per-message clause segmentation and context helpers."""

    def __init__(self, text: str) -> None:
        self.text = text
        spans: list[tuple[int, int]] = []
        prev = 0
        for boundary in _CLAUSE_BOUNDARY_RE.finditer(text):
            spans.append((prev, boundary.start()))
            prev = boundary.end()
        spans.append((prev, len(text)))
        self._starts = [s for s, _ in spans]
        self._clauses = [
            _Clause(
                start=s,
                text=text[s:e],
                next_text=text[spans[i + 1][0]:spans[i + 1][1]] if i + 1 < len(spans) else "",
            )
            for i, (s, e) in enumerate(spans)
        ]
        self.personal = bool(_PERSONAL_RE.search(text)) or any(
            _HI_PROGRESSIVE_RE.search(c.text) and not _HI_GENERIC_PERSON_RE.search(c.text)
            for c in self._clauses
        )
        self.rule_starts: list[int] = []
        self._rule_ends: dict[int, int] = {}

    def set_rule_spans(self, spans: list[tuple[int, int]]) -> None:
        """Register every red-flag match span (needed for list-scoped negation)."""
        self.rule_starts = sorted(start for start, _ in spans)
        for start, end in spans:
            self._rule_ends[start] = max(end, self._rule_ends.get(start, end))

    def clause_at(self, pos: int) -> _Clause:
        return self._clauses[bisect.bisect_right(self._starts, pos) - 1]

    def _hi_post_negated(self, end: int, clause: _Clause) -> bool:
        """Hinglish post-position negation: ``seene mein dard nahi hai``.

        Walks a short list (``dard aur saans ki takleef nahi``) whose items are
        other red-flag matches, bounded by :data:`_HI_LIST_MAX` items.
        """
        if clause.voided:
            return False
        clause_end = clause.start + len(clause.text)
        pos = end
        for _ in range(_HI_LIST_MAX):
            cue = _HI_POST_NEG_RE.match(self.text, pos, clause_end)
            if cue is not None:
                return not _HI_REAFFIRM_RE.search(self.text, cue.end(), clause_end)
            sep = _HI_LIST_SEP_RE.match(self.text, pos, clause_end)
            if sep is None:
                return False
            next_end = self._rule_ends.get(sep.end())
            if next_end is None:
                tail = _HI_TAIL_ITEM_NEG_RE.match(self.text, sep.end(), clause_end)
                return tail is not None and not _HI_REAFFIRM_RE.search(
                    self.text, tail.end(), clause_end
                )
            pos = next_end
        return False

    def is_negated(self, start: int, end: int) -> bool:
        """Whether the match ``[start, end)`` lies in the scope of a negation cue."""
        rule_starts = self.rule_starts
        clause = self.clause_at(start)
        if clause.post_negated or _POST_NEG_INLINE_RE.match(self.text, end):
            return True
        if self._hi_post_negated(end, clause):
            return True
        rel = start - clause.start
        for cue_start, cue_end in clause.negation_cues:
            # The cue must begin before the match; a cue verb may overlap the
            # match ("don't want" / "want to die").
            if cue_start >= rel or rel - cue_end > _NEG_WINDOW_CHARS:
                continue
            segment = clause.text[cue_end:rel]
            if _NEG_HARD_TERM_RE.search(segment):
                continue
            soft = _NEG_SOFT_TERM_RE.search(segment)
            if soft is None:
                return True
            # Lists stay negated only when the cue's first object is a red flag.
            first_soft = clause.start + cue_end + soft.start()
            if any(clause.start + cue_end <= s < first_soft for s in rule_starts):
                return True
        return False

    def status(self, start: int, end: int, *, exertion: bool) -> str:
        """Classify one rule match as affirmed / negated / suppressed."""
        if self.is_negated(start, end):
            return "negated"
        clause = self.clause_at(start)
        if clause.historical:
            return "suppressed"
        if clause.informational and not self.personal:
            return "suppressed"
        if exertion and clause.heavy_exertion:
            return "suppressed"
        return "affirmed"


# ======================================================================
# 3. Blood-pressure extraction
# ======================================================================

_NUM_GUARD = r"(?!\d|\.\d|\s*[/\\]\s*\d|\s+(?:over|by)\s+\d)"
_UNIT_GUARD = (
    r"(?!\s*(?:%|percent|mg|mcg|ug|ml|mmol|kg|kgs|lbs?|pounds|units?|iu|bpm|beats|steps"
    r"|calories|cal|kcal|days?|weeks?|months?|years?|yrs?|hours?|hrs?|minutes?|mins?|times"
    r"|rs|rupees|dollars|pills?|tablets?|tabs?|cm|mm(?!\s*hg)|meters?|metres?|km|miles?|ft"
    r"|feet|degrees?|saal|sal|mahine|mahina|din|ghante|ghanta|rupaye|rupaiye|rupay|goli"
    r"|goliyan|kilo|baar|dafa|umar)\b)"
)
_CMP = (
    r"(?:(?P<strict>over|above|more than|greater than|higher than|past|beyond|crossed"
    r"|exceeding|exceeds|>)\s*|(?:at least|>=|in the)\s+)?"
)
_SUFFIX = r"(?:\s*(?:\+|plus\b|something\b|ish\b|-ish\b|odd\b|s\b|'s\b))?"
_FILLER = (
    r"(?:\s+(?:is|was|are|were|'s|of|at|reads?|reading|showed|shows?|says|said|came|comes"
    r"|out|now|still|currently|today|tonight|again|just|already|up|reaching|around|about"
    r"|approximately|approx|roughly|nearly|almost|like|maybe|probably|went|gone|going|has|had"
    r"|have|been|got|gets|getting|keeps?|kept|stays?|stayed|hit|hits|reached|touched"
    r"|touching|spiked|jumped|shot|rose|climbed|dropped|fell|down|measured|measures|showing"
    r"|registered|it|it's|me|my|only|consistently|constantly|always|very|really|so|high|too"
    r"|to|hai|hain|tha|thi|hua|hui|gaya|gayi|raha|rahi|aaya|aayi|aa|abhi|aaj|bahut|kaafi|zyada"
    r"|jyada|ka|ki|mera|meri|bhi|toh|chala|chali|badh|badhkar|lagbhag|kareeb|upar|naapa)){0,5}"
)
_SINGLE_TAIL = r"\s*[:=-]?\s*" + _CMP + r"(?P<num>\d{2,3})" + _NUM_GUARD + _SUFFIX + _UNIT_GUARD

_BP_PAIR_RE = re.compile(
    r"(?<![\d/])(?<!\d\.)(?P<sys>\d{2,3})\s*(?:/|\\|\bover\b|\bby\b)\s*(?P<dia>\d{2,3})"
    r"(?!\d|\.\d)" + _UNIT_GUARD
)
_SYS_SINGLE_RE = re.compile(
    r"\b(?:systolic(?:\s+(?:bp|blood pressure|pressure|reading|number|value|level))?"
    r"|(?:top|upper|first)\s+(?:number|reading|value)"
    r"|(?:upar|uppar|ooper)\s+(?:wala|wali|vala|ka)(?:\s+(?:number|reading))?)"
    + _FILLER + _SINGLE_TAIL
)
_DIA_SINGLE_RE = re.compile(
    r"\b(?:diastolic(?:\s+(?:bp|blood pressure|pressure|reading|number|value|level))?"
    r"|(?:bottom|lower|second)\s+(?:number|reading|value)"
    r"|(?:neeche|niche|nichla|nichala)\s+(?:wala|wali|vala|ka)(?:\s+(?:number|reading))?)"
    + _FILLER + _SINGLE_TAIL
)
_BP_SINGLE_RE = re.compile(
    r"\b(?:bp|blood pressure|pressure|cuff)(?:\s+(?:reading|readings|monitor|machine|cuff"
    r"|level|levels|numbers?))?" + _FILLER + _SINGLE_TAIL
)
_CTX_SINGLE_RE = re.compile(
    r"\b(?:it|it's|its|now|numbers?|readings?|cuff|monitor|machine|to|hit|hits|reached"
    r"|reaching|touched|crossed)"
    r"(?:\s+(?:is|was|'s|are|were|reads?|said|says|showed|shows|came|out|at|now|still|like"
    r"|about|around|up|to)){0,3}\s*[:=-]?\s*" + _CMP + r"(?P<num>\d{3})" + _NUM_GUARD + _SUFFIX
    + _UNIT_GUARD
)
_BP_CONTEXT_RE = re.compile(
    r"\b(?:bp|blood pressure|pressure|systolic|diastolic|mm ?hg|hypertension|hypertensive"
    r"|cuff|readings?|monitor|machine|top number|bottom number|numbers|measured|measuring"
    r"|checked|checking|rechecked|raktchap|raktchaap|rakt chaap|rakt chap)\b"
)
_MEASUREMENT_LABEL_RE = re.compile(
    r"\b(?:(?P<bp>bp|blood pressure|pressure|systolic|diastolic|cuff)|sugar|glucose|cholesterol"
    r"|ldl|hdl|triglycerides?|a1c|hba1c|heart rate|pulse|hr|weight|weigh|temp|temperature"
    r"|fever|oxygen|o2|spo2|sats?|saturation|potassium|sodium|creatinine|egfr|platelets?"
    r"|h[a]?emoglobin|uric acid|tsh|vitamin|calories|steps|score|iq|dhadkan|nabz|nabj|wazan"
    r"|vajan|bukhar|taapmaan|tapman)\b"
)
_TARGET_RE = re.compile(
    r"(?:under|below|less than|lower than|<|target|goal|aim(?:ing)? for|keep (?:it|my bp|bp)?)"
    r"\s*(?:of|is|at|to)?\s*$"
)
#: "if / unless / in case / whenever ..." — a conditional threshold, not a measurement
#: ("call if BP is over 180/110"). "when" is excluded: it is often temporal.
_CONDITIONAL_RE = re.compile(
    r"\b(?:if|unless|in case|whenever|agar|yadi|jab\s+bhi)\b[^.;?!\n]{0,30}$"
)
_THRESHOLD_TAIL_RE = re.compile(
    r"(?:over|above|higher than|more than|greater than|exceeds|exceeding|>=?|crosses|past"
    r"|beyond)\s*$"
)
#: Hindi postpositions follow the number: "180 se upar" (above 180) is a threshold,
#: "130/80 se neeche rakhna" (keep below) and "130/80 hona chahiye" are targets.
_HI_THRESHOLD_AFTER_RE = re.compile(
    r"^\s*(?:mm ?hg\s+)?(?:se|ke)\s+(?:upar|ooper|uppar|zyada|jyada|adhik|paar)\b"
)
_HI_TARGET_AFTER_RE = re.compile(
    r"^\s*(?:mm ?hg\s+)?(?:(?:se|ke)\s+(?:neeche|niche|kam|andar)\s+(?:\w+\s+)?(?:rakh\w*"
    r"|hona\s+chahiye|rehna\s+chahiye|rahna\s+chahiye)|(?:hona|rehna|rahna)\s+chahiye"
    r"|tak\s+(?:rakh\w*|lana|laana))"
)


@dataclass(frozen=True)
class _Reading:
    """One plausible BP reading; a component that was not stated is ``0``."""

    systolic: int
    diastolic: int
    start: int
    end: int
    labelled: bool


def _other_measurement(before: str) -> bool:
    """Whether the nearest measurement label before a number is not a BP label."""
    labels = list(_MEASUREMENT_LABEL_RE.finditer(before))
    return bool(labels) and labels[-1].group("bp") is None


def _plausible(sys_value: int, dia_value: int) -> bool:
    return 50 <= sys_value <= 300 and 20 <= dia_value <= 200 and sys_value > dia_value


def _not_a_reading(before: str, after: str, personal: bool) -> bool:
    """Target ("below 130/80") or conditional threshold ("if it is over 180/110").

    ``before`` / ``after`` are the 40 / 30 characters around the number; Hinglish
    puts the comparison after it ("agar bp 180 se upar jaaye"). A conditional
    threshold still counts as a reading when the message carries a present-tense /
    first-person marker ("what if it's over 200/120 right now?").
    """
    if _TARGET_RE.search(before) or _HI_TARGET_AFTER_RE.match(after):
        return True
    return (
        not personal
        and bool(_CONDITIONAL_RE.search(before))
        and bool(_THRESHOLD_TAIL_RE.search(before) or _HI_THRESHOLD_AFTER_RE.match(after))
    )


def _single_value(match: re.Match[str]) -> int:
    value = int(match.group("num"))
    return value + 1 if match.group("strict") else value


def _extract_readings(text: str, has_context: bool, personal: bool) -> list[_Reading]:
    """Extract plausible BP readings from normalised text, most specific format first."""
    readings: list[_Reading] = []
    claimed: set[int] = set()
    for m in _BP_PAIR_RE.finditer(text):
        sys_value, dia_value = int(m.group("sys")), int(m.group("dia"))
        if has_context and 40 <= sys_value < dia_value:
            # "bp 120/190": most likely typed in the wrong order — read it swapped.
            sys_value, dia_value = dia_value, sys_value
        before = text[max(0, m.start() - 40):m.start()]
        if (
            _plausible(sys_value, dia_value)
            and not _other_measurement(before)
            and not _not_a_reading(before, text[m.end():m.end() + 30], personal)
        ):
            readings.append(_Reading(sys_value, dia_value, m.start(), m.end(), False))
            claimed.update((m.start("sys"), m.start("dia")))
    singles = (
        (_DIA_SINGLE_RE, "dia", True),
        (_SYS_SINGLE_RE, "sys", True),
        (_BP_SINGLE_RE, "sys", True),
        (_CTX_SINGLE_RE, "sys", has_context),
    )
    for pattern, component, enabled in singles:
        if not enabled:
            continue
        for m in pattern.finditer(text):
            if m.start("num") in claimed:
                continue
            before = text[max(0, m.start("num") - 40):m.start("num")]
            after = text[m.end():m.end() + 30]
            if _other_measurement(before) or _not_a_reading(before, after, personal):
                continue
            value = _single_value(m)
            if not (50 <= value <= 300 if component == "sys" else 20 <= value <= 200):
                continue
            claimed.add(m.start("num"))
            if component == "sys":
                readings.append(_Reading(value, 0, m.start(), m.end(), True))
            else:
                readings.append(_Reading(0, value, m.start(), m.end(), True))
    readings.sort(key=lambda r: r.start)
    return readings


def _is_crisis(reading: _Reading, has_context: bool) -> bool:
    high_sys = reading.systolic >= SYSTOLIC_CRISIS
    high_dia = reading.diastolic >= DIASTOLIC_CRISIS
    if reading.labelled or has_context:
        return high_sys or high_dia
    return high_sys and high_dia


def _report_bp(
    reading: _Reading, readings: list[_Reading], analysis: _Analysis
) -> tuple[int, int]:
    """Fill a missing component from a single-value reading in the same clause."""
    systolic, diastolic = reading.systolic, reading.diastolic
    clause = analysis.clause_at(reading.start)
    for other in readings:
        if analysis.clause_at(other.start) is not clause:
            continue
        if systolic == 0 and other.systolic:
            systolic = other.systolic
        if diastolic == 0 and other.diastolic:
            diastolic = other.diastolic
    return systolic, diastolic


# ======================================================================
# 4. Red-flag rules
# ======================================================================

_CANT = (
    r"(?:can't|cannot|can not|couldn't|could not|unable to|not able to|inability to|hard to"
    r"|difficult to|struggling to|struggle to|trouble|difficulty|difficulties|problems?"
    r"|(?:a )?hard time|labou?red)"
)
_CHEST = r"(?:chest|sternum|breastbone|breast bone)"
_DET = r"(?:(?:my|the|his|her|their|your)\s+)"


def _self_report(conditions: str) -> tuple[str, ...]:
    """Patterns for 'I'm having a <condition>' style self/bystander reports."""
    subject = (
        r"(?:i'm|i am|i think i'm|i think i am|i might be|i may be|i could be|am i"
        r"|is (?:he|she|my \w+)|(?:he|she)'s|(?:he|she|my \w+) (?:is|might be|may be))"
    )
    cond = rf"(?:(?:a|an)\s+)?(?:\w+\s+)?(?:{conditions})\b"
    return (
        rf"\b{subject}\s+(?:\w+\s+)?(?:having|getting|suffering)\s+{cond}",
        rf"\bi (?:think|believe|fear|guess) i (?:just |may have |might have )?(?:had|have had)"
        rf"\s+{cond}",
        rf"\bi (?:just|may have just|might have just|might have|may have) had\s+{cond}",
        rf"\b(?:feels?|felt|feeling) like (?:i'm having |i am having |i'm getting )?{cond}",
        rf"\b(?:i have|i've got|i'm having|i am having|having) (?:all )?(?:the )?"
        rf"(?:symptoms|signs) of\s+{cond}",
    )


#: (rule id, pattern, breathlessness-style exertion filter) in reporting order.
_RED_FLAG_RULES: tuple[tuple[str, re.Pattern[str], bool], ...] = (
    (
        "chest_pain",
        _rx(
            rf"\b{_CHEST}(?:'s|\s+(?:is|was|feels?|felt|has been|had been|keeps?|kept|started"
            r"|starts|getting|gets|got|became|becomes|going|went|are|area|wall))?(?:\s+(?:really"
            r"|very|so|super|kind of|kinda|sort of|a bit|a little|bit|little|extremely|quite"
            r"|pretty|too|all|like|suddenly|now|still|just|more|getting|being))*\s+(?:pains?"
            r"|painful|aches?|aching|achy|hurts?|hurting|tight|tightness|tightening"
            r"|tighten(?:s|ed)?"
            r"|heavy|heaviness|pressure|pressing|squeez\w*|crush\w*|burn\w*|discomfort"
            r"|constrict\w*|compress\w*|sharp|stabbing|stab\w*|pinch\w*|clench\w*|clamp\w*"
            r"|killing me)\b",
            rf"\b{_CHEST} (?:feels?|felt) like (?:\w+\s+){{0,5}}?(?:crush\w*|squeez\w*|pressure"
            r"|elephant|weight|tight\w*|vice|vise|explod\w*|burst\w*|on fire|burning|heart attack"
            r"|being (?:crushed|squeezed|stabbed))",
            r"\b(?:pains?|aches?|aching|tightness|tight (?:feeling|sensation)|pressure|heaviness"
            r"|heavy (?:feeling|sensation|weight)|weight|squeez\w*|crush\w*|burn\w*|discomfort"
            r"|constriction|compression|pressing|stabbing|elephant|band|vice|vise"
            r"|ton of bricks)\b(?:\s+\w+){0,4}?\s+(?:in|on|across|around|over|behind|at|of"
            rf"|through|inside|within|under|into|to)\s+{_DET}?(?:\w+\s+){{0,3}}?{_CHEST}\b",
            r"\bpain\s+(?:in|around)\s+my\s+heart\b",
            r"\b(?:someone|somebody|something|an? \w+)\s+(?:is\s+)?(?:sitting|standing|pressing"
            rf"|pushing|squeezing)\s+(?:on|down on)\s+{_DET}?{_CHEST}\b",
        ),
        False,
    ),
    (
        "heart_attack",
        _rx(
            *_self_report(r"heart attack|cardiac arrest"),
            r"\b(?:sense|feeling) of (?:impending )?doom\b|\bimpending doom\b",
        ),
        False,
    ),
    (
        "aortic_dissection",
        _rx(
            r"\b(?:tearing|ripping)(?:[\s,]+\w+){0,4}?[\s,]+(?:pain|ache|sensation|feeling)\b",
            r"\b(?:pain|ache)(?:[\s,]+\w+){0,4}?[\s,]+(?:tearing|ripping)\b",
            r"\b(?:tearing|ripping)(?:[\s,]+\w+){0,3}?[\s,]+(?:chest|back|shoulder blades?)\b",
        ),
        False,
    ),
    (
        "breathlessness",
        _rx(
            r"\bshort(?:ness)?\s+(?:of\s+)?breath\w*",
            r"\bbreathless\w*|\bout of breath\b|\bair hunger\b",
            rf"\b{_CANT}\s+(?:\w+\s+)?(?:breath(?:e|ing)?|catch(?:ing)? (?:my |his |her )?breath"
            r"|get(?:ting)? (?:enough |any )?(?:air|breath|oxygen))\b"
            r"(?! through (?:my|the|his|her) nose)",
            r"\bgasp(?:ing|ed|s)?\b|\b(?:struggling|fighting) for (?:air|breath)\b",
            r"\b(?:stopped|not|isn't|is not|wasn't|was not|has stopped|aren't|are not) breathing\b",
            r"\bbreathing (?:is |has (?:become|gotten|got) |feels |became )?(?:very |really |so "
            r"|extremely )?(?:hard|difficult|labou?red|laboring|labouring)\b",
            r"\b(?:can't|cannot|unable to|not able to) lie (?:down )?flat\b",
        ),
        True,
    ),
    (
        "stroke_face",
        _rx(
            r"\b(?:face|facial)(?:'s|\s+(?:is|was|feels?|felt|has|had|looks?|looked|went"
            r"|has gone|gone|seems?|started|starts|became|getting|keeps?))?(?:\s+(?:really|very"
            r"|so|a bit|a little|kind of|kinda|suddenly|all|now|totally|completely|partly"
            r"|partially|slightly|been))*\s+(?:droop\w*|drop(?:ping|ped|s)?|sagg\w*|numb\w*"
            r"|tingl\w*|paraly[sz]\w*|weak\w*|lopsided|uneven|crooked|twisted|frozen|slack"
            r"|fallen|falling|limp|asymmetric\w*)\b",
            r"\b(?:one|left|right|half|other)\s+(?:side\s+of\s+)?(?:(?:my|his|her|the)\s+)?"
            r"face\b(?:\s+\w+){0,4}?\s+(?:droop\w*|drop(?:ping|ped|s)?|sagg\w*|numb\w*|tingl\w*"
            r"|paraly[sz]\w*|weak\w*|lopsided|uneven|crooked|twisted|frozen|dead|slack|fallen"
            r"|falling|limp|heavy|weird|funny|strange|odd|off)\b",
            r"\b(?:smile|mouth|lips?)\s+(?:(?:is|are|was|looks?|looked|feels?|seems?|has|went|got)"
            r"\s+)?(?:\w+\s+)?(?:uneven|crooked|lopsided|droop\w*|twisted|sagg\w*"
            r"|drop(?:ping|ped)|slant\w*|one[- ]sided|asymmetric\w*)\b",
            r"\b(?:uneven|crooked|lopsided|droopy|one[- ]sided)\s+(?:smile|mouth)\b",
        ),
        False,
    ),
    (
        "stroke_arm",
        _rx(
            rf"\b{_CANT}\s+(?:\w+\s+)?(?:move|lift|raise|feel|use|hold up|grip with|control|bend)"
            r"\s+(?:(?:my|his|her|the)\s+)?(?:(?:left|right|one)\s+)?(?:arm|leg|hand|side"
            r"|fingers)\b",
            r"\b(?:(?:my|his|her|left|right|one)\s+)?(?:(?:left|right)\s+)?(?:arm|leg|hand)\s+"
            r"(?:(?:is|was|feels?|felt|went|has gone|has been|became|suddenly|just|keeps?"
            r"|seems?)\s+)?(?:(?:really|very|so|suddenly|all|completely|totally|kind of|a bit"
            r"|a little|now|gone)\s+)*(?:weak\w*|numb\w*|dead|limp|paraly[sz]\w*|useless|floppy"
            r"|clumsy|clumsiness|tingly|tingling|won't move|not moving|giving way|gave way)\b",
            r"\barm\s+(?:is |feels? |felt )?(?:\w+\s+)?heavy\b",
            r"\b(?:weak\w*|numb\w*|paraly[sz]\w*|limp|clumsy|clumsiness)\s+(?:in|of|on|down)\s+"
            r"(?:(?:my|his|her|the|one)\s+)?(?:(?:left|right)\s+)?(?:arm|leg|hand|side)\b",
            r"\b(?:numb\w*|weak\w*|paraly[sz]\w*|tingl\w*|dead)\s+(?:\w+\s+){0,2}?(?:on|down|along"
            r"|in|of)\s+(?:the\s+|my\s+|one\s+)?(?:left|right|one|whole|entire)\s+(?:\w+\s+)?"
            r"side\b",
            r"\b(?:one[- ]sided|sudden|unilateral)\s+(?:weakness|numbness|paralysis)\b",
            r"\b(?:left|right|one|whole|entire)\s+side\s+(?:of\s+(?:my|his|her|the)\s+)?"
            r"(?:body\s+)?"
            r"(?:(?:is|feels?|felt|went|has gone|has been|became|keeps|suddenly)\s+)?(?:\w+\s+)?"
            r"(?:numb\w*|weak\w*|paraly[sz]\w*|tingl\w*|dead|limp|heavy)\b",
        ),
        False,
    ),
    (
        "stroke_numbness",
        _rx(r"\bnumb(?:ness)?(?:[\s,]+\w+){0,6}?[\s,]+(?:face|arms?|legs?|side|body)\b"),
        False,
    ),
    (
        "stroke_speech",
        _rx(
            r"\bslurr?(?:ed|ing|s)?\s+(?:\w+\s+)?(?:speech|words?|talking)\b|\baphasi\w*",
            r"\b(?:speech|words|talking)\s+(?:(?:is|are|was|were|sounds?|sounded|seems?"
            r"|has been|have been|became|getting|came out|comes out|coming out|keep coming out)"
            r"\s+){0,2}(?:\w+\s+)?"
            r"(?:slurr?(?:ed|ing|y)|garbled|jumbled|mixed up|weird|strange|funny|wrong"
            r"|not making sense|muddled|incoherent|nonsense|gibberish)\b",
            rf"\b{_CANT}\s+(?:\w+\s+)?(?:talk(?:ing)?|speak(?:ing)?|get(?:ting)? (?:my |the )?"
            r"words out|find(?:ing)? (?:my |the )?words|form(?:ing)? words|say(?:ing)? words)\b"
            r"(?!\s+(?:to|with|about|for|right now|now|on the phone|much|long|at the moment"
            r"|today|english|hindi)\b)",
            rf"\b{_CANT}\s+(?:\w+\s+)?understand(?:ing)?\s+(?:what\s+(?:\w+\s+){{1,3}}?(?:is|are)"
            r"\s+saying|people|anyone|speech|spoken words|words)\b",
        ),
        False,
    ),
    (
        "stroke_vision",
        _rx(
            r"\bvision\s+(?:(?:is|has|has been|getting|went|is going|has gone)\s+)?(?:loss|lost"
            r"|chang\w*|blurr\w*|blur|blurry|darken\w*|black\w*|dim\w*|cut out|cuts out|gone"
            r"|going)\b",
            r"\b(?:lost|losing|loss of|lose)\s+(?:(?:my|the|his|her|all|some|part of my)\s+)?"
            r"(?:vision|sight|eyesight)\b",
            r"\b(?:blind\w*|can't see|cannot see|can not see|couldn't see|unable to see"
            r"|dark(?:ness)?"
            r"|black(?:ness)?|blurr\w*|blurry)\s+(?:\w+\s+){0,2}?(?:in|out of|from|with)\s+"
            r"(?:one|my|the|his|her|an?)\s+(?:(?:left|right|one)\s+)?eye\b",
            r"\bsudden(?:ly)?\s+(?:\w+\s+){0,2}?(?:blind\w*|blurr\w*|blurry|double vision"
            r"|vision|can't see|cannot see)",
            r"\bdouble vision\b|\bseeing double\b|\bvisual darkness\b|\bblack spots\b",
            r"\b(?:spots|blindness|flash\w*|curtain|shadow)\s+(?:\w+\s+){0,3}?(?:in|over|across"
            r"|down)\s+(?:my\s+|the\s+|one\s+)?(?:vision|eyes?|sight|visual field)\b",
            r"\bseeing (?:black |dark |bright |white )?(?:spots|flashes|flashing lights)\b",
        ),
        False,
    ),
    (
        "stroke_balance",
        _rx(
            r"\bsudden(?:ly)?\s+(?:\w+\s+){0,2}?(?:loss of balance|lost (?:my )?balance"
            r"|off balance|unsteady|unsteadiness|dizzy|dizziness|vertigo|trouble walking"
            r"|difficulty walking|can't walk|cannot walk|unable to walk|lost coordination"
            r"|loss of coordination|clumsy|clumsiness)\b",
            r"\binability to (?:walk|stand)\b",
            r"\b(?:can't|cannot|can not|couldn't|could not|unable to|not able to)\s+(?:walk|stand)"
            r"(?:\s+up)?\s+(?:straight|properly|steadily|steady|without (?:falling|help|holding)"
            r"|at all)\b",
            r"\bkeeps? (?:falling over|losing (?:my |his |her )?balance)\b",
        ),
        False,
    ),
    (
        "stroke_confusion",
        _rx(
            r"\bsudden(?:ly)?\s+(?:\w+\s+){0,2}?(?:confus\w*|disorient\w*|can't think"
            r" (?:straight|clearly)|memory loss)\b",
            r"\b(?:completely|totally|very|extremely|really)\s+(?:confused|disoriented)\b"
            r"(?!\s+(?:about|by|with|as to|on|over|why|what|which|how|whether|if|regarding)\b)",
            r"\bdisorient\w*\b",
            r"\b(?:don't|do not|doesn't|does not|didn't) (?:know|recogni[sz]e) (?:where (?:i am"
            r"|he is|she is|they are)|who (?:i am|he is|she is)|me|us|his family|her family"
            r"|anyone)\b",
        ),
        False,
    ),
    (
        "stroke_self_report",
        _rx(*_self_report(r"stroke|mini[- ]?stroke|tia")),
        False,
    ),
    (
        "severe_headache",
        _rx(
            r"\bworst\s+(?:\w+\s+){0,2}?(?:headache|head\s*ache|head pain|pain)\s+(?:of|in)\s+"
            r"my\s+(?:entire\s+|whole\s+)?life\b",
            r"\bworst\s+headache\s+(?:i(?:'ve| have)?\s+(?:ever\s+)?(?:had|felt|experienced)"
            r"|ever)\b",
            r"\bthunder\s*clap\w*",
            r"\bsudden(?:ly)?\s+(?:\w+\s+){0,2}?(?:headache|head\s*ache|head pain)\b",
            r"\b(?:headache|head\s*ache|head pain|pain in my head)\s+(?:\w+\s+){0,3}?(?:came on"
            r"|started|began|hit|struck|came)\s+(?:\w+\s+){0,2}?(?:suddenly|out of nowhere"
            r"|all of a sudden|in seconds|instantly)\b",
            r"\b(?:blinding|excruciating|explosive|exploding|unbearable|agonizing|agonising)\s+"
            r"(?:\w+\s+)?(?:headache|head\s*ache|head pain)\b",
            r"\bhead\s+(?:feels?\s+like\s+(?:it(?:'s| is)\s+)?|is\s+)(?:going to\s+)?"
            r"(?:explod\w*|burst\w*|split open)\b",
        ),
        False,
    ),
    (
        "syncope",
        _rx(
            r"\bfaint(?:ed|ing|s)\b|\bfaint(?:ed)?\s+(?:spell|episode)s?\b",
            r"\b(?:feel|feeling|felt|going to|gonna|about to|almost|nearly|might|may)\s+"
            r"(?:\w+\s+)?faint\b",
            r"\b(?:i|he|she|they|you|we)\s+(?:\w+\s+){0,2}?(?<!\ba )(?<!\bthe )faint\b",
            r"\bpass(?:ed|es|ing)?\s+out\b|\bblack(?:ed|ing|s)?\s+out\b|\bblackouts?\b",
            r"\bcollaps(?:e|ed|es|ing)\b|\bkeel(?:ed|s)? over\b|\bsyncop\w*",
            r"\b(?:lost|lose|losing|loss of)\s+consciousness\b|\bunconscious\b",
            r"\b(?:unresponsive|(?:he|she|they|he's|she's|they're|is|are)\s+(?:\w+\s+)?"
            r"not responding)\b(?!\s+to\s+(?:\w+\s+)?(?:treatment|meds|medications?|medicines?"
            r"|therapy|drugs?|pills?|tablets?))",
            r"\bwon't wake up\b|\b(?:can't|cannot|couldn't) wake (?:him|her|them|my \w+)\b",
        ),
        False,
    ),
    (
        "seizure",
        _rx(
            r"(?<!anti-)\bseizures?\b(?!\s+(?:medications?|medicines?|meds?|drugs?|pills?"
            r"|tablets?|disorder|threshold|risk|history))",
            r"\bseizing\b|\bconvuls\w*|\bhaving (?:a )?fit\b|\bepileptic (?:fit|attack)\b",
        ),
        False,
    ),
    (
        "hemoptysis",
        _rx(
            r"\bcough\w*\s+(?:up\s+)?(?:some\s+|a lot of\s+|lots of\s+)?(?:blood|bloody|pink"
            r"|frothy)\w*",
            r"\bpink\s+(?:\w+\s+)?(?:frothy|foamy|bubbl\w*|froth|foam|sputum|phlegm|spit|mucus)\b",
            r"\b(?:frothy|foamy)\s+(?:pink\s+)?(?:sputum|phlegm|spit|mucus|saliva)\b",
            r"\bblood(?:y)?\s+(?:in\s+(?:my\s+)?)?(?:sputum|phlegm|spit|mucus)\b",
            r"\bspit(?:ting)?\s+(?:up\s+)?blood\b|\bh(?:a)?emoptysis\b",
        ),
        False,
    ),
    (
        "cyanosis",
        _rx(
            r"\b(?:lips?|face|fingers?|fingertips|fingernails|nails|skin|tongue|mouth)\s+"
            r"(?:(?:are|is|look|looks|looked|turning|turned|went|going|gone|have gone|has gone"
            r"|getting|became)\s+)*(?:(?:a bit|slightly|very|really)\s+)?(?:blue|bluish|blueish"
            r"|purple|grey|gray|ashen)\b",
            r"\b(?:blue|bluish|blueish|purple)\s+(?:lips|face|fingers|fingertips|tongue)\b"
            r"|\bcyano\w*",
        ),
        False,
    ),
    (
        "angioedema",
        _rx(
            r"\b(?:tongue|throat|lips?)\s+(?:(?:is|are|has|have|feels?|started|keeps|getting|got)"
            r"\s+)?(?:\w+\s+)?(?:swell\w*|swollen|closing|puff\w*)\b",
            r"\bswell\w*\s+(?:of|in|on)\s+(?:my\s+|the\s+)?(?:tongue|throat|lips?)\b",
            r"\bswollen\s+(?:tongue|lips|throat)\b|\bthroat\s+(?:is\s+)?(?:closing|tightening)\b",
        ),
        False,
    ),
    (
        "severe_bleeding",
        _rx(
            r"\b(?:bleeding|blood|nosebleed|nose bleed)\s+(?:\w+\s+){0,4}?(?:won't|will not|wont"
            r"|doesn't|does not|didn't|can't|isn't|is not)\s+(?:stop|stopping)\b",
            r"\b(?:can't|cannot|couldn't|unable to)\s+stop\s+(?:the\s+)?bleed\w*",
            r"\b(?:profuse\w*|uncontroll\w*|massive)\s+bleed\w*|\bbleed\w*\s+(?:very\s+|really\s+)?"
            r"(?:heavily|profusely|everywhere)\b",
            r"\b(?:vomit\w*|throw\w* up|threw up|puk\w*)\s+(?:up\s+)?blood\b",
            r"\bblood\s+(?:is\s+)?(?:pouring|gushing|spurting|streaming)\b",
        ),
        False,
    ),
    (
        "blood_in_urine_or_stool",
        _rx(
            r"\bblood(?:y)?\s+(?:in\s+(?:my\s+|the\s+|his\s+|her\s+)?)?(?:urine|stools?|pee|poop"
            r"|poo|wee)\b",
            r"\b(?:peeing|pissing|passing|urinating)\s+blood\b|\bblack,?\s+tarry\s+stools?\b",
            r"\b(?:red|bloody|pink|cola)[- ]colou?red\s+urine\b",
        ),
        False,
    ),
    (
        "self_harm",
        _rx(
            r"\bsuicid\w*|\bself[- ]?harm\w*|\bkill(?:ing)?\s+my\s*self\b",
            r"\bend(?:ing)?\s+(?:my\s+(?:own\s+)?life|it all)\b|\btake\s+my\s+(?:own\s+)?life\b",
            r"\b(?:want|wanna|wish|wishing|plan(?:ning)?|thinking (?:about|of))\s+"
            r"(?:to\s+)?(?:die|be dead|not be here|overdose)\b",
            r"\bbetter off dead\b|\bno (?:reason|point) (?:to|in) (?:live|living|going on)\b",
            r"\b(?:hurt|harm|cut|cutting|hurting|harming)\s+myself\b",
            r"\bdon't want to (?:live|be alive|wake up|be here)\b|\bwish i (?:was|were) dead\b",
            r"\bwish i (?:wasn't|weren't|was not|were not) (?:here|alive|around)\b",
        ),
        False,
    ),
    (
        "hypertensive_crisis_report",
        _rx(
            r"\b(?:having|in|going through|experiencing)\s+(?:a\s+)?hypertensive\s+"
            r"(?:crisis|emergency|urgency)\b"
        ),
        False,
    ),
)

# ----------------------------------------------------------------------
# 4b. Hinglish / Indian-English red flags (Roman-script Hindi, after normalisation)
# ----------------------------------------------------------------------
# Each pattern needs at least one Hindi token (or an Indian-English idiom such as
# "chest is paining"), so purely English messages keep exactly the behaviour of
# the English rules above. Negation, informational, historical and exertion
# filters are shared; Hinglish post-position negation ("... nahi hai") is handled
# by :meth:`_Analysis._hi_post_negated`.

_HI_IN = r"(?:mein|me|main|mai|mei|mey|men|par|pe)"
_HI_PROG = r"(?:raha|rahi|rahe)"
_HI_INTENS = r"(?:bahut|kaafi|thodi|thoda|bhi|badi|zyada|jyada|koi)"
_HI_CHEST_FILL = (
    r"(?:bahut|ekdam|thoda|thodi|halka|halki|tez|zor|zyada|jyada|kaafi|ajeeb|ajib|aisa|aisi"
    r"|alag|bhayankar|bhayanak|sa|si|ek|kuch|koi|bhi|hi|to|toh|left|right|baayi|baayen|baaye"
    r"|daayi|daayen|daaye|beech|bich|taraf|side|hisse|hissa|wala|wali|achanak|lagatar|very"
    r"|severe|sharp|mild|ka|ki|ke|mein|me|main|andar)"
)
_HI_PAIN_HI = r"(?:dard|jalan|bharipan|dabav|jakdan|jakdahat|chubhan|khinchav|bojh)"
_HI_PAIN_ALL = (
    r"(?:dard|jalan|bharipan|dabav|jakdan|jakdahat|chubhan|khinchav|bojh|pain|tightness"
    r"|pressure|heaviness|discomfort|ache|aching)"
)
#: "struck by" verb group: "aa gaya", "pad raha", "ho gayi", "maar gaya", "hua hai" …
_HI_STRUCK = (
    r"(?:(?:aa|pad|par|ho|lag)\s+(?:raha|rahi|rahe|gaya|gayi|gaye|chuka|chuki)|aaya|aayi|aya"
    r"|ayi|pada|padi|pade|maar\s+(?:gaya|gayi|diya|di)|mar\s+(?:gaya|gayi)|maara|mara"
    r"|(?:hua|hui)\s+(?:hai|h|hain))"
)
#: "... aaya tha" (had, pluperfect) is history, not a current event.
_HI_NOT_PAST = r"(?!\s+(?:tha|thi|the)\b)"
_HI_HEART_ATTACK = (
    r"(?:heart\s+attack|hart\s+attack|dil\s+ka\s+daura|dil\s+ka\s+attack|heart\s+ka\s+attack"
    r"|cardiac\s+arrest)"
)
_HI_STROKE = (
    r"(?:lakwa|lakwe|paralysis|brain\s+stroke|stroke|brain\s+attack|brain\s+hemorrhage"
    r"|brain\s+haemorrhage)"
)
_HI_HEAD = r"(?:sar|sir|sarr|sirr)"
_HI_HEADACHE = rf"(?:{_HI_HEAD}\s*(?:(?:{_HI_IN}|ka)\s+)?dard|headache|head\s*ache)"
_HI_STAMMER = r"la[dr]h?kh?a?[dr]h?a\w*"
_HI_LIMB = r"(?:haath|baazu|pair|pairon|paon|paanv|taang|arm|leg|hand)"
_HI_SIDE = (
    r"(?:ek|aadhe|aadha|left|right|baayi|baayin|baayen|baaye|baaya|daayi|daayen|daaye|daaya"
    r"|ulti|ulta|seedhi|seedha)"
)
_HI_WANT = r"(?:chahta|chahti|chahte|chahiye|chahu|chahun|chahoon)"

_HINGLISH_RULES: tuple[tuple[str, re.Pattern[str], bool], ...] = (
    (
        "chest_pain_hi",
        _rx(
            rf"\b(?:seene|seena|chhati)\s+(?:(?:{_HI_IN}|ka|ki|ke)\s+)?(?:{_HI_CHEST_FILL}\s+){{0,4}}"
            rf"{_HI_PAIN_ALL}\b",
            rf"\b(?:chest|heart)\s+(?:{_HI_IN}|ka|ki|ke)\s+(?:{_HI_CHEST_FILL}\s+){{0,4}}"
            rf"{_HI_PAIN_ALL}\b",
            rf"\b(?:chest|heart)\s+(?:{_HI_CHEST_FILL}\s+){{0,3}}{_HI_PAIN_HI}\b",
            rf"\bdil\s+(?:(?:{_HI_IN}|ka|ki|ke)\s+)?(?:{_HI_CHEST_FILL}\s+){{0,3}}(?:dard|pain)\b",
            rf"\b(?:seene|seena|chhati|chest)\s+(?:{_HI_IN}\s+)?(?:{_HI_CHEST_FILL}\s+){{0,3}}"
            rf"(?:bhaari|jakad\w*|jakd\w*|dab\s+(?:{_HI_PROG}|gaya|gayi)|daba\s+hua|dabi\s+hui"
            rf"|dukh\w*|jal\s+{_HI_PROG}|jalta|jalti|bhinch\w*|ainth\w*)",
            rf"\b(?:dard|pain|jalan|dabav|bharipan|jakdan)\s+(?:(?:ho|uth|raha|rahi|rahe|hai|h|sa"
            rf"|si|bhi|ek|bahut|tez|hota|hoti)\s+){{0,4}}(?:seene|chhati|chest|heart|dil)\s+"
            rf"{_HI_IN}\b",
            r"\b(?:seene|chhati|chest)\s+se\s+(?:\w+\s+){0,3}?(?:tak|mein|me)\s+(?:\w+\s+)?"
            r"(?:dard|pain)\b",
            # Indian English
            r"\bchest\s+(?:is\s+)?paining\b|\buneasiness\s+(?:in|on|of)\s+(?:my\s+|the\s+)?chest\b"
            r"|\bchest\s+uneasiness\b",
        ),
        False,
    ),
    (
        "heart_attack_hi",
        _rx(
            rf"\b{_HI_HEART_ATTACK}\s+(?:(?:sa|jaisa|shayad|to|toh|abhi)\s+)?{_HI_STRUCK}"
            rf"{_HI_NOT_PAST}",
            rf"\b{_HI_HEART_ATTACK}\s+(?:jaisa|jaisi|jaise|sa|si)\s+(?:\w+\s+)?(?:lag|mehsoos"
            r"|feel)\w*",
            rf"\b(?:lagta|lag\s+raha|lag\s+rahi|shayad)\s+(?:hai\s+)?(?:\w+\s+){{0,3}}?"
            rf"{_HI_HEART_ATTACK}\s+(?:hai|h|ho\s+(?:raha|rahi|gaya|gayi)|aa\s+(?:raha|rahi|gaya"
            r"|gayi))\b",
            rf"\bkahin\s+(?:\w+\s+){{0,3}}?{_HI_HEART_ATTACK}\s+(?:(?:to|toh)\s+)?nahi\b",
        ),
        False,
    ),
    (
        "breathlessness_hi",
        _rx(
            r"\bsaans\s+(?:(?:theek|sahi|achhe|acche|properly|bilkul|puri|poori|thodi|zara|se"
            r"|bhi|ekdam)\s+){0,3}nahi\s+(?:aa|a|aati|aata|le|li|lee|ho|ban)\b",
            r"\bsaans\s+(?:le|lena|lene|lete|li)\s+nahi\s+(?:pa|paa|sak|ho)\w*",
            rf"\bsaans\s+(?:le|lena|lene|lete)\s+(?:(?:{_HI_IN}|se|waqt|samay)\s+)?"
            rf"(?:{_HI_INTENS}\s+){{0,2}}(?:takleef|dikkat|pareshani|problem|mushkil|trouble"
            r"|difficulty|issue|kathinai)\b",
            rf"\bsaans\s+(?:ki|ka|ke|{_HI_IN})\s+(?:{_HI_INTENS}\s+){{0,2}}(?:takleef|dikkat"
            r"|pareshani|problem|mushkil|trouble|kami|tangi|ghutan)\b",
            r"\bsaans\s+(?:(?:bahut|kaafi|thodi|thoda|bhi|ekdam|jaldi|zor\s+se|tez)\s+){0,2}"
            r"(?:phool|phul|fool|ful|chadh|chad|charh|ukhad|ukhar|atak|ruk|band|ghut|bhaari)\w*",
            r"\bdam\s+(?:\w+\s+)?(?:ghut|ghutan|nikal|ukhad|ukhar|phool|phul|fool)\w*",
            r"\b(?:haanf|hanf|haamf|hamf|haanph|hanph)\w*",
            rf"\bbreath(?:ing)?\s+(?:(?:{_HI_IN}|ki|ka|ke)\s+)?(?:{_HI_INTENS}\s+){{0,2}}"
            r"(?:takleef|dikkat|pareshani)\b",
            rf"\bbreath(?:ing)?\s+{_HI_IN}\s+(?:{_HI_INTENS}\s+){{0,2}}(?:problem|issue|trouble"
            r"|difficulty)\b|\bbreath\s+nahi\s+(?:aa|a|le|ho)\b",
            # Indian English
            r"\bbreathing\s+(?:difficulty|trouble)\b",
            r"\b(?:feel\w*|getting|having|get|got)\s+(?:(?:some|a|lot of|very|so|bit of)\s+)?"
            r"suffocat\w*|\bsuffocation\s+(?:ho|lag|feel)\w*",
        ),
        True,
    ),
    (
        "syncope_hi",
        _rx(
            r"\bbehosh\b(?!\s+(?:kar|karke|karne|karna|karte|kiya|kiye))",
            r"\bbehoshi\s+(?:(?:si|jaisi|ka|ki|bahut|thodi|thoda)\s+){0,2}(?:aa|aayi|aati|chha"
            rf"|chhaa|ho|hui|daura|lag|mehsoos)\w*{_HI_NOT_PAST}",
            rf"\bhosh\s+(?:(?:{_HI_IN}|bhi|hi|to|toh|bilkul)\s+){{0,2}}nahi\b",
            rf"\bhosh\s+(?:{_HI_IN}\s+)?(?:kho\w*|chala\s+gaya|chali\s+gayi)",
            r"\bchakkar\s+(?:\w+\s+){0,3}?(?:gir|gira|giri|gire|girna|girne|girte|girti|girta"
            r"|behosh)\b",
            r"\baankh\w*\s+(?:ke|ki)\s+(?:aage|samne|saamne)\s+(?:\w+\s+)?andhera\b",
        ),
        False,
    ),
    (
        "stroke_face_hi",
        _rx(
            r"\b(?:muh|chehra|chehre|honth)\s+(?:(?:ka|ki|ke|ek|aadha|aadhe|left|right|baaya"
            r"|baayan|daaya|daayan|hissa|hisse|taraf|side|thoda|thodi|bilkul|ekdam|achanak|bhi)"
            r"\s+){0,3}(?:tedha|latak\w*|latk\w*|tirchha|tircha|jhuk\w*|sunn|ek\s+taraf\s+(?:ho"
            r"|khinch|mud|jhuk)\w*|khinch\w*|bigad\w*)"
            # "muh tedha karna" (pulling a face) is deliberate, not a stroke sign
            r"(?!\s+(?:kar|kiya|kiye|karke|karna|karta|karti|karte|banaya|bana)\b)",
            r"\b(?:tedha|tirchha|tircha)\s+(?:muh|chehra|chehre)\b",
        ),
        False,
    ),
    (
        "stroke_limb_hi",
        _rx(
            rf"\b{_HI_LIMB}\s+(?:(?:{_HI_LIMB}|{_HI_SIDE}|bhi|bilkul|ekdam|achanak|ka|ki|ke|wala"
            rf"|thoda|thodi|poora|pura)\s+){{0,3}}(?:nahi\s+(?:hil|uth|utha|chal|mud)\w*|sunn\w*"
            r"|sun\s+(?:ho|pad|par)\w*|bejaan|be\s+jaan|kaam\s+nahi|latak\w*|lakwa)",
            rf"\b{_HI_SIDE}\s+(?:taraf|side|hissa|hisse|ang|part)\s+(?:\w+\s+){{0,3}}?(?:sunn\w*"
            r"|lakwa|kamzori|kamzor\w*|nahi\s+(?:hil|chal|uth)\w*|kaam\s+nahi|bejaan|be\s+jaan"
            r"|latak\w*)",
            rf"\b(?:haath|baazu|pair)\s+{_HI_IN}\s+(?:\w+\s+)?(?:kamzori|sunnpan|jaan\s+nahi"
            r"|taakat\s+nahi|takat\s+nahi)\b",
        ),
        False,
    ),
    (
        "stroke_speech_hi",
        _rx(
            rf"\b(?:aawaz|zubaan|jeebh|boli|bolna|bolne|bolte|baat|baatein|speech|words|bol)\s+"
            rf"(?:(?:bhi|thodi|thoda|bahut|achanak|ekdam|bilkul|se|mein|me|ki|ka)\s+){{0,2}}"
            rf"{_HI_STAMMER}",
            r"\b(?:zubaan|jeebh)\s+(?:(?:bhi|thodi|thoda|bahut|achanak|ekdam)\s+){0,2}(?:atak"
            r"|ulajh|bhaari|sunn|tedhi)\w*",
            r"\b(?:bol|bolne|(?<!se )(?<!ko )baat)\s+(?:(?:hi|bhi|theek\s+se|sahi\s+se|saaf"
            r"|kuch|properly|ekdam|bilkul)\s+){0,2}nahi\s+(?:kar\s+)?(?:pa|paa|sak|paya|payi)\w*",
            rf"\b(?:bolne|bolte|baat\s+karne|baat\s+karte)\s+(?:{_HI_IN}|waqt|samay)\s+"
            rf"(?:{_HI_INTENS}\s+){{0,2}}(?:takleef|dikkat|pareshani|problem|mushkil)\b",
            r"\b(?:shabd|alfaaz|words)\s+(?:\w+\s+)?(?:nahi\s+(?:nikal|mil|aa)|atak)\w*",
        ),
        False,
    ),
    (
        "stroke_vision_hi",
        _rx(
            r"\bachanak\s+(?:(?:se|hi)\s+)?(?:\w+\s+){0,2}?(?:dikhna|dikhai|dikhayi|nazar)\s+"
            r"(?:\w+\s+)?(?:band|kam|chala)\w*",
            r"\b(?:ek|left|right|baayi|daayi)\s+aankh\w*\s+(?:se|mein|me|main)\s+(?:\w+\s+){0,2}?"
            r"(?:dikhna\s+band|dikhai\s+nahi|dikhayi\s+nahi|dikh\s+nahi|kuch\s+nahi\s+dikh"
            r"|nahi\s+dikh|andhera|dhundhla)\w*",
            r"\b(?:sab|sabkuch|cheezein|chizein|har\s+cheez)\s+(?:\w+\s+)?do\s+do\s+(?:dikh"
            r"|nazar)\w*",
        ),
        False,
    ),
    (
        "stroke_self_report_hi",
        _rx(
            rf"\b{_HI_STROKE}\s+(?:(?:ka|ki)\s+(?:attack|daura|asar)\s+)?(?:(?:sa|jaisa|shayad|to"
            rf"|toh)\s+)?{_HI_STRUCK}{_HI_NOT_PAST}",
            rf"\bkahin\s+(?:\w+\s+){{0,3}}?{_HI_STROKE}\s+(?:(?:to|toh)\s+)?nahi\b",
            rf"\b(?:lagta|lag\s+raha|lag\s+rahi|shayad)\s+(?:hai\s+)?(?:\w+\s+){{0,3}}?"
            rf"{_HI_STROKE}\s+(?:hai|h|ho\s+(?:raha|rahi|gaya|gayi)|aa\s+(?:raha|rahi|gaya|gayi))\b",
        ),
        False,
    ),
    (
        "severe_headache_hi",
        _rx(
            r"\b(?:(?:bahut|ekdam|itna|itni|aisa|bahut\s+hi)\s+(?:tez|zor\s+ka|zordar)"
            r"|bhayankar|bhayanak|asahniya|asehniya|jaanleva|jaan\s+leva|zabardast"
            r"|bardasht\s+se\s+bahar|sabse\s+(?:bura|tez|zyada|kharab|bhayankar))\s+"
            rf"(?:(?:sa|wala|vala)\s+)?{_HI_HEADACHE}\b",
            rf"\b{_HI_HEAD}\s+(?:{_HI_IN}|ka)\s+(?:(?:bahut|ekdam|itna|achanak|se|hi)\s+){{0,2}}"
            r"(?:tez|bhayankar|bhayanak|asahniya|jaanleva|zabardast|zordar)\s+(?:sa\s+)?dard\b",
            rf"\b{_HI_HEAD}\s+(?:(?:dard|{_HI_IN}|bhi|se|mano|jaise|aisa|lag\s+raha)\s+){{0,3}}"
            r"(?:phat|fat|phatt|fatt)\w*\s+(?:raha|rahi|rahe|ja|jaa|jayega|jaega|jaaega|gaya"
            r"|jata|jaata|sa)\b",
            r"\bachanak\s+(?:(?:se|hi|bahut|ekdam|tez|zordar|bhayankar)\s+){0,3}"
            rf"(?:{_HI_HEAD}\s+(?:(?:{_HI_IN}|ka)\s+)?(?:(?:bahut|tez)\s+){{0,2}}dard"
            r"|headache)\b",
        ),
        False,
    ),
    (
        "hemoptysis_hi",
        _rx(
            rf"\b(?:khansi|khaans\w*|khans\w*|cough\w*)\s+(?:(?:{_HI_IN}|ke\s+saath|ke\s+sath|se"
            r"|ke\s+baad|karte\s+waqt|karne\s+par|waqt)\s+)?(?:(?:bhi|thoda|thodi|bahut|kaafi)"
            r"\s+)?khoon\b",
            r"\bkhoon\s+(?:wali|vali|wala|vala|ki|ka|bhari)\s+(?:khansi|balgam|thook|kaff)\b",
            rf"\b(?:balgam|thook|thuk|kaff|phlegm|sputum)\s+(?:{_HI_IN}|ke\s+saath|ke\s+sath|se)"
            r"\s+(?:(?:bhi|thoda|thodi|bahut|kaafi)\s+)?khoon\b",
        ),
        False,
    ),
    (
        "hematemesis_hi",
        _rx(
            r"\bkhoon\s+(?:ki|wali|vali|ka)\s+(?:ulti|ultiyan)\b",
            rf"\b(?:ulti|ultiyan|vomit\w*)\s+(?:{_HI_IN}|ke\s+saath|ke\s+sath|se)\s+"
            r"(?:(?:bhi|thoda|thodi|bahut|kaafi)\s+)?khoon\b",
        ),
        False,
    ),
    (
        "self_harm_hi",
        _rx(
            r"\b(?:marna|marne|mar\s+(?:jaana|jana|jaun|jaaun|jaane))\s+(?:(?:hi|bhi)\s+)?"
            rf"(?:{_HI_WANT}|ka\s+(?:mann|man|dil)|ki\s+(?:soch|ichha|iccha|khwahish))\w*",
            rf"\b(?:jeena|jeene|zinda|zindagi|jeevan)\s+(?:(?:bhi|ab|aur|hi)\s+)?(?:nahi|na)\s+"
            rf"(?:{_HI_WANT}|rehna|rahna)\w*",
            r"\bjeene\s+(?:ka|ki|ke)\s+(?:koi\s+)?(?:mann|man|dil|ichha|iccha|wajah|matlab|maksad"
            r"|irada)\s+(?:nahi|khatam|na)\b",
            r"\b(?:khudkushi|aatmhatya)\w*|\bmujhe\s+(?:bas\s+)?marna\s+(?:hai|h)\b",
            r"\b(?:khud\s+ko|khudko|apne\s+aap\s+ko|apneaap\s+ko|apne\s+ko)\s+(?:\w+\s+)?(?:khatam"
            r"|maar|nuksaan|nuksan|chot|hurt|kaat|harm)\w*",
            r"\b(?:zindagi|jeevan)\s+(?:\w+\s+)?khatam\s+(?:kar|karna|karne|karke)\w*",
        ),
        False,
    ),
    (
        "seizure_hi",
        _rx(
            r"\bmirgi\s+(?:(?:ka|ke|ki)\s+)?(?:(?:daura|attack|fit|jhatke)\s+)?(?:(?:aa|pad)\s+"
            rf"(?:{_HI_PROG}|gaya|gayi|gaye)|aaya|aayi|aaye|pada|padi|pade|ho\s+(?:{_HI_PROG}"
            rf"|gaya|gayi)|hua|hui)\b{_HI_NOT_PAST}",
            r"(?<!dil ka )(?<!heart ka )(?<!behoshi ka )(?<!lakwe ka )\b(?:daura|jhatke|fits?)\s+"
            rf"(?:(?:ka|ke|sa|jaisa)\s+)?(?:(?:aa|pad)\s+(?:{_HI_PROG}|gaya|gayi|gaye)|aaya|aayi"
            rf"|aaye|pada|padi|pade)\b{_HI_NOT_PAST}",
            r"\b(?:haath\s+pair|sharir|shareer|body)\s+(?:\w+\s+)?(?:akad|ainth)\w*\s+(?:gaya|gayi"
            rf"|gaye|{_HI_PROG})\b",
            r"\bmuh\s+se\s+(?:jhaag|jhag|fen|phen)\b",
        ),
        False,
    ),
)

#: Helper patterns that only escalate in combination (composite rules).
_PALPITATIONS_RE = _rx(
    r"\bpalpitations?\b|\bracing heart\b",
    r"\b(?:heart|heartbeat|pulse)\s+(?:(?:is|was|keeps|started)\s+)?(?:\w+\s+)?(?:racing"
    r"|pounding|fluttering|flutters|skipping|irregular|beating (?:really |very |so )?(?:fast"
    r"|hard|irregularly|out of rhythm))\b",
    r"\b(?:dil|heart)\s+(?:ki\s+)?(?:dhadkan\s+)?(?:(?:bahut|ekdam|itna|itni)\s+)?(?:tez|zor\s+se"
    r"|zor\s+zor\s+se)\s+(?:dhadak|chal|bhaag|ho\s+(?:gayi|rahi))\w*",
    r"\bdhadkan\s+(?:(?:bahut|ekdam|bhi)\s+)?(?:tez|badh|bekaabu|irregular|aniyamit)\w*",
)
_PREGNANCY_RE = _rx(
    r"\bpregnant\b(?<!get pregnant)(?<!become pregnant)(?<!getting pregnant)"
    r"(?<!becoming pregnant)",
    r"\b(?:my|this|current)\s+pregnancy\b|\b\d{1,2}\s*(?:weeks?|wks?)\s+(?:along|gestation)\b",
    r"\bpost[- ]?partum\b|\b(?:just|recently)\s+(?:gave birth|had (?:a|my) baby|delivered)\b",
    r"\b(?:first|second|third|1st|2nd|3rd)\s+trimester\b",
    r"\b(?:garbhvati|garbhwati|garbhavati)\b|\bpet\s+se\s+(?:hoon|hai|h)\b",
)
_PREECLAMPSIA_SYMPTOM_RE = _rx(
    r"\b(?:severe|bad|terrible|awful|horrible|pounding|throbbing|really bad|very bad"
    r"|persistent|constant|intense|worst|blinding|splitting)\s+(?:\w+\s+)?(?:headache"
    r"|head\s*ache|head pain)\b",
    r"\bheadache\s+(?:that\s+)?(?:won't|will not|doesn't|does not|wont)\s+go away\b",
    r"\b(?:blurr\w*|blurry|double|changes in (?:my )?|change in (?:my )?|loss of|dim(?:med)?"
    r"|spotty)\s*vision\b|\bvision\s+(?:\w+\s+)?(?:chang\w*|blurr\w*|blurry|problems?"
    r"|disturb\w*|loss|spots|dim\w*)\b",
    r"\bseeing (?:spots|stars|flash\w*)|\bspots in (?:my )?(?:vision|eyes)\b|\bflashing lights\b",
    r"\b(?:sudden|rapid|severe|a lot of)\s+(?:\w+\s+)?swelling\b",
    r"\bswelling\s+(?:\w+\s+)?(?:in|of)\s+(?:my\s+)?(?:face|hands|eyes|eyelids|fingers)\b",
    r"\b(?:face|hands|eyes|eyelids|fingers)\s+(?:(?:are|is|got|have|has|became|went)\s+)?"
    r"(?:(?:suddenly|really|very)\s+)?(?:swollen|puffy|swelling|swelled)\b",
    r"\bupper\s+(?:abdominal|belly|stomach|tummy)\s+pain\b|\bepigastric\b"
    r"|\bpain\s+(?:under|below)\s+(?:my\s+)?(?:right\s+)?ribs?\b",
    rf"\b(?:bahut|tez|bhayankar)\s+(?:\w+\s+)?{_HI_HEADACHE}\b|\bdhundhla\s+(?:dikh|nazar)\w*",
    r"\b(?:chehre|chehra|haath|pair|pairon|aankh\w*)\s+(?:\w+\s+){0,2}?(?:sujan|soojan|sooj|sujh)\w*",
)
_LOW_BP_SYMPTOM_RE = _rx(
    r"\b(?:dizz\w*|light[- ]?headed\w*|faint\w*|woozy|weak\w*|confus\w*|clammy|cold sweats?"
    r"|drows\w*|pale|giddy|giddiness|chakkar|kamzori|ghabrahat|behosh\w*|pasina|pasine)\b"
)

_ALL_RULES = _RED_FLAG_RULES + _HINGLISH_RULES
#: Rules after which racing / pounding palpitations are reported as severe.
_PALPITATION_TRIGGERS = ("syncope", "chest_pain", "syncope_hi", "chest_pain_hi")
#: Rule ids of the Hinglish / Indian-English lexicon (subset of :data:`RULE_IDS`).
HINGLISH_RULE_IDS: tuple[str, ...] = tuple(rule_id for rule_id, _, _ in _HINGLISH_RULES)

#: Every rule id this module can report, in reporting order.
RULE_IDS: tuple[str, ...] = (
    "bp_crisis",
    "bp_low_symptomatic",
    *(rule_id for rule_id, _, _ in _RED_FLAG_RULES),
    *(rule_id for rule_id, _, _ in _HINGLISH_RULES),
    "palpitations_severe",
    "preeclampsia",
)


# ======================================================================
# 5. Public API
# ======================================================================


def _affirmed_spans(pattern: re.Pattern[str], analysis: _Analysis) -> list[tuple[int, int]]:
    """Affirmed (non-negated, non-suppressed) match spans of a helper pattern."""
    return [
        m.span()
        for m in pattern.finditer(analysis.text)
        if analysis.status(m.start(), m.end(), exertion=False) == "affirmed"
    ]


def classify_detailed(message: str) -> EmergencyResult:
    """Classify a raw patient message and explain the decision.

    Deterministic, LLM-free and network-free; runs before any LLM call.

    Args:
        message: The raw (un-redacted) patient message.

    Returns:
        An :class:`EmergencyResult`; ``action`` is ``ActionEnum.ESCALATE`` if any
        rule fired, otherwise ``None``.
    """
    text = normalize_text(message) if message else ""
    if not text:
        return EmergencyResult(action=None)
    analysis = _Analysis(text)
    fired: dict[str, list[tuple[int, int]]] = {}
    negated: set[str] = set()
    suppressed: set[str] = set()

    # --- blood pressure --------------------------------------------------
    has_context = bool(_BP_CONTEXT_RE.search(text))
    readings = [
        r
        for r in _extract_readings(text, has_context, analysis.personal)
        if not analysis.clause_at(r.start).historical
    ]
    crisis = next((r for r in readings if _is_crisis(r, has_context)), None)
    low = next(
        (r for r in readings if has_context and 0 < r.systolic < SYSTOLIC_LOW), None
    )
    decisive = crisis or low or (readings[0] if readings else None)
    bp = _report_bp(decisive, readings, analysis) if decisive else None
    if crisis is not None:
        fired["bp_crisis"] = [(crisis.start, crisis.end)]

    # --- red-flag symptoms -------------------------------------------------
    raw_matches = [
        (rule_id, m.start(), m.end(), exertion)
        for rule_id, pattern, exertion in _ALL_RULES
        for m in pattern.finditer(text)
    ]
    analysis.set_rule_spans([(start, end) for _, start, end, _ in raw_matches])
    for rule_id, start, end, exertion in raw_matches:
        verdict = analysis.status(start, end, exertion=exertion)
        if verdict == "affirmed":
            fired.setdefault(rule_id, []).append((start, end))
        elif verdict == "negated":
            negated.add(rule_id)
        else:
            suppressed.add(rule_id)

    # --- composite rules ---------------------------------------------------
    if low is not None:
        symptoms = _affirmed_spans(_LOW_BP_SYMPTOM_RE, analysis)
        if symptoms:
            fired["bp_low_symptomatic"] = [(low.start, low.end), *symptoms]
    if any(rule_id in fired for rule_id in _PALPITATION_TRIGGERS):
        palpitations = _affirmed_spans(_PALPITATIONS_RE, analysis)
        if palpitations:
            fired["palpitations_severe"] = palpitations
    pregnancy = _affirmed_spans(_PREGNANCY_RE, analysis)
    if pregnancy:
        warning_signs = _affirmed_spans(_PREECLAMPSIA_SYMPTOM_RE, analysis)
        if warning_signs:
            fired["preeclampsia"] = pregnancy + warning_signs

    matched = [rule_id for rule_id in RULE_IDS if rule_id in fired]
    spans = sorted({span for rule_id in matched for span in fired[rule_id]})
    evidence = list(dict.fromkeys(text[s:e] for s, e in spans))
    return EmergencyResult(
        action=ActionEnum.ESCALATE if matched else None,
        matched_rules=matched,
        negated=[r for r in RULE_IDS if r in negated and r not in fired],
        bp=bp,
        evidence_spans=evidence,
        suppressed=[r for r in RULE_IDS if r in suppressed and r not in fired],
    )


def classify(message: str) -> ActionEnum | None:
    """Evaluate a raw patient message against the deterministic emergency rules.

    This function runs **before** any LLM invocation in the pipeline and must
    complete well within 10 ms (typically < 2 ms).

    Args:
        message: The raw patient message text.

    Returns:
        ``ActionEnum.ESCALATE`` if an emergency is detected, ``None`` if the
        message is safe to continue through the pipeline.
    """
    return classify_detailed(message).action
