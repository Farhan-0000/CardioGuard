"""
Cardiovascular drug-name normaliser (brand → generic) for MedBridge.

Patients usually name their medicines by brand ("Telma 40", "Norvasc",
"Met XL"), while guidelines use generic (INN) names. This module maps brand
names — US/EU and Indian — to their generic ingredients and drug class so that

* the Context Extractor stores medications under one canonical generic name
  (the state projector then de-duplicates brand/generic mentions), and
* query expansion can add the generic name to the guideline search query.

Only mappings the authors are confident about are included (they are cited in
the project report); accuracy matters more than coverage. Combination products
use ``" + "`` between ingredients (``"telmisartan + hydrochlorothiazide"``).

Everything here is deterministic, dependency-free and fast (< 1 ms per call).
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from functools import lru_cache

# ---------------------------------------------------------------------------
# Generic name → drug class
# ---------------------------------------------------------------------------

_ACEI = "ACE inhibitor"
_ARB = "angiotensin receptor blocker"
_CCB_DHP = "dihydropyridine calcium channel blocker"
_CCB_NDHP = "non-dihydropyridine calcium channel blocker"
_BB = "beta-blocker"
_THIAZIDE = "thiazide diuretic"
_THIAZIDE_LIKE = "thiazide-like diuretic"
_LOOP = "loop diuretic"
_K_SPARING = "potassium-sparing diuretic"
_MRA = "mineralocorticoid receptor antagonist"
_NITRATE = "nitrate"
_STATIN = "statin"
_ANTIPLATELET = "antiplatelet"
_P2Y12 = "P2Y12 inhibitor antiplatelet"
_DOAC = "direct oral anticoagulant"
_VKA = "vitamin K antagonist anticoagulant"
_SGLT2 = "SGLT2 inhibitor"
_GLP1 = "GLP-1 receptor agonist"
_ANTIARRHYTHMIC = "antiarrhythmic"

GENERIC_CLASS: dict[str, str] = {
    # Renin-angiotensin system
    "lisinopril": _ACEI,
    "enalapril": _ACEI,
    "ramipril": _ACEI,
    "perindopril": _ACEI,
    "benazepril": _ACEI,
    "quinapril": _ACEI,
    "captopril": _ACEI,
    "losartan": _ARB,
    "valsartan": _ARB,
    "telmisartan": _ARB,
    "olmesartan": _ARB,
    "irbesartan": _ARB,
    "candesartan": _ARB,
    "azilsartan": _ARB,
    "aliskiren": "direct renin inhibitor",
    # Calcium channel blockers
    "amlodipine": _CCB_DHP,
    "nifedipine": _CCB_DHP,
    "felodipine": _CCB_DHP,
    "cilnidipine": _CCB_DHP,
    "lercanidipine": _CCB_DHP,
    "diltiazem": _CCB_NDHP,
    "verapamil": _CCB_NDHP,
    # Diuretics
    "hydrochlorothiazide": _THIAZIDE,
    "bendroflumethiazide": _THIAZIDE,
    "chlorthalidone": _THIAZIDE_LIKE,
    "indapamide": _THIAZIDE_LIKE,
    "metolazone": _THIAZIDE_LIKE,
    "furosemide": _LOOP,
    "torsemide": _LOOP,
    "bumetanide": _LOOP,
    "amiloride": _K_SPARING,
    "triamterene": _K_SPARING,
    "spironolactone": _MRA,
    "eplerenone": _MRA,
    "finerenone": "non-steroidal mineralocorticoid receptor antagonist",
    # Beta-blockers
    "metoprolol": _BB,
    "metoprolol tartrate": _BB,
    "metoprolol succinate": _BB,
    "atenolol": _BB,
    "bisoprolol": _BB,
    "carvedilol": _BB,
    "nebivolol": _BB,
    "propranolol": _BB,
    "nadolol": _BB,
    "labetalol": _BB,
    # Other antihypertensives
    "clonidine": "central alpha-2 agonist",
    "methyldopa": "central alpha-2 agonist",
    "doxazosin": "alpha-1 blocker",
    "prazosin": "alpha-1 blocker",
    "hydralazine": "direct vasodilator",
    # Anti-anginal
    "isosorbide mononitrate": _NITRATE,
    "isosorbide dinitrate": _NITRATE,
    "nitroglycerin": _NITRATE,
    "ranolazine": "anti-anginal (late sodium current inhibitor)",
    "trimetazidine": "anti-anginal (metabolic modulator)",
    "nicorandil": "anti-anginal (potassium channel opener)",
    "ivabradine": "If channel inhibitor",
    # Lipid lowering
    "atorvastatin": _STATIN,
    "rosuvastatin": _STATIN,
    "simvastatin": _STATIN,
    "pravastatin": _STATIN,
    "pitavastatin": _STATIN,
    "ezetimibe": "cholesterol absorption inhibitor",
    "evolocumab": "PCSK9 inhibitor",
    "alirocumab": "PCSK9 inhibitor",
    "inclisiran": "PCSK9 small interfering RNA",
    "bempedoic acid": "ATP citrate lyase inhibitor",
    "fenofibrate": "fibrate",
    "gemfibrozil": "fibrate",
    "icosapent ethyl": "omega-3 fatty acid",
    # Antithrombotics
    "aspirin": _ANTIPLATELET,
    "clopidogrel": _P2Y12,
    "ticagrelor": _P2Y12,
    "prasugrel": _P2Y12,
    "apixaban": _DOAC,
    "rivaroxaban": _DOAC,
    "dabigatran": _DOAC,
    "edoxaban": _DOAC,
    "warfarin": _VKA,
    "acenocoumarol": _VKA,
    # Cardio-metabolic
    "dapagliflozin": _SGLT2,
    "empagliflozin": _SGLT2,
    "canagliflozin": _SGLT2,
    "semaglutide": _GLP1,
    "liraglutide": _GLP1,
    "dulaglutide": _GLP1,
    "metformin": "biguanide",
    # Heart failure / rhythm
    "digoxin": "cardiac glycoside",
    "vericiguat": "soluble guanylate cyclase stimulator",
    "amiodarone": _ANTIARRHYTHMIC,
    "dronedarone": _ANTIARRHYTHMIC,
    "dofetilide": _ANTIARRHYTHMIC,
    "flecainide": _ANTIARRHYTHMIC,
    "propafenone": _ANTIARRHYTHMIC,
}

# Combination products whose class is not simply the sum of their parts.
_COMBINATION_CLASS: dict[str, str] = {
    "sacubitril + valsartan": "angiotensin receptor-neprilysin inhibitor (ARNI)",
}

# Alternative generic spellings / abbreviations → canonical generic.
GENERIC_ALIASES: dict[str, str] = {
    "hctz": "hydrochlorothiazide",
    "frusemide": "furosemide",
    "torasemide": "torsemide",
    "chlortalidone": "chlorthalidone",
    "glyceryl trinitrate": "nitroglycerin",
    "gtn": "nitroglycerin",
    "acetylsalicylic acid": "aspirin",
    "sacubitril valsartan": "sacubitril + valsartan",
    "sacubitril/valsartan": "sacubitril + valsartan",
    "valsartan sacubitril": "sacubitril + valsartan",
    "valsartan/sacubitril": "sacubitril + valsartan",
    # Sacubitril is only marketed combined with valsartan (ARNI).
    "sacubitril": "sacubitril + valsartan",
}

# ---------------------------------------------------------------------------
# Brand → generic
# ---------------------------------------------------------------------------

_HCTZ = "hydrochlorothiazide"

_BRANDS_INTERNATIONAL: dict[str, str] = {
    # ACE inhibitors (+ combinations)
    "Zestril": "lisinopril",
    "Prinivil": "lisinopril",
    "Zestoretic": f"lisinopril + {_HCTZ}",
    "Vasotec": "enalapril",
    "Vaseretic": f"enalapril + {_HCTZ}",
    "Lotensin": "benazepril",
    "Lotrel": "amlodipine + benazepril",
    "Altace": "ramipril",
    "Tritace": "ramipril",
    "Accupril": "quinapril",
    "Coversyl": "perindopril",
    "Aceon": "perindopril",
    "Capoten": "captopril",
    # ARBs (+ combinations), ARNI, renin inhibitor
    "Cozaar": "losartan",
    "Hyzaar": f"losartan + {_HCTZ}",
    "Diovan": "valsartan",
    "Diovan HCT": f"valsartan + {_HCTZ}",
    "Exforge": "amlodipine + valsartan",
    "Benicar": "olmesartan",
    "Benicar HCT": f"olmesartan + {_HCTZ}",
    "Azor": "amlodipine + olmesartan",
    "Tribenzor": f"olmesartan + amlodipine + {_HCTZ}",
    "Micardis": "telmisartan",
    "Micardis HCT": f"telmisartan + {_HCTZ}",
    "Twynsta": "telmisartan + amlodipine",
    "Avapro": "irbesartan",
    "Avalide": f"irbesartan + {_HCTZ}",
    "Atacand": "candesartan",
    "Edarbi": "azilsartan",
    "Tekturna": "aliskiren",
    "Entresto": "sacubitril + valsartan",
    # Calcium channel blockers
    "Norvasc": "amlodipine",
    "Istin": "amlodipine",
    "Procardia": "nifedipine",
    "Adalat": "nifedipine",
    "Plendil": "felodipine",
    "Cardizem": "diltiazem",
    "Calan": "verapamil",
    "Isoptin": "verapamil",
    # Diuretics and MRAs
    "Microzide": _HCTZ,
    "Hygroton": "chlorthalidone",
    "Thalitone": "chlorthalidone",
    "Zaroxolyn": "metolazone",
    "Lasix": "furosemide",
    "Bumex": "bumetanide",
    "Demadex": "torsemide",
    "Midamor": "amiloride",
    "Dyazide": f"triamterene + {_HCTZ}",
    "Maxzide": f"triamterene + {_HCTZ}",
    "Aldactone": "spironolactone",
    "Inspra": "eplerenone",
    "Kerendia": "finerenone",
    # Beta-blockers
    "Lopressor": "metoprolol tartrate",
    "Toprol-XL": "metoprolol succinate",
    "Tenormin": "atenolol",
    "Tenoretic": "atenolol + chlorthalidone",
    "Zebeta": "bisoprolol",
    "Ziac": f"bisoprolol + {_HCTZ}",
    "Cardicor": "bisoprolol",
    "Coreg": "carvedilol",
    "Bystolic": "nebivolol",
    "Nebilet": "nebivolol",
    "Inderal": "propranolol",
    "Corgard": "nadolol",
    "Trandate": "labetalol",
    # Other antihypertensives
    "Catapres": "clonidine",
    "Aldomet": "methyldopa",
    "Cardura": "doxazosin",
    "Minipress": "prazosin",
    "Apresoline": "hydralazine",
    "BiDil": "isosorbide dinitrate + hydralazine",
    # Anti-anginal
    "Imdur": "isosorbide mononitrate",
    "Isordil": "isosorbide dinitrate",
    "Nitrostat": "nitroglycerin",
    "Ranexa": "ranolazine",
    "Corlanor": "ivabradine",
    "Procoralan": "ivabradine",
    # Lipid lowering
    "Lipitor": "atorvastatin",
    "Crestor": "rosuvastatin",
    "Zocor": "simvastatin",
    "Pravachol": "pravastatin",
    "Livalo": "pitavastatin",
    "Zetia": "ezetimibe",
    "Ezetrol": "ezetimibe",
    "Vytorin": "ezetimibe + simvastatin",
    "Caduet": "amlodipine + atorvastatin",
    "Repatha": "evolocumab",
    "Praluent": "alirocumab",
    "Leqvio": "inclisiran",
    "Nexletol": "bempedoic acid",
    "Tricor": "fenofibrate",
    "Lopid": "gemfibrozil",
    "Vascepa": "icosapent ethyl",
    # Antithrombotics
    "Ecotrin": "aspirin",
    "Plavix": "clopidogrel",
    "Brilinta": "ticagrelor",
    "Brilique": "ticagrelor",
    "Effient": "prasugrel",
    "Efient": "prasugrel",
    "Eliquis": "apixaban",
    "Xarelto": "rivaroxaban",
    "Pradaxa": "dabigatran",
    "Savaysa": "edoxaban",
    "Lixiana": "edoxaban",
    "Coumadin": "warfarin",
    "Jantoven": "warfarin",
    "Sintrom": "acenocoumarol",
    # Cardio-metabolic
    "Farxiga": "dapagliflozin",
    "Forxiga": "dapagliflozin",
    "Jardiance": "empagliflozin",
    "Invokana": "canagliflozin",
    "Ozempic": "semaglutide",
    "Wegovy": "semaglutide",
    "Victoza": "liraglutide",
    "Trulicity": "dulaglutide",
    "Glucophage": "metformin",
    # Heart failure / rhythm
    "Lanoxin": "digoxin",
    "Verquvo": "vericiguat",
    "Cordarone": "amiodarone",
    "Pacerone": "amiodarone",
    "Multaq": "dronedarone",
    "Tikosyn": "dofetilide",
    "Tambocor": "flecainide",
    "Rythmol": "propafenone",
}

_BRANDS_INDIA: dict[str, str] = {
    # Telmisartan family
    "Telma": "telmisartan",
    "Telma-H": f"telmisartan + {_HCTZ}",
    "Telma-AM": "telmisartan + amlodipine",
    "Tazloc": "telmisartan",
    "Telsartan": "telmisartan",
    # Other ARBs / ACE inhibitors / ARNI
    "Losar": "losartan",
    "Repace": "losartan",
    "Losacar": "losartan",
    "Olmezest": "olmesartan",
    "Cardace": "ramipril",
    "Envas": "enalapril",
    "Vymada": "sacubitril + valsartan",
    # Calcium channel blockers
    "Amlong": "amlodipine",
    "Stamlo": "amlodipine",
    "Amlokind": "amlodipine",
    "Amlodac": "amlodipine",
    "Amlopres": "amlodipine",
    "Amlopres-AT": "amlodipine + atenolol",
    "Cilacar": "cilnidipine",
    "Nicardia": "nifedipine",
    "Depin": "nifedipine",
    "Calcigard": "nifedipine",
    "Dilzem": "diltiazem",
    "Calaptin": "verapamil",
    # Beta-blockers
    "Concor": "bisoprolol",
    "Metolar": "metoprolol",
    "Met XL": "metoprolol succinate",
    "Revelol": "metoprolol",
    "Seloken": "metoprolol",
    "Betaloc": "metoprolol",
    "Nebicard": "nebivolol",
    "Nebistar": "nebivolol",
    "Aten": "atenolol",
    "Betacard": "atenolol",
    "Carloc": "carvedilol",
    "Cardivas": "carvedilol",
    "Ciplar": "propranolol",
    # Diuretics
    "Dytor": "torsemide",
    "Aquazide": _HCTZ,
    "Lorvas": "indapamide",
    "Natrilix": "indapamide",
    # Central / alpha agents
    "Arkamin": "clonidine",
    "Alphadopa": "methyldopa",
    "Prazopress": "prazosin",
    # Anti-anginal
    "Ivabrad": "ivabradine",
    "Sorbitrate": "isosorbide dinitrate",
    "Monotrate": "isosorbide mononitrate",
    "Nitrocontin": "nitroglycerin",
    "Angispan-TR": "nitroglycerin",
    "Flavedon": "trimetazidine",
    "Nikoran": "nicorandil",
    # Lipid lowering
    "Rosuvas": "rosuvastatin",
    "Rozavel": "rosuvastatin",
    "Storvas": "atorvastatin",
    "Atorva": "atorvastatin",
    "Atocor": "atorvastatin",
    "Tonact": "atorvastatin",
    "Atorlip": "atorvastatin",
    # Antithrombotics
    "Ecosprin": "aspirin",
    "Ecosprin-AV": "aspirin + atorvastatin",
    "Clopilet": "clopidogrel",
    "Deplatt": "clopidogrel",
    "Acitrom": "acenocoumarol",
    "Warf": "warfarin",
    # Cardio-metabolic
    "Glycomet": "metformin",
}

BRAND_TO_GENERIC: dict[str, str] = {**_BRANDS_INTERNATIONAL, **_BRANDS_INDIA}
INDIAN_BRANDS: frozenset[str] = frozenset(_BRANDS_INDIA)

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

FUZZY_CUTOFF = 0.88
_FUZZY_MIN_LEN = 5
# In free-text scanning, fuzzy matching is restricted to longer words so that
# ordinary English words never match short brand names.
_SCAN_FUZZY_MIN_LEN = 7

_RELEASE_SUFFIXES: frozenset[str] = frozenset(
    {"er", "xl", "xr", "sr", "la", "cr", "tr", "od", "mr", "cd", "zok", "retard"}
)
_SALT_WORDS: frozenset[str] = frozenset({
    "besylate", "besilate", "maleate", "mesylate", "hydrochloride", "hcl", "potassium",
    "sodium", "calcium", "magnesium", "medoxomil", "cilexetil", "fumarate", "erbumine",
    "arginine", "tablet", "tablets", "tab", "tabs", "capsule", "capsules",
})
_STRENGTH_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|iu)?\b")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*")
# Two scanned tokens form one name only when nothing but spaces, a hyphen or a
# slash separates them ("Met XL", "Telma-AM", "valsartan/sacubitril"); a strength
# in between ("Telma 40 am") means they are separate words.
_BIGRAM_GAP = re.compile(r"\s*[-/]?\s*")
# Ordinary English words that collide with short brand suffixes ("Amlopres at
# night", "Telma am"). Written in lower case after a brand they are read as
# English; upper case ("Telma AM") or hyphenated ("Telma-AM") still matches.
_SUFFIX_ENGLISH_WORDS: frozenset[str] = frozenset({
    "a", "am", "an", "as", "at", "by", "in", "is", "it", "me", "my", "no", "of", "on", "or",
    "pm", "so", "to", "up", "us", "we",
})
# Times of day after a suffix mean the suffix is the English word ("AT NIGHT").
_TIME_WORDS: frozenset[str] = frozenset({
    "night", "nights", "bedtime", "morning", "mornings", "noon", "evening", "evenings",
    "lunch", "breakfast", "dinner", "daytime", "midday",
})
# Common words that must never be fuzzy-matched to a drug while scanning text.
_SCAN_STOPWORDS: frozenset[str] = frozenset({
    "efficient", "pressure", "patient", "control", "concern", "contain", "content",
    "condition", "consider", "continue", "medicine", "medication", "tablets",
    "morning", "evening", "between", "without", "diastolic", "systolic",
})


@dataclass(frozen=True)
class DrugInfo:
    """A recognised drug: canonical generic name, the brand as known, and its class."""

    generic: str
    brand: str | None
    drug_class: str


def _normalise_text(name: str) -> str:
    """Lower-case, drop trademark symbols, unify hyphens/slashes/underscores to spaces."""
    text = name.lower().replace("®", "").replace("™", "")
    text = re.sub(r"[-_/,()]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_strength(text: str) -> str:
    return re.sub(r"\s+", " ", _STRENGTH_RE.sub(" ", text)).strip()


def _strip_trailing(text: str, words: frozenset[str]) -> str:
    parts = text.split()
    while len(parts) > 1 and parts[-1] in words:
        parts.pop()
    return " ".join(parts)


def _class_for(generic: str) -> str:
    if generic in _COMBINATION_CLASS:
        return _COMBINATION_CLASS[generic]
    parts = [p.strip() for p in generic.split("+")]
    return " + ".join(GENERIC_CLASS.get(p, "cardiovascular medicine") for p in parts)


def _build_index() -> dict[str, DrugInfo]:
    index: dict[str, DrugInfo] = {}
    for generic in GENERIC_CLASS:
        index[_normalise_text(generic)] = DrugInfo(generic, None, GENERIC_CLASS[generic])
    for alias, generic in GENERIC_ALIASES.items():
        index.setdefault(_normalise_text(alias), DrugInfo(generic, None, _class_for(generic)))
    for brand, generic in BRAND_TO_GENERIC.items():
        info = DrugInfo(generic, brand, _class_for(generic))
        key = _normalise_text(brand)
        index[key] = info
        index.setdefault(key.replace(" ", ""), info)
        # "Toprol" → Toprol-XL; skipped for short bases ("Met" XL) to avoid common words.
        base = _strip_trailing(key, _RELEASE_SUFFIXES)
        if base != key and len(base) >= _FUZZY_MIN_LEN:
            index.setdefault(base, info)
    return index


_INDEX: dict[str, DrugInfo] = _build_index()
_FUZZY_KEYS: list[str] = sorted(k for k in _INDEX if len(k) >= _FUZZY_MIN_LEN)
_SCAN_FUZZY_KEYS: list[str] = [k for k in _FUZZY_KEYS if len(k) >= _SCAN_FUZZY_MIN_LEN]


def _candidates(text: str) -> list[str]:
    """Progressively relaxed lookup keys for one normalised name."""
    out: list[str] = []
    for cand in (
        text,
        _strip_strength(text),
        _strip_trailing(_strip_strength(text), _SALT_WORDS),
        _strip_trailing(_strip_trailing(_strip_strength(text), _SALT_WORDS), _RELEASE_SUFFIXES),
    ):
        if cand and cand not in out:
            out.append(cand)
    return out


def normalize_drug(name: str) -> DrugInfo | None:
    """Resolve a brand or generic drug name to its canonical generic.

    Matching is case-, hyphen- and space-insensitive and ignores strengths
    ("40 mg"), salt names ("besylate") and release suffixes ("XL", "ER", "SR").
    If no exact key matches, a fuzzy match (``difflib`` ratio ≥ 0.88) is tried
    for names of at least 5 characters, which tolerates typos such as
    "amlodipin" or "telmisatran".

    Returns ``None`` for unknown names (including fabricated drugs).
    """
    text = _normalise_text(name or "")
    if not text:
        return None
    candidates = _candidates(text)
    for cand in candidates:
        for key in (cand, cand.replace(" ", "")):
            if key in _INDEX:
                return _INDEX[key]
    for cand in candidates:
        if len(cand) >= _FUZZY_MIN_LEN:
            match = difflib.get_close_matches(cand, _FUZZY_KEYS, n=1, cutoff=FUZZY_CUTOFF)
            if match:
                return _INDEX[match[0]]
    return None


@lru_cache(maxsize=4096)
def _fuzzy_scan(word: str) -> DrugInfo | None:
    """Fuzzy match for one free-text word (7+ characters, not a common word)."""
    if len(word) < _SCAN_FUZZY_MIN_LEN or word in _SCAN_STOPWORDS:
        return None
    close = difflib.get_close_matches(word, _SCAN_FUZZY_KEYS, n=1, cutoff=FUZZY_CUTOFF)
    return _INDEX[close[0]] if close else None


def _is_name_bigram(text: str, spans: list[re.Match[str]], i: int) -> bool:
    """Whether tokens ``i`` and ``i + 1`` can be read as one two-word drug name.

    They must be adjacent (only spaces, a hyphen or a slash between them), and a
    lower-case second token that is an ordinary English word, or a suffix
    followed by a time of day, is read as English rather than as a brand suffix.
    """
    first, second = spans[i], spans[i + 1]
    if not _BIGRAM_GAP.fullmatch(text[first.end():second.start()]):
        return False
    word = second.group(0)
    if word.islower() and word in _SUFFIX_ENGLISH_WORDS:
        return False
    if len(word) <= 3 and i + 2 < len(spans):
        following = spans[i + 2]
        between = text[second.end():following.start()]
        if not between.strip(" ,") and following.group(0).lower() in _TIME_WORDS:
            return False
    return True


@dataclass(frozen=True)
class DrugMention:
    """One recognised drug mention in free text, with its character span."""

    info: DrugInfo
    start: int
    end: int
    fuzzy: bool  # matched only by typo tolerance, not by an exact lexicon key


def find_drug_mentions(text: str) -> list[DrugMention]:
    """Every recognised drug mention in ``text`` with its span, in order (repeats kept).

    The scanner behind :func:`find_drugs`: tokens and bigrams are looked up in
    the lexicon index, and words of 7+ characters fall back to fuzzy matching
    (flagged ``fuzzy=True``).
    """
    source = text or ""
    spans = list(_TOKEN_RE.finditer(source))
    tokens = [m.group(0) for m in spans]
    mentions: list[DrugMention] = []
    i = 0
    while i < len(tokens):
        match: DrugInfo | None = None
        fuzzy = False
        width = 1
        if i + 1 < len(tokens) and _is_name_bigram(source, spans, i):
            match = _INDEX.get(_normalise_text(f"{tokens[i]} {tokens[i + 1]}"))
            width = 2 if match else 1
        if match is None:
            key = _strip_strength(_normalise_text(tokens[i]))
            match = _INDEX.get(key) or _INDEX.get(key.replace(" ", ""))
            if match is None:
                match = _fuzzy_scan(key)
                fuzzy = match is not None
        if match is not None:
            mentions.append(
                DrugMention(match, spans[i].start(), spans[i + width - 1].end(), fuzzy)
            )
        i += width
    return mentions


def find_drugs(text: str) -> list[DrugInfo]:
    """Find every recognised drug mentioned in free text, in order of appearance.

    Scans tokens and bigrams (so "Met XL", "Toprol XL", "isosorbide mononitrate"
    are recognised as one drug) for brand names, generic names and aliases.
    Fuzzy matching is applied only to words of 7+ characters to avoid
    false positives on ordinary English words. Duplicates are removed.
    """
    found: list[DrugInfo] = []
    seen: set[tuple[str, str | None]] = set()
    for mention in find_drug_mentions(text):
        ident = (mention.info.generic, mention.info.brand)
        if ident not in seen:
            seen.add(ident)
            found.append(mention.info)
    return found


def generic_components(generic: str) -> list[str]:
    """Ingredients of a canonical generic (``"telmisartan + amlodipine"`` → two names)."""
    return [part.strip() for part in (generic or "").split("+") if part.strip()]


# ---------------------------------------------------------------------------
# Lenient recognition (allow-lists only)
# ---------------------------------------------------------------------------

# Informal names and abbreviations patients use for lexicon medicines. Used only
# by :func:`find_drugs_lenient` (never to record a medicine or write a note):
# "ASA" is also an organisation and "HCT" also haematocrit.
INFORMAL_DRUG_NAMES: dict[str, str] = {
    "asa": "aspirin",
    "nitro": "nitroglycerin",
    "ntg": "nitroglycerin",
    "hct": _HCTZ,
    "amlo": "amlodipine",
    "rosuva": "rosuvastatin",
    "spiro": "spironolactone",
    "metop": "metoprolol",
    "digitalis": "digoxin",
}
# Common English words that the 5+ character typo tolerance of normalize_drug
# maps to a lexicon name ("among" → Amlong, "replace" → Repace).
_LENIENT_STOPWORDS: frozenset[str] = _SCAN_STOPWORDS | frozenset({
    "along", "among", "replace", "concord", "cadet", "divan", "aspiring", "theoretic",
    "oppressor", "capote", "carlo", "conor", "triace", "isatin", "fient", "trior", "tricorn",
    "propanone", "hydrazine", "transudate", "nitrosate",
})
_WORD_RE = re.compile(r"[A-Za-z]+")


def find_drugs_lenient(text: str) -> list[DrugInfo]:
    """Drugs plausibly named in ``text``, with the typo tolerance of :func:`normalize_drug`.

    :func:`find_drugs` plus every word of 5+ characters that :func:`normalize_drug`
    resolves ("Lipitr", "Eliqis", "asprin", "norvas") and a few informal names
    (:data:`INFORMAL_DRUG_NAMES`: "ASA", "nitro", "HCT"). Meant for allow-lists
    (which drugs the patient may have meant), never for recording a medicine or
    writing a trusted note. Duplicates are removed; order of first appearance.
    """
    found = list(find_drugs(text))
    for word in _WORD_RE.findall(text or ""):
        info = _lenient_word(word.lower())
        if info is not None and info not in found:
            found.append(info)
    return found


@lru_cache(maxsize=4096)
def _lenient_word(lower: str) -> DrugInfo | None:
    """One lower-case word resolved for :func:`find_drugs_lenient` (cached)."""
    if lower in INFORMAL_DRUG_NAMES:
        return normalize_drug(INFORMAL_DRUG_NAMES[lower])
    if len(lower) >= _FUZZY_MIN_LEN and lower not in _LENIENT_STOPWORDS:
        return normalize_drug(lower)
    return None


# ---------------------------------------------------------------------------
# Trusted drug-reference notes for the LLM prompts
# ---------------------------------------------------------------------------

MAX_REFERENCE_NOTES = 10
REFERENCE_TRUNCATED_NOTE = (
    "(list shortened: more recognised medicines are named than are listed here)"
)
MAX_BRANDS_PER_NOTE = 4
MAX_NOTE_CHARS = 240
_DEFAULT_CLASS = "cardiovascular medicine"
# Words that turn a brand into another product name ("Losar H", "Concor AM",
# "Coversyl Plus", "Eliquis XR"): the brand alone does not describe that product.
_PRODUCT_SUFFIXES: frozenset[str] = _RELEASE_SUFFIXES | frozenset({
    "h", "hct", "hctz", "am", "at", "plus", "forte", "max", "ultra", "gold", "neo", "xt", "cc",
    "ls", "lite", "mini", "duo", "trio", "combi", "co", "mite", "beta", "d", "m", "t", "n",
    "ez", "extra", "super", "strong", "pro", "advance", "advanced", "rapid", "fast", "dual",
    "triple", "new", "prime", "junior", "total", "complete", "natural", "herbal", "cardio",
    "nova", "one",
})
# Product suffixes that, written in lower case after a brand, are ordinary words
# ("Concor plus Telma", "Telma at night").
_SUFFIX_WORDS_READ_AS_ENGLISH: frozenset[str] = _SUFFIX_ENGLISH_WORDS | frozenset({
    "plus", "max", "extra", "super", "strong", "pro", "new", "fast", "rapid", "dual", "triple",
    "total", "complete", "natural", "herbal", "mini", "lite", "duo", "prime", "advance",
    "advanced", "co", "d", "m", "t", "n", "one", "gold", "la", "od", "er", "cardio",
})
# Capitalised words that are not part of a product name ("Eliquis I think", "Concor OK?").
_NOT_A_PRODUCT_WORD: frozenset[str] = frozenset({
    "i", "ok", "okay", "and", "or", "but", "the", "a", "an", "is", "are", "was", "were", "with",
    "for", "from", "to", "in", "on", "at", "of", "my", "me", "daily", "every", "each", "tablet",
    "tablets", "tab", "tabs", "pill", "pills", "mg", "mcg", "please", "thanks", "thank", "too",
    "also", "then", "now", "today", "yesterday", "tomorrow", "morning", "night", "evening",
    "pm", "bp", "dose", "doses", "safe", "yes", "no", "not",
})
_MODIFIER_GAP = re.compile(r"[ \t]*([-/]?)[ \t]*")
_NEXT_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def _mostly_capitalised(text: str) -> bool:
    """Whether the text is written in capitals or Title Case, so a capital carries no signal.

    Sentence-initial words, "I" and lexicon drug names are not counted.
    """
    counted: list[str] = []
    previous_end = 0
    for match in _WORD_RE.finditer(text):
        word = match.group(0)
        gap = text[previous_end:match.start()]
        sentence_start = (previous_end == 0 and not gap.strip()) or any(c in gap for c in ".!?:\n")
        previous_end = match.end()
        if sentence_start or word == "I" or word.lower() in _INDEX:
            continue
        counted.append(word)
    return bool(counted) and sum(w[0].isupper() for w in counted) * 2 > len(counted)


def _modified_brand(text: str, mention: DrugMention, title_case: bool) -> bool:
    """Whether a brand mention is directly followed by a word that makes it another product.

    "Losar H", "Concor AM", "Coversyl Plus", "Eliquis XR", "Lipitor Forte" or a
    capitalised unknown word ("Eliquis Supreme"): the lexicon knows only the
    brand, so a note for the brand would misdescribe that product (a missing
    ingredient) or vouch for a product that may not exist. Lower-case English
    readings ("Concor plus Telma", "Telma at night") and other drug names are
    not modifiers. Two-word lexicon names ("Met XL") never reach this check.
    """
    if mention.info.brand is None:
        return False
    gap = _MODIFIER_GAP.match(text, mention.end)
    if gap is None:
        return False
    token = _NEXT_TOKEN_RE.match(text, gap.end())
    if token is None:
        return False
    word = token.group(0)
    lower = word.lower()
    joined = bool(gap.group(1))  # "Telma-AM", "Losar/H"
    if lower in _INDEX:
        return False  # another medicine ("Concor Telma")
    if lower in _PRODUCT_SUFFIXES:
        if word.islower() and lower in _SUFFIX_WORDS_READ_AS_ENGLISH and not joined:
            return False
        rest = text[token.end():token.end() + 16].lstrip(" ,").lower()
        if len(word) <= 3 and rest.split(" ", 1)[0].strip(".,;:!?") in _TIME_WORDS:
            return False  # "Amlopres AT NIGHT"
        return True
    return (word[0].isupper() and not title_case and lower not in _NOT_A_PRODUCT_WORD
            and len(word) > 1)


@lru_cache(maxsize=512)
def brands_of(generic: str) -> tuple[str, ...]:
    """Lexicon brands whose generic is exactly ``generic`` (sorted, case-insensitive)."""
    return tuple(sorted(
        (brand for brand, mapped in BRAND_TO_GENERIC.items() if mapped == generic),
        key=lambda brand: (brand.lower(), brand),
    ))


def _describe_generic(generic: str) -> str:
    """``"quinapril (ACE inhibitor)"``; combinations name each ingredient with its class."""
    if generic in _COMBINATION_CLASS:
        return f"{generic} ({_COMBINATION_CLASS[generic]})"
    return " + ".join(
        f"{part} ({GENERIC_CLASS.get(part, _DEFAULT_CLASS)})" for part in generic_components(generic)
    )


def drug_reference_note(info: DrugInfo) -> str:
    """One canonical note for a recognised drug, built only from the curated lexicon.

    * brand:        ``"Accupril: brand name of quinapril (ACE inhibitor)"``
    * combination:  ``"Telma-H: brand name of a combination of telmisartan (angiotensin
      receptor blocker) + hydrochlorothiazide (thiazide diuretic)"``
    * generic:      ``"quinapril: generic medicine (ACE inhibitor); brands include Accupril"``

    Brand and generic notes carry the same generic name and class, so the two
    spellings of one medicine are described symmetrically. At most
    ``MAX_BRANDS_PER_NOTE`` brands are listed (alphabetical order) and the note
    never exceeds ``MAX_NOTE_CHARS`` characters.
    """
    generic = info.generic
    combination = len(generic_components(generic)) > 1
    if info.brand:
        kind = "a combination of " if combination else ""
        return f"{info.brand}: brand name of {kind}{_describe_generic(generic)}"[:MAX_NOTE_CHARS]
    kind = "generic combination medicine" if combination else "generic medicine"
    note = f"{generic}: {kind} ({info.drug_class})"
    brands = list(brands_of(generic)[:MAX_BRANDS_PER_NOTE])
    while brands and len(f"{note}; brands include {', '.join(brands)}") > MAX_NOTE_CHARS:
        brands.pop()
    if brands:
        note += f"; brands include {', '.join(brands)}"
    return note[:MAX_NOTE_CHARS]


def _reference_infos(text: str) -> list[DrugInfo]:
    """Exact, unmodified drug mentions in ``text`` (see :func:`drug_reference_notes`)."""
    title_case = _mostly_capitalised(text)
    return [
        mention.info for mention in find_drug_mentions(text)
        if not mention.fuzzy and not _modified_brand(text, mention, title_case)
    ]


def drug_reference_notes(*texts: str | None) -> list[str]:
    """Canonical lexicon notes for every drug named in ``texts`` (see :func:`drug_reference_note`).

    Only exact lexicon matches (brand, generic or alias) produce a note: unknown
    or fabricated names, and typo-level fuzzy matches, produce none, so a note
    always vouches for a real medicine. A brand directly followed by a word that
    makes it another product ("Losar H", "Coversyl Plus", "Eliquis Max") gets no
    note either: the lexicon knows only the brand, not that product. The notes
    are built from canonical lexicon spellings only (never from the text
    itself), in order of first mention and without duplicates. Capping is left
    to the prompt builder (:data:`MAX_REFERENCE_NOTES`).
    """
    notes: list[str] = []
    for text in texts:
        for info in _reference_infos(text or ""):
            note = drug_reference_note(info)
            if note not in notes:
                notes.append(note)
    return notes


def all_brand_generic_pairs() -> list[DrugInfo]:
    """Every brand → generic mapping (sorted by brand), e.g. for the RABBITS benchmark."""
    return [
        DrugInfo(generic, brand, _class_for(generic))
        for brand, generic in sorted(BRAND_TO_GENERIC.items(), key=lambda kv: kv[0].lower())
    ]
