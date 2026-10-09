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


def find_drugs(text: str) -> list[DrugInfo]:
    """Find every recognised drug mentioned in free text, in order of appearance.

    Scans tokens and bigrams (so "Met XL", "Toprol XL", "isosorbide mononitrate"
    are recognised as one drug) for brand names, generic names and aliases.
    Fuzzy matching is applied only to words of 7+ characters to avoid
    false positives on ordinary English words. Duplicates are removed.
    """
    tokens = _TOKEN_RE.findall(text or "")
    found: list[DrugInfo] = []
    seen: set[tuple[str, str | None]] = set()
    i = 0
    while i < len(tokens):
        match: DrugInfo | None = None
        width = 1
        if i + 1 < len(tokens):
            match = _INDEX.get(_normalise_text(f"{tokens[i]} {tokens[i + 1]}"))
            width = 2 if match else 1
        if match is None:
            key = _strip_strength(_normalise_text(tokens[i]))
            match = _INDEX.get(key) or _INDEX.get(key.replace(" ", ""))
            if match is None:
                match = _fuzzy_scan(key)
        if match is not None:
            ident = (match.generic, match.brand)
            if ident not in seen:
                seen.add(ident)
                found.append(match)
        i += width
    return found


def all_brand_generic_pairs() -> list[DrugInfo]:
    """Every brand → generic mapping (sorted by brand), e.g. for the RABBITS benchmark."""
    return [
        DrugInfo(generic, brand, _class_for(generic))
        for brand, generic in sorted(BRAND_TO_GENERIC.items(), key=lambda kv: kv[0].lower())
    ]
