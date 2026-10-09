"""Unit tests for the cardiovascular brand → generic drug normaliser (medbridge.nlp.drugs)."""
from __future__ import annotations

import pytest

from medbridge.nlp.drugs import (
    BRAND_TO_GENERIC,
    GENERIC_CLASS,
    INDIAN_BRANDS,
    DrugInfo,
    all_brand_generic_pairs,
    find_drugs,
    normalize_drug,
)

INDIAN_EXAMPLES = {
    "Telma": "telmisartan",
    "Telma-H": "telmisartan + hydrochlorothiazide",
    "Telma-AM": "telmisartan + amlodipine",
    "Amlong": "amlodipine",
    "Stamlo": "amlodipine",
    "Amlokind": "amlodipine",
    "Concor": "bisoprolol",
    "Ecosprin": "aspirin",
    "Ecosprin-AV": "aspirin + atorvastatin",
    "Rosuvas": "rosuvastatin",
    "Rozavel": "rosuvastatin",
    "Storvas": "atorvastatin",
    "Atorva": "atorvastatin",
    "Losar": "losartan",
    "Repace": "losartan",
    "Cilacar": "cilnidipine",
    "Metolar": "metoprolol",
    "Met XL": "metoprolol succinate",
    "Nebicard": "nebivolol",
    "Cardace": "ramipril",
    "Envas": "enalapril",
    "Dytor": "torsemide",
    "Clopilet": "clopidogrel",
    "Deplatt": "clopidogrel",
    "Acitrom": "acenocoumarol",
    "Warf": "warfarin",
    "Ciplar": "propranolol",
    "Arkamin": "clonidine",
    "Vymada": "sacubitril + valsartan",
    "Ivabrad": "ivabradine",
    "Sorbitrate": "isosorbide dinitrate",
}

INTERNATIONAL_EXAMPLES = {
    "Norvasc": "amlodipine", "Zestril": "lisinopril", "Prinivil": "lisinopril",
    "Cozaar": "losartan", "Hyzaar": "losartan + hydrochlorothiazide", "Diovan": "valsartan",
    "Benicar": "olmesartan", "Micardis": "telmisartan", "Avapro": "irbesartan",
    "Atacand": "candesartan", "Lasix": "furosemide", "Microzide": "hydrochlorothiazide",
    "Aldactone": "spironolactone", "Inspra": "eplerenone", "Lopressor": "metoprolol tartrate",
    "Toprol-XL": "metoprolol succinate", "Tenormin": "atenolol", "Coreg": "carvedilol",
    "Bystolic": "nebivolol", "Cardizem": "diltiazem", "Calan": "verapamil",
    "Vasotec": "enalapril", "Lotensin": "benazepril", "Altace": "ramipril",
    "Accupril": "quinapril", "Exforge": "amlodipine + valsartan",
    "Lotrel": "amlodipine + benazepril", "Lipitor": "atorvastatin", "Crestor": "rosuvastatin",
    "Zocor": "simvastatin", "Pravachol": "pravastatin", "Zetia": "ezetimibe",
    "Repatha": "evolocumab", "Plavix": "clopidogrel", "Brilinta": "ticagrelor",
    "Effient": "prasugrel", "Eliquis": "apixaban", "Xarelto": "rivaroxaban",
    "Coumadin": "warfarin", "Pradaxa": "dabigatran", "Savaysa": "edoxaban",
    "Entresto": "sacubitril + valsartan", "Farxiga": "dapagliflozin", "Jardiance": "empagliflozin",
    "Lanoxin": "digoxin", "Catapres": "clonidine", "Apresoline": "hydralazine",
    "Imdur": "isosorbide mononitrate", "Nitrostat": "nitroglycerin", "Ranexa": "ranolazine",
    "Corlanor": "ivabradine", "Kerendia": "finerenone", "Verquvo": "vericiguat",
    "Cordarone": "amiodarone", "Multaq": "dronedarone",
}


class TestDrugMap:
    def test_size_requirements(self) -> None:
        assert len(BRAND_TO_GENERIC) >= 70
        assert len(INDIAN_BRANDS) >= 30

    @pytest.mark.parametrize(("brand", "generic"), sorted({**INDIAN_EXAMPLES, **INTERNATIONAL_EXAMPLES}.items()))
    def test_required_mappings(self, brand: str, generic: str) -> None:
        assert BRAND_TO_GENERIC[brand] == generic

    def test_indian_examples_flagged(self) -> None:
        assert set(INDIAN_EXAMPLES) <= INDIAN_BRANDS
        assert "Norvasc" not in INDIAN_BRANDS

    def test_every_ingredient_has_a_class(self) -> None:
        ingredients = {p.strip() for g in BRAND_TO_GENERIC.values() for p in g.split("+")}
        missing = {i for i in ingredients if i not in GENERIC_CLASS and i != "sacubitril"}
        assert missing == set()

    def test_pairs_helper(self) -> None:
        pairs = all_brand_generic_pairs()
        assert len(pairs) == len(BRAND_TO_GENERIC)
        assert all(isinstance(p, DrugInfo) and p.brand and p.generic for p in pairs)
        assert [p.brand.lower() for p in pairs] == sorted(p.brand.lower() for p in pairs)


class TestNormalizeDrug:
    @pytest.mark.parametrize(("name", "generic", "brand"), [
        ("Telma 40", "telmisartan", "Telma"),
        ("telma-h", "telmisartan + hydrochlorothiazide", "Telma-H"),
        ("TELMA H 40", "telmisartan + hydrochlorothiazide", "Telma-H"),
        ("TelmaAM", "telmisartan + amlodipine", "Telma-AM"),
        ("Met XL 25", "metoprolol succinate", "Met XL"),
        ("Toprol XL", "metoprolol succinate", "Toprol-XL"),
        ("toprol", "metoprolol succinate", "Toprol-XL"),
        ("Norvasc 5mg", "amlodipine", "Norvasc"),
        ("Ecosprin-AV 75", "aspirin + atorvastatin", "Ecosprin-AV"),
        ("Concor 2.5 mg", "bisoprolol", "Concor"),
        ("Angispan TR", "nitroglycerin", "Angispan-TR"),
    ])
    def test_brand_variants(self, name: str, generic: str, brand: str) -> None:
        info = normalize_drug(name)
        assert info is not None and (info.generic, info.brand) == (generic, brand)

    @pytest.mark.parametrize(("name", "generic"), [
        ("lisinopril", "lisinopril"),
        ("Amlodipine besylate", "amlodipine"),
        ("olmesartan medoxomil 20 mg", "olmesartan"),
        ("HCTZ", "hydrochlorothiazide"),
        ("frusemide", "furosemide"),
        ("glyceryl trinitrate", "nitroglycerin"),
        ("metoprolol succinate ER", "metoprolol succinate"),
    ])
    def test_generics_and_aliases(self, name: str, generic: str) -> None:
        info = normalize_drug(name)
        assert info is not None and info.generic == generic and info.brand is None

    @pytest.mark.parametrize(("typo", "generic"), [
        ("amlodipin", "amlodipine"), ("telmisatran", "telmisartan"), ("lisinoprill", "lisinopril"),
        ("Lipitorr", "atorvastatin"), ("rosuvastatn", "rosuvastatin"),
    ])
    def test_fuzzy_matching(self, typo: str, generic: str) -> None:
        info = normalize_drug(typo)
        assert info is not None and info.generic == generic

    @pytest.mark.parametrize("name", ["Cardiozepam", "met", "", "   ", "aspirine-x-ultra", "tea",
                                      "Norvacs"])  # 7-letter transposition: ratio 0.857 < 0.88
    def test_unknown_names(self, name: str) -> None:
        assert normalize_drug(name) is None

    def test_classes(self) -> None:
        assert normalize_drug("Telma").drug_class == "angiotensin receptor blocker"
        assert normalize_drug("Telma-H").drug_class == (
            "angiotensin receptor blocker + thiazide diuretic"
        )
        assert "ARNI" in normalize_drug("Entresto").drug_class


class TestFindDrugs:
    def test_brands_generics_bigrams_in_order(self) -> None:
        found = find_drugs(
            "I take Telma 40 and Met XL 25, plus ecosprin daily; my doctor added amlodipin. "
            "Also isosorbide mononitrate."
        )
        assert [(d.generic, d.brand) for d in found] == [
            ("telmisartan", "Telma"),
            ("metoprolol succinate", "Met XL"),
            ("aspirin", "Ecosprin"),
            ("amlodipine", None),
            ("isosorbide mononitrate", None),
        ]

    def test_no_false_positives_on_common_words(self) -> None:
        text = "I met my doctor, it was efficient; the concern is my pressure in the morning."
        assert find_drugs(text) == []

    def test_duplicates_removed(self) -> None:
        assert len(find_drugs("Telma, telma and TELMA")) == 1

    def test_fabricated_drug_not_found(self) -> None:
        assert find_drugs("Is Cardiozepam better than the BP vaccine?") == []


class TestCombinationSuffixesAndArni:
    """Regression (llm-prompts-07): '<brand> at/am/h' false combos; sacubitril aliases."""

    @pytest.mark.parametrize(("text", "expected"), [
        ("I take Telma 40 am and Concor 5 pm", [("telmisartan", "Telma"), ("bisoprolol", "Concor")]),
        ("Amlopres 5 at night gives me swollen ankles", [("amlodipine", "Amlopres")]),
        ("Amlopres at night", [("amlodipine", "Amlopres")]),
        ("AMLOPRES AT NIGHT", [("amlodipine", "Amlopres")]),
        ("telma am", [("telmisartan", "Telma")]),
    ])
    def test_english_words_after_a_brand_are_not_suffixes(
        self, text: str, expected: list[tuple[str, str]]
    ) -> None:
        assert [(d.generic, d.brand) for d in find_drugs(text)] == expected

    @pytest.mark.parametrize(("text", "generic"), [
        ("Telma-AM 40", "telmisartan + amlodipine"),
        ("Telma AM 40 daily", "telmisartan + amlodipine"),
        ("Amlopres-AT", "amlodipine + atenolol"),
        ("Amlopres AT 50", "amlodipine + atenolol"),
        ("telma h 40", "telmisartan + hydrochlorothiazide"),
        ("Ecosprin AV 75", "aspirin + atorvastatin"),
        ("met xl 50", "metoprolol succinate"),
        ("Diovan HCT", "valsartan + hydrochlorothiazide"),
    ])
    def test_real_combination_and_release_names_still_match(self, text: str, generic: str) -> None:
        assert [d.generic for d in find_drugs(text)] == [generic]

    @pytest.mark.parametrize("name", [
        "sacubitril", "valsartan/sacubitril", "valsartan sacubitril", "sacubitril/valsartan",
        "Sacubitril-Valsartan 49/51", "Entresto",
    ])
    def test_sacubitril_spellings_resolve_to_the_arni(self, name: str) -> None:
        info = normalize_drug(name)
        assert info is not None and info.generic == "sacubitril + valsartan"
        assert "ARNI" in info.drug_class
        assert [d.generic for d in find_drugs(f"I take {name} twice a day")] == [
            "sacubitril + valsartan"
        ]

    def test_plain_valsartan_is_still_an_arb(self) -> None:
        assert [d.generic for d in find_drugs("valsartan 80 daily")] == ["valsartan"]

    def test_strength_between_tokens_breaks_a_two_word_generic(self) -> None:
        assert [d.generic for d in find_drugs("metoprolol 50 succinate")] == ["metoprolol"]
