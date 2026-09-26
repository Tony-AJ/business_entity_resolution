import numpy as np
import pandas as pd

from entity_resolution import config as C
from entity_resolution.normalize import (
    DEFAULT,
    EXTRA_COLUMNS,
    NORM_COLUMNS,
    NormaliseConfig,
    apply_token_map,
    basic_norm,
    fit_token_map,
    normalise_names,
    normalise_records,
)

FLAGS = ["non_latin", "domain_form", "addr_non_latin"]
STRINGS = [c for c in NORM_COLUMNS + EXTRA_COLUMNS if c not in [*FLAGS, "addr_tokens"]]
NAME_COLS = ["name_norm", "name_core", "legal_form", "name_first", "name_sorted", "name_squash"]
DEVANAGARI = "गल्फ प्राइवेट लिमिटेड"  # "Gulf Private Limited"

# One row per trap: legal forms, a US region, empty fields, France, a Devanagari name and
# address, a domain form and an honorific.
MIXED = [
    ("S2-01", "Sharma Traders Pvt. Ltd.", "MG Road, Pune 411001", "India"),
    ("S2-02", "ACME Corporation", "1795 Westchester Dr, High Point, NC", "US"),
    ("S2-03", "", "", "US"),
    ("S2-04", "Boulangerie Dupont SARL", "5 Rue de la Paix, 75002 Paris", "France"),
    ("S3-05", DEVANAGARI, "लिंकिंग रोड, मुंबई, महाराष्ट्र", "India"),
    ("S3-06", "pelletierrexford.com", "12 Main St", "US"),
    ("S3-07", "Mr Vistaar Traders", "Delhi", "India"),
]


def _src(*rows: tuple) -> pd.DataFrame:
    """Source frame (entity_id, business_name, business_address, country), all "str"."""
    return pd.DataFrame(list(rows), columns=list(C.SOURCE_COLUMNS)).astype("str")


def _names(*names: str, cfg: NormaliseConfig = DEFAULT) -> pd.DataFrame:
    """Normalised rows holding these names (empty address, country US)."""
    return normalise_records(_src(*[(f"S2-{i}", n, "", "US") for i, n in enumerate(names)]), cfg)


def _addrs(*addrs: str, cfg: NormaliseConfig = DEFAULT) -> pd.DataFrame:
    """Normalised rows holding these addresses (fixed name, country India)."""
    return normalise_records(_src(*[(f"S2-{i}", "Acme", a, "India") for i, a in enumerate(addrs)]),
                             cfg)


def test_basic_norm_accents_ampersand_punct():
    """Accents and case fold, & becomes "and", apostrophes vanish, punctuation splits."""
    assert _names("Café & Co.")["name_norm"].tolist() == ["cafe and co"]
    raw = pd.Series(["Café & Co.", "Léarning Àmicale", "O'Reilly's  Books-Inc"], dtype="str")
    assert basic_norm(raw).tolist() == ["cafe and co", "learning amicale", "oreillys books inc"]


def test_legal_forms_removed_and_canonical():
    """Legal forms leave name_core and land, canonical and in order, in legal_form."""
    out = _names("Sharma Traders Pvt. Ltd.", "Sharma Traders Private Limited",
                 "ACME Corporation", "Acme Corp", "Acme L.L.C.")
    assert out["name_core"].tolist() == ["sharma traders"] * 2 + ["acme"] * 3
    assert out["legal_form"].tolist() == ["pvt ltd", "pvt ltd", "corp", "corp", "llc"]
    assert out["name_norm"].iloc[4] == "acme llc"  # dotted initials are joined first


def test_legal_form_position_independent():
    """A leading legal form is removed just like a trailing one."""
    out = _names("LLC Moncada Learning Center", "Moncada Learning Center LLC")
    assert out["name_core"].tolist() == ["moncada learning center"] * 2
    assert out["legal_form"].tolist() == ["llc", "llc"]


def test_transliterated_legal_words_only_on_non_latin_rows():
    """Indic legal words are stripped from script rows; a Latin "Li" or "Pra" stays."""
    out = _names(DEVANAGARI, "Bruce Li Pra Traders")
    assert out["legal_form"].tolist() == ["pvt ltd", ""]
    assert out["name_core"].iloc[1] == "bruce li pra traders"
    assert out["name_core"].iloc[0].isascii() and out["name_core"].iloc[0] != ""


def test_dangling_and_dropped():
    """An "and" left at either end by legal-form removal goes; an inner one stays."""
    out = _names("Dover & Co", "Smith and Company", "Patel & Sons")
    assert out["name_norm"].iloc[0] == "dover and co"
    assert out["name_core"].tolist() == ["dover", "smith", "patel and sons"]


def test_honorifics_stripped_from_core():
    """Leading Mr / Smt / M/s / The leave name_core but stay in name_norm."""
    out = _names("Mr Vistaar Traders", "Smt Lakshmi Stores", "M/s Patel Brothers",
                 "The Dover Company")
    assert out["name_core"].tolist() == ["vistaar traders", "lakshmi stores", "patel brothers",
                                         "dover"]
    assert out["name_norm"].iloc[0] == "mr vistaar traders"
    assert out["name_first"].tolist() == ["vistaar", "lakshmi", "patel", "dover"]
    assert out["legal_form"].iloc[3] == "co"


def test_squash_domain_and_leet():
    """Domain, handle and leet forms share name_squash with the plain name."""
    out = _names("pelletierrexford.com", "Pelletier Rexford Corp", "#high1andministries",
                 "5tar flxe llc", "gl0ba traders")
    assert out["name_squash"].tolist() == ["pelletierrexford", "pelletierrexford",
                                           "highlandministries", "starflxe", "globatraders"]
    assert out["domain_form"].tolist() == [True, False, True, False, False]
    # leet digits are folded in name_core too, but name_norm keeps the written form
    assert out["name_core"].iloc[4] == "globa traders"
    assert out["name_norm"].iloc[4] == "gl0ba traders"


def test_squash_keeps_pure_numbers():
    """Digits of an all-digit token are never folded to letters."""
    out = _names("39/56 Asia")
    assert out["name_core"].iloc[0] == "39 56 asia"
    assert out["name_squash"].iloc[0] == "3956asia"


def test_sorted_key():
    """name_sorted holds the sorted distinct tokens: swaps and doubled words agree."""
    out = _names("Bustamante Sanchez", "Sanchez Bustamante", "Colonial Colonial Inn")
    assert out["name_sorted"].tolist() == ["bustamante sanchez", "bustamante sanchez",
                                           "colonial inn"]
    assert out["name_first"].tolist() == ["bustamante", "sanchez", "colonial"]


def test_address_abbrev_and_numbers():
    """Street types shrink to one short form; numbers, region and last words are split out."""
    out = _addrs("1795 Westchester Dr, High Point, NC",
                 "1795 Westchester Drive, High Point, North Carolina")
    for col, value in [("addr_norm", "1795 westchester dr high pt nc"), ("addr_nums", "1795"),
                       ("region", "nc"), ("addr_last", "pt nc")]:
        assert out[col].tolist() == [value, value]
    assert out["addr_tokens"].tolist() == [6, 6]
    more = _addrs("Government Street, Mobile", "Government Saint, Mobile", "5 R. de la Paix",
                  "8th Avenue", "Plot 0033, Sector 5, Noida", "25233B Main Rd",
                  "12 Main St, NULL, Springfield")
    assert more["addr_norm"].tolist() == [
        "government st mobile", "government st mobile", "5 rue de la paix", "8 ave",
        "plot 33 sec 5 noida", "25233 b main rd", "12 main st springfield"]
    assert more["addr_nums"].iloc[4] == "33 5"


def test_region_codes_across_forms_and_scripts():
    """Full name, postal code and native script of a region give one code, also in addr_norm."""
    out = _addrs("Linking Road, Mumbai, Maharashtra", "Linking Road, Mumbai, MH",
                 "Linking Road, Mumbai, महाराष्ट्र",
                 "12 Cours de l'Intendance, Bordeaux, Nouvelle-Aquitaine")
    assert out["region"].tolist() == ["mh", "mh", "mh", "naq"]
    assert out["addr_norm"].tolist()[:3] == ["linking rd mumbai mh"] * 3
    assert out["addr_norm"].iloc[3].endswith("bordeaux naq")
    assert out["addr_non_latin"].tolist() == [False, False, True, False]


def test_postcode_extraction():
    """A 6-digit PIN anywhere, else a 5-digit code only when it ends the address."""
    out = _addrs("MG Road, Pune 411001", "411001 Pune, MG Road", "100 Elm St, Tyler, TX 75701",
                 "75701 Elm St, Tyler, TX", "Near SBI ATM, MG Road, Pune")
    assert out["postcode"].tolist() == ["411001", "411001", "75701", "", ""]


def test_transliterate_only_non_latin():
    """Latin rows match plain basic normalisation; a Devanagari row becomes ASCII."""
    raw = ["Café & Co.", "Léarning Àmicale", "O'Reilly Books", DEVANAGARI]
    out = _names(*raw)
    assert out["non_latin"].tolist() == [False, False, False, True]
    latin = out["name_norm"].iloc[:3].tolist()
    assert latin == basic_norm(pd.Series(raw[:3], dtype="str")).tolist()
    # without transliteration Latin rows are byte-identical and the script row loses its text
    off = _names(*raw, cfg=NormaliseConfig(transliterate=False))
    assert off["name_norm"].iloc[:3].tolist() == latin
    assert off["name_norm"].iloc[3] == ""
    script = out["name_norm"].iloc[3]
    assert script.isascii() and script == script.lower() and len(script.split()) == 3


def test_region_map_override():
    """A region map passed in the config replaces the static one."""
    learned = NormaliseConfig(region_map={"ka": "karnataka"})
    out = _addrs("Bangalore, KA", "Chennai, Tamil Nadu", cfg=learned)
    assert out["region"].tolist() == ["karnataka", ""]
    assert out["addr_norm"].tolist() == ["bangalore karnataka", "chennai tamil nadu"]
    assert _addrs("Bangalore, KA", "Chennai, Tamil Nadu")["region"].tolist() == ["ka", "tn"]


def test_empty_fields():
    """Empty or missing name and address give "" everywhere, 0 tokens, no NaN."""
    out = normalise_records(_src(("S2-1", "", "", "US"), ("S2-2", None, None, "US"),
                                 ("S2-3", "Acme", "", "US")))
    for col in [c for c in STRINGS if c not in (C.ENTITY_ID, C.COUNTRY)]:
        assert out[col].iloc[:2].tolist() == ["", ""], col
    assert out["addr_tokens"].tolist() == [0, 0, 0]
    assert not out[FLAGS].to_numpy().any()
    assert not out.isna().to_numpy().any()
    assert out["name_addr"].iloc[2] == "acme"


def test_schema():
    """Columns, dtypes, row count and order of the input, index reset."""
    df = _src(*MIXED)
    df.index = [70, 10, 50, 20, 60, 30, 40]
    out = normalise_records(df)
    assert list(out.columns) == NORM_COLUMNS + EXTRA_COLUMNS
    assert all(out[c].dtype == "str" for c in STRINGS)
    assert all(out[c].dtype == bool for c in FLAGS)
    assert out["addr_tokens"].dtype == np.int16
    assert out[C.ENTITY_ID].tolist() == df[C.ENTITY_ID].tolist()
    assert out.index.equals(pd.RangeIndex(len(df)))
    assert out[C.COUNTRY].tolist() == df[C.COUNTRY].tolist()


def test_chunking_matches_one_chunk():
    """Small chunks (non-Latin rows in only some of them) give the one-chunk frame."""
    df = _src(*MIXED)
    whole = normalise_records(df)
    for rows in (1, 2, 3):
        pd.testing.assert_frame_equal(normalise_records(df, NormaliseConfig(chunk_rows=rows)),
                                      whole)


def test_empty_frame_keeps_schema():
    """Zero rows in, zero rows out with every column and dtype."""
    out = normalise_records(_src())
    assert len(out) == 0
    assert list(out.columns) == NORM_COLUMNS + EXTRA_COLUMNS
    assert out.dtypes.equals(normalise_records(_src(*MIXED)).dtypes)


def test_normalise_is_idempotent():
    """basic_norm of name_norm and of addr_norm changes nothing."""
    out = normalise_records(_src(*MIXED))
    for col in ("name_norm", "addr_norm"):
        assert basic_norm(out[col]).tolist() == out[col].tolist()


def test_deterministic():
    """Two runs give the same frame."""
    df = _src(*MIXED)
    pd.testing.assert_frame_equal(normalise_records(df), normalise_records(df))


def test_no_country_branching():
    """Country selects no rule: France gets "sarl", and a relabelled row normalises alike."""
    row = ("S1-10", "Boulangerie Dupont SARL", "5 Rue de la Paix, Paris", "France")
    fr = normalise_records(_src(row))
    assert fr["legal_form"].iloc[0] == "sarl"
    assert fr["name_core"].iloc[0] == "boulangerie dupont"
    assert fr["addr_norm"].iloc[0] == "5 rue de la paix paris"
    for country in ("US", "India", "Atlantis"):
        other = normalise_records(_src((*row[:3], country)))
        assert other[C.COUNTRY].iloc[0] == country
        pd.testing.assert_frame_equal(other.drop(columns=C.COUNTRY), fr.drop(columns=C.COUNTRY))


# ---------------------------------------------------------- learned token map ----
SHAKTI = "शक्ति ट्रेडिंग"  # anyascii "skti tredimg"


def _token_map_case() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(truth pairs, s1n, pooln, raw pool) for fit_token_map.

    "skti" meets "shakti" 3 times and "sakthi" once, "tredimg" meets "trading" 4 times;
    "syam" / "stors" meet their Latin form once; the Latin pool rows "Sunrise Traders"
    meet "sunrise trading" 3 times but are not written in a non-Latin script; S3-9 spells
    the transliteration in Latin letters.
    """
    s1 = _src(*[(f"S1-{i}", n, "Pune", "India") for i, n in enumerate(
        ["Shakti Trading"] * 3 + ["Sakthi Trading", "Shyam Stores"] + ["Sunrise Trading"] * 3)])
    pool = _src(*[(f"S{2 if i < 5 else 3}-{i}", n, "Pune", "India") for i, n in enumerate(
        [SHAKTI] * 4 + ["श्याम स्टोर्स"] + ["Sunrise Traders"] * 3 + ["Skti Tredimg"])])
    truth = pd.DataFrame({C.S1_ID: s1[C.ENTITY_ID], C.ENTITY_ID: pool[C.ENTITY_ID].iloc[:8]})
    return truth.astype("str"), normalise_records(s1), normalise_records(pool), pool


def test_fit_token_map_learns_alignment():
    """Alignments seen >= min_count times with >= min_share are learned, coincidences not."""
    truth, s1n, pooln, _ = _token_map_case()
    assert fit_token_map(truth, s1n, pooln) == {"skti": "shakti", "tredimg": "trading"}
    assert fit_token_map(truth, s1n, pooln, min_count=5) == {}
    assert fit_token_map(truth, s1n, pooln, min_share=0.8) == {"tredimg": "trading"}
    assert fit_token_map(truth.iloc[:0], s1n, pooln) == {}


def test_apply_token_map_changes_only_non_latin_rows():
    """Script rows get the mapped name columns; Latin rows and address columns stay."""
    truth, s1n, pooln, pool = _token_map_case()
    token_map = fit_token_map(truth, s1n, pooln)
    before = pooln.copy()
    out = apply_token_map(pooln, token_map)
    pd.testing.assert_frame_equal(pooln, before)  # input untouched
    assert list(out.columns) == list(pooln.columns) and out.dtypes.equals(pooln.dtypes)
    script = pooln["non_latin"].to_numpy()
    assert script.tolist() == [True] * 5 + [False] * 4
    pd.testing.assert_frame_equal(out[~script], pooln[~script])  # "Skti Tredimg" included
    fixed = out.iloc[:4]
    assert fixed["name_norm"].eq("shakti trading").all()
    assert fixed["name_squash"].eq("shaktitrading").all()
    assert fixed["name_first"].eq("shakti").all()
    assert fixed["name_addr"].eq("shakti trading pune").all()
    assert out["name_norm"].iloc[4] == pooln["name_norm"].iloc[4]  # no mapped token
    unchanged = [c for c in out.columns if c not in [*NAME_COLS, "name_addr"]]
    pd.testing.assert_frame_equal(out[unchanged], pooln[unchanged])
    # the cached route equals normalising the raw names with the map from scratch
    direct = normalise_names(pool[C.NAME], token_map=token_map)
    pd.testing.assert_frame_equal(out[NAME_COLS], direct[NAME_COLS])
    pd.testing.assert_frame_equal(apply_token_map(pooln, {}), pooln)


def test_french_departements_map_to_their_region() -> None:
    """A département in the state slot reads as its region's code, like the region itself."""
    import pandas as pd

    from entity_resolution.normalize import normalise_addresses
    out = normalise_addresses(pd.Series(["5 Rue Lafayette, Lille, Nord",
                                         "5 Rue Lafayette, Lille, Hauts-de-France",
                                         "3 Rue Racine, Bordeaux, Gironde",
                                         "1 Quai, Nantes, Loire-Atlantique",
                                         "2 Rue, Calais, Pas-de-Calais"]))
    assert out["region"].tolist() == ["hdf", "hdf", "naq", "pdl", "hdf"]


# ------------------------------------------------------------------- rules v4 ----
def test_v4_number_marker_leaves_addresses():
    """The pool's "N° 32" / "Nº 32" / "n°32" read like S1's "32": no stray "n" token."""
    out = _addrs("N° 32 R DES LAURIERS, PORNIC", "Nº 32 R. des Lauriers, Pornic",
                 "n°32 rue des lauriers, pornic", "32 Rue des Lauriers, Pornic")
    assert out["addr_norm"].tolist() == ["32 rue des lauriers pornic"] * 4
    assert out["addr_nums"].tolist() == ["32"] * 4
    # only the marker goes: "No 32" and a lone degree sign are unchanged, names untouched
    other = _addrs("No 32 Rue X", "Temp 5° Rue X")
    assert other["addr_norm"].tolist() == ["no 32 rue x", "temp 5 rue x"]
    assert _names("N° 1 Pizza")["name_norm"].iloc[0] == "n 1 pizza"


def test_v4_compagnie_is_a_legal_form_like_cie():
    """ "Compagnie" and "Cie", swapped by the French pool, both leave name_core as "co"."""
    out = _names("Bordeaux France Compagnie", "Bordeaux France Cie", "Meta & Compagnie SA")
    assert out["name_core"].tolist() == ["bordeaux france", "bordeaux france", "meta"]
    assert out["legal_form"].tolist() == ["co", "co", "co sa"]
    assert out["name_norm"].iloc[0] == "bordeaux france compagnie"
