"""Phonetic feature group (plan group C, 07 phonetic §): Soundex + Double Metaphone.

Mirrors the fixture style of ``test_features.py``: normalised frames are built inline from
already-normalised ``name_core`` / ``addr_norm`` strings, so these tests do not depend on
normalize.py. Independent reference values come straight from ``jellyfish``/``metaphone``
(the same libraries ``features._phonetic`` calls), not from re-deriving the algorithm.
"""
from __future__ import annotations

import jellyfish
import numpy as np
import pandas as pd
import pytest
from metaphone import doublemetaphone

from entity_resolution import config as C
from entity_resolution.blocking import PAIR_COLUMNS
from entity_resolution.features import FEATURE_COLUMNS, NAN_FEATURES, build_features

NAN = np.nan
PHONETIC = FEATURE_COLUMNS["phonetic"]


def _records(*rows: dict) -> pd.DataFrame:
    """Minimal normalised frame: only the columns the phonetic group reads."""
    df = pd.DataFrame(list(rows))
    return df.astype({C.ENTITY_ID: "str", "non_latin": bool, "name_core": "str",
                      "addr_norm": "str"})


def _rec(entity_id: str, name_core: str = "", addr_norm: str = "",
        non_latin: bool = False) -> dict:
    return {C.ENTITY_ID: entity_id, "non_latin": non_latin, "name_core": name_core,
            "addr_norm": addr_norm}


def _pairs(*rows: tuple) -> pd.DataFrame:
    """Candidate pairs from (s1, pool, pass, sim_name_char, sim_name_addr_word, sim_addr_char).

    The phonetic group reads none of these, so the sims are always NaN placeholders.
    """
    df = pd.DataFrame(list(rows), columns=PAIR_COLUMNS)
    return df.astype({C.S1_ID: "str", C.ENTITY_ID: "str", "pass": np.uint8,
                      **{c: np.float32 for c in PAIR_COLUMNS[3:]}})


def _phonetic_codes(token: str) -> tuple[str, str]:
    """(Soundex, Double Metaphone primary-or-secondary) of one token, "" if not applicable."""
    if not any(c.isalpha() for c in token):
        return "", ""
    primary, secondary = doublemetaphone(token)
    return jellyfish.soundex(token), primary or secondary


def _expected(name_l: str, name_r: str) -> tuple[float, float, float]:
    """Reference (exact_match, jaccard, token_overlap_ratio) for two space-separated strings.

    Same definition as ``features._phonetic_column``: each token contributes a Soundex code
    and (independently) a Double Metaphone code, skipped when blank; Soundex and Metaphone
    codes never collide as strings (one always ends in three digits, the other never does),
    so ``"D:"`` only makes that explicit.
    """
    codes_l = [_phonetic_codes(t) for t in name_l.split()]
    codes_r = [_phonetic_codes(t) for t in name_r.split()]
    sx_l, dm_l = [s for s, _ in codes_l if s], [f"D:{d}" for _, d in codes_l if d]
    sx_r, dm_r = [s for s, _ in codes_r if s], [f"D:{d}" for _, d in codes_r if d]
    set_l, set_r = {*sx_l, *dm_l}, {*sx_r, *dm_r}
    exact = float(sx_l == sx_r and dm_l == dm_r and bool(set_l) and bool(set_r))
    if not set_l or not set_r:
        return NAN, NAN, NAN
    common = len(set_l & set_r)
    return exact, common / len(set_l | set_r), common / len(set_l)


def _row(pairs: pd.DataFrame, s1n: pd.DataFrame, pooln: pd.DataFrame, i: int) -> pd.Series:
    return build_features(pairs, s1n, pooln, groups=("phonetic",)).iloc[i]


def test_soundex_variant_scores_high():
    """"Smith Bakery" / "Smyth Bakery": same Soundex on every token (first letter unchanged)."""
    s1n = _records(_rec("S1-1", "smith bakery"))
    pooln = _records(_rec("S2-1", "smyth bakery"))
    row = _row(_pairs(("S1-1", "S2-1", 0, NAN, NAN, NAN)), s1n, pooln, 0)
    exact, jaccard, overlap = _expected("smith bakery", "smyth bakery")
    assert (row["phonetic_exact_match"], row["phonetic_jaccard"],
            row["phonetic_token_overlap_ratio"]) == (exact, jaccard, overlap)
    assert row["phonetic_exact_match"] == 1.0  # every token identical under both systems


def test_metaphone_variant_scores_high_despite_different_soundex():
    """"Kumar Traders" / "Cumar Traders": changed first letter breaks Soundex, not Metaphone.

    Soundex alone would score this pair 0 on its changed token (K560 vs C560); Double
    Metaphone still agrees (KMR), so the combined jaccard/overlap stay well above zero -
    the reason the feature computes both systems instead of just Soundex.
    """
    s1n = _records(_rec("S1-1", "kumar traders"))
    pooln = _records(_rec("S2-1", "cumar traders"))
    row = _row(_pairs(("S1-1", "S2-1", 0, NAN, NAN, NAN)), s1n, pooln, 0)
    exact, jaccard, overlap = _expected("kumar traders", "cumar traders")
    assert row["phonetic_exact_match"] == exact == 0.0  # Soundex disagrees on "kumar"/"cumar"
    assert row["phonetic_jaccard"] == pytest.approx(jaccard)
    assert row["phonetic_token_overlap_ratio"] == pytest.approx(overlap)
    assert jaccard > 0.5 and overlap > 0.5  # "traders" plus the shared Metaphone code


def test_unrelated_pair_scores_low():
    """Two names sharing no letters, sound or spelling: every phonetic feature is 0."""
    s1n = _records(_rec("S1-1", "smith bakery"))
    pooln = _records(_rec("S2-1", "kumar traders"))
    row = _row(_pairs(("S1-1", "S2-1", 0, NAN, NAN, NAN)), s1n, pooln, 0)
    assert row["phonetic_exact_match"] == 0.0
    assert row["phonetic_jaccard"] == 0.0
    assert row["phonetic_token_overlap_ratio"] == 0.0


def test_address_phonetic_mirrors_name_phonetic():
    """``addr_phonetic_*`` is the same computation, run on ``addr_norm`` instead of name_core."""
    s1n = _records(_rec("S1-1", addr_norm="rajendra nagar road"))
    pooln = _records(_rec("S2-1", addr_norm="rajendar nagar road"))
    row = _row(_pairs(("S1-1", "S2-1", 0, NAN, NAN, NAN)), s1n, pooln, 0)
    exact, jaccard, overlap = _expected("rajendra nagar road", "rajendar nagar road")
    assert (row["addr_phonetic_exact_match"], row["addr_phonetic_jaccard"],
            row["addr_phonetic_token_overlap_ratio"]) == (exact, jaccard, overlap)
    # the name columns were never set: empty on both sides, so they carry no evidence
    assert row[["phonetic_exact_match", "phonetic_jaccard",
                "phonetic_token_overlap_ratio"]].isna().all()


def test_phonetic_features_nan_on_non_latin():
    """A transliterated record's phonetic codes are noise, not signal: NaN, never 0.

    ``non_latin`` (set during normalisation from the *original* script, before ``anyascii``
    transliterates it to ASCII) is what gates this, not a fresh script check on ``name_core``:
    by the time features run, ``name_core`` is already Latin text, so a same-stage regex would
    never fire. The two names below are unrelated strings that would otherwise score 0, not
    NaN - the non_latin flag must override that.
    """
    s1n = _records(_rec("S1-1", "shakti traders", "5 rajendra nagar", non_latin=True))
    pooln = _records(_rec("S2-1", "xyz unrelated", "9 unrelated road", non_latin=False))
    row = _row(_pairs(("S1-1", "S2-1", 0, NAN, NAN, NAN)), s1n, pooln, 0)
    assert row[PHONETIC].isna().all()
    assert set(PHONETIC) <= NAN_FEATURES


def test_phonetic_features_nan_on_empty_column():
    """No tokens on either side (empty name/address): NaN, the existing "no evidence" policy."""
    s1n = _records(_rec("S1-1"))
    pooln = _records(_rec("S2-1"))
    row = _row(_pairs(("S1-1", "S2-1", 0, NAN, NAN, NAN)), s1n, pooln, 0)
    assert row[PHONETIC].isna().all()


def test_phonetic_features_ignore_non_alpha_tokens():
    """House numbers / postcodes contribute no phonetic code, matching or not."""
    s1n = _records(_rec("S1-1", addr_norm="acme road 411001"))
    pooln = _records(_rec("S2-1", addr_norm="acme road 411002"))
    row = _row(_pairs(("S1-1", "S2-1", 0, NAN, NAN, NAN)), s1n, pooln, 0)
    # "411001" != "411002" as text, but neither has a letter, so it never enters the comparison
    assert row["addr_phonetic_exact_match"] == 1.0
    assert row["addr_phonetic_jaccard"] == 1.0


def test_phonetic_docs_code_distinct_strings_like_every_row():
    """Coding each distinct string once and taking it back to the rows gives the documents of
    coding every row: repeats, a null, an empty string and a digits-only token included."""
    import pyarrow as pa

    from entity_resolution.features import (
        _phonetic_docs,
        _phonetic_metaphone_token,
        _phonetic_soundex_token,
    )
    from entity_resolution.normalize import map_tokens
    arr = pa.array(["kumar trading", None, "", "cumar  trading", "kumar trading", "12 b"])
    sx, dm = _phonetic_docs(arr)
    assert sx.to_pylist() == map_tokens(arr, _phonetic_soundex_token.__wrapped__).to_pylist()
    assert dm.to_pylist() == map_tokens(arr, _phonetic_metaphone_token.__wrapped__).to_pylist()
    assert sx[0].as_py() == sx[4].as_py() == "K560 T635"
