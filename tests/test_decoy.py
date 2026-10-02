"""Tests of the final version's decoy features (decoy.py), the tight decoder and macro_tight."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from entity_resolution import config as C
from entity_resolution.decision import decode_tight
from entity_resolution.decoy import (
    DECOY_COLUMNS,
    GROUP_COLUMNS,
    HOUSE_NUMBER_COLUMNS,
    UNMATCHED_COLUMNS,
    decoy_features,
    first_number,
    group_features,
    house_number_features,
    idf_table,
    others_max,
    unmatched_features,
    with_columns,
)
from entity_resolution.evaluate import macro_tight
from entity_resolution.split import Fold
from entity_resolution.twostage import Stage1Output

NAN = float("nan")


def test_first_number() -> None:
    assert first_number(np.array(["12 4", "7", ""], dtype=object)).tolist() == ["12", "7", ""]


def test_house_number_relation_by_hand() -> None:
    """Equal, truncated, neighbouring, one-digit-noised, missing and distant numbers."""
    hl = np.array(["12", "2014", "7541", "200", "", "15", "", "30"], dtype=object)
    hr = np.array(["12", "201", "7532", "600", "15", "", "", "57"], dtype=object)
    f = house_number_features(hl, hr)
    assert list(f.columns) == HOUSE_NUMBER_COLUMNS
    expect = {  # eq, substr, onedigit, near, |gap|, missing
        0: (1, 0, 0, 0, 0, 0), 1: (0, 1, 0, 0, 1813, 0), 2: (0, 0, 0, 1, 9, 0),
        3: (0, 0, 1, 0, 400, 0), 7: (0, 0, 0, 0, 27, 0)}
    for i, (eq, sub, one, near, gap, miss) in expect.items():
        assert f.iloc[i, :4].tolist() == [eq, sub, one, near]
        assert f["hn_absdiff"].iloc[i] == pytest.approx(np.log1p(gap), rel=1e-6)
        assert f["hn_missing"].iloc[i] == miss
    for i, miss in ((4, 1), (5, 2), (6, 3)):           # a side without a number: NaN
        assert f.iloc[i, :5].isna().all() and f["hn_missing"].iloc[i] == miss


def test_others_max() -> None:
    key = pd.DataFrame({"s1": [0, 0, 0, 1], "num": [5, 5, 7, 5]})
    n, best = others_max(key, np.array([0.9, 0.4, 0.8, 0.3]))
    assert n.tolist() == [1, 1, 0, 0]
    assert best.tolist() == pytest.approx([0.4, 0.9, 0.0, 0.0])


def test_group_features_by_hand() -> None:
    """One S1 entity: an S2 and an S3 record at its number, a decoy at 14, no-number record."""
    pairs = pd.DataFrame({C.S1_ID: ["S1-A"] * 4,
                          C.ENTITY_ID: ["S2-1", "S3-2", "S2-3", "S3-4"]})
    p1 = np.array([0.9, 0.8, 0.6, 0.3])
    hl = np.array(["12"] * 4, dtype=object)
    hr = np.array(["12", "12", "14", ""], dtype=object)
    name_r = np.array(["acme", "acme", "acme holdings", "acme"], dtype=object)
    addr_r = np.array(["12 main st", "12 main st", "14 main st", ""], dtype=object)
    g = group_features(pairs, p1, hl, hr, name_r, addr_r)
    assert list(g.columns) == GROUP_COLUMNS
    np.testing.assert_allclose(g["grp_num_n"], [1, 1, 0, NAN])
    np.testing.assert_allclose(g["grp_num_p1max"], [0.8, 0.9, 0.0, NAN], rtol=1e-6)
    np.testing.assert_allclose(g["grp_num_xsrc"], [1, 1, 0, NAN])
    np.testing.assert_allclose(g["grp_s1num_n"], [2, 2, 2, 2])
    np.testing.assert_allclose(g["grp_name_n"], [2, 2, 0, 2])
    np.testing.assert_allclose(g["grp_name_p1max"], [0.8, 0.9, 0.0, 0.9], rtol=1e-6)
    np.testing.assert_allclose(g["grp_addr_n"], [1, 1, 0, NAN])


def _records(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    """Normalised-record stand-ins: entity_id, name_core, addr_norm, addr_nums."""
    return pd.DataFrame(rows, columns=[C.ENTITY_ID, "name_core", "addr_norm", "addr_nums"])


def test_unmatched_idf_by_hand() -> None:
    """A decoy's extra rare word, an identical pair and a word the pool never holds."""
    pooln = _records([("S2-1", "acme holdings", "14 main", "14"), ("S2-2", "acme", "12 main", "12"),
                      ("S2-3", "globex", "", "")])
    s1n = _records([("S1-A", "acme", "12 main", "12"), ("S1-Z", "zeta", "", "")])
    idf = idf_table(pooln["name_core"])
    assert idf == pytest.approx({"acme": np.log(1.5), "holdings": np.log(3), "globex": np.log(3)})
    pairs = pd.DataFrame({C.S1_ID: ["S1-A", "S1-A", "S1-Z"],
                          C.ENTITY_ID: ["S2-1", "S2-2", "S2-3"]})
    u = unmatched_features(pairs, s1n, pooln, idf)
    assert list(u.columns) == UNMATCHED_COLUMNS
    rare, common = np.log(3), np.log(1.5)
    np.testing.assert_allclose(u.iloc[0], [0, rare, 0, rare, rare / (rare + common), 1],
                               rtol=1e-6)
    np.testing.assert_allclose(u.iloc[1], [0, 0, 0, 0, 0, 0])
    # "zeta" is unknown: the table's median (log 3); no numbers on either side: no conflict
    np.testing.assert_allclose(u.iloc[2], [rare, rare, rare, rare, 1.0, 0], rtol=1e-6)


def test_decoy_features_assemble_in_column_order() -> None:
    pooln = _records([("S2-1", "acme holdings", "14 main", "14"),
                      ("S3-2", "acme", "12 main", "12")])
    s1n = _records([("S1-A", "acme", "12 main", "12")])
    pairs = pd.DataFrame({C.S1_ID: ["S1-A", "S1-A"], C.ENTITY_ID: ["S2-1", "S3-2"]})
    o = Stage1Output(pairs, pd.DataFrame({"p1": np.array([0.4, 0.9], np.float32)}), 2)
    feats = decoy_features(o, s1n, pooln)
    assert list(feats.columns) == DECOY_COLUMNS and len(feats) == 2
    # 14 vs 12 is a one-digit change (not "near", which excludes it); 12 vs 12 is equal
    assert feats["hn_eq"].tolist() == [0, 1] and feats["hn_onedigit"].tolist() == [1, 0]
    assert feats["grp_s1num_n"].tolist() == [1, 1] and feats["uidf_max_r"].iloc[0] > 0
    out = with_columns(o, feats)
    assert list(out.X.columns) == ["p1", *DECOY_COLUMNS] and out.n_all == 2


def _scored(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[C.S1_ID, C.ENTITY_ID, "prob"])


def test_decode_tight_weight_trades_recall_for_precision() -> None:
    """Two likely candidates: plain F0.5 keeps both, a false merge at x8 keeps the best only."""
    s = _scored([("a", "x", 0.95), ("a", "y", 0.9)])
    assert decode_tight(s, 1.0, 0.0)[C.ENTITY_ID].tolist() == ["x", "y"]
    assert decode_tight(s, 8.0, 0.0)[C.ENTITY_ID].tolist() == ["x"]
    assert decode_tight(s, 1.0, 0.0, max_matches=1)[C.ENTITY_ID].tolist() == ["x"]


def test_decode_tight_empty_one_to_one_and_order() -> None:
    """An unlikely lone candidate stays unmatched; a pool record keeps one owner."""
    s = _scored([("b", "z", 0.02), ("c", "x", 0.8), ("a", "x", 0.9), ("a", "w", 0.97)])
    out = decode_tight(s, 3.0, 0.4)
    assert out[[C.S1_ID, C.ENTITY_ID]].values.tolist() == [["a", "w"], ["a", "x"]]
    assert list(out.columns) == [C.S1_ID, C.ENTITY_ID, "prob"]
    assert decode_tight(s.iloc[:0], 3.0, 0.4).empty


def test_macro_tight_charges_false_merges() -> None:
    """A perfect entity scores 1; a merged singleton -fp_weight + 1 (0 at x1)."""
    s1 = pd.DataFrame({C.ENTITY_ID: ["S1-a", "S1-b"]})
    pool = pd.DataFrame({C.ENTITY_ID: ["S2-x", "S2-y"]})
    fold = Fold("t", s1, pool, pool.iloc[:0], pd.DataFrame({C.S1_ID: ["S1-a"],
                                                            C.ENTITY_ID: ["S2-x"]}))
    pred = pd.DataFrame({C.S1_ID: ["S1-a", "S1-b"], C.ENTITY_ID: ["S2-x", "S2-y"]})
    assert macro_tight(pred, fold, 1.0) == pytest.approx(0.5)
    assert macro_tight(pred, fold, 3.0) == pytest.approx(-0.5)
