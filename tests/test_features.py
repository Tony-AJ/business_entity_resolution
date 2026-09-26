"""Pair features (07 §7, 12 §5): values, NaN policy, context, chunking, determinism.

Normalised frames are built inline from already-normalised strings: ``_record`` derives the
NORM_COLUMNS keys as 05 §4 defines them, so these tests do not depend on normalize.py.
"""
from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from rapidfuzz import fuzz

from entity_resolution import config as C
from entity_resolution.blocking import PAIR_COLUMNS
from entity_resolution.evidence import TokenEvidence
from entity_resolution.features import (
    DEFAULT_GROUPS,
    EVIDENCE_GROUPS,
    FEATURE_COLUMNS,
    FREQ_COLUMNS,
    NAN_FEATURES,
    REGISTRY,
    STATS_GROUPS,
    build_features,
    feature_names,
    iter_chunks,
    pool_stats,
)
from entity_resolution.normalize import NORM_COLUMNS

NAN = np.nan
ALL_GROUPS = tuple(REGISTRY)
# a learned token evidence table for the toy's words (the tok_evidence group needs one)
EVIDENCE = TokenEvidence(pool={"trader": 0.4, "globex": -2.5, "hill": 1.5},
                         s1={"traders": 1.2, "kitten": -2.0}, strong=1.0)
FLAGS = ["pass_exact", "pass_name_char", "pass_name_addr", "first_eq", "sorted_eq",
         "prefix4_eq", "legal_eq", "legal_missing_l", "legal_missing_r", "region_eq", "last_eq",
         "addr_empty_r", "is_s3", "non_latin_r", "addr_empty_l"]
TRISTATE = ["num_shared_any", "num_first_eq", "postcode_eq"]  # 1 / 0, NaN when a side has none
NAME_SIMS = [*FEATURE_COLUMNS["name_fuzzy"], "tok_jaccard", "tok_dice", "len_ratio_name"]
ADDR_SIMS = ["ad_token_set", "ad_partial", "ad_ratio", "ad_jaccard", "ad_contain"]
UNIT_RANGE = [*FEATURE_COLUMNS["blocking"][3:], *NAME_SIMS, "num_jaccard", *ADDR_SIMS,
              "ctx_gap_name", "ctx_gap_addr", "ad_contain_r", "num_contain_l", "num_contain_r",
              "addr_len_ratio"]
# 12 §5: equal under an L/R swap; nm_partial, ad_partial and ad_contain are asymmetric by
# design, and ctx_*, *_l / *_r and addr_empty_r are one-sided
SYMMETRIC = [c for c in [*FEATURE_COLUMNS["name_fuzzy"], *FEATURE_COLUMNS["name_tokens"],
                         *FEATURE_COLUMNS["numeric"], *ADDR_SIMS, "legal_eq", "region_eq",
                         "last_eq", "len_ratio_name"]
             if c not in {"nm_partial", "ad_partial", "ad_contain", "tok_len_l", "tok_len_r"}]


def _record(entity_id: str, core: str = "", legal: str = "", addr: str = "", region: str = "",
            non_latin: bool = False, country: str = "US") -> dict:
    """One normalised record from an already-normalised core name, legal form and address."""
    words, tokens = core.split(), addr.split()
    numbers = [t for t in tokens if t.isdigit()]
    return {C.ENTITY_ID: entity_id, C.COUNTRY: country, "non_latin": non_latin,
            "name_norm": " ".join([*words, *legal.split()]), "name_core": core,
            "legal_form": legal, "name_first": words[0] if words else "",
            "name_sorted": " ".join(sorted(words)), "name_squash": core.replace(" ", ""),
            "addr_norm": addr, "addr_nums": " ".join(numbers),
            "postcode": next((t for t in numbers if len(t) in (5, 6)), ""), "region": region,
            "addr_last": " ".join([t for t in tokens if not t.isdigit()][-2:]),
            "addr_tokens": len(tokens), "name_addr": f"{core} {addr}".strip()}


def _records(*records: dict) -> pd.DataFrame:
    """Normalised frame: NORM_COLUMNS with their dtypes (str, bool non_latin, int16)."""
    text = {c: "str" for c in NORM_COLUMNS if c not in ("non_latin", "addr_tokens")}
    df = pd.DataFrame(list(records), columns=NORM_COLUMNS).astype(
        {**text, "non_latin": bool, "addr_tokens": "int16"})
    for i, col in enumerate(FREQ_COLUMNS):  # rates pipeline.add_frequencies adds in real runs
        df[col] = np.arange(len(df), dtype=np.float32) * (i + 1)
    df["name_core_nofill"] = df["name_core"]  # load_normalised adds it; no filler learned here
    return df


def _pairs(*rows: tuple) -> pd.DataFrame:
    """Candidate pairs from (s1, pool, pass, sim_name_char, sim_name_addr_word, sim_addr_char)."""
    df = pd.DataFrame(list(rows), columns=PAIR_COLUMNS)
    return df.astype({C.S1_ID: "str", C.ENTITY_ID: "str", "pass": np.uint8,
                      **{c: np.float32 for c in PAIR_COLUMNS[3:]}})


def _toy() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(pairs, s1n, pooln), grouped by S1 id.

    Row 0 pairs identical records; S1-1 vs S3-1 is a typo variant (transliterated pool row);
    S2-3 is a candidate of two S1 groups (in-degree 2); S1-3 and S2-2 have empty addresses;
    S1-4 has an empty name; the sims are NaN where the pass did not propose the pair.
    """
    s1n = _records(
        _record("S1-1", "acme traders", "pvt ltd", "12 main street springfield 411001",
                "maharashtra"),
        _record("S1-2", "kitten", "", "5 oak lane dover", "delaware"),
        _record("S1-3", "a b", "llc", ""),
        _record("S1-4", "", "", "7 hill road pune 411007", "maharashtra"))
    pooln = _records(
        _record("S2-1", "acme traders", "pvt ltd", "12 main street springfield 411001",
                "maharashtra"),
        _record("S3-1", "acme trader", "", "12 main st springfield 411002", "maharashtra",
                non_latin=True),
        _record("S2-2", "sitting"),
        _record("S3-2", "b c", "llc", "9 elm road"),
        _record("S2-3", "globex", "inc", "4 oak avenue austin"),
        _record("S2-4", "hill cafe", "", "7 hill road pune", "maharashtra"))
    pairs = _pairs(("S1-1", "S2-1", 1, 1.0, 0.9, NAN), ("S1-1", "S3-1", 8, 0.8, NAN, NAN),
                   ("S1-1", "S2-3", 16, NAN, 0.25, NAN), ("S1-2", "S2-2", 8, 0.5, NAN, NAN),
                   ("S1-2", "S2-3", 8, 0.35, NAN, NAN), ("S1-3", "S3-2", 16, NAN, 0.3, NAN),
                   ("S1-4", "S2-4", 48, NAN, 0.6, 0.7))
    return pairs, s1n, pooln


def test_registry_names_unique_and_count():
    every = feature_names(ALL_GROUPS)
    assert len(every) == len(set(every)) == sum(len(v) for v in FEATURE_COLUMNS.values())
    assert feature_names() == [c for g in DEFAULT_GROUPS for c in FEATURE_COLUMNS[g]]
    assert len(feature_names()) == 47 and "pool_context" not in DEFAULT_GROUPS
    assert set(FEATURE_COLUMNS) == set(REGISTRY) and NAN_FEATURES <= set(every)
    pairs, s1n, pooln = _toy()
    assert list(build_features(pairs, s1n, pooln, ALL_GROUPS,
                               evidence=EVIDENCE).columns) == every
    assert list(build_features(pairs, s1n, pooln).columns) == feature_names()
    with pytest.raises(ValueError, match="unknown feature groups"):
        feature_names(["blocking", "nope"])


def test_identical_records_score_one():
    pairs, s1n, pooln = _toy()
    row = build_features(pairs, s1n, pooln).iloc[0]  # S1-1 vs S2-1: the same record
    assert (row[[*NAME_SIMS, "num_jaccard", *ADDR_SIMS]] == 1.0).all()
    assert (row[["first_eq", "sorted_eq", "prefix4_eq", "legal_eq", *TRISTATE, "region_eq",
                 "last_eq"]] == 1.0).all()
    assert row["tok_common"] == row["tok_len_l"] == row["tok_len_r"] == 2


def test_disjoint_records_score_zero():
    s1n = _records(_record("S1-1", "abcd", "", "12 main street dover 411001"))
    pooln = _records(_record("S2-1", "xyzw", "", "99 oak lane austin 73301"))
    row = build_features(_pairs(("S1-1", "S2-1", 8, 0.3, NAN, NAN)), s1n, pooln).iloc[0]
    assert (row[FEATURE_COLUMNS["name_fuzzy"]] == 0).all()  # not one character in common
    assert row["tok_jaccard"] == row["tok_dice"] == row["tok_common"] == 0
    assert row["first_eq"] == row["sorted_eq"] == row["prefix4_eq"] == 0
    assert row["ad_jaccard"] == row["ad_contain"] == row["num_jaccard"] == 0
    assert (row[TRISTATE] == 0).all()
    assert row["ad_token_set"] < 0.5 and row["ad_ratio"] < 0.5


def test_known_values():
    s1n = _records(_record("S1-1", "kitten", "", "1 main street 411001"),
                   _record("S1-2", "a b", "", "2 park road 560001"),
                   _record("S1-3", "c", "", "3 hill lane"))
    pooln = _records(_record("S2-1", "sitting", "", "1 main street 411001"),
                     _record("S2-2", "b c", "", "2 park road 560002"),
                     _record("S2-3", "c", "", "3 hill lane 400001"))
    pairs = _pairs(*[(f"S1-{i}", f"S2-{i}", 8, 0.5, NAN, NAN) for i in (1, 2, 3)])
    X = build_features(pairs, s1n, pooln)
    assert X.loc[0, "nm_ratio"] == pytest.approx(fuzz.ratio("kitten", "sitting") / 100)
    assert X.loc[0, "core_lev"] == pytest.approx(1 - 3 / 7)  # 3 edits, longer name 7 chars
    assert X.loc[0, "len_ratio_name"] == pytest.approx(6 / 7)
    assert X.loc[1, "tok_jaccard"] == pytest.approx(1 / 3)  # {a, b} vs {b, c}
    assert X.loc[1, "tok_dice"] == pytest.approx(1 / 2)
    assert X.loc[1, "num_jaccard"] == pytest.approx(1 / 3)  # {2, 560001} vs {2, 560002}
    assert X.loc[1, "num_first_eq"] == 1
    assert X.loc[0, "postcode_eq"] == 1 and X.loc[1, "postcode_eq"] == 0
    assert np.isnan(X.loc[2, "postcode_eq"])  # S1-3 has no postcode


def test_address_extra_values():
    addresses = [("12 main street pune 411001", "12 main street pune 411001"),  # identical
                 ("main street dover", "main street dover near main gate"),  # pool adds words
                 ("12 park road 411001", "12 park road 411999 gate 7"),  # same postal area
                 ("1 hill lane 411001", "1 hill lane 560001"),  # another postal area
                 ("3 elm road 411001", "3 elm road"),  # the pool has no postcode
                 ("", "9 elm road 411001")]  # the S1 address is empty
    s1n = _records(*[_record(f"S1-{i}", "acme", "", a) for i, (a, _) in enumerate(addresses)])
    pooln = _records(*[_record(f"S2-{i}", "acme", "", b) for i, (_, b) in enumerate(addresses)])
    pairs = _pairs(*[(f"S1-{i}", f"S2-{i}", 8, 0.5, NAN, NAN) for i in range(len(addresses))])
    X = build_features(pairs, s1n, pooln, ("numeric", "address", "address_extra"))
    shares = [c for c in FEATURE_COLUMNS["address_extra"] if c != "addr_empty_l"]  # NaN-able
    assert (X.loc[0, shares] == 1).all() and X.loc[0, "addr_empty_l"] == 0
    # {main, street, dover} in {main, street, dover, near, gate}: the repeated main counts once
    assert X.loc[1, "ad_contain"] == 1 and X.loc[1, "ad_contain_r"] == pytest.approx(3 / 5)
    assert X.loc[1, "addr_len_ratio"] == pytest.approx(3 / 5)
    assert X.loc[1, ["num_contain_l", "num_contain_r", "postcode_prefix_eq"]].isna().all()
    # numbers {12, 411001} vs {12, 411999, 7}; 3 of the pool's 6 address tokens are in S1's 4
    assert X.loc[2, "num_contain_l"] == pytest.approx(1 / 2)
    assert X.loc[2, "num_contain_r"] == pytest.approx(1 / 3)
    assert X.loc[2, "ad_contain_r"] == pytest.approx(3 / 6)
    assert X.loc[2, "addr_len_ratio"] == pytest.approx(4 / 6)
    assert X.loc[2, "postcode_prefix_eq"] == 1 and X.loc[2, "postcode_eq"] == 0  # 411 area
    assert X.loc[3, "postcode_prefix_eq"] == 0  # 411001 vs 560001
    assert X.loc[4, "num_contain_l"] == pytest.approx(1 / 2) and X.loc[4, "num_contain_r"] == 1
    assert np.isnan(X.loc[4, "postcode_prefix_eq"])
    assert X["addr_empty_l"].tolist() == [0, 0, 0, 0, 0, 1]
    assert X.loc[5, shares].isna().all()  # no S1 address: no tokens, numbers or postcode


def test_token_sets_ignore_stray_spaces():
    s1n = _records(_record("S1-1", "a b", "", "12 main street"))
    pooln = _records(_record("S2-1", "b c", "", "12 main road"))
    pairs = _pairs(("S1-1", "S2-1", 8, 0.5, NAN, NAN))
    clean = build_features(pairs, s1n, pooln)
    s1n.loc[0, ["name_core", "addr_norm"]] = [" a  b", "12  main street "]  # not normalised
    stray = build_features(pairs, s1n, pooln)
    cols = ["tok_jaccard", "tok_common", "tok_len_l", "ad_jaccard", "ad_contain"]
    pd.testing.assert_frame_equal(stray[cols], clean[cols])


def test_nan_policy():
    pairs, s1n, pooln = _toy()
    X = build_features(pairs, s1n, pooln, ALL_GROUPS, evidence=EVIDENCE)
    with_nan = set(X.columns[X.isna().any()])
    assert with_nan <= NAN_FEATURES
    assert {"sim_addr_char", "nm_ratio", "tok_jaccard", "ad_token_set", "postcode_eq",
            "ctx_gap_name", "len_ratio_name"} <= with_nan  # the toy exercises the NaN cases
    assert X[FLAGS].isin([0.0, 1.0]).all().all()  # isin is False on NaN
    tri = X[[*TRISTATE, "postcode_prefix_eq"]]  # opt-in: default-group tests read TRISTATE
    assert (tri.isin([0.0, 1.0]) | tri.isna()).all().all()
    unit = X[UNIT_RANGE]
    assert (((unit >= 0) & (unit <= 1)) | unit.isna()).all().all()


def test_missing_fields():
    pairs, s1n, pooln = _toy()
    X = build_features(pairs, s1n, pooln)
    empty_pool_addr = X.iloc[3]  # S1-2 vs S2-2: pool address empty
    assert empty_pool_addr[ADDR_SIMS].isna().all() and empty_pool_addr["addr_empty_r"] == 1
    assert empty_pool_addr[["num_jaccard", *TRISTATE]].isna().all()
    empty_s1_addr = X.iloc[5]  # S1-3 vs S3-2: S1 address empty, pool address present
    assert empty_s1_addr[ADDR_SIMS].isna().all() and empty_s1_addr["addr_empty_r"] == 0
    empty_name = X.iloc[6]  # S1-4 vs S2-4: S1 name empty
    assert empty_name[NAME_SIMS].isna().all()
    assert empty_name["tok_common"] == empty_name["tok_len_l"] == empty_name["first_eq"] == 0
    assert empty_name["tok_len_r"] == 2


def test_context_rank_and_gap():
    s1n = _records(_record("S1-1", "acme", "", "1 main street"), _record("S1-2", "zeta"))
    pooln = _records(_record("S2-1", "acme", "", "1 main street"), _record("S2-2", "acme"),
                     _record("S2-3", "acme", "", "1 main road"),
                     *[_record(f"S3-{i}", "zeta") for i in range(1, 5)])
    pairs = _pairs(("S1-1", "S2-1", 8, 0.5, NAN, NAN), ("S1-1", "S2-2", 8, 0.9, NAN, NAN),
                   ("S1-1", "S2-3", 8, 0.7, NAN, NAN), ("S1-2", "S3-1", 8, 0.6, NAN, NAN),
                   ("S1-2", "S3-2", 16, NAN, 0.4, NAN), ("S1-2", "S3-3", 8, 0.6, NAN, NAN),
                   ("S1-2", "S3-4", 8, 0.2, NAN, NAN))
    X = build_features(pairs, s1n, pooln, ("address", "context"))
    assert X["ctx_rank_name"].tolist() == [3, 1, 2, 1, 4, 1, 3]  # ties share, NaN last
    assert X["ctx_gap_name"].tolist() == pytest.approx([0.4, 0.0, 0.2, 0.0, NAN, 0.0, 0.4],
                                                       nan_ok=True)
    assert X["ctx_n_cands"].tolist() == [3, 3, 3, 4, 4, 4, 4]
    # address: identical 1.0 first, "main road" second, the empty pool address NaN last
    assert X["ctx_rank_addr"].tolist()[:3] == [1, 3, 2]
    assert X.loc[2, "ctx_gap_addr"] == pytest.approx(1 - X.loc[2, "ad_token_set"])
    assert np.isnan(X.loc[1, "ctx_gap_addr"])


def test_pool_indegree_partition_wide():
    pairs, s1n, pooln = _toy()
    # chunk_rows=1 puts every S1 group in its own chunk; S2-3 competes in S1-1 and S1-2
    X = build_features(pairs, s1n, pooln, ("pool_context",), chunk_rows=1)
    hub = (pairs[C.ENTITY_ID] == "S2-3").to_numpy()
    assert X["ctx_pool_indegree"][hub].tolist() == [2, 2]
    assert (X["ctx_pool_indegree"][~hub] == 1).all()


def test_iter_chunks_never_splits_group():
    ids = ["S1-a"] * 3 + ["S1-b"] + ["S1-c"] * 4 + ["S1-d"] * 2
    pairs = pd.DataFrame({C.S1_ID: pd.Series(ids, dtype="str")})
    for rows in (1, 2, 3, 5, 100):
        slices = list(iter_chunks(pairs, rows))
        assert slices[0].start == 0 and slices[-1].stop == len(ids)
        assert all(a.stop == b.start for a, b in pairwise(slices))
        for sl in slices:
            inside = ids[sl]
            assert all(inside.count(i) == ids.count(i) for i in set(inside))  # whole groups
            assert len(inside) >= rows or sl.stop == len(ids)
    assert [(s.start, s.stop) for s in iter_chunks(pairs, 2)] == [(0, 3), (3, 8), (8, 10)]
    with pytest.raises(ValueError, match="grouped"):
        iter_chunks(pd.DataFrame({C.S1_ID: ["S1-a", "S1-b", "S1-a"]}), 2)


def test_chunking_invariant():
    pairs, s1n, pooln = _toy()
    whole = build_features(pairs, s1n, pooln, ALL_GROUPS, evidence=EVIDENCE)
    for rows in (1, 2, 4):
        pd.testing.assert_frame_equal(build_features(pairs, s1n, pooln, ALL_GROUPS, rows,
                                                     evidence=EVIDENCE), whole)


def test_deterministic_and_float32():
    pairs, s1n, pooln = _toy()
    pairs.index = pd.Index([f"p{i}" for i in range(len(pairs))])
    first = build_features(pairs, s1n, pooln, ALL_GROUPS, evidence=EVIDENCE)
    pd.testing.assert_frame_equal(build_features(pairs, s1n, pooln, ALL_GROUPS,
                                                 evidence=EVIDENCE), first)
    assert (first.dtypes == np.float32).all()
    assert first.index.equals(pairs.index)


def test_empty_pairs_keep_columns():
    pairs, s1n, pooln = _toy()
    X = build_features(pairs.iloc[:0], s1n, pooln)
    assert X.shape == (0, len(feature_names())) and list(X.columns) == feature_names()
    assert (X.dtypes == np.float32).all()
    assert list(iter_chunks(pairs.iloc[:0], 10)) == []


def test_symmetric_similarities():
    pairs, s1n, pooln = _toy()
    X = build_features(pairs, s1n, pooln)
    swapped = pairs.rename(columns={C.S1_ID: C.ENTITY_ID, C.ENTITY_ID: C.S1_ID})[PAIR_COLUMNS]
    swapped = swapped.sort_values(C.S1_ID, kind="stable")  # regroup by the new S1 side
    Y = build_features(swapped, pooln, s1n).loc[X.index]  # pool records play S1, and back
    pd.testing.assert_frame_equal(Y[SYMMETRIC], X[SYMMETRIC])
    assert Y["tok_len_l"].equals(X["tok_len_r"]) and Y["tok_len_r"].equals(X["tok_len_l"])


def test_address_extra_mirrors_under_swap():
    pairs, s1n, pooln = _toy()
    groups = ("address", "address_extra")
    X = build_features(pairs, s1n, pooln, groups)
    swapped = pairs.rename(columns={C.S1_ID: C.ENTITY_ID, C.ENTITY_ID: C.S1_ID})[PAIR_COLUMNS]
    swapped = swapped.sort_values(C.S1_ID, kind="stable")  # regroup by the new S1 side
    Y = build_features(swapped, pooln, s1n, groups).loc[X.index]
    # the one-sided columns trade places under an L/R swap: *_r is the reverse of *_l
    for a, b in (("ad_contain", "ad_contain_r"), ("num_contain_l", "num_contain_r"),
                 ("addr_empty_l", "addr_empty_r")):
        assert Y[a].equals(X[b]) and Y[b].equals(X[a])
    both = ["postcode_prefix_eq", "addr_len_ratio"]
    pd.testing.assert_frame_equal(Y[both], X[both])


def test_groups_alone_match_full_build():
    pairs, s1n, pooln = _toy()
    whole = build_features(pairs, s1n, pooln, ALL_GROUPS, evidence=EVIDENCE)
    for g in ALL_GROUPS:  # context alone computes ad_token_set itself
        alone = build_features(pairs, s1n, pooln, (g,), evidence=EVIDENCE)
        pd.testing.assert_frame_equal(alone, whole[FEATURE_COLUMNS[g]])


def test_groups_are_functions_of_aligned_rows():
    pairs, s1n, pooln = _toy()
    left = s1n.set_index(C.ENTITY_ID).loc[pairs[C.S1_ID]].reset_index()
    right = pooln.set_index(C.ENTITY_ID).loc[pairs[C.ENTITY_ID]].reset_index()
    whole = build_features(pairs, s1n, pooln, ALL_GROUPS, evidence=EVIDENCE)
    for g, group in REGISTRY.items():
        extra = {"stats": pool_stats(pooln, (g,))} if g in STATS_GROUPS else {}
        if g in EVIDENCE_GROUPS:
            extra["evidence"] = EVIDENCE
        pd.testing.assert_frame_equal(group(pairs, left, right, **extra),
                                      whole[FEATURE_COLUMNS[g]])


def test_bad_input_raises():
    pairs, s1n, pooln = _toy()
    with pytest.raises(ValueError, match="source1_entity_id values are not in"):
        build_features(pairs, s1n.iloc[1:], pooln)
    with pytest.raises(ValueError, match="entity_id values are not in"):
        build_features(pairs, s1n, pooln.iloc[1:])
    with pytest.raises(ValueError, match="grouped"):
        build_features(pairs.iloc[[0, 3, 1]], s1n, pooln)
    with pytest.raises(ValueError, match="lack"):
        build_features(pairs.drop(columns="pass"), s1n, pooln)


def test_frequency_group_passes_rates_and_core_eq() -> None:
    """The frequency group copies the side rates and flags equal non-empty core names."""
    from entity_resolution.features import REGISTRY
    pairs = pd.DataFrame({"source1_entity_id": ["S1-1", "S1-1"], "entity_id": ["S2-1", "S2-2"]})
    left = pd.DataFrame({"name_core": ["acme", "acme"], "freq_same": [2.0, 2.0],
                         "freq_other": [5.0, 5.0], "freq_first_other": [7.0, 7.0]})
    right = pd.DataFrame({"name_core": ["acme", ""], "freq_same": [3.0, np.nan],
                          "freq_other": [4.0, np.nan], "freq_first_other": [1.0, 1.0]})
    out = REGISTRY["frequency"](pairs, left.astype({"name_core": "str"}),
                                right.astype({"name_core": "str"}))
    assert out["core_eq"].tolist() == [1.0, 0.0]
    assert out["fq_s1_l"].tolist() == [2.0, 2.0] and out["fq_pool_r"].iloc[0] == 3.0
    assert np.isnan(out["fq_s1_r"].iloc[1])


# ------------------------------------------------ pool-statistics groups (C2, C5) ----
IDF_NAME = [c for c in FEATURE_COLUMNS["idf"] if c.startswith("idf_name")]
IDF_ADDR = [c for c in FEATURE_COLUMNS["idf"] if c.startswith("idf_addr")]


def _cafes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One S1 "acme cafe" against three pool records: identical, other rare word, bare."""
    s1n = _records(_record("S1-1", "acme cafe", "", "1 main road"))
    pooln = _records(_record("S2-1", "acme cafe", "", "1 main road"),
                     _record("S2-2", "zenith cafe", "", "2 main road"),
                     _record("S2-3", "cafe"),
                     _record("S2-4", "blue cafe", "", "3 main road"))
    pairs = _pairs(("S1-1", "S2-1", 1, 1.0, NAN, NAN), ("S1-1", "S2-2", 8, 0.5, NAN, NAN),
                   ("S1-1", "S2-3", 8, 0.6, NAN, NAN))
    return pairs, s1n, pooln


def test_idf_known_values():
    pairs, s1n, pooln = _cafes()
    X = build_features(pairs, s1n, pooln, ("idf",))
    n = 4  # pool records of the country; idf = ln((1 + n) / (1 + df)) + 1
    rare, cafe, best = np.log(5 / 2) + 1, np.log(5 / 5) + 1, np.log(1 + n) + 1
    assert X.loc[0, IDF_NAME].tolist() == pytest.approx([1.0, rare / best, 1.0, 1.0])
    # acme cafe vs zenith cafe: only the common "cafe" is shared
    assert X.loc[1, "idf_name_cos"] == pytest.approx(cafe**2 / (rare**2 + cafe**2))
    assert X.loc[1, "idf_name_top"] == pytest.approx(cafe / best)
    assert X.loc[1, "idf_name_cover_l"] == pytest.approx(cafe / (rare + cafe))
    assert X.loc[1, "idf_name_cover_r"] == pytest.approx(cafe / (rare + cafe))
    # acme cafe vs cafe: the pool name is fully covered, the S1 name only by its common word
    assert X.loc[2, "idf_name_cos"] == pytest.approx(cafe / np.sqrt(rare**2 + cafe**2))
    assert X.loc[2, "idf_name_cover_l"] == pytest.approx(cafe / (rare + cafe))
    assert X.loc[2, "idf_name_cover_r"] == pytest.approx(1.0)
    # addresses: "main" and "road" are in 3 of 4 records, each house number in one
    num, word = np.log(5 / 2) + 1, np.log(5 / 4) + 1
    assert X.loc[1, "idf_addr_cos"] == pytest.approx(2 * word**2 / (num**2 + 2 * word**2))
    assert X.loc[1, "idf_addr_top"] == pytest.approx(word / best)
    assert X.loc[2, IDF_ADDR].isna().all()  # the pool address is empty


def test_stats_are_counted_per_country():
    s1n = _records(_record("S1-1", "cafe roma", "", "1 rue", country="France"),
                   _record("S1-2", "cafe roma", "", "1 main st"))
    pooln = _records(_record("S2-1", "cafe roma", "", "1 rue", country="France"),
                     *[_record(f"S2-{i}", "cafe roma", "", f"{i} main st") for i in (2, 3, 4)])
    pairs = _pairs(("S1-1", "S2-1", 1, 1.0, NAN, NAN),
                   *[("S1-2", f"S2-{i}", 1, 1.0, NAN, NAN) for i in (2, 3, 4)])
    groups = ("idf", "token_freq")
    X = build_features(pairs, s1n, pooln, groups)
    # every name token is in every record of its country: idf 1, N = 1 (France) or 3 (US)
    assert X["idf_name_top"].tolist() == pytest.approx([1 / (np.log(2) + 1)]
                                                       + [1 / (np.log(4) + 1)] * 3)
    assert X["freq_name_l"].tolist() == [1, 3, 3, 3]  # France holds one "cafe roma", US three
    assert X["freq_name_r"].tolist() == [1, 3, 3, 3]
    assert X["freq_addr_l"].tolist() == [1, 0, 0, 0]  # "1 main st" is in no US pool record
    assert X["freq_addr_r"].tolist() == [1, 1, 1, 1]
    # a mixed-country fold (val) gives France the same values as its own partition (test)
    alone = build_features(pairs.iloc[:1], s1n.iloc[:1], pooln.iloc[:1], groups)
    pd.testing.assert_frame_equal(alone, X.iloc[:1])


def test_token_freq_missing_and_counts():
    pairs, s1n, pooln = _toy()
    X = build_features(pairs, s1n, pooln, ("token_freq",))
    assert np.isnan(X.loc[6, "freq_name_l"])  # S1-4 has no name
    assert np.isnan(X.loc[3, "freq_addr_r"]) and np.isnan(X.loc[5, "freq_addr_l"])
    named = X["freq_name_r"].notna()
    assert (X.loc[named, "freq_name_r"] >= 1).all()  # the pool record counts itself
    assert X.loc[0, "freq_name_l"] == 1 and X.loc[1, "freq_name_l"] == 1  # one "acme traders"


def test_ctx_idf_rank_gap_and_same_name():
    s1n = _records(_record("S1-1", "acme cafe", "", "1 main road"))
    pooln = _records(_record("S2-1", "acme cafe", "", "1 main road"),
                     _record("S2-2", "acme cafe", "", "9 oak lane"),
                     _record("S2-3", "zenith cafe", "", "1 main road"))
    pairs = _pairs(*[("S1-1", f"S2-{i}", 8, 0.5, NAN, NAN) for i in (1, 2, 3)])
    X = build_features(pairs, s1n, pooln, ("idf", "ctx_idf"))
    assert X["ctx_rank_idf_name"].tolist() == [1, 1, 3]  # the two exact names tie
    assert X["ctx_gap_idf_name"].tolist() == pytest.approx(
        [0.0, 0.0, 1 - X.loc[2, "idf_name_cos"]])
    assert X["ctx_rank_idf_addr"].tolist() == [1, 3, 1]
    assert X.loc[1, "ctx_gap_idf_addr"] == pytest.approx(1.0)  # no address token shared
    assert X["ctx_n_same_name"].tolist() == [2, 2, 2]
    alone = build_features(pairs, s1n, pooln, ("ctx_idf",))  # computes the cosines itself
    pd.testing.assert_frame_equal(alone, X[FEATURE_COLUMNS["ctx_idf"]])


def test_stats_group_ranges():
    pairs, s1n, pooln = _toy()
    X = build_features(pairs, s1n, pooln, tuple(g for g in ALL_GROUPS if g in STATS_GROUPS))
    unit = X[[*FEATURE_COLUMNS["idf"], "ctx_gap_idf_name", "ctx_gap_idf_addr"]]
    assert (((unit >= 0) & (unit <= 1)) | unit.isna()).all().all()
    counts = X[[*FEATURE_COLUMNS["token_freq"], "ctx_n_same_name"]]
    assert (((counts >= 0) & (counts == counts.round())) | counts.isna()).all().all()
    ranks = X[["ctx_rank_idf_name", "ctx_rank_idf_addr"]]
    assert (ranks >= 1).all().all() and (ranks == ranks.round()).all().all()


def test_stats_groups_need_pool_statistics():
    pairs, s1n, pooln = _toy()
    left = s1n.set_index(C.ENTITY_ID).loc[pairs[C.S1_ID]].reset_index()
    right = pooln.set_index(C.ENTITY_ID).loc[pairs[C.ENTITY_ID]].reset_index()
    for g in STATS_GROUPS:
        with pytest.raises(ValueError, match="needs pool statistics"):
            REGISTRY[g](pairs, left, right)
    assert pool_stats(pooln, DEFAULT_GROUPS) is None  # the v001 groups read no statistics
    with pytest.raises(ValueError, match="pooln lacks columns"):
        pool_stats(pooln.drop(columns=[C.COUNTRY]), ("idf",))


def test_shared_stats_match_whole_build():
    # pipeline.score features one slice at a time with statistics of the whole pool
    pairs, s1n, pooln = _toy()
    groups = ("idf", "ctx_idf", "token_freq")
    whole = build_features(pairs, s1n, pooln, groups)
    stats = pool_stats(pooln, groups)
    parts = [build_features(pairs.iloc[sl], s1n, pooln, groups, stats=stats)
             for sl in iter_chunks(pairs, 1)]
    pd.testing.assert_frame_equal(pd.concat(parts), whole)


def test_nofill_group_values():
    """Similarities of the filler-free core names, word-set equality, fillers removed."""
    s1n = _records(_record("S1-1", "acme center"), _record("S1-2", "globex"), _record("S1-3"))
    s1n["name_core_nofill"] = ["acme", "globex", ""]
    pooln = _records(_record("S2-1", "center acme services"), _record("S2-2", "globex initech"),
                     _record("S2-3", "hooli"))
    pooln["name_core_nofill"] = ["acme", "globex initech", "hooli"]
    pairs = _pairs(("S1-1", "S2-1", 128, NAN, NAN, NAN), ("S1-2", "S2-2", 16, NAN, 0.5, NAN),
                   ("S1-3", "S2-3", 16, NAN, 0.3, NAN))
    X = build_features(pairs, s1n, pooln, ("nofill",))
    assert list(X.columns) == FEATURE_COLUMNS["nofill"]
    assert X.iloc[0].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 2.0]    # "acme" both sides
    row = X.iloc[1]                                                # globex vs globex initech
    assert row["nofill_ratio"] == pytest.approx(fuzz.ratio("globex", "globex initech") / 100)
    assert row["nofill_token_set"] == 1.0 and row["nofill_jaccard"] == 0.5
    assert row["nofill_eq"] == row["fill_n_l"] == row["fill_n_r"] == 0
    empty = X.iloc[2]                                              # empty S1 name
    assert empty[["nofill_ratio", "nofill_token_set", "nofill_jaccard"]].isna().all()
    assert empty["nofill_eq"] == empty["fill_n_l"] == empty["fill_n_r"] == 0
    assert "nofill" not in DEFAULT_GROUPS
    with pytest.raises(ValueError, match="name_core_nofill"):
        build_features(pairs, s1n.drop(columns="name_core_nofill"), pooln, ("nofill",))


def test_tok_evidence_group_values():
    """Learned log-odds of each side's one-sided words: sum, extremes, strong and unknown."""
    ev = TokenEvidence(pool={"center": 2.0, "holdings": -3.0, "acme": -0.2},
                       s1={"group": 0.5, "holdings": -1.5}, strong=1.0)
    s1n = _records(_record("S1-1", "acme"), _record("S1-2", "hooli holdings group"))
    pooln = _records(_record("S2-1", "acme holdings center zork"), _record("S2-2", "acme"),
                     _record("S2-3", "hooli"))
    pairs = _pairs(("S1-1", "S2-1", 16, NAN, 0.5, NAN), ("S1-1", "S2-2", 1, NAN, NAN, NAN),
                   ("S1-2", "S2-3", 16, NAN, 0.3, NAN))
    X = build_features(pairs, s1n, pooln, ("tok_evidence",), evidence=ev)
    assert list(X.columns) == FEATURE_COLUMNS["tok_evidence"]
    pool = X.filter(like="te_pool").iloc[0].tolist()   # holdings -3, center +2, zork unknown
    assert pool == [-1.0, -3.0, 2.0, 1.0, 1.0, 1.0]
    assert X.iloc[0].filter(like="te_s1").tolist()[::3] == [0.0, 0.0]  # sum 0, decoy 0
    assert X.iloc[1][["te_pool_min", "te_s1_max"]].isna().all()       # nothing one-sided
    s1_side = X.filter(like="te_s1").iloc[2].tolist()  # holdings -1.5, group +0.5
    assert s1_side == [-1.0, -1.5, 0.5, 1.0, 0.0, 0.0]
    assert "tok_evidence" not in DEFAULT_GROUPS
    with pytest.raises(ValueError, match="evidence"):
        build_features(pairs, s1n, pooln, ("tok_evidence",))
