"""hardneg.py: hard-negative mining, rule kinds and the easy-negative cap (docs/plan/09 §3,
§4, §9). Fit weights, the config and weight_fn: test_hardneg_weights.py, which reuses the
frame builders defined here.

Pure unit tests on small synthetic frames built here: ``meta`` (ids, country, pass, label),
``X`` (the float32 feature columns the rules read), ``scored`` and truth pairs. No dataset,
no snapshot and no model fit; the round-2 loop is left to the snapshot tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from entity_resolution import config as C
from entity_resolution.hardneg import (
    LABEL,
    HardNegConfig,
    easy_keep,
    fit_weights,
    is_easy_negative,
    mine_by_rule,
    mine_false_positives,
    mine_near_misses,
    singleton_mask,
)

# Feature values that trigger no rule and are not easy: every test overrides only what it needs
NEUTRAL = {"core_ratio": 0.5, "ad_jaccard": 0.5, "ad_token_set": 0.5, "core_token_set": 0.8,
           "ctx_rank_name": 2.0, "sim_name_char": 0.8}
EASY = {"sim_name_char": 0.2, "ad_jaccard": 0.0}       # far names and far addresses (09 §4)
HARD_POS = {"sim_name_char": 0.9, "ad_jaccard": 0.8}   # a positive no rule calls easy


def make(rows: list[tuple]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(meta, X)`` from ``(s1 id, pool id, label[, feature overrides])`` tuples.

    ``meta`` carries the snapshot's columns (``pass`` uint8, ``label`` int8); ``X`` is float32
    with the ``NEUTRAL`` features plus any override column (e.g. ``core_eq``), same index.
    """
    feats = [{**NEUTRAL, **(r[3] if len(r) > 3 else {})} for r in rows]
    meta = pd.DataFrame({C.S1_ID: [r[0] for r in rows], C.ENTITY_ID: [r[1] for r in rows],
                         C.COUNTRY: ["US"] * len(rows)}, dtype="str")
    meta["pass"] = np.ones(len(rows), dtype=np.uint8)
    meta[LABEL] = np.array([r[2] for r in rows], dtype=np.int8)
    X = pd.DataFrame(feats, index=meta.index).astype(np.float32)
    return meta, X


def truth_of(meta: pd.DataFrame) -> pd.DataFrame:
    """The labelled-positive pairs of ``meta`` as a truth-pairs frame."""
    return meta.loc[meta[LABEL] == 1, [C.S1_ID, C.ENTITY_ID]].reset_index(drop=True)


def rule_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per 09 §3 kind plus a positive that looks like every kind at once.

    Rows: 0 positive of A hitting every rule; 1 A same-name / different-address; 2 A
    same-address / different-name; 3 A rank-1 non-match; 4 A neutral; 5, 6 singleton B.
    """
    every_rule = {"core_ratio": 1.0, "ad_jaccard": 0.0, "ad_token_set": 1.0,
                  "core_token_set": 0.0, "ctx_rank_name": 1.0}
    return make([
        ("A", "P0", 1, every_rule),
        ("A", "P1", 0, {"core_ratio": 1.0, "ad_jaccard": 0.1}),
        ("A", "P2", 0, {"ad_token_set": 0.95, "core_token_set": 0.3}),
        ("A", "P3", 0, {"ctx_rank_name": 1.0}),
        ("A", "P4", 0),
        ("B", "P5", 0),
        ("B", "P6", 0),
    ])


def build_easy_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """100 matched S1 with one positive and 10 easy negatives each (1,000 easy), plus 20
    singleton S1 with 5 easy-looking candidates each (100 decoys, exempt from the cap).
    Also used by test_hardneg_weights.py."""
    rows: list[tuple] = []
    for i in range(100):
        rows.append((f"M{i:03d}", f"Q{i:03d}-pos", 1, HARD_POS))
        rows += [(f"M{i:03d}", f"Q{i:03d}-{j}", 0, EASY) for j in range(10)]
    for i in range(20):
        rows += [(f"S{i:03d}", f"R{i:03d}-{j}", 0, EASY) for j in range(5)]
    return make(rows)


@pytest.fixture(scope="module")
def easy_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """``build_easy_frame()``, built once per module."""
    return build_easy_frame()


def scored_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(scored, truth)``: six scored pairs on a non-default index, three of them true."""
    scored = pd.DataFrame({C.S1_ID: ["A", "A", "A", "B", "B", "C"],
                           C.ENTITY_ID: ["P1", "P2", "P3", "P4", "P5", "P6"],
                           "prob": np.array([0.9, 0.5, 0.2, 0.7, 0.49, 0.1], dtype=np.float32)},
                          index=[10, 11, 12, 13, 14, 15])
    scored[[C.S1_ID, C.ENTITY_ID]] = scored[[C.S1_ID, C.ENTITY_ID]].astype("str")
    truth = pd.DataFrame({C.S1_ID: ["A", "B", "B", "Z"], C.ENTITY_ID: ["P1", "P4", "P5", "P9"]},
                         dtype="str")
    return scored, truth


def pair_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    """The (s1 id, pool id) pairs of a frame as a set."""
    return set(zip(frame[C.S1_ID], frame[C.ENTITY_ID], strict=True))


# ---------------------------------------------------------------- model-mined rows (09 §3)

def test_mine_false_positives_label_zero_only() -> None:
    """FPs: not true pairs, prob >= threshold (inclusive), label 0 int8, index kept."""
    scored, truth = scored_frame()
    fp = mine_false_positives(scored, truth, threshold=0.5)
    assert list(fp.index) == [11]                     # A-P2 at exactly 0.5; A-P1, B-P4 are true
    assert set(fp.index) <= set(scored.index)
    assert not pair_set(fp) & pair_set(truth)
    assert (fp["prob"] >= 0.5).all()
    assert fp[LABEL].dtype == np.int8 and (fp[LABEL] == 0).all()
    assert list(mine_false_positives(scored, truth, threshold=0.05).index) == [11, 12, 15]


def test_mine_near_misses_label_one_only() -> None:
    """Near misses: true pairs with prob < threshold, label 1 int8, index kept."""
    scored, truth = scored_frame()
    fn = mine_near_misses(scored, truth, threshold=0.5)
    assert list(fn.index) == [14]                     # B-P5 at 0.49; B-P4 at 0.7 is caught
    assert pair_set(fn) <= pair_set(truth)
    assert (fn["prob"] < 0.5).all()
    assert fn[LABEL].dtype == np.int8 and (fn[LABEL] == 1).all()
    assert list(mine_near_misses(scored, truth, threshold=0.95).index) == [10, 13, 14]


def test_mined_frames_partition_true_and_false_pairs() -> None:
    """At one threshold no row is both a FP and a near miss, and the columns are kept."""
    scored, truth = scored_frame()
    for t in (0.1, 0.5, 0.9):
        fp, fn = mine_false_positives(scored, truth, t), mine_near_misses(scored, truth, t)
        assert not set(fp.index) & set(fn.index)
        assert list(fp.columns) == [*scored.columns, LABEL] == list(fn.columns)


# ---------------------------------------------------------------- rule kinds (09 §3)

@pytest.mark.parametrize(("kind", "expected"), [
    ("same_name_diff_addr", [1]),
    ("same_addr_diff_name", [2]),
    ("top_cosine_nonmatch", [3]),
    ("singleton_decoy", [5, 6]),
])
def test_mine_by_rule_kinds(kind: str, expected: list[int]) -> None:
    """Each kind selects exactly its rows; the all-rules positive (row 0) never."""
    meta, X = rule_frame()
    hit = mine_by_rule(meta, X, kind)
    assert hit.dtype == bool and hit.shape == (len(meta),)
    assert list(np.flatnonzero(hit)) == expected
    assert not hit[meta[LABEL].to_numpy() == 1].any()


def test_mine_by_rule_thresholds_and_nan() -> None:
    """Strict ``<`` at 0.2 / 0.5, ``>=`` near 0.9, and NaN never satisfies a rule."""
    meta, X = make([
        ("A", "P0", 1),
        ("A", "P1", 0, {"core_ratio": 1.0, "ad_jaccard": 0.2}),           # not < 0.2
        ("A", "P2", 0, {"ad_token_set": 0.91, "core_token_set": 0.49}),   # both hit
        ("A", "P3", 0, {"ad_token_set": 0.95, "core_token_set": 0.5}),    # not < 0.5
        ("A", "P4", 0, {"core_ratio": 1.0, "ad_jaccard": np.nan}),
        ("A", "P5", 0, {"ad_token_set": np.nan, "core_token_set": 0.0}),
        ("A", "P6", 0, {"ctx_rank_name": np.nan}),
    ])
    assert not mine_by_rule(meta, X, "same_name_diff_addr").any()
    assert list(np.flatnonzero(mine_by_rule(meta, X, "same_addr_diff_name"))) == [2]
    assert not mine_by_rule(meta, X, "top_cosine_nonmatch").any()


def test_same_addr_threshold_inclusive_on_float32() -> None:
    """09 §3 says ``ad_token_set >= 0.9``: a float32 feature of exactly 0.9 must be selected."""
    meta, X = make([("A", "P0", 1),
                    ("A", "P1", 0, {"ad_token_set": 0.9, "core_token_set": 0.0})])
    assert list(np.flatnonzero(mine_by_rule(meta, X, "same_addr_diff_name"))) == [1]


def test_same_name_uses_core_eq_when_present() -> None:
    """With ``core_eq`` the name test reads it and ignores ``core_ratio``; without, ratio >= 1."""
    rows = [("A", "P0", 1),
            ("A", "P1", 0, {"core_eq": 0.0, "core_ratio": 1.0, "ad_jaccard": 0.0}),
            ("A", "P2", 0, {"core_eq": 1.0, "core_ratio": 0.3, "ad_jaccard": 0.0})]
    meta, X = make(rows)
    assert list(np.flatnonzero(mine_by_rule(meta, X, "same_name_diff_addr"))) == [2]
    fallback = mine_by_rule(meta, X.drop(columns="core_eq"), "same_name_diff_addr")
    assert list(np.flatnonzero(fallback)) == [1]


def test_singleton_decoy_uses_given_mask() -> None:
    """An explicit ``singleton`` mask replaces the label fallback, still label-0 rows only."""
    meta, X = rule_frame()
    given = np.zeros(len(meta), dtype=bool)
    given[[0, 4]] = True                              # row 0 is a positive: never selected
    assert list(np.flatnonzero(mine_by_rule(meta, X, "singleton_decoy", given))) == [4]


def test_mine_by_rule_unknown_kind_raises() -> None:
    """A kind outside ``HN_KINDS`` is an error, not an empty mask."""
    meta, X = rule_frame()
    with pytest.raises(ValueError, match="unknown kind"):
        mine_by_rule(meta, X, "easy_negative")


@pytest.mark.parametrize(("kind", "drop", "missing"), [
    ("same_name_diff_addr", ["ad_jaccard"], "ad_jaccard"),
    ("same_name_diff_addr", ["core_ratio"], "core_ratio"),
    ("same_addr_diff_name", ["ad_token_set"], "ad_token_set"),
    ("same_addr_diff_name", ["core_token_set"], "core_token_set"),
    ("top_cosine_nonmatch", ["ctx_rank_name"], "ctx_rank_name"),
])
def test_mine_by_rule_missing_feature_raises(kind: str, drop: list[str], missing: str) -> None:
    """A snapshot without a feature a rule needs fails loudly and names the feature."""
    meta, X = rule_frame()
    with pytest.raises(ValueError, match=missing):
        mine_by_rule(meta, X.drop(columns=drop), kind)


def test_singleton_decoy_needs_no_feature() -> None:
    """``singleton_decoy`` reads labels only, so an empty feature frame is fine."""
    meta, X = rule_frame()
    empty = X.drop(columns=list(X.columns))
    assert list(np.flatnonzero(mine_by_rule(meta, empty, "singleton_decoy"))) == [5, 6]


# ---------------------------------------------------------------- singletons, easy negatives

def test_singleton_mask_with_and_without_truth() -> None:
    """Truth decides the metric's singleton; without it an entity with no label-1 row counts.

    C has a true pair that blocking lost (no label-1 candidate): a singleton by labels only.
    """
    meta, _ = make([("A", "P1", 1), ("A", "P2", 0), ("B", "P3", 0), ("C", "P4", 0),
                    ("C", "P5", 0)])
    truth = pd.DataFrame({C.S1_ID: ["A", "C"], C.ENTITY_ID: ["P1", "P9"]}, dtype="str")
    assert singleton_mask(meta, truth).tolist() == [False, False, True, False, False]
    assert singleton_mask(meta).tolist() == [False, False, True, True, True]


def test_is_easy_negative_rules() -> None:
    """Easy = label 0, far name and far address; NaN, singletons and positives never are."""
    meta, X = make([
        ("A", "P0", 1, EASY),                                       # positive
        ("A", "P1", 0, EASY),                                       # easy
        ("A", "P2", 0, {"sim_name_char": np.nan, "ad_jaccard": 0.0}),
        ("A", "P3", 0, {"sim_name_char": 0.2, "ad_jaccard": np.nan}),
        ("A", "P4", 0, {"sim_name_char": 0.5, "ad_jaccard": 0.0}),  # not < 0.5
        ("A", "P5", 0, {"sim_name_char": 0.2, "ad_jaccard": 0.1}),  # not < 0.1
        ("B", "P6", 0, EASY),                                       # singleton decoy
    ])
    y = meta[LABEL].to_numpy()
    easy = is_easy_negative(X, y, singleton_mask(meta))
    assert easy.tolist() == [False, True, False, False, False, False, False]


def test_easy_negative_cap_ratio(easy_frame) -> None:
    """HN2: of 1,000 easy negatives 150-250 survive at 0.2; positives and decoys keep 1."""
    meta, X = easy_frame
    weight, counts = fit_weights(meta, X, HardNegConfig(easy_neg_keep=0.2))
    y = meta[LABEL].to_numpy()
    single = singleton_mask(meta)
    easy = is_easy_negative(X, y, single)
    assert counts["easy"] == int(easy.sum()) == 1000
    kept = int((weight[easy] > 0).sum())
    assert 150 <= kept <= 250
    assert counts["easy_dropped"] == 1000 - kept == int((weight == 0).sum())
    assert (weight[y == 1] == 1).all() and (weight[single] == 1).all()
    assert set(np.unique(weight)) <= {0.0, 1.0}


def test_singleton_decoys_never_dropped(easy_frame) -> None:
    """``easy_neg_keep=0`` drops every easy negative but no candidate of a singleton."""
    meta, X = easy_frame
    weight, counts = fit_weights(meta, X, HardNegConfig(easy_neg_keep=0.0))
    single = singleton_mask(meta)
    assert single.sum() == 100 and (weight[single] == 1).all()
    assert counts["easy_dropped"] == 1000
    assert counts["kept_rows"] == len(meta) - 1000


def test_easy_keep_is_order_free(easy_frame) -> None:
    """The keep decision depends on the pair, not on row order or index."""
    meta, _ = easy_frame
    keep = dict(zip(zip(meta[C.S1_ID], meta[C.ENTITY_ID], strict=True),
                    easy_keep(meta, 0.2, seed=7), strict=True))
    shuffled = meta.sample(frac=1.0, random_state=3)
    for k, pair in zip(easy_keep(shuffled, 0.2, seed=7),
                       zip(shuffled[C.S1_ID], shuffled[C.ENTITY_ID], strict=True), strict=True):
        assert keep[pair] == k


def test_easy_keep_seed_and_share(easy_frame) -> None:
    """Another seed draws another sample; share 1 keeps all, share 0 keeps none."""
    meta, _ = easy_frame
    assert not np.array_equal(easy_keep(meta, 0.2, seed=7), easy_keep(meta, 0.2, seed=8))
    assert easy_keep(meta, 1.0).all()
    assert not easy_keep(meta, 0.0).any()


def test_fit_weights_order_free(easy_frame) -> None:
    """Shuffling the fit rows gives every pair the same weight."""
    meta, X = easy_frame
    cfg = HardNegConfig(easy_neg_keep=0.2, singleton_weight=2.0)
    weight, _ = fit_weights(meta, X, cfg)
    order = np.random.default_rng(1).permutation(len(meta))
    w_shuf, _ = fit_weights(meta.iloc[order], X.iloc[order], cfg)
    np.testing.assert_array_equal(w_shuf, weight[order])
