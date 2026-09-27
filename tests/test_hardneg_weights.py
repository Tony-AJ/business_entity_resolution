"""hardneg.py: fit weights, HardNegConfig and weight_fn (docs/plan/09 §4, §9, weight-based).

Pure unit tests on the synthetic frames of test_hardneg.py (imported from it, as
test_snapshot.py imports snapshot_fixtures). No dataset, snapshot or model fit.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from entity_resolution import config as C
from entity_resolution.hardneg import (
    HN_KINDS,
    LABEL,
    HardNegConfig,
    easy_keep,
    fit_weights,
    is_easy_negative,
    singleton_mask,
    weight_fn,
)
from test_hardneg import build_easy_frame, make, rule_frame, truth_of


@pytest.fixture(scope="module")
def easy_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """``test_hardneg.build_easy_frame()``, built once per module."""
    return build_easy_frame()


# ---------------------------------------------------------------- fit_weights

def test_default_config_changes_nothing(easy_frame) -> None:
    """``HardNegConfig()`` gives every row weight 1 and computes no easy counts."""
    meta, X = easy_frame
    weight, counts = fit_weights(meta, X, HardNegConfig())
    assert weight.dtype == np.float64 and (weight == 1.0).all()
    assert "easy" not in counts and "fps" not in counts
    assert counts["kept_rows"] == len(meta) and counts["weight_sum"] == float(len(meta))


def test_singleton_weight(easy_frame) -> None:
    """HN3: every candidate of a singleton S1 weighs ``singleton_weight``, the rest 1."""
    meta, X = easy_frame
    weight, counts = fit_weights(meta, X, HardNegConfig(singleton_weight=3.0))
    single = singleton_mask(meta)
    assert (weight[single] == 3.0).all() and (weight[~single] == 1.0).all()
    assert counts["singleton_decoys"] == 100


def test_singleton_weight_uses_truth() -> None:
    """With ``truth`` an entity whose true pair was lost by blocking is not a singleton."""
    meta, X = make([("A", "P1", 1), ("B", "P2", 0), ("C", "P3", 0)])
    truth = pd.DataFrame({C.S1_ID: ["A", "C"], C.ENTITY_ID: ["P1", "P9"]}, dtype="str")
    cfg = HardNegConfig(singleton_weight=2.0)
    assert fit_weights(meta, X, cfg)[0].tolist() == [1.0, 2.0, 2.0]
    assert fit_weights(meta, X, cfg, truth=truth)[0].tolist() == [1.0, 2.0, 1.0]


def test_rule_kinds_weight() -> None:
    """Rows of ``rule_kinds`` weigh ``rule_weight``, the others 1; counts per kind."""
    meta, X = rule_frame()
    cfg = HardNegConfig(rule_kinds=("same_name_diff_addr", "top_cosine_nonmatch"),
                        rule_weight=2.5)
    weight, counts = fit_weights(meta, X, cfg)
    assert weight.tolist() == [1.0, 2.5, 1.0, 2.5, 1.0, 1.0, 1.0]
    assert counts["same_name_diff_addr"] == 1 and counts["top_cosine_nonmatch"] == 1
    assert "same_addr_diff_name" not in counts


def test_rule_and_singleton_weights_take_the_max() -> None:
    """A row hit by several boosts gets the largest, not the product or sum."""
    meta, X = rule_frame()
    cfg = HardNegConfig(rule_kinds=("singleton_decoy",), rule_weight=2.0, singleton_weight=3.0)
    weight, _ = fit_weights(meta, X, cfg)
    assert weight.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0]


def test_fp_and_fn_masks_add_extra_weight() -> None:
    """Round 2: an FP row gains ``fp_weight - 1``, a near miss ``near_miss_weight - 1``."""
    meta, X = rule_frame()
    fp = np.zeros(len(meta), dtype=bool)
    fp[[1, 5]] = True
    fn = np.zeros(len(meta), dtype=bool)
    fn[0] = True
    cfg = HardNegConfig(singleton_weight=2.0, fp_weight=3.0, near_miss_weight=2.0)
    weight, counts = fit_weights(meta, X, cfg, fp=fp, fn=fn)
    # row 5 is a singleton decoy (2) and an FP (+2)
    assert weight.tolist() == [2.0, 3.0, 1.0, 1.0, 1.0, 4.0, 2.0]
    assert counts["fps"] == 2 and counts["fns"] == 1


def test_dropped_easy_fp_comes_back_at_fp_weight_minus_one(easy_frame) -> None:
    """09 §7 adds copies to the round-1 set: a dropped easy FP ends at ``fp_weight - 1``."""
    meta, X = easy_frame
    y = meta[LABEL].to_numpy()
    easy = is_easy_negative(X, y, singleton_mask(meta))
    fp = np.zeros(len(meta), dtype=bool)
    fp[np.flatnonzero(easy)[:3]] = True
    weight, _ = fit_weights(meta, X, HardNegConfig(easy_neg_keep=0.0, fp_weight=3.0), fp=fp)
    assert (weight[fp] == 2.0).all()


@pytest.mark.parametrize("which", ["fp", "fn"])
def test_wrong_shape_mask_raises(which: str) -> None:
    """A round-2 mask that is not one bool per fit row is rejected, naming the mask."""
    meta, X = rule_frame()
    bad = np.zeros(len(meta) - 1, dtype=bool)
    with pytest.raises(ValueError, match=f"{which}s mask"):
        fit_weights(meta, X, HardNegConfig(), **{which: bad})


def test_meta_and_x_length_mismatch_raises() -> None:
    """``meta`` and ``X`` must describe the same rows."""
    meta, X = rule_frame()
    with pytest.raises(ValueError, match="rows"):
        fit_weights(meta, X.iloc[:-1], HardNegConfig())


def test_counts_consistent(easy_frame) -> None:
    """Every ``hn_counts`` key agrees with the weights and the masks it summarises."""
    meta, X = easy_frame
    y = meta[LABEL].to_numpy()
    rng = np.random.default_rng(5)
    fp = (y == 0) & (rng.random(len(meta)) < 0.05)
    fn = (y == 1) & (rng.random(len(meta)) < 0.2)
    cfg = HardNegConfig(easy_neg_keep=0.2, singleton_weight=2.0, rule_kinds=("singleton_decoy",),
                        rule_weight=1.5)
    weight, counts = fit_weights(meta, X, cfg, fp=fp, fn=fn)
    single = singleton_mask(meta)
    easy = is_easy_negative(X, y, single)
    assert {"rows", "positives", "singleton_decoys", "easy", "easy_dropped", "fps", "fns",
            "kept_rows", "weight_sum", "singleton_decoy"} <= set(counts)
    assert counts["rows"] == len(meta)
    assert counts["positives"] == int(y.sum()) == 100
    assert counts["singleton_decoys"] == int((single & (y == 0)).sum())
    assert counts["easy"] == int(easy.sum())
    assert counts["easy_dropped"] == int((easy & ~easy_keep(meta, 0.2, cfg.seed)).sum())
    assert counts["fps"] == int(fp.sum()) and counts["fns"] == int(fn.sum())
    assert counts["kept_rows"] == int((weight > 0).sum())
    assert counts["weight_sum"] == pytest.approx(float(weight.sum()))
    json.dumps(counts)                                  # metrics.json-ready: plain ints / floats


def test_fit_weights_deterministic(easy_frame) -> None:
    """Two calls with the same inputs give identical weights and counts."""
    meta, X = easy_frame
    cfg = HardNegConfig(easy_neg_keep=0.3, singleton_weight=2.0)
    w1, c1 = fit_weights(meta, X, cfg)
    w2, c2 = fit_weights(meta, X, cfg)
    np.testing.assert_array_equal(w1, w2)
    assert c1 == c2


def test_fit_weights_seed_changes_easy_sample_only(easy_frame) -> None:
    """Another ``seed`` moves which easy negatives drop; nothing else changes."""
    meta, X = easy_frame
    y = meta[LABEL].to_numpy()
    easy = is_easy_negative(X, y, singleton_mask(meta))
    w7, _ = fit_weights(meta, X, HardNegConfig(easy_neg_keep=0.2, seed=7))
    w8, _ = fit_weights(meta, X, HardNegConfig(easy_neg_keep=0.2, seed=8))
    assert not np.array_equal(w7[easy], w8[easy])
    np.testing.assert_array_equal(w7[~easy], w8[~easy])


# ---------------------------------------------------------------- config and weight_fn

@pytest.mark.parametrize("kwargs", [
    {"easy_neg_keep": -0.1}, {"easy_neg_keep": 1.1}, {"easy_neg_keep": float("nan")},
    {"singleton_weight": 0.5}, {"rule_weight": 0.0}, {"fp_weight": 0.9},
    {"near_miss_weight": 0.5},
    {"round2_threshold": 0.0}, {"round2_threshold": 1.0}, {"round2_threshold": 1.5},
    {"rule_kinds": ("same_name_diff_addr", "bogus")},
])
def test_config_rejects_bad_settings(kwargs: dict) -> None:
    """Settings that would silently mean something else raise at construction."""
    with pytest.raises(ValueError):
        HardNegConfig(**kwargs)


def test_config_record_is_json_and_round_trips() -> None:
    """``record()`` serialises to JSON, and a list of kinds is stored as a tuple."""
    cfg = HardNegConfig(easy_neg_keep=0.2, singleton_weight=3.0, rule_kinds=list(HN_KINDS),
                        round2=True, round2_threshold=0.8, fp_weight=1.0)
    assert cfg.rule_kinds == HN_KINDS
    rec = json.loads(json.dumps(cfg.record()))
    assert rec["easy_neg_keep"] == 0.2 and rec["rule_kinds"] == list(HN_KINDS)
    assert HardNegConfig(**rec) == cfg
    hash(cfg)                                           # frozen and hashable after the tuple


def test_weight_fn_matches_fit_weights(easy_frame) -> None:
    """The ``evaluate_params`` wrapper returns ``fit_weights``' weight and fills ``counts``."""
    meta, X = easy_frame
    y = meta[LABEL].to_numpy()
    fp = (y == 0) & (np.arange(len(meta)) % 50 == 0)
    cfg = HardNegConfig(easy_neg_keep=0.2, singleton_weight=2.0)
    truth = truth_of(meta)
    expected, expected_counts = fit_weights(meta, X, cfg, truth, fp=fp)
    counts: dict = {}
    fn_ = weight_fn(cfg, truth, fp=fp, counts=counts)
    np.testing.assert_array_equal(fn_(meta, X), expected)
    assert counts == expected_counts
    np.testing.assert_array_equal(weight_fn(cfg)(meta, X), fit_weights(meta, X, cfg)[0])
