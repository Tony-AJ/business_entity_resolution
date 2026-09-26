"""Tests of the test-shaped mock fold (mock.py) and the 1-to-1 filter it relies on."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from entity_resolution import config as C
from entity_resolution.decision import DecisionRule, decide, one_to_one_filter
from entity_resolution.mock import Shape, build_mock, target_shape
from entity_resolution.split import Fold


def world(n_s1: dict[str, int], per_s1: int = 3, unowned: float = 0.5, seed: int = 0):
    """Train and val folds over countries: each S1 owns ``per_s1`` records, plus unowned ones.

    Every fifth S1 is on val; the val pool holds its entities' records and a fifth of the
    unowned ones, as ``split.load_fold`` would put them.
    """
    rng = np.random.default_rng(seed)
    s1, pool, pairs = [], [], []
    k = 0
    for country, n in n_s1.items():
        for i in range(n):
            sid = f"S1-{country}{i:06d}"
            s1.append((sid, country))
            for _ in range(int(rng.integers(0, per_s1 + 1))):
                pid = f"S{2 + k % 2}-{k:08d}"
                pool.append((pid, country, sid))
                pairs.append((sid, pid))
                k += 1
        for _ in range(int(n * unowned)):
            pool.append((f"S{2 + k % 2}-{k:08d}", country, ""))
            k += 1
    s1 = pd.DataFrame(s1, columns=[C.ENTITY_ID, C.COUNTRY])
    pool = pd.DataFrame(pool, columns=[C.ENTITY_ID, C.COUNTRY, "owner"])
    pairs = pd.DataFrame(pairs, columns=[C.S1_ID, C.ENTITY_ID])
    on_val = s1.index % 5 == 0
    val_ids = set(s1[C.ENTITY_ID][on_val])
    pool_val = pool["owner"].isin(val_ids) | ((pool["owner"] == "") & (pool.index % 5 == 0))

    def fold(name: str, s1_mask: np.ndarray, pool_mask: np.ndarray) -> Fold:
        p = pool[pool_mask].drop(columns="owner")
        ids = set(s1[C.ENTITY_ID][s1_mask])
        return Fold(name, s1[s1_mask].reset_index(drop=True),
                    p[p[C.ENTITY_ID].str.startswith("S2-")].reset_index(drop=True),
                    p[p[C.ENTITY_ID].str.startswith("S3-")].reset_index(drop=True),
                    pairs[pairs[C.S1_ID].isin(ids)].reset_index(drop=True))

    train = fold("train", ~on_val, ~pool_val.to_numpy())
    val = fold("val", on_val, pool_val.to_numpy())
    rest = train.s1[C.ENTITY_ID]
    tune = rest[rest.index % 4 == 1]            # a quarter of the train fold
    fit_sample = rest[rest.index % 4 == 2][::2]  # the matcher's training entities
    return train, val, tune, fit_sample


def pool_ids(fold: Fold) -> set[str]:
    return set(fold.s2[C.ENTITY_ID]) | set(fold.s3[C.ENTITY_ID])


def test_shape_matches_the_target_per_country() -> None:
    """A bigger country is cut to the test pool size, then to the test pool-per-S1 ratio."""
    train, val, tune, fit_sample = world({"US": 4000, "India": 2000})
    shape = {"US": Shape(s1=900, pool=5000), "India": Shape(s1=1000, pool=10**6)}
    mock = build_mock(train, val, tune, fit_sample, shape)
    info = mock.info.set_index(C.COUNTRY)
    assert info.loc["US", "keep_frac"] < 1.0
    assert abs(info.loc["US", "pool_kept"] - 5000) < 400            # hashed, so approximate
    assert info.loc["India", "keep_frac"] == 1.0                     # train smaller: whole
    for c in ("US", "India"):
        assert info.loc[c, "pool_per_s1"] == pytest.approx(shape[c].ratio, rel=0.02) or (
            info.loc[c, "present_fit"] == 0)                          # ran out of fit S1
    pool = pool_ids(mock.fold)
    assert set(mock.fold.pairs[C.ENTITY_ID]) <= pool                  # clusters stay whole
    assert set(mock.fold.pairs[C.S1_ID]) <= set(mock.fold.s1[C.ENTITY_ID])


def test_roles_and_drops() -> None:
    """Training entities never compete; val and tune entities of kept clusters always do."""
    train, val, tune, fit_sample = world({"US": 3000})
    shape = {"US": Shape(s1=500, pool=4000)}
    mock = build_mock(train, val, tune, fit_sample, shape)
    present = set(mock.fold.s1[C.ENTITY_ID])
    assert not present & set(fit_sample)
    assert set(mock.ids("val")) <= set(val.s1[C.ENTITY_ID])
    assert set(mock.ids("tune")) <= set(tune)
    kept_val = {s for s in val.s1[C.ENTITY_ID]
                if set(val.pairs[C.ENTITY_ID][val.pairs[C.S1_ID] == s]) & pool_ids(mock.fold)}
    assert kept_val <= present                                        # never dropped
    part = mock.part("val")
    assert set(part.s1[C.ENTITY_ID]) == set(mock.ids("val"))
    assert set(part.pairs[C.S1_ID]) <= set(part.s1[C.ENTITY_ID])
    # a dropped entity's records stay as unowned decoys
    dropped = set(train.s1[C.ENTITY_ID]) - present
    orphan = set(train.pairs[C.ENTITY_ID][train.pairs[C.S1_ID].isin(dropped)])
    assert orphan & pool_ids(mock.fold)
    assert not orphan & set(mock.fold.pairs[C.ENTITY_ID])


def test_order_free_and_unknown_country() -> None:
    """Row order cannot change the fold; a country absent from the shape is kept whole."""
    train, val, tune, fit_sample = world({"US": 1500, "Mars": 300})
    shape = {"US": Shape(s1=400, pool=2500)}
    a = build_mock(train, val, tune, fit_sample, shape)
    shuffled = Fold("train", train.s1.sample(frac=1, random_state=1), train.s2.iloc[::-1],
                    train.s3.sample(frac=1, random_state=2), train.pairs.iloc[::-1])
    b = build_mock(shuffled, val, tune, fit_sample, shape)
    assert set(a.fold.s1[C.ENTITY_ID]) == set(b.fold.s1[C.ENTITY_ID])
    assert pool_ids(a.fold) == pool_ids(b.fold)
    mars = a.info.set_index(C.COUNTRY).loc["Mars"]
    assert mars["keep_frac"] == 1.0
    n_mars = (train.s1[C.COUNTRY] == "Mars").sum() + (val.s1[C.COUNTRY] == "Mars").sum()
    n_mars_first = train.s1[C.COUNTRY][train.s1[C.ENTITY_ID].isin(set(fit_sample))].eq("Mars")
    assert mars["s1_present"] == n_mars - n_mars_first.sum()


def test_rejects_bad_input() -> None:
    train, val, tune, fit_sample = world({"US": 200})
    with pytest.raises(ValueError, match="drop_first"):
        build_mock(train, val, tune, val.s1[C.ENTITY_ID], {})
    no_country = Fold("train", train.s1[[C.ENTITY_ID]], train.s2, train.s3, train.pairs)
    with pytest.raises(ValueError, match="country"):
        build_mock(no_country, val, tune, fit_sample, {})


def test_target_shape_counts_the_test_split(dataset_dir: Path) -> None:
    shape = target_shape(dataset_dir)
    assert shape["France"] == Shape(1, 1)
    assert shape["India"] == Shape(1, 1)
    assert shape["US"] == Shape(0, 1)


def test_one_to_one_filter_then_subset_equals_whole_decision() -> None:
    """Filtering the partition, then deciding a subset, gives the subset's whole-run matches."""
    rng = np.random.default_rng(3)
    n = 3000
    scored = pd.DataFrame({
        C.S1_ID: [f"S1-{i:04d}" for i in rng.integers(0, 300, n)],
        C.ENTITY_ID: [f"S2-{i:04d}" for i in rng.integers(0, 500, n)],
        "prob": rng.random(n).round(2).astype(np.float32),   # rounding makes ties
    }).drop_duplicates([C.S1_ID, C.ENTITY_ID]).reset_index(drop=True)
    rule = DecisionRule(0.3, 0.5, 0.4, 3, True)
    whole = decide(scored, rule)
    kept = one_to_one_filter(scored)
    assert not kept[C.ENTITY_ID].duplicated().any()
    assert kept.index.is_monotonic_increasing
    subset = sorted(set(scored[C.S1_ID]))[::3]
    part = decide(kept[kept[C.S1_ID].isin(subset)], rule)
    expect = whole[whole[C.S1_ID].isin(subset)].reset_index(drop=True)
    pd.testing.assert_frame_equal(part, expect)
