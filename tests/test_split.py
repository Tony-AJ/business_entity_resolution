import numpy as np
import pandas as pd
import pytest

from entity_resolution import config as C
from entity_resolution import split
from entity_resolution.split import hash_unit, load_fold, pool_val_mask, s1_val_mask


def test_hash_is_deterministic_and_order_independent():
    ids = pd.Series([f"S1-{i}" for i in range(1000)])
    u = hash_unit(ids)
    shuffled = ids.sample(frac=1, random_state=0)
    assert np.array_equal(hash_unit(shuffled), u[shuffled.index])
    assert ((0 <= u) & (u < 1)).all()
    assert not np.array_equal(hash_unit(ids, seed=7), u)  # seed changes the split


def test_validation_share_matches_fraction():
    ids = pd.Series([f"S1-{i}" for i in range(20_000)])
    assert s1_val_mask(ids, frac=0.2).mean() == pytest.approx(0.2, abs=0.01)


def test_matched_records_follow_their_source1_fold():
    pairs = pd.DataFrame({C.S1_ID: ["S1-a", "S1-b"], C.ENTITY_ID: ["S2-a", "S2-b"]})
    pool = pd.Series(["S2-a", "S2-b"])
    assert pool_val_mask(pool, pairs, {"S1-a"}).tolist() == [True, False]
    assert pool_val_mask(pool, pairs, {"S1-b"}).tolist() == [False, True]


def test_folds_partition_train_data(dataset_dir):
    val, train = (load_fold(f, dataset_dir, frac=0.5) for f in ("val", "train"))
    for part in ("s1", "s2", "s3"):
        a, b = set(getattr(val, part)[C.ENTITY_ID]), set(getattr(train, part)[C.ENTITY_ID])
        assert not a & b
    assert len(val.s1) + len(train.s1) == 3
    for fold in (val, train):
        s1 = set(fold.s1[C.ENTITY_ID])
        pool = set(fold.s2[C.ENTITY_ID]) | set(fold.s3[C.ENTITY_ID])
        assert set(fold.pairs[C.S1_ID]) <= s1  # labels never cross folds
        assert set(fold.pairs[C.ENTITY_ID]) <= pool
        assert set(fold.truth()) == s1  # singletons included


def test_fold_membership_is_cached(dataset_dir, monkeypatch):
    first = load_fold("val", dataset_dir, frac=0.5)
    monkeypatch.setattr(split, "pool_val_mask", lambda *a, **k: pytest.fail("recomputed"))
    again = load_fold("val", dataset_dir, frac=0.5)
    assert again.s1.equals(first.s1) and again.pairs.equals(first.pairs)


def test_columns_are_limited(dataset_dir):
    fold = load_fold("train", dataset_dir, columns=[C.NAME], frac=0.5)
    assert list(fold.s1.columns) == [C.ENTITY_ID, C.NAME]
    assert fold.summary()["fold"] == "train"
