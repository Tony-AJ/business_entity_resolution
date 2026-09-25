"""trainset.py: inner split invariants (11 §12), S1 sampling and pair labels (07 §7)."""
import numpy as np
import pandas as pd
import pytest

from entity_resolution import config as C
from entity_resolution.split import Fold, hash_unit, load_fold
from entity_resolution.trainset import INNER_FRAC, INNER_SEED, inner_split, label_pairs, sample_s1

PARTS = ("s1", "s2", "s3")


def records(ids) -> pd.DataFrame:
    """Raw source records (SOURCE_COLUMNS, str) whose name and address derive from the id."""
    ids = list(ids)
    return pd.DataFrame({C.ENTITY_ID: ids, C.NAME: [f"name {i}" for i in ids],
                         C.ADDRESS: [f"addr {i}" for i in ids], C.COUNTRY: ["US"] * len(ids)},
                        dtype="str")


@pytest.fixture(scope="module")
def train() -> Fold:
    """4,000 S1 ids with 0-5 matches (5% singletons) and 30% unmatched pool records."""
    rng = np.random.default_rng(0)
    n_s1 = 4000
    n_match = rng.integers(1, 6, n_s1)
    n_match[rng.random(n_s1) < 0.05] = 0
    n_matched = int(n_match.sum())
    n_pool = n_matched + round(n_matched * 0.3 / 0.7)
    in_s2 = rng.random(n_pool) < 0.5
    pool_ids = np.array([f"S{2 if s2 else 3}-{j:07d}" for j, s2 in enumerate(in_s2)])
    s1_ids = np.array([f"S1-{i:07d}" for i in range(n_s1)])
    pairs = pd.DataFrame({C.S1_ID: np.repeat(s1_ids, n_match), C.ENTITY_ID: pool_ids[:n_matched]},
                         dtype="str")
    return Fold("train", records(s1_ids), records(pool_ids[in_s2]), records(pool_ids[~in_s2]),
                pairs)


def ids(fold: Fold, part: str) -> set[str]:
    """Entity ids of one part ("s1", "s2" or "s3") of a fold."""
    return set(getattr(fold, part)[C.ENTITY_ID])


def pool(fold: Fold) -> set[str]:
    """Entity ids of the fold's pool (S2 and S3)."""
    return ids(fold, "s2") | ids(fold, "s3")


def test_inner_split_sides_are_disjoint(train):
    fit, tune = inner_split(train)
    assert (fit.name, tune.name) == ("fit", "tune")
    for part in PARTS:
        assert not ids(fit, part) & ids(tune, part)
        assert ids(fit, part) | ids(tune, part) == ids(train, part)
        assert len(getattr(fit, part)) + len(getattr(tune, part)) == len(getattr(train, part))
    assert len(fit.pairs) + len(tune.pairs) == len(train.pairs)


def test_inner_split_pairs_never_cross_sides(train):
    for side in inner_split(train):
        assert set(side.pairs[C.S1_ID]) <= ids(side, "s1")
        assert set(side.pairs[C.ENTITY_ID]) <= pool(side)


def test_inner_split_matched_pool_follows_owner(train):
    fit, tune = inner_split(train)
    tune_s1 = ids(tune, "s1")
    owner_on_tune = train.pairs[train.pairs[C.S1_ID].isin(tune_s1)][C.ENTITY_ID]
    own_hash_says_fit = owner_on_tune[hash_unit(owner_on_tune, INNER_SEED) >= INNER_FRAC]
    assert len(own_hash_says_fit) > 0                         # the case exists ...
    assert set(own_hash_says_fit) <= pool(tune)               # ... and follows its owner
    owner_on_fit = train.pairs[~train.pairs[C.S1_ID].isin(tune_s1)][C.ENTITY_ID]
    own_hash_says_tune = owner_on_fit[hash_unit(owner_on_fit, INNER_SEED) < INNER_FRAC]
    assert len(own_hash_says_tune) > 0 and set(own_hash_says_tune) <= pool(fit)


def test_inner_split_is_deterministic(train):
    fit, tune = inner_split(train)
    again = inner_split(train)
    for a, b in zip((fit, tune), again, strict=True):
        for part in (*PARTS, "pairs"):
            assert getattr(a, part).equals(getattr(b, part))
    shuffled = Fold("train", *(getattr(train, p).sample(frac=1, random_state=1)
                               for p in (*PARTS, "pairs")))
    for a, b in zip((fit, tune), inner_split(shuffled), strict=True):
        for part in PARTS:
            assert ids(a, part) == ids(b, part)


def test_inner_split_share(train):
    fit, tune = inner_split(train)
    assert len(tune.s1) / len(train.s1) == pytest.approx(INNER_FRAC, abs=0.02)
    matched = set(train.pairs[C.ENTITY_ID])
    unmatched = pool(train) - matched
    assert len(unmatched & pool(tune)) / len(unmatched) == pytest.approx(INNER_FRAC, abs=0.03)
    assert len(fit.s1) > len(tune.s1)


def test_inner_split_on_fixture(dataset_dir):
    train = load_fold("train", dataset_dir, frac=0.5)       # S1-00002 + S2-00002 (12 §1)
    fit, tune = inner_split(train)
    for side in (fit, tune):
        for part in PARTS:
            assert list(getattr(side, part).columns) == list(C.SOURCE_COLUMNS)
    assert all(len(getattr(fit, p)) == 0 for p in (*PARTS, "pairs"))    # fit side is empty
    assert ids(tune, "s1") == {"S1-00002"} and pool(tune) == {"S2-00002"}
    assert tune.pairs[C.ENTITY_ID].tolist() == ["S2-00002"]


def test_label_pairs():
    truth = pd.DataFrame({C.S1_ID: ["S1-a", "S1-a", "S1-b", "S1-c"],
                          C.ENTITY_ID: ["S2-1", "S3-1", "S2-2", "S3-9"]}, dtype="str")
    pairs = pd.DataFrame({C.S1_ID: ["S1-b", "S1-a", "S1-a", "S1-b", "S1-z", "S1-a"],
                          C.ENTITY_ID: ["S2-2", "S3-1", "S2-2", "S3-1", "S2-1", "S3-1"]},
                         dtype="str", index=[10, 3, 7, 1, 5, 0])
    pairs["pass"] = np.uint8([1, 2, 4, 8, 1, 2])           # other columns ride along untouched
    out = label_pairs(pairs, truth)
    assert out["label"].dtype == np.int8
    # S1-a/S2-2 and S1-b/S3-1 combine ids of true pairs but are not pairs; S1-c/S3-9 is absent
    assert out["label"].tolist() == [1, 1, 0, 0, 0, 1]
    assert out.index.equals(pairs.index)
    pd.testing.assert_frame_equal(out.drop(columns="label"), pairs)
    assert len(out) == len(pairs)
    assert label_pairs(pairs, truth.iloc[:0])["label"].sum() == 0
    assert label_pairs(pairs.iloc[:0], truth).empty


def test_sample_s1_order_independent():
    s1 = records([f"S1-{i:05d}" for i in range(2000)])
    s1.index = s1.index + 100
    sample = sample_s1(s1, 500)
    shuffled = sample_s1(s1.sample(frac=1, random_state=0), 500)
    assert set(sample[C.ENTITY_ID]) == set(shuffled[C.ENTITY_ID])
    assert sample.index.equals(pd.RangeIndex(len(sample)))
    assert sample[C.ENTITY_ID].is_monotonic_increasing        # original row order kept
    assert len(sample) == pytest.approx(500, abs=4 * np.sqrt(500))
    assert set(sample[C.ENTITY_ID]) <= set(sample_s1(s1, 800)[C.ENTITY_ID])   # nested in n
    assert set(sample[C.ENTITY_ID]) != set(sample_s1(s1, 500, seed=8)[C.ENTITY_ID])
    whole = sample_s1(s1, 5000)
    assert whole.equals(s1.reset_index(drop=True))
    assert sample_s1(s1, 0).empty
    with pytest.raises(ValueError):
        sample_s1(s1, -1)
