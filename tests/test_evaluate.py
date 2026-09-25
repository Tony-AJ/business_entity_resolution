"""evaluate.py: the vectorised metric equals metrics.py; blocking report, slices, samples."""
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from entity_resolution import config as C
from entity_resolution import metrics
from entity_resolution.evaluate import (
    SAMPLE_COLUMNS,
    SLICE_COLUMNS,
    blocking_report,
    entity_counts,
    entity_f05_from_counts,
    error_samples,
    harder_fold,
    macro_f05_from_counts,
    pair_recall,
    pairs_to_lists,
    score_pairs,
    slice_report,
)
from entity_resolution.normalize import NORM_COLUMNS
from entity_resolution.split import Fold, load_fold


def records(ids, countries="US") -> pd.DataFrame:
    """Raw source records (SOURCE_COLUMNS, str) whose name and address derive from the id."""
    ids = list(ids)
    countries = [countries] * len(ids) if isinstance(countries, str) else list(countries)
    return pd.DataFrame({C.ENTITY_ID: ids, C.NAME: [f"name {i}" for i in ids],
                         C.ADDRESS: [f"addr {i}" for i in ids], C.COUNTRY: countries},
                        dtype="str")


def pairs_frame(rows) -> pd.DataFrame:
    """A (source1_entity_id, entity_id) pairs frame of str columns."""
    return pd.DataFrame(rows, columns=[C.S1_ID, C.ENTITY_ID], dtype="str")


def make_fold(s1_ids, pool_ids, truth_rows, countries="US") -> Fold:
    """A "val" fold: S1 records, pool records split into S2/S3 by prefix, truth pairs."""
    pool = pd.Series(pool_ids, dtype="str")
    return Fold("val", records(s1_ids, countries), records(pool[pool.str.startswith("S2-")]),
                records(pool[pool.str.startswith("S3-")]), pairs_frame(truth_rows))


def random_case(seed: int = 0, n_s1: int = 500) -> tuple[Fold, pd.DataFrame]:
    """A fold with 0-4 true matches per S1 (one owner per pool record, extra decoys) and a
    prediction with found and false pairs, duplicate rows, absent entities, S1 ids outside
    the fold, and a few pool ids that no fold record has (counted as false positives)."""
    rng = np.random.default_rng(seed)
    s1_ids = np.array([f"S1-{i:05d}" for i in range(n_s1)])
    n_true = rng.integers(0, 5, n_s1)
    t_s1 = np.repeat(np.arange(n_s1), n_true)
    n_pool = len(t_s1) + n_s1                       # the matched records plus decoys
    pool_ids = np.array([f"S{2 + j % 2}-{j:05d}" for j in range(n_pool)])
    fold = make_fold(s1_ids, pool_ids, zip(s1_ids[t_s1], pool_ids[:len(t_s1)], strict=True))
    found = rng.random(len(t_s1)) < 0.7
    fp_s1, fp_pool = rng.integers(0, n_s1, n_s1 // 2), rng.integers(0, n_pool, n_s1 // 2)
    p_s1 = np.r_[t_s1[found], fp_s1]
    p_pool = pool_ids[np.r_[np.arange(len(t_s1))[found], fp_pool]]
    dup = rng.integers(0, len(p_s1), 40)            # duplicate rows collapse (set semantics)
    p_s1, p_pool = np.r_[p_s1, p_s1[dup]], np.r_[p_pool, p_pool[dup]]
    present = ~(rng.random(n_s1) < 0.1)[p_s1]        # 10% of the entities predict nothing
    pred = pairs_frame(zip(s1_ids[p_s1][present], p_pool[present], strict=True))
    outside = pairs_frame([("S1-99999", pool_ids[0]), ("S1-99998", "S2-77777")])
    unknown = pairs_frame([(s1_ids[1], "S3-88888"), (s1_ids[2], "S2-88889")])
    pred = pd.concat([pred, outside, unknown], ignore_index=True).sample(frac=1, random_state=1)
    return fold, pred.reset_index(drop=True)


def reference(pred: pd.DataFrame, fold: Fold):
    """``(pred lists, truth lists)`` for the dict-based functions of metrics.py."""
    ids = fold.s1[C.ENTITY_ID]
    return pairs_to_lists(pred, ids), pairs_to_lists(fold.pairs, ids)


def assert_same(got: dict, want: dict) -> None:
    """Same keys, exactly equal values (NaN equal to NaN)."""
    assert list(got) == list(want)
    for key, value in want.items():
        assert got[key] == value or (math.isnan(got[key]) and math.isnan(value)), key


def normalised(ids, **columns) -> pd.DataFrame:
    """A normalised frame with NORM_COLUMNS: "" strings, non_latin False, addr_tokens 3,
    then the given columns."""
    n = len(ids)
    frame = {c: [""] * n for c in NORM_COLUMNS}
    frame.update({C.ENTITY_ID: list(ids), C.COUNTRY: ["US"] * n, "non_latin": [False] * n,
                  "addr_tokens": [3] * n})
    frame.update(columns)
    out = pd.DataFrame(frame)[NORM_COLUMNS]
    return out.astype({"non_latin": bool, "addr_tokens": np.int16})


def test_macro_f05_from_counts_equals_macro_fbeta():
    fold, pred = random_case()
    pred_lists, truth_lists = reference(pred, fold)
    assert sum(not v for v in truth_lists.values()) > 50          # singletons
    assert sum(not v for v in pred_lists.values()) > 50           # absent entities
    assert pred.duplicated().any()
    with pytest.warns(UserWarning, match="2 predicted pairs"):     # the two unknown pool ids
        got = score_pairs(pred, fold)
    assert_same(got, metrics.breakdown(pred_lists, truth_lists))
    with pytest.warns(UserWarning):
        tp, n_pred, n_true = entity_counts(pred, fold)
    assert macro_f05_from_counts(tp, n_pred, n_true) == metrics.macro_fbeta(pred_lists,
                                                                           truth_lists)


@pytest.mark.parametrize(("tp", "n_pred", "n_true", "f"),
                         [(0, 0, 0, 1.0), (0, 1, 0, 0.0), (0, 0, 2, 0.0), (2, 3, 2, 5 / 7),
                          (1, 1, 1, 1.0)])
def test_counts_edge_cases(tp, n_pred, n_true, f):
    assert entity_f05_from_counts([tp], [n_pred], [n_true])[0] == pytest.approx(f, abs=1e-15)


def test_entity_f05_from_counts_matches_entity_fbeta():
    fold, pred = random_case(seed=3)
    pred_lists, truth_lists = reference(pred, fold)
    with pytest.warns(UserWarning):
        tp, n_pred, n_true = entity_counts(pred, fold)
    want = [metrics.entity_fbeta(pred_lists[k], truth_lists[k]) for k in truth_lists]
    assert np.array_equal(entity_f05_from_counts(tp, n_pred, n_true), want)   # bitwise
    grid = [(t, p, n) for n in range(7) for p in range(7) for t in range(min(p, n) + 1)]
    tp, n_pred, n_true = (np.array(x) for x in zip(*grid, strict=True))
    want = [metrics.entity_fbeta({f"t{i}" for i in range(t)} | {f"f{i}" for i in range(p - t)},
                                 {f"t{i}" for i in range(n)}) for t, p, n in grid]
    assert np.array_equal(entity_f05_from_counts(tp, n_pred, n_true), want)
    with pytest.raises(ValueError):
        macro_f05_from_counts([], [], [])


def test_entity_counts_aligned_to_fold_rows():
    fold = make_fold(["S1-b", "S1-a", "S1-c"], ["S2-1", "S2-2", "S3-1"],
                     [("S1-a", "S2-1"), ("S1-a", "S3-1"), ("S1-b", "S2-2")])
    pred = pairs_frame([("S1-a", "S2-1"), ("S1-a", "S2-1"), ("S1-a", "S2-2"), ("S1-c", "S3-1")])
    tp, n_pred, n_true = entity_counts(pred, fold)
    assert (tp.tolist(), n_pred.tolist(), n_true.tolist()) == ([0, 1, 0], [0, 2, 1], [1, 2, 0])


def test_pairs_to_lists_every_s1_present():
    pairs = pairs_frame([("S1-b", "S2-1"), ("S1-b", "S2-1"), ("S1-b", "S3-2"), ("S1-x", "S2-9"),
                         ("S1-a", "S3-1")])
    out = pairs_to_lists(pairs, pd.Series(["S1-c", "S1-b", "S1-a"]))
    assert list(out) == ["S1-c", "S1-b", "S1-a"]
    assert out == {"S1-c": [], "S1-b": ["S2-1", "S3-2"], "S1-a": ["S3-1"]}


def test_pair_recall():
    truth = pairs_frame([("S1-a", "S2-1"), ("S1-a", "S3-1"), ("S1-b", "S2-2")])
    cand = pairs_frame([("S1-a", "S2-1"), ("S1-a", "S2-1"), ("S1-b", "S2-2"),
                        ("S1-b", "S3-1"), ("S1-c", "S2-9")])   # S1-b/S3-1: known ids, no pair
    assert pair_recall(cand, truth) == 2 / 3
    assert math.isnan(pair_recall(cand, truth.iloc[:0]))


def test_blocking_report_matches_candidate_report(dataset_dir):
    val = load_fold("val", dataset_dir, frac=0.5)   # S1-00001 (2 matches), S1-00003 singleton
    cand = pairs_frame([("S1-00001", "S2-00001"), ("S1-00001", "S3-00002"),
                        ("S1-00003", "S3-00001")])
    cand_lists, truth_lists = reference(cand, val)
    assert_same(blocking_report(cand, val),
                metrics.candidate_report(cand_lists, truth_lists, pool_size=3))
    fold, pred = random_case(seed=5)
    cand_lists, truth_lists = reference(pred, fold)
    with pytest.warns(UserWarning):
        got = blocking_report(pred, fold)
    assert_same(got, metrics.candidate_report(cand_lists, truth_lists,
                                              pool_size=len(fold.s2) + len(fold.s3)))


def test_harder_fold_keeps_pool_drops_s1():
    fold, _ = random_case(seed=2, n_s1=300)
    h = harder_fold(fold)
    assert h.name == "harder"
    assert 0 < len(h.s1) < len(fold.s1)
    assert h.s2.equals(fold.s2) and h.s3.equals(fold.s3)
    kept = set(h.s1[C.ENTITY_ID])
    assert set(h.pairs[C.S1_ID]) <= kept
    assert len(h.pairs) == fold.pairs[C.S1_ID].isin(kept).sum()   # kept entities keep all pairs
    again = harder_fold(fold)
    assert again.s1.equals(h.s1) and again.pairs.equals(h.pairs)
    shuffled = Fold("val", fold.s1.sample(frac=1, random_state=0), fold.s2, fold.s3, fold.pairs)
    assert set(harder_fold(shuffled).s1[C.ENTITY_ID]) == kept


def test_harder_fold_share():
    s1 = records([f"S1-{i:05d}" for i in range(4000)])
    fold = Fold("val", s1, records([]), records([]), pairs_frame([]))
    assert 1 - len(harder_fold(fold).s1) / 4000 == pytest.approx(0.2, abs=0.02)


def slice_case(seed: int = 7):
    """random_case plus normalised frames with varied countries, names, postcodes, addresses."""
    rng = np.random.default_rng(seed)
    fold, pred = random_case(seed)
    s1_ids, pool_ids = fold.s1[C.ENTITY_ID], pd.concat([fold.s2[C.ENTITY_ID], fold.s3[C.ENTITY_ID]])
    s1n = normalised(s1_ids, country=rng.choice(["US", "India", "France"], len(s1_ids)),
                     name_core=rng.choice(["", "acme", "globex", "initech", "umbrella"] +
                                          [f"solo {i}" for i in range(400)], len(s1_ids)),
                     postcode=rng.choice(["", "411001"], len(s1_ids)),
                     addr_tokens=rng.integers(0, 3, len(s1_ids)))
    pooln = normalised(pool_ids, non_latin=rng.random(len(pool_ids)) < 0.2,
                       addr_tokens=rng.integers(0, 4, len(pool_ids)))
    return fold, pred, s1n.sample(frac=1, random_state=0), pooln   # looked up by id, any order


def test_slice_report_rows_sum_to_totals():
    fold, pred, s1n, pooln = slice_case()
    with pytest.warns(UserWarning):
        report = slice_report(pred, fold, s1n, pooln)
        overall = score_pairs(pred, fold)
    assert list(report.columns) == SLICE_COLUMNS
    assert list(dict.fromkeys(report["family"])) == [
        "country", "source", "non_latin", "ambiguous", "singleton", "n_matches", "postcode",
        "addr_empty"]
    for family, part in report[report["family"] != "source"].groupby("family", sort=False):
        total, rest = part.iloc[-1], part.iloc[:-1]
        assert total["slice"] == "all" and total["entities"] == len(fold.s1), family
        assert total["f_beta"] == overall["f_beta"]
        assert rest["entities"].sum() == len(fold.s1), family
        for col in ("tp", "n_pred", "n_true"):
            assert rest[col].sum() == total[col], (family, col)
        filled = rest[rest["entities"] > 0]
        assert (filled["f_beta"] * filled["entities"]).sum() == pytest.approx(
            total["f_beta"] * total["entities"], rel=1e-12), family
    rows = report.set_index(["family", "slice"])
    assert rows.loc[("n_matches", "5+"), "entities"] == 0         # at most 4 true matches here
    assert math.isnan(rows.loc[("n_matches", "5+"), "f_beta"])
    # flags recomputed the slow way
    s1x = s1n.set_index(C.ENTITY_ID).loc[fold.s1[C.ENTITY_ID]]
    shared = s1x.groupby([C.COUNTRY, "name_core"])["name_core"].transform("size") > 1
    assert rows.loc[("ambiguous", "yes"), "entities"] == (shared & (s1x["name_core"] != "")).sum()
    non_latin = set(pooln.loc[pooln["non_latin"], C.ENTITY_ID])
    has_nl = fold.pairs.loc[fold.pairs[C.ENTITY_ID].isin(non_latin), C.S1_ID].nunique()
    assert rows.loc[("non_latin", "yes"), "entities"] == has_nl
    empty_r = set(pooln.loc[pooln["addr_tokens"] == 0, C.ENTITY_ID])
    with_empty = set(fold.pairs.loc[fold.pairs[C.ENTITY_ID].isin(empty_r), C.S1_ID])
    empty = with_empty | set(s1x.index[s1x["addr_tokens"] == 0])
    assert rows.loc[("addr_empty", "yes"), "entities"] == len(empty)


def test_slice_report_country_open_set():
    fold = make_fold(["S1-1", "S1-2", "S1-3"], ["S2-1", "S3-1"], [("S1-1", "S2-1")],
                     countries=["US", "France", "France"])
    s1n = normalised(fold.s1[C.ENTITY_ID], country=["US", "France", "France"])
    pooln = normalised(["S2-1", "S3-1"])
    report = slice_report(pairs_frame([("S1-1", "S2-1")]), fold, s1n, pooln)
    country = report[report["family"] == "country"].set_index("slice")
    assert country.index.tolist() == ["France", "US", "all"]
    assert country.loc["France", "entities"] == 2 and country.loc["France", "f_beta"] == 1.0
    assert "domain_form" not in set(report["family"])
    with_domain = pooln.assign(domain_form=[True, False])     # extra column after NORM_COLUMNS
    report = slice_report(pairs_frame([("S1-1", "S2-1")]), fold, s1n, with_domain)
    domain = report[report["family"] == "domain_form"].set_index("slice")["entities"]
    assert domain.to_dict() == {"yes": 1, "no": 2, "all": 3}


def test_slice_report_source_projection():
    fold = make_fold(["S1-a", "S1-b"], ["S2-1", "S3-1", "S3-2"],
                     [("S1-a", "S2-1"), ("S1-a", "S3-1")])
    s1n, pooln = normalised(["S1-a", "S1-b"]), normalised(["S2-1", "S3-1", "S3-2"])
    report = slice_report(pairs_frame([("S1-a", "S2-1")]), fold, s1n, pooln)
    source = report[report["family"] == "source"].set_index("slice")
    assert source.index.tolist() == ["S2", "S3"]
    assert source.loc["S2", ["pair_precision", "pair_recall", "f_beta"]].tolist() == [1, 1, 1]
    assert source.loc["S3", "pair_recall"] == 0 and source.loc["S3", "f_beta"] == 0
    assert source["entities"].tolist() == [1, 1]            # S1-b has no S2/S3 pair at all
    with pytest.raises(ValueError, match="NORM_COLUMNS"):
        slice_report(pairs_frame([]), fold, s1n.iloc[:, ::-1], pooln)


def test_score_pairs_foreign_ids():
    fold = make_fold(["S1-a", "S1-b"], ["S2-1", "S2-2", "S3-1"],
                     [("S1-a", "S2-1"), ("S1-b", "S3-1")])
    pred = pairs_frame([("S1-a", "S2-1"), ("S1-a", "S2-999"), ("S1-zz", "S2-2")])
    with pytest.warns(UserWarning, match="1 predicted pair"):
        got = score_pairs(pred, fold)
    assert_same(got, metrics.breakdown({"S1-a": ["S2-1", "S2-999"]},
                                       {"S1-a": {"S2-1"}, "S1-b": {"S3-1"}}))
    assert got["pair_precision"] == 0.5                      # the unknown id is a false positive
    with warnings.catch_warnings():
        warnings.simplefilter("error")                        # known ids: no warning
        score_pairs(pred.iloc[[0, 2]], fold)
    with pytest.raises(ValueError, match="truth is empty"):
        score_pairs(pred, Fold("val", fold.s1.iloc[:0], fold.s2, fold.s3, fold.pairs.iloc[:0]))


def error_case():
    """S1-a: 3 true, predicts one of them plus a false one; S1-b: matched, predicts nothing;
    S1-c: singleton with a prediction; S1-d: perfect."""
    fold = make_fold(["S1-a", "S1-b", "S1-c", "S1-d"],
                     ["S2-1", "S3-1", "S2-2", "S2-3", "S3-4", "S3-5", "S2-9"],
                     [("S1-a", "S2-1"), ("S1-a", "S3-1"), ("S1-a", "S2-2"), ("S1-b", "S2-3"),
                      ("S1-d", "S3-4")])
    pred = pairs_frame([("S1-a", "S2-1"), ("S1-a", "S2-9"), ("S1-c", "S3-5"), ("S1-d", "S3-4")])
    return fold, pred


def test_error_samples_missed():
    fold, pred = error_case()
    missed = error_samples(pred, fold, "missed")
    assert list(missed.columns) == SAMPLE_COLUMNS
    assert pair_rows(missed) == [("S1-a", "S2-2"), ("S1-a", "S3-1")]
    row = missed.iloc[1]                                      # both records side by side
    assert (row["name_l"], row["addr_l"]) == ("name S1-a", "addr S1-a")
    assert (row["name_r"], row["addr_r"]) == ("name S3-1", "addr S3-1")
    assert missed["prob"].isna().all()
    scored = pairs_frame([("S1-a", "S3-1"), ("S1-a", "S2-1")]).assign(
        prob=np.float32([0.25, 0.75]))
    assert error_samples(pred, fold, "missed", scored=scored)["prob"].tolist()[1] == 0.25


def pair_rows(frame: pd.DataFrame) -> list[tuple[str, str]]:
    """The (source1_entity_id, entity_id) pairs of a frame, in row order."""
    return list(zip(frame[C.S1_ID], frame[C.ENTITY_ID], strict=True))


def test_error_samples_kinds_and_sampling():
    fold, pred = error_case()
    assert pair_rows(error_samples(pred, fold, "false_merge")) == [("S1-a", "S2-9")]
    assert pair_rows(error_samples(pred, fold, "false_singleton")) == [("S1-b", "S2-3")]
    assert pair_rows(error_samples(pred, fold, "singleton_merge")) == [("S1-c", "S3-5")]
    with pytest.raises(ValueError, match="kind"):
        error_samples(pred, fold, "typo")
    big, big_pred = random_case(seed=4)
    with pytest.warns(UserWarning):
        one = error_samples(big_pred, big, "false_merge", n=5, seed=1)
    with pytest.warns(UserWarning):
        again = error_samples(big_pred.sample(frac=1, random_state=9), big, "false_merge", n=5,
                              seed=1)
    assert len(one) == 5
    pd.testing.assert_frame_equal(one, again)                # a draw by id hash, order-free
