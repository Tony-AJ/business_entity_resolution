"""Evaluation on pairs frames: the challenge metric, blocking recall, slices, error samples.

``metrics.py`` is the reference implementation of macro F0.5: per-entity sets and Python
loops over dicts, fine for a val-size ``breakdown`` but not for the 10-50M candidate pairs of
a fold. This module computes the same numbers from pairs frames (``source1_entity_id``,
``entity_id``) with vectorised counts, equal to the reference bit for bit
(11_VALIDATION_AND_METRICS §4-§8):

    entity_f05_from_counts, macro_f05_from_counts   the metric on count arrays
    entity_counts                                   (tp, n_pred, n_true) per fold S1 entity
    score_pairs, blocking_report                    the keys of metrics.breakdown / candidate_report
    pairs_to_lists, pair_recall, pair_in            id lists (val size only), recall, membership
    slice_report, error_samples                     per-slice scores; wrong pairs side by side
    harder_fold                                     val with 20% of S1 dropped, pool kept (§6)

How: ids become integer positions once (pyarrow ``index_in``, no Python objects); a pair is
the int64 key ``s1_position * n_pool_codes + pool_code``; set semantics is a sort-unique of
those keys and per-entity counts are ``np.bincount`` over S1 positions. Nothing loops in
Python over pairs or entities, except ``pairs_to_lists``, whose output is a dict of lists.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from . import config as C
from .data import isin
from .split import Fold, hash_unit

B2 = C.BETA * C.BETA  # 0.25: the same float operations as metrics.entity_fbeta

ERROR_KINDS = ("false_merge", "missed", "false_singleton", "singleton_merge")
SLICE_COLUMNS = ["family", "slice", "entities", "f_beta", "pair_precision", "pair_recall",
                 "n_true", "n_pred", "tp"]
SAMPLE_COLUMNS = [C.S1_ID, C.ENTITY_ID, "prob", "name_l", "addr_l", "name_r", "addr_r"]
N_MATCH_SLICES = ("0", "1", "2", "3-4", "5+")
N_MATCH_EDGES = (1, 2, 3, 5)  # np.digitize edges giving the N_MATCH_SLICES bins


# ---------------------------------------------------------------- metric ----
def entity_f05_from_counts(tp, n_pred, n_true) -> np.ndarray:
    """Per-entity F0.5 from count arrays: the formula of ``metrics.entity_fbeta``, bitwise.

    P = tp / n_pred, R = tp / n_true, F = 1.25·P·R / (0.25·P + R); an entity without true
    matches scores 1.0 for an empty prediction and 0.0 otherwise; no true positive scores 0.0.
    Same float64 operations in the same order as the reference, so the values are identical.
    """
    tp, n_pred, n_true = (np.asarray(a, dtype=np.float64) for a in (tp, n_pred, n_true))
    with np.errstate(divide="ignore", invalid="ignore"):
        p, r = tp / n_pred, tp / n_true            # NaN/inf where a count is 0: masked below
        f = (1 + B2) * p * r / (B2 * p + r)
    return np.where(n_true == 0, (n_pred == 0) * 1.0, np.where(tp == 0, 0.0, f))


def macro_f05_from_counts(tp, n_pred, n_true) -> float:
    """Mean per-entity F0.5 over ALL the entities given (one element each, singletons too).

    ``math.fsum`` and one division is exactly what ``statistics.fmean`` computes, so the value
    equals ``metrics.macro_fbeta`` bit for bit, in any entity order.
    """
    f = entity_f05_from_counts(tp, n_pred, n_true)
    if f.size == 0:
        raise ValueError("no entities to average")
    return math.fsum(f.ravel().tolist()) / f.size  # tolist: fsum reads Python floats faster


# ------------------------------------------------------------ id lookups ----
def _as_arrow(values) -> pa.Array | pa.ChunkedArray:
    """``values`` (Series, Index, array or list of ids) as Arrow large_string, zero-copy when
    they already are (pandas' default ``str`` dtype)."""
    arr = pa.array(values)
    return arr if arr.type == pa.large_string() else arr.cast(pa.large_string())


def positions(values, keys) -> np.ndarray:
    """Position of each of ``values`` in ``keys`` (int64, -1 where absent).

    A vectorised ``Index.get_indexer`` through pyarrow ``index_in``: 4x faster on millions of
    Arrow strings and no Python objects. ``keys`` must be unique (the first match wins).
    """
    found = pc.index_in(_as_arrow(values), value_set=_as_arrow(keys)).fill_null(-1)
    return np.asarray(found, dtype=np.int64)


def _sorted_unique(a: np.ndarray) -> np.ndarray:
    """Sorted distinct values of an int array (numpy 2.5's hash-based ``np.unique`` is ~50x
    slower on int64 keys than a sort plus a neighbour mask)."""
    s = np.sort(a)
    return s[np.r_[True, s[1:] != s[:-1]]] if len(s) else s


def _member(query: np.ndarray, sorted_keys: np.ndarray) -> np.ndarray:
    """``np.isin(query, sorted_keys)`` for sorted unique ``sorted_keys``, by binary search."""
    if len(sorted_keys) == 0:
        return np.zeros(len(query), dtype=bool)
    at = np.minimum(np.searchsorted(sorted_keys, query), len(sorted_keys) - 1)
    return sorted_keys[at] == query


def _pair_keys(pairs: pd.DataFrame, s1_keys, pool_keys) -> np.ndarray:
    """int64 key per row, ``s1 position * len(pool_keys) + pool position``; -1 where either id
    is not among the keys."""
    s1, pool = positions(pairs[C.S1_ID], s1_keys), positions(pairs[C.ENTITY_ID], pool_keys)
    return np.where((s1 >= 0) & (pool >= 0), s1 * max(len(pool_keys), 1) + pool, -1)


def pair_in(pairs: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
    """Boolean per row of ``pairs``: is its (source1_entity_id, entity_id) a pair of ``other``?

    Keys live in the id space of ``other`` (the small side, e.g. truth pairs), so each row of
    ``pairs`` costs two hash lookups and a binary search, never a string merge.
    """
    s1_keys, pool_keys = pd.unique(other[C.S1_ID]), pd.unique(other[C.ENTITY_ID])
    ref = _sorted_unique(_pair_keys(other, s1_keys, pool_keys))
    keys = _pair_keys(pairs, s1_keys, pool_keys)
    return (keys >= 0) & _member(keys, ref)


def pairs_to_lists(pairs: pd.DataFrame, s1_ids: Iterable[str]) -> dict[str, list[str]]:
    """``{s1_id: [pool ids]}`` for every id of ``s1_ids``, in that order, ``[]`` when absent.

    Duplicate pairs collapse (first occurrence order kept); rows for other S1 ids are dropped.
    This is the input of ``metrics.breakdown`` / ``candidate_report``: val size only (02 §4.5;
    ~2 s for 441k entities and 1.6M pairs, 5x faster than ``groupby().agg(list)``).
    """
    out: dict[str, list[str]] = {s1: [] for s1 in s1_ids}
    keys = list(out)
    s1 = positions(pairs[C.S1_ID], keys)                 # -1: not among s1_ids, dropped
    pool, pool_ids = pd.factorize(pairs[C.ENTITY_ID], use_na_sentinel=False)
    first = ~pd.Series(s1 * max(len(pool_ids), 1) + pool).duplicated().to_numpy()
    rows = np.flatnonzero((s1 >= 0) & first)
    if len(rows) == 0:
        return out
    rows = rows[np.argsort(s1[rows], kind="stable")]    # grouped by entity, row order kept
    pos, ids = s1[rows], np.asarray(pool_ids, dtype=object)[pool[rows]]
    start = np.flatnonzero(np.r_[True, pos[1:] != pos[:-1]])
    for p, group in zip(pos[start].tolist(), np.split(ids, start[1:]), strict=True):
        out[keys[p]] = group.tolist()
    return out


def pair_recall(cand_pairs: pd.DataFrame, truth_pairs: pd.DataFrame) -> float:
    """Share of the distinct truth pairs found among the candidates (NaN without truth)."""
    s1_keys, pool_keys = pd.unique(truth_pairs[C.S1_ID]), pd.unique(truth_pairs[C.ENTITY_ID])
    truth = _sorted_unique(_pair_keys(truth_pairs, s1_keys, pool_keys))
    if len(truth) == 0:
        return float("nan")
    cand = _pair_keys(cand_pairs, s1_keys, pool_keys)
    return int(_member(truth, _sorted_unique(cand[cand >= 0])).sum()) / len(truth)


# ------------------------------------------------------ counts on a fold ----
@dataclass(frozen=True)
class _PairSets:
    """Distinct predicted and true pairs of the fold's S1 entities as sorted int64 keys."""

    n_codes: int         # key = S1 row position in fold.s1 * n_codes + pool code
    pool_ids: pd.Index   # pool id of each code
    pred: np.ndarray
    true: np.ndarray


def _pair_sets(pred_pairs: pd.DataFrame, fold: Fold) -> _PairSets:
    """Encode ``pred_pairs`` and ``fold.pairs`` in one key space; rows for S1 ids outside the
    fold are dropped (``metrics.breakdown`` ignores them too)."""
    s1_ids = fold.s1[C.ENTITY_ID]
    ps, ts = positions(pred_pairs[C.S1_ID], s1_ids), positions(fold.pairs[C.S1_ID], s1_ids)
    both = pd.concat([pred_pairs[C.ENTITY_ID], fold.pairs[C.ENTITY_ID]], ignore_index=True)
    codes, pool_ids = pd.factorize(both, use_na_sentinel=False)  # one code per distinct pool id
    n_codes, k = max(len(pool_ids), 1), len(pred_pairs)

    def unique_keys(s1: np.ndarray, pool: np.ndarray) -> np.ndarray:
        """Sorted distinct keys of the rows whose S1 id is in the fold (set semantics)."""
        ok = s1 >= 0
        return _sorted_unique(s1[ok] * n_codes + pool[ok])

    return _PairSets(n_codes, pool_ids, unique_keys(ps, codes[:k]), unique_keys(ts, codes[k:]))


def _counts(pred_pairs: pd.DataFrame,
            fold: Fold) -> tuple[_PairSets, np.ndarray, np.ndarray, np.ndarray]:
    """(pair sets, tp, n_pred, n_true), counts aligned to ``fold.s1`` rows.

    A predicted pool id that is not a record of the fold is a false positive like any other,
    never dropped; a warning gives the count, since it means the wrong split was scored.
    """
    sets = _pair_sets(pred_pairs, fold)
    if len(sets.pred):
        pool = pd.concat([fold.s2[C.ENTITY_ID], fold.s3[C.ENTITY_ID]], ignore_index=True)
        known = isin(pd.Series(sets.pool_ids), pool)            # per pool code
        n_foreign = int((~known[sets.pred % sets.n_codes]).sum())
        if n_foreign:
            warnings.warn(f"{n_foreign:,} predicted pairs name a pool id that is not a record of "
                          f"fold {fold.name!r}: counted as false positives (wrong split?)",
                          stacklevel=3)
    n = len(fold.s1)
    n_pred = np.bincount(sets.pred // sets.n_codes, minlength=n)
    n_true = np.bincount(sets.true // sets.n_codes, minlength=n)
    tp = np.bincount(sets.true[_member(sets.true, sets.pred)] // sets.n_codes, minlength=n)
    return sets, tp, n_pred, n_true


def entity_counts(pred_pairs: pd.DataFrame,
                  fold: Fold) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(tp, n_pred, n_true)`` per S1 entity, aligned to ``fold.s1`` rows (int64).

    Set semantics (duplicate pairs count once); rows for S1 ids outside the fold are ignored;
    a predicted pool id that is not a fold record counts in ``n_pred`` and warns.
    """
    _, tp, n_pred, n_true = _counts(pred_pairs, fold)
    return tp, n_pred, n_true


def _fmean(f: np.ndarray) -> float:
    """``statistics.fmean`` of an array, NaN when empty (as ``breakdown`` reports it)."""
    return math.fsum(f.tolist()) / len(f) if len(f) else float("nan")


def score_pairs(pred_pairs: pd.DataFrame, fold: Fold) -> dict[str, float | int]:
    """``metrics.breakdown`` of a predicted pairs frame on ``fold``, vectorised.

    Same keys and the same values as ``breakdown(pairs_to_lists(pred_pairs, ids),
    pairs_to_lists(fold.pairs, ids))`` with ``ids = fold.s1.entity_id``: every S1 entity
    counts, singletons included, and an entity without rows is an empty prediction.
    """
    if len(fold.s1) == 0:
        raise ValueError("truth is empty")
    _, tp, n_pred, n_true = _counts(pred_pairs, fold)
    f = entity_f05_from_counts(tp, n_pred, n_true)
    single = n_true == 0
    tp_s, pred_s, true_s = int(tp.sum()), int(n_pred.sum()), int(n_true.sum())
    nan = float("nan")
    return {
        "f_beta": _fmean(f),
        "f_beta_singletons": _fmean(f[single]),
        "f_beta_matched": _fmean(f[~single]),
        "pair_precision": tp_s / pred_s if pred_s else nan,
        "pair_recall": tp_s / true_s if true_s else nan,
        "entities": len(f),
        "singletons": int(single.sum()),
    }


def blocking_report(cand_pairs: pd.DataFrame, fold: Fold) -> dict[str, float | int]:
    """``metrics.candidate_report`` of a candidate pairs frame on ``fold``, vectorised.

    ``pool_size`` is the fold's pool (S2 + S3 records); the ceiling F0.5 is a perfect matcher
    that predicts exactly the true pairs among each entity's candidates.
    """
    if len(fold.s1) == 0:
        raise ValueError("truth is empty")
    _, tp, n_cand, n_true = _counts(cand_pairs, fold)
    matched = n_true > 0
    pool_size = len(fold.s2) + len(fold.s3)
    kept, n_true_s = int(tp.sum()), int(n_true.sum())
    hit, n_matched = int((tp[matched] > 0).sum()), int(matched.sum())
    nan = float("nan")
    return {
        "pair_recall": kept / n_true_s if n_true_s else nan,
        "entity_recall": hit / n_matched if n_matched else nan,
        "ceiling_f_beta": _fmean(entity_f05_from_counts(tp, tp, n_true)),
        "candidates_mean": float(n_cand.mean()),
        "candidates_p95": float(np.percentile(n_cand, 95)),
        "candidates_max": int(n_cand.max()),
        "candidate_pairs": int(n_cand.sum()),
        # same expression as the reference (np.int64 / int), so the same float
        "reduction_ratio": (float(1 - n_cand.sum() / (len(n_cand) * pool_size))
                            if pool_size else nan),
    }


# ---------------------------------------------------------- harder fold ----
def harder_fold(fold: Fold, drop_frac: float = 0.2, seed: int = 99) -> Fold:
    """``fold`` without ``drop_frac`` of its S1 entities (by id hash), pool untouched (11 §6).

    The dropped entities' true matches stay in the pool as unowned records: the extra decoys
    of test, where ~40% of the pool is unmatched against 26% in train.
    """
    keep = ~(hash_unit(fold.s1[C.ENTITY_ID], seed) < drop_frac)
    kept_ids = pd.Index(fold.s1[C.ENTITY_ID][keep])
    pairs = fold.pairs[isin(fold.pairs[C.S1_ID], kept_ids)]
    return Fold("harder", fold.s1[keep].reset_index(drop=True), fold.s2, fold.s3,
                pairs.reset_index(drop=True))


# ---------------------------------------------------------- slice report ----
