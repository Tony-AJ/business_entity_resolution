"""Stage-2 features from stage-1 probabilities: competition between candidates (plan C5).

At test density a common name brings many candidates that all look alike on their own; what
separates the true record from a same-name decoy is often the competition around the pair:
does this S1 entity have a clearly better candidate, does another S1 entity claim this pool
record more strongly, how many candidates of either side are likely matches? A stage-1
matcher scores every candidate pair of a partition; ``competition_features`` turns its
probabilities ``p1`` into per-pair context, and a stage-2 matcher learns from the pair
features plus that context.

    p1                 stage-1 probability of the pair
    s1_rank            rank of p1 among the S1 entity's candidates (1 = best)
    s1_best_other      best p1 among the entity's OTHER candidates (0 without any)
    s1_gap             p1 - s1_best_other (> 0 only for the entity's best candidate)
    s1_p1_sum          sum of p1 over the entity's candidates (expected match count)
    s1_n_likely        entity candidates with p1 >= LIKELY
    pool_rank          rank of p1 among the S1 entities that have this pool record
    pool_best_other    best p1 of this pool record with OTHER S1 entities (0 without any)
    pool_gap           p1 - pool_best_other (> 0 only for the record's best entity)
    pool_p1_sum        sum of p1 over the S1 entities holding this record (a record has at
                       most one true owner, so a sum well above 1 means ambiguity)
    pool_n_likely      S1 entities with p1 >= LIKELY for this pool record
    pool_degree        S1 entities that have this pool record as a candidate

Every column is computed over the pairs passed in, so pass the whole partition (a country
of the mock fold or of test): a subset would miss rivals. Nothing loops over pairs in Python.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

LIKELY = 0.5
STACK_COLUMNS = ["p1", "s1_rank", "s1_best_other", "s1_gap", "s1_p1_sum", "s1_n_likely",
                 "pool_rank", "pool_best_other", "pool_gap", "pool_p1_sum", "pool_n_likely",
                 "pool_degree"]


def group_stats(codes: np.ndarray, p1: np.ndarray,
                 n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row: rank of p1 inside its group (1 = best) and the best p1 of the other rows.

    Ties keep the input order (stable sort), so two rows with the same p1 get ranks 1 and 2
    and each sees the other's p1 as its best rival. A row alone in its group has rival 0.
    """
    n = len(p1)
    rank = np.empty(n, dtype=np.float32)
    top1 = np.zeros(n_groups, dtype=np.float32)
    top2 = np.zeros(n_groups, dtype=np.float32)
    if n == 0:
        return rank, np.zeros(0, dtype=np.float32)
    order = np.lexsort((-p1, codes))                # by group, then p1 descending, stable
    g, v = codes[order], p1[order]
    first = np.r_[True, g[1:] != g[:-1]]            # the best row of each group
    at = np.arange(n)
    rank[order] = at - np.maximum.accumulate(np.where(first, at, 0)) + 1
    second = np.r_[False, first[:-1]] & ~first      # right after a group's best, same group
    top1[g[first]] = v[first]
    top2[g[second]] = v[second]
    return rank, np.where(rank == 1, top2[codes], top1[codes]).astype(np.float32)


def competition_features(pairs: pd.DataFrame, p1: np.ndarray) -> pd.DataFrame:
    """``STACK_COLUMNS`` for every row of ``pairs`` (S1 id, pool id) given stage-1 ``p1``.

    ``p1`` is aligned to ``pairs`` rows (NaN counts as 0). Returns float32 columns on
    ``pairs.index``.
    """
    p = np.nan_to_num(np.asarray(p1, dtype=np.float32), nan=0.0)
    if len(p) != len(pairs):
        raise ValueError(f"p1 has {len(p)} values for {len(pairs)} pairs")
    s1, s1_ids = pd.factorize(pairs[C.S1_ID], use_na_sentinel=False)
    pool, pool_ids = pd.factorize(pairs[C.ENTITY_ID], use_na_sentinel=False)
    likely = (p >= LIKELY).astype(np.float32)
    s1_rank, s1_best_other = group_stats(s1, p, len(s1_ids))
    pool_rank, pool_best_other = group_stats(pool, p, len(pool_ids))
    s1_sum = np.bincount(s1, weights=p, minlength=len(s1_ids)).astype(np.float32)
    s1_likely = np.bincount(s1, weights=likely, minlength=len(s1_ids)).astype(np.float32)
    pool_likely = np.bincount(pool, weights=likely, minlength=len(pool_ids)).astype(np.float32)
    pool_sum = np.bincount(pool, weights=p, minlength=len(pool_ids)).astype(np.float32)
    degree = np.bincount(pool, minlength=len(pool_ids)).astype(np.float32)
    cols = {
        "p1": p, "s1_rank": s1_rank, "s1_best_other": s1_best_other,
        "s1_gap": p - s1_best_other, "s1_p1_sum": s1_sum[s1], "s1_n_likely": s1_likely[s1],
        "pool_rank": pool_rank, "pool_best_other": pool_best_other,
        "pool_gap": p - pool_best_other, "pool_p1_sum": pool_sum[pool],
        "pool_n_likely": pool_likely[pool], "pool_degree": degree[pool],
    }
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in cols.items()},
                        index=pairs.index)
