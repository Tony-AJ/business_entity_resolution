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

``anchor_features`` adds a second kind of context. An S1 entity has ~3.5 true records, and
they describe the same business, so they resemble each other; a same-name decoy belongs to
another business and its address differs from theirs. Each candidate is compared with its
**anchor**, the entity's best OTHER candidate by p1:

    anc_p1           p1 of the anchor (NaN without one)
    anc_addr_ts      token-set similarity of the two pool addresses (NaN if either is empty)
    anc_addr_ratio   plain similarity ratio of the two pool addresses
    anc_name_ts      token-set similarity of the two pool names
    anc_nums_eq      the two pool addresses carry the same numbers (NaN if either has none)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .evaluate import positions
from .features import _RATIO, _TOKEN_SET, _arrow, _eq, _fuzzy

LIKELY = 0.5
STACK_COLUMNS = ["p1", "s1_rank", "s1_best_other", "s1_gap", "s1_p1_sum", "s1_n_likely",
                 "pool_rank", "pool_best_other", "pool_gap", "pool_p1_sum", "pool_n_likely",
                 "pool_degree"]
ANCHOR_COLUMNS = ["anc_p1", "anc_addr_ts", "anc_addr_ratio", "anc_name_ts", "anc_nums_eq"]
ANCHOR_CHUNK = 1_000_000   # rows per rapidfuzz batch in anchor_features


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


def competition_features(pairs: pd.DataFrame, p1: np.ndarray,
                         keep: np.ndarray | None = None) -> pd.DataFrame:
    """``STACK_COLUMNS`` for the rows of ``pairs`` (S1 id, pool id) given stage-1 ``p1``.

    ``p1`` is aligned to ``pairs`` rows (NaN counts as 0). Every statistic is computed over
    all rows; with ``keep`` (a boolean row mask) only the kept rows are returned, on a fresh
    RangeIndex, so a 25M-row partition never materialises 25M x 12 values. Without ``keep``
    the frame is on ``pairs.index``. float32 throughout.
    """
    p = np.nan_to_num(np.asarray(p1, dtype=np.float32), nan=0.0)
    if len(p) != len(pairs):
        raise ValueError(f"p1 has {len(p)} values for {len(pairs)} pairs")
    rows = np.flatnonzero(keep) if keep is not None else slice(None)
    s1, s1_ids = pd.factorize(pairs[C.S1_ID], use_na_sentinel=False)
    pool, pool_ids = pd.factorize(pairs[C.ENTITY_ID], use_na_sentinel=False)
    likely = (p >= LIKELY).astype(np.float32)
    cols: dict[str, np.ndarray] = {"p1": p[rows]}
    rank, best_other = group_stats(s1, p, len(s1_ids))
    cols["s1_rank"], cols["s1_best_other"] = rank[rows], best_other[rows]
    cols["s1_gap"] = cols["p1"] - cols["s1_best_other"]
    s1k = s1[rows]
    cols["s1_p1_sum"] = np.bincount(s1, weights=p, minlength=len(s1_ids))[s1k]
    cols["s1_n_likely"] = np.bincount(s1, weights=likely, minlength=len(s1_ids))[s1k]
    del rank, best_other, s1, s1k
    rank, best_other = group_stats(pool, p, len(pool_ids))
    cols["pool_rank"], cols["pool_best_other"] = rank[rows], best_other[rows]
    cols["pool_gap"] = cols["p1"] - cols["pool_best_other"]
    pk = pool[rows]
    cols["pool_p1_sum"] = np.bincount(pool, weights=p, minlength=len(pool_ids))[pk]
    cols["pool_n_likely"] = np.bincount(pool, weights=likely, minlength=len(pool_ids))[pk]
    cols["pool_degree"] = np.bincount(pool, minlength=len(pool_ids))[pk]
    index = pairs.index if keep is None else pd.RangeIndex(len(cols["p1"]))
    return pd.DataFrame({k: np.asarray(cols[k], dtype=np.float32) for k in STACK_COLUMNS},
                        index=index)


def best_other_rows(codes: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Per row: the row index of the best OTHER row of its group by p1, -1 when alone.

    The group's best row points at the second best; every other row at the best (ties
    broken by input order, as in ``group_stats``).
    """
    n = len(p1)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((-p1, codes))
    g = codes[order]
    first = np.r_[True, g[1:] != g[:-1]]
    second = np.r_[False, first[:-1]] & ~first
    n_groups = int(codes.max()) + 1
    best = np.full(n_groups, -1, dtype=np.int64)
    runner = np.full(n_groups, -1, dtype=np.int64)
    best[g[first]] = order[first]
    runner[g[second]] = order[second]
    return np.where(np.arange(n) == best[codes], runner[codes], best[codes])


def anchor_features(pairs: pd.DataFrame, p1: np.ndarray, pooln: pd.DataFrame) -> pd.DataFrame:
    """``ANCHOR_COLUMNS`` for every row of ``pairs`` (S1 id, pool id), anchors among them.

    ``p1`` is aligned to ``pairs``; ``pooln`` holds the normalised pool records the pairs
    name (``name_norm``, ``addr_norm``, ``addr_nums``). Returns float32 on ``pairs.index``.
    """
    p = np.nan_to_num(np.asarray(p1, dtype=np.float32), nan=0.0)
    codes, _ = pd.factorize(pairs[C.S1_ID], use_na_sentinel=False)
    anchor = best_other_rows(codes, p)
    rows = np.flatnonzero(anchor >= 0)
    at = positions(pairs[C.ENTITY_ID], pooln[C.ENTITY_ID])
    if (at < 0).any():
        raise ValueError(f"{int((at < 0).sum())} pool ids of pairs are not in pooln")

    def side(column: str, idx: np.ndarray):
        return _arrow(pooln[column].iloc[idx].reset_index(drop=True))

    out = {c: np.full(len(pairs), np.nan, dtype=np.float32) for c in ANCHOR_COLUMNS}
    out["anc_p1"][rows] = p[anchor[rows]]
    # rapidfuzz reads Python strings: 1M rows at a time bounds that copy; the anchor side goes
    # first (it repeats along the entity's rows, so it is converted once per run) and every
    # scorer is symmetric
    for lo in range(0, len(rows), ANCHOR_CHUNK):
        r = rows[lo:lo + ANCHOR_CHUNK]
        mine, theirs = at[r], at[anchor[r]]
        ts, ratio = _fuzzy(side("addr_norm", theirs), side("addr_norm", mine),
                           (_TOKEN_SET, _RATIO))
        (name_ts,) = _fuzzy(side("name_norm", theirs), side("name_norm", mine), (_TOKEN_SET,))
        eq, both = _eq(side("addr_nums", theirs), side("addr_nums", mine))
        out["anc_addr_ts"][r], out["anc_addr_ratio"][r] = ts, ratio
        out["anc_name_ts"][r] = name_ts
        out["anc_nums_eq"][r] = np.where(both, eq, np.nan)
    return pd.DataFrame(out, index=pairs.index)


COHESION_COLUMNS = ["coh_addr", "coh_name", "coh_support"]
COHESION_ROWS = 1_000_000   # kept rows per batch (their within-entity comparisons: ~7x that)
SUPPORT_P1, SUPPORT_SIM = 0.5, 0.8


def within_group_pairs(starts: np.ndarray, sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every ordered pair (b, c), b != c, of rows inside the same contiguous group.

    ``starts``/``sizes`` describe contiguous row groups; returns row indices ``b`` (each row
    repeated size - 1 times, rows in order) and ``c`` (the other rows of its group, in order).
    """
    start = np.repeat(starts, sizes)                               # group start of each row
    off = np.arange(len(start)) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    rows = start + off                                             # absolute row index
    nb = np.repeat(sizes, sizes) - 1                               # partners per row
    b = np.repeat(rows, nb)
    j = np.arange(len(b)) - np.repeat(np.cumsum(nb) - nb, nb)      # 0 .. size-2 per row
    c = np.repeat(start, nb) + np.where(j < np.repeat(off, nb), j, j + 1)
    return b, c


def cohesion_features(pairs: pd.DataFrame, p1: np.ndarray, pooln: pd.DataFrame) -> pd.DataFrame:
    """``COHESION_COLUMNS``: each candidate against ALL other candidates of its S1 entity.

    coh_addr     p1-weighted mean token-set similarity of this pool address to the entity's
                 other candidates' addresses (NaN without another candidate or weight)
    coh_name     the same on pool names
    coh_support  other candidates with p1 >= 0.5 whose address is >= 0.8 similar to this one

    The anchor features compare with the single best other candidate; cohesion asks how
    well a record fits the whole group the entity's likely records form. ``pairs`` must be
    grouped by S1 id (kept pairs are). Returns float32 on ``pairs.index``.
    """
    n = len(pairs)
    p = np.nan_to_num(np.asarray(p1, dtype=np.float32), nan=0.0)
    out = {c: np.full(n, np.nan, dtype=np.float32) for c in COHESION_COLUMNS}
    if n == 0:
        return pd.DataFrame(out, index=pairs.index)
    ids = pairs[C.S1_ID].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    sizes = np.diff(np.r_[starts, n])
    at = positions(pairs[C.ENTITY_ID], pooln[C.ENTITY_ID])
    if (at < 0).any():
        raise ValueError(f"{int((at < 0).sum())} pool ids of pairs are not in pooln")

    def side(column: str, idx: np.ndarray):
        return _arrow(pooln[column].iloc[idx].reset_index(drop=True))

    g = 0
    while g < len(starts):                            # batches of whole groups
        h = int(np.searchsorted(np.cumsum(sizes[g:]), COHESION_ROWS)) + g + 1
        h = min(max(h, g + 1), len(starts))
        b, c = within_group_pairs(starts[g:h], sizes[g:h])
        if len(b):
            (addr,) = _fuzzy(side("addr_norm", at[c]), side("addr_norm", at[b]), (_TOKEN_SET,))
            (name,) = _fuzzy(side("name_norm", at[c]), side("name_norm", at[b]), (_TOKEN_SET,))
            w = p[c].astype(np.float64)
            lo = int(starts[g])
            m = int(starts[h - 1] + sizes[h - 1]) - lo
            for col, sim in (("coh_addr", addr), ("coh_name", name)):
                ok = ~np.isnan(sim)
                num = np.bincount(b[ok] - lo, weights=w[ok] * sim[ok], minlength=m)
                den = np.bincount(b[ok] - lo, weights=w[ok], minlength=m)
                val = np.full(m, np.nan)
                np.divide(num, den, out=val, where=den > 0)
                out[col][lo:lo + m] = val
            sup = (p[c] >= SUPPORT_P1) & (np.nan_to_num(addr) >= SUPPORT_SIM)
            out["coh_support"][lo:lo + m] = np.bincount(b[sup] - lo, minlength=m)
        g = h
    single = np.repeat(sizes == 1, sizes)
    out["coh_support"][single] = 0.0
    return pd.DataFrame(out, index=pairs.index)
