"""Decoy-signature features of the final version's stage 2 (v125, v126; plan C4 / C5).

The test's uncertain candidates are the generator's decoys: a second business named like the
S1 entity (plus or minus a word such as "Holdings") at a *nearby house number* (7541 vs 7532),
with its own Source 2 and Source 3 records. True records carry the S1's number, a truncated
or zero-padded form of it (201 for 2014, 00326), or one source's own corrupted number shared
by all of that source's records (600 for 200). No stage-1 feature reads either signal, so
stage 2 gets three groups, computed on the kept pairs only (``DECOY_COLUMNS``):

* house-number relation (``HOUSE_NUMBER_COLUMNS``) of the first address numbers of both sides;
* candidate groups of the same S1 entity (``GROUP_COLUMNS``): how many of its other candidates
  share the pair's number, name or address, and the best stage-1 probability among them;
* unmatched-token IDF (``UNMATCHED_COLUMNS``): the rarity of the core-name words only one side
  holds. A decoy adds a rare word ("Midtown"); a true pair's extra words are common filler.

The code is moved unchanged from the scripts that produced the submitted file,
``experiments/v126_decoy_groups/run_v126.py`` (house numbers, groups) and
``experiments/v125_unmatched_idf/run_v125.py`` (unmatched IDF), so its values are theirs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .twostage import Stage1Output

HOUSE_NUMBER_COLUMNS = ["hn_eq", "hn_substr", "hn_onedigit", "hn_near", "hn_absdiff",
                        "hn_missing"]
GROUP_COLUMNS = ["grp_num_n", "grp_num_p1max", "grp_num_xsrc", "grp_s1num_n", "grp_name_n",
                 "grp_name_p1max", "grp_addr_n"]
UNMATCHED_COLUMNS = ["uidf_max_l", "uidf_max_r", "uidf_sum_l", "uidf_sum_r", "uidf_share",
                     "unum_conflict"]
DECOY_COLUMNS = [*HOUSE_NUMBER_COLUMNS, *GROUP_COLUMNS, *UNMATCHED_COLUMNS]


def first_number(nums: np.ndarray) -> np.ndarray:
    """First number of each space-joined ``addr_nums`` string ('' when there is none)."""
    return pd.Series(nums, dtype="str").str.split(" ", n=1).str[0].fillna("").to_numpy()


def house_number_features(hl: np.ndarray, hr: np.ndarray) -> pd.DataFrame:
    """House-number relation of the first address numbers of both sides (NaN if one lacks one).

    hn_eq        the numbers are equal
    hn_substr    one contains the other (a truncation, a dropped unit digit)
    hn_onedigit  same length, one digit differs (a noised digit)
    hn_near      within 20 and none of the above: the decoy's neighbouring number
    hn_absdiff   log(1 + |difference|)
    hn_missing   1 the S1 side lacks a number, 2 the pool side, 3 both (never NaN)
    """
    both = (hl != "") & (hr != "")
    eq = both & (hl == hr)
    sub = both & ~eq & np.fromiter(((a in b) or (b in a) for a, b in zip(hl, hr, strict=True)),
                                   dtype=bool, count=len(hl))
    one = both & ~eq & np.fromiter(
        (len(a) == len(b) and sum(x != y for x, y in zip(a, b, strict=True)) == 1
         for a, b in zip(hl, hr, strict=True)), dtype=bool, count=len(hl))
    il = pd.to_numeric(pd.Series(hl), errors="coerce").to_numpy(np.float64)
    ir = pd.to_numeric(pd.Series(hr), errors="coerce").to_numpy(np.float64)
    gap = np.abs(il - ir)
    near = both & ~eq & ~sub & ~one & (gap <= 20)
    na = np.where(both, 0.0, np.nan)
    out = pd.DataFrame({
        "hn_eq": eq + na, "hn_substr": sub + na, "hn_onedigit": one + na,
        "hn_near": near + na, "hn_absdiff": np.log1p(gap) + na,
        "hn_missing": (hl == "").astype(np.float32) + 2 * (hr == "").astype(np.float32)})
    return out.astype(np.float32)


def others_max(key: pd.DataFrame, p1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: how many other rows share its ``key`` and the best ``p1`` among them (0 if none).

    One sort by key and descending p1: a group's best is its first row, and the best *other*
    row of the first row is the group's second.
    """
    order = np.lexsort((-p1, *[key[c].to_numpy() for c in reversed(key.columns)]))
    k = key.iloc[order].reset_index(drop=True)
    p = p1[order]
    start = np.r_[True, (k.iloc[1:].to_numpy() != k.iloc[:-1].to_numpy()).any(axis=1)]
    grp = np.cumsum(start) - 1
    size = np.bincount(grp)
    first = np.flatnonzero(start)
    top1 = p[first]
    top2 = np.where(size > 1, p[np.minimum(first + 1, len(p) - 1)], 0.0)
    is_first = start
    best_other = np.where(is_first, top2[grp], top1[grp])
    n_other = size[grp] - 1
    out_n, out_best = np.empty(len(p)), np.empty(len(p))
    out_n[order], out_best[order] = n_other, best_other
    return out_n, out_best


def group_features(pairs: pd.DataFrame, p1: np.ndarray, hl: np.ndarray, hr: np.ndarray,
                   name_r: np.ndarray, addr_r: np.ndarray) -> pd.DataFrame:
    """Candidate-set groups of each S1 entity: who shares the pair's number, name and address.

    grp_num_n       other candidates of the entity with the pool record's first number
    grp_num_p1max   the best stage-1 probability among them
    grp_num_xsrc    one of them comes from the other source (S2 vs S3): a business both
                    sources describe, the S1's own or a decoy's
    grp_s1num_n     candidates of the entity holding the S1's own first number
    grp_name_n      other candidates with the same core name, and grp_name_p1max their best p1
    grp_addr_n      other candidates with the same normalised address
    The number and address columns are NaN when the pool record has none.
    """
    s1 = pd.factorize(pairs[C.S1_ID])[0]
    is_s3 = pairs[C.ENTITY_ID].str.startswith("S3").to_numpy()
    n = len(pairs)
    has = hr != ""
    num_n, num_best, xsrc = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    if has.any():
        key = pd.DataFrame({"s1": s1[has], "num": pd.factorize(hr[has])[0]})
        num_n[has], num_best[has] = others_max(key, p1[has].astype(np.float64))
        # others of the group from the other source
        g = key["s1"].to_numpy().astype(np.int64) * (key["num"].max() + 1) + key["num"].to_numpy()
        gid = pd.factorize(g)[0]
        s3_in = np.bincount(gid, weights=is_s3[has].astype(np.float64))
        size = np.bincount(gid)
        s3_other = s3_in[gid] - is_s3[has]
        s2_other = (size[gid] - s3_in[gid]) - (~is_s3[has])
        xsrc[has] = np.where(is_s3[has], s2_other, s3_other) > 0
    s1num = np.bincount(s1, weights=((hl == hr) & has).astype(np.float64))[s1]
    name_key = pd.DataFrame({"s1": s1, "name": pd.factorize(name_r)[0]})
    name_n, name_best = others_max(name_key, p1.astype(np.float64))
    addr_has = addr_r != ""
    addr_n = np.full(n, np.nan)
    if addr_has.any():
        g = s1[addr_has].astype(np.int64) * (int(addr_has.sum()) + 1) + pd.factorize(
            addr_r[addr_has])[0]
        gid = pd.factorize(g)[0]
        addr_n[addr_has] = np.bincount(gid)[gid] - 1
    return pd.DataFrame({
        "grp_num_n": num_n, "grp_num_p1max": num_best, "grp_num_xsrc": xsrc,
        "grp_s1num_n": s1num, "grp_name_n": name_n, "grp_name_p1max": name_best,
        "grp_addr_n": addr_n}).astype(np.float32)


def idf_table(pool_names: pd.Series) -> dict[str, float]:
    """log(N / df) per token over the pool's core names (df = records holding the token)."""
    toks = pool_names.str.split()
    n = max(len(toks), 1)
    df = pd.Series([t for row in toks for t in set(row)]).value_counts()
    return dict(zip(df.index, np.log(n / df.to_numpy()), strict=True))


def unmatched_features(pairs: pd.DataFrame, s1n: pd.DataFrame, pooln: pd.DataFrame,
                       idf: dict[str, float]) -> pd.DataFrame:
    """Unmatched-token IDF of ``pairs`` (float32), from normalised name_core / addr_norm.

    uidf_max_l / uidf_max_r   highest IDF among the core-name tokens only that side holds
    uidf_sum_l / uidf_sum_r   summed IDF of those tokens
    uidf_share                unmatched IDF mass / IDF mass of both names' tokens
    unum_conflict             both addresses hold numbers and none is shared
    A token the pool never holds gets the median IDF of the table.
    """
    s1 = s1n.set_index(C.ENTITY_ID)
    pool = pooln.set_index(C.ENTITY_ID)
    ln = s1["name_core"].reindex(pairs[C.S1_ID]).fillna("").to_numpy()
    rn = pool["name_core"].reindex(pairs[C.ENTITY_ID]).fillna("").to_numpy()
    la = s1["addr_norm"].reindex(pairs[C.S1_ID]).fillna("").to_numpy()
    ra = pool["addr_norm"].reindex(pairs[C.ENTITY_ID]).fillna("").to_numpy()
    out = np.zeros((len(pairs), len(UNMATCHED_COLUMNS)), dtype=np.float32)
    default = float(np.median(list(idf.values()))) if idf else 0.0
    for i in range(len(pairs)):
        tl, tr = set(ln[i].split()), set(rn[i].split())
        only_l, only_r = tl - tr, tr - tl
        wl = [idf.get(t, default) for t in only_l]
        wr = [idf.get(t, default) for t in only_r]
        wall = [idf.get(t, default) for t in tl | tr]
        um = sum(wl) + sum(wr)
        out[i, 0] = max(wl, default=0.0)
        out[i, 1] = max(wr, default=0.0)
        out[i, 2] = sum(wl)
        out[i, 3] = sum(wr)
        out[i, 4] = um / max(sum(wall), 1e-9)
        nl = {t for t in la[i].split() if t.isdigit()}
        nr = {t for t in ra[i].split() if t.isdigit()}
        out[i, 5] = float(bool(nl) and bool(nr) and not (nl & nr))
    return pd.DataFrame(out, columns=UNMATCHED_COLUMNS, index=pairs.index)


def decoy_features(o: Stage1Output, s1n: pd.DataFrame, pooln: pd.DataFrame) -> pd.DataFrame:
    """Every ``DECOY_COLUMNS`` value for the kept pairs of ``o`` (row order of ``o.pairs``).

    ``s1n`` / ``pooln``: the normalised S1 and pool records of the partition (token map
    applied); the IDF table is the partition's own pool, so France gets French rarity.
    """
    s1 = s1n.drop_duplicates(C.ENTITY_ID).set_index(C.ENTITY_ID)
    pool = pooln.drop_duplicates(C.ENTITY_ID).set_index(C.ENTITY_ID)
    pairs = o.pairs.reset_index(drop=True)
    hl = first_number(s1["addr_nums"].reindex(pairs[C.S1_ID]).fillna("").to_numpy())
    hr = first_number(pool["addr_nums"].reindex(pairs[C.ENTITY_ID]).fillna("").to_numpy())
    name_r = pool["name_core"].reindex(pairs[C.ENTITY_ID]).fillna("").to_numpy()
    addr_r = pool["addr_norm"].reindex(pairs[C.ENTITY_ID]).fillna("").to_numpy()
    p1 = o.X["p1"].to_numpy(np.float64)
    idf = idf_table(pooln["name_core"])
    parts = [house_number_features(hl, hr), group_features(pairs, p1, hl, hr, name_r, addr_r),
             unmatched_features(pairs, s1n, pooln, idf).reset_index(drop=True)]
    return pd.concat(parts, axis=1)[DECOY_COLUMNS]


def with_columns(o: Stage1Output, feats: pd.DataFrame) -> Stage1Output:
    """The stage-1 output with ``feats`` (same rows) appended to its frame."""
    return Stage1Output(o.pairs.reset_index(drop=True),
                        pd.concat([o.X.reset_index(drop=True), feats], axis=1), o.n_all)
