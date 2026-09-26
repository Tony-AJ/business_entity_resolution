"""Test-shaped mock fold: local scores at the test's decoy density (mock-test protocol).

The fixed val fold (``split.load_fold``) scores 441k S1 entities against a 2.06M-record pool,
a 20% sample of train. Test scores 1.73M S1 against a 10.0M pool, so a test entity meets 3x
(US) to 6x (India) more same-name records of other businesses, and ~40% of the test pool
has no S1 owner at all (26% in train). The first two uploads scored 0.031 below val with
the same offset: val cannot rank the changes that matter at test density.

``build_mock`` rebuilds the test's shape from the whole train split, per country:

1. clusters (an S1 entity with its true pool records) and unowned pool records are kept by
   id hash with the probability that brings the country's pool to its test size (all of it
   when train is smaller);
2. kept S1 entities are dropped until pool records per S1 match the test's ratio: first the
   entities the matcher trained on (``drop_first``: their scores would be optimistic), then
   fit-side entities by hash. Their true records stay in the pool as unowned decoys, the
   way test holds records whose S1 is absent;
3. every remaining ("present") S1 entity is scored and competes in the pool-side 1-to-1,
   so a val entity meets the rivals it would meet on test.

Present entities carry a role: ``val`` (the fixed val fold), ``tune`` (the tune side of
``trainset.inner_split``) or ``fit`` (the rest of the fit side). Rules are tuned on ``tune``
and scored on ``val``; ``fit`` entities compete and can train stacked models. Val and tune
entities are never dropped in step 2, so every val entity in a kept cluster is scored.
"""
from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data import isin, load_source
from .split import Fold, hash_unit, pool_val_mask

MOCK_SEED = 5151        # cluster sampling; independent of the split and inner-split seeds
DROP_SEED = 5152        # order in which fit-side entities are dropped
ROLES = ("val", "tune", "fit")


@dataclass(frozen=True)
class Shape:
    """Record counts of one country: Source 1 entities and pool (Source 2 + 3) records."""

    s1: int
    pool: int

    @property
    def ratio(self) -> float:
        """Pool records per S1 entity."""
        return self.pool / self.s1 if self.s1 else float("nan")


def target_shape(dataset_dir: Path = C.DATASET) -> dict[str, Shape]:
    """Per-country record counts of the test split (counts only; test has no labels)."""
    counts = [load_source("test", s, dataset_dir, [C.COUNTRY])[C.COUNTRY].value_counts()
              for s in C.SOURCES]
    s1, pool = counts[0], counts[1].add(counts[2], fill_value=0)
    return {str(c): Shape(int(s1.get(c, 0)), int(pool.get(c, 0)))
            for c in s1.index.union(pool.index)}


@dataclass
class MockFold:
    """Present S1 entities, kept pool and their truth, with a role per present entity."""

    fold: Fold          # present S1, kept S2 / S3 records, truth pairs of the present S1
    role: pd.Series     # present S1 id -> "val" | "tune" | "fit" (index aligned to fold.s1)
    info: pd.DataFrame  # one row per country: how the shape was reached

    def ids(self, role: str) -> pd.Index:
        """Present S1 ids of one role."""
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        return pd.Index(self.fold.s1[C.ENTITY_ID][(self.role == role).to_numpy()])

    def part(self, role: str) -> Fold:
        """The present entities of ``role`` with their truth pairs and the whole kept pool."""
        ids = self.ids(role)
        s1 = self.fold.s1[isin(self.fold.s1[C.ENTITY_ID], ids)].reset_index(drop=True)
        pairs = self.fold.pairs[isin(self.fold.pairs[C.S1_ID], ids)].reset_index(drop=True)
        return Fold(f"mock_{role}", s1, self.fold.s2, self.fold.s3, pairs)


def build_mock(train: Fold, val: Fold, tune_ids: Collection[str],
               drop_first: Collection[str], shape: dict[str, Shape],
               seed: int = MOCK_SEED, drop_seed: int = DROP_SEED) -> MockFold:
    """The mock fold of ``train`` + ``val`` shaped like ``shape`` (see the module docstring).

    ``train`` and ``val`` must carry the ``country`` column; ``tune_ids`` are the train-fold
    S1 on the tune side of the inner split; ``drop_first`` are S1 ids dropped before any
    other (the matcher's training sample). A country missing from ``shape`` is kept whole.
    The result depends on ids and seeds only, never on row order.
    """
    frames = [train.s1, val.s1, train.s2, val.s2, train.s3, val.s3]
    if any(C.COUNTRY not in df.columns for df in frames):
        raise ValueError("train and val folds must be loaded with the country column")
    s1 = pd.concat([train.s1, val.s1], ignore_index=True)
    s2 = pd.concat([train.s2, val.s2], ignore_index=True)
    s3 = pd.concat([train.s3, val.s3], ignore_index=True)
    truth = pd.concat([train.pairs, val.pairs], ignore_index=True)
    ids = s1[C.ENTITY_ID]
    role = np.where(isin(ids, pd.Index(val.s1[C.ENTITY_ID])), "val",
                    np.where(isin(ids, pd.Index(list(tune_ids))), "tune", "fit"))
    first = isin(ids, pd.Index(list(drop_first)))
    if (first & (role != "fit")).any():
        raise ValueError("drop_first holds val or tune entities; only fit-side ones may go")
    cluster_rank, drop_rank = hash_unit(ids, seed), hash_unit(ids, drop_seed)
    s1_present = np.zeros(len(s1), dtype=bool)
    pool_keep = {2: np.zeros(len(s2), dtype=bool), 3: np.zeros(len(s3), dtype=bool)}
    rows = []
    for country in sorted(s1[C.COUNTRY].unique()):
        in_c = (s1[C.COUNTRY] == country).to_numpy()
        pool_c = {k: (df[C.COUNTRY] == country).to_numpy() for k, df in ((2, s2), (3, s3))}
        n_pool = int(pool_c[2].sum() + pool_c[3].sum())
        target = shape.get(country)
        if target is not None and (target.s1 == 0 or target.pool == 0):
            target = None                     # no usable test shape: keep the country whole
        frac = 1.0 if target is None or n_pool == 0 else min(1.0, target.pool / n_pool)
        keep_c = in_c & (cluster_rank < frac)                            # step 1: clusters
        kept_ids = pd.Index(ids[keep_c])
        n_pool_kept = 0
        for k, df in ((2, s2), (3, s3)):
            m = pool_c[k].copy()
            m[m] = pool_val_mask(df[C.ENTITY_ID][m], truth, kept_ids, frac, seed)
            pool_keep[k] |= m
            n_pool_kept += int(m.sum())
        present = keep_c & ~first                                        # step 2: drops
        if target is not None:
            want = int(round(n_pool_kept / target.ratio))
            extra = int(present.sum()) - want
            spare = np.flatnonzero(present & (role == "fit"))
            if extra > 0:
                present[spare[np.argsort(drop_rank[spare], kind="stable")[:extra]]] = False
        s1_present |= present
        n_present = int(present.sum())
        rows.append({
            C.COUNTRY: country, "s1_train": int(in_c.sum()), "pool_train": n_pool,
            "keep_frac": round(frac, 4), "s1_kept": int(keep_c.sum()), "pool_kept": n_pool_kept,
            "s1_present": n_present,
            "pool_per_s1": round(n_pool_kept / n_present, 3) if n_present else float("nan"),
            "test_pool_per_s1": round(target.ratio, 3) if target else float("nan"),
            **{f"present_{r}": int((present & (role == r)).sum()) for r in ROLES},
        })
    present_ids = pd.Index(ids[s1_present])
    fold = Fold("mock", s1[s1_present].reset_index(drop=True),
                s2[pool_keep[2]].reset_index(drop=True), s3[pool_keep[3]].reset_index(drop=True),
                truth[isin(truth[C.S1_ID], present_ids)].reset_index(drop=True))
    info = pd.DataFrame(rows)
    owned = isin(pd.concat([fold.s2[C.ENTITY_ID], fold.s3[C.ENTITY_ID]], ignore_index=True),
                 pd.Index(fold.pairs[C.ENTITY_ID]))
    info.attrs["unowned_share"] = float(1 - owned.mean()) if len(owned) else float("nan")
    return MockFold(fold, pd.Series(role[s1_present], index=fold.s1.index, name="role"), info)
