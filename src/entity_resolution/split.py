"""Fixed validation split of the training data, shared by every experiment.

Local F0.5 is only comparable across experiments when all of them score the same
held-out Source 1 entities against the same pool of Source 2/3 records. The split is
therefore a pure function of entity IDs and SPLIT_SEED: row order, sampling state or
machine cannot change it. Never tune it. Changing VAL_FRACTION or SPLIT_SEED starts a
new split, and local scores from before stop being comparable.

It mirrors the test set in miniature:

    * each Source 1 entity joins the validation fold with probability VAL_FRACTION,
      decided by a keyed hash of its ID;
    * a Source 2/3 record matched to a Source 1 entity follows that entity's fold, so
      all its true matches sit in the same pool;
    * unmatched Source 2/3 records are hashed with the same fraction, keeping the
      distractors that make singletons and false merges hard.
"""
from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data import load_source, load_truth_pairs

VAL_FRACTION = 0.2
SPLIT_SEED = C.SEED
FOLDS = ("train", "val")


def hash_unit(ids: pd.Series, seed: int = SPLIT_SEED) -> np.ndarray:
    """Deterministic pseudo-random number in [0, 1) per ID, independent of row order."""
    key = f"{seed:016d}"[-16:]  # hash_pandas_object wants a 16-character key
    hashed = pd.util.hash_pandas_object(ids, index=False, hash_key=key)
    return hashed.to_numpy() / 2.0**64


def s1_val_mask(s1_ids: pd.Series, frac: float = VAL_FRACTION,
                seed: int = SPLIT_SEED) -> np.ndarray:
    """True where a Source 1 entity belongs to the validation fold."""
    return hash_unit(s1_ids, seed) < frac


def pool_val_mask(pool_ids: pd.Series, truth_pairs: pd.DataFrame, val_s1: Collection[str],
                  frac: float = VAL_FRACTION, seed: int = SPLIT_SEED) -> np.ndarray:
    """True where a Source 2/3 record belongs to the validation pool.

    Matched records follow the fold of their Source 1 entity; unmatched ones are hashed.
    """
    owner = truth_pairs.drop_duplicates(C.ENTITY_ID).set_index(C.ENTITY_ID)[C.S1_ID]
    owner_of = pool_ids.map(owner)
    matched = owner_of.notna().to_numpy()
    return np.where(matched, owner_of.isin(val_s1).to_numpy(), hash_unit(pool_ids, seed) < frac)


@dataclass
class Fold:
    """Records and labels of one fold: ``val`` (held out) or ``train`` (fit models on)."""

    name: str
    s1: pd.DataFrame
    s2: pd.DataFrame
    s3: pd.DataFrame
    pairs: pd.DataFrame  # true (source1_entity_id, entity_id) pairs inside the fold

    def truth(self) -> dict[str, set[str]]:
        """Source 1 id -> set of true matches, singletons included as empty sets."""
        out: dict[str, set[str]] = {s1: set() for s1 in self.s1[C.ENTITY_ID]}
        for s1, match in zip(self.pairs[C.S1_ID], self.pairs[C.ENTITY_ID], strict=True):
            out[s1].add(match)
        return out

    def summary(self) -> dict[str, int | float]:
        n_s1 = len(self.s1)
        matched = self.pairs[C.S1_ID].nunique()
        return {"fold": self.name, "s1": n_s1, "s2": len(self.s2), "s3": len(self.s3),
                "true_pairs": len(self.pairs),
                "singleton_share": round(1 - matched / n_s1, 4) if n_s1 else float("nan")}


def load_fold(fold: str = "val", dataset_dir: Path = C.DATASET,
              columns: list[str] | None = None, frac: float = VAL_FRACTION,
              seed: int = SPLIT_SEED) -> Fold:
    """One fold of the fixed split, from the (cached) train files.

    ``columns`` limits the source columns loaded; ``entity_id`` is always included.
    """
    if fold not in FOLDS:
        raise ValueError(f"fold must be one of {FOLDS}, got {fold!r}")
    cols = None if columns is None else list(dict.fromkeys([C.ENTITY_ID, *columns]))
    want_val = fold == "val"
    pairs = load_truth_pairs(dataset_dir)

    s1 = load_source("train", 1, dataset_dir, cols)
    s1_val = s1_val_mask(s1[C.ENTITY_ID], frac, seed)
    val_ids = set(s1.loc[s1_val, C.ENTITY_ID])
    s1 = s1[s1_val == want_val].reset_index(drop=True)

    pool = {}
    for source in (2, 3):
        df = load_source("train", source, dataset_dir, cols)
        mask = pool_val_mask(df[C.ENTITY_ID], pairs, val_ids, frac, seed)
        pool[source] = df[mask == want_val].reset_index(drop=True)

    keep = pairs[C.S1_ID].isin(val_ids) == want_val
    return Fold(fold, s1, pool[2], pool[3], pairs[keep].reset_index(drop=True))
