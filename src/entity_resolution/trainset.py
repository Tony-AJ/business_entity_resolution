"""Training data for the matcher and the decision layer (07 §5, 11 §1.2).

    inner_split   the train fold split into (fit, tune) the way ``split.load_fold`` splits
                  train/val, one level down: tune gets 25% of S1 by id hash (seed 4242),
                  matched pool records follow their S1 owner, unmatched ones are hashed
    sample_s1     S1 entities drawn by id hash: order-free, deterministic, nested in ``n``
    label_pairs   candidate pairs plus ``label`` int8 (1 = true pair), row order untouched

Sample S1 entities, never pairs: every true match and every decoy of a sampled entity stays,
so the class balance and the group context match inference. The val fold is never read here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import isin
from .evaluate import pair_in
from .split import Fold, hash_unit, pool_val_mask

INNER_SEED = 4242   # independent of split.SPLIT_SEED, so the inner hash is not the outer one
INNER_FRAC = 0.25   # tune share of the train fold's S1 entities
SAMPLE_SEED = 7     # sample_s1 default: the same S1 sample in every version


def inner_split(train: Fold) -> tuple[Fold, Fold]:
    """``(fit, tune)`` folds of ``train``, named "fit" and "tune" (11 §1.2).

    The construction of ``split.load_fold`` keyed with ``INNER_SEED``: an S1 entity is on tune
    when ``hash_unit(id, 4242) < 0.25``; ``pool_val_mask`` is reused verbatim, so every true
    pair sits with its S1 entity, a pool record lands on exactly one side, and singletons and
    unmatched decoys fall on both sides in their natural share.
    """
    s1_ids = train.s1[C.ENTITY_ID]
    tune_s1 = hash_unit(s1_ids, INNER_SEED) < INNER_FRAC           # S1 by id hash
    tune_ids = pd.Index(s1_ids[tune_s1])

    def tune_side(pool: pd.DataFrame) -> np.ndarray:
        """Matched pool row -> the side of its S1 owner; unmatched -> its own hash < 0.25."""
        return pool_val_mask(pool[C.ENTITY_ID], train.pairs, tune_ids, INNER_FRAC, INNER_SEED)

    def part(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
        """Rows of ``df`` where ``mask`` holds, with a fresh index."""
        return df[mask].reset_index(drop=True)

    m2, m3 = tune_side(train.s2), tune_side(train.s3)
    in_tune = isin(train.pairs[C.S1_ID], tune_ids)                 # pairs follow their S1
    tune = Fold("tune", part(train.s1, tune_s1), part(train.s2, m2), part(train.s3, m3),
                part(train.pairs, in_tune))
    fit = Fold("fit", part(train.s1, ~tune_s1), part(train.s2, ~m2), part(train.s3, ~m3),
               part(train.pairs, ~in_tune))
    return fit, tune


def sample_s1(s1: pd.DataFrame, n: int, seed: int = SAMPLE_SEED) -> pd.DataFrame:
    """About ``n`` rows of ``s1``, chosen by id hash; all rows when ``n >= len(s1)``.

    Keeps the rows with ``hash_unit(entity_id, seed) < n / len(s1)``: the choice depends on the
    ids only, never on row order or sampling state, and a larger ``n`` keeps a superset. The
    count is binomial around ``n`` (±√n). Original row order is kept, the index is reset.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n >= len(s1):
        return s1.reset_index(drop=True)
    keep = hash_unit(s1[C.ENTITY_ID], seed) < n / len(s1)
    return s1[keep].reset_index(drop=True)


def label_pairs(pairs: pd.DataFrame, truth_pairs: pd.DataFrame) -> pd.DataFrame:
    """``pairs`` with the same rows, order and index, plus ``label`` int8: 1 for a true pair.

    A truth pair missing from ``pairs`` gets no row: a blocking miss is not trainable, only
    counted by the metric. Membership is a binary search on int64 pair keys
    (``evaluate.pair_in``) instead of a string merge, so 30M candidate pairs take seconds and
    the row order cannot change.
    """
    return pairs.assign(label=pair_in(pairs, truth_pairs).astype(np.int8))
