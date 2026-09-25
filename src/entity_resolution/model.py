"""Pairwise matcher: one match probability per candidate pair (plan group D).

Blueprint: docs/plan/08_MODEL_SELECTION.md. ``Matcher`` takes the float32 feature frame of
``features.build_features`` (columns ``features.feature_names(groups)``) with the 0/1
``label`` of ``trainset.label_pairs`` and returns one probability per pair, which
``decision.decide`` turns into match sets. Three backends share one interface:

    lgbm        LightGBM, the V1 model: native NaN, early stopping on the tune set
    logreg      D1 baseline: NaN -> -1 plus missing flags, scaling, logistic regression
    heuristic   no learning: best blocking similarity, 1.0 for exact-key pairs

The fitted column order is a contract: ``predict_proba`` refuses a frame whose columns
are missing, unexpected or reordered, because two swapped similarity columns would
otherwise score garbage without any error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BACKENDS = ("lgbm", "logreg", "heuristic")
HEURISTIC_SIMS = ("sim_name_char", "sim_name_addr_word", "sim_addr_char")
EXACT_FLAG = "pass_exact"
PARAMS_FILE = "params.json"
FEATURES_FILE = "feature_names.json"
MODEL_FILES = {"lgbm": "model.txt", "logreg": "model.joblib"}  # heuristic: JSON files only
TUNE = "tune"  # name of the early-stopping set in LightGBM's evaluation log


@dataclass
class MatcherParams:
    """Backend and LightGBM hyper-parameters; the defaults are the V1 point of 08 §3.

    Every field but ``backend``, ``n_estimators`` and ``early_stopping`` is the LightGBM
    parameter of the same name. ``logreg`` and ``heuristic`` ignore them all.
    """

    backend: str = "lgbm"            # "lgbm" | "logreg" | "heuristic"
    num_leaves: int = 63
    learning_rate: float = 0.05
    n_estimators: int = 2000         # boosting-round ceiling; early stopping picks the best
    early_stopping: int = 100        # rounds without a tune-logloss gain before stopping
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    min_data_in_leaf: int = 200
    lambda_l2: float = 1.0
    max_bin: int = 255
    scale_pos_weight: float = 1.0    # stays 1.0: reweighting would move every threshold
    seed: int = 42
    num_threads: int = 12

    def __post_init__(self) -> None:
        """Reject an unknown backend at construction, not after a long feature build."""
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")


def lgbm_params(p: MatcherParams) -> dict[str, object]:
    """LightGBM parameters for ``p`` (08 §1.1); everything else stays at LightGBM defaults.

    ``deterministic`` with ``force_row_wise`` makes the same seed and thread count grow
    the same trees (about 10% slower), which the KEEP margin of 13 §3 relies on.
    """
    return {
        "objective": "binary", "metric": "binary_logloss", "verbose": -1,
        "deterministic": True, "force_row_wise": True,
        "num_leaves": p.num_leaves, "learning_rate": p.learning_rate,
        "feature_fraction": p.feature_fraction, "bagging_fraction": p.bagging_fraction,
        "bagging_freq": p.bagging_freq, "min_data_in_leaf": p.min_data_in_leaf,
        "lambda_l2": p.lambda_l2, "max_bin": p.max_bin,
        "scale_pos_weight": p.scale_pos_weight, "seed": p.seed, "num_threads": p.num_threads,
    }


def logreg_pipeline() -> Pipeline:
    """The D1 baseline: NaN -> -1 plus one missing flag per NaN-bearing feature, scaled LR.

    -1 lies outside every similarity's [0, 1] range and the flags let the linear model
    price "undefined" apart from "low". ``keep_empty_features`` keeps a column that is
    all NaN in the fit sample (``sim_addr_char`` while pass P4 is off) instead of
    dropping it, so the model's inputs always line up with ``feature_names_``.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=-1, add_indicator=True,
                                 keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=300)),
    ])


def _to_float32(frame: pd.DataFrame) -> np.ndarray:
    """Frame values as float32, missing (NaN or pd.NA) as NaN; one float32 block is not copied."""
    return frame.to_numpy(dtype=np.float32, na_value=np.nan)


def heuristic_proba(X: pd.DataFrame) -> np.ndarray:
    """``heuristic`` backend: the best blocking similarity, 1.0 for exact-key pairs.

    ``nanmax(sim_name_char, sim_name_addr_word, sim_addr_char)`` clipped to [0, 1]; 0.0
    where all three are NaN (no similarity pass produced the pair); 1.0 where
    ``pass_exact == 1``, which wins over the NaN rule: a pair found only by an exact name
    key is the strongest evidence, not the weakest.
    """
    sims = _to_float32(X[list(HEURISTIC_SIMS)])
    prob = np.fmax.reduce(sims, axis=1)  # NaN-skipping max, without nanmax's all-NaN warning
    prob[np.isnan(prob)] = 0.0
    np.clip(prob, 0.0, 1.0, out=prob)    # float32 cosines can overshoot 1.0 by one ulp
    prob[_to_float32(X[[EXACT_FLAG]])[:, 0] == 1] = 1.0
    return prob
