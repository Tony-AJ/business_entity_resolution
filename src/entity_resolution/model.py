"""Pairwise matcher: one match probability per candidate pair (plan group D).

Blueprint: docs/plan/08_MODEL_SELECTION.md. ``Matcher`` takes the float32 feature frame of
``features.build_features`` (columns ``features.feature_names(groups)``) with the 0/1
``label`` of ``trainset.label_pairs`` and returns one probability per pair, which
``decision.decide`` turns into match sets. Four backends share one interface:

    lgbm        LightGBM, the V1 model: native NaN, early stopping on the tune set;
                ``device="gpu"`` trains on the GPU through OpenCL
    xgb         XGBoost hist (D4): the same leaf-wise trees, ``device="cuda"`` trains and
                predicts on the GPU (RTX 2050 here), native NaN, early stopping
    logreg      D1 baseline: NaN -> -1 plus missing flags, scaling, logistic regression
    heuristic   no learning: best blocking similarity, 1.0 for exact-key pairs

The fitted column order is a contract: ``predict_proba`` refuses a frame whose columns
are missing, unexpected or reordered, because two swapped similarity columns would
otherwise score garbage without any error.

``SeedEnsemble`` averages matchers that differ only in their seed (08 §9) behind the same
interface, and ``fit_matcher`` fits one ``Matcher`` or such an ensemble from one call.
``reliability`` measures calibration (08 §5) of any probabilities against 0/1 labels:
ECE over equal-count bins, Brier score and the reliability table, because the decision
layer's thresholds assume the probabilities mean what they say.
"""
from __future__ import annotations

import json
import math
import os
import time
import warnings
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # scikit-learn's compiled tree walker (BSD-3, pinned): a private API, so optional
    from sklearn.ensemble._hist_gradient_boosting._predictor import _predict_from_binned_data
    from sklearn.ensemble._hist_gradient_boosting.common import PREDICTOR_RECORD_DTYPE
except ImportError:  # pragma: no cover - another scikit-learn: LightGBM predicts alone
    _predict_from_binned_data = PREDICTOR_RECORD_DTYPE = None

BACKENDS = ("lgbm", "xgb", "logreg", "heuristic")
HEURISTIC_SIMS = ("sim_name_char", "sim_name_addr_word", "sim_addr_char")
EXACT_FLAG = "pass_exact"
PARAMS_FILE = "params.json"
FEATURES_FILE = "feature_names.json"
MODEL_FILES = {"lgbm": "model.txt", "xgb": "model.ubj",
               "logreg": "model.joblib"}  # heuristic: JSON files only
TUNE = "tune"  # name of the early-stopping set in LightGBM's evaluation log
CALIBRATION_BINS = 20  # equal-count bins of the reliability table (08 §5)
# prediction only: a row's value does not depend on the thread count, so every logical CPU
# is used; training keeps MatcherParams.num_threads, which does change LightGBM's trees
PREDICT_THREADS = os.cpu_count() or 1
TREE_WALK = True          # lgbm: predict through TreeWalker when the model allows it
WALK_BLOCK = 8192         # rows per TreeWalker task: their bins stay in the core's cache
GUARD_ROWS = 64           # rows of every TreeWalker chunk re-predicted by LightGBM itself
MISSING_BIN = 255         # TreeWalker's bin for NaN (thresholds use ranks 0..254)
ZERO_THRESHOLD = float(np.float32(1e-35))  # LightGBM's kZeroThreshold: |x| <= it reads as 0
EXP_LIMIT = 709.0         # math.exp raises past ~709.78 where C's exp returns inf


@dataclass
class MatcherParams:
    """Backend and LightGBM hyper-parameters; the defaults are the V1 point of 08 §3.

    Every field but ``backend``, ``n_estimators`` and ``early_stopping`` is the LightGBM
    parameter of the same name. ``logreg`` and ``heuristic`` ignore them all.
    """

    backend: str = "lgbm"            # "lgbm" | "xgb" | "logreg" | "heuristic"
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
    device: str = "cpu"              # lgbm: "cpu" | "gpu" (OpenCL); xgb: "cpu" | "cuda"
    min_child_weight: float = 1.0    # xgb only: minimum hessian sum in a leaf

    def __post_init__(self) -> None:
        """Reject an unknown backend at construction, not after a long feature build."""
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")


def lgbm_params(p: MatcherParams) -> dict[str, object]:
    """LightGBM parameters for ``p`` (08 §1.1); everything else stays at LightGBM defaults.

    ``deterministic`` with ``force_row_wise`` makes the same seed and thread count grow
    the same trees (about 10% slower), which the KEEP margin of 13 §3 relies on.
    """
    params = {
        "objective": "binary", "metric": "binary_logloss", "verbose": -1,
        "deterministic": True, "force_row_wise": True,
        "num_leaves": p.num_leaves, "learning_rate": p.learning_rate,
        "feature_fraction": p.feature_fraction, "bagging_fraction": p.bagging_fraction,
        "bagging_freq": p.bagging_freq, "min_data_in_leaf": p.min_data_in_leaf,
        "lambda_l2": p.lambda_l2, "max_bin": p.max_bin,
        "scale_pos_weight": p.scale_pos_weight, "seed": p.seed, "num_threads": p.num_threads,
    }
    if p.device != "cpu":
        params["device_type"] = p.device
    return params


def xgb_params(p: MatcherParams) -> dict[str, object]:
    """XGBoost parameters mirroring the LightGBM point: leaf-wise trees of ``num_leaves``.

    ``min_data_in_leaf`` has no XGBoost twin (``min_child_weight`` bounds the hessian sum);
    GPU hist is deterministic for a fixed seed.
    """
    return {
        "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
        "device": p.device, "grow_policy": "lossguide", "max_depth": 0,
        "max_leaves": p.num_leaves, "eta": p.learning_rate,
        "colsample_bytree": p.feature_fraction, "subsample": p.bagging_fraction,
        "min_child_weight": p.min_child_weight, "lambda": p.lambda_l2, "max_bin": p.max_bin,
        "scale_pos_weight": p.scale_pos_weight, "seed": p.seed, "nthread": p.num_threads,
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


def _libm_exp(values: np.ndarray) -> np.ndarray:
    """exp of each value through ``math.exp`` (the C library's exp, which LightGBM's
    ``std::exp`` calls too; numpy may use its own SIMD exp); inf past the float64 range."""
    big = values > EXP_LIMIT
    out = np.fromiter(map(math.exp, np.where(big, 0.0, values).tolist()), dtype=np.float64,
                      count=len(values))
    for i in np.flatnonzero(big):   # raw scores beyond +-709: never seen, kept exact anyway
        try:
            out[i] = math.exp(values[i])
        except OverflowError:
            out[i] = math.inf
    return out


class TreeWalker:
    """A LightGBM binary model predicted tree by tree over row blocks, bit-identical to
    ``Booster.predict`` (about 3.5x faster on 2000 trees here).

    Why: ``Booster.predict`` walks every tree for one row before the next row, and the nodes
    of 2000+ trees do not fit a core's cache, so most node reads miss. Here each block of
    ``WALK_BLOCK`` rows goes through one tree at a time (scikit-learn's compiled walker),
    blocks spread over ``PREDICT_THREADS`` threads.

    Exact by construction: a split sends x left iff ``x <= threshold``, and a feature's
    thresholds are at most 254 distinct floats (LightGBM bin edges), so x is replaced by its
    rank ``#{thresholds < x}`` (uint8) and the split by ``rank <= k`` with k the threshold's
    own rank: the same decision for every x. NaN gets ``MISSING_BIN`` and each split's NaN
    direction is LightGBM's (``default_left`` for missing type NaN or Zero, ``0 <= t`` for
    None, which reads NaN as 0); |x| <= ``ZERO_THRESHOLD`` reads as 0, as LightGBM's dense
    row reader drops it. Leaf values are summed from 0.0 in tree order in float64 (as
    ``GBDT::PredictRaw``) and turned into ``1 / (1 + exp(-sigmoid * raw))`` with libm's exp
    (``BinaryLogloss::ConvertOutput``). ``from_booster`` returns None for what it cannot
    express (categorical or linear trees, other objectives, > 254 thresholds on a feature, a
    Zero-missing split whose zero side differs from its default side); ``Matcher`` also
    re-predicts ``GUARD_ROWS`` rows of every chunk with LightGBM and falls back on a mismatch.
    """

    def __init__(self, trees: list[np.ndarray], edges: list[np.ndarray], sigmoid: float) -> None:
        """Node tables (``PREDICTOR_RECORD_DTYPE``), per-feature sorted thresholds, sigmoid."""
        self.trees, self.edges, self.sigmoid = trees, edges, sigmoid
        self.used = [j for j, e in enumerate(edges) if len(e)]   # features some split reads
        self._no_bitsets = np.zeros((0, 8), dtype=np.uint32)      # no categorical splits

    @classmethod
    def from_booster(cls, booster: lgb.Booster, num_iteration: int) -> TreeWalker | None:
        """The walker of ``booster``'s first ``num_iteration`` trees, or None if unsupported.

        Reads the text model (``model_to_string``), whose floats round-trip exactly (it is
        what ``Matcher.save`` writes and ``load`` predicts bit-identically from).
        """
        if _predict_from_binned_data is None:
            return None
        blocks = booster.model_to_string(num_iteration=num_iteration).split("\nTree=")
        head = dict(line.split("=", 1) for line in blocks[0].splitlines() if "=" in line)
        objective = head.get("objective", "").split()
        if not objective or objective[0] != "binary":
            return None
        sigmoid = next((float(t.split(":", 1)[1]) for t in objective if t.startswith("sigmoid:")),
                       1.0)
        n_features = int(head["max_feature_idx"]) + 1
        parsed = []
        for block in blocks[1:]:
            body = block.split("\nend of trees", 1)[0].splitlines()[1:]
            t = dict(line.split("=", 1) for line in body if "=" in line)
            if int(t.get("num_cat", 0)) or int(t.get("is_linear", 0)):
                return None
            parsed.append(t)
        splits = [t for t in parsed if int(t["num_leaves"]) > 1]
        feats = [np.array(t["split_feature"].split(), dtype=np.int64) for t in splits]
        thrs = [np.array(t["threshold"].split(), dtype=np.float64) for t in splits]
        all_f = np.concatenate(feats) if feats else np.zeros(0, np.int64)
        all_t = np.concatenate(thrs) if thrs else np.zeros(0)
        edges = [np.unique(all_t[all_f == j]) for j in range(n_features)]
        if any(len(e) >= MISSING_BIN for e in edges):
            return None
        all_rank = np.zeros(len(all_f), dtype=np.int64)   # each threshold's rank in its edges
        for j, e in enumerate(edges):
            at = all_f == j
            all_rank[at] = np.searchsorted(e, all_t[at])
        ranks = np.split(all_rank, np.cumsum([len(f) for f in feats])[:-1]) if feats else []
        trees, s = [], 0
        for t in parsed:
            leaf_value = np.array(t["leaf_value"].split(), dtype=np.float64)
            n_leaves = len(leaf_value)
            nodes = np.zeros(2 * n_leaves - 1, dtype=PREDICTOR_RECORD_DTYPE)
            nodes["is_leaf"][n_leaves - 1:] = 1   # internal nodes first, then the leaves
            nodes["value"][n_leaves - 1:] = leaf_value
            if n_leaves > 1:
                f, thr = feats[s], thrs[s]
                s += 1
                dtype = np.array(t["decision_type"].split(), dtype=np.int64)
                if (dtype & 1).any():   # kCategoricalMask
                    return None
                default_left, missing = (dtype & 2) > 0, (dtype >> 2) & 3  # 0 None 1 Zero 2 NaN
                zero_left = thr >= 0.0
                if ((missing == 1) & (default_left != zero_left)).any():
                    return None
                rank = ranks[s - 1]
                child = [np.array(t[k].split(), dtype=np.int64) for k in ("left_child",
                                                                        "right_child")]
                left, right = (np.where(c >= 0, c, n_leaves - 1 + ~c) for c in child)
                inner = slice(0, n_leaves - 1)
                nodes["feature_idx"][inner], nodes["num_threshold"][inner] = f, thr
                nodes["bin_threshold"][inner] = rank
                nodes["missing_go_to_left"][inner] = np.where(missing > 0, default_left,
                                                              zero_left)
                nodes["left"][inner], nodes["right"][inner] = left, right
            trees.append(nodes)
        return cls(trees, edges, sigmoid)

    def _bins(self, X: np.ndarray) -> np.ndarray:
        """uint8 threshold ranks of a float32 block (C order); unused features stay 0."""
        B = np.zeros(X.shape, dtype=np.uint8)
        for j in self.used:
            v = X[:, j].astype(np.float64)
            v[np.abs(v) <= ZERO_THRESHOLD] = 0.0
            b = np.searchsorted(self.edges[j], v)   # NaN sorts last; overwritten below
            b[np.isnan(v)] = MISSING_BIN
            B[:, j] = b
        return B

    def _block(self, X: np.ndarray) -> np.ndarray:
        """Probabilities of one block: every tree over all its rows, summed in tree order."""
        B = self._bins(X)
        raw = np.zeros(len(B), dtype=np.float64)
        leaf = np.empty(len(B), dtype=np.float64)
        for nodes in self.trees:
            _predict_from_binned_data(nodes, B, self._no_bitsets, MISSING_BIN, 1, leaf)
            raw += leaf
        return 1.0 / (1.0 + _libm_exp(-self.sigmoid * raw))

    def predict(self, X: np.ndarray, n_threads: int = PREDICT_THREADS) -> np.ndarray:
        """float64 probabilities of a float32 matrix, equal to ``Booster.predict``'s."""
        out = np.empty(len(X), dtype=np.float64)
        blocks = max(1, -(-len(X) // WALK_BLOCK))
        if n_threads > 1:   # whole rounds: no thread idles through the last round of a call
            blocks = -(-blocks // n_threads) * n_threads
        size = max(1, -(-len(X) // blocks))
        starts = range(0, len(X), size)

        def task(a: int) -> None:
            """One block into its slice of ``out`` (blocks never overlap)."""
            out[a:a + size] = self._block(X[a:a + size])

        if n_threads <= 1 or len(starts) <= 1:
            for a in starts:
                task(a)
        else:   # the walker releases the GIL; blocks go to whichever thread is free
            with ThreadPoolExecutor(min(n_threads, len(starts))) as pool:
                list(pool.map(task, starts))
        return out


class Matcher:
    """Pair classifier with a fixed column contract (02 §5, 08 §1).

    State after ``fit`` or ``load``: ``feature_names_`` (fitted column order),
    ``best_iteration_`` (boosting rounds used to predict; 0 for logreg and heuristic),
    ``model_`` (``lgb.Booster``, sklearn ``Pipeline`` or None) and ``fit_info_`` (rows,
    positive_rate, best_iteration, tune_logloss, tune_auc, fit_seconds; None where a
    number does not apply).
    """

    def __init__(self, params: MatcherParams | None = None) -> None:
        """Keep a copy of ``params`` (default ``MatcherParams()``).

        A copy, so that later edits of the caller's config cannot change what a fitted
        model says it was trained with.
        """
        self.params = replace(params) if params is not None else MatcherParams()
        self.feature_names_: list[str] | None = None
        self.best_iteration_: int = 0
        self.model_: lgb.Booster | xgb.Booster | Pipeline | None = None
        self.fit_info_: dict[str, float | int | None] = {}
        self._walker: TreeWalker | None = None   # lgbm fast path, see _tree_walker
        self._walker_of: tuple | None = None     # (model_, best_iteration_) it was built for

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        weight: pd.Series | np.ndarray | None = None,
    ) -> Matcher:
        """Learn from ``X`` (numeric features) and ``y`` (0/1 labels aligned to ``X``).

        ``X_val``/``y_val`` is the tune side of the inner split: LightGBM stops early on its
        binary logloss, and every backend reports tune logloss and AUC on it. Nothing is
        fitted on it. Without it, ``lgbm`` trains all ``n_estimators`` rounds and warns
        (tests only). ``weight`` (one value >= 0 per row of ``X``; None = unweighted) goes
        to the LightGBM Dataset or the logistic regression, for hard-negative experiments.
        The heuristic learns nothing: it only checks and records the columns, so an empty
        ``X`` is fine. ``lgbm`` bins ``X`` before boosting and drops its own reference, so a
        caller that passes its only reference (``fit_matcher`` with a loader) frees the raw
        matrix for the whole boosting run.
        """
        t0 = time.perf_counter()
        backend = self.params.backend
        n_rows = len(X)
        names = _feature_columns(X, "X")
        labels = _aligned_vector(y, X, "y", labels=True)
        w = None if weight is None else _aligned_vector(weight, X, "weight", labels=False)
        if (X_val is None) != (y_val is None):
            raise ValueError("pass X_val and y_val together")
        val_labels = None
        if X_val is not None:
            if mismatch := _column_mismatch(names, X_val):
                raise ValueError(f"X_val {mismatch}")
            _feature_columns(X_val, "X_val")
            val_labels = _aligned_vector(y_val, X_val, "y_val", labels=True)

        if backend == "heuristic":
            missing = [c for c in (EXACT_FLAG, *HEURISTIC_SIMS) if c not in names]
            if missing:
                raise ValueError(f"the heuristic backend needs the columns {missing}")
            model, best = None, 0
        elif len(X) == 0:
            raise ValueError(f"cannot fit the {backend} backend on an empty X")
        elif backend == "lgbm":
            train = self._lgbm_train_set(X, labels, w)
            del X   # the Dataset holds the binned rows; the float matrix is not needed again
            model, best = self._fit_lgbm(train, X_val, val_labels)
        elif backend == "xgb":
            model, best = self._fit_xgb(X, labels, w, X_val, val_labels)
        else:
            model, best = logreg_pipeline(), 0
            extra = {} if w is None else {"lr__sample_weight": w}
            model.fit(_to_float32(X), labels, **extra)
        self.feature_names_, self.model_, self.best_iteration_ = names, model, best

        tune_logloss = tune_auc = None
        if X_val is not None and len(X_val):
            tune_logloss, tune_auc = _tune_scores(val_labels, self.predict_proba(X_val))
        self.fit_info_ = {
            "rows": n_rows,
            "positive_rate": float(labels.mean()) if len(labels) else None,
            "best_iteration": best,
            "tune_logloss": tune_logloss,
            "tune_auc": tune_auc,
            "fit_seconds": round(time.perf_counter() - t0, 2),
        }
        return self

    def _lgbm_train_set(self, X: pd.DataFrame, y: np.ndarray,
                        weight: np.ndarray | None) -> lgb.Dataset:
        """The binned LightGBM training set, constructed now rather than inside ``lgb.train``.

        It is built with the training parameters, which are all ``lgb.train`` would pass to
        the lazy construction (it adds only boosting settings), so the bins and the model are
        the same; ``free_raw_data`` then drops the Dataset's view of the float matrix.
        """
        return lgb.Dataset(_to_float32(X), y, weight=weight, feature_name=list(X.columns),
                           params=lgbm_params(self.params), free_raw_data=True).construct()

    def _fit_lgbm(
        self,
        train: lgb.Dataset,
        X_val: pd.DataFrame | None,
        y_val: np.ndarray | None,
    ) -> tuple[lgb.Booster, int]:
        """Train the booster on the constructed ``train`` set; return it with the number of
        rounds to predict with."""
        p = self.params
        valid_sets, callbacks = [], []
        if X_val is None:
            warnings.warn("no tune set (X_val): training all n_estimators rounds without "
                          "early stopping; fine in tests, not for a model version",
                          UserWarning, stacklevel=3)
        elif len(X_val) == 0:
            raise ValueError("X_val is empty: early stopping needs tune rows")
        else:
            # reference=train bins the tune rows with the training histogram edges
            valid_sets.append(lgb.Dataset(_to_float32(X_val), y_val, reference=train))
            if p.early_stopping > 0:
                callbacks.append(lgb.early_stopping(p.early_stopping, first_metric_only=True,
                                                    verbose=False))
        booster = lgb.train(lgbm_params(p), train, num_boost_round=p.n_estimators,
                            valid_sets=valid_sets, valid_names=[TUNE] * len(valid_sets),
                            callbacks=callbacks)
        # best_iteration is 0 when nothing was early-stopped: every round counts then
        best = booster.best_iteration if booster.best_iteration > 0 else booster.current_iteration()
        return booster, int(best)

    def _fit_xgb(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        weight: np.ndarray | None,
        X_val: pd.DataFrame | None,
        y_val: np.ndarray | None,
    ) -> tuple[xgb.Booster, int]:
        """Train the XGBoost booster (GPU with ``device="cuda"``); return it with its rounds.

        ``QuantileDMatrix`` bins the rows once (on the GPU for cuda) instead of holding a
        float copy; the tune rows reuse the training bin edges (``ref``).
        """
        p = self.params
        names = list(X.columns)
        train = xgb.QuantileDMatrix(_to_float32(X), label=y, weight=weight, max_bin=p.max_bin,
                                    feature_names=names)
        evals, extra = [], {}
        if X_val is None:
            warnings.warn("no tune set (X_val): training all n_estimators rounds without "
                          "early stopping; fine in tests, not for a model version",
                          UserWarning, stacklevel=3)
        elif len(X_val) == 0:
            raise ValueError("X_val is empty: early stopping needs tune rows")
        else:
            evals = [(xgb.QuantileDMatrix(_to_float32(X_val), label=y_val, ref=train,
                                          max_bin=p.max_bin, feature_names=names), TUNE)]
            if p.early_stopping > 0:
                extra["early_stopping_rounds"] = p.early_stopping
        booster = xgb.train(xgb_params(p), train, num_boost_round=p.n_estimators, evals=evals,
                            verbose_eval=False, **extra)
        stopped = "early_stopping_rounds" in extra
        best = booster.best_iteration + 1 if stopped else booster.num_boosted_rounds()
        return booster, int(best)

    def predict_proba(self, X: pd.DataFrame, chunk_rows: int = 2_000_000) -> np.ndarray:
        """Match probability per row of ``X``: float32 in [0, 1], in row order.

        The columns must equal ``feature_names_`` (same names, same order); that check
        runs before any data is read and names what is missing, unexpected or reordered.
        Rows are scored ``chunk_rows`` at a time into one preallocated array, so peak
        memory is one float32 chunk rather than a list of chunks plus their concatenation.
        """
        names = self._fitted_names()
        if mismatch := _column_mismatch(names, X):
            raise ValueError(f"X {mismatch}")
        _feature_columns(X, "X")
        if chunk_rows < 1:
            raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
        out = np.empty(len(X), dtype=np.float32)
        for start in range(0, len(X), chunk_rows):
            chunk = X.iloc[start:start + chunk_rows]
            out[start:start + len(chunk)] = self._predict_chunk(chunk)
        return out

    def _predict_chunk(self, chunk: pd.DataFrame) -> np.ndarray:
        """Probabilities for one chunk whose columns are already checked."""
        backend = self.params.backend
        if backend == "heuristic":
            return heuristic_proba(chunk)
        data = _to_float32(chunk)  # LightGBM reads float32 without a float64 copy
        if backend == "lgbm":
            walker = self._tree_walker()
            if walker is not None:
                prob = walker.predict(data)
                if self._guard(data, prob):
                    return prob
            return self.model_.predict(data, num_iteration=self.best_iteration_,
                                       num_threads=PREDICT_THREADS)
        if backend == "xgb":
            return self.model_.inplace_predict(
                data, iteration_range=(0, self.best_iteration_)).astype(np.float32)
        return self.model_.predict_proba(data)[:, 1]

    def importance(self) -> pd.Series:
        """Share of the model's evidence per feature: >= 0, sums to 1, largest first.

        lgbm: total split gain over the rounds used to predict; logreg: |coefficient| on
        the standardised inputs (comparable across features), each missing-flag coefficient
        added to its feature; heuristic: 1/3 on each similarity it reads. All zeros when
        the model made no split. Ties keep ``feature_names_`` order.
        """
        names = self._fitted_names()
        backend = self.params.backend
        if backend == "lgbm":
            raw = self.model_.feature_importance("gain", iteration=self.best_iteration_)
        elif backend == "xgb":
            gain = self.model_[:self.best_iteration_].get_score(importance_type="total_gain")
            raw = [gain.get(n, 0.0) for n in names]
        elif backend == "logreg":
            coef = np.abs(self.model_.named_steps["lr"].coef_[0])
            raw = coef[:len(names)].copy()
            # flag columns follow the features; indicator_.features_ names their feature
            flagged = self.model_.named_steps["impute"].indicator_.features_
            np.add.at(raw, flagged, coef[len(names):])
        else:
            raw = [1.0 if n in HEURISTIC_SIMS else 0.0 for n in names]
        raw = np.asarray(raw, dtype=np.float64)
        total = raw.sum()
        share = raw / total if total > 0 else raw
        series = pd.Series(share, index=pd.Index(names, name="feature"), name="importance")
        return series.sort_values(ascending=False, kind="stable")

    def save(self, dir: Path) -> Path:
        """Write the fitted matcher to ``dir`` (created if needed) and return ``dir``.

        params.json (``MatcherParams`` plus ``best_iteration`` and ``fit_info``),
        feature_names.json and, by backend, model.txt (LightGBM text model, the rounds used
        to predict) or model.joblib (the sklearn pipeline); the heuristic writes the two
        JSON files only. params.json is removed first and written last, so an interrupted
        save cannot be loaded.
        """
        names = self._fitted_names()
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        (dir / PARAMS_FILE).unlink(missing_ok=True)
        backend = self.params.backend
        if backend == "lgbm":
            self.model_.save_model(str(dir / MODEL_FILES[backend]),
                                   num_iteration=self.best_iteration_)
        elif backend == "xgb":
            self.model_.save_model(str(dir / MODEL_FILES[backend]))
        elif backend == "logreg":
            joblib.dump(self.model_, dir / MODEL_FILES[backend])
        _write_json(dir / FEATURES_FILE, names)
        _write_json(dir / PARAMS_FILE, {**asdict(self.params),
                                        "best_iteration": self.best_iteration_,
                                        "fit_info": self.fit_info_})
        return dir

    @classmethod
    def load(cls, dir: Path) -> Matcher:
        """Rebuild a matcher written by ``save``; its predictions are bit-identical.

        Reads params.json first and dispatches on its ``backend``.
        """
        dir = Path(dir)
        meta = json.loads((dir / PARAMS_FILE).read_text(encoding="utf-8"))
        best_iteration = int(meta.pop("best_iteration"))
        fit_info = meta.pop("fit_info")
        unknown = sorted(set(meta) - {f.name for f in fields(MatcherParams)})
        if unknown:
            raise ValueError(f"{dir / PARAMS_FILE}: unknown MatcherParams fields {unknown}")
        matcher = cls(MatcherParams(**meta))
        names = json.loads((dir / FEATURES_FILE).read_text(encoding="utf-8"))
        backend = matcher.params.backend
        model = None
        if backend == "lgbm":
            model = lgb.Booster(model_file=str(dir / MODEL_FILES[backend]))
            if model.num_feature() != len(names):
                raise ValueError(f"{dir}: model.txt has {model.num_feature()} features, "
                                 f"feature_names.json {len(names)}")
        elif backend == "xgb":
            model = xgb.Booster(model_file=str(dir / MODEL_FILES[backend]))
            model.set_param({"device": matcher.params.device})
        elif backend == "logreg":
            model = joblib.load(dir / MODEL_FILES[backend])
        matcher.feature_names_, matcher.model_ = names, model
        matcher.best_iteration_, matcher.fit_info_ = best_iteration, fit_info
        return matcher

    def _tree_walker(self) -> TreeWalker | None:
        """The ``TreeWalker`` of the fitted booster (built once), or None when ``TREE_WALK``
        is off, the model is unsupported or a guard check failed for this model."""
        if not TREE_WALK:
            return None
        of = self._walker_of
        if of is None or of[0] is not self.model_ or of[1] != self.best_iteration_:
            self._walker = TreeWalker.from_booster(self.model_, self.best_iteration_)
            self._walker_of = (self.model_, self.best_iteration_)
        return self._walker

    def _guard(self, data: np.ndarray, prob: np.ndarray) -> bool:
        """True when LightGBM itself gives exactly ``prob`` on ``GUARD_ROWS`` spread rows.

        On a mismatch the walker is dropped for this model, with a warning, and the caller
        predicts with LightGBM: the fast path may be off, never wrong.
        """
        rows = np.unique(np.linspace(0, len(data) - 1, min(len(data), GUARD_ROWS)).astype(int))
        ref = self.model_.predict(data[rows], num_iteration=self.best_iteration_,
                                  num_threads=PREDICT_THREADS)
        if np.array_equal(ref, prob[rows]):
            return True
        warnings.warn("TreeWalker disagrees with LightGBM on this model: predicting with "
                      "LightGBM from now on", RuntimeWarning, stacklevel=4)
        self._walker = None
        return False

    def _fitted_names(self) -> list[str]:
        """The fitted column order; raises when neither ``fit`` nor ``load`` ran."""
        if self.feature_names_ is None:
            raise RuntimeError("Matcher is not fitted: call fit() or Matcher.load() first")
        return self.feature_names_


class SeedEnsemble:
    """Mean probability of matchers that differ only in their seed (08 §9, the v068 idea).

    Duck-types the parts of ``Matcher`` the pipeline and snapshot.py use: ``predict_proba``,
    ``importance``, ``feature_names_``, ``best_iteration_``, ``fit_info_``, ``params``,
    ``save``. The mean is accumulated in float64 in seed order, so it is deterministic.
    """

    def __init__(self, matchers: Sequence[Matcher]) -> None:
        """Keep the fitted matchers; they must share one column order."""
        if not matchers:
            raise ValueError("an ensemble needs at least one matcher")
        names = matchers[0].feature_names_
        if any(m.feature_names_ != names for m in matchers):
            raise ValueError("ensemble members were fitted on different columns")
        self.matchers = list(matchers)
        self.feature_names_ = names
        self.params = matchers[0].params
        self.best_iteration_ = int(round(np.mean([m.best_iteration_ for m in matchers])))
        self.fit_info_: dict = {}

    def predict_proba(self, X: pd.DataFrame, chunk_rows: int = 2_000_000) -> np.ndarray:
        """Mean of the members' probabilities, float32."""
        acc = np.zeros(len(X), dtype=np.float64)
        for m in self.matchers:
            acc += m.predict_proba(X, chunk_rows)
        return (acc / len(self.matchers)).astype(np.float32)

    def importance(self) -> pd.Series:
        """Mean of the members' importance shares, largest first (ties in column order)."""
        share = pd.concat([m.importance().reindex(self.feature_names_) for m in self.matchers],
                          axis=1).mean(axis=1)
        return share.rename("importance").sort_values(ascending=False, kind="stable")

    def save(self, dir: Path) -> Path:
        """Each member under ``seed_<seed>/`` plus ``ensemble.json`` listing them."""
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        seeds = [m.params.seed for m in self.matchers]
        for m in self.matchers:
            m.save(dir / f"seed_{m.params.seed}")
        _write_json(dir / "ensemble.json", {"seeds": seeds, "fit_info": self.fit_info_})
        return dir

    @classmethod
    def load(cls, dir: Path) -> SeedEnsemble:
        """Rebuild an ensemble written by ``save``."""
        dir = Path(dir)
        meta = json.loads((dir / "ensemble.json").read_text(encoding="utf-8"))
        ens = cls([Matcher.load(dir / f"seed_{s}") for s in meta["seeds"]])
        ens.fit_info_ = meta["fit_info"]
        return ens


def fit_matcher(params: MatcherParams, X: pd.DataFrame | Callable[[], pd.DataFrame],
                y: np.ndarray, X_stop: pd.DataFrame, y_stop: np.ndarray,
                weight: np.ndarray | None = None,
                seeds: Sequence[int] | None = None) -> Matcher | SeedEnsemble:
    """``Matcher(params).fit`` exactly as ``pipeline.fit`` calls it, or one fit per seed.

    With ``seeds`` every member is ``params`` with that ``seed``; the ensemble's tune logloss
    and AUC are recomputed on its mean probabilities over the stop set. ``X`` may be a
    loader (no argument, returns the frame) called once per fit: its frame is then
    referenced by ``Matcher.fit`` alone, so ``lgbm`` frees it once binned (the snapshot's
    evaluation keeps ~2 GB of fit rows out of memory during boosting this way).
    """
    load = X if callable(X) else (lambda: X)
    if not seeds:
        return Matcher(params).fit(load(), y, X_stop, y_stop, weight=weight)
    t0 = time.perf_counter()
    members = [Matcher(replace(params, seed=int(s))).fit(load(), y, X_stop, y_stop,
                                                          weight=weight)
               for s in seeds]
    ens = SeedEnsemble(members)
    logloss = auc = None
    if len(X_stop):
        logloss, auc = _tune_scores(np.asarray(y_stop, dtype=np.int8), ens.predict_proba(X_stop))
    ens.fit_info_ = {"rows": len(y), "positive_rate": members[0].fit_info_["positive_rate"],
                     "best_iteration": ens.best_iteration_,
                     "best_iterations": [m.best_iteration_ for m in members],
                     "tune_logloss": logloss, "tune_auc": auc,
                     "fit_seconds": round(time.perf_counter() - t0, 2)}
    return ens


def reliability(prob: np.ndarray, label: np.ndarray,
                bins: int = CALIBRATION_BINS) -> tuple[float, float, pd.DataFrame]:
    """``(ece, brier, table)`` of probabilities against 0/1 labels (08 §5).

    Equal-count bins: pairs sorted by ``prob`` (stable) and cut into ``bins`` runs of equal
    size, so every bin has the same weight whatever the skew of the scores. ECE is the
    bin-size-weighted mean |mean prob - positive rate|; Brier the mean squared error. NaN
    and an empty table without rows.
    """
    p = np.asarray(prob, dtype=np.float64)
    y = np.asarray(label, dtype=np.float64)
    cols = ["bin", "n", "p_mean", "y_rate", "gap", "p_lo", "p_hi"]
    if len(p) == 0:
        return float("nan"), float("nan"), pd.DataFrame(columns=cols)
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], y[order]
    edges = np.linspace(0, len(p), bins + 1).round().astype(np.int64)
    rows = []
    for b in range(bins):   # 20 bins: a loop over bins, never over pairs
        lo, hi = edges[b], edges[b + 1]
        if hi > lo:
            pm, yr = ps[lo:hi].mean(), ys[lo:hi].mean()
            rows.append((b, int(hi - lo), pm, yr, pm - yr, ps[lo], ps[hi - 1]))
    table = pd.DataFrame(rows, columns=cols)
    ece = float((table["n"] * table["gap"].abs()).sum() / len(p))
    brier = float(np.mean((p - y) ** 2))
    return ece, brier, table


def _feature_columns(X: pd.DataFrame, what: str) -> list[str]:
    """Column names of a feature frame, refusing non-string, repeated or non-numeric columns.

    ``object`` and string columns are refused by name: they would fail deep inside
    LightGBM or, worse, convert.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{what} must be a pandas DataFrame, got {type(X).__name__}")
    names = list(X.columns)
    if bad := [c for c in names if not isinstance(c, str)]:
        raise ValueError(f"{what}: column names must be strings, got {bad[:5]}")
    if repeated := [c for c, n in Counter(names).items() if n > 1]:
        raise ValueError(f"{what}: repeated columns {repeated}")
    if bad := [f"{c} ({dtype})" for c, dtype in X.dtypes.items()
               if not pd.api.types.is_numeric_dtype(dtype)]:
        raise ValueError(f"{what}: feature columns must be numeric (float32), got {bad[:5]}")
    return names


def _aligned_vector(
    values: pd.Series | np.ndarray, X: pd.DataFrame, what: str, *, labels: bool
) -> np.ndarray:
    """``values`` as a 1-D array with one entry per row of ``X``.

    A Series must carry ``X.index`` exactly (labels built for other rows would train
    silently wrong). Labels must be 0/1 and come back as int8; weights must be finite
    and >= 0 and come back as float64.
    """
    arr = np.asarray(values)
    if arr.ndim != 1 or len(arr) != len(X):
        raise ValueError(f"{what} must hold one value per row: shape {arr.shape}, "
                         f"{len(X)} rows")
    if isinstance(values, pd.Series) and not values.index.equals(X.index):
        raise ValueError(f"{what} is not aligned to the index of its feature frame")
    if not (np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_):
        raise ValueError(f"{what} must be numeric, got dtype {arr.dtype}")
    if labels:
        if not np.isin(arr, (0, 1)).all():
            raise ValueError(f"{what} must hold 0/1 labels only")
        return arr.astype(np.int8)
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError(f"{what} must be finite and >= 0")
    return arr


def _column_mismatch(expected: list[str], X: pd.DataFrame) -> str:
    """Why the columns of ``X`` differ from ``expected``; "" when they are identical.

    Never reorders silently: a reordered frame is an error that names the columns whose
    position changed.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame, got {type(X).__name__}")
    got = list(X.columns)
    if got == expected:
        return ""
    want, have = set(expected), set(got)
    problems = []
    if missing := [c for c in expected if c not in have]:
        problems.append(f"missing {missing}")
    if unexpected := [c for c in got if c not in want]:
        problems.append(f"unexpected {unexpected}")
    if repeated := [c for c, n in Counter(got).items() if n > 1]:
        problems.append(f"repeated {repeated}")
    # compare the order of the shared columns only, so one missing column is not
    # reported as every later column being reordered
    shared_got = [c for c in dict.fromkeys(got) if c in want]
    shared_expected = [c for c in expected if c in have]
    if moved := [c for c, e in zip(shared_got, shared_expected, strict=True) if c != e]:
        problems.append(f"reordered {moved}")
    return "columns differ from the fitted feature_names_: " + "; ".join(problems)


def _tune_scores(y: np.ndarray, prob: np.ndarray) -> tuple[float, float | None]:
    """(logloss, AUC) of float32 probabilities on the tune set; AUC None with one class.

    Computed from the probabilities the decision layer sees, the same way for every
    backend, so the numbers compare across versions.
    """
    p = prob.astype(np.float64)
    logloss = float(log_loss(y, p, labels=[0, 1]))
    auc = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None
    return logloss, auc


def _write_json(path: Path, payload: object) -> None:
    """Pretty-printed JSON with a trailing newline, as metrics.json is written."""
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
