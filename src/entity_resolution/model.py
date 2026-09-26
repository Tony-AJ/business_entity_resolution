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
"""
from __future__ import annotations

import json
import time
import warnings
from collections import Counter
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

BACKENDS = ("lgbm", "xgb", "logreg", "heuristic")
HEURISTIC_SIMS = ("sim_name_char", "sim_name_addr_word", "sim_addr_char")
EXACT_FLAG = "pass_exact"
PARAMS_FILE = "params.json"
FEATURES_FILE = "feature_names.json"
MODEL_FILES = {"lgbm": "model.txt", "xgb": "model.ubj",
               "logreg": "model.joblib"}  # heuristic: JSON files only
TUNE = "tune"  # name of the early-stopping set in LightGBM's evaluation log


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
        ``X`` is fine.
        """
        t0 = time.perf_counter()
        backend = self.params.backend
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
            model, best = self._fit_lgbm(X, labels, w, X_val, val_labels)
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
            "rows": len(X),
            "positive_rate": float(labels.mean()) if len(labels) else None,
            "best_iteration": best,
            "tune_logloss": tune_logloss,
            "tune_auc": tune_auc,
            "fit_seconds": round(time.perf_counter() - t0, 2),
        }
        return self

    def _fit_lgbm(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        weight: np.ndarray | None,
        X_val: pd.DataFrame | None,
        y_val: np.ndarray | None,
    ) -> tuple[lgb.Booster, int]:
        """Train the booster; return it with the number of rounds to predict with."""
        p = self.params
        train = lgb.Dataset(_to_float32(X), y, weight=weight, feature_name=list(X.columns),
                            free_raw_data=True)
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
            return self.model_.predict(data, num_iteration=self.best_iteration_,
                                       num_threads=self.params.num_threads)
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

    def _fitted_names(self) -> list[str]:
        """The fitted column order; raises when neither ``fit`` nor ``load`` ran."""
        if self.feature_names_ is None:
            raise RuntimeError("Matcher is not fitted: call fit() or Matcher.load() first")
        return self.feature_names_




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


class SeedMean:
    """Matchers trained on different seeds or data slices, averaged.

    Stands in for a ``Matcher`` wherever a fitted model predicts: a stage-2 part of several
    seeds (``predict_stage2``, ``TwoStage``) or a bagged stage 1 (``stage1.fit_stage1``,
    ``pipeline.Fitted``): same ``feature_names_``, ``predict_proba`` = the mean probability.
    """

    def __init__(self, models: list[Matcher]) -> None:
        if not models:
            raise ValueError("SeedMean needs at least one model")
        names = list(models[0].feature_names_)
        if any(list(m.feature_names_) != names for m in models):
            raise ValueError("SeedMean models must read the same features")
        self.models, self.feature_names_ = list(models), names
        self.fit_info_: dict = {}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Mean match probability of the seeds per row of ``X``, float32."""
        return np.mean([m.predict_proba(X) for m in self.models], axis=0).astype(np.float32)

    def importance(self) -> pd.Series:
        """The seeds' mean gain share per feature, largest first."""
        return (pd.concat([m.importance() for m in self.models], axis=1).mean(axis=1)
                .sort_values(ascending=False))

    def save(self, out: Path) -> Path:
        """Each seed's model under ``out/seed<i>`` (``Matcher.save``)."""
        for i, m in enumerate(self.models):
            m.save(Path(out) / f"seed{i}")
        return Path(out)

    @classmethod
    def load(cls, out: Path) -> SeedMean:
        """Read back what ``save`` wrote, seeds in their saved order."""
        dirs = sorted(Path(out).glob("seed*"), key=lambda d: int(d.name[len("seed"):]))
        return cls([Matcher.load(d) for d in dirs])


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
