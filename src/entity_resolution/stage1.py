"""A stage-1 matcher trained on the GPU from features streamed off disk (plan D3 / D4).

v101, the stage 1 of the two-stage versions, learned from 200k fit-side entities blocked
against the fit pool (~60 % of the test's density). The mock fold only needs its training
entities to be absent from the mock, and far more train-fold entities are: the ones the mock
drops to reach the test's S1-per-pool ratio, and (US) the ones in clusters it does not keep.
``fit_stage1`` trains an XGBoost matcher on the GPU (``device="cuda"``) on all of them,
blocked against the train-fold pool of their country:

    candidate pairs (blocking cache)  -> features chunk by chunk  -> float32 .npy on disk
    disk chunks -> xgboost.DataIter -> QuantileDMatrix on the GPU (one chunk in RAM at a time)
    early stopping on the entities of a held-out id-hash slice

Host memory stays at one chunk of features; the GPU holds the binned matrix (~1 byte per
value). The val fold is never read: its pool records are not in the training pool.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config as C
from .data import isin
from .evidence import TokenEvidence
from .features import build_features, feature_names, iter_chunks, pool_stats
from .mock import MockFold
from .model import Matcher, MatcherParams, _tune_scores, xgb_params
from .pipeline import (
    Fitted,
    PipelineConfig,
    _tag,
    _with_frequencies,
    load_normalised,
    mem_guard,
    pool_of,
    prepare,
)
from .split import Fold, hash_unit
from .trainset import label_pairs

STOP_SEED = 7171


def absent_from_mock(mock: MockFold, train: Fold) -> pd.Index:
    """Train-fold S1 ids that are not present in the mock fold (safe to train stage 1 on)."""
    ids = train.s1[C.ENTITY_ID]
    return pd.Index(ids[~isin(ids, pd.Index(mock.fold.s1[C.ENTITY_ID]))])


def write_chunks(cfg: PipelineConfig, train: Fold, ids: pd.Index, token_map: dict,
                 work_dir: Path, tag: str = "stage1train",
                 timings: dict | None = None, fillers: list[str] | None = None,
                 evidence: TokenEvidence | None = None) -> dict:
    """Features and labels of ``ids``' candidate pairs, one .npy pair per chunk in ``work_dir``.

    Each country's entities are blocked against the train-fold pool of that country (cached
    under ``tag``); frequencies count the whole train fold, as in ``pipeline.fit``. Returns the
    manifest (also written as ``manifest.json``): chunk files, rows, positives, features.
    ``token_map``, ``fillers`` and ``evidence`` are the base version's (``Fitted``).
    """
    timings = {} if timings is None else timings
    work_dir.mkdir(parents=True, exist_ok=True)
    names = feature_names(cfg.feature_groups)
    chunks, rows, positives = [], 0, 0
    pool_ids = pool_of(train)[C.ENTITY_ID]
    s1_country = train.s1.set_index(C.ENTITY_ID)[C.COUNTRY]
    for country in sorted(train.s1[C.COUNTRY].unique()):
        t0 = time.perf_counter()
        mine = ids[(s1_country.reindex(ids) == country).to_numpy()]
        s1n = load_normalised("train", (1,), cfg, pd.Series(mine), token_map, fillers=fillers)
        pooln = load_normalised("train", (2, 3), cfg, pool_ids, token_map, country=country,
                                fillers=fillers)
        s1_all = load_normalised("train", (1,), cfg, train.s1[C.ENTITY_ID][
            (train.s1[C.COUNTRY] == country).to_numpy()], columns=[C.COUNTRY, "name_core",
                                                               "name_first"])
        s1n, pooln = _with_frequencies(cfg, s1n, pooln, s1_all)
        del s1_all
        pairs = prepare(s1n, pooln, cfg, _tag(f"{tag}_{country}", s1n, pooln, token_map),
                        fillers=fillers)
        stats = pool_stats(pooln, cfg.feature_groups) if len(pairs) else None
        for k, sl in enumerate(iter_chunks(pairs, cfg.chunk_rows)):
            part = pairs.iloc[sl]
            X = build_features(part, s1n, pooln, groups=cfg.feature_groups,
                               chunk_rows=cfg.chunk_rows, stats=stats, evidence=evidence)
            y = label_pairs(part, train.pairs)["label"].to_numpy(np.int8)
            stop = hash_unit(part[C.S1_ID], STOP_SEED)
            stem = work_dir / f"{country}_{k:04d}"
            np.save(f"{stem}_X.npy", X.to_numpy(dtype=np.float32))
            np.save(f"{stem}_y.npy", y)
            np.save(f"{stem}_h.npy", stop.astype(np.float32))
            chunks.append(str(stem))
            rows += len(y)
            positives += int(y.sum())
            del X, part
        timings[f"stage1_set_{country}_seconds"] = round(time.perf_counter() - t0, 2)
        del s1n, pooln, pairs
        mem_guard(f"stage1 set {country}")
    manifest = {"chunks": chunks, "rows": rows, "positives": positives, "features": names,
                "entities": len(ids)}
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


# Rows x features that train on a 4 GB card: 12.5M x 53 peaked at 3.7 GB and 18M x 53 ran
# out of memory, so the budget is 10M x 53.
GPU_CELLS = 530_000_000


class _Chunks(xgb.DataIter):
    """Disk chunks of ``write_chunks``, rows of the training or the held-out slice."""

    def __init__(self, stems: list[str], names: list[str], stop_frac: float,
                 held_out: bool, keep_frac: float = 1.0) -> None:
        self._stems, self._names, self._frac, self._held = stems, names, stop_frac, held_out
        self._keep = keep_frac          # share of the training entities used (by id hash)
        self._i = 0
        super().__init__()

    def next(self, input_data) -> bool:
        """Hand the next chunk (its training or held-out rows) to XGBoost."""
        while self._i < len(self._stems):
            stem = self._stems[self._i]
            self._i += 1
            h = np.load(f"{stem}_h.npy")
            if self._held:
                rows = h < self._frac
            else:                       # training slice, thinned by entity hash if asked
                rows = (h >= self._frac) & ((h - self._frac) < self._keep * (1 - self._frac))
            if not rows.any():
                continue
            X = np.load(f"{stem}_X.npy", mmap_mode="r")[rows]
            y = np.load(f"{stem}_y.npy")[rows]
            input_data(data=np.ascontiguousarray(X), label=y, feature_names=self._names)
            return True
        return False

    def reset(self) -> None:
        """Start over (XGBoost reads the data twice: sketch, then bin)."""
        self._i = 0


def fit_stage1(manifest: dict, params: MatcherParams, stop_frac: float = 0.05,
               max_rows: int | None = None) -> Matcher:
    """XGBoost on the chunks of ``manifest``; early stopping on the held-out id slice.

    ``params`` should say ``backend="xgb"``; ``device="cuda"`` trains on the GPU. At most
    ``max_rows`` training rows are used (whole entities, by id hash), by default
    ``GPU_CELLS`` / features. Returns a ``Matcher`` (feature names = the manifest's) usable
    as any fitted matcher.
    """
    if params.backend != "xgb":
        raise ValueError("fit_stage1 trains the xgb backend only")
    t0 = time.perf_counter()
    names = manifest["features"]
    if max_rows is None:
        max_rows = GPU_CELLS // len(names)
    keep = min(1.0, max_rows / max(manifest["rows"] * (1 - stop_frac), 1))
    train = xgb.QuantileDMatrix(_Chunks(manifest["chunks"], names, stop_frac, False, keep),
                                max_bin=params.max_bin)
    held = xgb.QuantileDMatrix(_Chunks(manifest["chunks"], names, stop_frac, True),
                               max_bin=params.max_bin, ref=train)
    booster = xgb.train(xgb_params(params), train, num_boost_round=params.n_estimators,
                        evals=[(held, "tune")], early_stopping_rounds=params.early_stopping,
                        verbose_eval=False)
    m = Matcher(params)
    m.model_, m.feature_names_ = booster, list(names)
    m.best_iteration_ = int(booster.best_iteration + 1)
    held_y = held.get_label()
    prob = booster.predict(held, iteration_range=(0, m.best_iteration_))
    logloss, auc = _tune_scores(held_y.astype(np.int8), prob.astype(np.float32))
    m.fit_info_ = {"rows": int(train.num_row()), "positive_rate":
                   manifest["positives"] / max(manifest["rows"], 1),
                   "best_iteration": m.best_iteration_, "tune_logloss": logloss,
                   "tune_auc": auc, "fit_seconds": round(time.perf_counter() - t0, 2),
                   "entity_share_used": keep}
    return m


def as_fitted(matcher: Matcher, base: Fitted, cfg: PipelineConfig) -> Fitted:
    """A pipeline version that is ``base`` (token map, fillers, evidence, rule) with
    ``matcher``."""
    return Fitted(matcher, base.rule, base.tune_table, cfg, base.token_map,
                  {"stage1": "xgb gpu", "fit_info": matcher.fit_info_}, base.fillers,
                  base.token_evidence)


def clean(work_dir: Path) -> None:
    """Remove the feature chunks once the model is trained (they are ~4 bytes x cells)."""
    shutil.rmtree(work_dir, ignore_errors=True)
