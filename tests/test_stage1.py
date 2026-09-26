"""Tests of the GPU stage-1 trainer (stage1.py) on synthetic chunks and the fixture."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from entity_resolution import config as C
from entity_resolution.decision import DecisionRule
from entity_resolution.mock import build_mock
from entity_resolution.model import Matcher, MatcherParams, SeedMean
from entity_resolution.pipeline import Fitted, PipelineConfig
from entity_resolution.split import load_fold
from entity_resolution.stage1 import absent_from_mock, fit_stage1, write_chunks
from test_pipeline import tiny_cfg


def test_fit_stage1_on_disk_chunks(tmp_path: Path) -> None:
    """Streams chunks, holds out the hashed slice, returns a working xgb Matcher."""
    rng = np.random.default_rng(0)
    stems = []
    for k in range(3):
        X = rng.random((4000, 5), dtype=np.float32)
        y = (X[:, 0] + 0.2 * rng.standard_normal(4000) > 0.5).astype(np.int8)
        stem = tmp_path / f"c_{k}"
        np.save(f"{stem}_X.npy", X)
        np.save(f"{stem}_y.npy", y)
        np.save(f"{stem}_h.npy", rng.random(4000).astype(np.float32))
        stems.append(str(stem))
    manifest = {"chunks": stems, "rows": 12000, "positives": 6000,
                "features": [f"f{i}" for i in range(5)], "entities": 100}
    m = fit_stage1(manifest, MatcherParams(backend="xgb", device="cpu", n_estimators=50,
                                           early_stopping=10, num_threads=2), stop_frac=0.2)
    assert m.fit_info_["tune_auc"] > 0.8 and 0 < m.best_iteration_ <= 50
    assert 10000 > m.fit_info_["rows"] > 8000                  # ~20 % held out
    X = pd.DataFrame(rng.random((10, 5), dtype=np.float32), columns=manifest["features"])
    again = Matcher.load(m.save(tmp_path / "m"))
    assert np.allclose(again.predict_proba(X), m.predict_proba(X))


def test_write_chunks_and_absent_ids(dataset_dir: Path, tmp_path: Path) -> None:
    """Every entity absent from the mock is trainable; chunks hold labelled features."""
    cfg = tiny_cfg(tmp_path, dataset_dir)
    train = load_fold("train", dataset_dir, columns=[C.COUNTRY], frac=0.5)
    val = load_fold("val", dataset_dir, columns=[C.COUNTRY], frac=0.5)
    drop = train.s1[C.ENTITY_ID][:1]
    mock = build_mock(train, val, tune_ids=[], drop_first=drop, shape={})
    ids = absent_from_mock(mock, train)
    assert list(ids) == list(drop)
    manifest = write_chunks(cfg, train, ids, {}, tmp_path / "chunks")
    saved = json.loads((tmp_path / "chunks" / "manifest.json").read_text())
    assert saved == manifest and manifest["entities"] == 1
    for stem in manifest["chunks"]:
        X = np.load(f"{stem}_X.npy")
        assert X.shape[1] == len(manifest["features"]) and X.dtype == np.float32
        assert len(np.load(f"{stem}_y.npy")) == len(X) == len(np.load(f"{stem}_h.npy"))


def test_fit_stage1_caps_training_rows(tmp_path: Path) -> None:
    """max_rows thins the training entities by hash; the held-out slice is untouched."""
    rng = np.random.default_rng(1)
    X = rng.random((6000, 3), dtype=np.float32)
    y = (X[:, 0] > 0.5).astype(np.int8)
    stem = tmp_path / "c_0"
    np.save(f"{stem}_X.npy", X)
    np.save(f"{stem}_y.npy", y)
    np.save(f"{stem}_h.npy", rng.random(6000).astype(np.float32))
    manifest = {"chunks": [str(stem)], "rows": 6000, "positives": int(y.sum()),
                "features": ["a", "b", "c"], "entities": 6000}
    m = fit_stage1(manifest, MatcherParams(backend="xgb", device="cpu", n_estimators=20,
                                           early_stopping=5, num_threads=2),
                   stop_frac=0.1, max_rows=2700)
    assert 2400 < m.fit_info_["rows"] < 3000 and m.fit_info_["entity_share_used"] == 0.5


def test_fit_stage1_bags_cover_every_training_row(tmp_path: Path) -> None:
    """bags=2 trains on two disjoint hash windows and averages them; Fitted round-trips."""
    rng = np.random.default_rng(2)
    X = rng.random((6000, 3), dtype=np.float32)
    y = (X[:, 0] + 0.2 * rng.standard_normal(6000) > 0.5).astype(np.int8)
    stem = tmp_path / "c_0"
    np.save(f"{stem}_X.npy", X)
    np.save(f"{stem}_y.npy", y)
    np.save(f"{stem}_h.npy", rng.random(6000).astype(np.float32))
    manifest = {"chunks": [str(stem)], "rows": 6000, "positives": int(y.sum()),
                "features": ["a", "b", "c"], "entities": 6000}
    params = MatcherParams(backend="xgb", device="cpu", n_estimators=20, early_stopping=5,
                           num_threads=2)
    m = fit_stage1(manifest, params, stop_frac=0.1, max_rows=2700, bags=2)
    assert isinstance(m, SeedMean) and len(m.models) == 2
    info = m.fit_info_
    assert info["bags"] == 2 and info["entity_share_used"] == 1.0
    assert all(2400 < r < 3000 for r in info["rows_per_bag"])   # ~45 % of 6000 each
    assert info["rows"] == sum(info["rows_per_bag"]) and info["tune_auc"] > 0.8
    Xp = pd.DataFrame(X[:50], columns=manifest["features"])
    mean = np.mean([b.predict_proba(Xp) for b in m.models], axis=0)
    assert np.allclose(m.predict_proba(Xp), mean)
    fitted = Fitted(m, DecisionRule(), pd.DataFrame({"f_beta": [np.nan]}), PipelineConfig())
    again = Fitted.load(fitted.save(tmp_path / "fitted"), PipelineConfig())
    assert isinstance(again.matcher, SeedMean)
    assert np.allclose(again.matcher.predict_proba(Xp), mean, atol=1e-6)
    single = fit_stage1(manifest, params, stop_frac=0.1, max_rows=2700)
    assert isinstance(single, Matcher) and single.fit_info_["entity_share_used"] == 0.5


def test_write_chunks_and_as_fitted_carry_learned_tables(filler_dir: Path,
                                                          tmp_path: Path) -> None:
    """With the fillers and the token evidence on, the chunks hold the nofill and tok_evidence
    groups (some filler pair reads equal), each table is required, as_fitted keeps both."""
    from dataclasses import replace
    from types import SimpleNamespace

    import pytest

    from entity_resolution.evidence import EvidenceConfig
    from entity_resolution.features import DEFAULT_GROUPS, FEATURE_COLUMNS
    from entity_resolution.normalize import NormaliseConfig
    from entity_resolution.pipeline import Fitted, learn_fillers, learn_token_evidence
    from entity_resolution.stage1 import as_fitted
    base = tiny_cfg(tmp_path, filler_dir)
    cfg = replace(base, normalise=NormaliseConfig(learn_fillers=True),
                  blocking=replace(base.blocking, nofill_max_group=50),
                  evidence=EvidenceConfig(learn=True, sample_share=1.0, min_support=1,
                                          prior=1.0),
                  feature_groups=(*DEFAULT_GROUPS, "nofill", "tok_evidence"))
    full = load_fold("train", filler_dir, frac=0.5)
    fillers, evidence = learn_fillers(cfg, full), learn_token_evidence(cfg, full)
    train = load_fold("train", filler_dir, columns=[C.COUNTRY], frac=0.5)
    ids = pd.Index(train.s1[C.ENTITY_ID])
    manifest = write_chunks(cfg, train, ids, {}, tmp_path / "chunks", fillers=fillers,
                            evidence=evidence)
    names = manifest["features"]
    assert names[-18:] == FEATURE_COLUMNS["nofill"] + FEATURE_COLUMNS["tok_evidence"]
    assert manifest["positives"] == len(ids)
    X = np.concatenate([np.load(f"{stem}_X.npy") for stem in manifest["chunks"]])
    assert X.shape[1] == len(names) and X[:, names.index("nofill_eq")].max() == 1.0
    assert X[:, names.index("te_pool_filler")].max() == 1.0      # "center" read as a filler
    with pytest.raises(ValueError, match="fillers"):
        write_chunks(cfg, train, ids, {}, tmp_path / "again", evidence=evidence)
    with pytest.raises(ValueError, match="evidence"):
        write_chunks(cfg, train, ids, {}, tmp_path / "again", fillers=fillers)
    stage1 = SimpleNamespace(fit_info_={})
    fitted = Fitted(stage1, None, pd.DataFrame(), cfg, {}, {}, fillers, evidence)
    out = as_fitted(stage1, fitted, cfg)
    assert out.fillers == fillers == ["center"] and out.token_evidence is evidence
