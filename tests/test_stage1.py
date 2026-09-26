"""Tests of the GPU stage-1 trainer (stage1.py) on synthetic chunks and the fixture."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from entity_resolution import config as C
from entity_resolution.mock import build_mock
from entity_resolution.model import Matcher, MatcherParams
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
