"""Feature snapshots (snapshot.py): cache key, build, round trip, resume.

Two synthetic datasets: the six-record conftest fixture (empty fit side, so the matcher is
the ``heuristic`` backend, as in test_pipeline.py) and the generated one of
tests/snapshot_fixtures.py (~250 S1 entities with typo / abbreviation variants and
same-name decoys), big enough for LightGBM to train on, so stored rows are checked against
``pipeline.fit`` + ``run_fold`` with the default ``lgbm`` backend. Nothing here reads the
real dataset/.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from entity_resolution import config as C
from entity_resolution.decision import Grid
from entity_resolution.evaluate import harder_fold
from entity_resolution.features import build_features
from entity_resolution.model import MatcherParams
from entity_resolution.pipeline import (
    PipelineConfig,
    fit,
    learn_token_map,
    load_normalised,
    pool_of,
    run_fold,
)
from entity_resolution.snapshot import SIDES, build_snapshot, load_snapshot, snapshot_key
from entity_resolution.split import load_fold
from snapshot_fixtures import make_dataset, tiny_blocking, tiny_cfg

FAST_LGBM = MatcherParams(num_leaves=7, learning_rate=0.1, n_estimators=80, early_stopping=10,
                          min_data_in_leaf=5, num_threads=2)
SMALL_GRID = Grid(tau_abs=(0.1, 0.9, 0.1), tau_rel=(0.0, 0.7), single_delta=(0.0, 0.1),
                  max_matches=(3, 11))


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """Pipeline run and snapshot of the generated dataset with the lgbm backend.

    The snapshot is built with its own blocking cache, so it cannot reuse the pipeline's
    candidate files: equality then also covers blocking determinism.
    """
    base = tmp_path_factory.mktemp("gen")
    dataset = make_dataset(base / "dataset")
    cfg = PipelineConfig(blocking=tiny_blocking(), model=FAST_LGBM, grid=SMALL_GRID,
                         n_fit_s1=100_000, n_stop_s1=100_000, chunk_rows=300,
                         dataset_dir=dataset, cache_dir=base / "cache")
    train = load_fold("train", dataset, columns=[], frac=0.5)
    val = load_fold("val", dataset, columns=[], frac=0.5)
    fitted = fit(cfg, train)
    pipe = run_fold(cfg, fitted, val)
    harder = run_fold(cfg, fitted, harder_fold(val), tag="harder")[0]
    path = build_snapshot(replace(cfg, cache_dir=base / "cache_snap"), train, val,
                          out_dir=base / "m4")
    return {"cfg": cfg, "train": train, "val": val, "fitted": fitted, "pipe": pipe,
            "harder": harder, "path": path, "base": base}


def test_generated_dataset_is_deterministic_and_varied(tmp_path: Path) -> None:
    """Same seed, same bytes; both countries, singletons and multi-match entities occur."""
    a, b = make_dataset(tmp_path / "a", n_s1=60), make_dataset(tmp_path / "b", n_s1=60)
    files = sorted(p.relative_to(a) for p in a.rglob("*.tsv"))
    assert len(files) == 7   # three sources per split plus the ground truth
    assert all((a / f).read_bytes() == (b / f).read_bytes() for f in files)
    train, val = load_fold("train", a, frac=0.5), load_fold("val", a, frac=0.5)
    assert len(train.s1) + len(val.s1) == 60
    assert set(pd.concat([train.s1, val.s1])[C.COUNTRY]) == {"US", "India"}
    sizes = [len(v) for fold in (train, val) for v in fold.truth().values()]
    assert min(sizes) == 0 and max(sizes) >= 2
    for fold in (train, val):   # every true match sits in its fold's pool
        assert set(fold.pairs[C.ENTITY_ID]) <= set(pool_of(fold)[C.ENTITY_ID])


def test_key_changes_with_data_config_only(dataset_dir: Path, tmp_path: Path) -> None:
    """Data settings change the key; model, grid and cache location do not."""
    cfg = tiny_cfg(tmp_path, dataset_dir)
    train = load_fold("train", dataset_dir, columns=[], frac=0.5)
    val = load_fold("val", dataset_dir, columns=[], frac=0.5)
    key, parts = snapshot_key(cfg, train, val, {})
    assert parts["blocking"] == cfg.blocking.key()
    same = [replace(cfg, model=MatcherParams(num_leaves=7)), replace(cfg, grid=Grid()),
            replace(cfg, cache_dir=tmp_path / "elsewhere")]
    assert all(snapshot_key(c, train, val, {})[0] == key for c in same)
    changed = [replace(cfg, blocking=replace(cfg.blocking, max_per_s1=10)),
               replace(cfg, feature_groups=("blocking", "name_fuzzy")),
               replace(cfg, n_fit_s1=5), replace(cfg, chunk_rows=10)]
    keys = {snapshot_key(c, train, val, {})[0] for c in changed}
    assert key not in keys and len(keys) == len(changed)
    assert snapshot_key(cfg, train, val, {"a": "b"})[0] != key
    assert snapshot_key(cfg, train, train, {})[0] != key   # other fold ids


def test_generated_fixture_trains_a_real_model(generated) -> None:
    """The lgbm fixture is not degenerate: both sides have positives, rounds were used."""
    snap = load_snapshot(generated["path"])
    for side in SIDES:
        stats = snap.manifest["sides"][side]
        assert 0 < stats["positives"] < stats["rows"], side
    assert generated["fitted"].matcher.best_iteration_ > 1
    assert generated["pipe"][0]["f_beta"] > 0.5


def test_round_trip_matches_pipeline_features(generated) -> None:
    """Stored val rows are run_fold's pairs, in order, with build_features' exact values."""
    cfg, val = generated["cfg"], generated["val"]
    _, pairs, _, _ = generated["pipe"]
    snap = load_snapshot(generated["path"])
    meta = snap.meta("val")
    assert meta[C.S1_ID].tolist() == pairs[C.S1_ID].tolist()
    assert meta[C.ENTITY_ID].tolist() == pairs[C.ENTITY_ID].tolist()
    assert (meta["pass"].to_numpy() == pairs["pass"].to_numpy()).all()
    tmap = learn_token_map(cfg, generated["train"])
    s1n = load_normalised("train", (1,), cfg, val.s1[C.ENTITY_ID], tmap)
    pooln = load_normalised("train", (2, 3), cfg, pool_of(val)[C.ENTITY_ID], tmap)
    expected = build_features(pairs, s1n, pooln, groups=cfg.feature_groups)
    got = snap.features("val", batch_rows=37)
    assert list(got.columns) == snap.manifest["feature_names"] == list(expected.columns)
    assert all(dtype == np.float32 for dtype in got.dtypes)
    np.testing.assert_array_equal(got.to_numpy(), expected.to_numpy())  # NaN == NaN here
    assert snap.labels("val").sum() == snap.manifest["sides"]["val"]["positives"]
    fold = snap.fold("val")
    assert set(fold.s1[C.ENTITY_ID]) == set(val.s1[C.ENTITY_ID])
    assert len(fold.s2) + len(fold.s3) == len(val.s2) + len(val.s3)
    assert len(snap.fold("harder").s1) == len(harder_fold(val).s1)
