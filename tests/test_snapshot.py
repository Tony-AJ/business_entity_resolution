"""Feature snapshots (snapshot.py): round trip, cache key, and equality with the pipeline.

Two synthetic datasets: the six-record conftest fixture (empty fit side, so the matcher is
the ``heuristic`` backend, as in test_pipeline.py) and the generated one of
tests/snapshot_fixtures.py (~250 S1 entities with typo / abbreviation variants and
same-name decoys), big enough for LightGBM to train on, so ``evaluate_params`` is checked
against ``pipeline.fit`` + ``run_fold`` with the default ``lgbm`` backend. The model-level
helpers it uses (``SeedEnsemble``, ``reliability``) are unit-tested in test_model.py.
Nothing here reads the real dataset/.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from entity_resolution import config as C
from entity_resolution.decision import Grid
from entity_resolution.evaluate import error_samples, harder_fold
from entity_resolution.features import build_features
from entity_resolution.model import MatcherParams, SeedEnsemble
from entity_resolution.pipeline import (
    PipelineConfig,
    fit,
    learn_token_map,
    load_normalised,
    pool_of,
    run_fold,
    run_test,
)
from entity_resolution.snapshot import (
    SIDES,
    build_snapshot,
    error_counts,
    evaluate_params,
    load_snapshot,
    snapshot_key,
)
from entity_resolution.split import load_fold
from entity_resolution.submission import split_ids, validate
from snapshot_fixtures import make_dataset, tiny_blocking, tiny_cfg

BREAKDOWN = ("f_beta", "f_beta_singletons", "f_beta_matched", "pair_precision",
             "pair_recall", "entities", "singletons", "cand_recall", "entity_recall",
             "ceiling_f_beta", "cands_mean", "cands_p95")
FAST_LGBM = MatcherParams(num_leaves=7, learning_rate=0.1, n_estimators=80, early_stopping=10,
                          min_data_in_leaf=5, num_threads=2)
SMALL_GRID = Grid(tau_abs=(0.1, 0.9, 0.1), tau_rel=(0.0, 0.7), single_delta=(0.0, 0.1),
                  max_matches=(3, 11))


def without_timings(metrics: dict) -> str:
    """Metrics minus run times and RSS, as JSON (NaN compares equal as a string)."""
    kept = {k: v for k, v in metrics.items()
            if not k.endswith("_seconds") and k != "peak_rss_gb"}
    return json.dumps(kept, sort_keys=True, default=str)


def assert_same_as_pipeline(metrics: dict, pipe_metrics: dict, fitted, rule,
                            artifacts: dict, scored: pd.DataFrame,
                            matches: pd.DataFrame) -> None:
    """The snapshot evaluation equals pipeline.fit + run_fold, bit for bit."""
    assert rule == fitted.rule
    assert metrics["tune_f_beta"] == float(fitted.tune_table["f_beta"].max())
    pd.testing.assert_frame_equal(artifacts["tune_table"], fitted.tune_table)
    for key in BREAKDOWN:
        a, b = metrics[key], pipe_metrics[key]
        assert a == b or (np.isnan(a) and np.isnan(b)), key
    pd.testing.assert_frame_equal(artifacts["val_scored"].reset_index(drop=True),
                                  scored.reset_index(drop=True))
    pd.testing.assert_frame_equal(artifacts["val_matches"], matches)


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


def test_build_is_cached_and_resumable(generated, tmp_path: Path) -> None:
    """A second build returns the same folder untouched; a partial build is completed."""
    cfg, train, val = generated["cfg"], generated["train"], generated["val"]
    path = generated["path"]
    before = (path / "manifest.json").stat().st_mtime_ns
    assert build_snapshot(cfg, train, val, out_dir=path.parent) == path
    assert (path / "manifest.json").stat().st_mtime_ns == before
    part = build_snapshot(cfg, train, val, out_dir=tmp_path, sides=("fit", "stop"))
    assert list(load_snapshot(part).manifest["sides"]) == ["fit", "stop"]
    with pytest.raises(ValueError, match="were not built"):
        load_snapshot(part, sides=("fit", "val"))
    fit_time = (part / "fit.parquet").stat().st_mtime_ns
    assert build_snapshot(cfg, train, val, out_dir=tmp_path) == part
    assert list(load_snapshot(part).sides) == list(SIDES)
    assert (part / "fit.parquet").stat().st_mtime_ns == fit_time
    with pytest.raises(ValueError, match="unknown sides"):
        build_snapshot(cfg, train, val, out_dir=tmp_path, sides=("fit", "test"))


def test_chunking_does_not_change_the_data(generated) -> None:
    """Another chunk_rows (a new key) stores identical features for every side."""
    cfg, base = generated["cfg"], generated["base"]
    path = build_snapshot(replace(cfg, chunk_rows=97), generated["train"], generated["val"],
                          out_dir=base / "m4")
    assert path != generated["path"]
    a, b = load_snapshot(generated["path"]), load_snapshot(path)
    for side in SIDES:
        np.testing.assert_array_equal(a.features(side).to_numpy(), b.features(side).to_numpy())
        pd.testing.assert_frame_equal(a.meta(side), b.meta(side))


def test_evaluate_equals_pipeline_lgbm(generated) -> None:
    """Default-path evaluation on the snapshot = pipeline.fit + run_fold (lgbm backend)."""
    cfg, fitted = generated["cfg"], generated["fitted"]
    pipe_metrics, _, scored, matches = generated["pipe"]
    snap = load_snapshot(generated["path"])
    artifacts: dict = {}
    metrics, matcher, rule = evaluate_params(snap, cfg.model, grid=cfg.grid, artifacts=artifacts)
    assert_same_as_pipeline(metrics, pipe_metrics, fitted, rule, artifacts, scored, matches)
    assert metrics["harder_f_beta"] == generated["harder"]["f_beta"]
    assert matcher.best_iteration_ == fitted.matcher.best_iteration_
    info = fitted.info["fit_info"]
    assert metrics["tune_logloss"] == info["tune_logloss"]
    assert metrics["tune_auc"] == info["tune_auc"]
    assert metrics["importance_top20"] == {
        k: float(v) for k, v in fitted.matcher.importance().head(20).items()}
    assert set(metrics["f_beta_by_country"]) == {"India", "US"}


def test_evaluate_equals_pipeline_heuristic(dataset_dir: Path, tmp_path: Path) -> None:
    """The same on the conftest fixture, whose fit side is empty (heuristic backend)."""
    cfg = tiny_cfg(tmp_path, dataset_dir)
    train = load_fold("train", dataset_dir, frac=0.5)
    val = load_fold("val", dataset_dir, frac=0.5)
    fitted = fit(cfg, train)
    pipe_metrics, _, scored, matches = run_fold(cfg, fitted, val)
    path = build_snapshot(cfg, train, val, out_dir=tmp_path / "m4")
    artifacts: dict = {}
    metrics, _, rule = evaluate_params(load_snapshot(path), cfg.model, grid=cfg.grid,
                                       artifacts=artifacts)
    assert_same_as_pipeline(metrics, pipe_metrics, fitted, rule, artifacts, scored, matches)


def test_error_counts_equal_error_samples(generated) -> None:
    """error_counts = the v001 notebook's len(error_samples(..., n=10**9)) per kind."""
    _, _, _, matches = generated["pipe"]
    val = generated["val"]
    counts = error_counts(matches, val)
    for kind, n in counts.items():
        assert n == len(error_samples(matches, val, kind, n=10**9)), kind
    assert sum(counts.values()) > 0


def test_batch_rows_do_not_change_results(generated) -> None:
    """Scoring in batches of 7 rows gives exactly the default-batch metrics."""
    snap = load_snapshot(generated["path"])
    cfg = generated["cfg"]
    a = evaluate_params(snap, cfg.model, grid=cfg.grid)[0]
    b = evaluate_params(snap, cfg.model, grid=cfg.grid, batch_rows=7)[0]
    assert without_timings(a) == without_timings(b)


def test_seed_average_is_deterministic(generated, tmp_path: Path) -> None:
    """Two runs with seeds (1, 2) agree exactly; one seed equals the single model."""
    snap = load_snapshot(generated["path"])
    cfg = generated["cfg"]
    arts_a: dict = {}
    a, ens, _ = evaluate_params(snap, cfg.model, grid=cfg.grid, seeds=(1, 2), artifacts=arts_a)
    b, _, _ = evaluate_params(snap, cfg.model, grid=cfg.grid, seeds=(1, 2))
    assert isinstance(ens, SeedEnsemble) and len(ens.matchers) == 2
    assert without_timings(a) == without_timings(b)
    assert a["seeds"] == [1, 2] and len(ens.fit_info_["best_iterations"]) == 2
    single = evaluate_params(snap, cfg.model, grid=cfg.grid)[0]
    one = evaluate_params(snap, cfg.model, grid=cfg.grid, seeds=(cfg.model.seed,))[0]
    for key in ("f_beta", "tune_f_beta", "rule", "tune_logloss"):
        assert one[key] == single[key], key
    X = snap.features("val")
    again = SeedEnsemble.load(ens.save(tmp_path / "ens"))
    np.testing.assert_array_equal(again.predict_proba(X), ens.predict_proba(X))
    np.testing.assert_array_equal(ens.predict_proba(X), arts_a["val_scored"]["prob"].to_numpy())


def test_weight_fn_and_feature_subset(generated) -> None:
    """weight_fn gets the fit meta and must return one weight per row; columns subset."""
    snap = load_snapshot(generated["path"])
    cfg = generated["cfg"]
    seen = {}

    def hard_negatives(meta: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
        """Double weight on negatives that an exact-key pass proposed."""
        seen["cols"] = list(meta.columns)
        exact = (meta["pass"].to_numpy() & 7) != 0
        return np.where(exact & (meta["label"].to_numpy() == 0), 2.0, 1.0)

    metrics = evaluate_params(snap, cfg.model, grid=cfg.grid, weight_fn=hard_negatives)[0]
    assert seen["cols"] == [C.S1_ID, C.ENTITY_ID, C.COUNTRY, "pass", "label"]
    assert metrics["weighted"] and 0.0 <= metrics["f_beta"] <= 1.0
    with pytest.raises(ValueError, match="weight_fn returned shape"):
        evaluate_params(snap, cfg.model, grid=cfg.grid,
                        weight_fn=lambda meta, X: np.ones(len(X) + 1))
    names = snap.manifest["feature_names"]
    sub = load_snapshot(generated["path"], columns=names[:10])
    metrics, matcher, _ = evaluate_params(sub, cfg.model, grid=cfg.grid, harder=False)
    assert matcher.feature_names_ == names[:10] and metrics["n_features"] == 10
    assert "harder_f_beta" not in metrics
    with pytest.raises(ValueError, match="unknown feature columns"):
        load_snapshot(generated["path"], columns=["nope"])


def test_calibration_metrics_are_sane(generated) -> None:
    """evaluate_params reports ECE / Brier in [0, 1] and one reliability table per side and
    subset (reliability itself: test_model.py), and slice_report among its artifacts."""
    snap = load_snapshot(generated["path"])
    artifacts: dict = {}
    metrics = evaluate_params(snap, generated["cfg"].model, grid=generated["cfg"].grid,
                              artifacts=artifacts)[0]
    for key in ("ece_tune", "brier_tune", "ece_tune_rank1", "ece_val", "brier_val"):
        assert 0.0 <= metrics[key] <= 1.0, key
    rel = artifacts["reliability"]
    assert set(zip(rel["side"], rel["subset"], strict=True)) == {
        ("tune", "all"), ("tune", "rank1"), ("val", "all")}
    assert set(artifacts["slices"]["family"]) >= {"country", "singleton", "n_matches"}


def test_to_fitted_runs_test_inference(dataset_dir: Path, tmp_path: Path) -> None:
    """A snapshot-trained matcher and rule go through pipeline.run_test to valid files."""
    cfg = tiny_cfg(tmp_path, dataset_dir)
    train = load_fold("train", dataset_dir, frac=0.5)
    val = load_fold("val", dataset_dir, frac=0.5)
    snap = load_snapshot(build_snapshot(cfg, train, val, out_dir=tmp_path / "m4"))
    artifacts: dict = {}
    _, matcher, rule = evaluate_params(snap, cfg.model, grid=cfg.grid, artifacts=artifacts)
    fitted = snap.to_fitted(cfg, matcher, rule, artifacts["tune_table"])
    matching, candidates, *_ = run_test(cfg, fitted, out_dir=tmp_path / "output")
    s1_ids, valid = split_ids("test", dataset_dir, check_ids=True)
    assert validate(matching, candidates, s1_ids, valid) == ([], [])
    with pytest.raises(ValueError, match="differ from the snapshot"):
        snap.to_fitted(replace(cfg, feature_groups=("blocking",)), matcher, rule,
                       artifacts["tune_table"])
