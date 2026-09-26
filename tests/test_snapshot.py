"""Feature snapshots (snapshot.py): cache key, round trip, and equality with the pipeline.

Two synthetic datasets: the six-record conftest fixture (empty fit side, so the matcher is
the ``heuristic`` backend, as in test_pipeline.py) and the generated one of
tests/snapshot_fixtures.py, which is checked first here. Nothing here reads the real
dataset/.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from entity_resolution import config as C
from entity_resolution.decision import Grid
from entity_resolution.model import MatcherParams
from entity_resolution.pipeline import pool_of
from entity_resolution.snapshot import snapshot_key
from entity_resolution.split import load_fold
from snapshot_fixtures import make_dataset, tiny_cfg


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
