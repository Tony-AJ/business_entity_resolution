"""Feature snapshots (snapshot.py), starting with their synthetic data.

The generated dataset of tests/snapshot_fixtures.py is what the snapshot tests train
LightGBM on, so it is checked first: deterministic, and varied enough to hold singletons,
multi-match entities and both countries. Nothing here reads the real dataset/.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from entity_resolution import config as C
from entity_resolution.pipeline import pool_of
from entity_resolution.split import load_fold
from snapshot_fixtures import make_dataset


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
