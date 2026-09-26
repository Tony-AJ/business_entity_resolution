"""Tests of the competition features (stacking.py) and the two-stage pipeline (twostage.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from entity_resolution import config as C
from entity_resolution.decision import Grid, tune
from entity_resolution.mock import build_mock
from entity_resolution.model import MatcherParams
from entity_resolution.pipeline import fit
from entity_resolution.split import load_fold
from entity_resolution.stacking import STACK_COLUMNS, competition_features, group_stats
from entity_resolution.submission import split_ids, validate
from entity_resolution.twostage import (
    TwoStage,
    TwoStageConfig,
    fit_stage2,
    fold_of,
    keep_mask,
    mock_scored,
    mock_stage1,
    predict_stage2,
    run_test_two_stage,
)
from test_pipeline import tiny_cfg


def test_competition_features_by_hand() -> None:
    """Ranks and best rivals on both sides; a lone record has rival 0."""
    pairs = pd.DataFrame({C.S1_ID: ["a", "a", "a", "b", "b", "c"],
                          C.ENTITY_ID: ["x", "y", "z", "x", "w", "x"]})
    p1 = np.array([0.9, 0.2, 0.6, 0.95, 0.1, 0.9], dtype=np.float32)
    f = competition_features(pairs, p1)
    assert list(f.columns) == STACK_COLUMNS
    assert f["s1_rank"].tolist() == [1, 3, 2, 1, 2, 1]
    assert f["s1_best_other"].tolist() == pytest.approx([0.6, 0.9, 0.9, 0.1, 0.95, 0.0])
    assert f["pool_best_other"].tolist() == pytest.approx([0.95, 0, 0, 0.9, 0, 0.95])
    assert f["pool_rank"].tolist() == [2, 1, 1, 1, 1, 3]     # tie a/c: input order
    assert f["pool_degree"].tolist() == [3, 1, 1, 3, 1, 3]
    assert f["s1_n_likely"].tolist() == [2, 2, 2, 1, 1, 1]
    assert f["pool_n_likely"].tolist() == [3, 0, 1, 3, 0, 3]
    assert f["s1_p1_sum"].iloc[0] == pytest.approx(1.7)
    assert f["s1_gap"].tolist() == pytest.approx([0.3, -0.7, -0.3, 0.85, -0.85, 0.9])
    assert f["pool_gap"].tolist() == pytest.approx([-0.05, 0.2, 0.6, 0.05, 0.1, -0.05])
    assert f["pool_p1_sum"].tolist() == pytest.approx([2.75, 0.2, 0.6, 2.75, 0.1, 2.75])
    with pytest.raises(ValueError, match="p1 has"):
        competition_features(pairs, p1[:3])


def test_group_stats_ties_and_empty() -> None:
    rank, other = group_stats(np.array([0, 0, 1]), np.array([0.5, 0.5, 0.2], np.float32), 2)
    assert rank.tolist() == [1, 2, 1] and other.tolist() == pytest.approx([0.5, 0.5, 0.0])
    rank, other = group_stats(np.zeros(0, np.int64), np.zeros(0, np.float32), 0)
    assert len(rank) == len(other) == 0


def test_keep_mask_floor_and_cap() -> None:
    ids = pd.Series(["a"] * 4 + ["b"] * 2)
    p1 = np.array([0.9, 0.005, 0.5, 0.4, 0.02, 0.3], dtype=np.float32)
    assert keep_mask(ids, p1, 0.01, 2).tolist() == [True, False, True, False, True, True]
    assert keep_mask(ids, p1, 0.0, 10).all()


def test_fold_of_is_stable() -> None:
    ids = pd.Series([f"S1-{i}" for i in range(1000)])
    parts = fold_of(ids, 3)
    assert set(parts) == {0, 1, 2}
    assert (fold_of(ids.iloc[::-1], 3)[::-1] == parts).all()


class Const:
    """A stand-in matcher predicting a constant."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value, dtype=np.float32)


def test_predict_stage2_out_of_fold() -> None:
    X = pd.DataFrame({"f": np.arange(4, dtype=np.float32)})
    models = [Const(0.2), Const(0.6)]
    assert predict_stage2(models, X).tolist() == pytest.approx([0.4] * 4)
    part = np.array([0, 1, -1, 1])
    assert predict_stage2(models, X, part).tolist() == pytest.approx([0.2, 0.6, 0.4, 0.6])


def test_two_stage_end_to_end(dataset_dir: Path, tmp_path: Path) -> None:
    """Stage 1, filter, stage 2 on the mock, rule, save/load, test files that validate."""
    cfg = tiny_cfg(tmp_path, dataset_dir)
    stage1 = fit(cfg, load_fold("train", dataset_dir, frac=0.5), tmp_path / "s1")
    train = load_fold("train", dataset_dir, columns=[C.COUNTRY], frac=0.5)
    val = load_fold("val", dataset_dir, columns=[C.COUNTRY], frac=0.5)
    mock = build_mock(train, val, tune_ids=[], drop_first=[], shape={})
    tcfg = TwoStageConfig(floor=0.0, max_cands=5, model=MatcherParams(backend="heuristic"))
    outs = mock_stage1(cfg, stage1, mock, tcfg)
    for o in outs.values():
        assert len(o.pairs) == len(o.X) and o.n_all >= len(o.pairs)
        assert list(o.X.columns[-len(STACK_COLUMNS):]) == STACK_COLUMNS
    models, info = fit_stage2(outs, mock, tcfg)
    assert len(models) == tcfg.folds and "rows" in info
    scored, report = mock_scored(outs, models, mock, tcfg, roles=("val",))
    assert not scored[C.ENTITY_ID].duplicated().any()
    val_part = mock.part("val")
    rule, table = tune(scored, val_part.s1[C.ENTITY_ID], val_part.pairs,
                       Grid(tau_abs=(0.3, 0.7, 0.2), tau_rel=(0.0,), single_delta=(0.0,),
                            max_matches=(11,)))
    ts = TwoStage(stage1, models, rule, tcfg, table, info)
    again = TwoStage.load(ts.save(tmp_path / "two"), cfg)
    assert again.rule == rule and again.tcfg == tcfg and len(again.models) == tcfg.folds
    matching, candidates, *_ = run_test_two_stage(cfg, again, out_dir=tmp_path / "output")
    s1_ids, valid = split_ids("test", dataset_dir, check_ids=True)
    assert validate(matching, candidates, s1_ids, valid) == ([], [])
