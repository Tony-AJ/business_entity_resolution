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
from entity_resolution.stacking import (
    ANCHOR_COLUMNS,
    STACK_COLUMNS,
    anchor_features,
    best_other_rows,
    competition_features,
    group_stats,
)
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
        tail = STACK_COLUMNS + ANCHOR_COLUMNS
        assert list(o.X.columns[-len(tail):]) == tail
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


def test_rule_json_round_trip_and_dispatch() -> None:
    """Both rule kinds survive JSON and apply_rule picks the matching decoder."""
    from entity_resolution.decision import (
        DecisionRule,
        ExpectedRule,
        apply_rule,
        decide,
        decide_expected,
        rule_from_json,
        rule_to_json,
    )
    scored = pd.DataFrame({C.S1_ID: ["S1-a", "S1-a", "S1-b"],
                           C.ENTITY_ID: ["S2-1", "S2-2", "S2-3"],
                           "prob": np.float32([0.9, 0.35, 0.4])})
    for rule in (DecisionRule(0.3, 0.0, 0.5, 11, True), ExpectedRule(1.2, 0.1, 5, True)):
        again = rule_from_json(rule_to_json(rule))
        assert again == rule
        ref = decide_expected(scored, rule) if isinstance(rule, ExpectedRule) else decide(
            scored, rule)
        pd.testing.assert_frame_equal(apply_rule(scored, rule), ref)
    assert rule_from_json({"tau_abs": 0.4, "tau_rel": 0.0, "tau_single": 0.5,
                           "max_matches": 11, "one_to_one": True, "tune_f_beta": 0.9}) == \
        DecisionRule(0.4, 0.0, 0.5, 11, True)            # rule.json written before kinds


def test_best_other_rows() -> None:
    codes = np.array([0, 0, 0, 1, 2, 2])
    p1 = np.array([0.2, 0.9, 0.5, 0.7, 0.4, 0.4], dtype=np.float32)
    assert best_other_rows(codes, p1).tolist() == [1, 2, 1, -1, 5, 4]
    assert len(best_other_rows(np.zeros(0, np.int64), np.zeros(0, np.float32))) == 0


def test_anchor_features_compare_with_best_other() -> None:
    """The true record resembles the entity's best record; the decoy does not."""
    pairs = pd.DataFrame({C.S1_ID: ["a", "a", "a", "b"],
                          C.ENTITY_ID: ["t1", "t2", "d1", "z"]})
    p1 = np.array([0.95, 0.6, 0.7, 0.8], dtype=np.float32)
    pooln = pd.DataFrame({
        C.ENTITY_ID: ["t1", "t2", "d1", "z"],
        "name_norm": ["acme corp", "acme corporation", "acme corp", "zeta"],
        "addr_norm": ["12 main st springfield", "12 main street springfield",
                      "400 oak ave dallas", ""],
        "addr_nums": ["12", "12", "400", ""],
    })
    f = anchor_features(pairs, p1, pooln)
    assert list(f.columns) == ANCHOR_COLUMNS
    assert f["anc_p1"].tolist()[:3] == pytest.approx([0.7, 0.95, 0.95])
    assert np.isnan(f["anc_p1"].iloc[3])                       # b has one candidate
    assert f.loc[1, "anc_addr_ts"] > f.loc[2, "anc_addr_ts"]   # true t2 vs decoy d1
    assert f.loc[1, "anc_nums_eq"] == 1.0 and f.loc[2, "anc_nums_eq"] == 0.0
    assert f.iloc[3].isna().all()
    with pytest.raises(ValueError, match="not in pooln"):
        anchor_features(pairs, p1, pooln.iloc[:3])


def test_stage1_cache_round_trip(dataset_dir: Path, tmp_path: Path) -> None:
    """mock_stage1 with a cache directory writes once and reads back identical outputs."""
    cfg = tiny_cfg(tmp_path, dataset_dir)
    stage1 = fit(cfg, load_fold("train", dataset_dir, frac=0.5), tmp_path / "s1")
    train = load_fold("train", dataset_dir, columns=[C.COUNTRY], frac=0.5)
    val = load_fold("val", dataset_dir, columns=[C.COUNTRY], frac=0.5)
    mock = build_mock(train, val, tune_ids=[], drop_first=[], shape={})
    tcfg = TwoStageConfig(floor=0.0, max_cands=5, model=MatcherParams(backend="heuristic"))
    first = mock_stage1(cfg, stage1, mock, tcfg, cache_dir=tmp_path / "cache")
    assert sorted(p.name for p in (tmp_path / "cache").glob("*.parquet")) == sorted(
        f"mock_{c}.parquet" for c in first)
    again = mock_stage1(cfg, stage1, mock, tcfg, cache_dir=tmp_path / "cache")
    for c, o in first.items():
        pd.testing.assert_frame_equal(again[c].pairs, o.pairs)
        pd.testing.assert_frame_equal(again[c].X, o.X)
        assert again[c].n_all == o.n_all


def test_within_group_pairs() -> None:
    from entity_resolution.stacking import within_group_pairs
    b, c = within_group_pairs(np.array([10, 13, 14]), np.array([3, 1, 2]))
    assert list(zip(b.tolist(), c.tolist(), strict=True)) == [
        (10, 11), (10, 12), (11, 10), (11, 12), (12, 10), (12, 11), (14, 15), (15, 14)]


def test_cohesion_features() -> None:
    """True records support each other; the decoy has no support; a lone record is 0/NaN."""
    from entity_resolution.stacking import COHESION_COLUMNS, cohesion_features
    pairs = pd.DataFrame({C.S1_ID: ["a", "a", "a", "b"], C.ENTITY_ID: ["t1", "t2", "d1", "z"]})
    p1 = np.array([0.95, 0.6, 0.7, 0.8], dtype=np.float32)
    pooln = pd.DataFrame({
        C.ENTITY_ID: ["t1", "t2", "d1", "z"],
        "name_norm": ["acme corp", "acme corporation", "acme corp", "zeta"],
        "addr_norm": ["12 main st springfield", "12 main st springfield", "400 oak ave dallas",
                      "9 elm rd"],
    })
    f = cohesion_features(pairs, p1, pooln)
    assert list(f.columns) == COHESION_COLUMNS
    assert f.loc[1, "coh_addr"] > f.loc[2, "coh_addr"]          # t2 fits the group, d1 not
    assert f["coh_support"].tolist() == [1.0, 1.0, 0.0, 0.0]    # t1<->t2 support each other
    assert np.isnan(f.loc[3, "coh_addr"])                        # b has no other candidate


def test_rival_features() -> None:
    """The record compares with the best OTHER S1 claiming it; lone records get NaN."""
    from entity_resolution.stacking import RIVAL_COLUMNS, rival_features
    pairs = pd.DataFrame({C.S1_ID: ["x", "y", "x", "z"], C.ENTITY_ID: ["r", "r", "q", "w"]})
    p1 = np.array([0.9, 0.8, 0.5, 0.7], dtype=np.float32)
    s1n = pd.DataFrame({C.ENTITY_ID: ["x", "y", "z"], "name_norm": ["acme", "acme", "zeta"],
                        "addr_norm": ["12 main st", "400 oak ave", "9 elm rd"]})
    pooln = pd.DataFrame({C.ENTITY_ID: ["r", "q", "w"], "name_norm": ["acme", "acme", "zeta"],
                          "addr_norm": ["12 main st", "5 pine rd", "9 elm rd"]})
    keep = np.array([True, True, False, True])
    f = rival_features(pairs, p1, keep, s1n, pooln, own_addr=np.array([1.0, 0.2, 1.0]))
    assert list(f.columns) == RIVAL_COLUMNS and len(f) == 3
    assert f.loc[0, "riv_addr_ts"] < 0.5                 # x's rival y lives elsewhere
    assert f.loc[1, "riv_addr_ts"] == 1.0                # y's rival x has r's address
    assert f.loc[0, "riv_addr_gap"] > 0.5 and f.loc[1, "riv_addr_gap"] < 0
    assert f.iloc[2].isna().all()                        # w has a single claimant


def test_stage1_with_every_optional_group(dataset_dir: Path, tmp_path: Path) -> None:
    """Anchors, cohesion and rivals together still give aligned frames end to end."""
    from entity_resolution.stacking import COHESION_COLUMNS, RIVAL_COLUMNS
    cfg = tiny_cfg(tmp_path, dataset_dir)
    stage1 = fit(cfg, load_fold("train", dataset_dir, frac=0.5), tmp_path / "s1")
    train = load_fold("train", dataset_dir, columns=[C.COUNTRY], frac=0.5)
    val = load_fold("val", dataset_dir, columns=[C.COUNTRY], frac=0.5)
    mock = build_mock(train, val, tune_ids=[], drop_first=[], shape={})
    tcfg = TwoStageConfig(floor=0.0, max_cands=5, cohesion=True, rivals=True,
                          model=MatcherParams(backend="heuristic"))
    outs = mock_stage1(cfg, stage1, mock, tcfg)
    for o in outs.values():
        tail = STACK_COLUMNS + ANCHOR_COLUMNS + COHESION_COLUMNS + RIVAL_COLUMNS
        assert list(o.X.columns[-len(tail):]) == tail and len(o.X) == len(o.pairs)
    models, _ = fit_stage2(outs, mock, tcfg)
    scored, _ = mock_scored(outs, models, mock, tcfg, roles=("val",))
    assert not scored[C.ENTITY_ID].duplicated().any()
