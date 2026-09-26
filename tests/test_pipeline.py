"""Integration and end-to-end tests of the pipeline on the synthetic challenge files (12 §3-4).

The fixture's vocabulary is tiny, so the TF-IDF passes use ``min_df=1, max_df=1.0``; the
fit side of the inner split is empty on it, so the matcher is the ``heuristic`` backend
and every stage must accept empty input.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from entity_resolution import config as C
from entity_resolution.blocking import PAIR_COLUMNS, BlockingConfig, TopKSpec, block, to_id_lists
from entity_resolution.data import load_source, read_id_lists
from entity_resolution.decision import Grid
from entity_resolution.evaluate import blocking_report
from entity_resolution.features import DEFAULT_GROUPS, build_features, feature_names
from entity_resolution.model import MatcherParams
from entity_resolution.normalize import normalise_records
from entity_resolution.pipeline import Fitted, PipelineConfig, fit, run_fold, run_test
from entity_resolution.split import load_fold
from entity_resolution.submission import split_ids, validate


def tiny_cfg(tmp_path: Path, dataset_dir: Path) -> PipelineConfig:
    """Pipeline configuration sized for the six-record fixture."""
    blocking = BlockingConfig(
        name_char=TopKSpec("name_core", top_k=5, min_sim=0.1, max_df=1.0, min_df=1),
        name_addr_word=TopKSpec("name_addr", "word", (1, 2), 5, 0.05, 1.0, min_df=1),
        n_threads=1)
    grid = Grid(tau_abs=(0.3, 0.7, 0.2), tau_rel=(0.0, 0.7), single_delta=(0.0, 0.1),
                max_matches=(11,))
    return PipelineConfig(blocking=blocking, model=MatcherParams(backend="heuristic"),
                          grid=grid, n_fit_s1=10, n_stop_s1=10, n_tune_s1=None,
                          chunk_rows=1000, dataset_dir=dataset_dir,
                          cache_dir=tmp_path / "cache")


def test_stages_compose_on_synthetic_dataset(dataset_dir: Path) -> None:
    """normalise -> block -> build_features line up on the fixture's train split."""
    s1n = normalise_records(load_source("train", 1, dataset_dir, cache=False))
    pool = [load_source("train", s, dataset_dir, cache=False) for s in (2, 3)]
    pooln = normalise_records(pd.concat(pool, ignore_index=True))
    cfg = tiny_cfg(Path("."), dataset_dir).blocking
    pairs = block(s1n, pooln, cfg)
    assert list(pairs.columns) == PAIR_COLUMNS
    assert not pairs.duplicated([C.S1_ID, C.ENTITY_ID]).any()
    assert pairs[C.S1_ID].is_monotonic_increasing
    country = dict(zip(s1n[C.ENTITY_ID], s1n[C.COUNTRY], strict=True))
    pool_country = dict(zip(pooln[C.ENTITY_ID], pooln[C.COUNTRY], strict=True))
    assert all(country[a] == pool_country[b]
               for a, b in zip(pairs[C.S1_ID], pairs[C.ENTITY_ID], strict=True))
    X = build_features(pairs, s1n, pooln)
    assert len(X) == len(pairs) and X.index.equals(pairs.index)
    assert list(X.columns) == feature_names()
    assert all(dtype == np.float32 for dtype in X.dtypes)
    assert set(to_id_lists(pairs, s1n[C.ENTITY_ID])) == set(s1n[C.ENTITY_ID])


def test_fit_and_run_fold_on_synthetic_dataset(dataset_dir: Path, tmp_path: Path) -> None:
    """fit on an empty fit side, score val, save/load reproduce the matches."""
    cfg = tiny_cfg(tmp_path, dataset_dir)
    train = load_fold("train", dataset_dir, frac=0.5)
    val = load_fold("val", dataset_dir, frac=0.5)
    fitted = fit(cfg, train, tmp_path / "art")
    metrics, pairs, scored, matches = run_fold(cfg, fitted, val)
    assert 0.0 <= metrics["f_beta"] <= 1.0
    assert blocking_report(pairs, val)["pair_recall"] == 1.0
    for key in ("f_beta", "f_beta_singletons", "f_beta_matched", "pair_precision",
                "pair_recall", "cand_recall", "entity_recall", "ceiling_f_beta",
                "cands_mean", "cands_p95", "blocking_seconds", "score_seconds"):
        assert key in metrics
    again = Fitted.load(tmp_path / "art", cfg)
    assert again.rule == fitted.rule
    _, _, _, matches2 = run_fold(cfg, again, val)
    assert matches2.equals(matches)
    # fillers are opt-in: a default version learns, logs and saves none
    assert fitted.fillers is None and again.fillers is None and "fillers" not in fitted.info
    assert not (tmp_path / "art" / "fillers.json").exists()


def test_end_to_end_on_synthetic_dataset(dataset_dir: Path, tmp_path: Path) -> None:
    """fit -> run_fold -> run_test writes two valid files covering every test S1 (France too)."""
    cfg = tiny_cfg(tmp_path, dataset_dir)
    fitted = fit(cfg, load_fold("train", dataset_dir, frac=0.5), tmp_path / "art")
    metrics, *_ = run_fold(cfg, fitted, load_fold("val", dataset_dir, frac=0.5))
    assert isinstance(metrics, dict)
    matching, candidates, *_ = run_test(cfg, fitted, out_dir=tmp_path / "output")
    assert matching.name == C.MATCHING_FILE and candidates.name == C.CANDIDATE_FILE
    s1_ids, valid = split_ids("test", dataset_dir, check_ids=True)
    assert validate(matching, candidates, s1_ids, valid) == ([], [])
    _, rows = read_id_lists(matching)
    assert [s1 for s1, _ in rows] == ["S1-00010", "S1-00011"]
    cands = dict(read_id_lists(candidates)[1])
    for s1, ids in rows:
        assert all(i.startswith(("S2-", "S3-")) for i in ids)
        assert set(ids) <= set(cands[s1])
    assert cands["S1-00010"], "the France entity gets candidates without any country list"


def test_add_frequencies_rates_per_country() -> None:
    """Core-name rates count the whole fold per country, per million; empty names are NaN."""
    from entity_resolution.pipeline import add_frequencies
    s1_all = pd.DataFrame({C.COUNTRY: ["US", "US", "US", "US", "India"],
                           "name_core": ["acme", "acme", "globex", "", "acme"],
                           "name_first": ["acme", "acme", "globex", "", "acme"]}).astype("str")
    s1n = s1_all.iloc[[0, 3]].reset_index(drop=True)  # a sample of the fold
    pooln = pd.DataFrame({C.COUNTRY: ["US", "US", "India", "India"],
                          "name_core": ["acme", "initech", "acme", "acme"],
                          "name_first": ["acme", "initech", "acme", "acme"]}).astype("str")
    s1f, poolf = add_frequencies(s1n, pooln, s1_all)
    assert s1f["freq_same"].iloc[0] == np.float32(1 / 3 * 1e6)       # 1 other of 3 US S1
    assert s1f["freq_other"].iloc[0] == np.float32(1 / 2 * 1e6)      # 1 of 2 US pool records
    assert np.isnan(s1f["freq_same"].iloc[1])                        # empty core name
    assert poolf["freq_same"].iloc[0] == 0.0                         # a unique name reads 0
    assert poolf["freq_same"].tolist()[2:] == [1e6, 1e6]             # India pool: 1 other of 1
    assert poolf["freq_other"].iloc[1] == 0.0                        # no S1 initech
    assert poolf["freq_other"].iloc[2] == np.float32(1e6)            # the one India S1 is acme


def test_frequency_group_end_to_end(dataset_dir: Path, tmp_path: Path) -> None:
    """With the opt-in frequency group the pipeline still fits, scores and writes valid files."""
    from dataclasses import replace

    from entity_resolution.features import DEFAULT_GROUPS
    cfg = replace(tiny_cfg(tmp_path, dataset_dir),
                  feature_groups=(*DEFAULT_GROUPS, "frequency"))
    fitted = fit(cfg, load_fold("train", dataset_dir, frac=0.5), tmp_path / "art")
    metrics, *_ = run_fold(cfg, fitted, load_fold("val", dataset_dir, frac=0.5))
    assert 0.0 <= metrics["f_beta"] <= 1.0
    matching, candidates, *_ = run_test(cfg, fitted, out_dir=tmp_path / "output")
    s1_ids, valid = split_ids("test", dataset_dir, check_ids=True)
    assert validate(matching, candidates, s1_ids, valid) == ([], [])


def test_dense_tune_pool_fits(dataset_dir: Path, tmp_path: Path) -> None:
    """tune_pool='train' blocks the tune side against the whole train pool and still fits."""
    from dataclasses import replace
    cfg = replace(tiny_cfg(tmp_path, dataset_dir), tune_pool="train")
    train = load_fold("train", dataset_dir, frac=0.5)
    fitted = fit(cfg, train, tmp_path / "art")
    assert 0.0 <= fitted.tune_table["f_beta"].max() <= 1.0
    bad = replace(cfg, tune_pool="everything")
    try:
        fit(bad, train)
    except ValueError as e:
        assert "tune_pool" in str(e)
    else:
        raise AssertionError("an unknown tune_pool must be refused")


def test_retune_keeps_matcher_and_saves(dataset_dir: Path, tmp_path: Path) -> None:
    """retune re-runs only the rule grid (here against the whole train pool)."""
    from dataclasses import replace

    from entity_resolution.pipeline import retune
    cfg = tiny_cfg(tmp_path, dataset_dir)
    train = load_fold("train", dataset_dir, frac=0.5)
    fitted = fit(cfg, train, tmp_path / "art")
    again = retune(replace(cfg, tune_pool="train"), fitted, train, tmp_path / "art2")
    assert again.matcher is fitted.matcher and again.token_map == fitted.token_map
    assert (tmp_path / "art2" / "rule.json").exists()
    assert "retuned_from" in again.info


def test_run_mock_scores_tunes_and_reports(dataset_dir: Path, tmp_path: Path) -> None:
    """The mock path: every present S1 scored per country, 1-to-1 across them, per-role use."""
    from entity_resolution.decision import DecisionRule
    from entity_resolution.mock import build_mock
    from entity_resolution.pipeline import mock_scores, run_mock, tune_mock
    cfg = tiny_cfg(tmp_path, dataset_dir)
    fitted = fit(cfg, load_fold("train", dataset_dir, frac=0.5), tmp_path / "art")
    train = load_fold("train", dataset_dir, columns=[C.COUNTRY], frac=0.5)
    val = load_fold("val", dataset_dir, columns=[C.COUNTRY], frac=0.5)
    mock = build_mock(train, val, tune_ids=train.s1[C.ENTITY_ID], drop_first=[], shape={})
    assert len(mock.fold.s1) == len(train.s1) + len(val.s1)       # no shape: all present
    scored, report = run_mock(cfg, fitted, mock)
    assert not scored[C.ENTITY_ID].duplicated().any()               # 1-to-1 across roles
    assert set(report.index) == {"tune", "val"}
    rule, table = tune_mock(scored, mock, cfg.grid)
    assert isinstance(rule, DecisionRule) and len(table)
    scores = mock_scores(scored, mock, rule)
    assert scores.index[0] == "all"
    assert set(scores.index[1:]) == set(mock.part("val").s1[C.COUNTRY])
    assert scores.loc["all", "entities"] == len(mock.ids("val"))
    assert 0.0 <= scores.loc["all", "f_beta"] <= 1.0


def test_config_record_round_trip(tmp_path: Path) -> None:
    """from_record rebuilds exactly what record wrote, nested specs and tuples included."""
    import json
    from dataclasses import replace

    from entity_resolution.blocking import TopKSpec
    from entity_resolution.features import DEFAULT_GROUPS
    cfg = replace(PipelineConfig(), feature_groups=(*DEFAULT_GROUPS, "frequency"),
                  model=MatcherParams(backend="xgb", device="cuda", num_leaves=127))
    cfg = replace(cfg, blocking=replace(cfg.blocking, cap_order="sim_first",
                                        addr_char=TopKSpec("addr_norm", "word", (1, 2), 10)))
    again = PipelineConfig.from_record(json.loads(json.dumps(cfg.record())))
    assert again == cfg and again.blocking.key() == cfg.blocking.key()


def test_stats_feature_groups_end_to_end(dataset_dir: Path, tmp_path: Path) -> None:
    """The pool-statistics groups run through fit, run_fold and run_test (France too)."""
    cfg = replace(tiny_cfg(tmp_path, dataset_dir),
                  feature_groups=DEFAULT_GROUPS + ("idf", "token_freq", "ctx_idf"))
    fitted = fit(cfg, load_fold("train", dataset_dir, frac=0.5), tmp_path / "art")
    metrics, pairs, scored, _ = run_fold(cfg, fitted, load_fold("val", dataset_dir, frac=0.5))
    assert 0.0 <= metrics["f_beta"] <= 1.0 and len(scored) == len(pairs)
    matching, candidates, *_ = run_test(cfg, fitted, out_dir=tmp_path / "output")
    s1_ids, valid = split_ids("test", dataset_dir, check_ids=True)
    assert validate(matching, candidates, s1_ids, valid) == ([], [])


# ------------------------------------------------------------ learned fillers ----
def test_learn_fillers_and_the_nofill_pass(filler_dir: Path, tmp_path: Path) -> None:
    """Fillers are learned from the train fold (cached), loaded as name_core_nofill, and the
    nofill pass meets the six filler variants; everything stays off by default."""
    from entity_resolution.normalize import NormaliseConfig
    from entity_resolution.pipeline import _static_key, learn_fillers, load_normalised, prepare
    base = tiny_cfg(tmp_path, filler_dir)
    cfg = replace(base, normalise=NormaliseConfig(learn_fillers=True))
    train = load_fold("train", filler_dir, frac=0.0)            # every entity in train
    assert learn_fillers(base, train) is None
    fillers = learn_fillers(cfg, train)
    assert fillers == ["center", "services"]                     # "holdings" is held by S1
    assert learn_fillers(cfg, train) == fillers and len(list(cfg.cache_dir.glob("fillers_*")))
    # the learned-filler switches never touch the static normalisation cache
    assert _static_key(NormaliseConfig(learn_fillers=True, filler_min_ratio=2.0)) == \
        _static_key(NormaliseConfig())
    assert "name_core_nofill" not in load_normalised("train", (2, 3), cfg).columns
    s1n = load_normalised("train", (1,), cfg, fillers=fillers)
    pooln = load_normalised("train", (2, 3), cfg, fillers=fillers)
    bare = dict(zip(pooln[C.ENTITY_ID], pooln["name_core_nofill"], strict=True))
    assert bare["S3-20010"] == "globex" and bare["S3-20016"] == "stark holdings"
    on = replace(cfg, blocking=replace(cfg.blocking, nofill_max_group=50))
    pairs = prepare(s1n, pooln, on, "t", fillers=fillers)
    bits = pairs["pass"].to_numpy()
    assert ((bits & 128) != 0).sum() == 8                        # every pair: equal once bare
    only = pairs[((bits & 128) != 0) & ((bits & 7) == 0)]        # ... and no other exact key
    assert set(zip(only[C.S1_ID], only[C.ENTITY_ID], strict=True)) == {
        (f"S1-2{i + 9:04d}", f"S{2 + i % 2}-2{i + 9:04d}") for i in range(6)}
    with pytest.raises(ValueError, match="fillers"):
        prepare(s1n, pooln, on, "t")                             # the pass needs the fillers


def test_filler_switches_end_to_end(filler_dir: Path, tmp_path: Path) -> None:
    """Every filler switch on: fit, save / load, run_fold, run_mock and valid test files."""
    import json

    from entity_resolution.mock import build_mock
    from entity_resolution.normalize import NormaliseConfig
    from entity_resolution.pipeline import run_mock
    base = tiny_cfg(tmp_path, filler_dir)
    cfg = replace(base, normalise=NormaliseConfig(learn_fillers=True),
                  blocking=replace(base.blocking, nofill_max_group=50),
                  feature_groups=(*DEFAULT_GROUPS, "nofill"))
    assert PipelineConfig.from_record(json.loads(json.dumps(cfg.record()))) == cfg
    train = load_fold("train", filler_dir, frac=0.5)
    fitted = fit(cfg, train, tmp_path / "art")
    assert fitted.fillers == ["center"] == fitted.info["fillers"]   # "services": a val pair
    assert Fitted.load(tmp_path / "art", cfg).fillers == fitted.fillers
    metrics, pairs, scored, _ = run_fold(cfg, fitted, load_fold("val", filler_dir, frac=0.5))
    assert 0.0 <= metrics["f_beta"] <= 1.0 and len(scored) == len(pairs)
    assert metrics["cand_recall"] == 1.0
    tr = load_fold("train", filler_dir, columns=[C.COUNTRY], frac=0.5)
    va = load_fold("val", filler_dir, columns=[C.COUNTRY], frac=0.5)
    mock = build_mock(tr, va, tune_ids=tr.s1[C.ENTITY_ID], drop_first=[], shape={})
    mock_scored, report = run_mock(cfg, fitted, mock)
    assert not mock_scored[C.ENTITY_ID].duplicated().any() and set(report.index) == {
        "tune", "val"}
    matching, candidates, *_ = run_test(cfg, fitted, out_dir=tmp_path / "output")
    s1_ids, valid = split_ids("test", filler_dir, check_ids=True)
    assert validate(matching, candidates, s1_ids, valid) == ([], [])


def test_learn_fillers_samples_whole_entities_past_the_cap(filler_dir: Path, tmp_path: Path,
                                                          monkeypatch) -> None:
    """Past FILLER_PAIRS pairs, fillers are learned on a hashed sample of S1 entities: a
    deterministic subset of the pairs, cached under its own key."""
    from entity_resolution import pipeline
    from entity_resolution.normalize import NormaliseConfig
    cfg = replace(tiny_cfg(tmp_path, filler_dir), normalise=NormaliseConfig(learn_fillers=True))
    train = load_fold("train", filler_dir, frac=0.0)
    full = pipeline.learn_fillers(cfg, train)
    monkeypatch.setattr(pipeline, "FILLER_PAIRS", 4)                 # 8 pairs: ~half kept
    sampled = pipeline.learn_fillers(cfg, train)
    assert set(sampled) <= set(full) and sampled == pipeline.learn_fillers(cfg, train)
    assert len(list(cfg.cache_dir.glob("fillers_*.json"))) == 2       # one file per cap
