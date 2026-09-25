"""Integration and end-to-end tests of the pipeline on the synthetic challenge files (12 §3-4).

The fixture's vocabulary is tiny, so the TF-IDF passes use ``min_df=1, max_df=1.0``; the
fit side of the inner split is empty on it, so the matcher is the ``heuristic`` backend
and every stage must accept empty input.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from entity_resolution import config as C
from entity_resolution.blocking import PAIR_COLUMNS, BlockingConfig, TopKSpec, block, to_id_lists
from entity_resolution.data import load_source, read_id_lists
from entity_resolution.decision import Grid
from entity_resolution.evaluate import blocking_report
from entity_resolution.features import build_features, feature_names
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
    assert s1f["freq_same"].iloc[0] == np.float32(2 / 4 * 1e6)       # 2 of 4 US S1 records
    assert s1f["freq_other"].iloc[0] == np.float32(1 / 2 * 1e6)      # 1 of 2 US pool records
    assert np.isnan(s1f["freq_same"].iloc[1])                        # empty core name
    assert poolf["freq_same"].tolist()[2:] == [1e6, 1e6]             # India pool: all acme
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
