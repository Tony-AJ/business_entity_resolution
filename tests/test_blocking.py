"""Blocking is scored on candidate recall, not read for logic, so the fixtures below
build minimal normalised-looking frames by hand rather than depending on ``normalize.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from entity_resolution import blocking
from entity_resolution import config as C
from entity_resolution.blocking import (
    PAIR_COLUMNS,
    BlockingConfig,
    TopKSpec,
    block,
    exact_pass,
    to_id_lists,
    topk_pass,
    union_passes,
)


def _two_country_frames() -> tuple[pd.DataFrame, pd.DataFrame, BlockingConfig]:
    """One US pair, one France pair (an "unseen" country), one US distractor."""
    s1 = pd.DataFrame({
        C.ENTITY_ID: ["S1-1", "S1-2"],
        C.COUNTRY: ["US", "France"],
        "name_core": ["acme corp", "boulangerie dupont"],
        "name_sorted": ["acme corp", "boulangerie dupont"],
        "name_squash": ["acmecorp", "boulangeriedupont"],
        "name_addr": ["acme corp 12 main st", "boulangerie dupont 5 rue de la paix"],
    })
    pool = pd.DataFrame({
        C.ENTITY_ID: ["S2-1", "S3-1", "S2-2"],
        C.COUNTRY: ["US", "US", "France"],
        "name_core": ["acme corp", "widget inc", "boulangerie dupont"],
        "name_sorted": ["acme corp", "widget inc", "boulangerie dupont"],
        "name_squash": ["acmecorp", "widgetinc", "boulangeriedupont"],
        "name_addr": ["acme corp 12 main st", "widget inc 9 side ave",
                     "boulangerie dupont 5 rue de la paix"],
    })
    cfg = BlockingConfig(
        name_char=TopKSpec("name_core", min_df=1, max_df=1.0, top_k=5, min_sim=0.05),
        name_addr_word=TopKSpec("name_addr", "word", (1, 1), 5, 0.05, 1.0, 1),
    )
    return s1, pool, cfg


# ------------------------------------------------------------------------- P1 ----
def test_exact_pass_respects_max_group():
    s1n = pd.DataFrame({"key": ["big", "small"]})
    pooln = pd.DataFrame({"key": ["big"] * 51 + ["small"] * 3})
    pairs = exact_pass(s1n, pooln, "key", max_group=50)
    assert (pairs["s1_idx"] == 0).sum() == 0  # oversized group skipped entirely
    assert set(pairs.loc[pairs["s1_idx"] == 1, "pool_idx"]) == {51, 52, 53}


def test_exact_pass_skips_empty_key():
    s1n = pd.DataFrame({"key": ["", "acme"]})
    pooln = pd.DataFrame({"key": ["", "acme"]})
    pairs = exact_pass(s1n, pooln, "key", max_group=50)
    assert len(pairs) == 1
    assert pairs.iloc[0][["s1_idx", "pool_idx"]].tolist() == [1, 1]


# ------------------------------------------------------------------------- P2/3 ----
def test_topk_pass_finds_typos():
    base = [f"acme trading company {i}" for i in range(20)]
    typo = [n.replace("company", "compnay") for n in base]
    s1n = pd.DataFrame({"text": base})
    pooln = pd.DataFrame({"text": typo})
    spec = TopKSpec("text", "char_wb", (3, 3), top_k=3, min_sim=0.1, max_df=1.0, min_df=1)
    pairs = topk_pass(s1n, pooln, spec, s1_chunk=50, pool_chunk=50, n_threads=1, seed=0)
    for i in range(20):
        assert i in pairs.loc[pairs["s1_idx"] == i, "pool_idx"].to_numpy()
    assert (pairs["sim"] > 0).all() and (pairs["sim"] <= 1).all()


def test_topk_pass_chunking_invariant():
    s1n = pd.DataFrame({"text": [f"acme trading company {i}" for i in range(10)]})
    pooln = pd.DataFrame({"text": [f"acme trading compnay {i}" for i in range(20)]})
    spec = TopKSpec("text", "char_wb", (3, 3), top_k=3, min_sim=0.05, max_df=1.0, min_df=1)
    full = topk_pass(s1n, pooln, spec, 50, 20, 1, 0)
    half = topk_pass(s1n, pooln, spec, 50, 10, 1, 0)
    full = full.sort_values(["s1_idx", "pool_idx"]).reset_index(drop=True)
    half = half.sort_values(["s1_idx", "pool_idx"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(full, half, check_exact=False, atol=1e-6)


# --------------------------------------------------------------------- union ----
def test_union_bitmask_and_max_sim():
    exact = pd.DataFrame({"s1_idx": [0], "pool_idx": [0]})
    topk = pd.DataFrame({"s1_idx": [0, 1], "pool_idx": [0, 1], "sim": [0.5, 0.7]})
    out = union_passes({"name_core": exact, "name_char": topk}, max_per_s1=10)

    row0 = out[(out["s1_idx"] == 0) & (out["pool_idx"] == 0)].iloc[0]
    assert row0["pass"] == 9  # name_core (1) | name_char (8)
    assert row0["sim_name_char"] == pytest.approx(0.5)
    assert pd.isna(row0["sim_name_addr_word"]) and pd.isna(row0["sim_addr_char"])

    row1 = out[(out["s1_idx"] == 1) & (out["pool_idx"] == 1)].iloc[0]
    assert row1["pass"] == 8  # name_char only


def test_union_cap_prefers_exact_then_sim():
    exact = pd.DataFrame({"s1_idx": [0], "pool_idx": [0]})
    topk = pd.DataFrame({"s1_idx": [0, 0], "pool_idx": [1, 2], "sim": [0.9, 0.2]})
    out = union_passes({"name_core": exact, "name_char": topk}, max_per_s1=2)
    assert set(out["pool_idx"]) == {0, 1}  # exact pair + higher-sim pair; low-sim dropped


# ------------------------------------------------------------------------- block ----
def test_block_partitions_by_country():
    s1, pool, cfg = _two_country_frames()
    pairs = block(s1, pool, cfg)
    us_matches = set(pairs.loc[pairs[C.S1_ID] == "S1-1", C.ENTITY_ID])
    fr_matches = set(pairs.loc[pairs[C.S1_ID] == "S1-2", C.ENTITY_ID])
    assert us_matches <= {"S2-1", "S3-1"}  # never the France pool record
    assert fr_matches == {"S2-2"}  # France, absent from any hard-coded country list


def test_block_output_schema():
    s1, pool, cfg = _two_country_frames()
    pairs = block(s1, pool, cfg)
    assert list(pairs.columns) == list(PAIR_COLUMNS)
    assert pairs["pass"].dtype == np.uint8
    for col in ("sim_name_char", "sim_name_addr_word", "sim_addr_char"):
        assert pairs[col].dtype == np.float32
    assert not pairs.duplicated([C.S1_ID, C.ENTITY_ID]).any()
    assert pairs[C.S1_ID].tolist() == sorted(pairs[C.S1_ID].tolist())


def test_to_id_lists_every_s1_present():
    pairs = pd.DataFrame({C.S1_ID: ["S1-1", "S1-1"], C.ENTITY_ID: ["S2-1", "S3-1"]})
    out = to_id_lists(pairs, pd.Series(["S1-1", "S1-2"]))
    assert out["S1-1"] == ["S2-1", "S3-1"]
    assert out["S1-2"] == []
    assert len(set(out["S1-1"])) == len(out["S1-1"])


def test_conftest_truth_pairs_are_candidates(dataset_dir):
    """On the shared conftest fixture, every true pair is a candidate (pair_recall == 1.0)."""
    import re

    from entity_resolution.data import load_source, load_truth_pairs

    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()

    def normalise(df: pd.DataFrame) -> pd.DataFrame:
        name = df[C.NAME].map(norm)
        addr = df[C.ADDRESS].map(norm)
        return pd.DataFrame({
            C.ENTITY_ID: df[C.ENTITY_ID],
            C.COUNTRY: df[C.COUNTRY],
            "name_core": name,
            "name_sorted": name.map(lambda t: " ".join(sorted(t.split()))),
            "name_squash": name.str.replace(" ", "", regex=False),
            "name_addr": name + " " + addr,
        })

    s1 = load_source("train", 1, dataset_dir)
    pool = pd.concat([load_source("train", 2, dataset_dir), load_source("train", 3, dataset_dir)],
                     ignore_index=True)
    truth = load_truth_pairs(dataset_dir)

    cfg = BlockingConfig(
        name_char=TopKSpec("name_core", min_df=1, max_df=1.0, top_k=10, min_sim=0.05),
        name_addr_word=TopKSpec("name_addr", "word", (1, 1), 10, 0.05, 1.0, 1),
    )
    pairs = block(normalise(s1), normalise(pool), cfg)
    kept = set(zip(pairs[C.S1_ID], pairs[C.ENTITY_ID], strict=True))
    truth_pairs = set(zip(truth[C.S1_ID], truth[C.ENTITY_ID], strict=True))
    assert truth_pairs <= kept


def test_block_cache_roundtrip(tmp_path, monkeypatch):
    s1, pool, cfg = _two_country_frames()
    cache_dir = tmp_path / "cache"
    first = block(s1, pool, cfg, cache_dir=cache_dir, tag="val")

    monkeypatch.setattr(blocking, "block_partition",
                        lambda *a, **k: pytest.fail("recomputed instead of using the cache"))
    second = block(s1, pool, cfg, cache_dir=cache_dir, tag="val")
    pd.testing.assert_frame_equal(first, second)
