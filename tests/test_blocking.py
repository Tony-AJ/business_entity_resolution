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
    PASS_BITS,
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


# =====================================================================================
# Edge-case coverage below. Everything here was probed interactively against the
# implementation first (see the PR/commit discussion); no bug survived that pass, so
# these tests lock in the *verified* current behaviour rather than guess at intent.
# =====================================================================================

def _tiny_cfg(**overrides) -> BlockingConfig:
    """A BlockingConfig with min_df=1, max_df=1.0 so tiny synthetic corpora don't get
    filtered by TF-IDF's document-frequency cutoffs (per 06_BLOCKING_STRATEGY.md SS8)."""
    defaults = dict(
        name_char=TopKSpec("name_core", min_df=1, max_df=1.0, top_k=5, min_sim=0.05),
        name_addr_word=TopKSpec("name_addr", "word", (1, 1), 5, 0.05, 1.0, 1),
    )
    defaults.update(overrides)
    return BlockingConfig(**defaults)


def _frame(entity_ids, countries, name_core, name_addr=None) -> pd.DataFrame:
    """Minimal normalised-looking frame: entity_id, country and the four name_* / name_addr
    columns block_partition needs, built from name_core alone unless name_addr is given."""
    name_core = list(name_core)
    name_addr = list(name_addr) if name_addr is not None else name_core
    return pd.DataFrame({
        C.ENTITY_ID: entity_ids,
        C.COUNTRY: countries,
        "name_core": name_core,
        "name_sorted": name_core,
        "name_squash": [n.replace(" ", "") for n in name_core],
        "name_addr": name_addr,
    })


# ------------------------------------------------------------------- A: exact_pass ----
def test_exact_pass_cross_product_multiple_to_multiple():
    s1n = pd.DataFrame({"key": ["a", "a", "b"]})
    pooln = pd.DataFrame({"key": ["a", "a", "b", "c"]})
    pairs = exact_pass(s1n, pooln, "key", max_group=50)
    assert set(pairs.loc[pairs["s1_idx"] == 0, "pool_idx"]) == {0, 1}
    assert set(pairs.loc[pairs["s1_idx"] == 1, "pool_idx"]) == {0, 1}
    assert set(pairs.loc[pairs["s1_idx"] == 2, "pool_idx"]) == {2}


def test_exact_pass_no_matching_keys_is_empty():
    s1n = pd.DataFrame({"key": ["a", "b"]})
    pooln = pd.DataFrame({"key": ["c", "d"]})
    pairs = exact_pass(s1n, pooln, "key", max_group=50)
    assert len(pairs) == 0
    assert list(pairs.columns) == ["s1_idx", "pool_idx"]


def test_exact_pass_group_exactly_at_max_group_is_kept():
    """Only groups strictly bigger than max_group are dropped."""
    s1n = pd.DataFrame({"key": ["k"]})
    pooln = pd.DataFrame({"key": ["k"] * 50})
    pairs = exact_pass(s1n, pooln, "key", max_group=50)
    assert len(pairs) == 50
    pairs51 = exact_pass(s1n, pd.DataFrame({"key": ["k"] * 51}), "key", max_group=50)
    assert len(pairs51) == 0


def test_exact_pass_s1_side_group_size_is_never_capped():
    """max_group only drops oversized POOL groups; a huge S1 group is untouched."""
    s1n = pd.DataFrame({"key": ["k"] * 200})
    pooln = pd.DataFrame({"key": ["k"]})  # single pool row, well under the cap
    pairs = exact_pass(s1n, pooln, "key", max_group=50)
    assert len(pairs) == 200  # every S1 row still matches the one pool row


def test_exact_pass_oversized_on_both_sides_still_dropped_by_pool_side():
    s1n = pd.DataFrame({"key": ["k"] * 200})
    pooln = pd.DataFrame({"key": ["k"] * 51})
    pairs = exact_pass(s1n, pooln, "key", max_group=50)
    assert len(pairs) == 0


def test_exact_pass_whitespace_only_key_is_not_treated_as_empty():
    """Only the literal "" is excluded; this documents that boundary, it is not a bug."""
    s1n = pd.DataFrame({"key": ["   "]})
    pooln = pd.DataFrame({"key": ["   "]})
    pairs = exact_pass(s1n, pooln, "key", max_group=50)
    assert len(pairs) == 1


def test_exact_pass_none_is_not_treated_as_empty_out_of_contract():
    """normalize.py's contract guarantees these columns are never null (05 SS: "every
    string column non-null"), so exact_pass only special-cases the "" sentinel. If that
    upstream guarantee is ever broken, two None keys will match each other; this test
    documents the current, contract-dependent behaviour rather than papering over it.
    """
    s1n = pd.DataFrame({"key": [None, "acme"]})
    pooln = pd.DataFrame({"key": [None, "acme"]})
    pairs = exact_pass(s1n, pooln, "key", max_group=50)
    assert set(zip(pairs["s1_idx"], pairs["pool_idx"], strict=True)) == {(0, 0), (1, 1)}


def test_exact_pass_empty_s1n_and_empty_pooln():
    s1_full = pd.DataFrame({"key": ["a", "b"]})
    pool_full = pd.DataFrame({"key": ["a", "b"]})
    empty_s1 = exact_pass(s1_full.iloc[0:0], pool_full, "key", max_group=50)
    empty_pool = exact_pass(s1_full, pool_full.iloc[0:0], "key", max_group=50)
    for out in (empty_s1, empty_pool):
        assert len(out) == 0
        assert out["s1_idx"].dtype == np.int32 and out["pool_idx"].dtype == np.int32


def test_exact_pass_different_columns_yield_independent_pass_bits():
    """name_sorted and name_squash must carry their own bits (2, 4), not just name_core's."""
    s1n = _frame(["S1-1"], ["US"], ["zzz yyy"])
    pooln = _frame(["S2-1"], ["US"], ["yyy zzz"])  # different name_core, same sorted tokens
    s1n["name_sorted"] = pooln["name_sorted"] = "yyy zzz"  # both sort to the same key
    pooln["name_squash"] = "othervalue"  # ensure no accidental squash match
    core = exact_pass(s1n, pooln, "name_core", 50)
    sorted_ = exact_pass(s1n, pooln, "name_sorted", 50)
    assert len(core) == 0  # "zzz yyy" != "yyy zzz"
    assert len(sorted_) == 1
    out = union_passes({"name_sorted": sorted_}, max_per_s1=10)
    assert out.iloc[0]["pass"] == PASS_BITS["name_sorted"] == 2


# -------------------------------------------------------------------- B: topk_pass ----
def test_topk_pass_empty_s1_and_empty_pool():
    s1n = pd.DataFrame({"text": ["acme corp"]})
    pooln = pd.DataFrame({"text": ["acme corp"]})
    spec = TopKSpec("text", min_df=1, max_df=1.0, top_k=3, min_sim=0.05)
    for a, b in ((s1n.iloc[0:0], pooln), (s1n, pooln.iloc[0:0])):
        out = topk_pass(a, b, spec, 50, 50, 1, 0)
        assert len(out) == 0
        assert list(out.columns) == ["s1_idx", "pool_idx", "sim"]


def test_topk_pass_single_row_each_side():
    s1n = pd.DataFrame({"text": ["acme corp"]})
    pooln = pd.DataFrame({"text": ["acme corp"]})
    spec = TopKSpec("text", min_df=1, max_df=1.0, top_k=3, min_sim=0.05)
    out = topk_pass(s1n, pooln, spec, 50, 50, 1, 0)
    assert out[["s1_idx", "pool_idx"]].iloc[0].tolist() == [0, 0]
    assert out["sim"].iloc[0] == pytest.approx(1.0)


def test_topk_pass_no_similarity_above_threshold_is_empty():
    s1n = pd.DataFrame({"text": ["xxxxx yyyyy zzzzz"]})
    pooln = pd.DataFrame({"text": ["qqqqq wwwww eeeee"]})  # zero shared 3-grams
    spec = TopKSpec("text", min_df=1, max_df=1.0, top_k=50, min_sim=0.0)
    out = topk_pass(s1n, pooln, spec, 50, 50, 1, 0)
    assert len(out) == 0


def test_topk_pass_threshold_is_strict_not_inclusive():
    """sparse_dot_topn's threshold keeps values strictly greater than min_sim."""
    s1n = pd.DataFrame({"text": ["acme corp trading"]})
    pooln = pd.DataFrame({"text": ["acme corp trading company extra words here padding"]})
    probe = TopKSpec("text", min_df=1, max_df=1.0, top_k=5, min_sim=0.0)
    sim = topk_pass(s1n, pooln, probe, 50, 50, 1, 0)["sim"].iloc[0]

    at_threshold = TopKSpec("text", min_df=1, max_df=1.0, top_k=5, min_sim=float(sim))
    below_threshold = TopKSpec("text", min_df=1, max_df=1.0, top_k=5, min_sim=float(sim) - 1e-6)
    assert len(topk_pass(s1n, pooln, at_threshold, 50, 50, 1, 0)) == 0
    assert len(topk_pass(s1n, pooln, below_threshold, 50, 50, 1, 0)) == 1


def test_topk_pass_top_k_one_keeps_only_the_best():
    s1n = pd.DataFrame({"text": ["acme corp trading"]})
    pooln = pd.DataFrame({"text": ["acme corp trading exact", "acme corp trad", "acme cor"]})
    spec = TopKSpec("text", min_df=1, max_df=1.0, top_k=1, min_sim=0.0)
    out = topk_pass(s1n, pooln, spec, 50, 50, 1, 0)
    assert len(out) == 1
    assert out["pool_idx"].iloc[0] == 0  # "acme corp trading exact" is the closest


def test_topk_pass_top_k_exceeds_available_candidates():
    s1n = pd.DataFrame({"text": ["acme corp"]})
    pooln = pd.DataFrame({"text": ["acme corp", "acme corpo"]})
    spec = TopKSpec("text", min_df=1, max_df=1.0, top_k=10, min_sim=0.0)
    out = topk_pass(s1n, pooln, spec, 50, 50, 1, 0)
    assert len(out) == 2  # no padding, no error, just however many exist


def test_topk_pass_empty_string_row_yields_no_candidates_for_that_row():
    s1n = pd.DataFrame({"text": ["", "acme corp"]})
    pooln = pd.DataFrame({"text": ["acme corp", "widget inc"]})
    spec = TopKSpec("text", min_df=1, max_df=1.0, top_k=5, min_sim=0.0)
    out = topk_pass(s1n, pooln, spec, 50, 50, 1, 0)
    assert 0 not in out["s1_idx"].to_numpy()  # the empty-text row matches nothing
    assert 1 in out["s1_idx"].to_numpy()


def test_topk_pass_all_empty_text_is_empty():
    s1n = pd.DataFrame({"text": ["", ""]})
    pooln = pd.DataFrame({"text": ["", ""]})
    spec = TopKSpec("text", min_df=1, max_df=1.0, top_k=5, min_sim=0.0)
    assert len(topk_pass(s1n, pooln, spec, 50, 50, 1, 0)) == 0


def test_topk_pass_none_value_raises_out_of_contract():
    """Mirrors test_exact_pass_none_is_not_treated_as_empty: a None reaching the
    vectoriser is out of normalize.py's contract and fails loudly, not silently."""
    s1n = pd.DataFrame({"text": ["acme corp", None]})
    pooln = pd.DataFrame({"text": ["acme corp", "widget inc"]})
    spec = TopKSpec("text", min_df=1, max_df=1.0, top_k=3, min_sim=0.05)
    with pytest.raises(ValueError):
        topk_pass(s1n, pooln, spec, 50, 50, 1, 0)


def test_topk_pass_duplicate_pool_text_each_row_is_its_own_candidate():
    s1n = pd.DataFrame({"text": ["acme corp"]})
    pooln = pd.DataFrame({"text": ["acme corp", "acme corp", "widget inc"]})
    spec = TopKSpec("text", min_df=1, max_df=1.0, top_k=5, min_sim=0.05)
    out = topk_pass(s1n, pooln, spec, 50, 50, 1, 0)
    assert set(out["pool_idx"]) == {0, 1}
    assert np.allclose(out["sim"].to_numpy(), 1.0)


def test_topk_pass_duplicate_s1_rows_retrieve_independently():
    s1n = pd.DataFrame({"text": ["acme corp", "acme corp"]})
    pooln = pd.DataFrame({"text": ["acme corp", "widget inc"]})
    spec = TopKSpec("text", min_df=1, max_df=1.0, top_k=5, min_sim=0.05)
    out = topk_pass(s1n, pooln, spec, 50, 50, 1, 0)
    assert set(out.loc[out["s1_idx"] == 0, "pool_idx"]) == {0}
    assert set(out.loc[out["s1_idx"] == 1, "pool_idx"]) == {0}


def test_topk_pass_s1_side_chunking_is_invariant():
    """Mirrors the existing pool_chunk invariance test, but forces the S1 loop to split."""
    s1n = pd.DataFrame({"text": [f"acme trading company {i}" for i in range(10)]})
    pooln = pd.DataFrame({"text": [f"acme trading compnay {i}" for i in range(10)]})
    spec = TopKSpec("text", min_df=1, max_df=1.0, top_k=3, min_sim=0.05)
    one_chunk = topk_pass(s1n, pooln, spec, s1_chunk=50, pool_chunk=50, n_threads=1, seed=0)
    many_chunks = topk_pass(s1n, pooln, spec, s1_chunk=3, pool_chunk=50, n_threads=1, seed=0)
    one_chunk = one_chunk.sort_values(["s1_idx", "pool_idx"]).reset_index(drop=True)
    many_chunks = many_chunks.sort_values(["s1_idx", "pool_idx"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(one_chunk, many_chunks, check_exact=False, atol=1e-6)


def test_topk_pass_deterministic_across_repeated_calls():
    s1n = pd.DataFrame({"text": [f"acme trading company {i}" for i in range(15)]})
    pooln = pd.DataFrame({"text": [f"acme trading compnay {i}" for i in range(15)]})
    spec = TopKSpec("text", min_df=1, max_df=1.0, top_k=3, min_sim=0.05)
    runs = [topk_pass(s1n, pooln, spec, 5, 7, 3, 0) for _ in range(3)]
    for r in runs[1:]:
        pd.testing.assert_frame_equal(runs[0], r)  # identical, including row order


# ------------------------------------------------------------------- C: union_passes ----
def test_union_passes_single_pass_only():
    exact = pd.DataFrame({"s1_idx": [0], "pool_idx": [0]})
    out = union_passes({"name_core": exact}, max_per_s1=10)
    assert out.iloc[0]["pass"] == PASS_BITS["name_core"]
    assert out[list(blocking.SIM_COLUMNS)].isna().all(axis=None)


def test_union_passes_pair_only_in_topk_no_exact():
    topk = pd.DataFrame({"s1_idx": [0], "pool_idx": [0], "sim": [0.6]})
    out = union_passes({"addr_char": topk}, max_per_s1=10)
    assert out.iloc[0]["pass"] == PASS_BITS["addr_char"]
    assert out.iloc[0]["sim_addr_char"] == pytest.approx(0.6)


def test_union_passes_pair_only_in_one_exact_pass_bit_is_exact():
    exact = pd.DataFrame({"s1_idx": [0], "pool_idx": [0]})
    out = union_passes({"name_sorted": exact}, max_per_s1=10)
    assert out.iloc[0]["pass"] == PASS_BITS["name_sorted"] == 2


def test_union_passes_duplicate_rows_within_one_pass_keep_max_sim():
    """Simulates a chunk-boundary duplicate: the same pair appearing twice in one
    pass's raw output (e.g. from two overlapping topk_pass chunks) must collapse to
    a single row with the higher similarity, not sum or average them."""
    topk = pd.DataFrame({"s1_idx": [0, 0], "pool_idx": [0, 0], "sim": [0.3, 0.8]})
    out = union_passes({"name_char": topk}, max_per_s1=10)
    assert len(out) == 1
    assert out.iloc[0]["sim_name_char"] == pytest.approx(0.8)


def test_union_passes_max_per_s1_one_keeps_the_single_best():
    exact = pd.DataFrame({"s1_idx": [0], "pool_idx": [0]})
    topk = pd.DataFrame({"s1_idx": [0], "pool_idx": [1], "sim": [0.9]})
    out = union_passes({"name_core": exact, "name_char": topk}, max_per_s1=1)
    assert len(out) == 1
    assert out.iloc[0]["pool_idx"] == 0  # exact still wins even against a high sim


def test_union_passes_max_per_s1_zero_keeps_nothing():
    topk = pd.DataFrame({"s1_idx": [0, 0], "pool_idx": [0, 1], "sim": [0.9, 0.5]})
    out = union_passes({"name_char": topk}, max_per_s1=0)
    assert len(out) == 0


def test_union_passes_max_per_s1_larger_than_available_keeps_all():
    topk = pd.DataFrame({"s1_idx": [0, 0], "pool_idx": [0, 1], "sim": [0.9, 0.5]})
    out = union_passes({"name_char": topk}, max_per_s1=1000)
    assert len(out) == 2


def test_union_passes_caps_are_independent_per_s1():
    topk = pd.DataFrame({"s1_idx": [0, 0, 0, 1, 1], "pool_idx": [0, 1, 2, 0, 1],
                         "sim": [0.9, 0.5, 0.1, 0.8, 0.2]})
    out = union_passes({"name_char": topk}, max_per_s1=2)
    counts = out.groupby("s1_idx").size()
    assert counts.loc[0] == 2 and counts.loc[1] == 2
    assert set(out.loc[out["s1_idx"] == 0, "pool_idx"]) == {0, 1}  # not the lowest-sim one


def test_union_passes_tie_break_is_deterministic_across_calls():
    topk = pd.DataFrame({"s1_idx": [0, 0, 0], "pool_idx": [5, 3, 8], "sim": [0.5, 0.5, 0.5]})
    runs = [union_passes({"name_char": topk}, max_per_s1=2) for _ in range(3)]
    for r in runs[1:]:
        pd.testing.assert_frame_equal(runs[0], r)
    assert set(runs[0]["pool_idx"]) == {3, 5}  # lower pool_idx wins a similarity tie


def test_union_passes_no_duplicate_pairs_when_every_pass_hits_the_same_pair():
    same_pair_exact = {k: pd.DataFrame({"s1_idx": [0], "pool_idx": [0]})
                       for k in ("name_core", "name_sorted", "name_squash")}
    same_pair_topk = {k: pd.DataFrame({"s1_idx": [0], "pool_idx": [0], "sim": [0.7]})
                      for k in ("name_char", "name_addr_word", "addr_char")}
    out = union_passes({**same_pair_exact, **same_pair_topk}, max_per_s1=10)
    assert len(out) == 1
    assert out.iloc[0]["pass"] == sum(PASS_BITS.values()) == 63


def test_union_passes_unknown_pass_name_raises():
    with pytest.raises(ValueError):
        union_passes({"not_a_real_pass": pd.DataFrame({"s1_idx": [0], "pool_idx": [0]})}, 10)


# -------------------------------------------------------------- D: block_partition/block ----
def test_block_country_present_only_in_s1_yields_no_candidates_no_crash():
    s1 = _frame(["S1-1"], ["Mars"], ["acme corp"])
    pool = _frame(["S2-1"], ["US"], ["acme corp"])
    pairs = block(s1, pool, _tiny_cfg())
    assert len(pairs) == 0
    assert list(pairs.columns) == list(PAIR_COLUMNS)


def test_block_country_present_only_in_pool_is_never_a_candidate():
    s1 = _frame(["S1-1"], ["US"], ["acme corp"])
    pool = _frame(["S2-1", "S2-2"], ["US", "Neptune"], ["acme corp", "acme corp"])
    pairs = block(s1, pool, _tiny_cfg())
    assert "S2-2" not in set(pairs[C.ENTITY_ID])


def test_block_empty_string_country_partitions_like_any_other():
    s1 = _frame(["S1-1"], [""], ["acme corp"])
    pool = _frame(["S2-1"], [""], ["acme corp"])
    pairs = block(s1, pool, _tiny_cfg())
    assert list(pairs[C.ENTITY_ID]) == ["S2-1"]


def test_block_dtype_stable_when_one_partition_is_empty():
    """One country (Mars) has an S1 row but zero pool rows, so its block_partition
    call returns the empty-frame branch; concatenating it with a real partition must
    not silently upcast pass/sim dtypes."""
    s1 = _frame(["S1-1", "S1-2"], ["US", "Mars"], ["acme corp", "no match here"])
    pool = _frame(["S2-1"], ["US"], ["acme corp"])
    pairs = block(s1, pool, _tiny_cfg())
    assert pairs["pass"].dtype == np.uint8
    for col in blocking.SIM_COLUMNS:
        assert pairs[col].dtype == np.float32


def test_block_repeated_calls_without_cache_are_identical():
    s1, pool, cfg = _two_country_frames()
    runs = [block(s1, pool, cfg) for _ in range(3)]
    for r in runs[1:]:
        pd.testing.assert_frame_equal(runs[0], r)


def test_block_invariants_hold_across_countries_and_cap():
    """One property test exercising every invariant from 06_BLOCKING_STRATEGY.md at once:
    id membership, no duplicate pairs, valid similarity range, valid pass bitmask, country
    match, and the max_per_s1 cap, on a fixture designed to make cross-country leakage or
    a broken cap show up immediately (identical names reused in every country)."""
    countries = ["US", "India", "France", "Germany"]
    s1_rows, pool_rows = [], []
    for i, country in enumerate(countries):
        s1_rows.append((f"S1-{i}", country, "acme corp", "acme corp 1 main st"))
        for j in range(6):
            pool_rows.append((f"S2-{i}-{j}", country, f"acme corp {j}", f"acme corp {j} main st"))
    s1 = pd.DataFrame(s1_rows, columns=[C.ENTITY_ID, C.COUNTRY, "name_core", "name_addr"])
    s1["name_sorted"] = s1["name_core"]
    s1["name_squash"] = s1["name_core"].str.replace(" ", "", regex=False)
    pool = pd.DataFrame(pool_rows, columns=[C.ENTITY_ID, C.COUNTRY, "name_core", "name_addr"])
    pool["name_sorted"] = pool["name_core"]
    pool["name_squash"] = pool["name_core"].str.replace(" ", "", regex=False)

    cfg = _tiny_cfg(name_char=TopKSpec("name_core", min_df=1, max_df=1.0, top_k=10, min_sim=0.01),
                    name_addr_word=TopKSpec("name_addr", "word", (1, 1), 10, 0.01, 1.0, 1),
                    max_per_s1=2)
    pairs = block(s1, pool, cfg)

    s1_ids, pool_ids = set(s1[C.ENTITY_ID]), set(pool[C.ENTITY_ID])
    assert set(pairs[C.S1_ID]) <= s1_ids
    assert set(pairs[C.ENTITY_ID]) <= pool_ids
    assert pairs[C.S1_ID].notna().all() and pairs[C.ENTITY_ID].notna().all()
    assert not pairs.duplicated([C.S1_ID, C.ENTITY_ID]).any()
    assert (pairs.groupby(C.S1_ID).size() <= 2).all()  # max_per_s1

    valid_bits = sum(PASS_BITS.values())
    assert ((pairs["pass"].astype(int) & ~valid_bits) == 0).all()

    country_of_s1 = dict(zip(s1[C.ENTITY_ID], s1[C.COUNTRY], strict=True))
    country_of_pool = dict(zip(pool[C.ENTITY_ID], pool[C.COUNTRY], strict=True))
    assert all(country_of_s1[s] == country_of_pool[p]
              for s, p in zip(pairs[C.S1_ID], pairs[C.ENTITY_ID], strict=True))

    for col in blocking.SIM_COLUMNS:
        vals = pairs[col].dropna()
        assert ((vals > 0) & (vals <= 1)).all()


# ------------------------------------------------------------------------ E: cache ----
def test_block_runs_without_a_cache_dir():
    s1, pool, cfg = _two_country_frames()
    pairs = block(s1, pool, cfg)  # cache_dir defaults to None
    assert list(pairs.columns) == list(PAIR_COLUMNS)


def test_block_cache_creates_files_on_first_call(tmp_path):
    s1, pool, cfg = _two_country_frames()
    cache_dir = tmp_path / "cache"
    assert not cache_dir.exists()
    block(s1, pool, cfg, cache_dir=cache_dir, tag="val")
    assert list(cache_dir.rglob("*.parquet"))


def test_block_different_configs_get_different_cache_entries(tmp_path):
    s1, pool, _ = _two_country_frames()
    cache_dir = tmp_path / "cache"
    cfg_wide = _tiny_cfg(max_per_s1=60)
    cfg_narrow = _tiny_cfg(max_per_s1=1)

    wide = block(s1, pool, cfg_wide, cache_dir=cache_dir, tag="val")
    narrow = block(s1, pool, cfg_narrow, cache_dir=cache_dir, tag="val")
    files = list(cache_dir.rglob("*.parquet"))
    assert len({f.name for f in files}) == 2  # distinct hashes, no collision
    assert not narrow.groupby(C.S1_ID).size().gt(1).any()  # the narrower cap actually applied
    assert len(narrow) <= len(wide)


def test_block_cache_tag_isolates_namespace(tmp_path):
    s1, pool, cfg = _two_country_frames()
    cache_dir = tmp_path / "cache"
    fit = block(s1, pool, cfg, cache_dir=cache_dir, tag="fit")
    tune = block(s1, pool, cfg, cache_dir=cache_dir, tag="tune")
    pd.testing.assert_frame_equal(fit, tune)  # same data, same config -> same result
    assert (cache_dir / "fit").exists() and (cache_dir / "tune").exists()


def test_block_corrupt_cache_file_fails_loudly(tmp_path):
    """No corrupt-cache recovery is required by the spec; this documents that a bad
    cache file surfaces a clear error instead of silently returning wrong candidates."""
    s1, pool, cfg = _two_country_frames()
    cache_dir = tmp_path / "cache"
    block(s1, pool, cfg, cache_dir=cache_dir, tag="val")
    for f in cache_dir.rglob("*.parquet"):
        f.write_bytes(b"not a parquet file")
    with pytest.raises(Exception):  # noqa: B017 - pyarrow's own exception type, not ours to pin
        block(s1, pool, cfg, cache_dir=cache_dir, tag="val")


# -------------------------------------------------------------------- F: to_id_lists ----
def test_to_id_lists_preserves_input_row_order():
    pairs = pd.DataFrame({C.S1_ID: ["S1-1", "S1-1", "S1-1"], C.ENTITY_ID: ["S3-9", "S2-1", "S2-5"]})
    out = to_id_lists(pairs, pd.Series(["S1-1"]))
    assert out["S1-1"] == ["S3-9", "S2-1", "S2-5"]


def test_to_id_lists_duplicate_pairs_are_not_deduplicated():
    """to_id_lists trusts block()'s own uniqueness guarantee; fed raw duplicates (as in
    a direct unit test) it preserves them as-is rather than silently deduping."""
    pairs = pd.DataFrame({C.S1_ID: ["S1-1", "S1-1"], C.ENTITY_ID: ["S2-1", "S2-1"]})
    out = to_id_lists(pairs, pd.Series(["S1-1"]))
    assert out["S1-1"] == ["S2-1", "S2-1"]


def test_to_id_lists_empty_pairs_frame_maps_everything_to_empty():
    pairs = pd.DataFrame({C.S1_ID: pd.Series([], dtype="str"),
                          C.ENTITY_ID: pd.Series([], dtype="str")})
    out = to_id_lists(pairs, pd.Series(["S1-1", "S1-2"]))
    assert out == {"S1-1": [], "S1-2": []}


def test_to_id_lists_unusual_but_valid_ids():
    pairs = pd.DataFrame({C.S1_ID: ["S1-Ünïcödé", "S1-Ünïcödé"], C.ENTITY_ID: ["S2-1", "S2#2"]})
    out = to_id_lists(pairs, pd.Series(["S1-Ünïcödé"]))
    assert out["S1-Ünïcödé"] == ["S2-1", "S2#2"]


def test_to_id_lists_multiple_s1_ids_each_get_their_own_candidates():
    pairs = pd.DataFrame({C.S1_ID: ["S1-1", "S1-2", "S1-2"], C.ENTITY_ID: ["S2-1", "S2-2", "S2-3"]})
    out = to_id_lists(pairs, pd.Series(["S1-1", "S1-2", "S1-3"]))
    assert out == {"S1-1": ["S2-1"], "S1-2": ["S2-2", "S2-3"], "S1-3": []}
