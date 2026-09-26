import numpy as np
import pandas as pd
import pytest

from entity_resolution import blocking
from entity_resolution import config as C
from entity_resolution.blocking import (
    EXACT_BITS,
    PAIR_COLUMNS,
    PASS_BITS,
    SIM_COLUMNS,
    BlockingConfig,
    TopKSpec,
    block,
    exact_pass,
    to_id_lists,
    union_passes,
)
from entity_resolution.data import load_ground_truth, load_sources
from entity_resolution.metrics import candidate_report
from entity_resolution.normalize import normalise_records

PAIR_DTYPES = ["str", "str", "uint8", "float32", "float32", "float32"]

S1 = [("S1-01", "Acme Corp", "12 Main St, Springfield, IL", "US"),
      ("S1-02", "Sharma Traders Pvt Ltd", "Near SBI ATM, MG Road, Pune", "India"),
      ("S1-03", "Boulangerie Dupont SARL", "5 Rue de la Paix, Paris", "France"),
      ("S1-04", "Zyxwv Qqq", "", "US"),
      ("S1-05", "Acme Corp", "Berlin", "Germany")]  # no pool record in its country
POOL = [("S2-01", "ACME Corporation", "12 Main Street, Springfield", "US"),
        ("S2-02", "Sharma Traders Private Limited", "MG Rd, Pune 411001", "India"),
        ("S2-03", "Boulangerie Dupont", "5 rue de la Paix, 75002 Paris", "France"),
        ("S3-01", "Acme Corp.", "12 Main St", "US"),
        ("S3-02", "Acme Corp", "12 Main St, Pune", "India"),  # same name, other country
        ("S3-03", "Globex LLC", "1 Elm Rd, Austin, TX", "US")]
NAMES = ["Kaveri Silk Emporium", "Meridian Freight Lines", "Blue Lotus Bakery",
         "Summit Ridge Dental", "Orchid Valley Farms", "Granite Peak Roofing",
         "Saffron Spice Kitchen", "Harbor Light Marine", "Crimson Oak Furniture",
         "Silver Birch Pharmacy", "Golden Wheat Mills", "Evergreen Pine Nursery",
         "Copper Kettle Diner", "Northwind Aviation", "Lakeside Auto Repair",
         "Maple Leaf Printing", "Riverstone Construction", "Sunflower Daycare",
         "Thunderbolt Electricals", "Velvet Rose Boutique"]


def _src(*rows: tuple) -> pd.DataFrame:
    """Source frame (entity_id, business_name, business_address, country), all "str"."""
    return pd.DataFrame(list(rows), columns=list(C.SOURCE_COLUMNS)).astype("str")


def _cfg(**kw) -> BlockingConfig:
    """Small-data configuration: every gram counts (min_df=1, max_df=1.0), one thread."""
    base = {"name_char": TopKSpec("name_core", top_k=10, min_sim=0.3, min_df=1, max_df=1.0),
            "name_addr_word": TopKSpec("name_addr", "word", (1, 2), 20, 0.2, min_df=1,
                                       max_df=1.0),
            "n_threads": 1, "vocab_sample": 10_000}
    return BlockingConfig(**{**base, **kw})


def _pairset(pairs: pd.DataFrame) -> set[tuple[str, str]]:
    """The (source1_entity_id, entity_id) pairs of a pairs frame."""
    return set(zip(pairs[C.S1_ID], pairs[C.ENTITY_ID], strict=True))


def _keys(*keys: str) -> pd.DataFrame:
    """A frame holding only the name_core exact key."""
    return pd.DataFrame({"name_core": pd.Series(keys, dtype="str")})


def _idx(s1: list[int], pool: list[int], sim: list[float] | None = None) -> pd.DataFrame:
    """One pass's positional pairs, with its similarities for a top-k pass."""
    df = pd.DataFrame({"s1_idx": np.int32(s1), "pool_idx": np.int32(pool)})
    return df if sim is None else df.assign(sim=np.float32(sim))


def _typo(name: str) -> str:
    """The name with its middle letter replaced ("Kaveri Silk" -> "Kaveri Silx")."""
    k = len(name) // 2 + (name[len(name) // 2] == " ")
    return name[:k] + "x" + name[k + 1:]


def _typo_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """20 S1 names and a pool of their one-typo copies (S1-i matches S2-i)."""
    s1 = _src(*[(f"S1-{i:02d}", n, "", "US") for i, n in enumerate(NAMES)])
    pool = _src(*[(f"S2-{i:02d}", _typo(n), "", "US") for i, n in enumerate(NAMES)])
    return normalise_records(s1), normalise_records(pool)


@pytest.fixture
def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalised S1 and pool over four countries, France and a pool-less one included."""
    return normalise_records(_src(*S1)), normalise_records(_src(*POOL))


def test_exact_pass_respects_max_group():
    """A pool key group above max_group yields nothing; one at the limit yields every pair."""
    s1n = _keys("acme", "globex")
    pooln = _keys("acme", "globex", "acme", "globex", "acme")
    out = exact_pass(s1n, pooln, "name_core", max_group=2)
    assert out[["s1_idx", "pool_idx"]].to_numpy().tolist() == [[1, 1], [1, 3]]
    assert out.dtypes.tolist() == [np.int32, np.int32]
    assert len(exact_pass(s1n, pooln, "name_core", max_group=3)) == 5


def test_exact_pass_skips_empty_key():
    """Empty keys never match, even each other."""
    out = exact_pass(_keys("", "acme"), _keys("", "", "acme"), "name_core", max_group=50)
    assert out.to_numpy().tolist() == [[1, 2]]


def test_topk_pass_finds_typos():
    """Name char 3-grams link each of 20 names to its one-typo copy within k=3."""
    s1n, pooln = _typo_frames()
    assert (s1n["name_core"] != pooln["name_core"]).all()
    spec = TopKSpec("name_core", top_k=3, min_sim=0.3, min_df=1, max_df=1.0)
    pairs = block(s1n, pooln, _cfg(exact_keys=(), name_char=spec, name_addr_word=None))
    assert {(f"S1-{i:02d}", f"S2-{i:02d}") for i in range(20)} <= _pairset(pairs)
    assert pairs.groupby(C.S1_ID).size().max() <= 3
    assert (pairs["pass"] == PASS_BITS["name_char"]).all()
    assert ((pairs["sim_name_char"] > 0) & (pairs["sim_name_char"] <= 1)).all()
    assert pairs[["sim_name_addr_word", "sim_addr_char"]].isna().all().all()


def test_union_bitmask_and_max_sim():
    """Pass bits OR together, each top-k pass keeps its best sim, absent passes are NaN."""
    parts = {"exact_core": _idx([0, 0], [1, 5]),
             "name_char": _idx([0, 0, 0, 1], [1, 1, 2, 3], [0.6, 0.8, 0.5, 0.4]),
             "name_addr_word": _idx([0], [2], [0.7])}
    out = union_passes(parts, max_per_s1=60)
    assert list(out.columns) == ["s1_idx", "pool_idx", "pass", *SIM_COLUMNS]
    assert out.dtypes.astype(str).tolist() == ["int32", "int32", "uint8"] + ["float32"] * 3
    assert out[["s1_idx", "pool_idx"]].to_numpy().tolist() == [[0, 1], [0, 2], [0, 5], [1, 3]]
    assert out["pass"].tolist() == [9, 24, 1, 8]  # exact_core|name_char, name_char|name_addr
    assert out["sim_name_char"].iloc[0] == np.float32(0.8)  # max of the pass's two rows
    assert out["sim_name_addr_word"].iloc[1] == np.float32(0.7)
    assert out["sim_name_addr_word"].isna().tolist() == [True, False, True, True]
    assert out["sim_addr_char"].isna().all()
    assert out[SIM_COLUMNS].iloc[2].isna().all()  # exact-only pair


def test_union_cap_prefers_exact_then_sim():
    """max_per_s1 keeps exact pairs first, then the best sims; ties go to the lower pool."""
    parts = {"exact_sorted": _idx([0], [9]),
             "name_char": _idx([0, 0, 1, 1, 1], [1, 2, 7, 4, 2], [0.9, 0.3, 0.5, 0.5, 0.5]),
             "name_addr_word": _idx([0], [3], [0.6])}

    def kept(cap: int) -> dict[int, list[int]]:
        """Pool positions kept per S1 position under the cap."""
        return union_passes(parts, cap).groupby("s1_idx")["pool_idx"].agg(list).to_dict()

    assert kept(2) == {0: [1, 9], 1: [2, 4]}  # 0.6 and 0.3 dropped; the 0.5 tie keeps 2, 4
    assert kept(1) == {0: [9], 1: [2]}  # the exact pair beats a 0.9 similarity
    assert kept(60) == {0: [1, 2, 3, 9], 1: [2, 4, 7]}
    assert len(union_passes({}, 60)) == 0


def test_block_partitions_by_country(frames):
    """No pair crosses countries; France and a pool-less country need nothing special."""
    s1n, pooln = frames
    pairs = block(s1n, pooln, _cfg())
    s1_country = dict(zip(s1n[C.ENTITY_ID], s1n[C.COUNTRY], strict=True))
    pool_country = dict(zip(pooln[C.ENTITY_ID], pooln[C.COUNTRY], strict=True))
    assert all(s1_country[a] == pool_country[b] for a, b in _pairset(pairs))
    assert ("S1-01", "S3-02") not in _pairset(pairs)  # same name, but India
    assert pairs.loc[pairs[C.S1_ID] == "S1-03", C.ENTITY_ID].tolist() == ["S2-03"]
    assert "S1-05" not in set(pairs[C.S1_ID])


def test_block_output_schema(frames):
    """PAIR_COLUMNS and dtypes; unique non-empty pairs sorted by S1, then pool order."""
    s1n, pooln = frames
    pairs = block(s1n, pooln, _cfg())
    assert list(pairs.columns) == PAIR_COLUMNS
    assert pairs.dtypes.astype(str).tolist() == PAIR_DTYPES
    assert len(pairs) > 0 and not pairs.duplicated([C.S1_ID, C.ENTITY_ID]).any()
    assert pairs[C.S1_ID].is_monotonic_increasing
    assert (pairs[C.S1_ID] != "").all() and (pairs[C.ENTITY_ID] != "").all()
    position = {e: i for i, e in enumerate(pooln[C.ENTITY_ID])}
    order = pairs[C.ENTITY_ID].map(position).groupby(pairs[C.S1_ID])
    assert order.apply(lambda s: s.is_monotonic_increasing).all()
    assert ((pairs["pass"] > 0) & (pairs["pass"] < 64)).all()


def test_s1_chunking_invariant():
    """Querying S1 in chunks of 3 gives the same pairs and sims as one chunk."""
    s1n, pooln = _typo_frames()
    pd.testing.assert_frame_equal(block(s1n, pooln, _cfg(s1_chunk=3)), block(s1n, pooln, _cfg()))


def test_topk_pass_chunking_invariant():
    """Querying the pool in two chunks gives the same pairs and sims as one chunk."""
    s1n, pooln = _typo_frames()
    full = block(s1n, pooln, _cfg(pool_chunk=len(pooln)))
    half = block(s1n, pooln, _cfg(pool_chunk=len(pooln) // 2))
    pd.testing.assert_frame_equal(full, half)


def test_name_char_searches_only_short_address_pool_rows():
    """With pool_max_addr_tokens=3 the char pass skips pool rows with longer addresses."""
    s1n = normalise_records(_src(("S1-01", "Kaveri Textiles", "12 MG Road, Pune 411001",
                                  "India")))
    far = "45 Linking Road, Bandra West, Mumbai, Maharashtra 400050"
    pooln = normalise_records(_src(("S2-01", "Globex Industries", far, "India"),
                                   ("S2-02", "Kaveri Textilez", far, "India"),
                                   ("S2-03", "Kaveri Textile", "Pune", "India")))
    assert pooln["addr_tokens"].tolist() == [8, 8, 1]

    def found(limit: int | None) -> set[tuple[str, str]]:
        """Pairs of a name-char-only run with this pool address limit."""
        spec = TopKSpec("name_core", top_k=10, min_sim=0.3, min_df=1, max_df=1.0,
                        pool_max_addr_tokens=limit)
        cfg = _cfg(exact_keys=(), name_char=spec, name_addr_word=None)
        return _pairset(block(s1n, pooln, cfg))

    assert found(3) == {("S1-01", "S2-03")}  # mapped back to the short row's own id
    assert found(None) == {("S1-01", "S2-02"), ("S1-01", "S2-03")}


def test_to_id_lists_every_s1_present():
    """Every S1 id gets a list, in the given order, [] when it has no candidate."""
    pairs = pd.DataFrame({C.S1_ID: ["S1-1", "S1-1", "S1-3"],
                          C.ENTITY_ID: ["S2-1", "S3-4", "S2-9"]}).astype("str")
    lists = to_id_lists(pairs, pd.Series(["S1-3", "S1-2", "S1-1"], dtype="str"))
    assert list(lists) == ["S1-3", "S1-2", "S1-1"]
    assert lists == {"S1-3": ["S2-9"], "S1-2": [], "S1-1": ["S2-1", "S3-4"]}


def test_to_id_lists_of_block_output(frames):
    """On block output: all S1 present, no duplicate candidates, [] for the unmatched."""
    s1n, pooln = frames
    lists = to_id_lists(block(s1n, pooln, _cfg()), s1n[C.ENTITY_ID])
    assert list(lists) == s1n[C.ENTITY_ID].tolist()
    assert all(len(v) == len(set(v)) for v in lists.values())
    assert lists["S1-04"] == [] and lists["S1-05"] == []
    assert lists["S1-01"] == ["S2-01", "S3-01"]


def test_conftest_truth_pairs_are_candidates(dataset_dir):
    """On the shared synthetic train split every true pair survives blocking."""
    src = load_sources("train", dataset_dir)
    s1n = normalise_records(src[1])
    # normalised per source, then joined, as the pipeline does
    pooln = pd.concat([normalise_records(src[2]), normalise_records(src[3])], ignore_index=True)
    pairs = block(s1n, pooln, _cfg())
    lists = to_id_lists(pairs, src[1][C.ENTITY_ID])
    assert candidate_report(lists, load_ground_truth(dataset_dir))["pair_recall"] == 1.0


def test_block_cache_roundtrip(frames, tmp_path, monkeypatch):
    """A second call with cache_dir reads the per-country files and computes nothing."""
    s1n, pooln = frames
    cfg = _cfg()
    first = block(s1n, pooln, cfg, cache_dir=tmp_path, tag="val")
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    parquet = [f"val/{c}/pairs_{cfg.key()}.parquet" for c in ("France", "Germany", "India", "US")]
    assert [f for f in files if f.endswith(".parquet")] == parquet
    assert not [f for f in files if f.endswith(".tmp")]
    pd.testing.assert_frame_equal(first, block(s1n, pooln, cfg))  # uncached run agrees
    monkeypatch.setattr(blocking, "block_partition",
                        lambda *a, **k: pytest.fail("cached partition recomputed"))
    again = block(s1n, pooln, cfg, cache_dir=tmp_path, tag="val")
    pd.testing.assert_frame_equal(again, first)


def test_config_key_is_stable_and_config_sensitive():
    """The cache key is a fixed-length hash that changes with any setting."""
    assert BlockingConfig().key() == BlockingConfig().key()
    assert len(BlockingConfig().key()) == 8
    assert BlockingConfig(max_per_s1=10).key() != BlockingConfig().key()
    assert _cfg().key() != _cfg(name_char=None).key()
    small = TopKSpec("name_core", min_df=1, max_df=1.0)
    assert _cfg(name_char=small).key() != _cfg(name_char=TopKSpec("name_core")).key()


def test_block_two_runs_identical(frames):
    """Blocking is deterministic."""
    s1n, pooln = frames
    pd.testing.assert_frame_equal(block(s1n, pooln, _cfg()), block(s1n, pooln, _cfg()))


def test_block_empty_s1(frames):
    """No S1 rows gives zero pairs with the full schema."""
    s1n, pooln = frames
    for empty in (s1n.iloc[:0], normalise_records(_src())):
        out = block(empty, pooln, _cfg())
        assert len(out) == 0
        assert list(out.columns) == PAIR_COLUMNS
        assert out.dtypes.astype(str).tolist() == PAIR_DTYPES


# ============================================================== extra edge cases ====
# 1. P1a exact: cross-product, no-match, empty frames
def test_exact_pass_cross_product_no_match_and_empty():
    """A shared key cross-joins every S1 row with every pool row; a differing key matches
    nothing; and either side being empty yields zero pairs, not an error."""
    out = exact_pass(_keys("acme", "acme"), _keys("acme", "acme"), "name_core", max_group=50)
    assert sorted(out[["s1_idx", "pool_idx"]].to_numpy().tolist()) == [
        [0, 0], [0, 1], [1, 0], [1, 1]]  # every S1 row paired with every pool row

    assert len(exact_pass(_keys("acme"), _keys("globex"), "name_core", max_group=50)) == 0

    empty = _keys()
    for s1n, pooln in ((empty, empty), (_keys("acme"), empty), (empty, _keys("acme"))):
        out = exact_pass(s1n, pooln, "name_core", max_group=50)
        assert len(out) == 0
        assert out.dtypes.tolist() == [np.int32, np.int32]


# 2. P1b/P1c: each exact key contributes its own, independent pass bit
def test_exact_pass_different_columns_yield_independent_pass_bits():
    """name_core, name_sorted and name_squash each set only their own bit for a pair that
    matches on exactly one of the three keys."""
    s1n = pd.DataFrame({
        C.ENTITY_ID: ["S1-01", "S1-02", "S1-03"], C.COUNTRY: ["US"] * 3,
        "name_core": ["acme corp", "globex inc", "initech llc"],
        "name_sorted": ["acme corp", "globex inc", "initech llc"],
        "name_squash": ["acmecorp", "globexinc", "initechllc"],
    }).astype({C.ENTITY_ID: "str", C.COUNTRY: "str"})
    pooln = pd.DataFrame({
        C.ENTITY_ID: ["S2-01", "S2-02", "S2-03"], C.COUNTRY: ["US"] * 3,
        "name_core": ["corp acme", "globex incorporated", "initechllc"],  # no core match
        "name_sorted": ["acme corp", "incorporated globex", "different"],  # only row 0
        "name_squash": ["different", "different", "initechllc"],  # only row 2
    }).astype({C.ENTITY_ID: "str", C.COUNTRY: "str"})
    cfg = BlockingConfig(exact_keys=("name_core", "name_sorted", "name_squash"),
                         name_char=None, name_addr_word=None, addr_char=None)
    pairs = block(s1n, pooln, cfg)
    ids = list(zip(pairs[C.S1_ID], pairs[C.ENTITY_ID], strict=True))
    by_pair = dict(zip(ids, pairs["pass"], strict=True))
    assert by_pair == {("S1-01", "S2-01"): PASS_BITS["exact_sorted"],
                       ("S1-03", "S2-03"): PASS_BITS["exact_squash"]}


# 3. Group-size cap: strict >, pool-side only
def test_exact_pass_group_cap_boundary_and_sides():
    """The cap is a strict `>` on the pool-side group size only; S1-side repeats and an
    oversized S1 group never affect it."""
    s1n = _keys(*(["acme"] * 10))
    at_cap = exact_pass(s1n, _keys(*(["acme"] * 2)), "name_core", max_group=2)
    over_cap = exact_pass(s1n, _keys(*(["acme"] * 3)), "name_core", max_group=2)
    assert len(at_cap) == 10 * 2   # exactly at the cap: kept, not skipped (> not >=)
    assert len(over_cap) == 0      # one row over the cap: the whole pool group is skipped

    big_s1 = _keys(*(["acme"] * 100))
    assert len(exact_pass(big_s1, _keys("acme"), "name_core", max_group=2)) == 100  # S1 uncapped
    assert len(exact_pass(big_s1, _keys(*(["acme"] * 3)), "name_core", max_group=2)) == 0


# 4. Empty-key handling: whitespace boundary, out-of-contract None, empty frames
def test_exact_pass_empty_key_edge_cases():
    """Only the literal empty string "" is filtered; whitespace is a real key; a stray
    None (out of the "always str" schema contract) never matches but never crashes either."""
    out = exact_pass(_keys(" ", "acme"), _keys(" ", " ", "acme"), "name_core", max_group=50)
    assert out.to_numpy().tolist() == [[0, 0], [0, 1], [1, 2]]

    s1_none = pd.DataFrame({"name_core": pd.Series(["acme", None], dtype="object")})
    pool_none = pd.DataFrame({"name_core": pd.Series([None, "acme"], dtype="object")})
    assert exact_pass(s1_none, pool_none, "name_core", max_group=50).to_numpy().tolist() == [[0, 1]]

    empty = _keys()
    for s1n, pooln in ((empty, empty), (_keys("acme"), empty), (empty, _keys("acme"))):
        assert len(exact_pass(s1n, pooln, "name_core", max_group=50)) == 0


# 5. P2 name char-gram: threshold exclusivity, top_k, duplicate text
def test_topk_threshold_is_exclusive():
    """min_sim keeps only sims strictly greater than it, matching sp_matmul_topn's `>`."""
    text = pd.Series(["acme corp"])
    at = blocking.TopK(TopKSpec("name_core", top_k=5, min_sim=1.0, min_df=1, max_df=1.0),
                       text, text, vocab_sample=10, seed=1, n_threads=1)
    assert len(at.query(text)) == 0  # an exact-equal similarity (1.0) is not kept

    below = blocking.TopK(TopKSpec("name_core", top_k=5, min_sim=0.999, min_df=1, max_df=1.0),
                          text, text, vocab_sample=10, seed=1, n_threads=1)
    out = below.query(text)
    assert len(out) == 1 and out["sim"].iloc[0] == pytest.approx(1.0, abs=1e-5)


def test_topk_pass_top_k_and_duplicate_text():
    """top_k caps results per query even with more ties; duplicate pool text rows stay
    distinct candidates (no collapsing by text)."""
    q = pd.Series(["acme corp"])
    pool = pd.Series(["acme corp", "acme corp", "acme corp"])
    capped = blocking.TopK(TopKSpec("name_core", top_k=1, min_sim=0.5, min_df=1, max_df=1.0),
                           q, pool, vocab_sample=10, seed=1, n_threads=1)
    out = capped.query(q)
    assert len(out) == 1 and out["sim"].iloc[0] == pytest.approx(1.0, abs=1e-5)

    uncapped = blocking.TopK(TopKSpec("name_core", top_k=10, min_sim=0.5, min_df=1, max_df=1.0),
                             q, pool, vocab_sample=10, seed=1, n_threads=1)
    assert sorted(uncapped.query(q)["pool_idx"].tolist()) == [0, 1, 2]


# 7. P4 address char-gram: dedicated activation test (P3 reuses the generic topk tests)
def test_addr_char_pass_activates_bit_32():
    """Setting cfg.addr_char runs the same TopK machinery over addr_norm and tags bit 32."""
    s1n = normalise_records(_src(("S1-01", "Zzzz Foobar", "12 Main Street, Springfield, IL",
                                  "US")))
    pooln = normalise_records(_src(("S2-01", "Totally Different Name", "12 Main St, Springfield",
                                    "US")))
    spec = TopKSpec("addr_norm", top_k=5, min_sim=0.3, min_df=1, max_df=1.0)
    cfg = _cfg(exact_keys=(), name_char=None, name_addr_word=None, addr_char=spec)
    pairs = block(s1n, pooln, cfg)
    assert _pairset(pairs) == {("S1-01", "S2-01")}
    assert pairs["pass"].iloc[0] == PASS_BITS["addr_char"]
    assert pairs["sim_addr_char"].iloc[0] > 0.3


# 8. Union bitmask: single pass, all six bits, unknown pass name raises
def test_union_single_pass_all_bits_and_unknown_pass():
    """A lone pass keeps its own bit and NaN sims elsewhere; all six passes OR to 63; an
    unknown pass name is a programming error and raises rather than silently matching."""
    single = union_passes({"exact_squash": _idx([0], [5])}, 60)
    assert single["pass"].tolist() == [PASS_BITS["exact_squash"]]
    assert single[SIM_COLUMNS].isna().all().all()

    topk_names = ("name_char", "name_addr_word", "addr_char")
    all_parts = {name: _idx([0], [9], [0.9] if name in topk_names else None)
                for name in PASS_BITS}
    assert union_passes(all_parts, 60)["pass"].tolist() == [63]

    with pytest.raises(KeyError):
        union_passes({"bogus_pass": _idx([0], [1])}, 60)


# 9. Max similarity retained per pass: duplicate rows within one pass keep the max
def test_union_duplicate_rows_within_one_pass_keep_max_sim():
    """Two rows of the same pass for the same pair keep the larger similarity, regardless
    of which one appears first."""
    forward = union_passes({"name_char": _idx([0, 0], [1, 1], [0.9, 0.2])}, 60)
    backward = union_passes({"name_char": _idx([0, 0], [1, 1], [0.2, 0.9])}, 60)
    assert forward["sim_name_char"].iloc[0] == backward["sim_name_char"].iloc[0] == np.float32(0.9)


# 10. Exact-first selection under a tight cap
def test_union_cap_one_exact_beats_near_perfect_similarity():
    """With max_per_s1 = 1, an exact-key pair wins over a 0.99-similarity top-k pair."""
    parts = {"exact_core": _idx([0], [1]), "name_char": _idx([0], [2], [0.99])}
    assert union_passes(parts, max_per_s1=1)["pool_idx"].tolist() == [1]


# 11. max_per_s1 enforcement: 0, larger-than-available, independent per S1, property test
def test_union_cap_enforcement_zero_larger_and_independent_per_group():
    """max_per_s1 = 0 drops everything; a cap above the candidate count keeps them all;
    the cap is enforced independently per S1 group."""
    parts = {"name_char": _idx([0, 0, 1], [1, 2, 5], [0.5, 0.9, 0.1])}
    assert len(union_passes(parts, max_per_s1=0)) == 0
    assert len(union_passes(parts, max_per_s1=1000)) == 3
    out = union_passes(parts, max_per_s1=1)
    assert out.groupby("s1_idx").size().to_dict() == {0: 1, 1: 1}


def test_union_cap_enforcement_property():
    """Property check over random inputs: max_per_s1 is never exceeded, the kept count is
    exactly min(cap, candidates), and a group's exact pairs are never dropped in favour of
    a non-exact one as long as they fit under the cap."""
    rng = np.random.default_rng(7)
    for _ in range(30):
        n_s1, n_pool = 5, 15
        n_rows = int(rng.integers(3, 25))
        s1_idx = rng.integers(0, n_s1, n_rows).astype(np.int32)
        pool_idx = rng.integers(0, n_pool, n_rows).astype(np.int32)
        sim = rng.random(n_rows).astype(np.float32)
        is_exact = rng.random(n_rows) < 0.4
        exact_df = pd.DataFrame({"s1_idx": s1_idx[is_exact], "pool_idx": pool_idx[is_exact]}
                                ).drop_duplicates()
        topk_df = pd.DataFrame({"s1_idx": s1_idx[~is_exact], "pool_idx": pool_idx[~is_exact],
                                "sim": sim[~is_exact]})
        topk_df = topk_df.groupby(["s1_idx", "pool_idx"], as_index=False)["sim"].max()
        cap = int(rng.integers(0, 8))
        parts = {"exact_core": exact_df.astype({"s1_idx": np.int32, "pool_idx": np.int32}),
                "name_char": topk_df.astype({"s1_idx": np.int32, "pool_idx": np.int32})}
        out = union_passes(parts, cap)

        all_pairs = pd.concat([exact_df.assign(is_exact=True),
                              topk_df[["s1_idx", "pool_idx"]].assign(is_exact=False)],
                             ignore_index=True).drop_duplicates(["s1_idx", "pool_idx"])
        for s1 in range(n_s1):
            group = all_pairs[all_pairs["s1_idx"] == s1]
            kept = out[out["s1_idx"] == s1]
            assert len(kept) <= cap
            assert len(kept) == min(cap, len(group))
            n_exact_in_group = int(group["is_exact"].sum())
            n_exact_kept = int((kept["pass"].to_numpy() & EXACT_BITS != 0).sum())
            assert n_exact_kept == min(cap, n_exact_in_group)


# 12. Country partitioning: S1-only, pool-only, empty-string country, plus a property test
def test_block_country_partition_edge_cases():
    """A pool-only country is silently skipped; an S1-only country yields no candidates
    (and no error); an empty-string country label partitions like any other value."""
    s1n = normalise_records(_src(("S1-01", "Acme Corp", "1 Main St", "US"),
                                 ("S1-02", "Ghost Co", "Nowhere", "Atlantis"),
                                 ("S1-03", "Blank Co", "1 Void St", "")))
    pooln = normalise_records(_src(("S2-01", "Acme Corp", "1 Main St", "US"),
                                   ("S2-02", "Blank Co", "1 Void St", ""),
                                   ("S3-01", "Orphan Co", "9 Lost Ave", "Wakanda")))
    pairs = block(s1n, pooln, _cfg())
    assert _pairset(pairs) == {("S1-01", "S2-01"), ("S1-03", "S2-02")}
    assert "S1-02" not in set(pairs[C.S1_ID])


def test_block_four_country_partition_never_crosses():
    """Property check: across many random 4-country assignments (including ""), no
    returned pair ever joins records from different countries."""
    rng = np.random.default_rng(3)
    countries = ["US", "India", "France", ""]
    for _ in range(10):
        s1 = _src(*[(f"S1-{i}", NAMES[i % len(NAMES)], "", rng.choice(countries))
                    for i in range(6)])
        pool = _src(*[(f"S2-{i}", NAMES[i % len(NAMES)], "", rng.choice(countries))
                      for i in range(10)])
        s1n, pooln = normalise_records(s1), normalise_records(pool)
        pairs = block(s1n, pooln, _cfg())
        s1_country = dict(zip(s1n[C.ENTITY_ID], s1n[C.COUNTRY], strict=True))
        pool_country = dict(zip(pooln[C.ENTITY_ID], pooln[C.COUNTRY], strict=True))
        assert all(s1_country[a] == pool_country[b] for a, b in _pairset(pairs))


# 13. Unseen countries: fictional labels never present in any static config
def test_block_handles_completely_novel_country_labels():
    """Country is an open set: fictional labels get candidates like any known one."""
    s1n = normalise_records(_src(("S1-01", "Acme Corp", "1 Main St", "Mars"),
                                 ("S1-02", "Acme Corp", "1 Main St", "Neptune")))
    pooln = normalise_records(_src(("S2-01", "Acme Corp", "1 Main St", "Mars"),
                                   ("S3-01", "Acme Corp", "1 Main St", "Neptune")))
    pairs = block(s1n, pooln, _cfg())
    assert _pairset(pairs) == {("S1-01", "S2-01"), ("S1-02", "S3-01")}


# 14. Determinism: repeated TopK.query, repeated union tie-breaks
def test_topk_query_is_deterministic_across_repeated_calls():
    """Calling TopK.query twice on the same inputs yields byte-identical output."""
    q = pd.Series(["Kaveri Silk Emporium", "Meridian Freight Lines"])
    pool = pd.Series(NAMES)
    space = blocking.TopK(TopKSpec("name_core", top_k=5, min_sim=0.1, min_df=1, max_df=1.0),
                          q, pool, vocab_sample=100, seed=1, n_threads=1)
    pd.testing.assert_frame_equal(space.query(q), space.query(q))


def test_union_tie_break_is_repeatable():
    """The pool-position tie-break for equal similarities is stable across repeated calls."""
    parts = {"name_char": _idx([0, 0, 0], [5, 2, 8], [0.5, 0.5, 0.5])}
    results = [union_passes(parts, max_per_s1=2)["pool_idx"].tolist() for _ in range(5)]
    assert all(r == [2, 5] for r in results)


# 15. Cache behaviour: disabled, config/tag isolation, corrupt file
def test_block_cache_disabled_writes_nothing(frames, tmp_path):
    """Without cache_dir, block() never touches the filesystem and matches a cached run."""
    s1n, pooln = frames
    uncached = block(s1n, pooln, _cfg())
    assert list(tmp_path.iterdir()) == []
    pd.testing.assert_frame_equal(uncached, block(s1n, pooln, _cfg(), cache_dir=tmp_path / "c"))


def test_block_cache_key_isolation_across_config_and_tag(frames, tmp_path):
    """Different configs and different tags each get their own cache entry; rereading one
    never returns another's content."""
    s1n, pooln = frames
    cfg1, cfg2 = _cfg(), _cfg(exact_keys=())
    assert cfg1.key() != cfg2.key()
    out1_val = block(s1n, pooln, cfg1, cache_dir=tmp_path, tag="val")
    out2_val = block(s1n, pooln, cfg2, cache_dir=tmp_path, tag="val")
    out1_test = block(s1n, pooln, cfg1, cache_dir=tmp_path, tag="test")
    files = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.parquet")}
    assert any(f.startswith("val/") and cfg1.key() in f for f in files)
    assert any(f.startswith("val/") and cfg2.key() in f for f in files)
    assert any(f.startswith("test/") and cfg1.key() in f for f in files)
    pd.testing.assert_frame_equal(block(s1n, pooln, cfg1, cache_dir=tmp_path, tag="val"), out1_val)
    pd.testing.assert_frame_equal(block(s1n, pooln, cfg2, cache_dir=tmp_path, tag="val"), out2_val)
    reread_test = block(s1n, pooln, cfg1, cache_dir=tmp_path, tag="test")
    pd.testing.assert_frame_equal(reread_test, out1_test)


def test_block_cache_corrupt_file_raises(frames, tmp_path):
    """A corrupted cache file surfaces an error rather than silently returning bad data."""
    s1n, pooln = frames
    cfg = _cfg()
    block(s1n, pooln, cfg, cache_dir=tmp_path, tag="val")
    any_parquet = next(tmp_path.rglob("*.parquet"))
    any_parquet.write_bytes(b"not a parquet file")
    with pytest.raises(Exception):  # noqa: B017 - any read failure must surface, not be swallowed
        block(s1n, pooln, cfg, cache_dir=tmp_path, tag="val")


# 16. to_id_lists: duplicates preserved, unicode ids, fully empty pairs
def test_to_id_lists_duplicates_unicode_and_fully_empty_pairs():
    """Duplicate rows in the input are not silently deduped; unicode ids work; an empty
    pairs frame still returns [] for every S1 id."""
    dup_pairs = pd.DataFrame({C.S1_ID: ["S1-1", "S1-1"], C.ENTITY_ID: ["S2-9", "S2-9"]}
                            ).astype("str")
    assert to_id_lists(dup_pairs, pd.Series(["S1-1"], dtype="str"))["S1-1"] == ["S2-9", "S2-9"]

    uni_pairs = pd.DataFrame({C.S1_ID: ["S1-★"], C.ENTITY_ID: ["S2-日本語"]}
                            ).astype("str")
    uni_lists = to_id_lists(uni_pairs, pd.Series(["S1-★", "S1-☆"], dtype="str"))
    assert uni_lists == {"S1-★": ["S2-日本語"], "S1-☆": []}

    empty_pairs = pd.DataFrame({C.S1_ID: pd.Series([], dtype="str"),
                                C.ENTITY_ID: pd.Series([], dtype="str")})
    assert to_id_lists(empty_pairs, pd.Series(["S1-1", "S1-2"], dtype="str")) == {
        "S1-1": [], "S1-2": []}


# 17. Chunking: s1_chunk and pool_chunk both active at once
def test_chunking_s1_and_pool_together(frames):
    """s1_chunk and pool_chunk can both be small simultaneously and still match the default."""
    s1n, pooln = frames
    baseline = block(s1n, pooln, _cfg())
    both_chunked = block(s1n, pooln, _cfg(s1_chunk=2, pool_chunk=2))
    pd.testing.assert_frame_equal(baseline, both_chunked)
