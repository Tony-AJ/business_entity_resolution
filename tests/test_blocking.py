import numpy as np
import pandas as pd
import pytest

from entity_resolution import blocking
from entity_resolution import config as C
from entity_resolution.blocking import (
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


def test_union_cap_order_sim_first_drops_unscored_exact_pairs() -> None:
    """exact_first keeps unscored exact pairs over scored ones; sim_first does the reverse."""
    import pandas as pd

    from entity_resolution.blocking import union_passes
    exact = pd.DataFrame({"s1_idx": [0, 0, 0], "pool_idx": [1, 2, 3]})
    word = pd.DataFrame({"s1_idx": [0, 0], "pool_idx": [4, 5], "sim": [0.9, 0.5]})
    a = union_passes({"exact_core": exact, "name_addr_word": word}, 3)
    assert sorted(a["pool_idx"].tolist()) == [1, 2, 3]
    b = union_passes({"exact_core": exact, "name_addr_word": word}, 3, "sim_first")
    assert sorted(b["pool_idx"].tolist()) == [1, 4, 5]          # scored first, then pool order
    try:
        union_passes({"exact_core": exact}, 3, "random")
    except ValueError as e:
        assert "cap_order" in str(e)
    else:
        raise AssertionError("an unknown cap_order must be refused")


def test_default_blocking_key_is_stable() -> None:
    """New fields at their default must not change the key of cached candidate sets."""
    from dataclasses import replace

    from entity_resolution.blocking import BlockingConfig
    assert BlockingConfig().key() == "66540dae"
    assert replace(BlockingConfig(), cap_order="sim_first").key() != "66540dae"


def test_name_num_pass_joins_name_and_a_shared_number() -> None:
    """Same core name AND a shared address number; groups over the limit are skipped."""
    import pandas as pd

    from entity_resolution.blocking import name_num_pass
    s1n = pd.DataFrame({"name_core": ["acme", "acme", "globex", ""],
                        "addr_nums": ["12 5", "99", "12", "12"]}).astype("str")
    pooln = pd.DataFrame({"name_core": ["acme", "acme", "globex", "acme", "acme"],
                          "addr_nums": ["5", "13", "12", "", "99 12"]}).astype("str")
    out = name_num_pass(s1n, pooln, max_group=5)
    assert list(zip(out["s1_idx"], out["pool_idx"], strict=True)) == [
        (0, 0), (0, 4), (1, 4), (2, 2)]
    capped = name_num_pass(s1n, pd.concat([pooln] * 3, ignore_index=True), max_group=2)
    assert len(capped) == 0                         # every key now has >= 3 pool records


def test_name_num_pairs_survive_the_sim_first_cap() -> None:
    """Under sim_first, name+number pairs rank before any cosine-scored pair."""
    import pandas as pd

    from entity_resolution.blocking import union_passes
    word = pd.DataFrame({"s1_idx": [0, 0], "pool_idx": [1, 2], "sim": [0.9, 0.8]})
    nn = pd.DataFrame({"s1_idx": [0], "pool_idx": [3]})
    out = union_passes({"name_addr_word": word, "exact_name_num": nn}, 2, "sim_first")
    assert sorted(out["pool_idx"].tolist()) == [1, 3]
