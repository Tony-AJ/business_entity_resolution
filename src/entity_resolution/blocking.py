"""Candidate generation: a union of cheap exact passes and TF-IDF top-k retrieval.

Runs entirely inside one country partition at a time (Source 1 never candidates against
a pool record of another country). No single key reaches the recall target on its own
(01 SS3): 15% of true pairs share no name token, but ~90% of those share address tokens.
Hence a union of name channels (exact + char-gram) and an address channel (word-level,
name + address together). See 06_BLOCKING_STRATEGY.md for the full rationale, budget and
experiment plan.
"""
from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn, zip_sp_matmul_topn

from . import config as C
from .config import SEED

# ----------------------------------------------------------------------- schema ----
PAIR_COLUMNS = (C.S1_ID, C.ENTITY_ID, "pass", "sim_name_char", "sim_name_addr_word",
                "sim_addr_char")
SIM_COLUMNS = ("sim_name_char", "sim_name_addr_word", "sim_addr_char")

# bitmask assigned to each pass, per 02_SYSTEM_ARCHITECTURE.md SS4.2
PASS_BITS: dict[str, int] = {
    "name_core": 1, "name_sorted": 2, "name_squash": 4,
    "name_char": 8, "name_addr_word": 16, "addr_char": 32,
}
_EXACT_MASK = PASS_BITS["name_core"] | PASS_BITS["name_sorted"] | PASS_BITS["name_squash"]
_SIM_COLUMN_OF = {"name_char": "sim_name_char", "name_addr_word": "sim_name_addr_word",
                  "addr_char": "sim_addr_char"}

# rows sampled to fit each pass's TF-IDF vocabulary (unsupervised, refit per partition)
VOCAB_SAMPLE_ROWS = 500_000


# ------------------------------------------------------------------------ config ----
@dataclass(frozen=True)
class TopKSpec:
    """One TF-IDF top-k retrieval channel: text column, vectoriser and cutoffs."""

    column: str
    analyzer: str = "char_wb"
    ngram: tuple[int, int] = (3, 3)
    top_k: int = 30
    min_sim: float = 0.30
    max_df: float = 0.2
    min_df: int = 2
    sublinear_tf: bool = True


@dataclass(frozen=True)
class BlockingConfig:
    """Exact keys plus up to three top-k channels, and the union/chunking knobs."""

    exact_keys: tuple[str, ...] = ("name_core", "name_sorted", "name_squash")
    exact_max_group: int = 50
    name_char: TopKSpec | None = TopKSpec("name_core")
    name_addr_word: TopKSpec | None = TopKSpec("name_addr", "word", (1, 1), 20, 0.20, 0.05)
    addr_char: TopKSpec | None = None
    max_per_s1: int = 60
    s1_chunk: int = 50_000
    pool_chunk: int = 2_000_000
    n_threads: int = 12


# ------------------------------------------------------------------------ passes ----
def exact_pass(s1n: pd.DataFrame, pooln: pd.DataFrame, key: str, max_group: int) -> pd.DataFrame:
    """Every ``(s1_idx, pool_idx)`` pair sharing ``key`` verbatim.

    Pool groups bigger than ``max_group`` are skipped here (the top-k passes still see
    those rows); an empty key never matches, on either side, so missing values never
    create a group.
    """
    s1_key = s1n[key].to_numpy()
    pool_key = pooln[key].to_numpy()
    oversized = set(pd.Series(pool_key).value_counts().loc[lambda c: c > max_group].index)

    s1_keep = s1_key != ""
    pool_keep = (pool_key != "") & ~pd.Series(pool_key).isin(oversized).to_numpy()

    left = pd.DataFrame({"s1_idx": np.arange(len(s1n), dtype=np.int32)[s1_keep],
                         "_key": s1_key[s1_keep]})
    right = pd.DataFrame({"pool_idx": np.arange(len(pooln), dtype=np.int32)[pool_keep],
                          "_key": pool_key[pool_keep]})
    merged = left.merge(right, on="_key", how="inner")
    return merged[["s1_idx", "pool_idx"]].astype("int32").reset_index(drop=True)


def _empty_topk() -> pd.DataFrame:
    return pd.DataFrame({"s1_idx": pd.Series([], dtype="int32"),
                         "pool_idx": pd.Series([], dtype="int32"),
                         "sim": pd.Series([], dtype="float32")})


def topk_pass(s1n: pd.DataFrame, pooln: pd.DataFrame, spec: TopKSpec, s1_chunk: int,
             pool_chunk: int, n_threads: int, seed: int) -> pd.DataFrame:
    """Top-``spec.top_k`` cosine matches per S1 row, via chunked sparse TF-IDF retrieval.

    The vectoriser vocabulary is fit once on a sample of S1 union pool text of this
    partition (unsupervised, no leakage), then both sides are transformed and multiplied
    in chunks so memory stays bounded (02_SYSTEM_ARCHITECTURE.md SS7).
    """
    if len(s1n) == 0 or len(pooln) == 0:
        return _empty_topk()
    s1_text, pool_text = s1n[spec.column], pooln[spec.column]
    corpus = pd.concat([s1_text, pool_text], ignore_index=True)
    if not (corpus != "").any():
        return _empty_topk()
    sample = (corpus if len(corpus) <= VOCAB_SAMPLE_ROWS
             else corpus.sample(n=VOCAB_SAMPLE_ROWS, random_state=seed))

    vec = TfidfVectorizer(analyzer=spec.analyzer, ngram_range=spec.ngram, min_df=spec.min_df,
                          max_df=spec.max_df, sublinear_tf=spec.sublinear_tf, dtype=np.float32)
    vec.fit(sample)

    def transform(texts: pd.Series) -> sp.csr_matrix:
        parts = [vec.transform(texts.iloc[i:i + VOCAB_SAMPLE_ROWS])
                for i in range(0, len(texts), VOCAB_SAMPLE_ROWS)]
        return parts[0] if len(parts) == 1 else sp.vstack(parts, format="csr")

    pool_chunks_t = []
    for p0 in range(0, len(pooln), pool_chunk):
        b = transform(pool_text.iloc[p0:p0 + pool_chunk])
        pool_chunks_t.append(b.T.tocsr())
        del b
        gc.collect()

    frames = []
    for a0 in range(0, len(s1n), s1_chunk):
        a = transform(s1_text.iloc[a0:a0 + s1_chunk])
        results = [sp_matmul_topn(a, bt, top_n=spec.top_k, threshold=spec.min_sim, sort=True,
                                  n_threads=n_threads)
                  for bt in pool_chunks_t]
        c = results[0] if len(results) == 1 else zip_sp_matmul_topn(top_n=spec.top_k,
                                                                     C_mats=results)
        coo = c.tocoo()
        frames.append(pd.DataFrame({"s1_idx": (coo.row + a0).astype(np.int32),
                                    "pool_idx": coo.col.astype(np.int32),
                                    "sim": coo.data.astype(np.float32)}))
    return pd.concat(frames, ignore_index=True) if frames else _empty_topk()


# ------------------------------------------------------------------------ union ----
def _empty_union() -> pd.DataFrame:
    cols = {"s1_idx": "int32", "pool_idx": "int32", "pass": "uint8"}
    out = {name: pd.Series([], dtype=dtype) for name, dtype in cols.items()}
    out.update({col: pd.Series([], dtype="float32") for col in SIM_COLUMNS})
    return pd.DataFrame(out)


def union_passes(parts: dict[str, pd.DataFrame], max_per_s1: int) -> pd.DataFrame:
    """OR the bitmask and keep the best similarity per pass across every (s1, pool) pair.

    Cap at ``max_per_s1`` candidates per S1 row: exact-pass pairs are kept first, the
    rest ordered by the best similarity among the three channels, ties broken by
    ``pool_idx`` so the result is deterministic.
    """
    frames = []
    for name, df in parts.items():
        if name not in PASS_BITS:
            raise ValueError(f"unknown pass {name!r}, not in PASS_BITS")
        if df.empty:
            continue
        sim_col = _SIM_COLUMN_OF.get(name)
        if sim_col is not None:
            part = df.groupby(["s1_idx", "pool_idx"], as_index=False)["sim"].max()
            part = part.rename(columns={"sim": sim_col})
        else:
            part = df[["s1_idx", "pool_idx"]].drop_duplicates()
        part["bit"] = np.uint8(PASS_BITS[name])
        frames.append(part)
    if not frames:
        return _empty_union()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    for col in SIM_COLUMNS:
        if col not in combined.columns:
            combined[col] = np.nan

    agg = combined.groupby(["s1_idx", "pool_idx"], as_index=False).agg(
        **{"pass": ("bit", "sum"),
           "sim_name_char": ("sim_name_char", "max"),
           "sim_name_addr_word": ("sim_name_addr_word", "max"),
           "sim_addr_char": ("sim_addr_char", "max")})
    agg["pass"] = agg["pass"].astype("uint8")

    is_exact = (agg["pass"].to_numpy() & _EXACT_MASK) != 0
    best_sim = agg[list(SIM_COLUMNS)].max(axis=1, skipna=True).fillna(-1.0).to_numpy()
    s1_idx, pool_idx = agg["s1_idx"].to_numpy(), agg["pool_idx"].to_numpy()
    # lexsort's last key is primary: group by S1, exact first, best sim desc, pool_idx tie-break
    order = np.lexsort((pool_idx, -best_sim, ~is_exact, s1_idx))
    ordered = agg.iloc[order].reset_index(drop=True)
    rank = ordered.groupby("s1_idx").cumcount()
    return ordered[rank.to_numpy() < max_per_s1].reset_index(drop=True)


# ---------------------------------------------------------------------- assembly ----
def _empty_pairs() -> pd.DataFrame:
    out = {C.S1_ID: pd.Series([], dtype="str"), C.ENTITY_ID: pd.Series([], dtype="str"),
          "pass": pd.Series([], dtype="uint8")}
    out.update({col: pd.Series([], dtype="float32") for col in SIM_COLUMNS})
    return pd.DataFrame(out)[list(PAIR_COLUMNS)]


def block_partition(s1n: pd.DataFrame, pooln: pd.DataFrame, cfg: BlockingConfig) -> pd.DataFrame:
    """Union of every configured pass inside one, already country-filtered, partition."""
    parts: dict[str, pd.DataFrame] = {}
    for key in cfg.exact_keys:
        parts[key] = exact_pass(s1n, pooln, key, cfg.exact_max_group)
    for name, spec in (("name_char", cfg.name_char), ("name_addr_word", cfg.name_addr_word),
                       ("addr_char", cfg.addr_char)):
        if spec is not None:
            parts[name] = topk_pass(s1n, pooln, spec, cfg.s1_chunk, cfg.pool_chunk,
                                    cfg.n_threads, SEED)

    unioned = union_passes(parts, cfg.max_per_s1)
    if unioned.empty:
        return _empty_pairs()

    s1_ids, pool_ids = s1n[C.ENTITY_ID].to_numpy(), pooln[C.ENTITY_ID].to_numpy()
    out = pd.DataFrame({
        C.S1_ID: s1_ids[unioned["s1_idx"].to_numpy()],
        C.ENTITY_ID: pool_ids[unioned["pool_idx"].to_numpy()],
        "pass": unioned["pass"].to_numpy(),
        "sim_name_char": unioned["sim_name_char"].to_numpy(),
        "sim_name_addr_word": unioned["sim_name_addr_word"].to_numpy(),
        "sim_addr_char": unioned["sim_addr_char"].to_numpy(),
    })
    out = out[list(PAIR_COLUMNS)].astype(
        {C.S1_ID: "str", C.ENTITY_ID: "str", "pass": "uint8", "sim_name_char": "float32",
         "sim_name_addr_word": "float32", "sim_addr_char": "float32"})
    assert list(out.columns) == list(PAIR_COLUMNS)
    return out


def _config_hash(cfg: BlockingConfig) -> str:
    """Stable hash of a config, for cache filenames (Python's own hash() is randomised)."""
    return hashlib.sha256(repr(cfg).encode()).hexdigest()[:16]


def block(s1n: pd.DataFrame, pooln: pd.DataFrame, cfg: BlockingConfig,
         cache_dir: Path | None = None, tag: str = "") -> pd.DataFrame:
    """Candidate pairs for every country present in ``s1n``.

    No country is enumerated anywhere: whatever labels appear in the data are blocked,
    Source 2/3 records of a different country are never candidates, and an unseen
    country (test adds France) is handled exactly like any other. Deterministic: the
    result is sorted by ``source1_entity_id`` and every pair is unique.
    """
    cfg_hash = _config_hash(cfg)
    frames = []
    for country in sorted(s1n[C.COUNTRY].unique()):
        cache_path = (Path(cache_dir) / tag / country / f"pairs_{cfg_hash}.parquet"
                     if cache_dir is not None else None)
        if cache_path is not None and cache_path.exists():
            frames.append(pd.read_parquet(cache_path))
            continue
        s1c = s1n[s1n[C.COUNTRY] == country].reset_index(drop=True)
        poolc = pooln[pooln[C.COUNTRY] == country].reset_index(drop=True)
        part = block_partition(s1c, poolc, cfg)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            part.to_parquet(cache_path, index=False)
        frames.append(part)

    out = pd.concat(frames, ignore_index=True) if frames else _empty_pairs()
    out = out.sort_values(C.S1_ID, kind="stable").reset_index(drop=True)
    assert list(out.columns) == list(PAIR_COLUMNS)
    dup = out.duplicated([C.S1_ID, C.ENTITY_ID])
    assert not dup.any(), f"{int(dup.sum())} duplicate candidate pairs"
    return out


def to_id_lists(pairs: pd.DataFrame, s1_ids: pd.Series) -> dict[str, list[str]]:
    """Every S1 id mapped to its candidate ids, ``[]`` when it has none.

    Val-size frames only: this materialises one Python list per Source 1 entity.
    """
    grouped = pairs.groupby(C.S1_ID)[C.ENTITY_ID].apply(list)
    return {sid: (grouped[sid] if sid in grouped.index else []) for sid in s1_ids}
