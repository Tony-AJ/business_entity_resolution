"""Candidate generation (plan group A, 06_BLOCKING_STRATEGY).

``block(s1n, pooln, cfg)`` returns every (Source 1, Source 2/3) pair worth scoring: the
union of cheap exact-key passes and TF-IDF top-k retrieval, run inside each country
partition (true pairs always share the country; the partition is whatever country values
exist, so France needs nothing special). Blocking sets the recall ceiling and its output
is what ``candidate_pairs.tsv`` reports, so it is deterministic.

Passes (bit in ``pass``):

    1  exact_core       equal ``name_core``        pool key groups > exact_max_group skipped
    2  exact_sorted     equal ``name_sorted``      (word order swaps, doubled words)
    4  exact_squash     equal ``name_squash``      (domain / handle / leet forms)
    8  name_char        name char n-gram TF-IDF top-k       (typos, transliterations)
    16 name_addr_word   name + address word TF-IDF top-k    (renames, script names,
                                                             same-name decoys ranked by
                                                             their address)
    32 addr_char        address char n-gram TF-IDF top-k    (optional)
    64 exact_name_num   equal ``name_core`` AND a shared address number   (optional: small
                        groups even for common names; ranked first by the sim-first cap)
    128 exact_nofill    equal words of ``name_core_nofill``, name_core without the learned
                        filler tokens   (optional: "acme center" meets "acme"; its own group cap)

Inside a partition everything is positional int32 until the end; S1 records are processed
in chunks so the per-chunk union and the ``max_per_s1`` cap bound the memory.
"""
from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from . import config as C
from .normalize import NOFILL, sorted_words

PASS_BITS = {"exact_core": 1, "exact_sorted": 2, "exact_squash": 4, "name_char": 8,
             "name_addr_word": 16, "addr_char": 32, "exact_name_num": 64, "exact_nofill": 128}
EXACT_BITS = PASS_BITS["exact_core"] | PASS_BITS["exact_sorted"] | PASS_BITS["exact_squash"]
SIM_COLUMNS = ["sim_name_char", "sim_name_addr_word", "sim_addr_char"]
PAIR_COLUMNS = [C.S1_ID, C.ENTITY_ID, "pass", *SIM_COLUMNS]
_EXACT_PASS = {"name_core": "exact_core", "name_sorted": "exact_sorted",
               "name_squash": "exact_squash"}
_TOPK_PASS = (("name_char", "sim_name_char"), ("name_addr_word", "sim_name_addr_word"),
              ("addr_char", "sim_addr_char"))


@dataclass(frozen=True)
class TopKSpec:
    """One TF-IDF top-k retrieval pass: which text, how it is tokenised, how many kept."""

    column: str
    analyzer: str = "char_wb"
    ngram: tuple[int, int] = (3, 3)
    top_k: int = 30
    min_sim: float = 0.30
    max_df: float = 0.2
    min_df: int = 2
    sublinear_tf: bool = True
    # search only pool records whose address has at most this many tokens (None = all):
    # the name char pass is for records the address cannot link (empty or city-only)
    pool_max_addr_tokens: int | None = None
    # cap on a term's document count in the partition, on top of max_df: query cost grows
    # with the postings of kept terms, so a relative cap alone makes the 5x larger test
    # partitions 5x slower per query; an absolute cap keeps the cost per S1 constant
    max_df_abs: int | None = None


@dataclass(frozen=True)
class BlockingConfig:
    """Passes and budgets of ``block`` (06 §2); the defaults are the V1 configuration."""

    exact_keys: tuple[str, ...] = ("name_core", "name_sorted", "name_squash")
    exact_max_group: int = 50
    # char 3-grams cost ~3 ms per S1 against a 0.8M pool when run on every record (hours on
    # test), so the name pass only searches pool records with a short or empty address
    name_char: TopKSpec | None = TopKSpec("name_core", top_k=10, min_sim=0.5,
                                          pool_max_addr_tokens=3, max_df_abs=20_000)
    # the workhorse: name + address word uni- and bigrams (bigrams keep the address signal
    # that max_df removes from frequent unigrams: "rajendra nagar", "5 52"); val recall with
    # the exact passes 0.987 India / 0.994 US at ~30 candidates per S1
    name_addr_word: TopKSpec | None = TopKSpec("name_addr", "word", (1, 2), 25, 0.20, 0.01,
                                               max_df_abs=10_000)
    addr_char: TopKSpec | None = None
    max_per_s1: int = 60
    # which pairs the per-S1 cap keeps first: "exact_first" (exact-key pairs, then by best
    # cosine) or "sim_first" (by best cosine; exact-key pairs no top-k pass found come last,
    # so at test density a crowd of same-name records cannot push variant names out)
    cap_order: str = "exact_first"
    # exact pass on (name_core, one address number): pool key groups up to this size; None =
    # off. A common name whose name_core group is too large to join still meets its true
    # records through a shared house / plot number
    name_num_max_group: int | None = None
    # exact pass on the sorted words of name_core_nofill (name_core without the learned
    # filler tokens, normalize.fit_fillers): pool key groups up to this size; None = off.
    # Needs the column, which pipeline.load_normalised adds for a version that learns fillers.
    # Train fold, 26 fillers: +24k US / +40k India true pairs at 50 (0.3 / 0.5 extra
    # candidates per S1), +34k / +50k at 200 (1.0 / 1.6 per S1)
    nofill_max_group: int | None = None
    s1_chunk: int = 50_000
    n_threads: int = 12
    vocab_sample: int = 500_000
    seed: int = C.SEED

    def key(self) -> str:
        """Short stable hash of the configuration (cache file names, logs).

        Fields added after the first cached runs are left out while at their default, so the
        default configuration keeps its key and its cached candidate sets.
        """
        d = asdict(self)
        if d.get("cap_order") == "exact_first":
            d.pop("cap_order")
        if d.get("name_num_max_group") is None:
            d.pop("name_num_max_group")
        if d.get("nofill_max_group") is None:
            d.pop("nofill_max_group")
        blob = json.dumps(d, sort_keys=True, default=str).encode()
        return hashlib.sha1(blob).hexdigest()[:8]


DEFAULT = BlockingConfig()


# ------------------------------------------------------------------ passes ----
def exact_pass(s1n: pd.DataFrame, pooln: pd.DataFrame, key: str, max_group: int) -> pd.DataFrame:
    """Pairs whose ``key`` column is equal and non-empty: ``s1_idx``, ``pool_idx`` (int32).

    Pool key groups larger than ``max_group`` are skipped (a name shared by hundreds of
    records says nothing; the top-k passes still see those records).
    """
    # keys become integer codes shared by both sides, so the join never carries strings (at
    # test density a partition holds tens of millions of exact pairs)
    codes, uniques = pd.factorize(pd.concat([s1n[key], pooln[key]], ignore_index=True),
                                  use_na_sentinel=False)
    s1_code, pool_code = codes[:len(s1n)], codes[len(s1n):]
    size = np.bincount(pool_code, minlength=len(uniques))
    ok = size <= max_group
    blank = np.flatnonzero(np.asarray(uniques) == "")
    ok[blank] = False
    pool = pd.DataFrame({"k": pool_code, "pool_idx": np.arange(len(pooln), dtype=np.int32)})
    pool = pool[ok[pool_code]]
    left = pd.DataFrame({"k": s1_code, "s1_idx": np.arange(len(s1n), dtype=np.int32)})
    left = left[ok[s1_code] & (size[s1_code] > 0)]
    out = left.merge(pool, on="k", how="inner")[["s1_idx", "pool_idx"]].astype(np.int32)
    return out.sort_values(["s1_idx", "pool_idx"], kind="stable").reset_index(drop=True)


NAME_NUM_MAX = 4   # address numbers per record used as name+number keys


def name_num_pass(s1n: pd.DataFrame, pooln: pd.DataFrame, max_group: int) -> pd.DataFrame:
    """Pairs sharing ``name_core`` and at least one of their first address numbers.

    Each record gets one key per number (``name_core|number``, first ``NAME_NUM_MAX`` of
    ``addr_nums``); keys held by more than ``max_group`` pool records are skipped. Output as
    ``exact_pass``: unique ``s1_idx``, ``pool_idx`` (int32), sorted.
    """
    def keys(df: pd.DataFrame) -> pd.DataFrame:
        nums = df["addr_nums"].str.split(" ", n=NAME_NUM_MAX, expand=True)
        parts = []
        for j in range(min(NAME_NUM_MAX, nums.shape[1])):
            n = nums[j].fillna("")
            ok = ((n != "") & (df["name_core"] != "")).to_numpy()
            parts.append(pd.DataFrame({"k": (df["name_core"] + "|" + n)[ok].to_numpy(),
                                       "i": np.flatnonzero(ok).astype(np.int32)}))
        if not parts:
            return pd.DataFrame({"k": pd.Series([], dtype="str"), "i": np.zeros(0, np.int32)})
        return pd.concat(parts, ignore_index=True).drop_duplicates()

    left, right = keys(s1n), keys(pooln)
    if len(left) == 0 or len(right) == 0:
        return pd.DataFrame({"s1_idx": np.zeros(0, np.int32), "pool_idx": np.zeros(0, np.int32)})
    codes, uniques = pd.factorize(pd.concat([left["k"], right["k"]], ignore_index=True),
                                  use_na_sentinel=False)
    lc, rc = codes[:len(left)], codes[len(left):]
    size = np.bincount(rc, minlength=len(uniques))
    pool = pd.DataFrame({"k": rc, "pool_idx": right["i"].to_numpy()})[size[rc] <= max_group]
    s1 = pd.DataFrame({"k": lc, "s1_idx": left["i"].to_numpy()})
    s1 = s1[(size[lc] > 0) & (size[lc] <= max_group)]
    out = s1.merge(pool, on="k", how="inner")[["s1_idx", "pool_idx"]].drop_duplicates()
    return out.astype(np.int32).sort_values(["s1_idx", "pool_idx"]).reset_index(drop=True)


def nofill_pass(s1n: pd.DataFrame, pooln: pd.DataFrame, max_group: int) -> pd.DataFrame:
    """Pairs whose filler-free core names hold the same words: ``exact_pass`` on the sorted
    distinct words of ``name_core_nofill`` (as ``name_sorted`` is of ``name_core``).

    "acme center" and "center acme services" both key "acme" when "center" and "services"
    are learned fillers. Output as ``exact_pass``; ValueError without the column.
    """
    for label, df in (("s1n", s1n), ("pooln", pooln)):
        if NOFILL not in df.columns:
            raise ValueError(f"{label} lacks {NOFILL!r}: learn fillers "
                             "(NormaliseConfig.learn_fillers) and load the records with them")
    key = "nofill_words"
    return exact_pass(pd.DataFrame({key: sorted_words(s1n[NOFILL])}),
                      pd.DataFrame({key: sorted_words(pooln[NOFILL])}), key, max_group)


class TopK:
    """A fitted TF-IDF space for one pass over one partition: pool matrix ready to query."""

    def __init__(self, spec: TopKSpec, s1_text: pd.Series, pool_text: pd.Series,
                 vocab_sample: int, seed: int, n_threads: int) -> None:
        """Fit the vocabulary on a sample of S1 ∪ pool and transform the whole pool once."""
        self.spec, self.n_threads = spec, n_threads
        both = pd.concat([s1_text, pool_text], ignore_index=True)
        sample = both.sample(min(vocab_sample, len(both)), random_state=seed)
        n = len(sample)
        max_df = spec.max_df
        if spec.max_df_abs is not None and len(both):
            max_df = min(max_df, spec.max_df_abs / len(both))
        # min_df is an absolute count; guard tiny partitions (tests) against max_df < min_df
        max_df = max_df if max_df * n >= spec.min_df else 1.0
        # single-character words count ("d and v", house numbers "5"); char analysers
        # ignore token_pattern, so it is only passed for words
        words = {"token_pattern": r"(?u)\b\w+\b"} if spec.analyzer == "word" else {}
        self.vec = TfidfVectorizer(analyzer=spec.analyzer, ngram_range=spec.ngram,
                                   min_df=min(spec.min_df, max(n, 1)), max_df=max_df,
                                   sublinear_tf=spec.sublinear_tf, dtype=np.float32,
                                   lowercase=False, **words)
        try:
            self.vec.fit(sample.to_numpy())
            self.ok = True
        except ValueError:  # empty vocabulary (tiny or empty partition)
            self.ok = False
            return
        self.pool_t = self._transform(pool_text).T.tocsr()  # V x n_pool, transposed once

    def _transform(self, text: pd.Series) -> sp.csr_matrix:
        """TF-IDF rows for ``text`` in 500k-row slices (bounded transient memory)."""
        parts = [self.vec.transform(text.iloc[i:i + 500_000].to_numpy())
                 for i in range(0, len(text), 500_000)]
        if not parts:
            return sp.csr_matrix((0, len(self.vec.vocabulary_)), dtype=np.float32)
        return sp.vstack(parts, format="csr").astype(np.float32)

    def query(self, s1_text: pd.Series) -> pd.DataFrame:
        """Top-k pool neighbours of each text above ``min_sim``: s1_idx (local), pool_idx, sim."""
        if not self.ok or len(s1_text) == 0 or self.pool_t.shape[1] == 0:
            return pd.DataFrame({"s1_idx": np.zeros(0, np.int32),
                                 "pool_idx": np.zeros(0, np.int32),
                                 "sim": np.zeros(0, np.float32)})
        a = self._transform(s1_text)
        res = sp_matmul_topn(a, self.pool_t, top_n=self.spec.top_k,
                             threshold=self.spec.min_sim, sort=True, n_threads=self.n_threads)
        coo = res.tocoo()
        return pd.DataFrame({"s1_idx": coo.row.astype(np.int32),
                             "pool_idx": coo.col.astype(np.int32),
                             "sim": np.minimum(coo.data, 1.0).astype(np.float32)})


def union_passes(parts: dict[str, pd.DataFrame], max_per_s1: int,
                 cap_order: str = "exact_first") -> pd.DataFrame:
    """One row per (s1_idx, pool_idx): OR of pass bits, max sim per top-k pass, capped.

    ``cap_order="exact_first"`` keeps exact pairs first, then the rest by their best
    similarity; ``"sim_first"`` ranks every pair by its best similarity and puts pairs no
    top-k pass scored last. Ties go to the lower pool position: deterministic.
    """
    if cap_order not in ("exact_first", "sim_first"):
        raise ValueError(f"cap_order must be 'exact_first' or 'sim_first', got {cap_order!r}")
    frames = []
    for name, df in parts.items():
        if len(df) == 0:
            continue
        f = pd.DataFrame({"s1_idx": df["s1_idx"].to_numpy(np.int32),
                          "pool_idx": df["pool_idx"].to_numpy(np.int32),
                          "pass": np.uint8(PASS_BITS[name])})
        for pname, col in _TOPK_PASS:
            f[col] = df["sim"].to_numpy(np.float32) if pname == name else np.float32(np.nan)
        frames.append(f)
    cols = ["s1_idx", "pool_idx", "pass", *SIM_COLUMNS]
    if not frames:
        return pd.DataFrame({c: np.zeros(0, np.int32 if "idx" in c else np.float32)
                             for c in cols}).astype({"pass": np.uint8})
    allp = pd.concat(frames, ignore_index=True)
    g = allp.groupby(["s1_idx", "pool_idx"], sort=False)
    out = g[SIM_COLUMNS].max()
    out["pass"] = g["pass"].agg(np.bitwise_or.reduce).astype(np.uint8)
    out = out.reset_index()
    best = out[SIM_COLUMNS].max(axis=1).fillna(0.0).to_numpy()
    exact_bits = EXACT_BITS | PASS_BITS["exact_name_num"] | PASS_BITS["exact_nofill"]
    exact = (out["pass"].to_numpy() & exact_bits) != 0
    if cap_order == "sim_first":            # by best sim desc; unscored exact pairs last
        best = np.where(np.isnan(out[SIM_COLUMNS].to_numpy()).all(axis=1), -1.0, best)
        # name + number pairs are rare and precise: they go first, whatever their cosine
        exact = (out["pass"].to_numpy() & PASS_BITS["exact_name_num"]) != 0
    # rank inside each S1: exact first (exact_first only), then by best sim desc, pool position
    order = np.lexsort((out["pool_idx"].to_numpy(), -best, ~exact, out["s1_idx"].to_numpy()))
    out = out.iloc[order]
    rank = out.groupby("s1_idx", sort=False).cumcount().to_numpy()
    out = out[rank < max_per_s1]
    out = out.sort_values(["s1_idx", "pool_idx"], kind="stable").reset_index(drop=True)
    out["s1_idx"] = out["s1_idx"].astype(np.int32)
    out["pool_idx"] = out["pool_idx"].astype(np.int32)
    return out[cols]


# --------------------------------------------------------------- partition ----
def block_partition(s1n: pd.DataFrame, pooln: pd.DataFrame,
                    cfg: BlockingConfig = DEFAULT) -> pd.DataFrame:
    """Candidate pairs (PAIR_COLUMNS) for one country partition.

    ``s1n`` and ``pooln`` are normalised frames of that partition. Output is sorted by
    ``source1_entity_id`` then pool id order; pairs are unique.
    """
    s1n = s1n.sort_values(C.ENTITY_ID, kind="stable").reset_index(drop=True)
    pooln = pooln.reset_index(drop=True)
    exact = {_EXACT_PASS[k]: exact_pass(s1n, pooln, k, cfg.exact_max_group)
             for k in cfg.exact_keys}
    if cfg.name_num_max_group is not None:
        exact["exact_name_num"] = name_num_pass(s1n, pooln, cfg.name_num_max_group)
    if cfg.nofill_max_group is not None:
        exact["exact_nofill"] = nofill_pass(s1n, pooln, cfg.nofill_max_group)
    spaces, pool_rows = {}, {}
    for name, _ in _TOPK_PASS:
        spec = getattr(cfg, name)
        if spec is None:
            continue
        rows = np.arange(len(pooln))
        if spec.pool_max_addr_tokens is not None:
            rows = np.flatnonzero(pooln["addr_tokens"].to_numpy() <= spec.pool_max_addr_tokens)
        pool_rows[name] = rows
        spaces[name] = TopK(spec, s1n[spec.column], pooln[spec.column].iloc[rows],
                            cfg.vocab_sample, cfg.seed, cfg.n_threads)
    out = []
    ex_s1 = {k: v["s1_idx"].to_numpy() for k, v in exact.items()}  # sorted by s1_idx
    for a0 in range(0, len(s1n), cfg.s1_chunk):
        a1 = min(a0 + cfg.s1_chunk, len(s1n))
        parts = {}
        for name, df in exact.items():
            lo, hi = np.searchsorted(ex_s1[name], [a0, a1])
            parts[name] = df.iloc[lo:hi]
        for name, space in spaces.items():
            q = space.query(s1n[space.spec.column].iloc[a0:a1])
            q["s1_idx"] += np.int32(a0)
            q["pool_idx"] = pool_rows[name][q["pool_idx"].to_numpy()].astype(np.int32)
            parts[name] = q
        out.append(union_passes(parts, cfg.max_per_s1, cfg.cap_order))
        del parts
        gc.collect()
    del spaces
    pairs = pd.concat(out, ignore_index=True) if out else union_passes({}, cfg.max_per_s1)
    res = pd.DataFrame({
        C.S1_ID: s1n[C.ENTITY_ID].to_numpy()[pairs["s1_idx"].to_numpy()],
        C.ENTITY_ID: pooln[C.ENTITY_ID].to_numpy()[pairs["pool_idx"].to_numpy()],
    }).astype("str")
    res["pass"] = pairs["pass"].to_numpy(np.uint8)
    for col in SIM_COLUMNS:
        res[col] = pairs[col].to_numpy(np.float32)
    return res


def block(s1n: pd.DataFrame, pooln: pd.DataFrame, cfg: BlockingConfig = DEFAULT,
          cache_dir: Path | None = None, tag: str = "") -> pd.DataFrame:
    """Candidate pairs for every country present in ``s1n`` (PAIR_COLUMNS, sorted by S1 id).

    With ``cache_dir`` each partition's pairs are stored as
    ``<cache_dir>/<tag>/<country>/pairs_<cfg.key()>.parquet`` and reused on the next call.
    """
    out = []
    for country in sorted(s1n[C.COUNTRY].unique()):
        path = None
        if cache_dir is not None:
            path = Path(cache_dir) / tag / _safe(country) / f"pairs_{cfg.key()}.parquet"
            if path.exists():
                out.append(pd.read_parquet(path))
                continue
        part = block_partition(s1n[s1n[C.COUNTRY] == country],
                               pooln[pooln[C.COUNTRY] == country], cfg)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            part.to_parquet(tmp, index=False)
            tmp.replace(path)
        out.append(part)
        gc.collect()
    if not out:
        return _empty_pairs()
    pairs = pd.concat(out, ignore_index=True)
    pairs = pairs.sort_values([C.S1_ID], kind="stable").reset_index(drop=True)
    return pairs[PAIR_COLUMNS]


def _safe(name: str) -> str:
    """A country label as a folder name."""
    return "".join(ch if ch.isalnum() else "_" for ch in name) or "_"


def _empty_pairs() -> pd.DataFrame:
    """A zero-row frame with the PAIR_COLUMNS schema."""
    df = pd.DataFrame({C.S1_ID: pd.Series([], dtype="str"),
                       C.ENTITY_ID: pd.Series([], dtype="str"),
                       "pass": pd.Series([], dtype=np.uint8)})
    for col in SIM_COLUMNS:
        df[col] = pd.Series([], dtype=np.float32)
    return df


def to_id_lists(pairs: pd.DataFrame, s1_ids: pd.Series) -> dict[str, list[str]]:
    """Source 1 id -> candidate ids, every id of ``s1_ids`` present (val-size frames only)."""
    grouped = pairs.groupby(C.S1_ID, sort=False)[C.ENTITY_ID].agg(list)
    return {s: grouped.get(s, []) for s in s1_ids}
