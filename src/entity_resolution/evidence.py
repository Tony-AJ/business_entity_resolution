"""Learned token evidence (plan group C5): which one-sided name words mean a match or a decoy.

The generator makes decoys by adding a word to a reference name ("X Holdings", "X Midtown",
"Groupe X", "X Participations": other businesses), while the pool writes its own words into
true matches' names ("X Center", "X Services"). Both read "the names agree but for one word"
to every similarity feature. ``fit_token_evidence`` learns from the train fold how each word
moves the odds of a match when only the pool name holds it, and when only the Source 1 name
holds it; the ``tok_evidence`` feature group (``features.py``) sums it over a pair's
one-sided words.

Learning pairs are near-duplicates of one country whose word sets (``name_sorted``) differ by
exactly one word, anchored on Source 1 names no other S1 record of the country shares (so the
label reflects the word, not a shared name):

    pool words = S1 words + {w}    w only in the pool name   ("acme holdings" vs S1 "acme")
    S1 words = pool words + {w}    w only in the S1 name     (S1 "acme holdings" vs "acme")

True pairs are matches; every other such pair is a decoy. Per side and word,

    LLR(w) = log((n1 + k p0) / (n0 + k (1 - p0))) - log(p0 / (1 - p0))

with n1 / n0 the matches / decoys whose one-word difference is w, p0 the side's match share
and k (``prior``) pseudo-pairs pulling rare words to 0; words under ``min_support`` pairs get
no value. Words never seen (French ones: train has no France) are unknown and counted apart.
Sets are compared through additive 64-bit hashes (the sum of the word hashes), so "S1 words =
pool words minus w" is one integer subtraction per word; a false match needs a 64-bit
collision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from . import config as C
from .data import isin
from .split import hash_unit

EVIDENCE_SEED = 7373  # hash seed of the anchoring S1 sample


@dataclass(frozen=True)
class EvidenceConfig:
    """Switches of the learned token evidence (``PipelineConfig.evidence``); off by default."""

    learn: bool = False            # fit learns the table (the tok_evidence group reads it)
    sample_share: float = 0.5      # hashed share of the fold's S1 entities anchoring pairs
    min_support: int = 30          # pairs (matches + decoys) a word needs to get a value
    prior: float = 20.0            # k: pseudo-pairs pulling a word's log-odds toward 0
    strong: float = 1.0            # |LLR| from which a word counts as a strong filler / decoy
    max_pool_group: int = 20       # S1-only pairs: pool names held by more records skipped


DEFAULT = EvidenceConfig()


@dataclass
class TokenEvidence:
    """Learned log-odds per word: ``pool`` when only the pool name holds it, ``s1`` when only
    the Source 1 name does; ``strong`` is the threshold of the strong-word counts."""

    pool: dict[str, float]
    s1: dict[str, float]
    strong: float = 1.0
    info: dict = field(default_factory=dict)   # pairs and matches per side, for the logs

    def record(self) -> dict:
        """JSON-ready dict (``Fitted.save``); ``from_record`` reads it back."""
        return {"pool": self.pool, "s1": self.s1, "strong": self.strong, "info": self.info}

    @classmethod
    def from_record(cls, d: dict) -> TokenEvidence:
        """The table ``record`` wrote."""
        return cls(dict(d["pool"]), dict(d["s1"]), float(d["strong"]), dict(d.get("info", {})))

    @cached_property
    def _arrays(self) -> dict[str, tuple[pa.Array, np.ndarray]]:
        """Per side: the words as an Arrow array and their log-odds (for vectorised lookups)."""
        return {side: (pa.array(list(table), pa.large_string()),
                       np.fromiter(table.values(), dtype=np.float64, count=len(table)))
                for side, table in (("pool", self.pool), ("s1", self.s1))}

    def values(self, side: str, words: pa.Array) -> np.ndarray:
        """Log-odds of each word on ``side`` ("pool" or "s1"); NaN for an unknown word."""
        keys, llr = self._arrays[side]
        pos = pc.index_in(words.cast(pa.large_string()), value_set=keys)
        pos = pc.fill_null(pos, -1).to_numpy()
        return np.append(llr, np.nan)[pos]  # -1 picks the appended NaN


# ------------------------------------------------------------------ learning ----
def _set_hashes(keys: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, pa.Array]:
    """Word-set hashes of space-separated word sets (``name_sorted``).

    Returns ``(set_hash, parent, word_hash, words)``: per record the wrap-around sum of its word
    hashes (0 for an empty name); per word occurrence its record and hash; and the words
    themselves (Arrow). Word hashes come from the words' text, so they agree across frames.
    """
    arr = pa.array(keys, from_pandas=True)
    arr = (arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr).cast(
        pa.large_string())
    lists = pc.utf8_split_whitespace(arr)
    flat = pc.list_flatten(lists)
    parent = pc.list_parent_indices(lists).to_numpy()
    keep = pc.not_equal(flat, "").to_numpy(zero_copy_only=False)  # "" splits to [""]
    flat, parent = flat.filter(pa.array(keep)), parent[keep]
    enc = pc.dictionary_encode(flat)
    distinct = np.asarray(enc.dictionary.to_pylist(), dtype=object)
    word_hash = pd.util.hash_array(distinct)[enc.indices.to_numpy()] if len(distinct) else \
        np.zeros(0, np.uint64)
    set_hash = np.zeros(len(arr), dtype=np.uint64)
    if len(parent):
        starts = np.flatnonzero(np.r_[True, parent[1:] != parent[:-1]])
        set_hash[parent[starts]] = np.add.reduceat(word_hash, starts)  # uint64 wraps around
    return set_hash, parent, word_hash, flat


def _one_word_pairs(s1: pd.DataFrame, pool: pd.DataFrame, anchor: np.ndarray,
                    max_pool_group: int) -> pd.DataFrame:
    """Near-duplicate pairs of one country: s1 / pool row, differing word, side.

    ``anchor`` marks the S1 rows whose names are unique in the country and sampled.
    """
    s_hash, s_parent, s_word_hash, s_words = _set_hashes(s1["name_sorted"])
    p_hash, p_parent, p_word_hash, p_words = _set_hashes(pool["name_sorted"])
    s_len = np.bincount(s_parent, minlength=len(s1))
    p_len = np.bincount(p_parent, minlength=len(pool))
    out = []
    # pool = S1 + {w}: drop each word of a pool name (2+ words) and look for an anchor name
    order = np.argsort(s_hash[anchor], kind="stable")
    a_rows = np.flatnonzero(anchor)[order]
    a_hash = s_hash[a_rows]
    ok = p_len[p_parent] >= 2
    variant = p_hash[p_parent[ok]] - p_word_hash[ok]
    pos = np.minimum(np.searchsorted(a_hash, variant), max(len(a_hash) - 1, 0))
    hit = (a_hash[pos] == variant) if len(a_hash) else np.zeros(len(variant), bool)
    occ = np.flatnonzero(ok)[hit]
    out.append(pd.DataFrame({"s1": a_rows[pos[hit]], "pool": p_parent[occ],
                             "word": p_words.take(pa.array(occ)).to_numpy(zero_copy_only=False),
                             "side": "pool"}))
    # S1 = pool + {w}: drop each word of an anchor name (2+ words), look for pool names
    order = np.argsort(p_hash, kind="stable")
    sorted_hash = p_hash[order]
    ok = anchor[s_parent] & (s_len[s_parent] >= 2)
    variant = s_hash[s_parent[ok]] - s_word_hash[ok]
    lo = np.searchsorted(sorted_hash, variant, "left")
    hi = np.searchsorted(sorted_hash, variant, "right")
    size = hi - lo
    take = (size >= 1) & (size <= max_pool_group)   # a crowd of equal pool names says nothing
    occ = np.flatnonzero(ok)[take]
    n = size[take]
    first = np.repeat(lo[take], n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    out.append(pd.DataFrame({"s1": np.repeat(s_parent[occ], n), "pool": order[first],
                             "word": np.repeat(s_words.take(pa.array(occ)).to_numpy(
                                 zero_copy_only=False), n),
                             "side": "s1"}))
    return pd.concat(out, ignore_index=True)


def near_duplicates(s1n: pd.DataFrame, pooln: pd.DataFrame, truth_pairs: pd.DataFrame,
                    cfg: EvidenceConfig = DEFAULT, seed: int = EVIDENCE_SEED) -> pd.DataFrame:
    """The labelled one-word near-duplicate pairs of a fold (module doc): word, side, match.

    ``s1n`` holds EVERY S1 record of the fold's countries (a name is unique only among all of
    them; the hashed ``sample_share`` anchor pairs) and ``pooln`` their pool, both with
    ``entity_id``, ``country`` and ``name_sorted``; ``truth_pairs`` are the fold's true pairs.
    One country at a time is enough (the pipeline loads them one by one to bound memory).
    """
    parts = []
    for country in sorted(set(s1n[C.COUNTRY]) & set(pooln[C.COUNTRY])):
        s1 = s1n[(s1n[C.COUNTRY] == country).to_numpy()].reset_index(drop=True)
        pool = pooln[(pooln[C.COUNTRY] == country).to_numpy()].reset_index(drop=True)
        names = s1["name_sorted"]
        unique = ~names.duplicated(keep=False).to_numpy() & (names != "").to_numpy()
        anchor = unique & (hash_unit(s1[C.ENTITY_ID], seed) < cfg.sample_share)
        pairs = _one_word_pairs(s1, pool, anchor, cfg.max_pool_group)
        parts.append(pd.DataFrame({
            C.S1_ID: s1[C.ENTITY_ID].to_numpy()[pairs["s1"].to_numpy()],
            C.ENTITY_ID: pool[C.ENTITY_ID].to_numpy()[pairs["pool"].to_numpy()],
            "word": pairs["word"].to_numpy(), "side": pairs["side"].to_numpy()}))
    pairs = (pd.concat(parts, ignore_index=True) if parts else
             pd.DataFrame(columns=[C.S1_ID, C.ENTITY_ID, "word", "side"]))
    truth = truth_pairs[[C.S1_ID, C.ENTITY_ID]]
    truth = truth[isin(truth[C.S1_ID], pd.Index(pairs[C.S1_ID].unique()))].assign(match=1)
    pairs = pairs.astype({C.S1_ID: "str", C.ENTITY_ID: "str"}).merge(
        truth.astype({C.S1_ID: "str", C.ENTITY_ID: "str"}), on=[C.S1_ID, C.ENTITY_ID],
        how="left")
    return pd.DataFrame({"word": pairs["word"].astype("str"),
                         "side": pairs["side"].astype("category"),
                         "match": pairs["match"].fillna(0).to_numpy(np.int8)})


def evidence_table(pairs: pd.DataFrame, cfg: EvidenceConfig = DEFAULT) -> TokenEvidence:
    """``TokenEvidence`` from labelled near-duplicates (``near_duplicates``, any countries)."""
    tables, info = {}, {}
    for side in ("pool", "s1"):
        tables[side], info[side] = _log_odds(pairs[(pairs["side"] == side).to_numpy()], cfg)
    return TokenEvidence(tables["pool"], tables["s1"], cfg.strong,
                         {**info, "config": asdict(cfg)})


def fit_token_evidence(s1n: pd.DataFrame, pooln: pd.DataFrame, truth_pairs: pd.DataFrame,
                       cfg: EvidenceConfig = DEFAULT,
                       seed: int = EVIDENCE_SEED) -> TokenEvidence:
    """Per-word log-odds of a match from the one-word near-duplicates of a fold (module doc):
    ``evidence_table(near_duplicates(...))``. Fit on the train fold only."""
    return evidence_table(near_duplicates(s1n, pooln, truth_pairs, cfg, seed), cfg)


def _log_odds(pairs: pd.DataFrame, cfg: EvidenceConfig) -> tuple[dict[str, float], dict]:
    """Shrunk log-odds per word of one side, words with at least ``min_support`` pairs."""
    match = pairs["match"].to_numpy(np.int64)  # int8 in the frame: sum in int64
    n_pairs, n_match = len(pairs), int(match.sum())
    info = {"pairs": n_pairs, "matches": n_match}
    if n_match == 0 or n_match == n_pairs:  # no contrast to learn from
        return {}, info
    p0 = n_match / n_pairs
    counts = pd.DataFrame({"word": pairs["word"].to_numpy(), "match": match}).groupby(
        "word", sort=True)["match"].agg(["sum", "size"])
    n1 = counts["sum"].to_numpy(np.float64)
    n0 = counts["size"].to_numpy(np.float64) - n1
    llr = (np.log((n1 + cfg.prior * p0) / (n0 + cfg.prior * (1 - p0)))
           - np.log(p0 / (1 - p0)))
    keep = (n1 + n0) >= cfg.min_support
    info["words"] = int(keep.sum())
    return dict(zip(counts.index[keep], np.round(llr[keep], 4).tolist(), strict=True)), info
