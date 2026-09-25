"""Pair features for the matcher (plan group C, 07_FEATURE_ENGINEERING).

``build_features(pairs, s1n, pooln)`` turns candidate pairs (``blocking.PAIR_COLUMNS``) and
the normalised records of both sides (``normalize.NORM_COLUMNS``) into the float32 frame the
model trains and predicts on: index ``pairs.index``, columns ``feature_names(groups)``.

Features come in groups (``REGISTRY``, 07 §2). A group is a pure function
``(pairs_chunk, left, right) -> DataFrame``; ``left`` and ``right`` hold the normalised S1 and
pool record of each pair, row for row. Conventions (07 §1):

* similarities are symmetric and lie in [0, 1]; they are NaN where undefined (a side is
  empty), which LightGBM handles natively. ``NAN_FEATURES`` lists every column that may be NaN;
* flags are 0/1 and never NaN; counts and ranks are whole numbers; every column is float32;
* no country feature (the country set is open) and no randomness: two runs are identical;
* chunks never split a Source 1 group, so no feature depends on ``chunk_rows``.

Speed is the main constraint (~25M test pairs): no Python loop touches individual pairs.
Strings stay in Arrow and go through its C++ kernels; rapidfuzz scores aligned pairs in C++
threads (``process.cpdist``); token-set overlaps are row-wise products of binary sparse
matrices over one vocabulary for both sides. Each S1 record repeats once per candidate, so
runs of equal strings are converted to Python and tokenised only once.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sp
from rapidfuzz import distance, fuzz, process

from .blocking import PASS_BITS, SIM_COLUMNS

FeatureGroup = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame]

_EXACT_BITS = PASS_BITS["exact_core"] | PASS_BITS["exact_sorted"] | PASS_BITS["exact_squash"]

# Column contract per group, in model order (07 §2). Changing it changes feature_names, so a
# trained model refuses the new frame (08): add or drop features only with a new version.
FEATURE_COLUMNS: dict[str, list[str]] = {
    "blocking": ["pass_exact", "pass_name_char", "pass_name_addr", *SIM_COLUMNS],
    "name_fuzzy": ["nm_ratio", "nm_partial", "nm_token_sort", "nm_token_set", "nm_jw",
                   "core_ratio", "core_token_set", "core_jw", "core_lev", "squash_ratio"],
    "name_tokens": ["tok_jaccard", "tok_dice", "tok_common", "tok_len_l", "tok_len_r",
                    "first_eq", "sorted_eq", "prefix4_eq"],
    "legal": ["legal_eq", "legal_missing_l", "legal_missing_r"],
    "numeric": ["num_jaccard", "num_shared_any", "num_first_eq", "postcode_eq"],
    "address": ["ad_token_set", "ad_partial", "ad_ratio", "ad_jaccard", "ad_contain",
                "region_eq", "last_eq", "addr_empty_r"],
    "context": ["ctx_rank_name", "ctx_gap_name", "ctx_rank_addr", "ctx_gap_addr", "ctx_n_cands"],
    "meta": ["is_s3", "non_latin_r", "len_ratio_name"],
    "pool_context": ["ctx_pool_indegree"],
}

# The only columns allowed to hold NaN (07 §2 "Missing"); every other column is always set.
NAN_FEATURES = frozenset({
    *SIM_COLUMNS, *FEATURE_COLUMNS["name_fuzzy"], "tok_jaccard", "tok_dice",
    *FEATURE_COLUMNS["numeric"], "ad_token_set", "ad_partial", "ad_ratio", "ad_jaccard",
    "ad_contain", "ctx_gap_name", "ctx_gap_addr", "len_ratio_name",
})

# Normalised columns each group reads: build_features aligns only these to the pairs.
_INPUTS: dict[str, tuple[str, ...]] = {
    "blocking": (),
    "name_fuzzy": ("name_norm", "name_core", "name_squash"),
    "name_tokens": ("name_core", "name_first", "name_sorted", "name_squash"),
    "legal": ("legal_form",),
    "numeric": ("addr_nums", "postcode"),
    "address": ("addr_norm", "region", "addr_last"),
    "context": ("addr_norm",),  # only when the address group does not run
    "meta": ("name_core", "non_latin"),
    "pool_context": (),
}
_ALL_INPUTS = tuple(dict.fromkeys(c for cols in _INPUTS.values() for c in cols))

# The order build_features computes groups in (the output order is feature_names'): address
# before context, which reuses its ad_token_set, and the address-side groups before the
# name-side ones, so fewer aligned string columns are alive at the same time.
_COMPUTE_ORDER = ("blocking", "legal", "numeric", "address", "context", "name_fuzzy",
                  "name_tokens", "meta", "pool_context")

# Columns build_features adds to each pairs chunk: values that need more than the chunk (the
# partition-wide in-degree) or that one group already computed for another (the context
# group ranks candidates on the address group's ad_token_set).
_INDEGREE = "ctx_pool_indegree"
_AD_TOKEN_SET = "ad_token_set"

# (scorer, divisor mapping its range to [0, 1]). All are symmetric and score equal strings at
# their maximum, which _fuzzy relies on to skip identical pairs.
Scorer = tuple[Callable[..., float], float]
_RATIO: Scorer = (fuzz.ratio, 100.0)
_PARTIAL: Scorer = (fuzz.partial_ratio, 100.0)
_TOKEN_SORT: Scorer = (fuzz.token_sort_ratio, 100.0)
_TOKEN_SET: Scorer = (fuzz.token_set_ratio, 100.0)
_JARO_WINKLER: Scorer = (distance.JaroWinkler.normalized_similarity, 1.0)
_LEVENSHTEIN: Scorer = (distance.Levenshtein.normalized_similarity, 1.0)


# ------------------------------------------------------------------ helpers ----
def _column(values: pd.Series) -> pa.ChunkedArray:
    """A column as Arrow chunks, zero-copy for Arrow-backed pandas columns.

    Strings come back as ``large_string`` (pandas' own Arrow type), so arrays from different
    frames compare and concatenate; missing strings become "" (the normalised convention).
    Chunks are kept: Parquet-loaded frames have several, and joining them copies the column.
    """
    arr = pa.array(values)
    if not isinstance(arr, pa.ChunkedArray):
        arr = pa.chunked_array([arr])
    if pa.types.is_string(arr.type) or pa.types.is_null(arr.type):  # null: empty object column
        arr = arr.cast(pa.large_string())
    if pa.types.is_large_string(arr.type) and arr.null_count:
        arr = pc.fill_null(arr, "")
    return arr


def _arrow(values: pd.Series) -> pa.Array:
    """One contiguous Arrow array for a column of pair-aligned rows (see ``_column``)."""
    arr = _column(values)
    return arr.chunk(0) if arr.num_chunks == 1 else arr.combine_chunks()


def _np(arr: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """NumPy copy of a null-free Arrow array (booleans are bit-packed in Arrow, hence a copy)."""
    return arr.to_numpy(zero_copy_only=False)


def _empty(arr: pa.Array) -> np.ndarray:
    """True where the string is empty."""
    return pc.binary_length(arr).to_numpy() == 0


def _eq(left: pa.Array, right: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """Per pair: (equal and both non-empty, both non-empty)."""
    both = ~(_empty(left) | _empty(right))
    return _np(pc.equal(left, right)) & both, both


def _ratio(num: np.ndarray, den: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``num / den`` as float32 where ``valid``, NaN elsewhere (never divides by zero)."""
    out = np.full(len(num), np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=valid)
    return out


def _tristate(flag: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """A 0/1 flag where ``valid``, NaN elsewhere."""
    return np.where(valid, flag, np.nan).astype(np.float32)


def _frame(index: pd.Index, columns: dict[str, np.ndarray]) -> pd.DataFrame:
    """A group's output: float32 columns in the given order, on the chunk's index."""
    data = {name: np.asarray(col, dtype=np.float32) for name, col in columns.items()}
    return pd.DataFrame(data, index=index, copy=False)


def _runs(values: pa.Array, repeats: bool) -> tuple[pa.Array, np.ndarray]:
    """The value of each run of equal neighbours and the run index of each row.

    Only the S1 side has runs (its record repeats once per candidate): ``repeats=False``
    skips the encoding, which would only copy the near-unique pool side.
    """
    if not repeats:
        return values, np.arange(len(values))
    ree = pc.run_end_encode(values)
    lengths = np.diff(ree.run_ends.to_numpy(), prepend=0)
    return ree.values, np.repeat(np.arange(len(lengths)), lengths)


def _py_strings(values: pa.Array, repeats: bool) -> list[str]:
    """Python strings for rapidfuzz, converting each run of repeated values once.

    Converting is the serial part of a ``cpdist`` call; on the S1 side this cuts its cost and
    memory to one string object per S1 record.
    """
    runs, run = _runs(values, repeats)
    strings = runs.to_numpy(zero_copy_only=False)
    return (strings[run] if repeats else strings).tolist()


def _fuzzy(left: pa.Array, right: pa.Array, scorers: Sequence[Scorer]) -> list[np.ndarray]:
    """rapidfuzz similarity of each aligned string pair, one float32 array per scorer.

    NaN where either side is empty. Equal strings get exactly 1.0 without calling rapidfuzz
    (every scorer used here returns its maximum on equal strings); the other pairs go through
    ``process.cpdist``, which scores row i against row i in C++ threads.
    """
    n = len(left)
    empty = _empty(left) | _empty(right)
    same = _np(pc.equal(left, right)) & ~empty
    todo = np.flatnonzero(~(empty | same))
    if 0 < len(todo) < n:
        idx = pa.array(todo)
        left, right = left.take(idx), right.take(idx)
    queries = _py_strings(left, repeats=True) if len(todo) else []
    choices = _py_strings(right, repeats=False) if len(todo) else []
    outs = []
    for scorer, scale in scorers:
        out = np.full(n, np.nan, dtype=np.float32)
        out[same] = 1.0
        if len(todo):
            score = process.cpdist(queries, choices, scorer=scorer, workers=-1, dtype=np.float32)
            out[todo] = score / np.float32(scale)
        outs.append(out)
    return outs


def _token_sets(left: pa.Array, right: pa.Array) -> tuple[np.ndarray, ...]:
    """Distinct-token overlap of aligned space-separated strings.

    Returns per pair ``(common, n_left, n_right, first_left, first_right)``: distinct tokens
    shared, distinct tokens on each side, and the vocabulary code of each side's first token
    (-1 for an empty string; codes compare across sides). Every run of equal strings becomes
    one row of a binary CSR matrix over a vocabulary shared by both sides; ``common`` is the
    row-wise size of the elementwise product of the two sides' rows.
    """
    docs_l, run_l = _runs(left, repeats=True)
    docs_r, run_r = _runs(right, repeats=False)
    docs = pa.concat_arrays([docs_l, docs_r])
    filled = ~_empty(docs)  # "" has no token (splitting it would give one empty token)
    tokens = pc.split_pattern(docs if filled.all() else docs.filter(pa.array(filled)), " ")
    vocab = pc.dictionary_encode(pc.list_flatten(tokens))
    codes = _np(vocab.indices).astype(np.int32)  # astype copies: scipy sorts it in place
    counts = np.zeros(len(docs), dtype=np.int64)
    counts[filled] = pc.list_value_length(tokens).to_numpy()
    offsets = np.concatenate([[0], np.cumsum(counts)])
    blank = pc.index(vocab.dictionary, "").as_py()
    if blank >= 0:  # empty tokens from doubled or edge spaces (never in normalised text)
        keep = codes != blank
        offsets = np.concatenate([[0], np.cumsum(keep)])[offsets]  # re-point rows past drops
        codes = codes[keep]
    has = offsets[1:] > offsets[:-1]
    first = np.full(len(docs), -1, dtype=np.int32)
    first[has] = codes[offsets[:-1][has]]  # before scipy reorders the tokens of each row
    mat = sp.csr_array((np.ones(len(codes), dtype=np.int8), codes, offsets.astype(np.int32)),
                       shape=(len(docs), max(len(vocab.dictionary), 1)))
    mat.sum_duplicates()  # sorts each row and merges repeated tokens: set semantics
    mat.data[:] = 1
    size = np.diff(mat.indptr)
    rows_l, rows_r = run_l, run_r + len(docs_l)
    common = np.diff(mat[rows_l].multiply(mat[rows_r]).indptr)  # stored entries = shared tokens
    return common, size[rows_l], size[rows_r], first[rows_l], first[rows_r]


def _group_starts(ids: pa.Array | pa.ChunkedArray) -> np.ndarray:
    """First row of every run of equal S1 ids; ValueError if an id has two separate runs."""
    n = len(ids)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    new = _np(pc.not_equal(ids.slice(1), ids.slice(0, n - 1)))
    starts = np.concatenate([[0], np.flatnonzero(new) + 1])
    if pc.count_distinct(ids.take(pa.array(starts))).as_py() != len(starts):
        raise ValueError("pairs must be grouped by source1_entity_id (blocking sorts them): "
                         "an id occurs in two separate runs")
    return starts


def _rank_and_gap(sim: np.ndarray, starts: np.ndarray,
                  group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank of ``sim`` inside its S1 group, and the gap to the group's best value.

    Rank 1 is the best; tied values share the tie's best rank (1, 2, 2, 4) and NaN ranks after
    every value. Gap = group best - this value, NaN where this value is NaN. Groups must be
    contiguous runs (``starts``); one argsort of an int64 key (group in the high 32 bits,
    descending value in the low 32) ranks every group at once.
    """
    n = len(sim)
    # A non-negative float32's bit pattern grows with its value; +1 frees 0 for NaN.
    bits = np.where(sim > 0, sim, np.float32(0)).astype(np.float32).view(np.int32)
    bits = bits.astype(np.int64) + 1
    bits[np.isnan(sim)] = 0
    key = (group.astype(np.int64) << 32) | (0x7FFFFFFF - bits)  # ascending key = best first
    order = np.argsort(key)
    ranked = key[order]
    tie = np.concatenate([[True], ranked[1:] != ranked[:-1]])
    tie_start = np.maximum.accumulate(np.where(tie, np.arange(n), 0))
    rank = np.empty(n, dtype=np.float32)
    # sorting keeps every group on its own rows, so its first sorted slot is its start row
    rank[order] = tie_start - starts[group[order]] + 1
    best = np.fmax.reduceat(sim, starts)  # fmax ignores NaN; NaN only if the group has none
    return rank, best[group] - sim
