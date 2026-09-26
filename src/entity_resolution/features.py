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

from collections.abc import Callable, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sp
from rapidfuzz import distance, fuzz, process

from . import config as C
from .blocking import PAIR_COLUMNS, PASS_BITS, SIM_COLUMNS

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
    "address_extra": ["ad_contain_r", "addr_empty_l", "num_contain_l", "num_contain_r",
                      "postcode_prefix_eq", "addr_len_ratio"],
}

# The only columns allowed to hold NaN (07 §2 "Missing"); every other column is always set.
NAN_FEATURES = frozenset({
    *SIM_COLUMNS, *FEATURE_COLUMNS["name_fuzzy"], "tok_jaccard", "tok_dice",
    *FEATURE_COLUMNS["numeric"], "ad_token_set", "ad_partial", "ad_ratio", "ad_jaccard",
    "ad_contain", "ctx_gap_name", "ctx_gap_addr", "len_ratio_name",
    "ad_contain_r", "num_contain_l", "num_contain_r", "postcode_prefix_eq", "addr_len_ratio",
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
    "address_extra": ("addr_norm", "addr_nums", "postcode"),
}
_ALL_INPUTS = tuple(dict.fromkeys(c for cols in _INPUTS.values() for c in cols))

# The order build_features computes groups in (the output order is feature_names'): address
# before context, which reuses its ad_token_set, and the address-side groups before the
# name-side ones, so fewer aligned string columns are alive at the same time.
_COMPUTE_ORDER = ("blocking", "legal", "numeric", "address", "address_extra", "context",
                  "name_fuzzy", "name_tokens", "meta", "pool_context")

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


# ------------------------------------------------------------ feature groups ----
def _blocking(pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Blocking evidence (06 §2): which passes proposed the pair, and their cosines.

    pass_exact       an exact key pass proposed it (name_core, name_sorted or name_squash)
    pass_name_char   the name char-gram top-k pass (P2) proposed it
    pass_name_addr   the name + address word top-k pass (P3) proposed it
    sim_*            the pass cosines copied from the pairs; NaN when that pass did not
                     propose the pair
    """
    bits = pairs["pass"].to_numpy()
    return _frame(pairs.index, {
        "pass_exact": (bits & _EXACT_BITS) != 0,
        "pass_name_char": (bits & PASS_BITS["name_char"]) != 0,
        "pass_name_addr": (bits & PASS_BITS["name_addr_word"]) != 0,
        **{c: pairs[c].to_numpy(dtype=np.float32, na_value=np.nan) for c in SIM_COLUMNS},
    })


def _name_fuzzy(pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Fuzzy name similarity (rapidfuzz), NaN when either name is empty.

    nm_*          on name_norm: ratio, partial_ratio, token_sort_ratio, token_set_ratio and
                  Jaro-Winkler
    core_*        on name_core (legal forms removed): ratio, token_set_ratio, Jaro-Winkler and
                  Levenshtein
    squash_ratio  ratio on name_squash (letters and digits only: domain and handle forms)

    07 lists ``core_indel``, the Indel normalised similarity; rapidfuzz defines ``fuzz.ratio``
    as exactly that (x100), so it would duplicate ``core_ratio``. ``core_lev`` (Levenshtein: a
    substitution is one edit, normalised by the longer name) is the distinct edit-distance
    view in its place.
    """
    scores: list[np.ndarray] = []
    for column, scorers in (("name_norm", (_RATIO, _PARTIAL, _TOKEN_SORT, _TOKEN_SET,
                                           _JARO_WINKLER)),
                            ("name_core", (_RATIO, _TOKEN_SET, _JARO_WINKLER, _LEVENSHTEIN)),
                            ("name_squash", (_RATIO,))):
        scores += _fuzzy(_arrow(left[column]), _arrow(right[column]), scorers)
    return _frame(pairs.index, dict(zip(FEATURE_COLUMNS["name_fuzzy"], scores, strict=True)))


def _name_tokens(pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Word overlap of the core names and equality of the derived name keys.

    tok_jaccard, tok_dice  Jaccard and Dice of the name_core token sets; NaN if either is empty
    tok_common             distinct tokens shared (0 when a side is empty)
    tok_len_l, tok_len_r   distinct tokens on each side
    first_eq               equal non-empty name_first
    sorted_eq              equal non-empty name_sorted (the same words in any order)
    prefix4_eq             equal non-empty first 4 characters of name_squash
    """
    common, n_l, n_r, _, _ = _token_sets(_arrow(left["name_core"]), _arrow(right["name_core"]))
    both = (n_l > 0) & (n_r > 0)
    squash_l, squash_r = _arrow(left["name_squash"]), _arrow(right["name_squash"])
    return _frame(pairs.index, {
        "tok_jaccard": _ratio(common, n_l + n_r - common, both),
        "tok_dice": _ratio(2 * common, n_l + n_r, both),
        "tok_common": common,
        "tok_len_l": n_l,
        "tok_len_r": n_r,
        "first_eq": _eq(_arrow(left["name_first"]), _arrow(right["name_first"]))[0],
        "sorted_eq": _eq(_arrow(left["name_sorted"]), _arrow(right["name_sorted"]))[0],
        "prefix4_eq": _eq(pc.utf8_slice_codeunits(squash_l, 0, 4),
                          pc.utf8_slice_codeunits(squash_r, 0, 4))[0],
    })


def _legal(pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Legal-form agreement.

    legal_eq                          equal and non-empty legal_form
    legal_missing_l, legal_missing_r  that side has no legal form
    """
    legal_l, legal_r = _arrow(left["legal_form"]), _arrow(right["legal_form"])
    return _frame(pairs.index, {
        "legal_eq": _eq(legal_l, legal_r)[0],
        "legal_missing_l": _empty(legal_l),
        "legal_missing_r": _empty(legal_r),
    })


def _numeric(pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Numbers in the addresses: house numbers and postcodes separate same-name decoys.

    num_jaccard     Jaccard of the addr_nums token sets
    num_shared_any  1 if any number is shared
    num_first_eq    1 if the first numbers (usually the house number) are equal
    postcode_eq     1 if the postcodes are equal
    Each is NaN when either side has no number (no postcode, for postcode_eq), else 0/1 or a
    share, so "no evidence" never reads as "disagreement".
    """
    common, n_l, n_r, first_l, first_r = _token_sets(_arrow(left["addr_nums"]),
                                                     _arrow(right["addr_nums"]))
    both = (n_l > 0) & (n_r > 0)
    post_eq, post_both = _eq(_arrow(left["postcode"]), _arrow(right["postcode"]))
    return _frame(pairs.index, {
        "num_jaccard": _ratio(common, n_l + n_r - common, both),
        "num_shared_any": _tristate(common > 0, both),
        "num_first_eq": _tristate(first_l == first_r, both),
        "postcode_eq": _tristate(post_eq, post_both),
    })


def _address(pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Address agreement: names rank the candidates, addresses decide (01 §12).

    ad_token_set, ad_partial, ad_ratio  rapidfuzz token_set_ratio, partial_ratio and ratio on
                                        addr_norm
    ad_jaccard                          Jaccard of the addr_norm token sets
    ad_contain                          share of the S1 address tokens found in the pool
                                        address (dropped components; asymmetric by design)
    region_eq, last_eq                  equal non-empty region / addr_last (city hint)
    addr_empty_r                        the pool address is empty: name-only evidence
    The similarities are NaN when either address is empty; the flags are 0/1.
    """
    addr_l, addr_r = _arrow(left["addr_norm"]), _arrow(right["addr_norm"])
    token_set, partial, ratio = _fuzzy(addr_l, addr_r, (_TOKEN_SET, _PARTIAL, _RATIO))
    common, n_l, n_r, _, _ = _token_sets(addr_l, addr_r)
    both = (n_l > 0) & (n_r > 0)
    return _frame(pairs.index, {
        "ad_token_set": token_set,
        "ad_partial": partial,
        "ad_ratio": ratio,
        "ad_jaccard": _ratio(common, n_l + n_r - common, both),
        "ad_contain": _ratio(common, n_l, both),
        "region_eq": _eq(_arrow(left["region"]), _arrow(right["region"]))[0],
        "last_eq": _eq(_arrow(left["addr_last"]), _arrow(right["addr_last"]))[0],
        "addr_empty_r": _empty(addr_r),
    })


def _context(pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Competition inside the S1 group: what the decision layer will see (07 §1).

    ctx_rank_name, ctx_gap_name  rank of sim_name_char in the group (1 = best, ties share
                                 the best rank, NaN last) and the group best minus this value
                                 (NaN where this pair has no sim_name_char)
    ctx_rank_addr, ctx_gap_addr  the same on ad_token_set
    ctx_n_cands                  candidates in the group

    ``pairs`` must hold whole S1 groups, which iter_chunks guarantees. ad_token_set comes
    from the chunk when build_features already computed it (address group), else from here.
    """
    n = len(pairs)
    if n == 0:  # reduceat rejects empty input
        return _frame(pairs.index, {c: np.zeros(0) for c in FEATURE_COLUMNS["context"]})
    starts = _group_starts(_arrow(pairs[C.S1_ID]))
    sizes = np.diff(np.append(starts, n))
    group = np.repeat(np.arange(len(starts)), sizes)
    if _AD_TOKEN_SET in pairs.columns:
        addr = pairs[_AD_TOKEN_SET].to_numpy(dtype=np.float32)
    else:
        (addr,) = _fuzzy(_arrow(left["addr_norm"]), _arrow(right["addr_norm"]), (_TOKEN_SET,))
    name = pairs["sim_name_char"].to_numpy(dtype=np.float32, na_value=np.nan)
    rank_name, gap_name = _rank_and_gap(name, starts, group)
    rank_addr, gap_addr = _rank_and_gap(addr, starts, group)
    return _frame(pairs.index, {
        "ctx_rank_name": rank_name,
        "ctx_gap_name": gap_name,
        "ctx_rank_addr": rank_addr,
        "ctx_gap_addr": gap_addr,
        "ctx_n_cands": sizes[group],
    })


def _meta(pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Record-level context.

    is_s3           the pool record comes from Source 3
    non_latin_r     the pool record was transliterated (05 §5)
    len_ratio_name  shorter / longer name_core length; NaN if either name is empty
    """
    len_l = pc.utf8_length(_arrow(left["name_core"])).to_numpy()
    len_r = pc.utf8_length(_arrow(right["name_core"])).to_numpy()
    return _frame(pairs.index, {
        "is_s3": _np(pc.starts_with(_arrow(pairs[C.ENTITY_ID]), C.SOURCE_PREFIX[3])),
        "non_latin_r": right["non_latin"].to_numpy(dtype=bool),
        "len_ratio_name": _ratio(np.minimum(len_l, len_r), np.maximum(len_l, len_r),
                                 (len_l > 0) & (len_r > 0)),
    })


def _pool_context(pairs: pd.DataFrame, left: pd.DataFrame,
                  right: pd.DataFrame) -> pd.DataFrame:
    """ctx_pool_indegree: in how many S1 groups the pool record is a candidate (decoy hubs).

    Pairs are unique, so this is the pool id's pair count. build_features counts it once over
    the whole pairs frame before chunking and passes it in the chunk; blocking partitions by
    country, so that count is partition-wide. Called on its own, the group counts in ``pairs``.
    """
    if _INDEGREE in pairs.columns:
        degree = pairs[_INDEGREE].to_numpy()
    else:
        codes = _np(pc.dictionary_encode(_arrow(pairs[C.ENTITY_ID])).indices)
        degree = np.bincount(codes, minlength=1)[codes]
    return _frame(pairs.index, {_INDEGREE: degree})


def _address_extra(pairs: pd.DataFrame, left: pd.DataFrame,
                   right: pd.DataFrame) -> pd.DataFrame:
    """Address evidence the address and numeric groups leave out (plan groups C3, C4).

    ad_contain_r        share of the pool address tokens found in the S1 address: the reverse
                        of ad_contain, so components only the pool has (a suite, a landmark)
                        show up
    addr_empty_l        the S1 address is empty: name-only evidence (mirrors addr_empty_r)
    num_contain_l       share of the S1 address numbers found among the pool's: a number one
                        side drops keeps a pair plausible, a changed number does not
    num_contain_r       share of the pool address numbers found among the S1's
    postcode_prefix_eq  equal first 3 postcode characters: the same postal area even when a
                        typo in the last digits or a neighbouring code breaks postcode_eq
    addr_len_ratio      fewer / more distinct addr_norm tokens (a bare city against a full
                        street address)
    Tokens are counted once (sets, as in ad_contain). Each share and postcode_prefix_eq is NaN
    when either side lacks the field (no address, no number, no postcode); the flag is 0/1.
    """
    addr_l, addr_r = _arrow(left["addr_norm"]), _arrow(right["addr_norm"])
    common, n_l, n_r, _, _ = _token_sets(addr_l, addr_r)
    both = (n_l > 0) & (n_r > 0)
    num_common, num_l, num_r, _, _ = _token_sets(_arrow(left["addr_nums"]),
                                                 _arrow(right["addr_nums"]))
    nums_both = (num_l > 0) & (num_r > 0)
    # a slice of a non-empty postcode is non-empty, so _eq's "both" is "both have a postcode"
    prefix_eq, prefix_both = _eq(pc.utf8_slice_codeunits(_arrow(left["postcode"]), 0, 3),
                                 pc.utf8_slice_codeunits(_arrow(right["postcode"]), 0, 3))
    return _frame(pairs.index, {
        "ad_contain_r": _ratio(common, n_r, both),
        "addr_empty_l": _empty(addr_l),
        "num_contain_l": _ratio(num_common, num_l, nums_both),
        "num_contain_r": _ratio(num_common, num_r, nums_both),
        "postcode_prefix_eq": _tristate(prefix_eq, prefix_both),
        "addr_len_ratio": _ratio(np.minimum(n_l, n_r), np.maximum(n_l, n_r), both),
    })


REGISTRY: dict[str, FeatureGroup] = {
    "blocking": _blocking,
    "name_fuzzy": _name_fuzzy,
    "name_tokens": _name_tokens,
    "legal": _legal,
    "numeric": _numeric,
    "address": _address,  # computed before context, which reuses its ad_token_set
    "context": _context,
    "meta": _meta,
    "pool_context": _pool_context,
    "address_extra": _address_extra,
}

# The v001 model's 47 features, listed rather than derived from REGISTRY so that a group
# added later is opt-in and the logged v001 run stays reproducible.
# pool_context is opt-in: training pairs come from sampled S1 entities (07 §5), so an
# in-degree counted on them is biased low against val and test, where every S1 competes.
DEFAULT_GROUPS: tuple[str, ...] = ("blocking", "name_fuzzy", "name_tokens", "legal", "numeric",
                                   "address", "context", "meta")


# ----------------------------------------------------------------- building ----
def feature_names(groups: Sequence[str] = DEFAULT_GROUPS) -> list[str]:
    """Feature columns of ``groups``, in order: the column contract the model stores (08)."""
    groups = tuple(groups)
    unknown = [g for g in groups if g not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown feature groups {unknown}; known: {list(REGISTRY)}")
    if len(set(groups)) != len(groups):
        raise ValueError(f"feature groups repeat: {list(groups)}")
    return [name for g in groups for name in FEATURE_COLUMNS[g]]


def _slices(starts: np.ndarray, n: int, chunk_rows: int) -> Iterator[slice]:
    """Slices of at least ``chunk_rows`` rows (except the last) that end on group ends."""
    ends = np.append(starts[1:], n)  # exclusive end row of every group
    start = 0
    while start < n:
        # first group end at or after start + chunk_rows (the frame end at the latest)
        stop = int(ends[np.searchsorted(ends, min(start + chunk_rows, n))])
        yield slice(start, stop)
        start = stop


def iter_chunks(pairs: pd.DataFrame, chunk_rows: int) -> Iterator[slice]:
    """Row slices of about ``chunk_rows`` pairs that never split a Source 1 group.

    Pairs are grouped by ``source1_entity_id`` (blocking sorts them). A slice takes
    ``chunk_rows`` rows, then extends to the end of the group it stopped in, so a group larger
    than ``chunk_rows`` becomes one slice. Raises ValueError when ``chunk_rows < 1`` or when an
    S1 id occurs in two separate runs (its group would be split).
    """
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
    return _slices(_group_starts(_column(pairs[C.S1_ID])), len(pairs), chunk_rows)


def _positions(ids: pa.ChunkedArray, records: pd.DataFrame, column: str) -> np.ndarray:
    """Row of ``records`` holding each id (one pyarrow hash lookup); ValueError if missing."""
    pos = pc.index_in(ids, value_set=_arrow(records[C.ENTITY_ID]))
    if pos.null_count:
        example = ids.filter(pc.is_null(pos))[0].as_py()
        raise ValueError(f"{pos.null_count} pair {column} values are not in the normalised "
                         f"records, e.g. {example!r}")
    return _np(pos)


def _inputs(group: str, groups: tuple[str, ...]) -> tuple[str, ...]:
    """Normalised columns ``group`` reads when computed together with ``groups``."""
    if group == "context" and "address" in groups:
        return ()  # ad_token_set comes from the address group
    return _INPUTS.get(group, _ALL_INPUTS)


def _aligned(columns: dict[str, pa.ChunkedArray], n: int) -> pd.DataFrame:
    """Pair-aligned record columns as a frame (index reset); strings stay Arrow-backed."""
    if not columns:
        return pd.DataFrame(index=pd.RangeIndex(n))
    return pa.table(columns).to_pandas()


def build_features(pairs: pd.DataFrame, s1n: pd.DataFrame, pooln: pd.DataFrame,
                   groups: Sequence[str] = DEFAULT_GROUPS,
                   chunk_rows: int = 2_000_000) -> pd.DataFrame:
    """Feature frame of the candidate ``pairs``: float32, index ``pairs.index``.

    ``pairs`` holds ``PAIR_COLUMNS``, grouped by ``source1_entity_id``; ``s1n`` and ``pooln``
    are the normalised records its ids point to. Columns are ``feature_names(groups)``. Work
    runs in chunks of about ``chunk_rows`` pairs that never split an S1 group, and fills one
    preallocated float32 array (4 bytes x features x pairs: at test scale, call it per slice
    of ``iter_chunks`` and score each). The pool in-degree (``pool_context``) is counted over
    all of ``pairs``, so pass a whole partition or fold when requesting it. Raises
    ValueError for an unknown group, a missing column or a pair id absent from the records.
    """
    groups = tuple(groups)
    names = feature_names(groups)
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
    missing = [c for c in PAIR_COLUMNS if c not in pairs.columns]
    if missing:
        raise ValueError(f"pairs lack columns {missing}")
    need = list(dict.fromkeys(c for g in groups for c in _inputs(g, groups)))
    for label, records in (("s1n", s1n), ("pooln", pooln)):
        lacking = [c for c in [C.ENTITY_ID, *need] if c not in records.columns]
        if lacking:
            raise ValueError(f"{label} lacks normalised columns {lacking}")
    out = np.empty((len(pairs), len(names)), dtype=np.float32)
    if len(pairs):
        _fill(out, pairs, s1n, pooln, groups, chunk_rows)
    return pd.DataFrame(out, index=pairs.index, columns=names, copy=False)


def _fill(out: np.ndarray, pairs: pd.DataFrame, s1n: pd.DataFrame, pooln: pd.DataFrame,
          groups: tuple[str, ...], chunk_rows: int) -> None:
    """Compute ``groups`` chunk by chunk into ``out`` (one row per pair, feature_names order).

    Per chunk, each normalised column is aligned to the pairs once (an Arrow take), handed to
    the groups that read it and released after the last of them, so only a few aligned
    string columns are alive at a time.
    """
    s1_ids, pool_ids = _column(pairs[C.S1_ID]), _column(pairs[C.ENTITY_ID])
    s1_pos = _positions(s1_ids, s1n, C.S1_ID)
    pool_pos = _positions(pool_ids, pooln, C.ENTITY_ID)
    order = sorted(groups, key=lambda g: (_COMPUTE_ORDER + (g,)).index(g))  # unknown ones last
    inputs = {g: _inputs(g, groups) for g in order}
    last = {c: i for i, g in enumerate(order) for c in inputs[g]}  # last group reading c
    s1_cols = {c: _column(s1n[c]) for c in last}
    pool_cols = {c: _column(pooln[c]) for c in last}
    carried = {}
    if "pool_context" in groups:  # partition-wide, before chunking (07 §4)
        carried[_INDEGREE] = np.bincount(pool_pos, minlength=len(pooln))[pool_pos]
    bounds = np.cumsum([0, *(len(FEATURE_COLUMNS[g]) for g in groups)])
    where = {g: slice(int(bounds[i]), int(bounds[i + 1])) for i, g in enumerate(groups)}
    for sl in _slices(_group_starts(s1_ids), len(pairs), chunk_rows):
        chunk = pairs.iloc[sl][PAIR_COLUMNS]
        if carried:
            chunk = chunk.assign(**{name: values[sl] for name, values in carried.items()})
        at_l, at_r = pa.array(s1_pos[sl]), pa.array(pool_pos[sl])
        live_l: dict[str, pa.ChunkedArray] = {}
        live_r: dict[str, pa.ChunkedArray] = {}
        for i, g in enumerate(order):
            for c in inputs[g]:
                if c not in live_l:
                    live_l[c], live_r[c] = s1_cols[c].take(at_l), pool_cols[c].take(at_r)
            left = _aligned({c: live_l[c] for c in inputs[g]}, len(chunk))
            right = _aligned({c: live_r[c] for c in inputs[g]}, len(chunk))
            feats = REGISTRY[g](chunk, left, right)
            if list(feats.columns) != FEATURE_COLUMNS[g]:
                raise RuntimeError(f"group {g!r} returned {list(feats.columns)}, "
                                   f"expected {FEATURE_COLUMNS[g]}")
            out[sl, where[g]] = feats.to_numpy(dtype=np.float32)
            if g == "address" and "context" in groups:
                chunk = chunk.assign(**{_AD_TOKEN_SET: feats[_AD_TOKEN_SET].to_numpy()})
            for c in inputs[g]:
                if last[c] == i:  # no later group reads it
                    del live_l[c], live_r[c]
