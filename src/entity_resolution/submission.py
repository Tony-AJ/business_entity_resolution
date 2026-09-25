"""Write and validate the two submission files.

Rules (docs/PROBLEM_STATEMENT.md, Output), checked for both files:

    * header is exactly ``source1_entity_id<TAB><list column>``
    * one row per Source 1 entity of the split: none missing, none duplicated, no strangers
    * ID lists hold only Source 2 / Source 3 IDs present in the split, no repeats
    * every matched ID is also a candidate of that entity (warning, as upstream)

The organisers' ``utils/validate_submission.py`` stays the final gate before upload.

Two writers produce the same bytes: ``write_submission`` from dicts of id lists (small
inputs), ``write_pairs`` from pair frames (the test split: ~50M candidate pairs).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from . import config as C
from .data import read_id_lists, read_tsv, source_path

MATCHABLE_SOURCES = (2, 3)
PAIR_ID_COLUMNS = (C.S1_ID, C.ENTITY_ID)
WRITE_BATCH = 100_000  # lines per write(): bounds the id strings alive at once (~130 MB)


def write_id_lists(
    path: Path, lists: Mapping[str, Iterable[str]], s1_ids: Sequence[str], column: str
) -> Path:
    """One row per entry of ``s1_ids``, in order; lists de-duplicated, empty when absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"{C.S1_ID}{C.SEP}{column}\n")
        for s1 in s1_ids:
            ids = dict.fromkeys(lists.get(s1, ()))  # order-preserving de-dup
            f.write(f"{s1}{C.SEP}{C.ID_LIST_SEP.join(ids)}\n")
    return path


def write_submission(
    matches: Mapping[str, Iterable[str]],
    candidates: Mapping[str, Iterable[str]],
    s1_ids: Sequence[str],
    out_dir: Path = C.OUTPUT,
) -> tuple[Path, Path]:
    return (write_id_lists(out_dir / C.MATCHING_FILE, matches, s1_ids, C.MATCHED_IDS),
            write_id_lists(out_dir / C.CANDIDATE_FILE, candidates, s1_ids, C.CANDIDATE_IDS))


def write_pairs(
    matches: pd.DataFrame,
    candidates: pd.DataFrame,
    s1_ids: Sequence[str],
    out_dir: Path = C.OUTPUT,
) -> tuple[Path, Path]:
    """Both submission files from pair frames, byte-identical to ``write_submission``.

    ``matches`` (``source1_entity_id``, ``entity_id``: ``decision.MATCH_COLUMNS``) and
    ``candidates`` (at least those two columns: the full candidate-pairs frame) give one
    row per entry of ``s1_ids``, in that order. Each row lists the entity's pool ids
    de-duplicated in frame order (matches therefore appear in whatever order the caller
    sorted them, e.g. by probability) and comma-joined, empty when it has none; rows of
    other Source 1 ids are ignored, as ``write_submission`` ignores their keys.

    Built for the test split (1.7M Source 1 ids, ~50M candidate pairs): grouping runs in
    Arrow and lines are written in batches, never through a Python dict of lists. Both
    frames are checked before anything is written: a missing column, or a listed id
    that is not a Source 2/3 id (such as an ``S1-`` self-match), raises ValueError.
    """
    s1 = _arrow_strings(s1_ids, "s1_ids").combine_chunks()
    if s1.null_count:
        raise ValueError(f"s1_ids holds {s1.null_count} missing ids")
    match_pairs = _pair_table(matches, "matches")          # check both frames
    candidate_pairs = _pair_table(candidates, "candidates")  # before writing either file
    return (_write_pair_lists(out_dir / C.MATCHING_FILE, match_pairs, s1, C.MATCHED_IDS),
            _write_pair_lists(out_dir / C.CANDIDATE_FILE, candidate_pairs, s1,
                              C.CANDIDATE_IDS))


def _arrow_strings(values: object, what: str) -> pa.ChunkedArray:
    """``values`` as an Arrow ``large_string`` column, zero-copy for pandas ``str`` columns.

    Accepts a Series, Index, list, NumPy array or Arrow array of strings; anything else
    (numbers, say) is refused rather than cast, since it could never match an id.
    """
    arr = values if isinstance(values, pa.Array | pa.ChunkedArray) else pa.array(values)
    if isinstance(arr, pa.Array):
        arr = pa.chunked_array([arr])
    kind = arr.type.value_type if pa.types.is_dictionary(arr.type) else arr.type
    if not (pa.types.is_null(kind) or pa.types.is_string(kind)
            or pa.types.is_large_string(kind) or pa.types.is_string_view(kind)):
        raise TypeError(f"{what} must hold strings, got Arrow type {arr.type}")
    return arr.cast(pa.large_string())


def _pair_table(frame: pd.DataFrame, what: str) -> pa.Table:
    """The (``source1_entity_id``, ``entity_id``) columns of a pair frame as an Arrow table.

    Refuses a frame without those columns and any ``entity_id`` that is not a Source 2/3
    id: an ``S1-`` id, an empty string or a missing value is a bug upstream.
    """
    missing = [c for c in PAIR_ID_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{what}: missing columns {missing}, got {list(frame.columns)}")
    table = pa.table({c: _arrow_strings(frame[c], f"{what}.{c}") for c in PAIR_ID_COLUMNS})
    ids = table[C.ENTITY_ID]
    prefixes = [pc.starts_with(ids, C.SOURCE_PREFIX[s]) for s in MATCHABLE_SOURCES]
    valid = pc.fill_null(reduce(pc.or_, prefixes), False)  # a missing id is invalid too
    bad = pc.filter(ids, pc.invert(valid))
    if len(bad):
        raise ValueError(f"{what}: {len(bad)} listed ids are not Source 2/3 ids, "
                         f"e.g. {bad.slice(0, 3).to_pylist()}")
    return table


def _write_pair_lists(path: Path, pairs: pa.Table, s1: pa.Array, column: str) -> Path:
    """One ``<s1 id><TAB><id,id,...>`` line per entry of ``s1``, as ``write_id_lists`` does.

    Pool ids are grouped as int32 dictionary codes, a fraction of the strings' memory,
    single-threaded so that each Source 1 id keeps its ids in frame order. Each batch of
    lines then drops repeats and decodes its own ids: only one batch of strings is alive.
    """
    encoded = pc.dictionary_encode(pairs[C.ENTITY_ID])
    # Arrow finalises one dictionary for all chunks, so codes are global; the length
    # check guards that assumption cheaply
    dictionary = (encoded.chunk(0).dictionary if encoded.num_chunks
                  else pa.array([], pa.large_string()))
    if any(len(chunk.dictionary) != len(dictionary) for chunk in encoded.chunks):
        raise RuntimeError("dictionary_encode returned per-chunk dictionaries")
    codes = pa.chunked_array([chunk.indices for chunk in encoded.chunks], pa.int32())
    del encoded
    grouped = (pa.table({C.S1_ID: pairs[C.S1_ID], "code": codes})
               .group_by(C.S1_ID, use_threads=False).aggregate([("code", "list")]))
    del codes
    keys = grouped[C.S1_ID].combine_chunks()
    lists = grouped["code_list"].combine_chunks()
    del grouped
    position = pc.index_in(s1, value_set=keys)  # row of `lists` per output id; null: none
    tab = pa.scalar(C.SEP, pa.large_string())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"{C.S1_ID}{C.SEP}{column}\n")
        for start in range(0, len(s1), WRITE_BATCH):
            batch = pc.take(lists, position.slice(start, WRITE_BATCH))
            lines = pc.binary_join_element_wise(s1.slice(start, WRITE_BATCH),
                                                _joined_ids(batch, dictionary), tab)
            f.write("\n".join(lines.to_pylist()) + "\n")
    return path


def _joined_ids(lists: pa.ListArray, dictionary: pa.Array) -> pa.Array:
    """Comma-joined ids per list of codes, repeats dropped (first kept, as ``dict.fromkeys``).

    Repeats are found exactly, on (list, code) keys; a null list (no pairs) gives "".
    """
    codes = pc.list_flatten(lists).to_numpy()
    parent = pc.list_parent_indices(lists).to_numpy()
    pair_key = parent.astype(np.int64) * max(len(dictionary), 1) + codes
    first = ~pd.Series(pair_key).duplicated().to_numpy()
    sizes = np.bincount(parent[first], minlength=len(lists))
    offsets = pa.array(np.concatenate(([0], np.cumsum(sizes))).astype(np.int32))
    names = pc.take(dictionary, pa.array(codes[first]))
    comma = pa.scalar(C.ID_LIST_SEP, pa.large_string())
    return pc.binary_join(pa.ListArray.from_arrays(offsets, names), comma)


def split_ids(
    split: str = "test", dataset_dir: Path = C.DATASET, check_ids: bool = False
) -> tuple[list[str], set[str] | None]:
    """Source 1 IDs of a split and, with ``check_ids``, the Source 2/3 IDs lists may use.

    Reads only the ID column. The full Source 2/3 ID set costs about 1 GB of RAM on the
    real test split, so, like the official validator, that check is opt-in; without it
    list entries are checked by ID prefix only.
    """
    def ids(source: int) -> list[str]:
        path = source_path(split, source, dataset_dir)
        return read_tsv(path, usecols=[C.ENTITY_ID])[C.ENTITY_ID].tolist()

    valid = {i for s in MATCHABLE_SOURCES for i in ids(s)} if check_ids else None
    return ids(1), valid


def check_id_lists(
    path: Path, column: str, s1_ids: Collection[str], valid_ids: Collection[str] | None
) -> tuple[list[str], dict[str, list[str]] | None]:
    """Rule violations in one file, plus its rows (None when it cannot be parsed).

    ``valid_ids=None`` checks list entries by their S2-/S3- prefix instead of existence.
    """
    try:
        header, rows = read_id_lists(path)
    except (OSError, ValueError) as e:
        return [str(e)], None
    name = path.name
    errors = []
    if header != [C.S1_ID, column]:
        errors.append(f"{name}: header must be {C.S1_ID}<TAB>{column}, got {header}")
    prefixes = tuple(C.SOURCE_PREFIX[s] for s in MATCHABLE_SOURCES)

    def invalid(i: str) -> bool:
        return i not in valid_ids if valid_ids is not None else not i.startswith(prefixes)

    counts = Counter(s1 for s1, _ in rows)
    expected = set(s1_ids)
    for label, bad in (
        ("duplicate rows", {s for s, n in counts.items() if n > 1}),
        ("missing Source 1 entities", expected - counts.keys()),
        ("rows for unknown Source 1 entities", counts.keys() - expected),
        ("lists with repeated IDs", {s1 for s1, ids in rows if len(ids) != len(set(ids))}),
        ("IDs that are not Source 2/3 records of this split",
         {i for _, ids in rows for i in ids if invalid(i)}),
    ):
        if bad:
            errors.append(f"{name}: {len(bad)} {label}, e.g. {sorted(bad)[:3]}")
    return errors, dict(rows)


def validate(
    matching: Path, candidates: Path, s1_ids: Collection[str],
    valid_ids: Collection[str] | None = None,
) -> tuple[list[str], list[str]]:
    """(errors, warnings) for a submission pair; no errors means safe to upload."""
    errors, matched = check_id_lists(matching, C.MATCHED_IDS, s1_ids, valid_ids)
    cand_errors, cands = check_id_lists(candidates, C.CANDIDATE_IDS, s1_ids, valid_ids)
    errors += cand_errors
    warnings = []
    if matched is not None and cands is not None:
        outside = sorted(s1 for s1, ids in matched.items()
                         if not set(ids) <= set(cands.get(s1, ())))
        if outside:
            warnings.append(f"{len(outside)} entities have matches that are not candidates, "
                            f"e.g. {outside[:3]}")
    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check output files against the submission rules.")
    ap.add_argument("--output-dir", type=Path, default=C.OUTPUT)
    ap.add_argument("--dataset-dir", type=Path, default=C.DATASET)
    ap.add_argument("--split", choices=C.SPLITS, default="test")
    ap.add_argument("--check-ids", action="store_true",
                    help="verify every listed ID exists in the split (about 1 GB RAM on test)")
    args = ap.parse_args(argv)
    s1_ids, valid = split_ids(args.split, args.dataset_dir, args.check_ids)
    errors, warnings = validate(args.output_dir / C.MATCHING_FILE,
                                args.output_dir / C.CANDIDATE_FILE, s1_ids, valid)
    for w in warnings:
        print(f"WARNING: {w}")
    for n, e in enumerate(errors, start=1):
        print(f"{n}. {e}")
    print("FAIL" if errors else "PASS")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
