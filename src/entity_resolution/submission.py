"""Write and validate the two submission files.

Rules (docs/PROBLEM_STATEMENT.md, Output), checked for both files:

    * header is exactly ``source1_entity_id<TAB><list column>``
    * one row per Source 1 entity of the split: none missing, none duplicated, no strangers
    * ID lists hold only Source 2 / Source 3 IDs present in the split, no repeats
    * every matched ID is also a candidate of that entity (warning, as upstream)

The organisers' ``utils/validate_submission.py`` stays the final gate before upload.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path

from . import config as C
from .data import load_sources, read_id_lists

MATCHABLE_SOURCES = (2, 3)


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


def split_ids(split: str = "test", dataset_dir: Path = C.DATASET) -> tuple[list[str], set[str]]:
    """Source 1 IDs of a split, and the Source 2/3 IDs its ID lists may use."""
    sources = load_sources(split, dataset_dir)
    s1_ids = sources[1][C.ENTITY_ID].tolist()
    valid = {i for s in MATCHABLE_SOURCES for i in sources[s][C.ENTITY_ID]}
    return s1_ids, valid


def check_id_lists(
    path: Path, column: str, s1_ids: Collection[str], valid_ids: Collection[str]
) -> tuple[list[str], dict[str, list[str]] | None]:
    """Rule violations in one file, plus its rows (None when it cannot be parsed)."""
    try:
        header, rows = read_id_lists(path)
    except (OSError, ValueError) as e:
        return [str(e)], None
    name = path.name
    errors = []
    if header != [C.S1_ID, column]:
        errors.append(f"{name}: header must be {C.S1_ID}<TAB>{column}, got {header}")
    counts = Counter(s1 for s1, _ in rows)
    expected = set(s1_ids)
    for label, bad in (
        ("duplicate rows", {s for s, n in counts.items() if n > 1}),
        ("missing Source 1 entities", expected - counts.keys()),
        ("rows for unknown Source 1 entities", counts.keys() - expected),
        ("lists with repeated IDs", {s1 for s1, ids in rows if len(ids) != len(set(ids))}),
        ("IDs that are not Source 2/3 records of this split",
         {i for _, ids in rows for i in ids if i not in valid_ids}),
    ):
        if bad:
            errors.append(f"{name}: {len(bad)} {label}, e.g. {sorted(bad)[:3]}")
    return errors, dict(rows)


def validate(
    matching: Path, candidates: Path, s1_ids: Collection[str], valid_ids: Collection[str]
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
    args = ap.parse_args(argv)
    s1_ids, valid = split_ids(args.split, args.dataset_dir)
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
