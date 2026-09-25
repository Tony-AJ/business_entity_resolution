"""Loaders for the challenge TSV files.

Every file is tab-separated because addresses and ID lists contain commas. Reading
without ``sep="\\t"`` silently yields one column, so loaders check the header. Fields
load as plain strings with NA detection off: an empty ``matched_entity_ids`` means
"no match", and a business can really be called "NA" or "None".
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as C


def read_tsv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read a challenge TSV with every field as a string; empty fields stay ""."""
    return pd.read_csv(path, sep=C.SEP, dtype=str, na_filter=False, encoding="utf-8",
                       usecols=usecols)


def source_path(split: str, source: int, dataset_dir: Path = C.DATASET) -> Path:
    if split not in C.SPLITS:
        raise ValueError(f"split must be one of {C.SPLITS}, got {split!r}")
    return dataset_dir / split / f"{split}_source{source}.tsv"


def load_source(split: str, source: int, dataset_dir: Path = C.DATASET) -> pd.DataFrame:
    """One source file, checked for the expected columns, ID prefix and unique IDs."""
    path = source_path(split, source, dataset_dir)
    df = read_tsv(path)
    missing = [c for c in C.SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}, got {list(df.columns)}")
    ids = df[C.ENTITY_ID]
    prefix = C.SOURCE_PREFIX[source]
    bad = ids[~ids.str.startswith(prefix)]
    if len(bad):
        raise ValueError(f"{path}: {len(bad)} ids lack prefix {prefix!r}, e.g. {bad.iloc[0]!r}")
    dup = ids[ids.duplicated()]
    if len(dup):
        raise ValueError(f"{path}: duplicate entity_id {dup.iloc[0]!r}")
    return df


def load_sources(split: str, dataset_dir: Path = C.DATASET) -> dict[int, pd.DataFrame]:
    """All three sources of a split, keyed 1, 2, 3."""
    return {s: load_source(split, s, dataset_dir) for s in C.SOURCES}


def parse_id_list(field: str) -> list[str]:
    """``"S2-1,S3-4"`` -> ``["S2-1", "S3-4"]``, ``""`` -> ``[]``. Order and duplicates kept."""
    return [t for t in (x.strip() for x in field.split(C.ID_LIST_SEP)) if t]


def read_id_lists(path: Path) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """Header and rows of a ``source1_entity_id<TAB>id,id,...`` file.

    Parsed line by line, not with pandas, so validation sees the file exactly as
    written: row order, duplicate rows and extra columns survive. A row with no
    second field reads as an empty list.
    """
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError(f"{path}: empty file")
    rows = []
    for n, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split(C.SEP)
        if len(parts) > 2:
            raise ValueError(f"{path}:{n}: expected 2 tab-separated fields, got {len(parts)}")
        rows.append((parts[0].strip(), parse_id_list(parts[1] if len(parts) == 2 else "")))
    return lines[0].split(C.SEP), rows


def load_ground_truth(dataset_dir: Path = C.DATASET) -> dict[str, set[str]]:
    """Train labels: Source 1 id -> set of matching S2/S3 ids (empty set = singleton)."""
    path = dataset_dir / "train" / C.GROUND_TRUTH_FILE
    header, rows = read_id_lists(path)
    if header != [C.S1_ID, C.MATCHED_IDS]:
        raise ValueError(f"{path}: unexpected header {header}")
    return {s1: set(ids) for s1, ids in rows}
