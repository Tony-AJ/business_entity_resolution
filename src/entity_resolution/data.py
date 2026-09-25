"""Loaders for the challenge TSV files.

Every file is tab-separated because addresses and ID lists contain commas. Reading
without ``sep="\\t"`` silently yields one column, so loaders check the header. Fields
load as plain strings with NA detection off: an empty ``matched_entity_ids`` means
"no match", and a business can really be called "NA" or "None". Quotes are decoded
CSV-style, as the files were written (``\"\"\"ehpad Club SAS\"`` -> ``"ehpad Club SAS``).

Each split holds ~12M records, so every parse is cached as Parquet in
``<dataset>/.cache/`` (rebuilt when the TSV is newer): later loads take seconds and
can read only the columns they need.
"""
from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from . import config as C

CACHE_DIRNAME = ".cache"


def read_tsv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Read a challenge TSV with every field as a string; empty fields stay ""."""
    return pd.read_csv(path, sep=C.SEP, dtype=str, na_filter=False, encoding="utf-8",
                       usecols=usecols)


def isin(values: pd.Series, allowed: Iterable[str]) -> np.ndarray:
    """Fast ``values.isin(allowed)`` for large ID columns, via pyarrow ``is_in``.

    On millions of Arrow-backed strings pandas' own ``isin`` is ~20x slower
    (12 s vs 0.5 s for 5M IDs against 2.5M).
    """
    allowed = allowed if isinstance(allowed, pd.Index | pd.Series) else pd.Index(list(allowed))
    arr = pa.array(values)
    mask = pc.is_in(arr, value_set=pa.array(allowed, type=arr.type))
    return mask.to_numpy(zero_copy_only=False)


def source_path(split: str, source: int, dataset_dir: Path = C.DATASET) -> Path:
    if split not in C.SPLITS:
        raise ValueError(f"split must be one of {C.SPLITS}, got {split!r}")
    return dataset_dir / split / f"{split}_source{source}.tsv"


def _cached(
    path: Path, build: Callable[[], pd.DataFrame], columns: list[str] | None, cache: bool
) -> pd.DataFrame:
    """Serve ``build()`` from a Parquet copy of ``path``, rebuilding it when stale."""
    target = path.parent.parent / CACHE_DIRNAME / f"{path.stem}.parquet"
    if cache and target.exists() and target.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_parquet(target, columns=columns)
    df = build()
    if cache:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(target)  # atomic: a crashed write never leaves a half cache
    return df[columns] if columns else df


def load_source(
    split: str,
    source: int,
    dataset_dir: Path = C.DATASET,
    columns: list[str] | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """One source file, checked for the expected columns, ID prefix and unique IDs.

    ``columns`` limits what is returned (and read, once cached); ``cache=False``
    forces a fresh TSV parse without touching the cache.
    """
    path = source_path(split, source, dataset_dir)
    return _cached(path, lambda: _parse_source(path, source), columns, cache)


def _parse_source(path: Path, source: int) -> pd.DataFrame:
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


def load_sources(
    split: str, dataset_dir: Path = C.DATASET, columns: list[str] | None = None
) -> dict[int, pd.DataFrame]:
    """All three sources of a split, keyed 1, 2, 3."""
    return {s: load_source(split, s, dataset_dir, columns) for s in C.SOURCES}


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


def load_truth_pairs(dataset_dir: Path = C.DATASET, cache: bool = True) -> pd.DataFrame:
    """Train labels as one row per true pair: ``source1_entity_id``, ``entity_id``.

    Singletons have no rows. This exploded form scales to the full ~2.2M entities;
    ``load_ground_truth`` gives a dict of sets for small subsets.
    """
    path = dataset_dir / "train" / C.GROUND_TRUTH_FILE

    def build() -> pd.DataFrame:
        gt = read_tsv(path)
        if list(gt.columns) != [C.S1_ID, C.MATCHED_IDS]:
            raise ValueError(f"{path}: unexpected header {list(gt.columns)}")
        pairs = gt.assign(**{C.ENTITY_ID: gt[C.MATCHED_IDS].str.split(C.ID_LIST_SEP)})
        pairs = pairs.explode(C.ENTITY_ID)[[C.S1_ID, C.ENTITY_ID]]
        pairs[C.ENTITY_ID] = pairs[C.ENTITY_ID].str.strip()
        return pairs[pairs[C.ENTITY_ID] != ""].astype("str").reset_index(drop=True)

    return _cached(path, build, None, cache)


def main() -> None:
    """Parse every challenge file once and fill the Parquet cache (``make cache``)."""
    for split in C.SPLITS:
        for source in C.SOURCES:
            t0 = time.perf_counter()
            n = len(load_source(split, source, columns=[C.ENTITY_ID]))
            print(f"{split}_source{source}: {n:,} rows ({time.perf_counter() - t0:.1f}s)")
            sys.stdout.flush()
    t0 = time.perf_counter()
    n = len(load_truth_pairs())
    print(f"train ground truth: {n:,} true pairs ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
