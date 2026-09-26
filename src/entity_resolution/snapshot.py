"""Feature snapshots for model experiments (M4): build the matcher's inputs once, reuse them.

Why: ``pipeline.fit`` + ``pipeline.run_fold`` re-block, re-feature and re-score every side on
every call (35-50 min per version on the 15.7 GB laptop), yet a model experiment (LightGBM
parameters, seeds, sample weights / hard negatives, calibration) changes none of that data.
A snapshot stores exactly what those two functions feed the matcher and the decision layer,
once per data configuration, as Parquet under ``<dataset>/.cache/m4/<key>/``:

    fit     fit-side S1 sample of the inner split: pairs, features, labels (training rows)
    stop    tune-side S1 sample (n_stop_s1): pairs, features, labels (early stopping)
    tune    every tune S1 (or n_tune_s1): pairs, features, labels, S1 ids, truth
            (``decision.tune``)
    val     the fixed val fold as ``run_fold`` scores it: pairs, features, labels, S1 ids and
            country, truth, pool ids, blocking report, normalised S1 rows and matched pool
            rows (``evaluate.slice_report``)
    harder  ``evaluate.harder_fold(val)`` re-blocked as ``run_fold(..., tag="harder")`` does

Equivalence with the pipeline is the contract. Every side is built by the pipeline's own
functions with the same arguments: ``normalise_split``, ``learn_token_map``,
``trainset.inner_split`` / ``sample_s1``, ``pipeline._side`` (read-only use of a private
helper: fit, stop and tune sides) or the ``load_normalised`` + ``prepare`` + ``_tag`` lines
of ``run_fold`` (val, harder), ``label_pairs`` and ``build_features``. Features are built
like the pipeline builds them: whole side at once for fit/stop (``fit``), per
``iter_chunks`` slice for tune/val/harder (``score``), so even the chunk-dependent
``pool_context`` group matches. Rows keep the pipeline's pair order; float32 values
round-trip Parquet bit for bit.

Key: a short hash of the code that produces the data (content of the data modules, not the
git commit, so edits to model.py or decision.py keep the snapshot valid), the normalisation,
blocking and feature configuration, the learned token map, sample sizes, seeds and the fold
id fingerprints. The model and grid are not in it: model experiments vary them.
The git commit is recorded in ``manifest.json`` for the record.

Memory: one side is in memory at a time and tune/val/harder features are streamed to Parquet
per chunk (peak about the pipeline's own ``score``).
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config as C
from . import pipeline as P
from .data import isin
from .evaluate import blocking_report, positions
from .features import build_features, feature_names, iter_chunks
from .split import Fold
from .trainset import INNER_FRAC, INNER_SEED, SAMPLE_SEED, label_pairs

FORMAT = 1                      # bump when the on-disk layout changes
SIDES = ("fit", "stop", "tune", "val", "harder")
WHOLE_SIDES = ("fit", "stop")   # featured in one build_features call, as pipeline.fit does
MANIFEST = "manifest.json"
COMPRESSION = "zstd"
LABEL = "label"
PASS = "pass"
META_COLUMNS = [C.S1_ID, C.ENTITY_ID, C.COUNTRY, PASS, LABEL]
# modules whose code decides the snapshot's rows and values: their content is in the key
DATA_MODULES = ("config", "data", "split", "normalize", "token_maps", "blocking", "features",
                "trainset", "evaluate", "pipeline")


# -------------------------------------------------------------------- key ----
def code_hash(modules: Sequence[str] = DATA_MODULES) -> str:
    """Short hash of the data modules' source (CRLF read as LF, so checkouts agree)."""
    h = hashlib.sha1()
    here = Path(__file__).parent
    for name in modules:
        h.update(name.encode("utf-8"))
        h.update((here / f"{name}.py").read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:8]


def snapshot_key(cfg: P.PipelineConfig, train: Fold, val: Fold,
                 token_map: dict[str, str]) -> tuple[str, dict]:
    """``(key, parts)``: the snapshot folder name and everything hashed into it.

    ``cfg.model`` and ``cfg.grid`` are left out on purpose (``evaluate_params`` varies them);
    so is ``normalise.chunk_rows``, which only bounds memory.
    """
    norm = asdict(cfg.normalise)
    norm.pop("chunk_rows", None)
    parts = {
        "format": FORMAT,
        "code": code_hash(),
        "normalise": P._hash(norm),
        "blocking": cfg.blocking.key(),
        "feature_groups": list(cfg.feature_groups),
        "token_map": P._hash(token_map),
        "samples": {"n_fit_s1": cfg.n_fit_s1, "n_stop_s1": cfg.n_stop_s1,
                    "n_tune_s1": cfg.n_tune_s1, "chunk_rows": cfg.chunk_rows,
                    "inner_seed": INNER_SEED, "inner_frac": INNER_FRAC,
                    "sample_seed": SAMPLE_SEED, "harder": [0.2, 99]},  # harder_fold defaults
        "folds": {"train_s1": P._ids_key(train.s1[C.ENTITY_ID]),
                  "train_pool": P._ids_key(P.pool_of(train)[C.ENTITY_ID]),
                  "train_pairs": len(train.pairs),
                  "val_s1": P._ids_key(val.s1[C.ENTITY_ID]),
                  "val_pool": P._ids_key(P.pool_of(val)[C.ENTITY_ID]),
                  "val_pairs": len(val.pairs)},
    }
    return P._hash(parts), parts


# ------------------------------------------------------------------ files ----
def _json_default(obj: object) -> object:
    """numpy scalars as Python numbers, anything else as its string."""
    return obj.item() if hasattr(obj, "item") else str(obj)


def _write_json(path: Path, payload: object) -> None:
    """Pretty JSON, written to a temporary file first so a reader never sees half of it."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    """A frame as Parquet, atomically (temporary file, then rename)."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False, compression=COMPRESSION)
    tmp.replace(path)


def _large(values: pd.Series) -> pa.Array | pa.ChunkedArray:
    """A string column as Arrow ``large_string`` (zero-copy for pandas' Arrow strings)."""
    return pa.array(values, type=pa.large_string())


def _schema(names: Sequence[str]) -> pa.Schema:
    """Side file schema: pair ids, S1 country, pass bits, label, then float32 features."""
    return pa.schema([pa.field(C.S1_ID, pa.large_string()),
                      pa.field(C.ENTITY_ID, pa.large_string()),
                      pa.field(C.COUNTRY, pa.large_string()),
                      pa.field(PASS, pa.uint8()), pa.field(LABEL, pa.int8()),
                      *(pa.field(n, pa.float32()) for n in names)])


def _country_of(ids: pd.Series, s1n: pd.DataFrame) -> pa.Array | pa.ChunkedArray:
    """S1 country (as normalised, the value ``decide_by_country`` partitions on) per id."""
    at = positions(ids, s1n[C.ENTITY_ID])
    if (at < 0).any():
        raise ValueError(f"{int((at < 0).sum()):,} S1 ids are not in the normalised records")
    return _large(s1n[C.COUNTRY]).take(pa.array(at))


def _write_side(folder: Path, side: str, pairs: pd.DataFrame, s1n: pd.DataFrame,
                pooln: pd.DataFrame, truth: pd.DataFrame, cfg: P.PipelineConfig) -> dict:
    """``<side>.parquet``: one row per candidate pair, in the pipeline's pair order.

    Fit/stop sides are featured in one ``build_features`` call over the whole side, like
    ``pipeline.fit``; the other sides per ``iter_chunks`` slice, like ``pipeline.score``, and
    each slice is written before the next is built, so only one chunk of features is alive.
    """
    names = feature_names(cfg.feature_groups)
    schema = _schema(names)
    n = len(pairs)
    label = label_pairs(pairs[[C.S1_ID, C.ENTITY_ID]], truth)[LABEL].to_numpy(np.int8)
    country = _country_of(pairs[C.S1_ID], s1n)
    s1_ids, pool_ids = _large(pairs[C.S1_ID]), _large(pairs[C.ENTITY_ID])
    passes = pairs[PASS].to_numpy(np.uint8)
    whole = side in WHOLE_SIDES
    X_all = (build_features(pairs, s1n, pooln, groups=cfg.feature_groups,
                            chunk_rows=cfg.chunk_rows) if whole else None)
    slices = ([slice(a, min(a + cfg.chunk_rows, n)) for a in range(0, n, cfg.chunk_rows)]
              if whole else iter_chunks(pairs, cfg.chunk_rows))
    path = folder / f"{side}.parquet"
    tmp = path.with_name(path.name + ".tmp")
    wrote = False
    with pq.ParquetWriter(tmp, schema, compression=COMPRESSION) as writer:
        for sl in slices:
            X = (X_all.iloc[sl] if whole else
                 build_features(pairs.iloc[sl], s1n, pooln, groups=cfg.feature_groups,
                                chunk_rows=cfg.chunk_rows))
            m = sl.stop - sl.start
            cols = [s1_ids.slice(sl.start, m), pool_ids.slice(sl.start, m),
                    country.slice(sl.start, m), pa.array(passes[sl]), pa.array(label[sl]),
                    *(pa.array(X[c].to_numpy(dtype=np.float32)) for c in names)]
            writer.write_table(pa.Table.from_arrays(cols, schema=schema))
            wrote = True
            del X
        if not wrote:  # an empty side still gets a readable file with the schema
            writer.write_table(schema.empty_table())
    tmp.replace(path)
    return {"rows": n, "positives": int(label.sum()), "file": path.name}


def _write_ids(folder: Path, side: str, s1: pd.DataFrame, s1n: pd.DataFrame,
               truth: pd.DataFrame) -> int:
    """``<side>_s1.parquet`` (every S1 of the side, candidates or not, with its country) and
    ``<side>_truth.parquet`` (the true pairs of those S1). Returns the S1 count."""
    ids = s1[C.ENTITY_ID].reset_index(drop=True)
    frame = pa.table({C.ENTITY_ID: _large(ids), C.COUNTRY: _country_of(ids, s1n)}).to_pandas()
    _write_frame(frame, folder / f"{side}_s1.parquet")
    own = truth[isin(truth[C.S1_ID], pd.Index(ids))][[C.S1_ID, C.ENTITY_ID]]
    _write_frame(own.reset_index(drop=True), folder / f"{side}_truth.parquet")
    return len(ids)


def _write_val_extras(folder: Path, fold: Fold, s1n: pd.DataFrame, pooln: pd.DataFrame) -> None:
    """What rebuilding the val ``Fold`` and ``slice_report`` need beyond the pairs.

    ``val_pool``: every pool id with its source (score_pairs' foreign-id check, the
    blocking report's pool size); ``val_s1n``: the normalised val S1 rows; ``val_pooln``:
    the normalised pool rows of the true pairs (all ``slice_report`` looks up).
    """
    pool = pd.concat([pd.DataFrame({C.ENTITY_ID: fold.s2[C.ENTITY_ID], "source": np.int8(2)}),
                      pd.DataFrame({C.ENTITY_ID: fold.s3[C.ENTITY_ID], "source": np.int8(3)})],
                     ignore_index=True)
    _write_frame(pool, folder / "val_pool.parquet")
    _write_frame(s1n.reset_index(drop=True), folder / "val_s1n.parquet")
    matched = pooln[isin(pooln[C.ENTITY_ID], pd.Index(fold.pairs[C.ENTITY_ID]))]
    _write_frame(matched.reset_index(drop=True), folder / "val_pooln.parquet")


def _fold_records(fold: Fold, tag: str, cfg: P.PipelineConfig, token_map: dict[str, str],
                  info: dict, timings: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalised S1 / pool records and candidate pairs of a labelled fold: the first lines
    of ``pipeline.run_fold`` with the same arguments, plus its blocking report."""
    t0 = time.perf_counter()
    s1n = P.load_normalised("train", (1,), cfg, fold.s1[C.ENTITY_ID], token_map)
    pooln = P.load_normalised("train", (2, 3), cfg, P.pool_of(fold)[C.ENTITY_ID], token_map)
    timings[f"{tag}_load_seconds"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    pairs = P.prepare(s1n, pooln, cfg, P._tag(tag, s1n, pooln, token_map))
    timings[f"{tag}_blocking_seconds"] = round(time.perf_counter() - t0, 2)
    if len(fold.s1):
        info[f"{tag}_blocking"] = blocking_report(pairs, fold)
    info[f"{tag}_pairs"] = len(pairs)
    return s1n, pooln, pairs


def _files_bytes(folder: Path, side: str) -> int:
    """Bytes on disk of a side's files."""
    return sum(p.stat().st_size for p in folder.glob(f"{side}*.parquet"))
