"""Feature snapshots for model experiments (M4): build the matcher's inputs once, reuse them.

Why: ``pipeline.fit`` + ``pipeline.run_fold`` re-block, re-feature and re-score every side on
every call (35-50 min per version on the 15.7 GB laptop), yet a model experiment (LightGBM
parameters, seeds, sample weights / hard negatives, calibration) changes none of that data.
A snapshot stores exactly what those two functions feed the matcher and the decision layer,
once per data configuration, as Parquet under ``<dataset>/.cache/m4/<key>/``:

    fit     fit-side S1 sample of the inner split: pairs, features, labels (training rows)
    stop    tune-side S1 sample (n_stop_s1): pairs, features, labels (early stopping)
    tune    every tune S1 (or n_tune_s1): pairs, features, labels, S1 ids, truth
            (``decision.tune``); blocked against the tune fold's pool, or the whole train
            pool with ``tune_pool="train"`` (``pipeline._tune_rule``'s "tunedense" side)
    val     the fixed val fold as ``run_fold`` scores it: pairs, features, labels, S1 ids and
            country, truth, pool ids, blocking report, normalised S1 rows and matched pool
            rows (``evaluate.slice_report``)
    harder  ``evaluate.harder_fold(val)`` re-blocked as ``run_fold(..., tag="harder")`` does

``evaluate_params`` then trains a matcher on fit/stop, tunes the rule on tune, freezes it on
val (and harder) and returns the metrics.json keys of 13 §2.2 plus calibration (08 §5), in
minutes instead of rebuilding features. The model-level helpers it calls, ``SeedEnsemble`` /
``fit_matcher`` (seed averaging) and ``reliability`` (calibration), live in model.py.

Equivalence with the pipeline is the contract. Every side is built by the pipeline's own
functions with the same arguments: ``normalise_split``, ``learn_token_map``,
``trainset.inner_split`` / ``sample_s1``, ``pipeline._side`` (read-only use of a private
helper: fit, stop and tune sides) or the ``load_normalised`` + ``prepare`` + ``_tag`` lines
of ``run_fold`` (val, harder, with its ``_with_frequencies``), ``label_pairs`` and
``build_features``. Features are built like the pipeline builds them: whole side at once
for fit/stop (``fit``), per ``iter_chunks`` slice for tune/val/harder (``score``), so even
the chunk-dependent ``pool_context`` group matches; the ``STATS_GROUPS`` read
``pool_stats`` of the side's whole pool, counted once per side as ``score`` does. Rows
keep the pipeline's pair order; float32 values round-trip Parquet bit for bit. Evaluation
reuses ``Matcher``, ``decision.tune``, ``pipeline.decide_by_country`` and
``evaluate.score_pairs``, so default ``PipelineConfig()`` and ``MatcherParams()`` reproduce
``pipeline.fit`` + ``run_fold`` exactly (same rule, ``tune_f_beta``, val ``f_beta``,
matches; tests/test_snapshot.py).

Key: a short hash of the code that produces the data (content of the data modules, not the
git commit, so edits to model.py or decision.py keep the snapshot valid), the normalisation,
blocking and feature configuration, the learned token map, sample sizes, seeds and the fold
id fingerprints. The model and grid are not in it: they are what ``evaluate_params`` varies.
The git commit is recorded in ``manifest.json`` for the record.

Memory: one side is in memory at a time and tune/val/harder features are streamed to Parquet
per chunk (peak about the pipeline's own ``score``); ``evaluate_params`` reads fit/stop into
one float32 matrix each and streams tune/val/harder in ``batch_rows`` batches.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config as C
from . import pipeline as P
from .data import isin
from .decision import DEFAULT_GRID, DecisionRule, Grid, tune
from .evaluate import (
    blocking_report,
    entity_counts,
    harder_fold,
    positions,
    score_pairs,
    slice_report,
)
from .features import build_features, feature_names, iter_chunks, pool_stats
from .model import Matcher, MatcherParams, SeedEnsemble, fit_matcher, reliability
from .split import Fold
from .tracking import git_commit
from .trainset import INNER_FRAC, INNER_SEED, SAMPLE_SEED, inner_split, label_pairs, sample_s1

FORMAT = 1                      # bump when the on-disk layout changes
TUNE_POOLS = {"fold": "tune", "train": "tunedense"}  # tune_pool -> pipeline._tune_rule's tag
SIDES = ("fit", "stop", "tune", "val", "harder")
WHOLE_SIDES = ("fit", "stop")   # featured in one build_features call, as pipeline.fit does
MANIFEST = "manifest.json"
COMPRESSION = "zstd"
LABEL = "label"
PASS = "pass"
META_COLUMNS = [C.S1_ID, C.ENTITY_ID, C.COUNTRY, PASS, LABEL]
RANK1_COLUMN = "ctx_rank_name"  # 08 §5: reliability of the pairs the decision layer sees first
# modules whose code decides the snapshot's rows and values: their content is in the key
DATA_MODULES = ("config", "data", "split", "normalize", "token_maps", "blocking", "features",
                "trainset", "evaluate", "pipeline")

WeightFn = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


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
    so is ``normalise.chunk_rows``, which only bounds memory. ``tune_pool`` is hashed only
    when it is not the default "fold", so default-pool keys are computed as before.
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
    if cfg.tune_pool != "fold":
        parts["tune_pool"] = cfg.tune_pool
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
    Pool statistics (``STATS_GROUPS``) are counted once over the whole ``pooln`` and shared
    by every slice, as ``pipeline.score`` does, so they never depend on the chunking.
    """
    names = feature_names(cfg.feature_groups)
    schema = _schema(names)
    n = len(pairs)
    label = label_pairs(pairs[[C.S1_ID, C.ENTITY_ID]], truth)[LABEL].to_numpy(np.int8)
    country = _country_of(pairs[C.S1_ID], s1n)
    s1_ids, pool_ids = _large(pairs[C.S1_ID]), _large(pairs[C.ENTITY_ID])
    passes = pairs[PASS].to_numpy(np.uint8)
    whole = side in WHOLE_SIDES
    stats = pool_stats(pooln, cfg.feature_groups) if n else None
    X_all = (build_features(pairs, s1n, pooln, groups=cfg.feature_groups,
                            chunk_rows=cfg.chunk_rows, stats=stats) if whole else None)
    slices = ([slice(a, min(a + cfg.chunk_rows, n)) for a in range(0, n, cfg.chunk_rows)]
              if whole else iter_chunks(pairs, cfg.chunk_rows))
    path = folder / f"{side}.parquet"
    tmp = path.with_name(path.name + ".tmp")
    wrote = False
    with pq.ParquetWriter(tmp, schema, compression=COMPRESSION) as writer:
        for sl in slices:
            X = (X_all.iloc[sl] if whole else
                 build_features(pairs.iloc[sl], s1n, pooln, groups=cfg.feature_groups,
                                chunk_rows=cfg.chunk_rows, stats=stats))
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
    of ``pipeline.run_fold`` with the same arguments, plus its blocking report. Name
    frequencies (``frequency`` group) are counted over the whole fold, as ``run_fold`` does."""
    t0 = time.perf_counter()
    s1n = P.load_normalised("train", (1,), cfg, fold.s1[C.ENTITY_ID], token_map)
    pooln = P.load_normalised("train", (2, 3), cfg, P.pool_of(fold)[C.ENTITY_ID], token_map)
    s1n, pooln = P._with_frequencies(cfg, s1n, pooln)  # s1n is the whole fold here
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


# ------------------------------------------------------------------ build ----
def build_snapshot(cfg: P.PipelineConfig, train: Fold, val: Fold, out_dir: Path | None = None,
                   timings: dict | None = None, sides: Sequence[str] = SIDES) -> Path:
    """Build (or complete) the snapshot of ``cfg``'s data and return its folder.

    ``train`` and ``val`` are ``split.load_fold`` folds (ids-only frames are enough:
    ``columns=[]``). ``out_dir`` holds the snapshot folders (default
    ``cfg.dataset_dir / ".cache" / "m4"``); the folder is ``<out_dir>/<key>``. Sides already
    recorded in its manifest are kept, so an interrupted build resumes where it stopped and
    a second call returns at once. ``timings`` receives ``<side>_{load,blocking,features}_
    seconds``. Candidate pairs come from (and go to) the pipeline's own blocking cache.
    ``cfg.tune_pool`` picks the tune side's pool exactly as ``pipeline._tune_rule`` does.
    """
    unknown = [s for s in sides if s not in SIDES]
    if unknown:
        raise ValueError(f"unknown sides {unknown}; known: {list(SIDES)}")
    if cfg.tune_pool not in TUNE_POOLS:
        raise ValueError(f"tune_pool must be one of {list(TUNE_POOLS)}, got {cfg.tune_pool!r}")
    timings = {} if timings is None else timings
    t0 = time.perf_counter()
    P.normalise_split("train", cfg)
    token_map = P.learn_token_map(cfg, train)
    timings["normalise_seconds"] = round(time.perf_counter() - t0, 2)
    key, parts = snapshot_key(cfg, train, val, token_map)
    root = Path(out_dir) if out_dir is not None else cfg.dataset_dir / ".cache" / "m4"
    folder = root / key
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / MANIFEST
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("key") != key:
            raise ValueError(f"{manifest_path} belongs to snapshot {manifest.get('key')!r}")
    else:
        record = cfg.record()
        for k in ("model", "grid"):   # varied by evaluate_params, not part of the data
            record.pop(k, None)
        manifest = {"format": FORMAT, "key": key, "key_parts": parts,
                    "created": datetime.now().isoformat(timespec="seconds"),
                    "commit": git_commit(), "feature_names": feature_names(cfg.feature_groups),
                    "token_map_size": len(token_map), "config": record, "sides": {}}
        _write_json(folder / "token_map.json", token_map)
    todo = [s for s in SIDES if s in sides and s not in manifest["sides"]]
    inner: dict[str, tuple[pd.DataFrame, Fold]] = {}
    if any(s in ("fit", "stop", "tune") for s in todo):
        fit_fold, tune_fold = inner_split(train)
        tune_s1 = (tune_fold.s1 if cfg.n_tune_s1 is None
                   else sample_s1(tune_fold.s1, cfg.n_tune_s1))
        rule_fold = tune_fold if cfg.tune_pool == "fold" else Fold(  # as pipeline._tune_rule
            "tune", train.s1, train.s2, train.s3, tune_fold.pairs)
        inner = {"fit": (sample_s1(fit_fold.s1, cfg.n_fit_s1), fit_fold),
                 "stop": (sample_s1(tune_fold.s1, cfg.n_stop_s1), tune_fold),
                 "tune": (tune_s1, rule_fold)}
    info: dict = {}
    for side in todo:
        t_side = time.perf_counter()
        if side in inner:
            s1, fold = inner[side]
            # the pipeline's side name: its blocking cache tag and timing / info keys
            name = TUNE_POOLS[cfg.tune_pool] if side == "tune" else side
            s1n, pooln, pairs = P._side(name, s1, fold, cfg, token_map, info, timings)
            info[f"{side}_blocking"] = info.get(f"{name}_blocking")
        else:
            fold = val if side == "val" else harder_fold(val)
            s1 = fold.s1
            s1n, pooln, pairs = _fold_records(fold, side, cfg, token_map, info, timings)
        t0 = time.perf_counter()
        stats = _write_side(folder, side, pairs, s1n, pooln, fold.pairs, cfg)
        timings[f"{side}_features_seconds"] = round(time.perf_counter() - t0, 2)
        stats["s1"] = _write_ids(folder, side, s1, s1n, fold.pairs)
        if side == "val":
            _write_val_extras(folder, fold, s1n, pooln)
        del s1n, pooln, pairs
        stats.update(blocking=info.get(f"{side}_blocking"), bytes=_files_bytes(folder, side),
                     seconds=round(time.perf_counter() - t_side, 2))
        manifest["sides"][side] = stats
        manifest["bytes_total"] = sum(s["bytes"] for s in manifest["sides"].values())
        manifest["peak_rss_gb"] = max(manifest.get("peak_rss_gb", 0.0),
                                      P.mem_guard(f"snapshot {side}"), P.peak_rss_gb())
        _write_json(manifest_path, manifest)   # after each side: a crash keeps finished sides
    return folder


# ------------------------------------------------------------------- load ----
@dataclass
class Snapshot:
    """A built snapshot: its folder, manifest, the sides in use and the feature columns in use.

    Nothing heavy is held; every accessor reads Parquet on demand. ``columns`` is the
    feature subset a matcher trains on (all features by default, in manifest order).
    """

    path: Path
    manifest: dict
    sides: tuple[str, ...]
    columns: list[str]

    @property
    def key(self) -> str:
        """The snapshot key (folder name)."""
        return self.manifest["key"]

    def _file(self, side: str, suffix: str = "") -> Path:
        """Path of one of ``side``'s files; refuses sides not loaded."""
        if side not in self.sides:
            raise KeyError(f"side {side!r} is not loaded (loaded: {list(self.sides)})")
        return self.path / f"{side}{suffix}.parquet"

    def rows(self, side: str) -> int:
        """Candidate pairs of ``side``."""
        return int(self.manifest["sides"][side]["rows"])

    def meta(self, side: str, columns: Sequence[str] = META_COLUMNS) -> pd.DataFrame:
        """Pair ids, S1 country, pass bits and label of every row of ``side``, in row order."""
        return pq.read_table(self._file(side), columns=list(columns)).to_pandas()

    def labels(self, side: str) -> np.ndarray:
        """0/1 labels of ``side`` (int8, row order)."""
        return self.column(side, LABEL).astype(np.int8)

    def column(self, side: str, name: str) -> np.ndarray:
        """One numeric column of ``side`` (a feature, ``label`` or ``pass``) as numpy."""
        return pq.read_table(self._file(side), columns=[name]).column(0).to_numpy()

    def iter_features(self, side: str, batch_rows: int = 1_000_000,
                      columns: Sequence[str] | None = None
                      ) -> Iterator[tuple[slice, pd.DataFrame]]:
        """``(row slice, float32 feature frame)`` batches of ``side``, in row order."""
        cols = list(self.columns if columns is None else columns)
        at = 0
        with pq.ParquetFile(self._file(side)) as pf:
            for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
                m = batch.num_rows
                block = np.empty((m, len(cols)), dtype=np.float32)
                for j, c in enumerate(cols):   # by name: batches keep the file's column order
                    block[:, j] = batch.column(c).to_numpy(zero_copy_only=False)
                yield slice(at, at + m), pd.DataFrame(block, columns=cols,
                                                      index=pd.RangeIndex(at, at + m), copy=False)
                at += m

    def features(self, side: str, columns: Sequence[str] | None = None,
                 batch_rows: int = 1_000_000) -> pd.DataFrame:
        """The whole float32 feature frame of ``side`` (RangeIndex), filled batch by batch into
        one preallocated matrix, so the peak is the matrix plus one batch."""
        cols = list(self.columns if columns is None else columns)
        out = np.empty((self.rows(side), len(cols)), dtype=np.float32)
        for sl, X in self.iter_features(side, batch_rows, cols):
            out[sl] = X.to_numpy()
        return pd.DataFrame(out, columns=cols, copy=False)

    def s1(self, side: str) -> pd.DataFrame:
        """Every S1 entity of ``side`` (``entity_id``, ``country``), candidates or not."""
        return pd.read_parquet(self._file(side, "_s1"))

    def truth(self, side: str) -> pd.DataFrame:
        """True pairs (``source1_entity_id``, ``entity_id``) of ``side``'s S1 entities."""
        return pd.read_parquet(self._file(side, "_truth"))

    def fold(self, side: str = "val") -> Fold:
        """The labelled fold ``score_pairs`` needs: ``val``, or ``harder_fold`` of it."""
        if side not in ("val", "harder"):
            raise ValueError(f"only val and harder have a fold, got {side!r}")
        self._file(side)   # refuses a side that is not loaded
        pool = pd.read_parquet(self.path / "val_pool.parquet")
        src = pool["source"].to_numpy()
        val = Fold("val", pd.read_parquet(self.path / "val_s1.parquet"),
                   pool.loc[src == 2, [C.ENTITY_ID]].reset_index(drop=True),
                   pool.loc[src == 3, [C.ENTITY_ID]].reset_index(drop=True),
                   pd.read_parquet(self.path / "val_truth.parquet"))
        return val if side == "val" else harder_fold(val)

    def normalised(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(val S1 rows, pool rows of val's true pairs), normalised: ``slice_report`` input."""
        return (pd.read_parquet(self.path / "val_s1n.parquet"),
                pd.read_parquet(self.path / "val_pooln.parquet"))

    def blocking(self, side: str) -> dict:
        """``evaluate.blocking_report`` of ``side``'s candidates, as computed at build time."""
        return self.manifest["sides"][side]["blocking"]

    def token_map(self) -> dict[str, str]:
        """The learned token map the snapshot was built with."""
        return json.loads((self.path / "token_map.json").read_text(encoding="utf-8"))

    def to_fitted(self, cfg: P.PipelineConfig, matcher: Matcher, rule: DecisionRule,
                  tune_table: pd.DataFrame) -> P.Fitted:
        """A ``pipeline.Fitted`` for ``run_test`` (shortlisted versions, template §9).

        ``cfg`` must describe the snapshot's data (blocking, feature groups) and the matcher
        must use every feature, since ``pipeline.score`` builds all of them.
        """
        parts = self.manifest["key_parts"]
        if cfg.blocking.key() != parts["blocking"] or \
                list(cfg.feature_groups) != parts["feature_groups"]:
            raise ValueError("cfg's blocking or feature groups differ from the snapshot's")
        if list(matcher.feature_names_) != self.manifest["feature_names"]:
            raise ValueError("run_test needs a matcher trained on every snapshot feature")
        return P.Fitted(matcher, rule, tune_table, cfg, self.token_map(),
                        {"snapshot": self.key})


def load_snapshot(path: Path, sides: Sequence[str] | None = None,
                  columns: Sequence[str] | None = None) -> Snapshot:
    """Open a built snapshot.

    ``sides``: the sides to use, all of which must be built (None = every built side, so
    ``harder`` is included when it exists; ``("fit", "stop", "tune", "val")`` skips it).
    ``columns``: an optional feature subset (feature-ablation experiments), in that order.
    """
    path = Path(path)
    manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{path}: snapshot format {manifest.get('format')}, expected {FORMAT}")
    sides = [s for s in SIDES if s in manifest["sides"]] if sides is None else list(sides)
    missing = [s for s in sides if s not in manifest["sides"]]
    if missing:
        raise ValueError(f"{path}: sides {missing} were not built "
                         f"(built: {list(manifest['sides'])})")
    names = manifest["feature_names"]
    cols = list(names if columns is None else columns)
    if unknown := [c for c in cols if c not in names]:
        raise ValueError(f"unknown feature columns {unknown}")
    if len(set(cols)) != len(cols):
        raise ValueError("feature columns repeat")
    return Snapshot(path, manifest, tuple(sides), cols)


# --------------------------------------------------------------- evaluate ----
def _score_side(snap: Snapshot, side: str, model: Matcher | SeedEnsemble,
                batch_rows: int) -> tuple[pd.DataFrame, np.ndarray]:
    """``(scored, labels)`` of ``side``: SCORED_COLUMNS in the pipeline's pair order."""
    meta = snap.meta(side, [C.S1_ID, C.ENTITY_ID, LABEL])
    prob = np.empty(len(meta), dtype=np.float32)
    for sl, X in snap.iter_features(side, batch_rows):
        prob[sl] = model.predict_proba(X)
    scored = pd.DataFrame({C.S1_ID: meta[C.S1_ID], C.ENTITY_ID: meta[C.ENTITY_ID],
                           "prob": prob})
    return scored, meta[LABEL].to_numpy(np.int8)


def error_counts(matches: pd.DataFrame, fold: Fold) -> dict[str, int]:
    """Pair counts of the four ``evaluate.error_samples`` kinds, from entity counts.

    Equal to ``len(error_samples(matches, fold, kind, n=10**9))`` (v001's numbers) without
    building the sample frames: false_merge / singleton_merge are the wrong predicted pairs
    of matched / singleton entities, missed the unpredicted true pairs of entities that
    predicted something, false_singleton those of entities that predicted nothing.
    """
    tp, n_pred, n_true = entity_counts(matches, fold)
    fp, fn, matched, predicted = n_pred - tp, n_true - tp, n_true > 0, n_pred > 0
    return {"false_merge": int(fp[matched].sum()), "missed": int(fn[predicted].sum()),
            "false_singleton": int(fn[~predicted].sum()),
            "singleton_merge": int(fp[~matched].sum())}


def _by_country(matches: pd.DataFrame, fold: Fold) -> dict[str, float]:
    """Val F0.5 per S1 country (CLAUDE.md §8 extra): ``score_pairs`` on each country's S1."""
    out = {}
    country = fold.s1[C.COUNTRY]
    for c in sorted(country.unique()):
        s1 = fold.s1[(country == c).to_numpy()].reset_index(drop=True)
        pairs = fold.pairs[isin(fold.pairs[C.S1_ID], pd.Index(s1[C.ENTITY_ID]))]
        out[str(c)] = score_pairs(matches, Fold(fold.name, s1, fold.s2, fold.s3, pairs))["f_beta"]
    return out


def evaluate_params(snap: Snapshot, params: MatcherParams | None = None, *,
                    weight_fn: WeightFn | None = None, grid: Grid | None = None,
                    calibration: bool = True, seeds: Sequence[int] | None = None,
                    harder: bool = True, batch_rows: int = 1_000_000,
                    artifacts: dict | None = None
                    ) -> tuple[dict, Matcher | SeedEnsemble, DecisionRule]:
    """Train on fit/stop, tune the rule on tune, score val (and harder) with it frozen.

    The same calls as ``pipeline.fit`` + ``run_fold``: ``Matcher(params).fit(X_fit, y_fit,
    X_stop, y_stop)``; ``decision.tune(scored_tune, tune S1 ids, tune truth, grid)``;
    ``pipeline.decide_by_country`` and ``evaluate.score_pairs`` on val. ``weight_fn(meta,
    X_fit)`` returns one weight >= 0 per fit row (``meta`` = ``Snapshot.meta("fit")``: ids,
    country, pass, label), for hard-negative experiments. ``seeds`` trains one model per
    seed and averages their probabilities. ``grid`` defaults to ``decision.DEFAULT_GRID``
    (pass ``cfg.grid`` to match a pipeline run with another grid). ``harder`` scores the
    harder side when the snapshot has it (``harder_f_beta``).

    Returns ``(metrics, matcher, rule)``: ``metrics`` holds the 13 §2.2 keys this stage
    produces (val scores, blocking, ``tune_f_beta``, ``harder_f_beta``, errors, timings,
    ``peak_rss_gb``) plus ``best_iteration``, ``tune_logloss``, ``tune_auc`` (stop set, as
    ``Matcher.fit_info_``), ``importance_top20``, ``f_beta_by_country`` and, with
    ``calibration``, ``ece_tune``, ``brier_tune``, ``ece_tune_rank1``, ``ece_val``,
    ``brier_val``. A dict passed as ``artifacts`` receives ``tune_table``, ``reliability``,
    ``importance``, ``val_scored``, ``val_matches`` and ``slices`` (``slice_report``).
    """
    params = MatcherParams() if params is None else params
    grid = DEFAULT_GRID if grid is None else grid
    timings: dict[str, float] = {}
    P.mem_guard("evaluate start")

    t0 = time.perf_counter()
    X_fit, y_fit = snap.features("fit", batch_rows=batch_rows), snap.labels("fit")
    X_stop, y_stop = snap.features("stop", batch_rows=batch_rows), snap.labels("stop")
    weight = None
    if weight_fn is not None:
        weight = np.asarray(weight_fn(snap.meta("fit"), X_fit), dtype=np.float64)
        if weight.shape != (len(X_fit),):
            raise ValueError(f"weight_fn returned shape {weight.shape}, expected "
                             f"({len(X_fit)},)")
    timings["load_seconds"] = round(time.perf_counter() - t0, 2)
    P.mem_guard("evaluate load")

    t0 = time.perf_counter()
    model = fit_matcher(params, X_fit, y_fit, X_stop, y_stop, weight, seeds)
    timings["fit_seconds"] = round(time.perf_counter() - t0, 2)
    del X_fit, y_fit, X_stop, y_stop, weight
    P.mem_guard("evaluate fit")

    t0 = time.perf_counter()
    scored_tune, y_tune = _score_side(snap, "tune", model, batch_rows)
    score_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    rule, table = tune(scored_tune, snap.s1("tune")[C.ENTITY_ID], snap.truth("tune"), grid)
    timings["tune_seconds"] = round(time.perf_counter() - t0, 2)
    calib: dict[str, float] = {}
    rel_tables = []
    if calibration:
        ece, brier, rel = reliability(scored_tune["prob"].to_numpy(), y_tune)
        calib.update(ece_tune=ece, brier_tune=brier)
        rel_tables.append(rel.assign(side="tune", subset="all"))
        if RANK1_COLUMN in snap.manifest["feature_names"]:
            top = snap.column("tune", RANK1_COLUMN) == 1
            ece1, _, rel1 = reliability(scored_tune["prob"].to_numpy()[top], y_tune[top])
            calib["ece_tune_rank1"] = ece1
            rel_tables.append(rel1.assign(side="tune", subset="rank1"))
    del scored_tune, y_tune
    P.mem_guard("evaluate tune")

    t0 = time.perf_counter()
    scored_val, y_val = _score_side(snap, "val", model, batch_rows)
    score_seconds += time.perf_counter() - t0
    val = snap.fold("val")
    t0 = time.perf_counter()
    matches = P.decide_by_country(scored_val, val.s1, rule)
    decide_seconds = time.perf_counter() - t0
    if calibration:
        ece, brier, rel = reliability(scored_val["prob"].to_numpy(), y_val)
        calib.update(ece_val=ece, brier_val=brier)
        rel_tables.append(rel.assign(side="val", subset="all"))
    del y_val
    blocking = snap.blocking("val") or {}
    errors = error_counts(matches, val)
    metrics: dict = {
        "snapshot": snap.key, "model_params": asdict(params),
        "seeds": [int(s) for s in seeds] if seeds else None, "weighted": weight_fn is not None,
        "n_features": len(snap.columns), "rule": asdict(rule),
        **score_pairs(matches, val),
        "cand_recall": blocking.get("pair_recall"),
        "entity_recall": blocking.get("entity_recall"),
        "ceiling_f_beta": blocking.get("ceiling_f_beta"),
        "cands_mean": blocking.get("candidates_mean"),
        "cands_p95": blocking.get("candidates_p95"),
        "tune_f_beta": float(table["f_beta"].max()),
        "f_beta_by_country": _by_country(matches, val),
        "n_fp": errors["false_merge"] + errors["singleton_merge"], "n_fn": errors["missed"],
        "n_false_singleton": errors["false_singleton"], "errors": errors,
    }
    P.mem_guard("evaluate val")

    if harder and "harder" in snap.sides:
        hard = snap.fold("harder")
        t0 = time.perf_counter()
        scored_h, _ = _score_side(snap, "harder", model, batch_rows)
        score_seconds += time.perf_counter() - t0
        t0 = time.perf_counter()
        matches_h = P.decide_by_country(scored_h, hard.s1, rule)
        decide_seconds += time.perf_counter() - t0
        metrics["harder_f_beta"] = (score_pairs(matches_h, hard)["f_beta"] if len(hard.s1)
                                    else float("nan"))
        del scored_h, matches_h
        P.mem_guard("evaluate harder")

    importance = model.importance()
    info = model.fit_info_
    metrics.update(best_iteration=int(model.best_iteration_),
                   tune_logloss=info.get("tune_logloss"), tune_auc=info.get("tune_auc"),
                   importance_top20={k: float(v) for k, v in importance.head(20).items()},
                   **calib, **timings, score_seconds=round(score_seconds, 2),
                   decide_seconds=round(decide_seconds, 2), peak_rss_gb=P.peak_rss_gb())
    if artifacts is not None:
        artifacts.update(tune_table=table, importance=importance, val_scored=scored_val,
                         val_matches=matches,
                         reliability=(pd.concat(rel_tables, ignore_index=True)
                                      if rel_tables else pd.DataFrame()))
        if (snap.path / "val_s1n.parquet").exists() and len(val.s1):
            s1n, pooln = snap.normalised()
            artifacts["slices"] = slice_report(matches, val, s1n, pooln)
    return metrics, model, rule
