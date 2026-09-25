"""End-to-end pipeline (02 §6): normalise -> block -> features -> matcher -> decision -> files.

Notebooks call these functions and never re-implement a stage:

    fitted = fit(cfg, train)                               # train fold only
    metrics, pairs, scored, matches = run_fold(cfg, fitted, val)   # fixed val fold, once
    run_test(cfg, fitted)                                  # the two submission files

What each side of the data is used for (11 §2):

    fit side of the inner split   S1 sample (whole fit pool kept) -> LightGBM training pairs
    tune side                     S1 sample -> early stopping; ALL tune S1 -> decision rule
                                  (the full fold keeps the pool-side 1-to-1 competition real)
    train fold pairs              learned transliteration token map (05 §6)
    val fold                      scored once with the frozen rule

Normalised records are computed once per split (static rules) and cached as Parquet
(``<cache_dir>/norm/``); a fold is an id subset of that cache, filtered in Arrow. The
learned token map is applied on load (only non-Latin rows change). Candidate pairs are
cached per tag, country and blocking-config hash (``blocking.block``), so a new model
version reuses the candidates of an unchanged blocking configuration. Scoring builds
features chunk by chunk and keeps only the scored pairs.
"""
from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import config as C
from .blocking import BlockingConfig, block
from .data import load_source
from .decision import SCORED_COLUMNS, DecisionRule, Grid, decide
from .features import DEFAULT_GROUPS, build_features, iter_chunks
from .model import Matcher, MatcherParams
from .normalize import (
    RULES_VERSION,
    NormaliseConfig,
    apply_token_map,
    fit_token_map,
    normalise_records,
)
from .split import Fold

TIMING_LABELS = ("load", "normalise", "blocking", "features", "fit", "tune", "score", "decide")


@dataclass
class PipelineConfig:
    """Everything that defines a pipeline version (logged as the version's configuration)."""

    normalise: NormaliseConfig = field(default_factory=NormaliseConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    feature_groups: tuple[str, ...] = DEFAULT_GROUPS
    model: MatcherParams = field(default_factory=MatcherParams)
    grid: Grid = field(default_factory=Grid)
    n_fit_s1: int = 200_000          # S1 entities sampled from the fit side (whole fit pool kept)
    n_stop_s1: int = 50_000          # tune-side S1 sample used for LightGBM early stopping
    n_tune_s1: int | None = None     # tune-side S1 entities for the rule grid; None = all
    chunk_rows: int = 1_000_000      # pairs featured and scored at a time
    dataset_dir: Path = C.DATASET
    cache_dir: Path = C.DATASET / ".cache" / "pipeline"

    def record(self) -> dict:
        """JSON-ready dict of the configuration (metrics.json, artifacts/config.json)."""
        return json.loads(json.dumps(asdict(self), default=str))


@dataclass
class Fitted:
    """A trained pipeline version: token map, matcher, frozen rule and the grid behind it."""

    matcher: Matcher
    rule: DecisionRule
    tune_table: pd.DataFrame
    config: PipelineConfig
    token_map: dict[str, str] = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def save(self, out: Path) -> Path:
        """Write model/, rule.json, tune_table.csv, token_map.json, config and fit info."""
        out.mkdir(parents=True, exist_ok=True)
        self.matcher.save(out / "model")
        (out / "rule.json").write_text(json.dumps(
            {**asdict(self.rule), "tune_f_beta": float(self.tune_table["f_beta"].max())},
            indent=2) + "\n")
        self.tune_table.to_csv(out / "tune_table.csv", index=False)
        (out / "token_map.json").write_text(json.dumps(self.token_map, sort_keys=True) + "\n")
        (out / "config.json").write_text(json.dumps(self.config.record(), indent=2) + "\n")
        (out / "fit_info.json").write_text(json.dumps(self.info, indent=2, default=float) + "\n")
        return out

    @classmethod
    def load(cls, out: Path, config: PipelineConfig) -> Fitted:
        """Read back what ``save`` wrote; ``config`` must be the version's configuration."""
        rule = json.loads((out / "rule.json").read_text())
        rule.pop("tune_f_beta", None)
        info_path = out / "fit_info.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        return cls(Matcher.load(out / "model"), DecisionRule(**rule),
                   pd.read_csv(out / "tune_table.csv"), config,
                   json.loads((out / "token_map.json").read_text()), info)


# ---------------------------------------------------------------- helpers ----
_PEAK_RSS = [0.0]


def mem_guard(stage: str = "", limit_gb: float = 13.0) -> float:
    """Current RSS in GB; remembers the peak and raises past ``limit_gb`` (02 §7)."""
    gc.collect()
    rss = psutil.Process().memory_info().rss / 1e9
    _PEAK_RSS[0] = max(_PEAK_RSS[0], rss)
    if rss > limit_gb:
        raise MemoryError(f"RSS {rss:.1f} GB > {limit_gb} GB after {stage}")
    return rss


def peak_rss_gb() -> float:
    """Largest RSS seen by ``mem_guard`` in this process."""
    return round(_PEAK_RSS[0], 2)


def _hash(obj) -> str:
    """Short stable hash of a JSON-able object (cache keys)."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:8]


def _ids_key(ids: pd.Series) -> str:
    """Order-free fingerprint of a set of entity ids (cache tags for samples)."""
    h = pd.util.hash_pandas_object(ids.astype("str"), index=False).to_numpy()
    return f"{len(ids)}_{int(h.sum(dtype=np.uint64)) & 0xFFFFFFFF:08x}"


def pool_of(fold: Fold) -> pd.DataFrame:
    """Source 2 and 3 records of a fold as one frame."""
    return pd.concat([fold.s2, fold.s3], ignore_index=True)


def _static_key(cfg: NormaliseConfig) -> str:
    """Cache key of the static normalisation (the learned-map switches are not part of it)."""
    d = asdict(cfg)
    for k in ("learn_token_map", "token_map_min_count", "token_map_min_share", "chunk_rows"):
        d.pop(k, None)
    if RULES_VERSION != 1:  # version 1 keeps the original key (no cache rebuild)
        d["rules_version"] = RULES_VERSION
    return _hash(d)


# ------------------------------------------------------------- normalise ----
def normalise_split(split: str, cfg: PipelineConfig) -> dict[int, Path]:
    """Static normalisation of every record of a split, computed once: source -> Parquet."""
    folder = cfg.cache_dir / "norm"
    key = _static_key(cfg.normalise)
    paths = {s: folder / f"{split}_source{s}_{key}.parquet" for s in C.SOURCES}
    for s, path in paths.items():
        if path.exists():
            continue
        raw = load_source(split, s, cfg.dataset_dir)
        norm = normalise_records(raw, cfg.normalise)
        del raw
        folder.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        norm.to_parquet(tmp, index=False)
        tmp.replace(path)
        del norm
        mem_guard(f"normalise {split} source {s}")
    return paths


def load_normalised(split: str, sources: tuple[int, ...], cfg: PipelineConfig,
                    ids: pd.Series | None = None,
                    token_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Normalised records of ``sources`` in ``split``, optionally only ``ids``, map applied.

    The id filter runs in Arrow before conversion, so only the subset reaches pandas.
    """
    paths = normalise_split(split, cfg)
    frames = []
    for s in sources:
        tbl = pq.read_table(paths[s])
        if ids is not None:
            value_set = pa.array(pd.Index(ids).astype("str"), type=tbl.schema.field(
                C.ENTITY_ID).type)
            tbl = tbl.filter(pc.is_in(tbl[C.ENTITY_ID], value_set=value_set))
        frames.append(tbl.to_pandas())
        del tbl
    out = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if token_map:
        out = apply_token_map(out, token_map, cfg.normalise)
    return out


def learn_token_map(cfg: PipelineConfig, train: Fold) -> dict[str, str]:
    """Transliteration token map fitted on the train fold's true pairs (cached as JSON)."""
    if not cfg.normalise.learn_token_map:
        return {}
    key = _hash([_static_key(cfg.normalise), cfg.normalise.token_map_min_count,
                 cfg.normalise.token_map_min_share, _ids_key(train.pairs[C.ENTITY_ID])])
    path = cfg.cache_dir / f"token_map_{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    pool_ids = train.pairs[C.ENTITY_ID]
    s1n = load_normalised("train", (1,), cfg, train.pairs[C.S1_ID].drop_duplicates())
    pooln = load_normalised("train", (2, 3), cfg, pool_ids)
    tmap = fit_token_map(train.pairs, s1n, pooln, cfg.normalise.token_map_min_count,
                         cfg.normalise.token_map_min_share)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tmap, sort_keys=True))
    return tmap


def prepare(s1n: pd.DataFrame, pooln: pd.DataFrame, cfg: PipelineConfig, tag: str,
            timings: dict | None = None) -> pd.DataFrame:
    """Candidate pairs for normalised S1 and pool records, cached under ``tag``."""
    t0 = time.perf_counter()
    pairs = block(s1n, pooln, cfg.blocking, cache_dir=cfg.cache_dir / "pairs", tag=tag)
    if timings is not None:
        timings["blocking_seconds"] = round(time.perf_counter() - t0, 2)
    mem_guard(f"blocking {tag}")
    return pairs


def _tag(name: str, s1n: pd.DataFrame, pooln: pd.DataFrame, token_map: dict) -> str:
    """Cache tag of a candidate set: which S1 and pool records, which learned map."""
    return (f"{name}_{_ids_key(s1n[C.ENTITY_ID])}_{_ids_key(pooln[C.ENTITY_ID])}"
            f"_m{_hash(token_map)}")


# ---------------------------------------------------------------- scoring ----
def score(pairs: pd.DataFrame, s1n: pd.DataFrame, pooln: pd.DataFrame, matcher: Matcher,
          cfg: PipelineConfig) -> pd.DataFrame:
    """SCORED_COLUMNS for every candidate pair, featured and predicted chunk by chunk."""
    prob = np.empty(len(pairs), dtype=np.float32)
    for sl in iter_chunks(pairs, cfg.chunk_rows):
        X = build_features(pairs.iloc[sl], s1n, pooln, groups=cfg.feature_groups,
                           chunk_rows=cfg.chunk_rows)
        prob[sl] = matcher.predict_proba(X)
        del X
    scored = pairs[[C.S1_ID, C.ENTITY_ID]].copy()
    scored["prob"] = prob
    return scored[SCORED_COLUMNS]


def decide_by_country(scored: pd.DataFrame, s1n: pd.DataFrame,
                      rule: DecisionRule) -> pd.DataFrame:
    """``decide`` per country partition (pool ids never cross countries, 10 §13)."""
    country = scored[C.S1_ID].map(s1n.set_index(C.ENTITY_ID)[C.COUNTRY])
    parts = [decide(scored[(country == c).to_numpy()], rule) for c in country.unique()]
    return pd.concat(parts, ignore_index=True) if parts else decide(scored, rule)


def sort_matches(matches: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    """Matches ordered by S1 id, then by probability (best first), with ``prob``."""
    m = matches.merge(scored, on=[C.S1_ID, C.ENTITY_ID], how="left")
    m = m.sort_values([C.S1_ID, "prob"], ascending=[True, False], kind="stable")
    return m.reset_index(drop=True)
