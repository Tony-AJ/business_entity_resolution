"""End-to-end pipeline (02 §6): normalise -> block -> features -> matcher -> decision -> files.

Notebooks call these functions and never re-implement a stage:

    fitted = fit(cfg, train)                               # train fold only
    metrics, pairs, scored, matches = run_fold(cfg, fitted, val)   # fixed val fold, once
    run_test(cfg, fitted)                                  # the two submission files

What each side of the data is used for (11 §2):

    fit side of the inner split   S1 sample (whole fit pool kept) -> LightGBM training pairs
    tune side                     S1 sample -> early stopping; ALL tune S1 -> decision rule
                                  (the full fold keeps the pool-side 1-to-1 competition real)
    train fold pairs              learned transliteration token map (05 §6); learned filler
                                  tokens (opt-in, NormaliseConfig.learn_fillers)
    val fold                      scored once with the frozen rule

Normalised records are computed once per split (static rules) and cached as Parquet
(``<cache_dir>/norm/``); a fold is an id subset of that cache, filtered in Arrow. The
learned token map is applied on load (only non-Latin rows change), and so are the learned
fillers: a version that learns them loads ``name_core_nofill`` too, which the ``nofill``
feature group and the ``BlockingConfig.nofill_max_group`` pass read. Everything on:

    PipelineConfig(normalise=NormaliseConfig(learn_fillers=True),
                   blocking=BlockingConfig(nofill_max_group=50),
                   feature_groups=(*groups, "nofill"))

The learned token evidence (``evidence.py``, opt-in ``PipelineConfig.evidence.learn``) is
fitted next to them and handed to ``build_features(evidence=)`` for the ``tok_evidence``
group: ``evidence=EvidenceConfig(learn=True)``, ``feature_groups=(..., "tok_evidence")``.

Candidate pairs are cached per tag, country and blocking-config hash (``blocking.block``),
so a new model version reuses the candidates of an unchanged blocking configuration.
Scoring builds features chunk by chunk and keeps only the scored pairs.
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
from .blocking import BlockingConfig, TopKSpec, block
from .data import isin, load_source
from .decision import (
    SCORED_COLUMNS,
    DecisionRule,
    ExpectedRule,
    Grid,
    apply_rule,
    decide,
    one_to_one_filter,
    tune,
)
from .evaluate import blocking_report, entity_counts, entity_tight_from_counts, score_pairs
from .evidence import EvidenceConfig, TokenEvidence, evidence_table, near_duplicates
from .features import DEFAULT_GROUPS, build_features, iter_chunks, pool_stats
from .mock import FP_WEIGHT, PUBLIC_OFFSET, MockFold
from .model import Matcher, MatcherParams, SeedMean
from .normalize import (
    RULES_VERSION,
    NormaliseConfig,
    add_nofill,
    apply_token_map,
    fit_fillers,
    fit_token_map,
    normalise_records,
)
from .split import Fold, hash_unit
from .submission import write_pairs
from .trainset import inner_split, label_pairs, sample_s1

TIMING_LABELS = ("load", "normalise", "blocking", "features", "fit", "tune", "score", "decide")
FILLER_PAIRS = 1_000_000  # true pairs learn_fillers counts at most (whole S1, by id hash)
FILLER_SEED = 5151        # hash seed of that sample


@dataclass
class PipelineConfig:
    """Everything that defines a pipeline version (logged as the version's configuration)."""

    normalise: NormaliseConfig = field(default_factory=NormaliseConfig)
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    feature_groups: tuple[str, ...] = DEFAULT_GROUPS
    # learned token evidence of one-sided name words (the tok_evidence group reads it)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    model: MatcherParams = field(default_factory=MatcherParams)
    grid: Grid = field(default_factory=Grid)
    n_fit_s1: int = 200_000          # S1 entities sampled from the fit side (whole fit pool kept)
    n_stop_s1: int = 50_000          # tune-side S1 sample used for LightGBM early stopping
    n_tune_s1: int | None = None     # tune-side S1 entities for the rule grid; None = all
    # pool the tune-side S1 are blocked against for the rule grid: "fold" = the tune fold's
    # own pool (val-like density); "train" = the whole train-fold pool, ~the test pool's
    # size, so the rule meets as many same-name decoys per entity as at test time
    tune_pool: str = "fold"
    chunk_rows: int = 1_000_000      # pairs featured and scored at a time
    dataset_dir: Path = C.DATASET
    cache_dir: Path = C.DATASET / ".cache" / "pipeline"

    def record(self) -> dict:
        """JSON-ready dict of the configuration (metrics.json, artifacts/config.json)."""
        return json.loads(json.dumps(asdict(self), default=str))

    @classmethod
    def from_record(cls, d: dict) -> PipelineConfig:
        """The configuration ``record`` wrote (``artifacts/config.json``), rebuilt exactly."""
        def spec(s: dict | None) -> TopKSpec | None:
            return None if s is None else TopKSpec(**{**s, "ngram": tuple(s["ngram"])})

        b = d["blocking"]
        blocking = BlockingConfig(**{**b, "exact_keys": tuple(b["exact_keys"]),
                                     **{k: spec(b[k]) for k in ("name_char", "name_addr_word",
                                                                "addr_char")}})
        g = d["grid"]
        grid = Grid(**{k: tuple(v) if isinstance(v, list) else v for k, v in g.items()})
        return cls(normalise=NormaliseConfig(**d["normalise"]), blocking=blocking,
                   feature_groups=tuple(d["feature_groups"]),
                   evidence=EvidenceConfig(**d.get("evidence", {})),
                   model=MatcherParams(**d["model"]),
                   grid=grid, **{k: d[k] for k in ("n_fit_s1", "n_stop_s1", "n_tune_s1",
                                                   "tune_pool", "chunk_rows")},
                   dataset_dir=Path(d["dataset_dir"]), cache_dir=Path(d["cache_dir"]))


@dataclass
class Fitted:
    """A trained pipeline version: token map, matcher, frozen rule and the grid behind it."""

    matcher: Matcher | SeedMean
    rule: DecisionRule
    tune_table: pd.DataFrame
    config: PipelineConfig
    token_map: dict[str, str] = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    fillers: list[str] | None = None  # learned filler tokens; None: the version learns none
    token_evidence: TokenEvidence | None = None  # learned word log-odds; None: none learned

    def save(self, out: Path) -> Path:
        """Write model/, rule.json, tune_table.csv, token_map.json, config and fit info, and
        fillers.json for a version that learns fillers."""
        out.mkdir(parents=True, exist_ok=True)
        self.matcher.save(out / "model")
        (out / "rule.json").write_text(json.dumps(
            {**asdict(self.rule), "tune_f_beta": float(self.tune_table["f_beta"].max())},
            indent=2) + "\n")
        self.tune_table.to_csv(out / "tune_table.csv", index=False)
        (out / "token_map.json").write_text(json.dumps(self.token_map, sort_keys=True) + "\n")
        if self.fillers is not None:
            (out / "fillers.json").write_text(json.dumps(self.fillers) + "\n")
        if self.token_evidence is not None:
            (out / "token_evidence.json").write_text(json.dumps(self.token_evidence.record())
                                                     + "\n")
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
        fillers_path = out / "fillers.json"
        fillers = json.loads(fillers_path.read_text()) if fillers_path.exists() else None
        ev_path = out / "token_evidence.json"
        evidence = (TokenEvidence.from_record(json.loads(ev_path.read_text()))
                    if ev_path.exists() else None)
        model = out / "model"               # a bagged stage 1 holds one folder per bag
        matcher = SeedMean.load(model) if (model / "seed0").exists() else Matcher.load(model)
        return cls(matcher, DecisionRule(**rule),
                   pd.read_csv(out / "tune_table.csv"), config,
                   json.loads((out / "token_map.json").read_text()), info, fillers, evidence)


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
    for k in ("learn_token_map", "token_map_min_count", "token_map_min_share", "chunk_rows",
              "learn_fillers", "filler_min_share", "filler_min_ratio"):
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
                    token_map: dict[str, str] | None = None,
                    columns: list[str] | None = None,
                    country: str | None = None,
                    fillers: list[str] | None = None) -> pd.DataFrame:
    """Normalised records of ``sources`` in ``split``, optionally only ``ids``, map applied.

    The id and ``country`` filters run in Arrow before conversion, so only the subset
    reaches pandas; ``columns`` limits what is read (``entity_id`` always included).
    ``fillers`` (``learn_fillers``; None = off, the default) adds ``name_core_nofill`` after
    the token map, when name_core is loaded.
    """
    paths = normalise_split(split, cfg)
    cols = None if columns is None else list(dict.fromkeys([C.ENTITY_ID, *columns]))
    frames = []
    for s in sources:
        tbl = pq.read_table(paths[s], columns=cols)
        if country is not None:
            tbl = tbl.filter(pc.equal(tbl[C.COUNTRY], country))
        if ids is not None:
            value_set = pa.array(pd.Index(ids).astype("str"), type=tbl.schema.field(
                C.ENTITY_ID).type)
            tbl = tbl.filter(pc.is_in(tbl[C.ENTITY_ID], value_set=value_set))
        frames.append(tbl.to_pandas())
        del tbl
    out = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if token_map:
        out = apply_token_map(out, token_map, cfg.normalise)
    if fillers is not None and "name_core" in out.columns:
        out = add_nofill(out, fillers)
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
    cols = ["name_norm", "non_latin"]  # all the alignment reads: keeps this step ~0.5 GB
    s1n = load_normalised("train", (1,), cfg, train.pairs[C.S1_ID].drop_duplicates(),
                          columns=cols)
    pooln = load_normalised("train", (2, 3), cfg, train.pairs[C.ENTITY_ID], columns=cols)
    tmap = fit_token_map(train.pairs, s1n, pooln, cfg.normalise.token_map_min_count,
                         cfg.normalise.token_map_min_share)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tmap, sort_keys=True))
    return tmap


def learn_fillers(cfg: PipelineConfig, train: Fold) -> list[str] | None:
    """Filler tokens fitted on the train fold's true pairs (cached as JSON); None when off.

    ``fit_fillers`` thresholds shares of pairs, so at most ``FILLER_PAIRS`` pairs (whole S1
    entities, by id hash) give the fold's list at a fraction of the memory.
    """
    n = cfg.normalise
    if not n.learn_fillers:
        return None
    key = _hash([_static_key(n), n.filler_min_share, n.filler_min_ratio, FILLER_PAIRS,
                 _ids_key(train.pairs[C.ENTITY_ID])])
    path = cfg.cache_dir / f"fillers_{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    pairs = train.pairs
    if len(pairs) > FILLER_PAIRS:
        pairs = pairs[hash_unit(pairs[C.S1_ID], FILLER_SEED) < FILLER_PAIRS / len(pairs)]
    s1n = load_normalised("train", (1,), cfg, pairs[C.S1_ID].drop_duplicates(),
                          columns=["name_core"])
    pooln = load_normalised("train", (2, 3), cfg, pairs[C.ENTITY_ID],
                            columns=["name_core", "non_latin"])
    fillers = fit_fillers(pairs, s1n, pooln, n.filler_min_share, n.filler_min_ratio)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fillers))
    return fillers


def learn_token_evidence(cfg: PipelineConfig, train: Fold) -> TokenEvidence | None:
    """Token evidence fitted on the train fold (``evidence.py``; cached as JSON); None when off.

    Reads country and name_sorted of every S1 and pool record of the fold, one country at a
    time (the near-duplicate pairs need the country's whole pool, and a name is unique only
    among all its S1 records); the pairs of every country make one table.
    """
    ev = cfg.evidence
    if not ev.learn:
        return None
    key = _hash([_static_key(cfg.normalise), asdict(ev), _ids_key(train.s1[C.ENTITY_ID]),
                 _ids_key(train.pairs[C.ENTITY_ID])])
    path = cfg.cache_dir / f"token_evidence_{key}.json"
    if path.exists():
        return TokenEvidence.from_record(json.loads(path.read_text()))
    cols = [C.COUNTRY, "name_sorted"]
    s1_all = load_normalised("train", (1,), cfg, train.s1[C.ENTITY_ID], columns=cols)
    pool_ids = pd.concat([train.s2[C.ENTITY_ID], train.s3[C.ENTITY_ID]], ignore_index=True)
    parts = []
    for country in sorted(s1_all[C.COUNTRY].unique()):
        s1n = s1_all[(s1_all[C.COUNTRY] == country).to_numpy()]
        pooln = load_normalised("train", (2, 3), cfg, pool_ids, columns=cols, country=country)
        parts.append(near_duplicates(s1n, pooln, train.pairs, ev))
        del s1n, pooln
        mem_guard(f"token evidence {country}")
    del s1_all
    table = evidence_table(pd.concat(parts, ignore_index=True), ev)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table.record()))
    return table


def prepare(s1n: pd.DataFrame, pooln: pd.DataFrame, cfg: PipelineConfig, tag: str,
            timings: dict | None = None, fillers: list[str] | None = None) -> pd.DataFrame:
    """Candidate pairs for normalised S1 and pool records, cached under ``tag``.

    With the nofill pass on, the tag also names the learned ``fillers`` it keys on (the
    frames must carry name_core_nofill); without it the fillers change no pair, so the tag
    of a version that learns them stays that of one that does not.
    """
    if cfg.blocking.nofill_max_group is not None:
        if fillers is None:
            raise ValueError("BlockingConfig.nofill_max_group needs the version's learned "
                             "fillers: learn them (NormaliseConfig.learn_fillers), pass fillers=")
        tag = f"{tag}_f{_hash(fillers)}"
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


def _per_million(df: pd.DataFrame, key: str, counts: pd.Series, totals: pd.Series,
                 self_counted: bool = False) -> np.ndarray:
    """Rate of ``df[key]`` in ``counts`` (by country) per million records; NaN for "".

    ``self_counted``: the records of ``df`` are themselves in ``counts`` (same side), so one
    is removed from the count and from the total. Otherwise a unique name would read
    1 / total, a floor that differs between folds of different sizes (fit, val, test) and
    would make the same name look commoner on the smaller val fold.
    """
    idx = pd.MultiIndex.from_arrays([df[C.COUNTRY], df[key]])
    n = np.nan_to_num(counts.reindex(idx).to_numpy(dtype=np.float64), nan=0.0)
    total = totals.reindex(df[C.COUNTRY]).to_numpy(dtype=np.float64)
    if self_counted:
        n, total = np.maximum(n - 1.0, 0.0), total - 1.0
    rate = n / np.maximum(total, 1.0) * 1e6
    rate[(df[key] == "").to_numpy()] = np.nan
    return rate.astype(np.float32)


def add_frequencies(s1n: pd.DataFrame, pooln: pd.DataFrame,
                    s1_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add ``features.FREQ_COLUMNS`` (core-name frequencies) to the S1 and pool frames.

    ``s1_all`` holds country, name_core and name_first of EVERY S1 record of the fold (a
    training sample would understate how common a name is); ``pooln`` must be the whole
    pool of the fold. Counts are per country, as rates per million records of that side;
    same-side rates count the OTHER records with the name (a unique name reads 0).
    """
    def count(df: pd.DataFrame, key: str) -> pd.Series:
        return df.groupby([C.COUNTRY, key], sort=False, observed=True).size()

    s1_tot, pool_tot = s1_all.groupby(C.COUNTRY).size(), pooln.groupby(C.COUNTRY).size()
    s1_core, s1_first = count(s1_all, "name_core"), count(s1_all, "name_first")
    pool_core, pool_first = count(pooln, "name_core"), count(pooln, "name_first")
    s1n = s1n.assign(freq_same=_per_million(s1n, "name_core", s1_core, s1_tot, True),
                     freq_other=_per_million(s1n, "name_core", pool_core, pool_tot),
                     freq_first_other=_per_million(s1n, "name_first", pool_first, pool_tot))
    pooln = pooln.assign(freq_same=_per_million(pooln, "name_core", pool_core, pool_tot, True),
                         freq_other=_per_million(pooln, "name_core", s1_core, s1_tot),
                         freq_first_other=_per_million(pooln, "name_first", s1_first, s1_tot))
    return s1n, pooln


def _with_frequencies(cfg: PipelineConfig, s1n: pd.DataFrame, pooln: pd.DataFrame,
                      s1_all: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``add_frequencies`` when the frequency group is configured, else the frames as given."""
    if "frequency" not in cfg.feature_groups:
        return s1n, pooln
    return add_frequencies(s1n, pooln, s1n if s1_all is None else s1_all)


# ---------------------------------------------------------------- scoring ----
def score(pairs: pd.DataFrame, s1n: pd.DataFrame, pooln: pd.DataFrame, matcher: Matcher,
          cfg: PipelineConfig, evidence: TokenEvidence | None = None) -> pd.DataFrame:
    """SCORED_COLUMNS for every candidate pair, featured and predicted chunk by chunk.

    Pool statistics (idf, name frequency) are counted once over the whole ``pooln`` and
    shared by every chunk, so they never depend on the chunking. ``evidence``: the version's
    learned token evidence (``Fitted.token_evidence``) for the tok_evidence group.
    """
    prob = np.empty(len(pairs), dtype=np.float32)
    stats = pool_stats(pooln, cfg.feature_groups) if len(pairs) else None
    for sl in iter_chunks(pairs, cfg.chunk_rows):
        X = build_features(pairs.iloc[sl], s1n, pooln, groups=cfg.feature_groups,
                           chunk_rows=cfg.chunk_rows, stats=stats, evidence=evidence)
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


# -------------------------------------------------------------------- fit ----
def _side(name: str, s1: pd.DataFrame, fold: Fold, cfg: PipelineConfig, token_map: dict,
          info: dict, timings: dict, fillers: list[str] | None = None
          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalised S1 / pool records and candidate pairs of one side of the inner split."""
    t0 = time.perf_counter()
    s1n = load_normalised("train", (1,), cfg, s1[C.ENTITY_ID], token_map, fillers=fillers)
    pooln = load_normalised("train", (2, 3), cfg, pool_of(fold)[C.ENTITY_ID], token_map,
                            fillers=fillers)
    if "frequency" in cfg.feature_groups:  # frequencies over the whole fold, not the sample
        s1_all = load_normalised("train", (1,), cfg, fold.s1[C.ENTITY_ID],
                                 columns=[C.COUNTRY, "name_core", "name_first"])
        s1n, pooln = _with_frequencies(cfg, s1n, pooln, s1_all)
        del s1_all
    timings[f"{name}_load_seconds"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    pairs = prepare(s1n, pooln, cfg, _tag(name, s1n, pooln, token_map), fillers=fillers)
    timings[f"{name}_blocking_seconds"] = round(time.perf_counter() - t0, 2)
    truth = fold.pairs[isin(fold.pairs[C.S1_ID], pd.Index(s1[C.ENTITY_ID]))]
    if len(s1):  # an empty side (tiny fixtures) has no report
        info[f"{name}_blocking"] = blocking_report(pairs, Fold(name, s1, fold.s2, fold.s3, truth))
    info[f"{name}_pairs"] = len(pairs)
    return s1n, pooln, pairs


def fit(cfg: PipelineConfig, train: Fold, out: Path | None = None,
        timings: dict | None = None) -> Fitted:
    """Learn the token map, train the matcher on the fit side, tune the rule on the tune side.

    The validation fold is never touched here (11 §2).
    """
    timings = {} if timings is None else timings
    info: dict = {}
    t0 = time.perf_counter()
    normalise_split("train", cfg)
    token_map = learn_token_map(cfg, train)
    fillers = learn_fillers(cfg, train)
    evidence = learn_token_evidence(cfg, train)
    timings["normalise_seconds"] = round(time.perf_counter() - t0, 2)
    info["token_map_size"] = len(token_map)
    if fillers is not None:  # logged with the version (fit_info.json)
        info["fillers"] = fillers
    if evidence is not None:
        info["token_evidence"] = {"pool_words": len(evidence.pool),
                                  "s1_words": len(evidence.s1)}
    fit_fold, tune_fold = inner_split(train)

    # fit side: features in memory for LightGBM
    fit_s1 = sample_s1(fit_fold.s1, cfg.n_fit_s1)
    s1n, pooln, pairs = _side("fit", fit_s1, fit_fold, cfg, token_map, info, timings, fillers)
    t0 = time.perf_counter()
    X_fit = build_features(pairs, s1n, pooln, groups=cfg.feature_groups,
                           chunk_rows=cfg.chunk_rows, evidence=evidence)
    y_fit = label_pairs(pairs, fit_fold.pairs)["label"].to_numpy(np.int8)
    del s1n, pooln, pairs
    mem_guard("fit features")

    # early-stopping sample of the tune side
    stop_s1 = sample_s1(tune_fold.s1, cfg.n_stop_s1)
    s1n, pooln, pairs = _side("stop", stop_s1, tune_fold, cfg, token_map, info, timings,
                              fillers)
    X_stop = build_features(pairs, s1n, pooln, groups=cfg.feature_groups,
                            chunk_rows=cfg.chunk_rows, evidence=evidence)
    y_stop = label_pairs(pairs, tune_fold.pairs)["label"].to_numpy(np.int8)
    timings["features_seconds"] = round(time.perf_counter() - t0, 2)
    del s1n, pooln, pairs
    info["fit_positive_rate"] = float(y_fit.mean()) if len(y_fit) else float("nan")
    mem_guard("stop features")

    t0 = time.perf_counter()
    matcher = Matcher(cfg.model).fit(X_fit, pd.Series(y_fit, index=X_fit.index),
                                     X_stop, pd.Series(y_stop, index=X_stop.index))
    timings["fit_seconds"] = round(time.perf_counter() - t0, 2)
    info["fit_info"] = matcher.fit_info_
    del X_fit, y_fit, X_stop, y_stop
    mem_guard("matcher fit")

    rule, table = _tune_rule(cfg, matcher, train, tune_fold, token_map, info, timings, fillers,
                             evidence)
    fitted = Fitted(matcher, rule, table, cfg, token_map, {**info, "timings": dict(timings)},
                    fillers, evidence)
    if out is not None:
        fitted.save(out)
    mem_guard("fit done")
    return fitted


def _tune_rule(cfg: PipelineConfig, matcher: Matcher, train: Fold, tune_fold: Fold,
               token_map: dict, info: dict, timings: dict,
               fillers: list[str] | None = None,
               evidence: TokenEvidence | None = None) -> tuple[DecisionRule, pd.DataFrame]:
    """Decision rule on the tune side (all S1 unless n_tune_s1), scored chunk by chunk."""
    tune_s1 = tune_fold.s1 if cfg.n_tune_s1 is None else sample_s1(tune_fold.s1, cfg.n_tune_s1)
    if cfg.tune_pool not in ("fold", "train"):
        raise ValueError(f"tune_pool must be 'fold' or 'train', got {cfg.tune_pool!r}")
    rule_fold = tune_fold if cfg.tune_pool == "fold" else Fold(  # every train S1 exists
        "tune", train.s1, train.s2, train.s3, tune_fold.pairs)
    name = "tune" if cfg.tune_pool == "fold" else "tunedense"
    s1n, pooln, pairs = _side(name, tune_s1, rule_fold, cfg, token_map, info, timings, fillers)
    t0 = time.perf_counter()
    scored = score(pairs, s1n, pooln, matcher, cfg, evidence)
    timings["score_seconds"] = round(time.perf_counter() - t0, 2)
    del s1n, pooln, pairs
    t0 = time.perf_counter()
    rule, table = tune(scored, tune_s1[C.ENTITY_ID], tune_fold.pairs, cfg.grid)
    timings["tune_seconds"] = round(time.perf_counter() - t0, 2)
    return rule, table


def retune(cfg: PipelineConfig, fitted: Fitted, train: Fold, out: Path | None = None,
           timings: dict | None = None) -> Fitted:
    """A new decision rule for an already trained matcher (E-group versions, 10 §6).

    Keeps ``fitted``'s matcher and token map; only the tune-side scoring and the grid run
    again under ``cfg`` (e.g. ``tune_pool="train"``). ``cfg`` must share the fitted
    version's features, else the matcher refuses the frame.
    """
    timings = {} if timings is None else timings
    info = {k: v for k, v in fitted.info.items() if k != "timings"}
    _, tune_fold = inner_split(train)
    rule, table = _tune_rule(cfg, fitted.matcher, train, tune_fold, fitted.token_map, info,
                             timings, fitted.fillers, fitted.token_evidence)
    out_fitted = Fitted(fitted.matcher, rule, table, cfg, fitted.token_map,
                        {**info, "timings": dict(timings), "retuned_from": asdict(fitted.rule)},
                        fitted.fillers, fitted.token_evidence)
    if out is not None:
        out_fitted.save(out)
    mem_guard("retune done")
    return out_fitted


# ---------------------------------------------------------------- run_* ----
def run_fold(cfg: PipelineConfig, fitted: Fitted, fold: Fold, tag: str | None = None
             ) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score one labelled fold (val, or a variant of it) with the frozen rule.

    Returns (metrics, candidate pairs, scored pairs, matches). ``metrics`` holds the
    ``metrics.breakdown`` keys, the blocking keys renamed as in 13 §2.2 and the timings.
    """
    timings: dict = {}
    tag = tag or fold.name
    t0 = time.perf_counter()
    s1n = load_normalised("train", (1,), cfg, fold.s1[C.ENTITY_ID], fitted.token_map,
                          fillers=fitted.fillers)
    pooln = load_normalised("train", (2, 3), cfg, pool_of(fold)[C.ENTITY_ID], fitted.token_map,
                            fillers=fitted.fillers)
    s1n, pooln = _with_frequencies(cfg, s1n, pooln)  # s1n is the whole fold here
    timings["normalise_seconds"] = round(time.perf_counter() - t0, 2)
    pairs = prepare(s1n, pooln, cfg, _tag(tag, s1n, pooln, fitted.token_map), timings,
                    fitted.fillers)
    t0 = time.perf_counter()
    scored = score(pairs, s1n, pooln, fitted.matcher, cfg, fitted.token_evidence)
    timings["score_seconds"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    matches = decide_by_country(scored, s1n, fitted.rule)
    timings["decide_seconds"] = round(time.perf_counter() - t0, 2)
    blocking = blocking_report(pairs, fold)
    metrics = {**score_pairs(matches, fold),
               "cand_recall": blocking["pair_recall"],
               "entity_recall": blocking["entity_recall"],
               "ceiling_f_beta": blocking["ceiling_f_beta"],
               "cands_mean": blocking["candidates_mean"],
               "cands_p95": blocking["candidates_p95"],
               **timings}
    mem_guard(f"run_fold {tag}")
    return metrics, pairs, scored, matches


# ------------------------------------------------------------ mock test ----
def mock_partition(cfg: PipelineConfig, mock: MockFold, country: str, token_map: dict,
                   tag: str = "mock", fillers: list[str] | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalised present S1, kept pool and candidate pairs of one mock country (cached).

    Frequencies are counted over the country's whole mock fold, as ``run_test`` counts them
    over the whole test country. ``fillers``: the version's learned fillers (``Fitted``).
    """
    s1 = mock.fold.s1
    s1_ids = s1[C.ENTITY_ID][(s1[C.COUNTRY] == country).to_numpy()]
    s1c = load_normalised("train", (1,), cfg, s1_ids, token_map, fillers=fillers)
    poolc = load_normalised("train", (2, 3), cfg, pool_of(mock.fold)[C.ENTITY_ID], token_map,
                            country=country, fillers=fillers)
    s1c, poolc = _with_frequencies(cfg, s1c, poolc)
    pairs = prepare(s1c, poolc, cfg, _tag(tag, s1c, poolc, token_map), fillers=fillers)
    return s1c, poolc, pairs


def run_mock(cfg: PipelineConfig, fitted: Fitted, mock: MockFold, tag: str = "mock",
             roles: tuple[str, ...] = ("tune", "val"), timings: dict | None = None
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every present entity of ``mock``, one country at a time, 1-to-1 over all of them.

    Returns ``(scored, report)``. ``scored``: SCORED_COLUMNS of the pairs of the ``roles``
    entities that survive the pool-side 1-to-1 against every present entity of their
    country, so ``decide`` / ``tune`` on any subset of them give what the whole-partition
    decision gives (``decision.one_to_one_filter``). ``report``: ``blocking_report`` of each
    role, i.e. candidate recall and candidates per S1 at the mock's density.
    """
    timings = {} if timings is None else timings
    for k in ("blocking_seconds", "score_seconds", "decide_seconds"):
        timings[k] = 0.0
    keep = pd.Index(mock.fold.s1[C.ENTITY_ID][mock.role.isin(roles).to_numpy()])
    out, cands = [], []
    for country in sorted(mock.fold.s1[C.COUNTRY].unique()):
        t0 = time.perf_counter()
        s1c, poolc, pairs = mock_partition(cfg, mock, country, fitted.token_map, tag,
                                           fitted.fillers)
        timings["blocking_seconds"] += round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        scored = score(pairs, s1c, poolc, fitted.matcher, cfg, fitted.token_evidence)
        timings["score_seconds"] += round(time.perf_counter() - t0, 2)
        del s1c, poolc
        t0 = time.perf_counter()
        kept = one_to_one_filter(scored)
        out.append(kept[isin(kept[C.S1_ID], keep)].reset_index(drop=True))
        cands.append(pairs.loc[isin(pairs[C.S1_ID], keep), [C.S1_ID, C.ENTITY_ID]])
        timings["decide_seconds"] += round(time.perf_counter() - t0, 2)
        del pairs, scored, kept
        mem_guard(f"run_mock {country}")
    scored = pd.concat(out, ignore_index=True)
    cand = pd.concat(cands, ignore_index=True)
    report = pd.DataFrame({r: blocking_report(cand, mock.part(r)) for r in roles}).T
    return scored, report


def _rows_of(scored: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    """Rows of ``scored`` whose S1 entity is in ``fold``."""
    return scored[isin(scored[C.S1_ID], pd.Index(fold.s1[C.ENTITY_ID]))]


def tune_mock(scored: pd.DataFrame, mock: MockFold, grid: Grid,
              fp_weight: float = 1.0) -> tuple[DecisionRule, pd.DataFrame]:
    """``decision.tune`` on the mock's tune entities (``scored`` from ``run_mock``).

    ``scored`` already went through the 1-to-1, so the grid must keep it on: a rule without
    it would be tuned on filtered pairs and then applied unfiltered at test time.
    """
    if not all(grid.one_to_one):
        raise ValueError("tune_mock needs grid.one_to_one == (True,): the mock is 1-to-1 filtered")
    part = mock.part("tune")
    return tune(_rows_of(scored, part), part.s1[C.ENTITY_ID], part.pairs, grid, fp_weight)


def mock_scores(scored: pd.DataFrame, mock: MockFold, rule: DecisionRule | ExpectedRule,
                role: str = "val") -> pd.DataFrame:
    """``score_pairs`` of ``rule`` on the mock's ``role`` entities: all, then per country.

    Also ``f_tight`` (false merges weighted ``FP_WEIGHT``) and ``est_public`` (``f_tight``
    minus ``PUBLIC_OFFSET``): the tight mock, calibrated to the leaderboard (``mock.py``).
    """
    part = mock.part(role)
    matches = apply_rule(_rows_of(scored, part), rule)
    rows = {"all": _with_tight(score_pairs(matches, part), matches, part)}
    for country in sorted(part.s1[C.COUNTRY].unique()):
        in_c = (part.s1[C.COUNTRY] == country).to_numpy()
        ids = pd.Index(part.s1[C.ENTITY_ID][in_c])
        sub = Fold(f"{part.name}_{country}", part.s1[in_c].reset_index(drop=True), part.s2,
                   part.s3, part.pairs[isin(part.pairs[C.S1_ID], ids)])
        rows[country] = _with_tight(score_pairs(matches, sub), matches, sub)
    return pd.DataFrame(rows).T


def _with_tight(scores: dict, matches: pd.DataFrame, fold: Fold) -> dict:
    """``scores`` plus the tight mock score and the public estimate of ``matches`` on ``fold``."""
    tp, n_pred, n_true = entity_counts(matches, fold)
    tight = float(entity_tight_from_counts(tp, n_pred, n_true, FP_WEIGHT).mean())
    return {**scores, "f_tight": tight, "est_public": tight - PUBLIC_OFFSET}


def run_test(cfg: PipelineConfig, fitted: Fitted, out_dir: Path = C.OUTPUT,
             timings: dict | None = None
             ) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write both submission files for the test split, one country partition at a time.

    Returns (matching path, candidate path, normalised S1, matches with ``prob``, per-S1
    summary with ``p_max`` and ``n_cands`` for S1 ids that have candidates). Only one
    partition's pool, pairs and features are in memory at once; the candidate file is
    written from exactly the pairs that were scored.
    """
    timings = {} if timings is None else timings
    for k in ("normalise_seconds", "blocking_seconds", "score_seconds", "decide_seconds"):
        timings[k] = 0.0
    t0 = time.perf_counter()
    s1n = load_normalised("test", (1,), cfg, token_map=fitted.token_map, fillers=fitted.fillers)
    timings["normalise_seconds"] += round(time.perf_counter() - t0, 2)
    matches, cands, p_max = [], [], []
    for country in sorted(s1n[C.COUNTRY].unique()):
        t0 = time.perf_counter()
        s1c = s1n[(s1n[C.COUNTRY] == country).to_numpy()].reset_index(drop=True)
        poolc = load_normalised("test", (2, 3), cfg, token_map=fitted.token_map,
                                country=country, fillers=fitted.fillers)
        s1c, poolc = _with_frequencies(cfg, s1c, poolc)  # the whole test S1 of the country
        timings["normalise_seconds"] += round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        pairs = prepare(s1c, poolc, cfg, _tag("test", s1c, poolc, fitted.token_map),
                        fillers=fitted.fillers)
        timings["blocking_seconds"] += round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        scored = score(pairs, s1c, poolc, fitted.matcher, cfg, fitted.token_evidence)
        timings["score_seconds"] += round(time.perf_counter() - t0, 2)
        del poolc
        t0 = time.perf_counter()
        matches.append(sort_matches(decide(scored, fitted.rule), scored))
        p_max.append(scored.groupby(C.S1_ID, sort=False)["prob"].agg(p_max="max",
                                                                      n_cands="size"))
        cands.append(pairs[[C.S1_ID, C.ENTITY_ID]])
        timings["decide_seconds"] += round(time.perf_counter() - t0, 2)
        del pairs, scored
        mem_guard(f"run_test {country}")
    match_frame = pd.concat(matches, ignore_index=True)
    cand_frame = pd.concat(cands, ignore_index=True)
    del cands
    s1_ids = load_source("test", 1, cfg.dataset_dir, columns=[C.ENTITY_ID])[C.ENTITY_ID]
    paths = write_pairs(match_frame[[C.S1_ID, C.ENTITY_ID]], cand_frame, s1_ids.tolist(),
                        out_dir)
    del cand_frame
    mem_guard("run_test")
    return (*paths, s1n, match_frame, pd.concat(p_max))
