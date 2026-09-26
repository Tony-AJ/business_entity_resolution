"""Two-stage matcher trained where the test's decoys are (plan C5 / D3 / A5).

Stage 1 is an already fitted pipeline version (token map, blocking, features, matcher; e.g.
v101). It scores every candidate pair of a partition, and its probability ``p1`` does two
jobs:

* **candidate filter**: a pair stays only if ``p1 >= floor`` and it is among its entity's
  ``max_cands`` best. The kept pairs are all stage 2 scores, so they are the final candidate
  set written to ``candidate_pairs.tsv``: a learned last blocking stage;
* **competition features** (``stacking.competition_features``) over all pairs of the
  partition: rank and best rival on the S1 side and on the pool side;
* **anchor features** (``stacking.anchor_features``) over the kept pairs: each candidate
  compared with its entity's best other candidate (true records of one business resemble
  each other, a same-name decoy does not);
* **extra feature groups** (``TwoStageConfig.extra_groups``, optional) over the kept pairs:
  pair features the stage-1 matcher never read, e.g. M3's idf, token_freq, ctx_idf and
  address_extra groups, computed for the ~5 kept pairs per S1 instead of every candidate.

Stage 2 is a LightGBM on the pair features plus the competition features, trained on the kept
pairs of the mock fold's ``fit`` entities (``mock.py``), so it learns at the test's decoy
density with every rival present. It is cross-fitted: fit entities are split into ``folds``
parts by id hash and one model is trained per held-out part; a fit entity is scored by the
model that never saw it (so its rivals in the 1-to-1 are as honest as on test), every other
entity (tune, val, test) by the mean of all models.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data import isin, load_source
from .decision import (
    SCORED_COLUMNS,
    DecisionRule,
    ExpectedRule,
    apply_rule,
    one_to_one_filter,
    rule_from_json,
    rule_to_json,
)
from .evaluate import blocking_report
from .features import (
    _ALL_INPUTS,
    FREQ_COLUMNS,
    build_features,
    feature_names,
    iter_chunks,
    pool_stats,
)
from .mock import MockFold
from .model import Matcher, MatcherParams
from .pipeline import (
    Fitted,
    PipelineConfig,
    _tag,
    _with_frequencies,
    load_normalised,
    mem_guard,
    mock_partition,
    prepare,
    sort_matches,
)
from .split import hash_unit
from .stacking import (
    anchor_features,
    cohesion_features,
    competition_features,
    group_stats,
    rival_features,
)
from .submission import write_pairs
from .trainset import label_pairs, sample_s1

FOLD_SEED = 6161


@dataclass
class TwoStageConfig:
    """Candidate filter and stage-2 training of a two-stage version."""

    floor: float = 0.01          # stage-1 probability a pair needs to stay a candidate
    max_cands: int = 16          # at most this many candidates per S1 entity, best p1 first
    folds: int = 2               # cross-fitting parts over the mock's fit entities
    seed: int = FOLD_SEED
    n_stop_s1: int = 50_000      # mock tune entities whose kept pairs drive early stopping
    anchors: bool = True         # add stacking.ANCHOR_COLUMNS to the stage-2 frame
    cohesion: bool = False       # add stacking.COHESION_COLUMNS (each vs all other candidates)
    rivals: bool = False         # add stacking.RIVAL_COLUMNS (the record vs its best rival S1)
    # mock roles stage 2 trains on, all scored out of fold. With "tune" among them, early
    # stopping reads each model's held-out part (not tune entities), so the rule tuned on
    # the tune entities still sees out-of-fold probabilities
    train_roles: tuple[str, ...] = ("fit",)
    model: MatcherParams = field(default_factory=lambda: MatcherParams(n_estimators=4000))
    # feature groups (features.REGISTRY) built on the kept pairs only and appended to the
    # stage-2 frame; the stage-1 matcher never reads them. A stage-1 cache holds one
    # combination: give each extra_groups value its own cache directory
    extra_groups: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Cross-fitting needs two parts: with one, fit entities would be scored in-sample.

        Unknown or repeated extra groups are refused here, before any pass runs.
        """
        if self.folds < 2:
            raise ValueError(f"folds must be >= 2 (out-of-fold scores), got {self.folds}")
        feature_names(self.extra_groups)

    def record(self) -> dict:
        """JSON-ready dict (metrics.json, artifacts)."""
        return json.loads(json.dumps(asdict(self), default=str))


@dataclass
class Stage1Output:
    """The kept pairs of one partition with the frame stage 2 reads (same row order)."""

    pairs: pd.DataFrame      # source1_entity_id, entity_id of the kept pairs
    X: pd.DataFrame          # pair features + STACK_COLUMNS (+ ANCHOR_COLUMNS), float32
    n_all: int               # candidate pairs before the filter


# normalised columns the stage-1 pass reads (features, anchors, cohesion, rivals); blocking's
# own texts (name_addr: name + address, the longest) are dropped once the pairs exist
_STAGE1_COLUMNS = (C.ENTITY_ID, C.COUNTRY, *_ALL_INPUTS, *FREQ_COLUMNS)


def trim(records: pd.DataFrame) -> pd.DataFrame:
    """``records`` without the normalised columns the stage-1 pass never reads."""
    return records[[c for c in records.columns if c in _STAGE1_COLUMNS]]


def keep_mask(s1_ids: pd.Series, p1: np.ndarray, floor: float, max_cands: int) -> np.ndarray:
    """Pairs that stay candidates: ``p1 >= floor`` and among the entity's ``max_cands`` best."""
    codes, uniques = pd.factorize(s1_ids, use_na_sentinel=False)
    rank, _ = group_stats(codes, np.nan_to_num(p1.astype(np.float32), nan=0.0), len(uniques))
    return (p1 >= floor) & (rank <= max_cands)


def stage1_partition(pairs: pd.DataFrame, s1n: pd.DataFrame, pooln: pd.DataFrame,
                     stage1: Fitted, cfg: PipelineConfig, tcfg: TwoStageConfig) -> Stage1Output:
    """Stage 1 over every candidate pair of one partition: filter, then competition features.

    Features are built chunk by chunk (chunks never split an S1 group, so the per-entity cap
    is exact); only the kept rows are held. Competition features use every pair; the
    ``extra_groups`` see only the kept pairs, with pool statistics of the whole partition.
    """
    both = sorted(set(tcfg.extra_groups) & set(cfg.feature_groups))
    if both:
        raise ValueError(f"extra_groups {both} are stage-1 groups already")
    p1 = np.empty(len(pairs), dtype=np.float32)
    keep = np.zeros(len(pairs), dtype=bool)
    parts = []
    stats = pool_stats(pooln, cfg.feature_groups) if len(pairs) else None  # once per partition
    for sl in iter_chunks(pairs, cfg.chunk_rows):
        X = build_features(pairs.iloc[sl], s1n, pooln, groups=cfg.feature_groups,
                           chunk_rows=cfg.chunk_rows, stats=stats)
        p = stage1.matcher.predict_proba(X)
        k = keep_mask(pairs[C.S1_ID].iloc[sl], p, tcfg.floor, tcfg.max_cands)
        p1[sl], keep[sl] = p, k
        parts.append(X[k].reset_index(drop=True))
        del X
    X = pd.concat(parts, ignore_index=True) if parts else build_features(
        pairs.iloc[:0], s1n, pooln, groups=cfg.feature_groups)
    parts.clear()                                   # the concat is the only copy kept
    extra = [competition_features(pairs[[C.S1_ID, C.ENTITY_ID]], p1, keep)]
    kept = pairs.loc[keep, [C.S1_ID, C.ENTITY_ID]].reset_index(drop=True)
    if tcfg.anchors:
        extra.append(anchor_features(kept, p1[keep], pooln).reset_index(drop=True))
    if tcfg.cohesion:
        extra.append(cohesion_features(kept, p1[keep], pooln).reset_index(drop=True))
    if tcfg.rivals:
        own = X["ad_token_set"].to_numpy() if "ad_token_set" in X.columns else None
        extra.append(rival_features(pairs[[C.S1_ID, C.ENTITY_ID]], p1, keep, s1n, pooln, own))
    if tcfg.extra_groups:  # kept pairs stay grouped by S1: the filter keeps their order
        extra.append(build_features(
            pairs[keep].reset_index(drop=True), s1n, pooln, groups=tcfg.extra_groups,
            chunk_rows=cfg.chunk_rows, stats=pool_stats(pooln, tcfg.extra_groups)))
    X = pd.concat([X, *extra], axis=1)
    return Stage1Output(kept, X, len(pairs))


def save_stage1(o: Stage1Output, path: Path) -> Path:
    """Write a partition's stage-1 output as one Parquet file (pairs + frame) and a sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.concat([o.pairs, o.X], axis=1).to_parquet(tmp, index=False)
    tmp.replace(path)
    path.with_suffix(".json").write_text(json.dumps({"n_all": o.n_all}) + "\n")
    return path


def load_stage1(path: Path) -> Stage1Output:
    """Read back what ``save_stage1`` wrote."""
    df = pd.read_parquet(path)
    pairs = df[[C.S1_ID, C.ENTITY_ID]].copy()
    X = df.drop(columns=[C.S1_ID, C.ENTITY_ID])
    n_all = json.loads(path.with_suffix(".json").read_text())["n_all"]
    return Stage1Output(pairs, X, n_all)


def _cached_stage1(cache_dir: Path | None, name: str, compute) -> Stage1Output:
    """``compute()`` or its cached result under ``cache_dir/name.parquet``.

    The cache is only valid for one stage-1 model, blocking configuration, filter and anchor
    switch: callers give each such combination its own directory.
    """
    if cache_dir is not None and (Path(cache_dir) / f"{name}.parquet").exists():
        return load_stage1(Path(cache_dir) / f"{name}.parquet")
    o = compute()
    if cache_dir is not None:
        save_stage1(o, Path(cache_dir) / f"{name}.parquet")
    return o


def mock_stage1(cfg: PipelineConfig, stage1: Fitted, mock: MockFold, tcfg: TwoStageConfig,
                tag: str = "mock", timings: dict | None = None,
                cache_dir: Path | None = None) -> dict[str, Stage1Output]:
    """``stage1_partition`` for every country of the mock fold (blocking cached per tag;
    the outputs too when ``cache_dir`` is given, see ``_cached_stage1``)."""
    timings = {} if timings is None else timings
    out = {}
    for country in sorted(mock.fold.s1[C.COUNTRY].unique()):
        t0 = time.perf_counter()

        def compute(country: str = country) -> Stage1Output:
            s1c, poolc, pairs = mock_partition(cfg, mock, country, stage1.token_map, tag)
            s1c, poolc = trim(s1c), trim(poolc)       # frees blocking's texts
            return stage1_partition(pairs, s1c, poolc, stage1, cfg, tcfg)

        out[country] = _cached_stage1(cache_dir, f"mock_{country}", compute)
        timings[f"stage1_{country}_seconds"] = round(time.perf_counter() - t0, 2)
        mem_guard(f"mock_stage1 {country}")
    return out


def fold_of(s1_ids: pd.Series, folds: int, seed: int = FOLD_SEED) -> np.ndarray:
    """Cross-fitting part (0..folds-1) of each S1 id, by id hash."""
    return np.minimum((hash_unit(s1_ids, seed) * folds).astype(np.int64), folds - 1)


def _rows(outs: dict[str, Stage1Output], ids: pd.Index,
          columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kept pairs and features (``columns`` only, if given) of ``ids``, countries stacked."""
    pairs, X = [], []
    for o in outs.values():
        m = isin(o.pairs[C.S1_ID], ids)
        pairs.append(o.pairs[m])
        X.append(o.X.loc[m, columns] if columns is not None else o.X[m])
    return (pd.concat(pairs, ignore_index=True), pd.concat(X, ignore_index=True))


def _train_ids(mock: MockFold, tcfg: TwoStageConfig) -> pd.Index:
    """Present mock entities stage 2 trains on (and scores out of fold)."""
    if "val" in tcfg.train_roles:
        raise ValueError("stage 2 never trains on val entities")
    return pd.Index(mock.fold.s1[C.ENTITY_ID][mock.role.isin(tcfg.train_roles).to_numpy()])


def fit_stage2(outs: dict[str, Stage1Output], mock: MockFold, tcfg: TwoStageConfig,
               columns: list[str] | None = None) -> tuple[list[Matcher], dict]:
    """One stage-2 matcher per cross-fitting part, trained on the other parts' fit entities.

    Early stopping reads the kept pairs of ``n_stop_s1`` mock tune entities (the rule is later
    tuned on all tune entities, as in ``pipeline.fit``). ``columns`` restricts the features
    (ablations); None = every column of the stage-1 output.
    """
    fit_ids = pd.Series(_train_ids(mock, tcfg))
    part_of_id = fold_of(fit_ids, tcfg.folds, tcfg.seed)
    held_out_stop = "tune" in tcfg.train_roles

    def stop_set(ids: pd.Index) -> tuple[pd.DataFrame, np.ndarray]:
        """Early-stopping rows: the kept pairs of a hashed sample of ``ids``."""
        sample = sample_s1(pd.DataFrame({C.ENTITY_ID: ids}), tcfg.n_stop_s1)[C.ENTITY_ID]
        stop_pairs, X_stop = _rows(outs, pd.Index(sample), columns)
        return X_stop, label_pairs(stop_pairs, mock.fold.pairs)["label"].to_numpy(np.int8)

    if not held_out_stop:
        X_stop, y_stop = stop_set(pd.Index(mock.ids("tune")))
    models, info, rows, positives = [], {}, 0, 0
    for f in range(tcfg.folds):
        if held_out_stop:        # the part this model never trains on
            X_stop, y_stop = stop_set(pd.Index(fit_ids[part_of_id == f]))
        # one copy per model: the frame of the entities outside part f, built directly
        pairs, X = _rows(outs, pd.Index(fit_ids[part_of_id != f]), columns)
        y = label_pairs(pairs, mock.fold.pairs)["label"].to_numpy(np.int8)
        del pairs
        m = Matcher(tcfg.model).fit(X, y, X_stop, y_stop)
        models.append(m)
        info[f"fold{f}"] = m.fit_info_
        rows, positives = rows + len(y), positives + int(y.sum())
        del X, y
        mem_guard(f"fit_stage2 fold {f}")
    info["rows"] = rows // max(tcfg.folds - 1, 1)          # each fit row trains folds-1 models
    info["positive_rate"] = positives / rows if rows else None
    return models, info


def predict_stage2(models: list[Matcher], X: pd.DataFrame,
                   part: np.ndarray | None = None) -> np.ndarray:
    """Stage-2 probability per row: the held-out model where ``part >= 0``, else the mean."""
    if part is None:
        return np.mean([m.predict_proba(X) for m in models], axis=0).astype(np.float32)
    out = np.empty(len(X), dtype=np.float32)
    rest = np.flatnonzero(part < 0)
    if len(rest):
        out[rest] = np.mean([m.predict_proba(X.iloc[rest]) for m in models], axis=0)
    for f, m in enumerate(models):
        rows = np.flatnonzero(part == f)
        if len(rows):
            out[rows] = m.predict_proba(X.iloc[rows])
    return out


def mock_scored(outs: dict[str, Stage1Output], models: list[Matcher], mock: MockFold,
                tcfg: TwoStageConfig, roles: tuple[str, ...] = ("tune", "val")
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage-2 scores on the mock, 1-to-1 across every present entity, rows of ``roles``.

    Fit entities are scored out of fold. Returns ``(scored, report)`` like
    ``pipeline.run_mock``; the report is the candidate set after the stage-1 filter. The
    models' own feature lists pick the columns they read.
    """
    fit_ids = _train_ids(mock, tcfg)
    keep = pd.Index(mock.fold.s1[C.ENTITY_ID][mock.role.isin(roles).to_numpy()])
    out, cands = [], []
    for o in outs.values():
        part = np.where(isin(o.pairs[C.S1_ID], fit_ids),
                        fold_of(o.pairs[C.S1_ID], tcfg.folds, tcfg.seed), -1)
        X = o.X if list(o.X.columns) == models[0].feature_names_ else o.X[
            models[0].feature_names_]
        scored = o.pairs.assign(prob=predict_stage2(models, X, part))[SCORED_COLUMNS]
        del X
        kept = one_to_one_filter(scored)
        out.append(kept[isin(kept[C.S1_ID], keep)].reset_index(drop=True))
        cands.append(o.pairs[isin(o.pairs[C.S1_ID], keep)])
    cand = pd.concat(cands, ignore_index=True)
    report = pd.DataFrame({r: blocking_report(cand, mock.part(r)) for r in roles}).T
    return pd.concat(out, ignore_index=True), report


@dataclass
class TwoStage:
    """A fitted two-stage version: stage-1 pipeline, stage-2 models, filter, rule."""

    stage1: Fitted
    models: list[Matcher]
    rule: DecisionRule | ExpectedRule
    tcfg: TwoStageConfig
    tune_table: pd.DataFrame
    info: dict = field(default_factory=dict)

    def save(self, out: Path) -> Path:
        """Stage-1 files as ``Fitted.save``, plus ``stage2_<k>/``, two_stage.json, rule."""
        self.stage1.save(out / "stage1")
        for k, m in enumerate(self.models):
            m.save(out / f"stage2_{k}")
        (out / "two_stage.json").write_text(json.dumps(self.tcfg.record(), indent=2) + "\n")
        (out / "rule.json").write_text(json.dumps(
            {**rule_to_json(self.rule), "tune_f_beta": float(self.tune_table["f_beta"].max())},
            indent=2) + "\n")
        self.tune_table.to_csv(out / "tune_table.csv", index=False)
        (out / "fit_info.json").write_text(json.dumps(self.info, indent=2, default=float) + "\n")
        return out

    @classmethod
    def load(cls, out: Path, cfg: PipelineConfig) -> TwoStage:
        """Read back what ``save`` wrote; ``cfg`` is the stage-1 pipeline configuration."""
        tc = json.loads((out / "two_stage.json").read_text())
        tcfg = TwoStageConfig(**{**tc, "model": MatcherParams(**tc["model"]),
                                 "train_roles": tuple(tc.get("train_roles", ("fit",))),
                                 "extra_groups": tuple(tc.get("extra_groups", ()))})
        rule = rule_from_json(json.loads((out / "rule.json").read_text()))
        models = [Matcher.load(out / f"stage2_{k}") for k in range(tcfg.folds)]
        info_path = out / "fit_info.json"
        return cls(Fitted.load(out / "stage1", cfg), models, rule, tcfg,
                   pd.read_csv(out / "tune_table.csv"),
                   json.loads(info_path.read_text()) if info_path.exists() else {})


def run_test_two_stage(cfg: PipelineConfig, ts: TwoStage, out_dir: Path = C.OUTPUT,
                       timings: dict | None = None, cache_dir: Path | None = None
                       ) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Both submission files for the test split, one country at a time.

    ``candidate_pairs.tsv`` holds the pairs that pass the stage-1 filter: exactly the pairs
    stage 2 scores. Returns (matching path, candidate path, normalised S1, matches with
    ``prob``, per-S1 ``p_max`` / ``n_cands`` over the kept candidates). With ``cache_dir``
    the stage-1 outputs are read from / written to it (``_cached_stage1``).
    """
    timings = {} if timings is None else timings
    s1n = load_normalised("test", (1,), cfg, token_map=ts.stage1.token_map)
    matches, cands, summary = [], [], []
    for country in sorted(s1n[C.COUNTRY].unique()):
        t0 = time.perf_counter()

        def compute(country: str = country) -> Stage1Output:
            s1c = s1n[(s1n[C.COUNTRY] == country).to_numpy()].reset_index(drop=True)
            poolc = load_normalised("test", (2, 3), cfg, token_map=ts.stage1.token_map,
                                    country=country)
            s1c, poolc = _with_frequencies(cfg, s1c, poolc)
            pairs = prepare(s1c, poolc, cfg, _tag("test", s1c, poolc, ts.stage1.token_map))
            s1c, poolc = trim(s1c), trim(poolc)       # frees blocking's texts
            return stage1_partition(pairs, s1c, poolc, ts.stage1, cfg, ts.tcfg)

        o = _cached_stage1(cache_dir, f"test_{country}", compute)
        names = ts.models[0].feature_names_
        X = o.X if list(o.X.columns) == names else o.X[names]
        scored = o.pairs.assign(prob=predict_stage2(ts.models, X))[SCORED_COLUMNS]
        del X
        matches.append(sort_matches(apply_rule(scored, ts.rule), scored))
        summary.append(scored.groupby(C.S1_ID, sort=False)["prob"].agg(p_max="max",
                                                                      n_cands="size"))
        cands.append(o.pairs)
        timings[f"test_{country}_seconds"] = round(time.perf_counter() - t0, 2)
        del o, scored
        mem_guard(f"run_test_two_stage {country}")
    match_frame = pd.concat(matches, ignore_index=True)
    cand_frame = pd.concat(cands, ignore_index=True)
    s1_ids = load_source("test", 1, cfg.dataset_dir, columns=[C.ENTITY_ID])[C.ENTITY_ID]
    paths = write_pairs(match_frame[[C.S1_ID, C.ENTITY_ID]], cand_frame, s1_ids.tolist(),
                        out_dir)
    return (*paths, s1n, match_frame, pd.concat(summary))

