"""Two-stage matcher trained where the test's decoys are (plan C5 / D3 / A5).

Stage 1 is an already fitted pipeline version (token map, blocking, features, matcher; e.g.
v101). It scores every candidate pair of a partition, and its probability ``p1`` does two
jobs:

* **candidate filter**: a pair stays only if ``p1 >= floor`` and it is among its entity's
  ``max_cands`` best. The kept pairs are all stage 2 scores, so they are the final candidate
  set written to ``candidate_pairs.tsv``: a learned last blocking stage;
* **competition features** (``stacking.competition_features``) over all pairs of the
  partition: rank and best rival on the S1 side and on the pool side.

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
from .features import build_features, iter_chunks
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
from .stacking import competition_features, group_stats
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
    model: MatcherParams = field(default_factory=lambda: MatcherParams(n_estimators=4000))

    def record(self) -> dict:
        """JSON-ready dict (metrics.json, artifacts)."""
        return json.loads(json.dumps(asdict(self), default=str))


@dataclass
class Stage1Output:
    """The kept pairs of one partition with the frame stage 2 reads (same row order)."""

    pairs: pd.DataFrame      # source1_entity_id, entity_id of the kept pairs
    X: pd.DataFrame          # pair features + STACK_COLUMNS, float32, RangeIndex
    n_all: int               # candidate pairs before the filter


def keep_mask(s1_ids: pd.Series, p1: np.ndarray, floor: float, max_cands: int) -> np.ndarray:
    """Pairs that stay candidates: ``p1 >= floor`` and among the entity's ``max_cands`` best."""
    codes, uniques = pd.factorize(s1_ids, use_na_sentinel=False)
    rank, _ = group_stats(codes, np.nan_to_num(p1.astype(np.float32), nan=0.0), len(uniques))
    return (p1 >= floor) & (rank <= max_cands)


def stage1_partition(pairs: pd.DataFrame, s1n: pd.DataFrame, pooln: pd.DataFrame,
                     stage1: Fitted, cfg: PipelineConfig, tcfg: TwoStageConfig) -> Stage1Output:
    """Stage 1 over every candidate pair of one partition: filter, then competition features.

    Features are built chunk by chunk (chunks never split an S1 group, so the per-entity cap
    is exact); only the kept rows are held. Competition features use every pair.
    """
    p1 = np.empty(len(pairs), dtype=np.float32)
    keep = np.zeros(len(pairs), dtype=bool)
    parts = []
    for sl in iter_chunks(pairs, cfg.chunk_rows):
        X = build_features(pairs.iloc[sl], s1n, pooln, groups=cfg.feature_groups,
                           chunk_rows=cfg.chunk_rows)
        p = stage1.matcher.predict_proba(X)
        k = keep_mask(pairs[C.S1_ID].iloc[sl], p, tcfg.floor, tcfg.max_cands)
        p1[sl], keep[sl] = p, k
        parts.append(X[k].reset_index(drop=True))
        del X
    comp = competition_features(pairs[[C.S1_ID, C.ENTITY_ID]], p1)
    X = pd.concat(parts, ignore_index=True) if parts else build_features(
        pairs.iloc[:0], s1n, pooln, groups=cfg.feature_groups)
    X = pd.concat([X, comp[keep].reset_index(drop=True)], axis=1)
    kept = pairs.loc[keep, [C.S1_ID, C.ENTITY_ID]].reset_index(drop=True)
    return Stage1Output(kept, X, len(pairs))


def mock_stage1(cfg: PipelineConfig, stage1: Fitted, mock: MockFold, tcfg: TwoStageConfig,
                tag: str = "mock", timings: dict | None = None) -> dict[str, Stage1Output]:
    """``stage1_partition`` for every country of the mock fold (blocking cached per tag)."""
    timings = {} if timings is None else timings
    out = {}
    for country in sorted(mock.fold.s1[C.COUNTRY].unique()):
        t0 = time.perf_counter()
        s1c, poolc, pairs = mock_partition(cfg, mock, country, stage1.token_map, tag)
        out[country] = stage1_partition(pairs, s1c, poolc, stage1, cfg, tcfg)
        timings[f"stage1_{country}_seconds"] = round(time.perf_counter() - t0, 2)
        del s1c, poolc, pairs
        mem_guard(f"mock_stage1 {country}")
    return out


def fold_of(s1_ids: pd.Series, folds: int, seed: int = FOLD_SEED) -> np.ndarray:
    """Cross-fitting part (0..folds-1) of each S1 id, by id hash."""
    return np.minimum((hash_unit(s1_ids, seed) * folds).astype(np.int64), folds - 1)


def _rows(outs: dict[str, Stage1Output], ids: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Kept pairs and features of the entities ``ids``, all countries stacked."""
    pairs, X = [], []
    for o in outs.values():
        m = isin(o.pairs[C.S1_ID], ids)
        pairs.append(o.pairs[m])
        X.append(o.X[m])
    return (pd.concat(pairs, ignore_index=True), pd.concat(X, ignore_index=True))


def fit_stage2(outs: dict[str, Stage1Output], mock: MockFold, tcfg: TwoStageConfig
               ) -> tuple[list[Matcher], dict]:
    """One stage-2 matcher per cross-fitting part, trained on the other parts' fit entities.

    Early stopping reads the kept pairs of ``n_stop_s1`` mock tune entities (the rule is later
    tuned on all tune entities, as in ``pipeline.fit``).
    """
    fit_ids = mock.ids("fit")
    pairs, X = _rows(outs, fit_ids)
    y = label_pairs(pairs, mock.fold.pairs)["label"].to_numpy(np.int8)
    part = fold_of(pairs[C.S1_ID], tcfg.folds, tcfg.seed)
    stop_ids = pd.Index(sample_s1(mock.part("tune").s1, tcfg.n_stop_s1)[C.ENTITY_ID])
    stop_pairs, X_stop = _rows(outs, stop_ids)
    y_stop = label_pairs(stop_pairs, mock.fold.pairs)["label"].to_numpy(np.int8)
    models, info = [], {"rows": len(X), "positive_rate": float(y.mean()) if len(y) else None}
    for f in range(tcfg.folds):
        train = part != f if tcfg.folds > 1 else np.ones(len(X), dtype=bool)
        m = Matcher(tcfg.model).fit(X[train], y[train], X_stop, y_stop)
        models.append(m)
        info[f"fold{f}"] = m.fit_info_
        mem_guard(f"fit_stage2 fold {f}")
    return models, info


def predict_stage2(models: list[Matcher], X: pd.DataFrame,
                   part: np.ndarray | None = None) -> np.ndarray:
    """Stage-2 probability per row: the held-out model where ``part >= 0``, else the mean."""
    mean = np.mean([m.predict_proba(X) for m in models], axis=0).astype(np.float32)
    if part is None or len(models) == 1:
        return mean
    out = mean.copy()
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
    ``pipeline.run_mock``; the report is the candidate set after the stage-1 filter.
    """
    fit_ids = mock.ids("fit")
    keep = pd.Index(mock.fold.s1[C.ENTITY_ID][mock.role.isin(roles).to_numpy()])
    out, cands = [], []
    for o in outs.values():
        part = np.where(isin(o.pairs[C.S1_ID], fit_ids),
                        fold_of(o.pairs[C.S1_ID], tcfg.folds, tcfg.seed), -1)
        scored = o.pairs.assign(prob=predict_stage2(models, o.X, part))[SCORED_COLUMNS]
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
        tcfg = TwoStageConfig(**{**tc, "model": MatcherParams(**tc["model"])})
        rule = rule_from_json(json.loads((out / "rule.json").read_text()))
        models = [Matcher.load(out / f"stage2_{k}") for k in range(tcfg.folds)]
        info_path = out / "fit_info.json"
        return cls(Fitted.load(out / "stage1", cfg), models, rule, tcfg,
                   pd.read_csv(out / "tune_table.csv"),
                   json.loads(info_path.read_text()) if info_path.exists() else {})


def run_test_two_stage(cfg: PipelineConfig, ts: TwoStage, out_dir: Path = C.OUTPUT,
                       timings: dict | None = None
                       ) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Both submission files for the test split, one country at a time.

    ``candidate_pairs.tsv`` holds the pairs that pass the stage-1 filter: exactly the pairs
    stage 2 scores. Returns (matching path, candidate path, normalised S1, matches with
    ``prob``, per-S1 ``p_max`` / ``n_cands`` over the kept candidates).
    """
    timings = {} if timings is None else timings
    s1n = load_normalised("test", (1,), cfg, token_map=ts.stage1.token_map)
    matches, cands, summary = [], [], []
    for country in sorted(s1n[C.COUNTRY].unique()):
        t0 = time.perf_counter()
        s1c = s1n[(s1n[C.COUNTRY] == country).to_numpy()].reset_index(drop=True)
        poolc = load_normalised("test", (2, 3), cfg, token_map=ts.stage1.token_map,
                                country=country)
        s1c, poolc = _with_frequencies(cfg, s1c, poolc)
        pairs = prepare(s1c, poolc, cfg, _tag("test", s1c, poolc, ts.stage1.token_map))
        o = stage1_partition(pairs, s1c, poolc, ts.stage1, cfg, ts.tcfg)
        del poolc, pairs
        scored = o.pairs.assign(prob=predict_stage2(ts.models, o.X))[SCORED_COLUMNS]
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

