"""The final submission end to end: v126_decoy_groups on v110's stage 1 (public F0.5 0.972).

    python -m entity_resolution.final                  # stage A, then stage B (~3.5 h on a GPU)
    python -m entity_resolution.final --stage b        # stage B alone, from stage A's caches
    python -m entity_resolution.final --verify output  # sha256 of both files vs the upload

Stage A: candidates and stage 1, as ``experiments/v110_m3_features`` ran them (~3 h):
  1. every record normalised (rules v3), the transliteration token map learned from the train
     fold's true pairs, the fixed validation split and the test-shaped mock fold;
  2. per-country blocking of the train-fold entities absent from the mock (212,941 entities,
     14.9M candidate pairs, 76 pair features) and the stage-1 matcher (XGBoost, GPU);
  3. stage 1 over every candidate of the mock fold and of the test split: the filter (p1 >=
     0.01 among the entity's 16 best, i.e. ``candidate_pairs.tsv``) and the competition,
     anchor and rival features of the kept pairs, cached per country.
Stage B: final matcher and decision layer, as ``experiments/v126_decoy_groups/run_v126.py``
ran them (~25 min):
  4. decoy-signature features of the kept pairs (``decoy.py``);
  5. stage 2 (XGBoost, 127 leaves) cross-fitted on the mock fold's fit + tune entities;
  6. isotonic calibration and the tight decoder's (w, miss), both chosen on the mock's tune
     entities with false merges costing x3, then scored on its val entities;
  7. test inference: ``output/matching_results.tsv``, ``output/candidate_pairs.tsv``, both
     validators and the sha256 of each file against the uploaded one (``SUBMITTED_SHA256``).

Every intermediate is cached under ``--cache-dir``, so a stage can run alone. Models and a
JSON report go to ``--models-dir`` (``models/final``). The defaults reproduce the uploaded
files byte for byte with the pinned requirements and a CUDA GPU; ``--device cpu`` trains the
same models without one (slower; tree splits may differ slightly from the GPU's).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .blocking import TopKSpec
from .data import isin
from .decision import DecisionRule
from .evaluate import blocking_report
from .features import DEFAULT_GROUPS
from .mock import MockFold, build_mock, target_shape
from .model import MatcherParams
from .normalize import NormaliseConfig
from .pipeline import (
    Fitted,
    PipelineConfig,
    learn_token_map,
    mem_guard,
    peak_rss_gb,
    pool_of,
)
from .split import Fold, load_fold
from .stage1 import absent_from_mock, clean, fit_stage1, write_chunks
from .trainset import inner_split, sample_s1
from .twostage import (
    TwoStageConfig,
    mock_stage1,
    stage1_test_outputs,
)

VERSION = "v126_decoy_groups"      # the uploaded version (x3 arm), on v110_m3_features
BLOCKING_KEY = "db1f33c4"          # v110's blocking configuration (asserted below)


def stage1_pipeline() -> PipelineConfig:
    """v110's pipeline configuration: normalisation rules v3, the 13 stage-1 feature groups
    and blocking B6 plus the name + address-number pass, at most 120 candidates per S1."""
    base = PipelineConfig().blocking
    b6 = replace(base, name_addr_word=replace(base.name_addr_word, top_k=50),
                 max_per_s1=100, exact_max_group=200,
                 addr_char=TopKSpec("addr_norm", "word", (1, 2), top_k=10, min_sim=0.3,
                                    max_df=0.01, max_df_abs=10_000),
                 cap_order="sim_first")
    cfg = PipelineConfig(
        normalise=NormaliseConfig(rules=3),
        feature_groups=(*DEFAULT_GROUPS, "frequency", "idf", "token_freq", "ctx_idf",
                        "address_extra"),
        model=MatcherParams(n_estimators=4000),    # as v110 recorded it; stage 1 is below
        blocking=replace(b6, name_num_max_group=100, max_per_s1=120))
    if cfg.blocking.key() != BLOCKING_KEY:     # the blocking code or its defaults drifted
        raise RuntimeError(f"blocking key {cfg.blocking.key()} is not v110's {BLOCKING_KEY}")
    return cfg


@dataclass(frozen=True)
class FinalConfig:
    """Every setting of the final version; the defaults are the submission's."""

    pipeline: PipelineConfig = field(default_factory=stage1_pipeline)
    # stage-1 training entities: train-fold S1 absent from the mock, sampled by id hash
    # (sample_s1, 110,000 -> 212,941 entities: the mock already dropped the seed-7 fit sample)
    n_stage1: int = 110_000
    stage1_model: MatcherParams = field(default_factory=lambda: MatcherParams(
        backend="xgb", device="cuda", num_leaves=127, learning_rate=0.1, n_estimators=3000,
        early_stopping=50))
    stage2: TwoStageConfig = field(default_factory=lambda: TwoStageConfig(
        floor=0.01, max_cands=16, cohesion=False, rivals=True, train_roles=("fit", "tune"),
        model=MatcherParams(backend="xgb", device="cuda", num_leaves=127, learning_rate=0.1,
                            n_estimators=3000, early_stopping=60)))
    select_w: float = 3.0              # the decoder is chosen for false merges costing x3
    decode_w: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
    misses: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.4)
    report_w: tuple[float, ...] = (1.45, 3.0, 6.0)   # tight weights reported on val entities

    def with_device(self, device: str) -> FinalConfig:
        """The same configuration with both matchers on ``device`` ("cuda" or "cpu")."""
        return replace(self, stage1_model=replace(self.stage1_model, device=device),
                       stage2=replace(self.stage2, model=replace(self.stage2.model,
                                                                 device=device)))

    def stage1_dir(self) -> Path:
        """Stage-1 outputs of the mock fold (``mock/``) and the test split (``test/``)."""
        t = self.stage2
        return (self.pipeline.cache_dir / "stage1" / f"v110_{self.pipeline.blocking.key()}"
                f"_f{t.floor}_k{t.max_cands}_a{int(t.anchors)}_c{int(t.cohesion)}"
                f"_r{int(t.rivals)}")


T0 = time.time()


def log(msg: str) -> None:
    """Timestamped progress line (minutes since the start of the run)."""
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def _mock(cfg: PipelineConfig) -> tuple[MockFold, Fold]:
    """The test-shaped mock fold and the train fold it was cut from (ids and seeds only)."""
    train = load_fold("train", cfg.dataset_dir, columns=[C.COUNTRY])
    val = load_fold("val", cfg.dataset_dir, columns=[C.COUNTRY])
    fit_fold, tune_fold = inner_split(train)
    mock = build_mock(train, val, tune_fold.s1[C.ENTITY_ID],
                      sample_s1(fit_fold.s1, cfg.n_fit_s1)[C.ENTITY_ID],
                      target_shape(cfg.dataset_dir))
    return mock, train


# ---------------------------------------------------------------- stage A ----
def stage_a(fc: FinalConfig, models_dir: Path) -> dict:
    """Normalisation, token map, mock fold, blocking, stage 1 and its cached outputs.

    Returns (and writes to ``models_dir/stage_a.json``) the stage-1 fit and the candidate
    recall of the filtered mock candidates.
    """
    cfg, tcfg = fc.pipeline, fc.stage2
    models_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    mock, train = _mock(cfg)
    token_map = learn_token_map(cfg, train)
    log(f"mock fold: {len(mock.fold.s1):,} S1, {len(pool_of(mock.fold)):,} pool records; "
        f"token map {len(token_map)} tokens")
    absent = absent_from_mock(mock, train)
    s1_ids = pd.Index(sample_s1(train.s1[isin(train.s1[C.ENTITY_ID], absent)],
                                fc.n_stage1)[C.ENTITY_ID])
    log(f"stage-1 training entities: {len(s1_ids):,} of {len(absent):,} absent from the mock")

    work = cfg.cache_dir / "stage1_chunks" / f"v110_{cfg.blocking.key()}"
    t0 = time.time()
    manifest = (json.loads((work / "manifest.json").read_text())
                if (work / "manifest.json").exists()
                else write_chunks(cfg, train, s1_ids, token_map, work, timings=timings))
    timings["stage1_set_seconds"] = round(time.time() - t0, 2)
    log(f"stage-1 training set: {manifest['rows']:,} pairs, "
        f"{len(manifest['features'])} features")
    t0 = time.time()
    model = fit_stage1(manifest, fc.stage1_model)
    timings["stage1_fit_seconds"] = round(time.time() - t0, 2)
    stage1 = Fitted(model, DecisionRule(), pd.DataFrame({"f_beta": [np.nan]}), cfg, token_map,
                    {"stage1": f"xgb {fc.stage1_model.device}", "fit_info": model.fit_info_})
    stage1.save(models_dir / "stage1")
    clean(work)
    log(f"stage 1 fitted: {json.dumps(model.fit_info_)}")
    del train
    mem_guard("stage A fit")

    outs = mock_stage1(cfg, stage1, mock, tcfg, timings=timings,
                       cache_dir=fc.stage1_dir() / "mock")
    kept = pd.concat([o.pairs for o in outs.values()], ignore_index=True)
    filtered = {r: blocking_report(kept, mock.part(r)) for r in ("tune", "val")}
    del outs, kept
    log(f"mock stage-1 outputs cached; filtered candidates (val): {json.dumps(filtered['val'])}")
    countries = stage1_test_outputs(cfg, stage1, tcfg, fc.stage1_dir() / "test", timings)
    log(f"test stage-1 outputs cached: {countries}")
    report = {"stage1_entities": len(s1_ids), "stage1_rows": manifest["rows"],
              "stage1_fit_info": model.fit_info_, "filtered_candidates": filtered,
              **timings, "peak_rss_gb": peak_rss_gb()}
    (models_dir / "stage_a.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    """Command line: stage A (stage B, the decision layer and the checks come next)."""
    ap = argparse.ArgumentParser(prog="python -m entity_resolution.final",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                    help="where XGBoost trains (default cuda)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="pipeline caches (default <dataset>/.cache/pipeline)")
    ap.add_argument("--models-dir", type=Path, default=C.ROOT / "models" / "final",
                    help="trained models and the run reports (default models/final)")
    args = ap.parse_args(argv)
    fc = FinalConfig().with_device(args.device)
    if args.cache_dir is not None:
        fc = replace(fc, pipeline=replace(fc.pipeline, cache_dir=args.cache_dir))
    log(f"final pipeline {VERSION}: stage a, device {args.device}, cache "
        f"{fc.pipeline.cache_dir}, models {args.models_dir}")
    stage_a(fc, args.models_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
