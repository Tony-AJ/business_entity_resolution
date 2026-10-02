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
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

from . import config as C
from .blocking import TopKSpec
from .data import isin, load_source
from .decision import DecisionRule, decode_tight
from .decoy import DECOY_COLUMNS, decoy_features, with_columns
from .evaluate import (
    blocking_report,
    entity_counts,
    entity_tight_from_counts,
    error_samples,
    macro_tight,
    score_pairs,
)
from .features import DEFAULT_GROUPS
from .mock import MockFold, build_mock, target_shape
from .model import MatcherParams
from .normalize import NormaliseConfig, apply_token_map
from .pipeline import (
    Fitted,
    PipelineConfig,
    learn_token_map,
    mem_guard,
    normalise_split,
    peak_rss_gb,
    pool_of,
)
from .split import Fold, load_fold
from .stage1 import absent_from_mock, clean, fit_stage1, write_chunks
from .submission import write_pairs
from .trainset import inner_split, label_pairs, sample_s1
from .twostage import (
    Stage1Output,
    TwoStageConfig,
    fit_stage2,
    load_stage1,
    mock_scored,
    mock_stage1,
    stage1_test_outputs,
)

VERSION = "v126_decoy_groups"      # the uploaded version (x3 arm), on v110_m3_features
# sha256 of the files uploaded to the leaderboard (submission 09, public F0.5 0.972)
SUBMITTED_SHA256 = {
    C.MATCHING_FILE: "617c9822921c959c6dbee6fceeda42642afb1974d23cbee4c86548ab86605fa5",
    C.CANDIDATE_FILE: "7164d366a0c00209003db6affadcacd1448479c76798a74ee7936e9ed6a12754",
}
BLOCKING_KEY = "db1f33c4"          # v110's blocking configuration (asserted below)
# stage B reads stage A's records as run_v126.py read them: the rules-v3 records of the
# normalisation cache, with the token map re-applied to the non-Latin rows by the current
# (v5) rules; stage A itself runs rules v3 throughout
TOKEN_MAP_RULES = NormaliseConfig()
RECORD_COLUMNS = [C.COUNTRY, "name_core", "addr_norm", "addr_nums", "name_norm", "non_latin"]


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


# ---------------------------------------------------------------- stage B ----
def _records(cfg: PipelineConfig, split: str, sources: tuple[int, ...], token_map: dict,
             ids: pd.Series | None = None, country: str | None = None) -> pd.DataFrame:
    """Stage A's normalised records (``RECORD_COLUMNS``), filtered in Arrow, token map
    applied with ``TOKEN_MAP_RULES``: exactly what run_v126.py read."""
    paths = normalise_split(split, cfg)
    frames = []
    for s in sources:
        tbl = pq.read_table(paths[s], columns=[C.ENTITY_ID, *RECORD_COLUMNS])
        if country is not None:
            tbl = tbl.filter(pc.equal(tbl[C.COUNTRY], country))
        if ids is not None:
            tbl = tbl.filter(pc.is_in(tbl[C.ENTITY_ID], value_set=pa.array(
                pd.Index(ids).astype("str"), type=tbl.schema.field(C.ENTITY_ID).type)))
        frames.append(tbl.to_pandas())
        del tbl
    out = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return apply_token_map(out, token_map, TOKEN_MAP_RULES)


def _stage1_output(path: Path) -> Stage1Output:
    """A cached stage-1 output, with a clear error when stage A has not run."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run stage A first (--stage a)")
    return load_stage1(path)


def _decoder(scored: pd.DataFrame, mock: MockFold, fc: FinalConfig
             ) -> tuple[IsotonicRegression, dict, pd.DataFrame]:
    """Isotonic calibration on the tune entities, the (w, miss) grid scored there by the
    tight score at ``fc.select_w``, and the chosen decoder's scores on the val entities."""
    tune_part, val_part = mock.part("tune"), mock.part("val")
    rows_t = scored[isin(scored[C.S1_ID], pd.Index(tune_part.s1[C.ENTITY_ID]))]
    rows_v = scored[isin(scored[C.S1_ID], pd.Index(val_part.s1[C.ENTITY_ID]))]
    y_t = label_pairs(rows_t[[C.S1_ID, C.ENTITY_ID]], mock.fold.pairs)["label"].to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(rows_t["prob"].to_numpy(np.float64), y_t)
    cal_t = rows_t.assign(prob=iso.predict(rows_t["prob"].to_numpy(np.float64)))
    cal_v = rows_v.assign(prob=iso.predict(rows_v["prob"].to_numpy(np.float64)))
    grid = []
    for w in fc.decode_w:
        for miss in fc.misses:
            counts = entity_counts(decode_tight(cal_t, w, miss), tune_part)
            grid.append({"w": w, "miss": miss,
                         **{f"tune_tight{s:g}": float(entity_tight_from_counts(*counts, s).mean())
                            for s in (1.0, fc.select_w)}})
    grid = pd.DataFrame(grid)
    best = grid.sort_values(f"tune_tight{fc.select_w:g}", ascending=False).iloc[0]
    w, miss = float(best["w"]), float(best["miss"])
    m_v = decode_tight(cal_v, w, miss)
    s = score_pairs(m_v, val_part)
    chosen = {"w": w, "miss": miss, "f_beta": float(s["f_beta"]),
              "pair_precision": float(s["pair_precision"]),
              "pair_recall": float(s["pair_recall"]),
              **{f"tight{t:g}": macro_tight(m_v, val_part, t) for t in fc.report_w},
              "errors": {k: len(error_samples(m_v, val_part, k, n=10**9))
                         for k in ("false_merge", "singleton_merge", "missed",
                                   "false_singleton")}}
    return iso, chosen, grid


def stage_b(fc: FinalConfig, models_dir: Path, out_dir: Path = C.OUTPUT) -> dict:
    """Decoy features, stage 2, calibration and decoder, test inference, both output files.

    Returns (and writes to ``models_dir/final_report.json``) the mock scores of the chosen
    decoder, the test counts and the sha256 of each written file against the upload.
    """
    cfg, tcfg = fc.pipeline, fc.stage2
    models_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    mock, train = _mock(cfg)
    token_map = learn_token_map(cfg, train)          # cached by stage A
    del train
    log(f"mock fold rebuilt; token map {len(token_map)} tokens")

    outs = {}
    t0 = time.time()
    for country in sorted(mock.fold.s1[C.COUNTRY].unique()):
        o = _stage1_output(fc.stage1_dir() / "mock" / f"mock_{country}.parquet")
        s1_ids = mock.fold.s1[C.ENTITY_ID][(mock.fold.s1[C.COUNTRY] == country).to_numpy()]
        s1n = _records(cfg, "train", (1,), token_map, ids=s1_ids)
        pooln = _records(cfg, "train", (2, 3), token_map, ids=pool_of(mock.fold)[C.ENTITY_ID],
                         country=country)
        outs[country] = with_columns(o, decoy_features(o, s1n, pooln))
        log(f"mock {country}: {len(o.pairs):,} kept pairs, {len(DECOY_COLUMNS)} decoy columns")
        del o, s1n, pooln
        mem_guard(f"stage B mock {country}")
    timings["mock_features_seconds"] = round(time.time() - t0, 2)

    t0 = time.time()
    models, fit_info = fit_stage2(outs, mock, tcfg)
    timings["stage2_fit_seconds"] = round(time.time() - t0, 2)
    for k, m in enumerate(models):
        m.save(models_dir / f"stage2_{k}")
    scored, _ = mock_scored(outs, models, mock, tcfg)
    importance = pd.concat([m.importance() for m in models], axis=1).mean(axis=1)
    importance = importance.sort_values(ascending=False)
    log(f"stage 2: best iterations "
        f"{ {k: v.get('best_iteration') for k, v in fit_info.items() if k.startswith('fold')} }"
        f"; decoy columns' gain share {float(importance.reindex(DECOY_COLUMNS).sum()):.4f}")
    del outs
    mem_guard("stage B mock scored")

    iso, chosen, grid = _decoder(scored, mock, fc)
    grid.to_csv(models_dir / "decoder_grid.csv", index=False)
    (models_dir / "isotonic.json").write_text(json.dumps(
        {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}) + "\n")
    log(f"decoder (chosen on tune for false merges x{fc.select_w:g}): {json.dumps(chosen)}")
    del scored
    mem_guard("stage B decoder")

    t0 = time.time()
    s1n_all = _records(cfg, "test", (1,), token_map)
    countries = sorted(s1n_all[C.COUNTRY].unique())
    test_parts = []
    for country in countries:
        o = _stage1_output(fc.stage1_dir() / "test" / f"test_{country}.parquet")
        s1n = s1n_all[(s1n_all[C.COUNTRY] == country).to_numpy()]
        pooln = _records(cfg, "test", (2, 3), token_map, country=country)
        o = with_columns(o, decoy_features(o, s1n, pooln))
        del s1n, pooln
        names = models[0].feature_names_
        X = o.X if list(o.X.columns) == names else o.X[names]
        prob = np.mean([m.predict_proba(X) for m in models], axis=0).astype(np.float32)
        test_parts.append(o.pairs.assign(prob=prob, country=country))
        log(f"test {country}: {len(o.pairs):,} candidate pairs scored")
        del o, X
        mem_guard(f"stage B test {country}")
    del s1n_all
    test = pd.concat(test_parts, ignore_index=True)
    del test_parts
    cand = test[[C.S1_ID, C.ENTITY_ID]]
    s1_ids = load_source("test", 1, cfg.dataset_dir, columns=[C.ENTITY_ID])[C.ENTITY_ID]
    parts = []
    for country in countries:
        sc = test.loc[test["country"] == country, [C.S1_ID, C.ENTITY_ID, "prob"]]
        sc = sc.assign(prob=iso.predict(sc["prob"].to_numpy(np.float64)))
        parts.append(decode_tight(sc, chosen["w"], chosen["miss"]))
    matches = pd.concat(parts, ignore_index=True).sort_values(
        [C.S1_ID, "prob"], ascending=[True, False], kind="stable")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_pairs(matches[[C.S1_ID, C.ENTITY_ID]], cand, s1_ids.tolist(), out_dir)
    timings["test_seconds"] = round(time.time() - t0, 2)
    country_of = test.drop_duplicates(C.S1_ID).set_index(C.S1_ID)["country"]
    test_info = {
        "s1": int(len(s1_ids)), "candidate_pairs": int(len(cand)),
        "candidates_by_country": {k: int(v) for k, v in test.groupby("country").size().items()},
        "matched_pairs": int(len(matches)), "s1_matched": int(matches[C.S1_ID].nunique()),
        "matched_by_country": {k: int(v) for k, v in
                               matches.groupby(matches[C.S1_ID].map(country_of)).size().items()},
    }
    log(f"test: {json.dumps(test_info)}")
    checks = check_outputs(out_dir, cfg.dataset_dir)
    report = {"version": VERSION, "config": {
                  "n_stage1": fc.n_stage1, "stage1_model": asdict(fc.stage1_model),
                  "stage2": tcfg.record(), "select_w": fc.select_w},
              "stage2_fit_info": fit_info,
              "decoy_gain_share": float(importance.reindex(DECOY_COLUMNS).sum()),
              "top_importance": importance.head(20).to_dict(), "mock_val": chosen,
              "test": test_info, "checks": checks, **timings, "peak_rss_gb": peak_rss_gb()}
    (models_dir / "final_report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n")
    return report


# ----------------------------------------------------------------- checks ----
def sha256(path: Path) -> str:
    """Hex sha256 of a file, read in 1 MB blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def compare_to_submitted(out_dir: Path) -> dict[str, dict]:
    """sha256 of both files in ``out_dir`` against ``SUBMITTED_SHA256``."""
    out = {}
    for name, want in SUBMITTED_SHA256.items():
        path = out_dir / name
        got = sha256(path) if path.exists() else None
        out[name] = {"sha256": got, "identical_to_upload": got == want}
    return out


def check_outputs(out_dir: Path, dataset_dir: Path = C.DATASET) -> dict:
    """Our checker (ids checked), the organisers' validator when present, and the sha256
    comparison with the uploaded files; every result is logged."""
    res = subprocess.run([sys.executable, "-m", "entity_resolution.submission", "--output-dir",
                          str(out_dir), "--dataset-dir", str(dataset_dir), "--check-ids"],
                         capture_output=True, text=True)
    checks = {"our_checker": "PASS" if res.returncode == 0 else (res.stdout + res.stderr)[-500:]}
    if C.OFFICIAL_VALIDATOR.exists():
        res = subprocess.run([sys.executable, str(C.OFFICIAL_VALIDATOR), "--matching",
                              str(out_dir / C.MATCHING_FILE), "--candidate",
                              str(out_dir / C.CANDIDATE_FILE), "--test-dir",
                              str(dataset_dir / "test")], capture_output=True, text=True)
        checks["official_validator"] = ("PASS" if res.returncode == 0
                                        else (res.stdout + res.stderr)[-500:])
    else:
        checks["official_validator"] = f"not run: {C.OFFICIAL_VALIDATOR} not found"
    checks["vs_upload"] = compare_to_submitted(out_dir)
    log(f"checks: {json.dumps(checks)}")
    return checks


def main(argv: list[str] | None = None) -> int:
    """Command line: run stage A and/or B, or compare existing outputs with the upload."""
    ap = argparse.ArgumentParser(prog="python -m entity_resolution.final",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=("a", "b", "all"), default="all",
                    help="a: candidates + stage 1; b: stage 2 + decision + output (default all)")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                    help="where XGBoost trains (default cuda)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="pipeline caches (default <dataset>/.cache/pipeline)")
    ap.add_argument("--models-dir", type=Path, default=C.ROOT / "models" / "final",
                    help="trained models and the run reports (default models/final)")
    ap.add_argument("--out-dir", type=Path, default=C.OUTPUT,
                    help="where both TSVs are written (default output/)")
    ap.add_argument("--verify", type=Path, metavar="DIR", default=None,
                    help="only compare DIR's two TSVs with the uploaded files (sha256)")
    args = ap.parse_args(argv)
    if args.verify is not None:
        result = compare_to_submitted(args.verify)
        for name, r in result.items():
            print(f"{name}: {r['sha256']} "
                  f"{'IDENTICAL to the upload' if r['identical_to_upload'] else 'DIFFERS'}")
        return 0 if all(r["identical_to_upload"] for r in result.values()) else 1
    fc = FinalConfig().with_device(args.device)
    if args.cache_dir is not None:
        fc = replace(fc, pipeline=replace(fc.pipeline, cache_dir=args.cache_dir))
    args.models_dir.mkdir(parents=True, exist_ok=True)
    log(f"final pipeline {VERSION}: stage {args.stage}, device {args.device}, cache "
        f"{fc.pipeline.cache_dir}, models {args.models_dir}, output {args.out_dir}")
    if args.stage in ("a", "all"):
        stage_a(fc, args.models_dir)
    if args.stage in ("b", "all"):
        report = stage_b(fc, args.models_dir, args.out_dir)
        same = all(r["identical_to_upload"] for r in report["checks"]["vs_upload"].values())
        log(f"done: output {'IDENTICAL to' if same else 'differs from'} the uploaded files; "
            f"peak RSS {peak_rss_gb()} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
