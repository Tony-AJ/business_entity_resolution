"""v122_full_stack: v110's recipe with every opt-in switch merged since, on the tight mock.

v110 (est_public 0.9731, the best benchmark so far) is dense blocking B6 + the name/number
pass, a GPU stage 1 trained on train-fold entities absent from the mock, M3's four groups in
stage 1, rival features and v107's rule tuning. v122 keeps all of it and turns on what was
merged after it, each built and tested on its own:

* normalisation rules v5 (French rules, ``+``/``et``, leet legal forms, own-country word);
* learned filler words: the nofill exact pass (``nofill_max_group=50``) and the nofill group;
* learned token evidence: the tok_evidence group (stage 1, so the filter reads it too);
* stage 1 in 2 bags (disjoint entity slices, each under the GPU cap) instead of one model;
* stage-2 extras on the kept pairs: interactions + missing_flags (M3, v044) and phonetic
  (M2, PR #10);
* stage 2 on fit + tune entities (out of fold), 127 leaves, 3 seeds averaged per part.

Decision (user rule): KEEP if est_public beats v110's 0.9731, else DROP. Test files are written
for a KEEP (or with ``--force-test``) to ``submissions/v122/`` and, for a KEEP, to ``output/``.

Every step is cached (normalisation, blocking per tag, stage-1 chunks and model, stage-1
outputs per country, stage-2 seed models, test stage-1 outputs), so a re-run resumes.

    .venv/Scripts/python experiments/v122_full_stack/run_v122.py > run.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from entity_resolution import config as C
from entity_resolution.blocking import TopKSpec
from entity_resolution.data import isin
from entity_resolution.decision import DecisionRule, apply_rule, one_to_one_filter, tune_expected
from entity_resolution.evaluate import blocking_report, error_samples
from entity_resolution.evidence import EvidenceConfig
from entity_resolution.features import DEFAULT_GROUPS, feature_names
from entity_resolution.mock import FP_WEIGHT, build_mock, target_shape
from entity_resolution.model import Matcher, MatcherParams, SeedMean
from entity_resolution.normalize import RULES_VERSION, NormaliseConfig
from entity_resolution.pipeline import (
    Fitted,
    PipelineConfig,
    learn_fillers,
    learn_token_evidence,
    learn_token_map,
    mem_guard,
    mock_scores,
    peak_rss_gb,
    tune_mock,
)
from entity_resolution.split import load_fold
from entity_resolution.stage1 import absent_from_mock, clean, fit_stage1, write_chunks
from entity_resolution.tracking import log_result
from entity_resolution.trainset import inner_split, sample_s1
from entity_resolution.twostage import (
    TwoStage,
    TwoStageConfig,
    fit_stage2,
    mock_scored,
    mock_stage1,
    run_test_two_stage,
)

VERSION = "v122_full_stack"
PARENT = "v110"
EXP_DIR = C.EXPERIMENTS / VERSION
ARTIFACTS = EXP_DIR / "artifacts"
M3_GROUPS = ("idf", "token_freq", "ctx_idf", "address_extra")
STAGE2_EXTRAS = ("interactions", "missing_flags", "phonetic")
N_STAGE1 = 110_000             # v110's stage-1 sample (212,941 entities over both countries)
STAGE1_BAGS = 2                # disjoint entity slices, each capped at GPU_CELLS / features rows
STAGE2_SEEDS = (42, 43, 44)
SCORE_COLS = ["f_beta", "f_tight", "est_public", "f_beta_singletons", "f_beta_matched",
              "pair_precision", "pair_recall"]
T0 = time.time()


def log(msg: str) -> None:
    """Timestamped progress line (the run log is the only view of a background run)."""
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:6.1f} min] {msg}",
          flush=True)


def configs() -> tuple[PipelineConfig, TwoStageConfig]:
    """v110's configuration with every later switch on (the module docstring's list)."""
    base = PipelineConfig()
    b6 = replace(base.blocking,
                 name_addr_word=replace(base.blocking.name_addr_word, top_k=50),
                 max_per_s1=100, exact_max_group=200,
                 addr_char=TopKSpec("addr_norm", "word", (1, 2), top_k=10, min_sim=0.3,
                                    max_df=0.01, max_df_abs=10_000),
                 cap_order="sim_first")
    v110_blocking = replace(b6, name_num_max_group=100, max_per_s1=120)
    assert v110_blocking.key() == "db1f33c4", v110_blocking.key()     # v110's blocking
    cfg = PipelineConfig(
        normalise=NormaliseConfig(learn_fillers=True),
        blocking=replace(v110_blocking, nofill_max_group=50),
        evidence=EvidenceConfig(learn=True),
        feature_groups=(*DEFAULT_GROUPS, "frequency", *M3_GROUPS, "nofill", "tok_evidence"),
        model=MatcherParams(n_estimators=4000))
    tcfg = TwoStageConfig(
        floor=0.01, max_cands=16, cohesion=False, rivals=True, train_roles=("fit", "tune"),
        model=MatcherParams(backend="xgb", device="cuda", num_leaves=127, n_estimators=4000),
        extra_groups=STAGE2_EXTRAS)
    return cfg, tcfg


S1_PARAMS = MatcherParams(backend="xgb", device="cuda", num_leaves=127, learning_rate=0.1,
                          n_estimators=3000, early_stopping=50)


def fit_or_load_stage1(cfg: PipelineConfig, train, mock, timings: dict) -> Fitted:
    """v110's GPU stage 1 (learned map, fillers and evidence attached), trained once."""
    out = ARTIFACTS / "stage1"
    token_map = learn_token_map(cfg, train)
    fillers = learn_fillers(cfg, train)
    evidence = learn_token_evidence(cfg, train)
    log(f"token map {len(token_map)} tokens; fillers {len(fillers or [])}; evidence "
        f"{'on' if evidence is not None else 'off'}")
    if (out / "config.json").exists():
        log("stage 1: loading the saved model")
        return Fitted.load(out, cfg)
    absent = absent_from_mock(mock, train)
    s1_ids = pd.Index(sample_s1(train.s1[isin(train.s1[C.ENTITY_ID], absent)],
                                N_STAGE1)[C.ENTITY_ID])
    log(f"stage 1: {len(absent):,} train entities absent from the mock, {len(s1_ids):,} sampled")
    work = cfg.cache_dir / "stage1_chunks" / f"v122_{cfg.blocking.key()}"
    t0 = time.time()
    manifest = (json.loads((work / "manifest.json").read_text())
                if (work / "manifest.json").exists()
                else write_chunks(cfg, train, s1_ids, token_map, work, timings=timings,
                                  fillers=fillers, evidence=evidence))
    timings["stage1_set_seconds"] = round(time.time() - t0, 2)
    log(f"stage-1 set: {manifest['rows']:,} pairs of {manifest['entities']:,} entities, "
        f"{len(manifest['features'])} features, positive rate "
        f"{manifest['positives'] / manifest['rows']:.4f}")
    t0 = time.time()
    s1_model = fit_stage1(manifest, S1_PARAMS, bags=STAGE1_BAGS)
    timings["stage1_fit_seconds"] = round(time.time() - t0, 2)
    log(f"stage-1 fit {timings['stage1_fit_seconds']:.0f} s: {s1_model.fit_info_}")
    stage1 = Fitted(s1_model, DecisionRule(), pd.DataFrame({"f_beta": [np.nan]}), cfg,
                    token_map, {"stage1": "xgb gpu", "fit_info": s1_model.fit_info_},
                    fillers, evidence)
    stage1.save(out)
    clean(work)
    return stage1


def fit_seeds(outs: dict, mock, tcfg: TwoStageConfig, timings: dict) -> tuple[list, dict]:
    """Stage 2 once per seed (each saved, so a re-run loads it); parts averaged by seed."""
    runs, infos = [], {}
    for seed in STAGE2_SEEDS:
        folder = ARTIFACTS / "stage2_seeds" / f"seed{seed}"
        t0 = time.time()
        if (folder / "part0").exists():
            models = [Matcher.load(folder / f"part{k}") for k in range(tcfg.folds)]
            info = json.loads((folder / "fit_info.json").read_text())
        else:
            seed_cfg = replace(tcfg, model=replace(tcfg.model, seed=seed))
            models, info = fit_stage2(outs, mock, seed_cfg)
            for k, m in enumerate(models):
                m.save(folder / f"part{k}")
            (folder / "fit_info.json").write_text(json.dumps(info, indent=1, default=float))
        runs.append(models)
        infos[f"seed{seed}"] = info
        timings[f"stage2_seed{seed}_seconds"] = round(time.time() - t0, 2)
        best = [info[f"fold{k}"].get("best_iteration") for k in range(tcfg.folds)]
        log(f"stage 2 seed {seed}: {time.time() - t0:.0f} s; best iterations {best}")
        mem_guard(f"stage 2 seed {seed}")
    models = [SeedMean([run[k] for run in runs]) for k in range(tcfg.folds)]
    return models, infos


def choose_rule(scored: pd.DataFrame, mock, cfg: PipelineConfig):
    """v107's rule choice: the threshold grid or expected-F0.5 decoding, the better on tune."""
    rule_t, table_t = tune_mock(scored, mock, cfg.grid, fp_weight=FP_WEIGHT)
    tune_part = mock.part("tune")
    rows_t = scored[isin(scored[C.S1_ID], pd.Index(tune_part.s1[C.ENTITY_ID]))]
    rule_e, table_e = tune_expected(rows_t, tune_part.s1[C.ENTITY_ID], tune_part.pairs,
                                    gammas=(0.7, 0.85, 1.0, 1.2, 1.5, 2.0),
                                    misses=(0.0, 0.05, 0.1, 0.2, 0.4), fp_weight=FP_WEIGHT)
    if table_e["f_beta"].max() > table_t["f_beta"].max():
        return rule_e, table_e, rule_t
    return rule_t, table_t, rule_t


def validate(out_dir: Path) -> str:
    """Our checker on both files of ``out_dir`` (the organisers' validator is on M1's machine)."""
    res = subprocess.run([sys.executable, "-m", "entity_resolution.submission", "--output-dir",
                          str(out_dir), "--check-ids"], capture_output=True, text=True)
    return (res.stdout + res.stderr).strip()[-2000:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force-test", action="store_true",
                    help="write the test files even when the version is dropped")
    ap.add_argument("--output-dir", type=Path, default=C.OUTPUT)
    args = ap.parse_args()

    cfg, tcfg = configs()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    parent = json.loads((C.EXPERIMENTS / "v110_m3_features" / "metrics.json").read_text())
    est_to_beat = float(parent["metrics"]["est_public"])
    stage1_cache = (cfg.cache_dir / "stage1" / f"v122_{cfg.blocking.key()}_f{tcfg.floor}"
                    f"_k{tcfg.max_cands}_a{int(tcfg.anchors)}_c{int(tcfg.cohesion)}"
                    f"_r{int(tcfg.rivals)}_x{len(tcfg.extra_groups)}")
    timings: dict[str, float] = {}
    log(f"{VERSION}: rules v{RULES_VERSION}, blocking {cfg.blocking.key()}, "
        f"{len(feature_names(cfg.feature_groups))} stage-1 features, extras {STAGE2_EXTRAS}; "
        f"est_public to beat (v110) {est_to_beat:.5f}")

    t0 = time.time()
    train = load_fold("train", columns=[C.COUNTRY])
    val = load_fold("val", columns=[C.COUNTRY])
    fit_fold, tune_fold = inner_split(train)
    fit_sample = sample_s1(fit_fold.s1, cfg.n_fit_s1)[C.ENTITY_ID]
    mock = build_mock(train, val, tune_fold.s1[C.ENTITY_ID], fit_sample, target_shape())
    del val, fit_fold, tune_fold, fit_sample
    timings["load_seconds"] = round(time.time() - t0, 2)
    log(f"mock built: {mock.info}")

    stage1 = fit_or_load_stage1(cfg, train, mock, timings)
    del train
    mem_guard("stage 1")

    t0 = time.time()
    outs = mock_stage1(cfg, stage1, mock, tcfg, timings=timings,
                       cache_dir=stage1_cache / "mock")
    timings["stage1_mock_seconds"] = round(time.time() - t0, 2)
    kept = pd.concat([o.pairs for o in outs.values()], ignore_index=True)
    filter_report = pd.DataFrame({r: blocking_report(kept, mock.part(r))
                                  for r in ("tune", "val")}).T
    del kept
    log(f"mock stage-1 pass {timings['stage1_mock_seconds']:.0f} s; pairs "
        f"{sum(o.n_all for o in outs.values()):,}; frame {next(iter(outs.values())).X.shape}\n"
        f"{filter_report[['pair_recall', 'entity_recall', 'ceiling_f_beta', 'candidates_mean']]}")

    # stage 1 alone (p1 through the threshold grid), the single-stage reference
    keep_ids = pd.Index(mock.fold.s1[C.ENTITY_ID][mock.role.isin(["tune", "val"]).to_numpy()])
    p1_scored = []
    for o in outs.values():
        s = one_to_one_filter(o.pairs.assign(prob=o.X["p1"].to_numpy()))
        p1_scored.append(s[isin(s[C.S1_ID], keep_ids)])
    p1_scored = pd.concat(p1_scored, ignore_index=True)
    rule1, _ = tune_mock(p1_scored, mock, cfg.grid, fp_weight=FP_WEIGHT)
    single = mock_scores(p1_scored, mock, rule1)
    del p1_scored
    log(f"stage 1 alone: est_public {single.loc['all', 'est_public']:.5f}, "
        f"F0.5 {single.loc['all', 'f_beta']:.5f}")

    t0 = time.time()
    models, seed_info = fit_seeds(outs, mock, tcfg, timings)
    scored, _ = mock_scored(outs, models, mock, tcfg)
    rule, table, rule_t = choose_rule(scored, mock, cfg)
    res = mock_scores(scored, mock, rule)
    timings["stage2_seconds"] = round(time.time() - t0, 2)
    scored.to_parquet(ARTIFACTS / "mock_scored.parquet", index=False)
    est = float(res.loc["all", "est_public"])
    log(f"two-stage: est_public {est:.5f} (v110 {est_to_beat:.5f}, delta "
        f"{est - est_to_beat:+.5f}); rule {rule}\n{res[SCORE_COLS]}")

    # one seed alone: what the seed averaging adds
    one = [m.models[0] for m in models]
    scored_one, _ = mock_scored(outs, one, mock, tcfg)
    rule_one, _, _ = choose_rule(scored_one, mock, cfg)
    res_one = mock_scores(scored_one, mock, rule_one)
    del scored_one
    log(f"seed {STAGE2_SEEDS[0]} alone: est_public {res_one.loc['all', 'est_public']:.5f}")

    part = mock.part("val")
    matches = apply_rule(scored[isin(scored[C.S1_ID], pd.Index(part.s1[C.ENTITY_ID]))], rule)
    counts = {k: len(error_samples(matches, part, k, n=10**9))
              for k in ("false_merge", "missed", "false_singleton", "singleton_merge")}
    importance = pd.concat([m.importance() for m in models], axis=1).mean(axis=1)
    importance = importance.sort_values(ascending=False)
    extra_cols = feature_names(STAGE2_EXTRAS)
    log(f"errors {counts}; extras' gain share "
        f"{float(importance.reindex(extra_cols).fillna(0).sum()):.5f}; top\n"
        f"{importance.head(12)}")

    decision = "KEEP" if est > est_to_beat else "DROP"
    ts = TwoStage(stage1, models, rule, tcfg, table,
                  {"fit": seed_info, "stage1_fit": stage1.info.get("fit_info"),
                   "filter_report": filter_report.to_dict("index")})
    ts.save(ARTIFACTS / "two_stage")
    record = {
        "hypothesis": "every switch merged after v110 (rules v5, fillers, token evidence, "
                      "stage-1 bags, stage-2 extras, stage 2 on fit + tune with 127 leaves "
                      "and 3 seeds) lifts v110's est_public",
        "rules_version": RULES_VERSION, "feature_groups": list(cfg.feature_groups),
        "blocking_config": asdict(cfg.blocking), "blocking_key": cfg.blocking.key(),
        "stage1_params": asdict(S1_PARAMS), "stage1_bags": STAGE1_BAGS,
        "stage1_fit_info": stage1.info.get("fit_info"), "two_stage": tcfg.record(),
        "stage2_seeds": list(STAGE2_SEEDS), "stage2_fit_info": seed_info,
        "rule": asdict(rule), "rule_kind": type(rule).__name__, "fp_weight": FP_WEIGHT,
        "threshold_rule": asdict(rule_t),
        **{k: float(res.loc["all", k]) for k in SCORE_COLS},
        **{f"mock_{c}": float(res.loc[c, "f_beta"]) for c in res.index if c != "all"},
        **{f"est_public_{c}": float(res.loc[c, "est_public"]) for c in res.index if c != "all"},
        "single_stage_est_public": float(single.loc["all", "est_public"]),
        "single_stage_f_beta": float(single.loc["all", "f_beta"]),
        "one_seed_est_public": float(res_one.loc["all", "est_public"]),
        "cand_recall_val": float(filter_report.loc["val", "pair_recall"]),
        "cands_mean_val": float(filter_report.loc["val", "candidates_mean"]),
        "errors_mock": counts, "extras_gain_share":
            float(importance.reindex(extra_cols).fillna(0).sum()),
        "top_importance": importance.head(25).to_dict(),
        "parent_est_public": est_to_beat, "delta_vs_v110": est - est_to_beat,
        "stage1_cache": str(stage1_cache), **timings, "peak_rss_gb": peak_rss_gb(),
        "decision": decision,
    }
    row = log_result(
        EXP_DIR, change=("v110 + rules v5, learned fillers (nofill pass + group), token "
                         "evidence, 2 stage-1 bags, stage-2 extras (interactions, "
                         "missing_flags, phonetic), stage 2 on fit + tune, 127 leaves, 3 seeds"),
        group="INT", mock_f05=float(res.loc["all", "f_beta"]),
        cand_recall=float(filter_report.loc["val", "pair_recall"]),
        notes=(f"est_public {est:.4f} (v110 {est_to_beat:.4f}, {est - est_to_beat:+.4f}); "
               f"stage 1 alone {single.loc['all', 'est_public']:.4f}; one seed "
               f"{res_one.loc['all', 'est_public']:.4f}; cands "
               f"{filter_report.loc['val', 'candidates_mean']:.1f}/S1; blocking "
               f"{cfg.blocking.key()}"),
        metrics=record, owner="M3", parent=PARENT, decision=decision)
    log(f"logged {row}")

    if decision != "KEEP" and not args.force_test:
        log(f"DROP: est_public {est:.5f} <= v110 {est_to_beat:.5f}; no test files")
        return
    t0 = time.time()
    sub_dir = C.ROOT / "submissions" / "v122"
    match_path, cand_path, s1n_test, test_matches, test_summary = run_test_two_stage(
        cfg, ts, out_dir=sub_dir, timings=timings, cache_dir=stage1_cache / "test")
    log(f"test files {time.time() - t0:.0f} s in {sub_dir}")
    country_of = s1n_test.set_index(C.ENTITY_ID)[C.COUNTRY]
    n_s1 = s1n_test.groupby(C.COUNTRY).size()
    by = test_matches[C.S1_ID].map(country_of)
    log("test by country:\n" + str(pd.DataFrame({
        "s1": n_s1,
        "cands_per_s1": test_summary["n_cands"].groupby(
            test_summary.index.map(country_of)).sum() / n_s1,
        "matched_share": test_matches.groupby(by)[C.S1_ID].nunique() / n_s1,
        "matches_per_s1": test_matches.groupby(by).size() / n_s1})))
    log("checker (submissions/v122): " + validate(sub_dir))
    if decision == "KEEP":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for p in (match_path, cand_path):
            shutil.copy2(p, args.output_dir / p.name)
        log(f"copied to {args.output_dir}; checker: " + validate(args.output_dir))
    log(f"done: total {time.time() - T0:.0f} s, peak RSS {peak_rss_gb()} GB")


if __name__ == "__main__":
    main()
