"""v010 full-validation: top_k 25->35 on the COMPLETE val fold (India + US).

Run from the repo root:
    .venv\Scripts\python.exe v010_validate.py
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from entity_resolution import config as C
from entity_resolution.blocking import BlockingConfig, TopKSpec, block, PASS_BITS
from entity_resolution.evaluate import blocking_report
from entity_resolution.pipeline import PipelineConfig, load_normalised, normalise_split, pool_of
from entity_resolution.split import load_fold, Fold
from entity_resolution.tracking import log_result, git_commit

EXP_DIR = C.EXPERIMENTS / "v010_entity_blocking"
DATASET_DIR = C.DATASET
CACHE_DIR = DATASET_DIR / ".cache" / "pipeline"

# top_k=35 config: only name_addr_word top_k changes (25 -> 35), all else identical
TOP_K_35_CFG = BlockingConfig(
    name_addr_word=TopKSpec("name_addr", "word", (1, 2), 35, 0.20, 0.01, max_df_abs=10_000)
)
BASELINE_CFG = BlockingConfig()  # top_k=25, the v001 config

PIPE_CFG = PipelineConfig(dataset_dir=DATASET_DIR, cache_dir=CACHE_DIR)

SWEEP_CACHE_TAG = f"val_441333_2064065_sweep"  # matches existing India parquet


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def main() -> None:
    sep = "=" * 60
    print(sep)
    print("v010 Full Validation: blocking top_k 25 -> 35")
    print(sep)

    # Load the token map learned during fit (v001 pipeline)
    token_map_path = CACHE_DIR / "token_map_984d97d9.json"
    if token_map_path.exists():
        token_map = json.loads(token_map_path.read_text())
        print(f"Token map: {len(token_map)} entries ({token_map_path.name})")
    else:
        candidates = sorted(CACHE_DIR.glob("token_map_*.json"))
        if candidates:
            token_map = json.loads(candidates[0].read_text())
            print(f"Token map: {len(token_map)} entries ({candidates[0].name})")
        else:
            print("WARNING: no token map found, using empty")
            token_map = {}

    # Ensure static normalised parquets exist (already built by v001)
    print("Checking normalised caches...")
    normalise_split("train", PIPE_CFG)
    print("Normalised caches: OK")

    # Load the full val fold
    print("\nLoading val fold...")
    t0 = time.perf_counter()
    val_fold = load_fold("val", DATASET_DIR)
    s1n = load_normalised("train", (1,), PIPE_CFG, val_fold.s1[C.ENTITY_ID], token_map)
    pooln = load_normalised("train", (2, 3), PIPE_CFG,
                            pool_of(val_fold)[C.ENTITY_ID], token_map)
    load_secs = time.perf_counter() - t0
    countries = sorted(s1n[C.COUNTRY].unique())
    print(f"  S1: {len(s1n):,} | Pool: {len(pooln):,} | True pairs: {len(val_fold.pairs):,}")
    print(f"  Countries: {countries}")
    print(f"  Load: {load_secs:.1f}s")
    gc.collect()

    # Run top_k=35 blocking (cache-aware - India parquet already exists from sweep)
    print(f"\nBlocking with top_k=35 (key={TOP_K_35_CFG.key()}) ...")
    print(f"  cache_dir: {CACHE_DIR / 'pairs'}")
    print(f"  tag: {SWEEP_CACHE_TAG}")
    t0 = time.perf_counter()
    pairs_35 = block(s1n, pooln, TOP_K_35_CFG,
                     cache_dir=CACHE_DIR / "pairs", tag=SWEEP_CACHE_TAG)
    blocking_secs = time.perf_counter() - t0
    peak_rss = rss_gb()
    print(f"  Pairs: {len(pairs_35):,}")
    print(f"  Blocking time: {blocking_secs:.1f}s ({blocking_secs/60:.2f} min)")
    print(f"  Peak RSS: {peak_rss:.2f} GB")

    # Full blocking report
    print("\nFull blocking report (top_k=35):")
    report_35 = blocking_report(pairs_35, val_fold)
    for k, v in report_35.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # Per-country recall
    print("\nPer-country recall (top_k=35):")
    country_recalls = {}
    for country in countries:
        s1c = s1n[s1n[C.COUNTRY] == country].reset_index(drop=True)
        poolc = pooln[pooln[C.COUNTRY] == country].reset_index(drop=True)
        s1c_ids = set(s1c[C.ENTITY_ID])
        poolc_ids = set(poolc[C.ENTITY_ID])
        pairs_c = pairs_35[pairs_35[C.S1_ID].isin(s1c_ids)].reset_index(drop=True)
        truth_c = val_fold.pairs[val_fold.pairs[C.S1_ID].isin(s1c_ids)].reset_index(drop=True)
        s2c = val_fold.s2[val_fold.s2[C.ENTITY_ID].isin(poolc_ids)].reset_index(drop=True)
        s3c = val_fold.s3[val_fold.s3[C.ENTITY_ID].isin(poolc_ids)].reset_index(drop=True)
        fold_c = Fold(country, s1c, s2c, s3c, truth_c)
        if len(truth_c) == 0:
            print(f"  {country}: no true pairs")
            continue
        r_c = blocking_report(pairs_c, fold_c)
        country_recalls[country] = r_c["pair_recall"]
        print(f"  {country}: pair_recall={r_c['pair_recall']:.4f} | "
              f"entity_recall={r_c['entity_recall']:.4f} | "
              f"cands_mean={r_c['candidates_mean']:.1f} | "
              f"ceiling_f05={r_c['ceiling_f_beta']:.4f}")

    # Per-pass recall
    print("\nPer-pass recall (top_k=35):")
    truth_set = set(zip(val_fold.pairs[C.S1_ID], val_fold.pairs[C.ENTITY_ID]))
    pass_recalls = {}
    for pass_name, bit in PASS_BITS.items():
        mask = (pairs_35["pass"].to_numpy() & np.uint8(bit)) != 0
        cand_s1 = pairs_35[C.S1_ID].to_numpy()[mask]
        cand_e = pairs_35[C.ENTITY_ID].to_numpy()[mask]
        in_truth = sum(1 for s1, e in zip(cand_s1, cand_e) if (s1, e) in truth_set)
        total = len(truth_set)
        recall = in_truth / total if total else float("nan")
        pass_recalls[pass_name] = recall
        print(f"  {pass_name:20s} (bit {bit:2d}): {int(mask.sum()):>9,} cands | "
              f"truth coverage {recall:.4f}")

    # Load baseline (v001, top_k=25) from its existing cache
    print("\nLoading baseline (top_k=25) pairs from cache...")
    base_cache_dir = CACHE_DIR / "pairs" / "val_441333_83c0d952_2064065_d436c334_mcd072aee"
    base_key = BASELINE_CFG.key()
    base_parts = []
    if base_cache_dir.exists():
        for country_dir in sorted(base_cache_dir.iterdir()):
            p = country_dir / f"pairs_{base_key}.parquet"
            if p.exists():
                base_parts.append(pd.read_parquet(p))
    if base_parts:
        pairs_base = pd.concat(base_parts, ignore_index=True)
        pairs_base = pairs_base.sort_values(C.S1_ID, kind="stable").reset_index(drop=True)
        print(f"  Baseline pairs loaded: {len(pairs_base):,} from cache")
        report_base = blocking_report(pairs_base, val_fold)
    else:
        print("  Baseline cache not found - running fresh baseline...")
        t0 = time.perf_counter()
        pairs_base = block(s1n, pooln, BASELINE_CFG,
                           cache_dir=CACHE_DIR / "pairs", tag=SWEEP_CACHE_TAG + "_base")
        print(f"  Baseline blocking: {time.perf_counter()-t0:.1f}s")
        report_base = blocking_report(pairs_base, val_fold)

    # Side-by-side comparison
    print(f"\n{sep}")
    print("COMPARISON: top_k=35 vs baseline (top_k=25, v001)")
    print(sep)
    metrics_to_compare = [
        "pair_recall", "entity_recall", "ceiling_f_beta",
        "candidates_mean", "candidates_p95", "candidates_max",
        "reduction_ratio", "candidate_pairs",
    ]
    for m in metrics_to_compare:
        v35 = report_35.get(m, float("nan"))
        vb = report_base.get(m, float("nan"))
        if isinstance(v35, float) and isinstance(vb, float):
            delta = v35 - vb
            print(f"  {m:25s}: baseline={vb:10.4f}  top_k=35={v35:10.4f}  delta={delta:+.4f}")
        else:
            print(f"  {m:25s}: baseline={vb}  top_k=35={v35}")

    # M2 target check
    print(f"\n{sep}")
    print("M2 TARGET CHECK")
    print(sep)
    pr = report_35["pair_recall"]
    cm = report_35["candidates_mean"]
    rt_min = blocking_secs / 60.0
    rss = peak_rss

    t_pr = pr >= 0.97
    t_cm = cm <= 40.0
    t_rt = rt_min <= 10.0
    t_rs = rss <= 3.0

    print(f"  Pair recall >= 0.97:    {pr:.4f}  {'PASS' if t_pr else 'FAIL'}")
    print(f"  Mean cands/S1 <= 40:    {cm:.2f}   {'PASS' if t_cm else 'FAIL'}")
    print(f"  Runtime <= 10 min:      {rt_min:.2f} min  {'PASS' if t_rt else 'FAIL'}")
    print(f"  Peak RAM <= 3 GB:       {rss:.2f} GB   {'PASS' if t_rs else 'FAIL'}")

    # Decision
    if t_pr and t_cm and t_rt:
        decision = "KEEP"
        reason = "all primary targets met (recall, candidates, runtime)"
    else:
        decision = "DROP"
        failed = []
        if not t_pr: failed.append(f"pair_recall={pr:.4f}<0.97")
        if not t_cm: failed.append(f"cands_mean={cm:.1f}>40")
        if not t_rt: failed.append(f"runtime={rt_min:.1f}min>10")
        reason = "; ".join(failed)

    print(f"\n  DECISION: {decision}")
    print(f"  Reason: {reason}")

    # Log results
    notes = (
        f"pair_recall={pr:.4f}; entity_recall={report_35['entity_recall']:.4f}; "
        f"ceiling_f05={report_35['ceiling_f_beta']:.4f}; "
        f"cands_mean={cm:.2f}/S1; cands_p95={report_35['candidates_p95']:.0f}; "
        f"cands_max={report_35['candidates_max']}; "
        f"runtime={blocking_secs:.0f}s; peak_rss={rss:.2f}GB; "
        f"delta_recall={pr - report_base['pair_recall']:+.4f} vs v001"
    )

    metrics_payload = {
        **report_35,
        "blocking_config_key": TOP_K_35_CFG.key(),
        "blocking_config_desc": "top_k=35 name_addr_word; all else v001 defaults",
        "baseline_pair_recall": report_base.get("pair_recall"),
        "baseline_cands_mean": report_base.get("candidates_mean"),
        "baseline_ceiling_f05": report_base.get("ceiling_f_beta"),
        "per_country_pair_recall": country_recalls,
        "per_pass_recall": pass_recalls,
        "blocking_seconds": round(blocking_secs, 2),
        "blocking_minutes": round(rt_min, 3),
        "peak_rss_gb": round(rss, 2),
        "target_pair_recall_pass": t_pr,
        "target_cands_pass": t_cm,
        "target_runtime_pass": t_rt,
        "target_ram_pass": t_rs,
        "decision": decision,
        "decision_reason": reason,
    }

    print(f"\nLogging to experiments.csv and {EXP_DIR}/metrics.json ...")
    row = log_result(
        EXP_DIR,
        change="blocking top_k 25->35 (name_addr_word pass)",
        group="A2",
        local_f05=None,       # blocking-only; do NOT report ceiling as model F0.5
        cand_recall=pr,
        notes=notes,
        owner="M2",
        parent="v001",
        decision=decision,
        metrics=metrics_payload,
    )
    print(f"  CSV row: {row}")

    # Final summary
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    print(f"  Config:          top_k=35 (name_addr_word), all else v001")
    print(f"  Pair recall:     {pr:.4f}  (baseline={report_base['pair_recall']:.4f})")
    print(f"  Entity recall:   {report_35['entity_recall']:.4f}")
    print(f"  Ceiling F0.5:    {report_35['ceiling_f_beta']:.4f}")
    print(f"  Cands mean/S1:   {cm:.2f}  (baseline={report_base['candidates_mean']:.2f})")
    print(f"  Cands p95/S1:    {report_35['candidates_p95']:.0f}")
    print(f"  Cands max/S1:    {report_35['candidates_max']}")
    print(f"  Runtime:         {blocking_secs:.1f}s ({rt_min:.2f} min)")
    print(f"  Peak RSS:        {rss:.2f} GB")
    print(f"  Decision:        {decision}")
    print(f"  Commit:          {git_commit()}")


if __name__ == "__main__":
    main()
