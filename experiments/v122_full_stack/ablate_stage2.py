"""v122 stage-2 ablation: which of v122's stage-2 changes carry its est_public.

Reads v122's cached stage-1 outputs of the mock (``run_v122.py`` must have finished its mock
pass) and re-trains only stage 2, one seed per arm:

    A  v110's stage 2: fit entities, 63 leaves, no stage-2 extras
    B  A on fit + tune entities
    C  B with 127 leaves
    D  C + the stage-2 extras (interactions, missing_flags, phonetic): v122 with one seed

Each arm gets v107's rule choice and est_public on the mock's val entities; the table goes to
``artifacts/ablation_stage2.json``. Stage 1 (rules v5, fillers, token evidence, bags) is common
to every arm, so the table separates stage 2's share of v122's change from stage 1's.

    .venv/Scripts/python experiments/v122_full_stack/ablate_stage2.py > ablation.log 2>&1
"""
from __future__ import annotations

import importlib.util
import json
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from entity_resolution import config as C
from entity_resolution.features import feature_names
from entity_resolution.mock import build_mock, target_shape
from entity_resolution.pipeline import Fitted, mem_guard, mock_scores
from entity_resolution.split import load_fold
from entity_resolution.trainset import inner_split, sample_s1
from entity_resolution.twostage import fit_stage2, mock_scored, mock_stage1

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("run_v122", HERE / "run_v122.py")
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def main() -> None:
    cfg, tcfg = run.configs()
    cache = (cfg.cache_dir / "stage1" / f"v122_{cfg.blocking.key()}_f{tcfg.floor}"
             f"_k{tcfg.max_cands}_a{int(tcfg.anchors)}_c{int(tcfg.cohesion)}"
             f"_r{int(tcfg.rivals)}_x{len(tcfg.extra_groups)}" / "mock")
    if not any(cache.glob("mock_*.parquet")):
        raise SystemExit(f"no stage-1 outputs in {cache}: run run_v122.py first")
    train = load_fold("train", columns=[C.COUNTRY])
    val = load_fold("val", columns=[C.COUNTRY])
    fit_fold, tune_fold = inner_split(train)
    mock = build_mock(train, val, tune_fold.s1[C.ENTITY_ID],
                      sample_s1(fit_fold.s1, cfg.n_fit_s1)[C.ENTITY_ID], target_shape())
    del train, val, fit_fold, tune_fold
    stage1 = Fitted.load(run.ARTIFACTS / "stage1", cfg)
    outs = mock_stage1(cfg, stage1, mock, tcfg, cache_dir=cache)
    extras = set(feature_names(run.STAGE2_EXTRAS))
    base_cols = [c for c in next(iter(outs.values())).X.columns if c not in extras]
    one_seed = replace(tcfg.model, seed=run.STAGE2_SEEDS[0])
    arms = {
        "A: v110 stage 2 (fit, 63 leaves, no extras)":
            (base_cols, replace(tcfg, train_roles=("fit",),
                                model=replace(one_seed, num_leaves=63))),
        "B: A on fit + tune": (base_cols, replace(tcfg, model=replace(one_seed,
                                                                        num_leaves=63))),
        "C: B with 127 leaves": (base_cols, replace(tcfg, model=one_seed)),
        "D: C + extras (v122, one seed)": (None, replace(tcfg, model=one_seed)),
    }
    rows = {}
    for name, (cols, arm_cfg) in arms.items():
        t0 = time.time()
        models, _ = fit_stage2(outs, mock, arm_cfg, columns=cols)
        scored, _ = mock_scored(outs, models, mock, arm_cfg)
        rule, _, _ = run.choose_rule(scored, mock, cfg)
        res = mock_scores(scored, mock, rule)
        rows[name] = {**{k: float(res.loc["all", k]) for k in run.SCORE_COLS},
                      "rule": str(rule), "seconds": round(time.time() - t0, 1)}
        print(f"{name:46s} est_public {rows[name]['est_public']:.5f}  F0.5 "
              f"{rows[name]['f_beta']:.5f}  {rows[name]['seconds']:.0f} s  {rule}", flush=True)
        del models, scored
        mem_guard(name)
    table = pd.DataFrame(rows).T
    print(table[["est_public", "f_beta", "pair_precision", "pair_recall"]].to_string())
    (run.ARTIFACTS / "ablation_stage2.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
