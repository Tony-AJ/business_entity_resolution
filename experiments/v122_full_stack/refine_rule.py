"""v122's rule on a finer grid: expected-F0.5 decoding around v122's gamma 1.5 / misses 0.2.

v107's rule choice (``run_v122.choose_rule``) searches gammas 0.7-2.0 and expected misses
0.0-0.4 in coarse steps. This script re-tunes the expected-F0.5 rule on a finer grid around
the chosen point, on the same mock tune entities and the same three-seed stage-2 scores
(``artifacts/mock_scored.parquet``), and scores the val entities with it. Nothing is retrained.
The table goes to ``artifacts/refine_rule.json``; a finer rule that beats v122 on val is a new
version (v123) whose test files come from v122's cached test stage-1 outputs.

    .venv/Scripts/python experiments/v122_full_stack/refine_rule.py > refine_rule.log 2>&1
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from entity_resolution import config as C
from entity_resolution.data import isin
from entity_resolution.decision import tune_expected
from entity_resolution.mock import FP_WEIGHT, build_mock, target_shape
from entity_resolution.pipeline import mock_scores
from entity_resolution.split import load_fold
from entity_resolution.trainset import inner_split, sample_s1

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("run_v122", HERE / "run_v122.py")
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)

GAMMAS = tuple(np.round(np.arange(1.0, 2.31, 0.1), 2))
MISSES = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
MAX_MATCHES = (8, 11, 15)


def main() -> None:
    cfg, _ = run.configs()
    train = load_fold("train", columns=[C.COUNTRY])
    val = load_fold("val", columns=[C.COUNTRY])
    fit_fold, tune_fold = inner_split(train)
    mock = build_mock(train, val, tune_fold.s1[C.ENTITY_ID],
                      sample_s1(fit_fold.s1, cfg.n_fit_s1)[C.ENTITY_ID], target_shape())
    del train, val, fit_fold, tune_fold
    scored = pd.read_parquet(run.ARTIFACTS / "mock_scored.parquet")
    tune_part = mock.part("tune")
    rows_t = scored[isin(scored[C.S1_ID], pd.Index(tune_part.s1[C.ENTITY_ID]))]
    fits = [tune_expected(rows_t, tune_part.s1[C.ENTITY_ID], tune_part.pairs, gammas=GAMMAS,
                          misses=MISSES, max_matches=k, fp_weight=FP_WEIGHT)
            for k in MAX_MATCHES]
    rule, _ = max(fits, key=lambda f: float(f[1]["f_beta"].max()))
    table = pd.concat([t for _, t in fits], ignore_index=True)
    res = mock_scores(scored, mock, rule)
    base = json.loads((HERE / "metrics.json").read_text())["metrics"]
    out = {"rule": asdict(rule), "tune_f_tight": float(table["f_beta"].max()),
           **{k: float(res.loc["all", k]) for k in run.SCORE_COLS},
           "v122_est_public": base["est_public"], "v122_rule": base["rule"],
           "delta_vs_v122": float(res.loc["all", "est_public"]) - base["est_public"]}
    print(json.dumps(out, indent=1))
    print(table.sort_values("f_beta", ascending=False).head(10).to_string())
    (run.ARTIFACTS / "refine_rule.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
