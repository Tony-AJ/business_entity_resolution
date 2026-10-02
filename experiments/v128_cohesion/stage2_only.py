"""v128 without its third stage: stage-2 probabilities decoded as the main run decodes.

v127's stage 3 gained +0.0004 tight@3 on the mock but scored 0.971 public against v126's
0.972 (no third stage): the stage-3 gain did not reach the test. This scores v128's stage-2
models alone (v126's columns + cohesion, unit-number and name-difference columns, lr 0.05)
on the mock's tune and val entities (out of fold, as the main run), decodes them with v126's
isotonic + tight decoder at x3 and x6, and writes the test files from the stage-2
probabilities the main run saved (``test_scored.parquet``, column ``p2``):
``submissions/v128s2/`` (x3) and ``submissions/v128s2_fp6/`` (x6).

    .venv/bin/python experiments/v128_cohesion/stage2_only.py > stage2_only.log 2>&1
"""
from __future__ import annotations

import importlib.util
import json
import time

import pandas as pd

from entity_resolution import config as C
from entity_resolution.data import load_source
from entity_resolution.mock import build_mock, target_shape
from entity_resolution.model import Matcher
from entity_resolution.pipeline import mem_guard, pool_of
from entity_resolution.split import load_fold
from entity_resolution.trainset import inner_split, sample_s1
from entity_resolution.twostage import load_stage1, mock_scored

V128_DIR = C.EXPERIMENTS / "v128_cohesion"
_spec = importlib.util.spec_from_file_location("run_v128", V128_DIR / "run_v128.py")
v128 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v128)
v127, v126 = v128.v127, v128.v126
T0 = time.time()


def log(msg: str) -> None:
    """Timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def main() -> None:
    token_map = json.loads((v126.V110 / "artifacts" / "stage1" / "token_map.json").read_text())
    models2 = [Matcher.load(v128.ARTIFACTS / f"stage2_{k}") for k in range(v126.TCFG.folds)]
    train = load_fold("train", columns=[C.COUNTRY])
    val = load_fold("val", columns=[C.COUNTRY])
    fit_fold, tune_fold = inner_split(train)
    mock = build_mock(train, val, tune_fold.s1[C.ENTITY_ID],
                      sample_s1(fit_fold.s1, v126.CFG.n_fit_s1)[C.ENTITY_ID], target_shape())
    del train, val, fit_fold, tune_fold
    outs = {}
    for country in sorted(mock.fold.s1[C.COUNTRY].unique()):
        o = load_stage1(v126.STAGE1_CACHE / "mock" / f"mock_{country}.parquet")
        s1_ids = mock.fold.s1[C.ENTITY_ID][(mock.fold.s1[C.COUNTRY] == country).to_numpy()]
        s1n = v126.load_norm("train", (1,), token_map, ids=s1_ids)
        pooln = v126.load_norm("train", (2, 3), token_map,
                               ids=pool_of(mock.fold)[C.ENTITY_ID], country=country)
        outs[country] = v128.featured(o, s1n, pooln)
        del o, s1n, pooln
        mem_guard(country)
    log("mock stage-2 frames built")
    scored, _ = mock_scored(outs, models2, mock, v126.TCFG)
    del outs
    scored.to_parquet(v128.ARTIFACTS / "mock_scored_stage2.parquet", index=False)
    arms, iso = v127.arms_for(scored, mock, "v128 stage 2 only")
    test = pd.read_parquet(v128.ARTIFACTS / "test_scored.parquet")
    test = test.assign(prob=test["p2"])
    s1_ids = load_source("test", 1, v126.CFG.dataset_dir, columns=[C.ENTITY_ID])[C.ENTITY_ID]
    info = v127.write_arms(test, arms, iso, s1_ids, {"x3": "v128s2", "x6": "v128s2_fp6"},
                           copy_x3=False)
    (V128_DIR / "stage2_only.json").write_text(
        json.dumps({"arms": arms, "test": info}, indent=1) + "\n")
    log("wrote stage2_only.json")


if __name__ == "__main__":
    main()
