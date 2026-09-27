"""v128_cohesion: v127's two refits with group-fit and unit-number columns, decoded at x6.

v126's decoy columns lifted the public score by +0.004 (0.968 -> 0.972), 2.5 times their mock
gain: false merges are what the public board punishes, about 6 times a missed pair. v127's
remaining false merges on the mock are records that fit the S1 but not the entity's other
records: an unrelated name at the S1's address (a co-located business), the S1's name with
an empty address, a second number that differs (apt 234 vs 225). Its misses below the rule
are mostly the reverse: records the entity's other records vouch for.

New stage-2 columns (on the kept pairs of v110's caches, next to v126's):

* cohesion (``stacking.cohesion_features``, never used since v104): p1-weighted mean
  address / name similarity of the record to the entity's other candidates, and how many
  likely candidates (p1 >= 0.5) share its address (>= 0.8 token-set);
* ``hn2_eq``: the second address numbers (unit, apartment, floor) agree, NaN without both;
  ``nums_set_eq``: the number sets agree;
* ``n_only_l`` / ``n_only_r``: core-name words only the S1 / only the pool name holds.

Stage 2 as v126 (lr 0.05 now), stage 3 as v127 plus cohesion recomputed from p2
(``coh_*_2``). Decoded as v126 (isotonic, tight decoder); the main arm is selected by
tight@6, the weight the public uploads imply, next to the x3 arm. Files:
``submissions/v128/`` (x6, copied to ``output/`` when its val tight@6 beats v127's x6 arm)
and ``submissions/v128_x3/``.

    .venv/bin/python experiments/v128_cohesion/run_v128.py > run.log 2>&1
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from entity_resolution import config as C
from entity_resolution.data import isin, load_source
from entity_resolution.mock import build_mock, target_shape
from entity_resolution.pipeline import mem_guard, peak_rss_gb, pool_of
from entity_resolution.split import load_fold
from entity_resolution.stacking import COHESION_COLUMNS, cohesion_features
from entity_resolution.tracking import log_result
from entity_resolution.trainset import inner_split, sample_s1
from entity_resolution.twostage import (
    Stage1Output,
    _train_ids,
    fit_stage2,
    fold_of,
    load_stage1,
    mock_scored,
)

VERSION = "v128_cohesion"
EXP_DIR = C.EXPERIMENTS / VERSION
ARTIFACTS = EXP_DIR / "artifacts"
V127_DIR = C.EXPERIMENTS / "v127_stage3"
_spec = importlib.util.spec_from_file_location("run_v127", V127_DIR / "run_v127.py")
v127 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v127)
v126 = v127.v126
EXTRA = [*COHESION_COLUMNS, "hn2_eq", "nums_set_eq", "n_only_l", "n_only_r"]
COH2 = [f"{c}_2" for c in COHESION_COLUMNS]
TCFG2 = replace(v126.TCFG, model=replace(v126.TCFG.model, learning_rate=0.05,
                                         n_estimators=4000, early_stopping=100))
T0 = time.time()


def log(msg: str) -> None:
    """Timestamped progress line (the run log is the only view of a background run)."""
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def extra_features(o: Stage1Output, s1n: pd.DataFrame, pooln: pd.DataFrame,
                   p: np.ndarray) -> pd.DataFrame:
    """Cohesion from ``p`` plus the unit-number and name-difference columns (``EXTRA``)."""
    pairs = o.pairs.reset_index(drop=True)
    pool = pooln.drop_duplicates(C.ENTITY_ID).reset_index(drop=True)
    coh = cohesion_features(pairs, p, pool).reset_index(drop=True)
    s1 = s1n.drop_duplicates(C.ENTITY_ID).set_index(C.ENTITY_ID)
    pool_ix = pool.set_index(C.ENTITY_ID)
    nl = s1["addr_nums"].reindex(pairs[C.S1_ID]).fillna("").to_numpy()
    nr = pool_ix["addr_nums"].reindex(pairs[C.ENTITY_ID]).fillna("").to_numpy()
    second = [(a.split()[1:2], b.split()[1:2]) for a, b in zip(nl, nr, strict=True)]
    hn2 = np.array([(x[0] == y[0]) if x and y else np.nan for x, y in second], dtype=np.float32)
    sets = np.array([(set(a.split()) == set(b.split())) if a and b else np.nan
                     for a, b in zip(nl, nr, strict=True)], dtype=np.float32)
    ln = s1["name_core"].reindex(pairs[C.S1_ID]).fillna("").to_numpy()
    rn = pool_ix["name_core"].reindex(pairs[C.ENTITY_ID]).fillna("").to_numpy()
    only = [(len(set(a.split()) - set(b.split())), len(set(b.split()) - set(a.split())))
            for a, b in zip(ln, rn, strict=True)]
    out = coh.assign(hn2_eq=hn2, nums_set_eq=sets,
                     n_only_l=np.array([x for x, _ in only], dtype=np.float32),
                     n_only_r=np.array([y for _, y in only], dtype=np.float32))
    return out[EXTRA].astype(np.float32)


def stage3_extra(o: Stage1Output, p2: np.ndarray, s1n: pd.DataFrame,
                 pooln: pd.DataFrame) -> pd.DataFrame:
    """v127's stage-3 columns plus cohesion recomputed from p2."""
    pool = pooln.drop_duplicates(C.ENTITY_ID).reset_index(drop=True)
    coh = cohesion_features(o.pairs.reset_index(drop=True), p2, pool).reset_index(drop=True)
    return pd.concat([v127.stage3_features(o, p2, s1n, pooln),
                      coh.add_suffix("_2")[COH2].astype(np.float32)], axis=1)


def featured(o: Stage1Output, s1n: pd.DataFrame, pooln: pd.DataFrame) -> Stage1Output:
    """v110's stage-1 frame + v126's columns + ``EXTRA`` (from p1)."""
    o = v126.with_new(o, v126.new_features(o, s1n, pooln))
    return v126.with_new(o, extra_features(o, s1n, pooln, o.X["p1"].to_numpy(np.float32)))


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    token_map = json.loads((v126.V110 / "artifacts" / "stage1" / "token_map.json").read_text())
    train = load_fold("train", columns=[C.COUNTRY])
    val = load_fold("val", columns=[C.COUNTRY])
    fit_fold, tune_fold = inner_split(train)
    mock = build_mock(train, val, tune_fold.s1[C.ENTITY_ID],
                      sample_s1(fit_fold.s1, v126.CFG.n_fit_s1)[C.ENTITY_ID], target_shape())
    del train, val, fit_fold, tune_fold
    fit_ids = _train_ids(mock, v126.TCFG)
    log("mock built")

    outs = {}
    for country in sorted(mock.fold.s1[C.COUNTRY].unique()):
        t0 = time.time()
        o = load_stage1(v126.STAGE1_CACHE / "mock" / f"mock_{country}.parquet")
        s1_ids = mock.fold.s1[C.ENTITY_ID][(mock.fold.s1[C.COUNTRY] == country).to_numpy()]
        s1n = v126.load_norm("train", (1,), token_map, ids=s1_ids)
        pooln = v126.load_norm("train", (2, 3), token_map,
                               ids=pool_of(mock.fold)[C.ENTITY_ID], country=country)
        outs[country] = featured(o, s1n, pooln)
        log(f"mock {country}: stage-2 frame {outs[country].X.shape} ({time.time() - t0:.0f} s)")
        del o, s1n, pooln
        mem_guard(country)

    t0 = time.time()
    models2, info2 = fit_stage2(outs, mock, TCFG2)
    for k, m in enumerate(models2):
        m.save(ARTIFACTS / f"stage2_{k}")
    imp2 = pd.concat([m.importance() for m in models2], axis=1).mean(axis=1)
    log(f"stage 2 fit {time.time() - t0:.0f} s: "
        f"{ {k: v.get('best_iteration') for k, v in info2.items() if k.startswith('fold')} }; "
        f"extra columns' gain {imp2.reindex(EXTRA).round(4).to_dict()}")

    for country, o in outs.items():
        part = np.where(isin(o.pairs[C.S1_ID], fit_ids),
                        fold_of(o.pairs[C.S1_ID], v126.TCFG.folds, v126.TCFG.seed), -1)
        p2 = v127.p2_of(o, models2, part)
        s1_ids = mock.fold.s1[C.ENTITY_ID][(mock.fold.s1[C.COUNTRY] == country).to_numpy()]
        s1n = v126.load_norm("train", (1,), token_map, ids=s1_ids)
        pooln = v126.load_norm("train", (2, 3), token_map,
                               ids=pool_of(mock.fold)[C.ENTITY_ID], country=country)
        outs[country] = v126.with_new(o, stage3_extra(o, p2, s1n, pooln))
        log(f"mock {country}: stage-3 frame {outs[country].X.shape}")
        del s1n, pooln, p2
        mem_guard(country)

    t0 = time.time()
    models3, info3 = fit_stage2(outs, mock, v126.TCFG)
    for k, m in enumerate(models3):
        m.save(ARTIFACTS / f"stage3_{k}")
    scored, _ = mock_scored(outs, models3, mock, v126.TCFG)
    scored.to_parquet(ARTIFACTS / "mock_scored.parquet", index=False)
    imp3 = pd.concat([m.importance() for m in models3], axis=1).mean(axis=1)
    log(f"stage 3 fit {time.time() - t0:.0f} s; top\n"
        f"{imp3.sort_values(ascending=False).head(12)}")
    del outs
    mem_guard("mock scored")

    arms, iso = v127.arms_for(scored, mock, "v128")
    ref = json.loads((V127_DIR / "metrics.json").read_text())["metrics"]["arms"]
    better = arms["x6"]["tight6"] > ref["x6"]["tight6"]
    log(f"val tight@6: v127 {ref['x6']['tight6']:.5f} -> v128 {arms['x6']['tight6']:.5f}; "
        f"tight@3: v127 {ref['x3']['tight3']:.5f} -> v128 {arms['x3']['tight3']:.5f}; "
        f"{'better' if better else 'not better'} at x6")
    del scored

    s1n_all = v126.load_norm("test", (1,), token_map)
    parts = []
    for country in ("France", "India", "US"):
        t0 = time.time()
        o = load_stage1(v126.STAGE1_CACHE / "test" / f"test_{country}.parquet")
        s1n = s1n_all[(s1n_all[C.COUNTRY] == country).to_numpy()]
        pooln = v126.load_norm("test", (2, 3), token_map, country=country)
        o = featured(o, s1n, pooln)
        p2 = v127.p2_of(o, models2, None)
        o = v126.with_new(o, stage3_extra(o, p2, s1n, pooln))
        del s1n, pooln
        prob = v127.p2_of(o, models3, None)
        parts.append(o.pairs.assign(prob=prob, p2=p2, country=country))
        log(f"test {country}: {len(o.pairs):,} pairs scored ({time.time() - t0:.0f} s)")
        del o, p2
        mem_guard(country)
    test = pd.concat(parts, ignore_index=True)
    del parts
    test.to_parquet(ARTIFACTS / "test_scored.parquet", index=False)
    s1_ids = load_source("test", 1, v126.CFG.dataset_dir, columns=[C.ENTITY_ID])[C.ENTITY_ID]
    test_info = v127.write_arms(test, arms, iso, s1_ids, {"x6": "v128", "x3": "v128_x3"},
                                copy_x3=False)
    if better:
        C.OUTPUT.mkdir(parents=True, exist_ok=True)
        for name in (C.MATCHING_FILE, C.CANDIDATE_FILE):
            shutil.copy2(C.ROOT / "submissions" / "v128" / name, C.OUTPUT / name)
        log("x6 arm copied to output/")
    main_arm = arms["x6"]
    record = {
        "hypothesis": "group-fit (cohesion), unit-number and name-difference columns cut the "
                      "remaining false merges; decoding at the public-implied x6 weight",
        "parent": "v127 (same caches, both refits redone)", "extra_columns": EXTRA,
        "stage3_extra": COH2, "stage2": TCFG2.record(), "stage2_fit_info": info2,
        "stage3_fit_info": info3, "extra_gain_stage2": imp2.reindex(EXTRA).to_dict(),
        "top_importance_stage3": imp3.sort_values(ascending=False).head(25).to_dict(),
        "arms": arms, "v127_arms": ref, "test": test_info, "peak_rss_gb": peak_rss_gb(),
    }
    row = log_result(
        EXP_DIR, change="v127 + cohesion, unit-number and name-difference columns (stage 2), "
                        "cohesion from p2 (stage 3); decoded at x6",
        group="C5", mock_f05=main_arm["f_beta"], cand_recall=None,
        notes=(f"val tight@6 {main_arm['tight6']:.4f} (v127 x6 {ref['x6']['tight6']:.4f}); "
               f"tight@3 of the x3 arm {arms['x3']['tight3']:.4f} (v127 "
               f"{ref['x3']['tight3']:.4f}); false merges {main_arm['errors']['false_merge']}"),
        metrics=record, owner="M1", parent="v127", decision="INVESTIGATE")
    log(f"logged {row}; done, peak RSS {peak_rss_gb()} GB")


if __name__ == "__main__":
    main()
