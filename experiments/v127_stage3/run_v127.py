"""v127_stage3: a third stage that re-reads the competition and the groups through p2.

Stage 2 is trained with competition features (rank, best rival, gap on the S1 side and the
pool side) and v126's candidate-group features, all computed from stage 1's p1. Its own
probability p2 is sharper (it reads every decoy signal stage 1 never saw), so the same
collective statistics recomputed from p2 tell each pair where it stands among the *stage-2*
rivals: a decoy record that stage 1 ranked first on the pool side but stage 2 demotes frees
its pool record for the true S1, and a number group whose best member stage 2 trusts lifts
the group. This is one more round of the stacking v104 started (pool_gap alone carried 72 %
of stage 2's gain).

Stage 3 reads v126's frame (v110 stage-1 columns + v126's new columns) plus the competition
columns of p2 (``*_2``) and the best p2 of the pair's number and name groups. p2 is v126's
stage-2 models' out-of-fold probability on its fit + tune entities (the cross-fitting parts
are the same id hash, so stage 3's part f never sees a p2 fitted on part f), the mean of the
models elsewhere, as on test. Stage 3 is cross-fitted like stage 2, then calibrated and
decoded exactly as v126 (isotonic, v124's tight decoder, tight@3 on tune; tight@6 arm).
Files: ``submissions/v127/`` (+ ``output/`` when it beats v126 on val tight@3) and
``submissions/v127_fp6/``.

    .venv/bin/python experiments/v127_stage3/run_v127.py > run.log 2>&1
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from entity_resolution import config as C
from entity_resolution.data import isin, load_source
from entity_resolution.evaluate import (
    entity_counts,
    entity_tight_from_counts,
    error_samples,
    score_pairs,
)
from entity_resolution.mock import build_mock, target_shape
from entity_resolution.model import Matcher
from entity_resolution.pipeline import mem_guard, peak_rss_gb, pool_of
from entity_resolution.split import load_fold
from entity_resolution.stacking import STACK_COLUMNS, competition_features
from entity_resolution.submission import write_pairs
from entity_resolution.tracking import log_result
from entity_resolution.trainset import inner_split, label_pairs, sample_s1
from entity_resolution.twostage import (
    Stage1Output,
    _train_ids,
    fit_stage2,
    fold_of,
    load_stage1,
    mock_scored,
    predict_stage2,
)

VERSION = "v127_stage3"
EXP_DIR = C.EXPERIMENTS / VERSION
ARTIFACTS = EXP_DIR / "artifacts"
V126_DIR = C.EXPERIMENTS / "v126_decoy_groups"
_spec = importlib.util.spec_from_file_location("run_v126", V126_DIR / "run_v126.py")
v126 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v126)
STAGE3 = [f"{c}_2" for c in STACK_COLUMNS] + ["grp_num_p2max", "grp_name_p2max"]
T0 = time.time()


def log(msg: str) -> None:
    """Timestamped progress line (the run log is the only view of a background run)."""
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def arms_for(sc: pd.DataFrame, mock, label: str) -> tuple[dict, IsotonicRegression]:
    """v126's decode: isotonic on tune, tight-decoder grid on tune, val of the x3 / x6 picks."""
    tune_part, val_part = mock.part("tune"), mock.part("val")
    rows_t = sc[isin(sc[C.S1_ID], pd.Index(tune_part.s1[C.ENTITY_ID]))]
    rows_v = sc[isin(sc[C.S1_ID], pd.Index(val_part.s1[C.ENTITY_ID]))]
    y_t = label_pairs(rows_t[[C.S1_ID, C.ENTITY_ID]], mock.fold.pairs)["label"].to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(rows_t["prob"].to_numpy(np.float64), y_t)
    cal_t = rows_t.assign(prob=iso.predict(rows_t["prob"].to_numpy(np.float64)))
    cal_v = rows_v.assign(prob=iso.predict(rows_v["prob"].to_numpy(np.float64)))
    grid = []
    for w in v126.DECODE_W:
        for miss in v126.MISSES:
            counts = entity_counts(v126.v124.decode_tight(cal_t, w, miss), tune_part)
            grid.append({"w": w, "miss": miss, **{
                f"tune_tight{s:g}": float(entity_tight_from_counts(*counts, s).mean())
                for s in (1.0, v126.SELECT_W, v126.STRICT_W)}})
    grid = pd.DataFrame(grid)
    arms = {}
    for sel in (v126.SELECT_W, v126.STRICT_W):
        best = grid.sort_values(f"tune_tight{sel:g}", ascending=False).iloc[0]
        m_v = v126.v124.decode_tight(cal_v, float(best["w"]), float(best["miss"]))
        s = score_pairs(m_v, val_part)
        errors = {k: len(error_samples(m_v, val_part, k, n=10**9))
                  for k in ("false_merge", "singleton_merge", "missed", "false_singleton")}
        arms[f"x{sel:g}"] = {
            "w": float(best["w"]), "miss": float(best["miss"]), "f_beta": float(s["f_beta"]),
            "pair_precision": float(s["pair_precision"]),
            "pair_recall": float(s["pair_recall"]),
            **{f"tight{t:g}": v126.v124.macro_tight(m_v, val_part, t)
               for t in (1.45, v126.SELECT_W, v126.STRICT_W)}, "errors": errors}
        log(f"{label} arm x{sel:g}: {json.dumps(arms[f'x{sel:g}'])}")
    return arms, iso


def write_arms(test: pd.DataFrame, arms: dict, iso: IsotonicRegression, s1_ids: pd.Series,
               dirs: dict[str, str], copy_x3: bool) -> dict:
    """Decode the test probabilities per arm, write and check each arm's files."""
    cand = test[[C.S1_ID, C.ENTITY_ID]]
    country_of = test.drop_duplicates(C.S1_ID).set_index(C.S1_ID)["country"]
    info = {}
    for arm, sub in dirs.items():
        a = arms[arm]
        parts = []
        for country in ("France", "India", "US"):
            sc = test.loc[test["country"] == country, [C.S1_ID, C.ENTITY_ID, "prob"]]
            sc = sc.assign(prob=iso.predict(sc["prob"].to_numpy(np.float64)))
            parts.append(v126.v124.decode_tight(sc, a["w"], a["miss"]))
        tm = pd.concat(parts, ignore_index=True).sort_values(
            [C.S1_ID, "prob"], ascending=[True, False], kind="stable")
        out = C.ROOT / "submissions" / sub
        paths = write_pairs(tm[[C.S1_ID, C.ENTITY_ID]], cand, s1_ids.tolist(), out)
        by = tm.groupby(tm[C.S1_ID].map(country_of)).size().to_dict()
        info[arm] = {"pairs": int(len(tm)), "by_country": {k: int(v) for k, v in by.items()},
                     "s1_matched": int(tm[C.S1_ID].nunique()), "dir": str(out)}
        log(f"test arm {arm}: {info[arm]}; check: {v126.validate(out)}")
        if arm == "x3" and copy_x3:
            C.OUTPUT.mkdir(parents=True, exist_ok=True)
            for p in paths:
                shutil.copy2(p, C.OUTPUT / p.name)
            log("x3 arm copied to output/")
    return info


def stage3_features(o: Stage1Output, p2: np.ndarray, s1n: pd.DataFrame,
                    pooln: pd.DataFrame) -> pd.DataFrame:
    """Competition columns of p2 (``*_2``) and the best p2 of each pair's number/name group."""
    s1 = s1n.drop_duplicates(C.ENTITY_ID).set_index(C.ENTITY_ID)
    pool = pooln.drop_duplicates(C.ENTITY_ID).set_index(C.ENTITY_ID)
    pairs = o.pairs.reset_index(drop=True)
    hl = v126._first(s1["addr_nums"].reindex(pairs[C.S1_ID]).fillna("").to_numpy())
    hr = v126._first(pool["addr_nums"].reindex(pairs[C.ENTITY_ID]).fillna("").to_numpy())
    name_r = pool["name_core"].reindex(pairs[C.ENTITY_ID]).fillna("").to_numpy()
    addr_r = pool["addr_norm"].reindex(pairs[C.ENTITY_ID]).fillna("").to_numpy()
    comp = competition_features(pairs, p2).add_suffix("_2").reset_index(drop=True)
    grp = v126.group_features(pairs, p2.astype(np.float64), hl, hr, name_r, addr_r)
    comp["grp_num_p2max"] = grp["grp_num_p1max"].to_numpy()
    comp["grp_name_p2max"] = grp["grp_name_p1max"].to_numpy()
    return comp[STAGE3].astype(np.float32)


def p2_of(o: Stage1Output, models: list[Matcher], part: np.ndarray | None) -> np.ndarray:
    """v126's stage-2 probability of every kept pair (out of fold where ``part >= 0``)."""
    names = models[0].feature_names_
    X = o.X if list(o.X.columns) == names else o.X[names]
    return predict_stage2(models, X, part)


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    token_map = json.loads((v126.V110 / "artifacts" / "stage1" / "token_map.json").read_text())
    models2 = [Matcher.load(v126.ARTIFACTS / f"stage2_{k}") for k in range(v126.TCFG.folds)]
    train = load_fold("train", columns=[C.COUNTRY])
    val = load_fold("val", columns=[C.COUNTRY])
    fit_fold, tune_fold = inner_split(train)
    mock = build_mock(train, val, tune_fold.s1[C.ENTITY_ID],
                      sample_s1(fit_fold.s1, v126.CFG.n_fit_s1)[C.ENTITY_ID], target_shape())
    del train, val, fit_fold, tune_fold
    fit_ids = _train_ids(mock, v126.TCFG)
    log("mock built; v126 stage-2 models loaded")

    outs = {}
    for country in sorted(mock.fold.s1[C.COUNTRY].unique()):
        t0 = time.time()
        o = load_stage1(v126.STAGE1_CACHE / "mock" / f"mock_{country}.parquet")
        s1_ids = mock.fold.s1[C.ENTITY_ID][(mock.fold.s1[C.COUNTRY] == country).to_numpy()]
        s1n = v126.load_norm("train", (1,), token_map, ids=s1_ids)
        pooln = v126.load_norm("train", (2, 3), token_map,
                               ids=pool_of(mock.fold)[C.ENTITY_ID], country=country)
        o = v126.with_new(o, v126.new_features(o, s1n, pooln))
        part = np.where(isin(o.pairs[C.S1_ID], fit_ids),
                        fold_of(o.pairs[C.S1_ID], v126.TCFG.folds, v126.TCFG.seed), -1)
        p2 = p2_of(o, models2, part)
        outs[country] = v126.with_new(o, stage3_features(o, p2, s1n, pooln))
        log(f"mock {country}: {len(o.pairs):,} pairs, stage-3 frame "
            f"{outs[country].X.shape} ({time.time() - t0:.0f} s)")
        del s1n, pooln, o, p2
        mem_guard(country)

    t0 = time.time()
    models3, info = fit_stage2(outs, mock, v126.TCFG)
    log(f"stage 3 fit {time.time() - t0:.0f} s: "
        f"{ {k: v.get('best_iteration') for k, v in info.items() if k.startswith('fold')} }")
    for k, m in enumerate(models3):
        m.save(ARTIFACTS / f"stage3_{k}")
    scored, _ = mock_scored(outs, models3, mock, v126.TCFG)
    scored.to_parquet(ARTIFACTS / "mock_scored.parquet", index=False)
    importance = pd.concat([m.importance() for m in models3], axis=1).mean(axis=1)
    importance = importance.sort_values(ascending=False)
    log(f"stage-3 columns' gain share {float(importance.reindex(STAGE3).fillna(0).sum()):.4f}"
        f"\ntop\n{importance.head(15)}")
    del outs
    mem_guard("mock scored")

    arms, iso = arms_for(scored, mock, "v127")
    ref = json.loads((V126_DIR / "metrics.json").read_text())["metrics"]["arms"]
    better = arms["x3"]["tight3"] > ref["x3"]["tight3"]
    log(f"val tight@3 v126 {ref['x3']['tight3']:.5f} -> v127 {arms['x3']['tight3']:.5f}; "
        f"{'better' if better else 'not better'}")
    del scored

    s1n_all = v126.load_norm("test", (1,), token_map)
    parts = []
    for country in ("France", "India", "US"):
        t0 = time.time()
        o = load_stage1(v126.STAGE1_CACHE / "test" / f"test_{country}.parquet")
        s1n = s1n_all[(s1n_all[C.COUNTRY] == country).to_numpy()]
        pooln = v126.load_norm("test", (2, 3), token_map, country=country)
        o = v126.with_new(o, v126.new_features(o, s1n, pooln))
        p2 = p2_of(o, models2, None)
        o = v126.with_new(o, stage3_features(o, p2, s1n, pooln))
        del s1n, pooln
        prob = p2_of(o, models3, None)
        parts.append(o.pairs.assign(prob=prob, p2=p2, country=country))
        log(f"test {country}: {len(o.pairs):,} pairs scored ({time.time() - t0:.0f} s)")
        del o, p2
        mem_guard(country)
    test = pd.concat(parts, ignore_index=True)
    del parts
    test.to_parquet(ARTIFACTS / "test_scored.parquet", index=False)
    s1_ids = load_source("test", 1, v126.CFG.dataset_dir, columns=[C.ENTITY_ID])[C.ENTITY_ID]
    test_info = write_arms(test, arms, iso, s1_ids, {"x3": "v127", "x6": "v127_fp6"},
                                copy_x3=better)
    main_arm = arms["x3"]
    record = {
        "hypothesis": "competition and group features recomputed from stage-2 probabilities "
                      "sharpen the pool-side and group decisions",
        "parent": "v126 (stage-2 models and frame)", "stage3_columns": STAGE3,
        "stage3_fit_info": info, "stage3_gain_share":
            float(importance.reindex(STAGE3).fillna(0).sum()),
        "top_importance": importance.head(25).to_dict(), "arms": arms,
        "v126_arms": ref, "test": test_info, "peak_rss_gb": peak_rss_gb(),
    }
    row = log_result(
        EXP_DIR, change="stage 3 on v126: competition + group features from p2; isotonic + "
                        "tight decode (x3)",
        group="D3", mock_f05=main_arm["f_beta"], cand_recall=None,
        notes=(f"val tight@3 {main_arm['tight3']:.4f} (v126 {ref['x3']['tight3']:.4f}); "
               f"F0.5 {main_arm['f_beta']:.4f}; false merges "
               f"{main_arm['errors']['false_merge']} (v126 "
               f"{ref['x3']['errors']['false_merge']}); x6 arm in submissions/v127_fp6"),
        metrics=record, owner="M1", parent="v126", decision="INVESTIGATE")
    log(f"logged {row}; done, peak RSS {peak_rss_gb()} GB")


if __name__ == "__main__":
    main()
