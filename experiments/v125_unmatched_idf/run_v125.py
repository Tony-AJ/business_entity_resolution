"""v125_unmatched_idf: stage 2 refit with unmatched-token IDF features, tight decode.

The mock error profile of v123's x3 rule: the 2,120 remaining false merges carry a median
probability of 0.957 and look like high-probability true pairs to every current feature
(same name-equal share, same number overlap). The classic separator for one-token decoys
(Chaudhuri et al., SIGMOD 2003) is the *symmetric difference* weighted by rarity: a decoy
adds a token ("Holdings", "Midtown") that no record of the true business carries, and on a
true pair the unmatched tokens are common filler. None of the 94 stage-1 features reads the
IDF of unmatched name tokens directly.

New pair features (computed on the kept pairs only, appended to v122's cached stage-1
frames; per-country IDF from the partition's own pool, so France gets French rarity with no
training-language dependence):

    uidf_max_l / uidf_max_r   highest IDF among core-name tokens only that side holds
    uidf_sum_l / uidf_sum_r   summed IDF of those tokens
    uidf_share                unmatched IDF mass / total IDF mass of both names
    unum_conflict             both addresses hold numbers and none is shared

Stage 2 is refitted exactly as v122's (one seed: v122 measured seed averaging at zero),
probabilities are isotonically calibrated and decoded with v124's tight-objective decoder.
Selection on the mock's tune entities by tight@3; val reports F0.5, tight scores and error
counts against v123's x3 arm. Test files: ``submissions/v125/`` and ``output/``.

    .venv/Scripts/python experiments/v125_unmatched_idf/run_v125.py > run.log 2>&1
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from entity_resolution import config as C
from entity_resolution.data import isin, load_source
from entity_resolution.evaluate import error_samples, score_pairs
from entity_resolution.mock import build_mock, target_shape
from entity_resolution.pipeline import load_normalised, mem_guard, pool_of
from entity_resolution.split import load_fold
from entity_resolution.submission import write_pairs
from entity_resolution.tracking import log_result
from entity_resolution.trainset import inner_split, label_pairs, sample_s1
from entity_resolution.twostage import Stage1Output, fit_stage2, load_stage1, mock_scored

VERSION = "v125_unmatched_idf"
EXP_DIR = C.EXPERIMENTS / VERSION
ARTIFACTS = EXP_DIR / "artifacts"
V122 = C.EXPERIMENTS / "v122_full_stack"
_spec = importlib.util.spec_from_file_location("run_v122", V122 / "run_v122.py")
v122 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v122)
_spec4 = importlib.util.spec_from_file_location(
    "run_v124", C.EXPERIMENTS / "v124_tight_decode" / "run_v124.py")
v124 = importlib.util.module_from_spec(_spec4)
_spec4.loader.exec_module(v124)
NEW = ["uidf_max_l", "uidf_max_r", "uidf_sum_l", "uidf_sum_r", "uidf_share", "unum_conflict"]
SELECT_W = 3.0
DECODE_W = (2.0, 3.0, 4.0, 5.0)
MISSES = (0.0, 0.05, 0.1, 0.2)
T0 = time.time()


def log(msg: str) -> None:
    """Timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def idf_table(pool_names: pd.Series) -> dict[str, float]:
    """log(N / df) per token over the pool's core names (df = records holding the token)."""
    toks = pool_names.str.split()
    n = max(len(toks), 1)
    df = pd.Series([t for row in toks for t in set(row)]).value_counts()
    return dict(zip(df.index, np.log(n / df.to_numpy()), strict=True))


def new_features(pairs: pd.DataFrame, s1n: pd.DataFrame, pooln: pd.DataFrame,
                 idf: dict[str, float]) -> pd.DataFrame:
    """The six new columns for ``pairs`` (float32), from normalised name_core / addr_norm."""
    s1 = s1n.set_index(C.ENTITY_ID)
    pool = pooln.set_index(C.ENTITY_ID)
    ln = s1["name_core"].reindex(pairs[C.S1_ID]).fillna("").to_numpy()
    rn = pool["name_core"].reindex(pairs[C.ENTITY_ID]).fillna("").to_numpy()
    la = s1["addr_norm"].reindex(pairs[C.S1_ID]).fillna("").to_numpy()
    ra = pool["addr_norm"].reindex(pairs[C.ENTITY_ID]).fillna("").to_numpy()
    out = np.zeros((len(pairs), len(NEW)), dtype=np.float32)
    default = float(np.median(list(idf.values()))) if idf else 0.0
    for i in range(len(pairs)):
        tl, tr = set(ln[i].split()), set(rn[i].split())
        only_l, only_r = tl - tr, tr - tl
        wl = [idf.get(t, default) for t in only_l]
        wr = [idf.get(t, default) for t in only_r]
        wall = [idf.get(t, default) for t in tl | tr]
        um = sum(wl) + sum(wr)
        out[i, 0] = max(wl, default=0.0)
        out[i, 1] = max(wr, default=0.0)
        out[i, 2] = sum(wl)
        out[i, 3] = sum(wr)
        out[i, 4] = um / max(sum(wall), 1e-9)
        nl = {t for t in la[i].split() if t.isdigit()}
        nr = {t for t in ra[i].split() if t.isdigit()}
        out[i, 5] = float(bool(nl) and bool(nr) and not (nl & nr))
    return pd.DataFrame(out, columns=NEW, index=pairs.index)


def with_new(o: Stage1Output, feats: pd.DataFrame) -> Stage1Output:
    """The stage-1 output with the new columns appended to its frame."""
    return Stage1Output(o.pairs, pd.concat([o.X, feats.reset_index(drop=True)], axis=1),
                        o.n_all)


def main() -> None:
    cfg, tcfg = v122.configs()
    tcfg = replace(tcfg, model=replace(tcfg.model, seed=v122.STAGE2_SEEDS[0]))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stage1_cache = (cfg.cache_dir / "stage1" / f"v122_{cfg.blocking.key()}_f{tcfg.floor}"
                    f"_k{tcfg.max_cands}_a{int(tcfg.anchors)}_c{int(tcfg.cohesion)}"
                    f"_r{int(tcfg.rivals)}_x{len(tcfg.extra_groups)}")
    train = load_fold("train", columns=[C.COUNTRY])
    val = load_fold("val", columns=[C.COUNTRY])
    fit_fold, tune_fold = inner_split(train)
    mock = build_mock(train, val, tune_fold.s1[C.ENTITY_ID],
                      sample_s1(fit_fold.s1, cfg.n_fit_s1)[C.ENTITY_ID], target_shape())
    token_map = json.loads((V122 / "artifacts" / "stage1" / "token_map.json").read_text())
    fillers = json.loads((V122 / "artifacts" / "stage1" / "fillers.json").read_text())
    del val, fit_fold, tune_fold
    # name_norm, non_latin and country: apply_token_map recomputes non-Latin rows
    cols = [C.ENTITY_ID, C.COUNTRY, "name_core", "addr_norm", "name_norm", "non_latin"]

    # mock: new features per country on the cached kept pairs
    outs = {}
    for country in sorted(mock.fold.s1[C.COUNTRY].unique()):
        t0 = time.time()
        o = load_stage1(stage1_cache / "mock" / f"mock_{country}.parquet")
        s1_ids = mock.fold.s1[C.ENTITY_ID][(mock.fold.s1[C.COUNTRY] == country).to_numpy()]
        s1n = load_normalised("train", (1,), cfg, s1_ids, token_map, columns=cols,
                              fillers=fillers)
        pooln = load_normalised("train", (2, 3), cfg, pool_of(mock.fold)[C.ENTITY_ID],
                                token_map, columns=cols, country=country, fillers=fillers)
        idf = idf_table(pooln["name_core"])
        outs[country] = with_new(o, new_features(o.pairs, s1n, pooln, idf))
        log(f"mock {country}: {len(o.pairs):,} kept pairs featured "
            f"({time.time() - t0:.0f} s)")
        del s1n, pooln, o
        mem_guard(country)

    models, info = fit_stage2(outs, mock, tcfg)
    log(f"stage 2 refit: {info}")
    scored, _ = mock_scored(outs, models, mock, tcfg)
    scored.to_parquet(ARTIFACTS / "mock_scored.parquet", index=False)
    importance = pd.concat([m.importance() for m in models], axis=1).mean(axis=1)
    log("new columns' gain share "
        f"{float(importance.reindex(NEW).fillna(0).sum()):.5f}; "
        f"{importance.reindex(NEW).round(5).to_dict()}")

    tune_part, val_part = mock.part("tune"), mock.part("val")
    rows_t = scored[isin(scored[C.S1_ID], pd.Index(tune_part.s1[C.ENTITY_ID]))]
    rows_v = scored[isin(scored[C.S1_ID], pd.Index(val_part.s1[C.ENTITY_ID]))]
    y_t = label_pairs(rows_t[[C.S1_ID, C.ENTITY_ID]], mock.fold.pairs)["label"].to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(rows_t["prob"].to_numpy(np.float64), y_t)

    def calibrated(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(prob=iso.predict(df["prob"].to_numpy(np.float64)).astype(np.float32))

    cal_t, cal_v = calibrated(rows_t), calibrated(rows_v)
    grid = []
    for w in DECODE_W:
        for miss in MISSES:
            m_t = v124.decode_tight(cal_t, w, miss)
            grid.append({"w": w, "miss": miss,
                         "tune_tight3": v124.macro_tight(m_t, tune_part, SELECT_W)})
    grid = pd.DataFrame(grid).sort_values("tune_tight3", ascending=False)
    w_best, miss_best = float(grid.iloc[0]["w"]), float(grid.iloc[0]["miss"])
    log("tune grid (top 3):\n" + grid.head(3).to_string(index=False))

    matches = v124.decode_tight(cal_v, w_best, miss_best)
    s = score_pairs(matches, val_part)
    errors = {k: len(error_samples(matches, val_part, k, n=10**9))
              for k in ("false_merge", "singleton_merge", "missed", "false_singleton")}
    tight145 = v124.macro_tight(matches, val_part, 1.45)
    tight3 = v124.macro_tight(matches, val_part, SELECT_W)
    v123_arms = json.loads((C.EXPERIMENTS / "v123_tight_rule" / "metrics.json").read_text())[
        "metrics"]["arms"]
    log(f"val: F0.5 {s['f_beta']:.5f}, tight@1.45 {tight145:.5f}, tight@3 {tight3:.5f}, "
        f"errors {errors} (v123 x3: F0.5 {float(v123_arms['3.0']['f_beta']):.5f}, "
        f"false merges {v123_arms['3.0']['errors_mock']['false_merge']})")

    # test: same features on the cached test stage-1 outputs, refit models, decode
    test_parts, cands = [], []
    s1n_all = load_normalised("test", (1,), cfg, token_map=token_map, columns=cols,
                              fillers=fillers)
    for country in ("France", "India", "US"):
        t0 = time.time()
        o = load_stage1(stage1_cache / "test" / f"test_{country}.parquet")
        s1n = s1n_all[(s1n_all[C.COUNTRY] == country).to_numpy()]
        pooln = load_normalised("test", (2, 3), cfg, token_map=token_map, columns=cols,
                                country=country, fillers=fillers)
        idf = idf_table(pooln["name_core"])
        o = with_new(o, new_features(o.pairs, s1n, pooln, idf))
        del s1n, pooln
        X = o.X[models[0].feature_names_]
        prob = np.mean([m.predict_proba(X) for m in models], axis=0).astype(np.float32)
        sc = calibrated(o.pairs.assign(prob=prob))
        dec = v124.decode_tight(sc, w_best, miss_best)
        dec = dec.sort_values([C.S1_ID, "prob"], ascending=[True, False], kind="stable")
        test_parts.append(dec)
        cands.append(o.pairs[[C.S1_ID, C.ENTITY_ID]])
        log(f"test {country}: {len(o.pairs):,} pairs, {len(dec):,} matches "
            f"({time.time() - t0:.0f} s)")
        del o, X, sc
        mem_guard(country)
    tm = pd.concat(test_parts, ignore_index=True)
    cand = pd.concat(cands, ignore_index=True)
    s1_ids = load_source("test", 1, cfg.dataset_dir, columns=[C.ENTITY_ID])[C.ENTITY_ID]
    out = C.ROOT / "submissions" / "v125"
    paths = write_pairs(tm[[C.S1_ID, C.ENTITY_ID]], cand, s1_ids.tolist(), out)
    check = subprocess.run([sys.executable, "-m", "entity_resolution.submission",
                            "--output-dir", str(out), "--check-ids"],
                           capture_output=True, text=True)
    log(f"wrote {out} ({len(tm):,} pairs): checker "
        f"{(check.stdout + check.stderr).strip()[-120:]}")
    C.OUTPUT.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.copy2(p, C.OUTPUT / p.name)
    log("copied to output/")

    record = {
        "hypothesis": "IDF-weighted unmatched-token features separate the confident false "
                      "merges (median prob 0.957) that no current feature separates",
        "new_columns": NEW, "new_gain_share": float(importance.reindex(NEW).fillna(0).sum()),
        "stage2": "v122 config, one seed, refit on the same kept pairs + new columns",
        "decode": {"w": w_best, "miss": miss_best, "select_weight": SELECT_W,
                   "calibration": "isotonic on mock tune"},
        "grid": grid.to_dict("records"),
        "val": {**{k: float(x) for k, x in s.items() if isinstance(x, (int, float))},
                "tight_1.45": tight145, "tight_3": tight3, "errors": errors},
        "stage2_fit_info": info, "test_pairs": int(len(tm)),
    }
    row = log_result(
        EXP_DIR, change="unmatched-token IDF + number-conflict features in stage 2; "
                        "isotonic + tight decode; one seed",
        group="C4", mock_f05=float(s["f_beta"]), cand_recall=None,
        notes=(f"val tight@3 {tight3:.4f}, F0.5 {s['f_beta']:.4f}, false merges "
               f"{errors['false_merge']} (v123 x3: 1904); new columns' gain share "
               f"{record['new_gain_share']:.4f}; judged by the public upload"),
        metrics=record, owner="M3", parent="v124", decision="INVESTIGATE")
    log(f"logged {row}")


if __name__ == "__main__":
    main()
