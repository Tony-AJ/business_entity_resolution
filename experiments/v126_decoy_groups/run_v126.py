"""v126_decoy_groups: stage 2 refit on v110's caches with decoy-signature features.

Test candidates are far less certain than the mock's (uncertain candidates per S1: mock
0.15, India 0.19, US 0.37, France 0.80) and the public score moves only with precision. The
uncertain test pairs show the generator's decoy: the S1 name plus or minus a word at a
*nearby house number* (7541 vs 7532), with its own S2 and S3 records. True records carry the
S1 number, a truncated or padded form of it (201 for 2014), or one source's own corrupted
number shared by all that source's records (600 for 200). On v110's mock (US), differing
first numbers are true 0.97 of the time when one contains the other, 0.65 for a one-digit
change, 0.41 within 20; among uncertain pairs a number no other candidate holds is true 0.37,
shared with one other 0.81. No stage-1 feature reads either signal.

New stage-2 columns on the kept pairs: house-number relation (``hn_*``), candidate groups of
the same S1 (``grp_*``: others sharing the pair's number, name or address, candidates with
the S1's number) and v125's unmatched-token IDF columns. Stage 2 as v122's (XGBoost GPU, 127
leaves, fit + tune out of fold, one seed, lr 0.1 for time); isotonic calibration and v124's
tight decoder, selected by tight@3 on tune; control = v110's mock scores, same decoder.
Files: ``submissions/v126/`` (+ ``output/``) and a tight@6 arm in ``submissions/v126_fp6/``.

    .venv/bin/python experiments/v126_decoy_groups/run_v126.py > run.log 2>&1
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
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
from entity_resolution.model import MatcherParams
from entity_resolution.normalize import apply_token_map
from entity_resolution.pipeline import PipelineConfig, mem_guard, peak_rss_gb, pool_of
from entity_resolution.split import load_fold
from entity_resolution.submission import write_pairs
from entity_resolution.tracking import log_result
from entity_resolution.trainset import inner_split, label_pairs, sample_s1
from entity_resolution.twostage import (
    Stage1Output,
    TwoStageConfig,
    fit_stage2,
    load_stage1,
    mock_scored,
)

VERSION = "v126_decoy_groups"
EXP_DIR = C.EXPERIMENTS / VERSION
ARTIFACTS = EXP_DIR / "artifacts"
V110 = C.EXPERIMENTS / "v110_m3_features"
CFG = PipelineConfig()
STAGE1_CACHE = CFG.cache_dir / "stage1" / "v110_db1f33c4_f0.01_k16_a1_c0_r1"
NORM_KEY = "47a4dda7"          # v110's normalisation cache (rules v3), mock and test alike
NORM_COLS = [C.COUNTRY, "name_core", "addr_norm", "addr_nums", "name_norm", "non_latin"]


def _load_module(name: str, path: Path):
    """Import an earlier experiment's run script (its decoder and feature helpers)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v124 = _load_module("run_v124", C.EXPERIMENTS / "v124_tight_decode" / "run_v124.py")
v125 = _load_module("run_v125", C.EXPERIMENTS / "v125_unmatched_idf" / "run_v125.py")

NUM_COLS = ["hn_eq", "hn_substr", "hn_onedigit", "hn_near", "hn_absdiff", "hn_missing"]
GRP_COLS = ["grp_num_n", "grp_num_p1max", "grp_num_xsrc", "grp_s1num_n", "grp_name_n",
            "grp_name_p1max", "grp_addr_n"]
NEW = [*NUM_COLS, *GRP_COLS, *v125.NEW]
TCFG = TwoStageConfig(
    floor=0.01, max_cands=16, cohesion=False, rivals=True, train_roles=("fit", "tune"),
    model=MatcherParams(backend="xgb", device="cuda", num_leaves=127, learning_rate=0.1,
                        n_estimators=3000, early_stopping=60))
SELECT_W = 3.0                   # tight weight of the main arm (public-validated, v123)
STRICT_W = 6.0                   # stricter arm, written next to it
DECODE_W = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
MISSES = (0.0, 0.05, 0.1, 0.2, 0.4)
T0 = time.time()


def log(msg: str) -> None:
    """Timestamped progress line (the run log is the only view of a background run)."""
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def load_norm(split: str, sources: tuple[int, ...], token_map: dict[str, str],
              ids: pd.Series | None = None, country: str | None = None) -> pd.DataFrame:
    """v110's normalised records (cache ``NORM_KEY``), filtered in Arrow, token map applied."""
    frames = []
    for s in sources:
        path = CFG.cache_dir / "norm" / f"{split}_source{s}_{NORM_KEY}.parquet"
        tbl = pq.read_table(path, columns=[C.ENTITY_ID, *NORM_COLS])
        if country is not None:
            tbl = tbl.filter(pc.equal(tbl[C.COUNTRY], country))
        if ids is not None:
            tbl = tbl.filter(pc.is_in(tbl[C.ENTITY_ID], value_set=pa.array(
                pd.Index(ids).astype("str"), type=tbl.schema.field(C.ENTITY_ID).type)))
        frames.append(tbl.to_pandas())
        del tbl
    out = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return apply_token_map(out, token_map, CFG.normalise)


def _first(nums: np.ndarray) -> np.ndarray:
    """First number of each space-joined ``addr_nums`` string ('' when there is none)."""
    return pd.Series(nums, dtype="str").str.split(" ", n=1).str[0].fillna("").to_numpy()


def num_features(hl: np.ndarray, hr: np.ndarray) -> pd.DataFrame:
    """House-number relation of the first address numbers of both sides (NaN if one lacks one).

    ``hn_substr``: one contains the other (truncation, a unit digit dropped); ``hn_onedigit``:
    same length, one digit differs (a noised digit); ``hn_near``: within 20 and neither of
    those (the decoy's neighbouring number); ``hn_missing``: 1 left, 2 right, 3 both lack one.
    """
    both = (hl != "") & (hr != "")
    eq = both & (hl == hr)
    sub = both & ~eq & np.fromiter(((a in b) or (b in a) for a, b in zip(hl, hr, strict=True)),
                                   dtype=bool, count=len(hl))
    one = both & ~eq & np.fromiter(
        (len(a) == len(b) and sum(x != y for x, y in zip(a, b, strict=True)) == 1
         for a, b in zip(hl, hr, strict=True)), dtype=bool, count=len(hl))
    il = pd.to_numeric(pd.Series(hl), errors="coerce").to_numpy(np.float64)
    ir = pd.to_numeric(pd.Series(hr), errors="coerce").to_numpy(np.float64)
    gap = np.abs(il - ir)
    near = both & ~eq & ~sub & ~one & (gap <= 20)
    na = np.where(both, 0.0, np.nan)
    out = pd.DataFrame({
        "hn_eq": eq + na, "hn_substr": sub + na, "hn_onedigit": one + na,
        "hn_near": near + na, "hn_absdiff": np.log1p(gap) + na,
        "hn_missing": (hl == "").astype(np.float32) + 2 * (hr == "").astype(np.float32)})
    return out.astype(np.float32)


def _others_max(key: pd.DataFrame, p1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: how many other rows share its ``key`` and the best p1 among them (0 if none)."""
    order = np.lexsort((-p1, *[key[c].to_numpy() for c in reversed(key.columns)]))
    k = key.iloc[order].reset_index(drop=True)
    p = p1[order]
    start = np.r_[True, (k.iloc[1:].to_numpy() != k.iloc[:-1].to_numpy()).any(axis=1)]
    grp = np.cumsum(start) - 1
    size = np.bincount(grp)
    first = np.flatnonzero(start)
    top1 = p[first]
    top2 = np.where(size > 1, p[np.minimum(first + 1, len(p) - 1)], 0.0)
    is_first = start
    best_other = np.where(is_first, top2[grp], top1[grp])
    n_other = size[grp] - 1
    out_n, out_best = np.empty(len(p)), np.empty(len(p))
    out_n[order], out_best[order] = n_other, best_other
    return out_n, out_best


def group_features(pairs: pd.DataFrame, p1: np.ndarray, hl: np.ndarray, hr: np.ndarray,
                   name_r: np.ndarray, addr_r: np.ndarray) -> pd.DataFrame:
    """Candidate-set groups of each S1: who shares the pair's number, name and address."""
    s1 = pd.factorize(pairs[C.S1_ID])[0]
    is_s3 = pairs[C.ENTITY_ID].str.startswith("S3").to_numpy()
    n = len(pairs)
    has = hr != ""
    num_n, num_best, xsrc = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    if has.any():
        key = pd.DataFrame({"s1": s1[has], "num": pd.factorize(hr[has])[0]})
        num_n[has], num_best[has] = _others_max(key, p1[has].astype(np.float64))
        # others of the group from the other source
        g = key["s1"].to_numpy().astype(np.int64) * (key["num"].max() + 1) + key["num"].to_numpy()
        gid = pd.factorize(g)[0]
        s3_in = np.bincount(gid, weights=is_s3[has].astype(np.float64))
        size = np.bincount(gid)
        s3_other = s3_in[gid] - is_s3[has]
        s2_other = (size[gid] - s3_in[gid]) - (~is_s3[has])
        xsrc[has] = np.where(is_s3[has], s2_other, s3_other) > 0
    s1num = np.bincount(s1, weights=((hl == hr) & has).astype(np.float64))[s1]
    name_key = pd.DataFrame({"s1": s1, "name": pd.factorize(name_r)[0]})
    name_n, name_best = _others_max(name_key, p1.astype(np.float64))
    addr_has = addr_r != ""
    addr_n = np.full(n, np.nan)
    if addr_has.any():
        g = s1[addr_has].astype(np.int64) * (int(addr_has.sum()) + 1) + pd.factorize(
            addr_r[addr_has])[0]
        gid = pd.factorize(g)[0]
        addr_n[addr_has] = np.bincount(gid)[gid] - 1
    return pd.DataFrame({
        "grp_num_n": num_n, "grp_num_p1max": num_best, "grp_num_xsrc": xsrc,
        "grp_s1num_n": s1num, "grp_name_n": name_n, "grp_name_p1max": name_best,
        "grp_addr_n": addr_n}).astype(np.float32)


def new_features(o: Stage1Output, s1n: pd.DataFrame, pooln: pd.DataFrame) -> pd.DataFrame:
    """Every new column for the kept pairs of ``o`` (row order of ``o.pairs``)."""
    s1 = s1n.drop_duplicates(C.ENTITY_ID).set_index(C.ENTITY_ID)
    pool = pooln.drop_duplicates(C.ENTITY_ID).set_index(C.ENTITY_ID)
    pairs = o.pairs.reset_index(drop=True)
    hl = _first(s1["addr_nums"].reindex(pairs[C.S1_ID]).fillna("").to_numpy())
    hr = _first(pool["addr_nums"].reindex(pairs[C.ENTITY_ID]).fillna("").to_numpy())
    name_r = pool["name_core"].reindex(pairs[C.ENTITY_ID]).fillna("").to_numpy()
    addr_r = pool["addr_norm"].reindex(pairs[C.ENTITY_ID]).fillna("").to_numpy()
    p1 = o.X["p1"].to_numpy(np.float64)
    idf = v125.idf_table(pooln["name_core"])
    parts = [num_features(hl, hr), group_features(pairs, p1, hl, hr, name_r, addr_r),
             v125.new_features(pairs, s1n, pooln, idf).reset_index(drop=True)]
    return pd.concat(parts, axis=1)[NEW]


def with_new(o: Stage1Output, feats: pd.DataFrame) -> Stage1Output:
    """The stage-1 output with the new columns appended to its frame."""
    return Stage1Output(o.pairs.reset_index(drop=True),
                        pd.concat([o.X.reset_index(drop=True), feats], axis=1), o.n_all)


def validate(out_dir: Path) -> str:
    """Our checker (ids checked) and the organisers' validator on both files of ``out_dir``."""
    res = subprocess.run([sys.executable, "-m", "entity_resolution.submission", "--output-dir",
                          str(out_dir), "--check-ids"], capture_output=True, text=True)
    msg = (res.stdout + res.stderr).strip()[-300:]
    if C.OFFICIAL_VALIDATOR.exists():
        res = subprocess.run([sys.executable, str(C.OFFICIAL_VALIDATOR), "--matching",
                              str(out_dir / C.MATCHING_FILE), "--candidate",
                              str(out_dir / C.CANDIDATE_FILE), "--test-dir",
                              str(C.DATASET / "test")], capture_output=True, text=True)
        msg += " | official: " + (res.stdout + res.stderr).strip()[-300:]
    return msg


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    token_map = json.loads((V110 / "artifacts" / "stage1" / "token_map.json").read_text())
    train = load_fold("train", columns=[C.COUNTRY])
    val = load_fold("val", columns=[C.COUNTRY])
    fit_fold, tune_fold = inner_split(train)
    mock = build_mock(train, val, tune_fold.s1[C.ENTITY_ID],
                      sample_s1(fit_fold.s1, CFG.n_fit_s1)[C.ENTITY_ID], target_shape())
    del train, val, fit_fold, tune_fold
    log(f"mock built; token map {len(token_map)} tokens")

    outs = {}
    for country in sorted(mock.fold.s1[C.COUNTRY].unique()):
        t0 = time.time()
        o = load_stage1(STAGE1_CACHE / "mock" / f"mock_{country}.parquet")
        s1_ids = mock.fold.s1[C.ENTITY_ID][(mock.fold.s1[C.COUNTRY] == country).to_numpy()]
        s1n = load_norm("train", (1,), token_map, ids=s1_ids)
        pooln = load_norm("train", (2, 3), token_map, ids=pool_of(mock.fold)[C.ENTITY_ID],
                          country=country)
        feats = new_features(o, s1n, pooln)
        outs[country] = with_new(o, feats)
        y = label_pairs(o.pairs, mock.fold.pairs)["label"].to_numpy()
        rates = {c: float(np.corrcoef(np.nan_to_num(feats[c].to_numpy(), nan=-1), y)[0, 1])
                 for c in NUM_COLS + GRP_COLS}
        log(f"mock {country}: {len(o.pairs):,} kept pairs featured ({time.time() - t0:.0f} s); "
            f"corr with label {json.dumps({k: round(v, 3) for k, v in rates.items()})}")
        del s1n, pooln, o, feats
        mem_guard(country)

    t0 = time.time()
    models, info = fit_stage2(outs, mock, TCFG)
    log(f"stage 2 fit {time.time() - t0:.0f} s: "
        f"{ {k: v.get('best_iteration') for k, v in info.items() if k.startswith('fold')} }")
    for k, m in enumerate(models):
        m.save(ARTIFACTS / f"stage2_{k}")
    scored, _ = mock_scored(outs, models, mock, TCFG)
    scored.to_parquet(ARTIFACTS / "mock_scored.parquet", index=False)
    importance = pd.concat([m.importance() for m in models], axis=1).mean(axis=1)
    importance = importance.sort_values(ascending=False)
    log(f"new columns' gain share {float(importance.reindex(NEW).fillna(0).sum()):.4f}; "
        f"{importance.reindex(NEW).round(4).to_dict()}\ntop\n{importance.head(15)}")
    del outs
    mem_guard("mock scored")

    tune_part, val_part = mock.part("tune"), mock.part("val")
    tune_ids, val_ids = pd.Index(tune_part.s1[C.ENTITY_ID]), pd.Index(val_part.s1[C.ENTITY_ID])

    def arms_of(sc: pd.DataFrame, label: str) -> tuple[dict, IsotonicRegression]:
        """Isotonic fit on tune, decode grid on tune, val scores of the x3 and x6 picks."""
        rows_t = sc[isin(sc[C.S1_ID], tune_ids)]
        rows_v = sc[isin(sc[C.S1_ID], val_ids)]
        y_t = label_pairs(rows_t[[C.S1_ID, C.ENTITY_ID]], mock.fold.pairs)["label"].to_numpy()
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(rows_t["prob"].to_numpy(np.float64), y_t)
        cal_t = rows_t.assign(prob=iso.predict(rows_t["prob"].to_numpy(np.float64)))
        cal_v = rows_v.assign(prob=iso.predict(rows_v["prob"].to_numpy(np.float64)))
        grid = []
        for w in DECODE_W:
            for miss in MISSES:
                counts = entity_counts(v124.decode_tight(cal_t, w, miss), tune_part)
                grid.append({"w": w, "miss": miss,
                             **{f"tune_tight{s:g}":
                                float(entity_tight_from_counts(*counts, s).mean())
                                for s in (1.0, SELECT_W, STRICT_W)}})
        grid = pd.DataFrame(grid)
        arms = {}
        for sel in (SELECT_W, STRICT_W):
            best = grid.sort_values(f"tune_tight{sel:g}", ascending=False).iloc[0]
            m_v = v124.decode_tight(cal_v, float(best["w"]), float(best["miss"]))
            s = score_pairs(m_v, val_part)
            errors = {k: len(error_samples(m_v, val_part, k, n=10**9))
                      for k in ("false_merge", "singleton_merge", "missed", "false_singleton")}
            arms[f"x{sel:g}"] = {
                "w": float(best["w"]), "miss": float(best["miss"]),
                "f_beta": float(s["f_beta"]), "pair_precision": float(s["pair_precision"]),
                "pair_recall": float(s["pair_recall"]),
                **{f"tight{t:g}": v124.macro_tight(m_v, val_part, t)
                   for t in (1.45, SELECT_W, STRICT_W)}, "errors": errors}
            log(f"{label} arm x{sel:g}: {json.dumps(arms[f'x{sel:g}'])}")
        return arms, iso

    arms, iso = arms_of(scored, "v126")
    control = pd.read_parquet(V110 / "artifacts" / "mock_scored.parquet")
    arms_control, _ = arms_of(control, "v110 control")
    del control, scored
    mem_guard("arms")

    # test: v110's cached stage-1 outputs, the same new columns, v126's models
    s1n_all = load_norm("test", (1,), token_map)
    test_parts = []
    for country in ("France", "India", "US"):
        t0 = time.time()
        o = load_stage1(STAGE1_CACHE / "test" / f"test_{country}.parquet")
        s1n = s1n_all[(s1n_all[C.COUNTRY] == country).to_numpy()]
        pooln = load_norm("test", (2, 3), token_map, country=country)
        o = with_new(o, new_features(o, s1n, pooln))
        del s1n, pooln
        names = models[0].feature_names_
        X = o.X if list(o.X.columns) == names else o.X[names]
        prob = np.mean([m.predict_proba(X) for m in models], axis=0).astype(np.float32)
        test_parts.append(o.pairs.assign(prob=prob, country=country))
        log(f"test {country}: {len(o.pairs):,} pairs scored ({time.time() - t0:.0f} s)")
        del o, X
        mem_guard(country)
    test = pd.concat(test_parts, ignore_index=True)
    del test_parts
    test.to_parquet(ARTIFACTS / "test_scored.parquet", index=False)
    cand = test[[C.S1_ID, C.ENTITY_ID]]
    s1_ids = load_source("test", 1, CFG.dataset_dir, columns=[C.ENTITY_ID])[C.ENTITY_ID]
    test_info = {}
    for arm, sub in (("x3", "v126"), ("x6", "v126_fp6")):
        a = arms[arm]
        parts = []
        for country in ("France", "India", "US"):
            sc = test.loc[test["country"] == country, [C.S1_ID, C.ENTITY_ID, "prob"]]
            sc = sc.assign(prob=iso.predict(sc["prob"].to_numpy(np.float64)))
            parts.append(v124.decode_tight(sc, a["w"], a["miss"]))
        tm = pd.concat(parts, ignore_index=True).sort_values(
            [C.S1_ID, "prob"], ascending=[True, False], kind="stable")
        out = C.ROOT / "submissions" / sub
        paths = write_pairs(tm[[C.S1_ID, C.ENTITY_ID]], cand, s1_ids.tolist(), out)
        country_of = test.drop_duplicates(C.S1_ID).set_index(C.S1_ID)["country"]
        by = tm.groupby(tm[C.S1_ID].map(country_of)).size().to_dict()
        test_info[arm] = {"pairs": int(len(tm)), "by_country": {k: int(v) for k, v in by.items()},
                          "s1_matched": int(tm[C.S1_ID].nunique()), "dir": str(out)}
        log(f"test arm {arm}: {test_info[arm]}; check: {validate(out)}")
        if arm == "x3":
            C.OUTPUT.mkdir(parents=True, exist_ok=True)
            for p in paths:
                shutil.copy2(p, C.OUTPUT / p.name)
            log("x3 arm copied to output/")

    main_arm = arms["x3"]
    record = {
        "hypothesis": "house-number relation and candidate-group features separate the "
                      "generator's nearby-number decoys that the test holds more of",
        "stage1": "v110 (cached mock + test stage-1 outputs, blocking db1f33c4)",
        "new_columns": NEW, "stage2": TCFG.record(), "stage2_fit_info": info,
        "new_gain_share": float(importance.reindex(NEW).fillna(0).sum()),
        "top_importance": importance.head(25).to_dict(),
        "arms": arms, "arms_v110_control": arms_control, "test": test_info,
        "peak_rss_gb": peak_rss_gb(),
    }
    row = log_result(
        EXP_DIR, change="v110 caches; stage 2 + house-number relation, candidate-group and "
                        "unmatched-IDF features; isotonic + tight decode (x3)",
        group="C4", mock_f05=main_arm["f_beta"], cand_recall=None,
        notes=(f"val tight@3 {main_arm['tight3']:.4f} (v110 control "
               f"{arms_control['x3']['tight3']:.4f}); F0.5 {main_arm['f_beta']:.4f}; false "
               f"merges {main_arm['errors']['false_merge']} (control "
               f"{arms_control['x3']['errors']['false_merge']}); x6 arm in submissions/v126_fp6"),
        metrics=record, owner="M1", parent="v110", decision="INVESTIGATE")
    log(f"logged {row}; done, peak RSS {peak_rss_gb()} GB")


if __name__ == "__main__":
    main()
