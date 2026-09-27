"""v124_tight_decode: decode v122's probabilities for the tight objective, calibrated.

v123 showed the leaderboard rewards precision beyond the mock's x1.45 (public 0.966 -> 0.968
with a x3-tuned rule). v123 still used rules whose *set score* is plain expected F0.5; the
grid weight only chose between them. Here the decoder itself maximises the expected tight
score per entity: with F = F0.5(prefix) and F* = F0.5 with the false positives removed,

    tight(prefix) = w * E[F] - (w - 1) * E[F*]      (evaluate.entity_tight_from_counts)

by plug-in expectations (Jansche 2007 style, as `decision._expected_keep`), on stage-2
probabilities calibrated with isotonic regression fitted on the mock's tune pairs. The
candidate set, the stage-1/2 models and the pool-side 1-to-1 are v122's, untouched.

Selection: decode weights w and expected misses are tuned on the mock's tune entities by
tight score at x3 (the weight the public upload validated); the val entities report F0.5,
tight at x1.45 and x3, and error counts against v123's x3 arm. Test files go to
``submissions/v124/`` and ``output/``.

    .venv/Scripts/python experiments/v124_tight_decode/run_v124.py > run.log 2>&1
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from entity_resolution import config as C
from entity_resolution.data import isin, load_source
from entity_resolution.decision import one_to_one_filter
from entity_resolution.evaluate import entity_counts, entity_tight_from_counts, score_pairs
from entity_resolution.mock import build_mock, target_shape
from entity_resolution.split import Fold, load_fold
from entity_resolution.submission import write_pairs
from entity_resolution.tracking import log_result
from entity_resolution.trainset import inner_split, label_pairs, sample_s1

VERSION = "v124_tight_decode"
EXP_DIR = C.EXPERIMENTS / VERSION
V122 = C.EXPERIMENTS / "v122_full_stack"
_spec = importlib.util.spec_from_file_location("run_v122", V122 / "run_v122.py")
v122 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v122)
SELECT_W = 3.0                    # tight weight the tune-side selection scores against
DECODE_W = (2.0, 3.0, 4.0, 5.0)   # decoder's own false-positive weight
MISSES = (0.0, 0.05, 0.1, 0.2, 0.4)
MAX_MATCHES = 11
EPS = 1e-7
T0 = time.time()


def log(msg: str) -> None:
    """Timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def decode_tight(scored: pd.DataFrame, w: float, miss: float,
                 max_matches: int = MAX_MATCHES) -> pd.DataFrame:
    """Per-entity prefix maximising the plug-in expected tight score; pool-side 1-to-1 first.

    ``scored``: source1_entity_id, entity_id, prob (calibrated). Returns the kept pairs.
    """
    s = one_to_one_filter(scored[["source1_entity_id", "entity_id", "prob"]])
    s = s.sort_values(["source1_entity_id", "prob"],
                      ascending=[True, False], kind="stable").reset_index(drop=True)
    s1, _ = pd.factorize(s["source1_entity_id"], use_na_sentinel=False)
    q = np.clip(s["prob"].to_numpy(np.float64), EPS, 1.0 - EPS)
    n = s1.max() + 1 if len(s1) else 0
    total = np.bincount(s1, weights=q, minlength=n) + miss     # E[n_true]
    empty = np.exp(np.bincount(s1, weights=np.log1p(-q), minlength=n) - miss)
    start = np.r_[True, s1[1:] != s1[:-1]] if len(s1) else np.zeros(0, dtype=bool)
    starts = np.flatnonzero(start)
    rank = np.arange(len(s1)) - np.maximum.accumulate(np.where(start, np.arange(len(s1)), 0))
    cum = np.cumsum(q)
    cum = cum - np.r_[0.0, cum][np.maximum.accumulate(np.where(start, np.arange(len(s1)), 0))]
    ef = 1.25 * cum / (0.25 * total[s1] + rank + 1)            # E[F0.5] of the prefix
    ef_star = 1.25 * cum / (0.25 * total[s1] + cum)            # E[F0.5] w/o false positives
    tight = w * ef - (w - 1.0) * ef_star
    tight[rank >= max_matches] = -np.inf
    best = np.full(n, -np.inf)
    best[s1[starts]] = np.maximum.reduceat(tight, starts)
    at_best = np.where(tight >= best[s1] - 1e-12, rank, np.iinfo(np.int64).max)
    k = np.zeros(n, dtype=np.int64)
    k[s1[starts]] = np.minimum.reduceat(at_best, starts)
    keep = (rank <= k[s1]) & (best[s1] > empty[s1])
    return s.loc[keep, ["source1_entity_id", "entity_id", "prob"]].reset_index(drop=True)


def macro_tight(matches: pd.DataFrame, fold: Fold, fp_weight: float) -> float:
    """Mean per-entity tight score of ``matches`` on ``fold``."""
    tp, n_pred, n_true = entity_counts(matches, fold)
    return float(entity_tight_from_counts(tp, n_pred, n_true, fp_weight).mean())


def main() -> None:
    cfg, _ = v122.configs()
    train = load_fold("train", columns=[C.COUNTRY])
    val = load_fold("val", columns=[C.COUNTRY])
    fit_fold, tune_fold = inner_split(train)
    mock = build_mock(train, val, tune_fold.s1[C.ENTITY_ID],
                      sample_s1(fit_fold.s1, cfg.n_fit_s1)[C.ENTITY_ID], target_shape())
    del train, val, fit_fold, tune_fold
    scored = pd.read_parquet(V122 / "artifacts" / "mock_scored.parquet")
    tune_part, val_part = mock.part("tune"), mock.part("val")
    rows_t = scored[isin(scored[C.S1_ID], pd.Index(tune_part.s1[C.ENTITY_ID]))]
    rows_v = scored[isin(scored[C.S1_ID], pd.Index(val_part.s1[C.ENTITY_ID]))]

    # isotonic calibration on the tune pairs (post-1-to-1 rows, as the decoder sees them)
    y_t = label_pairs(rows_t[[C.S1_ID, C.ENTITY_ID]], mock.fold.pairs)["label"].to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(rows_t["prob"].to_numpy(np.float64), y_t)
    before = float(np.abs(rows_t["prob"].mean() - y_t.mean()))
    after = float(np.abs(iso.predict(rows_t["prob"].to_numpy(np.float64)).mean() - y_t.mean()))
    log(f"isotonic on {len(rows_t):,} tune pairs (pos rate {y_t.mean():.4f}); "
        f"mean-prob bias {before:.5f} -> {after:.5f}")

    def calibrated(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(prob=iso.predict(df["prob"].to_numpy(np.float64)).astype(np.float32))

    cal_t, cal_v = calibrated(rows_t), calibrated(rows_v)
    grid = []
    for w in DECODE_W:
        for miss in MISSES:
            m_t = decode_tight(cal_t, w, miss)
            grid.append({"w": w, "miss": miss,
                         "tune_tight3": macro_tight(m_t, tune_part, SELECT_W)})
    grid = pd.DataFrame(grid).sort_values("tune_tight3", ascending=False)
    log("tune grid (top 5):\n" + grid.head(5).to_string(index=False))
    w_best, miss_best = float(grid.iloc[0]["w"]), float(grid.iloc[0]["miss"])

    arms = {}
    v123_arms = json.loads((C.EXPERIMENTS / "v123_tight_rule" / "metrics.json").read_text())[
        "metrics"]["arms"]
    for name, matches in {
            "v124 tight decode": decode_tight(cal_v, w_best, miss_best),
            "v124 uncalibrated": decode_tight(rows_v, w_best, miss_best)}.items():
        s = score_pairs(matches, val_part)
        arms[name] = {**{k: float(v) for k, v in s.items() if isinstance(v, (int, float))},
                      "tight_1.45": macro_tight(matches, val_part, 1.45),
                      "tight_3": macro_tight(matches, val_part, SELECT_W)}
        log(f"{name}: F0.5 {s['f_beta']:.5f}, tight@1.45 {arms[name]['tight_1.45']:.5f}, "
            f"tight@3 {arms[name]['tight_3']:.5f}")
    ref = v123_arms["3.0"]
    log(f"v123 x3 arm (public 0.968): mock F0.5 {ref['f_beta']:.5f}, "
        f"f_tight(1.45) {ref['f_tight']:.5f}")

    # the yardstick: tight@3 on val of v123's x3 rule (public 0.968), same pairs
    from entity_resolution.decision import apply_rule, rule_from_json
    m123 = apply_rule(rows_v, rule_from_json(v123_arms["3.0"]["rule"]))
    ref_tight3 = macro_tight(m123, val_part, SELECT_W)
    chosen = arms["v124 tight decode"]
    log(f"val tight@3: v123 x3 rule {ref_tight3:.5f} -> v124 {chosen['tight_3']:.5f} "
        f"({chosen['tight_3'] - ref_tight3:+.5f})")

    test = pd.read_parquet(V122 / "artifacts" / "test_scored.parquet")
    candidates = test[[C.S1_ID, C.ENTITY_ID]]
    s1_ids = load_source("test", 1, cfg.dataset_dir, columns=[C.ENTITY_ID])[C.ENTITY_ID]
    parts = []
    for country in sorted(test["country"].unique()):
        sc = calibrated(test.loc[test["country"] == country,
                                 [C.S1_ID, C.ENTITY_ID, "prob"]])
        parts.append(decode_tight(sc, w_best, miss_best))
    tm = pd.concat(parts, ignore_index=True)
    tm = tm.sort_values([C.S1_ID, "prob"], ascending=[True, False], kind="stable")
    country_of = test.drop_duplicates(C.S1_ID).set_index(C.S1_ID)["country"]
    by = tm.groupby(tm[C.S1_ID].map(country_of)).size()
    v123_pairs = 5_723_885
    log(f"test: {len(tm):,} pairs ({len(tm) - v123_pairs:+,} vs v123 fp3) {dict(by)}; "
        f"S1 matched {tm[C.S1_ID].nunique():,}")
    out = C.ROOT / "submissions" / "v124"
    paths = write_pairs(tm[[C.S1_ID, C.ENTITY_ID]], candidates, s1_ids.tolist(), out)
    check = subprocess.run([sys.executable, "-m", "entity_resolution.submission",
                            "--output-dir", str(out), "--check-ids"],
                           capture_output=True, text=True)
    log(f"wrote {out}: checker {(check.stdout + check.stderr).strip()[-120:]}")
    C.OUTPUT.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.copy2(p, C.OUTPUT / p.name)
    log("copied to output/")

    record = {
        "hypothesis": "decoding for the tight objective on calibrated probabilities beats "
                      "rule families whose set score is plain expected F0.5",
        "stage1_stage2": "v122 (no retraining)", "calibration": "isotonic on mock tune pairs",
        "select_weight": SELECT_W, "decode_w": w_best, "decode_miss": miss_best,
        "grid": grid.to_dict("records"), "arms": arms,
        "v123_fp3_val": {k: ref[k] for k in ("f_beta", "f_tight", "est_public")},
        "v123_fp3_val_tight3": ref_tight3,
        "delta_tight3_vs_v123": chosen["tight_3"] - ref_tight3,
        "test_pairs": int(len(tm)), "test_pairs_by_country": {k: int(v) for k, v in by.items()},
    }
    row = log_result(
        EXP_DIR, change=f"v122 probabilities, isotonic calibration, tight-objective "
                        f"expected decode (w {w_best:g}, miss {miss_best:g}); no retraining",
        group="E5", mock_f05=chosen["f_beta"], cand_recall=None,
        notes=(f"val tight@3 {chosen['tight_3']:.4f} vs v123-x3 rule; F0.5 "
               f"{chosen['f_beta']:.4f}; judged by the public upload"),
        metrics=record, owner="M3", parent="v123", decision="INVESTIGATE")
    log(f"logged {row}")


if __name__ == "__main__":
    main()
