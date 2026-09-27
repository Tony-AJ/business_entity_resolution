"""v123_tight_rule: v122's matcher with its rule tuned for costlier false merges.

Why: the public score stayed at 0.966 for v107, v110 and v122 while the tight mock's
est_public rose 0.9659 -> 0.9731 -> 0.9747. The gains that did reach the leaderboard were
precision gains (v101 -> v103 -> v107: 0.955 -> 0.961 -> 0.966, rule and decoding changes);
recall gains did not. v122's own probabilities also show a harder test than the mock:
candidates with 0.1 < p < 0.9 per S1 are 0.15 on mock val, 0.19 India, 0.37 US and 0.80
France (unseen in train). So false merges cost more on test than the mock's x1.45 says.

What: no model is retrained. v122's stage-2 probabilities of the mock (tune + val entities)
and of every kept test pair (``test_scored.parquet``, written from v122's cached stage-1
outputs) are decoded with rules tuned on the mock's tune entities for false-merge weights
1.45 (v122) to 6, v107's rule choice (expected-F0.5 or threshold grid, the better). The
candidate file is v122's (the kept pairs do not depend on the rule).

Checks: v122's own rule applied to ``test_scored.parquet`` must rebuild v122's uploaded
matching file byte for byte. Test files for weights 3 and 6 go to ``submissions/v123_fp3``
and ``submissions/v123_fp6``; weight 3 (the upload candidate) is copied to ``output/``.

    .venv/Scripts/python experiments/v123_tight_rule/run_v123.py > run.log 2>&1
"""
from __future__ import annotations

import filecmp
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from entity_resolution import config as C
from entity_resolution.data import isin, load_source
from entity_resolution.decision import (
    SCORED_COLUMNS,
    apply_rule,
    rule_from_json,
    rule_to_json,
    tune_expected,
)
from entity_resolution.evaluate import error_samples
from entity_resolution.mock import FP_WEIGHT, build_mock, target_shape
from entity_resolution.pipeline import mock_scores, sort_matches, tune_mock
from entity_resolution.split import load_fold
from entity_resolution.submission import write_pairs
from entity_resolution.tracking import log_result
from entity_resolution.trainset import inner_split, sample_s1

VERSION = "v123_tight_rule"
EXP_DIR = C.EXPERIMENTS / VERSION
V122 = C.EXPERIMENTS / "v122_full_stack"
_spec = importlib.util.spec_from_file_location("run_v122", V122 / "run_v122.py")
v122 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v122)
WEIGHTS = (FP_WEIGHT, 2.0, 3.0, 4.0, 6.0)   # false merge cost / missed match cost
WRITE = {3.0: "v123_fp3", 6.0: "v123_fp6"}   # weight -> submissions folder
UPLOAD = 3.0                                 # the weight copied to output/
GAMMAS = tuple(np.round(np.arange(0.7, 3.01, 0.1), 2))
MISSES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6)
KINDS = ("false_merge", "singleton_merge", "missed", "false_singleton")
T0 = time.time()


def log(msg: str) -> None:
    """Timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def tuned_rule(scored: pd.DataFrame, mock, grid, weight: float):
    """v107's rule choice at false-merge ``weight``: expected-F0.5 or threshold grid."""
    tune_part = mock.part("tune")
    rows_t = scored[isin(scored[C.S1_ID], pd.Index(tune_part.s1[C.ENTITY_ID]))]
    rule_e, table_e = tune_expected(rows_t, tune_part.s1[C.ENTITY_ID], tune_part.pairs,
                                    gammas=GAMMAS, misses=MISSES, fp_weight=weight)
    rule_t, table_t = tune_mock(scored, mock, grid, fp_weight=weight)
    return rule_e if table_e["f_beta"].max() > table_t["f_beta"].max() else rule_t


def test_matches(test: pd.DataFrame, rule) -> pd.DataFrame:
    """``run_test_two_stage``'s decision on the scored test pairs, one country at a time."""
    parts = []
    for country in sorted(test["country"].unique()):
        scored = test.loc[test["country"] == country, SCORED_COLUMNS]
        parts.append(sort_matches(apply_rule(scored, rule), scored))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    cfg, _ = v122.configs()
    train = load_fold("train", columns=[C.COUNTRY])
    val = load_fold("val", columns=[C.COUNTRY])
    fit_fold, tune_fold = inner_split(train)
    mock = build_mock(train, val, tune_fold.s1[C.ENTITY_ID],
                      sample_s1(fit_fold.s1, cfg.n_fit_s1)[C.ENTITY_ID], target_shape())
    del train, val, fit_fold, tune_fold
    scored = pd.read_parquet(v122.ARTIFACTS / "mock_scored.parquet")
    test = pd.read_parquet(v122.ARTIFACTS / "test_scored.parquet")
    candidates = test[[C.S1_ID, C.ENTITY_ID]]
    s1_ids = load_source("test", 1, cfg.dataset_dir, columns=[C.ENTITY_ID])[C.ENTITY_ID]
    log(f"mock built; {len(scored):,} mock rows, {len(test):,} test pairs")

    # v122's rule on the saved test probabilities must rebuild v122's uploaded file
    rule122 = rule_from_json(json.loads((V122 / "artifacts" / "two_stage" / "rule.json")
                                        .read_text()))
    with tempfile.TemporaryDirectory() as tmp:
        m, _ = write_pairs(test_matches(test, rule122)[[C.S1_ID, C.ENTITY_ID]], candidates,
                           s1_ids.tolist(), Path(tmp))
        same = filecmp.cmp(m, C.ROOT / "submissions" / "v122" / C.MATCHING_FILE, shallow=False)
    log(f"v122 rule {rule122} rebuilds submissions/v122 matching file: {same}")
    if not same:
        raise SystemExit("test_scored.parquet does not reproduce v122: stop")

    val_part = mock.part("val")
    val_rows = scored[isin(scored[C.S1_ID], pd.Index(val_part.s1[C.ENTITY_ID]))]
    base_pairs = None
    arms, files = {}, {}
    for w in WEIGHTS:
        rule = tuned_rule(scored, mock, cfg.grid, w)
        res = mock_scores(scored, mock, rule)
        picked = apply_rule(val_rows, rule)
        errors = {k: len(error_samples(picked, val_part, k, n=10**9)) for k in KINDS}
        tm = test_matches(test, rule)
        pairs = set(zip(tm[C.S1_ID], tm[C.ENTITY_ID], strict=True))
        base_pairs = pairs if base_pairs is None else base_pairs
        country = test.drop_duplicates(C.S1_ID).set_index(C.S1_ID)["country"]
        by = tm.groupby(tm[C.S1_ID].map(country)).size()
        arms[w] = {"rule": rule_to_json(rule), **{k: float(res.loc["all", k]) for k in
                                                   v122.SCORE_COLS},
                   **{f"mock_{c}": float(res.loc[c, "f_beta"]) for c in res.index if c != "all"},
                   "errors_mock": errors, "test_pairs": len(tm),
                   "test_pairs_by_country": {k: int(v) for k, v in by.items()},
                   "test_s1_matched": int(tm[C.S1_ID].nunique()),
                   "test_pairs_removed_vs_v122": len(base_pairs - pairs),
                   "test_pairs_added_vs_v122": len(pairs - base_pairs)}
        log(f"weight {w}: {rule}; mock F0.5 {arms[w]['f_beta']:.5f}, est_public(x1.45) "
            f"{arms[w]['est_public']:.5f}, errors {errors}; test pairs {len(tm):,} "
            f"(-{arms[w]['test_pairs_removed_vs_v122']:,} / "
            f"+{arms[w]['test_pairs_added_vs_v122']:,} vs v122) {arms[w]['test_pairs_by_country']}")
        if w in WRITE:
            out = C.ROOT / "submissions" / WRITE[w]
            files[w] = write_pairs(tm[[C.S1_ID, C.ENTITY_ID]], candidates, s1_ids.tolist(), out)
            check = subprocess.run([sys.executable, "-m", "entity_resolution.submission",
                                    "--output-dir", str(out), "--check-ids"],
                                   capture_output=True, text=True)
            log(f"wrote {out}: checker {(check.stdout + check.stderr).strip()[-200:]}")

    C.OUTPUT.mkdir(parents=True, exist_ok=True)
    for p in files[UPLOAD]:
        shutil.copy2(p, C.OUTPUT / p.name)
    log(f"output/ now holds weight {UPLOAD} ({WRITE[UPLOAD]})")

    chosen, base = arms[UPLOAD], arms[FP_WEIGHT]
    record = {
        "hypothesis": "v122 decoded for costlier false merges raises the public score: "
                      "precision gains reached the leaderboard, recall gains did not",
        "stage1_stage2": "v122 (no retraining)", "weights": list(WEIGHTS),
        "upload_weight": UPLOAD, "arms": {str(k): v for k, v in arms.items()},
        "rule": chosen["rule"], **{k: chosen[k] for k in v122.SCORE_COLS},
        "v122_est_public": base["est_public"], "v122_f_beta": base["f_beta"],
        "reproduces_v122": True, "evidence_test_harder": {
            "uncertain_cands_per_s1": {"mock_val": 0.1527, "India": 0.1927, "US": 0.3673,
                                       "France": 0.8015},
            "self_expected_f05": {"mock_val": 0.99212, "India": 0.99076, "US": 0.98517,
                                  "France": 0.96438}},
    }
    row = log_result(
        EXP_DIR, change=f"v122 matcher, rule tuned for false merges x{UPLOAD:g} (v122: x1.45); "
                        "no retraining",
        group="E2", mock_f05=chosen["f_beta"], cand_recall=None,
        notes=(f"mock F0.5 {chosen['f_beta']:.4f} (v122 {base['f_beta']:.4f}); false merges "
               f"{chosen['errors_mock']['false_merge']} vs {base['errors_mock']['false_merge']}; "
               f"est_public(x1.45) {chosen['est_public']:.4f}; judged by the public upload, "
               f"since v107-v122 left public at 0.966"),
        metrics=record, owner="M3", parent="v122", decision="INVESTIGATE")
    log(f"logged {row}")


if __name__ == "__main__":
    main()
