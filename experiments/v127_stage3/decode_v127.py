"""v127 decode at the false-merge weight the public uploads imply (x4.3), next to x3 and x6.

v122 -> v123 is the one public step that changed only the decision rule: 923 fewer false
merges on mock val for 5,808 more misses, public 0.966 -> 0.968. In per-entity loss terms
(mock val: L_FP 0.00266 -> 0.00184, L_FN 0.01428 -> 0.01582) a public gain of +0.002 means
the public board charges a false merge about 4.3 times a miss (3.1 to 5.5 over the rounding
of the two public scores). v126/v127 select their decoder by tight@3; this script selects it
by tight@4.3 as well, on the same tune entities, reports every arm on val with the public
estimate of that loss model, and writes the x4.3 arm to ``submissions/v127_fp43/``.

Reads a version's ``mock_scored.parquet`` and ``test_scored.parquet`` (v127 by default,
v126 with ``--version v126``: same frames, same decoder).

    .venv/bin/python experiments/v127_stage3/decode_v127.py > decode.log 2>&1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from entity_resolution import config as C
from entity_resolution.data import isin, load_source
from entity_resolution.evaluate import entity_counts, entity_tight_from_counts, score_pairs
from entity_resolution.mock import build_mock, target_shape
from entity_resolution.split import load_fold
from entity_resolution.trainset import inner_split, label_pairs, sample_s1

V127_DIR = C.EXPERIMENTS / "v127_stage3"
_spec = importlib.util.spec_from_file_location("run_v127", V127_DIR / "run_v127.py")
v127 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v127)
v126, v124 = v127.v126, v127.v126.v124
PUBLIC_W = 4.3                   # false-merge weight implied by public v122 -> v123
SELECT = (3.0, PUBLIC_W, 6.0)
V123 = {"public": 0.968, "l_fn": 0.015818, "l_fp": 0.001836}   # mock val loss split of v123 x3
T0 = time.time()


def log(msg: str) -> None:
    """Timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", choices=("v126", "v127"), default="v127")
    args = ap.parse_args()
    art = v127.ARTIFACTS if args.version == "v127" else v126.ARTIFACTS
    train = load_fold("train", columns=[C.COUNTRY])
    val = load_fold("val", columns=[C.COUNTRY])
    fit_fold, tune_fold = inner_split(train)
    mock = build_mock(train, val, tune_fold.s1[C.ENTITY_ID],
                      sample_s1(fit_fold.s1, v126.CFG.n_fit_s1)[C.ENTITY_ID], target_shape())
    del train, val, fit_fold, tune_fold
    tune_part, val_part = mock.part("tune"), mock.part("val")
    scored = pd.read_parquet(art / "mock_scored.parquet")
    rows_t = scored[isin(scored[C.S1_ID], pd.Index(tune_part.s1[C.ENTITY_ID]))]
    rows_v = scored[isin(scored[C.S1_ID], pd.Index(val_part.s1[C.ENTITY_ID]))]
    y_t = label_pairs(rows_t[[C.S1_ID, C.ENTITY_ID]], mock.fold.pairs)["label"].to_numpy()
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(rows_t["prob"].to_numpy(np.float64), y_t)
    cal_t = rows_t.assign(prob=iso.predict(rows_t["prob"].to_numpy(np.float64)))
    cal_v = rows_v.assign(prob=iso.predict(rows_v["prob"].to_numpy(np.float64)))
    log("mock built, isotonic fitted on tune")

    grid = []
    for w in (*v126.DECODE_W, 10.0, 12.0):
        for miss in v126.MISSES:
            counts = entity_counts(v124.decode_tight(cal_t, w, miss), tune_part)
            grid.append({"w": w, "miss": miss, **{
                f"t{s:g}": float(entity_tight_from_counts(*counts, s).mean()) for s in SELECT}})
    grid = pd.DataFrame(grid)
    arms = {}
    for sel in SELECT:
        best = grid.sort_values(f"t{sel:g}", ascending=False).iloc[0]
        m_v = v124.decode_tight(cal_v, float(best["w"]), float(best["miss"]))
        s = score_pairs(m_v, val_part)
        tp, n_pred, n_true = entity_counts(m_v, val_part)
        f = float(entity_tight_from_counts(tp, n_pred, n_true, 1.0).mean())
        l_fp = float(entity_tight_from_counts(tp, n_pred, n_true, 2.0).mean())
        l_fp = f - l_fp                                   # tight@2 = F - L_FP
        l_fn = 1.0 - f - l_fp
        est = V123["public"] - (l_fn - V123["l_fn"]) - PUBLIC_W * (l_fp - V123["l_fp"])
        arms[f"x{sel:g}"] = {"w": float(best["w"]), "miss": float(best["miss"]),
                             "f_beta": float(s["f_beta"]), "l_fn": l_fn, "l_fp": l_fp,
                             "est_public": est}
        log(f"arm x{sel:g}: {json.dumps(arms[f'x{sel:g}'])}")

    test = pd.read_parquet(art / "test_scored.parquet")
    s1_ids = load_source("test", 1, v126.CFG.dataset_dir, columns=[C.ENTITY_ID])[C.ENTITY_ID]
    key = f"x{PUBLIC_W:g}"
    info = v127.write_arms(test, {key: arms[key]}, iso, s1_ids,
                           {key: f"{args.version}_fp43"}, copy_x3=False)
    out = {"version": args.version, "public_w": PUBLIC_W, "v123_reference": V123,
           "arms": arms, "test_fp43": info[key]}
    path = V127_DIR / f"decode_arms_{args.version}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    log(f"wrote {path.name}; x{PUBLIC_W:g} arm files in submissions/{args.version}_fp43")


if __name__ == "__main__":
    main()
