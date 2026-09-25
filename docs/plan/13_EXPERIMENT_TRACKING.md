# 13 — Experiment tracking

Audience: everyone. How a version is created, what it records, how it is judged and how it
is committed. The mechanism is `src/entity_resolution/tracking.py` (`new_experiment`,
`log_result`, `set_public_score`, `timed`); git rules are in 14, uploads in 15, the final
package in 16.

## 1. One experiment = one version folder

```
experiments/vNNN_<slug>/
├── vNNN_<slug>.ipynb    documented notebook, committed with its outputs (< 2 MB)
├── metrics.json         everything log_result() recorded, committed
└── artifacts/           model/, rule.json, tune_table.csv, scored_val.parquet, errors.parquet: gitignored
```

Create it with `make experiment NAME=<slug> V=<number>`, `V` taken from your reserved range
below. M1 adds `V=` on day 1 (`tracking.new_experiment(slug, number=N)`, CLI
`python -m entity_resolution.tracking new <slug> --number N`). Without `V` the next free
number is taken, which lands in someone else's range: only M1 omits it. Slugs are snake_case
letters and digits (`p3_word_pass`, `lgbm_hardneg2`); the folder is `v012_p3_word_pass`.
Versions are never renumbered, reused or deleted: a dropped idea keeps its folder as history.

| Member | Role | Versions | Groups |
|---|---|---|---|
| M1 | lead, integration, uploads | v001–v009 (skeleton, baselines); v100–v129 (integrated runs, the only uploadable numbers) | all, INT |
| M2 | normalisation, blocking | v010–v039 | A, B |
| M3 | features, training pairs | v040–v059 | C |
| M4 | matcher, hard negatives | v060–v079 | D, HN |
| M5 | decision layer, error analysis | v080–v099 | E |

Gaps are fine (v010, v012, v015); numbers outside your range are not. A member who runs out
asks M1 for a block of ten from v130 upwards.

### 1.1 Notebook header

The template (`experiments/_template/experiment.ipynb`) opens with a table; fill every
field before the first run:

| Field | Value |
|---|---|
| Version | `v012_p3_word_pass` (the folder name) |
| Plan group | one plan ID A1–E5, `HN` for hard-negative rounds, `INT` for integrated v1xx runs |
| Parent version | the logged version this builds on (`v011`); `—` only for v001 |
| Author (owner) | your role tag and name: `M2 <name>` |
| Date | `2026-09-25` |
| Status | `running` → `kept` / `discarded` / `shortlisted` / `submitted` |
| Hypothesis | notebook §1: one falsifiable sentence with the expected effect, e.g. "P3 raises pair recall from 0.86 to ≥ 0.95 at ≤ 35 cands/S1" |

The parent is a real, logged version whose `metrics.json` you compare against. A version
that combines two ideas names the one whose config it copied and lists the other in `notes`.
The sections that follow are the template's: 2 Setup, 3 Data, 4 Method (4.1 normalisation …
4.5 decision rule; unchanged stages say "as in parent"), 5 Evaluation on the val fold,
6 Error analysis, 7 Log the result, 8 Conclusion, 9 Test inference (shortlisted only). Every
code cell has a markdown cell before it saying what and why, and every function has a
docstring: the organisers require "proper comments describing the functions".

## 2. The record

### 2.1 `experiments/experiments.csv`: one row per version

`tracking.COLUMNS` today: `version, date, group, change, local_f05, cand_recall,
public_f05, commit, notes`. M1 appends `owner, parent, decision` on day 1
(`log_result(..., owner=, parent=, decision=)`); earlier rows get empty values.

| Column | Filled by | Content |
|---|---|---|
| version | folder name | `v012` |
| date | `log_result` | ISO date of the run |
| group | you | plan ID A1–E5, `HN`, or `INT` |
| change | you | one line: what differs from the parent, with parameter values |
| local_f05 | `scores["f_beta"]` | macro F0.5 on `split.load_fold("val")`, 4 decimals; empty for blocking-only versions |
| cand_recall | `blocking["pair_recall"]` | blocking pair recall on val |
| public_f05 | `make public V=vNNN SCORE=x` | leaderboard score of uploaded versions; survives re-runs |
| commit | `tracking.git_commit()` | short HEAD hash, `-dirty` when a tracked file under `src/` is modified |
| notes | you | cands/S1, singleton F0.5, run time, anything needed to read the row |
| owner | you | `M1`–`M5` |
| parent | you | `v011` |
| decision | you | `KEEP` / `DROP` / `INVESTIGATE` (§3) |

Re-running a notebook rewrites its own row (rows are keyed by version and re-sorted on every
write), so the csv holds the latest run of each version; earlier numbers stay in git history.

### 2.2 `metrics.json`

`log_result(EXP_DIR, ..., metrics=record)` writes `{**row, "metrics": record}`. `record`
uses the standard keys below so any two versions can be diffed by a script and so the
results table of the final documentation (16 §4) is generated from the json files rather
than retyped. A key may be absent when its stage does not exist in that version (a
blocking-only version has no `rule`); a synonym for an existing key is never allowed.

| Group | Keys |
|---|---|
| Identity, configuration | `hypothesis` (str); `blocking_config` (dict, `dataclasses.asdict(cfg.blocking)`); `feature_groups` (list of registry names); `model_params` (dict, `asdict(cfg.model)`); `rule` (dict: `tau_abs`, `tau_rel`, `tau_single`, `max_matches`, `one_to_one`) |
| Val scores (`metrics.breakdown`) | `f_beta`, `f_beta_singletons`, `f_beta_matched`, `pair_precision`, `pair_recall` |
| Blocking (`metrics.candidate_report`) | `cand_recall` (its `pair_recall`), `entity_recall`, `ceiling_f_beta`, `cands_mean`, `cands_p95` (its `candidates_mean` / `candidates_p95`) |
| Robustness | `harder_f_beta` (same fitted rule scored on `evaluate.harder_fold(val)`, 11 §6), `tune_f_beta` (best grid point on the tune split) |
| Errors (`errors.tag_errors`) | `n_fp`, `n_fn`, `n_false_singleton` (matched entities predicted empty), `errors` (dict category → count, categories of 18) |
| Timings (`tracking.timed`) | `load_seconds`, `normalise_seconds`, `blocking_seconds`, `features_seconds`, `fit_seconds`, `tune_seconds`, `score_seconds`, `decide_seconds` |
| Resources, verdict | `peak_rss_gb` (max `psutil` RSS seen by `mem_guard`), `decision` |

One trap: `breakdown` and `candidate_report` both return `pair_recall`, and the template's
`{**blocking, **scores}` silently keeps the matcher's. Rename the blocking one to
`cand_recall` before merging, as below.

```python
record = {
    "hypothesis": HYPOTHESIS,
    "blocking_config": asdict(cfg.blocking), "feature_groups": list(cfg.feature_groups),
    "model_params": asdict(cfg.model), "rule": asdict(fitted.rule),
    **scores,                                       # breakdown(): f_beta ... pair_recall
    "cand_recall": blocking["pair_recall"], "entity_recall": blocking["entity_recall"],
    "ceiling_f_beta": blocking["ceiling_f_beta"],
    "cands_mean": blocking["candidates_mean"], "cands_p95": blocking["candidates_p95"],
    "harder_f_beta": harder["f_beta"], "tune_f_beta": float(fitted.tune_table["f_beta"].max()),
    "n_fp": int(n_fp), "n_fn": int(n_fn), "n_false_singleton": int(n_false_singleton),
    "errors": error_counts,                         # {"decoy_same_name": 12340, ...}
    **timings, "peak_rss_gb": peak_rss_gb, "decision": DECISION,
}
```

`timings` is filled by wrapping each stage, `with timed("blocking", timings): pairs =
block(...)`, which stores `timings["blocking_seconds"]` (wall time, 0.01 s). Use exactly the
eight labels above so run times compare across versions; `pipeline.run_fold` returns the
same names. `json.dumps(default=float)` handles numpy scalars; tuples become lists.

## 3. KEEP / DROP / INVESTIGATE

Judged against the parent's `metrics.json`, on the fixed val fold, with the standard slice
report (`evaluate.slice_report`, 11 §7: country, singleton vs matched, 1-match vs 2+ matches,
`non_latin`, ambiguous `name_core`, empty address).

| Decision | Condition | Consequence |
|---|---|---|
| KEEP | `f_beta` > parent + 0.002 **and** no slice `f_beta` falls by more than 0.01 **and** `harder_f_beta` not lower | may become the parent of the next version; a PR may carry the change into `main` (14 §5) |
| DROP | `f_beta` ≤ parent, or a slice falls by more than 0.01 without an offsetting, understood reason | folder and row stay; the next version branches from the parent again |
| INVESTIGATE | mixed signals: `cand_recall` up but `f_beta` down, singleton F0.5 up but matched F0.5 down, val up but harder-val down, run time or RSS more than doubled | parent for nobody until resolved; the next version isolates the cause, one change at a time |

Blocking-only versions (groups A and B without a matcher) use `ceiling_f_beta` in place of
`f_beta`, with `cands_mean` ≤ 40 and no per-pass recall lost on the `non_latin` or
ambiguous-name slices (06 §5). The 0.002 margin assumes a deterministic pipeline (fixed
seeds, `num_threads` fixed); M1 re-runs v001 twice on day 1 and raises the margin if the two
runs differ.

## 4. Commit protocol (a row without a reproducible commit is worthless)

1. Commit `src/` changes first, on your feature branch, before you run. `git_commit()`
   appends `-dirty` when a **tracked** file under `src/` has uncommitted changes; it does not
   see untracked files (`--untracked-files=no`), so `git add` and commit a new module before
   running or the csv will name a commit that lacks your code.
2. Run the notebook (VS Code, or headless: `make nb
   NB=experiments/v012_p3_word_pass/v012_p3_word_pass.ipynb`). Its last cell calls `log_result`.
3. Check the row (`grep '^v012,' experiments/experiments.csv`). If `commit` ends in `-dirty`,
   commit and re-run: candidates are cached by blocking-config hash, so a re-run is cheap.
4. Commit notebook, `metrics.json` and the csv row together:

```bash
git add experiments/v012_p3_word_pass/v012_p3_word_pass.ipynb \
        experiments/v012_p3_word_pass/metrics.json experiments/experiments.csv
git commit -m "exp(v012): P3 name+address word pass k=20, cand recall 0.9520"
```

Versions with a matcher use `exp(vNNN): <change>, local F0.5 0.xxxx`; blocking-only versions
`cand recall 0.xxxx` or `ceiling F0.5 0.xxxx`. Subject ≤ 72 characters (hook). Keep the
notebook under 2 MB: show `df.head(20)` and `describe()`, never full frames, and clear
large plots before the final execution.

## 5. From local versions to an upload

Expect 10–20 local versions per upload (11 §11): 15 uploads exist for the whole challenge
(15 §1), so the csv is where exploration happens. Shortlist rule, applied by M1 at each upload slot:

1. Only integrated versions (v1xx) are shortlisted: a module improvement counts once M1 has
   merged it and re-run the full pipeline under a v1xx number (14 §5).
2. Candidates are KEEP versions with `f_beta` ≥ last uploaded + 0.003 and `harder_f_beta` ≥
   last uploaded − 0.002.
3. The highest `f_beta` wins; ties go to the higher `f_beta_singletons`, because the test
   pool has more unmatched records than train (01 §2).
4. The header `Status` becomes `shortlisted`, then `submitted` after the upload; the
   eligibility checklist of 15 §2 runs before the button is pressed.

## 6. Daily log ritual

At **13:00** and **21:00 IST** every member posts one line per version run since the last
post, in the team chat, in this form, plus a "next: …" line naming the next hypothesis:

```
M2  v013  A3  P3 word pass k=20 max_df=0.05   cand_recall 0.952 (+0.090)  ceiling 0.982  cands 31/S1  KEEP  parent v012
M4  v062  D3  lgbm num_leaves 127             f_beta 0.8390 (+0.0011)  harder 0.8210 (-0.0030)  INVESTIGATE  parent v061
```

M1 replies with the integration state: which KEEPs are merged, the current best v1xx and its
upload status, the next PR window. The 21:00 post fixes the next morning's first PR window.
A version that was not posted is not merged.

## 7. Worked example (illustrative numbers)

Row in `experiments.csv`:

```
version,date,group,change,local_f05,cand_recall,public_f05,commit,notes,owner,parent,decision
v103,2026-09-26,C3,address feature group (addr_jaccard/nums/postcode) on v102,0.8412,0.9710,,a1b2c3d,"cands 31/S1; singleton F0.5 0.79; India P2 14 min",M1,v102,KEEP
```

`experiments/v103_addr_features/metrics.json`:

```json
{
  "version": "v103", "date": "2026-09-26", "group": "C3",
  "change": "address feature group (addr_jaccard/nums/postcode) on v102",
  "local_f05": "0.8412", "cand_recall": "0.9710", "public_f05": "", "commit": "a1b2c3d",
  "notes": "cands 31/S1; singleton F0.5 0.79; India P2 14 min",
  "owner": "M1", "parent": "v102", "decision": "KEEP",
  "metrics": {
    "hypothesis": "address features separate same-name decoys: matched F0.5 +0.01, singleton F0.5 +0.03",
    "blocking_config": {"exact_keys": ["name_core", "name_sorted", "name_squash"], "exact_max_group": 50,
      "name_char": {"column": "name_core", "analyzer": "char_wb", "ngram": [3, 3], "top_k": 30,
                    "min_sim": 0.3, "max_df": 0.2, "min_df": 2, "sublinear_tf": true},
      "name_addr_word": {"column": "name_addr", "analyzer": "word", "ngram": [1, 1], "top_k": 20,
                         "min_sim": 0.2, "max_df": 0.05, "min_df": 2, "sublinear_tf": true},
      "addr_char": null, "max_per_s1": 60, "s1_chunk": 50000, "pool_chunk": 2000000, "n_threads": 12},
    "feature_groups": ["blocking", "name_fuzzy", "name_tokens", "legal", "numeric", "address"],
    "model_params": {"backend": "lgbm", "num_leaves": 63, "learning_rate": 0.05, "n_estimators": 2000,
      "early_stopping": 100, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
      "min_data_in_leaf": 200, "lambda_l2": 1.0, "max_bin": 255, "scale_pos_weight": 1.0,
      "seed": 42, "num_threads": 12},
    "rule": {"tau_abs": 0.62, "tau_rel": 0.25, "tau_single": 0.55, "max_matches": 11, "one_to_one": true},
    "f_beta": 0.8412, "f_beta_singletons": 0.7904, "f_beta_matched": 0.8442,
    "pair_precision": 0.9312, "pair_recall": 0.8607,
    "cand_recall": 0.9710, "entity_recall": 0.9851, "ceiling_f_beta": 0.9748,
    "cands_mean": 31.2, "cands_p95": 58.0,
    "harder_f_beta": 0.8261, "tune_f_beta": 0.8455,
    "n_fp": 41210, "n_fn": 36875, "n_false_singleton": 2290,
    "errors": {"blocking_miss": 21050, "decoy_same_name": 12340, "address_drift": 8900,
               "transliteration": 4560, "below_threshold": 15825, "false_singleton": 2290},
    "load_seconds": 41.3, "normalise_seconds": 65.0, "blocking_seconds": 231.5, "features_seconds": 190.2,
    "fit_seconds": 402.7, "tune_seconds": 88.1, "score_seconds": 154.4, "decide_seconds": 22.9,
    "peak_rss_gb": 4.8, "decision": "KEEP"
  }
}
```

The parent v102 had `f_beta` 0.8290 and `harder_f_beta` 0.8154; no slice fell by more than
0.01, so v103 is KEEP and the harder-fold gain confirms it survives more distractors.

## 8. What never goes into git

- `experiments/*/artifacts/` (models, `rule.json`, tune tables, scored pairs, error frames),
  `models/`, `dataset/` with its `.cache/`, `output/`, and any `*.tsv`, `*.parquet`,
  `*.joblib`, `*.pkl`, `*.zip`, `.env`: gitignored and rejected by the pre-commit hook.
- Candidate sets: test candidates live in `dataset/.cache/pipeline/test/<country>/pairs_<hash>.parquet`
  and are regenerated from the committed `blocking_config`.
- Anything over 2 MB, including a notebook with heavy outputs.

What does go in: the notebook with outputs, `metrics.json`, the csv row and the `src/`
commit they ran on. That is enough to regenerate every artifact, and `metrics.json` alone
fills the results table of the final documentation (16 §4).
