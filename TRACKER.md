# Task tracker

Who is doing what, per the team split in [docs/plan/03_TEAM_WORKING_STRATEGY.md](docs/plan/03_TEAM_WORKING_STRATEGY.md).
Update your own rows when a task changes state and commit the change with your work
(`docs(tracker): ...`). Scores live in `experiments/experiments.csv`; uploads in
[LEADERBOARD.md](LEADERBOARD.md). The version table below mirrors them for a quick read.
Per-member detail: [M3 features](docs/trackers/M3_TRACKER.md).

Status: `todo` · `doing` · `done` · `blocked`. Owners: M1 lead / integration, M2
normalisation + blocking, M3 features, M4 models + hard negatives, M5 decision + errors.

Last updated: 2026-09-26 11:55 IST (day 2).

## Day 1 — Fri 25 Sep

| # | Task | Owner | Plan | Status | Notes |
|---|---|---|---|---|---|
| 1 | Research plan `docs/plan/00–18` | M1 | INT | done | Merged to main (PR #3) |
| 2 | `normalize.py`: static rules + learned Indic token map | M1 | B1–B4 | done | Legal forms, leet/domain forms, honorifics, street tokens, state codes incl. native-script names; 539-token map learned from train-fold pairs |
| 3 | `blocking.py`: exact keys + name/address word TF-IDF + short-address name char pass | M1 | A1–A5 | done | 10k-S1 val samples: pair recall 0.987 India / 0.994 US at ~30 candidates per S1 |
| 4 | `evaluate.py`, `trainset.py` (inner split), `decision.py` + tests | M1 | E1–E2 | done | Tune scores the 3,906-rule grid in ~4 s on 3M pairs |
| 5 | `model.py` (LightGBM `Matcher`), `submission.write_pairs`, `tracking` `V=` | M1 | D3 | done | 69 tests pass |
| 6 | `features.py`: 47 pair features, chunked | M1 | C1–C5 | done | ~250k pairs/s on 12 threads |
| 7 | Tests for `normalize.py` and `blocking.py` | M1 | INT | done | Found 4 bugs, fixed (rules version 2) |
| 8 | `pipeline.py`: `fit` / `run_fold` / `run_test` | M1 | INT | done | End-to-end smoke test validates both files on the fixture |
| 9 | Normalisation cache for train + test | M1 | INT | done | ~5 min per split; rules version 2 |
| 10 | v001 base model notebook: fit + val score | M1 | INT | done | Val macro F0.5 0.9844 (harder 0.9838, singletons 0.9840), cand recall 0.9906; loss is mostly recall (53.9k missed pairs vs 6.7k false merges); LightGBM hit its 2,000-round cap |
| 11 | v001 test inference + `make validate` | M1 | INT | done | 1,732,544 rows in both files, both validators PASS; France match rate 0.950 vs India 0.941 / US 0.943 |
| 12 | Upload #1 + `LEADERBOARD.md` entry | M1 | INT | doing | Entry written; upload `output/matching_results.tsv`, then `make public V=v001 SCORE=...` |
| 13 | Dense-val check: val S1 against the full train pool | M1 | INT | todo | Test pool is ~5× denser than val; estimates the public drop |

| 14a | v101: core-name frequency features + LightGBM cap 4000 | M1 | C5 | doing | Code on `feat/name-frequency`; runs after v001 |
| 14b | v102: rule tuned against a test-sized pool (`tune_pool="train"`) | M1 | E2 | todo | Option committed; val cannot show the gain, dense val can |

## Day 2 — Sat 26 Sep

| # | Task | Owner | Plan | Status | Notes |
|---|---|---|---|---|---|
| 14 | Blocking sweeps: `max_df`, `top_k`, P4 address char pass | M2 | A2–A5 | todo | v010–v039; keep ceiling F0.5 up at ≤ 40 candidates per S1 |
| 15 | Learned address token map (cities, script tokens) | M2 | B4 | todo | |
| 16 | IDF-weighted name/address similarities for every pair | M3 | C2 | done | `idf` (8) + `ctx_idf` (5) groups, idf per country over the pool (`pool_stats`); v040 KEEP, val 0.9844 → 0.9870 with #17 and #17a |
| 17 | Context features: name frequency, pool-side competition | M3 | C5 | done | `frequency` group (4): pool records of the country sharing the exact name / address; worth +0.0004 (v040 vs v041), all recall; in-degree stays opt-in (S1 sampling bias). Top loss in the v001 dry run: exact-name pool records with empty addresses score ~0.05 because the model cannot tell a rare name from a common one |
| 17a | Extra address evidence: reverse containment, numbers, postcode prefix | M3 | C3–C4 | done | `address_extra` group (6); 5.6 % of v040's gain, `num_contain_l` #7 |
| 18 | LightGBM coordinate search + 3-seed average | M4 | D3 | todo | v060–v079 |
| 19 | XGBoost on the RTX 2050 (`device=cuda`) vs LightGBM | M4 | D4 | todo | |
| 20 | Round-2 hard negatives from fit-side false positives | M4 | HN | todo | |
| 21 | Rule grid, singleton floor, expected-F0.5 decoding | M5 | E3–E5 | todo | v080–v099 |
| 22 | `errors.py`: 18-category error report | M5 | E | todo | Doc 18 |

## Day 3 — Sun 27 Sep

| # | Task | Owner | Plan | Status | Notes |
|---|---|---|---|---|---|
| 23 | Freeze 18:00, final `run_test`, package + methodology document | M1 | INT | todo | Checklist in doc 16; last upload by 22:00 IST |

## Versions

| Version | Owner | Plan | Change | Local F0.5 | Cand. recall | Public F0.5 | Status |
|---|---|---|---|---|---|---|---|
| v001 | M1 | INT | Base model: normalise + learned map, multi-pass blocking, 47 features, LightGBM, tuned 1-to-1 rule | 0.9844 | 0.9906 | – | logged |
| v040 | M3 | C2 | v001 + idf, frequency, ctx_idf, address_extra groups (70 features) | 0.9870 | 0.9906 | – | kept |
| v041 | M3 | C2 | v040 without frequency (ablation) | 0.9866 | 0.9906 | – | kept |
