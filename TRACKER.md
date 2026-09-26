# Task tracker

Who is doing what, per the team split in [docs/plan/03_TEAM_WORKING_STRATEGY.md](docs/plan/03_TEAM_WORKING_STRATEGY.md).
Update your own rows when a task changes state and commit the change with your work
(`docs(tracker): ...`). Scores live in `experiments/experiments.csv`; uploads in
[LEADERBOARD.md](LEADERBOARD.md). The version table below mirrors them for a quick read.

Status: `todo` · `doing` · `done` · `blocked` · `dropped`. Owners: M1 lead / integration, M2
normalisation + blocking, M3 features, M4 models + hard negatives, M5 decision + errors.
**From day 2 M1 owns every task** (the M2–M5 rows of the day-1 plan are folded into the
day-2 and day-3 tables below). ETA = expected completion time, IST.

Last updated: 2026-09-26 10:20 IST (day 2).

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
| 12 | Upload #1 + `LEADERBOARD.md` entry | M1 | INT | done | v001 public F0.5 **0.954** (local 0.9844, offset −0.030) |
| 13 | Dense-val check: val S1 against the full train pool | M1 | INT | dropped | Superseded by the mock-test protocol (#26) |

| 14a | v101: core-name frequency features + LightGBM cap 4000 | M1 | C5 | done | Val 0.9858 (+0.0014), harder 0.9852, false merges −29%; upload #2: public **0.955** (+0.001 only) |
| 14b | v102: v101's model, rule re-tuned against a test-sized pool (`retune`, `tune_pool="train"`) | M1 | E2 | dropped | Never finished; no competing S1 in that pool, so too pessimistic. Superseded by #26–#29 |

## Why public is 0.955 while local is 0.986

Local val scores 441k S1 against a 2.06M pool (a 20 % sample), test scores 1.73M S1 against
a 10.0M pool: each test entity meets ~3× (US) to ~6× (India) more same-name decoys, and ~40 %
of the test pool has no S1 owner (26 % in train). Two uploads gave the same −0.030 offset, so
val cannot rank changes that matter at test density. Day 2 starts by making local testing
test-shaped, then spends every model change against that number.

**Mock-test protocol (#26).** Per country, a subsample of the train data with the test's pool
size and pool-per-S1 ratio (US 3.82M pool, 5.76 per S1; India all 4.13M, 5.82 per S1): whole
clusters (S1 + its pool records) kept by id hash, then S1 dropped by hash (the matcher's
training sample first) so their records stay as unowned decoys. Every kept S1 is scored and
competes in the pool-side 1-to-1, as on test. The rule is tuned on the mock's tune-side S1 and
scored on its val-side S1: **mock F0.5** is the decision number from here on; plain val stays
as a secondary check. Target: mock F0.5 of v001 / v101 within ±0.005 of 0.954 / 0.955.

## Day 2 — Sat 26 Sep

| # | Task | Owner | Plan | ETA | Status | Notes |
|---|---|---|---|---|---|---|
| 24 | Re-plan: M1 takes over every day-2/3 task; this tracker | M1 | INT | 10:00 | done | |
| 25 | Record upload #2 (v101 public 0.955), day-1 budget 2 used | M1 | INT | 10:15 | done | `make public V=v101 SCORE=0.955` |
| 26 | Mock-test fold + global 1-to-1 scoring, tuning on mock tune S1, scoring on mock val S1; tests | M1 | INT / E2 | 11:30 | done | `mock.py`, `pipeline.run_mock` / `tune_mock` / `mock_scores`, `decision.one_to_one_filter`; mock = India 710k S1 / 4.13M pool (5.82), US 663k / 3.82M (5.76), 40.2 % of the pool unowned (test ~40 %) |
| 27 | Mock blocking cache: ~1.37M S1 × ~8.0M pool, per country | M1 | A | 12:30 | done | India 24.7M pairs (34.9 per S1) in 14 min, US 22.6M (34.1 per S1) in 12 min; cached |
| 28 | Calibrate: v001 and v101 on the mock test vs their public scores | M1 | INT | 13:15 | todo | Decides whether mock F0.5 replaces val as the KEEP/DROP number |
| 29 | v103: v101 matcher + rule tuned on the mock tune S1; test inference; **upload #3** | M1 | E2 | 12:00 | doing | Notebook queued behind #27; upload only if mock F0.5 ≥ v101 + 0.003 |
| 30 | v104: matcher trained on test-density candidates (fit sample vs the whole train-fold pool) | M1 | D3 / HN | 16:00 | todo | Was #20 (hard negatives): density brings the decoys |
| 31 | Mock error report: false merges vs misses by category (same-name decoy, empty address, unowned record, France-like cases) | M1 | E | 16:30 | todo | Was #22; picks the next fixes |
| 32 | France: French legal forms (SARL, SAS, SASU, EURL, SNC, SCI, SA) and street types (rue, av, bd, pl, imp, rte, chem, fbg, St/Ste) in the token maps; test-side sample check | M1 | B2 / B3 | 17:00 | todo | Was #15; 15 % of test S1, never seen in train |
| 33 | v104: two-stage matcher: v101 as stage 1 → competition features (rank, best rival, likely counts; S1 side and pool side) → stage 2 trained on the mock's fit entities, cross-fitted, XGBoost on the GPU | M1 | C5 / D3 | 13:00 | doing | Was #17; `stacking.py`, `twostage.py` done + tests; notebook queued after v103 |
| 34 | Stage-1 candidate filter: keep the top 16 per S1 with p1 ≥ 0.01, so `candidate_pairs.tsv` shrinks from ~35 per S1 at ≤ 0.002 recall loss | M1 | A5 | 13:00 | doing | Was #13b; part of v104 (`twostage.keep_mask`) |
| 35 | v106: best of v104–v105 + France + filter; test inference; **upload #4** | M1 | INT | 21:00 | todo | |
| 36 | Stage-2 capacity on the GPU (leaves 127/255, lr 0.03), expected-F0.5 set decoding, per-source thresholds | M1 | D3 / E5 | 23:00 | todo | Was #18, #19, #21; XGBoost CUDA backend in `model.py` (4× LightGBM CPU); `decision.decide_expected` / `tune_expected` done + tests |
| 37 | Best mock version of the day; **upload #5** | M1 | INT | 23:30 | todo | |
| 44 | Review of the unattended-run code (mock, two-stage, stacking, decision, XGBoost backend) | M1 (agent) | INT | 11:00 | doing | Read-only review; findings fixed before v104 runs |
| 45 | Methodology draft `docs/methodology.md` (organisers' template) | M1 (agent) | INT | 11:00 | doing | Final numbers filled on day 3 (#42) |
| 46 | Packaging: `scripts/package_submission.sh` + `make package` (doc 16 checklist) | M1 (agent) | INT | 11:00 | doing | Dry run only until the final version exists |
| 47 | v105: blocking budgets at mock density (word top-k 25→50, cap 60→100, exact groups 50→200, P4) | M1 | A2–A5 | 12:30 | todo | Notebook ready; runs after v104 |

## Day 3 — Sun 27 Sep

| # | Task | Owner | Plan | ETA | Status | Notes |
|---|---|---|---|---|---|---|
| 38 | Error-driven fixes from the mock report (blocking misses at density, empty-address pairs) | M1 | A / C | 11:00 | todo | Was #14, #16 |
| 39 | Final matcher: more training S1, 3 seeds, rule re-tuned on the mock | M1 | D3 / E2 | 14:00 | todo | XGBoost (#19) only if LightGBM plateaus |
| 40 | **Uploads #6–#8**: best mock versions, one change each | M1 | INT | 16:00 | todo | |
| 41 | Freeze 18:00; final `run_test`; `make validate` | M1 | INT | 18:30 | todo | Checklist in doc 16 |
| 42 | Package: README reproduction, `requirements.txt`, `Documentation_template.md` (1–2 pages), zip | M1 | INT | 21:00 | todo | |
| 43 | **Uploads #9–#10** (last by 22:00 IST); git tag of the submitted commit | M1 | INT | 22:00 | todo | |

## Versions

| Version | Owner | Plan | Change | Local F0.5 | Mock F0.5 | Cand. recall | Public F0.5 | Status |
|---|---|---|---|---|---|---|---|---|
| v101 | M1 | C5 | v001 + core-name frequency features; LightGBM cap 4000 | 0.9858 | – | 0.9906 | 0.955 | submitted |
| v001 | M1 | INT | Base model: normalise + learned map, multi-pass blocking, 47 features, LightGBM, tuned 1-to-1 rule | 0.9844 | – | 0.9906 | 0.954 | submitted |
