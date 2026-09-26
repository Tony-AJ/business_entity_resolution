# Task tracker

Who is doing what, per the team split in [docs/plan/03_TEAM_WORKING_STRATEGY.md](docs/plan/03_TEAM_WORKING_STRATEGY.md).
Update your own rows when a task changes state and commit the change with your work
(`docs(tracker): ...`). Scores live in `experiments/experiments.csv`; uploads in
[LEADERBOARD.md](LEADERBOARD.md). The version table below mirrors them for a quick read.
Per-member detail: [M3 features](docs/trackers/M3_TRACKER.md).

Status: `todo` · `doing` · `done` · `blocked` · `dropped`. Owners: M1 lead / integration, M2
normalisation + blocking, M3 features, M4 models + hard negatives, M5 decision + errors.
**From day 2 M1 owns every task** (the M2–M5 rows of the day-1 plan are folded into the
day-2 and day-3 tables below). ETA = expected completion time, IST.

Last updated: 2026-09-26 13:00 IST (day 2); M3's feature groups (PR #8) merged into main.

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

**Tight mock (after upload #3).** v103 gained +0.006 public but +0.0027 on the mock: the test
punishes false merges harder. Splitting each mock loss into false-merge (L_FP) and missed-match
(L_FN) parts, public = 1 − L_FN − 1.45·L_FP − 0.0072 fits both uploads (`mock.FP_WEIGHT`,
`PUBLIC_OFFSET`). From v107 on, rules are tuned on this tight score and versions compared by
**est_public** (it reproduces 0.955 / 0.961 exactly). Reweighting by name commonness did not
explain the gap (test's same-name mix ≈ the mock's).

## Day 2 — Sat 26 Sep

| # | Task | Owner | Plan | ETA | Status | Notes |
|---|---|---|---|---|---|---|
| 24 | Re-plan: M1 takes over every day-2/3 task; this tracker | M1 | INT | 10:00 | done | |
| 25 | Record upload #2 (v101 public 0.955), day-1 budget 2 used | M1 | INT | 10:15 | done | `make public V=v101 SCORE=0.955` |
| 26 | Mock-test fold + global 1-to-1 scoring, tuning on mock tune S1, scoring on mock val S1; tests | M1 | INT / E2 | 11:30 | done | `mock.py`, `pipeline.run_mock` / `tune_mock` / `mock_scores`, `decision.one_to_one_filter`; mock = India 710k S1 / 4.13M pool (5.82), US 663k / 3.82M (5.76), 40.2 % of the pool unowned (test ~40 %) |
| 27 | Mock blocking cache: ~1.37M S1 × ~8.0M pool, per country | M1 | A | 12:30 | done | India 24.7M pairs (34.9 per S1) in 14 min, US 22.6M (34.1 per S1) in 12 min; cached |
| 28 | Calibrate: v101 on the mock test vs its public score | M1 | INT | 13:15 | done | v101: val 0.9858, **mock 0.9677**, public 0.955: the mock closes 60 % of the gap and is the KEEP/DROP number from here; offset mock − public ≈ +0.013 (India pool on test is 14 % denser than the mock's; France unseen) |
| 29 | v103: v101 matcher + rule tuned on the mock tune S1; test inference; **upload #3** | M1 | E2 | 11:00 | done | **Public 0.961** (+0.006). | Mock **0.9704** (+0.0027; singletons 0.952 → 0.978; rule τ 0.725 / rel 0.7 / single 0.775); plain val 0.9833. Finding: **candidate recall at mock density 0.9655** (val 0.9906): blocking loses 3.5 % of true pairs at test density → #47 moved up |
| 30 | v104: matcher trained on test-density candidates (fit sample vs the whole train-fold pool) | M1 | D3 / HN | 16:00 | todo | Was #20 (hard negatives): density brings the decoys |
| 31 | Mock error report: false merges vs misses by category (same-name decoy, empty address, unowned record, France-like cases) | M1 | E | 16:30 | todo | Was #22; picks the next fixes |
| 32 | France: French legal forms (SARL, SAS, SASU, EURL, SNC, SCI, SA) and street types (rue, av, bd, pl, imp, rte, chem, fbg, St/Ste) in the token maps; test-side sample check | M1 | B2 / B3 | 17:00 | todo | Was #15; 15 % of test S1, never seen in train |
| 33 | v104: two-stage matcher: v101 as stage 1 → competition + anchor features → stage 2 trained on the mock's fit entities, cross-fitted, XGBoost on the GPU; 2 ablations | M1 | C5 / D3 | 12:05 | done | **Mock 0.9744** (+0.0040), est_public 0.9654 (tight rule 0.9657). Ablations: competition features +0.0018, anchors +0.0002; `pool_gap` carries 72 % of the gain. Stage-2 fits ~5 min each on the GPU |
| 34 | Stage-1 candidate filter: keep the top 16 per S1 with p1 ≥ 0.01, so `candidate_pairs.tsv` shrinks from ~35 per S1 at ≤ 0.002 recall loss | M1 | A5 | 12:05 | done | **34.5 → 4.6 candidates per S1** for −0.0011 candidate recall (mock). Loss split of true pairs: 3.45 % not a blocking candidate, 0.11 % filter, 0.94 % lost in 1-to-1 conflicts, 1.96 % below the rule |
| 35 | v106: best of v104–v105 + France + filter; test inference; **upload #4** | M1 | INT | 21:00 | todo | |
| 36 | Stage-2 capacity on the GPU (leaves 127/255, lr 0.03), expected-F0.5 set decoding, per-source thresholds | M1 | D3 / E5 | 23:00 | todo | Was #18, #19, #21; XGBoost CUDA backend in `model.py` (4× LightGBM CPU); `decision.decide_expected` / `tune_expected` done + tests |
| 37 | Best mock version of the day; **upload #5** | M1 | INT | 23:30 | todo | |
| 44 | Review of the unattended-run code (mock, two-stage, stacking, decision, XGBoost backend) | M1 (agent) | INT | 11:00 | done | No logic bugs; memory peak of the stage-1 pass (~12–15 GB) and 6 smaller issues fixed before v104 ran |
| 45 | Methodology draft `docs/methodology.md` (organisers' template) | M1 (agent) | INT | 11:00 | done | Committed; final numbers filled on day 3 (#42) |
| 46 | Packaging: `scripts/package_submission.sh` + `make package` (doc 16 checklist) | M1 (agent) | INT | 11:00 | done | Flat zip (output/, code/, Documentation_template.md at the root); dry run passes the preflight |
| 47 | v105: blocking budgets at mock density (word top-k 25→50, cap 60→100, exact groups 50→200, address-word pass, sim-first cap) | M1 | A2–A5 | 12:15 | done | **B6 adopted: candidate recall 0.9677 → 0.9780** (misses −32 %) at 65.5 per S1. Exact-first cap was the culprit (bigger exact groups alone: 0.9551). 8 min via one superset per country |
| 48 | v106: stage 1 retrained on the GPU (XGBoost, 360k entities absent from the mock) + B6 blocking + rival features + rules v3; two-stage on top; **upload** | M1 | D4 / A5 | 16:30 | doing | Started 12:15; re-blocks mock and test with B6 (1.9× pairs) |
| 49 | v102 (user request): v101 + rule tuned on the dense tune pool, for an extra public data point | M1 | E2 | 18:00 | todo | Queued after v108, v109 |
| 53 | v108: stage 2 on fit + tune entities (out of fold), 127 leaves, from v106's stage-1 cache | M1 | D3 | 17:00 | todo | Spec-driven stage-2 variant notebook; queued right after v106 |
| 54 | v109: name + address-number exact key on top of B6 (blocking study, derived from one superset) | M1 | A1 | 17:10 | todo | Targets B6's remaining "name close" misses (common names, groups > 200); derivation checked equal to direct blocking |
| 55 | Methodology draft updated with day-2 results | M1 (agent) | INT | 13:30 | doing | |
| 50 | Tight mock: false merges ×1.45 + offset, calibrated on uploads #2–#3; `fp_weight` tuning | M1 | INT / E2 | 11:15 | done | est_public reproduces both public scores |
| 51 | v107: v104's two-stage + rule tuned on the tight mock (+ expected-F0.5 candidate); **upload #4** | M1 | E2 / E5 | 12:15 | done | est_public **0.9659** (mock 0.9745); expected-F0.5 decoding won; 5 min from caches. Upload pending |
| 52 | France: département names (Nord, Gironde, Loire-Atlantique, Pas-de-Calais: 27 % of French address components) mapped to region codes, rules v3 | M1 | B | 12:15 | done | Committed during v105; v106 is the first version with it |
| 16 | IDF-weighted name/address similarities for every pair | M3 | C2 | – | done | `idf` (8) + `ctx_idf` (5) groups, idf per country over the pool (`pool_stats`); v040 KEEP, val 0.9844 → 0.9870 with #17 and #17a |
| 17 | Context features: name frequency, pool-side competition | M3 | C5 | – | done | `frequency` group (4; renamed `token_freq` on merge to main, where `frequency` is v101's core-name rates): pool records of the country sharing the exact name / address; worth +0.0004 (v040 vs v041), all recall; in-degree stays opt-in (S1 sampling bias). Top loss in the v001 dry run: exact-name pool records with empty addresses score ~0.05 because the model cannot tell a rare name from a common one |
| 17a | Extra address evidence: reverse containment, numbers, postcode prefix | M3 | C3–C4 | – | done | `address_extra` group (6); 5.6 % of v040's gain, `num_contain_l` #7 |
| 17b | M3's groups in the two-stage matcher: `TwoStageConfig.extra_groups` (stage 2, on the kept pairs) + tests; stage-1 cache refuses another config | M3 | C2–C5 | – | done | `638e774`, `024b47c`; 266 tests pass |
| 17c | v042: v104 two-stage + M3's groups in stage 2 on the mock (3 arms, v107 rule tuning) | M3 | C2 | – | done | **est_public 0.9679** (+0.0020 over the same-machine v107 reproduction 0.9659), mock F0.5 0.9762, false merges −22 %; without token_freq 0.9675 |
| 17d | v043: stage 1 = v101 + M3's groups, then v104 two-stage + v107 rule | M3 | C5 | – | done | est_public 0.9678 (tie with v042), mock F0.5 0.9762, singletons 0.9902; stage 1 alone plain val **0.9876** (v101 0.9858); 4.37 candidates per S1 (4.59) |
| 17e | Test inference of v042 (both TSVs + our checker) for M1's upload decision | M3 | INT | 19:30 | doing | files to `submissions/v042/`; the organisers' validator is not on this machine |

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
| v041 | M3 | C2 | v040 without frequency (ablation) | 0.9866 | – | 0.9906 | – | kept |
| v040 | M3 | C2 | v001 + idf, frequency (now `token_freq`), ctx_idf, address_extra groups (70 features) | 0.9870 | – | 0.9906 | – | kept |
| v042 | M3 | C2 | v104 two-stage + M3's groups in stage 2 (kept pairs); v107 rule tuning | – | 0.9762 | – | – | kept (est_public 0.9679) |
| v043 | M3 | C5 | stage 1 = v101 + M3's groups; v104 two-stage; v107 rule tuning | 0.9876 (stage 1) | 0.9762 | 0.9906 | – | kept (est_public 0.9678) |
