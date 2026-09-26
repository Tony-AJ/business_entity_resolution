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

Last updated: 2026-09-26 16:30 IST (day 2): night-build strategy below; v110 est_public 0.9731.

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
| 30 | v104: matcher trained on test-density candidates (fit sample vs the whole train-fold pool) | M1 | D3 / HN | 16:00 | done | Covered by design: v104's stage 2 trains on the mock's fit entities at test density; v110's stage 1 on train-fold entities at train-fold density |
| 31 | Mock error report: false merges vs misses by category (same-name decoy, empty address, unowned record, France-like cases) | M1 | E | 16:30 | doing | v111 adds the loss breakdown (never a candidate / 1-to-1 conflict / below the rule, by country) for v110 and v111 |
| 32 | France: French legal forms (SARL, SAS, SASU, EURL, SNC, SCI, SA) and street types (rue, av, bd, pl, imp, rte, chem, fbg, St/Ste) in the token maps; test-side sample check | M1 | B2 / B3 | 17:00 | doing | French legal forms and street types were in the token maps from v001; départements added in rules v3 (#52); test-side check of French records running (agent). On test, France matches like India/US (v107: 94.5 % of S1 matched, 3.25 matches per S1) |
| 33 | v104: two-stage matcher: v101 as stage 1 → competition + anchor features → stage 2 trained on the mock's fit entities, cross-fitted, XGBoost on the GPU; 2 ablations | M1 | C5 / D3 | 12:05 | done | **Mock 0.9744** (+0.0040), est_public 0.9654 (tight rule 0.9657). Ablations: competition features +0.0018, anchors +0.0002; `pool_gap` carries 72 % of the gain. Stage-2 fits ~5 min each on the GPU |
| 34 | Stage-1 candidate filter: keep the top 16 per S1 with p1 ≥ 0.01, so `candidate_pairs.tsv` shrinks from ~35 per S1 at ≤ 0.002 recall loss | M1 | A5 | 12:05 | done | **34.5 → 4.6 candidates per S1** for −0.0011 candidate recall (mock). Loss split of true pairs: 3.45 % not a blocking candidate, 0.11 % filter, 0.94 % lost in 1-to-1 conflicts, 1.96 % below the rule |
| 35 | v106: best of v104–v105 + France + filter; test inference; **upload #4** | M1 | INT | 21:00 | dropped | Superseded: upload #4 was v107 (public 0.966); v110 (#57) combines v105, v109, the GPU stage 1 and M3's groups |
| 36 | Stage-2 capacity on the GPU (leaves 127/255, lr 0.03), expected-F0.5 set decoding, per-source thresholds | M1 | D3 / E5 | 23:00 | doing | Expected-F0.5 decoding done (v107); stage-2 capacity (127 leaves, lr 0.05), fit + tune training and 3 seeds in v111; per-source thresholds only if the breakdown points there |
| 37 | Best mock version of the day; **upload #5** | M1 | INT | 23:30 | doing | v110 if it beats v107's est_public by > 0.001, else v111 |
| 44 | Review of the unattended-run code (mock, two-stage, stacking, decision, XGBoost backend) | M1 (agent) | INT | 11:00 | done | No logic bugs; memory peak of the stage-1 pass (~12–15 GB) and 6 smaller issues fixed before v104 ran |
| 45 | Methodology draft `docs/methodology.md` (organisers' template) | M1 (agent) | INT | 11:00 | done | Committed; final numbers filled on day 3 (#42) |
| 46 | Packaging: `scripts/package_submission.sh` + `make package` (doc 16 checklist) | M1 (agent) | INT | 11:00 | done | Flat zip (output/, code/, Documentation_template.md at the root); dry run passes the preflight |
| 47 | v105: blocking budgets at mock density (word top-k 25→50, cap 60→100, exact groups 50→200, address-word pass, sim-first cap) | M1 | A2–A5 | 12:15 | done | **B6 adopted: candidate recall 0.9677 → 0.9780** (misses −32 %) at 65.5 per S1. Exact-first cap was the culprit (bigger exact groups alone: 0.9551). 8 min via one superset per country |
| 48 | v106: stage 1 retrained on the GPU (XGBoost, 360k entities absent from the mock) + B6 blocking + rival features + rules v3; two-stage on top; **upload** | M1 | D4 / A5 | 16:30 | dropped | GPU out of memory at 18M stage-1 rows × 53 features; row cap now sized by feature count (`stage1.GPU_CELLS`); the design runs as v110 (#56) |
| 49 | v102 (user request): v101 + rule tuned on the dense tune pool, for an extra public data point | M1 | E2 | 18:00 | dropped | Superseded by the mock fold (v103 tunes at test density); the queued run was stopped by a machine restart at 14:07 |
| 53 | v108: stage 2 on fit + tune entities (out of fold), 127 leaves, from v106's stage-1 cache | M1 | D3 | 17:00 | dropped | Needed v106's stage-1 cache; the idea moves to v110's stage 1 (day 3) |
| 54 | v109: name + address-number exact key on top of B6 (blocking study, derived from one superset) | M1 | A1 | 17:10 | done | **Candidate recall 0.9780 → 0.9829** at 69.3 per S1 (cap 120), +1 % run time. Targets B6's remaining "name close" misses (common names, groups > 200); derivation checked equal to direct blocking |
| 55 | Methodology draft updated with day-2 results | M1 (agent) | INT | 13:30 | done | Committed 2a777ad: draft updated to v107 (public 0.966); final numbers on the final version |
| 56 | Merge PR #8 (M3: `idf`, `token_freq`, `ctx_idf`, `address_extra` feature groups) into main; conflicts resolved (`frequency` kept for v101's core-name rates, M3's renamed `token_freq`) | M1 | INT | 13:00 | done | Merge `1a7df62`, lint + 263 tests green |
| 57 | v110: v107's two-stage + M3's four groups (76 features) + B6 and name+number blocking (v109) + GPU stage 1 (110k entities, ~9M pairs) + rival features; tight rule; test inference; **upload #5** if est_public beats v107 (0.9659) by > 0.001 | M1 | C2 / A5 / D4 | 17:00 | doing | Started 14:25; stage-2 ablation without M3's columns measures their share |
| 58 | France test-side audit (read-only, agent): French S1 vs pool forms under rules v3 | M1 (agent) | B2 / B3 | 15:30 | done | Gaps: "N°" number marker (6.6 % of French pool addresses, 0 in S1), `et`/`+` for `&`, `bis`→`B`, leet legal forms (`5arl`), `compagnie`; most street abbreviations already mapped |
| 59 | Rules v4 (France-only: N° marker, compagnie) and v5 (`+`/`et`→and, leet legal forms, frs, bis/crs/psg/appt, own-country word out of name_core) on `feat/rules-v4-france` | M1 (agent) | B2 / B3 | 16:00 | done | v4 `25e99e8`: all 10.0M US/India test records byte-identical to v3. v5: US +885, India +1,053 / −6 identical true-pair names; needs a full re-run (#62) |
| 60 | v111: v110 stage 2 on fit + tune entities, 127 leaves, 3 seeds (`SeedMean`, `feat/seed-mean`); loss breakdown (never a candidate / 1-to-1 / below the rule) | M1 | D3 / E | 19:00 | todo | Stage 2 only, from v110's caches: ~25 min, GPU |
| 61 | v112: best of v110/v111 + rules v4, France re-run only (India/US read from the parent's test cache) | M1 | B2 | 19:30 | dropped | Folded into the night build (#70): rules v5 include v4's French rules |
| 62 | v113: full re-run with rules v5 (mock + test); judged by est_public | M1 | B2–B4 | Day 3 08:00 | dropped | Folded into the night build (#70) |
| 50 | Tight mock: false merges ×1.45 + offset, calibrated on uploads #2–#3; `fp_weight` tuning | M1 | INT / E2 | 11:15 | done | est_public reproduces both public scores |
| 51 | v107: v104's two-stage + rule tuned on the tight mock (+ expected-F0.5 candidate); **upload #4** | M1 | E2 / E5 | 12:15 | done | est_public **0.9659** (mock 0.9745); expected-F0.5 decoding won; 5 min from caches. **Public 0.966** (12:29): the tight mock was off by 0.0001 |
| 52 | France: département names (Nord, Gironde, Loire-Atlantique, Pas-de-Calais: 27 % of French address components) mapped to region codes, rules v3 | M1 | B | 12:15 | done | Committed during v105; v106 is the first version with it |
| 16 | IDF-weighted name/address similarities for every pair | M3 | C2 | – | done | `idf` (8) + `ctx_idf` (5) groups, idf per country over the pool (`pool_stats`); v040 KEEP, val 0.9844 → 0.9870 with #17 and #17a |
| 17 | Context features: name frequency, pool-side competition | M3 | C5 | – | done | `frequency` group (4; renamed `token_freq` on merge to main, where `frequency` is v101's core-name rates): pool records of the country sharing the exact name / address; worth +0.0004 (v040 vs v041), all recall; in-degree stays opt-in (S1 sampling bias). Top loss in the v001 dry run: exact-name pool records with empty addresses score ~0.05 because the model cannot tell a rare name from a common one |
| 17a | Extra address evidence: reverse containment, numbers, postcode prefix | M3 | C3–C4 | – | done | `address_extra` group (6); 5.6 % of v040's gain, `num_contain_l` #7 |

## Night build — from 16:30 IST day 2 (target: public ≥ 0.99)

The leader has crossed 0.99; our best public is 0.966 (v107) and v110 (est_public 0.9731,
upload #5, untouched) closes part of the gap. Waiting for one full test run per idea (~1.5–3.5 h
each) no longer fits, so the strategy changes:

1. **Implement every planned item now**, each behind an opt-in switch (feature group, config
   field or flag) that defaults to today's behaviour, with tests and `make lint test` green.
   No per-item test inference.
2. **Integrate at 21:30.** Every member pushes their branch (opt-in, rebased or merged on
   current main). M1 merges them with merge commits into `integration/night-build`, runs
   `make lint test`, and fixes conflicts.
3. **Night build at ~22:00:** one notebook (`v120_night_build`) with every switch on: stage-1
   set, GPU stage 1, mock pass, stage 2, tight-mock rule, test inference, both validators
   (~4 h, files ~02:00). The mock's est_public checks it before the upload.
4. **Day 3:** upload the night build first; the public score verifies it. Stage-2 and rule
   ablations then run from its caches in minutes (uploads #7–#10), and the final package is
   built from the best upload.

| # | Task | Owner | Plan | ETA | Status | Notes |
|---|---|---|---|---|---|---|
| 63 | Night-build plan, `integration/night-build` branch, merge gate | M1 | INT | 21:30 | doing | Members: push opt-in branches by 21:30 |
| 64 | Rules v5 (France + `+`/`et`, leet legal forms, own-country word; #59) merged | M1 | B2–B4 | 18:30 | todo | After v111's run, so v111's commit stays reproducible |
| 65 | Stage 2 on fit + tune entities, 127 leaves, 3 seeds (`SeedMean`) | M1 | D3 | 18:15 | doing | v111 decides it (starts itself when v110 ends) |
| 66 | Stage 1 on every training row (CPU hist or GPU bagging) instead of the 7M-row GPU cap | M1 | D3 | 19:30 | todo | v110 used 49 % of its 14.9M rows |
| 67 | Pool-sibling candidates: records near-identical to an entity's best candidate join its candidate set | M1 | A5 | 20:30 | todo | Targets the 2.1 % of true pairs lost before stage 2 |
| 68 | Alias names (`dba`, `aka`, `fka`, `t/a`, S3 only) split into name + alias | M1 | B / C | 21:00 | done | `feat/alias-names` (opt-in, off): aliases are 2–4 % of S3 names, S1 always matches the part after the marker, and v110 already finds 99.92 % of alias pairs (41 misses of 48k): upper bound +0.00006, so the night build leaves it off |
| 69 | Error-driven fixes from v111's loss breakdown (#31) | M1 | A–E | 21:00 | todo | |
| 70 | v120 night build: every switch on, test inference, package | M1 | INT | Day 3 02:00 | todo | Replaces v112 (France-only) and v113 (rules v5) |
| 71 | Day-3 uploads: night build first, then ablations from its caches | M1 | INT | Day 3 | todo | 5 uploads |
| 72 | M3: stage-2 extra feature groups (`feat/m3-stage2-features`) | M3 | C5 | 21:30 | doing | Opt-in; merged at the gate |
| 73 | M2: phonetic Soundex + Metaphone features, entity blocking (`newblocking`, `newfeatureblocking`) | M2 | A / C | 21:30 | doing | Branches start from v001-era main: rebase or merge current main, keep opt-in |

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
