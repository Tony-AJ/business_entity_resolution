# 00 — Master plan: Business Entity Resolution (Amazon ML Challenge 2026)

Written 25 Sep 2026 (day 1 of 3). This is the one-page-per-topic blueprint; the numbered
documents in `docs/plan/` carry the detail. A teammate reads this, `01`, `02`, `03` and
their own module document, then implements.

## 1. Executive summary

For each of 1.73M test Source 1 businesses we must list every Source 2/3 record of the same
business, scored by macro F0.5 per entity with singletons worth a full point. The data is
synthetic noise over a small name vocabulary: 48% of reference names collide with another
reference name, 15% of true pairs share no name token, and every true pair shares the
country. So **blocking must combine name and address channels inside a country partition,
the matcher must lean on address agreement, and the decision layer must prefer an empty
answer to a doubtful one**. The stack: vectorised normalisation with hand-typed and
pair-learned token maps → multi-pass blocking (exact keys + char-gram and word TF-IDF top-k
via `sparse_dot_topn`) → 48 pair features → LightGBM → a tuned set-decision rule with
pool-side 1-to-1 → validator. Five people each own one module behind fixed DataFrame
contracts; the lead ships a runnable skeleton today at 17:00 IST; 15 leaderboard uploads are
spent only on versions that beat the previous upload on the fixed validation fold.

## 2. Problem understanding (→ 01)

Facts: three sources, S1 deduplicated reference, 0..many matches, macro F0.5 with singleton
rule 1.0/0.0, two TSV outputs with strict validation, no external lookups, MIT/Apache model
≤ 8B parameters, package re-run and audited. Measured: singletons 5.6%; 3.7 matches per
matched entity; pool records belong to at most one S1; exact core-name equality 48% on true
pairs; zero-overlap pairs 14.7% (47% Indic script, 35% domain/handle forms, 18% renames)
rescued by addresses ~90% of the time; 62% of singletons have a name twin in the pool;
test has 5.8 pool records per S1 versus 4.7 in train (more decoys); France is 15% of test
and absent from train.

## 3. Research findings (→ 04, 17)

- Blocking literature (Papadakis surveys, similarity joins, sparse top-k) says multi-pass
  union of cheap keys and TF-IDF top-k gives the best recall/cost at this scale on CPU;
  embedding ANN helps semantic renames but not synthetic typos, and costs GPU time.
- Feature-based gradient boosting (LightGBM) remains the strongest practical matcher for
  structured, noisy records with abundant labels (7.6M pairs) and explains itself; deep
  matchers (DeepMatcher, Ditto) win on textual product data, not on short names+addresses,
  and are too slow for 24M inference pairs on our hardware.
- F-measure optimisation: thresholds must be tuned for the metric, not for accuracy;
  calibrated probabilities enable expected-F decoding; precision-heavy F needs conservative
  set selection.
- Hard negatives: blocking candidates are already hard; the model's own false positives and
  singleton decoys are the valuable extras.

## 4. Algorithm stack (→ 02, 04–10)

| Tier | Content |
|---|---|
| Baseline v001 (MUST, day 1) | normalise → exact core-name block (groups ≤ 50) → address token-set similarity → threshold; first val score, first upload |
| V1 (MUST, day 1–2) | passes P1 exact core/sorted/squash + P2 name char 3-gram top-30 + P3 name+address word top-20, cap 60; 48 features; LightGBM (63 leaves, lr 0.05, early stop on the tune split); rule τ_abs/τ_rel/τ_single/K tuned by grid on the tune split; pool-side 1-to-1 |
| V2 (SHOULD, day 2–3) | token maps learned from train pairs (script → Latin, state forms), round-2 hard negatives, per-source thresholds, expected-F0.5 decoding |
| EXPERIMENTAL | P4 address char-grams; multilingual-e5-small embeddings as an extra pass; XGBoost/CatBoost |
| OPTIONAL | MiniLM cross-encoder on the uncertain band only |
| REJECTED | full-pool cross-encoders, deep blocking, libpostal/usaddress (external trained data), GPL libraries |

## 5. Architecture (→ 02)

Ten stages behind four frames: normalised records → candidate pairs (`source1_entity_id`,
`entity_id`, `pass`, three sims) → float32 features → scored pairs (`prob`) → matches.
Modules `normalize`, `blocking`, `features`, `trainset`, `model`, `decision`, `evaluate`,
`errors`, `pipeline` plus the existing `data`, `split`, `metrics`, `submission`, `tracking`.
Everything chunked (2M pairs), per country, cached by config hash; test blocking ≈ 45 min,
features + predict ≈ 15 min, RAM < 6 GB.

## 6. Team (→ 03)

M1 lead/integration (pipeline, evaluate, uploads, package; v001–v009, v100–v129), M2
normalisation + blocking (v010–v039), M3 features + training pairs (v040–v059), M4 models +
hard negatives (v060–v079), M5 decision + error analysis (v080–v099). Walking skeleton by
17:00 IST day 1; every member replaces one stage; only integrated v1xx versions are uploaded.

## 7. Notebooks (→ project rules, 13)

One experiment = one `experiments/vNNN_<slug>/` notebook from the template (hypothesis,
setup, data, method, evaluation, error analysis, log, conclusion, test inference). The
requested 00–07 notebook series maps to plan groups: EDA + baseline → v001; blocking → A;
features → C; models → D; error analysis → §6 of every notebook; thresholds → E; final
pipeline → the shortlisted version's §9 driven by `pipeline.run_test`. Library code lives in
`src/entity_resolution/`; notebooks import it.

## 8. Experiments (→ 13)

Every version logs `experiments.csv` (version, date, group, change, local F0.5, candidate
recall, public F0.5, commit, notes, owner, parent, decision) and `metrics.json` (configs,
rule, breakdown, blocking report, harder-val, tune score, error counts, timings, RSS).
KEEP if val F0.5 > parent + 0.002 with no slice regressing > 0.01. Expect 10–20 local
versions per upload.

## 9. Validation (→ 11)

Fixed val fold (20%, seed 42) scored once per version with a frozen rule; train fold split
by hash into fit (75%) and tune (25%, seed 4242) for training, early stopping, calibration
and threshold tuning; a harder val variant (drop 20% of S1, keep their pool rows) mimics the
test decoy ratio. Metric = `metrics.breakdown`; blocking = `metrics.candidate_report`;
slices by country, source, script, ambiguity, singleton, match count.

## 10. Testing (→ 12)

`make lint test` before every commit (ruff + pytest on the synthetic fixture, < 30 s). Named
unit tests per module (05 §9, 06 §8, 07 §7, 08–10), integration tests across stages, and
an end-to-end smoke test that fits, scores, decides, writes both TSVs and passes
`submission.validate` on the synthetic dataset.

## 11. Git and integration (→ 14)

Branch per task from `main`, PR with merge commit, hooks enforce Conventional Commits,
≤ 400 lines per commit (notebooks excluded), no data files, identity guard. Merge gate:
tests green, notebook re-executed, CSV row with a clean commit, val report not regressing on
any slice, compared with the frozen parent version; M1 reruns the integrated pipeline after
every merge.

## 12. Three-day plan (→ 03 §4, 15)

| Day | Goal | Uploads |
|---|---|---|
| Fri 25 (today) | skeleton 17:00; v001 scored 19:00; modules v1 landed 21:00; integrated v100 | v001 baseline; best V1 if ≥ +0.05; precision-max probe |
| Sat 26 | blocking ≥ 0.97 recall; features C1–C5; LightGBM tuned; rule E1–E4; hard negatives HN2–HN4; learned token maps | up to 5, each ≥ +0.003 val over last upload |
| Sun 27 | E5 / P4 / D4 only if promising; freeze 18:00; final `run_test`, validate, package | ≤ 3 candidates + safety re-upload; last ≤ 22:00 IST |

## 13. Final package (→ 16)

`<team>_submission.zip`: `output/` (both TSVs), `code/business_entity_resolution/` (this
repo with the winning notebook copied under `src/`, README reproduction steps,
pinned `requirements.txt`), filled `Documentation_template.md`. Validator PASS with id
checks, matches ⊆ candidates, one row per test S1, France rows present, git tag.

## 14. Risk register

| Risk | Mitigation | Owner |
|---|---|---|
| Blocking recall < 0.97 on script names / renames | P3 address pass, learned token map, P4; slice recall tracked per version | M2 |
| P2 too slow on India (4.7M pool) | calibrate on 20k chunk; max_df 0.05 / 4-grams; cache test candidates once | M2, M1 |
| RAM > 8 GB free | chunked transform, float32/int32, per-country, `mem_guard`, never a full test feature matrix | all |
| Same-name decoys (48% ambiguous names) → false merges | address features, group context, 1-to-1, conservative τ, decoy negatives | M3–M5 |
| Test has more decoys than val (5.8 vs 4.7) | harder-val score, prefer higher τ_abs on ties, France/US/India slice checks on test output | M1, M5 |
| France unseen | no country enumeration; French tokens in maps; test-slice sanity before upload | M2, M1 |
| Threshold leakage into val | tune only on the inner tune split; val scored once with a frozen rule | M5 |
| Version-number or CSV collisions | reserved ranges, explicit `V=`, `merge=union` | M1 |
| Wasted uploads | eligibility checklist, validator with id checks, one hypothesis per upload | M1 |
| Licence / fair-play breach | permissive libraries only; dictionaries hand-typed or learned from train pairs; grep for network calls before packaging | M1 |
| Synthetic label noise | inspect 20 samples per error category before "fixing" it | M5 |
| Time | walking skeleton first; MUST before SHOULD; freeze at 18:00 day 3 | M1 |

## 15. References (→ 17)

45 verified references: Fellegi–Sunter, Christen, Papadakis blocking surveys, similarity
joins, BM25, MinHash/LSH, HNSW/FAISS, string-metric comparisons, Magellan, DeepMatcher,
Ditto, DeepBlocker, LightGBM/XGBoost/CatBoost, calibration, F-measure-optimal thresholding,
DPR and hard-negative contrastive learning, Sentence-BERT, Splink, plus library and model
licence checks (rapidfuzz MIT, sparse_dot_topn Apache-2.0, LightGBM MIT, anyascii ISC;
unidecode and python-Levenshtein GPL → banned).

## 16. Open contract items for day 1 (M1 resolves in the first PRs)

- `PipelineConfig.hard_negatives: HardNegConfig | None = None` (09 §11) and
  `PipelineConfig.dataset_dir` so the pipeline never reads `config.DATASET` directly (12 §3).
- `normalize` adds a `domain_form` bool column for the slice report (11 §7).
- `Fitted.tune_table` scores its rules in a column named `f_beta` (10 §4, 11 §9, 13 §2).
- `DecisionRule` gains `conflict_delta` / `tau_s3_offset` only if the E2/E5 variants win (10 §6, §8).
- On the synthetic test fixture the fit side of the inner split is empty, so integration
  tests use the `heuristic` backend and every stage must accept empty input (11 §12, 12 §3).

## 17. Document index

| File | What it contains | Who reads it |
|---|---|---|
| `01_PROBLEM_ANALYSIS.md` | Every requirement with its source, tagged fact/inference/recommendation; measured data facts; metric arithmetic | all |
| `02_SYSTEM_ARCHITECTURE.md` | Stage table, module map, exact signatures and DataFrame schemas, orchestration, budget, caches | all |
| `03_TEAM_WORKING_STRATEGY.md` | Roles, ownership, per-member brief, day-1 timetable, unblocking and integration rules | all |
| `04_ALGORITHM_RESEARCH.md` | Technique cards for normalisation, blocking, features, models, hard negatives, decision; chosen vs rejected | M1–M5 (own area) |
| `05_NORMALISATION_STRATEGY.md` | Ordered rules, token maps, squash/leet, transliteration, learned token alignment, tests, B experiments | M2 |
| `06_BLOCKING_STRATEGY.md` | Passes and parameters, chunked top-k algorithm, budget, recall evaluation, A experiments, tests | M2 |
| `07_FEATURE_ENGINEERING.md` | 48-feature registry, chunking, training-pair construction, C experiments, tests | M3 |
| `08_MODEL_SELECTION.md` | Model comparison, LightGBM parameters, training protocol, calibration, D experiments, tests | M4 |
| `09_HARD_NEGATIVE_MINING.md` | Negative sources, ratios, leakage rules, round-2 loop, HN experiments, tests | M4 |
| `10_DECISION_LAYER.md` | Rule family, 1-to-1, vectorised grid tuning, expected-F0.5 decoding, E experiments, tests | M5 |
| `11_VALIDATION_AND_METRICS.md` | Folds and inner split, leakage rules, metric reference, harder val, slices, per-version protocol | M1, M5 |
| `12_TESTING_STRATEGY.md` | Gate, test files per module, integration and smoke tests, fixtures | all |
| `13_EXPERIMENT_TRACKING.md` | Version folders, CSV and metrics.json fields, KEEP/DROP rule, commit protocol | all |
| `14_GIT_AND_INTEGRATION_WORKFLOW.md` | Branches, hooks, PR checklist, merge gate, conflict handling | all |
| `15_LEADERBOARD_STRATEGY.md` | Upload budget, eligibility, day-by-day slots, reading public scores | M1 |
| `16_FINAL_SUBMISSION_CHECKLIST.md` | Package tree, reproduction, compliance, documentation mapping, day-3 checklist | M1 |
| `17_RESEARCH_REFERENCES.md` | Verified references and library/model licence table | all |
| `18_ERROR_ANALYSIS_FRAMEWORK.md` | Error categories with detection rules, owners and fixes; per-version error report | M5, all |
