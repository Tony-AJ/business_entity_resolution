# Leaderboard submissions

Auditable log of every upload to the challenge portal, newest first. Budget: **5 per day,
15 in total** (25–27 Sep 2026, IST). Uploads are only for versions shortlisted by local
F0.5, never for exploring ideas (`.claude/rules/project-rules.md`, section 3).

## Budget

| Day | Date | Used | Left |
|---|---|---|---|
| 1 | 2026-09-25 | 2 | 3 (expired) |
| 2 | 2026-09-26 | 3 | 2 |
| 3 | 2026-09-27 | 0 | 5 |

## Before every upload

1. The version has a row in `experiments/experiments.csv` with its local F0.5 on the fixed
   validation split, and its commit has no `-dirty` suffix.
2. Its notebook, `metrics.json` and code are committed. Note the commit hash.
3. `make validate` passes (our checker and the organisers' validator).
4. Add the entry below before uploading, then fill in the public score.
5. Record the score in the registry: `make public V=vNNN SCORE=0.xxxx`.

## Entry template

```markdown
## Submission NN: YYYY-MM-DD HH:MM IST
- Version: vNNN_<slug>
- Commit: <short hash>
- Change vs previous submission: ...
- Model: ...
- Blocking: ...
- Threshold / decision rule: ...
- Local F0.5 (validation fold): 0.xxxx
- Public F0.5: 0.xxxx
- Notes: ...
```

<!-- Add submissions below this line, newest first. -->

## Submission 05: 2026-09-26 ~18:00 IST
- Version: v110_m3_features (on v107's two-stage)
- Commit: c9ec108 (src/ as of 2c1a443); results in 031f767 `exp(v110)`
- Change vs previous submission (v107): M3's four feature groups (idf, token_freq, ctx_idf,
  address_extra: 76 features); blocking B6 + name + address-number exact pass (v105, v109:
  mock candidate recall 0.9677 -> 0.9829 before the filter, 0.9788 after); stage 1 retrained on
  the GPU (XGBoost, 213k train-fold entities absent from the mock, 6.99M of 14.9M rows under the
  4 GB cap); stage 2 with rival features; rule re-tuned on the tight mock.
- Threshold / decision rule: expected-F0.5 decoding, gamma 1.5, expected misses 0.1, max 11, 1-to-1 (tuned for 1 - L_FN - 1.45 L_FP on the mock tune entities)
- Mock F0.5 0.9817; est_public 0.9731 (v107: mock 0.9745, est 0.9659, public 0.966); pair
  recall 0.952 (v107 0.934), precision 0.997
- Public F0.5: (fill in after upload)
- Notes: files in submissions/v110/; both validators PASS; test candidates 4.8 (India), 5.0 (US),
  6.1 (France) per S1; matched share 94.1 / 94.3 / 94.8 %.

## Submission 04: 2026-09-26 12:29 IST
- Version: v107_tight_rule (on v104_two_stage)
- Commit: 86da932 (src/), results in (this commit) `exp(v107)`; v104 in 86da932 `exp(v104)`
- Change vs previous submission (v103): two-stage matcher (v104) + rule tuned on the tight mock.
  v101 stays stage 1: it scores every candidate, keeps the 16 best per S1 with p1 >= 0.01
  (the final candidate set: 4.8-6.0 per S1 on test instead of ~35), and its probabilities give
  competition features (rank, best rival and gap on the S1 side and the pool side) and anchor
  features (each candidate against its entity's best other candidate). Stage 2 is XGBoost on
  the GPU, trained on the mock fold's fit entities with 2-part cross-fitting.
- Threshold / decision rule: expected-F0.5 decoding, gamma 1.5, expected misses 0.05, max 11, 1-to-1 (tuned for 1 - L_FN - 1.45 L_FP on the mock tune entities)
- Mock F0.5 0.9745; est_public 0.9659 (v103: mock 0.9704, est 0.9610, public 0.961)
- Public F0.5: 0.966 (est_public 0.9659: the tight mock predicted it to within 0.0001)
- Notes: files in submissions/v107/; both validators PASS.


## Submission 03: 2026-09-26 10:50 IST
- Version: v103_mock_rule
- Commit: 6de139a (src/), results in fc1a79e `exp(v103)`
- Change vs previous submission: decision rule re-tuned on the tune entities of the mock fold
  (test-shaped: the test's pool size and pool records per S1, every present S1 competing in the
  1-to-1); matcher, features, blocking and token map are v101's
- Model: v101 LightGBM (53 features, 1,666 rounds)
- Blocking: as v101 (config 66540dae, cached candidates); ~35 candidates per test S1
- Threshold / decision rule: tau_abs 0.725, tau_rel 0.7, tau_single 0.775, max_matches 11,
  one_to_one true (was 0.42 / 0.0 / 0.52 in v101)
- Local F0.5 (validation fold): 0.9833 (v101 0.9858); **mock F0.5 0.9704** (v101's rule on
  the mock: 0.9677, public 0.955)
- Public F0.5: **0.961** (uploaded 2026-09-26 10:50 IST; +0.006 over v101)
- Notes: first upload judged by mock F0.5. Expected public ~0.957-0.958 if the mock-public
  offset (+0.013 for v101) holds. Candidate recall on the mock is 0.9655 (val 0.9906): at test
  density blocking loses 3.5 % of true pairs (v105 studies bigger budgets). Both validators
  PASS; 1,732,544 rows, 105,286 empty (v101: 99,746). Files in submissions/v103/.
- Result: public gained +0.006 where the mock predicted +0.0027: the test rewards precision
  more than the mock does (offset mock − public 0.0127 → 0.0094). The mock is tightened next.

## Submission 02: 2026-09-25 ~19:45 IST
- Version: v101_name_frequency
- Commit: f64fc9c (src/), results in the `exp(v101)` commit
- Change vs previous submission: six core-name frequency features (how many S1 / pool records
  share each side's core name, per million, whole fold) + LightGBM cap 4,000 rounds
- Model: LightGBM as v001, 53 features, early stop at 1,666 rounds; tune AUC 0.99986
- Blocking: as v001 (config 66540dae, cached candidates)
- Threshold / decision rule: tau_abs 0.42, tau_rel 0.0, tau_single 0.52, max_matches 11,
  one_to_one true (tune F0.5 0.9862)
- Local F0.5 (validation fold): 0.9858 (harder-val 0.9852, singletons 0.9874)
- Public F0.5: 0.955 (uploaded 2026-09-25 19:56 IST)
- Notes: both validators PASS; test S1 matched France 0.949 / India 0.940 / US 0.943. Expected
  public ~0.956-0.960: the val gain is precision (false-merge pairs -29%), which counts more
  at test decoy density. Files in submissions/v101/. Result: public +0.001 over v001 against
  +0.0014 on val; the offset stayed at -0.031 because val does not see the test's decoy
  density (TRACKER.md, mock-test protocol).

## Submission 01: 2026-09-25 18:59 IST
- Version: v001_base_model
- Commit: 875079a (src/), results in 2f19a54 `exp(v001)`
- Change vs previous submission: first submission, the full V1 pipeline
- Model: LightGBM binary, 63 leaves, lr 0.05, 1,998 rounds (2,000 cap), 47 features
  (blocking, name_fuzzy, name_tokens, legal, numeric, address, context, meta), trained on
  6.72M candidate pairs of 200k fit-side S1 entities; tune AUC 0.9998
- Blocking: config 66540dae, per country: exact name_core / name_sorted / name_squash (pool
  groups <= 50) + name char 3-gram top-10 on short-address pool records + name+address word
  uni+bigram top-25 (max_df 0.01, max_df_abs 10k), cap 60; 34-37 candidates per test S1
- Threshold / decision rule: tau_abs 0.47, tau_rel 0.0, tau_single 0.52, max_matches 11,
  one_to_one true (tuned on all 441k tune-side S1, tune F0.5 0.9846)
- Local F0.5 (validation fold): 0.9844 (harder-val 0.9838, singletons 0.9840, matched 0.9844)
- Public F0.5: 0.954 (offset public - local = -0.030; running mean -0.030 over 1 upload)
- Notes: 1,732,544 rows in both files; our checker (--check-ids) and the organisers' validator
  PASS. Test S1 matched: France 0.950, India 0.941, US 0.943 (val 0.942); matches per S1
  3.3-3.4 as on val. Uploaded bytes kept in submissions/v001/. The -0.030 offset is the
  expected shift: test entities meet ~5x more same-name decoys than val entities (the val
  pool is a 20% sample) and France is unseen; the dense-val check measures the first part.
