# Leaderboard submissions

Auditable log of every upload to the challenge portal, newest first. Budget: **5 per day,
15 in total** (25–27 Sep 2026, IST). Uploads are only for versions shortlisted by local
F0.5, never for exploring ideas (`.claude/rules/project-rules.md`, section 3).

## Budget

| Day | Date | Used | Left |
|---|---|---|---|
| 1 | 2026-09-25 | 1 | 4 |
| 2 | 2026-09-26 | 0 | 5 |
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

## Submission 01: 2026-09-25 ~19:00 IST
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
- Public F0.5: (fill in after upload)
- Notes: 1,732,544 rows in both files; our checker (--check-ids) and the organisers' validator
  PASS. Test S1 matched: France 0.950, India 0.941, US 0.943 (val 0.942); matches per S1
  3.3-3.4 as on val. Uploaded bytes kept in experiments/v001_base_model/artifacts/uploaded.tsv.
