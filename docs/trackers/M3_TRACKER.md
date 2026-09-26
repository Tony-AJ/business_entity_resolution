# M3 task tracker: pair features + training pairs

> **Merged into main (PR #8):** the `frequency` group of v040 is called `token_freq` on main,
> where `frequency` already names v101's core-name rates (`fq_*`, used by saved models).

Everything the M3 role owns, with its status. The team-wide view stays in
[TRACKER.md](../../TRACKER.md); this file holds M3's detail. Update a row when its state
changes and commit it with the work (`docs(tracker): ...`).

| | |
|---|---|
| Role | M3: pair features and training pairs (plan group C) |
| Code | `src/entity_resolution/features.py`, `trainset.py` (`sample_s1`, `label_pairs`; `inner_split` is M1's, `mine_false_positives` M4's); M3's switch in `twostage.py` (`extra_groups`) |
| Tests | `tests/test_features.py`, `tests/test_trainset.py` (labels, sampling), `tests/test_twostage.py` (extra groups, stage-1 cache key) |
| Versions | v040–v059 (13 §1); used: v040–v043 |
| Branches | day 1–2 morning: `feat/features-c2-c5` (merged, PR #8); day 2 afternoon: `feat/m3-stage2-features` (local, not pushed yet: I-09) |
| Guides | [07 feature engineering](../plan/07_FEATURE_ENGINEERING.md) (M3's document, §10–§11), [03 §3 brief](../plan/03_TEAM_WORKING_STRATEGY.md), [18 error categories](../plan/18_ERROR_ANALYSIS_FRAMEWORK.md) |

Status: `done` · `doing` · `todo` · `blocked` · `skipped`.

Last updated: 2026-09-26 17:35 IST (day 2).

## Snapshot

- **Decision number since day 2:** est_public on the test-shaped mock (TRACKER "Mock-test
  protocol", "Tight mock"); plain val is a secondary check.
- **Best M3 versions: v042 and v043, a tie.** v042 (M3's groups in stage 2 of v104's two-stage)
  est_public **0.9679**, v043 (M3's groups in stage 1) **0.9678**, against **0.9659** for the
  same pipeline without them (the logged v107, reproduced exactly on this machine). Both KEEP.
  v043's stage 1 alone scores plain val **0.9876**, the best single-stage model so far.
- **Done:** 50 of 60 tasks; 1 doing (E-06, v042 test inference); 5 todo (the push and PR,
  status posts, day-3 process, and M1's upload decision); 4 skipped with reasons.
- **Next:** v042's test files to M1 (E-06/E-07), push `feat/m3-stage2-features` and open its
  PR when the user says so (I-09).

## 1. Module deliverables (03 §3, 02 §5)

Built by M1 in the day-1 walking skeleton, on M3's module; M3 owns and extends them.

| ID | Task | Status | Done in | Evidence |
|---|---|---|---|---|
| D-01 | `features.py` schema: `FEATURE_COLUMNS`, `REGISTRY`, `feature_names`, `NAN_FEATURES` | done | 25 Sep, `66a825a` | column contract the model stores |
| D-02 | `iter_chunks`: chunks that never split an S1 group | done | 25 Sep, `b81e9f6` | `test_iter_chunks_never_splits_group` |
| D-03 | `build_features`: chunked, float32, index = pairs index | done | 25 Sep, `b81e9f6` | ~250k pairs/s on 12 threads |
| D-04 | v001 groups: blocking, name_fuzzy, name_tokens, legal, numeric, address, context, meta (47 features) | done | 25 Sep, `b81e9f6` | v001 local F0.5 0.9844 |
| D-05 | `pool_context` (`ctx_pool_indegree`), opt-in | done | 25 Sep, `b81e9f6` | biased low on the sampled fit side, so off by default |
| D-06 | `trainset.sample_s1`: S1 sampled by id hash, never pairs | done | 25 Sep, `6197954` | `test_sample_s1_order_independent` |
| D-07 | `trainset.label_pairs`: `label` int8, row order kept | done | 25 Sep, `6197954` | `test_label_pairs` |

## 2. Feature work (TRACKER #16, #17, 17a–17b, 07 §3, §6, §10, §11)

| ID | Task | Status | Done in | Evidence |
|---|---|---|---|---|
| F-01 | Pool statistics per country (`pool_stats`, `CountryStats`), counted once and shared by every chunk; `pipeline.score` counts once per call | done | 26 Sep, `b53be04`, `0cee0b4` | `test_shared_stats_match_whole_build`, `test_stats_are_counted_per_country` |
| F-02 | Token matrix split out of `_token_sets` (refactor, outputs unchanged) | done | 26 Sep, `1fd7220` | the 17 day-1 feature tests pass unchanged at that commit |
| F-03 | C2 `idf` group (8): idf cosine, rarest shared token, coverage per side, for name and address, on every pair (TRACKER #16) | done | 26 Sep, `9d77eac` | 6.5 % of v040's gain; 5.6 % of v043's stage 1 |
| F-04 | C5 `ctx_idf` group (5): rank and gap of the idf cosines in the S1 group, exact-name candidates | done | 26 Sep, `9d77eac` | `ctx_rank_idf_addr` #13 in v043's stage 1 |
| F-05 | C5 `token_freq` group (4; `frequency` in v040): pool records of the country sharing the exact name / address (TRACKER #17) | done | 26 Sep, `9d77eac` | `freq_addr_r` #7 of v042's stage 2 |
| F-06 | C3/C4 `address_extra` group (6): reverse containment, number containment, postcode prefix, empty S1 address, length ratio (TRACKER #17a) | done | 26 Sep, `c1439ab` | 6.3 % of v043's stage 1; `num_contain_l` #7 |
| F-07 | Keep `DEFAULT_GROUPS` at the v001 set so logged versions reproduce; new groups opt in | done | 26 Sep, `9d77eac` | `len(feature_names()) == 47` |
| F-08 | Speed check of the new groups | done | 26 Sep | 600k synthetic pairs: idf 332k/s, token_freq 321k/s, address_extra ~650k/s (v001 groups 72k/s); `pool_stats` ~5 s per million records |
| F-09 | Pool-side competition without the S1-sampling bias (TRACKER #17, second half) | done | 26 Sep (M1) | superseded by `stacking.py`: `pool_rank`, `pool_gap`, `pool_p1_sum`, `pool_degree` over every present S1 of the mock / test partition; `pool_gap` carries ~71 % of stage 2's gain. `ctx_pool_indegree` stays opt-in |
| F-10 | `token_freq` pool-size robustness at the test's density | done | 26 Sep, v042 | the mock has the test's density: arm B (with) 0.9679 vs arm C (without) 0.9675, so the counts help there too |
| F-11 | Faster token_freq look-ups (per-record counts once per partition) | skipped | | in the two-stage matcher M3's groups run on the kept pairs only (6.3M of 47.3M mock pairs), so the look-up cost stopped mattering |
| F-12 | Drop features at zero gain for three consecutive versions (08 §8) | skipped | | `sim_addr_char`, `addr_empty_r`, `addr_empty_l`, `postcode_prefix_eq` stay at zero, but v101 / v104 / v107's saved models read the v001 columns and zero-gain columns cost nothing; a cleanup after the challenge |
| F-13 | Cross-field interactions (07 §3) | skipped | | trees learn them; stage 2 lives on competition features (95 % of its gain in v043); importance never justified them |
| F-14 | `TwoStageConfig.extra_groups`: M3's groups built on the kept pairs with pool statistics of the whole partition, into stage 2 on the mock and on test (TRACKER 17b) | done | 26 Sep, `638e774` | `test_extra_groups_ride_on_the_kept_pairs`, `test_extra_groups_are_checked` |
| F-15 | Stage-1 cache records its `TwoStageConfig` fields and refuses a reuse under another configuration (review finding: stale features were returned silently) | done | 26 Sep, `024b47c` | `test_stage1_cache_refuses_another_configuration` |

## 3. Experiments (C group, v040–v059)

| ID | Version | Change | Plain val F0.5 | Mock F0.5 | est_public | Decision | Status |
|---|---|---|---|---|---|---|---|
| E-01 | v040 | v001 + idf, frequency, ctx_idf, address_extra (70 features) | 0.9870 (+0.0026) | – | – | KEEP | done, `eec55b9` |
| E-02 | v041 | v040 without `frequency` (ablation) | 0.9866 (+0.0022) | – | – | KEEP vs v001, 0.0004 below v040 | done, `ff67b17` |
| E-03 | C1–C5 ladder of 07 §6 | | | | | | skipped: v001 already ships every one of these groups; importance per version replaces the ladder |
| E-04 | v042 | v104 two-stage + M3's groups in stage 2; 3 arms; v107 rule tuning | – | 0.9762 | **0.9679** (+0.0020) | KEEP | done, `d01fc5f` (TRACKER 17c) |
| E-05 | v043 | stage 1 = v101 + M3's groups; v104 two-stage; v107 rule tuning | 0.9876 (stage 1) | 0.9762 | 0.9678 (+0.0019) | KEEP | done, `5f3b842` (TRACKER 17d) |
| E-06 | v042 test inference: both TSVs, our checker; files in `submissions/v042/` | | | | | | doing (TRACKER 17e) |
| E-07 | Upload of v042 (or v043) | | | | | M1's decision (LEADERBOARD.md is M1's) | todo (M1) |

Every comparison is same-machine: v001 (0.98436, harder 0.98383), v101 (rule 0.42/0/0.52, 1,666
rounds) and v107 (est_public 0.96588, mock F0.5 0.97451) were re-run here and reproduced their
logged numbers exactly, so the deltas are the features' effect.

## 4. Tests (07 §7, 12 §5)

| ID | Test | Status | Done in |
|---|---|---|---|
| T-01 | 07 §7 named tests: `test_registry_names_unique_and_count`, `test_identical_records_score_one`, `test_disjoint_records_score_zero`, `test_known_values`, `test_nan_policy`, `test_missing_fields`, `test_context_rank_and_gap`, `test_pool_indegree_partition_wide`, `test_iter_chunks_never_splits_group`, `test_deterministic_and_float32` | done | 25 Sep, `b8799c0` |
| T-02 | 12 §5: `test_empty_pairs_keep_columns`, `test_symmetric_similarities`, plus chunking, isolation and bad-input tests | done | 25 Sep, `b8799c0` |
| T-03 | `test_trainset.py`: `test_label_pairs`, `test_sample_s1_order_independent` | done | 25 Sep, `6197954` |
| T-04 | New groups: `test_idf_known_values` (hand-computed idf), `test_stats_are_counted_per_country`, `test_frequency_missing_and_counts`, `test_ctx_idf_rank_gap_and_same_name`, `test_stats_group_ranges`, `test_stats_groups_need_pool_statistics`, `test_shared_stats_match_whole_build` | done | 26 Sep, `c27a5ff` |
| T-05 | `address_extra`: `test_address_extra_values`, `test_address_extra_mirrors_under_swap` | done | 26 Sep, `de3a870` |
| T-06 | Pipeline end to end with the stats groups, France partition included (`test_stats_feature_groups_end_to_end`) | done | 26 Sep, `0cee0b4` |
| T-07 | Two tests failing on Windows before any change (`test_data` UTF-8 fixture, `test_model` logreg last-bit drift) fixed so every commit is green | done | 26 Sep, `c8b7edc`, `1d1df63` |
| T-08 | Full suite green after PR #8: 217 tests | done | 26 Sep, `29fdd6a` |
| T-09 | Two-stage with extra groups end to end (mock stage 1, stage 2, save / load, test files that validate, France included) and config checks | done | 26 Sep, `638e774` |
| T-10 | Stage-1 cache: a reuse under another configuration raises; an older sidecar is still read | done | 26 Sep, `024b47c` |
| T-11 | Full suite on `feat/m3-stage2-features`: ruff + 266 pytest tests (XGBoost 3.1.1 installed in the venv for main's GPU backend) | done | 26 Sep, `024b47c` |

## 5. Analysis and error follow-up (07 §8, 08 §8, 18 §3)

| ID | Task | Status | Evidence |
|---|---|---|---|
| A-01 | Importance top 20 and every new feature's rank logged per version | done | v040–v043 `metrics.json`; v043 ranks M3's features in both stages |
| A-02 | Class means of the new features on val (TP, FN, FP, TN) | done | v040 §6.2: `num_contain_l` FP 0.62, TP 0.89, TN 0.08 |
| A-03 | Failure case: same-name same-street decoys separated by numbers and containment | done | number containment is the strongest FP/TP separator (A-02); v043 stage 1 #7 |
| A-04 | Failure case: transliterated names (`non_latin = yes` slice) | done | v040 slice +0.0021 (0.9840 → 0.9861) |
| A-05 | Failure case: true pairs with `ad_jaccard == 0` (bare-city addresses) | done | v042 §6.3: none of the 30,094 remaining candidate misses has two non-empty addresses without a shared token |
| A-06 | Doc 18 categories owned by M3, tagged on v042's mock errors | done | misses: `name_typo` 4,737, `dba_trade_name` 3,871, `missing_address_component` 366, `word_order` 25, `postal_code_mismatch` 3 (39,772 are blocking / filter losses, M1's); false merges: `same_address_different_business` 445, `same_name_different_business` 35; `landmark_address` needs errors.py's landmark regex (not built) |
| A-07 | Remaining false merges: a full record against a name-only pool record of a close name | done | v042 cut false merges 22 %; the survivors (`Simba International Ltd` / `Simba International`, empty address) have no address evidence left to use |

## 6. Integration and process (13, 14)

| ID | Task | Status | Evidence |
|---|---|---|---|
| I-01 | Push `feat/features-c2-c5` and open its PR | done | PR #8, merged by M1 at 12:44 |
| I-02 | M1 merge gate and integration | done | PR #8 in main; v042 / v043 are the integrated two-stage runs with M3's groups |
| I-03 | `src/` committed before each run (csv commit not `-dirty`) | done | v040 `6e7b015`, v041 `eec55b9`, v042 `638e774`, v043 `024b47c` |
| I-04 | Notebook + `metrics.json` + csv row committed together per version | done | `eec55b9`, `ff67b17`, `d01fc5f`, `5f3b842` (v043's `metrics.json` written with one line per nested table to fit the 400-line hook; same content) |
| I-05 | Doc 07 documents the groups and their two-stage results | done | §10 `6e7b015`; §11 `02f850d`, `0b2122c` |
| I-06 | TRACKER.md rows 16, 17, 17a–17e and the version table | done | `29fdd6a`, `2124541` |
| I-07 | 13:00 / 21:00 IST status posts (13 §6); lines below | todo | |
| I-08 | Remove the finished agent worktree and merged branches | done | worktree and 4 merged local branches removed |
| I-09 | Push `feat/m3-stage2-features` and open its PR (14 §4 checklist) | todo | waits for the user's go-ahead (CLAUDE.md: push only when asked) |
| I-10 | Environment: `xgboost==3.1.1` (main's pin) installed in this venv; `nvidia-nccl-cu12` is Linux-only and not needed on Windows | done | GPU training checked on the RTX 2050 |

## 7. Day 3 (27 Sep)

| ID | Task | Status |
|---|---|---|
| Z-01 | Last feature PR windows 10:30 and 14:30 IST (14 §7) | todo: I-09's PR |
| Z-02 | Features part of the final documentation: M3's groups in `docs/methodology.md` §4, v040 / v042 / v043 in its results table | done, `f22c3a9` |
| Z-03 | Freeze 18:00 IST: only fixes after it | todo |

## Status posts (13 §6 format)

```
M3  v042  C2  v104 two-stage + M3 groups in stage 2   mock 0.9762 (+0.0017)  est_public 0.9679 (+0.0020)  KEEP  parent v107
M3  v043  C5  stage 1 = v101 + M3 groups, two-stage   mock 0.9762 (+0.0017)  est_public 0.9678 (+0.0019)  KEEP  parent v107
next: v042 test files for M1's upload decision; PR of feat/m3-stage2-features
```

## Commits on `feat/m3-stage2-features`

| Commit | Change |
|---|---|
| `638e774` | feat(twostage): add opt-in extra feature groups for stage 2 |
| `d01fc5f` | exp(v042): M3 groups in stage 2, mock F0.5 0.9762, est_public 0.9679 |
| `02f850d` | docs(plan): add M3's two-stage integration and v042 results to 07 |
| `024b47c` | fix(twostage): refuse a stage-1 cache built under another config |
| `5f3b842` | exp(v043): M3 groups in stage 1, mock F0.5 0.9762, est_public 0.9678 |
| `0b2122c` | docs(plan): add v043 to 07's two-stage section |
| `2124541` | docs(tracker): record M3's two-stage work, v042 and v043 |
| `f22c3a9` | docs(methodology): add M3's feature groups and v040-v043 results |

Day 1–2 morning (`feat/features-c2-c5`, merged as PR #8): `c8b7edc`, `1d1df63`, `1fd7220`,
`c1439ab`, `de3a870`, `b53be04`, `9d77eac`, `c27a5ff`, `0cee0b4`, `26e25c7`, `6e7b015`,
`eec55b9`, `ff67b17`, `29fdd6a`, `87c2a46`.
