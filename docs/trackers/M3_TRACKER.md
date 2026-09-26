# M3 task tracker: pair features + training pairs

Everything the M3 role owns, with its status. The team-wide view stays in
[TRACKER.md](../../TRACKER.md); this file holds M3's detail. Update a row when its state
changes and commit it with the work (`docs(tracker): ...`).

| | |
|---|---|
| Role | M3: pair features and training pairs (plan group C) |
| Code | `src/entity_resolution/features.py`, `trainset.py` (`sample_s1`, `label_pairs`; `inner_split` is M1's, `mine_false_positives` M4's) |
| Tests | `tests/test_features.py`, `tests/test_trainset.py` (labels, sampling) |
| Versions | v040–v059 (13 §1) |
| Branch | `feat/features-c2-c5` (local, not pushed yet: see I-01) |
| Guides | [07 feature engineering](../plan/07_FEATURE_ENGINEERING.md) (M3's document), [03 §3 brief](../plan/03_TEAM_WORKING_STRATEGY.md), [18 error categories](../plan/18_ERROR_ANALYSIS_FRAMEWORK.md) |

Status: `done` · `doing` · `todo` · `blocked` · `skipped`.

Last updated: 2026-09-26 12:27 IST (day 2).

## Snapshot

- **Best M3 version: v040**, local macro F0.5 **0.9870** (v001 0.9844, +0.0026), harder-val
  0.9864, every slice up, KEEP. v041 (v040 without `frequency`) 0.9866, also KEEP.
- **Done:** 33 of 51 tasks, 10 of them from M1's day-1 skeleton of M3's module. Open: 15
  todo, 1 doing, 1 blocked; 1 skipped.
- **Next:** push the branch and open the PR (I-01), M1's integrated run (I-02), dense-val
  check of the `frequency` group (F-10).

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

## 2. Feature work (TRACKER #16, #17, 07 §3, §6, §10)

| ID | Task | Status | Done in | Evidence |
|---|---|---|---|---|
| F-01 | Pool statistics per country (`pool_stats`, `CountryStats`), counted once and shared by every chunk; `pipeline.score` counts once per call | done | 26 Sep, `b53be04`, `0cee0b4` | `test_shared_stats_match_whole_build`, `test_stats_are_counted_per_country` |
| F-02 | Token matrix split out of `_token_sets` (refactor, outputs unchanged) | done | 26 Sep, `1fd7220` | the 17 day-1 feature tests pass unchanged at that commit |
| F-03 | C2 `idf` group (8): idf cosine, rarest shared token, coverage per side, for name and address, on every pair (TRACKER #16) | done | 26 Sep, `9d77eac` | 6.5 % of v040's gain; `idf_addr_cos` #8 |
| F-04 | C5 `ctx_idf` group (5): rank and gap of the idf cosines in the S1 group, exact-name candidates | done | 26 Sep, `9d77eac` | 1.8 % of v040's gain; `ctx_rank_idf_addr` #14 |
| F-05 | C5 `frequency` group (4): pool records of the country sharing the exact name / address (TRACKER #17, name-frequency half) | done | 26 Sep, `9d77eac` | +0.0004 F0.5 (v040 vs v041), all of it recall |
| F-06 | C3/C4 `address_extra` group (6): reverse containment, number containment, postcode prefix, empty S1 address, length ratio (TRACKER #17a) | done | 26 Sep, `c1439ab` | 5.6 % of v040's gain; `num_contain_l` #7 |
| F-07 | Keep `DEFAULT_GROUPS` at the v001 set so logged versions reproduce; new groups opt in | done | 26 Sep, `9d77eac` | `len(feature_names()) == 47` |
| F-08 | Speed check of the new groups | done | 26 Sep | 600k synthetic pairs: idf 332k/s, frequency 321k/s, address_extra ~650k/s (v001 groups 72k/s); `pool_stats` ~5 s per million records |
| F-09 | Pool-side competition without the S1-sampling bias (TRACKER #17, second half) | todo | | `ctx_pool_indegree` stays opt-in; needs the in-degree counted over all fit-side S1 (a pipeline change, with M1) |
| F-10 | `frequency` pool-size robustness: dense-val check (TRACKER #13, M1) of v040 against v041 | todo | | learned on the 6.2M fit pool, still helps on the 2.1M val pool; test pools are per country and larger |
| F-11 | Faster frequency look-ups: per-record counts once per partition instead of a hash per chunk | todo | | val scoring 814 s (v040) against 532 s (v041); part of it is paging on a 12 GB machine |
| F-12 | Drop features at zero gain for three consecutive versions (08 §8) | doing | | `sim_addr_char`, `addr_empty_r` are at zero in v001, v040 and v041 (eligible); `addr_empty_l` at zero in v040, v041; changing a v001 group needs a new version |
| F-13 | Cross-field interactions (07 §3): `name_strong_addr_weak`, `addr_strong_name_weak`, `both_strong` | todo | | low priority: trees learn them; add only if importance justifies |

## 3. Experiments (C group, v040–v059)

| ID | Version | Change | Local F0.5 | Harder | Decision | Status |
|---|---|---|---|---|---|---|
| E-01 | v040 | v001 + idf, frequency, ctx_idf, address_extra (70 features) | 0.9870 (+0.0026) | 0.9864 | KEEP | done, `eec55b9` |
| E-02 | v041 | v040 without `frequency` (ablation) | 0.9866 (+0.0022) | 0.9861 | KEEP vs v001, 0.0004 below v040 | done, `ff67b17` |
| E-03 | C1–C5 ladder of 07 §6 (names only, + blocking, + address, + numeric, + context) | | | | | skipped: v001 already ships every one of these groups; importance per version replaces the ladder |
| E-04 | v042 | v040 minus the zero-gain features (F-12) | | | | todo |
| E-05 | v043 | v040 + interactions (F-13), only if F-12/E-04 leave room | | | | todo |

Every run so far: parent v001 re-run on this machine reproduced its logged scores exactly
(0.98436, harder 0.98383), so the deltas are the features' effect.

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
| T-08 | Full suite green: `ruff check src tests` + 217 pytest tests | done | 26 Sep, `29fdd6a` |

## 5. Analysis and error follow-up (07 §8, 08 §8, 18 §3)

| ID | Task | Status | Evidence |
|---|---|---|---|
| A-01 | Importance top 20 and every new feature's rank logged per version | done | v040/v041 `metrics.json`, `artifacts/importance.csv` |
| A-02 | Class means of the new features on val (TP, FN, FP, TN) | done | v040 §6.2: `num_contain_l` FP 0.62, TP 0.89, TN 0.08 |
| A-03 | Failure case: same-name same-street decoys separated by numbers and containment | done | number containment is the strongest FP/TP separator (A-02) |
| A-04 | Failure case: transliterated names (`non_latin = yes` slice) | done | v040 slice +0.0021 (0.9840 → 0.9861) |
| A-05 | Failure case: true pairs with `ad_jaccard == 0` (bare-city addresses) | todo | inspect v040's missed pairs of that kind |
| A-06 | Doc 18 categories owned by M3: #1 name_typo, #4 word_order, #5 dba_trade_name, #7 missing_address_component, #8 landmark_address, #10 postal_code_mismatch, #11 same_name_different_business, #12 same_address_different_business | blocked | needs `errors.tag_errors` (M5, TRACKER #22) |
| A-07 | New false merges of v040: pool records with an empty address and a shortened name (`Toss Systems`, `Kent Future`) | todo | name-only evidence stays the weak spot; next feature idea |

## 6. Integration and process (13, 14)

| ID | Task | Status | Evidence |
|---|---|---|---|
| I-01 | Push `feat/features-c2-c5` and open the PR with the 14 §4 checklist | todo | remote holds only `1d1df63` (published from this clone at 09:32) |
| I-02 | M1 merge gate, then the integrated v1xx run with `feature_groups = DEFAULT_GROUPS + ("idf", "frequency", "ctx_idf", "address_extra")` | todo (M1) | overlaps TRACKER 14a (M1's name-frequency v101) |
| I-03 | `src/` committed before each run (csv commit not `-dirty`) | done | v040 ran on `6e7b015`, v041 on `eec55b9` |
| I-04 | Notebook + `metrics.json` + csv row committed together per version | done | `eec55b9`, `ff67b17` |
| I-05 | Doc 07 §10 documents the new groups | done | `6e7b015` |
| I-06 | TRACKER.md rows 16, 17, 17a and the version table | done | `29fdd6a` |
| I-07 | 13:00 / 21:00 IST status posts (13 §6); lines below | todo | |
| I-08 | Remove the finished agent worktree (`.claude/worktrees/agent-a4304…`, branch `worktree-agent-a4304ba2cb49bacf0`) | todo | optional clean-up; `feat/address-extra` points at the same commit |

## 7. Day 3 (27 Sep)

| ID | Task | Status |
|---|---|---|
| Z-01 | Last feature PR windows 10:30 and 14:30 IST (14 §7) | todo |
| Z-02 | Features part of the final documentation: registry by group, importance top 20 (16 §4) | todo |
| Z-03 | Freeze 18:00 IST: only fixes after it | todo |

## Status posts (13 §6 format)

```
M3  v040  C2  v001 + idf, frequency, ctx_idf, address_extra   f_beta 0.9870 (+0.0026)  harder 0.9864 (+0.0026)  KEEP  parent v001
M3  v041  C2  v040 without frequency (ablation)                f_beta 0.9866 (+0.0022)  harder 0.9861 (+0.0022)  KEEP  parent v001
next: v042 = v040 minus zero-gain features; dense-val check of frequency with M1
```

## Commits on `feat/features-c2-c5`

| Commit | Change |
|---|---|
| `c8b7edc` | test(data): append fixture rows as UTF-8 on every platform |
| `1d1df63` | test(model): allow last-bit drift in chunked logreg predict |
| `1fd7220` | refactor(features): split the token matrix out of _token_sets |
| `c1439ab` | feat(features): add address_extra group (reverse containment, numbers) |
| `de3a870` | test(features): cover the address_extra group |
| `b53be04` | feat(features): add per-country pool statistics for stats groups |
| `9d77eac` | feat(features): add idf, frequency and ctx_idf feature groups |
| `c27a5ff` | test(features): cover idf, frequency and ctx_idf values and statistics |
| `0cee0b4` | feat(pipeline): share pool statistics across scoring chunks |
| `26e25c7` | Merge branch 'feat/address-extra' into feat/features-c2-c5 |
| `6e7b015` | docs(plan): document the opt-in v2 feature groups in 07 |
| `eec55b9` | exp(v040): four new feature groups, local F0.5 0.9870 |
| `ff67b17` | exp(v041): v040 without frequency, local F0.5 0.9866 |
| `29fdd6a` | docs(tracker): record the M3 feature groups and v040/v041 results |
