# 07 — Pair feature engineering (module `features.py`, owner M3, plan group C)

What you build: `build_features(pairs, s1n, pooln, groups, chunk_rows) -> DataFrame` of
float32 columns, `index == pairs.index`, computed in chunks that never split an S1 group.
Features live in `REGISTRY: dict[str, FeatureGroup]`; each group is a pure function
`(pairs_chunk, left, right) -> DataFrame` where `left`/`right` are the normalised S1 and pool
rows aligned to the chunk. `feature_names(groups)` returns the column order the model uses.

## 1. Principles

- Names rank, addresses decide (01 §12): give the model many address-agreement features,
  including ones that fire when addresses disagree.
- Every similarity is symmetric and in [0, 1]; missing → NaN (LightGBM handles it) plus an
  explicit flag where missingness is informative (empty address, no numbers).
- Group context features (rank, gap, competition) let the pairwise model see what the
  decision layer will see; they are the cheapest precision gain.
- No country enumeration; `country` is not a feature (open set). Source (S2/S3) is fixed.
- Deterministic and chunk-safe: no randomness, no state across chunks except the pool
  in-degree table computed once per partition.

## 2. Registry (48 features)

`L` = S1 side, `R` = pool side. rapidfuzz scorers are called through
`rapidfuzz.process.cpdist(L, R, scorer=..., workers=-1, dtype=np.float32)` (pairwise, C++
threads; ≈ 1M pairs/s per scorer), divided by 100.

| Group | Feature | Definition | Missing |
|---|---|---|---|
| blocking (6) | `pass_exact` | `(pass & 7) != 0` | 0/1 |
| | `pass_name_char`, `pass_name_addr` | bit 8 / bit 16 set | 0/1 |
| | `sim_name_char`, `sim_name_addr_word`, `sim_addr_char` | copied from the pairs frame | NaN if pass absent |
| name_fuzzy (10) | `nm_ratio`, `nm_partial`, `nm_token_sort`, `nm_token_set`, `nm_jw` | `fuzz.ratio`, `fuzz.partial_ratio`, `fuzz.token_sort_ratio`, `fuzz.token_set_ratio`, `distance.JaroWinkler.normalized_similarity` on `name_norm` | NaN if either empty |
| | `core_ratio`, `core_token_set`, `core_jw`, `core_indel`, `squash_ratio` | same scorers on `name_core`; `distance.Indel.normalized_similarity`; `fuzz.ratio` on `name_squash` | NaN if either empty |
| name_tokens (8) | `tok_jaccard`, `tok_dice` | on `name_core` token sets (binary word CSR: `A.multiply(B).sum(1)` row-wise) | NaN if either empty |
| | `tok_common`, `tok_len_l`, `tok_len_r` | shared token count, token counts | 0 |
| | `first_eq`, `sorted_eq`, `prefix4_eq` | `name_first` equal; `name_sorted` equal; first 4 chars of `name_squash` equal | 0/1 |
| legal (3) | `legal_eq` | `legal_form` equal and non-empty | 0/1 |
| | `legal_missing_l`, `legal_missing_r` | `legal_form == ""` | 0/1 |
| numeric (4) | `num_jaccard` | Jaccard of `addr_nums` token sets | NaN if either has no numbers |
| | `num_shared_any`, `num_first_eq` | any shared number; first number (house number) equal | 0/1, NaN if none |
| | `postcode_eq` | equal non-empty postcodes | 1 / 0 / NaN if either empty |
| address (8) | `ad_token_set`, `ad_partial`, `ad_ratio` | rapidfuzz on `addr_norm` | NaN if either empty |
| | `ad_jaccard` | token Jaccard of `addr_norm` | NaN if either empty |
| | `ad_contain` | share of L tokens present in R (asymmetric containment, catches dropped components) | NaN |
| | `region_eq`, `last_eq` | `region` equal non-empty; `addr_last` equal non-empty | 0/1 |
| | `addr_empty_r` | pool address empty | 0/1 |
| context (6) | `ctx_rank_name` | rank of `sim_name_char` within the S1 group (1 = best; NaN sims last) | int |
| | `ctx_gap_name` | best `sim_name_char` in group − this pair's | NaN if absent |
| | `ctx_rank_addr`, `ctx_gap_addr` | same on `ad_token_set` | |
| | `ctx_n_cands` | candidates in the S1 group | int |
| | `ctx_pool_indegree` | number of S1 groups this pool record appears in (partition-wide, computed before chunking) | int |
| meta (3) | `is_s3` | `entity_id` starts with `S3-` | 0/1 |
| | `non_latin_r` | pool row was transliterated | 0/1 |
| | `len_ratio_name` | `min(len)/max(len)` of `name_core` | NaN if either empty |

Total: 6 + 10 + 8 + 3 + 4 + 8 + 6 + 3 = 48.

## 3. Cross-field interactions (cheap, add in C4/C5 if importance justifies)

`name_strong_addr_weak = (core_token_set ≥ 0.9) & (ad_jaccard < 0.2)` (the decoy signature),
`addr_strong_name_weak = (ad_token_set ≥ 0.9) & (core_token_set < 0.5)` (the rename
signature), `both_strong`. Tree models learn these from the base features, so keep them
out of V1; log importance first.

## 4. Chunking

```
build_features(pairs, s1n, pooln, groups, chunk_rows):
    indeg = pairs.entity_id.value_counts()                     # once per call
    s1_pos = s1n.index_of(pairs.source1_entity_id); pool_pos = likewise  (pyarrow index_in)
    for sl in iter_chunks(pairs, chunk_rows):                  # boundaries on S1 group edges
        left  = s1n.take(s1_pos[sl]); right = pooln.take(pool_pos[sl])
        frames = [REGISTRY[g](pairs.iloc[sl], left, right) for g in groups]
        yield concat(frames, axis=1).astype(float32)
    concat all → assert list(columns) == feature_names(groups)
```

`iter_chunks`: pairs are sorted by `source1_entity_id`; take `chunk_rows` rows, then extend
the slice to the end of the current group (`searchsorted` on group codes). Group-context
features use `groupby(source1_entity_id)` inside the chunk only, which is correct because a
group never spans two chunks.

Cost: 24M test pairs × 13 rapidfuzz scorer calls ≈ 5 min on 12 threads; sparse Jaccard
≈ 1 min; total features + predict ≈ 15 min. Memory: a 2M-row chunk × 48 float32 = 0.4 GB.

## 5. Training-pair construction (`trainset.py`, same owner)

```
inner_split(train)  → fit_fold, tune_fold          (M1 provides; seed 4242, 25% tune)
sample_s1(fold.s1, n, seed=7): hash_unit(entity_id, seed) < n / len(fold.s1)   (order-free, deterministic)
label_pairs(pairs, truth_pairs): merge on both ids, indicator → label int8 (1 = true pair)
```

Sample S1 entities, never pairs: every true match and every decoy of a sampled entity is
kept, so the class balance and the group context match inference. Sizes: 200k fit S1 →
≈ 5–6M pairs, ≈ 12% positive; 100k tune S1 → ≈ 3M pairs. Truth pairs outside the candidate
set are lost to the model (they are blocking misses) but counted by the metric.

## 6. Experiments (C group, versions v040–v059)

| Version | Change | Keep if (val macro F0.5 with the integrated model) |
|---|---|---|
| C1 | name_fuzzy + name_tokens + legal only | reference |
| C2 | + blocking sims and P3 cosine | +0.005 or importance in top 10 |
| C3 | + address group | expected largest gain (decoys) |
| C4 | + numeric group, region/last from learned map | singleton F0.5 improves |
| C5 | + context group; then interactions | precision at equal recall improves |

Log per version: feature importance (gain) top 20, AUC on tune (diagnostic only), macro
F0.5 and singleton F0.5 on val, feature build seconds. Drop features with zero gain in three
consecutive versions.

## 7. Tests (`tests/test_features.py`)

| Test | Assertion |
|---|---|
| `test_registry_names_unique_and_count` | 48 unique names; `feature_names(all) == columns of build_features` |
| `test_identical_records_score_one` | every similarity == 1.0 for an identical pair; `tok_jaccard == 1` |
| `test_disjoint_records_score_zero` | unrelated strings → ratios near 0, `tok_jaccard == 0`, `first_eq == 0` |
| `test_known_values` | `fuzz.ratio("kitten","sitting")/100`, Jaccard of `{a,b}` vs `{b,c}` = 1/3, `postcode_eq` cases 1/0/NaN |
| `test_nan_policy` | NaN only in the columns allowed to be NaN; flags are never NaN |
| `test_missing_fields` | empty address → address sims NaN, `addr_empty_r == 1`; empty name → name sims NaN |
| `test_context_rank_and_gap` | a 3-candidate group: ranks 1,2,3 by `sim_name_char`; best gap 0 |
| `test_pool_indegree_partition_wide` | a pool id in two S1 groups has `ctx_pool_indegree == 2` in both chunks |
| `test_iter_chunks_never_splits_group` | with `chunk_rows` smaller than a group, every group stays whole |
| `test_deterministic_and_float32` | two runs identical; all dtypes float32 |
| `test_label_pairs` | only truth pairs get label 1; a truth pair outside candidates is absent |
| `test_sample_s1_order_independent` | same ids sampled after shuffling the frame |

## 8. Failure cases to inspect

- Decoys with `core_token_set == 1` and high `ad_token_set`: same-name, same-street
  different-suite businesses; check `num_first_eq` and `ad_contain` separate them.
- True pairs with `ad_jaccard == 0` (4–5%): usually one address is a bare city; make sure
  `addr_empty_r`/`addr_tokens` let the model fall back to name evidence with lower confidence.
- Transliterated names: `core_jw` and `nm_partial` are the most tolerant scorers; verify on
  the `non_latin_r == 1` slice.

## 9. Integration

Consumes `PAIR_COLUMNS` and the normalised columns listed in 05 §11. Produces the frame the
model trains and predicts on; the model stores `feature_names` and refuses a mismatch, so
adding a feature means a new model version. New dependency: `rapidfuzz==3.14.6` (MIT).

## 10. V2 groups (opt-in, v040+)

`DEFAULT_GROUPS` stays the v001 set (47 features: every §2 group except `pool_context`), so
logged versions reproduce. A version adds groups through `PipelineConfig.feature_groups`;
`feature_names` then appends their columns after the v001 ones.

| Group (plan) | Feature | Definition | Missing |
|---|---|---|---|
| idf (C2, #16) | `idf_name_cos` | cosine of binary-tf idf vectors of the `name_core` tokens, for every pair (the `sim_*` cosines exist only for pairs their pass proposed) | NaN if either empty |
| | `idf_name_top` | largest shared idf / largest possible idf (a shared rare token) | 0 if none shared |
| | `idf_name_cover_l`, `idf_name_cover_r` | share of that side's idf mass found on the other side | NaN if either empty |
| | `idf_addr_*` | the same four on `addr_norm` | NaN if either empty |
| frequency (C5, #17) | `freq_name_l`, `freq_name_r` | pool records of the country whose `name_core` equals the S1's / the pool record's (decoy risk) | NaN if that side empty |
| | `freq_addr_l`, `freq_addr_r` | the same on `addr_norm` (shared buildings) | NaN if that side empty |
| ctx_idf (C5) | `ctx_rank_idf_name`, `ctx_gap_idf_name` | rank and gap of `idf_name_cos` inside the S1 group (sees every candidate) | gap NaN where the cosine is |
| | `ctx_rank_idf_addr`, `ctx_gap_idf_addr` | the same on `idf_addr_cos` | |
| | `ctx_n_same_name` | candidates of the group with the S1's exact non-empty `name_core` | 0 |
| address_extra (C3/C4) | `ad_contain_r` | share of the pool address tokens found in the S1 address | NaN if either empty |
| | `addr_empty_l` | S1 address empty | 0/1 |
| | `num_contain_l`, `num_contain_r` | share of one side's address numbers found on the other | NaN if either has none |
| | `postcode_prefix_eq` | equal first 3 postcode characters | NaN if either empty |
| | `addr_len_ratio` | fewer / more distinct address tokens | NaN if either empty |

idf = ln((1 + N) / (1 + df)) + 1, with df the number of the N pool records of the pair's
country holding the token. idf, frequency and ctx_idf read these partition-wide counts
(`STATS_GROUPS`): `pool_stats(pooln)` computes them once per country, `build_features`
hands them to every chunk and `pipeline.score` computes them once for all its chunks, so
no value depends on `chunk_rows`. Counting per country makes a mixed val fold give each
record the values of its own country, as the per-country test partitions do. Only pool
records are counted: the pool is complete in every setting, while S1 is sampled on the fit
side (the bias that keeps `ctx_pool_indegree` opt-in). Cost on 600k synthetic pairs: idf
0.33M pairs/s, frequency 0.32M pairs/s, address_extra ~0.65M pairs/s, against 0.07M
pairs/s for the v001 groups; `pool_stats` ~5 s per million pool records.
