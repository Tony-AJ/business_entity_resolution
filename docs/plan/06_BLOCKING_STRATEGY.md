# 06 — Blocking / candidate generation (module `blocking.py`, owner M2, plan group A)

What you build: `block(s1n, pooln, cfg) -> pairs` (schema in `02_SYSTEM_ARCHITECTURE.md`
§4.2), a union of cheap exact passes and TF-IDF top-k retrieval, run inside each country
partition. Targets on the val fold: **pair recall ≥ 0.97, mean candidates per S1 ≤ 40,
p95 ≤ 100, full val run ≤ 10 min, peak RAM ≤ 3 GB**. The candidate set you return is what
the organisers audit (`candidate_pairs.tsv`), so it must be deterministic.

## 1. Why multi-pass (measured, 01 §3)

| Single pass | Pair recall | Cands/S1 |
|---|---|---|
| exact `name_core` within country | 0.47 | 39 (p95 173, max 1461) |
| exact first token | 0.76 | 5,670 (unusable) |
| name char-gram top-30 (expected from Jaccard stats) | ≈ 0.85 | ≤ 30 |
| + address word top-20 | ≈ 0.97 | ≤ 50 before cap |

No single key reaches the target: 15% of true pairs share no name token, but ~90% of those
share address tokens. Hence a union of name channels and an address channel.

## 2. Passes (all inside one country partition)

| Bit | Pass | Key / text | Params | Purpose |
|---|---|---|---|---|
| 1 | P1a exact | `name_core` | groups > `exact_max_group=50` skipped (P2/P3 still see them) | cheap, exact |
| 2 | P1b exact | `name_sorted` | same cap | word-order swaps |
| 4 | P1c exact | `name_squash` | same cap; skip `""` | domain/handle/leet forms |
| 8 | P2 top-k | `name_core`, char_wb 3-grams, sublinear tf, `max_df=0.2`, `min_df=2` | k=30, min cos 0.30 | typos, abbreviations, transliterations |
| 16 | P3 top-k | `name_addr`, word unigrams, `max_df=0.05` | k=20, min cos 0.20 | renames and script names linked by the address; ambiguous names disambiguated by address tokens |
| 32 | P4 top-k (optional) | `addr_norm`, char_wb 3-grams | k=10, min cos 0.40 | enable only if recall after P1–P3 < 0.97 |

Union: `groupby(s1_idx, pool_idx)`: `pass = sum(bits)`, sims = max per pass. Cap: keep exact
pairs first, then the rest ordered by `max(sim_name_char, sim_name_addr_word, sim_addr_char)`,
`max_per_s1 = 60`. Pool records from other countries are never candidates.

`min_sim` is a floor, not a tuning knob for precision: the model decides. Lower it only if
recall is short and the candidate budget allows.

## 3. Algorithm

```
block(s1n, pooln, cfg, cache_dir, tag):
  for country in s1n.country.unique():           # whatever labels exist; France included
      if cached(tag, country, hash(cfg)): load; continue
      s1c, poolc = rows of that country (positional int32 index kept)
      parts = {}
      for key in cfg.exact_keys: parts[key] = exact_pass(s1c, poolc, key, cfg.exact_max_group)
      for name, spec in (("name_char", cfg.name_char), ("name_addr_word", cfg.name_addr_word), ("addr_char", cfg.addr_char)):
          if spec: parts[name] = topk_pass(s1c, poolc, spec, cfg.s1_chunk, cfg.pool_chunk, cfg.n_threads, SEED)
      pairs = union_passes(parts, cfg.max_per_s1)     # int32 positions → string ids, PAIR_COLUMNS
      save(cache); yield
  concat, sort by source1_entity_id (stable), assert unique pairs
```

`exact_pass`: `merge` on the key after dropping pool key groups with more than `max_group`
members (`value_counts`) and empty keys. `topk_pass`: the chunked `sparse_dot_topn` routine in
02 §7 (`sp_matmul_topn(A, B.T, top_n, threshold, sort=True, n_threads)`; `zip_sp_matmul_topn`
when the pool is split into chunks). Fit the vectoriser vocabulary on a 500k-row sample of
S1 ∪ pool of the same partition (unsupervised, refit per split; no leakage concern).

## 4. Budget (test India 810k × 4.72M; calibrate on 20k S1 rows first)

- P2: pool matrix ≈ 132M nnz (0.18 GB S1 side), 1.06 GB transposed float32/int32, 2.1 GB peak
  during transpose; 10–30 min on 12 threads. The cost is Σ over grams of df_S1 × df_pool, so
  it is driven by frequent grams: `max_df=0.2` removes them. If the 20k-chunk timing
  extrapolates past 30 min: `max_df=0.05`, or `ngram=(4,4)` (postings ≈ 10× shorter).
- P3: 52M nnz, 1–2 min. P1: < 1 min.
- US ≈ 0.8× India; France ≈ 0.1×. Val fold: P2 2–3 min, everything ≈ 5 min.
- Output: 30M pairs as int32 = 0.6 GB; string ids add ≈ 0.8 GB; write per country to Parquet.

## 5. Candidate recall evaluation (every A experiment)

```
truth = fold.pairs; cand = block(...)
blocking_report(cand, fold)  → pair_recall, entity_recall, ceiling_f_beta, candidates_mean/p95/max, reduction_ratio
per-pass recall: for bit in PASS_BITS: recall of pairs with (pass & bit) != 0
per-slice recall: non_latin pool name, domain-form names, ambiguous name_core (S1 group > 1), country
misses: truth pairs not in cand → error_samples(kind="missed") with both records side by side
```

`ceiling_f_beta` is the macro F0.5 a perfect matcher would reach on these candidates: the
number to report next to recall, because recall lost on 1-match entities costs more than on
6-match entities.

## 6. Experiments (A group, versions v010–v039 as assigned)

| Version | Change | Keep if |
|---|---|---|
| A1 | P1a only (baseline, also v001's blocking) | – (reference: recall ≈ 0.47) |
| A2 | + P2 name char-grams, k ∈ {20, 30, 50}, min_sim ∈ {0.2, 0.3} | recall ≥ 0.85, cands ≤ 35 |
| A3 | + P3 name+address words, k ∈ {10, 20}, max_df ∈ {0.05, 0.1} | recall ≥ 0.95 |
| A4 | + P1b/P1c keys; leet/squash variants; 4-grams vs 3-grams | recall +, time − |
| A5 | union tuning: `max_per_s1` ∈ {40, 60, 100}, `exact_max_group` ∈ {30, 50, 100}; P4 on/off | recall ≥ 0.97 at cands ≤ 40 |

Report for each: pair recall, entity recall, ceiling F0.5, cands mean/p95/max, seconds,
peak RSS. Keep the config with the best ceiling F0.5 subject to the candidate budget; a
recall gain that doubles the candidate count is not a win unless ceiling F0.5 rises.

## 7. Failure cases to inspect

1. Common-name groups (`meridian`, `summit`): P1 skips groups > 50; P3 must bring back the
   right member via address tokens. Check recall on the ambiguous slice specifically.
2. Non-Latin names: without R0 transliteration P2 finds nothing; with it, expect partial
   overlap. If the slice stays < 0.9, add the learned name token map (05 §6) or P4.
3. Renames (`Dovaflux`): only P3/P4 can find them; if the address is also short (`"Delhi"`),
   the pair is unreachable — accept and count it.
4. Empty addresses (4.4% of pool): P3 degenerates to name words; fine.
5. Chunk boundary effects: `zip_sp_matmul_topn` re-takes top-k across pool chunks; verify
   with a two-chunk vs one-chunk equality test on a small partition.
6. Determinism: fixed vocabulary sample seed, `sort=True`, stable sorts; assert identical
   output on two runs.

## 8. Tests (`tests/test_blocking.py`, synthetic fixture with `min_df=1, max_df=1.0`)

| Test | Assertion |
|---|---|
| `test_exact_pass_respects_max_group` | a key shared by `max_group + 1` pool rows produces no pairs; a smaller group produces all |
| `test_exact_pass_skips_empty_key` | `""` keys never match |
| `test_topk_pass_finds_typos` | 20 names + one-typo copies, k=3 → every true pair present, `0 < sim ≤ 1` |
| `test_topk_pass_chunking_invariant` | `pool_chunk` = half vs full → identical pairs and sims |
| `test_union_bitmask_and_max_sim` | a pair from P1a and P2 has `pass == 9` and the P2 sim; NaN for absent passes |
| `test_union_cap_prefers_exact_then_sim` | with `max_per_s1 = 2`, exact pairs survive, the lowest-sim pair is dropped |
| `test_block_partitions_by_country` | no pair joins records with different `country`; an unseen country ("France") still gets candidates |
| `test_block_output_schema` | `PAIR_COLUMNS`, dtypes, unique pairs, sorted by S1 |
| `test_to_id_lists_every_s1_present` | S1 without candidates → `[]`; lists have no duplicates |
| `test_conftest_truth_pairs_are_candidates` | on the synthetic dataset, `pair_recall == 1.0` |
| `test_block_cache_roundtrip` | second call with `cache_dir` returns an identical frame without recomputation |

## 9. Integration

- `features.py` consumes `pass` and the three `sim_*` columns by name and adds its own
  similarities; never compute features here.
- `pipeline.prepare` calls `block` once per split and caches by config hash; the test
  candidates are reused by every later model version with the same `BlockingConfig`.
- `candidate_pairs.tsv` is written from the `pairs` frame the model scored: if you add a
  post-filter, it belongs in `block` (before the cache), not in the notebook.
- New dependencies for this module, pinned in the commit that first imports them:
  `scikit-learn==1.9.1`, `scipy==1.18.1`, `sparse_dot_topn==1.2.0`.
