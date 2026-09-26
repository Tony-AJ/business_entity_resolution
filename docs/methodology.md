<!-- Draft of 2026-09-26 (day 2). "TBD" marks numbers that come from the final version's
metrics.json, experiments.csv and LEADERBOARD.md on day 3 (TRACKER #42); every other number
is logged under experiments/. Remove this comment before submission. -->

# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TEAM NAME]  
**Team Members:** [MEMBERS]  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We combine blocking, pairwise classification and a set decision tuned for the exact metric:
per-country exact-key and TF-IDF blocking (validation pair recall 0.9906 at 33 candidates per
entity), LightGBM on 53 pair features, and a pool-side one-to-one rule with thresholds tuned
for macro F0.5 (validation 0.9858, public 0.955). Because test entities meet 3–6× more
same-name decoys than validation entities, we built a test-shaped mock fold and trained a
stage-2 XGBoost matcher with competition features on it (mock F0.5 TBD, public TBD).

---

## 2. Methodology

### 2.1 Problem Analysis

- **Structure.** Train: 2.21M Source 1 (S1) entities, 10.32M Source 2/3 ("pool") records,
  7.64M true pairs; test: 1.73M S1 (India 47 %, US 38 %, France 15 %, unseen in train), 9.97M
  pool records. A pool record has at most one S1 owner; matched entities average 3.7 matches;
  5.6 % of S1 are singletons; 26 % of the train pool is unowned against ≈ 40 % on test. True
  pairs always share the country.
- **Names.** True-pair names agree for 4.6 % as written, 25.6 % after case, accent and
  punctuation folding, 48.4 % without legal forms; 14.7 % share no token (Indic scripts,
  domain or handle forms, renames).
- **Ambiguity.** 48 % of S1 core names recur in S1 and 62 % of singletons have a same-name
  pool record: the address separates a match from a decoy.
- **Addresses.** 4.4 % of pool addresses are empty; postcodes are rare, but 78–83 % of true
  pairs share a number.
- **Metric.** A false merge costs as much as missing half of an entity's matches; any
  prediction for a singleton scores 0.

### 2.2 Solution Strategy

Pipeline: normalise → block per country → pair features → boosted matcher → one-to-one set
rule → both TSVs, vectorised and chunked (peak RAM 6.6 GB). Models are fitted on the train
fold of a fixed hashed split (20 % of S1 held out, seed 42), with an inner tune split (25 %,
seed 4242) for early stopping and thresholds.

**Approach Type:** Blocking + Classifier (two-stage gradient-boosted trees) with a
metric-tuned set-decision layer.  
**Core Innovation:** evaluating and training at the test's decoy density: a test-shaped mock
fold, and a stage-2 matcher trained on it that reads stage-1 probabilities as a learned final
blocking filter and as competition features.

Validation 0.9844 / 0.9858 became 0.954 / 0.955 public (v001 / v101). Validation scores 441k
S1 against a 2.06M pool, test 1.73M against 10.0M, so a test entity meets 3× (US) to 6×
(India) more same-name records; dropping 20 % of validation S1 but keeping the pool cost only
0.0005–0.0006. The mock fold (`mock.py`) rebuilds the test's shape per country from all of
train: clusters (an S1 entity with its records) and unowned records are kept by id hash at the
ratio test pool / train pool; S1 entities are dropped, stage 1's training sample first, until
pool records per S1 match the test (US 5.76, India 5.82), their records staying as decoys.
Result: 1.37M S1, 7.95M pool records, 40.2 % unowned (test ≈ 40 %), every entity competing in
the one-to-one. Validation entities are only scored, tune entities set early stopping and the
rule, other fit-side entities train stage 2.

---

## 3. Candidate Generation (Blocking)

Keys are built on normalised records: `anyascii` transliteration; folding of case,
punctuation, domain suffixes and leet digits; legal forms in their own column; a 536-token
transliteration map learned from train-fold true pairs; one code per region, whether written
as name, code or native script (US, India, France); unified street types, ordinals and city
names. Passes run inside each country value present; TF-IDF retrieval uses multi-threaded
sparse top-k (`sparse_dot_topn`).

- **Blocking keys used** (validation recall of each pass; passes overlap):

| Pass | Key | Parameters | Recall | Pairs/S1 |
|---|---|---|---|---|
| P1a/b/c exact | core name / sorted tokens / alphanumerics | pool groups ≤ 50 | 0.565 / 0.596 / 0.594 | 7.2–7.3 each |
| P2 top-k | core-name char 3-grams, pool records with ≤ 3 address tokens | k 10, cos ≥ 0.5 | 0.046 | 7.4 |
| P3 top-k | core name + address, word uni- and bigrams | k 25, cos ≥ 0.2, df ≤ 1 % | 0.983 | 21.6 |

  The union is capped at 60 per S1, exact pairs first. The two-stage version then keeps a pair
  only if its stage-1 probability p1 ≥ 0.01 and it ranks in its entity's top 16;
  `candidate_pairs.tsv` holds exactly these pairs.
- **Candidate pairs generated:** validation 14.55M (33.0 per S1); test 60.52M (34.9 per S1),
  3.5 × 10⁻⁶ of all S1 × pool pairs, every test entity with candidates; after the filter TBD
  (target ≈ 10 per S1).
- **How you ensured true matches were not lost:** the address channel P3 recovers pairs
  sharing no name token; members of common-name groups, which P1 skips above 50, are still
  retrieved by P3 through address words and bigrams; every version reports pair recall,
  entity recall and the ceiling F0.5 of a perfect matcher (validation 0.9906, 0.9993, 0.9971:
  blocking costs at most 0.003 F0.5). Recall at test density is re-measured on the mock (TBD;
  larger budgets in study v105), and the filter must lose ≤ 0.002 recall (TBD).

---

## 4. Matching Model

**Features used** (53 per pair, float32, NaN where a field is empty; no country feature):
- Name features (21): ten rapidfuzz scores (ratio, partial, token-sort, token-set,
  Jaro-Winkler, Levenshtein) across the full, core and squashed name; token Jaccard, Dice and
  counts; first-token, sorted-key, prefix and legal-form agreement.
- Address features (12): token-set, partial and plain ratio; token Jaccard and containment;
  region and last-token equality; empty pool address; number-set Jaccard, shared number,
  house-number and postcode equality.
- Other (20): blocking flags and cosines; rank and gap to the entity's best candidate;
  candidate count; source; transliteration flag; name-length ratio; core-name frequencies and
  equality. Stage 2 adds 12 competition features from p1 over the whole partition (p1; per S1
  entity and per pool record: rank, best rival, gap, p1 sum, likely count; the record's
  degree) and 5 anchor features comparing each candidate with its entity's best other
  candidate: one business's records resemble each other, a same-name decoy does not.

**Model type:** Stage 1 is LightGBM (63 leaves, learning rate 0.05, deterministic, 1,666
rounds by early stopping) on 6.72M candidate pairs of 200k sampled fit-side entities (tune
AUC 0.99986); sampling entities, not pairs, keeps every decoy. Stage 2 is XGBoost on CUDA
(v104: 63 leaves, ≤ 4,000 rounds; final settings TBD) on the filtered candidates of the
mock's fit entities, none seen by stage 1, cross-fitted in two parts by id hash so every fit
entity is scored out of fold.

**Threshold selection method:** each pool record stays only with its highest-probability S1
entity (the one-owner property); an entity keeps pairs with p ≥ τ_abs and p ≥ τ_rel · p_max,
at most K, and nothing unless p_max ≥ τ_single. A 3,906-rule grid is scored by exact macro
F0.5 over all tune entities, singletons and blocking misses included; ties go to the most
conservative rule. It was tuned on the inner tune split for v001 / v101 (τ_abs 0.47 / 0.42,
τ_rel 0, τ_single 0.52, K 11) and on the mock tune entities from v103 on. Expected-F0.5 set
decoding is implemented as an alternative (adopted: TBD).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** validation 0.9858 (v101; singletons 0.9874, harder variant
  0.9852), public 0.955; final version: mock TBD, public TBD.

| Version | Change | Val F0.5 | Mock F0.5 | Cand. recall val / mock | Public F0.5 |
|---|---|---|---|---|---|
| v001 | V1 pipeline, 47 features | 0.9844 | not scored | 0.9906 / – | 0.954 |
| v101 | + core-name frequencies | 0.9858 | TBD | 0.9906 / TBD | 0.955 |
| v103 | v101 matcher, rule tuned on the mock | TBD | TBD | 0.9906 / TBD | TBD |
| v104 | two-stage: top-16 filter + stage 2 on the mock | – | TBD | – / TBD | TBD |
| final | TBD | TBD | TBD | TBD | TBD |

v101 cut validation false merges by 29 % but gained only 0.001 public, so mock F0.5 decides
KEEP / DROP from v103 on (v101 on the mock against its public 0.955: TBD).

- **Common false positives (wrong merges):** validation loss is mostly recall (v101: 5,076
  wrong pairs, 319 on true singletons, against 51,267 missed). Wrong merges are near-duplicate
  decoys: same name and street, house number a few apart
  (`Optimal Mobility Group, 11 Wildwood Drive` vs `… LLC, 14 Wildwood Dr`); same-name records
  with an empty address; a name extended by a generic token at the same address
  (`Visoft Capital LLC` vs `Visoft Capital Partners`). They multiply at test density (mock
  counts TBD).
- **Common false negatives (missed matches):** records with an empty or city-only address
  (empty-address slice: pair recall 0.897 vs 0.980); renames at an identical address
  (`Delta Homecare` ↔ `Umbraectoorbi`, p = 0.12); Indic-script names beside a partial
  address; blocking misses (0.94 % of true pairs). Weakest slices: ambiguous core names
  (0.978 vs 0.990) and single-match entities (0.954).

---

## 6. Conclusion

Recall comes from blocking (name and name+address channels per country keep 99.1 % of true
pairs at 33 candidates per entity), precision from the decision layer (one owner per pool
record, thresholds tuned on the exact metric). The main lesson: a random 20 % validation fold
hid a 0.03 drop caused by denser same-name decoys, so scoring and the final matcher moved to
a test-shaped mock fold (gain TBD).

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` is this repository: the tested library
`src/entity_resolution/`, one documented notebook per experiment in `experiments/vNNN_<slug>/`
(committed with outputs and `metrics.json`; registry `experiments/experiments.csv`; uploads in
`LEADERBOARD.md`) and pytest on synthetic data in `tests/`. The final notebook is also copied
to `src/notebooks/`.

| Modules | Role |
|---|---|
| `config`, `data` | paths, schema, seeds; TSV loaders, Parquet cache, Arrow id filters |
| `split`, `trainset`, `mock` | fixed validation fold; inner fit/tune split, S1 sampling, labels; mock fold |
| `metrics`, `evaluate` | reference macro F0.5; vectorised bit-equal twin, blocking report, slices, errors |
| `token_maps`, `normalize` | hand-typed legal-form, street and region maps; normalisation, learned token map |
| `blocking` | per-country exact keys and TF-IDF top-k passes, union, cap, cache |
| `features` | pair-feature registry (rapidfuzz, sparse token overlaps), chunked by S1 group |
| `model` | `Matcher`: LightGBM, XGBoost (CUDA), logistic, heuristic; fixed column contract |
| `decision` | one-to-one set rule, exact macro-F0.5 grid tuning, expected-F0.5 decoding |
| `stacking`, `twostage` | competition and anchor features; stage-1 filter, cross-fitted stage 2, test run |
| `pipeline` | `fit`, `run_fold`, `run_mock`, `tune_mock`, `run_test` |
| `submission`, `tracking` | writing and validating both TSVs; experiment registry, timings |

Reproduction (Python 3.12, organisers' zip unzipped into `dataset/`, 12 CPU threads, 15 GB
RAM; stage 2 trains on a CUDA GPU, an RTX 2050 here, or on the CPU with `device="cpu"`):

```bash
make setup     # .venv from the pinned requirements.txt
make cache     # every TSV to Parquet once (~1 min)
make nb NB=experiments/v101_name_frequency/v101_name_frequency.ipynb   # stage 1: fit, validation, artifacts
make nb NB=experiments/<final>/<final>.ipynb                           # TBD: mock fold, stage 2, rule, test run
make validate  # our checker and the organisers' validator on output/
```

`output/matching_results.tsv` and `output/candidate_pairs.tsv` (1,732,544 rows each) are
written by `submission.write_pairs` from the pairs frame stage 2 scored. Models, rules, token
map and configuration go to `experiments/<version>/artifacts/`, caches to `dataset/.cache/`.
Seeds: 42 (split, LightGBM, vocabulary sample), 4242 (inner split), 7 (S1 samples), 99
(harder fold), 5151 / 5152 (mock), 6161 (cross-fitting); LightGBM runs in deterministic mode.
v001 took 27 min to fit, 12 min on validation and 51 min for test inference with blocking.

**Compliance.** No external data, lookups or network calls: every dictionary is hand-typed or
learned from the provided train pairs, and no pretrained language model is used. Libraries:
LightGBM 4.7.0 (MIT), XGBoost 3.1.1 (Apache-2.0), rapidfuzz 3.14.6 (MIT), sparse-dot-topn
1.2.0 (Apache-2.0), anyascii 0.3.3 (ISC), scikit-learn 1.9.1 (BSD-3), NumPy, pandas, SciPy
(BSD), pyarrow (Apache-2.0); GPL packages such as `unidecode` and `python-Levenshtein` are
excluded. Model size: stage 1 has 1,666 trees of ≤ 63 leaves (≈ 0.2M parameters), stage 2 at
most 2 × 4,000 such trees (≤ 1M). Country is only a partition key: no country filter, feature
or threshold, and France gets rows like any other country.

### B. Additional Results

Settings not listed above: the learned token map aligns Indic-script pool names by position
with their S1 names and keeps a token seen ≥ 3 times with a ≥ 50 % consistent partner; the
mock keeps 61.7 % of US clusters and all Indian ones; the rule grid spans τ_abs 0.30–0.90
(step 0.02), τ_rel {0, 0.5–0.95}, τ_single − τ_abs {0–0.30}, K {4, 6, 11}, then refines τ_abs
by 0.005 (15 s for 441k entities); stage 2 scores tune, validation and test entities with the
mean of its two models.

Validation slices (v101, macro F0.5): India 0.9850, US 0.9864; entities with an Indic-script
true record 0.9863, with a domain-form true record 0.9879; ambiguous core name (34 % of
entities) 0.9783 vs 0.9898; true singletons 0.9874; by true match count 1: 0.9541, 2: 0.9832,
3–4: 0.9881, 5+: 0.9898; empty address on either side 0.9594 vs 0.9902. Stage-1 gain shares:
`ad_token_set` 0.40, `sim_name_addr_word` 0.16, `core_token_set` 0.061, `ctx_rank_addr`
0.055, `num_jaccard` 0.052, `ad_jaccard` 0.038.

Test output (v001): candidates per S1 France 37.0, India 35.0, US 34.1; share of S1 with at
least one match France 0.950, India 0.941, US 0.943 (validation 0.942); 5.7 % of test entities
predicted empty against 5.6 % singletons in train. France, unseen in training, behaves like
the train countries.

TBD: v101 on the mock against its public 0.955; mock blocking recall per budget (v105: word
top-k 25 → 50, cap 60 → 100, exact groups 50 → 200, address char-gram pass); stage-1 filter
recall; v104 ablations (without anchor features; pair features only); stage-2 feature
importance.
