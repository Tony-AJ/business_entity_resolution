<!-- Draft of 2026-09-26 (day 2), updated with v103-v107. "TBD" marks numbers that do not exist
yet (the final version, v106 or later, and public scores after v107); they are filled on day 3
(TRACKER #42) from metrics.json, experiments.csv and LEADERBOARD.md. Remove before submission. -->

# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TEAM NAME]  
**Team Members:** [MEMBERS]  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We combine per-country multi-pass blocking, a two-stage gradient-boosted matcher and a
one-owner set decision tuned for macro F0.5. The key finding is that a random validation fold
under-represents the test's density of same-name decoys (validation 0.9858, public 0.955). We
therefore score, train and tune on a test-shaped mock fold calibrated on the leaderboard: the
rule tuned there reached public 0.961, and a stage-2 matcher with competition features reached
public 0.966, as the calibrated mock predicted (0.9659; final version TBD).

---

## 2. Methodology

### 2.1 Problem Analysis

- **Structure.** Train: 2.21M Source 1 (S1) entities, 10.32M Source 2/3 ("pool") records,
  7.64M true pairs; test: 1.73M S1 (India 47 %, US 38 %, France 15 %, unseen in train), 9.97M
  pool records. A pool record has at most one S1 owner; matched entities average 3.7 matches;
  5.6 % of S1 are singletons; 26 % of the train pool is unowned against ≈ 40 % on test. True
  pairs always share the country.
- **Names.** True-pair names agree for 4.6 % as written, 48.4 % after folding case, accents,
  punctuation and legal forms; 14.7 % share no token (Indic scripts, domain forms, renames).
  48 % of S1 core names recur in S1 and 62 % of singletons have a same-name pool record: the
  address separates a match from a decoy.
- **Density (the key finding).** Validation scores 441k S1 against a 2.06M pool (20 % of
  train), test 1.73M S1 against 9.97M: a test entity meets 3× (US) to 6× (India) more
  same-name records, and both day-1 uploads lost 0.030 from validation to public.
- **Addresses and metric.** 4.4 % of pool addresses are empty; postcodes are rare, but 78–83 %
  of true pairs share a number. A false merge costs as much as missing half of an entity's
  matches; any prediction for a singleton scores 0.

### 2.2 Solution Strategy

Pipeline: normalise → block per country → stage 1 and learned filter → stage 2 → one-to-one
set rule → both TSVs, vectorised and chunked (peak RAM 9.4 GB). A fixed hashed split holds out
20 % of S1 (seed 42); an inner split (25 %, seed 4242) supplies tune entities.

**Mock-test fold** (`mock.py`). Per country, train is reshaped to the test's pool size and pool
records per S1 (US 3.82M, 5.76; India all 4.13M, 5.82): clusters (an S1 entity with its
records) are kept by id hash, then S1 entities are dropped, stage 1's training sample first,
leaving their records as unowned decoys (40.2 % of the pool; test ≈ 40 %). All 1.37M present
S1 are scored and compete in the one-to-one: validation-fold entities give mock F0.5, tune
entities set early stopping and the rule, fit entities train stage 2.

**Calibration.** v101 scores 0.9858 on validation, 0.9677 on the mock and 0.955 public: the
mock reproduces 60 % of the drop (India's test pool is 14 % larger than all of train's; France
is unseen). v103, the same matcher with a mock-tuned rule, gained 0.006 public where the mock
predicted 0.0027. Splitting the mock loss into L_FP (F0.5 lost to wrong pairs) and L_FN (the
rest), public ≈ 1 − L_FN − 1.45·L_FP − 0.0072 fits both uploads. From v107 on, rules are tuned
on this *tight mock* and versions compared by its estimate, est_public.

**Approach Type:** Blocking + Classifier (two-stage gradient-boosted trees) with a metric-tuned
set decision.  
**Core Innovation:** working at the test's decoy density: a leaderboard-calibrated,
test-shaped mock fold decides every change and trains a stage-2 matcher that compares each pair
with its entity's other candidates and the pool record's rival entities.

---

## 3. Candidate Generation (Blocking)

Keys are built on normalised records: `anyascii` transliteration; case, punctuation,
domain-suffix and leet folding; legal forms split off; a 536-token transliteration map learned
from train-fold pairs; one code per region in any written form (from v106, French départements
also map to their region: 27 % of French test address components); unified street types,
ordinals and city names. Passes run per country; TF-IDF uses sparse top-k (`sparse_dot_topn`).

- **Blocking keys used** (validation recall of each V1 pass in brackets; B6 from v106):

| Pass | Key | V1 (v001–v107) | B6 |
|---|---|---|---|
| P1a/b/c exact | core name / sorted tokens / alphanumerics | pool groups ≤ 50 (0.565 / 0.596 / 0.594) | ≤ 200 |
| P2 top-k | core-name char 3-grams, pool records with ≤ 3 address tokens | k 10, cos ≥ 0.5 (0.046) | as V1 |
| P3 top-k | name + address word 1–2-grams | k 25, cos ≥ 0.2, df ≤ 1 % (0.983) | k 50 |
| P4 top-k | address word 1–2-grams | off | k 10, cos ≥ 0.3 |
| Cap | per S1 | 60, exact pairs first | 100, by best cosine |

  At test density the cap lost recall: ranked exact-first, a common name's same-name records
  fill it and push out true variants (groups ≤ 200 cut recall from 0.9768 to 0.9551). Ranked
  by cosine, B6 lifts recall at mock density from 0.9677 to 0.9780 (misses −32 %; v105, 40k
  sampled mock validation S1).
- **Learned last stage.** From v104 a pair stays only if stage 1 gives it p1 ≥ 0.01 and ranks
  it in its entity's top 16: 4.6 of 34.5 candidates per S1 on the mock, at −0.0011 recall.
  `candidate_pairs.tsv` holds exactly these pairs.
- **Candidate pairs generated:** validation 14.55M (33.0 per S1); test 60.52M from V1 blocking
  (34.9 per S1, 3.5 × 10⁻⁶ of all S1 × pool pairs), 4.8 (India) to 6.0 (France) per S1 after
  the filter (file 802 MB → 136 MB; 1.2 % of entities keep none). B6 blocks 1.9× more pairs
  before the filter; final counts TBD.
- **How you ensured true matches were not lost:** the address channels recover pairs sharing
  no name token and members of name groups too large for P1. Every version reports pair
  recall, entity recall and the ceiling F0.5 of a perfect matcher, on validation and, since
  v103, on the mock (V1: 0.9906 / 0.9993 / 0.9971 vs 0.9655 / 0.9965 / 0.9879). Of v104's
  mock true pairs, 3.45 % are never candidates, 0.11 % are filtered out, 0.94 % go to a rival
  S1 in the one-to-one and 1.96 % fall below the rule.

---

## 4. Matching Model

**Features used** (53 per pair, float32, NaN where a field is empty; no country feature):
- Name features (21): rapidfuzz ratio, partial, token-sort, token-set, Jaro-Winkler and
  Levenshtein on the full, core and squashed name; token Jaccard, Dice and counts;
  first-token, sorted-key, prefix and legal-form agreement.
- Address features (12): token-set, partial and plain ratio; token Jaccard and containment;
  region and last-token equality; empty pool address; number-set Jaccard, shared number,
  house-number and postcode equality.
- Other (20): blocking flags and cosines; rank and gap to the entity's best candidate;
  candidate count; source; transliteration flag; name-length ratio; core-name frequencies and
  equality.
- Stage 2 only, from the stage-1 probabilities p1: 12 competition features over every pair of
  the country partition (p1; per S1 entity and per pool record: rank, best rival, gap, p1 sum,
  likely count; the record's degree), 5 anchor features (the candidate against its entity's
  best other candidate: one business's records resemble each other, a decoy does not) and,
  from v106, 3 rival features (the record against its best rival S1's name and address).
- M3's groups (23; v042 adds them to stage 2 on the kept pairs, v043 to stage 1):
  IDF-weighted name and address agreement for every pair (cosine, rarest shared token,
  coverage per side; document frequencies counted per country over the pool), pool counts of
  the exact name and address (decoy risk), ranks of the IDF cosines among the entity's
  candidates and its number of exact-name candidates, reverse address and house-number
  containment, postcode prefix and address-length ratio. On the mock they raise est_public by
  0.0020 (v042: false merges −22 %); in stage 1 they carry 14 % of the gain (v043).

**Model type:** Stage 1 is v101's LightGBM (63 leaves, learning rate 0.05, 1,666 rounds by
early stopping) on 6.72M candidate pairs of 200k sampled fit-side entities. Its successor
(v106, running; TBD) is XGBoost on the GPU trained on up to 360k train-fold entities absent
from the mock, their ~20M pairs streamed from disk into a GPU `QuantileDMatrix`. Stage 2 is
XGBoost on an RTX 2050 (63 leaves, learning rate 0.05, early stopping on 50k mock tune
entities) on the 3.19M filtered pairs of the mock's fit entities, none seen by stage 1,
cross-fitted in two id-hash parts so fit entities are scored out of fold. `pool_gap` (p1 minus
the record's best rival) carries 72 % of its gain. Ablations (mock F0.5 0.9744): without
competition and anchor features 0.9726, still 0.0022 above v103 thanks to training at
density; without anchors 0.9742.

**Threshold selection method:** each pool record stays only with its highest-probability S1
entity. A threshold rule then keeps an entity's pairs with p ≥ τ_abs and p ≥ τ_rel · p_max, at
most K, and nothing unless p_max ≥ τ_single (3,906-rule grid). Expected-F0.5 decoding instead
keeps the ranked prefix of highest expected F0.5, reading q = p^γ as the chance a pair is true
and allowing m true matches missing from the candidates. Rules are scored by exact macro F0.5
over all tune entities, singletons and blocking misses included; ties go to the most
conservative. Rules were tuned on the inner tune split up to v101, on the mock tune entities
for v103 (τ_abs 0.725, τ_rel 0.7, τ_single 0.775) and v104, and for the tight score from v107,
where expected-F0.5 decoding (γ 1.5, m 0.05, K 11) beat the best threshold rule by under 10⁻⁵.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** validation 0.9858 (v101); mock 0.9745, est_public 0.9659 and
  public 0.966 (v107); final version TBD.

| Version | Change | Val F0.5 | Mock F0.5 | est_public | Public F0.5 |
|---|---|---|---|---|---|
| v001 | V1 pipeline, 47 features | 0.9844 | – | – | 0.954 |
| v101 | + core-name frequencies (53 features) | 0.9858 | 0.9677 | 0.955 (fit) | 0.955 |
| v103 | v101 matcher, rule tuned on the mock | 0.9833 | 0.9704 | 0.961 (fit) | 0.961 |
| v104 | two-stage: filter, stage 2 trained on the mock | – | 0.9744 | 0.9654 | – |
| v107 | v104, rule tuned for the tight mock | – | 0.9745 | 0.9659 | 0.966 |
| v040 | v001 + M3's four groups (70 features, single stage; M3) | 0.9870 | – | – | – |
| v042 | v104 two-stage + M3's groups in stage 2 (M3) | – | 0.9762 | 0.9679 | – |
| v043 | stage 1 = v101 + M3's groups; two-stage (M3) | 0.9876 (stage 1) | 0.9762 | 0.9678 | – |
| v044 | v042 + interaction and missing-number flags in stage 2 (M3) | – | 0.9765 | 0.9681 | – |
| final | TBD | TBD | TBD | TBD | TBD |

Validation could not rank what mattered (v101: false merges −29 %, public +0.001; v103:
validation −0.0025, public +0.006). The tight mock is fitted on these two uploads; v107's
public score, its first out-of-sample check, landed within 0.0001 of the estimate. The two-stage gain is larger in India, where
decoys are densest (mock 0.9651 → 0.9700; US 0.9762 → 0.9792).

- **Common false positives (wrong merges):** same-name decoys that the address cannot
  separate: an identical name with an empty pool address (`Classic Suisse LLC`); the same name
  and street with a house number a few apart (`Osprey Group, 231 Silvermine Avenue` vs
  `232 Silvermine Ave`, a true singleton); one word changed at the same address
  (`Cornerstone Medicals` vs `Cornerstone Mega Private Limited`, p = 0.997). False-merge
  pairs on the mock: 15.9k with v101's rule, 3.5k with v107.
- **Common false negatives (missed matches):** misses rose in exchange (64k → 74k). Never
  retrieved: Indic-script names (`Baba Food` ↔ बाबा फूड), domain forms (`Kolkata Service` ↔
  `kolkataservice.com`), truncations (`Ranjit & Brothers Corporation` ↔ `Ranjit &`) and close
  names of common businesses beside a partial address. Others are lost to a rival S1 in the
  one-to-one (0.94 % of true pairs) or are renames at the same address scored below the rule
  (`Capital Alphabet` ↔ `Calovera Labs`).

---

## 6. Conclusion

Recall comes from per-country name and name + address blocking with a similarity-ranked cap
(97.8 % of true pairs at test density), precision from the competition-aware stage 2 and a
one-owner set decision tuned on the leaderboard-calibrated metric. The main lesson: a random
20 % validation fold hid a 0.03 drop caused by denser same-name decoys; a test-shaped mock
fold then gained 0.006 public through the rule and an estimated 0.005 more through the
two-stage matcher (final TBD).

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
| `split`, `trainset`, `mock` | fixed validation fold; inner split, S1 sampling, labels; mock fold, tight-mock constants |
| `metrics`, `evaluate` | reference macro F0.5; vectorised bit-equal twin, tight score, blocking report, errors |
| `token_maps`, `normalize` | hand-typed legal-form, street and region maps; normalisation, learned token map |
| `blocking` | per-country exact keys and TF-IDF top-k passes, union, exact- or sim-first cap, cache |
| `features` | pair-feature registry (rapidfuzz, sparse token overlaps), chunked by S1 group |
| `model` | `Matcher`: LightGBM, XGBoost (CUDA), logistic, heuristic; fixed column contract |
| `decision` | one-to-one set rule, exact macro-F0.5 grid tuning, expected-F0.5 decoding |
| `stacking`, `twostage`, `stage1` | competition, anchor, cohesion, rival features; filter, cross-fitted stage 2, test run; GPU stage 1 from disk |
| `pipeline` | `fit`, `run_fold`, `run_mock`, `tune_mock`, `run_test` |
| `submission`, `tracking` | writing and validating both TSVs; experiment registry, timings |

Reproduction of v107 (upload #4; the final version's entry point is TBD): Python 3.12, the
organisers' zip in `dataset/`, 12 CPU threads, 15 GB RAM; XGBoost stages use a CUDA GPU (an
RTX 2050 here) or `device="cpu"`.

```bash
make setup     # .venv from the pinned requirements.txt
make cache     # every TSV to Parquet once (~1 min)
make nb NB=experiments/v101_name_frequency/v101_name_frequency.ipynb   # stage 1
make nb NB=experiments/v104_two_stage/v104_two_stage.ipynb             # mock fold, filter, stage 2
make nb NB=experiments/v107_tight_rule/v107_tight_rule.ipynb           # tight-mock rule, test run
make validate  # our checker and the organisers' validator on output/
```

`output/matching_results.tsv` and `output/candidate_pairs.tsv` (1,732,544 rows each) are
written by `submission.write_pairs` from the pairs frame stage 2 scored; models, rules and
configuration go to `experiments/<version>/artifacts/`, caches to `dataset/.cache/`. Seeds: 42
(split, LightGBM, vocabulary sample), 4242 (inner split), 7 (S1 samples), 99 (harder fold),
5151 / 5152 (mock), 6161 (cross-fitting), 7171 (GPU stage-1 early stopping); LightGBM runs
deterministically. Run times: v001 27 min to fit, 51 min for test inference with blocking;
v104 15 min for stage 1 over the mock, 2 × 5 min for stage 2, 21 min for the test run from
cached blocking.

**Compliance.** No external data, lookups or network calls: every dictionary is hand-typed or
learned from the provided train pairs, and no pretrained language model is used. Libraries:
LightGBM 4.7.0 (MIT), XGBoost 3.1.1 (Apache-2.0), rapidfuzz 3.14.6 (MIT), sparse-dot-topn
1.2.0 (Apache-2.0), anyascii 0.3.3 (ISC), scikit-learn 1.9.1 (BSD-3), NumPy, pandas, SciPy
(BSD), pyarrow (Apache-2.0); GPL packages such as `unidecode` and `python-Levenshtein` are
excluded. Model size: stage 1 has 1,666 trees of ≤ 63 leaves (≈ 0.2M parameters; v106's GPU
stage 1 at most 3,000 trees of ≤ 127 leaves, ≤ 0.8M), stage 2 at most 2 × 4,000 trees of
≤ 63 leaves (≤ 1M; v107: 1,813 + 1,645). Country is only a partition key: no country filter,
feature or threshold, and France gets rows like any other country.

### B. Additional Results

Settings: the learned token map aligns Indic-script pool names by position with their S1 names
and keeps a token seen ≥ 3 times with a ≥ 50 % consistent partner; the mock keeps 61.7 % of US
clusters and all Indian ones (340k validation, 340k tune, 693k fit entities); the rule grid
spans τ_abs 0.30–0.90, τ_rel {0, 0.5–0.95}, τ_single − τ_abs {0–0.30}, K {4, 6, 11}; the
expected-F0.5 grid γ 0.7–2 and m 0–0.4; stage 2 scores non-fit entities with the mean of its
two models.

Blocking at mock density (v105; pair recall at candidates per S1): V1 0.9677 at 34.3; word
top-50 and cap 100: 0.9768 at 58.1; plus exact groups ≤ 200: 0.9551 at 64.7; plus the
address-word pass: 0.9554 at 65.5; the same with the sim-first cap (B6): 0.9780 at 65.5
(India 0.9688, US 0.9870); B6 with cap 80: 0.9773; the sim-first cap alone on V1: 0.9678.

Mock F0.5, India / US: v101 0.9610 / 0.9750, v103 0.9651 / 0.9762, v104 0.9700 / 0.9792,
v107 0.9701 / 0.9792. False-merge / missed pairs on the mock validation entities: v101's rule
15.9k / 64k, v103 6.3k / 79k, v104 5.3k / 68k, v107 3.5k / 74k. Stage-2 gain shares (v104):
`pool_gap` 0.72, `p1` 0.19, `s1_gap` 0.014; no anchor feature above 0.002.

Validation slices (v101): ambiguous core name (34 % of entities) 0.9783 vs 0.9898; one true
match 0.9541; empty address on either side 0.9594 vs 0.9902. Stage-1 gain shares:
`ad_token_set` 0.40, `sim_name_addr_word` 0.16.

Test output (v107): candidates per S1 France 6.04, India 4.84, US 4.99; share of S1 with a
match France 0.945, India 0.937, US 0.941; 6.0 % of entities predicted empty against 5.6 %
singletons in train. France, unseen in training, behaves like the train countries.

TBD (day 3): v106 (the GPU stage 1 alone and under stage 2, the rival-feature ablation, B6 and
filter recall on the whole mock); v109 (an exact key on core name plus a shared address number
for B6's remaining close-name misses); v107's public score; the final version.
