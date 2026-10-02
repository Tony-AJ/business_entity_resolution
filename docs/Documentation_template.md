# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** **TBD**  
**Team Members:** **TBD**  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

Per-country multi-pass blocking, a two-stage gradient-boosted matcher and a one-owner,
calibrated set decision tuned for macro F0.5. Random validation hid the test's density of
same-name decoys, so every change was ranked on a leaderboard-calibrated, test-shaped mock
fold. The decisive step was reading the test's uncertain pairs: the generator's decoy is a
second business with the S1 name plus or minus a word at a *nearby house number*, with its
own records. Stage-2 features for that signature took public F0.5 from 0.968 to **0.972**
(v126); the final version, v126 decoded at the false-merge weight the public uploads imply, is `submissions/v126_fp6/`.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Structure.** A Source 2/3 ("pool") record has at most one Source 1 (S1) owner; 5.6 % of
  S1 are singletons; S1 average 3.46 true matches (1.67 S2, 1.79 S3); France, 15 % of test
  S1, is absent from train.
- **Noise.** True-pair names agree for 4.6 % as written, 48.4 % after folding case, accents,
  punctuation and legal forms; 14.7 % share no token (acronyms, domains, handles, "X dba Y").
  Each source writes its own view of a business (one S2 view can carry a corrupted house
  number shared by all its S2 records).
- **Decoys and density.** ≈ 40 % of the test pool has no S1 owner (26 % in train). The
  test's candidates are far less certain than the mock's: candidates v122 scores between 0.2
  and 0.8, per S1, are 0.15 on the mock, 0.19 India, 0.37 US, 0.80 France.

### 2.2 Solution Strategy

- **Mock fold.** Per country, train is reshaped to the test's pool size and pool records per
  S1; dropped S1 leave their records as unowned decoys (40.2 % of the pool). All 1.37M S1
  compete in the one-owner step, as on test. Decisions are tuned on its tune entities and
  scored on its val entities.
- **Tight score.** The public score moved only with precision: splitting the mock loss into
  false-merge (L_FP) and missed-match (L_FN) parts, the public steps v122→v123→v126 imply the
  board charges a false merge about 6× a miss. Decoders are selected by `F − (w−1)·L_FP`.

**Approach Type:** blocking + classifier (two-stage gradient-boosted trees) with a calibrated,
metric-tuned set decision.  
**Core Innovation:** decoy-aware candidate-group features, read from the unlabelled test and
verified on a test-shaped mock fold.

---

## 3. Candidate Generation (Blocking)

Normalisation: `anyascii` transliteration; case, punctuation, domain and leet folding; legal
forms split off; a 536-token transliteration map learned from train-fold pairs; region codes
(French départements included).

- **Blocking keys used** (per country): exact core name, sorted tokens or alphanumerics (pool
  groups ≤ 200); exact core name + a shared address number (≤ 100); TF-IDF top-k on core-name
  char 3-grams (10, short-address pool records), name + address word 1–2-grams (50) and
  address words (10); at most 120 per S1, best cosine first.
- **Candidate pairs generated:** 69.3 per S1 blocked at mock density; a learned filter then
  keeps pairs with stage-1 probability p1 ≥ 0.01 among the entity's best 16, exactly
  `candidate_pairs.tsv`: **5.09 per test S1** (8.82M pairs: France 6.13, India 4.80, US 5.04).
- **How you ensured true matches were not lost:** address passes catch pairs sharing no name
  token; the cosine-ranked cap keeps same-name crowds from pushing out variants. Pair recall
  at mock density: 0.9829 blocked, 0.9788 after the filter (4.4 kept per S1).

---

## 4. Matching Model

**Features used** (no country feature):
- **Stage 1 (76 per pair):** rapidfuzz ratio, partial, token-sort, token-set, Jaro-Winkler,
  Levenshtein on full, core and squashed names; token Jaccard/Dice; legal-form agreement;
  address token-set/partial ratios, Jaccard, containment, region, number, postcode agreement;
  IDF-weighted overlaps; how many records share each name and address; blocking cosines.
- **Stage 2 adds, from p1 (20):** competition (rank, best rival, gap on the S1 side and the
  pool side; `pool_gap` carries 67 % of the gain), anchor (vs the entity's best other
  candidate), rival (vs the record's best rival S1).
- **Stage 2 adds, for decoys (v126, 19):** house-number relation of the first address numbers
  (equal, one contains the other = truncation, one-digit change, gap ≤ 20 = the decoy's
  neighbouring number, log gap); candidate groups of the same S1 (other candidates sharing the
  pair's number, their best p1, from the other source; candidates holding the S1's number;
  shared name, shared address); IDF of the name words only one side holds.

**Model type:** stage 1: XGBoost on the GPU (127 leaves) on the candidate pairs of 213k
train-fold entities absent from the mock (7.0M rows). Stage 2: XGBoost on the GPU (127
leaves), cross-fitted in two parts on the mock's fit + tune entities (4.4M kept pairs), each
entity scored by the model that never saw it; one seed, learning rate 0.1, early stopping
on each model's held-out part (598 and 528 trees). With stage 1 (396 trees), under 0.4M tree nodes in total.

**Threshold selection method:** each pool record stays only with its highest-probability S1;
probabilities are isotonic-calibrated on the mock's tune entities; each entity keeps the
prefix of its ranked candidates that maximises the expected tight score
`w·E[F0.5] − (w−1)·E[F0.5 without false positives]` (plug-in expectations, ≤ 11 matches).
(w, expected misses) are selected on tune entities by the tight score at the public-implied
weight: final w 8, 0.4 expected misses (selected at ×6; the ×3 selection gave w 4, 0.4).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** validation fold 0.9870 (single stage, v040); mock and public below.

| Version | Change | Mock F0.5 | Mock tight@3 | Public |
|---|---|---|---|---|
| v001 | 47 features, LightGBM, tuned rule | – | – | 0.954 |
| v103 | rule tuned on the mock | 0.9704 | – | 0.961 |
| v107 | two-stage, stage 2 trained on the mock | 0.9745 | – | 0.966 |
| v110 | + 23 features, denser blocking, GPU stage 1, rival features | 0.9817 | 0.9767† | 0.966 |
| v123 | v122 matcher, decoder weighted for false merges ×3 | 0.9823 | 0.9787 | 0.968 |
| v126 | + decoy columns (house number, candidate groups, unmatched IDF) | 0.9828 | 0.9802 | **0.972** |
| v127 | + stage 3 (competition and groups from stage-2 probabilities) | 0.9829 | 0.9807 | 0.971 |
| v128 | + group-fit columns, stage 3, ×6 decode | 0.9821 | 0.9804 | 0.972 |
| **v126 ×6** | **final**: v126's probabilities, decoder selected at ×6 | 0.9813 | 0.9796 | **TBD** |

† through v126's decoder. Mock false merges (val entities): 3,169 (v110), 1,904 (v123), 1,259 (v126), 839 (final).

- **Common false positives (wrong merges):** an unrelated name at the S1's address (a
  co-located business), the S1's name with an empty address, a differing unit number
  (`225 Apt 225` vs `225 Apt 234`), a twin at the same address with a legal word added.
- **Common false negatives (missed matches):** of 57.8k missed true pairs on the mock (v127),
  24.9k never become candidates (heavily noised names with empty or partial addresses,
  domains and handles), 10.7k are won by a rival S1 in the one-owner step (mostly
  empty-address records of same-name businesses), 22.2k stay below the rule.

---

## 6. Conclusion

Dense per-country blocking gives recall; a competition- and decoy-aware stage 2 and a
calibrated one-owner decision give precision. Two lessons: random validation hid a 0.03 drop
from denser decoys, and reading the unlabelled test's uncertain pairs found the decoy
signature no feature read (+0.004 public, 2.5× its mock gain). Next: address-key and
name-prefix blocking passes for the 2.1 % of true pairs never generated.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` is this repository: the tested library
`src/entity_resolution/`, one folder per experiment in `experiments/` (registry
`experiments.csv`, uploads `LEADERBOARD.md`); the final version's notebook and metrics are
also in `src/notebooks/`. Entry point (`README.md`; Python 3.12, 15 GB RAM, CUDA GPU):

```bash
make setup && make cache
# v110: blocking, stage 1 and the cached stage-1 outputs (mock + test)
make nb NB=experiments/v110_m3_features/v110_m3_features.ipynb
# final: stage 2 with the decoy columns, both decodes; the x6 arm is the final file
.venv/bin/python experiments/v126_decoy_groups/run_v126.py > experiments/v126_decoy_groups/run.log
cp submissions/v126_fp6/*.tsv output/
make validate
```

**Compliance.** Only the provided data: no external data, API, lookup or pretrained language
model; dictionaries are hand-typed or learned from train pairs. Models: gradient-boosted trees
from XGBoost 3.1.1 (Apache-2.0) and, up to v107, LightGBM 4.7.0 (MIT); under 0.4M tree nodes,
far below 8B parameters; no GPL package. Country is only a partition key (no filter, feature
or threshold): France, unseen in train, gets a row per entity.

### B. Additional Results

Mock F0.5 India / US: v110 0.9802 / 0.9832. Stage-2 gain shares (v126): `pool_gap` 0.67, p1
0.16, the 19 decoy columns 0.025 (best: best p1 of the number group 0.005, log house-number
gap 0.005). Public-weight decode (v127 on the mock val entities): arms selected at x3 / x4.5 /
x6 / x7.5 land within 0.0003 of each other in estimated public score.
