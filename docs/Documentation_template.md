# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** **TBD**  
**Team Members:** **TBD**  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

Per-country multi-pass blocking, a two-stage gradient-boosted matcher and a one-owner set
decision tuned for macro F0.5. Random validation hid the test's density of same-name decoys,
so later versions were ranked on a leaderboard-calibrated, test-shaped mock fold: public F0.5
rose from 0.954 to 0.966 (v107); v110 is estimated at 0.973; final v120: **TBD**.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Structure.** A Source 2/3 ("pool") record has at most one Source 1 (S1) owner; 5.6 % of
  S1 are singletons; France, 15 % of test S1, is absent from train.
- **Noise.** True-pair names agree for 4.6 % as written, 48.4 % after folding case, accents,
  punctuation and legal forms; 14.7 % share no token. 48 % of S1 core names recur in S1, so
  addresses decide; 78–83 % of true pairs share an address number.
- **Density.** Test entities meet 3× (US) to 6× (India) more same-name records than
  validation ones, and ≈ 40 % of the test pool is unowned (26 % in train): both day-1
  uploads lost 0.030 from validation to public.

### 2.2 Solution Strategy

- **Mock fold.** Per country, train is reshaped to the test's pool size and pool records per
  S1; dropped S1 leave their records as unowned decoys (40.2 % of the pool). All 1.37M S1
  compete in the one-to-one, as on test.
- **Tight mock.** Splitting mock loss into wrong pairs (L_FP) and the rest (L_FN),
  public ≈ 1 − L_FN − 1.45·L_FP − 0.0072 fits uploads 2 and 3; this **est_public** ranks
  versions since v107 and predicted its 0.966 as 0.9659.

**Approach Type:** Blocking + Classifier (two-stage gradient-boosted trees) with a metric-tuned
set decision.  
**Core Innovation:** a leaderboard-calibrated, test-shaped mock fold that decides every change
and trains a competition-aware stage 2.

---

## 3. Candidate Generation (Blocking)

Normalisation: `anyascii` transliteration; case, punctuation, domain and leet folding; legal
forms split off; a 536-token transliteration map learned from train-fold pairs; region codes,
French départements included.

- **Blocking keys used** (per country): exact core name, sorted tokens or alphanumerics (pool
  groups ≤ 200); exact core name + a shared address number (≤ 100); TF-IDF top-k on core-name
  char 3-grams (10, short-address pool records), name + address word 1–2-grams (50) and
  address words (10); from v120, exact core name without the learned filler words (≤ 50);
  at most 120 per S1, best cosine first.
- **Candidate pairs generated:** v107: 60.52M blocked test pairs (34.9 per S1), 4.8–6.0 per
  S1 after the filter; v110: 69.3 per S1 blocked at mock density, 4.4 after the filter
  (test 4.8–6.1); v120: **TBD**.
- **How you ensured true matches were not lost:** address passes catch pairs sharing no name
  token; the cosine-ranked cap keeps same-name crowds from pushing out variants. Pair recall
  at mock density: 0.9677 (v001–v107), 0.9780 (v105), 0.9829 (v109). A learned filter keeps
  pairs with stage-1 probability p1 ≥ 0.01 in their entity's top 16, exactly
  `candidate_pairs.tsv`: 34.5 → 4.6 per S1 for −0.0011 recall (v104); v110 keeps 0.9788
  of true pairs (0.9829 before the filter) at 4.4 per S1.

---

## 4. Matching Model

**Features used** (76 per pair, no country feature):
- Name features: rapidfuzz ratio, partial, token-sort, token-set, Jaro-Winkler, Levenshtein on
  full, core and squashed names; token Jaccard, Dice; legal-form agreement.
- Address features: token-set, partial, plain ratios; Jaccard, containment both ways; region,
  number, house-number, postcode(-prefix) agreement.
- Other: blocking cosines, IDF-weighted overlaps, rank and gap to the entity's best candidate,
  and how many records share each name and address; v110's 23 new ones (`idf`,
  `token_freq`, `ctx_idf`, `address_extra`) lifted single-stage validation F0.5 from 0.9844 to
  0.9870 (v040); v120 adds 6 on the names without learned filler words (`center`,
  `services`, `lnc`…). Stage 2 adds 20 from the stage-1 probabilities p1: competition (rank,
  best rival, gap), anchor (vs the entity's best other candidate) and rival (vs the record's
  best rival S1); v120 adds 12 learned word-evidence features: the log-odds that a word only one
  name holds marks a true match's filler or a decoy (`holdings`, `group`, `midtown`…).

**Model type:** stage 1: XGBoost on the GPU (127 leaves) on the candidate pairs of 213k
train-fold entities absent from the mock (15M pairs): v110 one model on 7.0M rows (4 GB card),
v120 two models on disjoint halves of the entities, averaged (12.9M rows); LightGBM up to
v107. Stage 2: XGBoost on the GPU, cross-fitted in two parts on the mock (v110: fit entities,
63 leaves; v120: fit + tune entities, 127 leaves, mean of 3 seeds); `pool_gap` (p1 minus the
record's best rival) carried 72 % of its gain in v104.

**Threshold selection method:** each pool record stays only with its highest-probability S1;
expected-F0.5 decoding then keeps each entity's ranked prefix of highest expected F0.5 (pair
probability p^γ, m expected misses), competing with a 3,906-rule threshold grid; both are tuned
by exact macro F0.5 on the tight mock's tune entities (v107: γ 1.5, m 0.05, ≤ 11 matches;
v110: γ 1.5, m 0.1; v120: **TBD**).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** validation fold 0.9858 (v101), best 0.9870 (v040); mock and public
  below.

| Version | Change | Mock F0.5 | est_public | Public |
|---|---|---|---|---|
| v001 | 47 features, LightGBM, tuned rule | – | – | 0.954 |
| v101 | + core-name frequencies (53 features) | 0.9677 | 0.955* | 0.955 |
| v103 | rule tuned on the mock | 0.9704 | 0.961* | 0.961 |
| v104 | two-stage, stage 2 trained on the mock | 0.9744 | 0.9654 | – |
| v107 | rule tuned on the tight mock | 0.9745 | 0.9659 | 0.966 |
| v110 | + 23 features, denser blocking, GPU stage 1, rival features | 0.9817 | 0.9731 | **TBD** |
| **v120** | + rules v5, learned filler and decoy words, 2-bag stage 1, 3-seed stage 2 | **TBD** | **TBD** | **TBD** |

\* tight-mock calibration points. Validation missed what mattered (v103: −0.0025 validation,
+0.006 public).

- **Common false positives (wrong merges):** same-name decoys the address cannot separate: an
  empty pool address, or a house number apart (`Osprey Group, 231 Silvermine Avenue` vs `232
  Silvermine Ave`, a true singleton). Mock false-merge pairs: 15.9k (v101's rule), 3.5k
  (v107), 3.2k (v110), **TBD** (v120).
- **Common false negatives (missed matches):** misses rose in exchange (64k → 74k pairs), then
  fell to 55k in v110 (v120 **TBD**): Indic-script names (`Baba Food` ↔ बाबा फूड), domain forms, truncations, renames,
  pairs won by a rival S1 (0.94 % of true pairs).

---

## 6. Conclusion

Dense per-country blocking gives recall; the competition-aware stage 2 and one-owner decision
give precision. Lesson: random validation hid a 0.03 drop from denser same-name decoys; a
leaderboard-calibrated, test-shaped mock fold took public F0.5 from 0.954 to 0.966 (final
**TBD**).

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` is this repository: the tested library
`src/entity_resolution/`, one documented notebook per experiment in `experiments/` (registry
`experiments.csv`, uploads `LEADERBOARD.md`); the final notebook is also in `src/notebooks/`.
Entry point (`README.md`; Python 3.12, 15 GB RAM, CUDA GPU or `device="cpu"`):

```bash
make setup && make cache
make nb NB=experiments/v120_night_build/v120_night_build.ipynb   # writes output/*.tsv
make validate
```

**Compliance.** Only the provided data: no external data, API, lookup or pretrained language
model; dictionaries are hand-typed or learned from train pairs. Models: gradient-boosted trees
from XGBoost 3.1.1 (Apache-2.0) and, up to v107, LightGBM 4.7.0 (MIT); v120 has **TBD**
parameters (tree nodes), far below 8B; no GPL package. Country is only a partition key (no filter, feature
or threshold): France, unseen in train, gets a row per entity (v107: 94.5 % of French S1
matched; India 93.7 %, US 94.1 %).

### B. Additional Results

Mock F0.5 India / US: v107 0.9701 / 0.9792; v110 0.9802 / 0.9832; v120 **TBD**. Stage-2
ablations (est_public): v110 without M3's groups −0.0005, without rival features −0.0001; v120
without word evidence **TBD**.
