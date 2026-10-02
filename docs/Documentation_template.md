# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team Skill Hive  
**Team Members:** Raja Guru R, Tony AJ, Tarakesh, Prasanna S  
**Submission Date:** 2 October 2026 (final leaderboard upload: 27 September 2026)

---

## 1. Executive Summary

We block each country separately with exact-key and TF-IDF passes, filter the candidates
with a first gradient-boosted matcher, and decide with a second matcher. The second matcher
is trained on a **test-shaped mock fold** and sees how every candidate competes with its
rivals; its probabilities are calibrated and decoded per entity for a precision-heavy
objective. The decisive finding came from reading the test's uncertain pairs: the generator's
decoy is a second business named like the Source 1 entity at a *nearby house number*, with
its own records. Features for that signature took public F0.5 from 0.968 to **0.972** (final
version `v126_decoy_groups`). The code reproduces the uploaded file byte for byte.

---

## 2. Methodology

### 2.1 Problem Analysis

Data: train 2.21M Source 1 (S1) entities and 10.3M Source 2/3 ("pool") records (US 60 %,
India 40 %); test 1.73M S1 and 9.97M pool records, with France (15 % of test S1) absent from
train. Measured on the train labels:

- **Structure.** A pool record belongs to at most one S1 entity, S2 and S3 are not
  deduplicated against each other, and an S1 entity has 3.7 matches on average (at most
  11). 5.6 % of S1 entities are singletons, whose empty prediction scores 1.0 and any
  prediction 0.
- **Name noise.** On 220k true pairs, names are identical as written for 4.6 % and after
  folding case, accents, punctuation and legal forms for 48.4 %; 14.7 % share no token
  (Indic-script names, domains and handles such as `clumora.com` or `@sakashpoint`, renames).
  The generator also writes leet (`5tar Flxe`), injected accents, moved legal forms (`LLC
  Moncada Learning Center`) and prefix junk (`-- Holloway`).
- **Address noise.** Abbreviations, states in full, abbreviated or in a regional script (9.2 %
  of pool addresses hold a non-Latin token), empty addresses (4.4 %), reordered components.
  78 % (US) and 83 % (India) of true pairs share an address number.
- **Name ambiguity.** 48 % of S1 records share their core name with another S1 record, and
  21 % of unmatched pool records carry the core name of some S1 entity. The address, not the
  name, separates true matches from decoys.
- **Density.** About 40 % of the test pool belongs to no S1 entity (26 % in train), and a test
  entity meets 3-6 times more same-name records of other businesses than an entity of a
  random 20 % validation split. Our first two uploads scored 0.030 below validation for this
  reason.
- **Decoys (read from the unlabelled test, confirmed on labelled data).** Test candidates are
  far less certain than validation ones: candidates scored between 0.2 and 0.8, per S1, are
  0.15 on the mock fold but 0.19 (India), 0.37 (US) and 0.80 (France) on test. These pairs
  show the generator's decoy: the S1 name plus or minus a word ("Holdings", "Corp") at a
  nearby house number (7541 vs 7532), with its own S2 and S3 records. True records carry the
  S1's number, a truncated or zero-padded form of it (201 for 2014), or one source's own
  corrupted number shared by all of that source's records (600 for 200). On the labelled
  mock (US), pairs whose first numbers differ are true 97 % of the time when one contains the
  other, 65 % for a one-digit change and 41 % within 20. Among uncertain pairs, a number no
  other candidate holds is true 37 % of the time, a number shared with one other candidate
  81 %.

### 2.2 Solution Strategy

1. **Normalise** names and addresses with hand-typed rules and a transliteration map learned
   from train pairs.
2. **Block** inside each country with seven passes, at most 120 candidates per S1.
3. **Stage 1**: XGBoost on 76 pair features gives `p1`. A learned filter keeps `p1 >= 0.01`
   among each S1's 16 best: 5.09 candidates per test S1, which is the shipped
   `candidate_pairs.tsv`.
4. **Stage 2**: XGBoost on 115 columns, adding competition, anchor and rival features
   computed from `p1` and 19 decoy-signature features. It is trained on a mock fold built to
   the test's density.
5. **Decide**: one owner per pool record, isotonic calibration, then per entity the prefix of
   ranked candidates that maximises the expected F0.5 with false merges weighted x3.

Validation: a fixed 20 % split of train (seed 42) for the first versions, then the **mock
fold**. The mock reshapes all of train to the test's pool size and pool records per S1 for
each country (India 709,678 S1 / 4.13M pool, US 663,049 / 3.82M, 40.2 % of the pool unowned).
Its fit entities train stage 2, its tune entities choose calibration and decoder, and its val
entities score. Every change from v103 on was judged by its mock score. The public steps then
calibrated how much a false merge costs on the board (§5.2).

**Approach Type:** Blocking + two-stage classifier (gradient-boosted trees) + a calibrated,
metric-optimal set decoder.  
**Core Innovation:** a leaderboard-calibrated, test-shaped mock fold on which the second
stage learns candidate competition and the decoy signature (house-number relation and
candidate groups) read from the unlabelled test.

---

## 3. Candidate Generation (Blocking)

**Normalisation first.** Names: `anyascii` transliteration of Indic and accented text, case
and punctuation folding, `&` -> `and`, dotted initials joined, domain and handle forms
stripped, legal forms canonicalised and moved out of the core name (`Pvt Ltd`, `LLC`, `SARL`,
transliterated `praivet limited`), honorifics and leet digits folded. Addresses: each comma
component checked against every written form of a region (US states, Indian states incl.
native-script names, French regions and départements) and replaced by its code; ordinals,
digit-letter splits and street types canonicalised (`St`/`Street`/`Saint`, `R.`/`Rue`). A
transliteration map of 536 tokens is learned by aligning transliterated pool names with their
S1 partners over train-fold true pairs.

- **Blocking keys used** (within each country; true pairs always share it, and any country
  value forms its own partition):

| Pass | Key | Limit |
|---|---|---|
| exact core name | `name_core` | pool groups <= 200 |
| exact sorted name | sorted distinct words of `name_core` | pool groups <= 200 |
| exact squashed name | letters and digits only (domains, handles) | pool groups <= 200 |
| exact name + number | equal `name_core` and a shared address number | groups <= 100, kept first |
| name character TF-IDF | `name_core` char 3-grams, cosine >= 0.5 | top 10 (pool records with <= 3 address tokens) |
| name + address word TF-IDF | word 1-2-grams of name and address, cosine >= 0.2 | top 50 |
| address word TF-IDF | `addr_norm` word 1-2-grams, cosine >= 0.3 | top 10 |
| union | best cosine first, name + number pairs first | <= 120 per S1 |

- **Candidate pairs generated** (test): out of 6.72 trillion same-country S1 x pool pairs,
  blocking keeps **119,813,160** (69.2 per S1, reduction ratio 0.999982). The stage-1
  filter keeps **8,818,600** (5.09 per S1: France 6.13, India 4.80, US 5.04; reduction ratio
  0.9999987). These are exactly the pairs stage 2 scores and exactly `candidate_pairs.tsv`.
- **How you ensured true matches were not lost:** recall was measured at the test's density
  on the mock fold, where it is 1-2 points lower than on a random split. Larger top-k and caps,
  a sim-first cap (so same-name crowds cannot push out variants) and the name + number pass
  lifted pair recall from 0.9677 to **0.9829** (v105, v109). Address passes catch pairs that
  share no name token, and the exact squashed-name pass catches domains and handles. The
  filter costs 0.4 points (0.9788 kept, entity recall 0.9984, recall ceiling F0.5 0.9933)
  while cutting the test candidates 13.6-fold.

---

## 4. Matching Model

**Features used** (no country feature; country only partitions):

- **Name features (stage 1):** rapidfuzz ratio, partial ratio, token-sort, token-set,
  Jaro-Winkler and Levenshtein on the full, core and squashed names; token Jaccard and Dice,
  shared tokens, equal first token, equal sorted name, 4-character prefix; legal-form
  agreement and missing legal form per side; IDF-weighted name cosine, rarest shared token
  and coverage per side; how many S1 / pool records share each side's core name (per
  million).
- **Address features (stage 1):** token-set, partial and plain ratios, Jaccard, containment
  both ways, equal region, equal last words, empty address per side; address-number
  Jaccard, shared number, equal first number, number containment both ways, equal postcode
  and postcode prefix; IDF-weighted address cosine and coverage; length ratio.
- **Other (stage 1):** which blocking passes found the pair and their cosines; rank and gap of
  the pair among its S1's candidates by name, address and IDF similarity; same-name
  candidates; source (S2/S3); non-Latin pool name. In total 76 features in 13 groups.
- **Stage 2 adds 39 columns, computed on the kept pairs:**
  - competition (12): `p1`; rank, best rival, gap, sum and count of likely candidates on
    the S1 side and on the pool side. `pool_gap`, the pair's `p1` minus the pool record's
    best other S1, carries 67 % of stage 2's gain.
  - anchor (5) and rival (3): the candidate against its S1's best other candidate (true
    records of one business resemble each other) and against the S1 that competes for it.
  - decoy (19): the **house-number relation** of the first numbers (equal; one contains the
    other; one digit changed; within 20; log gap; which side lacks one); **candidate
    groups** of the same S1 (how many other candidates share the pair's number, name or
    address and the best `p1` among them, support from the other source, candidates holding
    the S1's number); **unmatched-word IDF** (rarity of the core-name words only one side
    holds, number conflict).

**Model type:** XGBoost (`hist`, leaf-wise, 127 leaves, learning rate 0.1) on a 4 GB GPU.
**Stage 1** is trained on the 14.9M candidate pairs of 212,941 train-fold entities absent from
the mock fold (7.0M rows fit the card, whole entities by id hash). It stops early on a 5 %
entity slice: 396 trees, AUC 0.99984. **Stage 2** is trained on the 4.55M kept pairs of the
mock fold's fit + tune entities and cross-fitted in two parts by id hash: each model stops
early on the part it never saw, and every mock entity is scored out of fold (598 and 528
trees, AUC 0.998); test pairs get the mean of both. In total 1,522 trees and 385,066 nodes are
used at prediction, under 1M learned numbers. No pretrained model and no external data.

**Threshold selection method:** there is no fixed threshold; the set is decoded.
(1) Each pool record stays only with the S1 that gives it the highest probability.
(2) Stage-2 probabilities are calibrated by isotonic regression on the mock's tune entities.
(3) Per entity, the decoder keeps the prefix `k` of its ranked candidates (at most 11) that
maximises the plug-in expected tight score `w·E[F0.5] - (w-1)·E[F0.5 without false
positives]`. Keeping nothing is chosen when P(no true match) = Π(1 - p)·e^(-miss) is higher.
`(w, miss)` is chosen on the tune entities by the tight score at x3 over w ∈ {2, 3, 4, 5, 6,
8} and miss ∈ {0, 0.05, 0.1, 0.2, 0.4}: **w = 4, miss = 0.4**.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** mock fold, val entities (340k S1, test-shaped): **0.9828** (pair
  precision 0.9987, recall 0.9508, singletons included). Random validation fold, single
  stage: 0.9870. **Public leaderboard: 0.972.**

### 5.1 Final version against its parent (mock fold, val entities, same decoder)

| | v110 | **v126 (final)** |
|---|---|---|
| macro F0.5 | 0.9805 | **0.9828** |
| tight score (false merges x3) | 0.9767 | **0.9802** |
| false-merge pairs on matched entities | 1,928 | **1,259** |
| pairs merged into true singletons | 228 | **158** |
| missed true pairs | 63,816 | **56,205** |

### 5.2 Leaderboard progression

| Version | Change | Mock F0.5 | Public F0.5 |
|---|---|---|---|
| v001 | base pipeline, 47 features, LightGBM, tuned rule | (val 0.9844) | 0.954 |
| v101 | + core-name frequencies | 0.9677 | 0.955 |
| v103 | rule tuned on the mock fold | 0.9704 | 0.961 |
| v107 | two-stage matcher, false merges x1.45 | 0.9745 | 0.966 |
| v110 | + 23 features, denser blocking, GPU stage 1 | 0.9817 | 0.966 |
| v123 | decoded with false merges x3 | 0.9823 | 0.968 |
| **v126** | **+ decoy features, isotonic + tight decoder** | **0.9828** | **0.972** |
| v127 / v128 | + a third stage / x6 decoding | 0.9829 / 0.9821 | 0.971 / 0.972 |

The public steps calibrated the objective. v103 gained 2.2 times its mock gain, and v122 ->
v123 -> v126 moved only with precision, implying the board charges a false merge about 6 times
a missed match. The decoy features gained 2.5 times their mock gain on the board.

- **Common false positives (wrong merges):** an unrelated name at the S1's address (a
  co-located business); the S1's name on a pool record with an empty address; a differing
  unit number (`225 Apt 225` vs `225 Apt 234`); a twin at the same address with a legal word
  added. Before v126, mostly the nearby-number decoy.
- **Common false negatives (missed matches):** of 57.8k missed true pairs on the mock, 24.9k
  never become candidates (heavily noised names with empty or partial addresses, domains,
  handles). 10.7k are taken by a rival S1 in the one-owner step, mostly empty-address records
  of same-name businesses. 22.2k stay below the decoder, with probabilities 0.4-0.95, because
  the x3 objective trades them for precision.

---

## 6. Conclusion

Dense per-country blocking gives recall. A stage 2 that knows each candidate's competition
and the decoy signature, plus a calibrated one-owner decoder, give precision. Two lessons
mattered most. Random validation hid a 0.03 drop caused by denser decoys, and a test-shaped
mock fold calibrated on the leaderboard fixed that. Reading the unlabelled test's uncertain
pairs then found the signal no feature read. Next steps: address-key and name-prefix
blocking passes for the 2.1 % of true pairs never generated, and the one-owner losses on
empty-address records.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` is our repository. All pipeline code is in `src/`: the
tested library `src/entity_resolution/` (427 tests) and the final version's notebook in
`src/notebooks/`. The single entry point `python -m entity_resolution.final`
(`src/entity_resolution/final.py`) regenerates both output files from the organisers' data:

```bash
make setup                    # Python 3.12 venv with the pinned requirements.txt
make cache                    # TSVs -> Parquet, ~1 min
make reproduce                # stage A (blocking, stage 1) + stage B (stage 2, decoder), ~3.5 h
make validate                 # our checker + the organisers' validator on output/
make verify                   # sha256 of output/*.tsv vs the uploaded files
```

Stage A runs the configuration of `experiments/v110_m3_features` and stage B that of
`experiments/v126_decoy_groups/run_v126.py`. A rerun on our machine (12 CPU threads, 15 GB
RAM, 4 GB RTX 2050) rebuilt `matching_results.tsv` (sha256 `617c9822…5fa5`) and
`candidate_pairs.tsv` (`7164d366…2754`) byte for byte. Module-by-module architecture,
runtimes and the full commands are in the package's `README.md`. Every experiment (30
versions, each with its notebook or script and `metrics.json`) is in `experiments/`, and every
upload in `LEADERBOARD.md`.

**Compliance.** Only the provided data: no external data, API, lookup or pretrained model.
The region and legal-form tables are hand-typed, and the token map is learned from train
pairs. Models: gradient-boosted trees trained with XGBoost (Apache-2.0); our code and models
are MIT-licensed. 1,522 trees and 385k nodes are used at prediction, far below 8B
parameters. Country is only a partition key (no filter, feature or threshold), so France gets
a row for each of its 259,452 entities.

### B. Additional Results

| Stage-2 gain share | | Stage-1 gain share | |
|---|---|---|---|
| `pool_gap` | 0.670 | `ctx_gap_addr` | 0.272 |
| `p1` | 0.164 | `ad_token_set` | 0.130 |
| `s1_gap` | 0.066 | `squash_ratio` | 0.104 |
| `s1_p1_sum` | 0.007 | `ctx_rank_addr` | 0.097 |
| 19 decoy columns | 0.025 | `sim_name_addr_word` | 0.067 |

| Test output | France | India | US | All |
|---|---|---|---|---|
| S1 entities | 259,452 | 809,986 | 663,106 | 1,732,544 |
| candidates per S1 | 6.13 | 4.80 | 5.04 | 5.09 |
| matched pairs | 819,245 | 2,652,987 | 2,214,041 | 5,686,273 |

94.1 % of test S1 entities get at least one match (in train, 94.4 % of S1 entities have one).
