# Business Entity Resolution: Team Skill Hive

**Amazon ML Challenge 2026.** Business records from three sources share no identifier. For
every Source 1 (S1) entity we find all Source 2 / Source 3 records that describe the same
business, scored by macro F0.5 per S1 entity, which weighs precision twice as much as recall.

Our final submission is **`v126_decoy_groups`, public F0.5 0.972**, our best upload. The
pipeline blocks per country, filters candidates with a learned first stage, matches with a
second stage trained where the test's decoys are, and decodes the matches for a
precision-heavy objective. This folder regenerates the uploaded files **byte for byte** from the
organisers' data with one command.

| | |
|---|---|
| Final version | `v126_decoy_groups` on `v110_m3_features`'s stage 1; submission 09, 27 Sep 2026 |
| Leaderboard | public F0.5 **0.972**, our best (first upload: 0.954); every upload in [LEADERBOARD.md](LEADERBOARD.md) |
| Local scores | mock fold (test-shaped validation) F0.5 **0.9828**, pair precision 0.9987, pair recall 0.9508 |
| Candidates | 5.09 per test S1 (8.82M pairs) in `candidate_pairs.tsv`; recall at test density 0.9788 |
| Model | 3 XGBoost boosters (Apache-2.0): 1,522 trees, 385k nodes at prediction; no pretrained model, no external data |
| Reproduce | `make setup && make reproduce`: ~3.5 h on 12 CPU threads, 15 GB RAM and a 4 GB CUDA GPU |
| Check the shipped files | `make verify OUT=../../output`: sha256 of both TSVs against the uploaded files, in seconds |

**Contents.** [1. Quick start](#1-quick-start) · [2. Reproduction in detail](#2-reproduction-in-detail)
· [3. How the solution works](#3-how-the-solution-works) · [4. Architecture, module by module](#4-architecture-module-by-module)
· [5. Validation strategy](#5-validation-strategy) · [6. Results](#6-results)
· [7. Rule compliance](#7-rule-compliance) · [8. Repository map](#8-repository-map)
· [9. Experiment history](#9-experiment-history) · [10. Team and licence](#10-team-and-licence)

The methodology write-up is `Documentation_template.md` at the root of the submission zip
(source: [docs/Documentation_template.md](docs/Documentation_template.md)). The challenge
brief is condensed in [docs/PROBLEM_STATEMENT.md](docs/PROBLEM_STATEMENT.md).

---

## 1. Quick start

Run every command from this folder (`code/business_entity_resolution/` inside the zip). Each
one names the files it writes.

### 1.1 Check the shipped outputs (seconds)

```bash
make setup                        # .venv with the pinned requirements.txt (Python 3.12), ~3 min
make verify OUT=../../output      # sha256 of the zip's output/*.tsv vs the uploaded files
```

`make verify` prints `IDENTICAL to the upload` for both files when they are the bytes we
uploaded (matching file `617c9822…5fa5`, candidate file `7164d366…2754`; full hashes in
[§2.4](#24-expected-outputs)).

### 1.2 Smoke test without the data (about 2 minutes)

```bash
make test                         # 427 tests on synthetic files, CPU only
```

`tests/test_final.py` runs **both stages of the final pipeline** on a generated dataset (300
train and 180 test entities, France only in test), checks both files with our checker and the
organisers' rules, and checks that stage B gives identical bytes twice.

### 1.3 Reproduce the submission end to end (~3.5 hours)

```bash
# the organisers' student resource zip, unzipped here -> dataset/student_resource/dataset/{train,test}
unzip <path/to/student_resource.zip> -d dataset/
make setup                        # pinned environment, ~3 min
make cache                        # every TSV parsed once to Parquet, ~1 min
make reproduce                    # stage A + stage B -> output/*.tsv, ~3.5 h on a 4 GB GPU
make validate                     # our checker + the organisers' validator on output/
make verify                       # output/*.tsv vs the uploaded files (sha256)
```

`make reproduce` runs `python -m entity_resolution.final`, the single entry point. It logs each
step, writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, runs both validators
and compares both files with the uploaded ones. Without a CUDA GPU, run `make reproduce
DEVICE=cpu` (see [§2.5](#25-determinism-and-tolerance)).

---

## 2. Reproduction in detail

### 2.1 Environment we used

| | |
|---|---|
| CPU | Intel Core i5-11400H, 6 cores / 12 threads |
| RAM | 15 GB (+15 GB swap); the pipeline peaks at 7.7 GB and aborts above 13 GB (`pipeline.mem_guard`) |
| GPU | NVIDIA GeForce RTX 2050, 4 GB, driver 615.71 (CUDA 13.4); used by XGBoost only |
| Disk | ~15 GB of caches under `dataset/.cache/` (normalised records, candidate pairs, stage-1 outputs) |
| OS | Fedora Linux 44, kernel 7.2.5 |
| Python | 3.12.14; every package pinned in [requirements.txt](requirements.txt) (xgboost 3.1.1, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, pyarrow 25.0.1, rapidfuzz 3.14.6) |

No network access is needed after `make setup`, and the pipeline never makes a network call.

### 2.2 Data setup

Unzip the organisers' `student_resource.zip` into `dataset/`. Two layouts are found
automatically ([src/entity_resolution/config.py](src/entity_resolution/config.py)):

```
dataset/student_resource/dataset/{train,test}/*.tsv     # the zip as shipped (also gives the
                                                       # organisers' validator in utils/)
dataset/{train,test}/*.tsv                              # or the two folders directly
```

Expected files: `train_source{1,2,3}.tsv`, `train_ground_truth.tsv`, `test_source{1,2,3}.tsv`.
Every path is set in `config.py`; caches go to `<dataset>/.cache/` and models to `models/`.

### 2.3 What `make reproduce` runs

[src/entity_resolution/final.py](src/entity_resolution/final.py), in two stages. Each step
caches its result, so a stage can run alone (`make reproduce STAGE=a` or `STAGE=b`) and a
rerun skips finished work.

| Step | What it does | Time* |
|---|---|---|
| A1 | normalise all 24.2M records (rules v3), learn the 536-token transliteration map, build the validation split and the test-shaped mock fold | ~12 min |
| A2 | block the 212,941 stage-1 training entities against the train-fold pool, 76 features per pair (14.9M pairs) | ~16 min |
| A3 | train stage 1: XGBoost on the GPU, early stopping on a held-out entity slice | ~4 min |
| A4 | block the mock fold (1.37M S1) and run stage 1 over its 95.8M candidates: filter + competition / anchor / rival features | ~68 min |
| A5 | the same on the test split (1.73M S1, 119.8M candidates), France included | ~85 min |
| B1 | decoy-signature features on the 6.0M kept mock pairs | ~4 min |
| B2 | stage 2: two cross-fitted XGBoost models on the mock's fit + tune entities | ~4 min |
| B3 | isotonic calibration and the decoder grid on the mock's tune entities, scores on its val entities | ~4 min |
| B4 | test inference (8.82M pairs), both TSVs, both validators, sha256 against the upload | ~10 min |

\*On the machine above; peak RAM 7.7 GB in stage A, 6.0 GB in stage B. Outputs: `output/` (the two TSVs), `models/final/` (stage-1 model,
stage-2 models, isotonic calibrator, decoder grid, `stage_a.json` and `final_report.json` with
every score and timing), caches under `dataset/.cache/pipeline/`. `make reproduce` overwrites
`output/`.

### 2.4 Expected outputs

| File | Rows | Pairs | sha256 |
|---|---|---|---|
| `matching_results.tsv` | 1,732,544 (one per test S1) | 5,686,273 | `617c9822921c959c6dbee6fceeda42642afb1974d23cbee4c86548ab86605fa5` |
| `candidate_pairs.tsv` | 1,732,544 | 8,818,600 | `7164d366a0c00209003db6affadcacd1448479c76798a74ee7936e9ed6a12754` |

1,629,743 test S1 (94.1 %) get at least one match: France 819,245 pairs, India 2,652,987, US
2,214,041. Every matched id is in the candidate file, as the rules require.

### 2.5 Determinism and tolerance

- Every sample and split is a hash of the entity id with a fixed seed, so nothing depends on
  row order or on a random state: validation split 42, inner fit/tune split 4242, S1 samples 7,
  mock fold 5151 / 5152, cross-fitting 6161, stage-1 early-stopping slice 7171, XGBoost 42,
  TF-IDF vocabulary sample 42.
- XGBoost's GPU `hist` is deterministic for a fixed seed, and so are blocking, features and
  decoding. **Verified:** on the machine above, a rerun of stage B from stage A's caches
  rebuilt both uploaded files byte for byte (`make verify` -> IDENTICAL).
- Stage A ran in the original experiment (`experiments/v110_m3_features/`) under the library
  of 26 Sep. `final.py` runs the same configuration on today's library. The normalisation
  rules that changed since are switched back by `NormaliseConfig(rules=3)`, and the rebuilt
  records are identical to the originals on all 24.2M records (checked file by file).
- `DEVICE=cpu` trains the same models on the CPU. Their splits can differ slightly from the
  GPU's, so the files may not be byte-identical, and we have not timed that run end to end.

### 2.6 Troubleshooting

| Symptom | Fix |
|---|---|
| `make setup`: `python3.12: not found` | install Python 3.12, or `make setup PYTHON=/path/to/python3.12` |
| `MemoryError: RSS ... > 13.0 GB` | close other programs; the run needs ~8 GB free. Steps resume from their caches. |
| XGBoost: no CUDA device | `make reproduce DEVICE=cpu`, or install the NVIDIA driver; the xgboost wheel bundles CUDA |
| `... not found: run stage A first` | `make reproduce STAGE=b` needs stage A's caches; run `make reproduce` |
| the organisers' validator is skipped | it is read from `dataset/student_resource/utils/`; unzip the whole student resource there |
| out of disk | caches need ~15 GB; delete `dataset/.cache/pipeline/` to start over |

---

## 3. How the solution works

```
 train + test TSVs: 24.2M records (S1 reference; S2 + S3 "pool"); US, India; test adds France
   |
   v
 [1] normalise names and addresses, learned transliteration map   normalize.py, token_maps.py
   |
   v
 [2] block inside each country: 4 exact-key passes + 3 TF-IDF passes, <= 120 per S1   blocking.py
   |      test: 119.8M candidate pairs (69 per S1) out of 6.7 trillion same-country pairs
   v
 [3] stage 1: 76 pair features -> XGBoost probability p1          features.py, stage1.py
   |
   v
 [4] learned filter: p1 >= 0.01 and among the S1's 16 best  ----------->  output/candidate_pairs.tsv
   |      test: 8.82M pairs (5.09 per S1); recall at test density 0.9788
   v
 [5] stage-2 features: competition, anchor, rival (from p1) + 19 decoy columns   stacking.py, decoy.py
   |
   v
 [6] stage 2: XGBoost trained on the test-shaped mock fold, cross-fitted         twostage.py
   |
   v
 [7] decision: one owner per pool record -> isotonic calibration -> tight F0.5 decoder   decision.py
   |
   v
 output/matching_results.tsv: 5.69M pairs, 94.1 % of test S1 matched
```

The ideas that moved the leaderboard, in order (details in [§6](#6-results)):

1. **A test-shaped validation fold.** Test entities meet 3-6 times more same-name records of
   other businesses than a random validation split does, and 40 % of the test pool belongs to
   no S1. Our first two uploads scored 0.030 below validation for that reason. The **mock
   fold** ([mock.py](src/entity_resolution/mock.py)) reshapes train to the test's pool size and
   pool records per S1. Every decision since v103 was judged on it.
2. **Two stages.** Stage 1 scores all ~70 candidates of an entity. Stage 2 sees how each
   candidate compares with its rivals on both sides (`pool_gap`, the p1 lead over the pool
   record's best other S1, carries 67 % of stage 2's gain) and is trained on the mock fold,
   at the test's decoy density.
3. **Decoding for precision.** The public steps showed the board charges a false merge about
   six times a missed match. A calibrated expected-score decoder keeps, per entity, the
   prefix of its ranked candidates that maximises the expected *tight* score.
4. **Decoy-signature features (v126).** Reading the uncertain test pairs showed the
   generator's decoy: the S1's name, plus or minus a word, at a **nearby house number**,
   with its own S2 and S3 records. Nineteen stage-2 columns read that signature (house-number
   relation, candidate groups, unmatched-word IDF). False merges fell by a third on the mock;
   public F0.5 rose from 0.968 to 0.972.

---

## 4. Architecture, module by module

All pipeline code is the library [src/entity_resolution/](src/entity_resolution/) (docstrings,
type hints, tests in [tests/](tests/)). [final.py](src/entity_resolution/final.py) wires it
into the final version; the experiment notebooks and scripts in `experiments/` used the same
library.

### 4.1 Data, splits and the mock fold

| Module | Role |
|---|---|
| [config.py](src/entity_resolution/config.py) | paths, file schema, column names; finds the dataset in either layout |
| [data.py](src/entity_resolution/data.py) | TSV loaders with schema, prefix and uniqueness checks, a Parquet cache, fast id filtering |
| [split.py](src/entity_resolution/split.py) | the fixed validation fold: 20 % of train S1 by id hash (seed 42), each with its true records |
| [trainset.py](src/entity_resolution/trainset.py) | inner fit / tune split (25 % tune), S1 samples by id hash, pair labels by binary search |
| [mock.py](src/entity_resolution/mock.py) | the **mock fold**: per country, whole clusters kept by id hash to reach the test's pool size, then S1 entities dropped (their records stay as unowned decoys) to reach the test's pool records per S1. Result: India 709,678 S1 / 4.13M pool, US 663,049 / 3.82M, 40.2 % of the pool unowned (test ~40 %). Roles: fit (stage-2 training), tune (calibration, decoder), val (scores) |

### 4.2 Normalisation

[normalize.py](src/entity_resolution/normalize.py) and [token_maps.py](src/entity_resolution/token_maps.py),
vectorised on Arrow strings (~5 min per split):

- **Names**: `anyascii` transliteration of Indic and accented text; case, punctuation and `&`
  folded; dotted initials joined (`L.L.C.` -> `llc`); domain and handle forms stripped; legal
  forms canonicalised and moved out of the core name (`Pvt Ltd`, `LLC`, `SARL`, `GmbH`,
  transliterated `praivet limited`); honorifics and leet digits folded. Keys: `name_core`,
  `name_sorted` (word order free), `name_squash` (letters and digits only), `name_first`.
- **Addresses**: each comma component tested against every written form of a region (US
  states, Indian states incl. native-script names, French regions and départements) and
  replaced by its code; ordinals, digit-letter splits, leading zeros; street types
  canonicalised (`St`/`Street`/`Saint`, `Rd`/`Road`, `R.`/`Rue`). Keys: `addr_norm`,
  `addr_nums`, `postcode`, `region`, `addr_last`.
- **Learned token map** (`fit_token_map`): transliterated tokens aligned with their Latin
  spelling over the train fold's true pairs (`sakti` -> `shakti`); 536 tokens.
- **Rules versions.** The final version uses rules v3. Later rules (v4/v5: French `N°`,
  `et`/`+` as `and`, leet legal forms, own country name) are switched off by
  `NormaliseConfig(rules=3)`, which rebuilds v110's records exactly.
- Country is only a partition key: every map holds every country's tokens, so France needs no
  special case and any other country would work the same way.

### 4.3 Blocking: candidate generation

[blocking.py](src/entity_resolution/blocking.py), run inside each country (true pairs always
share it). Passes, unioned per S1:

| Pass | Key / space | Limit |
|---|---|---|
| exact `name_core`, `name_sorted`, `name_squash` | equal keys | pool key groups <= 200 |
| exact name + number | equal `name_core` and a shared address number | groups <= 100, ranked first |
| name char TF-IDF | `name_core` char 3-grams, cosine >= 0.5 | top 10, pool records with <= 3 address tokens |
| name + address word TF-IDF | `name_core + addr_norm` word 1-2-grams, cosine >= 0.2 | top 50 |
| address word TF-IDF | `addr_norm` word 1-2-grams, cosine >= 0.3 | top 10 |
| union and cap | best cosine first, name + number pairs first | <= 120 per S1 |

Top-k retrieval is a multi-threaded sparse product (`sparse_dot_topn`). **Audit** (the
`candidate_pairs.tsv` we ship is the last row):

| Stage | Test pairs | Per S1 | Reduction ratio* | Recall at test density (mock val) |
|---|---|---|---|---|
| all same-country S1 x pool pairs | 6.72e12 | 3.9M | 0 | 1 |
| blocking | 119,813,160 | 69.2 | 0.999982 | 0.9829 |
| stage-1 filter = `candidate_pairs.tsv` | 8,818,600 | 5.09 | 0.9999987 | 0.9788 (entity recall 0.9984) |

\*1 - pairs / same-country pairs. Per country after the filter: France 6.13, India 4.80, US
5.04 candidates per S1.

### 4.4 Pair features and stage 1

[features.py](src/entity_resolution/features.py) builds 76 features per pair, chunk by chunk
(vectorised Arrow and rapidfuzz on 12 threads), in 13 groups:

| Group | n | Examples |
|---|---|---|
| blocking | 6 | which passes found the pair, their cosines |
| name_fuzzy | 10 | rapidfuzz ratio, partial, token-sort, token-set, Jaro-Winkler, Levenshtein on full / core / squashed names |
| name_tokens | 8 | token Jaccard and Dice, shared tokens, equal first token, equal sorted name |
| legal | 3 | legal-form agreement, missing legal form per side |
| numeric | 4 | address-number Jaccard, shared number, equal first number, equal postcode |
| address | 8 | token-set, partial and plain ratios, Jaccard, containment, equal region, empty address |
| context | 5 | rank and gap of the pair among the S1's candidates by name and by address |
| meta | 3 | source (S2 or S3), non-Latin pool name, name length ratio |
| frequency | 6 | how many S1 / pool records share each side's core name (per million) |
| idf | 8 | IDF-weighted name and address cosine, rarest shared token, coverage per side |
| token_freq | 4 | pool frequency of each side's name and address |
| ctx_idf | 5 | rank and gap among the S1's candidates by IDF similarity, same-name candidates |
| address_extra | 6 | reverse containment, number containment both ways, postcode prefix, length ratio |

**Stage 1** ([stage1.py](src/entity_resolution/stage1.py), [model.py](src/entity_resolution/model.py)):
XGBoost `hist` on the GPU, leaf-wise, 127 leaves, learning rate 0.1. It is trained on the
candidate pairs of 212,941 train-fold entities that are **absent from the mock fold**, blocked
against the train-fold pool. Features stream from disk chunks through an `xgboost.DataIter`; a
4 GB card holds 7.0M of the 14.9M rows (whole entities, by id hash). Early stopping uses a 5 %
entity slice: 396 trees, held-out AUC 0.99984.

### 4.5 Learned filter and stage-2 features

[twostage.py](src/entity_resolution/twostage.py) runs stage 1 over every candidate of a
partition and keeps a pair when **p1 >= 0.01 and it is among its S1's 16 best**. The kept
pairs are all stage 2 scores, so they are exactly `candidate_pairs.tsv`. Over the kept pairs
it adds:

| Group | n | Source | What it reads |
|---|---|---|---|
| competition | 12 | [stacking.py](src/entity_resolution/stacking.py) | p1; rank, best rival and gap on the S1 side and on the pool side (`pool_gap`: p1 minus the pool record's best other S1) |
| anchor | 5 | stacking.py | the candidate vs its S1's best other candidate (true records of one business resemble each other) |
| rival | 3 | stacking.py | the pool record vs the S1 that competes for it |
| decoy | 19 | [decoy.py](src/entity_resolution/decoy.py) | house-number relation (equal, truncated, one digit changed, within 20, gap), candidate groups (others sharing the pair's number, name or address and their best p1, other-source support), unmatched-word IDF |

### 4.6 Stage 2

XGBoost on the GPU (127 leaves, learning rate 0.1, early stopping 60) on the 115 columns of
the kept pairs of the **mock fold's fit + tune entities** (4.55M pairs). It is cross-fitted in
two parts by id hash (seed 6161): each model trains on one part and stops early on the
other, and every mock entity is scored by the model that never saw it (598 and 528 trees,
held-out AUC 0.998). Test pairs get the mean of both models.

### 4.7 Decision layer

[decision.py](src/entity_resolution/decision.py):

1. **One owner per pool record** (`one_to_one_filter`): each S2/S3 record stays only with the S1
   that gives it the highest probability, across all entities of the country.
2. **Isotonic calibration** (scikit-learn), fitted on the mock's tune entities.
3. **Tight decoder** (`decode_tight`): per entity, keep the prefix of its ranked candidates
   that maximises `w * E[F0.5] - (w - 1) * E[F0.5 without false positives]`, plug-in
   expectations, at most 11 matches. "Keep nothing" wins when P(no true match) is higher.
   `(w, expected misses)` is chosen on the tune entities by the tight score with false
   merges costing x3, the weight the public uploads validated: **w = 4, miss = 0.4**.

### 4.8 Output and checks

[submission.py](src/entity_resolution/submission.py) writes both files (one row per test S1,
in the test file's order, ids comma-joined) and checks every rule: header, one row per S1, no
duplicates, S2/S3 ids only, ids that exist in the test split, matches within candidates. `make
validate` also runs the organisers' validator.

### 4.9 Supporting modules

| Module | Role |
|---|---|
| [pipeline.py](src/entity_resolution/pipeline.py) | stage wiring and caches: normalisation, token map, blocking per tag, scoring in chunks, the V1 single-stage `fit` / `run_fold` / `run_test` |
| [evaluate.py](src/entity_resolution/evaluate.py), [metrics.py](src/entity_resolution/metrics.py) | macro F0.5 exactly as the organisers define it (singletons included), the tight score, blocking reports, error samples |
| [tracking.py](src/entity_resolution/tracking.py) | experiment registry: `metrics.json` and the version's row in `experiments/experiments.csv` |
| [evidence.py](src/entity_resolution/evidence.py), [hardneg.py](src/entity_resolution/hardneg.py), [snapshot.py](src/entity_resolution/snapshot.py) | features and training schemes tried in later versions (token evidence, hard negatives); not used by the final version |

---

## 5. Validation strategy

- **Fixed validation fold**: 20 % of train S1 by id hash, never trained on. It scored v001 at
  0.9844, and the public board then gave 0.954.
- **Why it misled**: a 20 % fold also keeps only 20 % of the pool, so each validation entity
  meets 3-6 times fewer same-name records of other businesses than a test entity, and the
  test pool holds ~40 % unowned records (26 % in train). Two uploads lost the same 0.030.
- **The mock fold** (§4.1) has the test's pool size and pool records per S1, with every S1
  competing in the one-owner step as on test. Stage 2 trains on its fit (+ tune) entities.
  Calibration and the decoder are tuned on its tune entities, and **mock F0.5** is measured
  on its val entities.
- **The tight score**: splitting the mock's loss into false-merge and missed-match parts,
  the public steps fit a board that charges a false merge 1.45 times (v103), then ~6 times
  (v122 -> v123 -> v126) what macro F0.5 on the mock charges. Decoders are chosen by
  `F0.5 - (w - 1) * loss from false merges` at w = 3.
- Scoring is `evaluate.score_pairs`, identical to the organisers' formula: per-entity F0.5,
  a true singleton with an empty prediction scores 1, then the mean over all S1.

---

## 6. Results

### 6.1 Final version on the mock fold (val entities, 340k S1)

| | v110 (parent), same decoder | **v126 (final)** |
|---|---|---|
| macro F0.5 | 0.9805 | **0.9828** |
| tight score (false merges x3) | 0.9767 | **0.9802** |
| pair precision / recall | 0.9981 / 0.9442 | **0.9987 / 0.9508** |
| false-merge pairs (matched S1) | 1,928 | **1,259** |
| pairs merged into true singletons | 228 | **158** |
| missed true pairs | 63,816 | **56,205** |

Stage-2 gain shares: `pool_gap` 0.67, `p1` 0.16, `s1_gap` 0.07; the 19 decoy columns 0.025
(top: best p1 of the house-number group 0.005, log house-number gap 0.005). Losses that remain
(v127's audit of 57.8k missed pairs): 24.9k never become candidates (heavily noised names with
empty or partial addresses), 10.7k go to a rival S1 in the one-owner step, 22.2k stay below the
decoder.

### 6.2 Leaderboard

| # | Version | Change | Mock F0.5 | Public F0.5 |
|---|---|---|---|---|
| 01 | v001 | base pipeline: normalisation, multi-pass blocking, 47 features, LightGBM, tuned 1-to-1 rule | (val 0.9844) | 0.954 |
| 02 | v101 | + core-name frequency features | 0.9677 | 0.955 |
| 03 | v103 | rule tuned on the mock fold | 0.9704 | 0.961 |
| 04 | v107 | two-stage matcher, rule tuned for false merges x1.45 | 0.9745 | 0.966 |
| 05 | v110 | + 23 features, denser blocking, GPU stage 1, rival features | 0.9817 | 0.966 |
| 06 | v121 | v110 with v107's French rows (diagnostic) | - | 0.965 |
| 07 | v122 | rules v5, learned fillers, token evidence, 2 stage-1 bags, 3-seed stage 2 | 0.9831 | 0.966 |
| 08 | v123 | v122 decoded for false merges x3 | 0.9823 | 0.968 |
| 09 | **v126** | **v110's stage 1 + decoy features, isotonic + tight decoder (x3)** | **0.9828** | **0.972** |
| 10 | v127 | + a third stage on v126's probabilities | 0.9829 | 0.971 |
| 11 | v128 | + cohesion and unit-number columns, decoded at x6 | 0.9821 | 0.972 |

A teammate's upload between 08 and 09 scored 0.969. Every upload, with its commit and files,
is logged in [LEADERBOARD.md](LEADERBOARD.md).

### 6.3 What did not help

- A third stage (v127): +0.0004 on the mock, -0.001 public.
- v122's bundle (rules v5, learned filler words, token evidence, bagged stage 1, seed
  averaging): +0.0014 mock F0.5, public flat at 0.966. Only precision moved the board.
- France-specific rows from an older version (v121): 0.965, so France was not the gap.
- Alias splitting (`dba`, `aka`): its upper bound was +0.00006 (v110 already found 99.92 % of
  alias pairs), so it stayed off.

---

## 7. Rule compliance

| Rule | How we comply | Evidence |
|---|---|---|
| No external data, APIs, geocoding or lookups | Only the provided TSVs. Region and legal-form tables are hand-typed general knowledge; the token map is learned from train pairs. No network calls | [token_maps.py](src/entity_resolution/token_maps.py); `grep -rnE "^\s*(import\|from)\s+(requests\|urllib\|http\|socket)" src/` finds nothing, and `src/` holds no URL |
| Final model MIT / Apache-2.0 | Our code and trained models: MIT ([LICENSE](LICENSE)). XGBoost, which trains them: Apache-2.0. No pretrained weights | `pip show xgboost` |
| At most 8B parameters | 3 boosters with 1,522 trees and 385,066 nodes at prediction (1,692 trees and 428,076 nodes stored), under 1M learned numbers | `Booster.trees_to_dataframe()` on `models/final/` |
| Country is an open set | Country only partitions blocking and statistics; no country filter, feature or one-hot. France, absent from train, gets its own partition | `final.py` loops over the countries found in the data |
| One output row per test S1; matches within candidates | written by `submission.write_pairs`, checked by our checker and the organisers' validator | `make validate`; [§2.4](#24-expected-outputs) |
| `candidate_pairs.tsv` = the set the model scores | the stage-1 filter's kept pairs are exactly what stage 2 scores and what we write | [twostage.py](src/entity_resolution/twostage.py) `stage1_partition` |

Libraries (all open source): xgboost (Apache-2.0), scikit-learn, scipy, numpy, pandas,
psutil (BSD-3), pyarrow and sparse-dot-topn (Apache-2.0), rapidfuzz (MIT), anyascii (ISC).
LightGBM (MIT) trained earlier versions only. The xgboost wheel pulls in NVIDIA's NCCL
library for multi-GPU communication; a single-GPU run does not use it.

---

## 8. Repository map

```
src/entity_resolution/   the library: every stage of the pipeline, documented and tested
  final.py               the final version end to end: python -m entity_resolution.final
src/notebooks/           (in the zip) the final version's notebook and metrics.json
experiments/             one folder per version: notebook or run script, metrics.json, run log
  v110_m3_features/      stage A of the final version as it first ran (notebook, 3.3 h)
  v126_decoy_groups/     stage B as it first ran (run_v126.py + run.log) and its notebook
  experiments.csv        one row per version: change, local / mock / public F0.5, commit
tests/                   427 pytest tests on synthetic files (incl. the final pipeline)
docs/                    challenge brief and guidelines, the methodology write-up, the plan
scripts/                 package_submission.sh (the zip), v010_validate.py (a blocking sweep)
LEADERBOARD.md           every leaderboard upload with its files and score
TRACKER.md               task log of the three days
Makefile                 setup, cache, reproduce, verify, validate, test, lint, package
requirements.txt         pinned runtime environment (Python 3.12)
```

`dataset/`, `output/`, `models/`, `submissions/` and `experiments/*/artifacts/` are generated
locally and never committed.

---

## 9. Experiment history

Every version is a folder `experiments/vNNN_<slug>/` with a documented notebook (hypothesis,
method, evaluation, error analysis, conclusion) or run script, a `metrics.json`, and a row in
[experiments/experiments.csv](experiments/experiments.csv) with its commit. The research plan
written before any model is [docs/plan/00_MASTER_PLAN.md](docs/plan/00_MASTER_PLAN.md) (19
documents: problem analysis, architecture, per-module strategy, validation, testing). `v124`
appears twice in the registry (M3's `v124_tight_decode` and M2's
`v124_address_extra_targeted`); both are kept as history.

The registry's `group` column names the plan item each version tests:

| Group | Plan items |
|---|---|
| A. Blocking | A1 exact normalised name, A2 name TF-IDF, A3 address TF-IDF, A4 character n-gram, A5 multi-pass blocking |
| B. Normalisation | B1 basic normalisation, B2 legal suffixes, B3 abbreviations, B4 token canonicalisation, B5 address components |
| C. Features | C1 string similarity, C2 TF-IDF, C3 address components, C4 numeric tokens, C5 country and context |
| D. Models | D1 logistic regression, D2 random forest, D3 LightGBM, D4 XGBoost |
| E. Decision layer | E1 global threshold, E2 threshold optimisation, E3 score gap, E4 singleton detection, E5 multi-match selection |
| INT, HN | integrated runs (v1xx), hard-negative rounds |

Other commands: `make experiment NAME=<slug>` (a new version from the template), `make nb
NB=<notebook>` (run a notebook headless), `make score PRED=<tsv> TRUTH=<tsv>` (macro F0.5 of a
matching file), `make package TEAM=<team> V=<version> OUT=<dir>` (the submission zip,
[scripts/package_submission.sh](scripts/package_submission.sh)).

---

## 10. Team and licence

**Team Skill Hive**

| Member | Main work |
|---|---|
| Raja Guru R | lead: research plan, pipeline and integration, mock fold, two-stage matcher, decision layer, final version |
| Tony AJ | pair-feature groups (IDF, token frequency, context IDF, address extras, interactions), versions v122-v125 (x3 decoding, tight decoder, unmatched-word IDF) |
| Tarakesh | models: LightGBM tree walker, seed ensembles, model snapshots, hard-negative weighting, calibration helpers |
| Prasanna S | blocking experiments (v010), phonetic features, learned address token map, v124_address_extra_targeted |

Code and trained models are released under the [MIT licence](LICENSE).
