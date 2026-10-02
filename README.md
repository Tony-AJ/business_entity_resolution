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
