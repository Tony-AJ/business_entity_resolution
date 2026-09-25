# 11 — Validation and metrics (`evaluate.py` + `trainset.inner_split`; owner M1 with M5)

What you build: the instrument every other module is judged by. `evaluate.py` turns pairs
frames into the challenge metric, its slices and the blocking report; `trainset.inner_split`
gives the model and the decision layer data that is never the validation fold. Signatures
are fixed in `02_SYSTEM_ARCHITECTURE.md` §5; this document says what they compute, why, and
how they are tested. Numbers come from `01_PROBLEM_ANALYSIS.md`. The code that already
exists (`src/entity_resolution/metrics.py`, `split.py`) is the reference; nothing here
replaces it.

## 1. Three levels of data

| Level | Made by | S1 / pool (approx.) | Purpose |
|---|---|---|---|
| **val** | `split.load_fold("val")`: `VAL_FRACTION = 0.2`, `SPLIT_SEED = 42` | 441k / 2.06M | the one reported score |
| **tune** | `trainset.inner_split(train)[1]`: `INNER_FRAC = 0.25`, `INNER_SEED = 4242` | 441k / 2.06M; 100k S1 sampled | early stopping, calibration, threshold grid |
| **fit** | `trainset.inner_split(train)[0]`: the other 75% | 1.32M / 6.2M; 200k S1 sampled | model training, hard-negative mining |

### 1.1 The fixed validation fold (exists, frozen)

- Hash-based, a pure function of ids: an S1 entity is in val when `hash_unit(id, 42) < 0.2`;
  a matched S2/S3 record follows its S1 entity (`pool_val_mask`); an unmatched pool record
  is hashed with the same fraction. Row order, machine and sampling state cannot change it;
  `val_ids` is cached under `dataset/.cache/`.
- **Never change `VAL_FRACTION` or `SPLIT_SEED`.** A new split makes every row already in
  `experiments/experiments.csv` incomparable (project rules §4).
- The val pool holds the true matches of val S1 entities plus the val share of unmatched
  records (26% of the pool, 01 §2). Records matched to *other* val S1 entities are the
  natural decoys that make singletons and same-name entities hard: never drop them, never
  restrict the pool to "records with a candidate".
- It is used **only** to score a frozen pipeline (§2). Nothing is fitted, tuned or chosen
  on it.

### 1.2 The inner split (`trainset.inner_split`, owner M1)

The same construction one level down, keyed with `INNER_SEED` so it is independent of the
outer hash. `pool_val_mask` is reused verbatim, so the invariants of the outer split hold:
every true pair sits with its S1 entity, a pool record is on exactly one side, singletons
and unmatched decoys land on both sides in their natural share.

```python
INNER_SEED, INNER_FRAC, SAMPLE_SEED = 4242, 0.25, 7

def inner_split(train: Fold) -> tuple[Fold, Fold]:
    """Split the train fold into (fit, tune) exactly the way load_fold splits train/val."""
    s1_ids = train.s1[C.ENTITY_ID]
    tune_s1 = hash_unit(s1_ids, INNER_SEED) < INNER_FRAC           # S1 by id hash
    tune_ids = pd.Index(s1_ids[tune_s1])

    def tune_side(pool: pd.DataFrame) -> np.ndarray:
        # matched pool row -> the side of its S1 owner (train.pairs); unmatched -> hash < 0.25
        return pool_val_mask(pool[C.ENTITY_ID], train.pairs, tune_ids, INNER_FRAC, INNER_SEED)

    def part(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
        return df[mask].reset_index(drop=True)

    m2, m3 = tune_side(train.s2), tune_side(train.s3)
    in_tune = isin(train.pairs[C.S1_ID], tune_ids)                 # pairs follow their S1
    tune = Fold("tune", part(train.s1, tune_s1), part(train.s2, m2), part(train.s3, m3),
                part(train.pairs, in_tune))
    fit = Fold("fit", part(train.s1, ~tune_s1), part(train.s2, ~m2), part(train.s3, ~m3),
               part(train.pairs, ~in_tune))
    return fit, tune
```

`pipeline.fit` then takes `sample_s1(fit.s1, 200_000, seed=7)` and `sample_s1(tune.s1,
100_000, seed=7)` and keeps both pools whole. Whole pools give every sampled entity at least
the decoy density of test: the matches of unsampled S1 entities stay in the pool as unowned
records, which is what test's ≈ 40% unmatched pool looks like, and `label_pairs` correctly
labels them 0 for every sampled entity.

## 2. What each level is for

| Level | Allowed | Forbidden |
|---|---|---|
| fit | `Matcher.fit`; hard negatives (`mine_false_positives` on fit-side scored pairs, round 2); learned token maps (`fit_region_map`, `fit_token_map`) may use the whole train fold's pairs | reading tune or val labels |
| tune | LightGBM early stopping (`X_val, y_val`), probability calibration, `decision.tune` grid → `DecisionRule`, `tune_f_beta` | selecting features by tune F0.5 over many rounds without confirming on val |
| val | `run_fold(cfg, fitted, val)` once per version → `score_pairs` (the keys of `metrics.breakdown`), `blocking_report`, `slice_report`, `tag_errors` | any parameter, threshold or feature choice; re-running after "one more tweak" |

"One run per version, frozen rule": `Fitted.rule` is fixed by `decision.tune` on the tune
split *before* val is loaded. If val must be re-run (a bug in the pipeline, not in the
idea), the notebook says so under `notes`. A version re-tuned after seeing its val score is
a new version number.

## 3. Leakage rules

Copy this checklist into the notebook's conclusion and tick it.

- [ ] Vectorisers (TF-IDF vocabularies, n-grams) are unsupervised and **refit per split**
      (fit, tune, val, test) on that split's S1 ∪ pool; never fitted on one split and reused
      on another.
- [ ] Learned token maps (`fit_region_map`, `fit_token_map`) are fitted on the **train
      fold's pairs only** (`load_fold("train").pairs`), saved to `artifacts/`, then applied
      to every split as a static map.
- [ ] No feature reads a label: `build_features(pairs, s1n, pooln)` never sees `truth`;
      `ctx_*` features use candidates, not matches.
- [ ] No threshold, cap, calibration or early-stopping round comes from val; `Fitted.rule`
      and `Fitted.tune_table` exist before `run_fold(val)` is called.
- [ ] The val pool is complete: records matched to other val S1 entities (decoys) are kept;
      no filter by candidate presence or name overlap.
- [ ] Hard negatives are mined on fit, not on tune or val.
- [ ] `harder_fold` (§6) drops S1 entities only; its pool is the val pool untouched.
- [ ] The commit logged by `log_result` is not `-dirty`: the score is reproducible.

## 4. The metric, as implemented

`src/entity_resolution/metrics.py` is the reference implementation (project rules §4).
Per entity, verbatim:

```python
def entity_fbeta(pred: Collection[str], truth: Collection[str], beta: float = BETA) -> float:
    pred, truth = set(pred), set(truth)
    if not truth:
        return 0.0 if pred else 1.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(pred), tp / len(truth)
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)
```

Rules encoded there (01 §8): a singleton scores 1.0 for an empty prediction and 0.0 for any
prediction; an entity with matches and no true positive scores 0.0; otherwise F0.5 of that
entity's precision and recall. `macro_fbeta` averages over **every** S1 entity of `truth`,
singletons included, entities missing from `pred` counting as empty. The brief's example
(`{S2-00047, S2-00193, S3-00812}` vs `{S2-00047, S3-00812}` → 5/7 = 0.714) is
`tests/test_metrics.py::test_problem_statement_example`.

| Function | Keys |
|---|---|
| `metrics.breakdown(pred, truth)` | `f_beta`, `f_beta_singletons`, `f_beta_matched`, `pair_precision`, `pair_recall`, `entities`, `singletons` |
| `metrics.candidate_report(candidates, truth, pool_size)` | `pair_recall`, `entity_recall`, `ceiling_f_beta`, `candidates_mean`, `candidates_p95`, `candidates_max`, `candidate_pairs`, `reduction_ratio` |

Both take `dict[str, Collection[str]]` and loop in Python: fine on val (441k entities, a
few seconds) through `pairs_to_lists`, never on frames of test size. `evaluate.score_pairs`
and `evaluate.blocking_report` produce the same keys from pairs frames with vectorised
counts; the vectorised twin of the metric is:

```python
def entity_f05_from_counts(tp, n_pred, n_true) -> np.ndarray:
    """Per-entity F0.5 from count arrays: the formula of metrics.entity_fbeta (10 §4)."""
    tp, n_pred, n_true = (np.asarray(a, dtype=np.float64) for a in (tp, n_pred, n_true))
    with np.errstate(divide="ignore", invalid="ignore"):
        p, r = tp / n_pred, tp / n_true            # NaN/inf where a count is 0: masked below
        f = 1.25 * p * r / (0.25 * p + r)
    return np.where(n_true == 0, (n_pred == 0) * 1.0, np.where(tp == 0, 0.0, f))


def macro_f05_from_counts(tp, n_pred, n_true) -> float:
    """Mean over ALL S1 entities of the fold, one element each; fsum like statistics.fmean."""
    f = entity_f05_from_counts(tp, n_pred, n_true)
    return math.fsum(f) / len(f)
```

In words: P = tp / n_pred, R = tp / n_true, F = 1.25·P·R / (0.25·P + R); `n_true == 0` →
F = (n_pred == 0); `tp == 0` → 0; mean over all S1 entities. `decision.tune`,
`decision.evaluate_rules` and `errors.py` call these same two functions (10 §4), so tuning,
evaluation and error tagging cannot disagree.

Counts come from three `groupby(source1_entity_id).size()` calls reindexed on `fold.s1`
ids with `fill_value=0`: `n_true` from `fold.pairs`, `n_pred` from
`pred_pairs.drop_duplicates()` (set semantics), `tp` from the inner merge of the two.
`tests/test_evaluate.py::test_macro_f05_from_counts_equals_macro_fbeta` proves equality
with `metrics.macro_fbeta` on random data with singletons, duplicates and absent entities.

## 5. Pair construction for training and evaluation

```
candidates = block(s1n, pooln, cfg)          PAIR_COLUMNS
positives  = candidates ∩ fold.pairs         label 1   (trainset.label_pairs, 07 §5)
negatives  = candidates \ fold.pairs         label 0   (every other candidate)
misses     = fold.pairs \ candidates         not trainable; counted by the metric
```

`label_pairs` is an inner-merge indicator on both ids, `label` int8. Positives are only
truth pairs *inside* the candidate set: a true pair that blocking never produced cannot be
learned or predicted, so it costs `pair_recall` and `f_beta` but never a training row. That
is why `ceiling_f_beta` (a perfect matcher on these candidates) is reported next to every
model score: `ceiling_f_beta − f_beta` is the model's gap, `1 − ceiling_f_beta` is
blocking's. Expected sizes (07 §5): 200k fit S1 → ≈ 5–6M pairs, ≈ 12% positive; 100k tune
S1 → ≈ 3M pairs. Sample S1 entities, never pairs, so every decoy of a sampled entity is
present and the group context matches inference.

## 6. The harder fold (`evaluate.harder_fold`)

Test has 5.8 pool records per S1 entity against 4.7 in train, so ≈ 40% of its pool is
unmatched against 26% (01 §2): val alone under-estimates false merges.
`harder_fold(val, drop_frac=0.2, seed=99)` drops 20% of val S1 entities
(`hash_unit(s1_ids, 99) < 0.2`) **but keeps their pool rows**: their true matches become
unowned records, exactly the decoys test adds.

```python
def harder_fold(fold: Fold, drop_frac: float = 0.2, seed: int = 99) -> Fold:
    keep = ~(hash_unit(fold.s1[C.ENTITY_ID], seed) < drop_frac)
    kept_ids = pd.Index(fold.s1[C.ENTITY_ID][keep])
    pairs = fold.pairs[isin(fold.pairs[C.S1_ID], kept_ids)]
    return Fold("harder", fold.s1[keep].reset_index(drop=True), fold.s2, fold.s3,
                pairs.reset_index(drop=True))
```

Arithmetic: 441k → 353k S1 entities over the same 2.06M pool is 5.8 records per entity;
the unmatched share of the pool rises from 26% to 26% + 0.2 × 74% ≈ 41%. Every version
reports both `f_beta` (val) and `harder_f_beta`; a version whose harder-val drops while val
rises is buying recall with false merges and will lose on test. Run it as
`run_fold(cfg, fitted, harder_fold(val))` with tag `harder`: the pool normalisation is
identical to `val` and comes from the cache; only the kept S1 groups are re-blocked,
featured (`ctx_pool_indegree` changes) and scored, ≈ 80% of the val cost.

## 7. Slice reports (`evaluate.slice_report(pred_pairs, fold, s1n, pooln)`)

One frame, one row per slice, columns `family`, `slice`, `entities`, `f_beta`,
`pair_precision`, `pair_recall`, `n_true`, `n_pred`, `tp`. Every family that partitions
the entities also gets an `all` row, so rows can be checked against totals.

| Family | Slices | Level |
|---|---|---|
| `country` | one per value present (never an enumerated list) | entity |
| `source` | `S2`, `S3`: pred and truth projected to that source's ids; entities with a true or predicted pair there | entity, projected |
| `non_latin` | entity has ≥ 1 true match whose pool row has `non_latin` | entity |
| `domain_form` | entity has ≥ 1 true match whose pool name was a domain/handle form (R3 fired). Needs a `domain_form` bool next to `non_latin`, an added column 05 §11 allows; ask M2 | entity |
| `ambiguous` | S1 `name_core` shared by > 1 S1 record of the same country | entity |
| `singleton` | `singleton`, `matched` | entity |
| `n_matches` | `0`, `1`, `2`, `3-4`, `5+` true matches | entity |
| `postcode` | S1 `postcode != ""` vs `""` | entity |
| `addr_empty` | S1 `addr_tokens == 0`, or ≥ 1 true match with an empty pool address | entity |

Each row's `f_beta` is `macro_f05_from_counts` over the entities of the slice; pair P/R are
sums over those entities. The KEEP rule (13 §3) and the PR checklist (14 §4) compare the
`country`, `singleton`, `n_matches`, `non_latin`, `ambiguous` and `addr_empty` families
against the parent version; the other families are diagnostics. For blocking experiments the same families apply to candidates
(06 §5): `slice_report(cand_pairs, ...)` with candidates in place of predictions gives
per-slice `pair_recall`, which is per-slice candidate recall.

## 8. Multi-match and singleton semantics in evaluation

- Per-entity **set** semantics: a prediction is the set of pool ids for that S1 id;
  duplicates collapse (`drop_duplicates` in `score_pairs`, `dict.fromkeys` in the writer).
- An S1 entity with no row in `pred_pairs` is an empty prediction: 1.0 if it is a singleton,
  0.0 otherwise. `pairs_to_lists(pairs, s1_ids)` materialises every S1 id with `[]` when
  absent, so `breakdown` and `score_pairs` see the same thing.
- Rows for S1 ids outside the fold are ignored, as `breakdown` ignores them. A predicted
  pool id that is not a fold record is a false positive, never dropped silently;
  `score_pairs` logs a warning with the count (it means the wrong split was scored).
- The 1-to-1 property (01 §1) is enforced by `decision.decide`, not by the metric: the
  metric counts the same pool id under two entities, and at most one of them is right.
- `max_matches = 11` in `DecisionRule` mirrors the maximum in train; the metric has no cap.

## 9. Per-version evaluation protocol

Every notebook's §5–§7 (template) runs exactly this, in order:

1. `fitted = pipeline.fit(cfg, train, EXP_DIR / "artifacts")`: fit and tune sides only;
   record `tune_f_beta` (best row of `Fitted.tune_table`) and `rule` (`Fitted.rule`).
2. `metrics, cands, matches = pipeline.run_fold(cfg, fitted, val)`, once.
3. `blocking_report(cands, val)` → `cand_recall`, `cands_mean`, `cands_p95`,
   `ceiling_f_beta`. If `cand_recall < 0.97` the version is a blocking experiment,
   whatever it was meant to be.
4. `score_pairs(matches, val)` → `f_beta`, `f_beta_singletons`, `f_beta_matched`,
   `pair_precision`, `pair_recall`; the decision rule stays as tuned in step 1 (frozen).
5. `run_fold(cfg, fitted, harder_fold(val))` → `harder_f_beta`.
6. `slice_report(matches, val, s1n, pooln)`: print it, keep it as `artifacts/slices.csv`,
   name the worst slice in the conclusion.
7. `errors.tag_errors(matches, val, s1n, pooln, scored)` → `n_fp`, `n_fn`,
   `n_false_singleton`, `errors` (count per category of `18_ERROR_ANALYSIS_FRAMEWORK.md`),
   `artifacts/errors.parquet`; `error_samples(kind=...)` for the top category.
8. `tracking.log_result(EXP_DIR, change=..., group=..., local_f05=metrics["f_beta"],
   cand_recall=metrics["cand_recall"], notes=..., metrics=metrics)`.

9. Judge KEEP / DROP / INVESTIGATE against the parent version (13 §3) and record it as
   `decision`.

Standard `metrics` dict: the keys of `13_EXPERIMENT_TRACKING.md` §2.2. Add a key when a
stage is new; never rename one or add a synonym.

| Group | Keys | Source |
|---|---|---|
| identity | `hypothesis`, `blocking_config`, `feature_groups`, `model_params`, `rule` | notebook header; `asdict(cfg.blocking)`, `list(cfg.feature_groups)`, `asdict(cfg.model)`, `asdict(fitted.rule)` |
| val scores | `f_beta`, `f_beta_singletons`, `f_beta_matched`, `pair_precision`, `pair_recall` | `score_pairs(matches, val)` |
| blocking | `cand_recall`, `entity_recall`, `ceiling_f_beta`, `cands_mean`, `cands_p95` | `blocking_report(cands, val)`: its `pair_recall` becomes `cand_recall` and `candidates_*` become `cands_*`, so the matcher's `pair_recall` is not overwritten |
| robustness | `harder_f_beta`, `tune_f_beta` | `score_pairs` on `harder_fold(val)`; best row of `Fitted.tune_table` (column `f_beta`, as in 10 §4 and 13 §2.2) |
| errors | `n_fp`, `n_fn`, `n_false_singleton`, `errors` | `tag_errors`: false predicted pairs, true pairs not predicted, matched entities predicted empty, count per category of doc 18 |
| timings | `load_seconds`, `normalise_seconds`, `blocking_seconds`, `features_seconds`, `fit_seconds`, `tune_seconds`, `score_seconds`, `decide_seconds` | `tracking.timed` with exactly these eight labels; `run_fold` returns the same names |
| resources, verdict | `peak_rss_gb`, `decision` | max `psutil` RSS seen by `mem_guard` (02 §7); KEEP / DROP / INVESTIGATE |

## 10. Test-set sanity checks before an upload

Test has no labels, so `run_test` output is checked against what val taught us. All of
these go into the notebook's §9 and into the `LEADERBOARD.md` entry.

| Check | Expect |
|---|---|
| rows in `matching_results.tsv` and `candidate_pairs.tsv` | exactly **1,732,544**, one per test S1 id, in `test_source1.tsv` order |
| countries in the output | every value of test S1 `country`, France included, each with candidates and matches |
| per country: share of S1 with ≥ 1 match, matches per matched S1 | within a few points of val (94% matched, 3.7 per matched entity); France inside the US–India range |
| per country: candidates per S1 (mean, p95) | ≈ 1.2× val (pool/S1 5.8 vs 4.7); most entities below the `max_per_s1` cap |
| `p_max` histogram per country (best prob per S1) | bimodal like val; mass below `tau_single` ≈ the singleton share seen on harder-val |
| matches ⊆ candidates | `submission.validate` gives no warning (both files from the same `pairs` frame) |
| `make validate` | our checker and the organisers' validator both `PASS`; run `--check-ids` once per blocking config |
| ids | no `S1-` id in any list, no duplicates, no empty `source1_entity_id` |
| provenance | `experiments.csv` row with a clean commit; `artifacts/rule.json` equals the rule used |

## 11. How many local experiments per upload

Fifteen uploads over three days cannot explore (01 §11). Expect **10–20 local versions per
upload**; the csv is where exploration happens (13 §5). A version qualifies for upload when
all of these hold:

1. It is an integrated `v1xx` version judged KEEP against its parent (13 §3: `f_beta` >
   parent + 0.002, no slice down by more than 0.01, `harder_f_beta` not lower).
2. `f_beta` on val ≥ the last uploaded version's `f_beta` + **0.003**. The standard error of
   the macro mean over 441k entities is ≈ 0.0005, so 0.003 is a real gain, not noise.
3. `harder_f_beta` does not regress against the last uploaded version (13 §5 tolerates
   −0.002 for run-to-run noise, nothing more).
4. `cand_recall ≥ 0.97`, `make validate` passes, the §10 table is filled in; notebook,
   `metrics.json` and csv row committed, `commit` not `-dirty`.

Among qualifiers the highest `f_beta` wins; ties go to the higher `f_beta_singletons`, since
the test pool has more unmatched records than train. Doubts go to the lead. Record the
public score with `make public V=vNNN SCORE=...`; trust local over public
(`15_LEADERBOARD_STRATEGY.md`).

## 12. Tests

`tests/test_trainset.py`, inner split (the rest of the file is 07 §7). The synthetic Fold
is built in memory: 4,000 S1 ids, 0–5 matches each, 30% unmatched pool records, 5%
singletons, `rng = np.random.default_rng(0)`.

| Test | Assertion |
|---|---|
| `test_inner_split_sides_are_disjoint` | fit and tune share no `entity_id` in s1, s2 or s3; their union is the input |
| `test_inner_split_pairs_never_cross_sides` | `tune.pairs` S1 ids ⊆ `tune.s1`, pool ids ⊆ tune pool; same for fit |
| `test_inner_split_matched_pool_follows_owner` | a pool record whose owner is on tune is on tune even when its own hash says fit (assert on a found example, `hash_unit(id, 4242) ≥ 0.25`) |
| `test_inner_split_is_deterministic` | shuffled input rows give equal id sets on both sides; two calls equal |
| `test_inner_split_share` | tune share of S1 ≈ 0.25 ± 0.02; unmatched-pool tune share ≈ 0.25 ± 0.03 |
| `test_inner_split_on_fixture` | `load_fold("train", dataset_dir, frac=0.5)` splits without error, parts keep `SOURCE_COLUMNS`; fit side is empty (12 §1), tune holds `S1-00002` |

`tests/test_evaluate.py`:

| Test | Assertion |
|---|---|
| `test_macro_f05_from_counts_equals_macro_fbeta` | 500 random entities (0–4 true matches, random predictions with duplicates, singletons, absent entities): `score_pairs(...)["f_beta"] == metrics.macro_fbeta(...)` to 1e-12, every `breakdown` key equal |
| `test_counts_edge_cases` | `(tp, n_pred, n_true)`: (0,0,0)→1, (0,1,0)→0, (0,0,2)→0, (2,3,2)→5/7, (1,1,1)→1 |
| `test_entity_f05_from_counts_matches_entity_fbeta` | the per-entity array equals `metrics.entity_fbeta` element-wise on the same random data (bitwise, `pairs_toy` included) |
| `test_pairs_to_lists_every_s1_present` | absent S1 → `[]`; duplicates collapsed; order of `s1_ids` kept |
| `test_pair_recall` | 2 of 3 truth pairs among candidates → 2/3; no truth → NaN |
| `test_blocking_report_matches_candidate_report` | same numbers as `metrics.candidate_report` on the fixture's val fold |
| `test_harder_fold_keeps_pool_drops_s1` | `len(h.s1) < len(val.s1)`, `h.s2.equals(val.s2)`, `h.s3.equals(val.s3)`, pairs only for kept S1, `name == "harder"`, deterministic |
| `test_harder_fold_share` | 4,000 synthetic S1: dropped share ≈ 0.2 ± 0.02 |
| `test_slice_report_rows_sum_to_totals` | per partitioning family: `entities` sum to `len(fold.s1)`; `tp`, `n_pred`, `n_true` sum to the `all` row; `Σ f_beta × entities == f_beta_all × entities_all` |
| `test_slice_report_country_open_set` | a fold with countries `{"US", "France"}` gives exactly those two rows |
| `test_slice_report_source_projection` | an entity with one S2 and one S3 truth pair, S2 predicted: `S2` row P = R = 1, `S3` row R = 0 |
| `test_score_pairs_foreign_ids` | rows for S1 ids outside the fold are ignored; an unknown pool id counts as a false positive and warns |
| `test_error_samples_missed` | `kind="missed"` lists exactly the truth pairs absent from predictions, both records side by side |

Everything runs on in-memory frames or the `dataset_dir` fixture; no test reads
`dataset/` (12 §1).
