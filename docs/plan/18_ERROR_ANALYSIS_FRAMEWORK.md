# 18 — Error analysis framework (module `errors.py`, owner M5)

What you build: `tag_errors(pred_pairs, fold, s1n, pooln, scored=None) -> DataFrame`
(`02_SYSTEM_ARCHITECTURE.md` §5), one row per wrong or missing pair with an error type, a
category with an owner, and its share of the lost macro F0.5; plus `summarise`, `samples`
and `to_metrics`, which §6 of every experiment notebook runs. Error categories are what
decides what the next version tries.

## 1. Why categories, not counts

01 §3 measured where the difficulty is: 14.7% of true pairs share no name token (of those,
47% have the pool name in an Indic script, 35% are domain/handle forms, 18% renames or
heavy typos); 48% of S1 core names are shared with another S1 record; 62% of singletons
have a name twin in the pool; singletons are 5.6% of entities and each one is worth a full
point. A version that loses 0.01 macro F0.5 can lose it on any of these, and the fix belongs
to a different module each time. The frame built here names the pattern, the owner and the
F0.5 at stake, so the next experiment is chosen by impact, not by anecdote.

## 2. `tag_errors` contract

```python
def tag_errors(pred_pairs, fold: Fold, s1n, pooln, scored=None) -> pd.DataFrame
```

Inputs: `pred_pairs` (`MATCH_COLUMNS`, the output of `decision.decide`), the scored fold
(`split.Fold`: raw records and truth pairs), the normalised frames of that fold
(`NORM_COLUMNS`, 02 §4.1) and, when available, the scored candidates (`SCORED_COLUMNS`) with
`scored.attrs["rule"]` set by `pipeline.run_fold` (10 §13). `tau_abs` comes from that
attribute, or is estimated as the smallest `prob` of a predicted pair when it is missing.

Output columns, one row per erroneous pair:

| Column | Meaning |
|---|---|
| `source1_entity_id`, `entity_id` | the pair |
| `error_type` | `false_merge` (predicted, not true; entity has matches), `missed` (true, not predicted; entity predicted something), `false_singleton` (true pair of an entity whose prediction is empty), `missed_singleton` (predicted pair of a true singleton) |
| `category` | one of the 18 labels of §3 by precedence, else `other` |
| `sub_flag` | `domain_form` (dba_trade_name), `cands_present` / `cands_absent` (false_singleton), else `""` |
| `stage` | where the pair was lost, independent of `category` (§3.1) |
| `prob` | matcher probability; NaN if the pair was not scored (not a candidate, or `scored=None`) |
| `in_candidates` | pair present in `scored` (NA when `scored=None`) |
| `country`, `source` | the S1 country; `S2` / `S3` from the pool id prefix |
| `entity_f05` | the entity's F0.5 under this prediction (`evaluate.entity_f05_from_counts`, 10 §4) |
| `impact` | `(1 − entity_f05) / (n_entities · rows of this entity)`: the entity's loss shared equally over its rows |
| `suspicious_label` | §8: the pair looks right although the label says otherwise |
| `name_l`, `addr_l`, `name_r`, `addr_r` | raw `business_name` / `business_address` of both records, for the samples |
| feature columns | the 07 features the rules read (`core_ratio`, `core_token_set`, `tok_jaccard`, `tok_len_l`, `tok_len_r`, `ad_token_set`, `ad_ratio`, `ad_contain`, `ad_jaccard`, `postcode_eq`, `non_latin_r`, `squash_ratio`) plus the extras of §3.2 |

The four error types partition every wrong pair, so an entity with `entity_f05 < 1` always
has at least one row and Σ `impact` over all rows = 1 − macro F0.5 (§7).

Algorithm (vectorised; the frame is ≤ 1% of the pairs, so the per-row string helpers of
§3.2 are the only Python-level loops allowed in this module):

```python
truth = fold.pairs[MATCH_COLUMNS]; pred = pred_pairs[MATCH_COLUMNS].drop_duplicates()
pred = pred[isin(pred[C.S1_ID], fold.s1[C.ENTITY_ID])]                  # foreign S1 ids: warn, drop
m = truth.merge(pred, how="outer", indicator=True)                      # both | left_only | right_only
n_true, n_pred, tp = bincounts over fold.s1 positions → entity_f05 = entity_f05_from_counts(...)
err = m[m["_merge"] != "both"]; error_type from (side, n_pred == 0, n_true == 0) with np.select
join scored on the pair → prob, in_candidates; join s1n / pooln → normalised columns; join raw text
X = features.build_features(err[MATCH_COLUMNS], s1n, pooln,
                            groups=("name_fuzzy", "name_tokens", "legal", "numeric", "address", "meta"))
extras of §3.2; category = np.select(conditions in §3 order, CATEGORIES, default="other")
stage (§3.1), sub_flag, impact, suspicious_label (§8); return err[ERROR_COLUMNS]
```

The `blocking` and `context` feature groups are skipped: missed pairs outside the candidate
set have no `pass` / `sim_*` columns and no group context.

## 3. The 18 categories

Evaluate in this order; the first true condition is the category. `L` = S1 record, `R` = pool
record; feature names are those of 07 §2, `addr_tokens_l/r` the 05 column on each side;
`miss` = `error_type ∈ {missed, false_singleton}`, `false` = `error_type ∈ {false_merge,
missed_singleton}`. Owners: M2 normalisation/blocking, M3 features, M4 model, M5 decision.

| # | Category | Condition | Owner | Targeted improvement |
|---|---|---|---|---|
| 1 | `name_typo` | miss ∧ 0.6 ≤ `core_ratio` < 0.95 ∧ `tok_len_l == tok_len_r` | M2 / M3 | not a candidate: P2 `min_sim` 0.3 → 0.2 or 4-grams; a candidate: `core_jw`, `squash_ratio`, leet map in `name_squash` |
| 2 | `name_abbreviation` | miss ∧ `abbrev(L.name_core, R.name_core)` (§3.2) | M2 | B3 abbreviation map; learned name token map (05 §6) |
| 3 | `legal_suffix` | miss ∧ `L.name_core == R.name_core` ∧ `L.legal_form != R.legal_form` | M2 | B2 legal map (missing or French forms); `legal_eq` must not dominate the model |
| 4 | `word_order` | miss ∧ `L.name_sorted == R.name_sorted` ∧ `L.name_core != R.name_core` | M2 / M3 | P1b key present; `nm_token_sort` weight |
| 5 | `dba_trade_name` | miss ∧ `tok_jaccard == 0` ∧ `non_latin_r == 0`; `sub_flag = domain_form` when `domain_form(R)` or `squash_ratio ≥ 0.9` | M2 / M3 | P3/P4 address channel; `name_squash` for domain forms; address features must carry the pair |
| 6 | `address_abbreviation` | miss ∧ `ad_token_set ≥ 0.9` ∧ `ad_ratio_raw < 0.7` | M2 | B5 street/abbreviation map; check it is applied before blocking |
| 7 | `missing_address_component` | miss ∧ `max(ad_contain, ad_contain_r) ≥ 0.9` ∧ `|addr_tokens_l − addr_tokens_r| ≥ 2` | M3 / M2 | containment and `num_first_eq` features; P3 word channel |
| 8 | `landmark_address` | miss ∧ `LANDMARK_RE` matches `L.addr_norm` or `R.addr_norm` ∧ `ad_token_set < 0.9` | M2 / M3 | B5 landmark-phrase stripping; `addr_last` / `num_*` features |
| 9 | `transliteration` | miss ∧ `non_latin_r == 1` | M2 | learned token map (05 §6), P4 address char-grams; anyascii coverage |
| 10 | `postal_code_mismatch` | miss ∧ both postcodes non-empty ∧ `postcode_eq == 0` | M3 / M4 | `postcode_eq` as weak evidence (the generator moves PINs); decoy negatives with equal postcode |
| 11 | `same_name_different_business` | false ∧ `core_token_set ≥ 0.9` ∧ `ad_jaccard < 0.2` | M3 / M4 / M5 | address features; decoy negatives (`trainset.mine_false_positives`); higher `tau_abs` |
| 12 | `same_address_different_business` | false ∧ `ad_token_set ≥ 0.9` ∧ `core_token_set < 0.5` | M3 / M4 | name features on the `addr_strong_name_weak` slice; hard negatives from shared buildings |
| 13 | `false_singleton` | `error_type == false_singleton`; `sub_flag` = `cands_present` if `in_candidates` else `cands_absent` | M5 (present) / M2 (absent) | lower `tau_single` / `tau_rel`; a new blocking pass |
| 14 | `missed_multi_match` | miss ∧ entity `tp ≥ 1` (partially recalled) | M5 / M4 | `tau_rel`, `max_matches`; more many-match positives in training |
| 15 | `blocking_failure` | miss ∧ `in_candidates == False` | M2 | new pass, lower `min_sim`, higher `max_per_s1` |
| 16 | `model_failure` | (miss ∧ `prob < tau_abs − 0.2`) ∨ (false ∧ `prob ≥ 0.9`) | M4 | features for the slice, hard negatives, retrain |
| 17 | `threshold_failure` | (miss ∧ `tau_abs − 0.2 ≤ prob < tau_abs`) ∨ (false ∧ `tau_abs ≤ prob < tau_abs + 0.1`) | M5 | retune; `tau_rel`; expected-F0.5 decoding (10 §7) |
| 18 | `cross_country` | `L.country != R.country` | M2 | must be zero: a blocking partition bug; the notebook asserts it |

`other` is the residual (a false merge with mid prob and no decoy signature; a missed pair
with `prob ≥ tau_abs` lost to 1-to-1 or the cap). If `other` exceeds 10% of the impact, a
category is missing and this table gets a new row.

### 3.1 `stage` (computed independently of the precedence)

| Row | `stage` |
|---|---|
| miss, not in candidates | `blocking` |
| miss, `prob < tau_abs − 0.2` | `model` |
| miss, `tau_abs − 0.2 ≤ prob < tau_abs` | `threshold` |
| miss, `prob ≥ tau_abs` | `decision` (1-to-1, `tau_rel`, `tau_single` or the cap removed it) |
| false, `prob ≥ 0.9` / `tau_abs ≤ prob < tau_abs + 0.1` / otherwise | `model` / `threshold` / `mid` |
| `scored=None` | `unknown` |

`category` says *why* the pair was hard and `stage` says *where* it was lost, so the §4
report cross-tabulates the two: `transliteration × blocking` goes to M2, `transliteration ×
model` to M3/M4.

### 3.2 Extra columns computed in `errors.py`

- `ad_ratio_raw`: `fuzz.ratio` on `normalize.basic_norm(business_address)` of both sides
  (before abbreviation expansion): the "before" side of category 6.
- `ad_contain_r`: share of R tokens present in L (07's `ad_contain` is L-in-R).
- `abbrev(a, b)`: the tokens of the shorter core name are, in order, prefixes (≥ 2 chars) of
  tokens of the longer one, or one side is a single token equal to the initials of the
  other's tokens (`sbi` ↔ `state bank of india`, `intl` ↔ `international`).
- `domain_form(R)`: raw name matches `^[@#]|\.(com|in|net|org|co|io)\b`, or has no spaces
  and `squash_ratio ≥ 0.9` (`ORTHOPEDICHEALTHCOM`).
- `LANDMARK_RE = r"\b(near|nr|opp|opposite|behind|beside|next to|adjacent to|in front of)\b"`.

## 4. The per-version report

`summarise(err, n_entities, parent=None) -> DataFrame`: one row per category (and
`other`), sorted by impact: `count`, `share` of rows, `impact` = Σ `impact` of the
category's rows (the macro F0.5 lost to it), `impact_share` = impact / (1 − macro F0.5),
then `count` / `impact` per `country` and per `source`, the impact per `stage`
(`blocking`, `model`, `threshold`, `decision`), and `delta_impact` against the parent
version's `metrics.json["errors"]` when `parent` is given. `samples(err, category=None,
n=20, seed=C.SEED) -> DataFrame`: 20 rows per category with `name_l | addr_l | name_r |
addr_r | prob | error_type | stage | sub_flag`, both records side by side, drawn with
`sample(random_state=seed)` and shown worst `entity_f05` first. `to_metrics(summary) ->
dict` gives `{category: {"count": int, "impact": float}}` plus `"cross_country"` and
`"suspicious_label"` counts.

Storage: the full frame in `artifacts/errors.parquet` (gitignored, regenerated by the
notebook); the summary and the samples printed in the notebook's §6 (outputs committed);
the `to_metrics` dict in `metrics.json` under `"errors"` via `tracking.log_result(...,
metrics={..., "errors": ...})`, so neighbouring versions in `experiments.csv` can be
compared without reopening notebooks. The top three categories by impact go into the
`notes` column of the experiments row.

## 5. Notebook §6 template (8 lines: 2 markdown + 6 code; every experiment runs it)

```markdown
## 6. Error analysis
Categories by F0.5 impact (`errors.tag_errors`); the top category names the next experiment's hypothesis.
```

```python
from entity_resolution import errors
err = errors.tag_errors(matches, val, s1n, pooln, scored=scored_val)            # one row per wrong/missing pair
err.to_parquet(ARTIFACTS / "errors.parquet", index=False); assert (err["category"] == "cross_country").sum() == 0
summary = errors.summarise(err, n_entities=len(val.s1), parent=parent_errors); display(summary)   # count, impact, slices
display(errors.samples(err, category=summary.index[0], n=20))                      # both records side by side
scores["errors"] = errors.to_metrics(summary)                                      # lands in metrics.json under "errors"
```

`matches` comes from `pipeline.run_fold` (§5 of the notebook), `s1n` / `pooln` from
`pipeline.prepare(val.s1, pool(val), cfg, "val")` (cached), `scored_val` from
`artifacts/scored_val.parquet` (written by `run_fold`, 02 §8) with `attrs["rule"]` restored
from `rule.json`; `parent_errors` is the parent version's `metrics.json["errors"]` or
`None`; `scores` is the dict passed to `log_result`. Group A versions (blocking only) run the
same cell with the `heuristic` matcher so `blocking_failure` is measured on the same frame.

## 6. From error counts to the next experiment

1. Sort the summary by `impact`, not `count`: 200 missed pairs on 6-match entities cost less
   than 40 false merges on singletons.
2. One category per experiment. The hypothesis cell of the next version names it:
   "Category: `transliteration` × `blocking`, impact 0.012 in v041; change: learned token
   map; expected: impact ≤ 0.006, macro F0.5 +0.005".
3. The follow-up is kept under the usual rule (val macro F0.5 +0.002) **and** its §6 shows
   the targeted category's impact falling; a version that fixes one category by inflating
   another is discarded, and both impacts are quoted in the conclusion.
4. `delta_impact` against the parent is the regression check: any category whose impact
   rises by more than 0.002 is named in the conclusion even when the total improved.
5. Group E reads the `stage` split: `threshold` and `decision` impact is M5's ceiling;
   `blocking` impact cannot be touched by the decision layer and is handed to M2.

## 7. Tests (`tests/test_errors.py`)

Hand-built pairs on the synthetic dataset of `tests/conftest.py` (three train S1 entities,
one singleton, France only in test) plus a few extra rows per category.

| Test | Assertion |
|---|---|
| `test_error_types_partition` | on a prediction with one of each mistake: exactly one row per wrong pair, each with the expected `error_type`; a perfect prediction gives an empty frame |
| `test_each_category_triggers` | one hand-built pair per category (`Sharma Traders` ↔ `Sharma Tradres`; `intl` ↔ `international`; `Pvt Ltd` ↔ `Private Limited`; `Traders Sharma`; `clumora.com`; `Rd` ↔ `Road`; dropped `Pune 411001`; `Near SBI ATM`; `शर्मा ट्रेडर्स`; PIN 411001 vs 411002; same name at another address; same address with another name; empty prediction; partial recall; a pair outside `scored`; `prob` 0.1 / 0.45 with `tau_abs = 0.5`; a forged US–India pair) gets exactly that category |
| `test_precedence` | a pair satisfying `name_typo` and `transliteration` is `name_typo`; a false merge satisfying 11 and 16 is `same_name_different_business` |
| `test_cross_country_zero` | the synthetic dataset with correct blocking yields no `cross_country` row; the forged US–India pair yields one |
| `test_impact_sums_to_loss` | Σ `impact` == 1 − `metrics.breakdown(...)["f_beta"]` within 1e-9; the sum over rows of matched entities == `(1 − f_beta_matched) · matched / entities` |
| `test_stage_and_sub_flags` | `stage` for probs 0.1 / 0.45 / 0.7 with `tau_abs = 0.5` is `model` / `threshold` / `decision`; `cands_present` vs `cands_absent`; `domain_form` |
| `test_suspicious_label` | a "false merge" with `core_token_set ≥ 0.95` and `ad_token_set ≥ 0.95` is flagged; nothing else is |
| `test_scored_none` | with `scored=None`: `prob` NaN, `stage == "unknown"`, categories 15–17 never fire, the rest unchanged |
| `test_samples_deterministic` | two calls give identical frames; at most `n` rows per category |
| `test_foreign_s1_dropped` | predictions for an S1 id outside the fold are dropped with a warning and do not change `impact` |

## 8. Failure cases

1. **Rule order**: a pair often satisfies several conditions (a transliterated name with a
   typo; a decoy that is also a threshold miss). The precedence of §3 is the documented
   resolution: lexical causes first (1–10), decoy signatures (11–12), structural labels
   (13–17), `cross_country` last because it is asserted separately anyway; `stage` keeps
   the structural information for every row regardless of the category. Never reorder
   without changing the table and `test_precedence` together.
2. **Synthetic label noise**: the generator can produce a "false merge" that is plainly the
   same business (identical core name and address). Rows of `false` type with
   `core_token_set ≥ 0.95` and `ad_token_set ≥ 0.95` get `suspicious_label = True`; they are
   reported separately and excluded from the impact ranking when choosing the next
   experiment, because no module can fix them. If they exceed 1% of matched entities, put
   the share in the notes so leaderboard expectations are adjusted.
3. **Features recomputed with another config**: `tag_errors` rebuilds features for the error
   pairs from the `s1n` / `pooln` passed in; frames normalised with a different
   `NormaliseConfig` than the run silently shift every threshold-based category.
4. **Prediction ids outside the fold** are ignored by `metrics.breakdown` and must be dropped
   here too, or the impact identity of §7 breaks.
5. **Cost**: the frame comes from an outer merge of truth and predictions (≈ 1.6M rows on
   val) and features are computed on the error rows only; never recompute features for all
   candidates here, `scored` already carries `prob`.
