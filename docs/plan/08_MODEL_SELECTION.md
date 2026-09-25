# 08 — Model selection (module `model.py`, owner M4, plan group D)

What you build: `Matcher`, the pairwise classifier of `02_SYSTEM_ARCHITECTURE.md` §5. It
takes the float32 feature frame of `features.build_features` (07) with labels from
`trainset.label_pairs`, and returns one probability per candidate pair, which `decision.py`
(10) turns into match sets. Versions v060–v079 belong to M4: v060–v069 for the D group
(this document), v070–v079 for hard negatives (09). Log `group="D3"` etc. in
`tracking.log_result`.

## 1. What to implement

```python
# model.py (M4) — the signatures are the contract of 02 §5
@dataclass
class MatcherParams:
    backend: str = "lgbm"            # "lgbm" | "logreg" | "heuristic"
    num_leaves: int = 63; learning_rate: float = 0.05; n_estimators: int = 2000
    early_stopping: int = 100; feature_fraction: float = 0.8; bagging_fraction: float = 0.8
    bagging_freq: int = 1; min_data_in_leaf: int = 200; lambda_l2: float = 1.0
    max_bin: int = 255; scale_pos_weight: float = 1.0; seed: int = 42; num_threads: int = 12

class Matcher:
    def __init__(self, params: MatcherParams = MatcherParams()) -> None
    def fit(self, X, y, X_val=None, y_val=None) -> "Matcher"
    def predict_proba(self, X, chunk_rows: int = 2_000_000) -> np.ndarray   # float32
    def importance(self) -> pd.Series                                        # gain per feature
    def save(self, dir: Path) -> Path
    @classmethod
    def load(cls, dir: Path) -> "Matcher"
```

State after `fit`: `feature_names_: list[str]` (`list(X.columns)`), `best_iteration_: int`,
`model_` (`lgb.Booster`, sklearn `Pipeline`, or `None` for `heuristic`) and `fit_info_: dict`
(`rows`, `positive_rate`, `best_iteration`, `tune_logloss`, `tune_auc`, `fit_seconds`).

### 1.1 `fit(X, y, X_val, y_val)`

- `X` is a float32 `DataFrame` whose columns are exactly `features.feature_names(groups)`;
  `y` is the int8 `label` aligned to `X.index`. Assert both, refuse `object` dtypes and
  `len(y) != len(X)`.
- `lgbm`: `lgb.Dataset(X.to_numpy(np.float32), y, feature_name=feature_names_,
  free_raw_data=True)`; `lgb.train(params, train, num_boost_round=n_estimators,
  valid_sets=[tune], callbacks=[lgb.early_stopping(early_stopping, first_metric_only=True),
  lgb.log_evaluation(0)])`. Early stopping watches **tune `binary_logloss`**;
  `best_iteration_ = booster.best_iteration`. Without `X_val` (tests only) train
  `n_estimators` rounds and warn.
- `logreg`: `Pipeline([SimpleImputer(strategy="constant", fill_value=-1, add_indicator=True),
  StandardScaler(), LogisticRegression(C=1.0, max_iter=300)])`. `-1` lies outside every
  similarity's [0, 1] range and `add_indicator` supplies the missing flags. sklearn upcasts
  to float64 (6M × 68 ≈ 3.3 GB): fit the baseline on a 2M-pair subsample if RAM is short.
  `X_val` is only used to report logloss/AUC.
- `heuristic`: stores the feature names, learns nothing. Requires `pass_exact`,
  `sim_name_char`, `sim_name_addr_word`, `sim_addr_char` among the columns.

`MatcherParams` → LightGBM parameters (everything else at LightGBM defaults):

| MatcherParams field | LightGBM | Fixed in the mapping |
|---|---|---|
| `num_leaves`, `learning_rate`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `min_data_in_leaf`, `lambda_l2`, `max_bin`, `scale_pos_weight`, `seed`, `num_threads` | same names | `objective="binary"`, `metric="binary_logloss"`, `verbose=-1` |
| `n_estimators` | `num_boost_round` | `deterministic=True`, `force_row_wise=True`: same seed → same trees (≈ 10% slower) |
| `early_stopping` | `lgb.early_stopping(stopping_rounds)` | `first_metric_only=True` |

### 1.2 `predict_proba(X, chunk_rows)`

Refuse a column mismatch **before** touching the data: `list(X.columns) != feature_names_`
raises `ValueError` naming missing, unexpected and reordered columns. Never reorder
silently: two swapped similarity columns would score without any error. Then
`out = np.empty(len(X), np.float32)` and fill `out[i:i + chunk_rows]` from
`model_.predict(X.iloc[i:i + chunk_rows].to_numpy(np.float32), num_iteration=best_iteration_)`.
LightGBM reads C-contiguous float32 without a float64 copy; a 2M × 48 chunk is 0.4 GB. The
preallocated output avoids the list-of-chunks plus `np.concatenate` doubling. `heuristic`:
`prob = nanmax(sim_name_char, sim_name_addr_word, sim_addr_char)`, then `1.0` where
`pass_exact == 1` and `0.0` where every sim is NaN. `logreg`: `predict_proba(chunk)[:, 1]`.

### 1.3 `importance`, `save`, `load`

- `importance()`: `pd.Series(model_.feature_importance("gain"), index=feature_names_)`,
  normalised to sum 1, sorted descending. `logreg`: `|coef|` on the scaled inputs
  (comparable across features), indicator columns folded into their feature. `heuristic`:
  1/3 on each of the three sims, 0 elsewhere.
- `save(dir)`: `params.json` (`asdict(params)` + `best_iteration` + `fit_info`),
  `feature_names.json`, and `model.txt` (`booster.save_model`) or `model.joblib`
  (`joblib.dump(pipeline)`; joblib ships with scikit-learn). `heuristic` writes the two JSON
  files only. `load(dir)` reads `params.json` first and dispatches on `backend`. The round
  trip reproduces predictions bit for bit (§10). `Fitted.save` (02 §5) calls this with
  `experiments/vNNN_<slug>/artifacts/model/`.

## 2. Model comparison

Workload: ≈ 6M training pairs × 48 float32 similarity features with NaN (fit side, 07 §5);
≈ 24M test pairs to score; 12 CPU cores, 15 GB RAM, RTX 2050 with 4 GB. Learning-based
matchers on similarity vectors are the standard design [22]; Magellan [16] ships tree
learners as its default matcher for the same reasons that decide the table.

| Model | Accuracy on noisy tabular sims | Precision control | Train 6M×48 | Score 24M pairs | Explainability | Effort | Licence | Verdict |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | linear in the sims: cannot express "same name AND different address" without hand-made interactions (07 §3) | good: calibrated by construction, one threshold | 2–4 min lbfgs after scaling; float64 copies | < 1 min | coefficients | low | BSD-3 (sklearn) | D1 baseline, sanity floor |
| Random forest | handles interactions; weaker than boosting on NaN and monotone signals | vote fractions rarely reach 0/1; calibrate before thresholding [26a] | full depth infeasible (> 15 GB of nodes); `max_depth=16`: 5–10 min | 5–10 min, multi-GB | impurity importance, biased | low | BSD-3 | D2 reference only, never shipped |
| LightGBM [24] | best in class: native NaN, interactions, leaf-wise growth fits the decoy signature | log-loss + early stopping is near-calibrated; verify (§5) | 3–5 min, 600–1200 trees | 2–4 min | gain importance, SHAP via `pred_contrib` | low | MIT | **V1 and default** |
| XGBoost [25a] | same class; depth-wise growth converges a little slower | same | 5–10 min CPU `hist`; 1–2 min on the GPU (not bit-reproducible) | 3–6 min | gain, SHAP | low | Apache-2.0 | D4, only if LightGBM plateaus |
| CatBoost [25b] | same class; ordered boosting helps small data, irrelevant at 6M rows | symmetric trees, well calibrated | 10–20 min CPU | 2–4 min | gain, SHAP | low | Apache-2.0 | D4 alternative |
| Neural pair classifier (DeepMatcher [17]) | strong on raw text with many labels; blind to our group-context features; learns the generator's vocabulary | overconfident softmax, needs temperature scaling | hours on the RTX 2050 (sequence models over 6M text pairs) | 1–2 h | none | high | BSD-3 | no: time, GPU, nothing to add on tabular sims |
| Sentence-embedding cosine (bi-encoder) | poor alone: same-name decoys collide; embeddings ignore house numbers and PINs | one threshold, uncalibrated | encode 12M records: 1–2 h GPU | fast once encoded | none | medium | Apache-2.0 (MiniLM); pretrained weights count only as a model (01 §9) | no as matcher; at most one feature |
| Cross-encoder (Ditto [18]) | best pairwise text accuracy in the literature | good after calibration | fine-tuning: hours | 24M pairs at ≈ 0.5–1k pairs/s = 7–13 h | none | high | Apache-2.0 code; checkpoint licence varies | no: inference budget |
| Hybrid: embedding cosine as a GBDT feature | small gain on renames and transliterations; the rest is LightGBM | as LightGBM | + encoding | + encoding | as LightGBM | medium | as above | optional C-group feature after C5; not a model change |

## 3. Decision: LightGBM as V1

`MatcherParams()` is the starting point, not a grid: `objective binary`, `num_leaves 63`,
`learning_rate 0.05`, `n_estimators 2000` with `early_stopping 100` on tune logloss,
`feature_fraction 0.8`, `bagging_fraction 0.8`, `bagging_freq 1`, `min_data_in_leaf 200`,
`lambda_l2 1.0`, `max_bin 255`, `scale_pos_weight 1.0`, `seed 42`, `num_threads 12`,
`verbose -1`. Expected: early stop at 600–1200 trees, 3–5 min on the fit sample, ≤ 1.5 GB
(the binned dataset is ≈ 0.3 GB, the float32 frame 1.15 GB).

Why these values: 63 leaves with ≥ 200 rows per leaf on 6M rows gives smooth, non-memorising
trees over 48 bounded similarities; 0.05 with early stopping costs a few minutes and buys
1–3% lower logloss than 0.1; column and row subsampling decorrelate trees on collinear
features (`nm_ratio`, `core_ratio`, `core_indel` move together). **`scale_pos_weight` stays
1.0**: the ≈ 12% positive rate is the natural rate at inference; reweighting would shift
every probability and make thresholds incomparable between versions and against the tune
fold. Precision is bought in the decision layer, never by distorting the likelihood.

- D1 baseline: `MatcherParams(backend="logreg")` (§1.1) is the floor a linear model reaches;
  everything after it must beat it clearly.
- `heuristic` backend: keeps `pipeline.fit` / `run_fold` / `run_test` runnable on day 1 while
  features and the model are being written, and is the "no model" reference row.
- XGBoost / CatBoost (D4) are tried only if two consecutive D3 versions gain < 0.002.

## 4. Training data

```
fit_fold, tune_fold = trainset.inner_split(train)                      # seed 4242, 25% tune
fit_s1  = sample_s1(fit_fold.s1, 200_000)  → pairs = prepare(..., "fit") → X = build_features(...)
                                            y = label_pairs(pairs, fit_fold.pairs).label
tune_s1 = sample_s1(tune_fold.s1, 100_000) → X_t, y_t likewise, tag "tune"
matcher = Matcher(cfg.model).fit(X, y, X_t, y_t)
```

Sizes (07 §5): ≈ 5–6M fit pairs, ≈ 12% positive; ≈ 3M tune pairs. The fit side grows the
trees. The tune side does three jobs — early stopping, the reliability check (§5) and
`decision.tune` — and nothing else is ever fitted on it. The val fold is read only by
`pipeline.run_fold`; a model that has seen val ids in any form is invalid and its version
says so in `notes`. Whole S1 entities are sampled, never pairs, so the model sees each
entity's complete candidate group with the context features of 07 §2 exactly as at
inference.

## 5. Calibration

Thresholds are invariant to monotone maps, so calibration only matters when the decision
layer adds or compares probabilities across entities (expected-F0.5 decoding, 10 / E5).
Boosting with the exponential loss pushes probabilities away from 0 and 1 [26a]; log-loss
boosting with early stopping is much closer, but it is checked, not assumed. Every D version
logs on tune:

- a reliability table: 20 equal-count bins of `prob`, mean predicted vs observed positive
  rate, plus `ece_tune` (bin-weighted mean gap) and the Brier score;
- the same table restricted to `ctx_rank_name == 1`, the pairs the decision layer sees first.

If `ece_tune > 0.02` **and** the decoder needs probabilities, fit
`IsotonicRegression(out_of_bounds="clip")` [26c] on the tune predictions (Platt's sigmoid
[26b] when tune has < 200k pairs), save it as `calibrator.joblib` next to the model and
apply it inside `predict_proba`. V1 ships without a calibrator.

## 6. Evaluation

Tune logloss and AUC are diagnostics: they say whether a change moved the ranking, never
whether to keep it. The decision criterion for every D version is macro F0.5 on the val fold
after the decision layer, `evaluate.score_pairs(matches, val_fold)` = `metrics.breakdown`:
`f_beta` decides; `f_beta_singletons` and `f_beta_matched` explain; `pair_precision` and
`pair_recall` show which side moved. Each notebook also logs fit seconds, `best_iteration_`,
tune logloss/AUC and the top-20 gain importances through `tracking.log_result(metrics=...)`.

## 7. Hyperparameter search

| Axis | Values | Expectation |
|---|---|---|
| `num_leaves` | 31, 63, 127 | 127 memorises the synthetic vocabulary; 31 underfits interactions |
| `min_data_in_leaf` | 100, 200, 500 | larger = smoother; helps decoys |
| `learning_rate` | 0.03, 0.05 | 0.03 needs ≈ 2× the trees for ≤ 0.001 |

Run it as coordinate descent (one axis from the default, keep the best, next axis: 7 fits
≈ 30 min), not as the 18-point grid. Select by **tune macro F0.5 after `decision.tune`**
(retune the rule per point), tie-break by tune logloss; report the chosen point on val once.
Optuna (MIT) with TPE over the same ranges plus `lambda_l2 ∈ [0.1, 10]` and
`feature_fraction ∈ [0.5, 1.0]` is optional (≤ 25 trials, same objective). Never search on val.

## 8. Feature importance

`Matcher.importance()` (gain, normalised) is logged in every version: the top 20 in the
notebook and in `metrics.json` (`importance_top20`), the full series in
`artifacts/model/importance.csv`. Rules: (i) a feature with zero gain in three consecutive
versions is dropped from `features.REGISTRY` by M3 (07 §6); that changes `feature_names`, so
it is a new model version; (ii) no address feature in the top 15 is a bug (addresses decide,
01 §12): inspect before tuning anything; (iii) `ctx_*` features dominating while
`f_beta_singletons` falls means group size is leaking into the probability; check with
`evaluate.harder_fold`.

## 9. Experiments (D group)

| Version | ID | Change | Keep if (val macro F0.5 after `decision.tune`) |
|---|---|---|---|
| v060 | D1 | `logreg` baseline on the C-group feature set | reference; must beat `heuristic` by ≥ 0.02 |
| v061 | D2 | `RandomForestClassifier(300, max_depth=16, min_samples_leaf=200, n_jobs=12)` on a 1M-pair subsample, notebook only | reference only: reported, never shipped |
| v062 | D3 | LightGBM `MatcherParams()` | ≥ D1 + 0.01 (expected far more) |
| v063–v067 | D3 | coordinate search of §7, one axis per version | +0.002 over the previous best |
| v068 | D3 | 3-seed average (42, 43, 44) of the best point, averaged in the notebook | +0.002; else keep the single model |
| v069 | D4 | XGBoost (`hist`) or CatBoost with matched settings | only after a D3 plateau; keep if +0.003 |

## 10. Tests (`tests/test_model.py`)

Fixture: 3,000 synthetic pairs from `conftest` (48 columns, NaN where 07 allows), labels from
a noisy rule; `MatcherParams(n_estimators=60, num_threads=2)` so every test runs in < 2 s.

| Test | Assertion |
|---|---|
| `test_fit_is_deterministic` | two fits with seed 42 → `np.array_equal` predictions; seed 43 differs |
| `test_predict_proba_range_dtype_length` | float32, `0 ≤ p ≤ 1`, no NaN, `len == len(X)` |
| `test_chunked_predict_equals_full` | `chunk_rows=7` and `chunk_rows=10**9` give identical arrays |
| `test_save_load_roundtrip` | predictions, `feature_names_`, `best_iteration_`, params equal after `load` |
| `test_column_mismatch_raises` | missing, extra and reordered columns each raise `ValueError` naming the column |
| `test_early_stopping_on_tune` | with `X_val`: `0 < best_iteration_ ≤ n_estimators`, `fit_info_["tune_logloss"]` set |
| `test_heuristic_equals_nanmax` | `prob == nanmax(sims)`; 1.0 where `pass_exact == 1`; 0.0 where all NaN |
| `test_logreg_handles_nan` | NaN in fit and predict → finite probabilities, no warning |
| `test_importance_matches_feature_names` | index == `feature_names_`, non-negative, sums to 1 |
| `test_rejects_object_dtype_and_misaligned_y` | `object` column or `len(y) != len(X)` → `ValueError` |

## 11. Failure cases

1. **Overfitting the synthetic vocabulary.** The generator renames from a finite word list
   (`Dovaflux`, `N+ Inc Services`); deep trees memorise their similarity profile. Guards: no
   raw-token features (similarities only), `min_data_in_leaf ≥ 200`, the fit–tune logloss
   gap (> 0.03 = overfit) and the tune–val gap of `f_beta_matched`.
2. **Miscalibration under the test pool/S1 ratio** (5.8 vs 4.7 in train: ≈ 40% of the test
   pool unmatched vs 26%). More decoys per entity shift `ctx_n_cands`, `ctx_pool_indegree`
   and the prior; true-pair probabilities drift down and the tuned thresholds move off their
   optimum. Check: score `evaluate.harder_fold(val, drop_frac=0.2)` with the same rule; a
   drop > 0.01 means the rule needs the relative thresholds of E3/E4, and the France / US /
   India `p_max` histograms are compared before every upload (02 §9.6).
3. **Memory when scoring.** Never materialise the 24M × 48 test matrix (4.6 GB float32):
   `pipeline.score` builds features per `iter_chunks` slice and calls `predict_proba` on it;
   inside `predict_proba` the output is preallocated. `free_raw_data=True`, `del X` after
   `fit`, and `mem_guard` (RSS < 6 GB) at the stage boundary.
4. **Non-determinism.** Without `deterministic=True` / `force_row_wise=True` LightGBM can
   grow different trees across runs on 12 threads; `test_fit_is_deterministic` catches it.
5. **Silent column reorder.** A notebook assembling `X` by hand in a different group order
   would score garbage; the mismatch check in §1.2 is the only guard, so it never relaxes.

## 12. Integration

- Consumes `features.feature_names(cfg.feature_groups)` as the column contract; produces
  `prob` for `decision.SCORED_COLUMNS` (`source1_entity_id`, `entity_id`, `prob` float32)
  through `pipeline.score(pairs, s1n, pooln, matcher, cfg)`, the only caller of
  `predict_proba` outside tests.
- `pipeline.fit` calls `Matcher(cfg.model).fit(X, y, X_t, y_t)`; the hard-negative variant
  (09 §7) wraps that call and refits once. `Fitted.save` / `load` delegate to
  `Matcher.save` / `load`.
- Adding a feature changes `feature_names` and needs a new model version; the saved
  `feature_names.json` records what a version was trained on.
- Dependencies, pinned in the commit that first imports them: `lightgbm==4.7.0` (MIT);
  `scikit-learn==1.9.1` (BSD-3, already required by 06; provides joblib, the imputer, the
  scaler, LR and isotonic regression). `xgboost`, `catboost` (Apache-2.0) and `optuna` (MIT)
  only in the D4 / §7 versions that use them, and in `requirements.txt` only if shipped.

## 13. References (numbers as in `17_RESEARCH_REFERENCES.md`)

[16] Konda et al. 2016, Magellan. [17] Mudgal et al. 2018, DeepMatcher. [18] Li et al. 2020,
Ditto. [22] Köpcke, Thor & Rahm 2010, evaluation of ER approaches. [24] Ke et al. 2017,
LightGBM. [25a] Chen & Guestrin 2016, XGBoost. [25b] CatBoost. [26a] Niculescu-Mizil &
Caruana 2005, predicting good probabilities. [26b] Platt scaling. [26c] Zadrozny & Elkan,
isotonic calibration.
