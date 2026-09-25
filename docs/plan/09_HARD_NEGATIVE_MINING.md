# 09 — Hard negative mining (module `trainset.py`, owner M4, plan group D/HN)

What you build: the training-set assembly that decides **which candidate pairs the matcher
learns from and how much each one counts**: `mine_false_positives` (02 §5) plus the helpers
of §3–§4 and §7, wired into `pipeline.fit` behind an optional `HardNegConfig` (§11).
Versions v070–v079 (08 §9 uses v060–v069); log `group="HN4"` etc. in `tracking.log_result`.

## 1. Why

Measured on train (01 §3): 48% of S1 records share their core name with another S1 record;
62% of singletons have their core name in the pool of the same country; 21% of unmatched
pool records carry the core name of some S1 entity. The negatives that matter are therefore
not random pool records (never candidates, trivially separable) but same-name,
similar-address neighbours, and F0.5 punishes accepting one of them twice: a false merge
costs as much as missing half of the matches, and on a singleton it costs the whole point
(01 §8). Precision on hard negatives is the model's contribution to the metric; recall is
blocking's.

Retrieval reaches the same conclusion: DPR [29] gains most from BM25-mined negatives
(lexically similar non-matches), not random ones; Robinson et al. [30] show the hardest
negatives help until they are mislabelled positives, the label-noise risk of §10. In ER,
Bilenko & Mooney [15] and the ALIAS active learner [34] pick training pairs from the
similar-but-uncertain region for the same reason.

## 2. V1: blocking already mines

A candidate pair exists (06 §2) because the two records share a `name_core`, `name_sorted`
or `name_squash` key, or are top-k neighbours in name char-grams (P2) or name+address words
(P3). A candidate with `label == 0` is already a hard negative in the DPR sense: it is the
kind of record the model meets at inference and the only kind it can be wrong about. V1
(HN1) trains on **all candidates of the 200k sampled fit S1 entities** (07 §5: ≈ 5–6M pairs,
≈ 12% positive) with no mining, capping or weighting: `PipelineConfig.hard_negatives = None`,
the setting behind every D-group number of 08. Pairs outside the candidate set are never
added: the model would never meet them.

## 3. Extra sources for V2

Rules are evaluated on labelled candidate pairs (`PAIR_COLUMNS` + `label` from
`label_pairs`) and the feature frame `X` of `build_features` (07 §2, `index == pairs.index`):

| `kind` | Rule, on pairs with `label == 0` | Columns | Why it is hard |
|---|---|---|---|
| `same_name_diff_addr` | `s1n.name_core == pooln.name_core` and `ad_jaccard < 0.2` | `name_core` (join), `ad_jaccard` | the decoy signature (07 §3); 48% name collisions |
| `same_addr_diff_name` | `ad_token_set ≥ 0.9` and `core_token_set < 0.5` | `ad_token_set`, `core_token_set` | co-located different businesses; stops "address alone" merges |
| `top_cosine_nonmatch` | `ctx_rank_name == 1` (best P2 cosine in its S1 group) | `ctx_rank_name` | the model's most tempting candidate is wrong |
| `singleton_decoy` | every candidate of an S1 entity with no truth pair | `label`, truth | **most valuable**: the only evidence of what "no match" looks like; 62% have name twins |

Two further sources come from a fitted model (§7): **round-2 false positives** (`prob ≥ 0.5`,
`label == 0`) and **near-miss false negatives** (`prob < 0.5`, `label == 1`).

```python
# trainset.py additions (M4)
HN_KINDS = ("same_name_diff_addr", "same_addr_diff_name", "top_cosine_nonmatch", "singleton_decoy")
EXTRA_SEED = 8                                   # second S1 sample, disjoint from SAMPLE_SEED's

@dataclass(frozen=True)
class HardNegConfig:
    easy_neg_keep: float = 1.0                   # HN2: 0.2 keeps 20% of easy negatives
    singleton_weight: int = 1                    # HN3: total copies of each decoy row (1 = as is)
    n_extra_s1: int = 0                          # HN3/HN5: size of the second S1 sample
    rule_kinds: tuple[str, ...] = ()             # kinds kept from the second sample
    round2: bool = False                         # HN4
    round2_threshold: float = 0.5; fp_weight: int = 2; keep_near_misses: bool = True
    seed: int = SAMPLE_SEED

def mine_by_rule(pairs, s1n, pooln, kind: str, X: pd.DataFrame | None = None) -> pd.DataFrame
def mine_false_positives(scored, truth_pairs, threshold: float = 0.5) -> pd.DataFrame
def mine_near_misses(scored, truth_pairs, threshold: float = 0.5) -> pd.DataFrame
def is_easy_negative(X: pd.DataFrame, y: pd.Series, singleton: np.ndarray) -> np.ndarray
def build_trainset(pairs, X, y, s1n, pooln, cfg: HardNegConfig, extra=None) -> tuple[pd.DataFrame, pd.Series]
```

- `mine_by_rule` returns the selected rows of `pairs` (same index) plus an `hn_source` str
  column = `kind`; unknown kinds raise `ValueError`. With `X=None` it builds only the
  `name_fuzzy`, `address` and `context` groups through `features.build_features`; from
  `pipeline.fit` the existing `X` is passed and nothing is recomputed.
- `mine_false_positives(scored, truth_pairs, threshold)`: `scored` has `SCORED_COLUMNS`;
  returns the rows with `prob ≥ threshold` whose pair is absent from `truth_pairs`, with
  `label = 0` (int8), keeping `scored.index` so the caller can `X.loc[...]`.
  `mine_near_misses` is the mirror: truth pairs with `prob < threshold`, `label = 1`.
- **Extra sample (HN3/HN5)**: `n_extra_s1` entities drawn by `sample_s1` with `EXTRA_SEED`
  from `fit_fold.s1` minus the base sample, blocked and featurised through `prepare(...,
  tag="fit_extra")` (cached like any tag), labelled, and filtered to `rule_kinds`. Only the
  pairs matching a rule are appended (the whole group for `singleton_decoy`): more hard
  negatives without their easy siblings. Cost ≈ 8 min for 200k extra S1 on the fit pool.

## 4. Sampling ratios and weights

| Rule | Setting | Reason |
|---|---|---|
| keep all positives | always | ≈ 12% of the pairs; each true pair carries recall |
| cap easy negatives | `is_easy_negative`: `sim_name_char < 0.5 and ad_jaccard < 0.1` (NaN compares False, so P1/P3-only pairs are never easy); keep `easy_neg_keep` of them by `hash_unit(source1_entity_id + "|" + entity_id, seed) < easy_neg_keep` | they teach nothing; fewer rows = faster fits; hash sampling is order-free |
| never drop singleton decoys | `singleton = ~data.isin(pairs.source1_entity_id, truth_pairs.source1_entity_id)` exempts them from the cap | the only "no match" evidence |
| singleton decoys ×w | `singleton_weight − 1` extra copies of each decoy row (already present once) | HN3; try 2 and 3 |
| round-2 FPs ×2 | `fp_weight − 1` extra copies (present once already); near misses one extra copy | HN4 |

Weighting is by duplication because `Matcher.fit(X, y, X_val, y_val)` (02 §5) takes no
sample weights; duplicates count towards `min_data_in_leaf`, which is acceptable at 200. If
duplication proves too slow, add an optional `sample_weight: np.ndarray | None = None` to
`Matcher.fit` by a PR to 02 (backwards compatible) and map it to `lgb.Dataset(weight=)`.
Expected mix after HN2 with `easy_neg_keep = 0.2`: ≈ 3.5–4M rows, ≈ 17% positive.

## 5. Leakage rules

1. Everything in this document runs on the **fit side** of `trainset.inner_split(train)`
   (seed 4242): base sample, extra sample, rule mining, round-2 scoring. Fit and tune pools
   are disjoint by construction (02 §2), so no mined pair touches a tune record.
2. Round 2 scores the **fit pairs with the model trained on them** (in-sample). In-sample
   false positives are still valid negatives: the label is the truth, not the model, and a
   pair the model gets wrong even after seeing it is the hardest kind. If in-sample mining
   yields < 20k FPs, use the cross-fit variant (HN4b): split the base sample's S1 ids in two
   by `hash_unit`, fit on each half, score the other, union the FPs.
3. The tune side is used once per candidate model, for `decision.tune` and the §7
   comparison; nothing is mined from it. The val fold is read only by `pipeline.run_fold`.
4. `hn_source`, `label` and `prob` never enter `X`: `build_trainset` asserts
   `list(X.columns) == feature_names(cfg.feature_groups)` before returning.

## 6. Expected effect and how to measure it

Hard negatives move precision at fixed recall (the model separates decoys from matches at
the same threshold) and, above all, singleton F0.5: singletons are 5.6% of entities, so the
maximum available gain is `0.056 × (1 − f_beta_singletons)`. Expectations, not commitments:
HN3/HN4 +0.003 to +0.010 on val `f_beta`, mostly through `f_beta_singletons`; HN2 ±0.001
with 30–40% shorter fits; HN5 ≤ +0.003.

Measure every HN version on val after `decision.tune`:

- `evaluate.score_pairs(matches, val_fold)` → `metrics.breakdown`: `f_beta` decides;
  `f_beta_singletons`, `f_beta_matched`, `pair_precision`, `pair_recall` explain;
- false-merge count from `errors.tag_errors(matches, val_fold, s1n, pooln, scored)`, split
  into singleton victims and matched entities, with 20 examples via `evaluate.error_samples`;
- diagnostics on tune: precision at recall 0.90/0.95 on the pair PR curve, tune logloss;
- `hn_counts` in `metrics.json`: rows per source, positives, easy dropped, decoys, FPs, FNs.

## 7. Round-2 loop (`pipeline.fit`, when `cfg.hard_negatives.round2`)

```python
def fit_matcher(cfg, fit_fold, tune_fold, pairs, X, y, s1n, pooln, pairs_t, X_t, y_t, s1t, poolt, tune_s1):
    hn = cfg.hard_negatives
    X1, y1 = build_trainset(pairs, X, y, s1n, pooln, hn, extra)            # HN2/HN3/HN5
    m1 = Matcher(cfg.model).fit(X1, y1, X_t, y_t)                         # round 1
    if not hn.round2:
        return m1
    scored = pairs[[S1_ID, ENTITY_ID]].assign(prob=m1.predict_proba(X))    # fit side, in-sample
    fps = mine_false_positives(scored, fit_fold.pairs, hn.round2_threshold)
    fns = mine_near_misses(scored, fit_fold.pairs, hn.round2_threshold) if hn.keep_near_misses else fps.iloc[:0]
    n_copies = hn.fp_weight - 1                                           # FPs are in X1 once already
    X2 = pd.concat([X1, *([X.loc[fps.index]] * n_copies), X.loc[fns.index]])  # FPs ×fp_weight, FNs ×2
    y2 = pd.concat([y1, *([y.loc[fps.index]] * n_copies), y.loc[fns.index]])
    m2 = Matcher(cfg.model).fit(X2, y2, X_t, y_t)                         # refit from scratch
    f1, f2 = (tune_f05(m, cfg, pairs_t, s1t, poolt, tune_s1, tune_fold) for m in (m1, m2))
    return m2 if f2 >= f1 + 0.002 else m1                                  # keep only a clear gain

def tune_f05(m, cfg, pairs_t, s1t, poolt, tune_s1, tune_fold) -> float:
    scored_t = pipeline.score(pairs_t, s1t, poolt, m, cfg)
    rule, table = decision.tune(scored_t, tune_s1.entity_id, tune_fold.pairs, cfg.grid)
    return best tune macro F0.5 in `table`
```

Round 2 refits from scratch (no warm start) so the early-stopping count is re-derived; both
models are early-stopped on the same tune pairs, and the winner's rule is re-tuned by the
normal `pipeline.fit` flow. Log `round2_kept`, `len(fps)`, `len(fns)` and both tune scores.
One round only: a third round mines the noise of §10.

## 8. Experiments (HN, v070–v079 as assigned)

| ID | Change (`HardNegConfig`) | Keep if (val, after `decision.tune`) |
|---|---|---|
| HN1 | `None`: all candidates of the base sample (= best D3 of 08) | reference |
| HN2 | `easy_neg_keep=0.2` | `f_beta ≥ HN1 − 0.001` and fit time −30%; else keep 1.0 |
| HN3 | `singleton_weight ∈ {2, 3}`; then `n_extra_s1=100_000, rule_kinds=("singleton_decoy",)` | `f_beta_singletons +0.01` and `f_beta ≥ best + 0.001` |
| HN4 | `round2=True, fp_weight=2` (HN4b: cross-fit) | tune +0.002 (§7) and val `f_beta +0.002`; false merges −10% |
| HN5 | `n_extra_s1=200_000, rule_kinds=HN_KINDS` on top of the best of HN2–HN4 | `f_beta +0.002`; else drop the extra sample |

One change per version, compared against the previous best with the same blocking, features
and grid; thresholds are always re-tuned (§10.2). The HN sequence starts after D3 has a
stable parameter set (08 §9), because a parameter change would confound the comparison.

## 9. Tests (`tests/test_trainset.py` additions)

Fixture: the `conftest` synthetic fold, split with `inner_split`, blocked with the test
`BlockingConfig`, featurised, labelled; a fake `scored` frame with hand-set `prob`.

| Test | Assertion |
|---|---|
| `test_mine_false_positives_label_zero_only` | every row is absent from `truth_pairs`, `label == 0`, `prob ≥ threshold`, index ⊂ `scored.index` |
| `test_mine_near_misses_label_one_only` | every row is a truth pair with `prob < threshold`, `label == 1` |
| `test_mine_by_rule_kinds` | a same-name/different-address pair lands in `same_name_diff_addr` only; a same-address/different-name pair in `same_addr_diff_name`; the rank-1 non-match in `top_cosine_nonmatch`; a singleton's candidates all in `singleton_decoy`; label-1 rows never selected; unknown kind raises |
| `test_no_tune_or_val_ids_in_mined` | S1 and pool ids of every mined frame ⊂ fit-side ids; ∩ tune ids = ∅; ∩ val ids = ∅ |
| `test_easy_negative_cap_ratio` | with 1,000 easy negatives and `easy_neg_keep=0.2`, 150–250 survive; positives untouched; NaN-sim pairs never counted as easy |
| `test_singleton_decoys_never_dropped` | `easy_neg_keep=0.0` keeps every candidate of the singleton S1 entities |
| `test_weights_by_duplication` | `singleton_weight=3` → each decoy row appears 3 times; `fp_weight=2` → each FP twice; `y` aligned |
| `test_build_trainset_deterministic` | same `seed` → identical index, labels and row order; a different seed changes the easy-negative sample only |
| `test_build_trainset_columns_unchanged` | output columns == `feature_names(groups)`; no `hn_source`, `label`, `prob` |

## 10. Failure cases

1. **Label noise in the synthetic truth.** A "false positive" may be a match the generator
   did not label (21% of unmatched pool records carry an S1 core name; some are near-exact
   copies). Robinson et al. [30] show such pairs, weighted up, teach the model to reject
   true matches. Before trusting any mined set, inspect 20 random rows side by side
   (`evaluate.error_samples`) and 20 of the highest-`prob` FPs; if more than 2 of 20 look
   like real matches, raise `round2_threshold` to 0.8 and drop `fp_weight` to 1.
2. **Class-balance drift.** Capping, duplication and extra samples change the positive rate,
   so `prob` is not comparable across versions: never reuse a `rule.json`, never compare raw
   `prob` histograms between versions, always let `pipeline.fit` re-run `decision.tune`.
   Compare versions on val macro F0.5 only.
3. **Over-mining → recall collapse.** Too many hard negatives make the model reject genuine
   variants (transliterations, renames): watch `pair_recall` and `f_beta_matched` on val; a
   drop > 0.005 at equal `f_beta` is a warning even when singletons improve.
4. **Duplicated groups and context features.** Extra-sample pairs bring `ctx_*` features
   computed on their full group before filtering, which is correct; never recompute context
   on a filtered frame.
5. **Memory.** Duplication is `pd.concat` of float32 slices (≤ 0.3 GB per copy); keep
   `mem_guard` around `build_trainset` and free `X` after `fit` (08 §11.3).

## 11. Integration

- `PipelineConfig` gains `hard_negatives: HardNegConfig | None = None` (02 §5, by PR). With
  `None`, `pipeline.fit` is byte-for-byte the V1 flow of 02 §6; otherwise it calls
  `build_trainset` before `Matcher.fit` and, when `round2`, the §7 loop instead of the single
  fit. `Fitted`, `run_fold`, `run_test` and the pairs cache key (blocking config hash only)
  are unchanged.
- `hn_counts` and `round2_kept` go into `metrics.json` through `log_result(metrics=...)`;
  the `hn_source` breakdown is printed in notebook §4.4 next to the importances.
- No new dependency: hashing is `split.hash_unit`, sampling is `sample_s1`, everything else
  is pandas.

## 12. References (numbers as in `17_RESEARCH_REFERENCES.md`)

[15] Bilenko & Mooney 2003, learnable string similarity with selected training pairs.
[29] Karpukhin et al. 2020, DPR: BM25-mined hard negatives. [30] Robinson et al. 2021,
contrastive learning with hard negatives and their false-negative risk. [34] Sarawagi &
Bhamidipaty 2002, ALIAS: active learning of uncertain pairs for deduplication.
