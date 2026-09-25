# 10 — Decision layer (module `decision.py`, owner M5, plan group E)

What you build: `decide(scored, rule) -> matches` and `tune(scored, s1_ids, truth_pairs, grid)
-> (rule, table)` (signatures in `02_SYSTEM_ARCHITECTURE.md` §5). Input: the scored pairs
frame (`SCORED_COLUMNS` = `source1_entity_id`, `entity_id`, `prob` float32). Output: the
matches frame (`MATCH_COLUMNS` = `source1_entity_id`, `entity_id`), one row per kept pair.
Versions v080–v099. Everything here is vectorised pandas/numpy: no Python loop over pairs or
entities. All E versions reuse the D-group winner's `Fitted.matcher` and its cached scored
frames (tune sample and `scored_val.parquet`), so a decision experiment runs in minutes.

## 1. Goal: the match set that maximises macro F0.5

The matcher gives every candidate pair a probability; the decision layer turns the candidates
of each Source 1 entity into a **set** (zero, one or many ids). The score is the mean over
all S1 entities of the per-entity F0.5 (`metrics.entity_fbeta`, `BETA = 0.5`):

- empty truth, empty prediction → 1.0; empty truth, any prediction → 0.0;
- no true positive → 0.0; otherwise F0.5 = 1.25·P·R/(0.25·P + R) =
  **1.25·tp / (n_pred + 0.25·n_true)**, the closed form on counts that the tuner uses (§4).

Per-entity arithmetic (01 §8), n_true = 4 unless stated:

| Situation | P | R | F0.5 |
|---|---|---|---|
| all 4 found, nothing extra | 1 | 1 | 1.000 |
| 2 of 4 found, nothing extra | 1 | 0.5 | 0.833 |
| all 4 found + 1 false | 0.8 | 1 | 0.833 |
| 1 prediction, wrong (entity has matches) | 0 | 0 | 0.000 |
| singleton, empty prediction | – | – | 1.000 |
| singleton, any prediction | – | – | 0.000 |

Missing half of the matches costs as much as one false merge; a wrong-only list costs
everything. Expected F0.5 of keeping versus dropping one doubtful candidate with P(true) = q
next to s certain matches (no other truth):

| s certain + 1 doubtful | E[F \| keep] | E[F \| drop] | keep only if |
|---|---|---|---|
| 0 + 1 (possible singleton) | q | 1 − q | q > 0.50 |
| 1 + 1 | 0.556 + 0.444·q | 1 − 0.167·q | q > 0.73 |
| 4 + 1 | 0.833 + 0.167·q | 1 − 0.048·q | q > 0.78 |

Hence: the bar for an extra match rises with the number of confident matches the entity
already has (relative threshold, §2); a lone candidate needs p > 0.5 to beat the empty list
(singleton threshold); doubtful candidates are dropped, not kept. Lipton et al. [27] prove
the F1 case for calibrated scores (optimal threshold = F*/2) and the same argument gives
F*/(1 + β²) = 0.8·F* for β = 0.5: an entity whose reachable F0.5 is 0.9 needs p ≳ 0.72,
which is why the grid runs up to 0.90.

## 2. The rule family

```python
@dataclass(frozen=True)
class DecisionRule:
    tau_abs: float = 0.5      # absolute floor: keep a pair only if prob >= tau_abs
    tau_rel: float = 0.0      # relative floor: prob >= tau_rel * p_max of its S1 entity
    tau_single: float = 0.5   # singleton floor: the entity predicts nothing unless p_max >= tau_single
    max_matches: int = 11     # cap per S1 entity (train maximum is 11 matches)
    one_to_one: bool = True   # a pool record is kept only for its highest-prob S1 entity
```

Semantics, applied in this order:

1. **Pool-side 1-to-1 first** (`one_to_one`): every S2/S3 record belongs to at most one S1
   entity in the training data (01 §1), so each `entity_id` is kept only for the S1 with the
   highest `prob`; its other rows are removed before any threshold. `p_max` below is computed
   on the survivors: an entity whose best candidate was claimed by a stronger rival sees a
   lower `p_max`, which is the right signal.
2. **Thresholds per S1**: keep a pair iff `prob >= tau_abs` and `prob >= tau_rel * p_max` and
   `p_max >= tau_single`. `tau_single >= tau_abs` always (§4); when `p_max < tau_single` the
   entity's list is empty.
3. **Cap**: keep at most `max_matches` pairs per S1, by `prob` rank.
4. **Tie-breaks**, fixed once for the module: rows are sorted by `prob` descending,
   `source1_entity_id` ascending, `entity_id` ascending before every step, with a stable
   sort. A pool record scored equally by two entities goes to the lexicographically smaller
   S1 id; the cap keeps the lexicographically smaller `entity_id` at equal `prob`.

The thresholds are monotone in `prob` inside an entity, so the kept pairs are always a
prefix of the entity's ranked list and the cap is `rank < max_matches` on the survivors.
`prob` is compared as float64 in `decide` and `tune` alike (cast once), so both give
bitwise-identical decisions.

## 3. `decide` (vectorised)

```python
SCORED_COLUMNS = [C.S1_ID, C.ENTITY_ID, "prob"]; MATCH_COLUMNS = [C.S1_ID, C.ENTITY_ID]
SORT = dict(by=["prob", C.S1_ID, C.ENTITY_ID], ascending=[False, True, True], kind="stable")

def decide(scored: pd.DataFrame, rule: DecisionRule) -> pd.DataFrame:
    """Match set per S1 entity: 1-to-1, absolute/relative/singleton floors, cap (02 §4.5)."""
    assert list(scored.columns[:3]) == SCORED_COLUMNS and not scored.duplicated(MATCH_COLUMNS).any()
    df = scored.assign(prob=scored["prob"].astype("float64")).sort_values(**SORT)
    if rule.one_to_one:                                   # step 1: highest-prob S1 owns the pool record
        df = df.drop_duplicates(C.ENTITY_ID, keep="first")
    p_max = df.groupby(C.S1_ID, sort=False)["prob"].transform("max")
    keep = (df["prob"] >= rule.tau_abs) & (df["prob"] >= rule.tau_rel * p_max) & (p_max >= rule.tau_single)
    df = df[keep.to_numpy()]                              # step 2: a prefix of each ranked group
    rank = df.groupby(C.S1_ID, sort=False).cumcount()     # order inside a group is still prob desc
    out = df[(rank < rule.max_matches).to_numpy()][MATCH_COLUMNS]       # step 3: cap
    return out.sort_values(MATCH_COLUMNS, kind="stable").reset_index(drop=True)
```

Cost: one sort of the scored frame (24M test pairs ≈ 20 s), two group-bys; memory ≈ 3× the
input. Entities with no surviving pair simply have no rows: `submission.write_pairs` and
`evaluate.pairs_to_lists` add their empty lists.

## 4. `tune`: grid search on counts

```python
@dataclass(frozen=True)
class Grid:
    tau_abs: tuple[float, float, float] = (0.30, 0.90, 0.02)   # start, stop inclusive, step: 31 values
    tau_rel: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    single_delta: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)  # tau_single = tau_abs + delta
    max_matches: tuple[int, ...] = (4, 6, 11)
    one_to_one: tuple[bool, ...] = (True,)
    refine_span: float = 0.03; refine_step: float = 0.005; tie_tol: float = 1e-4
    def rules(self, one_to_one: bool) -> list[DecisionRule]: ...   # product: 31 · 7 · 6 · 3 = 3,906 by default
```

`single_delta < 0` is rejected (`ValueError`): a singleton floor below the absolute floor is a
no-op and would only create ties. `tune` scores every rule with
`evaluate.macro_f05_from_counts(tp, n_pred, n_true)` over **all** `s1_ids` of the tune
sample: `n_true` counts every truth pair of those entities, including pairs that blocking
missed, and entities without candidates contribute `n_pred = 0` (a correct empty list for a
singleton, 0.0 for a matched entity), exactly as `metrics.macro_fbeta` scores a submission.

```python
def entity_f05_from_counts(tp, n_pred, n_true) -> np.ndarray:    # evaluate.py (M1); also used by errors.py
    """Per-entity F0.5 from count arrays, same formula as metrics.entity_fbeta, so bitwise equal."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p = tp / n_pred; r = tp / n_true                          # NaN/inf where a count is 0: masked below
        f = 1.25 * p * r / (0.25 * p + r)
    return np.where(n_true == 0, (n_pred == 0) * 1.0, np.where(tp == 0, 0.0, f))

def macro_f05_from_counts(tp, n_pred, n_true) -> float:
    f = entity_f05_from_counts(tp, n_pred, n_true)
    return math.fsum(f) / len(f)                                  # fsum: identical to statistics.fmean

def tune(scored, s1_ids, truth_pairs, grid=Grid()):
    """Best DecisionRule on the tune sample by macro F0.5, plus the full table of rules tried."""
    idx = pd.Index(s1_ids); n = len(idx)
    truth = truth_pairs[isin(truth_pairs[C.S1_ID], idx)]                    # blocking misses included
    n_true = np.bincount(idx.get_indexer(truth[C.S1_ID]), minlength=n)
    rows = []
    for o2o in grid.one_to_one:
        df, p_max, rank = _prepare(scored, o2o)   # sort, optional drop_duplicates, p_max, rank: once per flag
        s1 = idx.get_indexer(df[C.S1_ID]); assert (s1 >= 0).all(); prob = df["prob"].to_numpy()
        is_true = label_pairs(df, truth)["label"].to_numpy().astype(bool)
        for rule in grid.rules(o2o):                                        # ≈ 15 ms each on 3M pairs
            m = (prob >= rule.tau_abs) & (prob >= rule.tau_rel * p_max) \
                & (p_max >= rule.tau_single) & (rank < rule.max_matches)
            tp = np.bincount(s1[m & is_true], minlength=n); n_pred = np.bincount(s1[m], minlength=n)
            rows.append((*astuple(rule), macro_f05_from_counts(tp, n_pred, n_true), n_pred.sum(),
                         tp.sum() / max(n_pred.sum(), 1), tp.sum() / n_true.sum(), (n_pred > 0).mean(), "grid"))
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    best = _pick(table, grid.tie_tol)
    for ta in np.arange(best.tau_abs - grid.refine_span, best.tau_abs + grid.refine_span + 1e-9, grid.refine_step):
        ...  # same evaluation with tau_single = ta + (best.tau_single - best.tau_abs), stage "refine"
    best = _pick(table, grid.tie_tol)
    return DecisionRule(**best[RULE_FIELDS]), table
```

`_pick`: among rules with `f_beta >= max(f_beta) - tie_tol`, take the highest `tau_abs`, then the
highest `tau_rel`, then the highest `tau_single`, then the lowest `max_matches`: ties go to
the most conservative rule (§9). The four comparisons and two `bincount`s on 3M pairs take
≈ 15 ms, so 3,906 rules ≈ 1 min and the 13-rule refinement is free; the cost does not
depend on the number of entities. `evaluate_rules(scored, s1_ids, truth_pairs, rules)`
exposes the inner loop for arbitrary rule lists (E-group notebooks and the tests). Table
columns: `tau_abs, tau_rel, tau_single, max_matches, one_to_one, f_beta, n_pred,
pair_precision, pair_recall, match_rate, stage`.

## 5. Tuning data: the tune side of the inner split, nothing else

`trainset.inner_split(train)` splits the train fold by id hash (seed 4242, 25% tune) into
fit and tune folds; `pipeline.fit` trains the matcher on fit-side pairs and scores the tune
sample (100k S1, ≈ 3M pairs), and `tune` sees only that scored frame plus the tune fold's
truth pairs. The val fold is scored **once per version** with the frozen rule by `run_fold`;
nothing about the rule is ever chosen by looking at val, and the tune → val gap of macro
F0.5 is logged in every E version (§12). The tune side also drives the matcher's early
stopping; that shared use is accepted (the rule is fitted on the same kind of scores it will
see at inference), but nothing from val leaks into either.

## 6. Experiments (E group, v080–v099)

Keep a change only if tune macro F0.5 rises by ≥ 0.002 **and** val macro F0.5 with the
frozen rule rises (any amount); log both plus singleton F0.5, matched F0.5, match rate, mean
matches per S1 and the tune − val gap.

| Version | Change | What to look at |
|---|---|---|
| E1 | global `tau_abs` only: `tau_rel = 0`, `single_delta = 0`, `max_matches = 11`, `one_to_one = False` | reference for group E; F0.5 vs `tau_abs` curve and its flatness around the optimum |
| E2 | 1-to-1 on with the full grid; conflict policies `keep_best` (default), `drop_all_if_gap_below(δ)` for δ ∈ {0.05, 0.1, 0.2}, `none` (§8) | false merges on shared pool records before/after; pairs removed by 1-to-1 |
| E3 | score gap: `tau_rel` grid jointly with `tau_abs`; variant: absolute gap `p_max − prob ≤ δ` instead of the ratio | matched-entity F0.5, mean matches per S1 (3.7 in train) |
| E4 | singleton floor: `single_delta` grid; histogram of `p_max` for true singletons vs matched entities on tune; AUC of `p_max` as a singleton detector (diagnostic only) | singleton F0.5 (5.6% of entities, one full point each); false singletons created |
| E5 | multi-match: `max_matches` grid; per-source floors (`tau_abs` + offset for `S3-` ids, offset ∈ {−0.1, …, +0.1}); expected-F0.5 set decoding (§7) with isotonic calibration | entities with ≥ 4 matches; calibration curve on tune |

E2's conflict policy and E5's per-source offset need fields that `DecisionRule` in 02 §5
does not have (`conflict_delta: float = 0.0`, `tau_s3_offset: float = 0.0`); a winning
variant adds them by a PR against 02 with defaults that reproduce the old behaviour.

## 7. Expected-F0.5 set decoding (experimental, E5)

Ye et al. [28] compare two routes to an F-optimal decision: tune a threshold empirically (§4,
robust to miscalibration) or predict the set with the highest expected F under the model's
probabilities (better when the probabilities are calibrated). The second route, per S1
entity with candidates sorted by `prob` descending, p₁ ≥ … ≥ p_m (m ≤ 60 after blocking's
`max_per_s1`, run after the 1-to-1 step):

```python
def expected_f05_sets(prob_matrix, g=0.0, max_k=11):       # prob_matrix: n_s1 × 60, NaN-padded, sorted desc
    """Per S1 entity the prefix size k (0 = empty) with the highest expected F0.5 under independence."""
    p = np.nan_to_num(prob_matrix); n_s1, m = p.shape
    dist = np.zeros((n_s1, m + 1)); dist[:, 0] = 1.0         # Poisson-binomial P(TP_k = t) for k = 0
    tail = p.sum(1) + g                                      # expected true matches not yet in the set
    best_k = np.zeros(n_s1, int); best_ef = np.prod(1 - p, axis=1)   # E[F | empty] = P(no true match)
    t = np.arange(m + 1)
    for k in range(1, max_k + 1):                            # DP step: fold candidate k into the distribution
        pk = p[:, k - 1][:, None]
        dist = np.concatenate([dist[:, :1] * (1 - pk), dist[:, 1:] * (1 - pk) + dist[:, :-1] * pk], 1)
        tail -= p[:, k - 1]                                  # T = TP_k + tail_k, tail_k = Σ_{i>k} p_i + g
        ef = (dist * 1.25 * t / (k + 0.25 * (t + tail[:, None]))).sum(1)   # the t = 0 term is 0: no TP → 0
        better = ef > best_ef; best_k[better] = k; best_ef[better] = ef[better]
    return best_k
```

Denominator: F0.5 = 1.25·TP/(k + 0.25·T) with T = TP_k + (true matches outside the prefix);
the outside part is replaced by its expectation Σ_{i>k} p_i + g, where g ≥ 0 is an optional
recall-gap constant for truth outside the candidate set (≈ (1 − pair recall) × mean matches,
≈ 0.1 on val). The TP distribution is exact (Poisson-binomial by dynamic programming, O(60²)
per entity, vectorised over entities). E[F | empty] = Π(1 − p_i): the empty list wins when
no candidate is likely, and the answer is cut at `max_k`. Requirements: `prob` must be
calibrated, so fit isotonic regression on half of the tune S1 (by id hash), evaluate the
decoder on the other half, and compare it with the tuned `DecisionRule` on that same half.
Keep only under the §6 criterion; the grid rule stays the default because it is robust to
the test's shift in decoy density (§9).

## 8. 1-to-1 conflict policies (E2)

| Policy | Rule when several S1 entities score the same pool record | Rationale |
|---|---|---|
| `keep_best` (default, `one_to_one=True`) | the highest-prob S1 keeps it; ties → smaller S1 id | matches the data (one owner per pool record); cheap |
| `drop_all_if_gap_below(δ)` | as `keep_best`, but if the best and second-best S1 probs differ by less than δ, no entity keeps the record | an ambiguous owner is a coin flip; a false merge costs more than a miss |
| `none` (`one_to_one=False`) | every S1 may keep it | E1 reference; expected to lose on same-name decoys (48% ambiguous core names, 01 §3) |

Implementation of the second policy: after the sort, `gap = prob − prob.groupby(entity_id)
.shift(−1)` on the first two rows of each pool id; drop every row of the pool id where
`gap < δ`.

## 9. France and the open-set caveat

- No per-country thresholds or grids: France has no training data, so nothing
  country-specific can be tuned; the rule is one global `DecisionRule` (01 §4). Per-source
  thresholds (S2 vs S3) are allowed because the sources are a fixed set.
- Before every upload, `evaluate.slice_report` on the test output compares the France slice
  with US and India: match rate (share of S1 with ≥ 1 match), mean matches per S1, and the
  `p_max` histogram in 0.05 bins. A France match rate more than 10 points below the others
  means normalisation (05 §7) is failing on French forms, not that the rule is wrong.
- The test pool/S1 ratio is 5.8 against 4.7 in train: ≈ 40% of the test pool is unmatched
  against 26% in train (01 §2), so every entity sees more decoys than on the tune split and
  the true optimum shifts towards a stricter rule. Therefore `_pick` prefers the largest
  `tau_abs` within `tie_tol = 1e-4` of the optimum, and every shortlisted version also
  reports the frozen rule on `evaluate.harder_fold(val)` (20% of matched pool records
  removed, seed 99, raising the unmatched share and the singleton count). If the harder fold
  prefers `tau_abs + 0.05` by more than 0.005, say so in the notes; the lead decides whether
  the uploaded version biases the rule (15_LEADERBOARD_STRATEGY).

## 10. Outputs and artifacts

- `experiments/vNNN_<slug>/artifacts/rule.json`: `dataclasses.asdict(rule)` plus `tune_f05`,
  `n_tune_s1`, `grid` (asdict) and the `src/` commit; written by `Fitted.save`.
- `artifacts/tune_table.csv`: the full table of §4 (3,919 rows with the refinement), so the
  F0.5 surface can be plotted without re-running.
- The rule is a field of `pipeline.Fitted(matcher, rule, tune_table, config)` and is loaded
  with the model; `metrics.json` records `rule` and `tune_f05` next to the val score.

## 11. Tests (`tests/test_decision.py`)

A toy scored frame with three S1 entities (`S1-a`, `S1-b`, `S1-c`) and eight pool ids,
probabilities chosen so every rule component changes the answer.

| Test | Assertion |
|---|---|
| `test_one_to_one_keeps_highest_prob_s1` | a pool id scored 0.9 by `S1-b` and 0.7 by `S1-a` stays only with `S1-b`; at 0.8/0.8 it goes to `S1-a` (smaller id) |
| `test_tau_abs` | with `tau_rel=0, tau_single=tau_abs`, exactly the pairs with `prob >= tau_abs` remain |
| `test_tau_rel` | `S1-a` with probs 0.9/0.6/0.4 and `tau_rel=0.7` keeps only 0.9 |
| `test_tau_single_empties_entity` | `S1-c` with `p_max = 0.55 < tau_single = 0.6` has no rows although `0.55 >= tau_abs` |
| `test_max_matches_cap` | `max_matches=2` keeps the two highest; equal probs → smaller `entity_id` |
| `test_tune_equals_metrics` | for 20 seeded random rules, `evaluate_rules(...).f_beta` equals `metrics.macro_fbeta(pairs_to_lists(decide(scored, rule), s1_ids), truth)` exactly, on a frame with an entity without candidates and a truth pair outside the candidates |
| `test_tie_prefers_conservative` | two `tau_abs` values with equal F0.5 → `tune` returns the higher one |
| `test_grid_size_and_validation` | default `Grid().rules(True)` has 3,906 entries; `single_delta=(-0.1,)` raises `ValueError` |
| `test_deterministic` | shuffled input rows and a second run give identical `decide` output and identical `tune` result |
| `test_schema` | output has `MATCH_COLUMNS`, unique pairs, sorted; duplicate input pairs raise |

## 12. Failure cases

1. **Rules that do not transfer**: tune F0.5 − val F0.5 > 0.01 means the 100k-entity tune
   sample or a too-fine grid is being overfitted; raise `n_tune_s1`, coarsen the grid, and
   check that the F0.5 surface is flat (±0.002) within ±0.05 of the chosen `tau_abs`.
2. **Probability shift**: any change in features, training pairs, hard negatives or model
   parameters moves the probability scale; `pipeline.fit` retunes automatically and no
   version may copy `rule.json` from another. The `p_max` histogram in `slice_report` shows
   the shift.
3. **Degenerate grids**: `tau_single < tau_abs` (rejected), a `tau_rel` so high that every
   entity keeps one match (mean matches per S1 collapses well below 3.7), grids whose
   optimum sits on the boundary (0.30 or 0.90: extend the range).
4. **Ties and dtype**: float32 probabilities compared against float64 thresholds differ
   between `tune` and `decide` unless both cast to float64 first (§2).
5. **Entities emptied by 1-to-1**: an entity whose every candidate is claimed by others ends
   up empty; intended, but counted (`errors.tag_errors`, stage `decision`).

## 13. Integration

```python
# pipeline.fit (M1): the rule is tuned once per version, on the tune split only
scored_t = score(pairs_t, s1t, poolt, matcher, cfg)                       # SCORED_COLUMNS
rule, table = tune(scored_t, tune_s1[C.ENTITY_ID], tune_fold.pairs, cfg.grid)
return Fitted(matcher, rule, table, cfg)                                  # Fitted.save → rule.json, tune_table.csv

# pipeline.run_fold(cfg, fitted, fold) / run_test(cfg, fitted, out_dir): decide per country partition
s1n, pooln, pairs = prepare(fold.s1, pool(fold), cfg, fold.name)          # cached per country
scored = score(pairs, s1n, pooln, fitted.matcher, cfg)
scored.attrs["rule"] = asdict(fitted.rule)                                # read by errors.tag_errors
part = scored[C.S1_ID].map(s1n.set_index(C.ENTITY_ID)[C.COUNTRY])         # whatever country values exist
matches = pd.concat([decide(scored[(part == c).to_numpy()], fitted.rule) for c in part.unique()],
                    ignore_index=True)
```

Pool ids never cross countries (blocking partitions by whatever `country` values exist), so
running `decide` per partition gives exactly the same result as one call on the whole frame
while keeping peak memory at one country's pairs. `run_fold` then calls
`evaluate.score_pairs(matches, fold)`, `slice_report` and `errors.tag_errors(matches, fold,
s1n, pooln, scored)`; `run_test` calls `submission.write_pairs(matches, pairs, test_s1_ids)`,
which adds the empty rows for entities without matches. `decision.py` depends only on
`config`, `evaluate.macro_f05_from_counts`, `trainset.label_pairs` and `data.isin`.

References: [27] Z. C. Lipton, C. Elkan, B. Naryanaswamy, "Optimal thresholding of
classifiers to maximize F1 measure", ECML-PKDD 2014. [28] N. Ye, K. M. A. Chai, W. S. Lee,
H. L. Chieu, "Optimizing F-measures: a tale of two approaches", ICML 2012.
