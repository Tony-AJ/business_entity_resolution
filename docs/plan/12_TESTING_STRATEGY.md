# 12 — Testing strategy (everyone; `tests/` layout, the `make lint test` gate)

Five people change one pipeline in three days. The tests are the executable form of the
contracts in `02_SYSTEM_ARCHITECTURE.md` §4–§5 and the net that catches a schema drift
before it costs an upload. They run in seconds on synthetic data and never touch
`dataset/`. Big-data questions (recall, RAM, time) are notebook checks (§6), not tests.

## 1. The gate

`make lint test` passes before **every commit** (project rules §3, "green before commit")
and before **every merge** into `main` (§9). What it runs (`pyproject.toml`, `Makefile`):

| Command | Does | Config |
|---|---|---|
| `make lint` | `ruff check src tests` | rules `E, F, W, I, B`; line length 100; `target-version = py312` |
| `make test` | `pytest` | `testpaths = ["tests"]`, `pythonpath = ["src"]`, `addopts = "--strict-markers -q"` |

- `I` sorts imports (`ruff check --fix` does it for you); `B` is bugbear: `zip(...,
  strict=True)`, no mutable defaults, no `except:`; `E501` is the 100-column limit.
- `--strict-markers`: an unregistered `@pytest.mark.x` is an error. We register none; a
  test that needs one is probably too big.
- Budget: the whole suite **< 30 s** (today: 60 tests in ≈ 2 s). One test may not exceed
  2 s without a comment saying why. No test reads `dataset/`, the network or a cache outside
  `tmp_path`. A conftest autouse fixture `no_real_dataset` monkeypatches `data._cached` and
  `split.val_ids` to `pytest.fail` when the path lies under `C.ROOT / "dataset"`.
- Dependencies are pinned in `requirements.txt` by the commit that first imports them; a
  missing package is a setup error, never `importorskip`.

The `dataset_dir` fixture (`tests/conftest.py`) writes the challenge layout under
`tmp_path/dataset/{train,test}/` per test and returns the root:

| Split | Rows |
|---|---|
| train S1 | `S1-00001` Acme Corp, "12 Main St, Springfield, IL", US; `S1-00002` Sharma Traders Pvt Ltd, "Near SBI ATM, MG Road, Pune", India; `S1-00003` name `NA`, empty address, US |
| train S2 | `S2-00001` ACME Corporation, "12 Main Street, Springfield"; `S2-00002` Sharma Traders Private Limited, "MG Rd, Pune 411001" |
| train S3 | `S3-00001` Acme Corp., "12 Main St"; `S3-00002` Globex LLC, Austin TX (unmatched decoy) |
| truth | `S1-00001 → S2-00001,S3-00001`; `S1-00002 → S2-00002`; `S1-00003 →` (singleton) |
| test S1 | `S1-00010` Boulangerie Dupont SARL, "5 Rue de la Paix, Paris", France; `S1-00011` Patel & Sons, "Station Road, Surat", India |
| test S2/S3 | `S2-00010` Boulangerie Dupont, "5 rue de la Paix, 75002 Paris", France; `S3-00010` Patel and Sons, "Stn Rd, Surat"; `S3-00011` Initech, Dallas, US (decoy) |

Traps planted on purpose: commas inside addresses (TSV parsing), a business literally
named `NA` (`na_filter=False`), an empty address, a singleton, France only in test (open
country set), legal-form variants (Corp/Corporation, Pvt Ltd/Private Limited, SARL),
abbreviations (St/Street, Rd/Road, Stn), `&`/and. Fold membership is fixed by the id hashes
(`split.hash_unit`), so it never changes: with `frac=0.2` val = `{S1-00001}` + `{S2-00001,
S3-00001}`; with `frac=0.5` val = `{S1-00001, S1-00003}` + `{S2-00001, S3-00001, S3-00002}`
(a matched entity, a singleton, a decoy) and train = `{S1-00002}` + `{S2-00002}`.
`inner_split` of that train fold puts `S1-00002` on **tune** (`hash_unit(id, 4242) =
0.209`), so the **fit side is empty**: integration tests use `backend="heuristic"` and every
stage must accept an empty frame (§5).

## 2. One test file per module

Existing files stay as they are and grow. Tests named in 05 §9, 06 §8 and 07 §7 are the
module owner's minimum; they are referenced, not repeated.

| Module | File | Owner | Tests |
|---|---|---|---|
| `config.py` | `test_config.py` | exists | paths hang off `ROOT`; prefixes cover sources |
| `data.py` | `test_data.py` | exists | Parquet cache and staleness, `isin`, truth pairs, `NA`/empty fields, CSV quotes, France kept, comma file / wrong prefix / duplicate id rejected, `parse_id_list`, `read_id_lists` |
| `split.py` | `test_split.py` | exists | hash determinism, 20% share, matched records follow their S1, folds partition the fixture, cached membership, column limiting |
| `metrics.py` | `test_metrics.py` | exists | `candidate_report`, the 5/7 example, singleton and empty edge cases, precision > recall, missing entities, `breakdown`, CLI |
| `normalize.py` | `test_normalize.py` | M2 | 05 §9 (13 tests) + `test_empty_frame_keeps_schema`, `test_normalise_is_idempotent` (`basic_norm` of `name_norm` is a no-op), `test_no_country_branching` (a France row goes through the same code; `legal_form == "sarl"`) |
| `blocking.py` | `test_blocking.py` | M2 | 06 §8 (11 tests) + `test_block_empty_s1`, `test_block_two_runs_identical` |
| `features.py` | `test_features.py` | M3 | 07 §7 (first 10) + `test_empty_pairs_keep_columns`, `test_symmetric_similarities` (§5) |
| `trainset.py` | `test_trainset.py` | M3 / M1 | 07 §7 `test_label_pairs`, `test_sample_s1_order_independent` + the inner-split tests of 11 §12 |
| `model.py` | `test_model.py` | M4 | below |
| `decision.py` | `test_decision.py` | M5 | 10 §11 (10 tests) + below |
| `evaluate.py` | `test_evaluate.py` | M1 | 11 §12 |
| `errors.py` | `test_errors.py` | M5 | below; categories of `18_ERROR_ANALYSIS_FRAMEWORK.md` |
| `submission.py` | `test_submission.py` | exists + M1 | 9 existing + `write_pairs` tests below |
| `tracking.py` | `test_tracking.py` | exists + M1 | 7 existing + `number=` tests below |
| `pipeline.py` | `test_pipeline.py` | M1 | §3 and §4 |

`tests/test_model.py` (M4; `08_MODEL_SELECTION.md` owns the full list, this is the minimum).
`X` is a 300-row synthetic frame with the `feature_names`
columns, NaN in the allowed ones, and a label that is a noisy function of `core_token_set`.

| Test | Assertion |
|---|---|
| `test_fit_is_deterministic` | two `Matcher(MatcherParams(seed=42, num_threads=1)).fit(X, y)` give identical `predict_proba(X)` |
| `test_probs_float32_in_unit_interval` | dtype float32, all in [0, 1], `len == len(X)`; `chunk_rows=7` equals one chunk |
| `test_save_load_round_trip` | `save(tmp)` then `Matcher.load(tmp)`: same `feature_names`, same params, equal probabilities |
| `test_column_mismatch_raises` | an extra, a missing and a reordered column each raise `ValueError` naming the columns |
| `test_heuristic_backend_is_nanmax_of_sims` | `backend="heuristic"`: `prob == nanmax(sim_name_char, sim_name_addr_word, sim_addr_char)` row-wise, 0.0 when all are NaN; `fit` on an empty `X` is a no-op |
| `test_logreg_backend_imputes_nan` | `backend="logreg"` fits on `X` with NaN (−1 plus flags) and predicts |
| `test_early_stopping_uses_tune` | with `X_val, y_val` and `early_stopping=5`, `n_estimators=200` on a separable problem, the best iteration is < 200 |
| `test_importance_covers_features` | `importance()` index == `feature_names`, values ≥ 0 |

`tests/test_decision.py` (M5): the ten tests of 10 §11 (`test_one_to_one_keeps_highest_prob_s1`,
`test_tau_abs`, `test_tau_rel`, `test_tau_single_empties_entity`, `test_max_matches_cap`,
`test_tune_equals_metrics`, `test_tie_prefers_conservative`, `test_grid_size_and_validation`,
`test_deterministic`, `test_schema`) run on the `pairs_toy` fixture (§7), which is built to
hold their cases: `S2-x` scored 0.9 by `S1-b` and 0.7 by `S1-a`; `S1-a` left with
0.9/0.6/0.4 after 1-to-1; `S1-c` with `p_max = 0.55`; equal 0.80 probabilities under
`S1-b` for the cap tie-break; `S1-d` without candidates and the truth pair `S1-a → S3-a9`
outside the candidates. Additions: `test_decide_is_idempotent` (§5) and
`test_known_answers_on_pairs_toy` (the four rules worked in §7; `tune(grid=Grid())` reaches
0.6558, the best the default grid allows).

`tests/test_errors.py` (M5): one test per category of `errors.tag_errors` (categories and
detection rules of `18_ERROR_ANALYSIS_FRAMEWORK.md`), each triggered by one hand-built pair
on a 3-entity fold; `test_cross_country_is_zero` (a cross-country pair can never be produced
because `block` partitions by country, so the category exists as a guard and is 0 on any
pipeline output); `test_categories_partition_errors` (every false or missed pair has exactly
one tag, `n_fp + n_fn == len(tagged)`, `n_false_singleton` counts matched entities predicted
empty); `test_perfect_prediction_has_no_rows`.

`tests/test_submission.py`, `write_pairs` additions (M1): `test_write_pairs_equals_write_id_lists`
(bytes of `write_pairs(matches, candidates, s1_ids, tmp)` == bytes of `write_submission`
on `pairs_to_lists` of the same frames), `test_write_pairs_warns_when_match_not_candidate`
(file still written, `validate` reports the warning), `test_write_pairs_every_s1_present`
(row per id of `s1_ids`, in order, empty list when absent), `test_write_pairs_only_s2_s3_ids`
(an `S1-` id in `matches.entity_id` raises), `test_write_pairs_ids_exist_when_check_ids`
(`validate` with `split_ids(check_ids=True)` accepts the fixture output; an id absent from
the test split is an error), `test_write_pairs_tsv_format_exact` (header, `\n` endings, no
quoting, comma-joined, empty field for singletons: byte-equal to a literal).

`tests/test_tracking.py` additions (M1): `test_new_experiment_with_number`
(`new_experiment("slug", root, template, number=7)` → `v007_slug`, artifacts dir, placeholder
filled) and `test_new_experiment_number_collision_raises` (`number=7` again → `ValueError`
naming the folder; versions are never reused).

## 3. Integration tests (`tests/test_pipeline.py`)

The synthetic vocabulary is tiny, so every `TopKSpec` in these tests uses `min_df=1,
max_df=1.0` (06 §8); the defaults would drop every term. `PipelineConfig` carries
`dataset_dir: Path = C.DATASET` next to `cache_dir` (addition to 02 §5, M1) so the pipeline
never reads `C.DATASET` directly; tests point both at `tmp_path`.

- `test_stages_compose_on_synthetic_dataset`: `normalise_records` on fixture train S1 and
  pool → `block` → `build_features`. Asserts `list(pairs.columns[:5]) == PAIR_COLUMNS`,
  unique pairs sorted by S1, no cross-country pair; `len(X) == len(pairs)`,
  `X.index.equals(pairs.index)`, `list(X.columns) == feature_names(groups)`, every dtype
  float32; `to_id_lists` has a key for every S1 id.
- `test_fit_and_run_fold_on_synthetic_dataset`: `train = load_fold("train", dataset_dir,
  frac=0.5)`, `val = load_fold("val", dataset_dir, frac=0.5)`; `cfg` with
  `MatcherParams(backend="heuristic")`, `Grid(tau_abs=(0.3, 0.7, 0.2), tau_rel=(0.0, 0.7),
  single_delta=(0.0, 0.1), max_matches=(11,))` (12 rules, 10 §4), tiny `n_fit_s1`/`n_tune_s1`; `fitted = pipeline.fit(cfg, train, tmp_path / "art")` (fit side
  empty, must not fail); `metrics, cands, matches = run_fold(cfg, fitted, val)`. Asserts
  `0 <= metrics["f_beta"] <= 1`, `blocking_report(cands, val)["pair_recall"] == 1.0`
  (`acme` is the exact `name_core` of both true matches), `metrics` holds every key of
  11 §9; `Fitted.save` / `Fitted.load` reproduce `matches` exactly.

## 4. End-to-end smoke test

`tests/test_pipeline.py::test_end_to_end_on_synthetic_dataset`, budget **< 5 s**:

1. `fitted = pipeline.fit(cfg, load_fold("train", dataset_dir, frac=0.5), tmp_path / "art")`.
2. `run_fold(cfg, fitted, load_fold("val", dataset_dir, frac=0.5))` returns a metrics dict.
3. `matching, candidates = run_test(cfg, fitted, out_dir=tmp_path / "output")`: both TSVs
   exist, named `C.MATCHING_FILE` and `C.CANDIDATE_FILE`.
4. `s1_ids, valid = submission.split_ids("test", dataset_dir, check_ids=True)`;
   `submission.validate(matching, candidates, s1_ids, valid) == ([], [])`: no errors, and
   no "matches that are not candidates" warning.
5. `read_id_lists(matching)` has exactly the rows `S1-00010` (France) and `S1-00011`, in
   `test_source1.tsv` order; every listed id starts with `S2-`/`S3-`; for every row
   `set(matches) <= set(candidates)`.
6. The France entity has at least one candidate (the pipeline never enumerates countries).

## 5. Property and invariant tests

Cheap, seeded (`np.random.default_rng(0)`), no new dependency. Each module owner adds the
rows that apply; the pipeline test checks the last two across stages.

| Invariant | Where |
|---|---|
| flags (`pass_*`, `*_eq`, `*_missing_*`, `addr_empty_r`, `is_s3`, `non_latin_r`) are never NaN, only 0/1 | `test_features` |
| every similarity is in [0, 1] or NaN; NaN only in the columns 07 §2 allows | `test_features` |
| symmetric features equal under L/R swap: every `nm_*`, `core_*`, `tok_*`, `num_*`, `ad_*` and `*_eq` similarity except the asymmetric-by-design `nm_partial`, `ad_partial` and `ad_contain`; `ctx_*`, `*_l`/`*_r` counts and `addr_empty_r` are one-sided and excluded | `test_features` |
| `decide(decide(scored))` with probabilities carried == `decide(scored)` | `test_decision` |
| `block` output has unique `(source1_entity_id, entity_id)`, sorted by S1, no empty ids | `test_blocking` |
| chunk invariance: `build_features(chunk_rows=2)`, `topk_pass(pool_chunk=half)`, `predict_proba(chunk_rows=7)` equal the unchunked result | each module |
| determinism: two runs of `normalise_records`, `block`, `build_features`, `decide` are frame-equal | each module |
| normalised frames contain no None/NaN: missing is `""`, `addr_tokens == 0` | `test_normalize` |
| **empty input → empty output with the full schema** for every stage (`normalise_records`, `block`, `build_features`, `Matcher.predict_proba`, `decide`, `score_pairs` raises `ValueError("truth is empty")` like `breakdown`) | each module, `test_pipeline` |
| `write_pairs` == `write_submission ∘ pairs_to_lists` | `test_submission` |

## 6. Large-data checks (notebook checklist, not unit tests)

These need the real folds and run inside the experiment notebook, timed with
`tracking.timed`, outputs committed with the notebook:

1. **Recall on val**: `blocking_report(cands, val)["pair_recall"] ≥ 0.97`, cands mean ≤ 40,
   p95 ≤ 100; per-pass and per-slice recall (06 §5); `error_samples(kind="missed")` inspected.
2. **RAM guard**: `mem_guard` asserts RSS < 6 GB at every stage boundary; `peak_rss_gb`
   logged (02 §7).
3. **Timing on a 20k S1 chunk** before any full run: extrapolate normalise, each blocking
   pass, features + predict to the full fold and to test India (02 §7, 06 §4); P2 above
   30 min extrapolated → `max_df=0.05` or 4-grams first.
4. **Determinism on real data**: `block` twice on one country chunk, `pd.util.hash_pandas_
   object(pairs).sum()` equal.
5. **Schema at boundaries**: the `assert list(df.columns[:n]) == COLS` lines (02 §3) run on
   real frames; `predict_proba` accepts the real feature frame.
6. **Test output sanity** (11 §10) before an upload; `make validate` PASS.

## 7. Fixtures

`dataset_dir` is shared by everyone, so extend it only by adding rows or columns that keep
the existing tests green. Several tests pin its exact content: `test_folds_partition_
train_data` (3 train S1), `test_truth_pairs_explode_lists_and_drop_singletons` and
`test_ground_truth_maps_singletons_to_empty_set` (every pair), `test_file_format_is_exact`
(the two test S1 rows), `test_sources_load_as_plain_strings` (`SOURCE_COLUMNS`). In
practice: train rows, truth and columns are frozen; test-split S2/S3 rows can be added.
Module tests build their own small frames inline (a `pd.DataFrame` literal beats a fixture
nobody else reads); only split, data, submission and pipeline tests need `dataset_dir`. A
module that needs a richer dataset adds a **new** fixture rather than growing this one.

`tests/conftest.py` and its fixtures belong to M1 (14 §6): a fixture change is a PR that
M1 reviews. Second shared fixture, `pairs_toy`, for decision and evaluation tests. It is the
toy frame of 10 §11: three scored S1 entities, eight pool ids, one entity without
candidates, one truth pair outside the candidates.

```python
@pytest.fixture
def pairs_toy():
    """(scored, truth_pairs, s1_ids) where every decision component changes the answer.

    S1-a: true a1 0.90 and a2 0.60, decoy a3 0.40, and x at 0.70 that S1-b claims at 0.90
    (1-to-1); its third true match a9 was missed by blocking.  S1-b: true x and b1 0.80,
    decoy b2 0.80 (cap tie-break).  S1-c: a singleton whose best candidate scores 0.55
    (false-merge trap).  S1-d: one true match and no candidates at all.
    """
    scored = pd.DataFrame({
        "source1_entity_id": ["S1-a"] * 4 + ["S1-b"] * 3 + ["S1-c"] * 2,
        "entity_id": ["S2-a1", "S2-x", "S3-a2", "S2-a3", "S2-x", "S3-b1", "S3-b2",
                      "S2-c1", "S3-c2"],
        "prob": np.float32([0.90, 0.70, 0.60, 0.40, 0.90, 0.80, 0.80, 0.55, 0.30]),
    })
    truth = pd.DataFrame({
        "source1_entity_id": ["S1-a", "S1-a", "S1-a", "S1-b", "S1-b", "S1-d"],
        "entity_id": ["S2-a1", "S3-a2", "S3-a9", "S2-x", "S3-b1", "S2-d1"],
    })
    return scored, truth, pd.Series(["S1-a", "S1-b", "S1-c", "S1-d"])
```

Worked answers, `DecisionRule(tau_abs, tau_rel, tau_single, max_matches)` with
`one_to_one=True`; `S1-d` always scores 0 and `S1-a` at best 10/11 because `a9` is
unreachable:

| Rule | S1-a | S1-b | S1-c | macro F0.5 |
|---|---|---|---|---|
| R0 `(0.5, 0.0, 0.5, 11)` | `{a1, a2}` 0.9091 | `{x, b1, b2}` 0.7143 | `{c1}` 0.0, false merge | **0.4058** |
| R1 `(0.5, 0.0, 0.6, 11)` | same | same | empty 1.0 | **0.6558** |
| R2 `(0.5, 0.0, 0.6, 2)` | same | `{x, b1}` 1.0, tie → smaller id | empty 1.0 | **0.7273** |
| R3 `(0.5, 0.7, 0.6, 2)` | `{a1}` 0.7143: 0.60 < 0.7 × 0.90 | `{x, b1}` 1.0 | empty 1.0 | **0.6786** |

`tune` with the default `Grid()` stops at 0.6558: `max_matches ∈ {4, 6, 11}` cannot split
`S1-b`'s 0.80 tie, so R2 is out of its reach; `test_known_answers_on_pairs_toy` asserts both.

## 8. When a test fails in someone else's module

1. Do not edit their module or their tests to make yours pass, and never `skip` or `xfail`
   them: a red test is information. Tests follow the module owner (14 §6); touching another
   owner's module is a PR with that owner as reviewer, even for a one-line fix.
2. Reproduce on a clean checkout of `main`. If `main` is green and your branch is red, the
   failure is yours: your change broke a contract (02 §4), whichever file holds the
   assertion. Fix your side, or propose the contract change through 02 (M1 merges it, with
   05/06/07/11 updated in the same PR).
3. If `main` itself is red, say so in chat and to M1 at once: 14 §9 applies (revert of the
   breaking merge, fix forward on the author's branch, new PR); nothing else merges until
   `main` is green.
4. Otherwise open an issue or a PR comment with the test name, the assertion, the commit
   hash and the smallest input that fails, and ping the owner. The owner fixes it on a
   `fix/` branch the same day; the lead (M1) arbitrates disagreements.

## 9. PR checklist, test lines

- [ ] New module → new `tests/test_<module>.py`; every public function has at least one
      test; the module's schema constant is asserted in at least one test.
- [ ] `make lint test` output pasted in the PR description (ruff `All checks passed!` and
      pytest's `N passed in X s` line), run on the branch head after the last commit.
- [ ] Suite still under 30 s; no test reads `dataset/`; no new pytest marks.
- [ ] Test names say behaviour and expectation (`test_<what>_<expectation>`), the fixture
      content they rely on is stated in a comment when it is not obvious.
