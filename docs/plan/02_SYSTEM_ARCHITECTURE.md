# 02 — System architecture and module contracts

Audience: everyone; this is the contract every module is written against. Change it only by
a PR that the lead (M1) merges, and update every document that quotes it.

## 1. Pipeline

```
raw TSV (dataset/)                       data.load_source / load_sources      [exists]
  → validation + Parquet cache           data._parse_source, make cache       [exists]
  → fixed folds                          split.load_fold("train"|"val")       [exists]
  → inner fit/tune split (train only)    trainset.inner_split                 [M1]
  → normalisation                        normalize.normalise_records          [M2]
  → blocking, per country                blocking.block  → pairs frame        [M2]
  → candidate recall report              evaluate.blocking_report             [M1]
  → pair features (chunked)              features.build_features              [M3]
  → training pairs + labels              trainset.sample_s1 / label_pairs     [M3]
  → hard negatives (round 2)             trainset.mine_false_positives        [M4]
  → matcher                              model.Matcher.fit / predict_proba    [M4]
  → decision rule, tuned on tune split   decision.tune / decide               [M5]
  → macro F0.5 + slices + error tags     evaluate.*, errors.tag_errors        [M5]
  → matching_results.tsv + candidate_pairs.tsv   submission.write_pairs       [M1]
  → validator                            make validate                        [exists]
```

`pipeline.py` (M1) wires these; notebooks call the pipeline, never re-implement a stage.

## 2. Stage table

| Stage | Input | Output | Algorithm / structure | Metric | Failure modes | Tests |
|---|---|---|---|---|---|---|
| Load + cache | TSV | Parquet, str columns | `read_tsv(dtype=str, na_filter=False)` | rows, time | comma-parsed file, duplicate ids | test_data (exists) |
| Folds | truth pairs | `Fold(s1, s2, s3, pairs)` | id hashing, seed 42, 20% val | fold sizes | changed seed/fraction → incomparable scores | test_split (exists) |
| Inner split | train Fold | fit Fold, tune Fold | same hashing, seed 4242, 25% tune | disjointness | pool records shared across sides | test_trainset |
| Normalise | records | normalised frame (§4.1) | vectorised regex, token maps, anyascii | none (unit tests) | NaN, empty string loss, non-Latin untouched | test_normalize |
| Block | normalised S1 + pool | pairs frame (§4.2) | exact keys + TF-IDF top-k (`sparse_dot_topn`) | pair recall, cands/S1, seconds | recall < 0.97, group explosions, RAM | test_blocking |
| Features | pairs, S1, pool | float32 matrix (§4.3) | rapidfuzz, sparse Jaccard, group context | none (unit tests) | NaN where not allowed, chunk splits a group | test_features |
| Train set | pairs, truth | labelled pairs | merge indicator, S1 sampling by hash | positive rate | leakage from tune/val | test_trainset |
| Matcher | X, y | probabilities | LightGBM binary; LR baseline; heuristic stub | tune logloss, AUC | miscalibration, column mismatch | test_model |
| Decision | scored pairs | matches frame | 1-to-1 + thresholds + cap, grid tuned | macro F0.5 on tune | tuned on val (leak), France-specific rules | test_decision |
| Evaluate | matches, Fold | breakdown, slices, errors | `metrics.breakdown`, vectorised counts | macro F0.5, singleton F0.5 | Python loops on 10M rows | test_evaluate |
| Write | matches, candidates | two TSVs | pyarrow group-by → id lists | validator PASS | id not in test, missing S1 | test_submission |

## 3. Module map (`src/entity_resolution/`)

| File | Owner | Depends on |
|---|---|---|
| `normalize.py` | M2 | `config` |
| `blocking.py` | M2 | `config`, `normalize` column names |
| `features.py` | M3 | `config`, pairs/normalised column names |
| `trainset.py` | M3 (`inner_split` by M1) | `split`, `data.isin` |
| `model.py` | M4 | `features.feature_names` |
| `decision.py` | M5 | `config` |
| `evaluate.py`, `errors.py` | M1 / M5 | `metrics`, `split` |
| `pipeline.py` | M1 | all of the above |
| `submission.py` (+`write_pairs`), `tracking.py` (+`number=`) | M1 | existing |

Conventions: pandas 3 Arrow `str` columns (never `object`); positional `int32` indices inside a
module, string ids at boundaries; float32 features; functions are pure and chunk-safe;
schema constants (`NORM_COLUMNS`, `PAIR_COLUMNS`, `SCORED_COLUMNS`, `MATCH_COLUMNS`) are
asserted at every boundary with `assert list(df.columns[:n]) == COLS`.

## 4. Data contracts

### 4.1 Normalised records (`normalize.NORM_COLUMNS`)

| Column | dtype | Meaning |
|---|---|---|
| `entity_id`, `country` | str | verbatim |
| `non_latin` | bool | row contained non-Latin letters before transliteration |
| `name_norm` | str | transliterated, NFKD accent-stripped, casefolded, `&`→` and `, punctuation→space, collapsed |
| `name_core` | str | `name_norm` minus legal-form tokens |
| `legal_form` | str | canonical legal tokens, space-joined, `""` if none |
| `name_first` | str | first token of `name_core` |
| `name_sorted` | str | `name_core` tokens sorted, space-joined |
| `name_squash` | str | `name_core` letters+digits only, leet-folded, domain suffix removed |
| `addr_norm` | str | address normalised like `name_norm`, abbreviations expanded |
| `addr_nums` | str | numeric tokens of the address in order, space-joined |
| `postcode` | str | 5- or 6-digit token if any, else `""` |
| `region` | str | canonical state/region token from the region map, else `""` |
| `addr_last` | str | last two non-numeric tokens (city hint) |
| `addr_tokens` | int16 | token count of `addr_norm` |
| `name_addr` | str | `name_core + " " + addr_norm` (P3 retrieval text) |

### 4.2 Candidate pairs (`blocking.PAIR_COLUMNS`), sorted by `source1_entity_id`, unique pairs

| Column | dtype | Meaning |
|---|---|---|
| `source1_entity_id`, `entity_id` | str | the pair (pool id is S2 or S3) |
| `pass` | uint8 | bitmask: exact_core 1, exact_sorted 2, exact_squash 4, name_char 8, name_addr_word 16, addr_char 32 |
| `sim_name_char` | float32 | P2 cosine, NaN if P2 did not produce the pair |
| `sim_name_addr_word` | float32 | P3 cosine, NaN likewise |
| `sim_addr_char` | float32 | P4 cosine, NaN likewise |

`to_id_lists(pairs, s1_ids)` gives `dict[str, list[str]]` with every S1 present (empty list
if no candidate) for `metrics.candidate_report`; use it on val-size frames only.

### 4.3 Features: float32 frame, `index == pairs.index`, columns `features.feature_names(groups)`.
NaN allowed only where a feature is undefined (documented per feature in 07); the model
handles NaN natively (LightGBM) or after `-1` imputation plus flags (LR).

### 4.4 Scored pairs (`decision.SCORED_COLUMNS`): `source1_entity_id`, `entity_id`, `prob` (float32).

### 4.5 Matches (`decision.MATCH_COLUMNS`): `source1_entity_id`, `entity_id`. Id lists are only
materialised by `submission.write_pairs` or `evaluate.pairs_to_lists` (val-size).

## 5. Signatures

```python
# normalize.py (M2)
@dataclass(frozen=True)
class NormaliseConfig:
    transliterate: bool = True; strip_legal: bool = True; expand_abbrev: bool = True
    region_map: Mapping[str, str] | None = None      # None → static REGION_ABBREV
def transliterate(s: pd.Series) -> pd.Series          # anyascii on non-Latin rows only
def basic_norm(s: pd.Series) -> pd.Series
def normalise_names(names: pd.Series, cfg: NormaliseConfig) -> pd.DataFrame
def normalise_addresses(addr: pd.Series, cfg: NormaliseConfig) -> pd.DataFrame
def normalise_records(df: pd.DataFrame, cfg: NormaliseConfig = NormaliseConfig()) -> pd.DataFrame
def fit_region_map(pairs: pd.DataFrame, s1n: pd.DataFrame, pooln: pd.DataFrame,
                   min_count: int = 50) -> dict[str, str]          # train fold only (B4)

# blocking.py (M2)
@dataclass(frozen=True)
class TopKSpec:
    column: str; analyzer: str = "char_wb"; ngram: tuple[int, int] = (3, 3); top_k: int = 30
    min_sim: float = 0.30; max_df: float = 0.2; min_df: int = 2; sublinear_tf: bool = True
@dataclass(frozen=True)
class BlockingConfig:
    exact_keys: tuple[str, ...] = ("name_core", "name_sorted", "name_squash")
    exact_max_group: int = 50
    name_char: TopKSpec | None = TopKSpec("name_core")
    name_addr_word: TopKSpec | None = TopKSpec("name_addr", "word", (1, 1), 20, 0.20, 0.05)
    addr_char: TopKSpec | None = None
    max_per_s1: int = 60; s1_chunk: int = 50_000; pool_chunk: int = 2_000_000; n_threads: int = 12
def exact_pass(s1n, pooln, key: str, max_group: int) -> pd.DataFrame     # s1_idx, pool_idx (int32)
def topk_pass(s1n, pooln, spec: TopKSpec, s1_chunk, pool_chunk, n_threads, seed) -> pd.DataFrame  # + sim
def union_passes(parts: dict[str, pd.DataFrame], max_per_s1: int) -> pd.DataFrame
def block_partition(s1n, pooln, cfg: BlockingConfig) -> pd.DataFrame     # PAIR_COLUMNS, one country
def block(s1n, pooln, cfg: BlockingConfig, cache_dir: Path | None = None, tag: str = "") -> pd.DataFrame
def to_id_lists(pairs: pd.DataFrame, s1_ids: pd.Series) -> dict[str, list[str]]

# features.py (M3)
FeatureGroup = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.DataFrame]
REGISTRY: dict[str, FeatureGroup]      # blocking, name_fuzzy, name_tokens, legal, numeric, address, context, meta
def feature_names(groups: Sequence[str]) -> list[str]
def iter_chunks(pairs: pd.DataFrame, chunk_rows: int) -> Iterator[slice]  # never splits an S1 group
def build_features(pairs, s1n, pooln, groups=tuple(REGISTRY), chunk_rows=2_000_000) -> pd.DataFrame

# trainset.py (M3; inner_split by M1)
INNER_SEED = 4242; INNER_FRAC = 0.25; SAMPLE_SEED = 7
def inner_split(train: Fold) -> tuple[Fold, Fold]                 # ("fit", "tune")
def sample_s1(s1: pd.DataFrame, n: int, seed: int = SAMPLE_SEED) -> pd.DataFrame
def label_pairs(pairs: pd.DataFrame, truth_pairs: pd.DataFrame) -> pd.DataFrame   # + label int8
def mine_false_positives(scored, truth_pairs, threshold: float = 0.5) -> pd.DataFrame

# model.py (M4)
@dataclass
class MatcherParams:
    backend: str = "lgbm"            # "lgbm" | "logreg" | "heuristic"
    num_leaves: int = 63; learning_rate: float = 0.05; n_estimators: int = 2000
    early_stopping: int = 100; feature_fraction: float = 0.8; bagging_fraction: float = 0.8
    bagging_freq: int = 1; min_data_in_leaf: int = 200; lambda_l2: float = 1.0
    max_bin: int = 255; scale_pos_weight: float = 1.0; seed: int = 42; num_threads: int = 12
class Matcher:
    def fit(self, X, y, X_val=None, y_val=None) -> "Matcher"
    def predict_proba(self, X, chunk_rows: int = 2_000_000) -> np.ndarray   # float32, refuses column mismatch
    def importance(self) -> pd.Series
    def save(self, dir: Path) -> Path;  @classmethod def load(cls, dir: Path) -> "Matcher"

# decision.py (M5)
@dataclass(frozen=True)
class DecisionRule:
    tau_abs: float = 0.5; tau_rel: float = 0.0; tau_single: float = 0.5
    max_matches: int = 11; one_to_one: bool = True
@dataclass(frozen=True)
class Grid: ...                                                   # see 10_DECISION_LAYER.md
def decide(scored: pd.DataFrame, rule: DecisionRule) -> pd.DataFrame            # MATCH_COLUMNS
def tune(scored, s1_ids: pd.Series, truth_pairs, grid: Grid = Grid()) -> tuple[DecisionRule, pd.DataFrame]

# evaluate.py (M1) / errors.py (M5)
def pairs_to_lists(pairs, s1_ids) -> dict[str, list[str]]
def pair_recall(cand_pairs, truth_pairs) -> float
def macro_f05_from_counts(tp: np.ndarray, n_pred: np.ndarray, n_true: np.ndarray) -> float
def score_pairs(pred_pairs, fold: Fold) -> dict                   # metrics.breakdown
def blocking_report(cand_pairs, fold: Fold) -> dict               # metrics.candidate_report
def slice_report(pred_pairs, fold, s1n, pooln) -> pd.DataFrame
def error_samples(pred_pairs, fold, s1n, pooln, kind: str, n: int = 20) -> pd.DataFrame
def harder_fold(fold: Fold, drop_frac: float = 0.2, seed: int = 99) -> Fold
def tag_errors(pred_pairs, fold: Fold, s1n, pooln, scored=None) -> pd.DataFrame   # errors.py

# pipeline.py (M1)
@dataclass
class PipelineConfig:
    normalise: NormaliseConfig = NormaliseConfig(); blocking: BlockingConfig = BlockingConfig()
    feature_groups: tuple[str, ...] = tuple(REGISTRY); model: MatcherParams = MatcherParams()
    grid: Grid = Grid(); n_fit_s1: int = 200_000; n_tune_s1: int = 100_000
    cache_dir: Path = C.DATASET / ".cache" / "pipeline"
@dataclass
class Fitted:
    matcher: Matcher; rule: DecisionRule; tune_table: pd.DataFrame; config: PipelineConfig
    def save(self, dir: Path) -> Path;  @classmethod def load(cls, dir: Path) -> "Fitted"
def prepare(s1, pool, cfg, tag: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]  # s1n, pooln, pairs (cached)
def score(pairs, s1n, pooln, matcher, cfg) -> pd.DataFrame        # SCORED_COLUMNS, chunked
def fit(cfg, train: Fold, out: Path) -> Fitted
def run_fold(cfg, fitted, fold: Fold) -> tuple[dict, pd.DataFrame, pd.DataFrame]   # metrics, candidates, matches
def run_test(cfg, fitted, out_dir: Path = C.OUTPUT) -> tuple[Path, Path]

# submission.py (M1, addition)
def write_pairs(matches, candidates, s1_ids: Sequence[str], out_dir: Path = C.OUTPUT) -> tuple[Path, Path]
# tracking.py (M1, addition)
def new_experiment(slug, root=C.EXPERIMENTS, template=TEMPLATE, number: int | None = None) -> Path
```

The notebook template stubs map 1:1: `normalise` → `normalise_records`, `block` → `block` +
`to_id_lists`, `pair_features` → `build_features`, `fit_matcher` → `Matcher.fit`,
`decide` → `decide`. Notebooks import these; they never copy code.

## 6. Orchestration (`pipeline.fit` and `run_*`)

```
fit(cfg, train):
  fit_fold, tune_fold = inner_split(train)
  fit_s1  = sample_s1(fit_fold.s1, cfg.n_fit_s1);   tune_s1 = sample_s1(tune_fold.s1, cfg.n_tune_s1)
  s1n, pooln, pairs = prepare(fit_s1, pool(fit_fold), cfg, "fit")        # normalise + block, cached
  X = build_features(pairs, s1n, pooln); y = label_pairs(pairs, fit_fold.pairs).label
  s1t, poolt, pairs_t = prepare(tune_s1, pool(tune_fold), cfg, "tune"); X_t, y_t likewise
  matcher = Matcher(cfg.model).fit(X, y, X_t, y_t)
  scored_t = score(pairs_t, s1t, poolt, matcher, cfg)
  rule, table = tune(scored_t, tune_s1.entity_id, tune_fold.pairs, cfg.grid)
  return Fitted(matcher, rule, table, cfg)
run_fold(cfg, fitted, fold):   prepare(fold.s1, pool(fold), cfg, fold.name) → score → decide → score_pairs,
                               blocking_report, slice_report, timings
run_test(cfg, fitted):         load_sources("test") → prepare("test") → score → decide (per country) →
                               write_pairs(matches, pairs, test_s1_ids)
```

`pool(fold) = concat(fold.s2, fold.s3)`. The candidates written to `candidate_pairs.tsv` are
exactly the `pairs` frame that `score` consumed.

## 7. Budget and chunking (calibrate on a 20k-row S1 chunk before running anything big)

| Step | Test, per country (India 810k × 4.72M) | Val (441k × 2.06M) | Peak RAM |
|---|---|---|---|
| Normalise | 2–3 min | 1 min | 1 GB |
| P1 exact merges | < 1 min | seconds | 0.5 GB |
| P2 name char 3-gram, k=30, max_df 0.2 | 10–30 min | 2–3 min | 2.5 GB (pool matrix transposed once) |
| P3 word name+addr, k=20 | 1–2 min | < 1 min | 1 GB |
| Features + predict, 2M-row chunks | 10–15 min (≈24M pairs) | 3 min | 1.5 GB transient |
| Decide + write | 5 min | 1 min | 1 GB |

Rules that keep RAM under 6 GB: read only needed columns from the per-country normalised
Parquet; fit the TF-IDF vocabulary on a 500k-row sample, then transform in 500k-row chunks
(float32 data, int32 indices); transpose the pool matrix once per pass and `del` the
original; keep pairs as int32 positions until `block_partition` returns; never hold a full
test feature matrix (score per chunk); process countries sequentially with `gc.collect()`;
assert `psutil` RSS < 6 GB at stage boundaries (`mem_guard`). If India P2 extrapolates above
30 min, raise `min_df`, lower `max_df` to 0.05, or switch to 4-grams.

```python
def topk_pass(s1n, pooln, spec, s1_chunk, pool_chunk, n_threads, seed):
    vec = TfidfVectorizer(analyzer=spec.analyzer, ngram_range=spec.ngram, min_df=spec.min_df,
                          max_df=spec.max_df, sublinear_tf=spec.sublinear_tf, dtype=np.float32)
    vec.fit(sample(concat([s1n[spec.column], pooln[spec.column]]), 500_000, seed))
    tf = lambda t: sp.vstack([vec.transform(t[i:i + 500_000]) for i in range(0, len(t), 500_000)], "csr")
    bts = []
    for p0 in range(0, len(pooln), pool_chunk):
        B = tf(pooln[spec.column].iloc[p0:p0 + pool_chunk]); bts.append(B.T.tocsr()); del B; gc.collect()
    out = []
    for a0 in range(0, len(s1n), s1_chunk):
        A = tf(s1n[spec.column].iloc[a0:a0 + s1_chunk])
        cs = [sp_matmul_topn(A, BT, top_n=spec.top_k, threshold=spec.min_sim, sort=True, n_threads=n_threads)
              for BT in bts]
        Cm = cs[0] if len(cs) == 1 else zip_sp_matmul_topn(cs, top_n=spec.top_k)
        coo = Cm.tocoo()
        out.append(pd.DataFrame({"s1_idx": coo.row + a0, "pool_idx": coo.col, "sim": coo.data}))
    return pd.concat(out, ignore_index=True)
```

## 8. Artifacts and caches

```
dataset/.cache/pipeline/<tag>/<country>/normalised_{s1,pool}.parquet   (tag = fit|tune|val|test)
dataset/.cache/pipeline/<tag>/<country>/pairs_<blocking-config-hash>.parquet
experiments/vNNN_<slug>/artifacts/{model/, rule.json, tune_table.csv, scored_val.parquet, errors.parquet}
output/{matching_results.tsv, candidate_pairs.tsv}
```

Caches are keyed by the config hash, so a new model version reuses the test candidates of an
unchanged blocking config. Everything under `dataset/`, `output/` and `artifacts/` is gitignored.

## 9. Failure modes the architecture guards against

1. Blocking miss → invisible downstream: `blocking_report` runs in every notebook; recall
   per pass and per slice (`non_latin`, ambiguous names) is logged.
2. Leakage: vectorisers are unsupervised and refit per split; `fit_region_map` and every
   learned dictionary use the train fold's pairs only; val is only read by `run_fold`.
3. Group-wise features across chunk boundaries: `iter_chunks` aligns to S1 groups.
4. Schema drift between five people: column constants asserted; `predict_proba` refuses
   unknown or missing columns; the `heuristic` backend keeps the pipeline runnable while a
   stage is being replaced.
5. Python-object blow-ups: id lists exist only at the writer; everything else is columnar.
6. France: no country enumeration anywhere; `slice_report` on the test output compares the
   France slice (match rate, cands/S1, `p_max` histogram) with US/India before an upload.
