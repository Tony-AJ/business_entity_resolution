# Business Entity Resolution

Amazon ML Challenge 2026. Business records from three sources share no identifiers. For
every Source 1 entity, find all Source 2 / Source 3 records that describe the same
business. Scoring is macro F0.5 per Source 1 entity, so precision counts twice as much as
recall.

- Task, data, output format: [docs/PROBLEM_STATEMENT.md](docs/PROBLEM_STATEMENT.md)
- Timeline, upload budget, required artefacts: [docs/GUIDELINES.md](docs/GUIDELINES.md)
- Team rules (notebooks, versioning, leaderboard, experiment plan, git):
  [.claude/rules/project-rules.md](.claude/rules/project-rules.md)
- Research plan and module blueprints (architecture, contracts, team split, per-module
  strategy, validation, testing, git, leaderboard): [docs/plan/00_MASTER_PLAN.md](docs/plan/00_MASTER_PLAN.md)

## Setup

Requires Python 3.12.

```bash
make setup    # .venv: pinned requirements.txt + editable package + pytest, ruff, jupytext
make hooks    # enable the versioned git hooks in .githooks/
```

Unzip the organisers' `student_resource.zip` into `dataset/` (gitignored), giving
`dataset/student_resource/dataset/{train,test}/`. A flat `dataset/{train,test}/` works
too. Then parse everything once:

```bash
make cache    # ~1 min: every TSV to Parquet in <dataset>/.cache/
```

## Experiment workflow

1. `make experiment NAME=name_tfidf` creates `experiments/vNNN_name_tfidf/` with the
   documented notebook template.
2. Fill the notebook: hypothesis, method, evaluation on the fixed validation fold, error
   analysis. Run it in VS Code/Jupyter, or headless with `make nb NB=<path>`.
3. Its last cell calls `log_result(...)`, writing `metrics.json` and the version's row in
   `experiments/experiments.csv`.
4. Commit it: `exp(vNNN): <change>, local F0.5 0.xxxx`.
5. Shortlisted versions only: run test inference, `make validate`, upload, add an entry
   to [LEADERBOARD.md](LEADERBOARD.md), then `make public V=vNNN SCORE=0.xxxx`.

## Commands

| Command | What it does |
|---|---|
| `make cache` | parse all TSVs once into a Parquet cache |
| `make experiment NAME=<slug> V=<n>` | `experiments/vNNN_<slug>/` from the template; `V` from your reserved range |
| `make nb NB=<notebook>` | execute a notebook headless, saving outputs in place |
| `make public V=vNNN SCORE=<f>` | record a leaderboard score in `experiments.csv` |
| `make validate` | check `output/*.tsv` with our checker and the organisers' validator |
| `make score PRED=<tsv> TRUTH=<tsv>` | macro F0.5 of a matching file against labels |
| `make package TEAM=<name> V=<vNNN_slug>` | the final submission zip in `build/` ([Final package](#final-package)) |
| `make test` / `make lint` | pytest on synthetic data / ruff |

## Layout

```
src/entity_resolution/     shared, tested library used by every notebook
├── config.py              paths, file schema, constants
├── data.py                TSV loaders, Parquet cache, fast ID filtering
├── split.py               fixed validation split (20% of Source 1, hashed)
├── mock.py                test-shaped mock fold: the test's pool size and pool-per-S1 ratio
├── metrics.py             macro F0.5, breakdown, blocking candidate report
├── token_maps.py          legal forms, street tokens, regions (hand-typed domain knowledge)
├── normalize.py           name/address normalisation, learned transliteration map
├── blocking.py            candidate generation: exact keys + TF-IDF top-k, per country
├── features.py            47 pair features, chunked
├── trainset.py            inner fit/tune split, S1 sampling, pair labels
├── model.py               LightGBM / XGBoost (CUDA GPU) / logistic / heuristic matcher
├── decision.py            1-to-1 set rule, its macro F0.5 grid tuning, expected-F0.5 decoding
├── evaluate.py            vectorised metric, blocking report, slices, error samples
├── pipeline.py            fit / run_fold / run_mock / run_test: the stages wired together
├── stacking.py            competition and anchor features from stage-1 probabilities
├── twostage.py            stage-1 filter (final candidate set) + stage-2 matcher trained on the mock
├── submission.py          write and validate the two output files
└── tracking.py            experiment registry, timings
experiments/
├── _template/             documented notebook template
├── vNNN_<slug>/           one folder per experiment: notebook, metrics.json, artifacts/
└── experiments.csv        one row per version: change, local/mock/public F0.5, commit
LEADERBOARD.md             every leaderboard upload, with budget
TRACKER.md                 who does what, task status per day
docs/                      challenge brief and guidelines
scripts/                   package_submission.sh: the final zip (make package)
tests/                     pytest on synthetic TSVs
.githooks/                 identity, commit format, size and data guards
```

`dataset/`, `output/`, `models/` and `experiments/*/artifacts/` stay local.

## Local testing: the mock fold

The fixed val fold is a 20 % sample of train: its entities meet 3–6× fewer same-name records
of other businesses than test entities do, and the first two uploads scored 0.031 below
val. `mock.build_mock` rebuilds the test's shape from all of train, per country (the test's
pool size and pool records per S1; dropped S1 leave their records as unowned decoys, 40 %
of the pool as on test). Every present S1 is scored and competes in the pool-side 1-to-1;
rules are tuned on its tune entities and **mock F0.5** is measured on its val entities
(`pipeline.run_mock`, `tune_mock`, `mock_scores`). It is the KEEP/DROP number from v103 on.

## Status

The V1 pipeline of the research plan is in place (`src/entity_resolution/`, tested) and
`v001_base_model` is its first run. Progress per member: [TRACKER.md](TRACKER.md).

## Reproducing the submission

From a clean clone with the organisers' zip unzipped into `dataset/` (see Setup):

```bash
make setup                   # pinned environment (requirements.txt)
make cache                   # raw TSVs -> Parquet, ~1 min
make nb NB=experiments/v001_base_model/v001_base_model.ipynb
make validate                # our checker + the organisers' validator on output/
```

The notebook runs every stage through `entity_resolution.pipeline`: normalisation (cached
per split under `dataset/.cache/pipeline/norm/`), the transliteration map learned from the
train fold's pairs, blocking per country (cached under `dataset/.cache/pipeline/pairs/`),
features, LightGBM trained on the fit side of the train fold, the decision rule tuned on the
tune side, one scoring of the validation fold, and test inference writing
`output/matching_results.tsv` and `output/candidate_pairs.tsv`. Model, rule, token map and
configuration are saved under `experiments/v001_base_model/artifacts/`. Seeds are fixed
(split 42, inner split 4242, samples 7, LightGBM 42, deterministic mode), so a rerun
reproduces the files.

## Final package

`make package` builds the zip the organisers ask for in `build/` (gitignored), from the
repository at HEAD, without touching the working tree. The checklist is
[docs/plan/16](docs/plan/16_FINAL_SUBMISSION_CHECKLIST.md).

```bash
make package TEAM=<team> V=vNNN_<slug> OUT=submissions/vNNN ARGS=--dry-run   # preflight only
make package TEAM=<team> V=vNNN_<slug> OUT=submissions/vNNN                  # build the zip
```

```
build/<team>_submission.zip            these three at the zip root, no wrapping folder
├── output/                            both TSVs from OUT (default output/)
├── code/business_entity_resolution/   git archive HEAD: tracked files only
│   └── src/notebooks/                 vNNN_<slug>.ipynb + metrics.json, copied from HEAD
└── Documentation_template.md          docs/methodology.md (ARGS="--doc <path>" for another)
```

| Stage | What it does |
|---|---|
| preflight | refuses uncommitted changes to tracked files; needs V's notebook and `metrics.json` committed, both TSVs with the right header and the write-up; HEAD must not track `dataset/`, `output/`, `.venv`, `artifacts/` or a file over 50 MiB |
| validate | our checker, then the organisers' validator (`ARGS=--check-ids` adds the ID check to both, a few GB of RAM) |
| build | stages `build/<team>_submission/`, re-checks the staged code tree, zips it |
| report | sizes; sha256 of the zip and both TSVs in `build/<team>_submission.sha256`; `unzip -l` in `build/<team>_submission.contents.txt` |

`OUT=submissions/vNNN` ships the exact bytes of that upload. Then finish checklist §1
step 5 (unpack into a temp dir, validate inside it, fresh-venv smoke test) and record the
size and sha256 in [LEADERBOARD.md](LEADERBOARD.md).
