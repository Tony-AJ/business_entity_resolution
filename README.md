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
| `make experiment NAME=<slug>` | next `experiments/vNNN_<slug>/` from the template |
| `make nb NB=<notebook>` | execute a notebook headless, saving outputs in place |
| `make public V=vNNN SCORE=<f>` | record a leaderboard score in `experiments.csv` |
| `make validate` | check `output/*.tsv` with our checker and the organisers' validator |
| `make score PRED=<tsv> TRUTH=<tsv>` | macro F0.5 of a matching file against labels |
| `make test` / `make lint` | pytest on synthetic data / ruff |

## Layout

```
src/entity_resolution/     shared, tested library used by every notebook
├── config.py              paths, file schema, constants
├── data.py                TSV loaders, Parquet cache, fast ID filtering
├── split.py               fixed validation split (20% of Source 1, hashed)
├── metrics.py             macro F0.5, breakdown, blocking candidate report
├── submission.py          write and validate the two output files
└── tracking.py            experiment registry, timings
experiments/
├── _template/             documented notebook template
├── vNNN_<slug>/           one folder per experiment: notebook, metrics.json, artifacts/
└── experiments.csv        one row per version: change, local/public F0.5, commit
LEADERBOARD.md             every leaderboard upload, with budget
docs/                      challenge brief and guidelines
tests/                     pytest on synthetic TSVs
.githooks/                 identity, commit format, size and data guards
```

`dataset/`, `output/`, `models/` and `experiments/*/artifacts/` stay local.

## Status

Project foundation is in place: cached loading, the fixed validation split, the
leaderboard metric with a blocking report, submission writing and validation, the
experiment registry and notebook template. The research plan is written
(`docs/plan/`); model work starts with `v001`, following the plan and the project rules.

## Reproducing the submission

To be written with the pipeline: exact commands from raw `dataset/` to
`output/matching_results.tsv` and `output/candidate_pairs.tsv`, then `make validate`.
