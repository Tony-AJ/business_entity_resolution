# Business Entity Resolution

Amazon ML Challenge 2026. Business records from three sources share no identifiers. For
every Source 1 entity, find all Source 2 / Source 3 records that describe the same
business. Scoring is macro F0.5 per Source 1 entity, so precision counts twice as much as
recall. Data, output format and rules: [docs/PROBLEM_STATEMENT.md](docs/PROBLEM_STATEMENT.md).

## Setup

Requires Python 3.12.

```bash
make setup    # .venv: pinned requirements.txt + editable package + pytest, ruff
make hooks    # enable the versioned git hooks in .githooks/
```

Put the organisers' files under `dataset/` (gitignored):

```
dataset/
├── train/   train_source{1,2,3}.tsv, train_ground_truth.tsv
└── test/    test_source{1,2,3}.tsv
```

## Commands

| Command | What it does |
|---|---|
| `make test` | unit tests on tiny synthetic TSVs; no dataset needed |
| `make lint` | ruff over `src/` and `tests/` |
| `make score PRED=<tsv> TRUTH=<tsv>` | macro F0.5 of a matching file against ground truth, split into singletons, matched entities and pair precision/recall |
| `make validate` | checks `output/*.tsv` against every submission rule for the test split |

## Layout

```
src/entity_resolution/
├── config.py       paths, file schema, metric constants
├── data.py         TSV loaders: string dtype, NA detection off, header and ID checks
├── metrics.py      macro F0.5, singletons included
└── submission.py   write and validate matching_results.tsv / candidate_pairs.tsv
tests/              pytest; synthetic fixtures in conftest.py
docs/               challenge brief
.githooks/          pre-commit and commit-msg hooks
```

`dataset/`, `output/` and `models/` stay local.

## Status

Done: data loading, the leaderboard metric, submission writing and validation.

Proposed next steps:

1. validation split held out from train, by Source 1 entity;
2. name and address normalisation (legal suffixes, abbreviations, transliteration);
3. blocking / candidate generation, which caps recall;
4. pairwise matching model with a threshold tuned for F0.5;
5. one command from `dataset/` to both output files, plus the methodology document.

## Version control

- One branch per task, named `<type>/<topic>` (e.g. `feat/blocking-tfidf`). Merge into
  `main` through a PR with a merge commit, not a squash, so every commit stays in history.
- Small commits, one logical change each. The pre-commit hook rejects commits over 400
  changed lines.
- Conventional Commits: `type(scope): summary`, at most 72 characters, type one of `feat fix
  docs style refactor perf test build ci chore revert`.
- Never commit data, outputs, model binaries or `.env`. `.gitignore` and the pre-commit hook
  both guard this.
- Commit under your own git identity. The hooks reject AI assistant identities and AI
  co-author trailers. For a strict check in your clone, run
  `git config guard.requiredEmail <your email>` and the hook rejects any other identity.
- AI assistant rules are personal: keep your own `CLAUDE.md` and
  `.claude/settings.local.json`. Both are gitignored.

## Reproducing the submission

To be written with the pipeline: exact commands from raw `dataset/` to
`output/matching_results.tsv` and `output/candidate_pairs.tsv`. Then check both files with
`make validate`, and with the organisers' validator from their `student_resource/` folder:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
