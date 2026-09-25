# Project rules: Business Entity Resolution (Amazon ML Challenge 2026)

Shared team rules, committed and loaded automatically by Claude Code for everyone.
Personal preferences go in your own gitignored `CLAUDE.md`. Challenge rules:
`docs/PROBLEM_STATEMENT.md` and `docs/GUIDELINES.md`.

## 1. Code is written as documented notebooks

- All experiment, training and inference code is written as Jupyter notebooks, one per
  experiment: `experiments/vNNN_<slug>/vNNN_<slug>.ipynb`, created with
  `make experiment NAME=<slug>` from `experiments/_template/experiment.ipynb`.
- Every code cell is preceded by a markdown cell explaining **what** it does and **why**.
  Every function has a docstring and non-obvious lines get a comment: the organisers
  require "source code ... with proper comments describing the functions".
- Notebooks follow the template: hypothesis, setup, data, method (one subsection per
  stage), evaluation, error analysis, log, conclusion, and test inference for
  shortlisted versions.
- Shared code lives in `src/entity_resolution/` (loading, split, metric, submission I/O,
  experiment tracking) so every experiment is measured the same way. Move notebook code
  there, with docstrings and tests, as soon as a second notebook needs it.
- Commit notebooks with their outputs: they are the record of the run. Every file stays
  under 2 MB (the pre-commit hook enforces it), so trim heavy outputs.

## 2. Model version history is mandatory

- One experiment = one folder `experiments/vNNN_<slug>/`, numbered in order. Versions are
  never renumbered, reused or deleted: discarded ideas stay as history.
- Every experiment ends with `tracking.log_result(...)`, which writes `metrics.json` and
  the version's row in `experiments/experiments.csv` (version, date, plan group, change,
  local F0.5, candidate recall, public F0.5, commit, notes).
- Commit `src/` changes **before** running, so the logged commit reproduces the score; a
  `-dirty` commit in the csv means it will not. Then commit the notebook, `metrics.json`
  and the csv row together as `exp(vNNN): <change>, local F0.5 0.xxxx`.
- Models, candidate sets and predictions go to the version's `artifacts/` (gitignored);
  the notebook must be able to regenerate them.

## 3. Local experiments first, leaderboard last

- Upload budget: 5 per day for 3 days, 15 in total. Never use the leaderboard to explore.
- Flow: many local experiments (10–50) → shortlist 2–5 by local F0.5 → upload.
- An upload needs a logged local F0.5 on the fixed split, committed code and notebook,
  and `make validate` passing.
- Log every upload in `LEADERBOARD.md` and record its score with
  `make public V=vNNN SCORE=0.xxxx`.
- Trust local F0.5 over the public score: public is a subset of test, and the final
  ranking uses the private part.

## 4. Optimise the real metric

- The primary objective is **macro F0.5 per Source 1 entity, singletons included**, on the
  fixed validation fold: `split.load_fold("val")` scored with `metrics.breakdown`.
- Accuracy, ROC-AUC, F1 and micro-F1 are diagnostics, never the decision criterion.
- Blocking experiments report `metrics.candidate_report` (pair recall, F0.5 ceiling,
  candidates per entity) and run time (`tracking.timed`).
- Fit only on the `train` fold; the `val` fold is for scoring. Never change
  `VAL_FRACTION` or `SPLIT_SEED`: a new split makes every earlier local score incomparable.

## 5. Experiment plan, in priority order

Tag every experiment with its plan ID in `group`.

- **A. Blocking** (measure candidate recall, candidate count, run time): A1 exact
  normalised name, A2 name TF-IDF, A3 address TF-IDF, A4 character n-gram, A5 multi-pass
  blocking.
- **B. Normalisation**: B1 basic normalisation, B2 legal suffix normalisation, B3
  abbreviation normalisation, B4 token canonicalisation, B5 address component extraction.
- **C. Features**: C1 string similarity, C2 TF-IDF, C3 address components, C4 numeric
  token matching, C5 country features.
- **D. Models**: D1 logistic regression, D2 random forest, D3 LightGBM, D4 XGBoost. Skip
  the rest once LightGBM clearly dominates.
- **E. Decision layer** (critical for F0.5): E1 global threshold, E2 threshold
  optimisation, E3 score gap, E4 singleton detection, E5 multi-match selection.

## 6. Hard constraints

- No external data, APIs, geocoding or lookups: only the provided data. Breaking this
  means disqualification.
- The final model is MIT or Apache-2.0 licensed with at most 8B parameters. Check the
  licence of any pretrained model before using it.
- Country is an open set: test adds France, which train lacks. Never hard-code, filter or
  one-hot on {US, India}; every test entity gets an output row.
- Scale: ~12M records per split on 15 GB machines. Load through `entity_resolution.data`
  (Parquet cache, `columns=`), filter IDs with `data.isin`, avoid Python loops over
  millions of rows, and develop on samples before confirming on the full fold.
- Output files are written only by `submission.write_submission` and checked with
  `make validate`.

## 7. Git

- One branch per task (`<type>/<topic>`), small Conventional Commits (at most 400 changed
  lines, notebooks excluded), `exp(vNNN): ...` for experiment runs, and merge commits
  rather than squashes. Enable the hooks once per clone with `make hooks`.
- Commit under your own git identity. The hooks reject AI identities and AI co-author
  trailers; `git config guard.requiredEmail <your email>` also makes them reject any
  other identity in your clone.
- Never commit data, outputs, artifacts or any file over 2 MB.

## 8. Deliverables

- Leaderboard uploads: `output/matching_results.tsv`.
- Final zip: `output/` with both TSVs; `code/business_entity_resolution/` (this repo, with
  the winning version's notebooks copied under `src/`, since all source must sit there),
  `README.md` and `requirements.txt`; and the filled `Documentation_template.md` (1–2
  pages: approach, blocking, model and features, results, conclusion).
