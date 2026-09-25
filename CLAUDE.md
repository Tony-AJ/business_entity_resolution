# CLAUDE.md

Amazon ML Challenge 2026: **Business Entity Resolution**. For each Source 1 business
record, find every Source 2 / Source 3 record that describes the same business. Scored by
macro F0.5 per Source 1 entity (precision-heavy). Full brief:
[docs/PROBLEM_STATEMENT.md](docs/PROBLEM_STATEMENT.md).

The rules below bind every AI assistant working in this repo.

## 1. File access

- Read, Write and Edit files inside this repo directly, in every permission mode, auto or
  not. `.claude/settings.json` pre-approves `Read(/**)` and `Edit(/**)` (Edit also covers
  Write), so don't pause to ask before reading or changing project files.
- Files outside the repo, deletions and other shell side effects follow the normal
  permission flow.

## 2. Git identity: rajaguru2004 only

- Every commit is authored **and** committed as `rajaguru2004 <rajaguru20042@gmail.com>`.
- Never commit under a Claude / Anthropic identity. Never add AI attribution anywhere in
  git or GitHub: no `Co-Authored-By: Claude ...`, no `Claude-Session:` trailer, no
  "Generated with Claude Code" line in commits, PR titles or bodies, tags or releases.
- Never override identity: no `--author`, no `-c user.name=` / `-c user.email=`, no
  `GIT_AUTHOR_*` / `GIT_COMMITTER_*` variables.
- Before committing, check `git config user.email`. If it is not `rajaguru20042@gmail.com`,
  stop and tell the user.
- Enforced by `.githooks/` and by `attribution` in `.claude/settings.json`. Never skip hooks
  (`--no-verify`, `-n`); a rejected commit gets fixed, not forced.

One-time setup per clone:

```bash
git config core.hooksPath .githooks                      # enable versioned hooks
git config user.name rajaguru2004
git config user.email rajaguru20042@gmail.com
git config guard.requiredEmail rajaguru20042@gmail.com   # hook rejects any other identity
```

## 3. Version control

- **Branch per task**: `<type>/<topic>` from an up-to-date `main` (e.g.
  `feat/blocking-tfidf`). Never commit directly to `main`; it changes only by merging
  reviewed branches or PRs, with a merge commit (not squash) so every step survives.
- **Small commits, one logical change each**: a module with its tests, a config change, a
  doc update. Never pile a whole feature or unrelated files into one commit. Aim for under
  ~200 changed lines; the pre-commit hook rejects more than 400. `ALLOW_LARGE_COMMIT=1` is
  only for vendored or generated files, and only after the user agrees.
- **Record every change**: commit each finished step as you go, so the tree shows how the
  work evolved, and end every task with a clean `git status`. History is append-only: no
  amend, rebase or squash of pushed commits, no force-push.
- **Conventional Commits**: `type(scope): imperative summary`, at most 72 characters, type
  one of `feat fix docs style refactor perf test build ci chore revert`. The body says
  *why* when it isn't obvious. The commit-msg hook enforces the format.
- **Stage explicitly**: `git add <paths>`, then review `git diff --cached`. Never
  `git add -A` or `git add .`.
- **Never commit** `dataset/`, `output/`, `models/`, `.env`, archives or model binaries.
  `.gitignore` and the pre-commit hook both guard this.
- Push and open PRs only when the user asks.

## 4. Challenge rules (hard constraints)

- **No external data**: no web lookups, entity-resolution or geocoding APIs, business
  registries or internet augmentation. Only the provided training data. Breaking this
  means disqualification.
- **Model licence and size**: the final model is MIT or Apache-2.0 licensed with at most 8B
  parameters. Check the licence before adding any pretrained model or embedding.
- **Country is an open set**: test adds France, which is absent from train. Never
  hard-code, filter or one-hot on {US, India}; every test entity gets an output row.
- **TSV everywhere**: `pd.read_csv(path, sep="\t", dtype=str, na_filter=False)`.
  Addresses and ID lists contain commas, an empty list means no match, and "NA" can be a
  real business name.
- **Output**: `output/matching_results.tsv` and `output/candidate_pairs.tsv`, one row per
  test Source 1 entity, only `S2-`/`S3-` IDs that exist in test, no duplicates, matches a
  subset of candidates. `candidate_pairs.tsv` is exactly the set the matcher scores.
- **Metric**: macro F0.5 per Source 1 entity, singletons included. An empty prediction for a
  true singleton scores 1.0, and a false merge costs more than a missed link, so tune for
  precision.
- **Reproducible**: all code under `src/`, pinned `requirements.txt`, fixed seeds, and a
  `README.md` that regenerates both output files from the raw data.
