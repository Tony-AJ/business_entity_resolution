# 14 — Git and integration workflow

Audience: everyone. Companion to 13 (what a version records) and 15 (uploads). Enforced by
`.githooks/pre-commit` and `.githooks/commit-msg`, enabled per clone with `make hooks`.

## 1. Branch model

```
                  PR #3 (M2)          PR #5 (M3)        PR #6 (M4)         exp PR (M1)
main   ●──────────────M───────────────────M─────────────────M──────────────────M──────▶   M = merge commit
        \            /  \                /                 /                  /
feat/normalize  ●─●─●    \              /                 /                  /
feat/features             ●────●────●──┘                 /                  /
feat/model                     ●────────●────●──────────┘                  /
exp/v103_addr_features                                         ●────●─────┘
```

- `main` changes only by merging a reviewed PR with a **merge commit** (GitHub "Create a
  merge commit"; never squash, never rebase-merge). The hooks let `Merge …` subjects through.
- History is append-only: no `--amend`, `rebase` or squash of anything pushed, no
  `push --force`, no branch deletion before the challenge ends. Mistakes are fixed by a new
  commit or a revert (§9).
- One branch per task, `<type>/<topic>`: `feat/normalize`, `feat/blocking` (M2),
  `feat/features` (M3), `feat/model` (M4), `feat/decision` (M5), plus `docs/…`, `fix/…`,
  `exp/…` (optional, for M1's integrated runs). Branch from an up-to-date `main`; long-lived
  branches merge `origin/main` into themselves (§8), never the other way round.
- Nobody commits on `main` directly; M1 turns on GitHub branch protection (PR required, no
  force-push, no deletion) if the repository plan allows it. The single exception is §9.

## 2. One-time setup per clone

```bash
git clone git@github.com:Tony-AJ/business_entity_resolution.git && cd business_entity_resolution
make setup                                    # .venv: pinned requirements.txt, editable package, pytest/ruff/jupytext
make hooks                                    # git config core.hooksPath .githooks
git config user.name  "<your GitHub login>"
git config user.email "<your email>"
git config guard.requiredEmail "<your email>"  # pre-commit rejects any other author/committer in this clone
unzip ~/Downloads/student_resource.zip -d dataset/   # dataset/student_resource/dataset/{train,test}; gitignored
make cache                                    # ~1 min: every TSV to Parquet under dataset/.cache
make lint test                                # green before the first commit
```

Verify with `git config --get core.hooksPath` (`.githooks`) and `git var GIT_AUTHOR_IDENT`.
AI assistants work under your identity: the hooks reject `claude`/`anthropic` in author,
committer or trailers; your `CLAUDE.md` and `.claude/settings.local.json` are gitignored.

## 3. Commit rules, exactly as the hooks check them

`pre-commit` (on every `git commit`; on a merge commit only checks 1–4 run):

| # | Check | Rule | Escape |
|---|---|---|---|
| 1 | AI identity | author or committer containing `claude` or `anthropic` (case-insensitive) is rejected | none |
| 2 | Required identity | when `guard.requiredEmail` is set, author and committer must be `<that email>` | none |
| 3 | Data files | staged paths under `dataset/`, `output/`, `models/`, any `.env`, any `.tsv .parquet .joblib .pkl .zip` are rejected, except under `tests/fixtures/` | none |
| 4 | File size | any staged blob over `MAX_FILE_KB` (2048 KB) is rejected | trim notebook outputs |
| 5 | Commit size | added + deleted lines over all staged files except `*.ipynb` must be ≤ `MAX_COMMIT_LINES` (400) | `ALLOW_LARGE_COMMIT=1 git commit …`, only for generated or vendored files, only after M1 agrees |

`commit-msg`:

| # | Check | Rule |
|---|---|---|
| 6 | AI attribution | any line `Co-Authored-By: … claude/anthropic`, `Claude-Session:`, `Generated with [Claude …` or `noreply@anthropic.com` is rejected |
| 7 | Format | subject matches `^(feat\|fix\|docs\|style\|refactor\|perf\|test\|build\|ci\|chore\|revert\|exp)(\(scope\))?!?: <text>`, scope `[a-z0-9._/-]+`; `Merge`, `Revert`, `fixup!`, `squash!`, `amend!` subjects are exempt |
| 8 | Length | subject ≤ 72 characters |

Never bypass with `--no-verify`/`-n`; a rejected commit is fixed, not forced. GitHub does
not run these hooks, so M1 also rejects a PR whose commits break them. Scope = module or
area: `feat(blocking): add P3 name+address word pass`, `test(normalize): cover Devanagari
transliteration`, `exp(v012): P3 word pass, cand recall 0.9520`, `build(deps): pin rapidfuzz
3.14.6`. The body says why when the subject does not.

### 3.1 Splitting work into ≤ 400-line commits

A module lands as three or more commits, in this order, each green under `make lint test`:

1. `feat(<module>): …` the module, or one coherent part of it (one blocking pass, one feature
   group, one decision rule), with docstrings, type hints and the contract asserts of 02 §3.
   A 600-line module is two commits: contract, constants and the first function; the rest.
2. `test(<module>): …` its tests on the synthetic fixtures under `tests/`.
3. `exp(vNNN): …` the notebook, `metrics.json` and csv row (13 §4). Notebooks do not count
   towards the 400 lines, only towards the 2 MB byte limit.

Stage explicitly: `git add <paths>`, then `git diff --cached --stat`. Never `git add -A` or
`git add .`, which is how a `.tsv` or an `artifacts/` file gets staged. Commit each finished
step as you go, and end every task with a clean `git status`.

## 4. PR checklist (pasted into the PR description by the author)

```
- [ ] branch <type>/<topic>; origin/main merged in within the last 3 hours; no conflicts
- [ ] `make lint test` output pasted: all green, with the test count
- [ ] notebook re-executed headless (`make nb NB=experiments/vNNN_<slug>/vNNN_<slug>.ipynb`), outputs committed, < 2 MB
- [ ] experiments.csv row present; commit column without -dirty; owner, parent, decision filled (13 §3)
- [ ] metrics.json with the standard keys (13 §2.2); parent version named in the header and the csv
- [ ] pasted: val breakdown (f_beta, singletons, matched, pair P/R); blocking report (cand_recall, entity_recall,
      ceiling, cands mean/p95, seconds); slice report vs parent (country, singleton/matched, 1-match, non_latin, ambiguous)
- [ ] error categories affected: tag_errors counts before → after, one example per category that changed
- [ ] contract unchanged (NORM_COLUMNS, PAIR_COLUMNS, SCORED_COLUMNS, MATCH_COLUMNS, signatures of 02 §5), or 02 and every
      dependent document and module are updated in this PR and their owners are reviewers
- [ ] no data files, artifacts, outputs, other people's notebooks, README.md or LEADERBOARD.md edits (M1 only)
- [ ] any new dependency pinned in requirements.txt in the commit that first imports it; licence MIT/BSD/Apache/ISC
- [ ] run time and peak RSS of the changed stage on the val fold, extrapolated to test India (02 §7)
```

Reviewer: M1 for every PR into `main`, plus the owner of any module the PR touches outside
the author's own (§6). Review means reading the diff and the pasted numbers, not re-running:
the numbers come from a non-dirty commit, so they can be reproduced if doubted.

## 5. Merge gate (operated by M1)

1. **Frozen parent.** The PR names its parent version; M1 compares the pasted numbers with
   that parent's `metrics.json` as committed on `main`, not with what anyone remembers. The
   parent must be a KEEP (or v001).
2. **KEEP rule of 13 §3**: `f_beta` > parent + 0.002, no slice regresses by more than 0.01,
   `harder_f_beta` not lower. One improved metric does not merge a PR: a recall gain with a
   0.02 drop on the singleton slice is INVESTIGATE, not KEEP.
3. **Blocking-only PRs** (no matcher change) are judged on `cand_recall`, `ceiling_f_beta`
   and `cands_mean`/`cands_p95` per S1: merge if ceiling F0.5 rises with `cands_mean` ≤ 40
   (06 §6), or if recall holds and candidates or seconds fall by at least 20 %.
4. **Normalisation-only PRs** are judged through blocking (per-pass recall, exact groups
   ≤ 50) and the agreement rates of 05 §1 recomputed on val.
5. **Integrated re-run.** After merging, M1 runs the full pipeline on `main` under a fresh
   v1xx number (on an `exp/v10x_<slug>` branch, PR like any other). Only a v1xx number with
   its own row and `metrics.json` can be uploaded (15 §2). If the integrated run is not KEEP
   against the previous v1xx, the merge is reverted (§9) and the author gets the slice report.
6. Merge with a merge commit, then post in chat: PR merged, new v1xx and its `f_beta`, and
   "re-merge origin/main now".

## 6. Five people, no conflicts

| Rule | Detail |
|---|---|
| One module, one owner | `normalize.py`, `blocking.py` M2; `features.py`, `trainset.py` M3; `model.py` M4; `decision.py`, `errors.py` M5; `pipeline.py`, `evaluate.py`, `submission.py`, `tracking.py`, `config.py` M1 (02 §3). Touching another owner's module needs that owner as reviewer; a one-line fix is still a PR |
| Tests follow the module | `tests/test_<module>.py` belongs to the module owner; `tests/conftest.py` fixtures are M1's |
| Reserved version numbers | 13 §1: nobody creates a folder outside their range, so folder names never collide |
| `experiments/experiments.csv` | `.gitattributes` (M1, day 1): `experiments/experiments.csv merge=union`. Rows from both sides survive a merge; afterwards re-sort and check for duplicate versions with the two commands in §8 (`make csv-sort` once M1 adds it) |
| Notebooks | a notebook is edited by its owner only; two people never edit the same `.ipynb`. A conflicted notebook is never merged by hand: take one side (`git checkout --ours <nb>`) and re-execute |
| `README.md`, `LEADERBOARD.md`, `docs/plan/02_*` | M1 only; others propose text in the PR description or chat |
| `requirements.txt` | append-only, one pinned line per new package, in the commit that first imports it |
| `config.py` | M1 only; new constants are requested in chat and added within the hour |

## 7. Integration sequence

**Day 1, Friday 25 Sep.** 15:00–17:00: M1 builds the walking skeleton on `feat/skeleton`:
`pipeline.py` with trivial normalisation (`basic_norm`), exact `name_norm` block, the
`heuristic` matcher backend, a fixed `DecisionRule`, `submission.write_pairs`, and `run_test`
producing `output/*.tsv` that pass `make validate`. Merged and logged as **v001** by ~17:00,
uploaded ~18:00 as the calibration point (15 §3). Meanwhile M2–M5 set up their clone (§2),
read 01, 02 and their module document, and start their module on their branch against the
contract columns, with tests on synthetic fixtures. 17:00–21:00: first PRs, smallest useful
slice first: M2 normalisation B1–B2 then blocking A2 (P2); M3 features C1 with
`trainset.label_pairs`; M4 `Matcher` with the LightGBM backend on the features that exist;
M5 `decide`/`tune` E1–E2 with `evaluate.slice_report` and `errors.tag_errors`. ~21:00: M1
merges what is KEEP and runs **v101** (V1), uploaded ~21:30 if `f_beta` ≥ v001 + 0.05; then
**v102** = v101 with a higher `tau_abs` (precision-max probe), uploaded ~23:00.

**Day 2, Saturday 26 Sep.** PR windows close at 10:30, 13:30, 16:30, 19:30 and 22:00; after
each, M1 merges and runs an integrated v10x whose numbers decide the 11:00, 14:00, 17:00,
20:00 and 22:30 upload slots (15 §3). Everyone re-merges `origin/main` after each
integration post (§8). Expected order: A3 blocking, C3 address features, D3 LightGBM tuned,
E2–E4 decision, HN hard negatives, then C4/C5, A5, E5.

**Day 3, Sunday 27 Sep.** Last feature and decision PR windows at 10:30 and 14:30.
**Freeze at 18:00 IST**: `main` accepts only `fix:` commits approved by M1 (validator
failures, France-slice anomalies, packaging, README). No new experiments after 18:00 except
the final `run_test`; everyone else works through 16.

## 8. Conflict resolution recipe (before every PR and after every integration post)

```bash
git fetch origin
git merge origin/main                    # into your branch; never rebase a pushed branch
# src/ conflicts: resolve by hand, keeping the contract columns
# experiments.csv: the union driver keeps both sides; re-sort, then no version may appear twice
.venv/bin/python -c "from entity_resolution import tracking as t, config as C; t._write_rows(C.EXPERIMENTS_CSV, t._read_rows(C.EXPERIMENTS_CSV))"
cut -d, -f1 experiments/experiments.csv | sort | uniq -d       # must print nothing
# a notebook: git checkout --ours <nb>   (your version), then re-execute it
make lint test                           # green again
make nb NB=experiments/vNNN_<slug>/vNNN_<slug>.ipynb   # only if src/ changed under you; then re-check the csv commit column
git add <resolved paths> && git commit   # the default "Merge remote-tracking branch 'origin/main' …" subject passes the hooks
git push
```

If `src/` changed, the numbers in your PR are stale: re-execute, re-paste, and name the new
parent if M1 integrated a v1xx in between.

## 9. When a merge breaks `main`

Symptoms: `make lint test` red on `main`, the integrated v1xx run fails, or the v1xx run is
DROP/INVESTIGATE against the previous v1xx.

```bash
git switch main && git pull --ff-only
git revert -m 1 <merge-commit-sha>       # keeps history; parent 1 is main's side of the merge
make lint test
git push origin main                     # the only commit made on main without a PR; M1 only; announced in chat
```

Never force-push, never delete the branch, never rewrite the merge. The author fixes forward
on the same branch, re-runs the notebook against the restored parent and opens a new PR; the
reverted commits stay visible and the version folders stay. `Revert "…"` subjects pass the
hooks; a revert over 400 lines is a merge-like operation and may use `ALLOW_LARGE_COMMIT=1`.
