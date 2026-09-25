# 16 — Final submission checklist

Audience: M1 runs it on day 3 (Sunday 27 Sep); M2–M5 own the compliance items of §3 for
their modules. The challenge ends 27 Sep 23:59 IST; the zip is finished by 23:00.

## 1. Package

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv            byte-identical to the last leaderboard upload (15 §7)
│   └── candidate_pairs.tsv             from the same run_test: the pairs frame the model scored
├── code/business_entity_resolution/    this repository at tag final-submission, tracked files only
│   ├── src/                            entity_resolution/ package + notebooks/ (final version's notebook, metrics.json)
│   ├── README.md                       with "Reproducing the submission" filled (§2)
│   ├── requirements.txt                pinned; every third-party import of src/ present
│   ├── experiments/                    every version folder (notebook, metrics.json) and experiments.csv: version history
│   ├── LEADERBOARD.md, docs/, tests/, Makefile, pyproject.toml, .githooks/
└── Documentation_template.md           filled methodology write-up (§4)
```

The rules require all source under `src/`, so the final version's notebook and
`metrics.json` are copied to `src/notebooks/` and committed (they stay in `experiments/` too).
`git archive` exports tracked files only, so `dataset/`, `output/`, `artifacts/`, `.venv`,
`.git`, caches and the filled template's drafts are excluded by construction.

```bash
# 0. on main at the final commit, clean tree
git switch main && git pull --ff-only && test -z "$(git status --porcelain)" && echo clean
# 1. outputs of the final version, validated twice (16 §5 timeline: after the final run_test)
make validate
.venv/bin/python -m entity_resolution.submission --output-dir output --check-ids
.venv/bin/python dataset/student_resource/utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/student_resource/dataset/test --check-ids
# 2. final notebook under src/ (rules: all source under src/)
mkdir -p src/notebooks && cp experiments/vNNN_<slug>/vNNN_<slug>.ipynb experiments/vNNN_<slug>/metrics.json src/notebooks/
git add src/notebooks && git commit -m "chore(final): copy vNNN notebook under src for the package"
# 3. requirements: every third-party import pinned, pins equal to the venv that produced the outputs
.venv/bin/python - <<'EOF'
import pathlib, re, subprocess, sys
mods = set()
for p in pathlib.Path("src").rglob("*.py"):
    mods |= set(re.findall(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", p.read_text(), re.M))
for p in pathlib.Path("src/notebooks").glob("*.ipynb"):
    mods |= set(re.findall(r'"\s*(?:from|import)\s+([A-Za-z_]\w*)', p.read_text()))
third = sorted(m for m in mods if m not in sys.stdlib_module_names and m != "entity_resolution")
norm = lambda s: s.lower().replace("_", "-")
alias = {"sklearn": "scikit-learn"}
pins = {norm(l.split("==")[0]): l.strip() for l in open("requirements.txt") if "==" in l}
frozen = subprocess.run([".venv/bin/pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
frozen = {norm(l.split("==")[0]): l.strip() for l in frozen if "==" in l}
print("third-party imports:", third)
print("UNPINNED:", [m for m in third if norm(alias.get(m, m)) not in pins] or "none")
print("PIN != VENV:", [v for k, v in pins.items() if frozen.get(k) != v] or "none")
EOF
# 4. stage and zip
TEAM=<team_name>; STAGE=build/${TEAM}_submission
rm -rf build && mkdir -p "$STAGE/output" "$STAGE/code/business_entity_resolution"
git archive HEAD | tar -x -C "$STAGE/code/business_entity_resolution"
cp output/matching_results.tsv output/candidate_pairs.tsv "$STAGE/output/"
cp docs/Documentation_template.md "$STAGE/Documentation_template.md"
(cd build && zip -r -q "${TEAM}_submission.zip" "${TEAM}_submission")
unzip -l "build/${TEAM}_submission.zip" | tee "build/${TEAM}_submission.contents.txt" | tail -1; ls -l build/*.zip
# 5. verify: unpack into a temp dir, validate from inside it, then a fresh-venv smoke test of the code copy
REPO=$(pwd); T=$(mktemp -d) && unzip -q "build/${TEAM}_submission.zip" -d "$T" && cd "$T/${TEAM}_submission"
python3 "$REPO/dataset/student_resource/utils/validate_submission.py" --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir "$REPO/dataset/student_resource/dataset/test" --check-ids
cmp output/matching_results.tsv "$REPO/output/matching_results.tsv" && echo identical
cd code/business_entity_resolution && python3.12 -m venv .venv && .venv/bin/pip install -q -r requirements.txt -e ".[dev]" \
    && .venv/bin/pytest -q && .venv/bin/python -c "import entity_resolution.pipeline; print('import ok')"
```

`build/` and `*.zip` are gitignored. Expected size 20–150 MB (notebooks with outputs). If
the portal or form caps the size below that, drop the notebooks of versions that were never
uploaded (keep `experiments.csv`, every `metrics.json` and every uploaded version) and say so
in the README; never drop `output/` or `src/`.

## 2. README: "Reproducing the submission"

The section (currently "To be written") is filled by M1 on day 3 between 20:30 and 22:00
and must contain, with real numbers:

```markdown
## Reproducing the submission
1. Python 3.12: `make setup` (pinned requirements.txt, editable package, nbconvert).
2. Unzip the organisers' student_resource.zip into dataset/ → dataset/student_resource/dataset/{train,test}.
3. `make cache` (~1 min: every TSV to Parquet).
4. Final version vNNN_<slug> (commit <sha>, tag final-submission):
   `make nb NB=experiments/vNNN_<slug>/vNNN_<slug>.ipynb` (identical copy in src/notebooks/), which fits on the
   train fold (inner fit/tune split), scores the val fold (macro F0.5 0.xxxx) and runs test inference; or headless:
   `python -m entity_resolution.pipeline --config experiments/vNNN_<slug>/artifacts/config.json --fit --run-test`.
   Wall time ≈ N h on 12 threads, peak RAM ≈ N GB (15 GB machine).
5. Outputs: output/matching_results.tsv and output/candidate_pairs.tsv; then `make validate`.
Seeds: 42 (config.SEED, validation split, LightGBM), 4242 (inner split), 7 (S1 sampling), 99 (harder fold);
num_threads fixed at 12; expected val F0.5 0.xxxx (± 0.0005 across machines).
```

Data → `make cache` → notebook or pipeline CLI → `output/` → `make validate`: nothing else,
and nothing read from outside `dataset/`. If the pipeline CLI does not exist by day 3, the
notebook is the only entry point and the README says so.

## 3. Compliance

| Item | Requirement | Owner |
|---|---|---|
| No external data or lookups | the grep below prints nothing except documentation URLs inside docstrings; every token map in `normalize.py` is hand-typed (05 §3) or fitted by `fit_region_map`/`fit_token_map` on train-fold pairs, and the notebook shows that fit; nothing is read from outside `dataset/` | M2, M1 |
| Model licence and size | LightGBM (MIT); `booster.num_trees()` (best iteration) and `num_leaves` recorded in `metrics.json` and the documentation, e.g. 1,400 trees × 63 leaves ≈ 88k leaves (thresholds + leaf values ≈ 0.2M parameters, far below 8B). TF-IDF vectorisers are unsupervised blocking components; note their vocabulary sizes. Any embedding model (none in the MUST path, 01 §9) is named with licence (MIT or Apache-2.0 only), parameter count ≤ 8B and weight hash | M4 |
| Library licences | numpy, pandas, scikit-learn, scipy, psutil (BSD); pyarrow, sparse_dot_topn (Apache-2.0); rapidfuzz, LightGBM (MIT); anyascii (ISC). GPL packages (`unidecode`, `python-Levenshtein`) absent from `requirements.txt` (grep below empty) | M1 |
| Fixed seeds | 42 / 4242 / 7 / 99 as in §2; `deterministic=True` for LightGBM; two runs of the final notebook give the same val F0.5 | M4, M1 |
| Pinned versions | the script of §1 step 3 prints `UNPINNED: none` and `PIN != VENV: none` | M1 |
| Function docstrings | the `ast` check below prints nothing (public functions and classes); every code cell of every notebook has a markdown cell before it | each owner |
| Notebook outputs present | the final notebook was executed with `make nb` and never cleared: the `jq` count below is 0 | M1 |
| File sizes | the `find` below prints nothing outside `output/` and `artifacts/` | M1 |
| Country open set | the country grep below shows no filter or one-hot on a country name; France rows present in `output/` with a plausible match rate (15 §2) | M5 |

```bash
grep -rnEi 'https?://|requests|urllib|httpx|socket|geocod|nominatim|libpostal|api[_-]?key|boto3|wget|curl' \
    src/ tests/ experiments/ --include='*.py' --include='*.ipynb'
grep -iE 'unidecode|levenshtein' requirements.txt                       # must be empty
.venv/bin/python - <<'EOF'
import ast, pathlib
for p in pathlib.Path("src").rglob("*.py"):
    for n in ast.walk(ast.parse(p.read_text())):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_") \
                and not ast.get_docstring(n):
            print(p, n.lineno, n.name)                                  # every hit needs a docstring
EOF
jq '[.cells[] | select(.cell_type == "code" and (.outputs | length) == 0)] | length' src/notebooks/vNNN_<slug>.ipynb
find . -path ./.git -prune -o -path ./dataset -prune -o -path ./.venv -prune -o -type f -size +2M -print
grep -rnE '"(US|India|France)"' src/                                    # no filter or one-hot on a country
```

## 4. `Documentation_template.md` mapping

The filled copy lives at `docs/Documentation_template.md` (committed; the organisers' blank
original stays under `dataset/student_resource/`). Main body 1–2 pages; appendices unbounded.
Numbers come from `metrics.json` of the final version and `experiments.csv`, never retyped
from memory; each section is drafted by its owner and edited by M1 on day 3.

| Template section | Source | Drafted by |
|---|---|---|
| Header: team name, members, date | LEADERBOARD.md | M1 |
| 1. Executive Summary (2–3 sentences) | 00 (master plan) §1; final local and public F0.5 | M1 |
| 2.1 Problem Analysis | 01 §3 (measured noise table), §4 (open-set country), §5 (singletons), §8 (metric arithmetic) | M1 |
| 2.2 Solution Strategy; Approach Type: Blocking + Classifier; Core Innovation | 00 §4–§5 and 02 §1 (pipeline), 01 §12 (what follows) | M1 |
| 3. Candidate Generation: blocking keys, candidate pairs generated, how true matches were kept | 06 §2–§3 (passes), final `metrics.json`: `cand_recall`, `entity_recall`, `ceiling_f_beta`, `cands_mean`/`p95`, total candidate pairs on test (`wc -l`/sum of list lengths), per-pass recall table | M2 |
| 4. Matching Model: name / address / other features, model type, threshold selection | 07 §2 (registry by group, importance top 20), 08 (model selection: LightGBM params), 09 (hard-negative mining), 10 (`DecisionRule`, `tune` on the tune split, macro F0.5 grid) | M3, M4, M5 |
| 5. Results & Error Analysis: F0.5 (macro), common false positives, common false negatives | `experiments.csv` progression (uploaded versions, local vs public), `metrics.json` (`f_beta`, singletons, matched, `harder_f_beta`), 18 (error analysis framework: category counts from `errors`, two examples each) | M5, M1 |
| 6. Conclusion (2–3 sentences) | what moved the score (from the csv deltas), what did not, lessons | M1 |
| Appendix A. Code Artefacts | README §2 reproduction steps, module map 02 §3, entry points | M1 |
| Appendix B. Additional Results | full `experiments.csv` table, blocking per-pass recall, feature importance, tune grid, slice report, LEADERBOARD.md offsets | M1 |

## 5. Day-3 timeline (IST, Sunday 27 Sep)

| Time | Step |
|---|---|
| 10:30, 14:30 | last PR windows; 11:00, 15:00 upload slots (15 §3) |
| 18:00 | **freeze**: `main` takes only M1-approved `fix:` commits; 18:00 upload slot |
| 18:00–19:30 | final `run_test` of the chosen version from its merge commit (test candidates cached by blocking hash; scoring + decide ≈ 30–40 min) |
| 19:30–20:00 | `make validate`, both `--check-ids` runs, test-slice sanity (15 §2); decide the final version (15 §7) |
| 20:30 | safety re-upload of the final version; `make public`; LEADERBOARD.md entry |
| 20:30–22:00 | README §2, `docs/Documentation_template.md` §4, requirements check, notebook copy under `src/`, compliance table §3, commit, `git tag -a final-submission -m "vNNN local 0.xxxx public 0.xxxx" <merge sha>`, push when the user says so |
| 22:00 | last-resort upload slot, only if 20:30 failed; nothing after 22:00 |
| 22:00–23:00 | build the zip (§1), unpack into a temp dir, run the validator from inside it, fresh-venv smoke test, save the contents listing, record size and sha256 (`sha256sum build/*.zip`) in LEADERBOARD.md |
| 23:00 | hand in the zip through the organisers' channel; keep the confirmation screenshot; 23:59 hard end |

## 6. Final checklist

```
Outputs
- [ ] output/matching_results.tsv produced by the final version's run_test at commit <sha>; never hand-edited
- [ ] output/candidate_pairs.tsv from the same run: the pairs frame that score() consumed
- [ ] headers exactly `source1_entity_id<TAB>matched_entity_ids` and `source1_entity_id<TAB>candidate_entity_ids`
- [ ] exactly 1,732,544 data rows in each file, one per test S1 entity, no duplicate source1_entity_id
- [ ] our checker and the organisers' validator PASS, both with --check-ids
- [ ] matches ⊆ candidates for every entity (no "matches that are not candidates" warning)
- [ ] France rows present; match rate and cands/S1 comparable to US/India; numbers noted in LEADERBOARD.md
- [ ] matching_results.tsv byte-identical (cmp) to the last leaderboard upload
Code
- [ ] final notebook and metrics.json copied to src/notebooks/ and committed
- [ ] make lint test green at the tagged commit
- [ ] every third-party import pinned in requirements.txt; pins equal to pip freeze of the producing venv
- [ ] fresh venv from requirements.txt installs the package, imports entity_resolution.pipeline, passes pytest
- [ ] README "Reproducing the submission" filled: exact commands, run time, RAM, expected val F0.5
- [ ] every public function and class has a docstring; ruff clean
- [ ] final notebook has outputs in every code cell; every file in the repo < 2 MB
- [ ] no external data, API, geocoding or lookup: grep clean; token maps hand-typed or learned from train pairs
- [ ] model licence MIT (LightGBM); trees × leaves recorded; no model over 8B parameters; no GPL package
- [ ] seeds 42 / 4242 / 7 / 99 fixed and documented; two runs of the final notebook agree
- [ ] no dataset/, output/ (except the two TSVs), artifacts/, .venv, .git, .env or cache files inside the zip
History
- [ ] experiments.csv has a row for every version; the final row has a non-dirty commit, public_f05 and decision
- [ ] LEADERBOARD.md lists every upload with its public score and offset; budget table shows the slots used
- [ ] git tag final-submission on the main merge commit of the final version; pushed with the branch
- [ ] git status clean; no rebase, squash or force-push anywhere in the history (git log --merges shows every PR)
Document
- [ ] docs/Documentation_template.md: every section filled, team name, members and date at the top, 1–2 pages plus appendices
- [ ] every number in the document equals metrics.json / experiments.csv (local) or LEADERBOARD.md (public)
- [ ] Appendix A entry point equals the README reproduction commands
Package
- [ ] zip named <team_name>_submission.zip with the tree of §1, nothing at the top level except output/, code/, Documentation_template.md
- [ ] zip unpacked into a temp dir; validator PASS with --check-ids from inside it
- [ ] size and sha256 recorded; contents listing saved to build/<team_name>_submission.contents.txt and its line count noted
- [ ] zip handed in by 23:00 IST; confirmation screenshot kept by M1
```
