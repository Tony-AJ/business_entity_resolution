#!/usr/bin/env bash
# Build the final submission zip (docs/plan/16_FINAL_SUBMISSION_CHECKLIST.md §1):
#
#   build/<team>_submission.zip           (staged in build/<team>_submission/; no wrapping folder)
#   ├── output/                           matching_results.tsv, candidate_pairs.tsv (--outputs)
#   ├── code/business_entity_resolution/  `git archive HEAD`: tracked files only, so dataset/,
#   │   │                                 output/, artifacts/, .venv and .git stay out
#   │   └── src/notebooks/                the final version's notebook + metrics.json
#   └── Documentation_template.md         the filled write-up (--doc, default
#                                         docs/Documentation_template.md)
#
# Steps: a read-only preflight, both validators, staging under build/ (gitignored), a check
# of the staged code tree, the zip, then sizes, sha256 and an `unzip -l` listing. The working
# tree is never modified. --dry-run runs only the preflight and prints the other steps.
#
# Written for bash 3.2+ with GNU or BSD tools (no mapfile, no GNU-only flags).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/package_submission.sh --team NAME --version vNNN_slug [options]

Builds build/NAME_submission.zip from the two output TSVs, the repository at HEAD (plus the
version's notebook and metrics.json under src/notebooks/) and the methodology write-up.

  --team NAME          team name; the zip is NAME_submission.zip (letters, digits, . _ -)
  --version vNNN_slug  committed experiment folder of the final version (v105_dense_blocking)
  --outputs DIR        folder holding both TSVs (default: output; e.g. submissions/v105)
  --doc PATH           filled write-up, shipped as Documentation_template.md
                       (default: docs/Documentation_template.md)
  --check-ids          pass --check-ids to both validators: every listed ID must exist in
                       the test split (a few GB of RAM)
  --dry-run            read-only preflight, then print each step without running it
  -h, --help           show this help

Paths are relative to the repository root. The code in the zip is HEAD, so the script
refuses to build while tracked files have uncommitted changes.
EOF
}

# ---------------------------------------------------------------- helpers ----
die()    { printf 'package_submission: %s\n' "$*" >&2; exit 1; }
step()   { printf '\n==> %s\n' "$*"; }
ok()     { printf '  ok    %s\n' "$*"; }
note()   { printf '  note  %s\n' "$*"; }
bad()    { printf '  FAIL  %s\n' "$*"; PROBLEMS=$((PROBLEMS + 1)); }  # tallied after preflight
indent() { printf '%s\n' "$1" | sed 's/^/          /'; }
q()      { printf '%q' "$1"; }  # shell-quote one word for a `run` command line

# Print a command line, then run it in this shell unless --dry-run
run() {
  printf '  $ %s\n' "$1"
  (( DRY_RUN )) || eval "$1"
}

# Byte count as B / KiB / MiB / GiB
human() {
  awk -v b="$1" 'BEGIN { split("B KiB MiB GiB", u, " "); i = 1
    while (b >= 1024 && i < 4) { b /= 1024; i++ }
    printf(i == 1 ? "%d %s\n" : "%.1f %s\n", b, u[i]) }'
}

bytes() { wc -c < "$1" | tr -d ' '; }  # file size (BSD wc pads it with spaces)

if ! command -v sha256sum >/dev/null; then  # macOS ships shasum, not GNU sha256sum
  sha256sum() { shasum -a 256 "$@"; }
fi

# -------------------------------------------------------------- arguments ----
TEAM='' VERSION='' OUTPUTS=output DOC=docs/Documentation_template.md CHECK_IDS='' DRY_RUN=0
while (( $# )); do
  case $1 in
    --team | --version | --outputs | --doc)
      (( $# >= 2 )) || die "$1 needs a value (see --help)"
      case $1 in
        --team) TEAM=$2 ;;
        --version) VERSION=$2 ;;
        --outputs) OUTPUTS=$2 ;;
        --doc) DOC=$2 ;;
      esac
      shift 2 ;;
    --check-ids) CHECK_IDS=' --check-ids'; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) die "unknown argument '$1' (see --help)" ;;
  esac
done

VERSION=${VERSION#experiments/}; VERSION=${VERSION%/}  # accept a tab-completed folder path
OUTPUTS=${OUTPUTS%/}
[[ -n $TEAM && -n $VERSION ]] || { usage >&2; die "--team and --version are required"; }
team_re='^[A-Za-z0-9][A-Za-z0-9._-]*$'
version_re='^v[0-9]{3}_[a-z0-9]+(_[a-z0-9]+)*$'  # as tracking.new names them: vNNN_<slug>
[[ $TEAM =~ $team_re ]] \
  || die "--team '$TEAM': use letters, digits, '.', '_' or '-' (it names the zip)"
[[ $VERSION =~ $version_re ]] \
  || die "--version '$VERSION': expected an experiment folder such as v105_dense_blocking"

# ------------------------------------------------------------------ paths ----
# Every path below is relative to the root of the clone that holds this script
ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null) \
  || die "not inside a git clone of the repository"
cd "$ROOT"
export GIT_OPTIONAL_LOCKS=0  # `git status` must not rewrite .git/index: preflight is read-only

PY=.venv/bin/python
OFFICIAL_VALIDATOR=dataset/student_resource/utils/validate_submission.py
# Same lookup as entity_resolution.config.DATASET: the first layout holding train/ (flat
# dataset/ or the organisers' unzipped student_resource/), falling back to dataset/
if [[ -d dataset/train ]]; then DATASET=dataset
elif [[ -d dataset/student_resource/dataset/train ]]; then DATASET=dataset/student_resource/dataset
else DATASET=dataset
fi
MATCHING=$OUTPUTS/matching_results.tsv
CANDIDATES=$OUTPUTS/candidate_pairs.tsv
NB=experiments/$VERSION/$VERSION.ipynb
METRICS=experiments/$VERSION/metrics.json
NAME=${TEAM}_submission                    # the zip unpacks to this folder
STAGE=build/$NAME
CODE=$STAGE/code/business_entity_resolution
ZIP=build/$NAME.zip
LISTING=build/$NAME.contents.txt           # `unzip -l` of the zip (checklist §6, Package)
SUMS=build/$NAME.sha256                    # sha256 of the zip and both TSVs
MAX_FILE_BYTES=$((50 * 1024 * 1024))       # no single file over 50 MiB in the code tree
# Code-tree paths (relative to its root) that must never ship: data, outputs, models, local
# builds, per-experiment artifacts, virtualenvs, git internals, caches and secrets
FORBIDDEN='^(dataset|output|models|submissions|build)/|^experiments/[^/]+/artifacts/'
FORBIDDEN+='|(^|/)(\.?venv|\.git|__pycache__|\.ipynb_checkpoints|\.pytest_cache|\.ruff_cache)/'
FORBIDDEN+='|(^|/)\.env$'
PROBLEMS=0

# -------------------------------------------------------------- preflight ----
# Read-only checks. Each failure is printed and counted; a real run stops after them, a
# dry run goes on to print the steps and reports the count at the end.
branch=$(git symbolic-ref --short -q HEAD || echo 'detached HEAD')
tags=$(git tag --points-at HEAD | xargs)
if (( DRY_RUN )); then
  echo "DRY RUN: read-only preflight, then the steps a real run would take (none is run)"
fi
head_info="$(git rev-parse --short HEAD) on $branch (tags: ${tags:-none})"
printf '  %-8s %s\n' repo "$ROOT" HEAD "$head_info" zip "$ZIP" version "$VERSION" \
  outputs "$OUTPUTS" doc "$DOC" test "$DATASET/test"  # one "label value" line per pair

step "preflight (read-only)"
for tool in tar zip unzip; do
  command -v "$tool" >/dev/null || bad "$tool not found on PATH"
done

# The zip's code is HEAD, so uncommitted edits of tracked files would silently be left out.
# Untracked files are fine: they are simply not packaged (ignored ones are not even listed).
status=$(git status --porcelain)
dirty=$(printf '%s\n' "$status" | grep -v '^[?][?] ' | grep . || true)
untracked=$(printf '%s\n' "$status" | grep -c '^[?][?] ' || true)
if [[ -n $dirty ]]; then
  bad "uncommitted changes to tracked files; commit (or stash) them first:"
  indent "$dirty"
else
  ok "no uncommitted changes to tracked files"
fi
if (( untracked )); then
  note "$untracked untracked path(s) left out: git archive exports HEAD only"
fi

# The version's notebook and metrics.json are copied from HEAD, like the rest of the code
if git cat-file -e "HEAD:$NB" 2>/dev/null && git cat-file -e "HEAD:$METRICS" 2>/dev/null; then
  ok "$NB and metrics.json committed -> src/notebooks/"
else
  committed=$(git ls-tree -d --name-only HEAD experiments/ \
    | sed -n 's|^experiments/\(v[0-9]\{3\}_\)|\1|p' | xargs)
  bad "$NB or metrics.json not committed at HEAD (committed versions: ${committed:-none})"
fi
others=$(git -c core.quotePath=false ls-tree --name-only HEAD src/notebooks/ \
  | grep -vxE "src/notebooks/($VERSION\.ipynb|metrics\.json)" | xargs || true)
if [[ -n $others ]]; then note "src/notebooks/ at HEAD also holds $others (shipped as well)"; fi

# Both TSVs: present, non-empty, right header. The validators check the rest in a real run.
check_tsv() {  # FILE LIST_COLUMN
  local header=''
  if [[ ! -f $1 || ! -s $1 ]]; then bad "$1: missing or empty"; return 0; fi
  IFS= read -r header < "$1" || true  # first line only: the files are ~1 GB
  if [[ $header == "source1_entity_id"$'\t'"$2" ]]; then
    ok "$1 ($(human "$(bytes "$1")"))"
  else
    bad "$1: header $(q "$header"), expected source1_entity_id<TAB>$2"
  fi
}
check_tsv "$MATCHING" matched_entity_ids
check_tsv "$CANDIDATES" candidate_entity_ids

if [[ -f $DOC && -s $DOC ]]; then
  ok "$DOC ($(human "$(bytes "$DOC")")) -> Documentation_template.md"
  git ls-files --error-unmatch -- "$DOC" >/dev/null 2>&1 \
    || note "$DOC is not committed (checklist §4 keeps the filled write-up in git)"
else
  bad "$DOC: missing or empty (--doc)"
fi

# What the validators need: found now rather than after the build has started
if [[ ! -x $PY ]]; then bad "$PY not found (make setup)"
elif [[ ! -f $OFFICIAL_VALIDATOR ]]; then
  bad "$OFFICIAL_VALIDATOR not found (unzip student_resource.zip into dataset/)"
elif [[ ! -f $DATASET/test/test_source1.tsv ]]; then
  bad "$DATASET/test/test_source1.tsv not found: the validators need the test split"
else ok "validators: $PY, $OFFICIAL_VALIDATOR, test split $DATASET/test"
fi

# Preview of step 3 on HEAD's file list, which is exactly what git archive exports
tree=$(git -c core.quotePath=false ls-tree -r -l HEAD)  # "<mode> <type> <sha> <size>\t<path>"
read -r n_files n_bytes <<<"$(printf '%s\n' "$tree" \
  | awk -F'\t' '{ split($1, a, " "); n++; s += a[4] } END { print n + 0, s + 0 }')"
forbidden=$(printf '%s\n' "$tree" | cut -f2- | grep -E "$FORBIDDEN" || true)
big=$(printf '%s\n' "$tree" \
  | awk -F'\t' -v max="$MAX_FILE_BYTES" '{ split($1, a, " ") } a[4] + 0 > max { print $2 }')
if [[ -n $forbidden ]]; then bad "HEAD tracks paths that must not ship:"; indent "$forbidden"; fi
if [[ -n $big ]]; then bad "HEAD tracks files over 50 MiB:"; indent "$big"; fi
if [[ -z $forbidden$big ]]; then
  ok "code at HEAD: $n_files files, $(human "$n_bytes"); none over 50 MiB or from" \
     "dataset/, output/, .venv, artifacts/"
fi

if (( PROBLEMS && ! DRY_RUN )); then die "$PROBLEMS problem(s) above: nothing was built"; fi

# ------------------------------------------------------------------ build ----
step "1/5 validate $OUTPUTS: our checker, then the organisers' validator"
if (( DRY_RUN )); then echo "  (dry run: not run, both read the full test split)"; fi
ours="$PY -m entity_resolution.submission --output-dir $(q "$OUTPUTS")"
ours+=" --dataset-dir $(q "$DATASET")$CHECK_IDS"
official="$PY $OFFICIAL_VALIDATOR --matching $(q "$MATCHING") --candidate $(q "$CANDIDATES")"
official+=" --test-dir $(q "$DATASET/test")$CHECK_IDS"
run "$ours" || die "our checker rejected $OUTPUTS: nothing was built"
run "$official" || die "the organisers' validator rejected $OUTPUTS: nothing was built"

step "2/5 stage $STAGE/ (the working tree is not touched)"
run "rm -rf $(q "$STAGE") $(q "$ZIP") $(q "$LISTING") $(q "$SUMS")"
run "mkdir -p $(q "$STAGE/output") $(q "$CODE/src/notebooks")"
run "git archive --format=tar HEAD | tar -xf - -C $(q "$CODE")"
# Copied from the extracted tree, i.e. HEAD's committed notebook, never a working-tree edit
run "cp $(q "$CODE/$NB") $(q "$CODE/$METRICS") $(q "$CODE/src/notebooks/")"
run "cp -p $(q "$MATCHING") $(q "$CANDIDATES") $(q "$STAGE/output/")"
run "cp -p $(q "$DOC") $(q "$STAGE/Documentation_template.md")"

step "3/5 check the staged code tree"
if (( DRY_RUN )); then
  echo "  (dry run: nothing staged; the preflight ran the same checks on HEAD's file list)"
else
  forbidden=$( (cd "$CODE" && find . ! -type d | sed 's|^\./||') | grep -E "$FORBIDDEN" || true)
  big=$(find "$CODE" -type f -size +"$MAX_FILE_BYTES"c)
  [[ -z $forbidden ]] || die "the staged code tree holds paths that must not ship:"$'\n'"$forbidden"
  [[ -z $big ]] || die "the staged code tree holds files over 50 MiB:"$'\n'"$big"
  ok "$(find "$CODE" -type f | wc -l | tr -d ' ') files, $(du -sh "$CODE" | cut -f1) on disk;" \
     "none over 50 MiB or from dataset/, output/, .venv, artifacts/"
fi

step "4/5 zip"
# The stage's three entries go at the zip root, as in the organisers' tree (student_resource
# README: the write-up is dropped "straight into the zip"), with no wrapping folder.
# -r takes dotfiles too; -y stores symlinks (.claude/skills/caveman) as links, so nothing
# outside the stage can be pulled in. A stale zip was removed in step 2: zip would update it.
run "(cd $(q "$STAGE") && zip -r -q -y $(q "../$NAME.zip") output code Documentation_template.md)"
run "unzip -l $(q "$ZIP") > $(q "$LISTING")"

step "5/5 sizes and sha256"
hashed="$(q "$NAME.zip") $(q "$NAME/output/matching_results.tsv")"
hashed+=" $(q "$NAME/output/candidate_pairs.tsv")"
run "(cd build && sha256sum $hashed) > $(q "$SUMS")"  # paths relative to build/: sha256sum -c
if (( DRY_RUN )); then
  if (( PROBLEMS )); then
    printf '\nDRY RUN: %d problem(s) above would stop a real run. Nothing was run or written.\n' \
      "$PROBLEMS"
    exit 1
  fi
  printf '\nDRY RUN: preflight clean, nothing was run or written. Drop --dry-run to build %s.\n' \
    "$ZIP"
  exit 0
fi
for f in "$ZIP" "$STAGE/output/matching_results.tsv" "$STAGE/output/candidate_pairs.tsv" \
         "$STAGE/Documentation_template.md"; do
  printf '  %-58s %10s\n' "$f" "$(human "$(bytes "$f")")"
done
sed 's/^/  /' "$SUMS"
printf '\nBuilt %s (sha256: %s; listing: %s, %s lines).\n' \
  "$ZIP" "$SUMS" "$LISTING" "$(wc -l < "$LISTING" | tr -d ' ')"
echo "Next (checklist §1 step 5): unzip -d into a temp dir (output/, code/ and the write-up land at"
echo "its root), run the organisers' validator there with --check-ids, fresh-venv smoke test of"
echo "code/business_entity_resolution; record size and sha256 in LEADERBOARD.md."
