"""Experiment registry: one numbered folder per experiment plus experiments.csv.

Every experiment lives in ``experiments/vNNN_<slug>/``:

    vNNN_<slug>.ipynb   the documented notebook (committed with its outputs)
    metrics.json        everything log_result() recorded (committed)
    artifacts/          models, candidate sets, predictions (gitignored)

and owns one row of ``experiments/experiments.csv``. Each row carries the git commit
of the library code it ran on, so any score traces back to exact code and answers
"which version gave that score?". Versions are never renumbered or reused.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from . import config as C

COLUMNS = ("version", "date", "group", "change", "local_f05", "cand_recall", "public_f05",
           "commit", "notes", "owner", "parent", "decision")
DECISIONS = ("", "KEEP", "DROP", "INVESTIGATE")  # 13 §3; "" while undecided
TEMPLATE = C.EXPERIMENTS / "_template" / "experiment.ipynb"
PLACEHOLDER = "__EXPERIMENT__"
_VERSION_DIR = re.compile(r"^v(\d{3})_([a-z0-9]+(?:_[a-z0-9]+)*)$")
_VERSION_NUMBER = re.compile(r"^v(\d{3})(?:_|$)")  # the number of any vNNN[_...] entry


def new_experiment(
    slug: str, root: Path = C.EXPERIMENTS, template: Path = TEMPLATE, number: int | None = None
) -> Path:
    """Create ``vNNN_<slug>/`` holding a copy of the notebook template.

    Without ``number``, NNN is one above the highest existing version, so numbers only
    ever grow. With ``number`` (taken from the member's reserved range, 13 §1), NNN is
    that number, refused when any ``vNNN`` folder already exists whatever its slug:
    versions are never reused. The template's ``__EXPERIMENT__`` placeholder becomes the
    folder name, so the notebook knows where to log whatever directory Jupyter runs it from.
    """
    if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", slug):
        raise ValueError(f"slug must be snake_case letters and digits, got {slug!r}")
    # every entry named vNNN or vNNN_<anything> holds its number, even with a malformed slug
    taken = {int(m.group(1)): p.name for p in sorted(root.glob("v*"))
             if (m := _VERSION_NUMBER.match(p.name))}
    if number is None:
        number = max(taken, default=0) + 1
    elif number in taken:
        raise ValueError(f"version v{number:03d} is taken by {root / taken[number]}; "
                         "versions are never reused")
    if not 1 <= number <= 999:
        raise ValueError(f"version number must be in 1..999, got {number}")
    exp = root / f"v{number:03d}_{slug}"
    (exp / "artifacts").mkdir(parents=True)
    notebook = template.read_text(encoding="utf-8").replace(PLACEHOLDER, exp.name)
    (exp / f"{exp.name}.ipynb").write_text(notebook, encoding="utf-8")
    return exp


def git_commit(cwd: Path = C.ROOT) -> str:
    """Short HEAD hash, with ``-dirty`` when library code under src/ is uncommitted.

    Only src/ counts: the running notebook itself is always modified by its outputs.
    """
    def git(*args: str) -> str:
        done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                              check=True)
        return done.stdout.strip()

    try:
        sha = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no", "--", "src")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    """Rows of experiments.csv; columns an older header lacks are filled with "" on write.

    A second header line, which a ``merge=union`` of two headers can leave behind, is
    skipped rather than kept as a version called "version".
    """
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("version") != "version"]


def _write_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["version"]))


def _fmt(score: float | None) -> str:
    return "" if score is None else f"{score:.4f}"


def log_result(
    exp_dir: Path,
    *,
    change: str,
    group: str = "",
    local_f05: float | None = None,
    cand_recall: float | None = None,
    notes: str = "",
    owner: str = "",
    parent: str = "",
    decision: str = "",
    metrics: dict | None = None,
    csv_path: Path = C.EXPERIMENTS_CSV,
) -> dict[str, str]:
    """Record an experiment's outcome: ``metrics.json`` in its folder, its csv row.

    ``group`` is the plan ID (A1..E5); ``local_f05`` the macro F0.5 on the fixed
    validation split; ``cand_recall`` the blocking pair recall; ``owner`` the member tag
    (M1..M5); ``parent`` the version this one builds on (v011); ``decision`` KEEP, DROP
    or INVESTIGATE (13 §3, any case), empty while undecided; ``metrics`` any extra
    numbers (timings, candidate counts, thresholds). Re-running the notebook replaces
    the same row, keeping a recorded leaderboard score.
    """
    m = _VERSION_DIR.match(exp_dir.name)
    if not m:
        raise ValueError(f"{exp_dir.name!r} is not an experiment folder (vNNN_<slug>)")
    decision = decision.strip().upper()
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS[1:]} or empty, got {decision!r}")
    version = f"v{m.group(1)}"
    rows = _read_rows(csv_path)
    previous = next((r for r in rows if r["version"] == version), {})
    row = {
        "version": version, "date": date.today().isoformat(), "group": group,
        "change": change, "local_f05": _fmt(local_f05), "cand_recall": _fmt(cand_recall),
        "public_f05": previous.get("public_f05", ""), "commit": git_commit(), "notes": notes,
        "owner": owner, "parent": parent, "decision": decision,
    }
    _write_rows(csv_path, [r for r in rows if r["version"] != version] + [row])
    payload = {**row, "metrics": metrics or {}}
    (exp_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    return row


def set_public_score(version: str, public_f05: float,
                     csv_path: Path = C.EXPERIMENTS_CSV) -> None:
    """Record a leaderboard (public) F0.5 against an experiment's row after an upload."""
    rows = _read_rows(csv_path)
    row = next((r for r in rows if r["version"] == version), None)
    if row is None:
        raise ValueError(f"{version} has no row in {csv_path.name}; run log_result first")
    row["public_f05"] = _fmt(public_f05)
    _write_rows(csv_path, rows)


@contextmanager
def timed(label: str, sink: dict) -> Iterator[None]:
    """Store the wall time of the ``with`` block as ``sink[f"{label}_seconds"]``."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        sink[f"{label}_seconds"] = round(time.perf_counter() - t0, 2)


def _version_number(text: str) -> int:
    """``--number`` argument: ``12``, ``012`` or ``v012`` (as ``make public V=`` spells it)."""
    m = re.fullmatch(r"v?(\d{1,3})", text.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"expected a version number such as 12 or v012, "
                                         f"got {text!r}")
    return int(m.group(1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Experiment registry.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    new = sub.add_parser("new", help="create experiments/vNNN_<slug>/ (next free number, "
                                     "or --number)")
    new.add_argument("slug")
    new.add_argument("--number", type=_version_number, default=None,
                     help="version number from your reserved range (13 §1), e.g. 12 or v012")
    pub = sub.add_parser("public", help="record a leaderboard F0.5 for a version")
    pub.add_argument("version")
    pub.add_argument("score", type=float)
    args = ap.parse_args(argv)
    if args.cmd == "new":
        print(new_experiment(args.slug, number=args.number))
    else:
        set_public_score(args.version, args.score)
        print(f"{args.version}: public_f05 = {args.score:.4f}")


if __name__ == "__main__":
    main()
