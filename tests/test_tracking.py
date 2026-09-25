import csv
import json

import pytest

from entity_resolution.tracking import (
    COLUMNS,
    git_commit,
    log_result,
    new_experiment,
    set_public_score,
    timed,
)


@pytest.fixture
def root(tmp_path):
    template = tmp_path / "_template" / "experiment.ipynb"
    template.parent.mkdir()
    template.write_text("{}")
    return tmp_path


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_versions_only_grow(root):
    first = new_experiment("baseline", root, root / "_template" / "experiment.ipynb")
    second = new_experiment("name_blocking", root, root / "_template" / "experiment.ipynb")
    assert (first.name, second.name) == ("v001_baseline", "v002_name_blocking")
    assert (second / "v002_name_blocking.ipynb").is_file()
    assert (second / "artifacts").is_dir()


@pytest.mark.parametrize("slug", ["Baseline", "name-blocking", "a__b", ""])
def test_slug_must_be_snake_case(root, slug):
    with pytest.raises(ValueError, match="snake_case"):
        new_experiment(slug, root, root / "_template" / "experiment.ipynb")


def test_log_result_upserts_row_and_keeps_public_score(root):
    exp = new_experiment("baseline", root, root / "_template" / "experiment.ipynb")
    table = root / "experiments.csv"
    log_result(exp, change="baseline", group="A1", local_f05=0.5, csv_path=table)
    set_public_score("v001", 0.61, csv_path=table)
    log_result(exp, change="baseline rerun", group="A1", local_f05=0.8123456,
               cand_recall=0.9, metrics={"blocking_seconds": 3.2}, csv_path=table)
    [row] = rows(table)
    assert list(row) == list(COLUMNS)
    assert (row["change"], row["local_f05"], row["public_f05"]) == (
        "baseline rerun", "0.8123", "0.6100")
    saved = json.loads((exp / "metrics.json").read_text())
    assert saved["metrics"] == {"blocking_seconds": 3.2}
    assert saved["cand_recall"] == "0.9000"


def test_public_score_needs_a_logged_version(root):
    with pytest.raises(ValueError, match="no row"):
        set_public_score("v009", 0.7, csv_path=root / "experiments.csv")


def test_log_result_rejects_non_experiment_folder(tmp_path):
    with pytest.raises(ValueError, match="not an experiment folder"):
        log_result(tmp_path, change="x", csv_path=tmp_path / "e.csv")


def test_timed_records_seconds():
    sink = {}
    with timed("blocking", sink):
        pass
    assert sink["blocking_seconds"] >= 0


def test_git_commit_is_a_short_hash():
    assert git_commit() != "unknown"
