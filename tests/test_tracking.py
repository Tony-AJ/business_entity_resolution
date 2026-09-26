import csv
import json
from pathlib import Path

import pytest

from entity_resolution import tracking
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
    template.write_text('{"title": "__EXPERIMENT__"}')
    return tmp_path


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_versions_only_grow(root):
    first = new_experiment("baseline", root, root / "_template" / "experiment.ipynb")
    second = new_experiment("name_blocking", root, root / "_template" / "experiment.ipynb")
    assert (first.name, second.name) == ("v001_baseline", "v002_name_blocking")
    notebook = (second / "v002_name_blocking.ipynb").read_text()
    assert notebook == '{"title": "v002_name_blocking"}'  # placeholder filled
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
               mock_f05=0.75, cand_recall=0.9, metrics={"blocking_seconds": 3.2},
               csv_path=table)
    [row] = rows(table)
    assert list(row) == list(COLUMNS)
    assert (row["change"], row["local_f05"], row["mock_f05"], row["public_f05"]) == (
        "baseline rerun", "0.8123", "0.7500", "0.6100")
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


def test_new_experiment_with_number(root):
    template = root / "_template" / "experiment.ipynb"
    exp = new_experiment("lgbm_base", root, template, number=7)
    assert exp.name == "v007_lgbm_base" and (exp / "artifacts").is_dir()
    assert (exp / "v007_lgbm_base.ipynb").read_text() == '{"title": "v007_lgbm_base"}'
    assert new_experiment("next", root, template).name == "v008_next"  # auto goes above it


def test_new_experiment_number_collision_raises(root):
    template = root / "_template" / "experiment.ipynb"
    new_experiment("lgbm_base", root, template, number=7)
    (root / "v009_Bad-Name").mkdir()  # a malformed slug still holds its number
    for number, folder in ((7, "v007_lgbm_base"), (9, "v009_Bad-Name")):
        with pytest.raises(ValueError, match=folder):
            new_experiment("other_slug", root, template, number=number)
    assert new_experiment("auto", root, template).name == "v010_auto"  # never reuses v009
    assert sorted(p.name for p in root.glob("v*")) == [
        "v007_lgbm_base", "v009_Bad-Name", "v010_auto"]


@pytest.mark.parametrize("number", [0, -3, 1000])
def test_new_experiment_number_out_of_range(root, number):
    with pytest.raises(ValueError, match="1..999"):
        new_experiment("slug", root, root / "_template" / "experiment.ipynb", number=number)


def test_log_result_fills_owner_parent_decision(root):
    table = root / "experiments.csv"
    table.write_text("version,date,group,change,local_f05,cand_recall,public_f05,commit,notes\n"
                     "v001,2026-09-25,A1,baseline,0.5000,,0.6100,abc1234,first\n")  # old header
    exp = new_experiment("lgbm_base", root, root / "_template" / "experiment.ipynb", number=60)
    log_result(exp, change="lgbm defaults", group="D3", local_f05=0.8, owner="M4",
               parent="v001", decision="keep", csv_path=table)
    old, new = rows(table)
    assert list(old) == list(new) == list(COLUMNS)
    assert list(COLUMNS[-3:]) == ["owner", "parent", "decision"]
    assert (old["version"], old["public_f05"], old["notes"]) == ("v001", "0.6100", "first")
    assert (old["owner"], old["parent"], old["decision"]) == ("", "", "")
    assert (new["version"], new["owner"], new["parent"], new["decision"]) == (
        "v060", "M4", "v001", "KEEP")
    saved = json.loads((exp / "metrics.json").read_text())
    assert (saved["owner"], saved["parent"], saved["decision"]) == ("M4", "v001", "KEEP")
    with pytest.raises(ValueError, match="decision"):
        log_result(exp, change="x", decision="MAYBE", csv_path=table)


def test_stray_header_from_union_merge_is_dropped(root):
    table = root / "experiments.csv"
    header = ",".join(COLUMNS)
    table.write_text(f"{header}\nv001,2026-09-25,A1,a,,,,abc1234,,M1,,\n{header}\n")
    set_public_score("v001", 0.5, csv_path=table)
    assert [r["version"] for r in rows(table)] == ["v001"]


def test_cli_new_passes_number(monkeypatch, capsys):
    calls = []

    def fake_new_experiment(slug, number=None):
        calls.append((slug, number))
        return Path("experiments") / f"v{number or 1:03d}_{slug}"

    monkeypatch.setattr(tracking, "new_experiment", fake_new_experiment)
    for argv in (["new", "lgbm_base", "--number", "60"], ["new", "lgbm_base", "--number", "v061"],
                 ["new", "lgbm_base"]):
        tracking.main(argv)
    assert calls == [("lgbm_base", 60), ("lgbm_base", 61), ("lgbm_base", None)]
    assert capsys.readouterr().out.splitlines()[0].endswith("v060_lgbm_base")
    with pytest.raises(SystemExit):
        tracking.main(["new", "lgbm_base", "--number", "sixty"])
