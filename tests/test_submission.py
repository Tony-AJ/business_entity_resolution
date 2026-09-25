import pytest

from entity_resolution.submission import main, split_ids, validate, write_submission

MATCHES = {"S1-00010": ["S2-00010"]}
CANDIDATES = {"S1-00010": ["S2-00010", "S3-00011"], "S1-00011": ["S3-00010"]}
HEADER = "source1_entity_id\tmatched_entity_ids\n"


@pytest.fixture
def ids(dataset_dir):
    return split_ids("test", dataset_dir)


@pytest.fixture
def written(tmp_path, ids):
    s1_ids, _ = ids
    return write_submission(MATCHES, CANDIDATES, s1_ids, tmp_path / "output")


def test_written_files_pass(written, ids):
    assert validate(*written, *ids) == ([], [])


def test_file_format_is_exact(written):
    matching, candidates = written
    assert matching.read_text() == HEADER + "S1-00010\tS2-00010\nS1-00011\t\n"
    assert candidates.read_text().splitlines()[1] == "S1-00010\tS2-00010,S3-00011"


def test_writer_drops_repeated_ids(tmp_path, ids):
    matching, _ = write_submission({"S1-00010": ["S2-00010", "S2-00010"]}, {}, ids[0], tmp_path)
    assert "S1-00010\tS2-00010\n" in matching.read_text()


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("S1-00010\tS2-00010\n", "1 missing Source 1 entities"),
        ("S1-00010\tS2-00010\nS1-00011\t\nS1-00011\t\n", "1 duplicate rows"),
        ("S1-00010\tS2-00010\nS1-00011\t\nS1-99999\t\n", "rows for unknown Source 1"),
        ("S1-00010\tS2-00010,S2-00010\nS1-00011\t\n", "lists with repeated IDs"),
        ("S1-00010\tS1-00011\nS1-00011\t\n", "not Source 2/3 records"),  # self-match
        ("S1-00010\tS2-99999\nS1-00011\t\n", "not Source 2/3 records"),  # unknown ID
    ],
)
def test_each_rule_is_enforced(written, ids, body, error):
    matching, candidates = written
    matching.write_text(HEADER + body)
    errors, _ = validate(matching, candidates, *ids)
    assert any(error in e for e in errors), errors


def test_wrong_header_is_an_error(written, ids):
    matching, candidates = written
    matching.write_text("id\tmatches\nS1-00010\tS2-00010\nS1-00011\t\n")
    errors, _ = validate(matching, candidates, *ids)
    assert any("header must be" in e for e in errors)


def test_match_outside_candidates_warns(written, ids):
    matching, candidates = written
    matching.write_text(HEADER + "S1-00010\tS2-00010\nS1-00011\tS3-00011\n")
    assert validate(matching, candidates, *ids) == ([], [
        "1 entities have matches that are not candidates, e.g. ['S1-00011']"])


def test_cli_exit_codes(written, dataset_dir, capsys):
    out_dir = str(written[0].parent)
    args = ["--output-dir", out_dir, "--dataset-dir", str(dataset_dir)]
    assert main(args) == 0
    assert capsys.readouterr().out.strip().endswith("PASS")
    written[0].unlink()
    assert main(args) == 1
    assert "FAIL" in capsys.readouterr().out
