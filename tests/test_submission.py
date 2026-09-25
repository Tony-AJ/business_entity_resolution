import numpy as np
import pandas as pd
import pytest

from entity_resolution import submission
from entity_resolution.data import read_id_lists
from entity_resolution.submission import (
    main,
    split_ids,
    validate,
    write_pairs,
    write_submission,
)

MATCHES = {"S1-00010": ["S2-00010"]}
CANDIDATES = {"S1-00010": ["S2-00010", "S3-00011"], "S1-00011": ["S3-00010"]}
HEADER = "source1_entity_id\tmatched_entity_ids\n"
PAIR_COLUMNS = ["source1_entity_id", "entity_id"]


@pytest.fixture
def ids(dataset_dir):
    return split_ids("test", dataset_dir, check_ids=True)


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


def test_default_mode_checks_prefixes_only(written, dataset_dir):
    s1_ids, valid = split_ids("test", dataset_dir)
    assert valid is None
    matching, candidates = written
    matching.write_text(HEADER + "S1-00010\tS2-99999\nS1-00011\tS1-00010\n")
    errors, _ = validate(matching, candidates, s1_ids)
    assert len(errors) == 1 and "['S1-00010']" in errors[0]  # S1 self-match only


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


# ---------------------------------------------------------------- write_pairs ----


def pair_frame(lists: dict[str, list[str]], dtype: object = "str") -> pd.DataFrame:
    """Pair frame with one row per listed id, in dict and list order."""
    rows = [(s1, i) for s1, ids in lists.items() for i in ids]
    return pd.DataFrame(rows, columns=PAIR_COLUMNS, dtype=dtype)


def id_lists(pairs: pd.DataFrame) -> dict[str, list[str]]:
    """Reference conversion to the dict form, ids in frame order (small frames only)."""
    lists: dict[str, list[str]] = {}
    for s1, i in zip(pairs[PAIR_COLUMNS[0]], pairs[PAIR_COLUMNS[1]], strict=True):
        lists.setdefault(s1, []).append(i)
    return lists


def test_write_pairs_tsv_format_exact(tmp_path, ids):
    matching, candidates = write_pairs(pair_frame(MATCHES), pair_frame(CANDIDATES), ids[0],
                                       tmp_path)
    assert (matching.name, candidates.name) == ("matching_results.tsv", "candidate_pairs.tsv")
    assert matching.read_bytes() == (b"source1_entity_id\tmatched_entity_ids\n"
                                     b"S1-00010\tS2-00010\n"
                                     b"S1-00011\t\n")
    assert candidates.read_bytes() == (b"source1_entity_id\tcandidate_entity_ids\n"
                                       b"S1-00010\tS2-00010,S3-00011\n"
                                       b"S1-00011\tS3-00010\n")


@pytest.mark.parametrize(
    ("to_container", "dtype"),
    [(list, "str"), (pd.Series, "str"), (pd.Index, object), (np.array, object)],
    ids=["list", "series", "index", "ndarray_object_frames"],
)
def test_write_pairs_equals_write_id_lists(tmp_path, monkeypatch, to_container, dtype):
    monkeypatch.setattr(submission, "WRITE_BATCH", 7)  # many batches, one partial at the end
    rng = np.random.default_rng(0)
    s1_ids = [f"S1-{i:05d}" for i in rng.permutation(60)]  # file order is s1_ids order
    pool = [f"S{2 + i % 2}-{i:05d}" for i in range(40)]
    n = 500
    # 10 of the S1 ids have no pairs, S1-99999 is not in s1_ids, repeats make duplicate pairs
    candidates = pd.DataFrame({
        "source1_entity_id": rng.choice(s1_ids[:50] + ["S1-99999"], n),
        "entity_id": rng.choice(pool, n),
        "prob": rng.random(n).astype(np.float32),
    }).astype({"source1_entity_id": dtype, "entity_id": dtype})
    # concatenated pieces, as per-country frames arrive: multi-chunk Arrow columns
    candidates = pd.concat([candidates.iloc[:120], candidates.iloc[120:300],
                            candidates.iloc[300:]])
    matches = candidates.sort_values("prob", ascending=False).head(90)[PAIR_COLUMNS]
    got = write_pairs(matches, candidates, to_container(s1_ids), tmp_path / "pairs")
    want = write_submission(id_lists(matches), id_lists(candidates), s1_ids, tmp_path / "dicts")
    assert [p.read_bytes() for p in got] == [p.read_bytes() for p in want]


def test_write_pairs_every_s1_present(tmp_path):
    s1_ids = ["S1-3", "S1-1", "S1-2"]
    matches = pair_frame({"S1-1": ["S3-7", "S2-1"], "S1-9": ["S2-9"]})  # S1-9: not written
    candidates = pair_frame({"S1-1": ["S2-1", "S3-7", "S2-1"], "S1-2": ["S3-2"]})
    matching, cands = write_pairs(matches, candidates, s1_ids, tmp_path)
    assert read_id_lists(matching)[1] == [("S1-3", []), ("S1-1", ["S3-7", "S2-1"]),
                                          ("S1-2", [])]
    assert read_id_lists(cands)[1] == [("S1-3", []), ("S1-1", ["S2-1", "S3-7"]),
                                       ("S1-2", ["S3-2"])]
    matching, cands = write_pairs(pair_frame({}), pair_frame({}), s1_ids, tmp_path / "none")
    assert read_id_lists(matching)[1] == read_id_lists(cands)[1] == [(s, []) for s in s1_ids]


@pytest.mark.parametrize("which", ["matches", "candidates"])
def test_write_pairs_only_s2_s3_ids(tmp_path, ids, which):
    frames = {"matches": pair_frame(MATCHES), "candidates": pair_frame(CANDIDATES)}
    frames[which] = pair_frame({"S1-00010": ["S2-00010", "S1-00011"]})  # an S1 self-match
    with pytest.raises(ValueError, match=rf"{which}: 1 listed ids .*\['S1-00011'\]"):
        write_pairs(frames["matches"], frames["candidates"], ids[0], tmp_path / "out")
    assert not (tmp_path / "out").exists()  # both frames are checked before writing
    frames[which] = pair_frame(MATCHES).rename(columns={"entity_id": "pool_id"})
    with pytest.raises(ValueError, match=r"missing columns \['entity_id'\]"):
        write_pairs(frames["matches"], frames["candidates"], ids[0], tmp_path / "out")


def test_write_pairs_warns_when_match_not_candidate(tmp_path, ids):
    # S3-00011 is a candidate of S1-00010 only
    matches = pair_frame({"S1-00010": ["S2-00010"], "S1-00011": ["S3-00011"]})
    written = write_pairs(matches, pair_frame(CANDIDATES), ids[0], tmp_path)
    assert all(p.is_file() for p in written)
    assert validate(*written, *ids) == ([], [
        "1 entities have matches that are not candidates, e.g. ['S1-00011']"])


def test_write_pairs_ids_exist_when_check_ids(tmp_path, ids):
    written = write_pairs(pair_frame(MATCHES), pair_frame(CANDIDATES), ids[0], tmp_path)
    assert validate(*written, *ids) == ([], [])
    unknown = write_pairs(pair_frame({"S1-00010": ["S2-99999"]}), pair_frame(CANDIDATES),
                          ids[0], tmp_path / "unknown")
    errors, _ = validate(*unknown, *ids)
    assert any("not Source 2/3 records of this split" in e for e in errors), errors
