import pytest

from entity_resolution import config as C
from entity_resolution.data import (
    load_ground_truth,
    load_source,
    load_sources,
    parse_id_list,
    read_id_lists,
)


def test_sources_load_as_plain_strings(dataset_dir):
    s1 = load_source("train", 1, dataset_dir)
    assert list(s1.columns) == list(C.SOURCE_COLUMNS)
    row = s1.set_index(C.ENTITY_ID).loc["S1-00003"]
    assert row[C.NAME] == "NA"      # not NaN
    assert row[C.ADDRESS] == ""     # empty stays empty
    assert s1.loc[0, C.ADDRESS] == "12 Main St, Springfield, IL"


def test_unseen_country_is_kept(dataset_dir):
    test = load_sources("test", dataset_dir)
    assert set(test) == {1, 2, 3}
    assert "France" in set(test[1][C.COUNTRY])


def test_comma_separated_file_is_rejected(dataset_dir):
    path = dataset_dir / "train" / "train_source1.tsv"
    path.write_text("entity_id,business_name,business_address,country\nS1-1,A,B,US\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_source("train", 1, dataset_dir)


def test_wrong_id_prefix_is_rejected(dataset_dir):
    train = dataset_dir / "train"
    (train / "train_source1.tsv").write_text((train / "train_source2.tsv").read_text())
    with pytest.raises(ValueError, match="lack prefix 'S1-'"):
        load_source("train", 1, dataset_dir)


def test_duplicate_ids_are_rejected(dataset_dir):
    path = dataset_dir / "train" / "train_source2.tsv"
    with path.open("a") as f:
        f.write("S2-00001\tDup\tDup\tUS\n")
    with pytest.raises(ValueError, match="duplicate entity_id"):
        load_source("train", 2, dataset_dir)


@pytest.mark.parametrize(
    ("field", "ids"),
    [
        ("", []),
        ("S2-00001", ["S2-00001"]),
        ("S2-00001,S3-00002", ["S2-00001", "S3-00002"]),
        (" S2-00001 , S3-00002,", ["S2-00001", "S3-00002"]),
        ("S2-00001,S2-00001", ["S2-00001", "S2-00001"]),  # kept so validation can flag it
    ],
)
def test_parse_id_list(field, ids):
    assert parse_id_list(field) == ids


def test_ground_truth_maps_singletons_to_empty_set(dataset_dir):
    assert load_ground_truth(dataset_dir) == {
        "S1-00001": {"S2-00001", "S3-00001"},
        "S1-00002": {"S2-00002"},
        "S1-00003": set(),
    }


def test_read_id_lists_keeps_rows_as_written(tmp_path):
    path = tmp_path / "m.tsv"
    path.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\nS1-1\t\nS1-2\n")
    header, rows = read_id_lists(path)
    assert header == ["source1_entity_id", "matched_entity_ids"]
    assert rows == [("S1-1", ["S2-1"]), ("S1-1", []), ("S1-2", [])]


def test_read_id_lists_rejects_extra_columns(tmp_path):
    path = tmp_path / "m.tsv"
    path.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\tS3-1\n")
    with pytest.raises(ValueError, match="expected 2 tab-separated fields"):
        read_id_lists(path)
