import os

import pandas as pd
import pytest

from entity_resolution import config as C
from entity_resolution import data
from entity_resolution.data import (
    load_ground_truth,
    load_source,
    load_sources,
    load_truth_pairs,
    parse_id_list,
    read_id_lists,
)


def test_parquet_cache_serves_repeat_loads(dataset_dir, monkeypatch):
    first = load_source("train", 2, dataset_dir)
    assert (dataset_dir / ".cache" / "train_source2.parquet").is_file()
    monkeypatch.setattr(data, "read_tsv", lambda *a, **k: pytest.fail("re-parsed TSV"))
    cached = load_source("train", 2, dataset_dir)
    assert cached.equals(first)
    assert list(load_source("train", 2, dataset_dir, columns=[C.ENTITY_ID]).columns) == [
        C.ENTITY_ID]


def test_stale_cache_is_rebuilt(dataset_dir):
    load_source("train", 2, dataset_dir)
    path = dataset_dir / "train" / "train_source2.tsv"
    with path.open("a", encoding="utf-8") as f:
        f.write("S2-00003\tNew Co\t1 New St\tUS\n")
    cache = dataset_dir / ".cache" / "train_source2.parquet"
    os.utime(path, (cache.stat().st_mtime + 5, cache.stat().st_mtime + 5))
    assert "S2-00003" in set(load_source("train", 2, dataset_dir)[C.ENTITY_ID])


def test_isin_matches_pandas():
    values = pd.Series(["S2-1", "S2-2", "S3-1", ""], dtype="str")
    for allowed in (pd.Index(["S2-2", "S3-1"]), {"S2-2", "S3-1"}, []):
        assert data.isin(values, allowed).tolist() == values.isin(list(allowed)).tolist()


def test_truth_pairs_explode_lists_and_drop_singletons(dataset_dir):
    pairs = load_truth_pairs(dataset_dir)
    assert list(pairs.columns) == [C.S1_ID, C.ENTITY_ID]
    assert sorted(map(tuple, pairs.to_numpy().tolist())) == [
        ("S1-00001", "S2-00001"), ("S1-00001", "S3-00001"), ("S1-00002", "S2-00002")]


def test_sources_load_as_plain_strings(dataset_dir):
    s1 = load_source("train", 1, dataset_dir)
    assert list(s1.columns) == list(C.SOURCE_COLUMNS)
    row = s1.set_index(C.ENTITY_ID).loc["S1-00003"]
    assert row[C.NAME] == "NA"      # not NaN
    assert row[C.ADDRESS] == ""     # empty stays empty
    assert s1.loc[0, C.ADDRESS] == "12 Main St, Springfield, IL"


def test_csv_style_quotes_are_decoded(dataset_dir):
    # Real files escape quotes CSV-style: """ehpad Club SAS" means "ehpad Club SAS.
    path = dataset_dir / "test" / "test_source1.tsv"
    with path.open("a", encoding="utf-8") as f:
        f.write('S1-00012\t"""ehpad Club SAS"\t"Fédération du ""ehpad"\tFrance\n')
    row = load_source("test", 1, dataset_dir).set_index(C.ENTITY_ID).loc["S1-00012"]
    assert row[C.NAME] == '"ehpad Club SAS'
    assert row[C.ADDRESS] == 'Fédération du "ehpad'


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
    with path.open("a", encoding="utf-8") as f:
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
