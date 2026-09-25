import pytest

from entity_resolution.metrics import breakdown, entity_fbeta, macro_fbeta, main


def test_problem_statement_example():
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    truth = {"S2-00047", "S3-00812"}
    assert entity_fbeta(pred, truth) == pytest.approx(5 / 7)  # 0.714 in the brief


@pytest.mark.parametrize(
    ("pred", "truth", "score"),
    [
        (set(), set(), 1.0),  # singleton, correctly left empty
        ({"S2-1"}, set(), 0.0),  # false merge on a singleton
        (set(), {"S2-1"}, 0.0),  # every match missed
        ({"S3-9"}, {"S2-1"}, 0.0),  # every prediction wrong
        ({"S2-1", "S3-2"}, {"S2-1", "S3-2"}, 1.0),
    ],
)
def test_entity_edge_cases(pred, truth, score):
    assert entity_fbeta(pred, truth) == score


def test_precision_weighs_more_than_recall():
    truth = {"a", "b"}
    half_wrong = entity_fbeta({"a", "b", "x", "y"}, truth)  # P=0.5, R=1
    half_missed = entity_fbeta({"a"}, truth)  # P=1, R=0.5
    assert half_wrong < half_missed


def test_macro_counts_missing_entities_as_empty():
    truth = {"S1-1": {"S2-1"}, "S1-2": set()}
    assert macro_fbeta({"S1-1": ["S2-1"]}, truth) == 1.0
    assert macro_fbeta({}, truth) == 0.5


def test_breakdown():
    truth = {"S1-1": {"S2-1", "S3-1"}, "S1-2": set(), "S1-3": set()}
    pred = {"S1-1": ["S2-1"], "S1-2": ["S2-9"]}
    r = breakdown(pred, truth)
    assert r["f_beta_matched"] == pytest.approx(entity_fbeta({"S2-1"}, {"S2-1", "S3-1"}))
    assert r["f_beta_singletons"] == 0.5
    assert r["pair_precision"] == 0.5
    assert r["pair_recall"] == 0.5
    assert (r["entities"], r["singletons"]) == (3, 2)


def test_cli_scores_files(tmp_path, capsys):
    header = "source1_entity_id\tmatched_entity_ids\n"
    (tmp_path / "pred.tsv").write_text(header + "S1-1\tS2-1\nS1-2\t\n")
    (tmp_path / "truth.tsv").write_text(header + "S1-1\tS2-1,S3-1\nS1-2\t\n")
    main(["--pred", str(tmp_path / "pred.tsv"), "--truth", str(tmp_path / "truth.tsv")])
    out = capsys.readouterr().out
    assert "f_beta_singletons  1.0000" in out
    assert "pair_recall  0.5000" in out
