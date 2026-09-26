"""decision.py: rule components on the pairs_toy frame, decide == 10 §3, tune == metrics."""
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from entity_resolution import config as C
from entity_resolution import metrics
from entity_resolution.decision import (
    MATCH_COLUMNS,
    SCORED_COLUMNS,
    SPLIT,
    TABLE_COLUMNS,
    DecisionRule,
    Grid,
    _Scorer,
    decide,
    evaluate_rules,
    tune,
)
from entity_resolution.evaluate import macro_f05_from_counts, pairs_to_lists


@pytest.fixture
def pairs_toy():
    """(scored, truth_pairs, s1_ids) where every decision component changes the answer (12 §7).

    S1-a: true a1 0.90 and a2 0.60, decoy a3 0.40, and x at 0.70 that S1-b claims at 0.90
    (1-to-1); its third true match a9 was missed by blocking.  S1-b: true x and b1 0.80,
    decoy b2 0.80 (cap tie-break).  S1-c: a singleton whose best candidate scores 0.55
    (false-merge trap).  S1-d: one true match and no candidates at all.
    """
    scored = pd.DataFrame({
        C.S1_ID: ["S1-a"] * 4 + ["S1-b"] * 3 + ["S1-c"] * 2,
        C.ENTITY_ID: ["S2-a1", "S2-x", "S3-a2", "S2-a3", "S2-x", "S3-b1", "S3-b2",
                      "S2-c1", "S3-c2"],
        "prob": np.float32([0.90, 0.70, 0.60, 0.40, 0.90, 0.80, 0.80, 0.55, 0.30]),
    })
    truth = pd.DataFrame({
        C.S1_ID: ["S1-a", "S1-a", "S1-a", "S1-b", "S1-b", "S1-d"],
        C.ENTITY_ID: ["S2-a1", "S3-a2", "S3-a9", "S2-x", "S3-b1", "S2-d1"],
    })
    return scored, truth, pd.Series(["S1-a", "S1-b", "S1-c", "S1-d"])


def reference_decide(scored: pd.DataFrame, rule: DecisionRule) -> pd.DataFrame:
    """10 §3 verbatim (pandas sort, drop_duplicates, group-bys): what decide must reproduce."""
    sort = dict(by=["prob", C.S1_ID, C.ENTITY_ID], ascending=[False, True, True], kind="stable")
    df = scored.assign(prob=scored["prob"].astype("float64")).sort_values(**sort)
    if rule.one_to_one:
        df = df.drop_duplicates(C.ENTITY_ID, keep="first")
    p_max = df.groupby(C.S1_ID, sort=False)["prob"].transform("max")
    keep = ((df["prob"] >= rule.tau_abs) & (df["prob"] >= rule.tau_rel * p_max)
            & (p_max >= rule.tau_single))
    df = df[keep.to_numpy()]
    rank = df.groupby(C.S1_ID, sort=False).cumcount()
    out = df[(rank < rule.max_matches).to_numpy()][MATCH_COLUMNS]
    return out.sort_values(MATCH_COLUMNS, kind="stable").reset_index(drop=True)


def random_scored(seed: int, n_s1: int = 40, n_pool: int = 90, n_rows: int = 400):
    """Random (scored, truth, s1_ids) with coarse probabilities (many exact ties), pool ids
    scored by several S1s, truth pairs outside the candidates, an entity without candidates
    (S1-nocand, matched) and a singleton without candidates (S1-single)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        C.S1_ID: [f"S1-{i:03d}" for i in rng.integers(0, n_s1, n_rows)],
        C.ENTITY_ID: [f"S{2 + j % 2}-{j:03d}" for j in rng.integers(0, n_pool, n_rows)],
    }).drop_duplicates(ignore_index=True)
    scored = df.assign(prob=(rng.integers(0, 21, len(df)) / 20).astype(np.float32))
    truth = pd.concat([df[rng.random(len(df)) < 0.35],
                       pd.DataFrame({C.S1_ID: ["S1-000", "S1-001", "S1-nocand"],
                                     C.ENTITY_ID: ["S2-900", "S3-901", "S2-902"]})],
                      ignore_index=True)
    ids = pd.Series(sorted({*df[C.S1_ID], "S1-nocand", "S1-single"}))
    return scored, truth, ids


def random_rules(seed: int, n: int = 20) -> list[DecisionRule]:
    """Seeded random rules: tau values on the 0.05 grid of the probabilities (exact ties) or
    anywhere, both one_to_one flags, caps 1-6 and 11."""
    rng = np.random.default_rng(seed)
    rules = []
    for _ in range(n):
        on_grid = rng.random() < 0.5
        tau_abs = float(rng.integers(0, 21) / 20 if on_grid else rng.random())
        rules.append(DecisionRule(tau_abs, float(rng.choice([0.0, 0.5, 0.8, 0.95, rng.random()])),
                                  tau_abs + float(rng.choice([0.0, 0.05, 0.1, rng.random() / 3])),
                                  int(rng.choice([1, 2, 3, 6, 11])), bool(rng.random() < 0.7)))
    return rules


def pair_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    """The (source1_entity_id, entity_id) pairs of a frame."""
    return set(zip(frame[C.S1_ID], frame[C.ENTITY_ID], strict=True))


def test_one_to_one_keeps_highest_prob_s1(pairs_toy):
    scored, _, _ = pairs_toy
    everything = DecisionRule(0.0, 0.0, 0.0, 11, one_to_one=True)
    kept = pair_set(decide(scored, everything))
    assert ("S1-b", "S2-x") in kept and ("S1-a", "S2-x") not in kept     # 0.9 beats 0.7
    tied = scored.assign(prob=np.where(scored[C.ENTITY_ID] == "S2-x", np.float32(0.8),
                                       scored["prob"].to_numpy()))
    kept = pair_set(decide(tied, everything))
    assert ("S1-a", "S2-x") in kept and ("S1-b", "S2-x") not in kept     # 0.8/0.8: smaller id
    shared = pair_set(decide(scored, DecisionRule(0.0, 0.0, 0.0, 11, one_to_one=False)))
    assert {("S1-a", "S2-x"), ("S1-b", "S2-x")} <= shared


def test_tau_abs(pairs_toy):
    scored, _, _ = pairs_toy
    for tau in (0.3, 0.55, 0.6, 0.75, 0.8, 0.9, 0.95):
        out = decide(scored, DecisionRule(tau, 0.0, tau, 11, one_to_one=False))
        assert pair_set(out) == pair_set(scored[scored["prob"].astype("float64") >= tau])


def test_tau_rel(pairs_toy):
    scored, _, _ = pairs_toy
    out = decide(scored, DecisionRule(0.0, 0.7, 0.0, 11))       # S1-a: 0.9 / 0.6 / 0.4 left
    assert out.loc[out[C.S1_ID] == "S1-a", C.ENTITY_ID].tolist() == ["S2-a1"]
    out = decide(scored, DecisionRule(0.0, 0.0, 0.0, 11))
    assert out.loc[out[C.S1_ID] == "S1-a", C.ENTITY_ID].tolist() == ["S2-a1", "S2-a3", "S3-a2"]


def test_tau_single_empties_entity(pairs_toy):
    scored, _, _ = pairs_toy
    assert "S1-c" not in set(decide(scored, DecisionRule(0.5, 0.0, 0.6, 11))[C.S1_ID])
    assert "S1-c" in set(decide(scored, DecisionRule(0.5, 0.0, 0.5, 11))[C.S1_ID])  # 0.55 kept


def test_max_matches_cap(pairs_toy):
    scored, _, _ = pairs_toy
    out = decide(scored, DecisionRule(0.5, 0.0, 0.5, max_matches=2))
    # S1-b: x 0.9, then b1 and b2 tied at 0.8 -> the smaller entity_id
    assert out.loc[out[C.S1_ID] == "S1-b", C.ENTITY_ID].tolist() == ["S2-x", "S3-b1"]
    assert out.groupby(C.S1_ID).size().max() == 2


def test_known_answers_on_pairs_toy(pairs_toy):
    scored, truth, ids = pairs_toy
    rules = [DecisionRule(0.5, 0.0, 0.5, 11), DecisionRule(0.5, 0.0, 0.6, 11),
             DecisionRule(0.5, 0.0, 0.6, 2), DecisionRule(0.5, 0.7, 0.6, 2)]   # R0-R3 of 12 §7
    table = evaluate_rules(scored, ids, truth, rules)
    assert table["f_beta"].round(4).tolist() == [0.4058, 0.6558, 0.7273, 0.6786]
    rule, table = tune(scored, ids, truth)
    # the default caps {4, 6, 11} cannot split S1-b's 0.80 tie, so R2 is out of reach
    assert table["f_beta"].max() == pytest.approx(0.6558, abs=5e-5)
    assert rule == DecisionRule(0.6, 0.6, 0.8, 4, True)   # the most conservative of the ties
    assert len(table) == 3906 + 13
    assert list(table.columns) == TABLE_COLUMNS
    assert table["stage"].value_counts().to_dict() == {"grid": 3906, "refine": 13}


def test_tune_equals_metrics(pairs_toy):
    cases = [pairs_toy, random_scored(0), random_scored(1)]
    for case, (scored, truth, ids) in enumerate(cases):
        truth_lists = pairs_to_lists(truth, ids)
        rules = random_rules(case)
        table = evaluate_rules(scored, ids, truth, rules)
        for rule, f_beta, n_pred in zip(rules, table["f_beta"], table["n_pred"], strict=True):
            matches = decide(scored, rule)
            assert f_beta == metrics.macro_fbeta(pairs_to_lists(matches, ids), truth_lists)
            assert n_pred == len(matches)


def test_histogram_scorer_is_bitwise_macro_f05():
    rng = np.random.default_rng(0)
    n = 50_000
    n_true, n_pred = rng.integers(0, 12, n), rng.integers(0, 40, n)
    tp = np.minimum(rng.integers(0, 12, n), np.minimum(n_pred, n_true))
    for dense in (True, False):                              # histogram and sort paths
        scorer = _Scorer(n_true, width=40)
        scorer.dense = dense
        f_beta, tp_s, pred_s, matched = scorer(scorer.code(tp, n_pred))
        assert f_beta == macro_f05_from_counts(tp, n_pred, n_true)
        assert (tp_s, pred_s, matched) == (tp.sum(), n_pred.sum(), (n_pred > 0).sum())
    # a Veltkamp half times a count below 2**27 is an exact product, and the halves add up
    f, count = rng.random(2000), rng.integers(1, 2**27, 2000)
    t = SPLIT * f
    hi = t - (t - f)
    assert np.array_equal(hi + (f - hi), f)
    for half in (hi, f - hi):
        assert all(Fraction(h) * int(c) == Fraction(h * float(c))
                   for h, c in zip(half, count, strict=True))


def test_decide_equals_reference_implementation():
    for seed in range(3):
        scored, _, _ = random_scored(seed)
        for rule in random_rules(10 + seed):
            pd.testing.assert_frame_equal(decide(scored, rule), reference_decide(scored, rule))


def test_tie_prefers_conservative(pairs_toy):
    scored, truth, ids = pairs_toy
    # tau_abs 0.42 and 0.50 keep the same pairs (a3 = 0.40 is out either way): equal F0.5
    grid = Grid(tau_abs=(0.42, 0.50, 0.08), tau_rel=(0.0,), single_delta=(0.2,),
                max_matches=(11,), refine_span=0.0)
    rule, table = tune(scored, ids, truth, grid)
    assert table.loc[table["stage"] == "grid", "tau_abs"].tolist() == [0.42, 0.5]
    assert table["f_beta"].nunique() == 1
    assert rule.tau_abs == 0.5


def test_grid_size_and_validation():
    grid = Grid()
    assert len(grid.rules(True)) == 3906
    values = grid.tau_abs_values()
    assert (len(values), values[0], values[3], values[-1]) == (31, 0.3, 0.36, 0.9)
    assert all(r.tau_single >= r.tau_abs and not r.one_to_one for r in grid.rules(False))
    assert len(grid.refine_rules(DecisionRule(0.5, 0.0, 0.6, 11))) == 13
    with pytest.raises(ValueError, match="single_delta"):
        Grid(single_delta=(-0.1,)).rules(True)
    with pytest.raises(ValueError):
        Grid(tau_abs=(0.5, 0.3, 0.1))


def test_deterministic():
    scored, truth, ids = random_scored(2)
    shuffled = scored.sample(frac=1, random_state=3)
    for rule in random_rules(4, n=5):
        once = decide(scored, rule)
        pd.testing.assert_frame_equal(decide(shuffled, rule), once)
        pd.testing.assert_frame_equal(decide(scored, rule), once)
    grid = Grid(tau_abs=(0.3, 0.9, 0.1), one_to_one=(True, False))
    rule, table = tune(scored, ids, truth, grid)
    rule2, table2 = tune(shuffled, ids.sample(frac=1, random_state=4), truth, grid)
    assert rule == rule2
    pd.testing.assert_frame_equal(table, table2)


def test_decide_is_idempotent():
    scored, _, _ = random_scored(5)
    for rule in random_rules(6, n=8):
        once = decide(scored, rule)
        carried = once.merge(scored, on=MATCH_COLUMNS)        # probabilities carried along
        pd.testing.assert_frame_equal(decide(carried, rule), once)


def test_schema(pairs_toy):
    scored, truth, ids = pairs_toy
    out = decide(scored, DecisionRule())
    assert list(out.columns) == MATCH_COLUMNS
    assert not out.duplicated().any()
    assert out.index.equals(pd.RangeIndex(len(out)))
    pd.testing.assert_frame_equal(out, out.sort_values(MATCH_COLUMNS, ignore_index=True))
    empty = decide(scored.iloc[:0], DecisionRule())
    assert list(empty.columns) == MATCH_COLUMNS and empty.empty
    duplicated = pd.concat([scored, scored.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        decide(duplicated, DecisionRule())
    with pytest.raises(ValueError, match="duplicate"):
        tune(duplicated, ids, truth)
    with pytest.raises(ValueError, match="scored must start with"):
        decide(scored[["prob", C.S1_ID, C.ENTITY_ID]], DecisionRule())
    assert SCORED_COLUMNS == [C.S1_ID, C.ENTITY_ID, "prob"]


def test_tune_rejects_s1_outside_the_sample(pairs_toy):
    scored, truth, ids = pairs_toy
    with pytest.raises(ValueError, match="not in s1_ids"):
        evaluate_rules(scored, ids[ids != "S1-c"], truth, [DecisionRule()])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_rules(scored, pd.concat([ids, ids]), truth, [DecisionRule()])


def test_expected_decoding_by_hand() -> None:
    """One clear match, a coin-flip pair, a hopeless singleton: the prefix with the best E[F]."""
    from entity_resolution.decision import ExpectedRule, decide_expected
    scored = pd.DataFrame({
        C.S1_ID: ["S1-a", "S1-a", "S1-a", "S1-b", "S1-c"],
        C.ENTITY_ID: ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5"],
        "prob": np.float32([0.95, 0.9, 0.05, 0.3, 0.6]),
    })
    out = decide_expected(scored, ExpectedRule())
    assert sorted(zip(out[C.S1_ID], out[C.ENTITY_ID], strict=True)) == [
        ("S1-a", "S2-1"), ("S1-a", "S2-2"), ("S1-c", "S2-5")]
    # E[F] of keeping S1-b's 0.3 pair: 1.25*0.3/(0.25*0.3+1) = 0.349 < P(empty) = 0.7
    capped = decide_expected(scored, ExpectedRule(max_matches=1))
    assert (capped[C.S1_ID] == "S1-a").sum() == 1
    strict = decide_expected(scored, ExpectedRule(miss=2.0))    # expecting misses: keep more
    assert set(strict[C.ENTITY_ID]) >= {"S2-1", "S2-2"}


def test_tune_expected_scores_what_decide_expected_keeps(pairs_toy) -> None:
    """The table's F0.5 for the chosen rule equals metrics on decide_expected's output."""
    from entity_resolution.decision import decide_expected, tune_expected
    scored, truth, s1_ids = pairs_toy
    rule, table = tune_expected(scored, s1_ids, truth, gammas=(0.5, 1.0, 2.0), misses=(0.0, 0.5))
    assert len(table) == 6
    pred = pairs_to_lists(decide_expected(scored, rule), s1_ids)
    ref = metrics.macro_fbeta(pred, pairs_to_lists(truth, s1_ids))
    assert table["f_beta"].max() == pytest.approx(ref, abs=1e-12)


def test_tune_with_fp_weight_is_never_looser(pairs_toy) -> None:
    """Weighting false merges more can only keep fewer or equal pairs on the chosen rule."""
    scored, truth, s1_ids = pairs_toy
    plain, _ = tune(scored, s1_ids, truth)
    tight, table = tune(scored, s1_ids, truth, fp_weight=3.0)
    assert len(decide(scored, tight)) <= len(decide(scored, plain))
    assert table["f_beta"].max() <= 1.0
