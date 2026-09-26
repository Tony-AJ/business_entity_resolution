"""Matcher contract (docs/plan/08 §1, §10): determinism, column guard, backends, persistence.

Every test runs on a small synthetic pair frame; LightGBM fits use at most 200 rounds on
two threads, so each takes milliseconds.
"""
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from entity_resolution.model import (
    BACKENDS,
    CALIBRATION_BINS,
    HEURISTIC_SIMS,
    Matcher,
    MatcherParams,
    reliability,
)

FAST = {"n_estimators": 60, "num_threads": 2}
SAVED_FILES = {"lgbm": {"model.txt"}, "logreg": {"model.joblib"}, "heuristic": set()}


def make_pairs(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic candidate pairs shaped like build_features output, and their labels.

    Blocking flags and similarities (NaN where the pass did not produce the pair), name
    and address similarities with NaN where undefined, a few counts and flags. The label
    is a noisy function of core_token_set and ad_token_set, so a model has something to
    learn but cannot be perfect. The index does not start at 0, to exercise alignment.
    """
    rng = np.random.default_rng(seed)

    def with_nan(values: np.ndarray, share: float) -> np.ndarray:
        """``values`` as float32 with a random ``share`` of them set to NaN."""
        return np.where(rng.random(n) < share, np.nan, values).astype(np.float32)

    core = rng.random(n)
    addr = rng.random(n)
    pass_exact = (rng.random(n) < 0.15).astype(np.float32)
    pass_name_char = rng.random(n) < 0.7
    noisy_core = np.clip(core + 0.1 * rng.standard_normal(n), 0, 1)
    X = pd.DataFrame({
        "pass_exact": pass_exact,
        "pass_name_char": pass_name_char.astype(np.float32),
        "sim_name_char": np.where(pass_name_char, noisy_core, np.nan).astype(np.float32),
        "sim_name_addr_word": with_nan((core + addr) / 2, 0.5),
        "sim_addr_char": with_nan(addr, 0.8),
        "core_token_set": with_nan(core, 0.05),
        "ad_token_set": with_nan(addr, 0.1),
        "postcode_eq": with_nan((rng.random(n) < addr).astype(float), 0.3),
        "num_first_eq": with_nan((rng.random(n) < 0.5).astype(float), 0.2),
        "ctx_n_cands": rng.integers(1, 61, n).astype(np.float32),
        "is_s3": (rng.random(n) < 0.5).astype(np.float32),
    }, index=pd.RangeIndex(1000, 1000 + n))
    score = 5 * core + 2 * addr + 1.5 * pass_exact - 4.5 + rng.logistic(size=n)
    y = pd.Series((score > 0).astype(np.int8), index=X.index, name="label")
    return X, y


@pytest.fixture(scope="module")
def data():
    """(fit frame, fit labels, tune frame, tune labels); tests never modify them."""
    return (*make_pairs(3000, seed=0), *make_pairs(1500, seed=1))


def fitted(data, backend: str = "lgbm", **params) -> Matcher:
    """A matcher of ``backend`` fitted on the fit frame, early-stopped on the tune frame."""
    X, y, Xt, yt = data
    return Matcher(MatcherParams(backend=backend, **{**FAST, **params})).fit(X, y, Xt, yt)


def test_fit_is_deterministic(data):
    X = data[0]
    first = fitted(data, seed=42).predict_proba(X)
    assert np.array_equal(first, fitted(data, seed=42).predict_proba(X))
    assert not np.array_equal(first, fitted(data, seed=43).predict_proba(X))


@pytest.mark.parametrize("backend", BACKENDS)
def test_predict_proba_range_dtype_length(data, backend):
    Xt = data[2]
    prob = fitted(data, backend).predict_proba(Xt)
    assert prob.dtype == np.float32 and prob.shape == (len(Xt),)
    assert not np.isnan(prob).any() and ((prob >= 0) & (prob <= 1)).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_chunked_predict_equals_full(data, backend):
    matcher, Xt = fitted(data, backend), data[2]
    chunked = matcher.predict_proba(Xt, chunk_rows=7)
    full = matcher.predict_proba(Xt, chunk_rows=10**9)
    if backend == "logreg":
        # BLAS summation order depends on batch size, so float32 output drifts by a few ulp
        np.testing.assert_allclose(chunked, full, rtol=2e-6, atol=1e-7)
    else:
        assert np.array_equal(chunked, full)  # tree and nanmax backends are bit-deterministic


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_frame_gives_empty_float32(data, backend):
    prob = fitted(data, backend).predict_proba(data[2].iloc[:0])
    assert prob.dtype == np.float32 and prob.shape == (0,)


@pytest.mark.parametrize("backend", BACKENDS)
def test_save_load_roundtrip(data, backend, tmp_path):
    matcher, Xt = fitted(data, backend), data[2]
    saved = matcher.save(tmp_path / "model")
    assert {p.name for p in saved.iterdir()} == {"params.json", "feature_names.json",
                                                 *SAVED_FILES[backend]}
    params = json.loads((saved / "params.json").read_text())
    assert params["backend"] == backend and params["fit_info"] == matcher.fit_info_
    loaded = Matcher.load(saved)
    assert np.array_equal(loaded.predict_proba(Xt), matcher.predict_proba(Xt))  # bit for bit
    assert loaded.feature_names_ == matcher.feature_names_ == list(Xt.columns)
    assert loaded.best_iteration_ == matcher.best_iteration_
    assert loaded.params == matcher.params and loaded.fit_info_ == matcher.fit_info_


def swap(columns: pd.Index, a: str, b: str) -> list[str]:
    """``columns`` with ``a`` and ``b`` exchanged."""
    order = list(columns)
    i, j = order.index(a), order.index(b)
    order[i], order[j] = order[j], order[i]
    return order


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda X: X.drop(columns="sim_addr_char"), r"missing \['sim_addr_char'\]"),
        (lambda X: X.assign(extra=np.float32(0)), r"unexpected \['extra'\]"),
        (lambda X: X[swap(X.columns, "sim_name_char", "core_token_set")],
         r"reordered \['core_token_set', .*'sim_name_char'\]"),
        # the column check runs before the data is read: the object column is not reached
        (lambda X: X.drop(columns="is_s3").astype({"ctx_n_cands": object}),
         r"missing \['is_s3'\]"),
    ],
    ids=["missing", "extra", "reordered", "checked_before_data"],
)
def test_column_mismatch_raises(data, change, message):
    matcher = fitted(data)
    with pytest.raises(ValueError, match=message):
        matcher.predict_proba(change(data[2]))


def test_early_stopping_on_tune(data):
    matcher = fitted(data, n_estimators=200, early_stopping=5, learning_rate=0.3)
    assert 0 < matcher.best_iteration_ < 200
    info = matcher.fit_info_
    assert set(info) == {"rows", "positive_rate", "best_iteration", "tune_logloss",
                         "tune_auc", "fit_seconds"}
    assert info["rows"] == len(data[0]) and 0 < info["positive_rate"] < 1
    assert info["best_iteration"] == matcher.best_iteration_
    assert 0 < info["tune_logloss"] < 0.69 and 0.8 < info["tune_auc"] <= 1


def test_fit_without_tune_trains_every_round_and_warns(data):
    X, y = data[:2]
    with pytest.warns(UserWarning, match="without early stopping"):
        matcher = Matcher(MatcherParams(**FAST)).fit(X, y)
    assert matcher.best_iteration_ == FAST["n_estimators"]
    assert matcher.fit_info_["tune_logloss"] is None and matcher.fit_info_["tune_auc"] is None


def test_heuristic_equals_nanmax(data):
    X, y = data[:2]
    # fit on an empty frame is a no-op: nothing is learnt, the columns are recorded
    matcher = Matcher(MatcherParams(backend="heuristic")).fit(X.iloc[:0], y.iloc[:0])
    assert matcher.model_ is None and matcher.fit_info_["rows"] == 0
    prob = matcher.predict_proba(X)
    sims = X[list(HEURISTIC_SIMS)].to_numpy()
    all_nan = np.isnan(sims).all(axis=1)
    exact = X["pass_exact"].to_numpy() == 1
    assert all_nan.sum() and exact.sum() and (all_nan & exact).sum()  # every case occurs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # nanmax of an all-NaN row
        expected = np.nanmax(sims, axis=1).astype(np.float32)
    rest = ~exact & ~all_nan
    assert np.array_equal(prob[rest], expected[rest])
    assert (prob[exact] == 1.0).all()                 # exact key wins, even without sims
    assert (prob[all_nan & ~exact] == 0.0).all()
    refit = Matcher(MatcherParams(backend="heuristic")).fit(X, y)
    assert np.array_equal(refit.predict_proba(X), prob)


def test_heuristic_needs_its_columns(data):
    X, y = data[:2]
    with pytest.raises(ValueError, match=r"\['pass_exact', 'sim_addr_char'\]"):
        Matcher(MatcherParams(backend="heuristic")).fit(
            X.drop(columns=["pass_exact", "sim_addr_char"]), y)


def test_logreg_handles_nan(data):
    X, _, Xt, _ = data
    assert X.isna().any().sum() >= 5 and Xt.isna().any().sum() >= 5
    # a column that is all NaN in the fit sample (P4 off) and NaN where fit had none
    harder = (X.assign(sim_addr_char=np.float32(np.nan)), data[1],
              Xt.assign(is_s3=Xt["is_s3"].where(Xt.index % 7 > 0)), data[3])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no convergence, imputer or feature-name warning
        matcher = fitted(harder, "logreg")
        prob = matcher.predict_proba(harder[2])
    assert np.isfinite(prob).all()
    assert matcher.fit_info_["tune_auc"] > 0.8


@pytest.mark.parametrize("backend", BACKENDS)
def test_importance_matches_feature_names(data, backend):
    matcher = fitted(data, backend)
    imp = matcher.importance()
    assert sorted(imp.index) == sorted(matcher.feature_names_) and imp.index.is_unique
    assert (imp >= 0).all() and imp.sum() == pytest.approx(1.0)
    assert imp.is_monotonic_decreasing  # largest first, for the top-20 log
    if backend == "heuristic":
        assert set(imp.index[:3]) == set(HEURISTIC_SIMS) and imp.iloc[0] == pytest.approx(1 / 3)
    else:
        assert imp.index[0] == "core_token_set"  # the label's main driver


def test_rejects_object_dtype_and_misaligned_y(data):
    X, y = data[:2]
    matcher = Matcher(MatcherParams(**FAST))
    with pytest.raises(ValueError, match="is_s3"):
        matcher.fit(X.astype({"is_s3": object}), y)
    with pytest.raises(ValueError, match="is_s3"):
        matcher.fit(X.assign(is_s3="yes"), y)  # pandas str column
    with pytest.raises(ValueError, match="one value per row"):
        matcher.fit(X, y.iloc[:-1])
    with pytest.raises(ValueError, match="not aligned"):
        matcher.fit(X, y.reset_index(drop=True))
    with pytest.raises(ValueError, match="0/1"):
        matcher.fit(X, y.replace({1: 2}))
    assert matcher.feature_names_ is None  # nothing was fitted


def test_weight_reaches_the_model(data):
    X, y, Xt, yt = data
    heavy_positives = np.where(y == 1, 5.0, 1.0)
    for backend in ("lgbm", "logreg"):
        params = MatcherParams(backend=backend, **FAST)
        base = Matcher(params).fit(X, y, Xt, yt).predict_proba(Xt)
        heavy = Matcher(params).fit(X, y, Xt, yt, weight=heavy_positives).predict_proba(Xt)
        assert heavy.mean() > base.mean() + 0.05
    unit = Matcher(MatcherParams(**FAST)).fit(X, y, Xt, yt, weight=np.ones(len(X)))
    assert np.array_equal(unit.predict_proba(Xt), fitted(data).predict_proba(Xt))
    with pytest.raises(ValueError, match="weight"):
        Matcher(MatcherParams(**FAST)).fit(X, y, weight=np.ones(len(X) - 1))


def test_unfitted_and_unknown_backend_raise(data):
    with pytest.raises(RuntimeError, match="not fitted"):
        Matcher().predict_proba(data[2])
    with pytest.raises(ValueError, match="backend"):
        MatcherParams(backend="xgboost")


def test_reliability_separates_calibrated_from_overconfident():
    rng = np.random.default_rng(0)
    p = rng.random(100_000)
    ece, brier, table = reliability(p, rng.random(100_000) < p)  # calibrated by construction
    assert ece < 0.02 and 0.0 < brier < 0.25
    assert list(table.columns) == ["bin", "n", "p_mean", "y_rate", "gap", "p_lo", "p_hi"]
    assert len(table) == CALIBRATION_BINS and table["n"].sum() == 100_000
    assert table["p_mean"].is_monotonic_increasing
    ece_bad, brier_bad, _ = reliability(np.full(1000, 0.9), np.zeros(1000))
    assert ece_bad == pytest.approx(0.9) and brier_bad == pytest.approx(0.81)


def test_reliability_by_hand():
    # sorted: (0.1, 0) (0.2, 1) | (0.8, 1) (0.9, 1) -> gaps -0.35 and -0.15, two pairs each
    ece, brier, table = reliability(np.array([0.1, 0.9, 0.2, 0.8]), np.array([0, 1, 1, 1]),
                                    bins=2)
    assert ece == pytest.approx(0.25) and brier == pytest.approx(0.175)
    assert table["n"].tolist() == [2, 2]
    assert table["gap"].tolist() == pytest.approx([-0.35, -0.15])
    assert table[["p_lo", "p_hi"]].to_numpy().tolist() == [[0.1, 0.2], [0.8, 0.9]]
    # fewer pairs than bins: empty bins are dropped, never divided by zero
    ece, _, table = reliability(np.array([0.3, 0.7]), np.array([0, 1]), bins=5)
    assert table["n"].tolist() == [1, 1] and ece == pytest.approx(0.3)
    ece, brier, table = reliability(np.zeros(0), np.zeros(0))
    assert np.isnan(ece) and np.isnan(brier) and table.empty
