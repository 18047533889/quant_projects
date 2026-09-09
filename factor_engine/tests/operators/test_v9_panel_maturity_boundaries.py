"""Q02 scoped public CPU boundary tests; not end-to-end PIT certification."""
import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


NAMES = (
    "panel_rolling_pcr_forecast", "panel_rolling_pls_forecast",
    "panel_rolling_elastic_net_forecast", "panel_regime_conditioned_forecast",
    "panel_mixture_of_experts_score",
)


@pytest.fixture(scope="module", autouse=True)
def registry():
    load_all()


def fixture():
    rng = np.random.default_rng(902)
    idx = pd.date_range("2020-01-01", periods=110)
    def frame(a):
        return pd.DataFrame(a, index=idx, columns=["A", "B"])
    x1, x2 = (frame(rng.normal(size=(110, 2))) for _ in range(2))
    y = 1.2 + 2.3 * x1 - .7 * x2 + frame(rng.normal(scale=.12, size=(110, 2)))
    state = frame(rng.normal(size=(110, 2)))
    return [y, x1, x2, state]


def run(name, frames, horizon=3):
    y, x1, x2, state = frames
    kw = dict(window=60, label_horizon=horizon)
    if "pcr" in name or "pls" in name:
        kw["n_components"] = 2
    if "regime" in name:
        kw.update(market_state=state, n_regimes=2)
    if "mixture" in name:
        kw.update(market_state=state, n_experts=2)
    return OperatorRegistry.get(name, "pandas_numpy", mode="any").calculate(y, x1, x2, **kw)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("horizon", [1, 5])
def test_newest_unavailable_label_and_future_features_cannot_change_query(name, horizon):
    frames = fixture()
    t = 92
    base = run(name, frames, horizon)
    assert np.isfinite(base.iloc[t]).all(), name
    changed = [f.copy() for f in frames]
    # Prior-fit policy: s + H <= t - 1. At s=t-H the label is
    # unavailable at that fit cutoff, even though it matures at decision t.
    changed[0].iloc[t-horizon:] = 1e9
    for f in changed[1:]:
        f.iloc[t+1:] = -1e8
    got = run(name, changed, horizon)
    np.testing.assert_allclose(got.iloc[:t+1], base.iloc[:t+1], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("name", NAMES)
def test_actual_public_prefix_and_window_replay(name):
    frames = fixture()
    base = run(name, frames)
    assert np.isfinite(base.iloc[90:]).all().all(), name
    prefix = run(name, [f.iloc[:94] for f in frames])
    pd.testing.assert_frame_equal(prefix, base.iloc[:94])
    # Each query uses at most 60 preceding physical rows. This tests window
    # replay only; it does not assert runtime checkpoint support.
    chunk = run(name, [f.iloc[30:] for f in frames])
    pd.testing.assert_frame_equal(chunk.loc[base.index[90:]], base.iloc[90:])


def test_pcr_matches_independent_matured_ols_and_includes_boundary_label():
    frames = fixture()
    t, h = 92, 5
    start, last = t - 60, t - 1 - h
    y, x1, x2, _ = frames
    X = np.column_stack([np.ones(last-start+1), x1.iloc[start:last+1, 0], x2.iloc[start:last+1, 0]])
    beta = np.linalg.lstsq(X, y.iloc[start:last+1, 0].to_numpy(), rcond=None)[0]
    expected = np.array([1, x1.iloc[t, 0], x2.iloc[t, 0]]) @ beta
    actual = run(NAMES[0], frames, h).iloc[t, 0]
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    changed = [f.copy() for f in frames]
    changed[0].iloc[last, 0] += 30
    influenced = run(NAMES[0], changed, h).iloc[t, 0]
    assert abs(influenced-actual) > 1e-3


@pytest.mark.parametrize("name", NAMES)
def test_future_only_member_is_not_in_historical_stock_models(name):
    frames = fixture()
    base = run(name, frames)
    expanded = []
    for f in frames:
        g = f.copy()
        g["NEW"] = np.nan
        g.loc[g.index[98:], "NEW"] = 1e6
        expanded.append(g)
    got = run(name, expanded)
    assert np.isfinite(base.iloc[92]).all(), name
    pd.testing.assert_frame_equal(got.loc[:, ["A", "B"]], base)
    assert got["NEW"].iloc[:98].isna().all()
