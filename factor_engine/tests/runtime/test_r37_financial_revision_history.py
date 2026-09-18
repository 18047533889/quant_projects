import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.fundamental.expectation_v2 import (
    fin_expectation_revision, fin_expectation_revision_count,
    fin_expectation_revision_magnitude, fin_expectation_revision_pct,
    fin_expectation_revision_speed,
)
from factor_engine.cleaned_operators.fundamental.transforms_repairs_v2 import (
    fin_restated_flag, fin_revision_count, fin_revision_delta,
    fin_revision_direction, fin_revision_magnitude, fin_revision_pct,
)
from factor_engine.runtime.execution_contract import forward_impact, own_history_requirement

LAGGED = (
    "fin_revision_delta", "fin_revision_pct", "fin_revision_direction",
    "fin_expectation_revision", "fin_expectation_revision_pct",
)
EXPECTATION_WINDOWS = (
    "fin_expectation_revision_speed", "fin_expectation_revision_count",
    "fin_expectation_revision_magnitude",
)
REVISION_WINDOWS = (
    "fin_revision_count", "fin_revision_magnitude", "fin_restated_flag",
)


@pytest.mark.parametrize("canonical", LAGGED)
def test_revision_lag_contract_is_one_row(canonical):
    assert own_history_requirement(canonical).rows == 1
    assert forward_impact(canonical) == 1


@pytest.mark.parametrize("canonical", EXPECTATION_WINDOWS)
def test_expectation_window_contract_default_keyword_and_unknown(canonical):
    assert own_history_requirement(canonical).rows == 60
    assert forward_impact(canonical) == 60
    assert own_history_requirement(canonical, {"window_days": 10}).rows == 10
    assert forward_impact(canonical, {"window_days": 10}) == 10
    assert own_history_requirement(canonical, {"window_days": 2.5}).is_full_history
    assert forward_impact(canonical, {"window_days": 2.5}) is None


@pytest.mark.parametrize("canonical", REVISION_WINDOWS)
def test_revision_window_contract_default_keyword_and_unknown(canonical):
    assert own_history_requirement(canonical).rows == 252
    assert forward_impact(canonical) == 252
    params = {"window_days": 10, "coverage_threshold": 0.8}
    assert own_history_requirement(canonical, params).rows == 10
    assert forward_impact(canonical, params) == 10
    params["window_days"] = 2.5
    assert own_history_requirement(canonical, params).is_full_history
    assert forward_impact(canonical, params) is None


def test_lag_kernels_preserve_gap_and_period_transition():
    index = pd.date_range("2025-01-01", periods=5)
    x = pd.DataFrame({"a": [10.0, 12.0, np.nan, 15.0, 20.0]}, index=index)
    period = pd.DataFrame({"a": [1, 1, 1, 1, 2]}, index=index)
    delta = np.array([0.0, 2.0, np.nan, np.nan, 0.0])
    pct = np.array([0.0, 0.2, np.nan, np.nan, 0.0])
    for out in (fin_revision_delta(x, period), fin_expectation_revision(x, period)):
        np.testing.assert_allclose(out["a"], delta, equal_nan=True)
    for out in (fin_revision_pct(x, period), fin_expectation_revision_pct(x, period)):
        np.testing.assert_allclose(out["a"], pct, equal_nan=True)
    np.testing.assert_allclose(
        fin_revision_direction(x, period)["a"], np.sign(delta), equal_nan=True
    )


@pytest.mark.parametrize("window", [3, 10])
def test_windows_need_exact_overlap_and_match_independent_oracles(window):
    rows, suffix = 40, 25
    index = pd.date_range("2025-02-01", periods=rows)
    values = 10.0 + np.arange(rows, dtype=float)
    x = pd.DataFrame({"a": values}, index=index)
    period = pd.DataFrame({"a": np.ones(rows)}, index=index)
    full = (
        fin_revision_count(x, period, window_days=window, coverage_threshold=0.8),
        fin_expectation_revision_count(x, period, window_days=window),
        fin_expectation_revision_speed(x, period, window_days=window),
        fin_revision_magnitude(x, period, window_days=window, coverage_threshold=0.8),
        fin_expectation_revision_magnitude(x, period, window_days=window),
        fin_restated_flag(x, period, window_days=window, coverage_threshold=0.8),
    )
    start = suffix - window
    sx, sp = x.iloc[start:], period.iloc[start:]
    exact = (
        fin_revision_count(sx, sp, window_days=window, coverage_threshold=0.8),
        fin_expectation_revision_count(sx, sp, window_days=window),
        fin_expectation_revision_speed(sx, sp, window_days=window),
        fin_revision_magnitude(sx, sp, window_days=window, coverage_threshold=0.8),
        fin_expectation_revision_magnitude(sx, sp, window_days=window),
        fin_restated_flag(sx, sp, window_days=window, coverage_threshold=0.8),
    )
    for chunk, whole in zip(exact, full):
        np.testing.assert_allclose(
            chunk.loc[index[suffix]:], whole.loc[index[suffix]:], equal_nan=True
        )
    pct = values[suffix-window+1:suffix+1] / values[suffix-window:suffix] - 1.0
    expected = (window, window, pct.sum(), np.abs(pct).sum(), np.abs(pct).sum(), 1.0)
    for result, oracle in zip(full, expected):
        assert result.iloc[suffix, 0] == pytest.approx(oracle)
    short = fin_revision_count(
        x.iloc[suffix-window+1:], period.iloc[suffix-window+1:],
        window_days=window, coverage_threshold=0.8,
    )
    assert short.loc[index[suffix], "a"] == pytest.approx(window - 1)
