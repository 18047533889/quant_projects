import numpy as np
import pytest

from quant_evaluator.metrics.portfolio_stats import (
    _rolling_sharpe_per_window,
    compute_calmar_ratio,
    compute_sortino_ratio,
)


@pytest.mark.parametrize("denominator", ["negative", "all"])
def test_sortino_no_downside_is_nan_without_divide_warnings(denominator):
    returns = np.full((30, 2), 0.01, dtype=np.float64)
    returns[:, 1] = np.linspace(-0.02, 0.03, 30)

    with np.errstate(divide="raise", invalid="raise"):
        actual = compute_sortino_ratio(
            returns, min_periods=20, downside_denominator=denominator,
        )

    assert np.isnan(actual[0])
    assert np.isfinite(actual[1])


def test_calmar_monotonic_returns_is_nan_without_dividing_by_zero():
    returns = np.full((30, 1), 0.01, dtype=np.float64)

    with np.errstate(divide="raise", invalid="raise"):
        actual = compute_calmar_ratio(returns, min_periods=20)

    assert np.isnan(actual[0])


def test_rolling_sharpe_flat_windows_are_nan_and_mixed_windows_stay_finite():
    returns = np.array([0.01, 0.01, 0.01, 0.01, 0.02, 0.03], dtype=np.float64)

    with np.errstate(divide="raise", invalid="raise"):
        actual = _rolling_sharpe_per_window(
            returns, window=3, periods_per_year=252, min_periods=3,
        )

    assert np.isnan(actual[:4]).all()
    assert np.isfinite(actual[4:]).all()


def test_rolling_sharpe_cancellation_fallback_matches_finite_window_oracle():
    returns = 1e12 + np.arange(9, dtype=np.float64) * 0.25
    returns[2] = np.nan
    window_size = 5
    min_periods = 4

    actual = _rolling_sharpe_per_window(
        returns, window=window_size, periods_per_year=252,
        min_periods=min_periods,
    )
    expected = np.full(returns.shape, np.nan)
    for start in range(returns.size - window_size + 1):
        values = returns[start:start + window_size]
        values = values[np.isfinite(values)]
        if values.size >= min_periods:
            std = np.std(values, ddof=1)
            if std > 1e-10:
                expected[start + window_size - 1] = (
                    np.mean(values) / std * np.sqrt(252)
                )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0.0, equal_nan=True)


def test_rolling_sharpe_long_flat_series_has_no_finite_windows():
    returns = np.full(3000, 0.01, dtype=np.float64)

    actual = _rolling_sharpe_per_window(
        returns, window=20, periods_per_year=252, min_periods=20,
    )

    assert np.isnan(actual).all()


def test_rolling_sharpe_flat_tail_after_volatile_prefix_stays_nan():
    prefix_size = 2500
    window_size = 32
    volatile_prefix = np.resize(
        np.array([0.08, -0.06, 0.04, -0.03], dtype=np.float64), prefix_size,
    )
    returns = np.concatenate((volatile_prefix, np.full(400, 0.01)))

    actual = _rolling_sharpe_per_window(
        returns, window=window_size, periods_per_year=252,
        min_periods=window_size,
    )

    flat_tail_start = prefix_size + window_size - 1
    assert np.isnan(actual[flat_tail_start:]).all()


def test_rolling_sharpe_missing_near_constant_large_offset_matches_window_oracle():
    returns = 1e12 + np.arange(257, dtype=np.float64) * 0.25
    returns[[3, 18, 71, 72, 190]] = np.nan
    returns[[109, 220]] = np.inf
    returns[150] = -np.inf
    window_size = 19
    min_periods = 16

    actual = _rolling_sharpe_per_window(
        returns, window=window_size, periods_per_year=252,
        min_periods=min_periods,
    )
    expected = np.full(returns.shape, np.nan)
    for start in range(returns.size - window_size + 1):
        values = returns[start:start + window_size]
        values = values[np.isfinite(values)]
        if values.size >= min_periods:
            std = np.std(values, ddof=1)
            if std > 1e-10:
                expected[start + window_size - 1] = (
                    np.mean(values) / std * np.sqrt(252)
                )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0.0, equal_nan=True)
