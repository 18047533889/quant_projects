"""Precision regression tests for conditioned rolling-Sharpe windows."""
import numpy as np
import pytest

from quant_evaluator.metrics.portfolio_stats import _rolling_sharpe_per_window


def _finite_slice_oracle(returns, window, periods_per_year, min_periods):
    expected = np.full(returns.shape, np.nan, dtype=np.float64)
    for start in range(returns.size - window + 1):
        values = returns[start:start + window]
        values = values[np.isfinite(values)]
        if values.size < min_periods:
            continue
        std = np.std(values, ddof=1)
        if std > 1e-10:
            expected[start + window - 1] = (
                np.mean(values) / std * np.sqrt(periods_per_year)
            )
    return expected


@pytest.mark.parametrize("noise_scale", [1e-5, 5e-5, 1e-4])
@pytest.mark.parametrize("offset", [0.01, -0.02, 0.0, 1e6])
@pytest.mark.parametrize("window", [2, 20, 252])
def test_conditioned_returns_match_direct_finite_slice_oracle(noise_scale, offset, window):
    rng = np.random.default_rng(20261003)
    returns = offset + rng.normal(0.0, noise_scale, 3000)
    returns[::113] = np.nan
    returns[37::257] = np.inf
    min_periods = max(2, window - 2)
    actual = _rolling_sharpe_per_window(
        returns, window=window, periods_per_year=252, min_periods=min_periods,
    )
    expected = _finite_slice_oracle(returns, window, 252, min_periods)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10, equal_nan=True)


def test_near_flat_prefix_results_are_causal_under_future_append():
    rng = np.random.default_rng(20261003)
    returns = 0.01 + rng.normal(0.0, 1e-5, 3000)
    extended = np.concatenate((returns, [1e6]))
    actual = _rolling_sharpe_per_window(
        returns, window=20, periods_per_year=252, min_periods=20,
    )
    extended_actual = _rolling_sharpe_per_window(
        extended, window=20, periods_per_year=252, min_periods=20,
    )
    np.testing.assert_array_equal(actual, extended_actual[:returns.size])



def test_near_flat_prefix_is_unchanged_by_appended_high_offset_regime():
    rng = np.random.default_rng(20261004)
    prefix = 0.01 + rng.normal(0.0, 1e-5, 3000)
    tail = 25.0 + rng.normal(0.0, 0.1, 300)
    extended = np.concatenate((prefix, tail))
    actual = _rolling_sharpe_per_window(
        prefix, window=20, periods_per_year=252, min_periods=20,
    )
    extended_actual = _rolling_sharpe_per_window(
        extended, window=20, periods_per_year=252, min_periods=20,
    )
    np.testing.assert_array_equal(actual, extended_actual[:prefix.size])


def test_rejected_window_count_growth_does_not_change_prior_windows():
    rng = np.random.default_rng(902)
    prefix = 0.01 + rng.normal(0.0, 1e-5, 80)
    prefix[0] = 0.1
    prefix[[13, 29]] = np.nan
    tail = 0.01 + rng.normal(0.0, 1e-5, 300)
    extended = np.concatenate((prefix, tail))
    actual = _rolling_sharpe_per_window(
        prefix, window=20, periods_per_year=252, min_periods=18,
    )
    extended_actual = _rolling_sharpe_per_window(
        extended, window=20, periods_per_year=252, min_periods=18,
    )
    np.testing.assert_array_equal(actual, extended_actual[:prefix.size])
