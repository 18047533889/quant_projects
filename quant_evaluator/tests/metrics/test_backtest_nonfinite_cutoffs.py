"""Regression checks for finite-only portfolio quantile cutoffs."""

import numpy as np
import pytest

from quant_evaluator.metrics.portfolio_stats import (
    _compute_long_short_returns_reference,
    compute_long_short_returns,
)


@pytest.mark.parametrize("tie_policy", ["min", "max"])
@pytest.mark.parametrize("missing_return_policy", ["zero_fill", "drop", "fail"])
def test_nonfinite_factors_match_reference(tie_policy, missing_return_policy):
    factors = np.array([
        [-np.inf, 1., 2., 3., 4., 5., np.inf, np.nan],
        [np.inf, np.nan, 2., 2., 2., 3., 4., -np.inf],
        [np.nan, np.inf, -np.inf, np.nan, np.inf, -np.inf, np.nan, np.inf],
    ])
    returns = np.array([
        [.99, .01, .02, .03, .04, .05, .98, .97],
        [.96, .95, .02, .02, .02, .03, .04, .94],
        [.01, .02, .03, .04, .05, .06, .07, .08],
    ])
    got = compute_long_short_returns(
        factors, returns, tie_policy=tie_policy,
        missing_return_policy=missing_return_policy,
    )
    expected = _compute_long_short_returns_reference(
        factors, returns, tie_policy=tie_policy,
        missing_return_policy=missing_return_policy,
    )
    for actual, oracle in zip(got, expected):
        np.testing.assert_allclose(actual, oracle, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("tie_policy", ["min", "max"])
@pytest.mark.parametrize("missing_return_policy", ["zero_fill", "drop"])
def test_missing_selected_returns_match_reference(tie_policy, missing_return_policy):
    factors = np.array([[-np.inf, 1., 2., 3., 4., 5., np.inf, np.nan]])
    returns = np.array([[.99, np.nan, .02, .03, .04, np.nan, .98, .97]])
    got = compute_long_short_returns(
        factors, returns, tie_policy=tie_policy,
        missing_return_policy=missing_return_policy,
    )
    expected = _compute_long_short_returns_reference(
        factors, returns, tie_policy=tie_policy,
        missing_return_policy=missing_return_policy,
    )
    for actual, oracle in zip(got, expected):
        np.testing.assert_allclose(actual, oracle, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_fail_policy_rejects_missing_return_with_nonfinite_factors():
    factors = np.array([[-np.inf, 1., 2., 3., 4., 5., np.inf, np.nan]])
    returns = np.array([[.99, np.nan, .02, .03, .04, .05, .98, .97]])
    with pytest.raises(ValueError, match="non-finite"):
        compute_long_short_returns(factors, returns, missing_return_policy="fail")
    with pytest.raises(ValueError, match="non-finite"):
        _compute_long_short_returns_reference(factors, returns, missing_return_policy="fail")


def test_all_nonfinite_row_is_nan_without_invalid_warning():
    factors = np.array([[-np.inf, np.inf, np.nan, -np.inf]])
    returns = np.array([[.01, .02, .03, .04]])
    with np.errstate(invalid="raise"):
        result = compute_long_short_returns(factors, returns)
    for series in result:
        assert np.isnan(series[0])
