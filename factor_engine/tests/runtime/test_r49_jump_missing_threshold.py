"""Jump thresholds use finite return support without compressing slot positions."""
import numpy as np
import pandas as pd
import polars as pl
from factor_engine.cleaned_operators.intraday.higher_moments import _jump_mask
from factor_engine.cleaned_operators.intraday.polars_intraday_full import _jump_mask_long


def test_nonfinite_return_is_not_a_jump():
    returns = np.array([.01, -.02, .03, .1, np.nan, np.inf, -np.inf])
    actual = _jump_mask(returns, 1.)
    finite = np.isfinite(returns)
    threshold = np.sqrt(np.mean(returns[finite]**2))
    expected = finite & (np.abs(returns) > threshold)
    np.testing.assert_array_equal(actual, expected)


def test_missing_slots_do_not_lower_threshold_by_inflating_support():
    returns = np.array([.01,.07,.01,.15,.01,-.16,.01,.17,.01])
    close = np.r_[100*np.exp(np.r_[0., returns.cumsum()]), np.full(24, np.nan)]
    timestamps = pd.date_range("2024-01-03 09:30", periods=len(close), freq="min")
    panel = pl.DataFrame({"QuoteTime": timestamps, "A": close})
    result = _jump_mask_long(panel, 1.).sort("ts")
    raw = np.r_[np.nan, np.diff(np.log(close))]
    finite = np.isfinite(raw)
    expected = finite & (np.abs(raw) > np.sqrt(np.mean(raw[finite]**2)))
    np.testing.assert_array_equal(result["jump"].to_numpy(), expected)
    np.testing.assert_array_equal(_jump_mask(raw, 1.), expected)
    assert len(result) == len(close)
