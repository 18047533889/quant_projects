# -*- coding: utf-8 -*-
"""ts_cov operator parity tests across Pandas/Polars backends.

Tests ddof parameter, min_periods, NaN handling, and edge cases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")
import polars as pl

from api.cleaned_ops import make_cleaned_call_factory
from cleaned_operators import load_all


@pytest.fixture(scope="module")
def setup_operators():
    """Load all operators once for the module."""
    load_all()


def _pandas_ts_cov(x: pd.DataFrame, y: pd.DataFrame, window: int,
                   ddof: int = 1, min_periods: int | None = None) -> pd.DataFrame:
    """Call Pandas backend ts_cov."""
    from cleaned_operators.common.time_series import TSCov
    op = TSCov()
    return op._calculate_series(x, y, window=window, ddof=ddof, min_periods=min_periods)


def _polars_ts_cov(x: pd.DataFrame, y: pd.DataFrame, window: int,
                   ddof: int = 1, min_periods: int | None = None) -> pd.DataFrame:
    """Call Polars backend ts_cov and convert back to pandas."""
    from cleaned_operators.common.polars_ts_rolling import TSCovNative

    x_pl = pl.from_pandas(x.reset_index(drop=True))
    y_pl = pl.from_pandas(y.reset_index(drop=True))

    op = TSCovNative()
    result_pl = op._calculate_series(x_pl, y_pl, window=window, ddof=ddof, min_periods=min_periods)

    result_pd = result_pl.to_pandas()
    result_pd.index = x.index
    return result_pd


def test_ts_cov_ddof_0(setup_operators):
    """Test ts_cov with ddof=0 (population covariance)."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}, index=dates)
    y = pd.DataFrame({"A": [2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 8.0, 9.0, 11.0, 12.0]}, index=dates)

    window = 5
    ddof = 0

    pandas_result = _pandas_ts_cov(x, y, window=window, ddof=ddof)
    polars_result = _polars_ts_cov(x, y, window=window, ddof=ddof)

    # Check shape
    assert pandas_result.shape == polars_result.shape

    # Check parity (allow small floating point error)
    np.testing.assert_allclose(
        pandas_result.values,
        polars_result.values,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True
    )


def test_ts_cov_ddof_1(setup_operators):
    """Test ts_cov with ddof=1 (sample covariance, default)."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}, index=dates)
    y = pd.DataFrame({"A": [2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 8.0, 9.0, 11.0, 12.0]}, index=dates)

    window = 5
    ddof = 1

    pandas_result = _pandas_ts_cov(x, y, window=window, ddof=ddof)
    polars_result = _polars_ts_cov(x, y, window=window, ddof=ddof)

    # Check shape
    assert pandas_result.shape == polars_result.shape

    # Check parity
    np.testing.assert_allclose(
        pandas_result.values,
        polars_result.values,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True
    )

    # Verify ddof=1 values are larger than ddof=0 (dividing by smaller denominator)
    pandas_result_ddof0 = _pandas_ts_cov(x, y, window=window, ddof=0)
    # For window=5, ddof=1: divide by 4, ddof=0: divide by 5
    # So ddof=1 result should be 5/4 = 1.25x larger
    ratio = pandas_result.values[4, 0] / pandas_result_ddof0.values[4, 0]
    assert abs(ratio - 1.25) < 0.01


def test_ts_cov_min_periods(setup_operators):
    """Test ts_cov with custom min_periods."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}, index=dates)
    y = pd.DataFrame({"A": [2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 8.0, 9.0, 11.0, 12.0]}, index=dates)

    window = 5
    min_periods = 4

    pandas_result = _pandas_ts_cov(x, y, window=window, min_periods=min_periods)
    polars_result = _polars_ts_cov(x, y, window=window, min_periods=min_periods)

    # Check shape
    assert pandas_result.shape == polars_result.shape

    # Check parity
    np.testing.assert_allclose(
        pandas_result.values,
        polars_result.values,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True
    )

    # First value with min_periods=4 should be at index 3 (0-indexed)
    assert pd.notna(pandas_result.values[3, 0])
    assert pd.isna(pandas_result.values[2, 0])


def test_ts_cov_nan_handling(setup_operators):
    """Test ts_cov with NaN values - should skip NaN pairs."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    x = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0]}, index=dates)
    y = pd.DataFrame({"A": [2.0, np.nan, 5.0, 4.0, 5.0, 7.0, 8.0, 9.0, 11.0, 12.0]}, index=dates)

    window = 5
    min_periods = 2

    pandas_result = _pandas_ts_cov(x, y, window=window, min_periods=min_periods)
    polars_result = _polars_ts_cov(x, y, window=window, min_periods=min_periods)

    # Check shape
    assert pandas_result.shape == polars_result.shape

    # Check parity
    np.testing.assert_allclose(
        pandas_result.values,
        polars_result.values,
        rtol=1e-9,
        atol=1e-9,
        equal_nan=True
    )

    # Window at index 4 has 5 values but some NaN pairs
    # Should still compute with valid pairs only
    assert pd.notna(pandas_result.values[4, 0])


def test_ts_cov_constant_series(setup_operators):
    """Test ts_cov with constant series - covariance should be zero."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    x = pd.DataFrame({"A": [5.0] * 10}, index=dates)
    y = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}, index=dates)

    window = 5

    pandas_result = _pandas_ts_cov(x, y, window=window)
    polars_result = _polars_ts_cov(x, y, window=window)

    # Check shape
    assert pandas_result.shape == polars_result.shape

    # Check parity
    np.testing.assert_allclose(
        pandas_result.values,
        polars_result.values,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True
    )

    # Covariance with constant series should be zero
    valid_values = pandas_result.values[~np.isnan(pandas_result.values)]
    np.testing.assert_allclose(valid_values, 0.0, atol=1e-10)


def test_ts_cov_multi_column(setup_operators):
    """Test ts_cov with multiple columns."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    x = pd.DataFrame({
        "A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "B": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    }, index=dates)
    y = pd.DataFrame({
        "A": [2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 8.0, 9.0, 11.0, 12.0],
        "B": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5]
    }, index=dates)

    window = 5

    pandas_result = _pandas_ts_cov(x, y, window=window)
    polars_result = _polars_ts_cov(x, y, window=window)

    # Check shape
    assert pandas_result.shape == polars_result.shape
    assert pandas_result.shape[1] == 2  # Both columns

    # Check parity for both columns
    np.testing.assert_allclose(
        pandas_result.values,
        polars_result.values,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
