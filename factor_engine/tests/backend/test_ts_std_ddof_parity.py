# -*- coding: utf-8 -*-
"""Parity tests for ts_std ddof parameter between Pandas and Polars backends."""
import pytest
import pandas as pd
import polars as pl
import numpy as np


def test_ts_std_ddof_1_sample():
    """Test ts_std with ddof=1 (sample stddev) - default behavior."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # Pandas reference
    s_pd = pd.Series(data)
    expected = s_pd.rolling(3).std(ddof=1)

    # Polars implementation
    df_pl = pl.DataFrame({"value": data})
    result_pl = df_pl.select(pl.col("value").rolling_std(window_size=3, ddof=1))["value"].to_numpy()

    # Compare
    np.testing.assert_allclose(result_pl, expected.values, rtol=1e-10, equal_nan=True)


def test_ts_std_ddof_0_population():
    """Test ts_std with ddof=0 (population stddev)."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # Pandas reference
    s_pd = pd.Series(data)
    expected = s_pd.rolling(3).std(ddof=0)

    # Polars implementation
    df_pl = pl.DataFrame({"value": data})
    result_pl = df_pl.select(pl.col("value").rolling_std(window_size=3, ddof=0))["value"].to_numpy()

    # Compare
    np.testing.assert_allclose(result_pl, expected.values, rtol=1e-10, equal_nan=True)


def test_ts_std_with_nan():
    """Test ts_std handles NaN correctly."""
    data = [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, 8.0, 9.0, 10.0]

    # Pandas reference
    s_pd = pd.Series(data)
    expected = s_pd.rolling(4).std(ddof=1)

    # Polars implementation
    df_pl = pl.DataFrame({"value": data})
    result_pl = df_pl.select(pl.col("value").rolling_std(window_size=4, ddof=1))["value"].to_numpy()

    # Compare
    np.testing.assert_allclose(result_pl, expected.values, rtol=1e-10, equal_nan=True)


def test_ts_std_constant_values():
    """Test ts_std with all same values (should be 0.0 or NaN)."""
    data = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    # Pandas reference
    s_pd = pd.Series(data)
    expected = s_pd.rolling(3).std(ddof=1)

    # Polars implementation
    df_pl = pl.DataFrame({"value": data})
    result_pl = df_pl.select(pl.col("value").rolling_std(window_size=3, ddof=1))["value"].to_numpy()

    # Compare (should all be 0.0 after first window)
    np.testing.assert_allclose(result_pl, expected.values, rtol=1e-10, equal_nan=True)


def test_ts_std_min_periods_behavior():
    """Test ts_std with window=5 on small series."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    # Pandas reference with min_periods matching window
    s_pd = pd.Series(data)
    expected = s_pd.rolling(5, min_periods=5).std(ddof=1)

    # Polars implementation (min_samples=None means min_samples=window_size)
    df_pl = pl.DataFrame({"value": data})
    result_pl = df_pl.select(pl.col("value").rolling_std(window_size=5, ddof=1, min_samples=5))["value"].to_numpy()

    # Compare
    np.testing.assert_allclose(result_pl, expected.values, rtol=1e-10, equal_nan=True)


def test_ts_std_ddof_difference():
    """Verify ddof=0 and ddof=1 produce different results."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    df_pl = pl.DataFrame({"value": data})
    result_ddof0 = df_pl.select(pl.col("value").rolling_std(window_size=3, ddof=0))["value"].to_numpy()
    result_ddof1 = df_pl.select(pl.col("value").rolling_std(window_size=3, ddof=1))["value"].to_numpy()

    # They should be different (population vs sample)
    # For window=3: population_std * sqrt(3/2) = sample_std
    # So sample_std should be larger
    valid_mask = ~np.isnan(result_ddof0) & ~np.isnan(result_ddof1)
    assert np.all(result_ddof1[valid_mask] > result_ddof0[valid_mask])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
