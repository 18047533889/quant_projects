# -*- coding: utf-8 -*-
"""ts_corr operator parity tests across Pandas/Polars backends.

Tests:
1. Basic correlation computation
2. min_periods parameter handling
3. NaN handling (skip NaN pairs)
4. Perfect correlation (1.0)
5. Anti-correlation (-1.0)
6. Result range [-1, 1]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.api import ts_corr
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture
def simple_source():
    """Create a simple panel source for testing."""
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=10, freq="D"), ["A"]],
        names=["timestamp", "instrument"]
    )
    x_values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=idx)
    y_values = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0], index=idx)
    return InMemorySeriesSource(data={"x": x_values, "y": y_values})


@pytest.fixture
def nan_source():
    """Create a panel source with NaN values."""
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=10, freq="D"), ["A"]],
        names=["timestamp", "instrument"]
    )
    x_values = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=idx)
    y_values = pd.Series([2.0, 4.0, np.nan, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0], index=idx)
    return InMemorySeriesSource(data={"x": x_values, "y": y_values})


@pytest.fixture
def anti_corr_source():
    """Create a panel source with negative correlation."""
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=10, freq="D"), ["A"]],
        names=["timestamp", "instrument"]
    )
    x_values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=idx)
    y_values = pd.Series([20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0], index=idx)
    return InMemorySeriesSource(data={"x": x_values, "y": y_values})


def _run_ts_corr(source, window, min_periods=2, backend="pandas"):
    """Helper to run ts_corr on a source with specified parameters."""
    load_all()
    expr = ts_corr(col("x"), col("y"), window, min_periods=min_periods)
    eng = FactorEngine(backend=build_backend(backend), data_source=source, run_mode="research")
    return eng.run(Factor(name="test", expr=expr))["result"]


class TestTsCorrParity:
    """Verify ts_corr parity across backends."""

    def test_basic_perfect_correlation(self, simple_source):
        """Test perfect positive correlation (y = 2*x)."""
        pd_result = _run_ts_corr(simple_source, window=5, backend="pandas")
        pl_result = _run_ts_corr(simple_source, window=5, backend="polars")

        pd.testing.assert_series_equal(
            pd_result, pl_result,
            check_names=False, rtol=1e-9, atol=1e-10
        )

        # Should have perfect correlation (1.0) for all valid windows
        valid = pd_result[pd_result.notna()]
        assert len(valid) >= 6
        np.testing.assert_allclose(valid, 1.0, atol=1e-10)

    def test_min_periods_default(self, simple_source):
        """Test default min_periods=2 behavior."""
        pd_result = _run_ts_corr(simple_source, window=5, min_periods=2, backend="pandas")
        pl_result = _run_ts_corr(simple_source, window=5, min_periods=2, backend="polars")

        pd.testing.assert_series_equal(
            pd_result, pl_result,
            check_names=False, rtol=1e-9, atol=1e-10
        )

        # With min_periods=2, should get results starting from index 1
        assert pd_result.notna().sum() >= 9

    def test_min_periods_high(self, simple_source):
        """Test high min_periods value."""
        pd_result = _run_ts_corr(simple_source, window=5, min_periods=5, backend="pandas")
        pl_result = _run_ts_corr(simple_source, window=5, min_periods=5, backend="polars")

        pd.testing.assert_series_equal(
            pd_result, pl_result,
            check_names=False, rtol=1e-9, atol=1e-10
        )

        # With min_periods=5, should get results starting from index 4
        assert pd.isna(pd_result.iloc[0])
        assert pd.isna(pd_result.iloc[3])
        assert not pd.isna(pd_result.iloc[4])

    def test_nan_handling(self, nan_source):
        """Test NaN handling - skip NaN pairs."""
        pd_result = _run_ts_corr(nan_source, window=5, min_periods=2, backend="pandas")
        pl_result = _run_ts_corr(nan_source, window=5, min_periods=2, backend="polars")

        pd.testing.assert_series_equal(
            pd_result, pl_result,
            check_names=False, rtol=1e-9, atol=1e-10
        )

        # Should skip NaN pairs and compute with valid pairs
        valid = pd_result[pd_result.notna()]
        assert len(valid) >= 5

        # Correlation results should be in [-1, 1] range (allow 1e-12 for fp rounding)
        vals = valid.values
        assert np.all(vals >= -1.0 - 1e-12) and np.all(vals <= 1.0 + 1e-12)

    @pytest.mark.parametrize("use_numba", [False, True])
    def test_inf_handling_matches_finite_pair_contract(self, monkeypatch, use_numba):
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=5, freq="D"), ["A"]],
            names=["timestamp", "instrument"],
        )
        x_values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        y_values = pd.Series([2.0, np.inf, 6.0, 8.0, 10.0], index=idx)
        source = InMemorySeriesSource(data={"x": x_values, "y": y_values})
        if use_numba:
            monkeypatch.setenv("FACTOR_ENGINE_USE_NUMBA", "1")
        else:
            monkeypatch.delenv("FACTOR_ENGINE_USE_NUMBA", raising=False)

        pd_result = _run_ts_corr(source, window=3, backend="pandas")
        pl_result = _run_ts_corr(source, window=3, backend="polars")

        pd.testing.assert_series_equal(
            pd_result, pl_result, check_names=False, rtol=1e-9, atol=1e-10
        )
        # The Inf pair is skipped: windows ending at rows 2, 3, and 4
        # still contain two finite pairs and are perfectly correlated.
        np.testing.assert_allclose(
            pd_result.to_numpy(), [np.nan, np.nan, 1.0, 1.0, 1.0], equal_nan=True
        )

    def test_anti_correlation(self, anti_corr_source):
        """Test perfect negative correlation."""
        pd_result = _run_ts_corr(anti_corr_source, window=5, backend="pandas")
        pl_result = _run_ts_corr(anti_corr_source, window=5, backend="polars")

        pd.testing.assert_series_equal(
            pd_result, pl_result,
            check_names=False, rtol=1e-9, atol=1e-10
        )

        # Should have perfect negative correlation (-1.0)
        valid = pd_result[pd_result.notna()]
        assert len(valid) >= 6
        np.testing.assert_allclose(valid, -1.0, atol=1e-10)

    def test_result_range(self, simple_source):
        """Test that results are always in [-1, 1] range."""
        pd_result = _run_ts_corr(simple_source, window=5, backend="pandas")
        pl_result = _run_ts_corr(simple_source, window=5, backend="polars")

        # Check pandas results (allow 1e-12 for floating-point rounding)
        valid_pd = pd_result[pd_result.notna()]
        vals_pd = valid_pd.values
        assert np.all(vals_pd >= -1.0 - 1e-12) and np.all(vals_pd <= 1.0 + 1e-12)

        # Check polars results
        valid_pl = pl_result[pl_result.notna()]
        vals_pl = valid_pl.values
        assert np.all(vals_pl >= -1.0 - 1e-12) and np.all(vals_pl <= 1.0 + 1e-12)

    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=1, freq="D"), ["A"]],
            names=["timestamp", "instrument"]
        )
        x_values = pd.Series([1.0], index=idx)
        y_values = pd.Series([2.0], index=idx)
        source = InMemorySeriesSource(data={"x": x_values, "y": y_values})

        pd_result = _run_ts_corr(source, window=5, min_periods=2, backend="pandas")
        pl_result = _run_ts_corr(source, window=5, min_periods=2, backend="polars")

        # Both should return NaN when insufficient data
        assert pd.isna(pd_result.iloc[0])
        assert pd.isna(pl_result.iloc[0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
