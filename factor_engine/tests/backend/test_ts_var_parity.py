# -*- coding: utf-8 -*-
"""Focused parity tests for ts_var ddof parameter.

Tests verify that Polars backend ts_var matches Pandas backend
for ddof=0 (population variance), ddof=1 (sample variance), and edge cases.
"""

import pytest
import pandas as pd
import numpy as np

from factor_engine.api import ts_var
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


@pytest.fixture
def simple_source():
    """Create a simple panel source for focused testing."""
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=10, freq="D"), ["A"]],
        names=["timestamp", "instrument"]
    )
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=idx)
    return InMemorySeriesSource(data={"x": values})


def _run_ts_var(source, window, ddof=1, min_periods=1, backend="pandas"):
    """Helper to run ts_var on a source with specified parameters."""
    load_all()
    expr = ts_var(col("x"), window, ddof=ddof, min_periods=min_periods)
    eng = FactorEngine(backend=build_backend(backend), data_source=source)
    return eng.run(Factor(name="t", expr=expr))["result"]


class TestTsVarDdofParity:
    """Verify ddof parameter correctness for ts_var."""

    def test_ddof_0_population_variance(self, simple_source):
        """ddof=0 should compute population variance (divide by N)."""
        pd_result = _run_ts_var(simple_source, window=3, ddof=0, backend="pandas")
        pl_result = _run_ts_var(simple_source, window=3, ddof=0, backend="polars")

        pd.testing.assert_series_equal(
            pd_result, pl_result,
            check_names=False, rtol=1e-10, atol=1e-10
        )

        # Verify formula: for [1,2,3] with ddof=0: var = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 0.666...
        # Index 2 (third value) should have variance of first 3 elements
        expected_val = ((1-2)**2 + (2-2)**2 + (3-2)**2) / 3
        assert abs(pl_result.iloc[2] - expected_val) < 1e-10

    def test_ddof_1_sample_variance(self, simple_source):
        """ddof=1 should compute sample variance (divide by N-1, default)."""
        pd_result = _run_ts_var(simple_source, window=3, ddof=1, backend="pandas")
        pl_result = _run_ts_var(simple_source, window=3, ddof=1, backend="polars")

        pd.testing.assert_series_equal(
            pd_result, pl_result,
            check_names=False, rtol=1e-10, atol=1e-10
        )

        # Verify formula: for [1,2,3] with ddof=1: var = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 2 = 1.0
        expected_val = ((1-2)**2 + (2-2)**2 + (3-2)**2) / 2
        assert abs(pl_result.iloc[2] - expected_val) < 1e-10

    def test_min_periods_behavior(self, simple_source):
        """min_periods should control when variance starts being computed."""
        # With min_periods=3, first 2 values should be NaN
        pd_result = _run_ts_var(simple_source, window=3, ddof=1, min_periods=3, backend="pandas")
        pl_result = _run_ts_var(simple_source, window=3, ddof=1, min_periods=3, backend="polars")

        pd.testing.assert_series_equal(
            pd_result, pl_result,
            check_names=False, rtol=1e-10, atol=1e-10
        )

        # First 2 should be NaN when min_periods=3
        assert pd.isna(pl_result.iloc[0])
        assert pd.isna(pl_result.iloc[1])
        assert not pd.isna(pl_result.iloc[2])

    def test_nan_handling(self):
        """NaN values should be handled consistently (skipped in window)."""
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=8, freq="D"), ["A"]],
            names=["timestamp", "instrument"]
        )
        values = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0], index=idx)
        source = InMemorySeriesSource(data={"x": values})

        pd_result = _run_ts_var(source, window=3, ddof=1, backend="pandas")
        pl_result = _run_ts_var(source, window=3, ddof=1, backend="polars")

        pd.testing.assert_series_equal(
            pd_result, pl_result,
            check_names=False, rtol=1e-10, atol=1e-10
        )

    def test_all_zero_series(self):
        """All-zero input should produce zero variance (not NaN or inf)."""
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=6, freq="D"), ["A"]],
            names=["timestamp", "instrument"]
        )
        values = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], index=idx)
        source = InMemorySeriesSource(data={"x": values})

        pd_result = _run_ts_var(source, window=3, ddof=1, backend="pandas")
        pl_result = _run_ts_var(source, window=3, ddof=1, backend="polars")

        pd.testing.assert_series_equal(
            pd_result, pl_result,
            check_names=False, rtol=1e-10, atol=1e-10
        )

        # Non-NaN variance values should be exactly 0.0 (constant series has zero variance)
        non_nan = pl_result[~pl_result.isna()]
        assert (non_nan == 0.0).all()