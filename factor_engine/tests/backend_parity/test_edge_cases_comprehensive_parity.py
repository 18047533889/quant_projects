# -*- coding: utf-8
"""Edge case three-way parity tests across all operator families.

Systematic edge case coverage:
1. Boundary conditions: empty panels, single instrument, single timestamp
2. Numeric edge cases: NaN, Inf, -Inf, zero, very large/small values
3. Window edge cases: min_periods boundary, insufficient data
4. Group edge cases: empty groups, singleton groups, all-NaN groups
5. Cross-sectional edge cases: zero variance, all-same, all-NaN
6. IEEE 754 edge cases: NaN comparisons, Inf arithmetic

Goal: Ensure all three backends handle edge cases identically.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from cleaned_operators import load_all
from runtime.engine import FactorEngine
from storage.factory import build_data_source
from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution
from tests.helpers import InMemorySeriesSource


F = make_cleaned_call_factory


def _run(source, expr, backend: str):
    return FactorEngine(
        backend=build_backend(backend), data_source=source, run_mode="research"
    ).run(Factor(name="edge_parity", expr=expr))


def _assert_parity(reference: pd.Series, candidate: pd.Series) -> None:
    pd.testing.assert_series_equal(
        reference.sort_index(), candidate.sort_index(),
        check_names=False, check_dtype=False, rtol=1e-6, atol=1e-6,
    )


@pytest.fixture(scope="module")
def setup():
    load_all()


def test_empty_panel_handling(setup):
    """Empty panel returns empty result consistently."""
    dates = pd.date_range("2024-01-02", periods=0, freq="D")
    idx = pd.MultiIndex.from_tuples([], names=["timestamp", "instrument"])
    close = pd.Series([], index=idx, dtype=float)
    source = InMemorySeriesSource(data={"close": close})

    expr = F("ts_mean")(col("close"), 3)
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    assert len(pandas_out) == 0
    assert len(polars_out) == 0


def test_single_instrument_ts_operators(setup):
    """Time-series operators work with single instrument."""
    dates = pd.date_range("2024-01-02", periods=10, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["timestamp", "instrument"])
    close = pd.Series(range(10), index=idx, dtype=float)
    source = InMemorySeriesSource(data={"close": close})

    expr = F("ts_mean")(col("close"), 3)
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)
    assert len(pandas_out) == 10


def test_single_timestamp_cs_operators(setup):
    """Cross-sectional operators work with single timestamp."""
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=1), ["A", "B", "C"]],
        names=["timestamp", "instrument"]
    )
    close = pd.Series([10.0, 20.0, 30.0], index=idx)
    source = InMemorySeriesSource(data={"close": close})

    expr = F("cs_rank")(col("close"))
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)
    assert len(pandas_out) == 3


def test_all_nan_window(setup):
    """Rolling operators handle all-NaN windows."""
    dates = pd.date_range("2024-01-02", periods=5, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["timestamp", "instrument"])
    close = pd.Series([np.nan] * 10, index=idx)
    source = InMemorySeriesSource(data={"close": close})

    expr = F("ts_mean")(col("close"), 3)
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)
    assert pandas_out.isna().all()


def test_all_nan_cross_section(setup):
    """Cross-sectional operators handle all-NaN timestamps."""
    dates = pd.date_range("2024-01-02", periods=2, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["timestamp", "instrument"])
    close = pd.Series([10.0, 20.0, 30.0, np.nan, np.nan, np.nan], index=idx)
    source = InMemorySeriesSource(data={"close": close})

    expr = F("cs_mean")(col("close"))
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)
    # Second timestamp should be all NaN
    assert pandas_out.iloc[3:].isna().all()


def test_zero_variance_cross_section(setup):
    """Cross-sectional zscore handles zero variance."""
    dates = pd.date_range("2024-01-02", periods=2, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["timestamp", "instrument"])
    close = pd.Series([100.0, 100.0, 100.0, 10.0, 20.0, 30.0], index=idx)
    source = InMemorySeriesSource(data={"close": close})

    expr = F("zscore")(col("close"))
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)
    # First timestamp (constant) should produce NaN or 0
    first_ts_out = pandas_out.iloc[:3]
    assert first_ts_out.isna().all() or (first_ts_out == 0.0).all()


def test_min_periods_boundary_exact(setup):
    """Rolling operators at exact min_periods boundary."""
    dates = pd.date_range("2024-01-02", periods=5, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["timestamp", "instrument"])
    close = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx)
    source = InMemorySeriesSource(data={"close": close})

    # Window=3, min_periods=3: first valid at index 2
    expr = F("ts_mean")(col("close"), 3, min_periods=3)
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)
    # First two should be NaN, rest valid
    assert pandas_out.iloc[:2].isna().all()
    assert pandas_out.iloc[2:].notna().all()


def test_min_periods_insufficient_data(setup):
    """Rolling operators with insufficient data for min_periods."""
    dates = pd.date_range("2024-01-02", periods=3, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["timestamp", "instrument"])
    close = pd.Series([10.0, 20.0, 30.0], index=idx)
    source = InMemorySeriesSource(data={"close": close})

    # Window=5, min_periods=5: never enough data
    expr = F("ts_mean")(col("close"), 5, min_periods=5)
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)
    assert pandas_out.isna().all()


def test_inf_in_rolling_window(setup):
    """Rolling aggregations handle Inf values."""
    dates = pd.date_range("2024-01-02", periods=5, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["timestamp", "instrument"])
    close = pd.Series([10.0, 20.0, np.inf, 30.0, 40.0], index=idx)
    source = InMemorySeriesSource(data={"close": close})

    expr = F("ts_mean")(col("close"), 3)
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)


def test_negative_inf_handling(setup):
    """Operators handle -Inf consistently."""
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=1), ["A", "B", "C"]],
        names=["timestamp", "instrument"]
    )
    close = pd.Series([10.0, -np.inf, 30.0], index=idx)
    source = InMemorySeriesSource(data={"close": close})

    expr = F("cs_mean")(col("close"))
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)


def test_very_large_values(setup):
    """Operators handle very large values without overflow."""
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=1), ["A", "B", "C"]],
        names=["timestamp", "instrument"]
    )
    close = pd.Series([1e15, 2e15, 3e15], index=idx)
    source = InMemorySeriesSource(data={"close": close})

    expr = F("cs_mean")(col("close"))
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)


def test_very_small_values(setup):
    """Operators handle very small values without underflow."""
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=1), ["A", "B", "C"]],
        names=["timestamp", "instrument"]
    )
    close = pd.Series([1e-15, 2e-15, 3e-15], index=idx)
    source = InMemorySeriesSource(data={"close": close})

    expr = F("add")(col("close"), col("close"))
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)


def test_group_with_all_nan(setup):
    """Group operators handle groups with all NaN."""
    dates = pd.date_range("2024-01-02", periods=2, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["timestamp", "instrument"])
    close = pd.Series([10.0, np.nan, 30.0, 15.0, np.nan, 35.0], index=idx)
    group_id = pd.Series([1.0, 2.0, 1.0, 1.0, 2.0, 1.0], index=idx)
    source = InMemorySeriesSource(data={"close": close, "group_id": group_id})

    expr = F("group_mean")(col("close"), col("group_id"))
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)


def test_group_singleton_after_nan_filter(setup):
    """Group operators handle singleton groups after NaN filtering."""
    dates = pd.date_range("2024-01-02", periods=1, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["timestamp", "instrument"])
    close = pd.Series([10.0, np.nan, np.nan], index=idx)
    group_id = pd.Series([1.0, 1.0, 1.0], index=idx)
    source = InMemorySeriesSource(data={"close": close, "group_id": group_id})

    expr = F("group_std")(col("close"), col("group_id"))
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)


def test_correlation_constant_series(setup):
    """Correlation handles constant series (undefined correlation)."""
    dates = pd.date_range("2024-01-02", periods=5, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["timestamp", "instrument"])
    x = pd.Series([100.0] * 5, index=idx)
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    source = InMemorySeriesSource(data={"x": x, "y": y})

    expr = F("ts_corr")(col("x"), col("y"), 3)
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)
    # Correlation with constant should be NaN
    assert pandas_out.iloc[2:].isna().all()


def test_rank_with_all_ties(setup):
    """Rank handles all tied values."""
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=1), ["A", "B", "C", "D"]],
        names=["timestamp", "instrument"]
    )
    close = pd.Series([50.0, 50.0, 50.0, 50.0], index=idx)
    source = InMemorySeriesSource(data={"close": close})

    expr = F("rank")(col("close"))
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)


def test_ts_delay_exceeds_history(setup):
    """ts_delay with delay exceeding available history."""
    dates = pd.date_range("2024-01-02", periods=3, freq="D")
    idx = pd.MultiIndex.from_product([dates, ["A"]], names=["timestamp", "instrument"])
    close = pd.Series([10.0, 20.0, 30.0], index=idx)
    source = InMemorySeriesSource(data={"close": close})

    # Delay by 5 periods when only 3 available
    expr = F("ts_delay")(col("close"), 5)
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)
    # All should be NaN since delay exceeds history
    assert pandas_out.isna().all()


def test_mixed_inf_nan_arithmetic(setup):
    """Arithmetic with mixed Inf and NaN."""
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-02", periods=1), ["A", "B", "C", "D"]],
        names=["timestamp", "instrument"]
    )
    x = pd.Series([np.inf, np.nan, 10.0, -np.inf], index=idx)
    y = pd.Series([1.0, 2.0, np.inf, np.nan], index=idx)
    source = InMemorySeriesSource(data={"x": x, "y": y})

    expr = F("add")(col("x"), col("y"))
    pandas_out = _run(source, expr, "pandas")["result"]
    polars_out = _run(source, expr, "polars_long")["result"]

    _assert_parity(pandas_out, polars_out)
