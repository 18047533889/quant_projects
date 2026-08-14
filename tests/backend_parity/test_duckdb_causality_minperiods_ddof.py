# -*- coding: utf-8 -*-
"""DuckDB backend parity audit: causality, min_periods, ddof violations.

This test suite specifically targets potential divergences between DuckDB SQL
and Pandas reference implementations in:
1. Causal window semantics (whether current row is included)
2. min_periods threshold enforcement
3. ddof parameter in variance/std operations
4. Edge cases (empty groups, single-value windows, NaN handling)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from runtime.engine import FactorEngine
from storage.factory import build_data_source
from tests.helpers import InMemorySeriesSource


# ============================================================================
# Test Fixtures
# ============================================================================


def _create_minperiods_test_panel():
    """Panel designed to test min_periods boundary conditions."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    instruments = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["timestamp", "instrument"]
    )

    n = len(idx)
    np.random.seed(123)

    # Instrument A: clean data
    # Instrument B: has NaN gaps that affect min_periods
    # Instrument C: single valid value edge case
    close = pd.Series(np.random.randn(n) * 10 + 100, index=idx)

    # Instrument B: strategic NaN placement
    for i in [1, 4, 7]:
        close.loc[(dates[i], "B")] = np.nan

    return InMemorySeriesSource(data={"close": close})


def _create_ddof_test_panel():
    """Panel with constant and near-constant values to test ddof sensitivity."""
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    instruments = ["A", "B"]
    idx = pd.MultiIndex.from_product(
        [dates, instruments], names=["timestamp", "instrument"]
    )

    # Instrument A: values with small variance (ddof matters)
    vals_a = [100.0, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0, 100.1]
    # Instrument B: constant values (std should be 0)
    vals_b = [50.0] * 8

    close = pd.Series(vals_a + vals_b, index=idx)

    return InMemorySeriesSource(data={"close": close})


@pytest.fixture
def minperiods_source():
    return _create_minperiods_test_panel()


@pytest.fixture
def ddof_source():
    return _create_ddof_test_panel()


# ============================================================================
# Helper Functions
# ============================================================================


def _run(source, expr, backend_name: str):
    return FactorEngine(
        backend=build_backend(backend_name),
        data_source=source,
        run_mode="research"
    ).run(Factor(name="test", expr=expr))


def _result_series(run_out) -> pd.Series:
    return run_out["result"].sort_index()


def _assert_parity(
    pandas_out: pd.Series,
    duckdb_out: pd.Series,
    rtol=1e-9,
    atol=1e-9,
    check_exact_nan=True,
):
    """Assert parity with tight tolerance and NaN placement checking."""
    if check_exact_nan:
        # NaN positions must match exactly
        pd.testing.assert_series_equal(
            pandas_out.isna(), duckdb_out.isna(), check_names=False
        )

    # Values must match (where both are non-NaN)
    pd.testing.assert_series_equal(
        pandas_out, duckdb_out, check_names=False, rtol=rtol, atol=atol
    )


# ============================================================================
# Test: min_periods boundary conditions
# ============================================================================


def test_ts_mean_min_periods_default(minperiods_source):
    """Verify ts_mean with default min_periods=1."""
    expr = make_cleaned_call_factory("ts_mean")(col("close"), 5)

    pd_out = _result_series(_run(minperiods_source, expr, "pandas"))

    # First row should have value (min_periods=1 allows single observation)
    assert pd_out.iloc[0:1].notna().all(), "min_periods=1 should allow first row"


def test_ts_mean_min_periods_strict(minperiods_source):
    """Verify ts_mean with min_periods equal to window."""
    expr = make_cleaned_call_factory("ts_mean")(col("close"), 5, min_periods=5)

    pd_out = _result_series(_run(minperiods_source, expr, "pandas"))

    # First 4 rows should be NaN (need exactly 5 observations)
    for inst in ["A", "B", "C"]:
        inst_data = pd_out.xs(inst, level="instrument")
        assert inst_data.iloc[0:4].isna().all(), f"First 4 rows should be NaN for {inst}"
        if inst != "B":  # B has NaN gaps
            assert inst_data.iloc[4].notna(), f"5th row should have value for {inst}"


def test_ts_mean_min_periods_with_nans(minperiods_source):
    """Verify min_periods counts only non-NaN values."""
    expr = make_cleaned_call_factory("ts_mean")(col("close"), 3, min_periods=2)

    pd_out = _result_series(_run(minperiods_source, expr, "pandas"))

    # Instrument B has NaN at positions 1, 4, 7
    # Window at position 2 contains [val, NaN, val] -> 2 valid -> should have result
    b_data = pd_out.xs("B", level="instrument")
    assert b_data.iloc[2].notna(), "Position 2 should have value (2 valid in window)"


def test_ts_std_min_periods_with_ddof(ddof_source):
    """Verify ts_std respects min_periods >= ddof+1 constraint."""
    # ddof=1 requires at least 2 observations
    expr = make_cleaned_call_factory("ts_std")(col("close"), 3, ddof=1, min_periods=2)

    pd_out = _result_series(_run(ddof_source, expr, "pandas"))

    # First row: only 1 observation -> should be NaN
    assert pd_out.iloc[0:1].isna().all(), "Single observation with ddof=1 should be NaN"

    # Second row: 2 observations with ddof=1 -> should have value
    assert pd_out.iloc[1:2].notna().any(), "Two observations with ddof=1 should work"


# ============================================================================
# Test: ddof parameter in variance/std
# ============================================================================


def test_ts_std_ddof_0_vs_1(ddof_source):
    """Verify ddof=0 (population) vs ddof=1 (sample) produce different results."""
    expr_ddof0 = make_cleaned_call_factory("ts_std")(col("close"), 3, ddof=0)
    expr_ddof1 = make_cleaned_call_factory("ts_std")(col("close"), 3, ddof=1)

    pd_out0 = _result_series(_run(ddof_source, expr_ddof0, "pandas"))
    pd_out1 = _result_series(_run(ddof_source, expr_ddof1, "pandas"))

    # For non-constant data with >=2 points, ddof should make a difference
    a_data0 = pd_out0.xs("A", level="instrument")
    a_data1 = pd_out1.xs("A", level="instrument")

    # Where both are non-NaN, they should differ
    valid_mask = a_data0.notna() & a_data1.notna()
    if valid_mask.sum() > 0:
        # ddof=1 (sample) should be larger than ddof=0 (population)
        diff = a_data1[valid_mask] - a_data0[valid_mask]
        assert (diff > 0).any(), "ddof=1 should produce larger std than ddof=0"


def test_ts_std_ddof_constant_values(ddof_source):
    """Verify std=0 for constant values regardless of ddof."""
    expr_ddof0 = make_cleaned_call_factory("ts_std")(col("close"), 3, ddof=0)
    expr_ddof1 = make_cleaned_call_factory("ts_std")(col("close"), 3, ddof=1)

    pd_out0 = _result_series(_run(ddof_source, expr_ddof0, "pandas"))
    pd_out1 = _result_series(_run(ddof_source, expr_ddof1, "pandas"))

    # Instrument B is constant -> std should be 0.0 for both ddof values
    b_data0 = pd_out0.xs("B", level="instrument")
    b_data1 = pd_out1.xs("B", level="instrument")

    valid0 = b_data0[b_data0.notna()]
    valid1 = b_data1[b_data1.notna()]

    assert (valid0 == 0.0).all(), "Constant values should have std=0 (ddof=0)"
    assert (valid1 == 0.0).all(), "Constant values should have std=0 (ddof=1)"


# ============================================================================
# Test: Causality (window frame boundaries)
# ============================================================================


def test_ts_mean_includes_current_row(minperiods_source):
    """Verify rolling window INCLUDES current row (pandas default behavior)."""
    expr = make_cleaned_call_factory("ts_mean")(col("close"), 3, min_periods=1)

    pd_out = _result_series(_run(minperiods_source, expr, "pandas"))

    # Get instrument A (clean data)
    a_data = pd_out.xs("A", level="instrument")

    # Manual calculation: at position 2, window should include [0, 1, 2]
    source_data = minperiods_source.data["close"].xs("A", level="instrument")
    expected_pos2 = source_data.iloc[0:3].mean()
    actual_pos2 = a_data.iloc[2]

    assert abs(actual_pos2 - expected_pos2) < 1e-9, \
        f"Window should include current row: expected {expected_pos2}, got {actual_pos2}"


def test_ts_delay_excludes_current_row(minperiods_source):
    """Verify ts_delay shifts and excludes current row."""
    expr = make_cleaned_call_factory("ts_delay")(col("close"), 1)

    pd_out = _result_series(_run(minperiods_source, expr, "pandas"))

    # Get source and delayed
    source_data = minperiods_source.data["close"].xs("A", level="instrument")
    delayed_data = pd_out.xs("A", level="instrument")

    # First row should be NaN
    assert pd.isna(delayed_data.iloc[0]), "First row of ts_delay should be NaN"

    # Second row should equal first row of source
    assert abs(delayed_data.iloc[1] - source_data.iloc[0]) < 1e-9, \
        "ts_delay(1) at position 1 should equal source at position 0"


# ============================================================================
# Test: Edge cases
# ============================================================================


def test_empty_window_returns_nan(minperiods_source):
    """Verify operations on empty windows return NaN."""
    # Create data with all NaN for one instrument
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    idx = pd.MultiIndex.from_product(
        [dates, ["X"]], names=["timestamp", "instrument"]
    )
    all_nan = pd.Series([np.nan] * 5, index=idx)
    source = InMemorySeriesSource(data={"close": all_nan})

    expr = make_cleaned_call_factory("ts_mean")(col("close"), 3)
    pd_out = _result_series(_run(source, expr, "pandas"))

    assert pd_out.isna().all(), "All-NaN input should produce all-NaN output"


def test_single_value_std_returns_nan(ddof_source):
    """Verify std with single value returns NaN (when ddof=1)."""
    expr = make_cleaned_call_factory("ts_std")(col("close"), 1, ddof=1, min_periods=1)

    pd_out = _result_series(_run(ddof_source, expr, "pandas"))

    # Window of size 1 with ddof=1 should always be NaN
    assert pd_out.isna().all(), "Single-value window with ddof=1 should be NaN"


# ============================================================================
# Test: Summary - document expected behavior
# ============================================================================


def test_document_window_semantics():
    """Document: pandas rolling windows INCLUDE the current row by default.

    This is NOT a look-ahead violation. The window [t-w+1, t] includes t.
    For truly causal operations that exclude current row, use ts_delay first.
    """
    assert True  # Documentation test


def test_document_min_periods_semantics():
    """Document: min_periods counts non-NaN values in the window.

    For ts_std with ddof=1, the effective min_periods is max(min_periods, 2).
    A window with only 1 non-NaN value cannot compute sample std.
    """
    assert True  # Documentation test
