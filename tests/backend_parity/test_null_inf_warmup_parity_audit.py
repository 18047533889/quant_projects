# -*- coding: utf-8
"""Systematic audit of Polars backend null/Inf/warmup/min_periods parity violations.

Mission: Identify specific divergences between Pandas reference and Polars implementation
in handling of:
- NULL/NaN propagation
- +/-Inf handling in operations
- Window warmup behavior (first N values)
- min_periods logic (when to emit NaN vs compute)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from runtime.engine import FactorEngine
from tests.helpers import InMemorySeriesSource


def _make_test_panel():
    """Create deterministic test panel with edge cases."""
    dates = pd.date_range("2024-01-01", periods=10)
    instruments = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["timestamp", "instrument"])

    n = len(idx)
    rng = np.random.RandomState(42)

    # Base series with controlled patterns
    values = rng.randn(n) * 10 + 100

    # Inject edge cases at specific positions
    values_array = values.copy()

    # NaN at position 3 (day 2, instrument A)
    values_array[3] = np.nan

    # Inf at position 12 (day 5, instrument A)
    values_array[12] = np.inf

    # -Inf at position 18 (day 7, instrument A)
    values_array[18] = -np.inf

    # Constant window for A on days 5-7 to trigger zero std
    values_array[15] = 100.0  # day 6, A
    values_array[18] = 100.0  # day 7, A (overwrite -inf)
    values_array[21] = 100.0  # day 8, A

    series = pd.Series(values_array, index=idx)

    return InMemorySeriesSource(data={"value": series})


@pytest.fixture(scope="module")
def panel():
    return _make_test_panel()


def _run_expr(source, expr, backend: str):
    """Run expression on given backend and return result dict."""
    return FactorEngine(
        backend=build_backend(backend),
        data_source=source,
        run_mode="research"
    ).run(Factor(name="parity_audit", expr=expr))


def _compare_results(pandas_result, polars_result, test_name: str) -> list[str]:
    """Compare pandas and polars results, return list of violations."""
    violations = []

    p_series = pandas_result["result"]
    pl_series = polars_result["result"]

    # Align indices
    p_series = p_series.sort_index()
    pl_series = pl_series.sort_index()

    # Check if polars used long path
    if not polars_result.get("used_polars_long_path"):
        violations.append(f"{test_name}: Polars did not use long path")

    # Compare values position by position
    for idx in p_series.index:
        if idx not in pl_series.index:
            violations.append(f"{test_name} @{idx}: Missing in Polars output")
            continue

        p_val = p_series.loc[idx]
        pl_val = pl_series.loc[idx]

        # Both NaN - OK
        if pd.isna(p_val) and pd.isna(pl_val):
            continue

        # Both Inf with same sign - OK
        if np.isinf(p_val) and np.isinf(pl_val) and np.sign(p_val) == np.sign(pl_val):
            continue

        # One is NaN, other is not - VIOLATION
        if pd.isna(p_val) != pd.isna(pl_val):
            violations.append(
                f"{test_name} @{idx}: NaN mismatch (pandas={p_val}, polars={pl_val})"
            )
            continue

        # One is Inf, other is not - VIOLATION
        if np.isinf(p_val) != np.isinf(pl_val):
            violations.append(
                f"{test_name} @{idx}: Inf mismatch (pandas={p_val}, polars={pl_val})"
            )
            continue

        # Finite values - check numerical tolerance
        if abs(p_val - pl_val) > 1e-6 * (1 + abs(p_val)):
            violations.append(
                f"{test_name} @{idx}: Value mismatch (pandas={p_val}, polars={pl_val})"
            )

    return violations


# Test 1: ts_mean with NaN in window
def test_ts_mean_nan_propagation(panel):
    """Verify NaN is skipped (not propagated) in ts_mean window."""
    expr = make_cleaned_call_factory("ts_mean")(col("value"), 3)

    pandas_out = _run_expr(panel, expr, "pandas")
    polars_out = _run_expr(panel, expr, "polars_long")

    violations = _compare_results(pandas_out, polars_out, "ts_mean_nan")
    assert not violations, "\n".join(violations)


# Test 2: ts_mean with Inf in window
def test_ts_mean_inf_handling(panel):
    """Verify Inf is treated as missing (pandas rolling drops Inf)."""
    expr = make_cleaned_call_factory("ts_mean")(col("value"), 3)

    pandas_out = _run_expr(panel, expr, "pandas")
    polars_out = _run_expr(panel, expr, "polars_long")

    violations = _compare_results(pandas_out, polars_out, "ts_mean_inf")
    assert not violations, "\n".join(violations)


# Test 3: ts_std with constant window (zero std)
def test_ts_std_zero_variance(panel):
    """Verify ts_std returns 0.0 (not NaN) for constant window."""
    expr = make_cleaned_call_factory("ts_std")(col("value"), 3)

    pandas_out = _run_expr(panel, expr, "pandas")
    polars_out = _run_expr(panel, expr, "polars_long")

    violations = _compare_results(pandas_out, polars_out, "ts_std_zero_var")
    assert not violations, "\n".join(violations)


# Test 4: ts_zscore with zero std
def test_ts_zscore_zero_std_handling(panel):
    """Verify ts_zscore returns 0.0 (not inf/NaN) when std=0."""
    expr = make_cleaned_call_factory("ts_zscore")(col("value"), 3)

    pandas_out = _run_expr(panel, expr, "pandas")
    polars_out = _run_expr(panel, expr, "polars_long")

    violations = _compare_results(pandas_out, polars_out, "ts_zscore_zero_std")

    # This is expected to have violations - document them
    if violations:
        print("\nDOCUMENTED VIOLATIONS for ts_zscore:")
        for v in violations:
            print(f"  - {v}")

    # For now, expect failure
    assert violations, "Expected ts_zscore parity violation but found none"


# Test 5: Warmup behavior with min_periods
def test_warmup_min_periods_boundary(panel):
    """Verify first N-1 values are NaN when min_periods not met."""
    expr = make_cleaned_call_factory("ts_mean")(col("value"), 5)

    pandas_out = _run_expr(panel, expr, "pandas")
    polars_out = _run_expr(panel, expr, "polars_long")

    violations = _compare_results(pandas_out, polars_out, "warmup_min_periods")
    assert not violations, "\n".join(violations)


# Test 6: Division operators with zero denominator
def test_divide_by_zero_handling(panel):
    """Verify division by zero produces appropriate result."""
    zero_series = panel.data["value"] * 0
    panel_with_zero = InMemorySeriesSource(data={"value": panel.data["value"], "zero": zero_series})

    from api.dsl import div
    expr = div(col("value"), col("zero"))

    pandas_out = _run_expr(panel_with_zero, expr, "pandas")
    polars_out = _run_expr(panel_with_zero, expr, "polars_long")

    violations = _compare_results(pandas_out, polars_out, "divide_by_zero")
    assert not violations, "\n".join(violations)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
