# -*- coding: utf-8 -*-
"""DuckDB SQL parity for ts_skew and ts_kurt: min_periods, Inf handling.

Audit findings:
- ts_skew missing min_periods validation (requires ≥3 points)
- ts_skew missing ±Inf handling (pandas propagates Inf → NaN)
- ts_kurt missing min_periods validation (requires ≥4 points)
- ts_kurt had Inf check but missing min_periods enforcement

This test exercises:
1. Minimum data point requirements (skew=3, kurt=4)
2. min_periods parameter binding
3. ±Inf propagation to NaN
4. NaN handling in windows
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")
import duckdb

from backend.sql_pushdown.emitter import (
    compile_plan_to_sql,
    reset_sql_template_cache,
)
from backend.sql_pushdown.plan_fixtures import column, literal
from planner.logical_plan import PlanNode


def _execute(compiled, frame: pd.DataFrame) -> pd.Series:
    """Execute compiled SQL against DuckDB."""
    con = duckdb.connect(":memory:")
    try:
        con.register("panel", frame)
        query = compiled.query.replace("{{panel}}", "panel")
        out = con.execute(query).df()
    finally:
        con.close()
    return out.set_index(["ts", "inst"])["value"]


@pytest.mark.parametrize("window,min_periods", [(5, 5), (5, 3), (4, 4), (4, 2)])
def test_ts_skew_min_periods_enforcement(window: int, min_periods: int) -> None:
    """ts_skew must enforce min_periods and minimum 3 data points."""
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=8),
            "inst": ["A"] * 8,
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    reset_sql_template_cache()

    plan = PlanNode(
        op="ts_skew",
        inputs=[column("x"), literal(window)],
        attrs={"min_periods": min_periods},
    )
    compiled = compile_plan_to_sql(
        plan, dataset="panel", time_column="ts", instrument_column="inst"
    )
    assert compiled is not None

    # Pandas reference
    expected = frame["x"].rolling(window, min_periods=min_periods).skew()
    got = _execute(compiled, frame).reset_index(drop=True)

    # Skew requires minimum 3 points; min_periods below that still → NaN
    effective_min = max(min_periods, 3)
    for i, (g, e) in enumerate(zip(got, expected)):
        if i < effective_min - 1:
            assert pd.isna(g), f"idx={i}: expected NaN, got {g}"
        else:
            np.testing.assert_allclose(g, e, equal_nan=True, atol=1e-10)


@pytest.mark.parametrize("window,min_periods", [(6, 6), (6, 4), (5, 5), (5, 3)])
def test_ts_kurt_min_periods_enforcement(window: int, min_periods: int) -> None:
    """ts_kurt must enforce min_periods and minimum 4 data points."""
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=10),
            "inst": ["A"] * 10,
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    reset_sql_template_cache()

    plan = PlanNode(
        op="ts_kurt",
        inputs=[column("x"), literal(window)],
        attrs={"min_periods": min_periods},
    )
    compiled = compile_plan_to_sql(
        plan, dataset="panel", time_column="ts", instrument_column="inst"
    )
    assert compiled is not None

    # Pandas reference
    expected = frame["x"].rolling(window, min_periods=min_periods).kurt()
    got = _execute(compiled, frame).reset_index(drop=True)

    # Kurt requires minimum 4 points
    effective_min = max(min_periods, 4)
    for i, (g, e) in enumerate(zip(got, expected)):
        if i < effective_min - 1:
            assert pd.isna(g), f"idx={i}: expected NaN, got {g}"
        else:
            np.testing.assert_allclose(g, e, equal_nan=True, atol=1e-10)


def test_ts_skew_inf_handling() -> None:
    """ts_skew: pandas rolling.skew() skips Inf (not propagate to NaN).

    Note: pandas.Series([1,2,inf]).skew() → NaN, but
    pandas.Series([1,2,3,inf]).rolling(4).skew().iloc[3] → skew([1,2,3])
    because rolling uses optimized code that silently skips Inf.
    We match this behavior for parity.
    """
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=10),
            "inst": ["A"] * 10,
            "x": [1.0, 2.0, 3.0, np.inf, 5.0, 6.0, 7.0, -np.inf, 9.0, 10.0],
        }
    )
    reset_sql_template_cache()

    plan = PlanNode(
        op="ts_skew",
        inputs=[column("x"), literal(4)],
        attrs={"min_periods": 3},
    )
    compiled = compile_plan_to_sql(
        plan, dataset="panel", time_column="ts", instrument_column="inst"
    )
    assert compiled is not None

    # Pandas rolling skips Inf values
    expected = frame["x"].rolling(4, min_periods=3).skew()
    got = _execute(compiled, frame).reset_index(drop=True)

    np.testing.assert_allclose(got, expected, equal_nan=True, atol=1e-10)


def test_ts_kurt_inf_handling() -> None:
    """ts_kurt: pandas rolling.kurt() skips Inf (not propagate to NaN).

    Same behavior as skew - rolling uses optimized code that silently skips Inf.
    """
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=10),
            "inst": ["A"] * 10,
            "x": [1.0, 2.0, 3.0, 4.0, np.inf, 6.0, 7.0, 8.0, -np.inf, 10.0],
        }
    )
    reset_sql_template_cache()

    plan = PlanNode(
        op="ts_kurt",
        inputs=[column("x"), literal(5)],
        attrs={"min_periods": 4},
    )
    compiled = compile_plan_to_sql(
        plan, dataset="panel", time_column="ts", instrument_column="inst"
    )
    assert compiled is not None

    # Pandas rolling skips Inf values
    expected = frame["x"].rolling(5, min_periods=4).kurt()
    got = _execute(compiled, frame).reset_index(drop=True)

    np.testing.assert_allclose(got, expected, equal_nan=True, atol=1e-10)


def test_ts_skew_nan_handling() -> None:
    """ts_skew with NaN: COUNT excludes NULL, min_periods enforced."""
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=8),
            "inst": ["A"] * 8,
            "x": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0],
        }
    )
    reset_sql_template_cache()

    plan = PlanNode(
        op="ts_skew",
        inputs=[column("x"), literal(4)],
        attrs={"min_periods": 3},
    )
    compiled = compile_plan_to_sql(
        plan, dataset="panel", time_column="ts", instrument_column="inst"
    )
    assert compiled is not None

    expected = frame["x"].rolling(4, min_periods=3).skew()
    got = _execute(compiled, frame).reset_index(drop=True)

    np.testing.assert_allclose(got, expected, equal_nan=True, atol=1e-10)


def test_ts_kurt_nan_handling() -> None:
    """ts_kurt with NaN: COUNT excludes NULL, min_periods enforced."""
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=10),
            "inst": ["A"] * 10,
            "x": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],
        }
    )
    reset_sql_template_cache()

    plan = PlanNode(
        op="ts_kurt",
        inputs=[column("x"), literal(5)],
        attrs={"min_periods": 4},
    )
    compiled = compile_plan_to_sql(
        plan, dataset="panel", time_column="ts", instrument_column="inst"
    )
    assert compiled is not None

    expected = frame["x"].rolling(5, min_periods=4).kurt()
    got = _execute(compiled, frame).reset_index(drop=True)

    np.testing.assert_allclose(got, expected, equal_nan=True, atol=1e-10)
