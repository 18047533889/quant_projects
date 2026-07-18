# -*- coding: utf-8 -*-
"""DuckDB regression tests for audited fiscal-period SQL semantics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from planner.logical_plan import PlanNode


@pytest.fixture(scope="module", autouse=True)
def _load_registry() -> None:
    load_all()


def _execute_period_lag(panel: pd.DataFrame, *, periods=1, policy="latest_available"):
    duckdb = pytest.importorskip("duckdb")
    from backend.sql_pushdown.emitter import compile_plan_to_sql

    plan = PlanNode(
        "period_lag",
        [
            PlanNode("column", attrs={"name": "x"}),
            PlanNode("column", attrs={"name": "pid"}),
        ],
        attrs={"periods": periods, "revision_policy": policy},
    )
    compiled = compile_plan_to_sql(
        plan,
        dataset="panel",
        time_column="ts",
        instrument_column="inst",
    )
    connection = duckdb.connect()
    connection.register("panel", panel)
    return (
        connection.execute(compiled.query.replace("{{panel}}", "panel"))
        .df()
        .sort_values(["inst", "ts"])["value"]
        .to_numpy()
    )


def _pandas_period_lag(panel: pd.DataFrame, *, periods=1, policy="latest_available"):
    values = pd.DataFrame({"A": panel["x"].to_numpy()})
    period_ids = pd.DataFrame({"A": panel["pid"].to_numpy()})
    return OperatorRegistry.get("period_lag", "pandas_numpy").calculate(
        values,
        period_ids,
        periods,
        policy,
    )["A"].to_numpy()


def test_duckdb_period_lag_does_not_jump_across_missing_quarters():
    panel = pd.DataFrame(
        {
            "ts": [1, 2, 3],
            "inst": ["A", "A", "A"],
            "x": [10.0, 30.0, 40.0],
            "pid": ["2024Q1", "2024Q3", "2024Q4"],
        }
    )
    actual = _execute_period_lag(panel, periods=1)
    expected = _pandas_period_lag(panel, periods=1)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert np.isnan(actual[1])
    assert actual[2] == 30.0


def test_duckdb_period_lag_honours_revision_policy():
    panel = pd.DataFrame(
        {
            "ts": [1, 2, 3, 4],
            "inst": ["A"] * 4,
            "x": [10.0, 20.0, 22.0, 30.0],
            "pid": ["2024Q1", "2024Q2", "2024Q2", "2024Q3"],
        }
    )
    latest = _execute_period_lag(panel, policy="latest_available")
    first = _execute_period_lag(panel, policy="first_available")
    np.testing.assert_allclose(
        latest,
        _pandas_period_lag(panel, policy="latest_available"),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        first,
        _pandas_period_lag(panel, policy="first_available"),
        equal_nan=True,
    )
    assert latest[-1] == 22.0
    assert first[-1] == 20.0


def test_duckdb_period_parser_matches_mixed_encodings():
    panel = pd.DataFrame(
        {
            "ts": [1, 2, 3, 4],
            "inst": ["A"] * 4,
            "x": [10.0, 20.0, 30.0, 40.0],
            "pid": ["202401", "2024Q2", "20240930", "2024-12-31"],
        }
    )
    actual = _execute_period_lag(panel)
    expected = _pandas_period_lag(panel)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    np.testing.assert_allclose(
        actual,
        np.array([np.nan, 10.0, 20.0, 30.0]),
        equal_nan=True,
    )


def test_duckdb_period_lag_rejects_fractional_periods():
    panel = pd.DataFrame(
        {
            "ts": [1],
            "inst": ["A"],
            "x": [10.0],
            "pid": ["2024Q1"],
        }
    )
    with pytest.raises(ValueError):
        _execute_period_lag(panel, periods=1.5)
