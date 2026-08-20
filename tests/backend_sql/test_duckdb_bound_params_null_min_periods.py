# -*- coding: utf-8 -*-
"""Bounded DuckDB compile/execute parity checks for literals and windows."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")
import duckdb

from backend.sql_pushdown.emitter import (
    SqlPushdownFilter,
    compile_plan_to_sql,
    reset_sql_template_cache,
)
from backend.sql_pushdown.plan_fixtures import column, literal
from planner.logical_plan import PlanNode


def _execute(compiled, frame: pd.DataFrame) -> pd.Series:
    con = duckdb.connect(":memory:")
    try:
        con.register("panel", frame)
        query = compiled.query.replace("{{panel}}", "panel")
        out = con.execute(query).df()
    finally:
        con.close()
    return out.set_index(["ts", "inst"])["value"]


def test_duckdb_rebinds_literal_window_and_min_periods_with_nulls() -> None:
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=8),
            "inst": ["A"] * 8,
            "x": [1.0, np.nan, 3.0, np.inf, 5.0, 6.0, -np.inf, 8.0],
        }
    )
    reset_sql_template_cache()

    for window, min_periods in ((3, 2), (2, 1)):
        plan = PlanNode(
            op="ts_mean",
            inputs=[column("x"), literal(window)],
            attrs={"min_periods": min_periods},
        )
        compiled = compile_plan_to_sql(
            plan, dataset="panel", time_column="ts", instrument_column="inst"
        )
        assert compiled is not None
        got = _execute(compiled, frame).reset_index(drop=True)
        expected = (
            frame["x"]
            .replace([np.inf, -np.inf], np.nan)
            .rolling(window, min_periods=min_periods)
            .mean()
        )
        np.testing.assert_allclose(got, expected, equal_nan=True)


def test_template_cache_binds_filter_column_names() -> None:
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=3),
            "event_ts": pd.date_range("2024-02-01", periods=3),
            "inst": ["A", "A", "A"],
            "asset": ["B", "B", "B"],
            "x": [1.0, 2.0, 3.0],
        }
    )
    plan = PlanNode(op="add", inputs=[column("x"), literal(1.0)], attrs={})
    reset_sql_template_cache()

    first = compile_plan_to_sql(
        plan,
        dataset="panel",
        time_column="ts",
        instrument_column="inst",
        filt=SqlPushdownFilter(
            time_column="ts", instrument_column="inst", start="2024-01-02"
        ),
    )
    second = compile_plan_to_sql(
        plan,
        dataset="panel",
        time_column="ts",
        instrument_column="inst",
        filt=SqlPushdownFilter(
            time_column="event_ts",
            instrument_column="asset",
            start="2024-02-02",
        ),
    )
    assert first is not None and second is not None
    assert '"event_ts" >= \'2024-02-02\'' in second.query
    assert '"ts" >= \'2024-01-02\'' not in second.query
    got = _execute(second, frame)
    assert got.tolist() == [3.0, 4.0]
