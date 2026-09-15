"""Exact two-point Pearson semantics for real DuckDB SQL execution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from factor_engine.backend.sql_pushdown.emitter import compile_plan_to_sql
from factor_engine.backend.sql_pushdown.plan_fixtures import column, literal
from factor_engine.planner.logical_plan import PlanNode


def _execute(x, y, *, window=2, min_periods=None):
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=len(x)),
            "inst": ["A"] * len(x),
            "x": x,
            "y": y,
        }
    )
    attrs = {} if min_periods is None else {"min_periods": min_periods}
    plan = PlanNode(
        op="ts_corr",
        inputs=[column("x"), column("y"), literal(window)],
        attrs=attrs,
    )
    compiled = compile_plan_to_sql(
        plan, dataset="panel", time_column="ts", instrument_column="inst"
    )
    assert compiled is not None
    con = duckdb.connect(":memory:")
    try:
        con.register("panel", frame)
        out = con.execute(compiled.query.replace("{{panel}}", "panel")).fetchdf()
    finally:
        con.close()
    return out.sort_values(["ts", "inst"])["value"].to_numpy()


def test_two_point_corr_is_exact_sign_with_large_offsets():
    got = _execute(
        [1e12, 1e12 + 1, 1e12 + 4, 1e12 + 5],
        [1e12, 1e12 - 3, 1e12 + 8, 1e12 + 9],
    )
    np.testing.assert_array_equal(got, [np.nan, -1.0, 1.0, 1.0])


@pytest.mark.parametrize(
    "x,y",
    [
        ([1.0, 1.0], [2.0, 3.0]),
        ([1.0, 2.0], [3.0, 3.0]),
        ([1.0, np.nan], [2.0, 3.0]),
        ([1.0, np.inf], [2.0, 3.0]),
        ([1.0, 2.0], [2.0, -np.inf]),
    ],
)
def test_two_point_corr_constant_or_nonfinite_pair_is_null(x, y):
    got = _execute(x, y)
    assert np.isnan(got[-1])


def test_window_one_cannot_form_a_pearson_pair():
    got = _execute([1.0, 2.0], [3.0, 4.0], window=1)
    assert np.isnan(got).all()


def test_explicit_min_periods_two_keeps_default_two_pair_semantics():
    got = _execute([1.0, 2.0], [3.0, 1.0], min_periods=2)
    np.testing.assert_array_equal(got, [np.nan, -1.0])
