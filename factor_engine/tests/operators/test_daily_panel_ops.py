# -*- coding: utf-8 -*-
"""Shape-preserving daily operators: semantics, backend parity and causality."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.sql_pushdown.emitter import compile_plan_to_sql, plan_is_sql_capable
from cleaned_operators import load_all
from cleaned_operators.registry import OperatorRegistry
from planner.logical_plan import PlanNode


OPS = (
    "ts_count_if",
    "ts_sum_if",
    "ts_mean_if",
    "ts_std_if",
    "ts_last_if",
    "ts_days_since",
    "ts_true_streak",
    "cs_bucket",
    "cs_multi_resid",
    "cs_wls_resid",
    "period_lag",
    "ts_regression_tstat",
    "ts_trend_tstat",
    "ts_max_drawdown",
    "ts_partial_corr",
    "ts_nth_value",
)


@pytest.fixture(scope="module", autouse=True)
def _load() -> None:
    load_all()


@pytest.fixture()
def panels() -> dict[str, pd.DataFrame]:
    index = pd.Index(range(8), name="ts")
    columns = pd.Index(list("ABCDE"), name="inst")
    row = np.arange(8, dtype=float)[:, None]
    col = np.arange(5, dtype=float)[None, :]
    x = (
        10.0
        + row * 0.7
        + row**2 * 0.025
        + col * 1.3
        + (row * col) * 0.04
        + col**2 * 0.11
    )
    z = 3.0 + row**2 * 0.09 + col**2 * 0.23 + row * col * 0.05
    y = 1.4 * x - 0.35 * z + np.sin(row + col) * 0.4
    condition = ((row + col) % 3 != 1).astype(float)
    condition[2, 1] = np.nan
    condition[5, 3] = np.nan
    x[3, 2] = np.nan
    price = 20.0 + row + col
    price[4:, 1] -= 4.0
    price[6:, 3] -= 7.0
    period = np.repeat(np.array([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0], [4.0], [4.0]]), 5, axis=1)
    weight = np.repeat(np.arange(1.0, 6.0)[None, :], 8, axis=0)
    return {
        "x": pd.DataFrame(x, index=index, columns=columns),
        "y": pd.DataFrame(y, index=index, columns=columns),
        "z": pd.DataFrame(z, index=index, columns=columns),
        "condition": pd.DataFrame(condition, index=index, columns=columns),
        "price": pd.DataFrame(price, index=index, columns=columns),
        "period": pd.DataFrame(period, index=index, columns=columns),
        "weight": pd.DataFrame(weight, index=index, columns=columns),
    }


def _cases(p: dict[str, pd.DataFrame]):
    return {
        "ts_count_if": ([p["condition"], 4, 2], {}),
        "ts_sum_if": ([p["x"], p["condition"], 4, 1], {}),
        "ts_mean_if": ([p["x"], p["condition"], 4, 1], {}),
        "ts_std_if": ([p["x"], p["condition"], 4, 2, 1], {}),
        "ts_last_if": ([p["x"], p["condition"], 4], {}),
        "ts_days_since": ([p["condition"], 4], {}),
        "ts_true_streak": ([p["condition"]], {}),
        "cs_bucket": ([p["x"], 3, True], {}),
        "cs_multi_resid": ([p["y"], p["x"], p["z"]], {"min_obs": 4}),
        "cs_wls_resid": ([p["y"], p["x"], p["weight"]], {"min_obs": 4}),
        "period_lag": ([p["x"], p["period"], 1], {}),
        "ts_regression_tstat": ([p["y"], p["x"], 5], {"min_periods": 3}),
        "ts_trend_tstat": ([p["x"], 5], {"min_periods": 3}),
        "ts_max_drawdown": ([p["price"], 5, 2], {}),
        "ts_partial_corr": ([p["x"], p["y"], p["z"], 5], {"min_periods": 3}),
        "ts_nth_value": ([p["x"], 5, 2, "largest"], {"min_periods": 2}),
    }


def _col(name: str) -> PlanNode:
    return PlanNode(op="column", attrs={"name": name})


def _lit(value) -> PlanNode:
    return PlanNode(op="literal", attrs={"value": value})


def _plans() -> dict[str, PlanNode]:
    return {
        "ts_count_if": PlanNode("ts_count_if", [_col("condition"), _lit(4), _lit(2)]),
        "ts_sum_if": PlanNode("ts_sum_if", [_col("x"), _col("condition"), _lit(4), _lit(1)]),
        "ts_mean_if": PlanNode("ts_mean_if", [_col("x"), _col("condition"), _lit(4), _lit(1)]),
        "ts_std_if": PlanNode("ts_std_if", [_col("x"), _col("condition"), _lit(4), _lit(2), _lit(1)]),
        "ts_last_if": PlanNode("ts_last_if", [_col("x"), _col("condition"), _lit(4)]),
        "ts_days_since": PlanNode("ts_days_since", [_col("condition"), _lit(4)]),
        "ts_true_streak": PlanNode("ts_true_streak", [_col("condition")]),
        "cs_bucket": PlanNode("cs_bucket", [_col("x"), _lit(3), _lit(True)]),
        "cs_multi_resid": PlanNode("cs_multi_resid", [_col("y"), _col("x"), _col("z")], {"min_obs": 4}),
        "cs_wls_resid": PlanNode("cs_wls_resid", [_col("y"), _col("x"), _col("weight")], {"min_obs": 4}),
        "period_lag": PlanNode("period_lag", [_col("x"), _col("period"), _lit(1)]),
        "ts_regression_tstat": PlanNode("ts_regression_tstat", [_col("y"), _col("x"), _lit(5)], {"min_periods": 3}),
        "ts_trend_tstat": PlanNode("ts_trend_tstat", [_col("x"), _lit(5)], {"min_periods": 3}),
        "ts_max_drawdown": PlanNode("ts_max_drawdown", [_col("price"), _lit(5), _lit(2)]),
        "ts_partial_corr": PlanNode("ts_partial_corr", [_col("x"), _col("y"), _col("z"), _lit(5)], {"min_periods": 3}),
        "ts_nth_value": PlanNode("ts_nth_value", [_col("x"), _lit(5), _lit(2), _lit("largest")], {"min_periods": 2}),
    }


def test_condition_and_event_semantics() -> None:
    idx = pd.RangeIndex(6)
    x = pd.DataFrame({"A": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0]}, index=idx)
    cond = pd.DataFrame({"A": [1.0, 0.0, np.nan, 1.0, 0.0, 1.0]}, index=idx)
    count = OperatorRegistry.get("ts_count_if").calculate(cond, 3, 2)
    assert np.isnan(count["A"].iloc[0])
    assert count["A"].iloc[1] == 1.0
    assert count["A"].iloc[-1] == 2.0
    summed = OperatorRegistry.get("ts_sum_if").calculate(x, cond, 3, 1)
    assert np.isnan(summed["A"].iloc[3])
    assert summed["A"].iloc[-1] == 6.0
    last = OperatorRegistry.get("ts_last_if").calculate(x, cond, 4)
    assert last["A"].iloc[2] == 1.0
    assert last["A"].iloc[-1] == 6.0
    days = OperatorRegistry.get("ts_days_since").calculate(cond, 3)
    assert days["A"].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0, 0.0]
    streak = OperatorRegistry.get("ts_true_streak").calculate(cond)
    assert streak["A"].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]


def test_period_lag_changes_only_after_new_period(panels) -> None:
    result = OperatorRegistry.get("period_lag").calculate(
        panels["x"], panels["period"], 1
    )
    assert result.iloc[:2].isna().all().all()
    pd.testing.assert_series_equal(result.iloc[2], panels["x"].iloc[1], check_names=False)
    pd.testing.assert_series_equal(result.iloc[3], panels["x"].iloc[1], check_names=False)
    expected_previous_period = panels["x"].iloc[3].combine_first(panels["x"].iloc[2])
    pd.testing.assert_series_equal(result.iloc[4], expected_previous_period, check_names=False)


@pytest.mark.parametrize("op", OPS)
def test_polars_panel_backend_matches_pandas(op: str, panels) -> None:
    pl = pytest.importorskip("polars")
    args, kwargs = _cases(panels)[op]
    expected = OperatorRegistry.get(op, backend="pandas_numpy").calculate(*args, **kwargs)
    polars_args = [
        pl.DataFrame({str(c): arg[c].to_numpy() for c in arg.columns})
        if isinstance(arg, pd.DataFrame)
        else arg
        for arg in args
    ]
    actual_pl = OperatorRegistry.get(op, backend="polars").calculate(*polars_args, **kwargs)
    actual = pd.DataFrame(
        {c: actual_pl[str(c)].to_numpy() for c in expected.columns},
        index=expected.index,
    )
    np.testing.assert_allclose(
        actual.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=1e-8,
        atol=1e-8,
        equal_nan=True,
    )


@pytest.mark.parametrize("op", OPS)
def test_polars_long_backend_matches_pandas(op: str, panels) -> None:
    pl = pytest.importorskip("polars")
    from backend.polars_expr_emitter import (
        compile_plan_to_polars,
        plan_is_polars_long_capable,
    )

    plan = _plans()[op]
    assert plan_is_polars_long_capable(plan)
    long = pd.concat(
        {name: frame.stack(future_stack=True) for name, frame in panels.items()},
        axis=1,
    ).reset_index()
    compiled = compile_plan_to_polars(plan, pl.from_pandas(long).lazy())
    assert compiled is not None
    actual_long = compiled.frame.sort(["ts", "inst"]).collect().to_pandas()
    actual = actual_long.pivot(index="ts", columns="inst", values="_v").reindex(
        index=panels["x"].index,
        columns=panels["x"].columns,
    )
    args, kwargs = _cases(panels)[op]
    expected = OperatorRegistry.get(op, backend="pandas_numpy").calculate(*args, **kwargs)
    np.testing.assert_allclose(
        actual.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=1e-8,
        atol=1e-8,
        equal_nan=True,
    )


@pytest.mark.parametrize("op", OPS)
def test_duckdb_sql_matches_pandas(op: str, panels) -> None:
    duckdb = pytest.importorskip("duckdb")
    plan = _plans()[op]
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="panel",
        time_column="ts",
        instrument_column="inst",
    )
    assert compiled is not None
    long = (
        pd.concat(
            {name: frame.stack(future_stack=True) for name, frame in panels.items()},
            axis=1,
        )
        .reset_index()
    )
    con = duckdb.connect()
    con.register("panel", long)
    actual_long = con.execute(compiled.query.replace("{{panel}}", "panel")).df()
    actual = actual_long.pivot(index="ts", columns="inst", values="value").reindex(
        index=panels["x"].index,
        columns=panels["x"].columns,
    )
    args, kwargs = _cases(panels)[op]
    expected = OperatorRegistry.get(op, backend="pandas_numpy").calculate(*args, **kwargs)
    np.testing.assert_allclose(
        actual.to_numpy(dtype=float),
        expected.to_numpy(dtype=float),
        rtol=2e-6,
        atol=2e-7,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "op",
    [
        "ts_count_if",
        "ts_last_if",
        "ts_days_since",
        "ts_true_streak",
        "period_lag",
        "ts_regression_tstat",
        "ts_max_drawdown",
        "ts_partial_corr",
        "ts_nth_value",
    ],
)
def test_future_mutation_does_not_change_prefix(op: str, panels) -> None:
    args, kwargs = _cases(panels)[op]
    operator = OperatorRegistry.get(op, backend="pandas_numpy")
    baseline = operator.calculate(*args, **kwargs)
    mutated_args = []
    for arg in args:
        if isinstance(arg, pd.DataFrame):
            changed = arg.copy()
            if np.issubdtype(changed.to_numpy().dtype, np.number):
                changed.iloc[-2:] = changed.iloc[-2:] * -7.0 + 1234.0
            mutated_args.append(changed)
        else:
            mutated_args.append(arg)
    mutated = operator.calculate(*mutated_args, **kwargs)
    np.testing.assert_allclose(
        baseline.iloc[:-2].to_numpy(dtype=float),
        mutated.iloc[:-2].to_numpy(dtype=float),
        equal_nan=True,
    )
