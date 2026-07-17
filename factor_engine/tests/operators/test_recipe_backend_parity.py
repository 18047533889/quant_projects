# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all
from factor_recipes.planner_bridge import compile_recipe_plans
from factor_recipes.registry import FactorRecipeRegistry


@pytest.fixture(scope="module", autouse=True)
def _load() -> None:
    load_all()


def _data() -> dict[str, pd.DataFrame]:
    index = pd.Index(range(10), name="ts")
    columns = pd.Index(["A", "B", "C"], name="inst")
    row = np.arange(10, dtype=float)[:, None]
    col = np.arange(3, dtype=float)[None, :]
    close = 20.0 + row * 0.7 + col * 1.2 + np.sin(row + col) * 0.15
    open_ = close - 0.3
    close[3, 1] = np.nan
    return {
        "close": pd.DataFrame(close, index=index, columns=columns),
        "open": pd.DataFrame(open_, index=index, columns=columns),
    }


def _long(inputs):
    return pd.concat(
        {name: frame.stack(future_stack=True) for name, frame in inputs.items()}, axis=1
    ).reset_index()


CASES = {
    "bollinger_upper": ({"x": "close", "window": 4, "width": 2.0}, ("close",)),
    "intraday_return": ({"open": "open", "close": "close"}, ("open", "close")),
}


@pytest.mark.parametrize("name", CASES)
def test_recipe_plan_matches_pandas_on_polars_and_duckdb(name: str) -> None:
    pl = pytest.importorskip("polars")
    duckdb = pytest.importorskip("duckdb")
    from backend.polars_expr_emitter import compile_plan_to_polars, plan_is_polars_long_capable
    from backend.sql_pushdown.emitter import compile_plan_to_sql, plan_is_sql_capable

    bindings, fields = CASES[name]
    panels = _data()
    inputs = {field: panels[field] for field in fields}
    expected = FactorRecipeRegistry.compile_batch({name: (name, bindings)}).execute(
        inputs, backend="pandas_numpy"
    )[name]
    plan = compile_recipe_plans({name: (name, bindings)}).plans[name]
    long = _long(inputs)

    assert plan_is_polars_long_capable(plan)
    polars_plan = compile_plan_to_polars(plan, pl.from_pandas(long).lazy())
    assert polars_plan is not None
    polars_long = polars_plan.frame.collect().to_pandas()
    polars_result = polars_long.pivot(index="ts", columns="inst", values="_v").reindex(
        index=expected.index, columns=expected.columns
    )
    np.testing.assert_allclose(polars_result, expected, rtol=1e-9, atol=1e-9, equal_nan=True)

    assert plan_is_sql_capable(plan)
    sql_plan = compile_plan_to_sql(
        plan, dataset="panel", time_column="ts", instrument_column="inst"
    )
    assert sql_plan is not None
    connection = duckdb.connect()
    connection.register("panel", long)
    sql_long = connection.execute(sql_plan.query.replace("{{panel}}", "panel")).df()
    sql_result = sql_long.pivot(index="ts", columns="inst", values="value").reindex(
        index=expected.index, columns=expected.columns
    )
    np.testing.assert_allclose(sql_result, expected, rtol=2e-6, atol=2e-7, equal_nan=True)
