# -*- coding: utf-8 -*-
"""Real Pandas / pure-Polars / DuckDB execution evidence for production recipes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.context import ExecutionContext
from backend.factory import build_backend
from backend.sql_pushdown.emitter import compile_plan_to_sql
from cleaned_operators import load_all
from factor_recipes.planner_bridge import compile_recipe_plans
from tests.helpers import InMemorySeriesSource


RECIPE_CASES = {
    "annualized_volatility": {"returns": "returns", "window": 3, "ann_factor": 252.0},
    "bollinger_lower": {"x": "close", "window": 3, "width": 2.0},
    "bollinger_mid": {"x": "close", "window": 3},
    "bollinger_upper": {"x": "close", "window": 3, "width": 2.0},
    "current_ratio": {"current_assets": "current_assets", "current_liabilities": "current_liabilities"},
    "debt_to_equity": {"total_debt": "total_debt", "total_equity": "total_equity"},
    "intraday_return": {"open": "open", "close": "close"},
    "momentum": {"x": "close", "window": 2},
    "net_debt_ratio": {"short_debt": "short_debt", "long_debt": "long_debt", "cash": "cash", "total_assets": "total_assets"},
    "operating_margin": {"operating_income": "operating_income", "revenue": "revenue"},
    "overnight_gap": {"open": "open", "close": "close"},
    "quick_ratio": {"current_assets": "current_assets", "inventory": "inventory", "current_liabilities": "current_liabilities"},
    "rate_of_change": {"x": "close", "window": 2},
    "rolling_sharpe": {"returns": "returns", "window": 3, "ann_factor": 252.0},
    "stochastic_d": {"high": "high", "low": "low", "close": "close", "window": 3, "smooth_window": 2},
    "stochastic_k": {"high": "high", "low": "low", "close": "close", "window": 3},
    "williams_r": {"high": "high", "low": "low", "close": "close", "window": 3},
}


@pytest.fixture(scope="module")
def recipe_source():
    load_all()
    ts = pd.date_range("2024-01-01", periods=6)
    idx = pd.MultiIndex.from_product([ts, ["A", "B"]], names=["timestamp", "instrument"])
    base = np.array([10, 20, 11, 21, 12, 19, 13, 22, 12, 23, 14, 24], dtype=float)
    values = {
        "close": base,
        "open": base - 0.5,
        "high": base + 1.0,
        "low": base - 1.0,
        "returns": np.array([.01, .02, -.01, .03, .02, -.02, .04, .01, .03, .02, -.01, .05]),
        "current_assets": base * 5,
        "current_liabilities": base * 2,
        "inventory": base * 0.5,
        "total_debt": base * 3,
        "total_equity": base * 7,
        "short_debt": base,
        "long_debt": base * 2,
        "cash": base * 0.25,
        "total_assets": base * 10,
        "operating_income": base * 1.5,
        "revenue": base * 4,
    }
    return InMemorySeriesSource(data={k: pd.Series(v, index=idx) for k, v in values.items()})


def _plan(name: str):
    return compile_recipe_plans({name: (name, RECIPE_CASES[name])}).plans[name]


def _duckdb_result(plan, source) -> pd.Series:
    duckdb = pytest.importorskip("duckdb")
    frame = pd.DataFrame({
        "ts": source.data["close"].index.get_level_values("timestamp"),
        "inst": source.data["close"].index.get_level_values("instrument"),
        **{name: series.to_numpy() for name, series in source.data.items()},
    })
    compiled = compile_plan_to_sql(
        plan, dataset="panel", time_column="ts", instrument_column="inst"
    )
    assert compiled is not None
    con = duckdb.connect()
    con.register("panel", frame)
    result = con.execute(compiled.query.replace("{{panel}}", "panel")).df()
    return result.set_index(["ts", "inst"])["value"].sort_index()


@pytest.mark.parametrize("recipe", sorted(RECIPE_CASES))
def test_production_recipe_executes_on_three_real_backends(recipe_source, recipe: str) -> None:
    plan = _plan(recipe)
    pandas_out = build_backend("pandas").execute(
        plan, ExecutionContext(data_source=recipe_source, run_mode="research")
    ).sort_index()
    polars_out = build_backend("polars_long").execute(
        plan, ExecutionContext(data_source=recipe_source, run_mode="research")
    ).sort_index()
    duckdb_out = _duckdb_result(plan, recipe_source)
    pd.testing.assert_series_equal(pandas_out, polars_out, check_names=False, rtol=1e-9, atol=1e-9)
    pd.testing.assert_series_equal(pandas_out, duckdb_out, check_names=False, rtol=1e-9, atol=1e-9)
