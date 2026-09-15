"""Planning and executable SQL contract for canonical ts_delay aliases."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

import factor_engine.cleaned_operators as cleaned

cleaned.load_all()

from factor_engine.backend.operator_errors import OperatorParameterError
from factor_engine.backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.planner.logical_plan import PlanNode
from factor_engine.planner.optimizer import Optimizer


def _plan(attrs: dict) -> PlanNode:
    return PlanNode(
        op="ts_delay",
        inputs=[PlanNode(op="column", attrs={"name": "close"}, inputs=[])],
        attrs=dict(attrs),
    )


@pytest.mark.parametrize("name", ["n", "window", "d", "lag", "periods"])
@pytest.mark.parametrize("bad", [5.9, True, float("nan"), float("inf")])
def test_ts_delay_sql_plan_rejects_invalid_integer_aliases(name, bad) -> None:
    with pytest.raises(OperatorParameterError):
        Optimizer().optimize(_plan({name: bad}), production=True)


def test_ts_delay_sql_executes_alias_with_row_lag_oracle() -> None:
    duckdb = pytest.importorskip("duckdb")
    plan = Optimizer().optimize(_plan({"window": 2}), production=True)
    compiled = compile_plan_to_sql(
        plan,
        dataset="panel",
        time_column="ts",
        instrument_column="inst",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    frame = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=4).repeat(2),
            "inst": ["A", "B"] * 4,
            "close": [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
        }
    )
    con = duckdb.connect()
    try:
        con.register("panel", frame)
        actual = con.execute(compiled.query.replace("{{panel}}", "panel")).df()
    finally:
        con.close()
    actual = actual.sort_values(["inst", "ts"])["value"].to_numpy()
    expected = np.array([np.nan, np.nan, 1.0, 2.0, np.nan, np.nan, 10.0, 20.0])
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_ts_delay_polars_runtime_accepts_every_declared_compat_alias() -> None:
    operator = OperatorRegistry.get("ts_delay", backend="polars")
    assert dict(operator.metadata.param_aliases) == {
        "d": "n",
        "window": "n",
        "lag": "n",
        "periods": "n",
    }
    panel = pl.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
    expected = np.array([np.nan, np.nan, 1.0, 2.0])
    for alias in ("n", "d", "window", "lag", "periods"):
        actual = operator.calculate(panel, **{alias: 2})
        np.testing.assert_allclose(
            actual["A"].to_numpy(), expected, equal_nan=True, err_msg=alias
        )
