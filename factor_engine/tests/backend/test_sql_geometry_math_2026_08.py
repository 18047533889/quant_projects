# -*- coding: utf-8 -*-
"""2026-08 geometry/math expansion — DuckDB SQL pushdown parity tests.

The 5 pushdown ops (3 intraday volatility shape + 2 crossing quality) must
compile to DuckDB SQL and execute with exact parity to the pandas_numpy
reference (max abs diff < 1e-9, identical NaN pattern).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.sql_pushdown.emitter import compile_plan_to_sql, plan_is_sql_capable
from planner.logical_plan import PlanNode

_SQL_OPS: dict[str, tuple[tuple[str, ...], int]] = {
    "intraday_volatility_concentration": (("ret",), 240),
    "intraday_volatility_entropy": (("ret",), 240),
    "intraday_realized_semivariance_balance": (("ret",), 240),
    "ts_crossing_speed": (("close", "vol"), 20),
    "ts_crossing_acceleration": (("close", "vol"), 20),
}


@pytest.fixture(scope="module", autouse=True)
def _load():
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def _col(name: str) -> PlanNode:
    return PlanNode(op="column", attrs={"name": name})


def _plan(op: str, cols: tuple[str, ...], window: int) -> PlanNode:
    return PlanNode(op=op, inputs=[_col(c) for c in cols], attrs={"d": window})


def test_sql_geometry_math_compiles():
    from cleaned_operators import OperatorRegistry

    for op, (cols, w) in _SQL_OPS.items():
        plan = _plan(op, cols, w)
        assert plan_is_sql_capable(plan), f"{op} not SQL capable"
        compiled = compile_plan_to_sql(
            plan, dataset="cn_stock_daily", time_column="trade_date", instrument_column="ticker"
        )
        assert compiled is not None, f"{op} failed to compile"
        assert "cn_stock_daily" in compiled.query
        # the op is registered daily and SQL-implemented
        assert "pandas_numpy" in OperatorRegistry.backends_for(op)


@pytest.mark.parametrize("op", sorted(_SQL_OPS))
def test_sql_geometry_math_duckdb_parity(op):
    duckdb = pytest.importorskip("duckdb")
    from cleaned_operators import OperatorRegistry

    rng = np.random.default_rng(11)
    mi = pd.date_range("2024-01-01", periods=240, freq="1min")
    ret = pd.DataFrame(rng.normal(0, 0.01, (240, 2)), index=mi, columns=["A", "B"])
    ret.iloc[5, 0] = 0.05
    ret.iloc[6, 1] = -0.04
    close = pd.DataFrame(100 + np.cumsum(rng.normal(0, 0.1, (240, 2)), 0), index=mi, columns=["A", "B"])
    vol = pd.DataFrame(rng.lognormal(5, 0.5, (240, 2)), index=mi, columns=["A", "B"])

    cols, w = _SQL_OPS[op]
    long_rows = []
    for c in ["A", "B"]:
        for i, t in enumerate(mi):
            long_rows.append((t, c, float(ret.iloc[i][c]), float(close.iloc[i][c]), float(vol.iloc[i][c])))
    df = pd.DataFrame(long_rows, columns=["ts", "inst", "ret", "close", "vol"])
    conn = duckdb.connect()
    conn.register("df", df)
    conn.execute("CREATE TABLE t_test AS SELECT * FROM df")

    plan = _plan(op, cols, w)
    compiled = compile_plan_to_sql(plan, dataset="t_test", time_column="ts", instrument_column="inst")
    assert compiled is not None
    sql_res = conn.execute(compiled.query.replace("{{t_test}}", "t_test")).fetchdf()
    sql_v = sql_res.set_index(["ts", "inst"])["value"].sort_index()

    panels = {"ret": ret, "close": close, "vol": vol}
    args = tuple(panels[c] for c in cols)
    op_impl = OperatorRegistry.get(op, "pandas_numpy")
    pd_out = op_impl.calculate(*args, window=w)
    pd_v = pd_out.stack().rename("value").sort_index()
    pd_v.index.names = ["ts", "inst"]

    joint = pd.concat([sql_v, pd_v], axis=1, keys=["sql", "pd"]).dropna()
    assert len(joint) > 0, f"{op}: no overlapping finite rows"
    max_diff = float(np.nanmax((joint["sql"] - joint["pd"]).abs()))
    assert max_diff < 1e-9, f"{op}: SQL vs pandas maxdiff {max_diff}"
    full = pd.concat([sql_v, pd_v], axis=1, keys=["sql", "pd"])
    na_mismatch = int((full["sql"].isna() != full["pd"].isna()).sum())
    assert na_mismatch == 0, f"{op}: {na_mismatch} NaN-pattern mismatches"
