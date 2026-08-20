# -*- coding: utf-8
"""Alpha-language SQL subset: emitter compiles to real DuckDB SQL and the
executed result matches the pandas reference exactly (clean daily panels).

This bypasses ``data_access`` and exercises the emitter + DuckDB execution
directly, so it runs in environments without the optional data_access package.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")

import duckdb

from backend.cleaned_bridge import ensure_cleaned_loaded
from backend.sql_pushdown.emitter import compile_plan_to_sql, plan_is_sql_capable
from backend.sql_pushdown.plan_fixtures import column, literal
from cleaned_operators.registry import OperatorRegistry
from planner.logical_plan import PlanNode

ensure_cleaned_loaded()

# (op, source column, panel values, positional literals, pandas kwargs)
SQL_CASES = [
    ("event_frequency", "close", [10.0, 11.0, 10.5, 12.0, 11.5, 20.0, 21.0, 20.5, 22.0, 21.5],
     [5, 1], dict(window=5, min_periods=1)),
    ("ts_semivariance_balance", "ret", [0.01, 0.02, -0.01, 0.03, 0.04, 0.01, 0.02, 0.05, 0.03, 0.02],
     [5, 2], dict(window=5, min_periods=2)),
    ("ts_realized_quarticity", "ret", [0.01, 0.02, -0.01, 0.03, 0.04, 0.01, 0.02, 0.05, 0.03, 0.02],
     [5, 3], dict(window=5, min_periods=3)),
    ("ts_vol_of_vol", "ret", [0.01, 0.02, -0.01, 0.03, 0.04, 0.01, 0.02, 0.05, 0.03, 0.02],
     [3, 5], dict(inner_window=3, outer_window=5)),
    ("ts_vol_acceleration", "ret", [0.01, 0.02, -0.01, 0.03, 0.04, 0.01, 0.02, 0.05, 0.03, 0.02],
     [3, 2], dict(inner_window=3, lag=2)),
    ("ts_vol_term_structure", "ret", [0.01, 0.02, -0.01, 0.03, 0.04, 0.01, 0.02, 0.05, 0.03, 0.02],
     [3, 5], dict(short_window=3, long_window=5)),
]

DATES = [pd.Timestamp("2024-01-02") + pd.Timedelta(days=i) for i in range(5)]
_INST = ["A", "B"]


def _build_table(vals):
    rows = []
    half = len(vals) // 2
    for inst, offset in zip(_INST, (0, half)):
        for day, v in enumerate(vals[offset : offset + half]):
            rows.append({"ts": DATES[day].date(), "inst": inst, "close": float(v), "ret": float(v)})
    return pd.DataFrame(rows)


def _pandas_reference(op, vals, pkw):
    half = len(vals) // 2
    panel = pd.DataFrame({"A": vals[:half], "B": vals[half:]}, index=DATES)
    inst = OperatorRegistry.get(op, "pandas_numpy")
    return inst.calculate(panel, **pkw).stack().sort_index()


@pytest.mark.parametrize("op,col,vals,lits,pkw", SQL_CASES)
def test_alpha_language_sql_matches_pandas(op, col, vals, lits, pkw):
    plan = PlanNode(op=op, inputs=[column(col)] + [literal(v) for v in lits], attrs={})
    assert plan_is_sql_capable(plan), f"{op} not SQL-capable"
    sql = compile_plan_to_sql(plan, dataset="test_panel", time_column="ts", instrument_column="inst")
    assert sql is not None and sql.query.strip(), f"{op} compiled empty SQL"
    query = sql.query.replace("{{test_panel}}", "test_panel")

    con = duckdb.connect(":memory:")
    con.register("test_panel", _build_table(vals))
    out = con.execute(query).df()
    out["ts"] = pd.to_datetime(out["ts"])
    sql_series = out.set_index(["ts", "inst"])["value"].astype(float).sort_index()

    pd_series = _pandas_reference(op, vals, pkw)
    pd_series.index = pd_series.index.set_names(["ts", "inst"])
    aligned = pd_series.reindex(sql_series.index)

    assert len(sql_series) == len(aligned)
    np.testing.assert_allclose(
        aligned.to_numpy(), sql_series.to_numpy(),
        rtol=1e-5, atol=1e-6, equal_nan=True,
        err_msg=f"{op} duckdb-sql != pandas",
    )


def test_alpha_language_sql_capable_listed_in_sql_implemented():
    from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    missing = [op for op, *_ in SQL_CASES if op not in SQL_IMPLEMENTED_CANONICALS]
    assert not missing, f"SQL ops missing from SQL_IMPLEMENTED_CANONICALS: {missing}"
