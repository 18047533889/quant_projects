# -*- coding: utf-8
"""index_weight 的 DuckDB SQL emitter 真实执行 parity。

``index_weight``（normalize=True）的 pandas 参考实现按每个横截面（date）做
nansum 归一化，零/近零总和输出 NaN。SQL 长表语义下这正是
``_v / SUM(_v) OVER (PARTITION BY ts)``（SUM 天然忽略 NULL）。本测试直接对
emitter 产出的 SQL 在真实 DuckDB 上执行，再与 pandas 内核逐值对齐。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("polars")

from factor_engine.backend.sql_pushdown.emitter import (
    SqlDialect,
    compile_plan_to_sql,
)
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.relation.ops import IndexWeight
from factor_engine.planner.logical_plan import PlanNode

load_all()


def _wide_panel(dates, instruments, values) -> pd.DataFrame:
    return pd.DataFrame(
        {inst: [values[inst][i] for i in range(len(dates))] for inst in instruments},
        index=pd.DatetimeIndex(dates, name="date"),
        dtype=float,
    )


def _pandas_reference(panel: pd.DataFrame) -> pd.DataFrame:
    return IndexWeight()._calculate_series(panel)


def _index_weight_plan():
    from factor_engine.backend.sql_pushdown.plan_fixtures import column

    return PlanNode(op="index_weight", inputs=[column("weight")], attrs={"normalize": True})


def _sql_reference(panel: pd.DataFrame, *, normalize: bool = True) -> pd.DataFrame:
    import duckdb

    long_rows = []
    for date, row in panel.iterrows():
        for inst, value in row.items():
            long_rows.append(
                {"ts": pd.Timestamp(date).date(), "inst": inst, "weight": value}
            )
    con = duckdb.connect()
    con.register("panel", pd.DataFrame(long_rows))
    compiled = compile_plan_to_sql(
        _index_weight_plan(),
        dataset="panel",
        table="panel",
        time_column="ts",
        instrument_column="inst",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None, "index_weight SQL did not compile"
    sql = compiled.query.replace("{{panel}}", "panel")
    out = con.execute(sql).fetchdf()
    pivoted = out.pivot(index="ts", columns="inst", values="value").sort_index()
    pivoted.index = pd.DatetimeIndex(pivoted.index).as_unit("ns")
    pivoted.index.name = "date"
    pivoted.columns.name = None
    return pivoted[panel.columns]


def test_index_weight_sql_parity_normalize():
    dates = ["2024-01-02", "2024-01-03"]
    panel = _wide_panel(
        dates,
        ["A", "B", "C"],
        {"A": [10.0, 30.0], "B": [30.0, 60.0], "C": [60.0, 10.0]},
    )
    pd_ref = _pandas_reference(panel)
    sql_ref = _sql_reference(panel)
    pd.testing.assert_frame_equal(pd_ref, sql_ref, rtol=1e-9, atol=1e-12)


def test_index_weight_sql_parity_handles_nan_and_zero_total():
    import duckdb

    dates = ["2024-01-02"]
    # A row with NaN and a row whose total is zero.
    panel = _wide_panel(
        dates,
        ["A", "B", "C", "D"],
        {"A": [np.nan], "B": [30.0], "C": [60.0], "D": [0.0]},
    )
    pd_ref = _pandas_reference(panel)
    assert np.isnan(pd_ref.loc[pd.Timestamp("2024-01-02"), "A"])

    con = duckdb.connect()
    lf = pd.DataFrame(
        {
            "ts": [pd.Timestamp("2024-01-02").date()] * 4,
            "inst": ["A", "B", "C", "D"],
            "weight": [None, 30.0, 60.0, 0.0],
        }
    )
    con.register("panel", lf)
    compiled = compile_plan_to_sql(
        _index_weight_plan(),
        dataset="panel",
        table="panel",
        time_column="ts",
        instrument_column="inst",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    sql = compiled.query.replace("{{panel}}", "panel")
    out = con.execute(sql).fetchdf().pivot(index="ts", columns="inst", values="value")
    out.index = pd.DatetimeIndex(out.index).as_unit("ns")
    out.index.name = "date"
    out.columns.name = None
    out = out[panel.columns]
    pd.testing.assert_frame_equal(pd_ref, out, rtol=1e-9, atol=1e-12)


def test_index_weight_sql_emitter_registered():
    from factor_engine.backend.operator_capability import _sql_emitter_ok
    from factor_engine.backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    assert "index_weight" in SQL_IMPLEMENTED_CANONICALS
    assert _sql_emitter_ok("index_weight")
