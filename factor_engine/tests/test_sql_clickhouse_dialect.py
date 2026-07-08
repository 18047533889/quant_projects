# -*- coding: utf-8
"""ClickHouse SQL 方言 emitter 测试。"""
from __future__ import annotations

from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql
from planner.logical_plan import PlanNode


def _col(name: str) -> PlanNode:
    return PlanNode(op="column", attrs={"name": name})


def test_winsorize_clickhouse_uses_quantile_exact():
    plan = PlanNode(op="winsorize", inputs=[_col("close")], attrs={"a": 0.05})
    sql = compile_plan_to_sql(
        plan,
        table="panel_daily",
        time_column="trade_date",
        instrument_column="ticker",
        dialect=SqlDialect.CLICKHOUSE,
    )
    assert sql is not None
    assert "quantileExact(0.05)" in sql.query
    assert "quantile_cont" not in sql.query


def test_sign_clickhouse_dialect():
    plan = PlanNode(op="sign", inputs=[_col("close")])
    sql = compile_plan_to_sql(
        plan,
        table="panel_daily",
        time_column="trade_date",
        instrument_column="ticker",
        dialect=SqlDialect.CLICKHOUSE,
    )
    assert sql is not None
    assert "sign(_v)" in sql.query


def test_ts_beta_clickhouse_covar_samp():
    plan = PlanNode(
        op="ts_beta",
        inputs=[_col("ret"), _col("mkt")],
        attrs={"window": 10},
    )
    sql = compile_plan_to_sql(
        plan,
        table="panel_daily",
        time_column="trade_date",
        instrument_column="ticker",
        dialect=SqlDialect.CLICKHOUSE,
    )
    assert sql is not None
    assert "covarSamp" in sql.query
    assert "varSamp" in sql.query
