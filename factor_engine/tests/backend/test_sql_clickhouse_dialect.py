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


def _ch_sql(plan: PlanNode) -> str:
    compiled = compile_plan_to_sql(
        plan,
        table="panel_daily",
        time_column="trade_date",
        instrument_column="ticker",
        dialect=SqlDialect.CLICKHOUSE,
    )
    assert compiled is not None
    return compiled.query


def test_group_rank_clickhouse_uses_to_float64_rank():
    plan = PlanNode(op="group_rank", inputs=[_col("close"), _col("grp")])
    q = _ch_sql(plan)
    assert "toFloat64(RANK()" in q
    assert "isNull(x._v)" in q or "if(isNull" in q


def test_group_zscore_clickhouse_stddev():
    plan = PlanNode(op="group_zscore", inputs=[_col("close"), _col("grp")])
    q = _ch_sql(plan)
    assert "stddevSamp" in q
    assert "nullIf" in q


def test_group_percentile_clickhouse_dialect():
    plan = PlanNode(
        op="group_percentile",
        inputs=[_col("close"), _col("grp"), PlanNode(op="literal", inputs=[], attrs={"value": 0.5})],
    )
    q = _ch_sql(plan)
    assert "toFloat64(RANK()" in q
    assert "WHEN x._v IS NULL THEN 0.0" in q
    assert "<= 0.5" in q


def test_group_decay_linear_clickhouse_dialect():
    plan = PlanNode(
        op="group_decay_linear",
        inputs=[_col("close"), _col("grp"), PlanNode(op="literal", inputs=[], attrs={"value": 5})],
    )
    q = _ch_sql(plan)
    assert "toFloat64(RANK()" in q
    assert "WHEN x._v IS NULL THEN NULL" in q
    assert "nullIf" in q


def test_cs_resid_clickhouse_dialect():
    plan = PlanNode(op="cs_resid", inputs=[_col("close"), _col("open")])
    q = _ch_sql(plan)
    assert "covarSamp" in q
    assert "varSamp" in q
    assert "< 3" in q
