# -*- coding: utf-8 -*-
"""SQL 下推 emitter 单元测试。"""
from __future__ import annotations

from backend.sql_pushdown.emitter import compile_plan_to_sql, plan_is_sql_capable
from planner.logical_plan import PlanNode


def _col(name: str) -> PlanNode:
    return PlanNode(op="column", attrs={"name": name})


def _lit(v) -> PlanNode:
    return PlanNode(op="literal", attrs={"value": v})


def test_ts_mean_column_sql():
    plan = PlanNode(
        op="ts_mean",
        inputs=[_col("close")],
        attrs={"d": 3},
    )
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="cn_stock_daily",
        time_column="trade_date",
        instrument_column="ticker",
    )
    assert compiled is not None
    assert "cn_stock_daily" in compiled.query
    assert "AVG" in compiled.query
    assert "ROWS BETWEEN 2 PRECEDING" in compiled.query
    assert compiled.referenced_columns == frozenset({"close"})


def test_rank_ts_mean_sql():
    plan = PlanNode(
        op="rank",
        inputs=[
            PlanNode(
                op="ts_mean",
                inputs=[_col("close")],
                attrs={"d": 2},
            )
        ],
    )
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="panel",
        time_column="ts",
        instrument_column="inst",
    )
    assert compiled is not None
    assert "RANK()" in compiled.query
    assert "AVG" in compiled.query


def test_macd_not_sql_capable():
    plan = PlanNode(op="MACD", inputs=[_col("close")], attrs={"window": 12})
    assert not plan_is_sql_capable(plan)


def test_zscore_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(op="zscore", inputs=[_col("close")])
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "STDDEV" in compiled.query or "stddev" in compiled.query.lower()


def test_ts_corr_sql():
    from backend.sql_pushdown.emitter import compile_plan_to_sql

    plan = PlanNode(
        op="ts_corr",
        inputs=[_col("close"), _col("volume")],
        attrs={"d": 3},
    )
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
    )
    assert compiled is not None
    assert "corr" in compiled.query.lower()
    assert '"close"' in compiled.query and '"volume"' in compiled.query


def test_group_neutralize_sql():
    from backend.sql_pushdown.emitter import compile_plan_to_sql

    plan = PlanNode(
        op="group_neutralize",
        inputs=[_col("close"), _col("industry")],
    )
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
    )
    assert compiled is not None
    assert "PARTITION BY x.ts, g._v" in compiled.query


def test_where_sql():
    from backend.sql_pushdown.emitter import compile_plan_to_sql

    plan = PlanNode(
        op="where",
        inputs=[_col("flag"), _col("a"), _col("b")],
    )
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
    )
    assert compiled is not None
    assert "CASE WHEN" in compiled.query


