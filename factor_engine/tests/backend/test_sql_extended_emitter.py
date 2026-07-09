# -*- coding: utf-8
"""SQL group_rank / winsorize emitter 测试。"""
from __future__ import annotations

from backend.sql_pushdown.emitter import compile_plan_to_sql, plan_is_sql_capable
from planner.logical_plan import PlanNode


def _col(name: str) -> PlanNode:
    return PlanNode(op="column", attrs={"name": name})


def test_group_rank_sql():
    plan = PlanNode(op="group_rank", inputs=[_col("close"), _col("industry")])
    assert plan_is_sql_capable(plan)
    sql = compile_plan_to_sql(
        plan, dataset="d", time_column="t", instrument_column="i"
    )
    assert sql is not None
    assert "RANK()" in sql.query
    assert "g._v" in sql.query


def test_winsorize_sql():
    plan = PlanNode(op="winsorize", inputs=[_col("close")], attrs={"a": 0.05})
    assert plan_is_sql_capable(plan)
    sql = compile_plan_to_sql(
        plan, dataset="d", time_column="t", instrument_column="i"
    )
    assert sql is not None
    assert "quantile_cont" in sql.query


def test_if_else_sql():
    plan = PlanNode(
        op="if_else",
        inputs=[_col("flag"), _col("a"), _col("b")],
    )
    assert plan_is_sql_capable(plan)
    sql = compile_plan_to_sql(
        plan, dataset="d", time_column="t", instrument_column="i"
    )
    assert sql is not None
    assert "CASE WHEN" in sql.query


def test_group_winsorize_sql():
    plan = PlanNode(
        op="group_winsorize",
        inputs=[_col("close"), _col("industry")],
        attrs={"a": 0.05},
    )
    assert plan_is_sql_capable(plan)
    sql = compile_plan_to_sql(
        plan, dataset="d", time_column="t", instrument_column="i"
    )
    assert sql is not None
    assert "quantile_cont" in sql.query
    assert "g._v" in sql.query


def test_ts_beta_sql():
    plan = PlanNode(
        op="ts_beta",
        inputs=[_col("ret"), _col("mkt")],
        attrs={"window": 20},
    )
    assert plan_is_sql_capable(plan)
    sql = compile_plan_to_sql(
        plan, dataset="d", time_column="t", instrument_column="i"
    )
    assert sql is not None
    assert "covar_samp" in sql.query


def test_ts_ema_sql_duckdb():
    plan = PlanNode(op="ts_ema", inputs=[_col("close")], attrs={"window": 12})
    assert plan_is_sql_capable(plan)
    sql = compile_plan_to_sql(
        plan, dataset="d", time_column="t", instrument_column="i"
    )
    assert sql is not None
    assert "POW(" in sql.query


def test_batch_compile_sql():
    from backend.sql_pushdown.emitter import compile_plans_batch_to_sql

    p1 = PlanNode(op="ts_mean", inputs=[_col("close")], attrs={"window": 5})
    p2 = PlanNode(op="ts_std", inputs=[_col("open")], attrs={"window": 5})
    batch = compile_plans_batch_to_sql(
        {"s1": p1, "s2": p2},
        dataset="d",
        time_column="t",
        instrument_column="i",
    )
    assert batch is not None
    assert "sub_v_0" in batch.query
    assert "sub_v_1" in batch.query
    assert len(batch.column_aliases) == 2
