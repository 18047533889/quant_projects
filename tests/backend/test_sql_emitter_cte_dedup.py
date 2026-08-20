# -*- coding: utf-8
"""SQL emitter DAG-level CTE 去重。"""
from __future__ import annotations

from backend.sql_pushdown.emitter import compile_plan_to_sql, plan_is_sql_capable
from backend.sql_pushdown.plan_fixtures import column, minimal_plan
from planner.logical_plan import PlanNode


def test_sql_emitter_deduplicates_shared_subtree():
    left = minimal_plan("ts_mean")
    right = minimal_plan("ts_mean")
    plan = PlanNode(op="add", inputs=[left, right], attrs={})
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="test_panel",
        time_column="ts",
        instrument_column="inst",
    )
    assert compiled is not None
    sql = compiled.query
    assert " s1 AS " in sql or " s1 AS(" in sql.replace(" ", "")
