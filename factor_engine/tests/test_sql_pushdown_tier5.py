# -*- coding: utf-8 -*-
"""DuckDB SQL 下推 Tier5：ewm_corr / ewm_cov。"""
from __future__ import annotations

import pytest

from backend.sql_pushdown.emitter import compile_plan_to_sql, plan_is_sql_capable
from backend.sql_pushdown.sql_registry import SQL_CAPABLE_CANONICALS
from planner.logical_plan import PlanNode


def _col(name: str) -> PlanNode:
    return PlanNode(op="column", attrs={"name": name})


@pytest.mark.parametrize("op", ["ewm_corr", "ewm_cov"])
def test_tier5_ops_in_registry(op):
    assert op in SQL_CAPABLE_CANONICALS


@pytest.mark.parametrize("op,frag", [("ewm_cov", "xv"), ("ewm_corr", "xv")])
def test_tier5_emit_sql(op, frag):
    plan = PlanNode(
        op=op,
        inputs=[_col("close"), _col("volume")],
        attrs={"span": 10, "window": 10},
    )
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan, dataset="d", time_column="t", instrument_column="i"
    )
    assert compiled is not None
    assert frag in compiled.query
