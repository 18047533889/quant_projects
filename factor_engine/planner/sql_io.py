# -*- coding: utf-8
"""SQL 下推与列 IO 协调（避免 fully_sql 计划重复 prefetch）。"""
from __future__ import annotations

from typing import Any, Iterable

from planner.logical_plan import PlanNode
from planner.sql_lowerer import lower_to_physical_plan


def plan_is_fully_sql(plan: PlanNode) -> bool:
    """逻辑计划是否可整树 SQL 执行（无需 Python 列 prefetch）。"""
    return lower_to_physical_plan(plan).fully_sql


def should_skip_column_prefetch(
    plans: Iterable[PlanNode],
    *,
    input_dq_check: bool = False,
) -> bool:
    """fully_sql 且无 input DQ 时跳过 ``load_columns`` / prefetch。"""
    if input_dq_check:
        return False
    nodes = list(plans)
    if not nodes:
        return False
    return all(plan_is_fully_sql(p) for p in nodes)
