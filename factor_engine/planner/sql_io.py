# -*- coding: utf-8 -*-
"""SQL 下推与列 IO 协调（避免 fully_sql 计划重复 prefetch）。"""
from __future__ import annotations

from typing import Any, Iterable

from planner.logical_plan import PlanNode
from planner.sql_lowerer import lower_to_physical_plan


def plan_is_fully_sql(plan: PlanNode) -> bool:
    """逻辑计划是否可整树 SQL 执行（无需 Python 列 prefetch）。

    参数：
        plan: 逻辑计划根节点

    返回：
        ``True`` 表示可完全 SQL 下推；否则需 Python 后端参与
    """
    return lower_to_physical_plan(plan).fully_sql


def should_skip_column_prefetch(
    plans: Iterable[PlanNode],
    *,
    input_dq_check: bool = False,
    backend: Any | None = None,
) -> bool:
    """fully_sql / native-scan backend 且无 input DQ 时跳过 ``load_columns`` / prefetch。

    参数：
        plans: 待执行的一个或多个逻辑计划
        input_dq_check: 为 ``True`` 时强制需要 prefetch（数据质量检查）
        backend: 可选执行后端；``prefers_native_scan`` 为真时倾向跳过 prefetch

    返回：
        ``True`` 表示可跳过列预取；``False`` 表示仍需 prefetch
    """
    if input_dq_check:
        return False
    if backend is not None and getattr(backend, "prefers_native_scan", False):
        return True
    nodes = list(plans)
    if not nodes:
        return False
    return all(plan_is_fully_sql(p) for p in nodes)
