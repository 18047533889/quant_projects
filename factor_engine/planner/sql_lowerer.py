# -*- coding: utf-8 -*-
"""PlanNode → PhysicalPlan：切分 SQL 可编译子树与 Python fallback。"""
from __future__ import annotations

from planner.logical_plan import PlanNode
from planner.plan_hash import plan_cache_key
from planner.physical_plan import ExecKind, PhysicalNode, PhysicalPlan


def _is_sql_capable(plan: PlanNode) -> bool:
    from backend.sql_pushdown.sql_registry import is_sql_capable

    return is_sql_capable(plan)


def annotate_node(plan: PlanNode, *, parent_sql: bool = False) -> PhysicalNode:
    """递归标注执行种类（调试用 / 可视化）。"""
    if parent_sql and _is_sql_capable(plan):
        children = tuple(annotate_node(c, parent_sql=True) for c in plan.inputs)
        return PhysicalNode(kind=ExecKind.SQL, plan=plan, children=children)

    if _is_sql_capable(plan) and not parent_sql:
        sid = plan_cache_key(plan)
        return PhysicalNode(kind=ExecKind.SQL, plan=plan, sid=sid)

    children = tuple(annotate_node(c, parent_sql=False) for c in plan.inputs)
    if plan.op == "materialized_series":
        return PhysicalNode(
            kind=ExecKind.MATERIALIZED,
            plan=plan,
            sid=str(plan.attrs.get("sid", "")),
            children=children,
        )
    return PhysicalNode(kind=ExecKind.PYTHON, plan=plan, children=children)


def lower_to_physical_plan(plan: PlanNode) -> PhysicalPlan:
    """生成混合物理计划：整树 SQL 或切分 maximal SQL 子树。"""
    if _is_sql_capable(plan):
        return PhysicalPlan(root=plan, sql_subtrees={}, fully_sql=True)

    pending: dict[str, PlanNode] = {}

    def _transform(node: PlanNode) -> PlanNode:
        if _is_sql_capable(node):
            sid = plan_cache_key(node)
            pending[sid] = node
            return PlanNode(
                op="materialized_series",
                inputs=(),
                attrs={"sid": sid},
            )
        new_inputs = tuple(_transform(c) for c in node.inputs)
        return PlanNode(node.op, new_inputs, node.attrs)

    rewritten = _transform(plan)
    return PhysicalPlan(root=rewritten, sql_subtrees=pending, fully_sql=False)


def collect_sql_subtrees(plan: PlanNode) -> dict[str, PlanNode]:
    """返回需预计算的 SQL 子树 ``{sid: subplan}``。"""
    return lower_to_physical_plan(plan).sql_subtrees
