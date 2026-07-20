# -*- coding: utf-8 -*-
"""PlanNode → PhysicalPlan：切分 SQL 可编译子树与 Python fallback。"""
from __future__ import annotations

from planner.logical_plan import PlanNode
from planner.plan_hash import plan_cache_key
from planner.physical_plan import ExecKind, PhysicalNode, PhysicalPlan

# Bare leaves are SQL-capable but extracting them alone under a Python parent
# only adds DuckDB round-trips; the Python parent can read columns cheaper.
_SQL_LEAF_OPS = frozenset({"column", "literal"})


def _is_sql_capable(plan: PlanNode, *, mode: str = "production") -> bool:
    """查询算子注册表，判断节点是否支持 SQL 下推。"""
    from backend.sql_pushdown.sql_registry import (
        is_sql_capable, is_sql_parity_verified, is_sql_production_safe,
    )

    if mode == "production":
        return is_sql_production_safe(plan)
    if mode == "validation":
        return is_sql_parity_verified(plan)
    return is_sql_capable(plan)


def _is_extractable_sql_subtree(plan: PlanNode, *, mode: str = "production") -> bool:
    """Whether ``plan`` should be materialised as a standalone SQL subtree.

    Whole-plan SQL (including a bare ``column`` root) still uses
    ``_is_sql_capable`` at the root.  Partial extraction skips leaf-only nodes.
    """
    if plan.op in _SQL_LEAF_OPS:
        return False
    return _is_sql_capable(plan, mode=mode)


def annotate_node(plan: PlanNode, *, parent_sql: bool = False, mode: str = "production") -> PhysicalNode:
    """递归标注执行种类（调试用 / 可视化）。

    参数：
        plan: 待标注的逻辑计划节点
        parent_sql: 父节点是否已在 SQL 子树内（影响子节点是否继续展开）

    返回：
        带 ``ExecKind`` 与递归 ``children`` 的 ``PhysicalNode``
    """
    if parent_sql and _is_sql_capable(plan, mode=mode):
        children = tuple(annotate_node(c, parent_sql=True, mode=mode) for c in plan.inputs)
        return PhysicalNode(kind=ExecKind.SQL, plan=plan, children=children)

    if _is_extractable_sql_subtree(plan, mode=mode) and not parent_sql:
        sid = plan_cache_key(plan)
        return PhysicalNode(kind=ExecKind.SQL, plan=plan, sid=sid)

    children = tuple(annotate_node(c, parent_sql=False, mode=mode) for c in plan.inputs)
    if plan.op == "materialized_series":
        return PhysicalNode(
            kind=ExecKind.MATERIALIZED,
            plan=plan,
            sid=str(plan.attrs.get("sid", "")),
            children=children,
        )
    return PhysicalNode(kind=ExecKind.PYTHON, plan=plan, children=children)


def lower_to_physical_plan(plan: PlanNode, *, mode: str = "production") -> PhysicalPlan:
    """生成混合物理计划：整树 SQL 或切分 maximal SQL 子树。

    参数：
        plan: 逻辑计划根节点

    返回：
        ``PhysicalPlan``，含改写根、待预计算 SQL 子树及 ``fully_sql`` 标记
    """
    if _is_sql_capable(plan, mode=mode):
        return PhysicalPlan(root=plan, sql_subtrees={}, fully_sql=True)

    pending: dict[str, PlanNode] = {}

    def _transform(node: PlanNode) -> PlanNode:
        """将 maximal SQL 子树替换为 ``materialized_series`` 占位并登记 sid。"""
        if _is_extractable_sql_subtree(node, mode=mode):
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


def collect_sql_subtrees(plan: PlanNode, *, mode: str = "production") -> dict[str, PlanNode]:
    """返回需预计算的 SQL 子树 ``{sid: subplan}``。

    参数：
        plan: 逻辑计划根节点

    返回：
        结构 sid → SQL 可编译子计划的映射
    """
    return lower_to_physical_plan(plan, mode=mode).sql_subtrees
