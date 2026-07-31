# -*- coding: utf-8 -*-
"""PlanNode -> PhysicalPlan: split certified SQL subtrees from Python/SourceRef execution."""
from __future__ import annotations

from planner.logical_plan import PlanNode
from planner.plan_hash import plan_cache_key
from planner.physical_plan import ExecKind, PhysicalNode, PhysicalPlan

_SQL_LEAF_OPS = frozenset({"column", "literal"})


def _contains_source_ref(plan: PlanNode) -> bool:
    if plan.op == "column":
        try:
            from api.source_ref import decode_source_ref
            if decode_source_ref(str((plan.attrs or {}).get("name", ""))) is not None:
                return True
        except Exception:
            pass
    return any(_contains_source_ref(child) for child in plan.inputs)


def _is_sql_capable(plan: PlanNode, *, mode: str = "production") -> bool:
    # Opaque logical-source columns are not physical DuckDB columns.  Their
    # subtree must cross the SourceRef resolver before numerical execution.
    if _contains_source_ref(plan):
        return False
    from backend.sql_pushdown.sql_registry import is_sql_capable, is_sql_parity_verified, is_sql_production_safe
    if mode == "production": return is_sql_production_safe(plan)
    if mode == "validation": return is_sql_parity_verified(plan)
    return is_sql_capable(plan)


def _is_extractable_sql_subtree(plan: PlanNode, *, mode: str = "production") -> bool:
    if plan.op in _SQL_LEAF_OPS: return False
    return _is_sql_capable(plan, mode=mode)


def annotate_node(plan: PlanNode, *, parent_sql: bool = False, mode: str = "production") -> PhysicalNode:
    if parent_sql and _is_sql_capable(plan, mode=mode):
        return PhysicalNode(kind=ExecKind.SQL, plan=plan,
                            children=tuple(annotate_node(c,parent_sql=True,mode=mode) for c in plan.inputs))
    if _is_extractable_sql_subtree(plan, mode=mode) and not parent_sql:
        return PhysicalNode(kind=ExecKind.SQL, plan=plan, sid=plan_cache_key(plan))
    children=tuple(annotate_node(c,parent_sql=False,mode=mode) for c in plan.inputs)
    if plan.op == "materialized_series":
        return PhysicalNode(kind=ExecKind.MATERIALIZED, plan=plan,
                            sid=str(plan.attrs.get("sid","")), children=children)
    return PhysicalNode(kind=ExecKind.PYTHON, plan=plan, children=children)


def lower_to_physical_plan(plan: PlanNode, *, mode: str = "production") -> PhysicalPlan:
    if _is_sql_capable(plan, mode=mode):
        return PhysicalPlan(root=plan, sql_subtrees={}, fully_sql=True)
    pending: dict[str, PlanNode] = {}
    def _transform(node: PlanNode) -> PlanNode:
        if _is_extractable_sql_subtree(node, mode=mode):
            sid=plan_cache_key(node); pending[sid]=node
            return PlanNode(op="materialized_series", inputs=(), attrs={"sid":sid})
        return PlanNode(node.op, tuple(_transform(c) for c in node.inputs), node.attrs)
    return PhysicalPlan(root=_transform(plan), sql_subtrees=pending, fully_sql=False)


def collect_sql_subtrees(plan: PlanNode, *, mode: str = "production") -> dict[str, PlanNode]:
    return lower_to_physical_plan(plan, mode=mode).sql_subtrees
