# -*- coding: utf-8 -*-
"""PlanNode -> PhysicalPlan: split certified SQL subtrees from Python/SourceRef execution."""
from __future__ import annotations

from enum import Enum

from planner.logical_plan import PlanNode
from planner.plan_hash import plan_cache_key
from planner.physical_plan import ExecKind, PhysicalNode, PhysicalPlan

_SQL_LEAF_OPS = frozenset({"column", "literal"})


class SourceRefStatus(str, Enum):
    """R20-091..093: SourceRef 三态判定。

    - ``NOT_SOURCE_REF``：名字不带 SourceRef 前缀（普通列 / literal）；
    - ``VALID_SOURCE_REF``：带前缀且 payload 严格可解码；
    - ``MALFORMED_SOURCE_REF``：带前缀但 payload 损坏（bad base64 / bad JSON /
      缺 table/field）—— production 必须 hard fail，绝不吞成「不是 SourceRef」。
    """

    NOT_SOURCE_REF = "not_source_ref"
    VALID_SOURCE_REF = "valid_source_ref"
    MALFORMED_SOURCE_REF = "malformed_source_ref"


def classify_source_ref(plan: PlanNode) -> SourceRefStatus:
    """三态 SourceRef 分类（R20-091..093）。

    ``decode_source_ref`` 的宽泛版在异常时返回 None，会把「损坏的 SourceRef」
    误判为「不是 SourceRef」，导致 SQL detector 吞掉畸形身份。这里用
    ``decode_source_ref_strict`` 严格区分：带前缀但无法解码 = MALFORMED。
    """
    if plan.op == "column":
        name = (plan.attrs or {}).get("name")
        if isinstance(name, str):
            from api.source_ref import looks_like_source_ref

            if looks_like_source_ref(name):
                from api.source_ref import decode_source_ref_strict

                try:
                    decode_source_ref_strict(name)
                    return SourceRefStatus.VALID_SOURCE_REF
                except (ValueError, TypeError):
                    return SourceRefStatus.MALFORMED_SOURCE_REF
    return SourceRefStatus.NOT_SOURCE_REF


def _contains_source_ref(plan: PlanNode, *, production: bool = False) -> bool:
    """子树是否含 SourceRef（三态判定；malformed 在 production 下 hard fail）。

    R20-091..093: 旧的 ``except Exception: pass`` 会吞掉损坏 SourceRef 的解码
    异常——损坏身份被当成「不是 SourceRef」，从而可能被错误地当作 SQL 物理列
    下推。现在 malformed SourceRef 在 production 下抛 ``ValueError``（fail
    closed）；research 下按 opaque 处理（非 SQL-capable）。
    """
    status = classify_source_ref(plan)
    if status is SourceRefStatus.MALFORMED_SOURCE_REF:
        if production:
            raise ValueError(
                "malformed SourceRef in plan: "
                f"{str((plan.attrs or {}).get('name', ''))!r} is not a decodable "
                "source identity; production SQL lowering must not swallow a "
                "damaged SourceRef"
            )
        return True
    if status is SourceRefStatus.VALID_SOURCE_REF:
        return True
    return any(
        _contains_source_ref(child, production=production) for child in plan.inputs
    )


def _is_sql_capable(plan: PlanNode, *, mode: str = "production") -> bool:
    # Opaque logical-source columns are not physical DuckDB columns.  Their
    # subtree must cross the SourceRef resolver before numerical execution.
    # R20-091..093: production mode 下 malformed SourceRef 抛 ValueError
    #（fail-closed）；research 下按 opaque 处理。
    if _contains_source_ref(plan, production=(mode == "production")):
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
            sid = plan_cache_key(node)
            pending[sid] = node
            # R20-459: ``materialized_series`` 占位必须携带被抽取子树根的 output
            # semantic contract（unit / grain / availability / price-basis /
            # source identity）—— 下游 typing / PIT / SQL 层把占位当完整子树读。
            return PlanNode(
                op="materialized_series",
                inputs=(),
                attrs={"sid": sid},
                semantic_attrs=dict(node.semantic_attrs),
                node_id=node.node_id,
            )
        return PlanNode(
            node.op,
            tuple(_transform(c) for c in node.inputs),
            node.attrs,
            dict(node.semantic_attrs),
            node_id=node.node_id,
        )
    return PhysicalPlan(root=_transform(plan), sql_subtrees=pending, fully_sql=False)


def collect_sql_subtrees(plan: PlanNode, *, mode: str = "production") -> dict[str, PlanNode]:
    return lower_to_physical_plan(plan, mode=mode).sql_subtrees
