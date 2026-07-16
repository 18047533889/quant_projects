# -*- coding: utf-8 -*-
"""SQL 下推能力与 OperatorRegistry 对齐。

维护 SQL 可编译 canonical 集合，并为每个算子登记 ``backend='sql'`` 占位元数据，
供 planner / emitter 判断计划树是否可下推。
"""
from __future__ import annotations

from dataclasses import dataclass

from cleaned_operators.registry import OperatorRegistry
from planner.logical_plan import PlanNode

from backend.sql_tiers import (
    SQL_IMPLEMENTED_CANONICALS,
    SQL_PARITY_VERIFIED_CANONICALS,
    SQL_PRODUCTION_SAFE_CANONICALS,
)

# 向后兼容：emitter 有实现 ≠ production-safe
SQL_CAPABLE_CANONICALS: frozenset[str] = SQL_IMPLEMENTED_CANONICALS

__all__ = [
    "SQL_CAPABLE_CANONICALS",
    "SQL_IMPLEMENTED_CANONICALS",
    "SQL_PARITY_VERIFIED_CANONICALS",
    "SQL_PRODUCTION_SAFE_CANONICALS",
    "SqlCapableOperator",
    "is_sql_capable",
    "register_sql_backends",
    "resolve_canonical",
    "sql_backends_for",
]

_SQL_MARKERS_REGISTERED = False
_SQL_MARKERS_INCOMPLETE = False


@dataclass(frozen=True)
class SqlCapableOperator:
    """Registry 中 ``backend='sql'`` 的占位算子（编译走 emitter）。"""

    canonical: str

    @property
    def metadata(self):
        from cleaned_operators.base import OperatorMetadata

        return OperatorMetadata(
            name=self.canonical,
            category="sql",
            description=f"SQL 下推算子 {self.canonical}",
            param_names=[],
            return_type="series",
            tags=["sql", "pushdown"],
        )


def register_sql_backends() -> None:
    """为 SQL 可编译 canonical 登记 ``backend='sql'`` 元数据。

    可在 ``load_all()`` 之前被 ``sql_pushdown`` 包 import 触发；若当时尚无
    runtime 实现，会标记 incomplete，并在后续再次调用时补登记。
    """
    global _SQL_MARKERS_REGISTERED, _SQL_MARKERS_INCOMPLETE
    if _SQL_MARKERS_REGISTERED and not _SQL_MARKERS_INCOMPLETE:
        return
    incomplete = False
    for canon in SQL_CAPABLE_CANONICALS:
        if canon in {"column", "literal"}:
            continue
        # Attach sql marker to the resolved runtime canonical when possible, so we
        # never create an empty new primary name that later blocks rename merges.
        resolved = OperatorRegistry._aliases.get(canon, canon)
        if resolved in OperatorRegistry._operators:
            target = resolved
        elif canon in OperatorRegistry._operators:
            target = canon
        else:
            incomplete = True
            continue
        if "sql" in OperatorRegistry.backends_for(target):
            continue
        OperatorRegistry.register(
            SqlCapableOperator(target),
            canonical=target,
            backend="sql",
            source="backend/sql_pushdown",
        )
    _SQL_MARKERS_REGISTERED = True
    _SQL_MARKERS_INCOMPLETE = incomplete


def resolve_canonical(op: str) -> str:
    """将算子别名解析为 canonical 名称。"""
    return OperatorRegistry._aliases.get(op, op)


def is_sql_capable(plan: PlanNode) -> bool:
    """递归判断计划树是否全部由 SQL backend 支持的算子构成。"""
    register_sql_backends()
    canon = resolve_canonical(plan.op)
    if canon not in SQL_CAPABLE_CANONICALS:
        return False
    return all(is_sql_capable(c) for c in plan.inputs)


def sql_backends_for(name: str) -> list[str]:
    """返回算子名在 OperatorRegistry 中登记的后端列表（含 ``sql``）。"""
    register_sql_backends()
    return OperatorRegistry.backends_for(name)
