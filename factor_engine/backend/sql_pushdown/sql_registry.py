# -*- coding: utf-8 -*-
"""SQL 下推能力与 OperatorRegistry 对齐。"""
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
    """为 SQL 可编译 canonical 登记 ``backend='sql'`` 元数据。"""
    global _SQL_MARKERS_REGISTERED
    if _SQL_MARKERS_REGISTERED:
        return
    for canon in SQL_CAPABLE_CANONICALS:
        if canon in {"column", "literal"}:
            continue
        if "sql" in OperatorRegistry.backends_for(canon):
            continue
        OperatorRegistry.register(
            SqlCapableOperator(canon),
            canonical=canon,
            backend="sql",
            source="backend/sql_pushdown",
        )
    _SQL_MARKERS_REGISTERED = True


def resolve_canonical(op: str) -> str:
    return OperatorRegistry._aliases.get(op, op)


def is_sql_capable(plan: PlanNode) -> bool:
    """计划树是否全部由 SQL backend 支持。"""
    register_sql_backends()
    canon = resolve_canonical(plan.op)
    if canon not in SQL_CAPABLE_CANONICALS:
        return False
    return all(is_sql_capable(c) for c in plan.inputs)


def sql_backends_for(name: str) -> list[str]:
    register_sql_backends()
    return OperatorRegistry.backends_for(name)
