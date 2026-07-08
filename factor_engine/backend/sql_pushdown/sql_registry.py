# -*- coding: utf-8 -*-
"""SQL 下推能力与 OperatorRegistry 对齐。"""
from __future__ import annotations

from dataclasses import dataclass

from cleaned_operators.registry import OperatorRegistry
from planner.logical_plan import PlanNode

# 与 emitter 同步的 canonical 白名单
SQL_CAPABLE_CANONICALS: frozenset[str] = frozenset(
    {
        "column",
        "literal",
        "add",
        "subtract",
        "multiply",
        "divide",
        "neg",
        "abs",
        "sign",
        "log",
        "exp",
        "sqrt",
        "clip",
        "ts_mean",
        "ts_delay",
        "ts_delta",
        "ts_std",
        "ts_sum",
        "ts_max",
        "ts_min",
        "ts_pct",
        "ts_zscore",
        "ts_corr",
        "rank",
        "zscore",
        "scale",
        "cs_demean",
        "group_neutralize",
        "where",
        "if_else",
        "ts_median",
        "ts_var",
        "group_rank",
        "group_mean",
        "group_zscore",
        "winsorize",
        "group_winsorize",
        "ts_beta",
        "ts_mad",
        "ts_ema",
        "ts_rank",
        "ewm_mean",
        "ffill",
        "bfill",
        "fillna_const",
        "ts_decay_linear",
        "coalesce",
        "protected_div",
        "protected_log",
        "protected_sqrt",
        "nan_to_num",
        "fillna",
        "is_nan",
        "is_finite",
        "normalize",
        "group_normalize",
        "group_percentile",
        "group_decay_linear",
    }
)

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
