# -*- coding: utf-8
"""Plan 级性能退化守卫（正确但不可接受的路径不得进 production fastpath）。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanPerformanceBudget:
    max_parquet_scans: int = 1
    max_binary_joins: int = 3
    max_sql_cte_depth: int = 8
    max_row_inflation_ratio: float = 2.0
    max_intermediate_columns: int = 32


DEFAULT_PLAN_BUDGET = PlanPerformanceBudget()

SIMPLE_BINARY_OPS: frozenset[str] = frozenset(
    {"add", "subtract", "multiply", "divide", "maximum", "minimum"}
)


def simple_add_max_scans() -> int:
    """简单 add 不允许产生两个独立 parquet scan。"""
    return 1


def composite_must_not_recompute_same_subtree() -> bool:
    return True
