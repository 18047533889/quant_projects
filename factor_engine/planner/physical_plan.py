# -*- coding: utf-8 -*-
"""物理计划：标注 SQL / Python / 物化子树执行方式。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from planner.logical_plan import PlanNode


class ExecKind(str, Enum):
    """节点执行后端种类。"""

    SQL = "sql"  # 整棵 SQL 可编译子树
    PYTHON = "python"  # pandas/polars 算子
    MATERIALIZED = "materialized"  # 已 SQL 预计算的 Series 引用


@dataclass(frozen=True)
class PhysicalNode:
    """``sql_lowerer`` 输出：带执行语义的计划节点。"""

    kind: ExecKind
    plan: PlanNode
    sid: str | None = None
    children: tuple[PhysicalNode, ...] = field(default_factory=tuple)


@dataclass
class PhysicalPlan:
    """混合执行物理计划（root + 待预计算 SQL 子树）。"""

    root: PlanNode
    sql_subtrees: dict[str, PlanNode] = field(default_factory=dict)
    fully_sql: bool = False
