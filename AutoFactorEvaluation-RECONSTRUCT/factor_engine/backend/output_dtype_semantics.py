# -*- coding: utf-8
"""Long-table 输出 dtype 契约（第一阶段）。"""
from __future__ import annotations

from typing import Literal

LogicalRepr = Literal["float64_0_1_null"]


def logical_output_representation() -> LogicalRepr:
    """gt/is_null 等逻辑算子：Float64 0/1/NULL，非 native Boolean。"""
    return "float64_0_1_null"


def count_output_is_float64_phase1() -> bool:
    """``count`` 等本应为 Int64，第一阶段仍统一 Float64。"""
    return True
