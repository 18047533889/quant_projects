# -*- coding: utf-8
"""Plan 参数解析与校验（跨 PolarsLong / DuckDB 统一语义）。"""
from __future__ import annotations

from planner.logical_plan import PlanNode


class PlanParamError(ValueError):
    """非法 plan 参数（window / mode 等）。"""


def parse_positive_int_literal(raw: object, *, label: str = "window") -> int:
    """解析正整数 literal；拒绝非整数浮点、NaN、Inf、非正数。"""
    if raw is None:
        raise PlanParamError(f"{label} 未提供")
    if isinstance(raw, bool):
        raise PlanParamError(f"{label} 不能为 boolean")
    if isinstance(raw, float):
        if raw != raw or raw in (float("inf"), float("-inf")):
            raise PlanParamError(f"{label} 不能为 NaN/Inf")
        if raw != int(raw):
            raise PlanParamError(f"{label} 必须为整数 literal，收到 {raw!r}")
        val = int(raw)
    elif isinstance(raw, int):
        val = raw
    else:
        raise PlanParamError(f"{label} 必须为整数 literal，收到 {type(raw).__name__}")
    if val <= 0:
        raise PlanParamError(f"{label} 必须 > 0，收到 {val}")
    return val


def window_from_plan_node(node: PlanNode, *, default: int = 3) -> int:
    """从 attrs 或 positional literal 解析滚动窗口（Polars/DuckDB 共用）。"""
    from backend.window_spec import WindowSpec

    return WindowSpec.from_plan_node(node, default_size=default).size


def window_spec_from_plan_node(node: PlanNode, *, default: int = 3):
    """返回完整 ``WindowSpec``（含 min_periods / ddof / null_policy）。"""
    from backend.window_spec import WindowSpec

    return WindowSpec.from_plan_node(node, default_size=default)


def int_mode_from_plan_node(
    node: PlanNode,
    *,
    input_index: int,
    default: int = 0,
    label: str = "mode",
) -> int:
    """读取 positional literal 整数 mode（``input_index`` 为 inputs 下标）。"""
    if input_index < len(node.inputs):
        child = node.inputs[input_index]
        if child.op == "literal":
            raw = child.attrs.get("value")
            if raw is not None:
                if isinstance(raw, bool):
                    return default
                if isinstance(raw, (int, float)) and raw == int(raw):
                    return int(raw)
    for key in ("mode", label):
        if key in (node.attrs or {}) and node.attrs[key] is not None:
            return int(node.attrs[key])
    return default
