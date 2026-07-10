# -*- coding: utf-8
"""统一滚动窗口语义（Pandas / PolarsLong / DuckDB 共用 WindowSpec）。"""
from __future__ import annotations

from dataclasses import dataclass

from backend.plan_params import PlanParamError, parse_positive_int_literal
from planner.logical_plan import PlanNode


@dataclass(frozen=True)
class WindowSpec:
    """滚动窗口契约（第一阶段 rolling primitive 统一入口）。"""

    size: int
    min_periods: int = 1
    closed: str = "right"
    ddof: int = 1
    null_policy: str = "skip"

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise PlanParamError(f"window.size 必须 > 0，收到 {self.size}")
        if self.min_periods <= 0:
            raise PlanParamError(f"window.min_periods 必须 > 0，收到 {self.min_periods}")
        if self.min_periods > self.size:
            raise PlanParamError(
                f"window.min_periods({self.min_periods}) 不能大于 size({self.size})"
            )
        if self.closed not in {"right", "left", "both", "neither"}:
            raise PlanParamError(f"window.closed 非法: {self.closed!r}")
        if self.ddof < 0:
            raise PlanParamError(f"window.ddof 不能为负: {self.ddof}")
        if self.null_policy not in {"skip", "propagate", "zero"}:
            raise PlanParamError(f"window.null_policy 非法: {self.null_policy!r}")

    @classmethod
    def from_plan_node(
        cls,
        node: PlanNode,
        *,
        default_size: int = 3,
        default_min_periods: int = 1,
        default_ddof: int = 1,
    ) -> WindowSpec:
        """从 plan 节点解析 WindowSpec（拒绝非 literal 动态窗口）。"""
        attrs = node.attrs or {}
        size = default_size
        for key in ("d", "window", "n", "periods", "span"):
            if key in attrs and attrs[key] is not None:
                size = parse_positive_int_literal(attrs[key], label=key)
                break
        else:
            found_window = False
            for idx in range(1, len(node.inputs)):
                child = node.inputs[idx]
                if child.op != "literal":
                    if child.op not in {"column", "materialized_series", "plan_ref"}:
                        raise PlanParamError(
                            f"window 必须为整数 literal，收到动态输入 {child.op!r}"
                        )
                    continue
                val = child.attrs.get("value")
                if val is None:
                    continue
                size = parse_positive_int_literal(val, label="window")
                found_window = True
                break
            if not found_window and default_size == default_size:
                size = default_size

        min_periods = default_min_periods
        if "min_periods" in attrs and attrs["min_periods"] is not None:
            min_periods = parse_positive_int_literal(attrs["min_periods"], label="min_periods")

        ddof = default_ddof
        if "ddof" in attrs and attrs["ddof"] is not None:
            raw = attrs["ddof"]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise PlanParamError("window.ddof 必须为整数 literal")
            ddof = int(raw)

        closed = str(attrs.get("closed", "right"))
        null_policy = str(attrs.get("null_policy", "skip"))
        return cls(
            size=size,
            min_periods=min_periods,
            closed=closed,
            ddof=ddof,
            null_policy=null_policy,
        )
