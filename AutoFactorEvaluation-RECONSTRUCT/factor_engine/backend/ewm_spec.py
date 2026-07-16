# -*- coding: utf-8
"""EWM / Wilder 参数规格与校验。"""
from __future__ import annotations

from dataclasses import dataclass

from backend.plan_params import PlanParamError, parse_positive_int_literal
from planner.logical_plan import PlanNode


@dataclass(frozen=True)
class EwmSpec:
    span: int
    alpha: float
    adjust: bool = False
    min_periods: int = 1

    @classmethod
    def from_plan_node(cls, node: PlanNode, *, default_span: int = 20) -> "EwmSpec":
        """从 plan 解析 span；拒绝非整数/非正 span。"""
        raw = None
        for key in ("span", "window", "n", "periods"):
            if key in (node.attrs or {}) and node.attrs[key] is not None:
                raw = node.attrs[key]
                break
        if raw is None:
            for idx in range(1, len(node.inputs)):
                child = node.inputs[idx]
                if child.op == "literal":
                    val = child.attrs.get("value")
                    if val is not None:
                        raw = val
                        break
        span = parse_positive_int_literal(raw if raw is not None else default_span, label="span")
        alpha = 2.0 / (span + 1.0)
        min_p = 1
        if "min_periods" in (node.attrs or {}):
            min_p = parse_positive_int_literal(node.attrs["min_periods"], label="min_periods")
        adjust = bool((node.attrs or {}).get("adjust", False))
        return cls(span=span, alpha=alpha, adjust=adjust, min_periods=min_p)
