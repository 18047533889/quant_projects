# -*- coding: utf-8
"""统一滚动窗口语义（Pandas / PolarsLong / DuckDB 共用 WindowSpec）。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from factor_engine.backend.plan_params import PlanParamError, parse_positive_int_literal
from factor_engine.planner.logical_plan import PlanNode


class MissingWindowPolicy(Enum):
    """窗口内缺失值（NaN/SQL NULL）的参与策略（#359）。

    * ``PROPAGATE`` — 缺失值参与传播（SQL ``RESPECT NULLS`` 语义）
    * ``IGNORE`` — 忽略缺失值（SQL ``IGNORE NULLS`` 语义）
    * ``ZERO`` — 缺失值按 0 参与聚合
    """

    PROPAGATE = "propagate"
    IGNORE = "ignore"
    ZERO = "zero"


@dataclass(frozen=True)
class WindowSpec:
    """滚动窗口契约（第一阶段 rolling primitive 统一入口）。

    ``nan_policy``（#360）决定 NaN/SQL NULL 在窗口内如何参与聚合：
    ``propagate``（RESPECT NULLS）、``ignore``（IGNORE NULLS）、``zero``。
    SQL emitter 依据该字段映射 ``IGNORE NULLS`` / ``RESPECT NULLS`` 语义。
    """

    size: int
    min_periods: int = 1
    closed: str = "right"
    ddof: int = 1
    null_policy: MissingWindowPolicy = MissingWindowPolicy.IGNORE
    nan_policy: str = "propagate"

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
        # #361：SQL 只实现 population/sample 两档（ddof ∈ {0, 1}），
        # 别假装支持任意 ddof。
        if self.ddof not in {0, 1}:
            raise PlanParamError(f"window.ddof 仅支持 0 或 1，收到 {self.ddof}")
        # #359：null_policy 统一为 MissingWindowPolicy 枚举。字符串只在 parser
        # 边界映射：``"skip"`` 是旧外部别名 → IGNORE。
        np = self.null_policy
        if isinstance(np, str):
            if np == "skip":
                np = MissingWindowPolicy.IGNORE
            elif np == "ignore":
                np = MissingWindowPolicy.IGNORE
            elif np == "propagate":
                np = MissingWindowPolicy.PROPAGATE
            elif np == "zero":
                np = MissingWindowPolicy.ZERO
            else:
                raise PlanParamError(f"window.null_policy 非法: {np!r}")
        elif not isinstance(np, MissingWindowPolicy):
            raise PlanParamError(f"window.null_policy 非法: {np!r}")
        object.__setattr__(self, "null_policy", np)
        # #360：nan_policy 合法集 {propagate, ignore, zero}。
        if self.nan_policy not in {"propagate", "ignore", "zero"}:
            raise PlanParamError(f"window.nan_policy 非法: {self.nan_policy!r}")

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
                if child.op in {"column", "materialized_series", "plan_ref"}:
                    continue
                if child.op != "literal":
                    raise PlanParamError(
                        f"window 必须为整数 literal，收到动态输入 {child.op!r}"
                    )
                val = child.attrs.get("value")
                if val is None:
                    continue
                size = parse_positive_int_literal(val, label="window")
                found_window = True
                break
            if not found_window:
                size = default_size

        min_periods = default_min_periods
        if "min_periods" in attrs and attrs["min_periods"] is not None:
            min_periods = parse_positive_int_literal(attrs["min_periods"], label="min_periods")

        ddof = default_ddof
        if "ddof" in attrs and attrs["ddof"] is not None:
            raw = attrs["ddof"]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise PlanParamError("window.ddof 必须为整数 literal")
            if isinstance(raw, float) and raw != int(raw):
                raise PlanParamError(f"window.ddof 必须为整数 literal，收到 {raw}")
            ddof = int(raw)

        closed = str(attrs.get("closed", "right"))
        null_policy = str(attrs.get("null_policy", "skip"))
        nan_policy = str(attrs.get("nan_policy", "propagate"))
        return cls(
            size=size,
            min_periods=min_periods,
            closed=closed,
            ddof=ddof,
            null_policy=null_policy,
            nan_policy=nan_policy,
        )
