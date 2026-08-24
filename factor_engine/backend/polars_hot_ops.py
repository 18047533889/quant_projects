"""Polars 热路径算子集合（与 ``POLARS_PRODUCTION_SAFE`` 对齐）。"""

from __future__ import annotations

from typing import Any

from factor_engine.cleaned_operators.operator_policy import POLARS_PRODUCTION_SAFE

# 向后兼容别名
POLARS_TS_OPS: frozenset[str] = POLARS_PRODUCTION_SAFE
POLARS_HOT_OPS: frozenset[str] = POLARS_PRODUCTION_SAFE


def is_polars_hot_op(op: str) -> bool:
    """判断算子是否在 Polars 热路径白名单内。

    参数:
        op: 算子名称。

    返回:
        是否在 ``POLARS_HOT_OPS`` 集合内。
    """
    return str(op) in POLARS_HOT_OPS


def is_polars_ts_op(op: str) -> bool:
    """判断算子是否在 Polars 时序热路径白名单内。

    参数:
        op: 算子名称。

    返回:
        是否在 ``POLARS_TS_OPS`` 集合内。
    """
    return str(op) in POLARS_TS_OPS


def is_polars_backend_ctx(ctx: Any) -> bool:
    """判断执行上下文当前是否使用 Polars backend。

    参数:
        ctx: 执行上下文。

    返回:
        ``runtime_stats.backend`` 为 ``polars`` 时为 ``True``。
    """
    runtime = getattr(ctx, "runtime_stats", None) or {}
    return runtime.get("backend") == "polars"
