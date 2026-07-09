"""Polars 热路径算子集合（与 ``POLARS_PRODUCTION_SAFE`` 对齐）。"""

from __future__ import annotations

from typing import Any

from cleaned_operators.operator_policy import POLARS_PRODUCTION_SAFE

# 向后兼容别名
POLARS_TS_OPS: frozenset[str] = POLARS_PRODUCTION_SAFE
POLARS_HOT_OPS: frozenset[str] = POLARS_PRODUCTION_SAFE


def is_polars_hot_op(op: str) -> bool:
    return str(op) in POLARS_HOT_OPS


def is_polars_ts_op(op: str) -> bool:
    return str(op) in POLARS_TS_OPS


def is_polars_backend_ctx(ctx: Any) -> bool:
    runtime = getattr(ctx, "runtime_stats", None) or {}
    return runtime.get("backend") == "polars"
