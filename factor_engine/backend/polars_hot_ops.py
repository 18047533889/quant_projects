"""Polars 热路径算子集合（Top 20% 覆盖高频因子）。"""

from __future__ import annotations

from typing import Any

# 时序类：PolarsBackend 优先 native polars（与 pandas 对齐良好）
POLARS_TS_OPS: frozenset[str] = frozenset(
    {
        "ts_mean",
        "ts_sum",
        "ts_std",
        "ts_min",
        "ts_max",
        "ts_delay",
        "ts_delta",
        "ts_rank",
        "ts_corr",
        "ts_cov",
        "decay_linear",
        "abs",
        "log",
        "sign",
        "neg",
    }
)

# 全量热算子（含截面；截面可能与 pandas 有细微数值差）
POLARS_HOT_OPS: frozenset[str] = POLARS_TS_OPS | frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "rank",
        "zscore",
        "scale",
        "winsorize",
    }
)


def is_polars_hot_op(op: str) -> bool:
    return str(op) in POLARS_HOT_OPS


def is_polars_ts_op(op: str) -> bool:
    return str(op) in POLARS_TS_OPS


def is_polars_backend_ctx(ctx: Any) -> bool:
    runtime = getattr(ctx, "runtime_stats", None) or {}
    return runtime.get("backend") == "polars"
