# -*- coding: utf-8 -*-
"""Polars 流式执行策略：智能决策 collect(streaming=True)。

基于算子分析自动判断计划是否支持流式执行，目标：
- 内存峰值降低 50-70%（大数据集场景）
- 保守白名单：仅对已验证安全的算子组合启用
- 自动回退：检测到阻塞算子时降级到全量 collect
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from planner.logical_plan import PlanNode


@dataclass(frozen=True)
class StreamingCapability:
    """流式能力评估结果。"""

    can_stream: bool
    """是否可安全启用 streaming=True。"""

    reason: str
    """不支持原因（调试用）或"ok"。"""

    blocking_ops: frozenset[str]
    """阻塞流式的算子集合。"""

    python_udf_ratio: float
    """Python UDF 算子占比（0.0-1.0）。"""


# 需要全局排序的算子（阻塞流式）
_REQUIRES_GLOBAL_SORT = frozenset({
    "ts_rank",  # 窗口内排序需要完整窗口
})

# 需要完整分组物化的算子（阻塞流式）
_REQUIRES_FULL_GROUP = frozenset({
    "rank", "rank_pct", "cs_pct_rank",  # 截面排序需要完整截面
    "zscore", "normalize",  # 标准化需要完整截面统计量
    "cs_quantile", "quantile",  # 分位数需要完整分布
    "cs_mad", "cs_mad_zscore",  # 中位数绝对偏差需要完整数据
})

# 递归状态算子（部分支持，保守阻塞）
_STATEFUL_CAUTIOUS = frozenset({
    "KAMA",  # Kaufman 自适应移动平均（复杂状态机）
    "MACD",  # 多重 EWM 嵌套
})

# Python UDF 算子（非纯 Polars，流式性能可能不佳）
_PYTHON_UDF_OPS = frozenset({
    # map_groups 类
    "expanding_std", "ewm_corr", "ewm_cov",
    "ts_kurt", "ts_moment", "ts_max_buildup",
    "expanding_rank", "causal_linear_extrapolate",

    # rolling_map 类
    "ts_skew", "ts_quantile", "ts_product",
    "ts_median_abs_deviation", "ts_mean_abs_deviation",
    "ts_argmax", "ts_argmin", "ts_time_slope",
    "ts_decay_linear",
})

# 已验证流式安全的算子（白名单）
_STREAMING_SAFE = frozenset({
    # 逐行变换（完全流式安全）
    "add", "subtract", "multiply", "divide", "power",
    "abs", "neg", "sign", "log", "exp", "sqrt",
    "floor", "ceil", "clip", "fillna", "nan_to_num",
    "is_null", "is_not_null", "is_nan", "is_finite", "is_infinite",
    "gt", "lt", "eq", "ge", "le", "ne", "and_", "or_", "not_",
    "where", "coalesce",

    # 分组窗口（按 instrument 分组流式）
    "ts_mean", "ts_sum", "ts_std", "ts_var", "ts_min", "ts_max", "ts_median",
    "ts_delay", "ts_delta", "ts_pct", "ts_ratio",
    "ts_corr", "ts_cov", "ts_beta", "vwap",
    "ts_ema", "ema", "ewm_std", "ewm_var",
    "ffill",

    # 累积（streaming 友好）
    "cum_sum", "cum_max", "cum_min", "cum_prod", "cum_delta",
    "expanding_sum", "expanding_mean",

    # 价量技术指标（纯 Polars 实现）
    "RSI_WILDER", "ATR_WILDER",

    # 叶子节点
    "column", "literal", "materialized_series", "plan_ref",
})


def _resolve_op(op: str) -> str:
    """解析算子别名至 canonical 名称。"""
    from cleaned_operators.registry import OperatorRegistry
    return OperatorRegistry._aliases.get(op, op)


def collect_plan_ops(plan: PlanNode) -> frozenset[str]:
    """递归收集计划中所有算子（canonical 名称）。"""
    ops = {_resolve_op(plan.op)}
    for child in plan.inputs:
        ops.update(collect_plan_ops(child))
    return frozenset(ops)


def analyze_streaming_capability(plan: PlanNode) -> StreamingCapability:
    """分析计划的流式执行能力。

    Args:
        plan: 待分析的逻辑计划根节点。

    Returns:
        StreamingCapability 评估结果。

    策略:
        1. 白名单优先：所有算子在 _STREAMING_SAFE → can_stream=True
        2. 阻塞检测：存在 GLOBAL_SORT/FULL_GROUP/STATEFUL → can_stream=False
        3. UDF 限制：Python UDF 占比 >30% → can_stream=False（性能考虑）
        4. 保守回退：未知算子 → can_stream=False
    """
    ops = collect_plan_ops(plan)

    # 检查阻塞算子
    blocking = ops & (_REQUIRES_GLOBAL_SORT | _REQUIRES_FULL_GROUP | _STATEFUL_CAUTIOUS)
    if blocking:
        return StreamingCapability(
            can_stream=False,
            reason=f"blocking_ops={sorted(blocking)[:3]}",  # 最多显示 3 个
            blocking_ops=frozenset(blocking),
            python_udf_ratio=0.0,
        )

    # 检查 Python UDF 占比
    udf_ops = ops & _PYTHON_UDF_OPS
    if ops:
        udf_ratio = len(udf_ops) / len(ops)
    else:
        udf_ratio = 0.0

    if udf_ratio > 0.3:
        return StreamingCapability(
            can_stream=False,
            reason=f"high_python_udf_ratio={udf_ratio:.2f}",
            blocking_ops=frozenset(),
            python_udf_ratio=udf_ratio,
        )

    # 检查未知算子（保守策略）
    unknown = ops - _STREAMING_SAFE - _PYTHON_UDF_OPS
    if unknown:
        return StreamingCapability(
            can_stream=False,
            reason=f"unknown_ops={sorted(unknown)[:3]}",
            blocking_ops=frozenset(),
            python_udf_ratio=udf_ratio,
        )

    # 所有检查通过
    return StreamingCapability(
        can_stream=True,
        reason="ok",
        blocking_ops=frozenset(),
        python_udf_ratio=udf_ratio,
    )


def should_use_streaming(
    plan: PlanNode,
    *,
    force: bool | None = None,
    min_rows: int = 10000,
) -> tuple[bool, str]:
    """决策是否使用 streaming collect。

    Args:
        plan: 逻辑计划。
        force: 强制开关（None=自动，True=强制启用，False=强制禁用）。
        min_rows: 最小行数阈值（小数据集不启用，开销大于收益）。

    Returns:
        (should_stream, reason): 是否启用及原因。

    Examples:
        >>> should_use_streaming(plan)
        (True, "auto:ok")

        >>> should_use_streaming(plan, force=False)
        (False, "force_disabled")
    """
    if force is True:
        return True, "force_enabled"
    if force is False:
        return False, "force_disabled"

    # 环境变量覆盖（运维紧急开关）
    import os
    env_disable = os.environ.get("FACTOR_ENGINE_POLARS_STREAMING", "").lower()
    if env_disable in {"0", "false", "off"}:
        return False, "env_disabled"

    # 自动决策
    capability = analyze_streaming_capability(plan)

    if not capability.can_stream:
        return False, f"auto:{capability.reason}"

    # 小数据集不启用（streaming 有固定开销）
    # 注：这里无法精确预估行数，启用后通过 runtime_stats 监控实际收益

    return True, "auto:ok"


def streaming_collect(
    lf: Any,  # pl.LazyFrame
    plan: PlanNode,
    *,
    force: bool | None = None,
) -> Any:  # pl.DataFrame
    """智能流式 collect：自动决策是否启用 streaming=True。

    Args:
        lf: Polars LazyFrame。
        plan: 对应的逻辑计划（用于能力分析）。
        force: 强制开关。

    Returns:
        Polars DataFrame。

    用法:
        >>> lf = compile_polars_long_lazy(plan, ctx)
        >>> df = streaming_collect(lf.frame, plan)
    """
    should_stream, reason = should_use_streaming(plan, force=force)

    if should_stream:
        return lf.collect(streaming=True), reason
    else:
        return lf.collect(), reason
