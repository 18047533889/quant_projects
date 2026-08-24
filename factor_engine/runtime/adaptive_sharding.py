# -*- coding: utf-8 -*-
"""R27-084..093: 自适应分片 —— shard legality 是语义属性，shard size 动态决定。

要点
    - R27-084：不能随便按股票拆——含 rank / zscore / neutralize / group /
      cross-sectional regression / market/global state / relation graph 的因子
      按 asset shard 会**改变结果**。
    - R27-085：编译计划后得到 shard legality：TIME_SHARD_SAFE /
      ASSET_SHARD_SAFE / GROUP_SHARD_SAFE / FACTOR_SHARD_SAFE /
      SESSION_SHARD_SAFE / STATEFUL_CHECKPOINT_REQUIRED。
    - R27-086..091：factor_id / time / asset / session / stateful / hybrid shard
      各自的适用性。
    - R27-092/093：shard size 不是固定「每月 / 64 bucket」，而是
      ``predicted_peak_memory <= shard_memory_target``，再满足 calendar/session/
      group 边界。
    - R27-155/256：每种合法 shard 的 sharded-compute-merge 必须等价 full compute。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from factor_engine.planner.physical_factor_dag import (
    SHARD_ASSET_SAFE,
    SHARD_FACTOR_SAFE,
    SHARD_GROUP_SAFE,
    SHARD_SESSION_SAFE,
    SHARD_STATEFUL_CHECKPOINT,
    SHARD_TIME_SAFE,
)

#: 跨截面 / 全市场 / group 语义算子 → 禁止 asset shard（R27-084）
_CROSS_SECTION_OPS = frozenset(
    {
        "rank",
        "zscore",
        "cs_mean",
        "cs_std",
        "cs_sum",
        "cs_count",
        "cs_mad",
        "cs_mad_zscore",
        "cs_demean",
        "cs_quantile",
        "cs_pct_rank",
        "neutralize",
        "group_rank",
        "group_mean",
        "group_std",
        "group_zscore",
        "group_neutralize",
        "group_winsorize",
        "group_percentile",
        "group_normalize",
        "cs_regression",
        "cs_resid",
        "cs_multi_resid",
        "cs_wls_resid",
        "size_neutralize",
        "industry_size_neutralize",
    }
)


@dataclass(frozen=True)
class ShardLegality:
    """一个 task 的 shard 合法性判定（R27-085/245/246）。"""

    time_shard_safe: bool
    asset_shard_safe: bool
    group_shard_safe: bool
    factor_shard_safe: bool
    session_shard_safe: bool
    stateful_checkpoint_required: bool
    reason: str = ""

    @property
    def best_dimension(self) -> str:
        if self.time_shard_safe:
            return "time"
        if self.session_shard_safe:
            return "session"
        if self.factor_shard_safe:
            return "factor"
        return "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_shard_safe": self.time_shard_safe,
            "asset_shard_safe": self.asset_shard_safe,
            "group_shard_safe": self.group_shard_safe,
            "factor_shard_safe": self.factor_shard_safe,
            "session_shard_safe": self.session_shard_safe,
            "stateful_checkpoint_required": self.stateful_checkpoint_required,
            "reason": self.reason,
            "best_dimension": self.best_dimension,
        }


def _plan_ops(plan: Any) -> set[str]:
    ops: set[str] = set()
    stack = [plan]
    while stack:
        node = stack.pop()
        op = str(getattr(node, "op", "") or "")
        if op:
            ops.add(op)
        for child in getattr(node, "inputs", []) or []:
            stack.append(child)
    return ops


def classify_shard_legality(plan: Any) -> ShardLegality:
    """从计划推断 shard 合法性（R27-085 编译后调用）。

    - asset shard：只允许纯 instrument-separable 时序因子（无任何截面算子）。
    - time shard：截面 / group / 全市场因子安全（每日期保留完整 universe）。
    - session shard：intraday 优先 trade_date/session（R27-089，完整一天）。
    - stateful：有 checkpoint 才能 segment incremental，否则禁止任意切分
      （R27-090，由调用方在计划无 checkpoint 时强制禁止）。
    """
    ops = _plan_ops(plan)
    has_cross_section = bool(ops & _CROSS_SECTION_OPS)
    time_safe = True
    asset_safe = not has_cross_section
    session_safe = True
    factor_safe = True
    # time shard 需要 lookback overlap；这里保守：含滚动时序算子的 task 仍然
    # time-safe（wave/执行层负责 overlap）。
    reason = ""
    if has_cross_section:
        reason = "contains cross-sectional ops (rank/zscore/neutralize/group); asset-shard unsafe"
    return ShardLegality(
        time_shard_safe=time_safe,
        asset_shard_safe=asset_safe,
        group_shard_safe=time_safe,
        factor_shard_safe=factor_safe,
        session_shard_safe=session_safe,
        stateful_checkpoint_required=False,
        reason=reason,
    )


def adaptive_shard_size(
    *,
    predicted_peak_bytes: int,
    shard_memory_target: int,
    calendar_buckets: int = 12,
) -> int:
    """R27-092/093：``split_factor ≈ ceil(predicted / target)``，再 clamp。

    current shard 估计 20GB、target 8GB → split_factor ≈ ceil(20/8) = 3。
    ``calendar_buckets`` 是可供切分的日历单位上限（如月份数），split 不能超过它。
    """
    if shard_memory_target <= 0:
        return 1
    split = max(1, (predicted_peak_bytes + shard_memory_target - 1) // shard_memory_target)
    return min(max(1, calendar_buckets), split)


def shard_split_factor(
    legality: ShardLegality,
    *,
    predicted_peak_bytes: int,
    shard_memory_target: int,
    calendar_buckets: int = 12,
) -> int:
    """返回某个 task 的合法 shard 份数（不合法返回 1=不切）。"""
    if not (legality.time_shard_safe or legality.session_shard_safe or legality.factor_shard_safe):
        return 1
    return adaptive_shard_size(
        predicted_peak_bytes=predicted_peak_bytes,
        shard_memory_target=shard_memory_target,
        calendar_buckets=calendar_buckets,
    )
