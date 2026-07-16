# -*- coding: utf-8
"""缺失值处理契约：coalesce / ffill / nan_to_num。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CoalesceNanPolicy = Literal["null_only", "treat_nan_as_missing"]
FfillLimitPolicy = Literal["unlimited", "limited"]


@dataclass(frozen=True)
class CoalesceSpec:
    """目标契约：``coalesce`` 仅处理 SQL NULL。

    第一阶段 Pandas 内核仍按 ``coalesce_(NaN→fallback)`` 对齐；Polars 须与 Pandas parity。
    """

    null_only: bool = True
    nan_policy: CoalesceNanPolicy = "null_only"
    phase1_follows_pandas_nan_coalesce: bool = True


@dataclass(frozen=True)
class NanToNumSpec:
    """第一阶段 ``nan_to_num(x, num)``：NULL/NaN/±Inf 均替换为同一常数。"""

    unified_fill: bool = True
    default_nan: float = 0.0


@dataclass(frozen=True)
class FfillSpec:
    """日频第一阶段：按 instrument 无限 forward fill（research 路径）。

    Production 目标：``ffill(limit=N)`` / ``max_age``；无限 ffill 须显式声明。
    """

    limit_policy: FfillLimitPolicy = "unlimited"
    cross_session: bool = False
    cross_trading_day: bool = True
    production_requires_limit: bool = True
    default_research_unlimited: bool = True


COALESCE_SPEC = CoalesceSpec()
NAN_TO_NUM_SPEC = NanToNumSpec()
FFILL_SPEC = FfillSpec()


def coalesce_null_only() -> bool:
    """目标契约：``coalesce`` 仅处理 NULL（长期 SQL 语义）。"""
    return COALESCE_SPEC.null_only


def coalesce_phase1_pandas_nan_coalesce() -> bool:
    """第一阶段：NaN 与 Pandas ``coalesce_`` 一致，视为可替换缺失。"""
    return COALESCE_SPEC.phase1_follows_pandas_nan_coalesce


def coalesce_nan_is_not_missing() -> bool:
    return COALESCE_SPEC.nan_policy == "null_only"


def nan_to_num_unified_fill() -> bool:
    """单参数形式：NaN/Inf/NULL 同一填充值。"""
    return NAN_TO_NUM_SPEC.unified_fill


def ffill_unlimited_phase1() -> bool:
    """第一阶段 ffill 无 limit（production 目标改为 limit=N）。"""
    return FFILL_SPEC.limit_policy == "unlimited"


def ffill_production_requires_limit() -> bool:
    return FFILL_SPEC.production_requires_limit
