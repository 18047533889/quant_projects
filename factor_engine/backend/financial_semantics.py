# -*- coding: utf-8
"""金融算子公开定义：vwap / volatility / log_returns。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

VwapKind = Literal["rolling_vwap"]
VolatilityKind = Literal["rolling_std_annualized"]
LogReturnKind = Literal["log_price_ratio"]


@dataclass(frozen=True)
class VwapSpec:
    """``vwap(price, volume, window)`` = 滚动 sum(price*volume)/sum(volume)。"""

    kind: VwapKind = "rolling_vwap"
    zero_volume_is_null: bool = True
    null_volume_is_null: bool = True
    negative_volume_is_null: bool = False
    reset_per_session: bool = False
    accumulate_in: Literal["float64", "decimal128"] = "float64"
    use_compensated_sum: bool = False


@dataclass(frozen=True)
class VolatilitySpec:
    """``volatility(x, window)`` = rolling_std(x, ddof=1) * sqrt(252)。

    输入 **x 为收益率序列**（调用方负责 log_returns / pct_change），不内部计算收益。
    """

    kind: VolatilityKind = "rolling_std_annualized"
    input_is_return: bool = True
    annualization_factor: float = 252.0**0.5
    ddof: int = 1
    min_periods_ratio: float = 0.5
    zero_vol_output: Literal["zero", "null"] = "zero"


@dataclass(frozen=True)
class LogReturnsSpec:
    """``log_returns(price)`` = log(price / lag(price, 1))。"""

    kind: LogReturnKind = "log_price_ratio"
    non_positive_is_null: bool = True
    null_propagates: bool = True


VWAP_SPEC = VwapSpec()
VOLATILITY_SPEC = VolatilitySpec()
LOG_RETURNS_SPEC = LogReturnsSpec()


def vwap_is_rolling() -> bool:
    return VWAP_SPEC.kind == "rolling_vwap"


def volatility_annualization_factor() -> float:
    return VOLATILITY_SPEC.annualization_factor


def log_returns_non_positive_is_null() -> bool:
    return LOG_RETURNS_SPEC.non_positive_is_null


def vwap_accumulate_float64() -> bool:
    return VWAP_SPEC.accumulate_in == "float64"
