# -*- coding: utf-8 -*-
"""A-share limit-price behavior operators.

All operators read daily panels and remain causal.  ``tick_tolerance`` follows
the same convention as ``limit_up_close`` in ``ashare/ops.py``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator


def _metadata(name: str, description: str, params: list[str], *, unit: str = "ratio") -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="ashare",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "ashare", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:trading_state",
            f"unit:{unit}", "cost:1",
        ],
    )


def _safe_ratio(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    den = denominator.replace(0, np.nan) if hasattr(denominator, "replace") else denominator
    out = numerator / den
    return out.replace([np.inf, -np.inf], np.nan)


@register_operator(
    name="ashare_limit_distance",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_distance",
    source="ashare.limit_ops",
    status="experimental",
)
class AshareLimitDistance(SeriesOperator):
    """收盘价相对涨停价的距离：close / upper_limit - 1。"""

    metadata = _metadata(
        "ashare_limit_distance",
        "收盘价相对涨停价的距离 close/upper_limit - 1。",
        ["close", "upper_limit"],
    )

    def _calculate_series(self, close: pd.DataFrame, upper_limit: pd.DataFrame, **_: Any) -> pd.DataFrame:
        return _safe_ratio(close, upper_limit) - 1.0


@register_operator(
    name="ashare_limit_touch",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_touch",
    source="ashare.limit_ops",
    status="experimental",
)
class AshareLimitTouch(SeriesOperator):
    """收盘在 tick 容差内触及涨停价。"""

    metadata = _metadata(
        "ashare_limit_touch",
        "收盘在 tick 容差内触及涨停价。",
        ["close", "upper_limit", "tick_tolerance"],
        unit="boolean",
    )

    def _calculate_series(self, close: pd.DataFrame, upper_limit: pd.DataFrame, tick_tolerance: float = 0.005, **_: Any) -> pd.DataFrame:
        tolerance = float(tick_tolerance)
        if tolerance < 0:
            raise ValueError("tick_tolerance must be non-negative")
        valid = close.notna() & upper_limit.notna()
        return (close >= upper_limit - tolerance).astype(float).where(valid)


@register_operator(
    name="ashare_limit_one_price",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_one_price",
    source="ashare.limit_ops",
    status="experimental",
)
class AshareLimitOnePrice(SeriesOperator):
    """一字涨停：收盘同时处于涨停与跌停边界（封死）。"""

    metadata = _metadata(
        "ashare_limit_one_price",
        "一字板：收盘同时达到涨跌停价。",
        ["close", "upper_limit", "lower_limit"],
        unit="boolean",
    )

    def _calculate_series(self, close: pd.DataFrame, upper_limit: pd.DataFrame, lower_limit: pd.DataFrame, **_: Any) -> pd.DataFrame:
        valid = close.notna() & upper_limit.notna() & lower_limit.notna()
        out = ((close >= upper_limit) & (close <= lower_limit)).astype(float)
        return out.where(valid)


@register_operator(
    name="ashare_limit_failed",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_failed",
    source="ashare.limit_ops",
    status="experimental",
)
class AshareLimitFailed(SeriesOperator):
    """冲高回落：日内曾触及涨停但收盘未封住（窗口内触及过涨停且当前收盘低于涨停）。"""

    metadata = _metadata(
        "ashare_limit_failed",
        "窗口内曾触及涨停但当前收盘未封住。",
        ["close", "upper_limit", "window"],
        unit="boolean",
    )

    def _calculate_series(self, close: pd.DataFrame, upper_limit: pd.DataFrame, window: int = 20, **_: Any) -> pd.DataFrame:
        w = int(window)
        touch = (close >= upper_limit).astype(float)
        ever_touched = touch.rolling(w, min_periods=1).max()
        valid = close.notna() & upper_limit.notna() & ever_touched.notna()
        out = ((ever_touched > 0) & (close < upper_limit)).astype(float)
        return out.where(valid)


@register_operator(
    name="ashare_limit_open_break",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_open_break",
    source="ashare.limit_ops",
    status="experimental",
)
class AshareLimitOpenBreak(SeriesOperator):
    """开盘即涨停：open 达到涨停价。"""

    metadata = _metadata(
        "ashare_limit_open_break",
        "开盘达到涨停价。",
        ["open", "upper_limit"],
        unit="boolean",
    )

    def _calculate_series(self, open_px: pd.DataFrame, upper_limit: pd.DataFrame, **_: Any) -> pd.DataFrame:
        valid = open_px.notna() & upper_limit.notna()
        return (open_px >= upper_limit).astype(float).where(valid)
