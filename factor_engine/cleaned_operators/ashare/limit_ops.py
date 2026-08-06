# -*- coding: utf-8 -*-
"""A-share limit-price behavior operators (2026-08 audit fixes).

All operators read daily panels and remain causal.  ``tick_tolerance`` follows
the same convention as ``limit_up_close`` in ``ashare/ops.py``.

Semantics (corrected):
- ``ashare_limit_up_touch``: intraday high touched the upper limit.
- ``ashare_limit_down_touch``: intraday low touched the lower limit.
- ``ashare_limit_one_price``: one-price (一字) board — open/high/low/close all
  sit at the same limit price.
- ``ashare_limit_failed``: intraday touched the limit but failed to hold at
  close (炸板).
- ``ashare_open_at_upper_limit``: opened at the upper limit.
- ``ashare_limit_open_failed``: opened at the upper limit then broke below it
  intraday (开板).
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


def _tolerance(value: Any) -> float:
    tolerance = float(value)
    if tolerance < 0:
        raise ValueError("tick_tolerance must be non-negative")
    return tolerance


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
    name="ashare_limit_up_touch",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_up_touch",
    source="ashare.limit_ops",
    status="experimental",
)
class AshareLimitUpTouch(SeriesOperator):
    """日内最高价触及涨停价：high >= upper_limit - tolerance。"""

    metadata = _metadata(
        "ashare_limit_up_touch",
        "日内最高价触及涨停价。",
        ["high", "upper_limit", "tick_tolerance"],
        unit="boolean",
    )

    def _calculate_series(self, high: pd.DataFrame, upper_limit: pd.DataFrame, tick_tolerance: float = 0.005, **_: Any) -> pd.DataFrame:
        tolerance = _tolerance(tick_tolerance)
        valid = high.notna() & upper_limit.notna()
        return (high >= upper_limit - tolerance).astype(float).where(valid)


@register_operator(
    name="ashare_limit_down_touch",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_down_touch",
    source="ashare.limit_ops",
    status="experimental",
)
class AshareLimitDownTouch(SeriesOperator):
    """日内最低价触及跌停价：low <= lower_limit + tolerance。"""

    metadata = _metadata(
        "ashare_limit_down_touch",
        "日内最低价触及跌停价。",
        ["low", "lower_limit", "tick_tolerance"],
        unit="boolean",
    )

    def _calculate_series(self, low: pd.DataFrame, lower_limit: pd.DataFrame, tick_tolerance: float = 0.005, **_: Any) -> pd.DataFrame:
        tolerance = _tolerance(tick_tolerance)
        valid = low.notna() & lower_limit.notna()
        return (low <= lower_limit + tolerance).astype(float).where(valid)


@register_operator(
    name="ashare_limit_one_price",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_one_price",
    source="ashare.limit_ops",
    status="experimental",
)
class AshareLimitOnePrice(SeriesOperator):
    """一字板：open/high/low/close 全部处于同一涨（跌）停价（tick 容差内）。"""

    metadata = _metadata(
        "ashare_limit_one_price",
        "一字板：OHLC 全部处于同一涨跌停价。",
        ["open", "high", "low", "close", "upper_limit", "lower_limit", "side", "tick_tolerance"],
        unit="boolean",
    )

    def _calculate_series(
        self,
        open_px: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        upper_limit: pd.DataFrame,
        lower_limit: pd.DataFrame,
        side: str = "up",
        tick_tolerance: float = 0.005,
        **_: Any,
    ) -> pd.DataFrame:
        tolerance = _tolerance(tick_tolerance)
        direction = str(side).lower()
        limit = upper_limit if direction == "up" else lower_limit
        valid = (
            open_px.notna() & high.notna() & low.notna() & close.notna() & limit.notna()
        )
        at = (
            (open_px >= limit - tolerance)
            & (open_px <= limit + tolerance)
            & (high >= limit - tolerance)
            & (high <= limit + tolerance)
            & (low >= limit - tolerance)
            & (low <= limit + tolerance)
            & (close >= limit - tolerance)
            & (close <= limit + tolerance)
        )
        if direction == "up":
            out = at & (close >= upper_limit - tolerance)
        elif direction == "down":
            out = at & (close <= lower_limit + tolerance)
        else:
            raise ValueError("side must be 'up' or 'down'")
        return out.astype(float).where(valid)


@register_operator(
    name="ashare_limit_failed",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_failed",
    source="ashare.limit_ops",
    status="experimental",
)
class AshareLimitFailed(SeriesOperator):
    """当日炸板：盘中触及涨停（high >= 上限-容差）但收盘未封住（close < 上限-容差）。"""

    metadata = _metadata(
        "ashare_limit_failed",
        "当日触及涨停但收盘未封住。",
        ["high", "close", "upper_limit", "tick_tolerance"],
        unit="boolean",
    )

    def _calculate_series(self, high: pd.DataFrame, close: pd.DataFrame, upper_limit: pd.DataFrame, tick_tolerance: float = 0.005, **_: Any) -> pd.DataFrame:
        tolerance = _tolerance(tick_tolerance)
        valid = high.notna() & close.notna() & upper_limit.notna()
        touched = high >= upper_limit - tolerance
        held = close >= upper_limit - tolerance
        return (touched & ~held).astype(float).where(valid)


@register_operator(
    name="ashare_open_at_upper_limit",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_open_at_upper_limit",
    source="ashare.limit_ops",
    status="experimental",
)
class AshareOpenAtUpperLimit(SeriesOperator):
    """开盘即达涨停价：open >= upper_limit - tolerance。"""

    metadata = _metadata(
        "ashare_open_at_upper_limit",
        "开盘达到涨停价。",
        ["open", "upper_limit", "tick_tolerance"],
        unit="boolean",
    )

    def _calculate_series(self, open_px: pd.DataFrame, upper_limit: pd.DataFrame, tick_tolerance: float = 0.005, **_: Any) -> pd.DataFrame:
        tolerance = _tolerance(tick_tolerance)
        valid = open_px.notna() & upper_limit.notna()
        return (open_px >= upper_limit - tolerance).astype(float).where(valid)


@register_operator(
    name="ashare_limit_open_failed",
    category="ashare",
    business_category="trading_state",
    canonical="ashare_limit_open_failed",
    source="ashare.limit_ops",
    status="experimental",
)
class AshareLimitOpenFailed(SeriesOperator):
    """开盘涨停后开板：开盘达涨停价，但盘中跌破涨停价。"""

    metadata = _metadata(
        "ashare_limit_open_failed",
        "开盘涨停后盘中开板（low 跌破涨停价）。",
        ["open", "low", "upper_limit", "tick_tolerance"],
        unit="boolean",
    )

    def _calculate_series(self, open_px: pd.DataFrame, low: pd.DataFrame, upper_limit: pd.DataFrame, tick_tolerance: float = 0.005, **_: Any) -> pd.DataFrame:
        tolerance = _tolerance(tick_tolerance)
        valid = open_px.notna() & low.notna() & upper_limit.notna()
        opened_at_limit = open_px >= upper_limit - tolerance
        broke = low < upper_limit - tolerance
        return (opened_at_limit & broke).astype(float).where(valid)
