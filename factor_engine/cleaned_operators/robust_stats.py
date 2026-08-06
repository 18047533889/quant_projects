# -*- coding: utf-8 -*-
"""Robust time-series statistics operators.

These operators resist single extreme observations: quantile ranges, trimmed
means and median/MAD z-scores.  Every function consumes one or more
``timestamp x instrument`` panels and returns a panel with the same shape,
using only the current row and historical rows (causal).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator


def _metadata(name: str, description: str, params: list[str], *, domain: str, unit: str) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="robust_statistics",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "robust_statistics", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", "cost:1",
        ],
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _rolling_apply_2d(
    values: np.ndarray,
    window: int,
    fn: Any,
    min_periods: int = 1,
) -> np.ndarray:
    rows, cols = values.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - window + 1)
            chunk = values[start : row + 1, col]
            out[row, col] = fn(chunk)
    return out


@register_operator(
    name="ts_quantile_range",
    category="robust_statistics",
    business_category="robust_statistics",
    canonical="ts_quantile_range",
    source="robust_stats",
    status="experimental",
)
class TsQuantileRange(SeriesOperator):
    """滚动分位区间：Q(x, q_high) - Q(x, q_low)。q_low=0.25/q_high=0.75 即 IQR。"""

    metadata = _metadata(
        "ts_quantile_range",
        "滚动分位区间 Q(x,q_high)-Q(x,q_low)。",
        ["x", "window", "q_low", "q_high", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 20,
        q_low: float = 0.25,
        q_high: float = 0.75,
        min_periods: int = 1,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        lo = float(q_low)
        hi = float(q_high)
        if not (0.0 < lo < hi < 1.0):
            raise ValueError("ts_quantile_range requires 0 < q_low < q_high < 1")
        mp = max(1, int(min_periods))

        def _fn(chunk: np.ndarray) -> float:
            valid = chunk[np.isfinite(chunk)]
            if valid.size < mp:
                return np.nan
            return float(np.quantile(valid, hi) - np.quantile(valid, lo))

        return _frame_like(x, _rolling_apply_2d(x.to_numpy(dtype=float), w, _fn, mp))


@register_operator(
    name="ts_trimmed_mean",
    category="robust_statistics",
    business_category="robust_statistics",
    canonical="ts_trimmed_mean",
    source="robust_stats",
    status="experimental",
)
class TsTrimmedMean(SeriesOperator):
    """滚动截尾均值：删除最低/最高 trim_ratio 后对剩余求平均。"""

    metadata = _metadata(
        "ts_trimmed_mean",
        "滚动截尾均值（删除两侧 trim_ratio 后求平均）。",
        ["x", "window", "trim_ratio", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 20,
        trim_ratio: float = 0.1,
        min_periods: int = 1,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        trim = float(trim_ratio)
        if not (0.0 <= trim < 0.5):
            raise ValueError("ts_trimmed_mean requires 0 <= trim_ratio < 0.5")
        mp = max(1, int(min_periods))

        def _fn(chunk: np.ndarray) -> float:
            valid = chunk[np.isfinite(chunk)]
            if valid.size < mp:
                return np.nan
            ordered = np.sort(valid)
            cut = int(np.floor(trim * ordered.size))
            if cut * 2 >= ordered.size:
                return float(np.mean(ordered))
            return float(np.mean(ordered[cut : ordered.size - cut]))

        return _frame_like(x, _rolling_apply_2d(x.to_numpy(dtype=float), w, _fn, mp))


@register_operator(
    name="ts_robust_zscore",
    category="robust_statistics",
    business_category="robust_statistics",
    canonical="ts_robust_zscore",
    source="robust_stats",
    status="experimental",
)
class TsRobustZscore(SeriesOperator):
    """稳健 z-score：(x - center) / (scale)，center 中位数、scale MAD。

    ``center`` 取 ``"median"``/``"mean"``，``scale`` 取 ``"mad"``/``"std"``；
    ``clip`` 非空时将输出截断到 [-clip, clip]。
    """

    metadata = _metadata(
        "ts_robust_zscore",
        "稳健 z-score：(x-median)/(1.4826*MAD)，可选 clip。",
        ["x", "window", "center", "scale", "clip"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 20,
        center: Any = "median",
        scale: Any = "mad",
        clip: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        center_name = str(center or "median").lower()
        scale_name = str(scale or "mad").lower()

        def _fn(chunk: np.ndarray) -> float:
            valid = chunk[np.isfinite(chunk)]
            if valid.size == 0:
                return np.nan
            center_value = float(np.median(valid)) if center_name == "median" else float(np.mean(valid))
            if scale_name == "std":
                spread = float(np.std(valid))
            else:
                spread = float(np.median(np.abs(valid - center_value))) * 1.4826
            if not np.isfinite(spread) or spread <= 0.0:
                return np.nan
            value = (float(chunk[-1]) - center_value) / spread if np.isfinite(chunk[-1]) else np.nan
            if clip is not None and np.isfinite(value):
                bound = float(clip)
                value = max(-bound, min(bound, value))
            return value

        return _frame_like(x, _rolling_apply_2d(x.to_numpy(dtype=float), w, _fn))
