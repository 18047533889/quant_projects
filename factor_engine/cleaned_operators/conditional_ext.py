# -*- coding: utf-8 -*-
"""Extended conditional rolling operators (``ts_*_if``).

Extends the existing ``ts_mean_if``/``ts_sum_if``/``ts_std_if`` family with
min/max/quantile/corr/beta/regression-residual.  The ``condition`` panel is
truthy where it is finite and non-zero (same convention as ``daily_panel``).
All operators are causal daily-panel transforms.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.common.daily_panel import _aligned


def _metadata(name: str, description: str, params: list[str], *, domain: str, unit: str) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="time_series_condition",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_condition", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", "cost:1",
        ],
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _selected_mask(condition: pd.DataFrame) -> np.ndarray:
    cv = condition.to_numpy()
    return np.isfinite(cv) & (cv != 0)


def _min_max_rolling(values: np.ndarray, mask: np.ndarray, window: int, min_periods: int, op: str) -> np.ndarray:
    rows, cols = values.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - window + 1)
            x_chunk = values[start : row + 1, col]
            m_chunk = mask[start : row + 1, col]
            selected = x_chunk[m_chunk]
            if selected.size < min_periods:
                continue
            out[row, col] = float(np.min(selected)) if op == "min" else float(np.max(selected))
    return out


@register_operator(
    name="ts_min_if",
    category="time_series_condition",
    business_category="time_series_condition",
    canonical="ts_min_if",
    source="conditional_ext",
    status="experimental",
)
class TsMinIf(SeriesOperator):
    """滚动窗口内条件成立样本的最小值。"""

    metadata = _metadata(
        "ts_min_if",
        "窗口内 condition 成立时 x 的最小值。",
        ["x", "condition", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, condition: pd.DataFrame, window: int = 20, min_periods: int = 1, **_: Any) -> pd.DataFrame:
        x, condition = _aligned(x, condition)
        w = int(window)
        mp = max(1, int(min_periods))
        mask = _selected_mask(condition)
        return _frame_like(x, _min_max_rolling(x.to_numpy(dtype=float), mask, w, mp, "min"))


@register_operator(
    name="ts_max_if",
    category="time_series_condition",
    business_category="time_series_condition",
    canonical="ts_max_if",
    source="conditional_ext",
    status="experimental",
)
class TsMaxIf(SeriesOperator):
    """滚动窗口内条件成立样本的最大值。"""

    metadata = _metadata(
        "ts_max_if",
        "窗口内 condition 成立时 x 的最大值。",
        ["x", "condition", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, condition: pd.DataFrame, window: int = 20, min_periods: int = 1, **_: Any) -> pd.DataFrame:
        x, condition = _aligned(x, condition)
        w = int(window)
        mp = max(1, int(min_periods))
        mask = _selected_mask(condition)
        return _frame_like(x, _min_max_rolling(x.to_numpy(dtype=float), mask, w, mp, "max"))


@register_operator(
    name="ts_quantile_if",
    category="time_series_condition",
    business_category="time_series_condition",
    canonical="ts_quantile_if",
    source="conditional_ext",
    status="experimental",
)
class TsQuantileIf(SeriesOperator):
    """滚动窗口内条件成立样本的分位数。"""

    metadata = _metadata(
        "ts_quantile_if",
        "窗口内 condition 成立时 x 的 q 分位。",
        ["x", "condition", "window", "q", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, condition: pd.DataFrame, window: int = 20, q: float = 0.5, min_periods: int = 1, **_: Any) -> pd.DataFrame:
        x, condition = _aligned(x, condition)
        w = int(window)
        quantile = float(q)
        if not (0.0 <= quantile <= 1.0):
            raise ValueError("ts_quantile_if requires 0 <= q <= 1")
        mp = max(1, int(min_periods))
        mask = _selected_mask(condition)
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                selected = xv[start : row + 1, col][mask[start : row + 1, col]]
                if selected.size < mp:
                    continue
                out[row, col] = float(np.quantile(selected, quantile))
        return _frame_like(x, out)


def _pair_condition_rolling(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    window: int,
    min_periods: int,
    kind: str,
) -> np.ndarray:
    rows, cols = x.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - window + 1)
            x_chunk = x[start : row + 1, col]
            y_chunk = y[start : row + 1, col]
            m_chunk = mask[start : row + 1, col]
            xs = x_chunk[m_chunk]
            ys = y_chunk[m_chunk]
            if xs.size < min_periods:
                continue
            if kind == "corr":
                if np.std(xs) > 0 and np.std(ys) > 0:
                    out[row, col] = float(np.corrcoef(xs, ys)[0, 1])
            elif kind == "beta":
                var_x = float(np.var(xs))
                if var_x > 0 and np.isfinite(var_x):
                    cov = float(np.mean((xs - np.mean(xs)) * (ys - np.mean(ys))))
                    out[row, col] = cov / var_x
            elif kind == "resid":
                if np.std(xs) > 0 and xs.size >= 2:
                    coeffs = np.polyfit(xs, ys, 1)
                    out[row, col] = float(ys[-1] - np.polyval(coeffs, xs[-1]))
    return out


@register_operator(
    name="ts_corr_if",
    category="time_series_condition",
    business_category="time_series_condition",
    canonical="ts_corr_if",
    source="conditional_ext",
    status="experimental",
)
class TsCorrIf(SeriesOperator):
    """条件窗口内 x 与 y 的相关系数。"""

    metadata = _metadata(
        "ts_corr_if",
        "窗口内 condition 成立样本中 x 与 y 的相关系数。",
        ["x", "y", "condition", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, condition: pd.DataFrame, window: int = 20, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        x, y, condition = _aligned(x, y, condition)
        w = int(window)
        mp = max(2, int(min_periods))
        mask = _selected_mask(condition)
        return _frame_like(
            x,
            _pair_condition_rolling(x.to_numpy(dtype=float), y.to_numpy(dtype=float), mask, w, mp, "corr"),
        )


@register_operator(
    name="ts_beta_if",
    category="time_series_condition",
    business_category="time_series_condition",
    canonical="ts_beta_if",
    source="conditional_ext",
    status="experimental",
)
class TsBetaIf(SeriesOperator):
    """条件窗口内 y 对 x 的回归斜率（beta）。"""

    metadata = _metadata(
        "ts_beta_if",
        "窗口内 condition 成立样本中 y 对 x 的 beta。",
        ["y", "x", "condition", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, condition: pd.DataFrame, window: int = 20, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        x, y, condition = _aligned(x, y, condition)
        w = int(window)
        mp = max(2, int(min_periods))
        mask = _selected_mask(condition)
        return _frame_like(
            y,
            _pair_condition_rolling(x.to_numpy(dtype=float), y.to_numpy(dtype=float), mask, w, mp, "beta"),
        )


@register_operator(
    name="ts_regression_resid_if",
    category="time_series_condition",
    business_category="time_series_condition",
    canonical="ts_regression_resid_if",
    source="conditional_ext",
    status="experimental",
)
class TsRegressionResidIf(SeriesOperator):
    """条件窗口内 y 对 x 线性回归的当前残差。"""

    metadata = _metadata(
        "ts_regression_resid_if",
        "窗口内 condition 成立样本中 y 对 x 回归的当前残差。",
        ["y", "x", "condition", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, condition: pd.DataFrame, window: int = 20, min_periods: int = 3, **_: Any) -> pd.DataFrame:
        x, y, condition = _aligned(x, y, condition)
        w = int(window)
        mp = max(3, int(min_periods))
        mask = _selected_mask(condition)
        return _frame_like(
            y,
            _pair_condition_rolling(x.to_numpy(dtype=float), y.to_numpy(dtype=float), mask, w, mp, "resid"),
        )
