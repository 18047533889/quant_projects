# -*- coding: utf-8 -*-
"""Model-type rolling operators (2026-08 P2 expansion).

Robust / ridge / quantile regression residuals, AR coefficient, variance
ratio and structural-shift scores.  All kernels are causal rolling per-column
operations over ``timestamp x instrument`` panels.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.common.daily_panel import _aligned
from cleaned_operators.ts_model._rolling_core import pinball_quantile_fit


def _metadata(name: str, description: str, params: list[str], *, domain: str, unit: str) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="time_series_regression",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_regression", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", "cost:2",
        ],
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _rolling_apply_2d(values: np.ndarray, window: int, fn: Any, min_periods: int = 1) -> np.ndarray:
    rows, cols = values.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for row in range(rows):
            start = max(0, row - window + 1)
            out[row, col] = fn(values[start : row + 1, col])
    return out


def _regression_resid(y: np.ndarray, x: np.ndarray, *, method: str, min_periods: int, alpha: float = 0.0) -> float:
    valid = np.isfinite(y) & np.isfinite(x)
    if valid.sum() < max(min_periods, 3):
        return np.nan
    xs = x[valid]
    ys = y[valid]
    if np.std(xs) <= 0.0:
        return np.nan
    design = np.column_stack([np.ones(len(xs)), xs])
    if method == "ols":
        beta, *_ = np.linalg.lstsq(design, ys, rcond=None)
    elif method == "ridge":
        ridge = alpha * np.eye(design.shape[1])
        ridge[0, 0] = 0.0  # 不惩罚截距
        beta, *_ = np.linalg.lstsq(design.T @ design + ridge, design.T @ ys, rcond=None)
    elif method == "huber":
        beta = _huber_fit(design, ys)
    else:
        raise ValueError(f"unknown method: {method}")
    # 当前样本残差（最后一行）
    y_cur = y[-1]
    x_cur = x[-1]
    if not np.isfinite(y_cur) or not np.isfinite(x_cur):
        return np.nan
    return float(y_cur - (beta[0] + beta[1] * x_cur))


def _huber_fit(design: np.ndarray, ys: np.ndarray, *, delta: float = 1.345, iterations: int = 5) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(design, ys, rcond=None)
    for _ in range(iterations):
        resid = ys - design @ beta
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if scale <= 0.0:
            scale = np.std(resid)
        if scale <= 0.0:
            break
        z = resid / scale
        abs_z = np.abs(z)
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = np.where(abs_z <= delta, 1.0, delta / abs_z)
        # Huber IRLS minimizes sum(w_i e_i^2).  Weighted least squares solves
        # the normal equations with sqrt(w_i) applied to both design and
        # response (same convention as ``daily_panel._ols_residual``).
        # Weighting design/y by w (not sqrt(w)) minimized sum(w_i^2 e_i^2),
        # which over-shrinks outlier rows (review P0-01).
        root_w = np.sqrt(weight)
        wdesign = design * root_w[:, None]
        beta, *_ = np.linalg.lstsq(wdesign, ys * root_w, rcond=None)
    return beta


@register_operator(
    name="ts_huber_regression_resid",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_huber_regression_resid",
    source="regression_models",
    status="experimental",
)
class TsHuberRegressionResid(SeriesOperator):
    """Huber 稳健回归的当前样本残差。"""

    metadata = _metadata(
        "ts_huber_regression_resid",
        "Huber 稳健回归当前样本残差。",
        ["y", "x", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        y, x = _aligned(y, x)
        w = int(window)
        mp = max(3, int(min_periods))
        yv = y.to_numpy(dtype=float)
        xv = x.to_numpy(dtype=float)
        rows, cols = yv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                out[row, col] = _regression_resid(yv[start : row + 1, col], xv[start : row + 1, col], method="huber", min_periods=mp)
        return _frame_like(y, out)


@register_operator(
    name="ts_ridge_regression_resid",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_ridge_regression_resid",
    source="regression_models",
    status="experimental",
)
class TsRidgeRegressionResid(SeriesOperator):
    """Ridge 回归的当前样本残差。"""

    metadata = _metadata(
        "ts_ridge_regression_resid",
        "Ridge 回归当前样本残差。",
        ["y", "x", "window", "alpha", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, alpha: float = 0.1, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        y, x = _aligned(y, x)
        w = int(window)
        mp = max(3, int(min_periods))
        a = float(alpha)
        if a < 0.0:
            raise ValueError("alpha must be non-negative")
        yv = y.to_numpy(dtype=float)
        xv = x.to_numpy(dtype=float)
        rows, cols = yv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                out[row, col] = _regression_resid(yv[start : row + 1, col], xv[start : row + 1, col], method="ridge", min_periods=mp, alpha=a)
        return _frame_like(y, out)


@register_operator(
    name="ts_quantile_regression_slope",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_quantile_regression_slope",
    source="regression_models",
    status="experimental",
)
class TsQuantileRegressionSlope(SeriesOperator):
    """分位数回归斜率（单分位，pinball-loss 线性规划）。"""

    metadata = _metadata(
        "ts_quantile_regression_slope",
        "分位数回归斜率。",
        ["y", "x", "window", "q", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, window: int = 20, q: float = 0.5, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        y, x = _aligned(y, x)
        w = int(window)
        mp = max(3, int(min_periods))
        quantile = float(q)
        if not (0.0 < quantile < 1.0):
            raise ValueError("q must be in (0,1)")
        yv = y.to_numpy(dtype=float)
        xv = x.to_numpy(dtype=float)
        rows, cols = yv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                out[row, col] = _quantile_slope(yv[start : row + 1, col], xv[start : row + 1, col], quantile, mp)
        return _frame_like(y, out)


def _quantile_slope(y: np.ndarray, x: np.ndarray, q: float, min_periods: int) -> float:
    """True pinball-loss quantile slope (Koenker–Bassett LP)."""
    valid = np.isfinite(y) & np.isfinite(x)
    if valid.sum() < max(min_periods, 3):
        return np.nan
    xs = x[valid]
    ys = y[valid]
    if np.std(xs) <= 0.0:
        return np.nan
    design = np.column_stack([np.ones(len(xs)), xs])
    beta = pinball_quantile_fit(design, ys, float(q))
    if beta is None:
        return np.nan
    return float(beta[1])


@register_operator(
    name="ts_ar_coefficient",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_ar_coefficient",
    source="regression_models",
    status="experimental",
)
class TsArCoefficient(SeriesOperator):
    """AR(lag) 系数：x_t 对 x_{t-lag} 的回归斜率。"""

    metadata = _metadata(
        "ts_ar_coefficient",
        "AR(lag) 回归系数。",
        ["x", "window", "lag", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, lag: int = 1, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = int(window)
        lg = max(1, int(lag))
        mp = max(3, int(min_periods))
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                if row - lg < start:
                    continue
                segment = xv[start : row + 1, col]
                current = segment[lg:]
                lagged = segment[:-lg]
                valid = np.isfinite(current) & np.isfinite(lagged)
                if valid.sum() < mp or np.std(lagged[valid]) <= 0.0:
                    continue
                cov = float(np.mean((current[valid] - np.mean(current[valid])) * (lagged[valid] - np.mean(lagged[valid]))))
                var = float(np.var(lagged[valid]))
                out[row, col] = cov / var if var > 0.0 else np.nan
        return _frame_like(x, out)


@register_operator(
    name="ts_variance_ratio",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_variance_ratio",
    source="regression_models",
    status="experimental",
)
class TsVarianceRatio(SeriesOperator):
    """Lo–MacKinlay 方差比：Var(q-period) / (q * Var(1-period)) - 1。"""

    metadata = _metadata(
        "ts_variance_ratio",
        "方差比 Var(q)/q/Var(1) - 1。",
        ["x", "window", "q", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 60, q: int = 5, min_periods: int = 10, **_: Any) -> pd.DataFrame:
        w = int(window)
        periods = max(2, int(q))
        mp = max(6, int(min_periods))
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                segment = xv[start : row + 1, col]
                valid = np.isfinite(segment)
                vals = segment[valid]
                if vals.size < mp:
                    continue
                rets = np.diff(vals)
                if rets.size < mp:
                    continue
                var1 = float(np.var(rets))
                if var1 <= 0.0:
                    continue
                # q-period returns
                n = rets.size
                qrets = np.array([vals[i + periods] - vals[i] for i in range(0, n - periods + 1)])
                if qrets.size < 2:
                    continue
                varq = float(np.var(qrets))
                out[row, col] = varq / (periods * var1) - 1.0
        return _frame_like(x, out)


@register_operator(
    name="ts_cusum_break_score",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_cusum_break_score",
    source="regression_models",
    status="experimental",
)
class TsCusumBreakScore(SeriesOperator):
    """CUSUM 结构性突变得分：累积偏差相对波动率的比值。"""

    metadata = _metadata(
        "ts_cusum_break_score",
        "CUSUM 突变得分。",
        ["x", "window", "min_periods"],
        domain="price_volume",
        unit="level",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = int(window)
        mp = max(3, int(min_periods))
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                segment = xv[start : row + 1, col]
                valid = np.isfinite(segment)
                vals = segment[valid]
                if vals.size < mp:
                    continue
                mean = float(np.mean(vals))
                sd = float(np.std(vals))
                if sd <= 0.0:
                    continue
                # 到当前为止的累计标准化偏差。窗口总偏差对自身均值为 0，
                # 因此取路径上 |cumsum| 的最大值（末尾值无信息量）。
                running = np.cumsum(vals - mean) / (sd * np.sqrt(np.arange(1, len(vals) + 1)))
                out[row, col] = float(np.max(np.abs(running)))
        return _frame_like(x, out)


@register_operator(
    name="ts_level_shift_score",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_level_shift_score",
    source="regression_models",
    status="experimental",
)
class TsLevelShiftScore(SeriesOperator):
    """水平位移得分：前半与后半窗口均值差，除以滚动标准差。"""

    metadata = _metadata(
        "ts_level_shift_score",
        "前后半窗口均值差 / 滚动标准差。",
        ["x", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = int(window)
        mp = max(4, int(min_periods))
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                segment = xv[start : row + 1, col]
                valid = np.isfinite(segment)
                vals = segment[valid]
                if vals.size < mp:
                    continue
                half = max(1, len(vals) // 2)
                first = vals[:half]
                second = vals[-half:]
                sd = float(np.std(vals))
                if sd <= 0.0:
                    continue
                out[row, col] = float(np.mean(second) - np.mean(first)) / sd
        return _frame_like(x, out)


@register_operator(
    name="ts_vol_shift_score",
    category="time_series_regression",
    business_category="time_series_regression",
    canonical="ts_vol_shift_score",
    source="regression_models",
    status="experimental",
)
class TsVolShiftScore(SeriesOperator):
    """波动率位移得分：前后半窗口标准差的对数比。"""

    metadata = _metadata(
        "ts_vol_shift_score",
        "log(后半 std / 前半 std)。",
        ["x", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, min_periods: int = 5, **_: Any) -> pd.DataFrame:
        w = int(window)
        mp = max(4, int(min_periods))
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                start = max(0, row - w + 1)
                segment = xv[start : row + 1, col]
                valid = np.isfinite(segment)
                vals = segment[valid]
                if vals.size < mp:
                    continue
                half = max(1, len(vals) // 2)
                first = vals[:half]
                second = vals[-half:]
                sd1 = float(np.std(first))
                sd2 = float(np.std(second))
                if sd1 <= 0.0 or sd2 <= 0.0:
                    continue
                out[row, col] = float(np.log(sd2 / sd1))
        return _frame_like(x, out)
