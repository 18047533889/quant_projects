# -*- coding: utf-8 -*-
"""Wave-1 operator expansion: cyclical / seasonal decomposition statistics.

New canonicals across genuinely new thematic ground:
  * seasonal-lag comparison (weekday / monthly anchor ratios and z-scores)
  * seasonal effect strength / weekday anomaly
  * seasonal cycle phase / harmonic power (weekly periodicity strength)

All are real pandas_numpy implementations, strictly causal (seasonal lags only
read PAST rows — no forward-looking seasonal statistics), deterministic and
NaN-safe.  Names are prefixed ``cs1_`` and are globally unique.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.common.daily_panel import _aligned, _check_int


def _metadata(
    name: str, description: str, params: list[str], *, domain: str, unit: str,
    cost: int = 1, category: str,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category=category,
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            category, "wave1", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", f"cost:{cost}",
        ],
        output_unit=unit if unit.startswith(("same_as:", "unit(")) else None,
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


# ---------------------------------------------------------------------------
# 1. seasonal-lag comparison
# ---------------------------------------------------------------------------
@register_operator(
    name="cs1_weekday_lag_ratio",
    category="cyclical_decomposition",
    business_category="cyclical_decomposition",
    canonical="cs1_weekday_lag_ratio",
    source="wave1_seasonal",
    status="experimental")
class Cs1WeekdayLagRatio(SeriesOperator):
    """周内同位置滞后比：x[t] / x[t - lag_week]。

    默认 lag_week=5（一周交易日）比较"今天"与"上周同期"。>1 = 周度同比
    上升。输出 ratio。
    """

    metadata = _metadata(
        "cs1_weekday_lag_ratio",
        "当前值 / 周内同期滞后值（周度同比）。",
        ["x", "lag_week", "min_periods"],
        domain="price_volume",
        unit="ratio",
        category="cyclical_decomposition",
    )

    def _calculate_series(self, x: pd.DataFrame, lag_week: int = 5, min_periods: int = 1, **_: Any) -> pd.DataFrame:
        lw = _check_int(lag_week, "lag_week", 1)
        mp = _check_int(min_periods, "min_periods", 1)
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(xv[row, col]):
                    continue
                i = row - lw
                if i < 0:
                    continue
                v = xv[i, col]
                if np.isfinite(v) and np.abs(v) > 1e-12:
                    out[row, col] = xv[row, col] / v
        return _frame_like(x, out)


@register_operator(
    name="cs1_weekday_lag_zscore",
    category="cyclical_decomposition",
    business_category="cyclical_decomposition",
    canonical="cs1_weekday_lag_zscore",
    source="wave1_seasonal",
    status="experimental")
class Cs1WeekdayLagZscore(SeriesOperator):
    """周内同期滞后 z：当前值相对过去 lag_week 期均值的 z-score。

    捕捉相对自身周内位置的偏离（排除周度季节性的标准化）。当前值减过去
    lag_week 期均值再除以 std。输出 dimensionless。
    """

    metadata = _metadata(
        "cs1_weekday_lag_zscore",
        "(x[t] - mean(x[t-1..t-lag_week]))/std（周度季节标准化）。",
        ["x", "lag_week", "min_periods"],
        domain="price_volume",
        unit="dimensionless",
        cost=2,
        category="cyclical_decomposition",
    )

    def _calculate_series(self, x: pd.DataFrame, lag_week: int = 5, min_periods: int = 3, **_: Any) -> pd.DataFrame:
        lw = _check_int(lag_week, "lag_week", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > lw:
            raise ValueError("min_periods must be <= lag_week")
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(xv[row, col]):
                    continue
                lo = max(0, row - lw)
                hist = xv[lo:row, col]
                ok = np.isfinite(hist)
                if ok.sum() < mp:
                    continue
                h = hist[ok]
                m = float(np.mean(h))
                s = float(np.std(h, ddof=1))
                if s <= 1e-12:
                    out[row, col] = 0.0
                else:
                    out[row, col] = (xv[row, col] - m) / s
        return _frame_like(x, out)


@register_operator(
    name="cs1_monthly_lag_ratio",
    category="cyclical_decomposition",
    business_category="cyclical_decomposition",
    canonical="cs1_monthly_lag_ratio",
    source="wave1_seasonal",
    status="experimental")
class Cs1MonthlyLagRatio(SeriesOperator):
    """月度滞后比：x[t] / x[t - lag_month]（默认 20 个交易日）。

    月度同比（month-over-month 代理）。>1 = 环比上升。输出 ratio。
    """

    metadata = _metadata(
        "cs1_monthly_lag_ratio",
        "当前值 / 月度同期滞后值（月度环比）。",
        ["x", "lag_month", "min_periods"],
        domain="price_volume",
        unit="ratio",
        category="cyclical_decomposition",
    )

    def _calculate_series(self, x: pd.DataFrame, lag_month: int = 20, min_periods: int = 1, **_: Any) -> pd.DataFrame:
        lm = _check_int(lag_month, "lag_month", 1)
        mp = _check_int(min_periods, "min_periods", 1)
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(xv[row, col]):
                    continue
                i = row - lm
                if i < 0:
                    continue
                v = xv[i, col]
                if np.isfinite(v) and np.abs(v) > 1e-12:
                    out[row, col] = xv[row, col] / v
        return _frame_like(x, out)


@register_operator(
    name="cs1_seasonal_relative_rank",
    category="cyclical_decomposition",
    business_category="cyclical_decomposition",
    canonical="cs1_seasonal_relative_rank",
    source="wave1_seasonal",
    status="experimental")
class Cs1SeasonalRelativeRank(SeriesOperator):
    """季节性相对排名：当前值在过去 lag_week 个同期观测中的百分位。

    0 = 低于所有历史同期，1 = 高于所有历史同期。输出 ratio。
    """

    metadata = _metadata(
        "cs1_seasonal_relative_rank",
        "当前值在过去同期观测中的百分位。",
        ["x", "lag_week", "min_periods"],
        domain="price_volume",
        unit="ratio",
        cost=2,
        category="cyclical_decomposition",
    )

    def _calculate_series(self, x: pd.DataFrame, lag_week: int = 5, min_periods: int = 3, **_: Any) -> pd.DataFrame:
        lw = _check_int(lag_week, "lag_week", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > lw:
            raise ValueError("min_periods must be <= lag_week")
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(xv[row, col]):
                    continue
                lo = max(0, row - lw)
                hist = xv[lo:row, col]
                ok = np.isfinite(hist)
                if ok.sum() < mp:
                    continue
                h = hist[ok]
                rank = float(np.sum(h < xv[row, col])) + 0.5 * float(np.sum(h == xv[row, col]))
                out[row, col] = rank / h.size
        return _frame_like(x, out)


# ---------------------------------------------------------------------------
# 2. seasonal effect strength / weekday anomaly
# ---------------------------------------------------------------------------
@register_operator(
    name="cs1_weekday_effect_strength",
    category="cyclical_decomposition",
    business_category="cyclical_decomposition",
    canonical="cs1_weekday_effect_strength",
    source="wave1_seasonal",
    status="experimental")
class Cs1WeekdayEffectStrength(SeriesOperator):
    """周度效应强度：mean|x[t] - x[t-lag_week]| / mean|x[t] - x[t-1]|。

    分子 = 周度同比差，分母 = 日度差。比值 >1 = 周度模式比日度波动更显著。
    输出 ratio。
    """

    metadata = _metadata(
        "cs1_weekday_effect_strength",
        "周度同比差 / 日度差（周度效应显著性）。",
        ["x", "lag_week", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        cost=2,
        category="cyclical_decomposition",
    )

    def _calculate_series(self, x: pd.DataFrame, lag_week: int = 5, window: int = 40, min_periods: int = 8, **_: Any) -> pd.DataFrame:
        lw = _check_int(lag_week, "lag_week", 1)
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 4)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                idxs = np.arange(lo, row + 1)
                d1: list[float] = []
                d5: list[float] = []
                for i in idxs:
                    if i - 1 >= lo and np.isfinite(xv[i, col]) and np.isfinite(xv[i - 1, col]):
                        d1.append(abs(xv[i, col] - xv[i - 1, col]))
                    if i - lw >= lo and np.isfinite(xv[i, col]) and np.isfinite(xv[i - lw, col]):
                        d5.append(abs(xv[i, col] - xv[i - lw, col]))
                if len(d1) < mp or len(d5) < 1:
                    continue
                m1 = float(np.mean(d1))
                m5 = float(np.mean(d5))
                if m1 <= 1e-12:
                    out[row, col] = 0.0
                else:
                    out[row, col] = m5 / m1
        return _frame_like(x, out)


@register_operator(
    name="cs1_weekday_anomaly",
    category="cyclical_decomposition",
    business_category="cyclical_decomposition",
    canonical="cs1_weekday_anomaly",
    source="wave1_seasonal",
    status="experimental")
class Cs1WeekdayAnomaly(SeriesOperator):
    """周内异常度：当前值相对周内同期均值的标准化偏离。

    用过去 lag_week 个同期观测估计"正常"水平，输出当前值与其差（同单位）。
    高 = 当前周内异常高。
    """

    metadata = _metadata(
        "cs1_weekday_anomaly",
        "x[t] - mean(x 过去同期)（周内异常度）。",
        ["x", "lag_week", "min_periods"],
        domain="price_volume",
        unit="same_as:x",
        category="cyclical_decomposition",
    )

    def _calculate_series(self, x: pd.DataFrame, lag_week: int = 5, min_periods: int = 2, **_: Any) -> pd.DataFrame:
        lw = _check_int(lag_week, "lag_week", 2)
        mp = _check_int(min_periods, "min_periods", 2)
        if mp > lw:
            raise ValueError("min_periods must be <= lag_week")
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                if not np.isfinite(xv[row, col]):
                    continue
                lo = max(0, row - lw)
                hist = xv[lo:row, col]
                ok = np.isfinite(hist)
                if ok.sum() < mp:
                    continue
                m = float(np.mean(hist[ok]))
                out[row, col] = xv[row, col] - m
        return _frame_like(x, out)


# ---------------------------------------------------------------------------
# 3. seasonal cycle phase / harmonic power
# ---------------------------------------------------------------------------
@register_operator(
    name="cs1_week_cycle_phase",
    category="cyclical_decomposition",
    business_category="cyclical_decomposition",
    canonical="cs1_week_cycle_phase",
    source="wave1_seasonal",
    status="experimental")
class Cs1WeekCyclePhase(SeriesOperator):
    """周循环相位：与 lag_week 位移序列的最大相关滞后（周期相位）。

    在 1..lag_week 的滞后中找使 corr(x[t], x[t-lag]) 最强的 lag，输出该 lag
    对应相位的归一化 0..1（0 = 相位 0，1 = 满周期）。衡量序列的主导周内节奏。
    """

    metadata = _metadata(
        "cs1_week_cycle_phase",
        "周期相位：1..lag_week 中使自相关最强的滞后（归一化 0..1）。",
        ["x", "lag_week", "min_periods"],
        domain="price_volume",
        unit="ratio",
        cost=3,
        category="cyclical_decomposition",
    )

    def _calculate_series(self, x: pd.DataFrame, lag_week: int = 5, min_periods: int = 6, **_: Any) -> pd.DataFrame:
        lw = _check_int(lag_week, "lag_week", 2)
        mp = _check_int(min_periods, "min_periods", 3)
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - lw * 2)
                chunk = xv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                vals = chunk[ok]
                if vals.size < mp + 1:
                    continue
                best = -1.0
                best_lag = -1
                for lag in range(1, lw + 1):
                    x_ = vals[:-lag]
                    y_ = vals[lag:]
                    if x_.size < mp:
                        continue
                    sx = float(np.std(x_, ddof=1))
                    sy = float(np.std(y_, ddof=1))
                    if sx <= 1e-12 or sy <= 1e-12:
                        continue
                    c = abs(float(np.corrcoef(x_, y_)[0, 1]))
                    if c > best:
                        best = c
                        best_lag = lag
                if best_lag < 0:
                    continue
                out[row, col] = (best_lag - 1) / lw
        return _frame_like(x, out)


@register_operator(
    name="cs1_weekly_harmonic_power",
    category="cyclical_decomposition",
    business_category="cyclical_decomposition",
    canonical="cs1_weekly_harmonic_power",
    source="wave1_seasonal",
    status="experimental")
class Cs1WeeklyHarmonicPower(SeriesOperator):
    """周谐波功率：周周期自相关（lag_week 与 2·lag_week 的均值）强度。

    衡量序列中周度周期成分的相对强度。0 = 无周度周期，1 = 完美周度周期。
    输出 ratio。
    """

    metadata = _metadata(
        "cs1_weekly_harmonic_power",
        "lag_week / 2·lag_week 自相关的均值（周周期强度）。",
        ["x", "lag_week", "min_periods"],
        domain="price_volume",
        unit="ratio",
        cost=3,
        category="cyclical_decomposition",
    )

    def _calculate_series(self, x: pd.DataFrame, lag_week: int = 5, min_periods: int = 6, **_: Any) -> pd.DataFrame:
        lw = _check_int(lag_week, "lag_week", 2)
        mp = _check_int(min_periods, "min_periods", 3)
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - lw * 3)
                chunk = xv[lo:row + 1, col]
                ok = np.isfinite(chunk)
                if ok.sum() < mp:
                    continue
                vals = chunk[ok]
                cors: list[float] = []
                for lag in (lw, 2 * lw):
                    if vals.size <= lag:
                        continue
                    x_ = vals[:-lag]
                    y_ = vals[lag:]
                    if x_.size < mp:
                        continue
                    sx = float(np.std(x_, ddof=1))
                    sy = float(np.std(y_, ddof=1))
                    if sx <= 1e-12 or sy <= 1e-12:
                        continue
                    cors.append(abs(float(np.corrcoef(x_, y_)[0, 1])))
                if not cors:
                    continue
                out[row, col] = float(np.mean(cors))
        return _frame_like(x, out)


@register_operator(
    name="cs1_seasonal_residual_smoothness",
    category="cyclical_decomposition",
    business_category="cyclical_decomposition",
    canonical="cs1_seasonal_residual_smoothness",
    source="wave1_seasonal",
    status="experimental")
class Cs1SeasonalResidualSmoothness(SeriesOperator):
    """季节残差平滑度：去除周度滞后残差后的低波动占比。

    residual = x[t] - x[t-lag_week]（周度去趋势）；其窗口 std 越低 = 序列
    被周度模式解释得越好（平滑残差）。输出 ratio（越小越平滑，直接输出
    residual 的稳健离散度）。
    """

    metadata = _metadata(
        "cs1_seasonal_residual_smoothness",
        "周度残差 x[t]-x[t-lag_week] 的窗口离散度。",
        ["x", "lag_week", "window", "min_periods"],
        domain="price_volume",
        unit="ratio",
        cost=2,
        category="cyclical_decomposition",
    )

    def _calculate_series(self, x: pd.DataFrame, lag_week: int = 5, window: int = 40, min_periods: int = 8, **_: Any) -> pd.DataFrame:
        lw = _check_int(lag_week, "lag_week", 1)
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 4)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                lo = max(0, row - w + 1)
                resid: list[float] = []
                for i in range(lo, row + 1):
                    if i - lw >= lo:
                        a = xv[i, col]
                        b = xv[i - lw, col]
                        if np.isfinite(a) and np.isfinite(b):
                            resid.append(a - b)
                if len(resid) < mp:
                    continue
                r = np.array(resid)
                m = float(np.mean(r))
                s = float(np.std(r, ddof=1))
                den = abs(m)
                if den <= 1e-12:
                    out[row, col] = 0.0
                else:
                    out[row, col] = s / (den + 1e-12)
        return _frame_like(x, out)