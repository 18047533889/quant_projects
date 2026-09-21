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

from factor_engine.cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec, SeriesOperator, register_operator
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
# R62 vectorised kernels (trailing-window, zero per-row Python loops).
#
# All four kernels are strictly causal: row r reads only rows <= r, exactly as
# the reference per-row loops.  Windows are the *raw position* window
# ``x[lo(r):r+1]`` with ``lo(r) = max(0, r - W + 1)`` -- the index carries no
# information (no calendar-day / weekday arithmetic anywhere in this file), so
# lag_week / window are pure positional offsets, identical to the reference.
# ---------------------------------------------------------------------------
def _r62_win(rows: int, w: int):
    """Trailing-window frame: ``lo[:,None]`` is the window start row."""
    r = np.arange(rows)[:, None]
    lo = np.maximum(0, r - w + 1)
    i = lo + np.arange(w)[None, :]
    return lo, i


def _r62_pack(col: np.ndarray, i: np.ndarray, rows: int):
    """Left-packed finite subsequence of ``x[lo(r):r+1]``.

    Returns ``(C, cnt)`` where ``C[r, 0:cnt[r]]`` is the finite subsequence in
    raw order (the exact analogue of ``chunk[np.isfinite(chunk)]``) zero-padded
    after ``cnt[r]``.  One cumsum + one scatter -> no row loop.
    """
    r = np.arange(rows)[:, None]
    valid = i <= r
    idx = np.clip(i, 0, rows - 1)
    v = col[idx]
    m = valid & np.isfinite(v)
    cnt = m.sum(axis=1)
    pos = np.cumsum(m, axis=1) - 1
    C = np.zeros((rows, i.shape[1]), dtype=float)
    rr, cc = np.nonzero(m)
    C[rr, pos[rr, cc]] = v[rr, cc]
    return C, cnt


def _r62_lagcorr(C: np.ndarray, cnt: np.ndarray, lag: int, min_pairs: int):
    """Pearson corr of ``(C[:, :-lag], C[:, lag:])`` over the first cnt-lag pairs.

    Mirrors ``np.corrcoef`` exactly: centring uses ``sum/n`` and the correlation
    is ``num / sqrt(M2x * M2y)`` (the ddof=1 factors cancel), so overflow in the
    ``huge`` panel degrades to NaN in the same way as the reference.  Returns
    ``(corr, good)``; ``good`` folds in the sample floor and the reference's
    ``std(ddof=1) <= 1e-12`` degeneracy guard.
    """
    rows, W = C.shape
    n = cnt - lag
    ok = n >= min_pairs
    K = W - lag
    if K <= 0:
        return np.zeros(rows), np.zeros(rows, dtype=bool)
    A = C[:, :K]
    B = C[:, lag:]
    jj = np.arange(K)[None, :]
    mk = ok[:, None] & (jj < n[:, None])
    n1 = np.where(ok, n, 2).astype(float)  # degenerate rows use a benign 2
    inv = np.true_divide(1.0, n1 - 1.0)
    mx = np.where(mk, A, 0.0).sum(axis=1) / n1
    my = np.where(mk, B, 0.0).sum(axis=1) / n1
    ac = np.where(mk, A - mx[:, None], 0.0)
    bc = np.where(mk, B - my[:, None], 0.0)
    M2x = (ac * ac).sum(axis=1)
    M2y = (bc * bc).sum(axis=1)
    num = (ac * bc).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        # np.cov: c = dot(X, X.T) * (1/(n-1)); np.corrcoef: (c01/d0)/d1 then
        # clip the real part to [-1, 1] -- both matter for exact tie-breaks.
        sx = np.sqrt(M2x * inv)
        sy = np.sqrt(M2y * inv)
        corr = np.clip(((num * inv) / sx) / sy, -1.0, 1.0)
    good = ok & (sx > 1e-12) & (sy > 1e-12)
    return corr, good


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
                if np.isfinite(xv[max(0, row - lw):row, col]).sum() < mp:
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
                if np.isfinite(xv[max(0, row - lm):row, col]).sum() < mp:
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
        # R62: prefix-sum over the two difference series.
        #   d1 pairs: i in [lo+1 .. r] with both finite  (i-1 >= lo)
        #   d5 pairs: i in [lo+lw .. r] with both finite (i-lw >= lo)
        #   m1 = mean(d1), m5 = mean(d5); out = m5/m1 (0 when m1 <= 1e-12)
        lw = _check_int(lag_week, "lag_week", 1)
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 4)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        r = np.arange(rows)
        lo = np.maximum(0, r - w + 1)
        st1 = np.minimum(np.maximum(1, lo + 1), rows)
        st5 = np.minimum(lo + lw, rows)
        for col in range(cols):
            xc = xv[:, col]
            d1 = np.zeros(rows, dtype=float)
            c1 = np.zeros(rows, dtype=float)
            d5 = np.zeros(rows, dtype=float)
            c5 = np.zeros(rows, dtype=float)
            if rows > 1:
                a, b = xc[1:], xc[:-1]
                m = np.isfinite(a) & np.isfinite(b)
                d1[1:] = np.where(m, np.abs(a - b), 0.0)
                c1[1:] = m
            if rows > lw:
                a, b = xc[lw:], xc[:-lw]
                m = np.isfinite(a) & np.isfinite(b)
                d5[lw:] = np.where(m, np.abs(a - b), 0.0)
                c5[lw:] = m
            S1 = np.concatenate(([0.0], np.cumsum(d1)))
            C1 = np.concatenate(([0.0], np.cumsum(c1)))
            S5 = np.concatenate(([0.0], np.cumsum(d5)))
            C5 = np.concatenate(([0.0], np.cumsum(c5)))
            sum1 = S1[r + 1] - S1[st1]
            cnt1 = C1[r + 1] - C1[st1]
            sum5 = S5[r + 1] - S5[st5]
            cnt5 = C5[r + 1] - C5[st5]
            ok = (cnt1 >= mp) & (cnt5 >= 1)
            m1 = sum1 / np.where(cnt1 > 0, cnt1, 1.0)
            m5 = sum5 / np.where(cnt5 > 0, cnt5, 1.0)
            val = np.where(m1 <= 1e-12, 0.0, m5 / np.where(np.abs(m1) > 0.0, m1, 1.0))
            out[:, col] = np.where(ok, val, np.nan)
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
        # R62: pack the finite subsequence of x[max(0,r-2*lw):r+1] then score every
        # lag 1..lw.  best_lag = first lag attaining the max |corr| (strict '>');
        # out = (best_lag - 1) / lw.  Guard: vals.size >= mp+1 and, per lag,
        # x_.size >= mp and std(ddof=1) > 1e-12.
        lw = _check_int(lag_week, "lag_week", 2)
        mp = _check_int(min_periods, "min_periods", 3)
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        _, i = _r62_win(rows, 2 * lw + 1)
        for col in range(cols):
            C, cnt = _r62_pack(xv[:, col], i, rows)
            base = cnt >= (mp + 1)
            best = np.full(rows, -np.inf)
            best_lag = np.zeros(rows, dtype=int)
            for lag in range(1, lw + 1):
                corr, good = _r62_lagcorr(C, cnt, lag, mp)
                cc = np.where(good & base & np.isfinite(corr), np.abs(corr), -np.inf)
                upd = cc > best
                best = np.where(upd, cc, best)
                best_lag = np.where(upd, lag, best_lag)
            sel = best > -np.inf
            out[:, col] = np.where(sel, (best_lag - 1) / lw, np.nan)
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
        # R62: symmetric to cs1_week_cycle_phase but over lags (lw, 2*lw) and the
        # output is the mean of the accepted |corr| values.  Guard: vals.size >= mp
        # and per lag vals.size > lag and x_.size >= mp and std(ddof=1) > 1e-12.
        lw = _check_int(lag_week, "lag_week", 2)
        mp = _check_int(min_periods, "min_periods", 3)
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        _, i = _r62_win(rows, 3 * lw + 1)
        for col in range(cols):
            C, cnt = _r62_pack(xv[:, col], i, rows)
            ssum = np.zeros(rows, dtype=float)
            scnt = np.zeros(rows, dtype=float)
            for lag in (lw, 2 * lw):
                corr, good = _r62_lagcorr(C, cnt, lag, mp)
                good = good & (cnt >= mp) & (cnt > lag)
                ssum = ssum + np.where(good, np.abs(corr), 0.0)
                scnt = scnt + good
            out[:, col] = np.where(scnt > 0, ssum / np.maximum(scnt, 1.0), np.nan)
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
        # R62: residual r[i] = x[i] - x[i-lw] over i in [lo+lw .. r]; mean and
        # std(ddof=1) from a centred two-pass over the window matrix (matches
        # np.std, so the huge panel still overflows to inf and the tiny panel
        # still underflows to 0).  den = |mean|; out = 0 if den<=1e-12 else
        # s/(den+1e-12).
        lw = _check_int(lag_week, "lag_week", 1)
        w = _check_int(window, "window", 3)
        mp = _check_int(min_periods, "min_periods", 4)
        if mp > w:
            raise ValueError("min_periods must be <= window")
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        r = np.arange(rows)[:, None]
        lo = np.maximum(0, r - w + 1)
        j = np.arange(w)[None, :]
        i = lo + j
        pos_ok = (j >= lw) & (i <= r)
        idx = np.clip(i, 0, rows - 1)
        idx_l = np.clip(i - lw, 0, rows - 1)
        for col in range(cols):
            xc = xv[:, col]
            a = xc[idx]
            b = xc[idx_l]
            mm = pos_ok & np.isfinite(a) & np.isfinite(b)
            res = np.where(mm, a - b, 0.0)
            cnt = mm.sum(axis=1)
            mean = res.sum(axis=1) / np.where(cnt > 0, cnt, 1.0)
            cen = np.where(mm, res - mean[:, None], 0.0)
            ss = (cen * cen).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                s = np.sqrt(ss / np.maximum(cnt - 1, 1))
                den = np.abs(mean)
                val = np.where(den <= 1e-12, 0.0, s / (den + 1e-12))
            out[:, col] = np.where(cnt >= mp, val, np.nan)
        return _frame_like(x, out)


_H = lambda default, minimum=1: ParamSpec(
    dtype=int, min=minimum, default=default, param_role=ParamRole.HORIZON,
)
_N = lambda default, minimum=1: ParamSpec(
    dtype=int, min=minimum, default=default, param_role=ParamRole.ESTIMATOR_RESOLUTION,
)

Cs1WeekdayLagRatio.metadata.param_specs = {"lag_week": _H(5), "min_periods": _N(1)}
Cs1WeekdayLagZscore.metadata.param_specs = {"lag_week": _H(5, 2), "min_periods": _N(3, 2)}
Cs1MonthlyLagRatio.metadata.param_specs = {"lag_month": _H(20), "min_periods": _N(1)}
Cs1SeasonalRelativeRank.metadata.param_specs = {"lag_week": _H(5, 2), "min_periods": _N(3, 2)}
Cs1WeekdayEffectStrength.metadata.param_specs = {
    "lag_week": _H(5), "window": _H(40, 3), "min_periods": _N(8, 4),
}
Cs1WeekdayAnomaly.metadata.param_specs = {"lag_week": _H(5, 2), "min_periods": _N(2, 2)}
Cs1WeekCyclePhase.metadata.param_specs = {"lag_week": _H(5, 2), "min_periods": _N(6, 3)}
Cs1WeeklyHarmonicPower.metadata.param_specs = {"lag_week": _H(5, 2), "min_periods": _N(6, 3)}
Cs1SeasonalResidualSmoothness.metadata.param_specs = {
    "lag_week": _H(5), "window": _H(40, 3), "min_periods": _N(8, 4),
}
