# -*- coding: utf-8 -*-
"""Panel 算子共用的 numpy 1D 内核（非独立算子库）。"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


def _to_array(x: Any) -> np.ndarray:
    """统一转为 numpy 数组。"""
    if isinstance(x, pd.Series):
        return x.values
    if isinstance(x, (list, tuple)):
        return np.array(x, dtype=float)
    return np.asarray(x, dtype=float)

def _rolling_window(arr: np.ndarray, d: int) -> np.ndarray:
    """滑动窗口视图（避免复制数据）。"""
    shape = (arr.shape[0] - d + 1, d)
    strides = (arr.strides[0], arr.strides[0])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

def _rolling_apply(arr: np.ndarray, d: int, fn: Callable) -> np.ndarray:
    """在滑动窗口上应用函数。"""
    out = np.full_like(arr, np.nan, dtype=float)
    if d <= 0 or len(arr) < d:
        return out
    windows = _rolling_window(arr, d)
    out[d - 1:] = np.apply_along_axis(fn, 1, windows)
    return out

def signed_sqrt_(x) -> np.ndarray:
    """保留符号后开方: sign(x) * sqrt(|x|)。"""
    arr = _to_array(x)
    return np.sign(arr) * np.sqrt(np.abs(arr))

def sigmoid_(x) -> np.ndarray:
    """Sigmoid 映射: 1 / (1 + e^-x)。"""
    arr = _to_array(x)
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-arr))

def cap_(x, lo: float, hi: float) -> np.ndarray:
    """截断到区间 [lo, hi]。"""
    return np.clip(_to_array(x), lo, hi)

def coalesce_(*args) -> np.ndarray:
    """返回第一个非 NULL 值。"""
    arrays = [_to_array(a) for a in args]
    result = np.full_like(arrays[0], np.nan)
    for arr in arrays:
        mask = np.isnan(result)
        result[mask] = arr[mask]
    return result

def rank_(x) -> np.ndarray:
    """截面百分位排名 [0, 1]。"""
    arr = _to_array(x)
    valid = ~np.isnan(arr)
    result = np.full_like(arr, np.nan)
    if valid.sum() > 0:
        ranks = np.argsort(np.argsort(arr[valid]))
        result[valid] = ranks / (len(ranks) - 1) if len(ranks) > 1 else 0.5
    return result

def cs_resid_(y, x) -> np.ndarray:
    """截面线性回归残差: y - (α + βx)。"""
    y_arr = _to_array(y)
    x_arr = _to_array(x)
    valid = ~(np.isnan(y_arr) | np.isnan(x_arr))
    result = np.full_like(y_arr, np.nan)
    if valid.sum() < 3:
        return result
    A = np.vstack([x_arr[valid], np.ones(valid.sum())]).T
    try:
        beta, alpha = np.linalg.lstsq(A, y_arr[valid], rcond=None)[0]
        result[valid] = y_arr[valid] - (alpha + beta * x_arr[valid])
    except np.linalg.LinAlgError:
        pass
    return result

def cs_regression_(y, x, mode: int = 0) -> np.ndarray:
    """截面回归: mode=0 残差, 1 beta, 2 拟合值。"""
    y_arr = _to_array(y)
    x_arr = _to_array(x)
    valid = ~(np.isnan(y_arr) | np.isnan(x_arr))
    result = np.full_like(y_arr, np.nan)
    if valid.sum() < 3:
        return result
    A = np.vstack([x_arr[valid], np.ones(valid.sum())]).T
    try:
        beta, alpha = np.linalg.lstsq(A, y_arr[valid], rcond=None)[0]
        if mode == 0:
            result[valid] = y_arr[valid] - (alpha + beta * x_arr[valid])
        elif mode == 1:
            result[valid] = beta
        else:
            result[valid] = alpha + beta * x_arr[valid]
    except np.linalg.LinAlgError:
        pass
    return result

def price_spread_deviation_(x, d: int) -> np.ndarray:
    """相对窗口均值偏离: x_t / mean(x_{t-d+1:t}) - 1。"""
    arr = _to_array(x)
    rolling_mean = _rolling_apply(arr, d, lambda w: np.nanmean(w))
    with np.errstate(divide="ignore", invalid="ignore"):
        return arr / rolling_mean - 1.0

def ts_regression_slope_(x, y, d: int) -> np.ndarray:
    """窗口回归斜率: cov(x,y)/var(x)。"""
    x_arr, y_arr = _to_array(x), _to_array(y)
    result = np.full_like(x_arr, np.nan, dtype=float)
    if d <= 0:
        return result
    for i in range(d - 1, len(x_arr)):
        xw = x_arr[i - d + 1:i + 1]
        yw = y_arr[i - d + 1:i + 1]
        valid = ~(np.isnan(xw) | np.isnan(yw))
        if valid.sum() >= 3:
            cov = np.cov(xw[valid], yw[valid])[0, 1]
            var = np.var(xw[valid])
            result[i] = cov / var if var > 0 else np.nan
    return result

def ts_moment_(x, d: int, k: int) -> np.ndarray:
    """窗口 k 阶中心矩: E[(x-μ)^k]。"""
    arr = _to_array(x)
    return _rolling_apply(arr, d, lambda w: np.nanmean((w - np.nanmean(w)) ** k) if not np.all(np.isnan(w)) else np.nan)

def rank_corr_(x, y, d: int = 0) -> np.ndarray:
    """秩相关系数: corr(rank(x), rank(y))。
   当 d=0 时做截面秩相关；d>0 时做时序滚动秩相关。"""
    if d == 0:
        # 截面
        rx = rank_(x)
        ry = rank_(y)
        valid = ~(np.isnan(rx) | np.isnan(ry))
        if valid.sum() < 3:
            return np.full_like(_to_array(x), np.nan)
        result = np.full_like(_to_array(x), np.nan)
        result[valid] = np.corrcoef(rx[valid], ry[valid])[0, 1]
        return result
    else:
        # 时序：窗口内分别 rank，避免全局 rank 引入未来数据
        x_arr, y_arr = _to_array(x), _to_array(y)
        result = np.full_like(x_arr, np.nan, dtype=float)
        for i in range(d - 1, len(x_arr)):
            xw = x_arr[i - d + 1:i + 1]
            yw = y_arr[i - d + 1:i + 1]
            valid = ~(np.isnan(xw) | np.isnan(yw))
            if valid.sum() >= 3:
                rx = pd.Series(xw[valid]).rank(pct=True).values
                ry = pd.Series(yw[valid]).rank(pct=True).values
                corr = np.corrcoef(rx, ry)[0, 1]
                result[i] = corr if not np.isnan(corr) else np.nan
        return result

def ts_poly2_coeff_(x, d: int) -> np.ndarray:
    """对时间索引做二次拟合后的二次项系数: x ≈ a + bt + ct², return c。"""
    arr = _to_array(x)
    result = np.full_like(arr, np.nan, dtype=float)
    if d <= 0:
        return result
    t = np.arange(d, dtype=float)
    for i in range(d - 1, len(arr)):
        y = arr[i - d + 1:i + 1]
        valid = ~np.isnan(y)
        if valid.sum() < 3:
            continue
        coeffs = np.polyfit(t[valid], y[valid], 2)
        result[i] = coeffs[0]
    return result

def ts_poly2_resid_(y, x, d: int) -> np.ndarray:
    """二次拟合残差: y - (a + bx + cx²)。"""
    y_arr, x_arr = _to_array(y), _to_array(x)
    result = np.full_like(y_arr, np.nan, dtype=float)
    if d <= 0:
        return result
    for i in range(d - 1, len(y_arr)):
        yw = y_arr[i - d + 1:i + 1]
        xw = x_arr[i - d + 1:i + 1]
        valid = ~(np.isnan(yw) | np.isnan(xw))
        if valid.sum() < 3:
            continue
        coeffs = np.polyfit(xw[valid], yw[valid], 2)
        a, b, c = coeffs
        fitted = a + b * xw + c * xw ** 2
        result[i] = yw[-1] - fitted[-1]
    return result

def digital_count_(x, d: int, threshold: float, run: int) -> np.ndarray:
    """统计连续小波动片段: sum(1(|x_i/x_{i-1}-1| ≤ threshold)) for runs。"""
    arr = _to_array(x)
    result = np.full_like(arr, 0, dtype=float)
    if d <= 0:
        return result
    for i in range(1, min(d, len(arr))):
        with np.errstate(divide="ignore", invalid="ignore"):
            change = np.abs(arr[i] / arr[i - 1] - 1)
            if not np.isnan(change) and change <= threshold:
                result[i] = result[i - 1] + 1
            else:
                result[i] = 0
    # 只保留 >= run 的连续计数
    result[result < run] = 0
    return result

def ts_max_buildup_(x, d: int) -> np.ndarray:
    """统计持续创新高次数: sum(1(x_i = max(x_1..x_i)))。"""
    arr = _to_array(x)
    result = np.full_like(arr, 0, dtype=float)
    if len(arr) == 0:
        return result
    current_max = arr[0]
    count = 0
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            result[i] = count
            continue
        if arr[i] >= current_max or np.isnan(current_max):
            current_max = arr[i]
            count += 1
        result[i] = count
    # 仅取最近 d 期的累计
    if d > 0 and d < len(arr):
        result[:len(arr) - d] = 0
    return result

def ttm_(x) -> np.ndarray:
    """TTM 累加: 当前 + 前 3 报告期。"""
    arr = _to_array(x)
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(len(arr)):
        if i >= 3 and not np.any(np.isnan(arr[i - 3:i + 1])):
            result[i] = np.nansum(arr[max(0, i - 3):i + 1])
        else:
            result[i] = arr[i]
    return result

def quarter_(x) -> np.ndarray:
    """累计值转单季度: x_t - x_{t-1}（一季报返回当期值）。"""
    arr = _to_array(x)
    result = np.full_like(arr, np.nan, dtype=float)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        if not np.isnan(arr[i]) and not np.isnan(arr[i - 1]):
            result[i] = arr[i] - arr[i - 1]
    return result

def yoy_(x) -> np.ndarray:
    """同比增速: x_t / x_{t-4} - 1。"""
    arr = _to_array(x)
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < 5:
        return result
    with np.errstate(divide="ignore", invalid="ignore"):
        result[4:] = arr[4:] / arr[:-4] - 1.0
    return result

def avg2_(x) -> np.ndarray:
    """当期与上期均值: (x_t + x_{t-1}) / 2。"""
    arr = _to_array(x)
    result = np.full_like(arr, np.nan, dtype=float)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        if not np.isnan(arr[i]) and not np.isnan(arr[i - 1]):
            result[i] = (arr[i] + arr[i - 1]) / 2.0
    return result

def rolling_beta_to_market_(ret, index_ret, window: int) -> np.ndarray:
    """滚动 Beta: Cov(r_i, r_m) / Var(r_m)。"""
    return ts_regression_slope_(index_ret, ret, window)

def downside_beta_(ret, index_ret, window: int) -> np.ndarray:
    """市场下跌样本 Beta: 只在 r_m < 0 的样本上计算。"""
    ret_arr, idx_arr = _to_array(ret), _to_array(index_ret)
    result = np.full_like(ret_arr, np.nan, dtype=float)
    if window <= 0:
        return result
    for i in range(window - 1, len(ret_arr)):
        r = ret_arr[i - window + 1:i + 1]
        m = idx_arr[i - window + 1:i + 1]
        valid = ~(np.isnan(r) | np.isnan(m))
        mask = valid & (m < 0)
        if mask.sum() >= 3:
            cov = np.cov(r[mask], m[mask])[0, 1]
            var = np.var(m[mask])
            result[i] = cov / var if var > 0 else np.nan
    return result

def tail_beta_(ret, index_ret, window: int, q: float = 0.05) -> np.ndarray:
    """尾部 Beta: 在 r_m ≤ Q_q(r_m) 的样本上计算 Beta。"""
    ret_arr, idx_arr = _to_array(ret), _to_array(index_ret)
    result = np.full_like(ret_arr, np.nan, dtype=float)
    if window <= 0:
        return result
    for i in range(window - 1, len(ret_arr)):
        r = ret_arr[i - window + 1:i + 1]
        m = idx_arr[i - window + 1:i + 1]
        valid = ~(np.isnan(r) | np.isnan(m))
        threshold = np.nanpercentile(m[valid], q * 100) if valid.sum() > 0 else 0
        mask = valid & (m <= threshold)
        if mask.sum() >= 3:
            cov = np.cov(r[mask], m[mask])[0, 1]
            var = np.var(m[mask])
            result[i] = cov / var if var > 0 else np.nan
    return result

def residual_momentum_capm_(ret, index_ret, window: int) -> np.ndarray:
    """CAPM 残差动量: sum(epsilon), r_i = alpha + beta*r_m + epsilon。"""
    ret_arr, idx_arr = _to_array(ret), _to_array(index_ret)
    result = np.full_like(ret_arr, np.nan, dtype=float)
    if window <= 0:
        return result
    for i in range(window - 1, len(ret_arr)):
        r = ret_arr[i - window + 1:i + 1]
        m = idx_arr[i - window + 1:i + 1]
        valid = ~(np.isnan(r) | np.isnan(m))
        if valid.sum() < 5:
            continue
        A = np.vstack([m[valid], np.ones(valid.sum())]).T
        try:
            beta, alpha = np.linalg.lstsq(A, r[valid], rcond=None)[0]
            residual = r[valid] - (alpha + beta * m[valid])
            result[i] = np.nansum(residual)
        except np.linalg.LinAlgError:
            pass
    return result

def coskewness_to_market_(ret, index_ret, window: int) -> np.ndarray:
    """余偏度: E[(r_i-μ_i)(r_m-μ_m)²] / (σ_i * σ_m²)。"""
    ret_arr, idx_arr = _to_array(ret), _to_array(index_ret)
    result = np.full_like(ret_arr, np.nan, dtype=float)
    if window <= 0:
        return result
    for i in range(window - 1, len(ret_arr)):
        r = ret_arr[i - window + 1:i + 1]
        m = idx_arr[i - window + 1:i + 1]
        valid = ~(np.isnan(r) | np.isnan(m))
        if valid.sum() < 5:
            continue
        ri = r[valid] - np.nanmean(r[valid])
        rm = m[valid] - np.nanmean(m[valid])
        numer = np.nanmean(ri * rm ** 2)
        denom = np.nanstd(r[valid]) * np.nanvar(m[valid])
        result[i] = numer / denom if denom > 0 else np.nan
    return result

def idio_vol_(ret, index_ret, window: int) -> np.ndarray:
    """CAPM 残差波动率: std(epsilon)。"""
    ret_arr, idx_arr = _to_array(ret), _to_array(index_ret)
    result = np.full_like(ret_arr, np.nan, dtype=float)
    if window <= 0:
        return result
    for i in range(window - 1, len(ret_arr)):
        r = ret_arr[i - window + 1:i + 1]
        m = idx_arr[i - window + 1:i + 1]
        valid = ~(np.isnan(r) | np.isnan(m))
        if valid.sum() < 5:
            continue
        A = np.vstack([m[valid], np.ones(valid.sum())]).T
        try:
            beta, alpha = np.linalg.lstsq(A, r[valid], rcond=None)[0]
            residual = r[valid] - (alpha + beta * m[valid])
            result[i] = np.nanstd(residual, ddof=1)
        except np.linalg.LinAlgError:
            pass
    return result

def idio_skew_(ret, index_ret, window: int) -> np.ndarray:
    """CAPM 残差偏度: skew(epsilon)。"""
    ret_arr, idx_arr = _to_array(ret), _to_array(index_ret)
    result = np.full_like(ret_arr, np.nan, dtype=float)
    if window <= 0:
        return result
    for i in range(window - 1, len(ret_arr)):
        r = ret_arr[i - window + 1:i + 1]
        m = idx_arr[i - window + 1:i + 1]
        valid = ~(np.isnan(r) | np.isnan(m))
        if valid.sum() < 5:
            continue
        A = np.vstack([m[valid], np.ones(valid.sum())]).T
        try:
            beta, alpha = np.linalg.lstsq(A, r[valid], rcond=None)[0]
            residual = r[valid] - (alpha + beta * m[valid])
            result[i] = pd.Series(residual).skew()
        except np.linalg.LinAlgError:
            pass
    return result

def real_turnover_rate_(volume: np.ndarray, effective_float: np.ndarray) -> np.ndarray:
    """真实流通盘换手率: volume / effective_float_shares。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return _to_array(volume) / _to_array(effective_float)

