# -*- coding: utf-8 -*-
"""LQTP 来源的 numpy/pandas 算子片段（内联自历史 operators.py）。

仅被部分 ``fundamental`` / 遗留路径引用；新算子请优先写在对应业务模块并使用 ``@register_operator``。
"""
from __future__ import annotations

import warnings
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


# ============================================================
# 基础运算符
# ============================================================
# +, -, *, /, ==, !=, <, <=, >, >=, and, or, not
# 由 Python / pandas 原生支持，不单独实现函数


# ============================================================
# 标量函数 (Scalar Functions)
# ============================================================

def abs_(x) -> np.ndarray:
    """绝对值。"""
    return np.abs(_to_array(x))


def log_(x) -> np.ndarray:
    """自然对数。"""
    arr = _to_array(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(arr)
    return result


def sqrt_(x) -> np.ndarray:
    """平方根。"""
    arr = _to_array(x)
    with np.errstate(invalid="ignore"):
        return np.sqrt(arr)


def sign_(x) -> np.ndarray:
    """符号函数。"""
    return np.sign(_to_array(x))


def round_(x) -> np.ndarray:
    """四舍五入。"""
    return np.round(_to_array(x))


def signed_sqrt_(x) -> np.ndarray:
    """保留符号后开方: sign(x) * sqrt(|x|)。"""
    arr = _to_array(x)
    return np.sign(arr) * np.sqrt(np.abs(arr))


def sigmoid_(x) -> np.ndarray:
    """Sigmoid 映射: 1 / (1 + e^-x)。"""
    arr = _to_array(x)
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-arr))


def power_(x, n: float) -> np.ndarray:
    """幂函数: x^n。"""
    return np.power(_to_array(x), n)


def cap_(x, lo: float, hi: float) -> np.ndarray:
    """截断到区间 [lo, hi]。"""
    return np.clip(_to_array(x), lo, hi)


def where_(cond, x, y) -> np.ndarray:
    """条件选择: cond ? x : y。"""
    arr_x, arr_y = _to_array(x), _to_array(y)
    cond_arr = _to_array(cond) if not isinstance(cond, (bool, np.bool_)) else cond
    return np.where(cond_arr, arr_x, arr_y)


def iif_(cond, x, y) -> np.ndarray:
    """条件选择（where 别名）。"""
    return where_(cond, x, y)


def is_null_(x) -> np.ndarray:
    """NULL 判断。"""
    return np.isnan(_to_array(x))


def is_nan_(x) -> np.ndarray:
    """NaN 判断。"""
    return np.isnan(_to_array(x))


def nan_to_num_(x, replacement: float = 0.0) -> np.ndarray:
    """NULL/NaN 替换。"""
    arr = _to_array(x)
    arr[np.isnan(arr)] = replacement
    return arr


def coalesce_(*args) -> np.ndarray:
    """返回第一个非 NULL 值。"""
    arrays = [_to_array(a) for a in args]
    result = np.full_like(arrays[0], np.nan)
    for arr in arrays:
        mask = np.isnan(result)
        result[mask] = arr[mask]
    return result


# ============================================================
# 截面函数 (Cross-Sectional Functions)
# ============================================================
# 使用时: df.groupby("TradeDate")["field"].transform(rank)
# 或直接: rank(df["field"])

def rank_(x) -> np.ndarray:
    """截面百分位排名 [0, 1]。"""
    arr = _to_array(x)
    valid = ~np.isnan(arr)
    result = np.full_like(arr, np.nan)
    if valid.sum() > 0:
        ranks = np.argsort(np.argsort(arr[valid]))
        result[valid] = ranks / (len(ranks) - 1) if len(ranks) > 1 else 0.5
    return result


def zscore_(x) -> np.ndarray:
    """截面标准化: (x - μ) / σ。"""
    arr = _to_array(x)
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    if std == 0 or np.isnan(std):
        return np.full_like(arr, 0.0)
    return (arr - mean) / std


def cs_demean_(x) -> np.ndarray:
    """截面去均值: x - μ。"""
    arr = _to_array(x)
    return arr - np.nanmean(arr)


def scale_(x, a: float = 1.0) -> np.ndarray:
    """按截面绝对值和缩放: a * x / sum(|x|)。"""
    arr = _to_array(x)
    total = np.nansum(np.abs(arr))
    if total == 0 or np.isnan(total):
        return np.full_like(arr, 0.0)
    return a * arr / total


def winsorize_(x, n: float = 5.0) -> np.ndarray:
    """按截面 n% 分位裁剪（两端各 n%）。"""
    arr = _to_array(x)
    lo = np.nanpercentile(arr, n)
    hi = np.nanpercentile(arr, 100 - n)
    return np.clip(arr, lo, hi)


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


# ============================================================
# 时序函数 (Time-Series Functions)
# ============================================================
# 使用时: df.groupby("Symbol")["Close"].transform(ts_mean, d=5)

def ts_mean_(x, d: int) -> np.ndarray:
    """窗口均值: (1/d) * sum(x_{t-i}) for i=0..d-1。"""
    arr = _to_array(x)
    return _rolling_apply(arr, d, lambda w: np.nanmean(w))


def ts_sum_(x, d: int) -> np.ndarray:
    """窗口求和。"""
    arr = _to_array(x)
    return _rolling_apply(arr, d, lambda w: np.nansum(w))


def ts_std_(x, d: int) -> np.ndarray:
    """窗口标准差。"""
    arr = _to_array(x)
    return _rolling_apply(arr, d, lambda w: np.nanstd(w, ddof=1))


def ts_max_(x, d: int) -> np.ndarray:
    """窗口最大值。"""
    arr = _to_array(x)
    return _rolling_apply(arr, d, lambda w: np.nanmax(w))


def ts_min_(x, d: int) -> np.ndarray:
    """窗口最小值。"""
    arr = _to_array(x)
    return _rolling_apply(arr, d, lambda w: np.nanmin(w))


def ts_rank_(x, d: int) -> np.ndarray:
    """窗口内百分位排名 [0, 1]。"""
    arr = _to_array(x)
    return _rolling_apply(arr, d, lambda w: rank_(w)[-1] if not np.all(np.isnan(w)) else np.nan)


def ts_delta_(x, d: int) -> np.ndarray:
    """x_t - x_{t-d}。"""
    arr = _to_array(x)
    result = np.full_like(arr, np.nan, dtype=float)
    if d > 0 and d < len(arr):
        result[d:] = arr[d:] - arr[:-d]
    return result


def ts_pct_(x, d: int) -> np.ndarray:
    """x_t / x_{t-d} - 1。"""
    arr = _to_array(x)
    result = np.full_like(arr, np.nan, dtype=float)
    if d > 0 and d < len(arr):
        with np.errstate(divide="ignore", invalid="ignore"):
            result[d:] = arr[d:] / arr[:-d] - 1.0
    return result


def delay_(x, d: int) -> np.ndarray:
    """d 期前值 x_{t-d}。"""
    arr = _to_array(x)
    result = np.full_like(arr, np.nan, dtype=float)
    if d > 0 and d < len(arr):
        result[d:] = arr[:-d]
    return result


def decay_linear_(x, d: int) -> np.ndarray:
    """线性权重衰减均值: sum((d-i) * x_{t-i}) / sum(i), i=0..d-1。"""
    arr = _to_array(x)
    weights = np.arange(d, 0, -1, dtype=float)
    weight_sum = weights.sum()
    return _rolling_apply(arr, d, lambda w: np.nansum(w * weights) / weight_sum)


def ema_(x, d: int) -> np.ndarray:
    """指数加权移动平均: alpha=2/(d+1)。"""
    arr = _to_array(x)
    result = np.full_like(arr, np.nan, dtype=float)
    alpha = 2.0 / (d + 1)
    # 找到第一个非 NaN 值
    valid_idx = np.where(~np.isnan(arr))[0]
    if len(valid_idx) == 0:
        return result
    result[valid_idx[0]] = arr[valid_idx[0]]
    for i in range(valid_idx[0] + 1, len(arr)):
        if np.isnan(arr[i]):
            result[i] = result[i - 1]
        else:
            result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def price_spread_deviation_(x, d: int) -> np.ndarray:
    """相对窗口均值偏离: x_t / mean(x_{t-d+1:t}) - 1。"""
    arr = _to_array(x)
    rolling_mean = _rolling_apply(arr, d, lambda w: np.nanmean(w))
    with np.errstate(divide="ignore", invalid="ignore"):
        return arr / rolling_mean - 1.0


def ts_corr_(x, y, d: int) -> np.ndarray:
    """窗口相关系数: corr(x, y) over d periods。"""
    x_arr, y_arr = _to_array(x), _to_array(y)
    result = np.full_like(x_arr, np.nan, dtype=float)
    if d <= 0:
        return result
    for i in range(d - 1, len(x_arr)):
        xw = x_arr[i - d + 1:i + 1]
        yw = y_arr[i - d + 1:i + 1]
        valid = ~(np.isnan(xw) | np.isnan(yw))
        if valid.sum() >= 3:
            corr = np.corrcoef(xw[valid], yw[valid])[0, 1]
            result[i] = corr if not np.isnan(corr) else np.nan
    return result


def ts_cov_(x, y, d: int) -> np.ndarray:
    """窗口协方差: cov(x, y) over d periods。"""
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
            result[i] = cov if not np.isnan(cov) else np.nan
    return result


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


def ts_quantile_(x, d: int, q: float) -> np.ndarray:
    """窗口分位数: Q_q(x_t,...,x_{t-d+1})，q 在 [0,1]。"""
    arr = _to_array(x)
    return _rolling_apply(arr, d, lambda w: np.nanpercentile(w, q * 100))


def ts_skew_(x, d: int) -> np.ndarray:
    """窗口偏度。"""
    arr = _to_array(x)
    return _rolling_apply(arr, d, lambda w: pd.Series(w).skew() if w is not None else np.nan)


def ts_kurt_(x, d: int) -> np.ndarray:
    """窗口峰度。"""
    arr = _to_array(x)
    return _rolling_apply(arr, d, lambda w: pd.Series(w).kurt() if w is not None else np.nan)


def ts_moment_(x, d: int, k: int) -> np.ndarray:
    """窗口 k 阶中心矩: E[(x-μ)^k]。"""
    arr = _to_array(x)
    return _rolling_apply(arr, d, lambda w: np.nanmean((w - np.nanmean(w)) ** k) if not np.all(np.isnan(w)) else np.nan)


def ts_topk_sum_(x, d: int, k: int) -> np.ndarray:
    """窗口内 Top K 求和。"""
    arr = _to_array(x)
    return _rolling_apply(arr, d, lambda w: np.nansum(np.sort(w)[-k:]) if np.sum(~np.isnan(w)) >= k else np.nan)


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
        # 时序
        x_arr, y_arr = _to_array(x), _to_array(y)
        return ts_corr_(
            pd.Series(x_arr).rank(pct=True).values,
            pd.Series(y_arr).rank(pct=True).values,
            d,
        )


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


def ts_argmax_(x, d: int) -> np.ndarray:
    """窗口最大值位置（最近距离，0=最新）。"""
    arr = _to_array(x)
    result = np.full_like(arr, np.nan, dtype=float)
    if d <= 0:
        return result
    for i in range(d - 1, len(arr)):
        w = arr[i - d + 1:i + 1]
        if np.all(np.isnan(w)):
            continue
        result[i] = d - 1 - np.nanargmax(w)
    return result


def ts_argmin_(x, d: int) -> np.ndarray:
    """窗口最小值位置（最近距离，0=最新）。"""
    arr = _to_array(x)
    result = np.full_like(arr, np.nan, dtype=float)
    if d <= 0:
        return result
    for i in range(d - 1, len(arr)):
        w = arr[i - d + 1:i + 1]
        if np.all(np.isnan(w)):
            continue
        result[i] = d - 1 - np.nanargmin(w)
    return result


# ============================================================
# 财务报表函数
# ============================================================

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


# ============================================================
# 市场与风险函数
# ============================================================

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


# ============================================================
# 中性化函数
# ============================================================

def industry_neutralize_(x, industries: pd.Series | None = None) -> np.ndarray:
    """行业中性化: x - x_hat(industry)。
    需传入 industry 映射（Series, index=Symbol, value=industry_label）。
    """
    arr = _to_array(x)
    if industries is None:
        return zscore_(x)
    # 按行业去均值
    result = arr.copy()
    for ind in industries.unique():
        mask = industries.values == ind
        if mask.sum() > 1:
            result[mask] = arr[mask] - np.nanmean(arr[mask])
    return result


def size_neutralize_(x, market_cap: pd.Series | None = None) -> np.ndarray:
    """市值中性化: x - x_hat(size)。
    需传入市值映射。
    """
    if market_cap is None:
        return zscore_(x)
    return cs_resid_(x, np.log(market_cap.values + 1))


def neutralize_(x, industries=None, market_cap=None) -> np.ndarray:
    """先行业再市值中性化。"""
    result = industry_neutralize_(x, industries)
    if market_cap is not None:
        result = size_neutralize_(result, market_cap)
    return result


# ============================================================
# 流动性/换手率
# ============================================================

def real_turnover_rate_(volume: np.ndarray, effective_float: np.ndarray) -> np.ndarray:
    """真实流通盘换手率: volume / effective_float_shares。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return _to_array(volume) / _to_array(effective_float)



