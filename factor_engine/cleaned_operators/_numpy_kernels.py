# -*- coding: utf-8 -*-
"""Panel 算子共用的 numpy 1D 内核（非独立算子库）。

提供截面/时序/价量的逐列 numpy 实现，供 pandas 算子与 polars 桥接复用。
函数名以下划线结尾表示内部内核，非 DSL 直接暴露。
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


def _to_array(x: Any) -> np.ndarray:
    """统一将输入转为 numpy 浮点数组。

参数:
    x: ``pd.Series``、列表、元组或类数组对象。

返回:
    ``np.ndarray`` 浮点数组。
"""
    if isinstance(x, pd.Series):
        return x.values
    if isinstance(x, (list, tuple)):
        return np.array(x, dtype=float)
    return np.asarray(x, dtype=float)

def _rolling_window(arr: np.ndarray, d: int) -> np.ndarray:
    """构造滑动窗口 strided 视图（零拷贝）。

参数:
    arr: 一维 numpy 数组。
    d: 窗口长度。

返回:
    形状 ``(len(arr)-d+1, d)`` 的窗口视图数组。
"""
    shape = (arr.shape[0] - d + 1, d)
    strides = (arr.strides[0], arr.strides[0])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

def _rolling_apply(arr: np.ndarray, d: int, fn: Callable) -> np.ndarray:
    """在滑动窗口上逐窗应用自定义函数。

参数:
    arr: 一维 numpy 数组。
    d: 窗口长度。
    fn: 接收窗口数组返回标量的函数。

返回:
    与 ``arr`` 等长的一维结果数组，前 ``d-1`` 个为 NaN。
"""
    out = np.full_like(arr, np.nan, dtype=float)
    if d <= 0 or len(arr) < d:
        return out
    windows = _rolling_window(arr, d)
    out[d - 1:] = np.apply_along_axis(fn, 1, windows)
    return out

def signed_sqrt_(x) -> np.ndarray:
    """保留符号的开方：``sign(x) * sqrt(|x|)``。

参数:
    x: 输入序列。

返回:
    逐元素带符号平方根数组。
"""
    arr = _to_array(x)
    return np.sign(arr) * np.sqrt(np.abs(arr))

def sigmoid_(x) -> np.ndarray:
    """Sigmoid 映射：``1 / (1 + exp(-x))``。

参数:
    x: 输入序列。

返回:
    逐元素 sigmoid 数组。
"""
    arr = _to_array(x)
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-arr))

def cap_(x, lo: float, hi: float) -> np.ndarray:
    """将数值截断到区间 ``[lo, hi]``。

参数:
    x: 输入序列。
    lo: 下界。
    hi: 上界。

返回:
    截断后的 numpy 数组。
"""
    return np.clip(_to_array(x), lo, hi)

def coalesce_(*args) -> np.ndarray:
    """返回第一个非 NaN 值（SQL COALESCE 语义）。

参数:
    *args: 多个同长度输入序列。

返回:
    逐位置取首个有限值的数组。
"""
    arrays = [_to_array(a) for a in args]
    result = np.full_like(arrays[0], np.nan)
    for arr in arrays:
        mask = np.isnan(result)
        result[mask] = arr[mask]
    return result

def rank_(x) -> np.ndarray:
    """截面百分位排名，映射到 ``[0, 1]``。

参数:
    x: 一维截面样本。

返回:
    百分位排名数组，无效位置为 NaN。
"""
    arr = _to_array(x)
    valid = ~np.isnan(arr)
    result = np.full_like(arr, np.nan)
    if valid.sum() > 0:
        ranks = np.argsort(np.argsort(arr[valid]))
        result[valid] = ranks / (len(ranks) - 1) if len(ranks) > 1 else 0.5
    return result

def cs_resid_(y, x) -> np.ndarray:
    """截面线性回归残差：``y - (α + βx)``。

参数:
    y: 因变量一维数组。
    x: 自变量一维数组。

返回:
    残差数组，有效样本不足 3 时全 NaN。
"""
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
    """截面 OLS 回归，按 mode 返回不同量。

参数:
    y: 因变量一维数组。
    x: 自变量一维数组。
    mode: ``0`` 残差，``1`` beta，其他为拟合值。

返回:
    与 ``y`` 等长的结果数组。
"""
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
    """相对窗口均值偏离：``x_t / mean(x_{t-d+1:t}) - 1``。

参数:
    x: 一维时序数组。
    d: 滚动窗口长度。

返回:
    偏离度数组。
"""
    arr = _to_array(x)
    rolling_mean = _rolling_apply(arr, d, lambda w: np.nanmean(w))
    with np.errstate(divide="ignore", invalid="ignore"):
        return arr / rolling_mean - 1.0

def ts_regression_slope_(x, y, d: int) -> np.ndarray:
    """滚动窗口回归斜率：``cov(x,y)/var(x)``。

参数:
    x: 自变量一维数组。
    y: 因变量一维数组。
    d: 窗口长度。

返回:
    斜率数组。
"""
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
    """滚动窗口 k 阶中心矩：``E[(x-μ)^k]``。

参数:
    x: 一维时序数组。
    d: 窗口长度。
    k: 矩的阶数。

返回:
    k 阶中心矩数组。
"""
    arr = _to_array(x)
    return _rolling_apply(arr, d, lambda w: np.nanmean((w - np.nanmean(w)) ** k) if not np.all(np.isnan(w)) else np.nan)

def rank_corr_(x, y, d: int = 0) -> np.ndarray:
    """秩相关系数：``corr(rank(x), rank(y))``。

参数:
    x: 第一个序列。
    y: 第二个序列。
    d: ``0`` 为截面秩相关；``>0`` 为时序滚动秩相关。

返回:
    相关系数数组（截面模式为广播标量）。
"""
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
    """对时间索引做二次拟合，返回二次项系数 c。

参数:
    x: 一维时序数组。
    d: 拟合窗口长度。

返回:
    二次项系数 c 数组（``x ≈ a + bt + ct²``）。
"""
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
    """二次拟合残差：``y - (a + bx + cx²)`` 在窗口末 bar。

参数:
    y: 因变量一维数组。
    x: 自变量一-dimensional数组。
    d: 拟合窗口长度。

返回:
    窗口末 bar 的残差数组。
"""
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
    """统计连续小波动片段长度。

参数:
    x: 一维价格/收益数组。
    d: 回溯长度上限。
    threshold: 相邻变化率阈值。
    run: 最小连续计数，低于此清零。

返回:
    连续小波动计数数组。
"""
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
    """统计持续创新高次数。

参数:
    x: 一维时序数组。
    d: 仅保留最近 d 期的累计计数（``d>0`` 时）。

返回:
    创新高累计次数数组。
"""
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
    """TTM 累加：当前值 + 前 3 个报告期。

参数:
    x: 季度累计或单季值一维数组。

返回:
    滚动四季累加数组。
"""
    arr = _to_array(x)
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(len(arr)):
        if i >= 3 and not np.any(np.isnan(arr[i - 3:i + 1])):
            result[i] = np.nansum(arr[max(0, i - 3):i + 1])
        else:
            result[i] = arr[i]
    return result

def quarter_(x) -> np.ndarray:
    """累计值转单季度：``x_t - x_{t-1}``（首期为当期值）。

参数:
    x: 累计值一维数组。

返回:
    单季度值数组。
"""
    arr = _to_array(x)
    result = np.full_like(arr, np.nan, dtype=float)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        if not np.isnan(arr[i]) and not np.isnan(arr[i - 1]):
            result[i] = arr[i] - arr[i - 1]
    return result

def yoy_(x) -> np.ndarray:
    """同比增速：``x_t / x_{t-4} - 1``。

参数:
    x: 季度值一维数组。

返回:
    同比增速数组，前 4 期为 NaN。
"""
    arr = _to_array(x)
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < 5:
        return result
    with np.errstate(divide="ignore", invalid="ignore"):
        result[4:] = arr[4:] / arr[:-4] - 1.0
    return result

def avg2_(x) -> np.ndarray:
    """当期与上期均值：``(x_t + x_{t-1}) / 2``。

参数:
    x: 一维时序数组。

返回:
    两期均值数组。
"""
    arr = _to_array(x)
    result = np.full_like(arr, np.nan, dtype=float)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        if not np.isnan(arr[i]) and not np.isnan(arr[i - 1]):
            result[i] = (arr[i] + arr[i - 1]) / 2.0
    return result

def rolling_beta_to_market_(ret, index_ret, window: int) -> np.ndarray:
    """滚动 Beta：``Cov(r_i, r_m) / Var(r_m)``。

参数:
    ret: 个股收益一维数组。
    index_ret: 市场收益一维数组。
    window: 滚动窗口长度。

返回:
    Beta 数组。
"""
    return ts_regression_slope_(index_ret, ret, window)

def downside_beta_(ret, index_ret, window: int) -> np.ndarray:
    """下行 Beta：仅在市场收益 ``< 0`` 的样本上估计。

参数:
    ret: 个股收益一维数组。
    index_ret: 市场收益一维数组。
    window: 滚动窗口长度。

返回:
    下行 Beta 数组。
"""
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
    """尾部 Beta：在市场收益处于低分位样本上估计。

参数:
    ret: 个股收益一维数组。
    index_ret: 市场收益一维数组。
    window: 滚动窗口长度。
    q: 市场收益分位阈值，默认 ``0.05``。

返回:
    尾部 Beta 数组。
"""
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
    """CAPM 残差动量：窗口内残差 ``epsilon`` 之和。

参数:
    ret: 个股收益一维数组。
    index_ret: 市场收益一维数组。
    window: 滚动窗口长度。

返回:
    残差动量数组。
"""
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
    """余偏度：``E[(r_i-μ_i)(r_m-μ_m)²] / (σ_i * σ_m²)``。

参数:
    ret: 个股收益一维数组。
    index_ret: 市场收益一维数组。
    window: 滚动窗口长度。

返回:
    余偏度数组。
"""
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
    """CAPM 残差波动率：``std(epsilon)``。

参数:
    ret: 个股收益一维数组。
    index_ret: 市场收益一维数组。
    window: 滚动窗口长度。

返回:
    特质波动率数组。
"""
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
    """CAPM 残差偏度：``skew(epsilon)``。

参数:
    ret: 个股收益一维数组。
    index_ret: 市场收益一维数组。
    window: 滚动窗口长度。

返回:
    特质偏度数组。
"""
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
    """真实流通盘换手率：``volume / effective_float``。

参数:
    volume: 成交量数组。
    effective_float: 有效流通股本数组。

返回:
    换手率数组。
"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return _to_array(volume) / _to_array(effective_float)

