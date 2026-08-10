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
    """逐位置取首个**有限**值（SQL COALESCE + finite-sample 原则）。

    R19-026: ``±Inf`` 与 NaN 一样不可用。待填充位判定统一用
    ``~np.isfinite(result)``——首个参数里的 Inf 不会阻塞后续参数的有限值。
    """
    arrays = [_to_array(a) for a in args]
    result = np.full_like(arrays[0], np.nan)
    for arr in arrays:
        mask = ~np.isfinite(result)
        result[mask] = arr[mask]
    return result

_RANK_METHODS = ("average", "min", "max", "dense", "first")


def rank_(x, method: str = "average") -> np.ndarray:
    """截面百分位排名，映射到 ``[0, 1]``。

    R19-019/020: 并列 tie 统一 authority——默认 ``average``（并列给平均 rank，
    不再让``np.argsort(argsort)`` 的列顺序信号渗入）。只接受有限样本
    （``np.isfinite``，±Inf 与 NaN 一样不可用）。``method`` ∈
    {average, min, max, dense, first}，与 pandas ``Series.rank`` 语义一致。

    参数:
        x: 一维截面样本。
        method: 并列 rank 方法，默认 ``average``。

    返回:
        百分位排名数组，无效位置为 NaN。
    """
    arr = _to_array(x)
    if method not in _RANK_METHODS:
        raise ValueError(
            f"unsupported rank method {method!r}; expected one of {_RANK_METHODS}"
        )
    valid = np.isfinite(arr)
    result = np.full_like(arr, np.nan)
    n = int(valid.sum())
    if n == 0:
        return result
    ranks = pd.Series(arr[valid]).rank(method=method).to_numpy()
    if n > 1:
        result[valid] = (ranks - 1.0) / (n - 1.0)
    else:
        result[valid] = 0.5
    return result

def cs_resid_(y, x) -> np.ndarray:
    """截面线性回归残差：``y - (α + βx)``。

    R19-025: 样本 = ``np.isfinite(y) & np.isfinite(x)``（±Inf 不算合法统计样本）。

参数:
    y: 因变量一维数组。
    x: 自变量一维数组。

返回:
    残差数组，有效样本不足 3 时全 NaN。
"""
    y_arr = _to_array(y)
    x_arr = _to_array(x)
    valid = np.isfinite(y_arr) & np.isfinite(x_arr)
    result = np.full_like(y_arr, np.nan)
    n = int(valid.sum())
    if n < 3:
        return result
    yv = y_arr[valid]
    xv = x_arr[valid]
    mx = xv.mean()
    my = yv.mean()
    xc = xv - mx
    var_x = float((xc * xc).mean())
    if var_x == 0.0 or var_x != var_x:
        return result
    beta = float((xc * (yv - my)).mean()) / var_x
    alpha = my - beta * mx
    result[valid] = yv - (alpha + beta * xv)
    return result


def group_demean_panel_(x: np.ndarray, group: np.ndarray | None = None, fallback: str = "nan") -> np.ndarray:
    """Panel group demean: each row ``x - group_mean``.

    ``group`` may be numeric or object/string labels. Output shape matches ``x``.
    When a row has no usable group labels (``group is None`` or all-NA), the
    ``fallback`` policy decides: ``"nan"`` (default, PIT-correct — matches the
    SQL/Polars path, unknown membership never participates), ``"global"``
    (demean over the full finite row, legacy behavior), ``"keep_original"``.
    """
    xv = np.asarray(x, dtype=float)
    out = np.full(xv.shape, np.nan, dtype=float)
    if xv.ndim != 2:
        raise ValueError("group_demean_panel_ expects a 2D panel")
    if group is None:
        if fallback == "nan":
            return out
        if fallback == "keep_original":
            return np.where(np.isfinite(xv), xv, np.nan)
        with np.errstate(all="ignore"):
            means = np.nanmean(xv, axis=1, keepdims=True)
        return xv - means

    gv = np.asarray(group)
    if gv.shape != xv.shape:
        raise ValueError("group panel shape must match x")
    for i in range(xv.shape[0]):
        row = xv[i]
        g = gv[i]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        # pd.notna handles None/NaN for object and float groups
        g_ok = pd.notna(g)
        valid = finite & g_ok
        if not valid.any():
            # no usable group labels → PIT-correct default: leave NaN.  Only the
            # explicit ``fallback="global"`` policy demeans over the full row.
            if fallback == "global":
                mu = float(np.nanmean(row))
                out[i, finite] = row[finite] - mu
            elif fallback == "keep_original":
                out[i, finite] = row[finite]
            continue
        codes = pd.factorize(pd.Series(g[valid]), use_na_sentinel=True)[0]
        ok = codes >= 0
        if not ok.any():
            mu = float(np.nanmean(row[valid]))
            out[i, valid] = row[valid] - mu
            continue
        pos = np.flatnonzero(valid)[ok]
        c = codes[ok]
        vals = row[valid][ok]
        sums = np.bincount(c, weights=vals)
        cnts = np.bincount(c).astype(float)
        means = sums / cnts
        out[i, pos] = vals - means[c]
    return out


def _log_market_cap(market_cap: np.ndarray) -> np.ndarray:
    """合法市值 ``market_cap>0`` 且有限 → ``log(mc)``；否则 NaN。

    R19-024: 禁止 ``log(max(mc,1))`` 的 silent clip——``<=0``/NaN/±Inf 不再被
    伪造成 ``log(size)=0`` 的“极小公司”，而是作为缺失样本排除。
    """
    cv = np.asarray(market_cap, dtype=float)
    legal = np.isfinite(cv) & (cv > 0.0)
    out = np.full(cv.shape, np.nan, dtype=float)
    out[legal] = np.log(cv[legal])
    return out


def size_resid_panel_(y: np.ndarray, market_cap: np.ndarray) -> np.ndarray:
    """Panel size neutralize: each row residual of ``y ~ log(market_cap)``.

    R19-024: 市值合法性 = ``market_cap>0`` 且有限；非法市值 → NaN（不再 clamp）。
    样本由 ``cs_resid_`` 的 ``np.isfinite`` 统一判定。
    """
    yv = np.asarray(y, dtype=float)
    cv = _log_market_cap(market_cap)
    if yv.shape != cv.shape:
        raise ValueError("size_resid_panel_ shape mismatch")
    out = np.full(yv.shape, np.nan, dtype=float)
    for i in range(yv.shape[0]):
        out[i] = cs_resid_(yv[i], cv[i])
    return out


def industry_size_resid_panel_(
    y: np.ndarray, industry: np.ndarray, market_cap: np.ndarray
) -> np.ndarray:
    """行业+市值双中性（Frisch–Waugh–Lovell，R19-023）。

    ``y_tilde = demean(y | industry)``, ``size_tilde = demean(log_size | industry)``,
    ``resid = residual(y_tilde ~ size_tilde)``。size regressor 先做行业去均值，
    行业平均差异不再重新进入 residual projection（旧实现用原始 log(size)
    回归，不满足 FWL）。市值合法性同 ``size_resid_panel_``。
    """
    yv = np.asarray(y, dtype=float)
    log_size = _log_market_cap(market_cap)
    y_tilde = group_demean_panel_(yv, industry)
    size_tilde = group_demean_panel_(log_size, industry)
    out = np.full(yv.shape, np.nan, dtype=float)
    for i in range(yv.shape[0]):
        out[i] = cs_resid_(y_tilde[i], size_tilde[i])
    return out

def cs_regression_(y, x, mode: int = 0) -> np.ndarray:
    """截面 OLS 回归，按 mode 返回不同量。

    R19-025: 样本 = ``np.isfinite(y) & np.isfinite(x)``（±Inf 不算合法统计样本）。

参数:
    y: 因变量一维数组。
    x: 自变量一维数组。
    mode: ``0`` 残差，``1`` beta，其他为拟合值。

返回:
    与 ``y`` 等长的结果数组。
"""
    y_arr = _to_array(y)
    x_arr = _to_array(x)
    valid = np.isfinite(y_arr) & np.isfinite(x_arr)
    result = np.full_like(y_arr, np.nan)
    n = int(valid.sum())
    if n < 3:
        return result
    yv = y_arr[valid]
    xv = x_arr[valid]
    mx = xv.mean()
    my = yv.mean()
    xc = xv - mx
    var_x = float((xc * xc).mean())
    if var_x == 0.0 or var_x != var_x:
        return result
    beta = float((xc * (yv - my)).mean()) / var_x
    alpha = my - beta * mx
    if mode == 0:
        result[valid] = yv - (alpha + beta * xv)
    elif mode == 1:
        result[valid] = beta
    else:
        result[valid] = alpha + beta * xv
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

    def _nanmean_finite(w):
        wv = np.asarray(w, dtype=float)[np.isfinite(w)]
        return float(wv.mean()) if wv.size else np.nan

    rolling_mean = _rolling_apply(arr, d, _nanmean_finite)
    with np.errstate(divide="ignore", invalid="ignore"):
        return arr / rolling_mean - 1.0

def ts_regression_slope_(x, y, d: int) -> np.ndarray:
    """滚动窗口 OLS 斜率（exact OLS）。

    R19-021/022: 直接使用 centered sums
    ``beta = dot(xc, yc) / dot(xc, xc)``——不依赖 ``np.cov``(n-1) /
    ``np.var``(n) 的 ddof 混用（旧实现把 beta 系统性放大成
    ``beta_OLS * n/(n-1)``）。样本 = 窗口内 x/y 均有限（``np.isfinite``）。

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
        valid = np.isfinite(xw) & np.isfinite(yw)
        if int(valid.sum()) >= 3:
            xv = xw[valid]
            yv = yw[valid]
            xc = xv - xv.mean()
            sxx = float((xc * xc).sum())
            if sxx > 0.0:
                yc = yv - yv.mean()
                result[i] = float((xc * yc).sum()) / sxx
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

    def _moment_1d(w):
        wv = np.asarray(w, dtype=float)[np.isfinite(w)]
        if wv.size == 0:
            return np.nan
        mu = float(wv.mean())
        return float(np.mean((wv - mu) ** k))

    return _rolling_apply(arr, d, _moment_1d)

def ts_rank_corr_(x, y, window: int, method: str = "average") -> np.ndarray:
    """滚动窗口秩相关（Spearman）：``corr(rank(x), rank(y))`` 于 trailing window。

    R19-016..018: 每个时点 t 只使用窗口 ``[t-W+1, t]`` 内的样本——分别对 x 与 y
    ``rank``（window-local、prefix-invariant、禁止全样本 rank，消除 full-sample
    look-ahead），再对 paired 有限样本求 Pearson 相关。tie 按 ``method``
    （默认 ``average``，与全库 ``rank_`` tie authority 一致）。有效对 = 窗口内
    x 与 y 均有限（``np.isfinite``）。

参数:
    x: 第一个序列。
    y: 第二个序列。
    window: trailing 窗口长度。
    method: 并列 rank 方法，默认 ``average``。

返回:
    与 ``x`` 等长的相关系数数组，前 ``window-1`` 个为 NaN。
"""
    x_arr, y_arr = _to_array(x), _to_array(y)
    result = np.full_like(x_arr, np.nan, dtype=float)
    if window <= 0:
        return result
    for i in range(window - 1, len(x_arr)):
        xw = x_arr[i - window + 1:i + 1]
        yw = y_arr[i - window + 1:i + 1]
        valid = np.isfinite(xw) & np.isfinite(yw)
        if int(valid.sum()) < 3:
            continue
        rx = rank_(xw[valid], method=method)
        ry = rank_(yw[valid], method=method)
        corr = np.corrcoef(rx, ry)[0, 1]
        result[i] = corr if np.isfinite(corr) else np.nan
    return result


def cs_rank_corr_(x, y) -> np.ndarray:
    """逐日跨股票秩相关（截面模式）。

    R19-016..018: 对每个 date（行），跨 instruments 独立 rank x 与 y（默认
    ``average`` tie），然后对 paired 有限样本求 Pearson 相关。输入为 2D panel
    ``(dates, instruments)``；输出为每 date 一个标量（shape ``(dates,)``）。
    **若调用方把该标量广播回全股票面板，输出是 GLOBAL_STATE**——同一天内所有
    股票共享同一值，绝不伪装成个股 alpha。
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(1, -1)
        y_arr = y_arr.reshape(1, -1)
    if x_arr.shape != y_arr.shape:
        raise ValueError("cs_rank_corr_ shape mismatch")
    out = np.full(x_arr.shape[0], np.nan, dtype=float)
    for i in range(x_arr.shape[0]):
        rx = rank_(x_arr[i])
        ry = rank_(y_arr[i])
        valid = np.isfinite(rx) & np.isfinite(ry)
        if int(valid.sum()) < 3:
            continue
        corr = np.corrcoef(rx[valid], ry[valid])[0, 1]
        out[i] = corr if np.isfinite(corr) else np.nan
    return out


def rank_corr_(x, y, d: int = 0) -> np.ndarray:
    """秩相关（旧 dual-semantics 入口）。

    R19-016..018: 旧的 ``d==0`` full-sample 截面广播已**彻底移除**——它对 t 时点
    使用 t+1..T 数据，是 full-sample look-ahead。``d>0`` 时委托
    ``ts_rank_corr_``（trailing window）。跨截面秩相关请改用 ``cs_rank_corr_``。

    参数:
        x: 第一个序列。
        y: 第二个序列。
        d: trailing 窗口长度（必须 ``>0``）。
    """
    if d <= 0:
        raise ValueError(
            "rank_corr_(x, y, d) requires d > 0 (trailing window). "
            "Full-sample cross-sectional rank correlation was removed "
            "(R19-016..018); use cs_rank_corr_ on a 2-D (dates, instruments) "
            "panel instead."
        )
    return ts_rank_corr_(x, y, window=d)

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
        valid = np.isfinite(y)
        if int(valid.sum()) < 3:
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
        valid = np.isfinite(yw) & np.isfinite(xw)
        if int(valid.sum()) < 3:
            continue
        coeffs = np.polyfit(xw[valid], yw[valid], 2)
        a, b, c = coeffs
        fitted = a + b * xw + c * xw ** 2
        result[i] = yw[-1] - fitted[-1]
    return result

# R22-058: causal (prior) siblings of the in-sample ts_poly2_coeff/resid.  The
# model is fit on data STRICTLY <= t-1; the current observation is used only for
# evaluation (forecast error) — never jointly fit + scored (R22-054..055).
def ts_poly2_prior_coeff_(x, d: int) -> np.ndarray:
    """二次项系数 c，拟合窗口为 [t-d, t-1]（严格 prior，不含 t）。

参数:
    x: 一维时序数组。
    d: 拟合窗口长度。

返回:
    prior 二次项系数 c 数组（``x ≈ a + bt + ct²`` 在 <= t-1 拟合）。
"""
    arr = _to_array(x)
    result = np.full_like(arr, np.nan, dtype=float)
    if d <= 0:
        return result
    t = np.arange(d, dtype=float)
    for i in range(d, len(arr)):
        past = arr[i - d:i]
        valid = np.isfinite(past)
        if int(valid.sum()) < 3:
            continue
        coeffs = np.polyfit(t[valid], past[valid], 2)
        result[i] = coeffs[0]
    return result


def ts_poly2_forecast_error_(x, d: int) -> np.ndarray:
    """一次步外预测误差：``x_t - poly2_fit([t-d, t-1])`` 在 t。

    模型只在 <= t-1 拟合，当前观测仅用于评估（R22-055）。
"""
    arr = _to_array(x)
    result = np.full_like(arr, np.nan, dtype=float)
    if d <= 0:
        return result
    t = np.arange(d, dtype=float)
    for i in range(d, len(arr)):
        past = arr[i - d:i]
        valid = np.isfinite(past)
        if int(valid.sum()) < 3:
            continue
        coeffs = np.polyfit(t[valid], past[valid], 2)
        pred = coeffs[0] * d * d + coeffs[1] * d + coeffs[2]
        result[i] = arr[i] - pred
    return result


def ts_poly2_forecast_error_z_(x, d: int) -> np.ndarray:
    """标准化一步外预测误差：forecast_error / in-sample 残差 std(ddof=1)。

    需要 >= 4 个有效点（3 个系数 + 1 个残差自由度）。
"""
    arr = _to_array(x)
    result = np.full_like(arr, np.nan, dtype=float)
    if d <= 0:
        return result
    t = np.arange(d, dtype=float)
    for i in range(d, len(arr)):
        past = arr[i - d:i]
        valid = np.isfinite(past)
        if int(valid.sum()) < 4:
            continue
        tv = t[valid]
        pv = past[valid]
        coeffs = np.polyfit(tv, pv, 2)
        fitted = coeffs[0] * tv ** 2 + coeffs[1] * tv + coeffs[2]
        std = (pv - fitted).std(ddof=1)
        if not np.isfinite(std) or std <= 0:
            continue
        pred = coeffs[0] * d * d + coeffs[1] * d + coeffs[2]
        result[i] = (arr[i] - pred) / std
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
        valid = np.isfinite(r) & np.isfinite(m)
        mask = valid & (m < 0)
        if int(mask.sum()) >= 3:
            # R19-021/022: centered sums，avoid np.cov(n-1)/np.var(n) ddof mix
            mv = m[mask]
            mc = mv - mv.mean()
            sxx = float((mc * mc).sum())
            if sxx > 0.0:
                rc = r[mask] - r[mask].mean()
                result[i] = float((mc * rc).sum()) / sxx
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
        valid = np.isfinite(r) & np.isfinite(m)
        threshold = np.percentile(m[valid], q * 100) if int(valid.sum()) > 0 else 0
        mask = valid & (m <= threshold)
        if int(mask.sum()) >= 3:
            # R19-021/022: centered sums，avoid np.cov(n-1)/np.var(n) ddof mix
            mv = m[mask]
            mc = mv - mv.mean()
            sxx = float((mc * mc).sum())
            if sxx > 0.0:
                rc = r[mask] - r[mask].mean()
                result[i] = float((mc * rc).sum()) / sxx
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
        valid = np.isfinite(r) & np.isfinite(m)
        if int(valid.sum()) < 5:
            continue
        A = np.vstack([m[valid], np.ones(int(valid.sum()))]).T
        try:
            beta, alpha = np.linalg.lstsq(A, r[valid], rcond=None)[0]
            residual = r[valid] - (alpha + beta * m[valid])
            result[i] = float(np.sum(residual))
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
        valid = np.isfinite(r) & np.isfinite(m)
        if int(valid.sum()) < 5:
            continue
        rv = r[valid]
        mv = m[valid]
        ri = rv - rv.mean()
        rm = mv - mv.mean()
        numer = float(np.mean(ri * rm ** 2))
        denom = float(np.std(rv) * np.var(mv))
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
        valid = np.isfinite(r) & np.isfinite(m)
        if int(valid.sum()) < 5:
            continue
        A = np.vstack([m[valid], np.ones(int(valid.sum()))]).T
        try:
            beta, alpha = np.linalg.lstsq(A, r[valid], rcond=None)[0]
            residual = r[valid] - (alpha + beta * m[valid])
            result[i] = float(np.std(residual, ddof=1))
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
        valid = np.isfinite(r) & np.isfinite(m)
        if int(valid.sum()) < 5:
            continue
        A = np.vstack([m[valid], np.ones(int(valid.sum()))]).T
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
