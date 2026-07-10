"""快速滚动算子：优先 Numba 列向量化，避免 ``rolling.apply`` Python 回调。

供 ``time_series.py`` 与 polars 桥接复用的高性能滚动内核。
通过环境变量 ``FACTOR_ENGINE_DISABLE_NUMBA`` 可禁用 Numba 加速。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

_NUMBA_DISABLED = os.environ.get("FACTOR_ENGINE_DISABLE_NUMBA", "").lower() in (
    "1",
    "true",
    "yes",
)


def _get_linear_weighted_1d():
    """惰性加载 Numba 线性加权滚动内核。

返回:
    Numba JIT 函数或 ``None``（Numba 不可用或已禁用时）。
"""
    if _NUMBA_DISABLED:
        return None
    try:
        from numba import njit  # type: ignore
    except ImportError:
        return None

    @njit(cache=True)
    def _linear_weighted_1d(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
        n = arr.shape[0]
        wlen = weights.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            start = i - wlen + 1
            if start < 0:
                start = 0
            seg_len = i - start + 1
            s = 0.0
            ws = 0.0
            for k in range(seg_len):
                v = arr[start + k]
                if np.isfinite(v):
                    w = weights[wlen - seg_len + k]
                    s += v * w
                    ws += w
            out[i] = s / ws if ws > 0.0 else np.nan
        return out

    return _linear_weighted_1d


_linear_weighted_1d_jit = _get_linear_weighted_1d()


def _linear_weighted_1d_numpy(arr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """纯 numpy 实现的线性加权滚动均值（Numba 回退）。

参数:
    arr: 一维 numpy 数组。
    weights: 线性权重数组（长度 = 窗口）。

返回:
    与 ``arr`` 等长的一维加权均值数组。
"""
    n = arr.shape[0]
    wlen = len(weights)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - wlen + 1)
        seg = arr[start : i + 1]
        ww = weights[-len(seg) :]
        mask = np.isfinite(seg)
        if not mask.any():
            out[i] = np.nan
        else:
            s = seg[mask]
            w = ww[mask]
            out[i] = np.dot(s, w) / w.sum()
    return out


def rolling_linear_weighted(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """线性衰减加权滚动均值（``ts_decay_linear`` / ``WMA`` 语义）。

参数:
    x: 输入宽表 panel。
    window: 滚动窗口长度，``min_periods=1``。

返回:
    线性加权均值 panel。
"""
    weights = np.arange(1, window + 1, dtype=np.float64)
    fn = _linear_weighted_1d_jit or _linear_weighted_1d_numpy
    arr = x.to_numpy(dtype=np.float64, copy=False)
    out = np.empty_like(arr, dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = fn(arr[:, j], weights)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def rolling_regression(
    y: pd.DataFrame,
    x: pd.DataFrame,
    *,
    window: int,
    min_periods: int = 3,
    lag: int = 0,
    retval: str = "slope",
) -> pd.DataFrame:
    """滚动 OLS 回归，全 panel 向量化。

参数:
    y: 因变量宽表 panel。
    x: 自变量宽表 panel。
    window: 滚动窗口长度。
    min_periods: 最少有效样本数，默认 3。
    lag: 自变量滞后 bar 数，负值返回全 NaN。
    retval: 返回值类型，``slope`` / ``intercept`` / ``r_squared`` / ``residual``。

返回:
    指定回归统计量的 panel。
"""
    if lag < 0:
        return pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    if lag > 0:
        x = x.shift(lag)
    r_cov = y.rolling(window=window, min_periods=min_periods).cov(x)
    r_var_x = x.rolling(window=window, min_periods=min_periods).var()
    r_mean_y = y.rolling(window=window, min_periods=min_periods).mean()
    r_mean_x = x.rolling(window=window, min_periods=min_periods).mean()

    slope = r_cov / r_var_x
    if retval == "slope":
        return slope

    intercept = r_mean_y - slope * r_mean_x
    if retval == "intercept":
        return intercept

    if retval == "r_squared":
        corr = x.rolling(window=window, min_periods=min_periods).corr(y)
        return corr * corr

    if retval == "residual":
        resid = y - (slope * x + intercept)
        return resid.rolling(window=window, min_periods=min_periods).mean()

    return slope


def rolling_time_slope(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动对时间索引的 OLS 斜率（``Slope(close, n)`` 语义）。

参数:
    x: 输入宽表 panel。
    window: 滚动窗口长度。

返回:
    时间趋势斜率 panel。
"""
    t = np.arange(window, dtype=np.float64)
    t = t - t.mean()
    denom = float(np.dot(t, t))
    if denom == 0.0:
        return pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    weights = t / denom
    fn = _linear_weighted_1d_jit or _linear_weighted_1d_numpy

    def _weighted_dot(arr: np.ndarray, w: np.ndarray) -> np.ndarray:
        n = arr.shape[0]
        wlen = len(w)
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - wlen + 1)
            seg = arr[start : i + 1]
            ww = w[-len(seg) :]
            mask = np.isfinite(seg)
            if not mask.any():
                out[i] = np.nan
            else:
                out[i] = np.dot(seg[mask], ww[mask])
        return out

    arr = x.to_numpy(dtype=np.float64, copy=False)
    out = np.empty_like(arr, dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = _weighted_dot(arr[:, j], weights)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def rolling_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动窗口内最大值的位置（0-based，窗口内相对下标）。

参数:
    x: 输入宽表 panel。
    window: 滚动窗口长度。

返回:
    极大值位置 panel。
"""
    arr = x.to_numpy(dtype=np.float64, copy=False)
    n, m = arr.shape
    out = np.empty((n, m), dtype=np.float64)
    for j in range(m):
        col = arr[:, j]
        for i in range(n):
            start = max(0, i - window + 1)
            seg = col[start : i + 1]
            if seg.size == 0 or not np.isfinite(seg).any():
                out[i, j] = 0.0
            else:
                out[i, j] = float(np.nanargmax(seg))
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def rolling_argmin(x: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动窗口内最小值的位置（0-based，窗口内相对下标）。

参数:
    x: 输入宽表 panel。
    window: 滚动窗口长度。

返回:
    极小值位置 panel。
"""
    arr = x.to_numpy(dtype=np.float64, copy=False)
    n, m = arr.shape
    out = np.empty((n, m), dtype=np.float64)
    for j in range(m):
        col = arr[:, j]
        for i in range(n):
            start = max(0, i - window + 1)
            seg = col[start : i + 1]
            if seg.size == 0 or not np.isfinite(seg).any():
                out[i, j] = 0.0
            else:
                out[i, j] = float(np.nanargmin(seg))
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def rolling_beta(
    y: pd.DataFrame,
    x: pd.DataFrame,
    *,
    window: int,
    min_periods: int = 2,
) -> pd.DataFrame:
    """滚动 Beta：``Cov(y,x) / Var(x)``。

参数:
    y: 因变量宽表 panel。
    x: 自变量宽表 panel。
    window: 滚动窗口长度。
    min_periods: 最少有效样本数，默认 2。

返回:
    Beta 系数 panel。
"""
    cov = y.rolling(window=window, min_periods=min_periods).cov(x)
    var = x.rolling(window=window, min_periods=min_periods).var()
    return cov / var


def _rolling_top_bottom_1d_numpy(
    arr: np.ndarray,
    window: int,
    k: int,
    *,
    top: bool,
    stat: str,
) -> np.ndarray:
    """一维窗口内 top/bottom-k 的 mean/sum/std（因果窗口）。

参数:
    arr: 一维 numpy 数组。
    window: 滚动窗口长度。
    k: 取 top/bottom 的个数。
    top: ``True`` 取最大 k 个，``False`` 取最小 k 个。
    stat: 聚合方式，``mean`` / ``sum`` / ``std``。

返回:
    与 ``arr`` 等长的一维结果数组。
"""
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        seg = arr[start : i + 1]
        valid = seg[np.isfinite(seg)]
        if valid.size == 0:
            out[i] = np.nan
            continue
        take = min(k, valid.size)
        ordered = np.sort(valid)
        picked = ordered[-take:] if top else ordered[:take]
        if stat == "mean":
            out[i] = picked.mean()
        elif stat == "sum":
            out[i] = picked.sum()
        elif stat == "std":
            out[i] = picked.std(ddof=1) if picked.size >= 2 else np.nan
        else:
            out[i] = np.nan
    return out


def _cum_top_bottom_1d_numpy(
    arr: np.ndarray,
    k: int,
    *,
    top: bool,
    stat: str,
) -> np.ndarray:
    """一维扩展窗口内 top/bottom-k 的 mean/sum。

参数:
    arr: 一维 numpy 数组。
    k: 取 top/bottom 的个数。
    top: ``True`` 取最大 k 个，``False`` 取最小 k 个。
    stat: 聚合方式，``mean`` 或 ``sum``。

返回:
    与 ``arr`` 等长的一维结果数组。
"""
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        seg = arr[: i + 1]
        valid = seg[np.isfinite(seg)]
        if valid.size == 0:
            out[i] = np.nan
            continue
        take = min(k, valid.size)
        ordered = np.sort(valid)
        picked = ordered[-take:] if top else ordered[:take]
        out[i] = picked.mean() if stat == "mean" else picked.sum()
    return out


def _apply_colwise_1d(x: pd.DataFrame, fn) -> pd.DataFrame:
    """对 panel 每列应用一维 numpy 内核函数。

参数:
    x: 输入宽表 panel。
    fn: 接收一维数组返回一维结果的函数。

返回:
    逐列计算后的 panel。
"""
    arr = x.to_numpy(dtype=np.float64, copy=False)
    out = np.empty_like(arr, dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = fn(arr[:, j])
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def rolling_top_n_mean(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """滚动窗口内前 N 大值的均值。

参数:
    x: 输入宽表 panel。
    n: 取最大的 N 个值。

返回:
    top-N 均值 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _rolling_top_bottom_1d_numpy(col, n, n, top=True, stat="mean")
    )


def rolling_top_n_sum(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """滚动窗口内前 N 大值的求和。

参数:
    x: 输入宽表 panel。
    n: 取最大的 N 个值。

返回:
    top-N 求和 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _rolling_top_bottom_1d_numpy(col, n, n, top=True, stat="sum")
    )


def rolling_top_n_std(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """滚动窗口内前 N 大值的标准差。

参数:
    x: 输入宽表 panel。
    n: 取最大的 N 个值。

返回:
    top-N 标准差 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _rolling_top_bottom_1d_numpy(col, n, n, top=True, stat="std")
    )


def rolling_bottom_n_mean(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """滚动窗口内后 N 小值的均值。

参数:
    x: 输入宽表 panel。
    n: 取最小的 N 个值。

返回:
    bottom-N 均值 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _rolling_top_bottom_1d_numpy(col, n, n, top=False, stat="mean")
    )


def rolling_bottom_n_sum(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """滚动窗口内后 N 小值的求和。

参数:
    x: 输入宽表 panel。
    n: 取最小的 N 个值。

返回:
    bottom-N 求和 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _rolling_top_bottom_1d_numpy(col, n, n, top=False, stat="sum")
    )


def cum_top_n_mean(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """扩展窗口内前 N 大值的均值。

参数:
    x: 输入宽表 panel。
    n: 取最大的 N 个值。

返回:
    累积 top-N 均值 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _cum_top_bottom_1d_numpy(col, n, top=True, stat="mean")
    )


def cum_top_n_sum(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """扩展窗口内前 N 大值的求和。

参数:
    x: 输入宽表 panel。
    n: 取最大的 N 个值。

返回:
    累积 top-N 求和 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _cum_top_bottom_1d_numpy(col, n, top=True, stat="sum")
    )


def rolling_top_n_mean_window(x: pd.DataFrame, window: int, k: int) -> pd.DataFrame:
    """指定窗口内前 k 大值的均值。

参数:
    x: 输入宽表 panel。
    window: 滚动窗口长度。
    k: 取最大的 k 个值。

返回:
    窗口 top-k 均值 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _rolling_top_bottom_1d_numpy(col, window, k, top=True, stat="mean")
    )


def rolling_top_n_sum_window(x: pd.DataFrame, window: int, k: int) -> pd.DataFrame:
    """指定窗口内前 k 大值的求和。

参数:
    x: 输入宽表 panel。
    window: 滚动窗口长度。
    k: 取最大的 k 个值。

返回:
    窗口 top-k 求和 panel。
"""
    return _apply_colwise_1d(
        x, lambda col: _rolling_top_bottom_1d_numpy(col, window, k, top=True, stat="sum")
    )
