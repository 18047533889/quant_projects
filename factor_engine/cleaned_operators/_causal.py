"""因果性辅助：滚动/扩展计算，避免使用未来样本。

本模块提供 PIT-safe 的时序变换内核，供 pandas/polars 算子桥接调用。
所有函数仅使用截至当前时点的历史窗口，禁止引用未来 bar。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SPECTRAL_WINDOW = 20
DEFAULT_WAVELET_WINDOW = 20


def causal_lag(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """因果滞后：仅允许非负滞后，负值视为未来位移。

参数:
    x: 输入宽表 panel（index=时间, columns=标的）。
    n: 滞后 bar 数；``n < 0`` 时返回全 NaN。

返回:
    滞后后的 ``pd.DataFrame``；``n=0`` 时原样返回。
"""
    lag = int(n)
    if lag < 0:
        # A negative lag is a future reference.  Reject it instead of silently
        # returning an all-NaN panel, which hides the illegal formula as a
        # factor with zero backtest coverage.
        from factor_engine.backend.operator_errors import FutureReferenceError

        raise FutureReferenceError(f"causal_lag: negative lag {lag} references future data")
    if lag == 0:
        return x
    return x.shift(lag)


def rolling_rfft_feature(
    arr: np.ndarray,
    window: int,
    *,
    feature: str = "dominant",
) -> np.ndarray:
    """滚动 RFFT 特征：仅用截至当前时点的窗口做频谱分析。

参数:
    arr: 一维 numpy 数组。
    window: 滚动窗口长度。
    feature: 特征类型，``dominant`` / ``energy`` / 其他（均值）。

返回:
    与 ``arr`` 等长的一维特征数组，窗口不足处为 NaN。
"""
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    w = max(2, int(window))
    for i in range(n):
        seg = arr[max(0, i - w + 1): i + 1]
        valid = seg[np.isfinite(seg)]
        if valid.size < 2:
            continue
        spec = np.abs(np.fft.rfft(valid))
        if feature == "dominant":
            out[i] = spec[1:].max() if spec.size > 1 else float(spec[0])
        elif feature == "energy":
            out[i] = float(np.sum(spec ** 2))
        else:
            out[i] = float(spec.mean())
    return out


def rolling_panel_rfft(
    x: pd.DataFrame,
    window: int = DEFAULT_SPECTRAL_WINDOW,
    *,
    feature: str = "dominant",
) -> pd.DataFrame:
    """panel 逐列滚动 RFFT 特征。

参数:
    x: 输入宽表 panel。
    window: 滚动窗口长度，默认 ``DEFAULT_SPECTRAL_WINDOW``。
    feature: 频谱特征类型，默认 ``dominant``。

返回:
    与 ``x`` 同形的特征 panel。
"""
    arr = x.to_numpy(dtype=np.float64, copy=False)
    out = np.empty_like(arr, dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = rolling_rfft_feature(arr[:, j], window, feature=feature)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def expanding_svd_diagonal(x: pd.DataFrame) -> pd.DataFrame:
    """扩展窗口 SVD 奇异值：每个时点仅使用历史行。

参数:
    x: 输入宽表 panel（多列矩阵）。

返回:
    每行写入截至当期的奇异值对角元素，其余为 NaN。
"""
    arr = x.values.astype(float)
    t_rows, n_cols = arr.shape
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(2, t_rows):
        sub = arr[: i + 1]
        valid_rows = np.all(np.isfinite(sub), axis=1)
        if valid_rows.sum() < 2:
            continue
        c = sub[valid_rows]
        try:
            _, s, _ = np.linalg.svd(c, full_matrices=False)
            k = min(len(s), n_cols)
            out[i, :k] = s[:k]
        except np.linalg.LinAlgError:
            pass
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def expanding_eig_diagonal(x: pd.DataFrame) -> pd.DataFrame:
    """扩展窗口协方差特征值：每个时点仅使用历史行。

参数:
    x: 输入宽表 panel。

返回:
    每行写入截至当期的特征值，列数对齐输入宽度。
"""
    arr = x.values.astype(float)
    t_rows, n_cols = arr.shape
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(2, t_rows):
        sub = arr[: i + 1]
        valid_rows = np.all(np.isfinite(sub), axis=1)
        if valid_rows.sum() < 2:
            continue
        c = sub[valid_rows]
        try:
            if c.shape[0] >= c.shape[1]:
                cov = np.cov(c, rowvar=False)
            else:
                cov = np.cov(c.T, rowvar=False)
            if cov.ndim < 2:
                continue
            eigenvalues, _ = np.linalg.eig(cov)
            ev = np.real(eigenvalues)
            k = min(len(ev), n_cols)
            out[i, :k] = ev[:k]
        except (np.linalg.LinAlgError, ValueError):
            pass
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def rolling_wavelet_column(
    col: np.ndarray,
    window: int,
    *,
    threshold: float = 0.1,
    wavelet: str = "db4",
) -> np.ndarray:
    """单列滚动小波去噪：窗口内分解，仅取当前时点重构值。

参数:
    col: 一维 numpy 数组。
    window: 滚动窗口长度。
    threshold: 小波系数软阈值。
    wavelet: 小波基名称，默认 ``db4``。

返回:
    与 ``col`` 等长的一维去噪结果数组。
"""
    try:
        import pywt
    except ImportError:
        return np.full(len(col), np.nan, dtype=np.float64)

    n = len(col)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    w = max(8, int(window))
    for i in range(n):
        seg = col[max(0, i - w + 1): i + 1]
        valid = seg[np.isfinite(seg)]
        if valid.size < 8:
            if valid.size > 0:
                out[i] = valid[-1]
            continue
        try:
            max_level = pywt.dwt_max_level(len(valid), wavelet)
            level = min(3, max_level)
            coeffs = pywt.wavedec(valid, wavelet, level=level)
            coeffs[1:] = [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
            rec = pywt.waverec(coeffs, wavelet)
            out[i] = rec[-1]
        except Exception:
            out[i] = valid[-1]
    return out


def rolling_wavelet_panel(
    x: pd.DataFrame,
    window: int = DEFAULT_WAVELET_WINDOW,
    *,
    threshold: float = 0.1,
    wavelet_type: str = "db4",
) -> pd.DataFrame:
    """panel 逐列滚动小波去噪。

参数:
    x: 输入宽表 panel。
    window: 滚动窗口长度，默认 ``DEFAULT_WAVELET_WINDOW``。
    threshold: 小波系数软阈值。
    wavelet_type: 小波基名称。

返回:
    与 ``x`` 同形的去噪 panel。
"""
    arr = x.to_numpy(dtype=np.float64, copy=False)
    out = np.empty_like(arr, dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = rolling_wavelet_column(
            arr[:, j], window, threshold=threshold, wavelet=wavelet_type
        )
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def rolling_irfft_last(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动 RFFT→IRFFT，仅输出窗口末样本（因果）。

参数:
    arr: 一维 numpy 数组。
    window: 滚动窗口长度。

返回:
    与 ``arr`` 等长的一维重构末值数组。
"""
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    w = max(2, int(window))
    for i in range(n):
        seg = arr[max(0, i - w + 1): i + 1]
        valid = seg[np.isfinite(seg)]
        if valid.size < 2:
            continue
        try:
            spec = np.fft.rfft(valid)
            rec = np.fft.irfft(spec, n=len(valid))
            out[i] = rec[-1]
        except Exception:
            out[i] = valid[-1]
    return out


def rolling_panel_irfft(x: pd.DataFrame, window: int = DEFAULT_SPECTRAL_WINDOW) -> pd.DataFrame:
    """panel 逐列滚动 IRFFT 末值。

参数:
    x: 输入宽表 panel。
    window: 滚动窗口长度，默认 ``DEFAULT_SPECTRAL_WINDOW``。

返回:
    与 ``x`` 同形的重构 panel。
"""
    arr = x.to_numpy(dtype=np.float64, copy=False)
    out = np.empty_like(arr, dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = rolling_irfft_last(arr[:, j], window)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def causal_convolve_column(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """因果 FIR 卷积：``y[n] = sum_k b[k] * x[n-k]``。

参数:
    a: 输入信号一维数组。
    b: FIR 滤波器系数一维数组。

返回:
    与 ``a`` 等长的卷积结果数组。
"""
    from scipy.signal import lfilter

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not np.isfinite(b).any():
        return np.full(len(a), np.nan, dtype=np.float64)
    kernel = np.where(np.isfinite(b), b, 0.0)
    x = np.where(np.isfinite(a), a, 0.0)
    return lfilter(kernel, [1.0], x)


def causal_correlate_column(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """因果互相关：每个时点仅与截至当前的样本重叠。

参数:
    a: 第一个一维数组。
    b: 第二个一维数组。

返回:
    与 ``a`` 等长的互相关结果数组。
"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = len(a)
    out = np.full(n, np.nan, dtype=np.float64)
    b_valid = b[np.isfinite(b)]
    if b_valid.size == 0:
        return out
    m = len(b_valid)
    for i in range(n):
        k = min(i + 1, m)
        seg_a = a[i - k + 1: i + 1][::-1]
        seg_b = b_valid[:k]
        mask = np.isfinite(seg_a) & np.isfinite(seg_b)
        if not mask.any():
            continue
        out[i] = float(np.dot(seg_a[mask], seg_b[mask]))
    return out


def expanding_unwrap_panel(x: pd.DataFrame) -> pd.DataFrame:
    """扩展窗口相位展开：每个时点仅 unwrap 历史前缀。

参数:
    x: 输入宽表 panel（相位角）。

返回:
    与 ``x`` 同形的展开后 panel。
"""
    out = np.full(x.shape, np.nan, dtype=np.float64)
    arr = x.to_numpy(dtype=np.float64, copy=False)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        for i in range(len(col)):
            seg = col[: i + 1]
            valid = seg[np.isfinite(seg)]
            if valid.size == 0:
                continue
            out[i, j] = float(np.unwrap(valid)[-1])
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def expanding_geometric_mean(x: pd.DataFrame) -> pd.DataFrame:
    """扩展几何平均：基于对数绝对值的 expanding mean 还原。

参数:
    x: 输入宽表 panel。

返回:
    截至各时点的几何平均 panel。
"""
    log_abs = np.log(x.abs().replace(0, np.nan))
    return np.exp(log_abs.expanding(min_periods=1).mean())


def expanding_harmonic_mean(x: pd.DataFrame) -> pd.DataFrame:
    """扩展调和平均。

参数:
    x: 输入宽表 panel（非零值参与计算）。

返回:
    截至各时点的调和平均 panel。
"""
    inv = 1.0 / x.replace(0, np.nan)
    count = x.expanding(min_periods=1).count().astype(float)
    inv_sum = inv.expanding(min_periods=1).sum()
    return count / inv_sum


def expanding_panel_stat(x: pd.DataFrame, stat: str, **kwargs) -> pd.DataFrame:
    """按列扩展聚合统计，避免全样本广播引入未来值。

参数:
    x: 输入宽表 panel。
    stat: 统计量名称（``mean``/``sum``/``count``/``std``/``var``/``sem``/``product``）。
    **kwargs: 如 ``min_periods``、``ddof`` 等 pandas expanding 参数。

返回:
    扩展聚合结果 panel。

异常:
    ValueError: 不支持的 ``stat`` 名称。
"""
    stat = stat.lower()
    min_periods = int(kwargs.get("min_periods", 1))
    if stat == "mean":
        return x.expanding(min_periods=min_periods).mean()
    if stat == "sum":
        return x.expanding(min_periods=min_periods).sum()
    if stat == "count":
        return x.expanding(min_periods=min_periods).count().astype(float)
    if stat == "std":
        return x.expanding(min_periods=max(min_periods, 2)).std(ddof=int(kwargs.get("ddof", 1)))
    if stat == "var":
        return x.expanding(min_periods=max(min_periods, 2)).var(ddof=int(kwargs.get("ddof", 0)))
    if stat == "sem":
        return x.expanding(min_periods=max(min_periods, 2)).sem()
    if stat == "product":
        return x.expanding(min_periods=min_periods).apply(
            lambda s: float(np.prod(s[np.isfinite(s)])) if np.isfinite(s).any() else np.nan,
            raw=True,
        )
    raise ValueError(f"unsupported expanding stat: {stat}")


def expanding_first_not_null(x: pd.DataFrame) -> pd.DataFrame:
    """扩展窗口内首个非空值向前填充。

参数:
    x: 输入宽表 panel。

返回:
    截至各时点所见首个有限值的 panel。
"""
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
    for col in x.columns:
        seen = np.nan
        for i, v in enumerate(x[col].to_numpy(dtype=float, copy=False)):
            if np.isfinite(v) and not np.isfinite(seen):
                seen = v
            if np.isfinite(seen):
                out.iloc[i, out.columns.get_loc(col)] = seen
    return out


def expanding_argext(x: pd.DataFrame, which: str = "max") -> pd.DataFrame:
    """扩展窗口内极值位置（iloc 下标）。

参数:
    x: 输入宽表 panel。
    which: ``max`` 或 ``min``，指定取极大或极小位置。

返回:
    截至各时点窗口内极值位置的 panel。
"""
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
    for col in x.columns:
        s = x[col].to_numpy(dtype=float, copy=False)
        for i in range(len(s)):
            seg = s[: i + 1]
            valid = seg[np.isfinite(seg)]
            if valid.size == 0:
                continue
            idx = int(np.argmax(valid) if which == "max" else np.argmin(valid))
            out.iloc[i, out.columns.get_loc(col)] = float(idx)
    return out


def expanding_univariate(x: pd.DataFrame, fn, *, min_periods: int = 1) -> pd.DataFrame:
    """对每列历史前缀调用一元统计函数。

参数:
    x: 输入宽表 panel。
    fn: 接收 ``valid_array`` 返回标量的函数。
    min_periods: 最少有效样本数。

返回:
    扩展一元统计结果 panel。
"""
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
    for col in x.columns:
        vals = x[col].to_numpy(dtype=float, copy=False)
        for i in range(len(vals)):
            if i + 1 < min_periods:
                continue
            valid = vals[: i + 1]
            valid = valid[np.isfinite(valid)]
            if valid.size < min_periods:
                continue
            try:
                out.iloc[i, out.columns.get_loc(col)] = float(fn(valid))
            except Exception:
                pass
    return out


def expanding_bivariate(
    x: pd.DataFrame,
    y: pd.DataFrame,
    fn,
    *,
    min_periods: int = 3,
) -> pd.DataFrame:
    """对每列对齐前缀调用二元统计函数。

参数:
    x: 第一个输入宽表 panel。
    y: 第二个输入宽表 panel。
    fn: 接收 ``(x_valid, y_valid)`` 返回标量的函数。
    min_periods: 最少配对样本数。

返回:
    扩展二元统计结果 panel。
"""
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
    for col in x.columns:
        y_col = y[col] if col in y.columns else y.iloc[:, 0]
        xc = x[col].to_numpy(dtype=float, copy=False)
        yc = y_col.reindex(x.index).to_numpy(dtype=float, copy=False)
        for i in range(len(xc)):
            if i + 1 < min_periods:
                continue
            paired = np.column_stack([xc[: i + 1], yc[: i + 1]])
            paired = paired[np.all(np.isfinite(paired), axis=1)]
            if paired.shape[0] < min_periods:
                continue
            try:
                out.iloc[i, out.columns.get_loc(col)] = float(fn(paired[:, 0], paired[:, 1]))
            except Exception:
                pass
    return out


def causal_linear_extrapolate_panel(x: pd.DataFrame) -> pd.DataFrame:
    """仅用最近两个历史已知点做线性外推，不引用未来样本。

参数:
    x: 输入宽表 panel。

返回:
    因果线性外推后的 panel。
"""
    out = x.copy()
    for col in x.columns:
        arr = x[col].to_numpy(dtype=float, copy=False)
        filled = np.full_like(arr, np.nan, dtype=float)
        for i in range(len(arr)):
            if np.isfinite(arr[i]):
                filled[i] = arr[i]
                continue
            prev_idx = next((j for j in range(i - 1, -1, -1) if np.isfinite(arr[j])), None)
            if prev_idx is None:
                continue
            prev2_idx = next(
                (j for j in range(prev_idx - 1, -1, -1) if np.isfinite(arr[j])), None
            )
            if prev2_idx is not None and prev_idx > prev2_idx:
                slope = (arr[prev_idx] - arr[prev2_idx]) / (prev_idx - prev2_idx)
                filled[i] = arr[prev_idx] + slope * (i - prev_idx)
            else:
                filled[i] = arr[prev_idx]
        out[col] = filled
    return out


def expanding_two_sample(
    x: pd.DataFrame,
    y: pd.DataFrame,
    fn,
    *,
    min_periods: int = 2,
) -> pd.DataFrame:
    """两序列各自取前缀非空样本（不要求逐行对齐）。

参数:
    x: 第一个输入宽表 panel。
    y: 第二个输入宽表 panel。
    fn: 接收 ``(xa, ya)`` 返回标量的函数。
    min_periods: 每序列最少有效样本数。

返回:
    扩展双样本统计结果 panel。
"""
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns, dtype=float)
    for col in x.columns:
        y_col = y[col] if col in y.columns else y.iloc[:, 0]
        xc = x[col].to_numpy(dtype=float, copy=False)
        yc = y_col.reindex(x.index).to_numpy(dtype=float, copy=False)
        for i in range(len(xc)):
            xa = xc[: i + 1]
            ya = yc[: i + 1]
            xa = xa[np.isfinite(xa)]
            ya = ya[np.isfinite(ya)]
            if xa.size < min_periods or ya.size < min_periods:
                continue
            try:
                out.iloc[i, out.columns.get_loc(col)] = float(fn(xa, ya))
            except Exception:
                pass
    return out
