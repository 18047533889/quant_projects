"""因果性辅助：滚动/扩展计算，避免使用未来样本。"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SPECTRAL_WINDOW = 20
DEFAULT_WAVELET_WINDOW = 20


def causal_lag(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """只允许非负滞后；负值视为未来位移，返回 NaN。"""
    lag = int(n)
    if lag < 0:
        return pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    if lag == 0:
        return x
    return x.shift(lag)


def causal_bfill(x: pd.DataFrame) -> pd.DataFrame:
    """后向填充依赖未来值，因子链路中禁止。"""
    return x


def rolling_rfft_feature(
    arr: np.ndarray,
    window: int,
    *,
    feature: str = "dominant",
) -> np.ndarray:
    """仅用截至当前时点的窗口做 RFFT，输出标量特征。"""
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
    arr = x.to_numpy(dtype=np.float64, copy=False)
    out = np.empty_like(arr, dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = rolling_rfft_feature(arr[:, j], window, feature=feature)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def expanding_svd_diagonal(x: pd.DataFrame) -> pd.DataFrame:
    """扩展窗口 SVD：每个时点仅使用历史行。"""
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
    """扩展窗口协方差特征值：每个时点仅使用历史行。"""
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
    """滚动小波去噪：窗口内分解，仅取当前时点重构值。"""
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
    arr = x.to_numpy(dtype=np.float64, copy=False)
    out = np.empty_like(arr, dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = rolling_wavelet_column(
            arr[:, j], window, threshold=threshold, wavelet=wavelet_type
        )
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def rolling_irfft_last(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动 RFFT→IRFFT，仅输出窗口末样本（因果）。"""
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
    arr = x.to_numpy(dtype=np.float64, copy=False)
    out = np.empty_like(arr, dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = rolling_irfft_last(arr[:, j], window)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def causal_convolve_column(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """因果 FIR 卷积：y[n] = sum_k b[k] * x[n-k]。"""
    from scipy.signal import lfilter

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not np.isfinite(b).any():
        return np.full(len(a), np.nan, dtype=np.float64)
    kernel = np.where(np.isfinite(b), b, 0.0)
    x = np.where(np.isfinite(a), a, 0.0)
    return lfilter(kernel, [1.0], x)


def causal_correlate_column(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """因果互相关：每个时点仅与截至当前的样本重叠。"""
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
    """扩展窗口相位展开：每个时点仅 unwrap 历史前缀。"""
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
    log_abs = np.log(x.abs().replace(0, np.nan))
    return np.exp(log_abs.expanding(min_periods=1).mean())


def expanding_harmonic_mean(x: pd.DataFrame) -> pd.DataFrame:
    inv = 1.0 / x.replace(0, np.nan)
    count = x.expanding(min_periods=1).count().astype(float)
    inv_sum = inv.expanding(min_periods=1).sum()
    return count / inv_sum


def expanding_panel_stat(x: pd.DataFrame, stat: str, **kwargs) -> pd.DataFrame:
    """按列扩展聚合，避免全样本广播引入未来值。"""
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
    """扩展窗口内极值位置（iloc 下标）。"""
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
    """对每列前缀调用 fn(valid_array) -> scalar。"""
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
    """对每列对齐前缀调用 fn(x_valid, y_valid) -> scalar。"""
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


def causal_interpolate_panel(x: pd.DataFrame, method: str = "linear") -> pd.DataFrame:
    """仅用历史已知点做前向填充/外推，不引用未来样本。"""
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
            if method == "linear":
                prev2_idx = next(
                    (j for j in range(prev_idx - 1, -1, -1) if np.isfinite(arr[j])), None
                )
                if prev2_idx is not None and prev_idx > prev2_idx:
                    slope = (arr[prev_idx] - arr[prev2_idx]) / (prev_idx - prev2_idx)
                    filled[i] = arr[prev_idx] + slope * (i - prev_idx)
                else:
                    filled[i] = arr[prev_idx]
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
    """两序列各自取前缀非空样本（不要求逐行对齐）。"""
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
