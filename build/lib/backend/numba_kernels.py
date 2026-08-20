"""可选 Numba 加速的滚动核；未安装 numba 或禁用时不应导入失败。"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

_NUMBA_DISABLED = os.environ.get("FACTOR_ENGINE_DISABLE_NUMBA", "").lower() in (
    "1",
    "true",
    "yes",
)


def _get_move_mean() -> Callable[..., np.ndarray] | None:
    if _NUMBA_DISABLED:
        return None
    try:
        from numba import njit  # type: ignore
    except ImportError:
        return None

    @njit(cache=True)
    def _move_mean_1d(arr: np.ndarray, window: int, min_count: int) -> np.ndarray:
        n = arr.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            start = i - window + 1
            if start < 0:
                start = 0
            cnt = 0
            s = 0.0
            for j in range(start, i + 1):
                v = arr[j]
                # NaN AND ±Inf are missing in the pandas reference
                # ``rolling().mean()`` (pandas rolling aggregations skip
                # non-finite); the fast path must match it.
                if np.isfinite(v):
                    s += v
                    cnt += 1
            if cnt >= min_count:
                out[i] = s / cnt
            else:
                out[i] = np.nan
        return out

    return _move_mean_1d


def _get_move_rank_pct() -> Callable[..., np.ndarray] | None:
    if _NUMBA_DISABLED:
        return None
    try:
        from numba import njit  # type: ignore
    except ImportError:
        return None

    @njit(cache=True)
    def _move_rank_pct_1d(arr: np.ndarray, window: int, min_count: int) -> np.ndarray:
        n = arr.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            start = i - window + 1
            if start < 0:
                start = 0
            vals = arr[start : i + 1]
            cnt = 0
            last = arr[i]
            if not np.isfinite(last):
                out[i] = np.nan
                continue
            less = 0
            equal = 0
            for j in range(vals.shape[0]):
                v = vals[j]
                if not np.isfinite(v):
                    continue
                cnt += 1
                if v < last:
                    less += 1
                elif v == last:
                    equal += 1
            if cnt < min_count:
                out[i] = np.nan
            elif cnt <= 1:
                out[i] = 1.0
            else:
                rank_val = less + (equal + 1) / 2.0
                out[i] = rank_val / cnt
        return out

    return _move_rank_pct_1d


def _get_move_corr() -> Callable[..., np.ndarray] | None:
    if _NUMBA_DISABLED:
        return None
    try:
        from numba import njit  # type: ignore
    except ImportError:
        return None

    @njit(cache=True)
    def _move_corr_1d(x: np.ndarray, y: np.ndarray, window: int, min_count: int) -> np.ndarray:
        n = x.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            start = i - window + 1
            if start < 0:
                start = 0
            sx = sy = sxx = syy = sxy = 0.0
            cnt = 0
            for j in range(start, i + 1):
                xv = x[j]
                yv = y[j]
                # NaN AND ±Inf are missing in the pandas reference
                # ``rolling().corr()`` (pandas rolling aggregations skip
                # non-finite); the fast path must match it.
                if not (np.isfinite(xv) and np.isfinite(yv)):
                    continue
                cnt += 1
                sx += xv
                sy += yv
                sxx += xv * xv
                syy += yv * yv
                sxy += xv * yv
            if cnt < min_count:
                out[i] = np.nan
                continue
            num = cnt * sxy - sx * sy
            den_x = cnt * sxx - sx * sx
            den_y = cnt * syy - sy * sy
            if den_x <= 0.0 or den_y <= 0.0:
                out[i] = np.nan
            else:
                out[i] = num / np.sqrt(den_x * den_y)
        return out

    return _move_corr_1d


_move_mean_1d_jit = _get_move_mean()
_move_rank_pct_1d_jit = _get_move_rank_pct()
_move_corr_1d_jit = _get_move_corr()


def rolling_mean_1d(arr: np.ndarray, window: int, min_count: int) -> np.ndarray | None:
    """对 1D float 数组做与 ``rolling(...).mean()`` 同族语义的滑动均值；不可用则返回 ``None``。"""
    if _move_mean_1d_jit is None:
        return None
    a = np.asarray(arr, dtype=np.float64)
    return np.asarray(_move_mean_1d_jit(a, int(window), int(min_count)), dtype=float)


def rolling_rank_pct_1d(arr: np.ndarray, window: int, min_count: int = 1) -> np.ndarray | None:
    """滚动百分位 rank，对齐 ``rolling().rank(pct=True)`` 近似语义。"""
    if _move_rank_pct_1d_jit is None:
        return None
    a = np.asarray(arr, dtype=np.float64)
    return np.asarray(_move_rank_pct_1d_jit(a, int(window), int(min_count)), dtype=float)


def rolling_corr_1d(
    x: np.ndarray, y: np.ndarray, window: int, min_count: int = 3
) -> np.ndarray | None:
    if _move_corr_1d_jit is None:
        return None
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    return np.asarray(_move_corr_1d_jit(a, b, int(window), int(min_count)), dtype=float)


def rolling_corr_panel(
    x: np.ndarray, y: np.ndarray, window: int, min_count: int = 3
) -> np.ndarray | None:
    """2D panel (time, asset) 逐列 rolling corr。"""
    if _move_corr_1d_jit is None:
        return None
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape:
        return None
    out = np.empty_like(a, dtype=np.float64)
    for col in range(a.shape[1]):
        out[:, col] = _move_corr_1d_jit(a[:, col], b[:, col], int(window), int(min_count))
    return out


def rolling_rank_pct_panel(panel: np.ndarray, window: int, min_count: int = 1) -> np.ndarray | None:
    """2D panel (time, asset) 逐列 rolling rank pct。"""
    if _move_rank_pct_1d_jit is None:
        return None
    arr = np.asarray(panel, dtype=np.float64)
    out = np.empty_like(arr, dtype=np.float64)
    for col in range(arr.shape[1]):
        out[:, col] = _move_rank_pct_1d_jit(arr[:, col], int(window), int(min_count))
    return out


def _get_move_std() -> Callable[..., np.ndarray] | None:
    if _NUMBA_DISABLED:
        return None
    try:
        from numba import njit  # type: ignore
    except ImportError:
        return None

    @njit(cache=True)
    def _move_std_1d(arr: np.ndarray, window: int, min_count: int) -> np.ndarray:
        n = arr.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            start = i - window + 1
            if start < 0:
                start = 0
            s = 0.0
            s2 = 0.0
            cnt = 0
            for j in range(start, i + 1):
                v = arr[j]
                # NaN AND ±Inf are missing in the pandas reference
                # ``rolling().std()`` (pandas rolling aggregations skip
                # non-finite); the fast path must match it.
                if np.isfinite(v):
                    s += v
                    s2 += v * v
                    cnt += 1
            if cnt < min_count:
                out[i] = np.nan
            elif cnt == 1:
                out[i] = np.nan
            else:
                mean = s / cnt
                var = (s2 - cnt * mean * mean) / (cnt - 1)
                out[i] = np.sqrt(max(var, 0.0))
        return out

    return _move_std_1d


_move_std_1d_jit = _get_move_std()


def rolling_mean_panel(
    panel: np.ndarray, window: int, min_count: int = 1
) -> np.ndarray | None:
    """2D panel (time, asset) 逐列 rolling mean。"""
    if _move_mean_1d_jit is None:
        return None
    arr = np.asarray(panel, dtype=np.float64)
    out = np.empty_like(arr, dtype=np.float64)
    for col in range(arr.shape[1]):
        out[:, col] = _move_mean_1d_jit(arr[:, col], int(window), int(min_count))
    return out


def rolling_std_panel(
    panel: np.ndarray, window: int, min_count: int = 1
) -> np.ndarray | None:
    """2D panel (time, asset) 逐列 rolling std（ddof=1，对齐 pandas）。"""
    if _move_std_1d_jit is None:
        return None
    arr = np.asarray(panel, dtype=np.float64)
    out = np.empty_like(arr, dtype=np.float64)
    for col in range(arr.shape[1]):
        out[:, col] = _move_std_1d_jit(arr[:, col], int(window), int(min_count))
    return out
