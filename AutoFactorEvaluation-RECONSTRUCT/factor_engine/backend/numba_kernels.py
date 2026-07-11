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
                if np.isfinite(v):
                    s += v
                    cnt += 1
            if cnt >= min_count:
                out[i] = s / cnt
            else:
                out[i] = np.nan
        return out

    return _move_mean_1d


_move_mean_1d_jit = _get_move_mean()


def rolling_mean_1d(
    arr: np.ndarray, window: int, min_count: int
) -> np.ndarray | None:
    """对 1D float 数组做与 ``rolling(...).mean()`` 同族语义的滑动均值；不可用则返回 ``None``。"""
    if _move_mean_1d_jit is None:
        return None
    a = np.asarray(arr, dtype=np.float64)
    return np.asarray(_move_mean_1d_jit(a, int(window), int(min_count)), dtype=float)
