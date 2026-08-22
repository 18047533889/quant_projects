# -*- coding: utf-8 -*-
"""R21 §32 P10: Numba CPU kernel implementations for five core rolling operators.

Registers ts_mean, ts_std, ts_sum, ts_min, ts_max with the
:class:`~backend.numba_kernel_registry.NumbaKernelRegistry`.

Each kernel has:
- A canonical NumPy reference implementation (skip non-finite, matching pandas semantics).
- A compiled Numba fastmath=False/nogil=True implementation (when numba is available).
- Strict NaN/Inf handling: non-finite values are skipped in aggregations.
- fastmath=False: strict NaN/Inf/associativity semantics (R35 §15).

Pandas reference semantics:
- rolling().min_periods=1, non-finite values (NaN, +Inf, -Inf) are treated as missing.
- For std: ddof=1 (sample std), single-element or all-missing windows return NaN.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from backend.numba_kernel_registry import (  # noqa: E402
    NUMBA_AVAILABLE,
    NumbaKernelRegistry,
)

if NUMBA_AVAILABLE:
    from numba import njit  # noqa: E402

    _HAS_NUMBA = True
else:  # pragma: no cover
    _HAS_NUMBA = False


# --------------------------------------------------------------------------
# Reference kernels (canonical NumPy implementations)
# --------------------------------------------------------------------------

def _rolling_sum_reference(arr: np.ndarray, window: int, min_count: int) -> np.ndarray:
    """Rolling sum, non-finite values skipped, matching pandas rolling semantics."""
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        s = 0.0
        cnt = 0
        for j in range(start, i + 1):
            v = arr[j]
            if np.isfinite(v):
                s += v
                cnt += 1
        if cnt >= min_count:
            out[i] = s
    return out


def _rolling_mean_reference(arr: np.ndarray, window: int, min_count: int) -> np.ndarray:
    """Rolling mean, non-finite values skipped, matching pandas rolling semantics."""
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        s = 0.0
        cnt = 0
        for j in range(start, i + 1):
            v = arr[j]
            if np.isfinite(v):
                s += v
                cnt += 1
        if cnt >= min_count:
            out[i] = s / cnt
    return out


def _rolling_min_reference(arr: np.ndarray, window: int, min_count: int) -> np.ndarray:
    """Rolling min, non-finite values skipped, matching pandas rolling semantics."""
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        cur_min = np.inf
        cnt = 0
        for j in range(start, i + 1):
            v = arr[j]
            if np.isfinite(v):
                cnt += 1
                if v < cur_min:
                    cur_min = v
        if cnt >= min_count:
            out[i] = cur_min
    return out


def _rolling_max_reference(arr: np.ndarray, window: int, min_count: int) -> np.ndarray:
    """Rolling max, non-finite values skipped, matching pandas rolling semantics."""
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        cur_max = -np.inf
        cnt = 0
        for j in range(start, i + 1):
            v = arr[j]
            if np.isfinite(v):
                cnt += 1
                if v > cur_max:
                    cur_max = v
        if cnt >= min_count:
            out[i] = cur_max
    return out


def _rolling_std_reference(arr: np.ndarray, window: int, min_count: int) -> np.ndarray:
    """Rolling std (ddof=1), non-finite values skipped, matching pandas rolling semantics."""
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window + 1)
        s = 0.0
        s2 = 0.0
        cnt = 0
        for j in range(start, i + 1):
            v = arr[j]
            if np.isfinite(v):
                s += v
                s2 += v * v
                cnt += 1
        if cnt < min_count:
            pass  # already NaN
        elif cnt == 1:
            pass  # single element: NaN (ddof=1)
        else:
            mean = s / cnt
            var = (s2 - cnt * mean * mean) / (cnt - 1)
            out[i] = np.sqrt(max(var, 0.0))
    return out


# --------------------------------------------------------------------------
# Numba kernels (compiled, fastmath=False, nogil=True)
# --------------------------------------------------------------------------

if _HAS_NUMBA:

    @njit(cache=True, nogil=True, fastmath=False)
    def _rolling_sum_numba(arr: np.ndarray, window: int, min_count: int) -> np.ndarray:
        n = arr.shape[0]
        out = np.full(n, np.nan)
        for i in range(n):
            start = i - window + 1
            if start < 0:
                start = 0
            s = 0.0
            cnt = 0
            for j in range(start, i + 1):
                v = arr[j]
                if np.isfinite(v):
                    s += v
                    cnt += 1
            if cnt >= min_count:
                out[i] = s
        return out

    @njit(cache=True, nogil=True, fastmath=False)
    def _rolling_mean_numba(arr: np.ndarray, window: int, min_count: int) -> np.ndarray:
        n = arr.shape[0]
        out = np.full(n, np.nan)
        for i in range(n):
            start = i - window + 1
            if start < 0:
                start = 0
            s = 0.0
            cnt = 0
            for j in range(start, i + 1):
                v = arr[j]
                if np.isfinite(v):
                    s += v
                    cnt += 1
            if cnt >= min_count:
                out[i] = s / cnt
        return out

    @njit(cache=True, nogil=True, fastmath=False)
    def _rolling_min_numba(arr: np.ndarray, window: int, min_count: int) -> np.ndarray:
        n = arr.shape[0]
        out = np.full(n, np.nan)
        for i in range(n):
            start = i - window + 1
            if start < 0:
                start = 0
            cur_min = np.inf
            cnt = 0
            for j in range(start, i + 1):
                v = arr[j]
                if np.isfinite(v):
                    cnt += 1
                    if v < cur_min:
                        cur_min = v
            if cnt >= min_count:
                out[i] = cur_min
        return out

    @njit(cache=True, nogil=True, fastmath=False)
    def _rolling_max_numba(arr: np.ndarray, window: int, min_count: int) -> np.ndarray:
        n = arr.shape[0]
        out = np.full(n, np.nan)
        for i in range(n):
            start = i - window + 1
            if start < 0:
                start = 0
            cur_max = -np.inf
            cnt = 0
            for j in range(start, i + 1):
                v = arr[j]
                if np.isfinite(v):
                    cnt += 1
                    if v > cur_max:
                        cur_max = v
            if cnt >= min_count:
                out[i] = cur_max
        return out

    @njit(cache=True, nogil=True, fastmath=False)
    def _rolling_std_numba(arr: np.ndarray, window: int, min_count: int) -> np.ndarray:
        n = arr.shape[0]
        out = np.full(n, np.nan)
        for i in range(n):
            start = i - window + 1
            if start < 0:
                start = 0
            s = 0.0
            s2 = 0.0
            cnt = 0
            for j in range(start, i + 1):
                v = arr[j]
                if np.isfinite(v):
                    s += v
                    s2 += v * v
                    cnt += 1
            if cnt < min_count:
                pass  # NaN
            elif cnt == 1:
                pass  # NaN for ddof=1
            else:
                mean = s / cnt
                var = (s2 - cnt * mean * mean) / (cnt - 1)
                out[i] = np.sqrt(max(var, 0.0))
        return out

    _TS_SUM_NUMBA = _rolling_sum_numba
    _TS_MEAN_NUMBA = _rolling_mean_numba
    _TS_MIN_NUMBA = _rolling_min_numba
    _TS_MAX_NUMBA = _rolling_max_numba
    _TS_STD_NUMBA = _rolling_std_numba
else:  # pragma: no cover
    _TS_SUM_NUMBA = None
    _TS_MEAN_NUMBA = None
    _TS_MIN_NUMBA = None
    _TS_MAX_NUMBA = None
    _TS_STD_NUMBA = None


def _register() -> None:
    NumbaKernelRegistry.register(
        "ts_rolling", "ts_sum", _rolling_sum_reference, _TS_SUM_NUMBA,
        semantic_version="1.0", supported_dtypes=("float64",),
        supported_param_domain="window>=1, min_count>=1", nogil=True, parallel=False,
    )
    NumbaKernelRegistry.register(
        "ts_rolling", "ts_mean", _rolling_mean_reference, _TS_MEAN_NUMBA,
        semantic_version="1.0", supported_dtypes=("float64",),
        supported_param_domain="window>=1, min_count>=1", nogil=True, parallel=False,
    )
    NumbaKernelRegistry.register(
        "ts_rolling", "ts_min", _rolling_min_reference, _TS_MIN_NUMBA,
        semantic_version="1.0", supported_dtypes=("float64",),
        supported_param_domain="window>=1, min_count>=1", nogil=True, parallel=False,
    )
    NumbaKernelRegistry.register(
        "ts_rolling", "ts_max", _rolling_max_reference, _TS_MAX_NUMBA,
        semantic_version="1.0", supported_dtypes=("float64",),
        supported_param_domain="window>=1, min_count>=1", nogil=True, parallel=False,
    )
    NumbaKernelRegistry.register(
        "ts_rolling", "ts_std", _rolling_std_reference, _TS_STD_NUMBA,
        semantic_version="1.0", supported_dtypes=("float64",),
        supported_param_domain="window>=1, min_count>=1, ddof=1", nogil=True, parallel=False,
    )


_register()
