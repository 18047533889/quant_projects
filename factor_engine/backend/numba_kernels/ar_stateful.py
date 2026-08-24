# -*- coding: utf-8 -*-
"""R35 Phase F: Numba AR + stateful recurrence kernels.

Taskbook §11 / §111: recursive-state kernels (``state_t = f(state_{t-1}, x_t)``)
are ideal for Numba.  First batch here: an AR(order) prior-forecast kernel
(rolling Yule-Walker / least-squares fit on the trailing window, fit strictly
before the scored row) — the fast path behind ``ts_ar_prior_forecast``.

fastmath=False, nogil=True, cache=True.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from factor_engine.backend.numba_kernel_registry import (  # noqa: E402
    NUMBA_AVAILABLE,
    NumbaKernelRegistry,
)

if NUMBA_AVAILABLE:
    from numba import njit  # noqa: E402

    _HAS_NUMBA = True
else:  # pragma: no cover
    _HAS_NUMBA = False


# --------------------------------------------------------------------------
# Reference: AR(order) prior forecast via Yule-Walker on the trailing window.
# --------------------------------------------------------------------------

def _ar_prior_forecast_series(vals: np.ndarray, window: int, order: int) -> np.ndarray:
    """Rolling AR(order) one-step-ahead forecast fit on [row-w, row-1]."""
    n = len(vals)
    out = np.full(n, np.nan, dtype=float)
    o = max(1, int(order))
    w = max(o + 2, int(window))
    for row in range(n):
        fit_end = row - 1
        if fit_end < o:
            continue
        start = max(0, fit_end - w + 1)
        seg = vals[start : fit_end + 1]
        beta = _ar_yw(seg, o)
        if beta is None:
            continue
        if row < o:
            continue
        xv = vals[row - o : row][::-1]
        if not np.all(np.isfinite(xv)):
            continue
        out[row] = float(beta @ xv)
    return out


def _ar_yw(seg: np.ndarray, order: int) -> np.ndarray | None:
    """Yule-Walker AR(order) coefficients (no intercept)."""
    y = seg[np.isfinite(seg)]
    if len(y) < order * 3 + 1:
        return None
    yc = y - np.mean(y)
    acf = np.empty(order + 1)
    yy = float(np.dot(yc, yc))
    if yy <= 0.0:
        return None
    for k in range(order + 1):
        acf[k] = np.dot(yc[: len(yc) - k], yc[k:]) / yy
    # Toeplitz solve
    R = np.empty((order, order))
    for i in range(order):
        for j in range(order):
            R[i, j] = acf[abs(i - j)]
    r = acf[1:]
    try:
        a = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(a)):
        return None
    return a


# --------------------------------------------------------------------------
# Numba: same algorithm, compiled.
# --------------------------------------------------------------------------

if _HAS_NUMBA:

    @njit(cache=True, nogil=True, fastmath=False)
    def _ar_prior_forecast_numba(vals: np.ndarray, window: int, order: int) -> np.ndarray:
        n = vals.shape[0]
        out = np.full(n, np.nan)
        o = max(1, order)
        w = max(o + 2, window)
        for row in range(n):
            fit_end = row - 1
            if fit_end < o:
                continue
            start = fit_end - w + 1
            if start < 0:
                start = 0
            seg = vals[start : fit_end + 1]
            # finite subset
            m = seg.shape[0]
            yf = np.empty(m)
            cnt = 0
            for i in range(m):
                if np.isfinite(seg[i]):
                    yf[cnt] = seg[i]
                    cnt += 1
            if cnt < o * 3 + 1:
                continue
            y = yf[:cnt]
            mean = 0.0
            for i in range(cnt):
                mean += y[i]
            mean /= cnt
            yc = np.empty(cnt)
            for i in range(cnt):
                yc[i] = y[i] - mean
            yy = 0.0
            for i in range(cnt):
                yy += yc[i] * yc[i]
            if yy <= 0.0:
                continue
            acf = np.empty(o + 1)
            for k in range(o + 1):
                s = 0.0
                for i in range(cnt - k):
                    s += yc[i] * yc[i + k]
                acf[k] = s / yy
            # Toeplitz solve via numpy in numba is not available; use simple
            # Gaussian elimination on the small system.
            R = np.empty((o, o))
            for i in range(o):
                for j in range(o):
                    R[i, j] = acf[abs(i - j)]
            rv = acf[1:]
            a = _solve_small(R, rv, o)
            if a is None:
                continue
            if row < o:
                continue
            ok = True
            pred = 0.0
            for k in range(o):
                xv = vals[row - 1 - k]
                if not np.isfinite(xv):
                    ok = False
                    break
                pred += a[k] * xv
            if ok:
                out[row] = pred
        return out

    @njit(cache=True, nogil=True, fastmath=False)
    def _solve_small(A, b, n):
        """Gaussian elimination with partial pivoting for a small n x n system."""
        M = A.copy()
        v = b.copy()
        for col in range(n):
            # pivot
            piv = col
            best = abs(M[col, col])
            for r in range(col + 1, n):
                if abs(M[r, col]) > best:
                    best = abs(M[r, col])
                    piv = r
            if best <= 1e-15:
                return None
            if piv != col:
                for j in range(n):
                    tmp = M[col, j]
                    M[col, j] = M[piv, j]
                    M[piv, j] = tmp
                t = v[col]
                v[col] = v[piv]
                v[piv] = t
            pivv = M[col, col]
            for r in range(col + 1, n):
                f = M[r, col] / pivv
                if f == 0.0:
                    continue
                for j in range(col, n):
                    M[r, j] -= f * M[col, j]
                v[r] -= f * v[col]
        x = np.empty(n)
        for i in range(n - 1, -1, -1):
            s = v[i]
            for j in range(i + 1, n):
                s -= M[i, j] * x[j]
            x[i] = s / M[i, i]
        return x

    _AR_PRIOR_NUMBA = _ar_prior_forecast_numba
else:  # pragma: no cover
    _AR_PRIOR_NUMBA = None


def _register() -> None:
    NumbaKernelRegistry.register(
        "ar", "ar_prior_forecast", _ar_prior_forecast_series, _AR_PRIOR_NUMBA,
        semantic_version="1.0", supported_dtypes=("float64",),
        supported_param_domain="window>=order+2, order>=1", nogil=True, parallel=False,
    )


_register()
