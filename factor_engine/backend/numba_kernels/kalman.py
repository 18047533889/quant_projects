# -*- coding: utf-8 -*-
"""R35 Phase F: Numba Kalman kernels.

Taskbook §10 / §38: Kalman is the most Numba-appropriate model family — pure
numeric recurrence, fixed dtype, no SciPy optimizer, recursive state.  First
batch: scalar local-level, local-trend slope, and dynamic-beta filters.

Fastmath is OFF (strict NaN/Inf semantics).  ``nogil=True`` lets the FE
scheduler parallelise outer factor/block dims without oversubscription.
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
# Reference kernels (canonical NumPy implementations)
# --------------------------------------------------------------------------

def kalman_level_reference(vals: np.ndarray, q: float, r: float) -> np.ndarray:
    """Canonical local-level Kalman filter (matches ts_model.state_space)."""
    n = len(vals)
    mu = np.full(n, np.nan, dtype=float)
    mu_prev = np.nan
    p_prev = 1.0
    for t in range(n):
        x = vals[t]
        if not np.isfinite(x):
            if np.isfinite(mu_prev):
                mu[t] = mu_prev
                p_prev = p_prev + q
            continue
        if not np.isfinite(mu_prev):
            mu_prev = x
            p_prev = r
            mu[t] = x
            continue
        p_pred = p_prev + q
        k = p_pred / (p_pred + r)
        mu_prev = mu_prev + k * (x - mu_prev)
        p_prev = (1.0 - k) * p_pred
        mu[t] = mu_prev
    return mu


def kalman_trend_reference(vals: np.ndarray, q_level: float, q_trend: float, r: float) -> np.ndarray:
    """Local linear trend slope (level + slope Kalman)."""
    n = len(vals)
    slope = np.full(n, np.nan, dtype=float)
    level = np.nan
    sl = 0.0
    P = np.array([[1.0, 0.0], [0.0, 1.0]])
    Q = np.array([[q_level, 0.0], [0.0, q_trend]])
    for t in range(n):
        x = vals[t]
        if not np.isfinite(x):
            if np.isfinite(level):
                P = P + Q
            continue
        if not np.isfinite(level):
            level = x
            slope[t] = 0.0
            continue
        # predict
        s_prev, l_prev = sl, level
        P = P + Q
        # update with observation variance R
        H = np.array([1.0, 0.0])
        S = float(P[0, 0] + r)
        K = P[:, 0] / S
        innov = x - l_prev
        level = l_prev + K[0] * innov
        sl = s_prev + K[1] * innov
        P = P - np.outer(K, H) @ P
        slope[t] = sl
    return slope


def kalman_beta_reference(y: np.ndarray, x: np.ndarray, q: float, r: float) -> np.ndarray:
    """Dynamic beta (state = beta; y = beta*x + noise)."""
    n = len(y)
    beta = np.full(n, np.nan, dtype=float)
    b = np.nan
    P = 1.0
    for t in range(n):
        if not (np.isfinite(y[t]) and np.isfinite(x[t])):
            if np.isfinite(b):
                P = P + q
            continue
        if not np.isfinite(b):
            b = 0.0
            beta[t] = b
            P = r
            continue
        P = P + q
        H = x[t]
        S = float(H * P * H + r)
        K = P * H / S
        b = b + K * (y[t] - H * b)
        P = (1.0 - K * H) * P
        beta[t] = b
    return beta


# --------------------------------------------------------------------------
# Numba kernels (compiled, fastmath=False, nogil=True)
# --------------------------------------------------------------------------

if _HAS_NUMBA:

    @njit(cache=True, nogil=True, fastmath=False)
    def _kalman_level_numba(vals: np.ndarray, q: float, r: float) -> np.ndarray:
        n = vals.shape[0]
        mu = np.full(n, np.nan)
        mu_prev = np.nan
        p_prev = 1.0
        for t in range(n):
            x = vals[t]
            if not np.isfinite(x):
                if np.isfinite(mu_prev):
                    mu[t] = mu_prev
                    p_prev = p_prev + q
                continue
            if not np.isfinite(mu_prev):
                mu_prev = x
                p_prev = r
                mu[t] = x
                continue
            p_pred = p_prev + q
            k = p_pred / (p_pred + r)
            mu_prev = mu_prev + k * (x - mu_prev)
            p_prev = (1.0 - k) * p_pred
            mu[t] = mu_prev
        return mu

    @njit(cache=True, nogil=True, fastmath=False)
    def _kalman_trend_numba(vals: np.ndarray, q_level: float, q_trend: float, r: float) -> np.ndarray:
        n = vals.shape[0]
        slope = np.full(n, np.nan)
        level = np.nan
        sl = 0.0
        # P flat 2x2
        p00 = 1.0
        p01 = 0.0
        p10 = 0.0
        p11 = 1.0
        for t in range(n):
            x = vals[t]
            if not np.isfinite(x):
                if np.isfinite(level):
                    p00 += q_level
                    p11 += q_trend
                continue
            if not np.isfinite(level):
                level = x
                slope[t] = 0.0
                continue
            # predict
            p00 += q_level
            p11 += q_trend
            # update (H = [1, 0]).  K = P[:,0]/S where S = P[0,0]+r.
            # P_new = P - K H P  with H=[1,0] => P_new[i,j] = P[i,j] - K[i]*P[0,j]
            s = p00 + r
            k0 = p00 / s
            k1 = p10 / s
            innov = x - level
            level = level + k0 * innov
            sl = sl + k1 * innov
            p00_old = p00
            p01_old = p01
            p10_old = p10
            p11_old = p11
            p00 = p00_old - k0 * p00_old
            p01 = p01_old - k0 * p01_old
            p10 = p10_old - k1 * p00_old
            p11 = p11_old - k1 * p01_old
            slope[t] = sl
        return slope

    @njit(cache=True, nogil=True, fastmath=False)
    def _kalman_beta_numba(y: np.ndarray, x: np.ndarray, q: float, r: float) -> np.ndarray:
        n = y.shape[0]
        beta = np.full(n, np.nan)
        b = np.nan
        P = 1.0
        for t in range(n):
            if not (np.isfinite(y[t]) and np.isfinite(x[t])):
                if np.isfinite(b):
                    P = P + q
                continue
            if not np.isfinite(b):
                b = 0.0
                P = r
                beta[t] = b
                continue
            P = P + q
            H = x[t]
            s = H * P * H + r
            K = P * H / s
            b = b + K * (y[t] - H * b)
            P = (1.0 - K * H) * P
            beta[t] = b
        return beta

    _KALMAN_LEVEL_NUMBA = _kalman_level_numba
    _KALMAN_TREND_NUMBA = _kalman_trend_numba
    _KALMAN_BETA_NUMBA = _kalman_beta_numba
else:  # pragma: no cover
    _KALMAN_LEVEL_NUMBA = None
    _KALMAN_TREND_NUMBA = None
    _KALMAN_BETA_NUMBA = None


def _register() -> None:
    NumbaKernelRegistry.register(
        "kalman", "kalman_level", kalman_level_reference, _KALMAN_LEVEL_NUMBA,
        semantic_version="1.0", supported_dtypes=("float64",),
        supported_param_domain="q>=0, r>0, finite", nogil=True, parallel=False,
    )
    NumbaKernelRegistry.register(
        "kalman", "kalman_trend", kalman_trend_reference, _KALMAN_TREND_NUMBA,
        semantic_version="1.0", supported_dtypes=("float64",),
        supported_param_domain="q_level>=0, q_trend>=0, r>0", nogil=True, parallel=False,
    )
    NumbaKernelRegistry.register(
        "kalman", "kalman_beta", kalman_beta_reference, _KALMAN_BETA_NUMBA,
        semantic_version="1.0", supported_dtypes=("float64",),
        supported_param_domain="q>=0, r>0", nogil=True, parallel=False,
    )


_register()
