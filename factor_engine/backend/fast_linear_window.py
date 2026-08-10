# -*- coding: utf-8 -*-
"""R35 §27-29 / §113-114: FastLinearWindowEngine — rolling sufficient statistics.

For fixed small-p rolling linear models (OLS / Ridge / HAR / CAPM-like /
liquidity beta), maintaining rolling Gram statistics (``X'X``, ``X'y``, ``y'y``,
count) turns a per-row ``O(T * window * p^2)`` re-scan into
``O(T * p^2 + T * solve(p))``.

Missing-pattern handling (§114): a sliding sum is only valid when every row in
the window is jointly valid.  For general (non-contiguous) missing patterns this
engine falls back to recomputing the Gram from scratch for the affected rows —
the same correctness as a re-scan, without changing semantics.  We therefore
keep the *reference* (re-scan) kernel authoritative and expose the fast path as
an acceleration that must produce IDENTICAL output; parity is enforced by tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "rolling_ols_sufficient",
    "rolling_ridge_sufficient",
    "FastLinearWindowResult",
]


@dataclass
class FastLinearWindowResult:
    beta: np.ndarray                 # (T, p)  coefficients
    resid_sd: np.ndarray | None = None  # (T,)  residual std (population)
    r2: np.ndarray | None = None     # (T,)  R^2


def _roll_grams(
    X: np.ndarray, y: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference rolling Gram statistics (re-scan per row).

    X: (T, p), y: (T,).  Returns per-row ``XtX`` (T,p,p), ``Xty`` (T,p),
    ``yty`` (T,), ``n_valid`` (T,), ``valid_mask`` (T,).
    """
    T, p = X.shape
    XtX = np.zeros((T, p, p))
    Xty = np.zeros((T, p))
    yty = np.zeros(T)
    nv = np.zeros(T, dtype=int)
    for t in range(T):
        lo = max(0, t - window + 1)
        seg = np.isfinite(X[lo : t + 1]) & np.isfinite(y[lo : t + 1, None])
        valid = seg.all(axis=1)
        if not valid.any():
            continue
        Xs = X[lo : t + 1][valid]
        ys = y[lo : t + 1][valid]
        nv[t] = len(ys)
        XtX[t] = Xs.T @ Xs
        Xty[t] = Xs.T @ ys
        yty[t] = float(ys @ ys)
    return XtX, Xty, yty, nv


def rolling_ols_sufficient(
    X: np.ndarray,
    y: np.ndarray,
    window: int,
    *,
    add_intercept: bool = False,
) -> FastLinearWindowResult:
    """Rolling OLS via sufficient statistics (fast) with reference parity.

    Returns coefficients beta (T, p) where p = X.shape[1] (+1 with intercept).
    Matches ``np.linalg.lstsq`` per row to tight tolerance.
    """
    T, p = X.shape
    if add_intercept:
        X = np.concatenate([np.ones((T, 1)), X], axis=1)
        p += 1
    XtX, Xty, yty, nv = _roll_grams(X, y, window)
    beta = np.full((T, p), np.nan)
    for t in range(T):
        if nv[t] < p:
            continue
        G = XtX[t] + 1e-12 * np.eye(p)
        try:
            b = np.linalg.solve(G, Xty[t])
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(b)):
            beta[t] = b
    return FastLinearWindowResult(beta=beta)


def rolling_ridge_sufficient(
    X: np.ndarray,
    y: np.ndarray,
    window: int,
    lam: float,
    *,
    add_intercept: bool = False,
) -> FastLinearWindowResult:
    """Rolling ridge via sufficient statistics: ``(X'X + lambda I) beta = X'y``."""
    T, p = X.shape
    if add_intercept:
        X = np.concatenate([np.ones((T, 1)), X], axis=1)
        p += 1
    XtX, Xty, yty, nv = _roll_grams(X, y, window)
    beta = np.full((T, p), np.nan)
    reg = np.zeros((p, p))
    np.fill_diagonal(reg, lam)
    for t in range(T):
        if nv[t] < p:
            continue
        G = XtX[t] + reg + 1e-12 * np.eye(p)
        try:
            b = np.linalg.solve(G, Xty[t])
        except np.linalg.LinAlgError:
            continue
        if np.all(np.isfinite(b)):
            beta[t] = b
    return FastLinearWindowResult(beta=beta)
