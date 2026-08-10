# -*- coding: utf-8 -*-
"""R35 Phase B: independent model oracles (numpy/scipy only — no sklearn/statsmodels).

Each oracle re-derives a model statistic from first principles so the engine
kernel is compared against a genuinely independent implementation, not a
restatement of the same code path.

Oracles provided:
- :func:`pca_oracle` — standardized SVD PCA (active-space, mean-imputed), matching
  ``panel_model._pca_svd`` semantics but independently written.
- :func:`rolling_ols_oracle` — per-row rolling OLS slope via centered dot
  products (population ddof=0), the reference for ``_rolling_regression`` /
  ``_pairwise_rolling``.
- :func:`ar_oracle` — AR(order) coefficients via Yule-Walker (scipy.linalg.
  solve_toeplitz) on the trailing window, independent of the engine's fit.
- :func:`kalman_level_oracle` — scalar local-level Kalman filter by hand.
- :func:`garch_ml_oracle` — GARCH(1,1) variance-targeting MLE via scipy
  minimize (independent objective), for recovery / persistence checks.
- :func:`har_oracle` — HAR-RV OLS next-period forecast by explicit design-matrix
  construction.
"""
from __future__ import annotations

import numpy as np

from scipy import linalg
from scipy.optimize import minimize


def pca_oracle(X: np.ndarray, n_components: int, *, min_obs: int = 2) -> dict | None:
    """Standardised SVD PCA with per-column mean/std on finite rows, active
    sub-space only, imputation by column mean — matches ``_pca_svd``."""
    n, d = X.shape
    finite_count = np.isfinite(X).sum(axis=0)
    active = finite_count >= min_obs
    n_active = int(active.sum())
    if n_active < 2:
        return None
    sub = X[:, active]
    mu = np.nanmean(sub, axis=0)
    sd = np.nanstd(sub, axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Xc = np.where(np.isfinite(sub), sub, mu)
    Xs = (Xc - mu) / sd
    k = int(min(n_components, n_active - 1, Xs.shape[0] - 1))
    if k < 1:
        return None
    _, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    total = float(np.sum(s * s))
    return {
        "active": active,
        "k": k,
        "mu": mu,
        "sd": sd,
        "loadings": Vt[:k],
        "explained": s[:k] ** 2 / max(total, 1e-12),
    }


def rolling_ols_slope(y: np.ndarray, x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Per-row rolling slope via centered dot products (population ddof=0)."""
    n = len(y)
    out = np.full(n, np.nan)
    for row in range(n):
        start = max(0, row - window + 1)
        a, b = y[start : row + 1], x[start : row + 1]
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < min_periods:
            continue
        a, b = a[valid], b[valid]
        xbar = float(np.mean(b))
        denom = float(np.sum((b - xbar) ** 2))
        if denom <= 0.0:
            continue
        out[row] = float(np.sum((a - np.mean(a)) * (b - xbar)) / denom)
    return out


def ar_yule_walker(vals: np.ndarray, order: int) -> np.ndarray | None:
    """AR(order) coefficients by Yule-Walker on the trailing finite run."""
    finite = vals[np.isfinite(vals)]
    if len(finite) < order * 3 + 1:
        return None
    y = finite - np.mean(finite)
    acf = [np.dot(y[: len(y) - k], y[k:]) / max(float(np.dot(y, y)), 1e-12) for k in range(order + 1)]
    # acf[0]==1; solve R * a = r  (Toeplitz)
    R = linalg.toeplitz(acf[:order])
    r = np.asarray(acf[1 : order + 1])
    try:
        a = linalg.solve(R, r)
    except linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(a)):
        return None
    return a


def kalman_local_level(x: np.ndarray, q: float, r: float) -> np.ndarray:
    """Scalar local-level Kalman filter (one-pass, causal), matching the
    level-state semantics: x_t = mu_t + eps_t, mu_t = mu_{t-1} + w_t.
    Returns the filtered level for each row (predict + update)."""
    n = len(x)
    mu = np.full(n, np.nan)
    m = float(np.nanmean(x)) if np.isfinite(x).any() else 0.0
    P = float(np.var(x)) + r if np.isfinite(x).any() else 1.0
    for t in range(n):
        # predict
        m_p = m
        P_p = P + q
        if not np.isfinite(x[t]):
            m, P = m_p, P_p
            continue
        # update
        K = P_p / (P_p + r)
        m = m_p + K * (x[t] - m_p)
        P = (1.0 - K) * P_p
        mu[t] = m
    return mu


def garch_persistence_oracle(rets: np.ndarray, window: int) -> float | None:
    """GARCH(1,1) variance-targeting MLE persistence (alpha+beta) — an
    independent scipy-minimize objective (no sklearn)."""
    seg = rets[-window:]
    if len(seg) < 30 or np.std(seg) <= 1e-12:
        return None
    long_var = float(np.var(seg))

    def nll(p):
        a, b = p
        if a < 1e-6 or b < 1e-6 or a + b >= 0.999:
            return 1e12
        w = max(long_var * (1 - a - b), 1e-12)
        h = np.full(len(seg), long_var)
        for t in range(1, len(seg)):
            h[t] = w + a * seg[t - 1] ** 2 + b * h[t - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.sum(np.log(h) + seg**2 / h))

    res = minimize(nll, np.array([0.05, 0.9]), method="Nelder-Mead",
                   options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-6})
    if not getattr(res, "success", False):
        return None
    a, b = float(res.x[0]), float(res.x[1])
    if not (np.isfinite(a) and np.isfinite(b)) or a < 1e-6 or b < 1e-6 or a + b >= 0.999:
        return None
    return a + b


def har_forecast_oracle(rv: np.ndarray, window: int) -> float | None:
    """HAR-RV next-period forecast via explicit design matrix:
    y = RV_{s+1}, X = [1, RV_s, mean5_s, mean22_s]; predict from X_t."""
    seg = rv[-window:]
    n = len(seg)
    if n < 30:
        return None
    daily = seg
    weekly = np.convolve(seg, np.ones(5) / 5, mode="valid")
    # align: weekly[k] = mean(seg[k:k+5]); pad front with NaN
    w5 = np.full(n, np.nan)
    w5[4:] = weekly
    m22 = np.convolve(seg, np.ones(22) / 22, mode="valid")
    m = np.full(n, np.nan)
    m[21:] = m22
    X = np.column_stack([np.ones(n), daily, w5, m])
    target = np.concatenate([seg[1:], [np.nan]])
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(target)
    if valid.sum() < 25:
        return None
    Xs, y = X[valid], target[valid]
    beta, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    pred = float(np.dot(X[-1], beta))
    return float(np.sqrt(max(pred, 0.0)))
