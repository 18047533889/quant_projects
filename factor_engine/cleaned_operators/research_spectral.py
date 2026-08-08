# -*- coding: utf-8 -*-
"""Research-surface spectral / nonlinear-dependence primitives (2026-08-08).

These operators are statistically meaningful but either estimator-heavy or
assumption-laden, so they are registered on the *research* surface only and
stay out of the default production grammar:

* ``ts_bicoherence_max`` — maximum squared bicoherence over the frequency
  triangle (quadratic phase coupling).  Segment-averaged for variance control;
  fixed frequency grid and normalisation.
* ``ts_kernel_granger_score`` — kernel-ridge predictive improvement of ``x``
  for ``y`` under *blocked* out-of-sample evaluation: ``ln(MSE_restricted /
  MSE_full)``.  Fixed RBF kernel / ridge; no training-residual cheating.
* ``ts_residualized_hsic`` — HSIC between kernel-ridge residualisations of
  ``x`` and ``y`` on ``z`` (the honest approximation of conditional
  dependence; a true conditional-kernel statistic would require KCI, so the
  canonical is named for what it actually computes).
* ``ts_bds_statistic`` — the Brock–Dechert–Scheinkman statistic
  ``√n·(c_m − c_1^m)/σ_m`` for serial dependence (correlation integrals on a
  sup-norm embedding, triple correlation ``K`` via a two-pointer count).
* ``ts_sr_gaussian_mean_shift_score`` — a *named* Gaussian Shiryaev-Roberts
  mean-shift statistic (``W_t = max(0, W_{t-1} + μ·z_t − μ²/2)`` on
  baseline-standardised residuals), not a generic non-parametric SR.

All deterministic (no randomised kernels / bootstrap), strict-PIT, NaN
fail-closed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.gemini_v2_common import (
    frame_like,
    register_dual,
    trailing_contiguous_finite,
    trailing_contiguous_multi,
    union_research,
)

_EPS = 1e-12


def _rbf(a: np.ndarray, b: np.ndarray, sigma: float) -> np.ndarray:
    d2 = np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2.0 * a @ b.T
    return np.exp(-d2 / (2.0 * sigma * sigma + _EPS))


# ---------------------------------------------------------------------------
# bicoherence
# ---------------------------------------------------------------------------
def _bicoherence_max(v: np.ndarray, n_segments: int) -> float:
    n = v.shape[0]
    ns = max(2, int(n_segments))
    seg = n // ns
    if seg < 8:
        return np.nan
    max_freq = min(32, seg // 2)
    B_sum = np.zeros((max_freq, max_freq), dtype=complex)
    S_sum = np.zeros(max_freq, dtype=float)
    for s in range(ns):
        chunk = v[s * seg : (s + 1) * seg]
        if not np.all(np.isfinite(chunk)):
            return np.nan
        t = np.arange(seg, dtype=float)
        chunk = chunk - np.polyval(np.polyfit(t, chunk, 1), t)
        hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * t / (seg - 1.0))) if seg > 1 else np.ones(seg)
        X = np.fft.rfft(chunk * hann)
        Xf = X[1 : max_freq + 1]
        for f1 in range(1, max_freq + 1):
            x1 = Xf[f1 - 1]
            for f2 in range(1, max_freq - f1 + 1):
                B_sum[f1 - 1, f2 - 1] += x1 * Xf[f2 - 1] * np.conj(Xf[f1 + f2 - 1])
        S_sum += np.abs(Xf) ** 2
    best = 0.0
    for f1 in range(1, max_freq + 1):
        for f2 in range(1, max_freq - f1 + 1):
            denom = S_sum[f1 - 1] * S_sum[f2 - 1] * S_sum[f1 + f2 - 1]
            if denom <= _EPS:
                continue
            b2 = abs(B_sum[f1 - 1, f2 - 1]) ** 2 / denom
            if b2 > best:
                best = float(b2)
    return best if best > 0.0 else np.nan


def _ts_bicoherence_max(x: pd.DataFrame, window: int = 120, n_segments: int = 4) -> pd.DataFrame:
    if int(window) < 16:
        raise ValueError("ts_bicoherence_max requires window >= 16")
    rows, cols = x.shape
    w = int(window)
    arr = x.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = arr[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            if v.size < 16:
                continue
            val = _bicoherence_max(v, int(n_segments))
            if np.isfinite(val):
                out[r, c] = val
    return frame_like(x, out)


# ---------------------------------------------------------------------------
# kernel Granger (blocked OOS)
# ---------------------------------------------------------------------------
def _kernel_granger_score(y: np.ndarray, x: np.ndarray, lag: int) -> float:
    n = y.shape[0]
    lg = max(1, int(lag))
    n_avail = n - lg
    if n_avail < 24:
        return np.nan
    Y = y[lg:]
    Ylags = np.column_stack([y[lg - k : n - k] for k in range(1, lg + 1)])
    Xlags = np.column_stack([x[lg - k : n - k] for k in range(1, lg + 1)])
    train_n = int(0.7 * n_avail)
    test_n = n_avail - train_n
    if test_n < 6 or train_n < 12:
        return np.nan
    Ytr = Y[:train_n]
    Yte = Y[train_n:]
    Yl_tr = Ylags[:train_n]
    Yl_te = Ylags[train_n:]
    Xl_tr = Xlags[:train_n]
    Xl_te = Xlags[train_n:]

    def _kridge(K_tr, target, K_te):
        ntr = K_tr.shape[0]
        lam = 1e-3 * np.trace(K_tr) / ntr
        alpha = np.linalg.solve(K_tr + lam * np.eye(ntr), target)
        return K_te @ alpha

    sigma_full = float(np.median(np.sqrt(np.sum((Yl_tr[:, None, :] - Yl_tr[None, :, :]) ** 2, axis=2))))
    if not np.isfinite(sigma_full) or sigma_full <= _EPS:
        sigma_full = 1.0

    Kr = _rbf(Yl_tr, Yl_tr, sigma_full)
    Kte_r = _rbf(Yl_te, Yl_tr, sigma_full)
    pred_r = _kridge(Kr, Ytr, Kte_r)
    mse_r = float(np.mean((Yte - pred_r) ** 2))

    P_tr = np.hstack([Yl_tr, Xl_tr])
    P_te = np.hstack([Yl_te, Xl_te])
    sigma_full = float(np.median(np.sqrt(np.sum((P_tr[:, None, :] - P_tr[None, :, :]) ** 2, axis=2))))
    if not np.isfinite(sigma_full) or sigma_full <= _EPS:
        sigma_full = 1.0
    Kf = _rbf(P_tr, P_tr, sigma_full)
    Kte_f = _rbf(P_te, P_tr, sigma_full)
    pred_f = _kridge(Kf, Ytr, Kte_f)
    mse_f = float(np.mean((Yte - pred_f) ** 2))

    if mse_r <= _EPS or mse_f <= _EPS:
        return np.nan
    return float(np.log(mse_r / mse_f))


def _ts_kernel_granger_score(y: pd.DataFrame, x: pd.DataFrame, window: int = 120, lag: int = 2) -> pd.DataFrame:
    if int(window) < 30:
        raise ValueError("ts_kernel_granger_score requires window >= 30")
    rows, cols = y.shape
    w = int(window)
    arr_y = y.to_numpy(dtype=float)
    arr_x = x.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            lo = max(0, r - w + 1)
            run = trailing_contiguous_multi(arr_y[lo : r + 1, c], arr_x[lo : r + 1, c])
            if run is None:
                continue
            if run[0].shape[0] < 30:
                continue
            val = _kernel_granger_score(run[0], run[1], int(lag))
            if np.isfinite(val):
                out[r, c] = val
    return frame_like(y, out)


# ---------------------------------------------------------------------------
# residualised HSIC
# ---------------------------------------------------------------------------
def _residualized_hsic(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    n = x.shape[0]
    if n < 20:
        return np.nan
    zz = z.reshape(-1, 1)
    sigma = float(np.median(np.abs(z[:, None] - z[None, :])))
    if not np.isfinite(sigma) or sigma <= _EPS:
        sigma = 1.0
    Kzz = _rbf(zz, zz, sigma)
    lam = 1e-3 * np.trace(Kzz) / n
    proj = Kzz @ np.linalg.solve(Kzz + lam * np.eye(n), np.eye(n))
    rx = x - proj @ x
    ry = y - proj @ y

    sigma_x = float(np.median(np.abs(rx[:, None] - rx[None, :])))
    sigma_y = float(np.median(np.abs(ry[:, None] - ry[None, :])))
    if not np.isfinite(sigma_x) or sigma_x <= _EPS:
        sigma_x = 1.0
    if not np.isfinite(sigma_y) or sigma_y <= _EPS:
        sigma_y = 1.0
    Kx = _rbf(rx.reshape(-1, 1), rx.reshape(-1, 1), sigma_x)
    Ky = _rbf(ry.reshape(-1, 1), ry.reshape(-1, 1), sigma_y)
    H = np.eye(n) - np.ones((n, n)) / n
    hsic = float(np.trace(Kx @ H @ Ky @ H) / (n * n))
    return hsic if np.isfinite(hsic) else np.nan


def _ts_residualized_hsic(x: pd.DataFrame, y: pd.DataFrame, z: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    if int(window) < 24:
        raise ValueError("ts_residualized_hsic requires window >= 24")
    rows, cols = x.shape
    w = int(window)
    ax = x.to_numpy(dtype=float)
    ay = y.to_numpy(dtype=float)
    az = z.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            lo = max(0, r - w + 1)
            run = trailing_contiguous_multi(ax[lo : r + 1, c], ay[lo : r + 1, c], az[lo : r + 1, c])
            if run is None:
                continue
            if run[0].shape[0] < 24:
                continue
            val = _residualized_hsic(run[0], run[1], run[2])
            if np.isfinite(val):
                out[r, c] = val
    return frame_like(x, out)


# ---------------------------------------------------------------------------
# BDS statistic
# ---------------------------------------------------------------------------
def _bds_statistic(v: np.ndarray, m: int, distance_multiplier: float) -> float:
    n = v.shape[0]
    if n < 30 or m < 1:
        return np.nan
    eps = float(distance_multiplier) * float(np.std(v))
    if not np.isfinite(eps) or eps <= _EPS:
        return np.nan

    def _c_integral(dim: int) -> float:
        count = 0
        if dim == 1:
            for i in range(n):
                vi = v[i]
                for j in range(i + 1, n):
                    if abs(vi - v[j]) < eps:
                        count += 1
        else:
            for i in range(n - dim + 1):
                seg_i = v[i : i + dim]
                for j in range(i + 1, n - dim + 1):
                    if np.max(np.abs(seg_i - v[j : j + dim])) < eps:
                        count += 1
        return 2.0 * count / (n * (n - 1))

    c1 = _c_integral(1)
    cm = _c_integral(m)
    if c1 <= _EPS or cm < 0.0:
        return np.nan
    # triple correlation K: fraction of 3-index subsets with all pairwise < eps
    sv = np.sort(v)
    k_count = 0
    rptr = 0
    for i in range(n):
        if rptr < i:
            rptr = i
        while rptr + 1 < n and sv[rptr + 1] - sv[i] < eps:
            rptr += 1
        cnt = rptr - i
        k_count += cnt * (cnt - 1) // 2
    triples = n * (n - 1) * (n - 2) / 6.0
    K = k_count / triples if triples > 0 else 0.0
    if K <= _EPS:
        return np.nan
    sigma2 = 4.0 * (
        K ** m
        + 2.0 * sum(c1 ** (2 * j) * K ** (m - j) for j in range(1, m))
        + (m - 1) ** 2 * c1 ** (2 * m)
        - m ** 2 * K * c1 ** (2 * m - 2)
    )
    if sigma2 <= _EPS:
        return np.nan
    return float(np.sqrt(n) * (cm - c1 ** m) / np.sqrt(sigma2))


def _ts_bds_statistic(x: pd.DataFrame, window: int = 250, embedding_dim: int = 2, distance_multiplier: float = 1.5) -> pd.DataFrame:
    if int(window) < 30:
        raise ValueError("ts_bds_statistic requires window >= 30")
    rows, cols = x.shape
    w = int(window)
    arr = x.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = arr[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            if v.size < 30:
                continue
            val = _bds_statistic(v, int(embedding_dim), float(distance_multiplier))
            if np.isfinite(val):
                out[r, c] = val
    return frame_like(x, out)


# ---------------------------------------------------------------------------
# Gaussian Shiryaev-Roberts
# ---------------------------------------------------------------------------
def _sr_gaussian(v: np.ndarray, shift_sigma: float, baseline_window: int) -> float:
    n = v.shape[0]
    bw = max(4, int(baseline_window))
    if n < bw + 8:
        return np.nan
    base = v[:bw]
    mu = float(np.mean(base))
    sd = float(np.std(base))
    if not np.isfinite(mu) or not np.isfinite(sd) or sd <= _EPS:
        return np.nan
    shift = float(shift_sigma)
    w_stat = 0.0
    for t in range(n):
        z = (v[t] - mu) / sd
        w_stat = max(0.0, w_stat + shift * z - shift * shift / 2.0)
    return float(w_stat)


def _ts_sr_gaussian_mean_shift_score(x: pd.DataFrame, window: int = 120, shift_sigma: float = 1.0, baseline_window: int = 40) -> pd.DataFrame:
    if int(window) < 20:
        raise ValueError("ts_sr_gaussian_mean_shift_score requires window >= 20")
    rows, cols = x.shape
    w = int(window)
    arr = x.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = arr[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            if v.size < int(baseline_window) + 8:
                continue
            val = _sr_gaussian(v, float(shift_sigma), int(baseline_window))
            if np.isfinite(val):
                out[r, c] = val
    return frame_like(x, out)


_SPECS: dict[str, dict[str, Any]] = {
    "ts_bicoherence_max": {
        "fn": _ts_bicoherence_max,
        "params": ["x", "window", "n_segments"],
        "category": "research_spectral",
        "domain": "cross_spectral",
        "unit": "ratio",
        "cost": 7,
        "tags_extra": [],
        "output_unit": "ratio",
    },
    "ts_kernel_granger_score": {
        "fn": _ts_kernel_granger_score,
        "params": ["y", "x", "window", "lag"],
        "category": "research_nonlinear",
        "domain": "causality",
        "unit": "level",
        "cost": 9,
        "tags_extra": [],
        "output_unit": "level",
    },
    "ts_residualized_hsic": {
        "fn": _ts_residualized_hsic,
        "params": ["x", "y", "z", "window"],
        "category": "research_nonlinear",
        "domain": "dependence",
        "unit": "level",
        "cost": 9,
        "tags_extra": [],
        "output_unit": "level",
    },
    "ts_bds_statistic": {
        "fn": _ts_bds_statistic,
        "params": ["x", "window", "embedding_dim", "distance_multiplier"],
        "category": "research_dependence",
        "domain": "dependence",
        "unit": "level",
        "cost": 6,
        "tags_extra": [],
        "output_unit": "level",
    },
    "ts_sr_gaussian_mean_shift_score": {
        "fn": _ts_sr_gaussian_mean_shift_score,
        "params": ["x", "window", "shift_sigma", "baseline_window"],
        "category": "research_sequential",
        "domain": "change_point",
        "unit": "level",
        "cost": 4,
        "tags_extra": [],
        "output_unit": "level",
    },
}


def _register() -> None:
    for canonical, spec in _SPECS.items():
        register_dual(
            canonical,
            spec["fn"],
            spec["params"],
            category=spec["category"],
            domain=spec["domain"],
            unit=spec["unit"],
            cost=spec["cost"],
            source="research_spectral",
            tags_extra=spec["tags_extra"],
            output_unit=spec.get("output_unit"),
        )
    union_research(*_SPECS.keys())


_register()
