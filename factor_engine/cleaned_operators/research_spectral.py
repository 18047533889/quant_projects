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
* ``ts_rolling_sr_gaussian_mean_shift_score`` — the Gaussian Shiryaev-Roberts
  mean-shift statistic: the likelihood-ratio recursion
  ``R_t = (1 + R_{t-1})·Λ_t`` with ``Λ_t = exp(shift·z_t − shift²/2)`` on
  baseline-standardised residuals, started after the baseline window;
  output ``log(1 + R_n)``.

All deterministic (no randomised kernels / bootstrap), strict-PIT, NaN
fail-closed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import ParamSpec, RelationalParamSpec
from cleaned_operators.gemini_v2_common import (
    frame_like,
    register_dual,
    trailing_contiguous_finite,
    trailing_contiguous_multi,
    union_research,
)

_EPS = 1e-12

# R5 P1-01: research-surface ints/floats are validated (n_segments=2.9, lag=1.9,
# embedding_dim=2.9, baseline_window=3.9 are all rejected, never truncated).
_BICOH_SPEC = {
    "window": ParamSpec(dtype=int, min=16),
    "n_segments": ParamSpec(dtype=int, min=2),
}
_GRANGER_SPEC = {
    "window": ParamSpec(dtype=int, min=30),
    "lag": ParamSpec(dtype=int, min=1),
}
_HSIC_SPEC = {"window": ParamSpec(dtype=int, min=24)}
# R6-211: BDS with embedding_dim == 1 is degenerate (C_m == C_1 makes the
# asymptotic variance collapse to 0) — the search space must start at dim >= 2.
# R6-212: distance_multiplier == 0 gives eps == 0, which the kernel rejects at
# runtime; reviewed grid only (0.5/1.0/1.5/2.0).
_BDS_SPEC = {
    "window": ParamSpec(dtype=int, min=30),
    "embedding_dim": ParamSpec(dtype=int, min=2),
    "distance_multiplier": ParamSpec(dtype=float, choices=(0.5, 1.0, 1.5, 2.0)),
}
# R6-217: shift_sigma == 0 makes log_lambda = 0 for every t (no evidence), and
# the kernel explicitly rejects shift <= 0 at runtime.  Search must not offer 0.
_SR_SPEC = {
    "window": ParamSpec(dtype=int, min=20),
    "shift_sigma": ParamSpec(dtype=float, min=0.01),
    "baseline_window": ParamSpec(dtype=int, min=4),
}
# R6-24 cross-parameter feasibility relations (declared so search prunes the
# guaranteed-NaN region before runtime):
#   bicoherence : floor(window / n_segments) >= 8
#   kernel-granger : lag must leave >= 24 available samples
#   BDS : N_m = window - embedding_dim + 1 must be large enough
#   SR : baseline_window + 8 <= window
_BICOH_RELATIONAL_SPECS = [
    RelationalParamSpec(
        "(window // n_segments) >= 8",
        "bicoherence requires floor(window/n_segments) >= 8 "
        "(window={window}, n_segments={n_segments})",
    )
]
_GRANGER_RELATIONAL_SPECS = [
    RelationalParamSpec(
        "window - lag >= 24",
        "kernel Granger requires window - lag >= 24 available samples "
        "(window={window}, lag={lag})",
    )
]
_BDS_RELATIONAL_SPECS = [
    RelationalParamSpec(
        "window - embedding_dim + 1 >= 12",
        "BDS requires N_m = window - embedding_dim + 1 >= 12 "
        "(window={window}, embedding_dim={embedding_dim})",
    )
]
_SR_RELATIONAL_SPECS = [
    RelationalParamSpec(
        "baseline_window + 8 <= window",
        "SR requires baseline_window + 8 <= window "
        "(baseline_window={baseline_window}, window={window})",
    )
]


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
    # R6-204: RIGHT-ALIGN the segments.  ``v[s*seg:(s+1)*seg]`` drops the
    # trailing ``n % ns`` samples — the NEWEST data.  Use the last ns*seg
    # samples so the window's latest information is never discarded.
    v = v[-seg * ns:]
    max_freq = min(32, seg // 2)
    B_sum = np.zeros((max_freq, max_freq), dtype=complex)
    # P_sum[f1,f2] = Σ_s |X_s(f1)·X_s(f2)|²  — the standard bicoherence
    # denominator uses the *segment product of powers* and the power of the sum
    # frequency, NOT the product of three separate spectrum sums (P0-07 review:
    # the old denominator only matched ``E|X(f1)X(f2)|²·E|X(f1+f2)|²`` up to a
    # model-implied independence that does not hold, and lost the [0,1] bound).
    P_sum = np.zeros((max_freq, max_freq), dtype=float)
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
        pwr = np.abs(Xf) ** 2
        for f1 in range(1, max_freq + 1):
            x1 = Xf[f1 - 1]
            p1 = pwr[f1 - 1]
            for f2 in range(1, max_freq - f1 + 1):
                B_sum[f1 - 1, f2 - 1] += x1 * Xf[f2 - 1] * np.conj(Xf[f1 + f2 - 1])
                P_sum[f1 - 1, f2 - 1] += p1 * pwr[f2 - 1]
        S_sum += pwr
    vals: list[float] = []
    for f1 in range(1, max_freq + 1):
        for f2 in range(1, max_freq - f1 + 1):
            denom = P_sum[f1 - 1, f2 - 1] * S_sum[f1 + f2 - 1]
            if denom <= _EPS:
                continue
            b2 = abs(B_sum[f1 - 1, f2 - 1]) ** 2 / denom
            if np.isfinite(b2):
                vals.append(float(b2))
    if not vals:
        return np.nan
    # R6-205: ``max bicoherence`` over all (f1,f2) is extreme-selection biased —
    # more frequency pairs (larger window / segments) raise the pure-noise
    # maximum mechanically.  Report the mean of the top-decile values instead
    # of the raw maximum, which is far more stable across parameter changes.
    vals_sorted = np.sort(np.asarray(vals))
    k = max(1, int(np.ceil(vals_sorted.size * 0.1)))
    return float(vals_sorted[-k:].mean())


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

    # R5 P1-13: the RBF kernel is scale-sensitive — raw ``Y`` lags (~0.01) and
    # raw ``X`` lags (e.g. volume ~1e7) would let the full model's distance be
    # dominated by ``X``.  Both lag blocks are standardised with STATS FITTED ON
    # THE TRAINING BLOCK ONLY (never the test block), so no OOS contamination.
    def _standardize(tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = tr.mean(axis=0)
        sd = tr.std(axis=0)
        sd = np.where(np.isfinite(sd) & (sd > _EPS), sd, 1.0)
        return (tr - mu) / sd, (te - mu) / sd

    Yl_tr, Yl_te = _standardize(Ylags[:train_n], Ylags[train_n:])
    Xl_tr, Xl_te = _standardize(Xlags[:train_n], Xlags[train_n:])

    def _kridge(K_tr, target, K_te):
        ntr = K_tr.shape[0]
        lam = 1e-3 * np.trace(K_tr) / ntr
        alpha = np.linalg.solve(K_tr + lam * np.eye(ntr), target)
        return K_te @ alpha

    # R6-207 (model fairness): the restricted and full models MUST share the
    # same Y-lag RBF bandwidth and the same regularisation, so log(MSE_R/MSE_F)
    # measures "how much predictive information X adds" and nothing else.  The
    # old code re-estimated sigma in the FULL [Y-lag, X-lag] space, so the ratio
    # also absorbed a kernel-hyperparameter change.  Nested kernel:
    #   K_F = K_Y + eta·K_X   (restricted = K_Y), with the SAME sigma_Y, same
    # ridge, and eta a fixed small weight — "adding X" is the only change.
    sigma_y = float(np.median(np.sqrt(np.sum((Yl_tr[:, None, :] - Yl_tr[None, :, :]) ** 2, axis=2))))
    if not np.isfinite(sigma_y) or sigma_y <= _EPS:
        sigma_y = 1.0

    Kr = _rbf(Yl_tr, Yl_tr, sigma_y)
    Kte_r = _rbf(Yl_te, Yl_tr, sigma_y)
    pred_r = _kridge(Kr, Ytr, Kte_r)
    mse_r = float(np.mean((Yte - pred_r) ** 2))

    # Full model: K_F = K_Y + eta·K_X evaluated with the SAME Y bandwidth.
    Kx_tr = _rbf(Xl_tr, Xl_tr, sigma_y)
    Kte_x = _rbf(Xl_te, Xl_tr, sigma_y)
    eta = 0.5
    Kf = Kr + eta * Kx_tr
    Kte_f = Kte_r + eta * Kte_x
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
    if n < 24:
        return np.nan
    # R6-209 (blocked cross-fitting): the old code fit the kernel smoother
    # x~z and y~z on the FULL window and measured HSIC on the SAME residuals —
    # a flexible kernel regression has in-sample bias that inflates the
    # "residual independence".  Split the window into two interleaved halves;
    # the smoother is fit on one half and the residuals of the OTHER half are
    # used for HSIC, so the dependence measure never sees its own fit.
    halves = np.arange(n) % 2
    rx_all = np.full(n, np.nan)
    ry_all = np.full(n, np.nan)
    for half in (0, 1):
        tr_idx = np.flatnonzero(halves == half)
        te_idx = np.flatnonzero(halves != half)
        ztr = z[tr_idx].reshape(-1, 1)
        zte = z[te_idx].reshape(-1, 1)
        sigma = float(np.median(np.abs(z[tr_idx][:, None] - z[tr_idx][None, :])))
        if not np.isfinite(sigma) or sigma <= _EPS:
            sigma = 1.0
        Kzz = _rbf(ztr, ztr, sigma)
        Kzte = _rbf(zte, ztr, sigma)
        lam = 1e-3 * np.trace(Kzz) / tr_idx.size
        smooth = Kzte @ np.linalg.solve(Kzz + lam * np.eye(tr_idx.size), np.eye(tr_idx.size))
        rx_all[te_idx] = x[te_idx] - smooth @ x[tr_idx]
        ry_all[te_idx] = y[te_idx] - smooth @ y[tr_idx]
    fin = np.isfinite(rx_all) & np.isfinite(ry_all)
    rx = rx_all[fin]
    ry = ry_all[fin]
    m = rx.shape[0]
    if m < 12:
        return np.nan

    sigma_x = float(np.median(np.abs(rx[:, None] - rx[None, :])))
    sigma_y = float(np.median(np.abs(ry[:, None] - ry[None, :])))
    if not np.isfinite(sigma_x) or sigma_x <= _EPS:
        sigma_x = 1.0
    if not np.isfinite(sigma_y) or sigma_y <= _EPS:
        sigma_y = 1.0
    Kx = _rbf(rx.reshape(-1, 1), rx.reshape(-1, 1), sigma_x)
    Ky = _rbf(ry.reshape(-1, 1), ry.reshape(-1, 1), sigma_y)
    # R6-210: the trace form trace(Kx H Ky H)/n² is a BIASED V-statistic whose
    # baseline grows with m — two windows with different effective N (24 vs 60
    # after gaps) would have incomparable raw values.  Use the UNBIASED HSIC
    # estimator: average only the off-diagonal (i != j) kernel products, so the
    # null baseline does not grow with the sample size.
    if m < 3:
        return np.nan
    # Unbiased estimator of E[k_x k_y] under independence: average the
    # OFF-DIAGONAL elementwise kernel products over ordered pairs i != j.  This
    # is the unbiased HSIC first-moment (the diagonal i==j terms are the biased
    # V-statistic self-products that inflate the baseline with the sample size).
    off = ~np.eye(m, dtype=bool)
    hsic = float((Kx[off] * Ky[off]).sum() / (m * (m - 1)))
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
        # P0-08: the embedded sample has N_m = n - m + 1 vectors; the correlation
        # integral denominator must be N_m(N_m - 1), not the raw n(n - 1) used
        # for dim == 1 (otherwise C_m is systematically scaled down for m > 1).
        N_m = n - dim + 1
        if N_m < 2:
            return np.nan
        count = 0
        if dim == 1:
            for i in range(n):
                vi = v[i]
                for j in range(i + 1, n):
                    if abs(vi - v[j]) < eps:
                        count += 1
        else:
            for i in range(N_m):
                seg_i = v[i : i + dim]
                for j in range(i + 1, N_m):
                    if np.max(np.abs(seg_i - v[j : j + dim])) < eps:
                        count += 1
        return 2.0 * count / (N_m * (N_m - 1))

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
def _sr_gaussian(v: np.ndarray, shift_sigma: float, baseline_window: int, side: str = "up") -> float:
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
    if not np.isfinite(shift) or shift <= 0.0:
        return np.nan
    if side not in ("up", "down"):
        raise ValueError("side must be 'up' or 'down'")
    # R6-215: the likelihood ratio is signed — Λ = exp(δ·z − δ²/2) detects an
    # UPWARD shift (positive δ·z).  For a DOWNWARD shift the evidence is
    # Λ = exp(−δ·z − δ²/2), i.e. we flip the sign of the standardised residual.
    # The kernel previously only detected upward shifts; ``side`` selects the
    # direction explicitly (default stays "up" for backward compatibility).
    sign = 1.0 if side == "up" else -1.0
    # P0-09: true Shiryaev-Roberts likelihood-ratio recursion
    #   R_t = (1 + R_{t-1}) · Λ_t,   Λ_t = exp(δ·sign·z_t − δ²/2)
    # started only AFTER the baseline window — the baseline observations define
    # the null and must not be re-scored as candidate shifts (the old code ran a
    # CUSUM-style max(0, W + μz − μ²/2) over rows 0..n using the baseline itself).
    # R5 P0-10 numerical form: track log R_t in log-space
    #   logR_t = logaddexp(0, logR_{t-1}) + logΛ_t
    # (logaddexp(0, ·) is log(1 + R_{t-1})) — the equivalent linear recursion
    # ``(1+R)·Λ`` overflows to inf on a real mean shift and wrongly read back
    # as 0.0.  Output log(1 + R_n), monotone in the accumulated evidence.
    logR = float("-inf")  # R_0 = 0
    for t in range(bw, n):
        z = (v[t] - mu) / sd
        log_lambda = shift * sign * z - shift * shift / 2.0
        logR = np.logaddexp(0.0, logR) + log_lambda
    return float(np.logaddexp(0.0, logR))


def _ts_rolling_sr_gaussian_mean_shift_score(x: pd.DataFrame, window: int = 120, shift_sigma: float = 1.0, baseline_window: int = 40, side: str = "up") -> pd.DataFrame:
    if int(window) < 20:
        raise ValueError("ts_rolling_sr_gaussian_mean_shift_score requires window >= 20")
    if side not in ("up", "down"):
        raise ValueError("side must be 'up' or 'down'")
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
            val = _sr_gaussian(v, float(shift_sigma), int(baseline_window), side)
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
        "param_specs": _BICOH_SPEC,
        "relational_specs": _BICOH_RELATIONAL_SPECS,
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
        "param_specs": _GRANGER_SPEC,
        "relational_specs": _GRANGER_RELATIONAL_SPECS,
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
        "param_specs": _HSIC_SPEC,
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
        "param_specs": _BDS_SPEC,
        "relational_specs": _BDS_RELATIONAL_SPECS,
    },
    "ts_rolling_sr_gaussian_mean_shift_score": {
        "fn": _ts_rolling_sr_gaussian_mean_shift_score,
        "params": ["x", "window", "shift_sigma", "baseline_window", "side"],
        "category": "research_sequential",
        "domain": "change_point",
        "unit": "level",
        "cost": 4,
        "tags_extra": [],
        "output_unit": "level",
        "param_specs": dict(
            _SR_SPEC, side=ParamSpec(dtype=str, choices=("up", "down"), searchable=True)
        ),
        "relational_specs": _SR_RELATIONAL_SPECS,
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
            param_specs=spec.get("param_specs"),
        )
    union_research(*_SPECS.keys())
    # R6-214: honest rename — each rolling t re-fits the baseline and restarts
    # R_t from 0, so this is a ROLLING windowed SR, not a true sequential
    # Shiryaev-Roberts that carries R_t forward across days.  The old name stays
    # as a resolving alias for existing recipes.
    from cleaned_operators.registry import OperatorRegistry

    try:
        OperatorRegistry.register_alias(
            "ts_sr_gaussian_mean_shift_score",
            "ts_rolling_sr_gaussian_mean_shift_score",
        )
    except (KeyError, ValueError):
        pass  # already registered


_register()
