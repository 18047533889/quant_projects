# -*- coding: utf-8 -*-
"""Research-surface spectral / nonlinear-dependence primitives (2026-08-08).

These operators are statistically meaningful but either estimator-heavy or
assumption-laden, so they are registered on the *research* surface only and
stay out of the default production grammar:

* ``ts_bicoherence_top_decile_mean`` — mean squared bicoherence of the TOP
  DECILE of the frequency triangle (quadratic phase coupling), the
  extreme-selection-stable replacement for a raw ``max``.  Segment-averaged for
  variance control; fixed frequency grid and normalisation.  The legacy name
  ``ts_bicoherence_max`` resolves as a deprecated alias.
* ``ts_bicoherence_top_decile_excess`` — the top-decile mean MINUS a
  deterministic null baseline estimated from phase-randomised surrogates of the
  same window (same power spectrum, destroyed phases; fixed seed).  Under pure
  white noise the null subtraction drives the excess toward 0 regardless of
  ``window``/``n_segments``/``max_freq``, so a searchable config cannot chase
  the estimator's finite-sample bias; genuine quadratic phase coupling yields a
  positive excess.
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

from cleaned_operators.base import ParamSpec, RelationalParamSpec, ParamRole
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
    # Audit #59: bicoherence is a peak-CONCENTRATION statistic in the frequency
    # triangle — max_freq (and therefore the number of (f1, f2) pairs) grows with
    # window, so under pure noise the top-decile mean has a mechanical N
    # dependence.  window is an estimator-resolution knob (coarse grid only),
    # not a free economic parameter.
    "window": ParamSpec(dtype=int, min=16, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    "n_segments": ParamSpec(dtype=int, min=2, param_role=ParamRole.ESTIMATOR_RESOLUTION),
}
_GRANGER_SPEC = {
    "window": ParamSpec(dtype=int, min=30),
    "lag": ParamSpec(dtype=int, min=1),
}
_HSIC_SPEC = {
    "window": ParamSpec(dtype=int, min=24),
    "purge_gap": ParamSpec(dtype=int, min=1, param_role=ParamRole.ESTIMATOR_RESOLUTION),
}
# R6-211: BDS with embedding_dim == 1 is degenerate (C_m == C_1 makes the
# asymptotic variance collapse to 0) — the search space must start at dim >= 2.
# R6-212: distance_multiplier == 0 gives eps == 0, which the kernel rejects at
# runtime; reviewed grid only (0.5/1.0/1.5/2.0).
_BDS_SPEC = {
    "window": ParamSpec(dtype=int, min=30),
    "embedding_dim": ParamSpec(dtype=int, min=2),
    "distance_multiplier": ParamSpec(dtype=float, choices=(0.5, 1.0, 1.5, 2.0), param_role=ParamRole.ESTIMATOR_RESOLUTION),
}
# R6-217: shift_sigma == 0 makes log_lambda = 0 for every t (no evidence), and
# the kernel explicitly rejects shift <= 0 at runtime.  Search must not offer 0.
_SR_SPEC = {
    "window": ParamSpec(dtype=int, min=20),
    "shift_sigma": ParamSpec(dtype=float, min=0.01, param_role=ParamRole.ESTIMATOR_RESOLUTION),
    "baseline_window": ParamSpec(dtype=int, min=4, param_role=ParamRole.ESTIMATOR_RESOLUTION),
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
def _bicoherence_values(v: np.ndarray, n_segments: int) -> list[float]:
    """Squared-bicoherence values over the whole positive-frequency triangle.

    Shared by the top-decile-mean canonical, the legacy ``_bicoherence_max``
    compat helper and the phase-surrogate null of ``_bicoherence_top_decile_excess``.
    """
    n = v.shape[0]
    ns = max(2, int(n_segments))
    seg = n // ns
    if seg < 8:
        return []
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
            return []
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
    return vals


def _bicoherence_top_decile_mean(v: np.ndarray, n_segments: int) -> float:
    vals = _bicoherence_values(v, n_segments)
    if not vals:
        return np.nan
    # R6-205: ``max bicoherence`` over all (f1,f2) is extreme-selection biased —
    # more frequency pairs (larger window / segments) raise the pure-noise
    # maximum mechanically.  Report the mean of the top-decile values instead
    # of the raw maximum, which is far more stable across parameter changes.
    # R9-OP-028: the canonical NAME now says what the statistic is
    # (``ts_bicoherence_top_decile_mean``); the old ``*_max`` name is only a
    # deprecated alias so a research note cannot misread it as a maximum.
    vals_sorted = np.sort(np.asarray(vals))
    k = max(1, int(np.ceil(vals_sorted.size * 0.1)))
    return float(vals_sorted[-k:].mean())


def _bicoherence_max(v: np.ndarray, n_segments: int) -> float:
    """Legacy private helper: the raw maximum squared bicoherence.

    NOT the registered statistic (the canonical ``ts_bicoherence_max`` resolves
    to the top-decile mean per R9-OP-028).  Kept only so the stale round-5 audit
    test that imports ``_bicoherence_max`` keeps resolving; bounded in [0, 1].
    """
    vals = _bicoherence_values(v, n_segments)
    return float(max(vals)) if vals else np.nan


def _ts_bicoherence_top_decile_mean(x: pd.DataFrame, window: int = 120, n_segments: int = 4) -> pd.DataFrame:
    if int(window) < 16:
        raise ValueError("ts_bicoherence_top_decile_mean requires window >= 16")
    rows, cols = x.shape
    w = int(window)
    arr = x.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = arr[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            # Audit #58: no partial warmup — the front of a windowed bicoherence
            # must not be a shorter-segment spectrum.  Require the FULL window
            # (contiguous finite) or NaN.
            if v.size < w:
                continue
            val = _bicoherence_top_decile_mean(v, int(n_segments))
            if np.isfinite(val):
                out[r, c] = val
    return frame_like(x, out)


def _phase_surrogate(v: np.ndarray, seed: int) -> np.ndarray:
    """Phase-randomised surrogate of ``v`` (same power spectrum, destroyed phases).

    The magnitude spectrum is kept identical and the phases are randomised
    (preserving conjugate symmetry so the inverse FFT is real) — the standard
    Fourier surrogate that removes any quadratic phase coupling while keeping
    the linear autocorrelation.  Deterministic for a given ``seed``.
    """
    n = v.shape[0]
    X = np.fft.rfft(v)
    ph = np.random.default_rng(seed).uniform(0.0, 2.0 * np.pi, size=X.shape)
    ph[0] = 0.0  # DC bin stays real
    if n % 2 == 0:
        ph[-1] = 0.0  # Nyquist bin stays real for even n
    return np.fft.irfft(X * np.exp(1j * ph), n)


def _bicoherence_top_decile_excess(v: np.ndarray, n_segments: int, n_surrogates: int = 5) -> float:
    """Top-decile bicoherence MINUS a deterministic phase-surrogate null.

    The raw top-decile mean is biased under pure noise and the bias grows with
    ``window``/``n_segments``/``max_freq`` (P2-34), so a searchable config can
    chase the estimator's own bias.  Here the same statistic is evaluated on
    phase-randomised surrogates of the SAME window and the mean of those nulls
    is subtracted: under white noise the excess hovers near 0 for any parameter
    choice, while genuine quadratic phase coupling survives the phase scramble
    and gives a positive excess.
    """
    real = _bicoherence_top_decile_mean(v, n_segments)
    if not np.isfinite(real):
        return np.nan
    nulls: list[float] = []
    for s in range(max(1, int(n_surrogates))):
        surr = _phase_surrogate(v, seed=20260809 + s)
        nb = _bicoherence_top_decile_mean(surr, n_segments)
        if np.isfinite(nb):
            nulls.append(nb)
    if not nulls:
        return np.nan
    return float(real - float(np.mean(nulls)))


def _ts_bicoherence_top_decile_excess(
    x: pd.DataFrame, window: int = 120, n_segments: int = 4, n_surrogates: int = 5
) -> pd.DataFrame:
    if int(window) < 16:
        raise ValueError("ts_bicoherence_top_decile_excess requires window >= 16")
    rows, cols = x.shape
    w = int(window)
    arr = x.to_numpy(dtype=float)
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        col = arr[:, c]
        for r in range(rows):
            lo = max(0, r - w + 1)
            v = trailing_contiguous_finite(col[lo : r + 1])
            # Audit #58: no partial warmup (same contract as the top-decile-mean
            # canonical).
            if v.size < w:
                continue
            val = _bicoherence_top_decile_excess(v, int(n_segments), int(n_surrogates))
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

    # R9-OP-029 (regularisation fairness): ``lam`` MUST be defined from the
    # RESTRICTED training kernel only and shared verbatim by both models.  The
    # old ``_kridge`` recomputed ``lam = 1e-3·trace(K)/n`` *inside* each call —
    # restricted used K=Kr but full used K=Kr+0.5·Kx, so lambda_r != lambda_f
    # and ``MSE_R/MSE_F`` absorbed a regularisation change on top of "adding X".
    # R6-207 (model fairness): the restricted and full models share the same
    # Y-lag RBF bandwidth, so log(MSE_R/MSE_F) measures "how much predictive
    # information X adds" and nothing else.  P1-11/P1-12 refine the full kernel:
    # the X kernel uses X's OWN bandwidth (P1-11) and the sum is trace-normalised
    # by (1+eta) so trace(K_F) == trace(K_R) (P1-12) — "adding X" (with a fair
    # kernel and fair regularisation) is the only change.
    def _median_bandwidth(tr: np.ndarray) -> float:
        # median pairwise distance over the TRAINING block only (never the
        # test block, so no OOS contamination).
        d2 = (
            np.sum(tr[:, None, :] ** 2, axis=2)
            + np.sum(tr[None, :, :] ** 2, axis=2)
            - 2.0 * tr @ tr.T
        )
        sigma = float(np.median(np.sqrt(np.maximum(d2, 0.0))))
        if not np.isfinite(sigma) or sigma <= _EPS:
            sigma = 1.0
        return sigma

    # P1-11 (fairness): the X kernel's smoothness is set by X's OWN distance
    # geometry.  The old code reused ``sigma_y`` for Kx, so X's kernel width
    # depended on Y's scale/structure even though X is separately standardised.
    sigma_y = _median_bandwidth(Yl_tr)
    sigma_x = _median_bandwidth(Xl_tr)

    Kr = _rbf(Yl_tr, Yl_tr, sigma_y)
    ntr = Kr.shape[0]
    lambda_shared = 1e-3 * float(np.trace(Kr)) / ntr

    def _kridge(K_tr, target, K_te, lam):
        alpha = np.linalg.solve(K_tr + lam * np.eye(ntr), target)
        return K_te @ alpha

    Kte_r = _rbf(Yl_te, Yl_tr, sigma_y)
    pred_r = _kridge(Kr, Ytr, Kte_r, lambda_shared)
    mse_r = float(np.mean((Yte - pred_r) ** 2))

    # Full model: K_F = (K_Y + eta·K_X)/(1+eta) evaluated with X's OWN
    # bandwidth ``sigma_x`` and the SAME shared ridge ``lambda_shared``.
    # P1-12 (fairness): an RBF Gram matrix has all diagonal entries 1, so
    # ``trace(K_Y) = trace(K_X) = ntr`` and the old ``K_F = K_Y + eta·K_X`` had
    # ``trace(K_F) = (1+eta)·trace(K_R)`` — the full model was regularised ~1.5x
    # weaker relative to its own kernel magnitude, so ``log(MSE_R/MSE_F)`` mixed
    # in a regularisation change on top of "adding X".  Normalising the SUM by
    # ``(1+eta)`` makes ``trace(K_F) == trace(K_R)`` exactly, so the shared
    # ridge regularises both models equally and the score measures "how much
    # predictive information X adds" and nothing else.
    Kx_tr = _rbf(Xl_tr, Xl_tr, sigma_x)
    Kte_x = _rbf(Xl_te, Xl_tr, sigma_x)
    eta = 0.5
    scale = 1.0 + eta
    Kf = (Kr + eta * Kx_tr) / scale
    Kte_f = (Kte_r + eta * Kte_x) / scale
    pred_f = _kridge(Kf, Ytr, Kte_f, lambda_shared)
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
def _residualized_hsic(x: np.ndarray, y: np.ndarray, z: np.ndarray, purge_gap: int = 3) -> float:
    n = x.shape[0]
    if n < 24:
        return np.nan
    purge = max(1, int(purge_gap))
    # R6-209 (blocked cross-fitting) + P1-13 (purged CONTIGUOUS blocks): the
    # original code fit the kernel smoother x~z and y~z on the FULL window and
    # measured HSIC on the SAME residuals — a flexible kernel regression has
    # in-sample bias that inflates the "residual independence".  It then moved
    # to an INTERLEAVED even/odd split (train = 0,2,4…; test = 1,3,5…), but that
    # is NOT blocked time-series cross-fitting: every test point still has
    # immediate neighbours in the training set, so the conditional smoother
    # overfits the temporal neighbourhood.  Use contiguous blocks instead: the
    # window is split into a contiguous train block / purge gap / test block and
    # the cross-fit runs BOTH directions (early block -> late block and late
    # block -> early block).  The purge gap guarantees test points have no
    # immediate neighbours in the training set; HSIC is measured on residuals
    # the smoother never saw.
    split = n // 2
    fwd_tr = np.arange(0, split)
    fwd_te = np.arange(split + purge, n)
    bwd_tr = np.arange(split, n)
    bwd_te = np.arange(0, split - purge)
    if split < 12 or fwd_te.size < 6 or bwd_te.size < 6:
        return np.nan
    rx_all = np.full(n, np.nan)
    ry_all = np.full(n, np.nan)
    for tr_idx, te_idx in ((fwd_tr, fwd_te), (bwd_tr, bwd_te)):
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
    if m < 3:
        return np.nan
    # R9-OP-030 (mathematical correctness): the previous "unbiased HSIC" was
    # only ``mean_{i!=j} Kx[i,j]·Ky[i,j]`` — a raw cross-moment with NO kernel
    # centering, so under independence its expectation is ``E[Kx]·E[Ky]`` ≠ 0
    # and it could not be read as "residual dependence ≈ 0 under
    # independence".  Use the textbook **biased centred HSIC** V-statistic
    # (Gretton et al. 2005):
    #     H = I - 11'/m,   HSIC = tr(Kx·H·Ky·H) / (m-1)²
    # which is exactly the empirical distance covariance-style measure: it is
    # non-negative, ≈ 0 iff the (residualised) variables are independent, and
    # > 0 under any (non-linear) dependence.  (A true unbiased U-statistic
    # estimator is tracked separately; this canonical now measures what its
    # name claims.)
    H = np.eye(m) - np.ones((m, m)) / m
    KxH = Kx @ H
    HKyH = H @ Ky @ H
    hsic = float(np.trace(KxH @ HKyH)) / float((m - 1) * (m - 1))
    return hsic if np.isfinite(hsic) else np.nan


def _ts_residualized_hsic(
    x: pd.DataFrame, y: pd.DataFrame, z: pd.DataFrame, window: int = 120, purge_gap: int = 3
) -> pd.DataFrame:
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
            val = _residualized_hsic(run[0], run[1], run[2], int(purge_gap))
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
    "ts_bicoherence_top_decile_mean": {
        "fn": _ts_bicoherence_top_decile_mean,
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
    "ts_bicoherence_top_decile_excess": {
        "fn": _ts_bicoherence_top_decile_excess,
        "params": ["x", "window", "n_segments", "n_surrogates"],
        "category": "research_spectral",
        "domain": "cross_spectral",
        "unit": "ratio",
        "cost": 8,
        "tags_extra": [],
        "output_unit": "ratio",
        "param_specs": dict(
            _BICOH_SPEC, n_surrogates=ParamSpec(dtype=int, min=1)
        ),
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
        "params": ["x", "y", "z", "window", "purge_gap"],
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
            relational_specs=spec.get("relational_specs"),
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

    # R9-OP-028: honest rename — the statistic is the mean of the top-decile
    # bicoherence values, NOT a maximum.  ``ts_bicoherence_max`` stays as a
    # deprecated resolving alias so existing recipes keep loading, but the
    # canonical name no longer overstates the statistic.
    try:
        OperatorRegistry.register_alias(
            "ts_bicoherence_max",
            "ts_bicoherence_top_decile_mean",
        )
    except (KeyError, ValueError):
        pass  # already registered


_register()
