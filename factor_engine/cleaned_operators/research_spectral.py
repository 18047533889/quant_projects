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
* ``ts_kernel_granger_score`` — blocked out-of-sample loss comparison between
  a Y-lag RBF model and a fixed trace-normalized additive Y/X-lag RBF model:
  ``ln(MSE_restricted / MSE_full)``.  This is a modeled diagnostic, not a
  causal or conditional-information proof.

Timing honesty — BLOCKED_HISTORICAL_EVALUATION (M-9xx)
-------------------------------------------------------
``ts_kernel_granger_score`` is NOT a t-1 fit -> forecast t predictor.  It is a
**trailing-window blocked train/test predictive diagnostic score**: the window
is split into a contiguous train block and a later test block
(``train_n = int(0.7 * n_available)``), and the scaler (per-block mean/std) and
the RBF bandwidths (median pairwise distance) are fitted on the TRAINING block
only — the test block never touches any fit statistic, so
``ln(MSE_restricted/MSE_full)`` carries no in-sample / training-residual
leakage.  The output compares those two fixed regularized models within this
trailing window; it is NOT a next-bar forecast or a causal hypothesis test.
The same blocked-cross-fit
discipline applies to ``ts_residualized_hsic`` (contiguous train / purge gap /
test blocks, both directions).
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

from factor_engine.cleaned_operators.base import ParamSpec, RelationalParamSpec, ParamRole
from factor_engine.cleaned_operators.gemini_v2_common import (
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
    # R22-044: embedding_dim is an estimator-resolution knob (coarse grid only),
    # never a free economic parameter.
    "embedding_dim": ParamSpec(dtype=int, min=2, param_role=ParamRole.ESTIMATOR_RESOLUTION),
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
    # Direct pairwise differences avoid catastrophic cancellation in
    # ||a||² + ||b||² - 2<a,b>.  Both train and query retain one coordinate
    # system; separate centering would change cross-kernel geometry.
    # Normalize differences, not large coordinates, before squaring.  This
    # retains close-point precision and prevents finite bandwidths squaring
    # into inf.  hypot preserves the existing additive EPS bandwidth exactly
    # in real arithmetic.  Temporary pairwise buffers are bounded by row tiles.
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("RBF inputs must be two-dimensional with matching feature widths")
    bandwidth = np.hypot(float(sigma), np.sqrt(_EPS / 2.0))
    d2 = np.zeros((a.shape[0], b.shape[0]), dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for start in range(0, a.shape[0], 32):
            stop = min(start + 32, a.shape[0])
            for column in range(a.shape[1]):
                left = a[start:stop, column, None]
                right = b[None, :, column]
                difference = left - right
                overflow = np.isinf(difference) & np.isfinite(left) & np.isfinite(right)
                difference /= bandwidth
                if np.any(overflow):
                    # Only opposite-sign finite operands can overflow a
                    # subtraction. Scaling those operands first is safe and
                    # cannot hide a close-point cancellation.
                    fallback = left / bandwidth - right / bandwidth
                    difference[overflow] = fallback[overflow]
                np.square(difference, out=difference)
                d2[start:stop] += difference
        d2 *= -0.5
        np.exp(d2, out=d2)
    return d2


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
    if not np.isfinite(v).all():
        return []
    magnitude = float(np.max(np.abs(v)))
    if magnitude == 0.0:
        return []
    # One causal-window normalization preserves relative segment powers and
    # the degree-six numerator/denominator ratio without unit under/overflow.
    v = v / magnitude
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
        chunk = (np.zeros_like(chunk) if np.all(chunk == chunk[0])
                 else chunk - np.polyval(np.polyfit(t, chunk, 1), t))
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
    denominators = [
        P_sum[f1 - 1, f2 - 1] * S_sum[f1 + f2 - 1]
        for f1 in range(1, max_freq + 1)
        for f2 in range(1, max_freq - f1 + 1)
    ]
    energy_floor = np.finfo(float).eps * max(denominators, default=0.0)
    vals: list[float] = []
    for f1 in range(1, max_freq + 1):
        for f2 in range(1, max_freq - f1 + 1):
            denom = P_sum[f1 - 1, f2 - 1] * S_sum[f1 + f2 - 1]
            if denom <= energy_floor:
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
    if not np.isfinite(v).all() or v.size == 0:
        return np.nan
    magnitude = float(np.max(np.abs(v)))
    if magnitude == 0.0:
        return np.nan
    # The excess is dimensionless too. Normalize before surrogate FFTs so
    # their intermediate spectrum cannot overflow solely from physical units.
    v = v / magnitude
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
def _safe_train_standardize(
    train: np.ndarray, test: np.ndarray, *, preserve_constant_test: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Train-only affine coordinates without absolute-scale thresholds."""
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    was_vector = train.ndim == 1
    train_2d = train.reshape(-1, 1) if was_vector else train
    test_2d = test.reshape(-1, 1) if was_vector else test
    train_out = np.empty_like(train_2d)
    test_out = np.empty_like(test_2d)
    block_is_constant = bool(np.all(train_2d == train_2d[0]))
    for column in range(train_2d.shape[1]):
        tr = train_2d[:, column]
        te = test_2d[:, column]
        origin = tr[0]
        with np.errstate(over="ignore", invalid="ignore"):
            tr_dev = tr - origin
        # Coordinate choice is a training-only decision.  A hostile holdout may
        # fail closed later, but can never change fitted training geometry.
        if not np.all(np.isfinite(tr_dev)):
            magnitude = float(np.max(np.abs(tr)))
            if not np.isfinite(magnitude) or magnitude == 0.0:
                train_out[:, column] = 0.0
                test_out[:, column] = 0.0
                continue
            bounded_tr = tr / magnitude
            bounded_te = te / magnitude
            tr_dev = bounded_tr - bounded_tr[0]
            te_dev = bounded_te - bounded_tr[0]
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                te_dev = te - origin
        scale = float(np.max(np.abs(tr_dev)))
        if not np.isfinite(scale) or scale == 0.0:
            train_out[:, column] = 0.0
            if preserve_constant_test:
                level_scale = abs(float(origin))
                if not np.isfinite(level_scale) or level_scale == 0.0:
                    level_scale = 1.0
                test_out[:, column] = te_dev / level_scale
            elif block_is_constant:
                # Every training vector is identical, so the centered train and
                # cross kernels are zero regardless of the holdout coordinates.
                test_out[:, column] = 0.0
            else:
                # A moved holdout value in only one train-constant coordinate
                # changes a mixed-dimensional RBF cross kernel, but training
                # data provide no unit for that extrapolation.  Fail closed.
                test_out[:, column] = np.where(te == origin, 0.0, np.nan)
            continue
        tr_scaled = tr_dev / scale
        with np.errstate(over="ignore", invalid="ignore"):
            te_scaled = te_dev / scale
        mean = float(np.mean(tr_scaled))
        sd = float(np.std(tr_scaled))
        if not np.isfinite(sd) or sd == 0.0:
            train_out[:, column] = 0.0
            test_out[:, column] = 0.0
            continue
        train_out[:, column] = (tr_scaled - mean) / sd
        test_out[:, column] = (te_scaled - mean) / sd
    if was_vector:
        return train_out[:, 0], test_out[:, 0]
    return train_out, test_out


def _center_train_test_kernel(
    K_tr: np.ndarray, K_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Center a train Gram and test-to-train Gram using training geometry."""
    train_col_mean = K_tr.mean(axis=0, keepdims=True)
    train_mean = float(K_tr.mean())
    K_tr_c = K_tr - train_col_mean - train_col_mean.T + train_mean
    K_te_c = K_te - K_te.mean(axis=1, keepdims=True) - train_col_mean + train_mean
    return K_tr_c, K_te_c


def _kernel_ridge_with_intercept(
    K_tr_c: np.ndarray,
    target: np.ndarray,
    K_te_c: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Kernel ridge prediction with an unpenalized, train-only intercept."""
    target_mean = np.mean(target, axis=0)
    centered_target = target - target_mean
    alpha = np.linalg.solve(K_tr_c + lam * np.eye(K_tr_c.shape[0]), centered_target)
    return target_mean + K_te_c @ alpha


def _kernel_granger_score(y: np.ndarray, x: np.ndarray, lag: int) -> float:
    """BLOCKED_HISTORICAL_EVALUATION: contiguous train/test blocks, scaler and
    RBF bandwidths fitted on the TRAINING block only (never the test block), so
    the output is a predictive-diagnostic score, not a next-bar forecast."""
    n = y.shape[0]
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
        return np.nan
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
    Yl_tr, Yl_te = _safe_train_standardize(Ylags[:train_n], Ylags[train_n:])
    Xl_tr, Xl_te = _safe_train_standardize(Xlags[:train_n], Xlags[train_n:])
    target_is_constant = bool(np.all(Ytr == Ytr[0]))
    constant_target_holdout_moves = target_is_constant and bool(np.any(Yte != Ytr[0]))
    Ytr, Yte = _safe_train_standardize(Ytr, Yte, preserve_constant_test=True)
    if target_is_constant:
        # Both fitted models reduce exactly to the same unpenalized intercept;
        # unequal finite holdout values give equal positive losses.  If the
        # holdout is also constant, both losses are zero and log(0/0) is undefined.
        return 0.0 if constant_target_holdout_moves else np.nan

    # Both blocked models use training-only geometry and the same numeric ridge
    # coefficient.  They are nevertheless different fixed RKHS models, so the
    # resulting OOS loss ratio is a modeled predictive diagnostic rather than a
    # nested likelihood test or a pure causal-information measure.
    def _median_bandwidth(tr: np.ndarray) -> float:
        # median pairwise distance over the TRAINING block only (never the
        # test block, so no OOS contamination).
        distance = np.zeros((tr.shape[0], tr.shape[0]), dtype=float)
        for column in range(tr.shape[1]):
            delta = tr[:, column, None] - tr[None, :, column]
            distance = np.hypot(distance, delta)
        sigma = float(np.median(distance))
        if not np.isfinite(sigma) or sigma == 0.0:
            sigma = 1.0
        return sigma

    # P1-11 (fairness): the X kernel's smoothness is set by X's OWN distance
    # geometry.  The old code reused ``sigma_y`` for Kx, so X's kernel width
    # depended on Y's scale/structure even though X is separately standardised.
    sigma_y = _median_bandwidth(Yl_tr)
    sigma_x = _median_bandwidth(Xl_tr)

    Kr, Kte_r = _center_train_test_kernel(
        _rbf(Yl_tr, Yl_tr, sigma_y), _rbf(Yl_te, Yl_tr, sigma_y)
    )
    ntr = Kr.shape[0]
    trace_r = float(np.trace(Kr))
    reference_trace = trace_r if trace_r > _EPS else float(ntr)
    lambda_shared = 1e-3 * reference_trace / ntr
    pred_r = _kernel_ridge_with_intercept(Kr, Ytr, Kte_r, lambda_shared)
    mse_r = float(np.mean((Yte - pred_r) ** 2))

    # The full model is a separately fixed additive-kernel model.  Center and
    # trace-normalize the nonconstant X component on training data, then retain
    # the historical eta=.5 mixture scale.  A constant X has zero centered
    # trace and is an exact no-op.  This compares two blocked OOS models; it is
    # not a nested-hypothesis test or proof that X carries causal information.
    Kx_tr, Kte_x = _center_train_test_kernel(
        _rbf(Xl_tr, Xl_tr, sigma_x), _rbf(Xl_te, Xl_tr, sigma_x)
    )
    eta = 0.5
    trace_x = float(np.trace(Kx_tr))
    if trace_x <= _EPS:
        Kf, Kte_f = Kr, Kte_r
    else:
        component_scale = reference_trace / trace_x
        Kf = (Kr + eta * component_scale * Kx_tr) / (1.0 + eta)
        Kte_f = (Kte_r + eta * component_scale * Kte_x) / (1.0 + eta)
    pred_f = _kernel_ridge_with_intercept(Kf, Ytr, Kte_f, lambda_shared)
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
    # Response/residual geometry belongs to this complete retrospective window;
    # affine normalization makes HSIC unit-safe without reaching beyond it.
    x, _ = _safe_train_standardize(x, x)
    y, _ = _safe_train_standardize(y, y)
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
        ztr, zte = _safe_train_standardize(z[tr_idx], z[te_idx])
        ztr = ztr.reshape(-1, 1)
        zte = zte.reshape(-1, 1)
        sigma = float(np.median(np.abs(ztr[:, None, 0] - ztr[None, :, 0])))
        if not np.isfinite(sigma) or sigma == 0.0:
            sigma = 1.0
        Kzz, Kzte = _center_train_test_kernel(
            _rbf(ztr, ztr, sigma), _rbf(zte, ztr, sigma)
        )
        trace_z = float(np.trace(Kzz))
        lam = 1e-3 * (trace_z / tr_idx.size if trace_z > _EPS else 1.0)
        predictions = _kernel_ridge_with_intercept(
            Kzz, np.column_stack((x[tr_idx], y[tr_idx])), Kzte, lam
        )
        rx_all[te_idx] = x[te_idx] - predictions[:, 0]
        ry_all[te_idx] = y[te_idx] - predictions[:, 1]
    fin = np.isfinite(rx_all) & np.isfinite(ry_all)
    rx = rx_all[fin]
    ry = ry_all[fin]
    m = rx.shape[0]
    if m < 12:
        return np.nan

    rx, _ = _safe_train_standardize(rx, rx)
    ry, _ = _safe_train_standardize(ry, ry)

    sigma_x = float(np.median(np.abs(rx[:, None] - rx[None, :])))
    sigma_y = float(np.median(np.abs(ry[:, None] - ry[None, :])))
    if not np.isfinite(sigma_x) or sigma_x == 0.0:
        sigma_x = 1.0
    if not np.isfinite(sigma_y) or sigma_y == 0.0:
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
    # For these symmetric kernels, double centering and a Frobenius product
    # equal the trace above without any dense centering-matrix multiplies.
    Kxc = Kx - Kx.mean(axis=0, keepdims=True) - Kx.mean(axis=1, keepdims=True) + Kx.mean()
    Kyc = Ky - Ky.mean(axis=0, keepdims=True) - Ky.mean(axis=1, keepdims=True) + Ky.mean()
    hsic = float(np.sum(Kxc * Kyc)) / float((m - 1) * (m - 1))
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
def _bds_common_center_probability(v: np.ndarray, eps: float) -> float:
    """Probability that two distinct points neighbor one common center."""
    n = v.shape[0]
    if n < 3:
        return np.nan
    degree = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(v[i] - v[j]) < eps:
                degree[i] += 1
                degree[j] += 1
    return float(np.sum(degree * (degree - 1))) / float(n * (n - 1) * (n - 2))


def _bds_statistic(v: np.ndarray, m: int, distance_multiplier: float) -> float:
    n = v.shape[0]
    if n < 30 or m < 1 or m >= n:
        return np.nan
    # Preserve every representable original-unit deviation.  Only fall back to
    # bounded coordinates when subtracting opposite-sign extremes overflows.
    with np.errstate(over="ignore", invalid="ignore"):
        deviations = v - v[0]
    if not np.all(np.isfinite(deviations)):
        magnitude = float(np.max(np.abs(v)))
        if not np.isfinite(magnitude) or magnitude == 0.0:
            return np.nan
        bounded = v / magnitude
        deviations = bounded - bounded[0]
    scale = float(np.max(np.abs(deviations)))
    if not np.isfinite(scale) or scale == 0.0:
        return np.nan
    v = deviations / scale
    eps = float(distance_multiplier) * float(np.std(v))
    if not np.isfinite(eps) or eps <= 0.0:
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

    # Standard conditioned BDS convention: the effect compares C_m on N_m
    # embedded vectors with C_1 on the matching suffix v[m-1:].  The asymptotic
    # variance remains defined from the full one-dimensional indicator matrix.
    N_m = n - m + 1
    c1_variance = _c_integral(1)
    suffix = v[m - 1 :]
    suffix_count = 0
    for i in range(N_m):
        for j in range(i + 1, N_m):
            if abs(suffix[i] - suffix[j]) < eps:
                suffix_count += 1
    c1_effect = 2.0 * suffix_count / (N_m * (N_m - 1))
    cm = _c_integral(m)
    if c1_variance <= _EPS or c1_effect <= _EPS or cm < 0.0:
        return np.nan
    K = _bds_common_center_probability(v, eps)
    if K <= _EPS:
        return np.nan
    sigma2 = 4.0 * (
        K ** m
        + 2.0 * sum(c1_variance ** (2 * j) * K ** (m - j) for j in range(1, m))
        + (m - 1) ** 2 * c1_variance ** (2 * m)
        - m ** 2 * K * c1_variance ** (2 * m - 2)
    )
    if sigma2 <= _EPS:
        return np.nan
    return float(np.sqrt(N_m) * (cm - c1_effect ** m) / np.sqrt(sigma2))


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
            _BICOH_SPEC,
            # R22-041: n_surrogates is a null-estimator knob — fixed or a very
            # small coarse grid, never a large-scale alpha-parameter search.
            n_surrogates=ParamSpec(dtype=int, min=1, param_role=ParamRole.ESTIMATOR_RESOLUTION),
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
        "relational_specs": [
            RelationalParamSpec(
                "purge_gap <= window // 2 - 6",
                "HSIC purge_gap must leave at least six test rows in both folds",
            ),
        ],
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
    from factor_engine.cleaned_operators.registry import OperatorRegistry

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
