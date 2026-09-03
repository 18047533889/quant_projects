"""GPU temporal-stability kernels (spec §20, Wave 4).

Batch-vectorized over F (NO ``for f in range(F)``).  All inputs are daily IC
series of shape (T, F) — small relative to raw factor tensors — so the point
is cross-factor batch vectorization, not per-factor Python loops.

All estimators preserve the true-time-axis policy of the CPU reference in
:mod:`quant_evaluator.metrics.temporal`: lag-based estimators use
pairwise-finite lag alignment on the *original, uncompressed* calendar axis.
For lag ``k`` only original positions ``(t, t+k)`` where BOTH are finite
contribute.  Values are never NaN-compressed before lagging, so calendar gaps
never fabricate adjacent pairs.  The batch-vectorized form computes each lag's
pair-specific sums directly from the original arrays with a per-(t,f) boolean
mask, matching the CPU pair samples exactly.

CuPy is imported lazily so the module (and any importers) load on a
CPU-only environment.
"""

from __future__ import annotations

import numpy as np


def _import_cp():
    import cupy as cp
    return cp


def ic_autocorrelation_series(ic_series, max_lag=20, min_obs=30):
    """Batch-vectorized IC autocorrelation (ACF), parity with CPU
    ``compute_ic_autocorrelation`` / ``compute_autocorrelation``.

    Returns an array of shape (max_lag + 1, F).  ``acf[0]`` is 1.0 for factors
    with >= min_obs finite values; higher lags use pairwise-finite alignment on
    the original (uncompressed) calendar axis.  Each lag uses pair-specific
    means over the pairwise-finite sample, exactly as the CPU oracle.
    """
    cp = _import_cp()
    ic = cp.asarray(ic_series)
    if ic.ndim == 1:
        ic = ic[:, None]
    T, F = ic.shape
    acf = cp.full((max_lag + 1, F), cp.nan, dtype=cp.float64)

    finite = cp.isfinite(ic)
    n_f = cp.sum(finite, axis=0)  # (F,)
    acf[0, :] = cp.where(n_f >= min_obs, 1.0, cp.nan)

    for lag in range(1, max_lag + 1):
        if lag >= T:
            break
        # pairwise-finite pairs on the ORIGINAL calendar (t, t+lag)
        pair_mask = finite[lag:, :] & finite[:-lag, :]  # (T-lag, F)
        n_pairs = cp.sum(pair_mask, axis=0)             # (F,)
        pair_ok = n_pairs >= min_obs
        x_prev = cp.where(pair_mask, ic[:-lag, :], 0.0)  # x_t
        x_curr = cp.where(pair_mask, ic[lag:, :], 0.0)   # x_{t+lag}
        cnt = n_pairs.astype(cp.float64)
        s_prev = cp.sum(x_prev, axis=0)
        s_curr = cp.sum(x_curr, axis=0)
        s_pp = cp.sum(x_prev * x_prev, axis=0)
        s_cc = cp.sum(x_curr * x_curr, axis=0)
        s_pc = cp.sum(x_prev * x_curr, axis=0)
        num = cnt * s_pc - s_prev * s_curr
        denom = cp.sqrt(
            (cnt * s_pp - s_prev * s_prev) * (cnt * s_cc - s_curr * s_curr)
        )
        corr = num / denom
        corr = cp.where((denom <= 0) | (~cp.isfinite(denom)), cp.nan, corr)
        corr = cp.where(pair_ok, corr, cp.nan)
        acf[lag, :] = corr

    return acf.get()


def half_life(ic_series, min_periods=60):
    """Batch-vectorized AR(1) IC half-life.

    phi_f = sum(y * x) / sum(x * x) with x = ic[t], y = ic[t+1] over
    pairwise-finite adjacent pairs on the original calendar.  Returns shape
    (F,), NaN where n_finite < min_periods, too few pairs, denominator<=0,
    or phi outside (0, 1).
    """
    cp = _import_cp()
    ic = cp.asarray(ic_series)
    if ic.ndim == 1:
        ic = ic[:, None]
    T, F = ic.shape
    half = cp.full(F, cp.nan, dtype=cp.float64)
    if T < 2:
        return half.get()

    finite = cp.isfinite(ic)
    n_f = cp.sum(finite, axis=0)
    pair_mask = finite[1:, :] & finite[:-1, :]   # (T-1, F)
    n_pairs = cp.sum(pair_mask, axis=0)
    enough = (n_f >= min_periods) & (n_pairs >= min_periods - 1)

    x = cp.where(pair_mask, ic[:-1, :], 0.0)     # x_{t-1}
    y = cp.where(pair_mask, ic[1:, :], 0.0)      # x_t
    s_xy = cp.sum(x * y, axis=0)
    s_xx = cp.sum(x * x, axis=0)
    # divide by the true sum(x*x) (guard only against zero/negative denom)
    denom = cp.where(s_xx > 0, s_xx, 1.0)
    phi = s_xy / denom
    phi = cp.where((s_xx <= 0) | (~cp.isfinite(phi)), cp.nan, phi)
    in_range = (phi > 0) & (phi < 1)
    hl = -float(np.log(2.0)) / cp.log(phi)
    hl = cp.where(in_range & enough, hl, cp.nan)
    return hl.get()
