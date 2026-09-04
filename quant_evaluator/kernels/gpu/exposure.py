"""GPU exposure / purity kernels (spec §24/§25).

Batch-vectorized over the factor axis F on the GPU-friendly (T, F, N)
layout, matching the CPU reference in :mod:`quant_evaluator.metrics.exposure`
exactly:

  - ``batched_concentration_hhi``  -> ``compute_concentration_hhi``
  - ``batched_sector_exposure``    -> ``compute_sector_exposure``
  - ``batched_factor_loadings``    -> ``compute_factor_loadings``
  - ``batched_style_exposure``     -> ``compute_style_exposure``

Semantics preserved:
  - HHI is computed on GROSS (absolute) exposure, invariant to sign flips.
  - Sector exposure is a weighted mean of factor values per sector over the
    finite, positive-weight assets.
  - Factor loadings are cross-sectional OLS betas (with optional intercept);
    singular / insufficient periods are skipped (NaN), matching the CPU
    ``np.linalg.LinAlgError`` guard.

CuPy is imported lazily so the module loads on a CPU-only environment.
"""

from __future__ import annotations

import numpy as np


def _import_cp():
    import cupy as cp
    return cp


def _as_tfn(x):
    """Coerce to (T, F, N) float64 device array."""
    cp = _import_cp()
    a = cp.asarray(x, dtype=cp.float64)
    if a.ndim == 2:
        a = a[:, None, :]
    if a.ndim != 3:
        raise ValueError(f"expected (T, F, N) or (T, N), got ndim={a.ndim}")
    return a


def batched_concentration_hhi(factor_values, weights=None):
    """Herfindahl-Hirschman Index on gross exposure, (T, F).

    Matches ``compute_concentration_hhi``: per (t, f) the weights are
    normalized to sum 1 over the finite positive-weight assets, then
    ``share_i = |w_i * f_i| / sum_j |w_j * f_j|`` and ``HHI = sum share^2``.
    NaN where no valid assets or zero total gross exposure.
    """
    cp = _import_cp()
    x = _as_tfn(factor_values)
    T, F, N = x.shape
    if weights is None:
        w = cp.ones((T, 1, N), dtype=cp.float64)
    else:
        w = cp.asarray(weights, dtype=cp.float64)
        if w.ndim == 2:
            w = w[:, None, :]
    valid = cp.isfinite(x) & cp.isfinite(w) & (w > 0)  # (T, F, N)
    w0 = cp.where(valid, w, 0.0)
    wsum = cp.sum(w0, axis=2)  # (T, F)
    w_norm = w0 / cp.where((wsum > 0)[:, :, None], wsum[:, :, None], 1.0)
    gross = cp.abs(w_norm * cp.where(valid, x, 0.0))
    total = cp.sum(gross, axis=2)  # (T, F)
    shares = gross / cp.where((total > 0)[:, :, None], total[:, :, None], 1.0)
    hhi = cp.sum(shares * shares, axis=2)  # (T, F)
    any_valid = cp.sum(valid, axis=2) > 0
    hhi = cp.where(any_valid & (total > 0), hhi, cp.nan)
    return hhi


def batched_sector_exposure(factor_values, sector_labels, weights=None):
    """Weighted mean factor value per sector, (T, F, S) and counts (T, F, S).

    Matches ``compute_sector_exposure`` (static PIT labels, equal-weight
    default).  ``sector_labels`` is (N,) integer sector codes; NaN labels are
    excluded.  Returns (exposure, counts) on device.
    """
    cp = _import_cp()
    x = _as_tfn(factor_values)
    T, F, N = x.shape
    labels = cp.asarray(sector_labels, dtype=cp.float64)
    if labels.ndim != 1 or labels.shape[0] != N:
        raise ValueError(f"sector_labels must be (N,) == {(N,)}, got {labels.shape}")
    if weights is None:
        w = cp.ones((T, 1, N), dtype=cp.float64)
    else:
        w = cp.asarray(weights, dtype=cp.float64)
        if w.ndim == 2:
            w = w[:, None, :]

    uniq = cp.unique(labels[~cp.isnan(labels)])
    S = int(uniq.shape[0])
    exposure = cp.full((T, F, S), cp.nan, dtype=cp.float64)
    counts = cp.zeros((T, F, S), dtype=cp.int32)
    for s in range(S):
        sec = uniq[s]
        mask = (labels == sec)[None, None, :]  # (1, 1, N)
        valid = mask & cp.isfinite(x) & cp.isfinite(w) & (w > 0)  # (T, F, N)
        w0 = cp.where(valid, w, 0.0)
        x0 = cp.where(valid, x, 0.0)
        den = cp.sum(w0, axis=2)  # (T, F)
        num = cp.sum(x0 * w0, axis=2)  # (T, F)
        cnt = cp.sum(valid, axis=2)  # (T, F)
        mean = num / cp.maximum(den, 1.0)
        exposure[:, :, s] = cp.where(den > 0, mean, cp.nan)
        counts[:, :, s] = cnt.astype(cp.int32)
    return exposure, counts


def batched_factor_loadings(factor_values, risk_factors, intercept=True, min_obs=10):
    """Cross-sectional OLS factor loadings, (T, K+1) / (T, K) + R2 + residuals.

    Matches ``compute_factor_loadings``.  ``factor_values`` is (T, F, N) and
    ``risk_factors`` is (T, N, K) shared across factors.  Returns
    (loadings (T, F, K+1 or K), r_squared (T, F), residuals (T, F, N)).
    Singular / insufficient periods are NaN.
    """
    cp = _import_cp()
    x = _as_tfn(factor_values)
    T, F, N = x.shape
    rf = cp.asarray(risk_factors, dtype=cp.float64)
    if rf.ndim != 3 or rf.shape[0] != T or rf.shape[1] != N:
        raise ValueError(f"risk_factors must be (T, N, K) == {(T, N, 'K')}, got {rf.shape}")
    K = rf.shape[2]
    num_coefs = K + 1 if intercept else K
    loadings = cp.full((T, F, num_coefs), cp.nan, dtype=cp.float64)
    r_squared = cp.full((T, F), cp.nan, dtype=cp.float64)
    residuals = cp.full((T, F, N), cp.nan, dtype=cp.float64)

    for t in range(T):
        X = rf[t]  # (N, K)
        X_fin = cp.isfinite(X)  # (N, K)
        all_fin = cp.all(X_fin, axis=1)  # (N,)
        y = x[t]  # (F, N)
        y_fin = cp.isfinite(y)  # (F, N)
        valid = all_fin[None, :] & y_fin  # (F, N)
        n_valid = cp.sum(valid, axis=1)  # (F,)
        ok = n_valid >= min_obs
        if not bool(cp.any(ok)):
            continue
        # per-factor OLS (design matrix differs per factor because the valid
        # row set depends on that factor's y finiteness)
        for f in range(F):
            if not bool(ok[f]):
                continue
            row_mask = valid[f]  # (N,)
            Xv = X[row_mask]  # (n, K)
            if intercept:
                Xv = cp.column_stack([cp.ones(Xv.shape[0]), Xv])  # (n, C)
            yv = y[f, row_mask]  # (n,)
            XtX = Xv.T @ Xv  # (C, C)
            Xty = Xv.T @ yv  # (C,)
            try:
                beta = cp.linalg.solve(XtX, Xty)  # (C,)
            except Exception:
                continue
            if not bool(cp.all(cp.isfinite(beta))):
                continue
            loadings[t, f] = beta
            y_pred = Xv @ beta  # (n,)
            ss_res = cp.sum((yv - y_pred) ** 2)
            y_mean = cp.mean(yv)
            ss_tot = cp.sum((yv - y_mean) ** 2)
            if ss_tot > 0:
                r_squared[t, f] = 1.0 - ss_res / ss_tot
            # residuals over all N (NaN where invalid)
            Xfull = X
            if intercept:
                Xfull = cp.column_stack([cp.ones(N), Xfull])
            resid = y[f] - Xfull @ beta  # (N,)
            resid = cp.where(row_mask, resid, cp.nan)
            residuals[t, f] = resid
    return loadings, r_squared, residuals


def batched_style_exposure(factor_values, style_factors, style_names):
    """Time-averaged style loadings, (K,).

    Matches ``compute_style_exposure``: regresses each factor on the style
    factors (with intercept) and averages the non-intercept loadings over time.
    """
    cp = _import_cp()
    x = _as_tfn(factor_values)
    T, F, N = x.shape
    sf = cp.asarray(style_factors, dtype=cp.float64)
    K = sf.shape[2]
    if K != len(style_names):
        raise ValueError(
            f"style_factors has {K} factors but {len(style_names)} names provided"
        )
    loadings, _, _ = batched_factor_loadings(x, sf, intercept=True, min_obs=10)
    # loadings (T, F, K+1); skip intercept (first column)
    factor_loadings = loadings[:, :, 1:]  # (T, F, K)
    mean_exposures = cp.nanmean(factor_loadings, axis=0)  # (F, K)
    return mean_exposures
