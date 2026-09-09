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


def batched_factor_loadings(factor_values, risk_factors, intercept=True, min_obs=10, *, weights=None, standardized=False, return_diagnostics=False):
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
    if isinstance(min_obs,(bool,np.bool_)) or not isinstance(min_obs,(int,np.integer)) or min_obs<2:
        raise ValueError("min_obs must be an integer >=2")
    w = cp.ones((T,N)) if weights is None else cp.asarray(weights,dtype=cp.float64)
    if w.shape!=(T,N) or bool(cp.any(cp.isfinite(w)&(w<0))):
        raise ValueError("weights must be nonnegative (T,N)")
    num_coefs = K + 1 if intercept else K
    loadings = cp.full((T, F, num_coefs), cp.nan, dtype=cp.float64)
    r_squared = cp.full((T, F), cp.nan, dtype=cp.float64)
    residuals = cp.full((T, F, N), cp.nan, dtype=cp.float64)
    evidence={"raw_loadings":cp.full_like(loadings,cp.nan),
              "counts":cp.zeros((T,F),dtype=cp.int64),
              "rank":cp.zeros((T,F),dtype=cp.int64),
              "effective_df":cp.zeros((T,F),dtype=cp.int64)}

    for t in range(T):
        X = rf[t]  # (N, K)
        X_fin = cp.isfinite(X)  # (N, K)
        all_fin = cp.all(X_fin, axis=1)  # (N,)
        y = x[t]  # (F, N)
        y_fin = cp.isfinite(y)  # (F, N)
        valid = all_fin[None, :] & y_fin & cp.isfinite(w[t])[None,:] & (w[t]>0)[None,:]
        n_valid = cp.sum(valid, axis=1)  # (F,)
        evidence["counts"][t]=n_valid
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
            yv = y[f, row_mask]  # (n,)
            ww=w[t,row_mask]; ww=ww/cp.max(ww); ww=ww/ww.sum()
            xo=Xv[0]+cp.sum(ww[:,None]*(Xv-Xv[0]),axis=0) if intercept else cp.zeros(K)
            yo=yv[0]+cp.sum(ww*(yv-yv[0])) if intercept else 0.
            xc=Xv-xo; yc=yv-yo
            xs=cp.sqrt(cp.sum(ww[:,None]*xc*xc,axis=0)); xs=cp.where(xs>0,xs,1.)
            ys=cp.sqrt(cp.sum(ww*yc*yc)); ys=cp.where(ys>0,ys,1.)
            design=xc/xs
            if intercept:
                design=cp.column_stack((cp.ones(len(yv)),design))
            rootw=cp.sqrt(ww)
            u,singular,vh=cp.linalg.svd(design*rootw[:,None],full_matrices=False)
            rank=int(cp.sum(singular>cp.finfo(cp.float64).eps*max(design.shape)*singular[0]))
            coordinates=u[:,:rank].T@(yc/ys*rootw)
            beta=vh[:rank].T@(coordinates/singular[:rank])
            evidence["rank"][t,f]=rank
            evidence["effective_df"][t,f]=len(yv)-rank
            if len(yv)-rank<2: continue
            predicted=(u[:,:rank]@coordinates)*ys/rootw
            resid=yc-predicted
            condition=singular[0]/singular[rank-1] if rank else cp.inf
            tolerance=64*cp.finfo(cp.float64).eps*max(len(yv),num_coefs)*cp.maximum(cp.maximum(cp.linalg.norm(yc),cp.linalg.norm(predicted)),cp.finfo(cp.float64).tiny)
            if bool(cp.linalg.norm(resid)<=tolerance): resid=cp.zeros_like(resid)
            ss_tot=cp.sum(ww*(yv-(yv[0]+cp.sum(ww*(yv-yv[0]))))**2)
            if bool(ss_tot>0) and rank>int(intercept):
                r_squared[t,f]=1.-cp.sum(ww*resid**2)/ss_tot
            if rank==num_coefs:
                slopes=beta[int(intercept):]*ys/xs
                raw=cp.concatenate((cp.atleast_1d(yo+beta[0]*ys-xo@slopes),slopes)) if intercept else slopes
                evidence["raw_loadings"][t,f]=raw
                if standardized:
                    sx=cp.sqrt(cp.sum(ww[:,None]*xc*xc,axis=0))
                    raw[int(intercept):]=cp.where((sx>0)&cp.isfinite(r_squared[t,f]),slopes*sx/cp.sqrt(ss_tot),cp.nan)
                loadings[t,f]=raw
            residuals[t,f,row_mask]=resid
    result=(loadings,r_squared,residuals)
    return (*result,evidence) if return_diagnostics else result


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
