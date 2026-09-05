"""
GPU exposure / purity parity kernels (R61-FI-024).

Thin CuPy-lazy wrappers over the R60 batch kernels in
``quant_evaluator.kernels.gpu.exposure`` for the per-style exposure evidence
family.  When CuPy is absent every function fails closed with
:class:`OptionalDependencyMissing` — never silently falls back to CPU labels
under a GPU name.

Parity counterpart: the CPU kernels in ``metrics/exposure_evidence.py`` /
``metrics/exposure.py``; ``tests/test_exposure_evidence.py`` runs the
CPU-vs-GPU parity test on a CUDA host.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from quant_evaluator.contracts.errors import OptionalDependencyMissing

__all__ = [
    "gpu_style_exposure",
    "gpu_factor_loadings",
    "gpu_concentration_hhi",
    "gpu_sector_exposure",
    "gpu_neutralized_rank_ic",
]


def _import_cp():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - host without CuPy
        raise OptionalDependencyMissing(
            "cupy", "GPU exposure kernels require CuPy on a CUDA host"
        ) from exc
    return cp


def gpu_style_exposure(factor_values, style_factors, style_names):
    """(T, N) factor + (T, N, K) style panel -> (K,) mean signed loadings."""
    _import_cp()
    from quant_evaluator.kernels.gpu.exposure import batched_style_exposure

    return batched_style_exposure(factor_values, style_factors, style_names)


def gpu_factor_loadings(factor_values, risk_factors, intercept=True, min_obs=10):
    """(T, N) factor + (T, N, K) risk panel -> (loadings, r2, residuals)."""
    _import_cp()
    from quant_evaluator.kernels.gpu.exposure import batched_factor_loadings

    return batched_factor_loadings(
        factor_values, risk_factors, intercept=intercept, min_obs=min_obs
    )


def gpu_concentration_hhi(factor_values, weights=None):
    """(T, N) or (T, F, N) factor -> (T, F) HHI concentration."""
    _import_cp()
    from quant_evaluator.kernels.gpu.exposure import batched_concentration_hhi

    return batched_concentration_hhi(factor_values, weights)


def gpu_sector_exposure(factor_values, sector_labels, weights=None):
    """(T, N) factor + (N,) sector labels -> (exposure, counts)."""
    _import_cp()
    from quant_evaluator.kernels.gpu.exposure import batched_sector_exposure

    return batched_sector_exposure(factor_values, sector_labels, weights)


def gpu_neutralized_rank_ic(
    factor_values,
    forward_returns,
    exposure_panel,
    min_obs: int = 10,
):
    """GPU neutralized rank IC.

    Per date the OLS residuals come from the R60 GPU loading kernel
    (``kernels/gpu.exposure.batched_factor_loadings``); the per-date
    Spearman residual-vs-forward IC is computed on-device (average-tie rank
    Pearson).  CPU reference (authoritative):
    ``metrics/exposure_evidence.compute_neutralized_rank_ic``; parity is
    asserted in ``test_exposure_evidence.py`` on a CUDA host.
    """
    cp = _import_cp()
    from quant_evaluator.kernels.gpu.exposure import batched_factor_loadings

    fv = cp.asarray(factor_values, dtype=cp.float64)  # (T, N)
    fwd = cp.asarray(forward_returns, dtype=cp.float64)
    panel = cp.asarray(exposure_panel, dtype=cp.float64)  # (T, N, K)
    if fv.ndim != 2 or fwd.ndim != 2 or panel.ndim != 3:
        raise ValueError(
            "factor_values (T,N), forward_returns (T,N), exposure_panel (T,N,K) required"
        )
    _, _, residuals = batched_factor_loadings(
        fv, panel, intercept=True, min_obs=min_obs
    )
    # batched loading kernel treats the (T,N) factor as (T,1,N) -> residuals (T,1,N).
    residuals = residuals[:, 0, :]  # (T, N)
    daily: list[float] = []
    for t in range(fv.shape[0]):
        resid = residuals[t]  # (N,) on device
        y = fwd[t]  # (N,) on device
        valid = cp.isfinite(resid) & cp.isfinite(y)
        if int(cp.sum(valid)) < 2:
            continue
        r = resid[valid]
        yy = y[valid]
        r_rank = cp.argsort(cp.argsort(r)).astype(cp.float64)
        y_rank = cp.argsort(cp.argsort(yy)).astype(cp.float64)
        r_rank = r_rank - cp.mean(r_rank)
        y_rank = y_rank - cp.mean(y_rank)
        denom = cp.sqrt(cp.sum(r_rank ** 2) * cp.sum(y_rank ** 2))
        if float(denom) <= 1e-12:
            continue
        daily.append(float(cp.sum(r_rank * y_rank) / denom))
    if not daily:
        return np.nan
    return float(np.mean(daily))