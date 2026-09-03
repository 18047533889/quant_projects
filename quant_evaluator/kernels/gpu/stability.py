"""GPU rank stability (spec §12.4, Wave 2).

Matches CPU ``compute_rank_stability`` (metrics/temporal.py): Spearman
correlation between adjacent-day cross-sections over the pairwise-finite
set, min 10 valid, constant rejection.  Returns (T-1, F) series.
"""

from __future__ import annotations

from quant_evaluator.kernels.gpu.rank import batched_rank, batched_distinct_level_count


def _import_cp():
    import cupy as cp
    return cp


def batched_rank_stability(factor_values, lag: int = 1, min_obs: int = 10):
    """Compute rank stability over (T, F, N) factors.

    Returns (T-lag, F) Spearman stability series (NaN where insufficient or
    constant), matching CPU ``compute_rank_stability``.
    """
    cp = _import_cp()
    x = cp.asarray(factor_values)
    T, F, N = x.shape
    if lag >= T:
        raise ValueError(f"lag ({lag}) must be less than T ({T})")
    out = cp.full((T - lag, F), cp.nan, dtype=cp.float64)
    for t in range(T - lag):
        a = x[t]
        b = x[t + lag]
        finite = cp.isfinite(a) & cp.isfinite(b)
        # rank over the PAIRWISE-FINITE set (CPU spearmanr ranks these)
        a_rank = batched_rank(cp.where(finite, a, cp.nan), pct=False)
        b_rank = batched_rank(cp.where(finite, b, cp.nan), pct=False)
        n = cp.sum(finite, axis=1).astype(cp.float64)  # (F,)
        a0 = cp.where(finite, a_rank, 0.0)
        b0 = cp.where(finite, b_rank, 0.0)
        sa = cp.sum(a0, axis=1)
        sb = cp.sum(b0, axis=1)
        saa = cp.sum(a0 * a0, axis=1)
        sbb = cp.sum(b0 * b0, axis=1)
        sab = cp.sum(a0 * b0, axis=1)
        num = n * sab - sa * sb
        den = (n * saa - sa * sa) * (n * sbb - sb * sb)
        corr = num / cp.sqrt(cp.maximum(den, 0.0))
        # constant rejection: distinct levels of the pairwise-finite set
        dl = batched_distinct_level_count(cp.where(finite, a, cp.nan))  # (F,)
        dl_lag = batched_distinct_level_count(cp.where(finite, b, cp.nan))  # (F,)
        bad = (n < min_obs) | (den <= 0) | (dl < 2) | (dl_lag < 2)
        corr = cp.where(bad, cp.nan, corr)
        out[t] = corr
    return out  # (T-lag, F)
