"""GPU quantile returns (spec §12.3, Wave 2).

Reuses the shared batched quantile assignment; computes per-bucket mean
daily return over the pairwise-finite set with a min_assets floor, matching
CPU ``compute_quantile_returns`` (metrics/quantile.py) which returns
``(T, n_quantiles, F)``.
"""

from __future__ import annotations

from quant_evaluator.kernels.gpu.rank import batched_quantile_assignment


def _import_cp():
    import cupy as cp
    return cp


def batched_quantile_returns(
    factor_values,
    daily_returns,
    n_quantiles: int = 10,
    method: str = "max",
    min_assets: int = 10,
):
    """Compute quantile bucket mean returns over (T, F, N) factors.

    Args:
        factor_values: (T, F, N) device or host.
        daily_returns: (T, N) daily returns.
        n_quantiles: number of buckets.
        method: tie policy ('max' default, matches CPU assign_quantiles).
        min_assets: minimum members per bucket per (t,f) to report a mean.

    Returns:
        (T, n_quantiles, F) mean bucket returns (NaN where insufficient).
    """
    cp = _import_cp()
    x = cp.asarray(factor_values)
    r = cp.asarray(daily_returns)
    T, F, N = x.shape
    bucket = batched_quantile_assignment(x, n_quantiles=n_quantiles, method=method)  # (T,F,N)
    rb = r[:, None, :]  # (T,1,N)
    out = cp.full((T, n_quantiles, F), cp.nan, dtype=cp.float64)
    for q in range(n_quantiles):
        m = (bucket == q) & cp.isfinite(rb)  # (T,F,N)
        r0 = cp.where(m, rb, 0.0)
        cnt = cp.sum(m, axis=2)  # (T,F)
        s = cp.sum(r0, axis=2)   # (T,F)
        mean = s / cp.maximum(cnt, 1.0)
        mean = cp.where(cnt >= min_assets, mean, cp.nan)
        out[:, q, :] = mean
    return out  # (T, n_quantiles, F)
