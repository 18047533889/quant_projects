"""GPU quantile returns (spec §12.3, Wave 2).

Reuses the shared batched quantile assignment; computes per-bucket mean
daily return over the pairwise-finite set with a min_assets floor, matching
CPU ``compute_quantile_returns`` (metrics/quantile.py) which returns
``(T, n_quantiles, F)``.
"""

from __future__ import annotations

import numpy as np

from quant_evaluator.kernels.gpu.rank import batched_quantile_assignment
from quant_evaluator.kernels.gpu.quantile_numeric import repair_quantile_means_gpu


def _import_cp():
    import cupy as cp
    return cp


def batched_quantile_returns(
    factor_values,
    daily_returns,
    n_quantiles: int = 10,
    method: str = "max",
    min_assets: int = 10,
    return_counts: bool = False,
    *,
    workspace_bytes: int | None = None,
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
    if x.ndim != 3 or r.shape != (x.shape[0], x.shape[2]):
        raise ValueError("expected factors (T,F,N) and returns (T,N)")
    if isinstance(min_assets, (bool, np.bool_)) or not isinstance(min_assets, (int, np.integer)) or min_assets < 1:
        raise ValueError("min_assets must be a positive integer")
    if isinstance(n_quantiles, (bool, np.bool_)) or not isinstance(n_quantiles, (int, np.integer)) or n_quantiles < 1:
        raise ValueError("n_quantiles must be a positive integer")
    if method not in ("min", "max"):
        raise ValueError("quantile method must be min/max")
    budget = (1 << 30) if workspace_bytes is None else workspace_bytes
    if isinstance(budget, (bool, np.bool_)) or not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError("workspace_bytes must be a positive integer")
    budget = int(budget)
    T, F, N = x.shape
    # Caller-owned inputs and final outputs are admitted by the session.
    # Include gathered factors/float64 returns, int32 buckets, reductions, sort
    # scratch, plus the static normal-path guard/nonzero workspace. The repair
    # API separately admits measured guard-stack bytes and conditional exact
    # workspace against the remaining post-gather budget.
    repair_guard_row_bytes = 34 + 40 * n_quantiles
    tile_input_row_bytes = 20 * N  # factor/return cast peak + bucket + live returns
    row_bytes = (4 * ((x.dtype.itemsize + 8 + 8 + 8 + 4 + 4) * N
                      + 64 * n_quantiles) + N * 40
                 + tile_input_row_bytes + repair_guard_row_bytes)
    if T * F and row_bytes > budget:
        raise MemoryError("single quantile return cross-section exceeds workspace budget")
    chunk = max(1, int(budget // max(row_bytes, 1)))
    out = cp.full((T * F, n_quantiles), cp.nan, dtype=cp.float64)
    counts = cp.zeros((T * F, n_quantiles), dtype=cp.int64)
    for start in range(0, T * F, chunk):
        stop = min(start + chunk, T * F)
        rows = cp.arange(start, stop)
        # Gather only this tile; do not flatten a transposed full input copy.
        values = x[rows // F, rows % F, :]
        rb = r[rows // F, :]
        # The exact repair ABI and its accumulator operate on binary64 values.
        # Convert on-device before both the fast reduction and risk guard.
        rb = rb.astype(cp.float64, copy=False)
        bucket = batched_quantile_assignment(values, n_quantiles=n_quantiles,
                                             method=method, workspace_bytes=int(budget))
        for q in range(n_quantiles):
            m = (bucket == q) & cp.isfinite(rb)
            cnt = cp.sum(m, axis=1)
            total = cp.sum(cp.where(m, rb, 0.0), axis=1)
            out[start:stop, q] = cp.where(cnt >= min_assets, total / cp.maximum(cnt, 1), cp.nan)
            counts[start:stop, q] = cnt
        # Drop transient reduction/gather work before the independent guard and
        # repair scratch is admitted. bucket/rb are the repair's two live inputs.
        del values, m, cnt, total, rows
        tile_means = out[start:stop, :]
        tile_counts = counts[start:stop, :]
        repair_budget = budget - int(bucket.nbytes) - int(rb.nbytes)
        if repair_budget < 1:
            raise MemoryError("quantile repair inputs leave no workspace budget")
        repair_quantile_means_gpu(
            bucket, rb, tile_means, tile_counts, int(min_assets),
            workspace_bytes=int(repair_budget),
        )
        del bucket, rb, tile_means, tile_counts
    out = out.reshape(T, F, n_quantiles).transpose(0, 2, 1)
    counts = counts.reshape(T, F, n_quantiles).transpose(0, 2, 1)
    return (out, counts) if return_counts else out
