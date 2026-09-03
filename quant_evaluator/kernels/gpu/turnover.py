"""GPU turnover (spec §12.4, Wave 2).

Turnover_t = 0.5 * sum_i |w_{t,i} - w_{t-1,i}| on rank weights
(w = rank_pct - 0.5, NaN→0).  Reuses the shared batched rank.
"""

from __future__ import annotations

from quant_evaluator.kernels.gpu.rank import batched_rank_weights


def _import_cp():
    import cupy as cp
    return cp


def batched_turnover(factor_values, return_series: bool = False):
    """Compute turnover series over (T, F, N) factors.

    Args:
        factor_values: (T, F, N) device or host.
        return_series: if True, return (T, F) series; else (F,) mean.

    Returns:
        (F,) mean turnover (or (T, F) series).
    """
    cp = _import_cp()
    x = cp.asarray(factor_values)
    w = batched_rank_weights(x)  # (T, F, N), NaN→0
    # turnover_t = 0.5 * sum_i |w_t - w_{t-1}|
    diff = cp.abs(w[1:] - w[:-1])  # (T-1, F, N)
    turnover = 0.5 * cp.sum(diff, axis=2)  # (T-1, F)
    # pad first day with 0 (no previous)
    turnover = cp.concatenate([cp.zeros((1, turnover.shape[1])), turnover], axis=0)
    if return_series:
        return turnover
    # mean over t=1..T-1 (exclude the zero-padded first day, matching CPU)
    return cp.mean(turnover[1:], axis=0)  # (F,)
