"""GPU turnover (spec §12.4, Wave 2).

For an eligible cross-section (at least 10 finite signals), average-tie ranks
are normalized to sum one over finite assets; missing signals receive zero
proxy weight. Turnover_t = 0.5 * sum_i |w_{t,i} - w_{t-1,i}|. An ineligible
date has unknown weights and turnover, not a zero-cost cash position.
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
    # Match metrics.turnover._rank_weights_matrix: its default min_obs=10 is
    # part of the published rank-turnover contract. With fewer observations,
    # the CPU reference leaves that date unknown, so GPU must do the same.
    eligible = cp.isfinite(x).sum(axis=2) >= 10
    w = cp.where(eligible[..., None], cp.where(cp.isfinite(w), w, 0.), cp.nan)
    # turnover_t = 0.5 * sum_i |w_t - w_{t-1}|
    joint = cp.isfinite(w[1:]) & cp.isfinite(w[:-1])
    diff = cp.where(joint, cp.abs(w[1:] - w[:-1]), 0.)
    turnover = cp.where(cp.all(joint, axis=2), 0.5 * cp.sum(diff, axis=2), cp.nan)
    turnover = cp.concatenate([cp.full((1, turnover.shape[1]), cp.nan), turnover], axis=0)
    if return_series:
        return turnover
    # mean over t=1..T-1 (exclude the zero-padded first day, matching CPU)
    return cp.nanmean(turnover[1:], axis=0)  # (F,)


def batched_membership_turnover(factor_values, quantile=.9):
    """CPU temporal.compute_factor_turnover_rate semantics, shape (T-1,F).

    This diagnostic uses the jointly observed universe, not executed trades.
    Its denominator is the joint security count, not gross traded notional.
    """
    from .tradability import batched_factor_turnover_rate
    return _import_cp().asarray(batched_factor_turnover_rate(factor_values, quantile))
