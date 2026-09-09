"""Realized portfolio cost drag from an execution-trajectory cost leg."""

from __future__ import annotations

import numpy as np


def compute_turnover_cost(
    returns: np.ndarray,
    min_periods: int = 1,
) -> float | np.ndarray:
    """Return mean realized portfolio cost drag in basis points.

    ``returns`` is deliberately named for QE's portfolio-panel runtime binder,
    but its values must be the positive cost-rate magnitudes from the typed
    ``cost_drag`` trajectory leg.  It is not portfolio PnL, gross-minus-net,
    or factor-rank turnover.  NaN denotes an unobserved period; infinities and
    negative observed costs are invalid rather than silently reinterpreted.
    """
    if isinstance(min_periods, bool) or not isinstance(min_periods, (int, np.integer)):
        raise TypeError("min_periods must be a positive integer")
    if min_periods < 1:
        raise ValueError("min_periods must be at least 1")

    values = np.asarray(returns, dtype=np.float64)
    if values.ndim not in (1, 2):
        raise ValueError("returns must have shape (T,) or (T, F)")
    observed = ~np.isnan(values)
    if np.any(np.isinf(values)):
        raise ValueError("cost_drag contains infinite values")
    if np.any(values[observed] < 0.0):
        raise ValueError("cost_drag must contain non-negative cost-rate magnitudes")

    matrix = values[:, None] if values.ndim == 1 else values
    counts = np.sum(~np.isnan(matrix), axis=0)
    totals = np.nansum(matrix, axis=0)
    result = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    enough = counts >= min_periods
    result[enough] = totals[enough] / counts[enough] * 10_000.0
    return float(result[0]) if values.ndim == 1 else result


__all__ = ["compute_turnover_cost"]
