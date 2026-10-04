"""Mean-only rolling IC computation for requests without rich statistics."""
from __future__ import annotations

import numpy as np

from quant_evaluator.metrics.ic_summary import _rolling_mean_nan


def compute_rolling_ic_mean(
    ic_series: np.ndarray, window: int = 60, min_periods: int = 20,
) -> np.ndarray:
    """Return trailing finite-observation means as a float64 (T, F) array.

    A one-dimensional input represents one factor. NaN and both infinities
    do not count as observations. Each cell needs ``min_periods`` finite
    observations in its trailing (initially expanding) window. Constants
    have ordinary finite means; no variance/IR statistics are computed.
    Supported window/min_periods semantics match the rich IC statistics API.
    """
    arr = np.asarray(ic_series, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    time_count, factor_count = arr.shape
    out = np.full((time_count, factor_count), np.nan, dtype=np.float64)
    for factor in range(factor_count):
        values = arr[:, factor]
        # Preserve rich-statistics global admission and identical mean math.
        if np.count_nonzero(np.isfinite(values)) < min_periods:
            continue
        out[:, factor] = _rolling_mean_nan(values, window, min_periods)
    return out
