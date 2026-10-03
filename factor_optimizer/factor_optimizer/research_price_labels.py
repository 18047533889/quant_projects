"""Pure price-to-return labels for research inputs."""
from __future__ import annotations

import numpy as np


def price_to_return_labels(start_prices, end_prices):
    """Return simple end/start - 1 labels and a mask of usable observations.

    Both endpoints must be same-shaped two-dimensional real numeric arrays.
    An observation is valid only when both prices are finite and strictly
    positive and the resulting return is finite. Invalid labels remain NaN.
    Inputs are read-only from this function's perspective.
    """
    try:
        start = np.asarray(start_prices)
        end = np.asarray(end_prices)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "start_prices and end_prices must have the same two-dimensional shape"
        ) from exc

    if start.ndim != 2 or end.ndim != 2 or start.shape != end.shape:
        raise ValueError(
            "start_prices and end_prices must have the same two-dimensional shape")
    if start.dtype.kind not in "iuf" or end.dtype.kind not in "iuf":
        raise TypeError("start_prices and end_prices must be real numeric arrays")

    dtype = np.result_type(start.dtype, end.dtype, np.float64)
    start_values = start.astype(dtype, copy=False)
    end_values = end.astype(dtype, copy=False)
    endpoints_valid = (np.isfinite(start_values) & np.isfinite(end_values)
                       & (start_values > 0) & (end_values > 0))

    returns = np.full(start.shape, np.nan, dtype=dtype)
    valid = np.zeros(start.shape, dtype=bool)
    if np.any(endpoints_valid):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            candidates = end_values[endpoints_valid] / start_values[endpoints_valid] - 1
        finite_result = np.isfinite(candidates)
        valid[endpoints_valid] = finite_result
        returns[endpoints_valid] = np.where(finite_result, candidates, np.nan)
    return returns, valid
