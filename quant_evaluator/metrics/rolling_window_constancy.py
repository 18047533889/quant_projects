"""Linear-time detection of windows with identical finite observations."""
from __future__ import annotations

import numpy as np


def finite_constant_windows(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-window constant-finite masks and their common values.

    Non-finite entries are ignored for detection only. The input is never
    modified or filled. A window with no finite entries is not constant.
    The result arrays have length ``max(0, len(values) - window + 1)``.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if type(window) is not int or window <= 0:
        raise ValueError("window must be a positive integer")
    count = max(0, values.size - window + 1)
    constant = np.zeros(count, dtype=bool)
    common_value = np.full(count, np.nan, dtype=np.float64)
    if count == 0:
        return constant, common_value

    finite = np.isfinite(values)
    positions = np.arange(values.size, dtype=np.int64)
    next_finite = np.where(finite, positions, values.size)
    next_finite = np.minimum.accumulate(next_finite[::-1])[::-1]
    previous_finite = np.maximum.accumulate(np.where(finite, positions, -1))

    finite_positions = np.flatnonzero(finite)
    changes = np.zeros(values.size, dtype=np.int64)
    if finite_positions.size > 1:
        changes[finite_positions[1:]] = (
            values[finite_positions[1:]] != values[finite_positions[:-1]]
        )
    change_prefix = np.concatenate(([0], np.cumsum(changes, dtype=np.int64)))

    starts = np.arange(count, dtype=np.int64)
    ends = starts + window - 1
    first = next_finite[starts]
    last = previous_finite[ends]
    has_finite = (first < values.size) & (last >= first)
    has_change = np.ones(count, dtype=bool)
    has_change[has_finite] = (
        change_prefix[last[has_finite] + 1]
        != change_prefix[first[has_finite] + 1]
    )
    constant = has_finite & ~has_change
    common_value[constant] = values[first[constant]]
    return constant, common_value


__all__ = ["finite_constant_windows"]
