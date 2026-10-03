"""Versioned numeric kernels for standalone factor-preprocess z-scores.

The finite-input v2 policy is FP-local. It makes no claim that FactorEngine's
separate z-score operators have been admitted to or qualified against v2.
"""
from __future__ import annotations

from typing import Any

import numpy as np

FINITE_ANCHOR_CENTERED_V2 = "finite_anchor_centered_v2"


def finite_anchor_centered_zscore(
    values: np.ndarray,
    *,
    axis: Any = -1,
    ddof: float = 1,
    constant_value: float = 0.0,
) -> np.ndarray:
    """Stable Float64 z-score for finite members, preserving FP v1 edge policy.

    Inputs are first converted to Float64, matching NumPy's implicit statistics
    conversion for integer arrays (including its >2**53 rounding). Finite slices
    use a safe midpoint anchor and scaled central moments. Any slice containing
    +/-Inf uses the legacy NumPy formula so its existing policy remains intact.
    NaN remains missing. The output is Float64 for nonempty v2 inputs.
    """
    arr = np.asarray(values).astype(np.float64, copy=False)
    if arr.size == 0:
        return arr.copy()

    finite = np.isfinite(arr)
    count = np.sum(finite, axis=axis, keepdims=True, dtype=np.int64)
    low = np.min(np.where(finite, arr, np.inf), axis=axis, keepdims=True)
    high = np.max(np.where(finite, arr, -np.inf), axis=axis, keepdims=True)
    safe_low = np.where(count > 0, low, 0.0)
    safe_high = np.where(count > 0, high, 0.0)
    anchor = safe_low / 2.0 + safe_high / 2.0

    # The anchor lies inside the finite range. Scaling deltas bounds all squares.
    finite_value = np.where(finite, arr, anchor)
    delta = finite_value - anchor
    scale = np.max(np.abs(delta), axis=axis, keepdims=True)
    unit = np.divide(
        delta,
        scale,
        out=np.zeros_like(delta, dtype=np.float64),
        where=scale > 0.0,
    )
    unit_sum = np.sum(unit, axis=axis, keepdims=True, dtype=np.float64)
    mean = np.divide(
        unit_sum,
        count,
        out=np.zeros_like(unit_sum, dtype=np.float64),
        where=count > 0,
    )
    centered = np.where(finite, unit - mean, 0.0)
    divisor = count.astype(np.float64) - float(ddof)
    squared_sum = np.sum(centered * centered, axis=axis, keepdims=True, dtype=np.float64)
    variance = np.divide(
        squared_sum,
        divisor,
        out=np.full_like(squared_sum, np.nan, dtype=np.float64),
        where=divisor > 0.0,
    )
    std = np.sqrt(variance)
    with np.errstate(invalid="ignore", divide="ignore"):
        stable = np.where(std > 0.0, centered / std, constant_value)

    # Inf participates in the old NumPy moments; its NaN/constant outcomes are
    # deliberately preserved instead of silently dropping it from the sample.
    has_inf = np.any(np.isinf(arr), axis=axis, keepdims=True)
    if np.any(has_inf):
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            legacy_mean = np.nanmean(arr, axis=axis, keepdims=True)
            legacy_std = np.nanstd(arr, axis=axis, keepdims=True, ddof=ddof)
            legacy = np.where(
                legacy_std > 0.0,
                (arr - legacy_mean) / legacy_std,
                constant_value,
            )
        stable = np.where(has_inf, legacy, stable)

    return np.where(np.isnan(arr), np.nan, stable)
