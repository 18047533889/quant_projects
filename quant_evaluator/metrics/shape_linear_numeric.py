"""Selective exact arithmetic for float64 linear quantile-shape metrics.

Normal-scale inputs stay on the caller's established NumPy operation path.
Inputs that may exceed the absolute-error budget, overflow an intermediate, or
involve subnormals use exact binary-rational accumulation and one final round.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Iterable

import numpy as np


_FLOAT64 = np.finfo(np.float64)


def linear_risk_columns(values: np.ndarray, coefficient_l1: int) -> np.ndarray:
    """Vectorized absolute-error, overflow, and subnormal mask for (Q,F)."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("values must be a two-dimensional (Q,F) matrix")
    if coefficient_l1 <= 0:
        raise ValueError("coefficient_l1 must be positive")
    absolute = np.abs(data)
    largest = np.max(absolute, axis=0)
    nonfinite_columns = ~np.isfinite(largest)
    if np.any(nonfinite_columns):
        finite = np.isfinite(data[:, nonfinite_columns])
        largest[nonfinite_columns] = np.max(
            absolute[:, nonfinite_columns], axis=0, where=finite, initial=0.0,
        )
    subnormal = np.any((absolute > 0.0) & (absolute < _FLOAT64.tiny), axis=0)
    # Bound accumulated absolute rounding by 2*eps*(L1+2)*max_abs.
    # The overflow ceiling separately protects intermediates when the exact
    # weighted result is finite because of cancellation.
    l1 = float(coefficient_l1)
    abs_error_threshold = 1.0e-12 / (2.0 * _FLOAT64.eps * (l1 + 2.0))
    overflow_threshold = _FLOAT64.max / (2.0 * l1)
    return subnormal | (largest >= min(abs_error_threshold, overflow_threshold))


def _fraction_to_float64(value: Fraction) -> float:
    try:
        return float(value)
    except OverflowError:
        return float("-inf") if value < 0 else float("inf")


def exact_weighted_mean(values: Iterable[float], coefficients: Iterable[int],
                        divisor: int) -> float:
    """Round the exact weighted sum divided by ``divisor`` exactly once."""
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    pairs = zip(values, coefficients, strict=True)
    total = sum((Fraction.from_float(float(value)) * int(coefficient)
                 for value, coefficient in pairs), Fraction())
    return _fraction_to_float64(total / divisor)


def exact_mean_abs_pair_differences(left: Iterable[float],
                                    right: Iterable[float]) -> float:
    """Round the exact mean of absolute pair differences once."""
    differences = [abs(Fraction.from_float(float(b)) - Fraction.from_float(float(a)))
                   for a, b in zip(left, right, strict=True)]
    if not differences:
        raise ValueError("at least one pair is required")
    return _fraction_to_float64(sum(differences, Fraction()) / len(differences))
