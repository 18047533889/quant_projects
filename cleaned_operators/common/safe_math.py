"""
Safe mathematical operations to prevent numerical hazards.

This module provides utilities for division by zero, log of non-positive values,
sqrt of negative values, and power overflow protection.

Phase 1 Safety Fix - Created during operator audit remediation.
"""

import numpy as np
from typing import Union

# Type alias for numeric types
Numeric = Union[float, int, np.ndarray]


def safe_divide(
    numerator: Numeric,
    denominator: Numeric,
    fill_value: float = np.nan,
    epsilon: float = 1e-15
) -> Numeric:
    """
    Safely divide numerator by denominator, handling zero division.

    Args:
        numerator: The numerator value(s)
        denominator: The denominator value(s)
        fill_value: Value to return when denominator is zero (default: np.nan)
        epsilon: Threshold below which denominator is considered zero (default: 1e-15)

    Returns:
        Result of division, or fill_value where denominator is near zero
    """
    if isinstance(denominator, np.ndarray):
        result = np.full_like(numerator, fill_value, dtype=float)
        mask = np.abs(denominator) > epsilon
        result[mask] = numerator[mask] / denominator[mask]
        return result
    else:
        if abs(denominator) > epsilon:
            return numerator / denominator
        else:
            return fill_value


def safe_log(
    x: Numeric,
    fill_value: float = np.nan,
    epsilon: float = 1e-15
) -> Numeric:
    """
    Safely compute logarithm, handling non-positive values.

    Args:
        x: Input value(s)
        fill_value: Value to return when x <= 0 (default: np.nan)
        epsilon: Minimum positive value for log (default: 1e-15)

    Returns:
        Natural logarithm of x, or fill_value where x <= epsilon
    """
    if isinstance(x, np.ndarray):
        result = np.full_like(x, fill_value, dtype=float)
        mask = x > epsilon
        result[mask] = np.log(x[mask])
        return result
    else:
        if x > epsilon:
            return np.log(x)
        else:
            return fill_value


def safe_sqrt(
    x: Numeric,
    fill_value: float = np.nan,
    epsilon: float = -1e-15
) -> Numeric:
    """
    Safely compute square root, handling negative values.

    Args:
        x: Input value(s)
        fill_value: Value to return when x < 0 (default: np.nan)
        epsilon: Tolerance for negative values (default: -1e-15, allows small numerical errors)

    Returns:
        Square root of x, or fill_value where x < epsilon
    """
    if isinstance(x, np.ndarray):
        result = np.full_like(x, fill_value, dtype=float)
        mask = x >= epsilon
        # For small negative values due to numerical error, use 0
        safe_x = np.maximum(x, 0.0)
        result[mask] = np.sqrt(safe_x[mask])
        return result
    else:
        if x >= epsilon:
            return np.sqrt(max(x, 0.0))
        else:
            return fill_value


def safe_power(
    base: Numeric,
    exponent: Numeric,
    max_exp: float = 100.0,
    fill_value: float = np.nan
) -> Numeric:
    """
    Safely compute power, handling overflow.

    Args:
        base: Base value(s)
        exponent: Exponent value(s)
        max_exp: Maximum allowed exponent magnitude (default: 100)
        fill_value: Value to return on overflow (default: np.nan)

    Returns:
        base ** exponent, or fill_value when exponent exceeds max_exp
    """
    if isinstance(exponent, np.ndarray):
        result = np.full_like(base, fill_value, dtype=float)
        mask = np.abs(exponent) <= max_exp
        try:
            result[mask] = np.power(base[mask], exponent[mask])
        except (OverflowError, FloatingPointError):
            # If overflow still occurs, leave as fill_value
            pass
        return result
    else:
        if abs(exponent) <= max_exp:
            try:
                return base ** exponent
            except (OverflowError, FloatingPointError):
                return fill_value
        else:
            return fill_value
