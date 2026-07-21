"""Strict scalar parameter parsing shared by Pandas and Polars operators."""
from __future__ import annotations

import math
from typing import Any

from backend.operator_errors import OperatorParameterError


def strict_integer(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Parse an exact finite integer without bool or float truncation."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OperatorParameterError(f"{name} must be an integer, got {value!r}")
    if not math.isfinite(float(value)) or int(value) != value:
        raise OperatorParameterError(f"{name} must be an exact finite integer, got {value!r}")
    parsed = int(value)
    if minimum is not None and parsed < minimum:
        raise OperatorParameterError(f"{name} must be >= {minimum}, got {parsed}")
    if maximum is not None and parsed > maximum:
        raise OperatorParameterError(f"{name} must be <= {maximum}, got {parsed}")
    return parsed
