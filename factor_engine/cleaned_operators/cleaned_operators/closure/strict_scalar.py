# -*- coding: utf-8 -*-
"""Uniform strict scalar validators (Master Spec Part A-4).

The kernels themselves must never ``int(window)`` / ``max(2, int(x))`` a
runtime parameter value.  These four helpers are the single documented entry
point for operator code (and new registrations): each RAISES on any value that
would silently coerce, so a ``20.2`` window, a ``True`` "lag", a non-finite
epsilon or a value outside a declared choice-set is a contract error — never a
quiet truncation.

``strict_int`` / ``strict_bool`` delegate to the central validators in
``cleaned_operators.base`` (``strict_int_param`` / ``strict_bool_param``) so
the whole library has exactly ONE coercive core.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from factor_engine.cleaned_operators.base import (
    strict_bool_param,
    strict_int_param,
)

__all__ = ["strict_int", "strict_float", "strict_bool", "strict_enum"]


def strict_int(value: Any, name: str, *, lower: int | None = None, upper: int | None = None) -> int:
    """Return ``value`` as an int, raising unless it is already an exact
    integral value.  ``20.0`` is accepted (value-equal to 20); ``20.2``,
    ``True``, ``"20"``, ``NaN`` and ``Inf`` all raise (Part A-4).  Unlike the
    central ``strict_int_param`` (which keeps numeric-string compatibility for
    the legacy DSL), this closure entry point rejects strings outright."""
    if isinstance(value, str):
        raise TypeError(f"{name} must be an integer, not a string ({value!r})")
    return strict_int_param(value, name, lower=lower, upper=upper)


def strict_bool(value: Any, name: str) -> bool:
    """Return ``value`` as a strict bool (only ``bool`` / ``numpy.bool_``;
    ``1``/``0``/``"true"``/``0.5`` all raise)."""
    return strict_bool_param(value, name)


def strict_float(value: Any, name: str, *, lower: float | None = None, upper: float | None = None) -> float:
    """Return ``value`` as a finite float; ``True`` and non-finite (NaN/±Inf)
    raise.  An integral value is returned as a float — no truncation ever."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not bool")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if lower is not None and result < lower:
        raise ValueError(f"{name} must be >= {lower}, got {result}")
    if upper is not None and result > upper:
        raise ValueError(f"{name} must be <= {upper}, got {result}")
    return result


def strict_enum(value: Any, name: str, choices: tuple | frozenset | list) -> Any:
    """Return ``value`` unchanged iff it is an exact member of ``choices``
    (``"a"`` and ``"A"`` are different values; ``True`` and ``1`` are equal in
    Python, so a ``(1, 2)`` choice-set accepts ``True`` — declare bool choices
    explicitly when that distinction matters)."""
    if value not in choices:
        raise ValueError(f"{name}={value!r} is not an allowed choice {list(choices)}")
    return value
