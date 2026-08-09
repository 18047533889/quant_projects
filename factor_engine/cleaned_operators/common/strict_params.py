# -*- coding: utf-8 -*-
"""THE single strict scalar-parameter authority (Master Prompt NEW-005).

Every operator kernel in the tree must call exactly this module for scalar
validation — never a module-local ``int(value)``, never ``max(2, int(x))``,
never ``positive_int`` / ``_pos_int`` / ``_pi`` / ``_positive_int`` copies.

Contract (all raise :class:`backend.operator_errors.OperatorParameterError` on
violation; none clip, truncate or silently reinterpret):

* ``strict_int``                 — exact finite integral value (20.9 rejected)
* ``strict_nonnegative_int``     — >= 0
* ``strict_positive_int``        — >= 1
* ``strict_float``               — finite real (bool rejected)
* ``strict_probability``         — finite in [0, 1]
* ``strict_bool``                — ``type(value) is bool`` only
* ``strict_enum``                — exact membership in an allowed set
* ``strict_condition_bool``      — panel-like ConditionBool {0, 1, NaN}
* ``strict_event_bool``          — EventBool {0, 1, NaN}
* ``strict_signed_event``        — SignedEvent {-1, 0, 1, NaN}
* ``strict_universe_bool``       — UniverseBool {0, 1, NaN}

The bool / weight / positive-price semantic validators (NEW-047..051, NEW-253)
all reject ±Inf: the legal set is strictly {0,1,NaN} (or {-1,0,1,NaN} for
SignedEvent); an Inf is NOT a valid "truthy" and is NOT a missing value — it is
a data-quality error.

``strict_positive_int`` / ``strict_nonnegative_int`` / ``strict_int`` are thin
re-exports of :func:`cleaned_operators.base.strict_int_param` so kernels that
only need integer validation may import the single gate.  ``strict_bool``
re-exports ``cleaned_operators.base.strict_bool_param``.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from backend.operator_errors import OperatorParameterError
from cleaned_operators.base import strict_bool_param, strict_int_param


def strict_int(value: Any, name: str, *, minimum: int | None = None,
               maximum: int | None = None) -> int:
    """Exact finite integer (no bool, no float truncation)."""
    return strict_int_param(value, name, lower=minimum, upper=maximum)


def strict_nonnegative_int(value: Any, name: str) -> int:
    return strict_int_param(value, name, lower=0)


def strict_positive_int(value: Any, name: str) -> int:
    return strict_int_param(value, name, lower=1)


def strict_float(value: Any, name: str, *, minimum: float | None = None,
                 maximum: float | None = None) -> float:
    """Finite real scalar; bool and non-finite (incl. ±Inf) rejected."""
    if isinstance(value, (bool, np.bool_)):
        raise OperatorParameterError(f"{name} must be a real number, not bool")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise OperatorParameterError(
            f"{name} must be a real number, not {type(value).__name__} ({value!r})"
        )
    numeric = float(value)
    if not np.isfinite(numeric):
        raise OperatorParameterError(f"{name} must be finite, got {value!r}")
    if minimum is not None and numeric < minimum:
        raise OperatorParameterError(f"{name} must be >= {minimum}, got {numeric}")
    if maximum is not None and numeric > maximum:
        raise OperatorParameterError(f"{name} must be <= {maximum}, got {numeric}")
    return value


def strict_probability(value: Any, name: str) -> float:
    """Finite value in [0, 1] (a probability / ratio / quantile domain)."""
    if isinstance(value, (bool, np.bool_)):
        raise OperatorParameterError(f"{name} must be a probability, not bool")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise OperatorParameterError(f"{name} must be finite, got {value!r}")
    if not 0.0 <= numeric <= 1.0:
        raise OperatorParameterError(f"{name} must be in [0, 1], got {numeric}")
    return value


def strict_bool(value: Any, name: str) -> bool:
    return strict_bool_param(value, name)


def strict_enum(value: Any, name: str, choices: Sequence[Any]) -> Any:
    """Exact membership in an allowed set (string enums, reviewed grids)."""
    if value not in choices:
        raise OperatorParameterError(
            f"{name}={value!r} is not an allowed choice {list(choices)}"
        )
    return value


# ---------------------------------------------------------------------------
# Panel-like semantic boolean validators (NEW-047..051, NEW-253).
# These operate on the raw numpy array of a condition/event/member/state input.
# Legal set: {0, 1, NaN}; SignedEvent adds -1.  ±Inf is always an error.
# ---------------------------------------------------------------------------


def _validate_bool_panel_array(cv: np.ndarray, name: str, *, allow_negative_one: bool) -> None:
    missing = (cv != cv)  # NaN (works for float arrays, incl. object arrays of floats)
    if allow_negative_one:
        valid = (cv == 0.0) | (cv == 1.0) | (cv == -1.0)
    else:
        valid = (cv == 0.0) | (cv == 1.0)
    bad = ~missing & ~valid
    if np.any(bad):
        bad_values = sorted({str(v) for v in np.unique(cv[bad]).tolist()})[:10]
        legal = "{-1, 0, 1, NaN}" if allow_negative_one else "{0, 1, NaN}"
        raise OperatorParameterError(
            f"{name} must be a {legal} semantic bool; found {int(bad.sum())} "
            f"value(s) outside the legal set: {bad_values} (this includes ±Inf, "
            "which must never be read as True or False)"
        )


def strict_condition_bool(condition: Any, name: str = "condition") -> None:
    """Validate a ConditionBool panel input: legal set {0, 1, NaN}."""
    cv = np.asarray(condition.to_numpy(), dtype=float) if hasattr(condition, "to_numpy") else np.asarray(condition, dtype=float)
    _validate_bool_panel_array(cv, name, allow_negative_one=False)


def strict_event_bool(event: Any, name: str = "event") -> None:
    """Validate an EventBool panel input: legal set {0, 1, NaN}."""
    cv = np.asarray(event.to_numpy(), dtype=float) if hasattr(event, "to_numpy") else np.asarray(event, dtype=float)
    _validate_bool_panel_array(cv, name, allow_negative_one=False)


def strict_signed_event(event: Any, name: str = "event") -> None:
    """Validate a SignedEvent panel input: legal set {-1, 0, 1, NaN}."""
    cv = np.asarray(event.to_numpy(), dtype=float) if hasattr(event, "to_numpy") else np.asarray(event, dtype=float)
    _validate_bool_panel_array(cv, name, allow_negative_one=True)


def strict_universe_bool(mask: Any, name: str = "universe_mask") -> None:
    """Validate a UniverseBool panel input: legal set {0, 1, NaN}."""
    cv = np.asarray(mask.to_numpy(), dtype=float) if hasattr(mask, "to_numpy") else np.asarray(mask, dtype=float)
    _validate_bool_panel_array(cv, name, allow_negative_one=False)
