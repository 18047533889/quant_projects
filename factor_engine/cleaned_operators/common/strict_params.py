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
from cleaned_operators.base import (
    _coerce_declared_numeric_string,
    _normalise_integer,
    strict_bool_param,
    strict_int_param,
    strict_int_runtime,
)


def strict_int(value: Any, name: str, *, minimum: int | None = None,
               maximum: int | None = None) -> int:
    """Exact finite integer (no bool, no float truncation).

    R19-005: this is the RUNTIME kernel gate — a numeric string (``"20"``) is
    REJECTED.  Numeric-string conversion happens once at the DSL/binder
    declaration layer (:func:`cleaned_operators.base.bind_numeric_string_if_declared`);
    a string reaching a kernel means the parameter was not declared numeric.
    """
    return strict_int_runtime(value, name, lower=minimum, upper=maximum)


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
    # R16-051: return the canonicalized float, never the raw int/str value —
    # otherwise ``1`` and ``1.0`` coexist as two AST representations of the
    # same parameter (a false search-space duplicate).
    return numeric


def strict_probability(value: Any, name: str) -> float:
    """Finite value in [0, 1] (a probability / ratio / quantile domain).

    R16-052: a numeric string (``"0.5"``) is rejected — the strict gate must
    never smuggle a string into a kernel as a probability.
    """
    if isinstance(value, (bool, np.bool_)):
        raise OperatorParameterError(f"{name} must be a probability, not bool")
    if isinstance(value, str):
        raise OperatorParameterError(
            f"{name} must be a real number, not a string ({value!r})"
        )
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise OperatorParameterError(
            f"{name} must be a real number, not {type(value).__name__} ({value!r})"
        )
    numeric = float(value)
    if not np.isfinite(numeric):
        raise OperatorParameterError(f"{name} must be finite, got {value!r}")
    if not 0.0 <= numeric <= 1.0:
        raise OperatorParameterError(f"{name} must be in [0, 1], got {numeric}")
    return numeric


def strict_bool(value: Any, name: str) -> bool:
    return strict_bool_param(value, name)


def strict_enum(value: Any, name: str, choices: Sequence[Any]) -> Any:
    """Exact membership in an allowed set (string enums, reviewed grids).

    R16-053: membership is type-aware.  Python cross-type equality
    (``True == 1``, ``1 == 1.0``) must not smuggle a value past the gate — a
    bool is never accepted for numeric choices.  Within the numeric family an
    equivalent value is canonicalized to the DECLARED choice so ``1`` and ``1.0``
    cannot coexist as two representations of one parameter.
    """
    if isinstance(value, (bool, np.bool_)) and not any(
        isinstance(c, (bool, np.bool_)) for c in choices
    ):
        raise OperatorParameterError(
            f"{name} must be an allowed choice, not bool ({value!r})"
        )
    for choice in choices:
        if type(value) is type(choice) and value == choice:
            return value
        if isinstance(value, (int, float, np.integer, np.floating)) and isinstance(
            choice, (int, float, np.integer, np.floating)
        ):
            if float(value) == float(choice):
                return choice
    raise OperatorParameterError(
        f"{name}={value!r} is not an allowed choice {list(choices)}"
    )


def normalize_and_validate_scalar_param(
    canonical: str,
    param_name: str,
    value: Any,
    *,
    phase: str = "planning",
    declared_type: type | None = None,
    spec: Any | None = None,
) -> Any:
    """R19-002: the SINGLE unified scalar-parameter validation entry shared by
    the planning-time validator (``validate_plan_params``), the runtime call
    gate (``_normalise_call``) and the hash identity path.

    Planning and runtime consume the SAME declared ParamSpec type domain: np
    scalars (``np.int64``/``np.float64``), Decimal, enum members and explicit
    ``None`` are validated by exactly the same rules the kernel gate would
    apply, instead of being skipped because ``isinstance(value, (int, float,
    str, bool))`` is False (the old planning gap).  Structured / vector values
    (``list``/``tuple``/``dict``/``set``) are NOT scalar params and pass through
    unchanged — the vector / structured path owns them, matching the runtime.

    The declaration-layer numeric-string binder (:func:`cleaned_operators.base.
    bind_numeric_string_if_declared`) runs FIRST, so a numeric string is
    converted once here and the kernel gates (``strict_int_runtime``) never see
    it (R19-005).  ``phase`` only labels the error context
    (``planning``/``runtime``/``hash``); it never changes the validation rules.
    """
    if phase not in ("planning", "runtime", "hash"):
        raise ValueError(
            f"unknown phase {phase!r}; expected 'planning' | 'runtime' | 'hash'"
        )
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        # Structured / vector param: owned by the vector / structured path.
        return value
    coerced = _coerce_declared_numeric_string(value, param_name, declared_type, spec)
    try:
        return _normalise_integer(coerced, param_name, declared_type, spec)
    except OperatorParameterError as exc:
        raise OperatorParameterError(
            f"{canonical}.{param_name} [{phase}]: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Panel-like semantic boolean validators (NEW-047..051, NEW-253).
# These operate on the raw numpy array of a condition/event/member/state input.
# Legal set: {0, 1, NaN}; SignedEvent adds -1.  ±Inf is always an error.
# ---------------------------------------------------------------------------


def _coerce_numeric_panel(panel: Any, name: str) -> np.ndarray:
    """Convert a panel input to a float ndarray WITHOUT string coercion.

    R16-054: the operator semantic boundary must not turn a string column
    (``'0'``/``'1'``) into a legal condition — string coercion may only happen
    in a SourceAdapter with lineage.  Object arrays whose elements are real
    scalars / NaN / None (e.g. a plain Python list) are accepted; any string
    element is a hard error.
    """
    raw = panel.to_numpy() if hasattr(panel, "to_numpy") else np.asarray(panel)
    if raw.dtype.kind in "SU":
        raise OperatorParameterError(
            f"{name} must be a numeric bool panel, not string values (dtype {raw.dtype})"
        )
    if raw.dtype.kind == "O":  # object: only real scalars / NaN / None allowed
        for v in raw.ravel():
            if v is None:
                continue
            if isinstance(v, str):
                raise OperatorParameterError(
                    f"{name} must be a numeric bool panel, found string element {v!r}"
                )
            if not isinstance(v, (int, float, np.integer, np.floating)):
                raise OperatorParameterError(
                    f"{name} must be a numeric bool panel, found "
                    f"{type(v).__name__} element {v!r}"
                )
    try:
        return raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise OperatorParameterError(
            f"{name} must be convertible to numeric for semantic-bool validation"
        ) from exc


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
    cv = _coerce_numeric_panel(condition, name)
    _validate_bool_panel_array(cv, name, allow_negative_one=False)


def strict_event_bool(event: Any, name: str = "event") -> None:
    """Validate an EventBool panel input: legal set {0, 1, NaN}."""
    cv = _coerce_numeric_panel(event, name)
    _validate_bool_panel_array(cv, name, allow_negative_one=False)


def strict_signed_event(event: Any, name: str = "event") -> None:
    """Validate a SignedEvent panel input: legal set {-1, 0, 1, NaN}."""
    cv = _coerce_numeric_panel(event, name)
    _validate_bool_panel_array(cv, name, allow_negative_one=True)


def strict_universe_bool(mask: Any, name: str = "universe_mask") -> None:
    """Validate a UniverseBool panel input: legal set {0, 1, NaN}."""
    cv = _coerce_numeric_panel(mask, name)
    _validate_bool_panel_array(cv, name, allow_negative_one=False)
