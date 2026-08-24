# -*- coding: utf-8 -*-
"""FE-P0-021 integration test — production runtime already fails closed.

This test verifies that the REAL production execution path (planner +
_enforce_active_when) already implements fail-closed behavior for missing
active_when controllers, independent of the ParameterCanonicalizer fix.

The production runtime raises OperatorParameterError when a controller is
missing and has no bindable default.
"""
from __future__ import annotations

import pytest

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    OperatorParameterError,
    ParamSpec,
    _enforce_active_when,
)


def test_fe_p0_021_production_runtime_fails_closed_on_missing_controller():
    """Production runtime (_enforce_active_when) raises when controller missing."""
    # Define an operator with active_when but controller has no default
    metadata = OperatorMetadata(
        name="test_op",
        category="test",
        param_specs={
            "window": ParamSpec(dtype=int, min=2, max=100),
            "mode": ParamSpec(dtype=str, choices=("basic", "advanced")),  # NO default
            "adjustment": ParamSpec(
                dtype=str,
                choices=("none", "log", "sqrt"),
                active_when=("mode", ("advanced",)),
                default="none",
            ),
        },
    )

    # Call with adjustment but without controller mode -> must raise
    with pytest.raises(OperatorParameterError) as exc_info:
        _enforce_active_when(
            metadata,
            (),
            {"window": 20, "adjustment": "log"},
            defaults={"window": 10},  # mode has no default
        )

    assert "active_when controller 'mode'" in str(exc_info.value)
    assert "unbound and has no bindable default" in str(exc_info.value)
    assert "cannot judge active/inactive" in str(exc_info.value)
    assert "fail-closed" in str(exc_info.value)


def test_fe_p0_021_production_runtime_succeeds_with_controller_default():
    """Production runtime succeeds when controller has a bindable default."""
    metadata = OperatorMetadata(
        name="test_op",
        category="test",
        param_specs={
            "window": ParamSpec(dtype=int, min=2, max=100),
            "mode": ParamSpec(
                dtype=str,
                choices=("basic", "advanced"),
                default="basic",  # HAS default
            ),
            "adjustment": ParamSpec(
                dtype=str,
                choices=("none", "log", "sqrt"),
                active_when=("mode", ("advanced",)),
                default="none",
            ),
        },
    )

    # Controller has default -> can judge adjustment is inactive
    # adjustment="none" (its default) is allowed when inactive
    active, inactive = _enforce_active_when(
        metadata,
        (),
        {"window": 20, "adjustment": "none"},
        defaults={"window": 10, "mode": "basic"},
    )

    assert "adjustment" in inactive
    assert "adjustment" not in active


def test_fe_p0_021_production_runtime_succeeds_with_explicit_controller():
    """Production runtime succeeds when controller is explicitly provided."""
    metadata = OperatorMetadata(
        name="test_op",
        category="test",
        param_specs={
            "window": ParamSpec(dtype=int, min=2, max=100),
            "mode": ParamSpec(dtype=str, choices=("basic", "advanced")),
            "adjustment": ParamSpec(
                dtype=str,
                choices=("none", "log", "sqrt"),
                active_when=("mode", ("advanced",)),
                default="none",
            ),
        },
    )

    # Explicit controller -> can judge
    active, inactive = _enforce_active_when(
        metadata,
        (),
        {"window": 20, "mode": "advanced", "adjustment": "log"},
        defaults={"window": 10},
    )

    assert "adjustment" in active
    assert "adjustment" not in inactive


def test_fe_p0_021_production_runtime_rejects_inactive_param_non_default():
    """Production runtime rejects inactive parameter at non-default value."""
    metadata = OperatorMetadata(
        name="test_op",
        category="test",
        param_specs={
            "window": ParamSpec(dtype=int, min=2, max=100),
            "mode": ParamSpec(
                dtype=str,
                choices=("basic", "advanced"),
                default="basic",
            ),
            "adjustment": ParamSpec(
                dtype=str,
                choices=("none", "log", "sqrt"),
                active_when=("mode", ("advanced",)),
                default="none",
            ),
        },
    )

    # mode=basic makes adjustment inactive, but adjustment="log" (non-default) -> reject
    with pytest.raises(OperatorParameterError) as exc_info:
        _enforce_active_when(
            metadata,
            (),
            {"window": 20, "adjustment": "log"},
            defaults={"window": 10, "mode": "basic"},
        )

    assert "parameter 'adjustment' is inactive" in str(exc_info.value)
    assert "dead knob" in str(exc_info.value)
