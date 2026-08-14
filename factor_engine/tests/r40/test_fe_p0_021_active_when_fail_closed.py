# -*- coding: utf-8 -*-
"""FE-P0-021: parameter active_when controller missing must fail-closed in production.

When a ParamSpec declares active_when but the controller parameter is missing
and has no decidable default, the canonicalizer must:

- Production mode: raise ParameterContractError (fail-closed)
- Research mode: skip the check (fail-open, legacy behavior)

This ensures production paths never silently accept parameter combinations
whose active/inactive state is undecidable.
"""
from __future__ import annotations

import pytest

from parameter_canonicalizer import (
    ParameterCanonicalizer,
    ParameterContractError,
)


class _SpecStub:
    """Duck-typed stand-in for cleaned_operators.base.ParamSpec."""

    def __init__(
        self,
        dtype=None,
        min=None,
        max=None,
        choices=None,
        active_when=None,
        default=None,
    ):
        self.dtype = dtype
        self.min = min
        self.max = max
        self.choices = choices
        self.active_when = active_when
        self.default = default


# ---------------------------------------------------------------------------
# FE-P0-021: Production mode must fail-closed when controller is undecidable
# ---------------------------------------------------------------------------


def test_production_mode_raises_when_controller_missing_no_default():
    """Production canonicalizer must raise when active_when controller is
    missing and has no ParamSpec default."""
    specs = {
        "window": _SpecStub(dtype=int, min=2, max=100),
        "adjustment": _SpecStub(
            dtype=str,
            choices=("none", "log", "sqrt"),
            active_when=("mode", ("advanced",)),
            default="none",
        ),
        # mode has no default - undecidable
        "mode": _SpecStub(dtype=str, choices=("basic", "advanced")),
    }
    pc = ParameterCanonicalizer(
        "test_op",
        [],
        param_specs=specs,
        production_mode=True,
    )
    # Controller "mode" is missing and has no default -> production must fail-closed
    with pytest.raises(ParameterContractError, match="controller .* missing"):
        pc.canonical_key({"window": 20, "adjustment": "log"})

    with pytest.raises(ParameterContractError, match="FE-P0-021"):
        pc.canonical_key({"window": 20, "adjustment": "log"})


def test_research_mode_allows_missing_controller_no_default():
    """Research mode preserves legacy fail-open behavior: missing controller
    with no default is silently skipped (cannot judge active/inactive)."""
    specs = {
        "window": _SpecStub(dtype=int, min=2, max=100),
        "adjustment": _SpecStub(
            dtype=str,
            choices=("none", "log", "sqrt"),
            active_when=("mode", ("advanced",)),
            default="none",
        ),
        "mode": _SpecStub(dtype=str, choices=("basic", "advanced")),
    }
    pc = ParameterCanonicalizer(
        "test_op",
        [],
        param_specs=specs,
        production_mode=False,  # research/legacy mode
    )
    # Research mode: undecidable controller -> skip check, no error
    key = pc.canonical_key({"window": 20, "adjustment": "log"})
    assert ("window", "20") in key
    assert ("adjustment", "'log'") in key


def test_production_mode_succeeds_when_controller_has_spec_default():
    """Production mode succeeds when controller is missing but has a
    ParamSpec.default - the default is used to judge active/inactive."""
    specs = {
        "window": _SpecStub(dtype=int, min=2, max=100),
        "adjustment": _SpecStub(
            dtype=str,
            choices=("none", "log", "sqrt"),
            active_when=("mode", ("advanced",)),
            default="none",
        ),
        "mode": _SpecStub(
            dtype=str,
            choices=("basic", "advanced"),
            default="basic",  # controller has default
        ),
    }
    pc = ParameterCanonicalizer(
        "test_op",
        [],
        param_specs=specs,
        production_mode=True,
    )
    # Controller defaults to "basic", so "adjustment" is inactive.
    # Providing adjustment="none" (its default) is allowed.
    key = pc.canonical_key({"window": 20, "adjustment": "none"})
    assert ("window", "20") in key

    # Omitting inactive parameter entirely is also allowed
    key2 = pc.canonical_key({"window": 20})
    assert ("window", "20") in key2


def test_production_mode_succeeds_when_controller_explicitly_provided():
    """Production mode succeeds when controller is explicitly provided in params."""
    specs = {
        "window": _SpecStub(dtype=int, min=2, max=100),
        "adjustment": _SpecStub(
            dtype=str,
            choices=("none", "log", "sqrt"),
            active_when=("mode", ("advanced",)),
            default="none",
        ),
        "mode": _SpecStub(dtype=str, choices=("basic", "advanced")),
    }
    pc = ParameterCanonicalizer(
        "test_op",
        [],
        param_specs=specs,
        production_mode=True,
    )
    # Controller explicitly provided -> decidable
    key = pc.canonical_key({"window": 20, "mode": "advanced", "adjustment": "log"})
    assert ("window", "20") in key
    assert ("mode", "'advanced'") in key
    assert ("adjustment", "'log'") in key


def test_production_mode_rejects_inactive_parameter_with_non_default_value():
    """Production mode must reject an inactive parameter bound to a non-default
    value, even when controller is provided or has a default."""
    specs = {
        "window": _SpecStub(dtype=int, min=2, max=100),
        "adjustment": _SpecStub(
            dtype=str,
            choices=("none", "log", "sqrt"),
            active_when=("mode", ("advanced",)),
            default="none",
        ),
        "mode": _SpecStub(
            dtype=str,
            choices=("basic", "advanced"),
            default="basic",
        ),
    }
    pc = ParameterCanonicalizer(
        "test_op",
        [],
        param_specs=specs,
        production_mode=True,
    )
    # mode defaults to "basic" -> adjustment is inactive
    # Binding adjustment to non-default "log" must be rejected
    with pytest.raises(ValueError, match="inactive when"):
        pc.canonical_key({"window": 20, "adjustment": "log"})

    with pytest.raises(ValueError, match="dead knob"):
        pc.canonical_key({"window": 20, "adjustment": "log"})


def test_production_mode_allows_active_parameter():
    """Production mode allows active parameter to be varied."""
    specs = {
        "window": _SpecStub(dtype=int, min=2, max=100),
        "adjustment": _SpecStub(
            dtype=str,
            choices=("none", "log", "sqrt"),
            active_when=("mode", ("advanced",)),
            default="none",
        ),
        "mode": _SpecStub(
            dtype=str,
            choices=("basic", "advanced"),
            default="basic",
        ),
    }
    pc = ParameterCanonicalizer(
        "test_op",
        [],
        param_specs=specs,
        production_mode=True,
    )
    # mode="advanced" -> adjustment is ACTIVE and can be varied
    key = pc.canonical_key({"window": 20, "mode": "advanced", "adjustment": "sqrt"})
    assert ("adjustment", "'sqrt'") in key


def test_research_mode_allows_inactive_parameter_with_non_default():
    """Research mode preserves the legacy behavior: inactive parameter with
    non-default value is still rejected (R10 #18 applies to both modes)."""
    specs = {
        "window": _SpecStub(dtype=int, min=2, max=100),
        "adjustment": _SpecStub(
            dtype=str,
            choices=("none", "log", "sqrt"),
            active_when=("mode", ("advanced",)),
            default="none",
        ),
        "mode": _SpecStub(
            dtype=str,
            choices=("basic", "advanced"),
            default="basic",
        ),
    }
    pc = ParameterCanonicalizer(
        "test_op",
        [],
        param_specs=specs,
        production_mode=False,
    )
    # R10 #18 dead-knob rejection applies to both production and research mode
    with pytest.raises(ValueError, match="inactive when"):
        pc.canonical_key({"window": 20, "adjustment": "log"})


def test_production_mode_multiple_active_when_parameters():
    """Production mode with multiple active_when parameters must check all."""
    specs = {
        "window": _SpecStub(dtype=int, min=2, max=100),
        "adjustment": _SpecStub(
            dtype=str,
            active_when=("mode", ("advanced",)),
            default="none",
        ),
        "smoothing": _SpecStub(
            dtype=str,
            active_when=("mode", ("advanced",)),
            default="linear",
        ),
        "mode": _SpecStub(
            dtype=str,
            choices=("basic", "advanced"),
            default="basic",
        ),
    }
    pc = ParameterCanonicalizer(
        "test_op",
        [],
        param_specs=specs,
        production_mode=True,
    )
    # Both adjustment and smoothing inactive with mode="basic"
    key = pc.canonical_key({"window": 20, "adjustment": "none", "smoothing": "linear"})
    assert ("window", "20") in key

    # Both active with mode="advanced"
    key2 = pc.canonical_key({
        "window": 20,
        "mode": "advanced",
        "adjustment": "log",
        "smoothing": "spline",
    })
    assert ("adjustment", "'log'") in key2
    assert ("smoothing", "'spline'") in key2


def test_production_mode_controller_none_explicit_vs_missing():
    """Production mode must distinguish explicit None from missing controller."""
    specs = {
        "window": _SpecStub(dtype=int, min=2, max=100),
        "adjustment": _SpecStub(
            dtype=str,
            active_when=("mode", ("advanced",)),
            default="none",
        ),
        "mode": _SpecStub(dtype=str, default="basic"),
    }
    pc = ParameterCanonicalizer(
        "test_op",
        [],
        param_specs=specs,
        production_mode=True,
    )
    # Explicit mode=None (if allowed by choices) vs omitted
    # With controller having a default, omitted -> use default
    key = pc.canonical_key({"window": 20, "adjustment": "none"})
    assert ("window", "20") in key


def test_default_mode_is_research_not_production():
    """Verify default production_mode=False preserves legacy behavior."""
    specs = {
        "window": _SpecStub(dtype=int, min=2, max=100),
        "adjustment": _SpecStub(
            dtype=str,
            active_when=("mode", ("advanced",)),
            default="none",
        ),
        "mode": _SpecStub(dtype=str, choices=("basic", "advanced")),
    }
    # Default: production_mode not specified (defaults to False)
    pc = ParameterCanonicalizer("test_op", [], param_specs=specs)
    # Should NOT raise (fail-open legacy behavior)
    key = pc.canonical_key({"window": 20, "adjustment": "log"})
    assert ("window", "20") in key


def test_production_mode_param_without_active_when_unaffected():
    """Production mode only affects active_when validation; parameters without
    active_when are unaffected."""
    specs = {
        "window": _SpecStub(dtype=int, min=2, max=100),
        "alpha": _SpecStub(dtype=float, min=0.0, max=1.0),
    }
    pc = ParameterCanonicalizer(
        "test_op",
        [],
        param_specs=specs,
        production_mode=True,
    )
    key = pc.canonical_key({"window": 20, "alpha": 0.5})
    assert ("window", "20") in key
    assert ("alpha", "0.5") in key
