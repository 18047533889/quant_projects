# -*- coding: utf-8 -*-
"""FE-P0-021 — active_when controller missing/undecidable fails closed in production.

When an active_when controller is missing (not in params) and has no decidable
default, the production canonicalizer must raise ParameterContractError rather
than silently skipping validation (fail-open).

Research mode preserves the legacy fail-open behavior for backward compatibility.
Production mode is determined by resolve_run_mode(), which reads from:
  - Explicit run_mode parameter
  - FACTOR_ENGINE_RUN_MODE environment variable
  - QUANT_PRODUCTION_MODE environment variable
  - Default: "research"
"""
from __future__ import annotations

import os

import pytest

from factor_engine.parameter_canonicalizer import (
    ParameterCanonicalizer,
    ParameterContractError,
    ParamNormalizer,
)


class _SpecStub:
    """Duck-typed ParamSpec for testing."""

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


_MISSING = object()


# ---------------------------------------------------------------------------
# FE-P0-021: production mode fails closed on missing controller
# ---------------------------------------------------------------------------


def test_fe_p0_021_production_mode_missing_controller_no_default_raises():
    """Production: missing controller with no default → ParameterContractError."""
    specs = {
        "mode": _SpecStub(choices=["A", "B"]),  # controller has NO default
        "alpha": _SpecStub(active_when=("mode", "A"), default=0.5),
    }
    pc = ParameterCanonicalizer(
        "test_op",
        [],
        param_specs=specs,
        run_mode="production",  # explicit production mode
    )
    # Controller 'mode' is missing and has no default -> fail closed
    with pytest.raises(ParameterContractError) as exc_info:
        pc.canonical_key({"alpha": 0.7})
    assert "active_when controller 'mode'" in str(exc_info.value)
    assert "missing and has no decidable default" in str(exc_info.value)
    assert "FE-P0-021" in str(exc_info.value)


def test_fe_p0_021_production_mode_with_controller_value_succeeds():
    """Production: explicit controller value allows validation."""
    specs = {
        "mode": _SpecStub(choices=["A", "B"]),
        "alpha": _SpecStub(active_when=("mode", "A"), default=0.5),
    }
    pc = ParameterCanonicalizer("test_op", [], param_specs=specs, run_mode="production")
    # Controller is explicit -> validation proceeds
    key = pc.canonical_key({"mode": "A", "alpha": 0.7})
    assert key is not None  # succeeds


def test_fe_p0_021_production_mode_with_controller_default_succeeds():
    """Production: controller with decidable default allows validation."""
    specs = {
        "mode": _SpecStub(choices=["A", "B"], default="A"),  # has default
        "alpha": _SpecStub(active_when=("mode", "A"), default=0.5),
    }
    pc = ParameterCanonicalizer("test_op", [], param_specs=specs, run_mode="production")
    # Controller has default -> can decide alpha is active
    key = pc.canonical_key({"alpha": 0.7})
    assert key is not None  # succeeds


def test_fe_p0_021_production_mode_inactive_param_at_default_allowed():
    """Production: inactive parameter at its canonical default is allowed."""
    specs = {
        "mode": _SpecStub(choices=["A", "B"], default="B"),
        "alpha": _SpecStub(active_when=("mode", "A"), default=0.5),
    }
    pc = ParameterCanonicalizer("test_op", [], param_specs=specs, run_mode="production")
    # mode=B (default) makes alpha inactive, but alpha is at its default
    key = pc.canonical_key({"alpha": 0.5})
    assert key is not None  # allowed


def test_fe_p0_021_production_mode_inactive_param_non_default_raises():
    """Production: inactive parameter at non-default value raises."""
    specs = {
        "mode": _SpecStub(choices=["A", "B"], default="B"),
        "alpha": _SpecStub(active_when=("mode", "A"), default=0.5),
    }
    pc = ParameterCanonicalizer("test_op", [], param_specs=specs, run_mode="production")
    # mode=B makes alpha inactive, but alpha is non-default
    with pytest.raises(ValueError) as exc_info:
        pc.canonical_key({"alpha": 0.9})
    assert "parameter 'alpha' is inactive" in str(exc_info.value)


# ---------------------------------------------------------------------------
# FE-P0-021: research mode preserves fail-open behavior
# ---------------------------------------------------------------------------


def test_fe_p0_021_research_mode_missing_controller_no_default_skip_validation():
    """Research (default): missing controller with no default skips validation (fail-open)."""
    specs = {
        "mode": _SpecStub(choices=["A", "B"]),  # no default
        "alpha": _SpecStub(active_when=("mode", "A"), default=0.5),
    }
    pc = ParameterCanonicalizer(
        "test_op",
        [],
        param_specs=specs,
        # No run_mode specified -> defaults to "research"
    )
    # Controller missing with no default -> skips validation (legacy behavior)
    key = pc.canonical_key({"alpha": 0.7})
    assert key is not None  # succeeds (fail-open)


def test_fe_p0_021_research_mode_explicit():
    """Research mode can be explicitly specified."""
    specs = {
        "mode": _SpecStub(choices=["A", "B"]),
        "alpha": _SpecStub(active_when=("mode", "A"), default=0.5),
    }
    pc = ParameterCanonicalizer(
        "test_op",
        [],
        param_specs=specs,
        run_mode="research",  # explicit research mode
    )
    # Same as default behavior
    key = pc.canonical_key({"alpha": 0.7})
    assert key is not None


# ---------------------------------------------------------------------------
# FE-P0-021: environment-driven production mode
# ---------------------------------------------------------------------------


def test_fe_p0_021_production_from_factor_engine_run_mode_env(monkeypatch):
    """Production mode from FACTOR_ENGINE_RUN_MODE env."""
    monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")
    specs = {
        "mode": _SpecStub(choices=["A", "B"]),
        "alpha": _SpecStub(active_when=("mode", "A"), default=0.5),
    }
    # No explicit run_mode -> resolve_run_mode() reads env
    pc = ParameterCanonicalizer("test_op", [], param_specs=specs)
    with pytest.raises(ParameterContractError) as exc_info:
        pc.canonical_key({"alpha": 0.7})
    assert "FE-P0-021" in str(exc_info.value)


def test_fe_p0_021_production_from_quant_production_mode_env(monkeypatch):
    """Production mode from QUANT_PRODUCTION_MODE env."""
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    specs = {
        "mode": _SpecStub(choices=["A", "B"]),
        "alpha": _SpecStub(active_when=("mode", "A"), default=0.5),
    }
    pc = ParameterCanonicalizer("test_op", [], param_specs=specs)
    with pytest.raises(ParameterContractError) as exc_info:
        pc.canonical_key({"alpha": 0.7})
    assert "FE-P0-021" in str(exc_info.value)


def test_fe_p0_021_explicit_run_mode_overrides_env(monkeypatch):
    """Explicit run_mode parameter takes precedence over environment."""
    monkeypatch.setenv("FACTOR_ENGINE_RUN_MODE", "production")
    specs = {
        "mode": _SpecStub(choices=["A", "B"]),
        "alpha": _SpecStub(active_when=("mode", "A"), default=0.5),
    }
    # Explicit research mode overrides env production
    pc = ParameterCanonicalizer("test_op", [], param_specs=specs, run_mode="research")
    key = pc.canonical_key({"alpha": 0.7})
    assert key is not None  # succeeds (research mode)
