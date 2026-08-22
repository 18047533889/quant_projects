# -*- coding: utf-8 -*-
"""MODEL-P0-003: EmbargoSpec contract tests.

This module tests the explicit embargo specification that enforces mandatory
temporal separation between training and validation/test to prevent leakage.

Embargo is distinct from label purge:
- Label purge: removes training rows whose label interval overlaps validation
- Embargo: adds extra buffer days to ensure complete temporal separation

EmbargoSpec.days must be >= label horizon to ensure labels are fully mature.
"""
from __future__ import annotations

import pytest

from modeling.contracts import EmbargoSpec, LabelContract


# --------------------------------------------------------------------------- #
# EmbargoSpec construction and validation
# --------------------------------------------------------------------------- #
def test_embargo_spec_basic_construction():
    """Basic EmbargoSpec can be constructed with required fields."""
    spec = EmbargoSpec(
        days=5,
        rationale="5-day buffer for label maturity and feature lag",
        applies_to="validation",
    )
    assert spec.days == 5
    assert spec.rationale == "5-day buffer for label maturity and feature lag"
    assert spec.applies_to == "validation"


def test_embargo_spec_applies_to_validation():
    """EmbargoSpec can target validation only."""
    spec = EmbargoSpec(days=3, rationale="validation buffer", applies_to="validation")
    assert spec.applies_to == "validation"


def test_embargo_spec_applies_to_test():
    """EmbargoSpec can target test only."""
    spec = EmbargoSpec(days=3, rationale="test buffer", applies_to="test")
    assert spec.applies_to == "test"


def test_embargo_spec_applies_to_both():
    """EmbargoSpec can target both validation and test."""
    spec = EmbargoSpec(days=7, rationale="universal buffer", applies_to="both")
    assert spec.applies_to == "both"


def test_embargo_spec_negative_days_raises():
    """Embargo days must be >= 0."""
    with pytest.raises(ValueError, match="embargo days must be >= 0"):
        EmbargoSpec(days=-5, rationale="invalid", applies_to="validation")


def test_embargo_spec_invalid_applies_to_raises():
    """applies_to must be one of the valid values."""
    with pytest.raises(ValueError, match="applies_to must be"):
        EmbargoSpec(days=5, rationale="test", applies_to="invalid")


def test_embargo_spec_zero_days_allowed():
    """Zero embargo days is allowed (no buffer)."""
    spec = EmbargoSpec(days=0, rationale="no embargo", applies_to="validation")
    assert spec.days == 0


# --------------------------------------------------------------------------- #
# validate_against_label: embargo must be >= label horizon
# --------------------------------------------------------------------------- #
def test_embargo_sufficient_for_label_horizon():
    """Embargo >= label horizon is valid."""
    label = LabelContract(label_name="ret_5", horizon_bars=5)
    embargo = EmbargoSpec(days=5, rationale="matches horizon", applies_to="validation")

    violations = embargo.validate_against_label(label)
    assert violations == [], f"Expected no violations, got {violations}"


def test_embargo_exceeds_label_horizon():
    """Embargo > label horizon is valid (extra safety margin)."""
    label = LabelContract(label_name="ret_5", horizon_bars=5)
    embargo = EmbargoSpec(days=10, rationale="extra buffer", applies_to="validation")

    violations = embargo.validate_against_label(label)
    assert violations == []


def test_embargo_less_than_label_horizon_fails():
    """Embargo < label horizon violates contract."""
    label = LabelContract(label_name="ret_10", horizon_bars=10)
    embargo = EmbargoSpec(days=5, rationale="insufficient", applies_to="validation")

    violations = embargo.validate_against_label(label)
    assert len(violations) == 1
    assert "embargo days 5 < label horizon 10" in violations[0]
    assert "embargo must be >= horizon" in violations[0]


def test_embargo_zero_with_nonzero_horizon_fails():
    """Zero embargo with non-zero horizon violates contract."""
    label = LabelContract(label_name="ret_3", horizon_bars=3)
    embargo = EmbargoSpec(days=0, rationale="no buffer", applies_to="validation")

    violations = embargo.validate_against_label(label)
    assert len(violations) == 1
    assert "embargo days 0 < label horizon 3" in violations[0]


def test_embargo_validation_with_overlapping_label():
    """Overlapping label (high horizon) requires large embargo."""
    label = LabelContract(label_name="ret_20", horizon_bars=20, overlapping=True)
    embargo = EmbargoSpec(days=15, rationale="insufficient for 20-bar label", applies_to="both")

    violations = embargo.validate_against_label(label)
    assert len(violations) == 1
    assert "embargo days 15 < label horizon 20" in violations[0]


def test_embargo_validation_multiple_labels():
    """Embargo can be validated against multiple label contracts."""
    embargo = EmbargoSpec(days=10, rationale="10-day buffer", applies_to="validation")

    # Valid for 5-day and 10-day labels
    label_5 = LabelContract(label_name="ret_5", horizon_bars=5)
    label_10 = LabelContract(label_name="ret_10", horizon_bars=10)

    assert embargo.validate_against_label(label_5) == []
    assert embargo.validate_against_label(label_10) == []

    # Invalid for 15-day label
    label_15 = LabelContract(label_name="ret_15", horizon_bars=15)
    violations = embargo.validate_against_label(label_15)
    assert len(violations) == 1


# --------------------------------------------------------------------------- #
# Rationale documentation
# --------------------------------------------------------------------------- #
def test_embargo_spec_requires_rationale():
    """EmbargoSpec requires rationale for auditability."""
    # rationale is a required field in the dataclass
    spec = EmbargoSpec(
        days=5,
        rationale="Ensure 5-day label is fully mature before validation boundary",
        applies_to="validation",
    )
    assert len(spec.rationale) > 0
    assert "label" in spec.rationale.lower() or "mature" in spec.rationale.lower()


def test_embargo_spec_rationale_examples():
    """Common rationale patterns for embargo."""
    # Label maturity
    spec1 = EmbargoSpec(
        days=5,
        rationale="5-day forward return label requires 5 days to mature",
        applies_to="validation",
    )
    assert "label" in spec1.rationale

    # Feature lag
    spec2 = EmbargoSpec(
        days=3,
        rationale="Fundamental data has 3-day publication lag",
        applies_to="both",
    )
    assert "lag" in spec2.rationale

    # Execution timing
    spec3 = EmbargoSpec(
        days=1,
        rationale="Next-day VWAP execution requires 1-day buffer",
        applies_to="test",
    )
    assert "execution" in spec3.rationale or "VWAP" in spec3.rationale


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
def test_embargo_spec_frozen_immutable():
    """EmbargoSpec is frozen (immutable)."""
    spec = EmbargoSpec(days=5, rationale="test", applies_to="validation")
    with pytest.raises(Exception):  # FrozenInstanceError in dataclasses
        spec.days = 10


def test_embargo_spec_equality():
    """EmbargoSpec equality is value-based."""
    spec1 = EmbargoSpec(days=5, rationale="buffer", applies_to="validation")
    spec2 = EmbargoSpec(days=5, rationale="buffer", applies_to="validation")
    spec3 = EmbargoSpec(days=5, rationale="buffer", applies_to="test")

    assert spec1 == spec2
    assert spec1 != spec3


def test_embargo_spec_large_days():
    """EmbargoSpec can handle large embargo periods."""
    spec = EmbargoSpec(days=252, rationale="one-year embargo for regime shift", applies_to="both")
    assert spec.days == 252
