"""Tests for ComplexityProfile and ComplexityEstimator."""

import pytest
from factor_optimizer.complexity.profile import ComplexityProfile, ComplexityEstimator


def test_complexity_profile_creation():
    """Test ComplexityProfile creation."""
    profile = ComplexityProfile(
        operator_count=5,
        max_depth=3,
        lookback_periods=20,
        estimated_cost=10.0,
    )

    assert profile.operator_count == 5
    assert profile.max_depth == 3
    assert profile.lookback_periods == 20
    assert profile.estimated_cost == 10.0


def test_complexity_profile_defaults():
    """Test ComplexityProfile default values."""
    profile = ComplexityProfile()

    assert profile.operator_count == 0
    assert profile.max_depth == 0
    assert profile.estimated_cost == 0.0
    assert len(profile.domains) == 0


def test_complexity_profile_serialization():
    """Test ComplexityProfile serialization."""
    profile = ComplexityProfile(
        operator_count=3,
        max_depth=2,
        domains=["equity", "futures"],
        sources=["market", "fundamental"],
    )

    serialized = profile.to_dict()
    assert serialized["operator_count"] == 3
    assert serialized["domains"] == ["equity", "futures"]

    restored = ComplexityProfile.from_dict(serialized)
    assert restored.operator_count == profile.operator_count
    assert restored.domains == profile.domains


def test_complexity_profile_budget_check():
    """Test budget checking."""
    profile = ComplexityProfile(estimated_cost=50.0)

    assert profile.is_within_budget(100.0)
    assert profile.is_within_budget(50.0)
    assert not profile.is_within_budget(25.0)


def test_complexity_estimator_no_adapter():
    """Test estimator fails without adapter."""
    estimator = ComplexityEstimator()

    with pytest.raises(RuntimeError, match="FE adapter required"):
        estimator.estimate("some_factor")


def test_complexity_estimator_with_mock_adapter():
    """Test estimator with mock adapter."""

    class MockAdapter:
        def estimate_complexity(self, factor_def):
            return {
                "operator_count": 2,
                "max_depth": 1,
                "estimated_cost": 5.0,
            }

    estimator = ComplexityEstimator(fe_adapter=MockAdapter())
    profile = estimator.estimate("factor_definition")

    assert profile.operator_count == 2
    assert profile.max_depth == 1
    assert profile.estimated_cost == 5.0


def test_complexity_estimator_mutation_delta_window():
    """Test mutation complexity delta for window adjustment."""
    parent = ComplexityProfile(
        operator_count=5,
        lookback_periods=20,
        estimated_cost=10.0,
    )

    estimator = ComplexityEstimator()
    new_profile = estimator.estimate_mutation_delta(
        parent, "window_adjust", {"new_window": 40}
    )

    # Window doubled, cost should roughly double
    assert new_profile.lookback_periods == 40
    assert new_profile.estimated_cost > parent.estimated_cost


def test_complexity_estimator_mutation_delta_composition():
    """Test mutation complexity delta for composition."""
    parent = ComplexityProfile(
        operator_count=3,
        max_depth=2,
        estimated_cost=5.0,
    )

    estimator = ComplexityEstimator()
    new_profile = estimator.estimate_mutation_delta(
        parent, "linear_combination", {"weight_a": 0.5, "weight_b": 0.5}
    )

    # Composition adds operator and increases cost
    assert new_profile.operator_count == parent.operator_count + 1
    assert new_profile.max_depth >= parent.max_depth
    assert new_profile.estimated_cost > parent.estimated_cost


def test_complexity_estimator_mutation_delta_parameter_tune():
    """Test mutation complexity delta for parameter tuning."""
    parent = ComplexityProfile(
        operator_count=3,
        estimated_cost=5.0,
    )

    estimator = ComplexityEstimator()
    new_profile = estimator.estimate_mutation_delta(
        parent, "parameter_tune", {"parameter_name": "decay", "new_value": 0.9}
    )

    # Parameter tuning doesn't change structure much
    assert new_profile.operator_count == parent.operator_count
    assert new_profile.estimated_cost >= parent.estimated_cost


def test_complexity_estimator_metadata_tracking():
    """Test that mutation deltas track metadata."""
    parent = ComplexityProfile()

    estimator = ComplexityEstimator()
    new_profile = estimator.estimate_mutation_delta(
        parent, "window_adjust", {"new_window": 50}
    )

    assert "mutation_type" in new_profile.metadata
    assert new_profile.metadata["mutation_type"] == "window_adjust"
    assert new_profile.metadata["estimated_from_heuristics"] is True
