"""Integration tests for complexity profile with FE adapter."""

import pytest
from factor_optimizer.complexity.profile import ComplexityProfile, ComplexityEstimator
from factor_optimizer.complexity.budget import ComplexityBudget, BudgetTracker


def test_estimator_with_fe_adapter_mock():
    """Test complexity estimation with mock FE adapter."""

    class MockFEAdapter:
        """Mock FE adapter that simulates complexity analysis."""

        def estimate_complexity(self, factor_definition):
            """Simulate FE complexity estimation."""
            # Simulate FE returning detailed complexity info
            return {
                "operator_count": 8,
                "max_depth": 4,
                "lookback_periods": 60,
                "stateful_operators": 2,
                "cross_sectional_operators": 1,
                "nonlinear_operators": 1,
                "estimated_cost": 25.0,
                "domains": ["equity", "market"],
                "sources": ["price", "volume"],
                "estimated_latency_ms": 150.0,
                "memory_estimate": 50.0,
            }

    estimator = ComplexityEstimator(fe_adapter=MockFEAdapter())
    profile = estimator.estimate("rolling(close, 20).rank().ewm(10)")

    assert profile.operator_count == 8
    assert profile.max_depth == 4
    assert profile.lookback_periods == 60
    assert profile.estimated_cost == 25.0
    assert "equity" in profile.domains


def test_estimator_with_object_result():
    """Test complexity estimation when FE returns object with to_dict."""

    class FEComplexityResult:
        """Mock FE result object."""

        def to_dict(self):
            return {
                "operator_count": 5,
                "max_depth": 3,
                "lookback_periods": 20,
                "estimated_cost": 10.0,
            }

    class MockAdapter:
        def estimate_complexity(self, factor_def):
            return FEComplexityResult()

    estimator = ComplexityEstimator(fe_adapter=MockAdapter())
    profile = estimator.estimate("some_factor")

    assert profile.operator_count == 5
    assert profile.estimated_cost == 10.0


def test_estimator_fallback_for_unknown_format():
    """Test estimator fallback when FE returns unknown format."""

    class MockAdapter:
        def estimate_complexity(self, factor_def):
            return "unknown_format"

    estimator = ComplexityEstimator(fe_adapter=MockAdapter())
    profile = estimator.estimate("some_factor")

    # Should fallback to minimal profile
    assert profile.operator_count == 1
    assert profile.max_depth == 1
    assert profile.estimated_cost == 1.0
    assert profile.metadata["fe_result"] == "unknown_format"


def test_estimator_adapter_without_method():
    """Test estimator with adapter lacking estimate_complexity method."""

    class IncompleteAdapter:
        pass

    estimator = ComplexityEstimator(fe_adapter=IncompleteAdapter())
    profile = estimator.estimate("some_factor")

    # Should return minimal profile as fallback
    assert profile.operator_count == 1
    assert profile.max_depth == 1
    assert profile.estimated_cost == 1.0


def test_end_to_end_with_budget():
    """Test end-to-end workflow: estimate -> check budget -> track."""

    class MockFEAdapter:
        def estimate_complexity(self, factor_definition):
            # Simulate complexity based on factor name
            if "simple" in factor_definition:
                return {"operator_count": 3, "estimated_cost": 15.0, "lookback_periods": 10}
            elif "complex" in factor_definition:
                return {"operator_count": 15, "estimated_cost": 150.0, "lookback_periods": 200}
            else:
                return {"operator_count": 5, "estimated_cost": 30.0, "lookback_periods": 50}

    estimator = ComplexityEstimator(fe_adapter=MockFEAdapter())
    budget = ComplexityBudget(max_cost=100.0, max_operator_count=10, strict=False)
    tracker = BudgetTracker(budget=budget)

    # Estimate and track multiple factors
    factors = ["simple_factor", "medium_factor", "complex_factor"]
    results = []

    for factor in factors:
        profile = estimator.estimate(factor)
        is_within = tracker.record(profile, candidate_id=factor)
        results.append((factor, is_within))

    # Check results
    assert results[0][1] is True  # simple within budget
    assert results[1][1] is True  # medium within budget
    assert results[2][1] is False  # complex exceeds budget

    # Check tracker stats
    stats = tracker.stats()
    assert stats["total_evaluated"] == 3
    assert stats["violations_count"] == 1
    assert stats["max_observed_cost"] == 150.0


def test_mutation_delta_with_budget_enforcement():
    """Test mutation delta estimation with budget enforcement."""
    estimator = ComplexityEstimator()
    budget = ComplexityBudget(max_cost=100.0, strict=True)

    parent = ComplexityProfile(
        operator_count=5,
        lookback_periods=20,
        estimated_cost=40.0,
    )

    # Try window adjustment that would exceed budget
    new_profile = estimator.estimate_mutation_delta(
        parent, "window_adjust", {"new_window": 200}
    )

    # Cost should scale with window
    assert new_profile.lookback_periods == 200
    assert new_profile.estimated_cost > budget.max_cost

    # Budget should reject it
    with pytest.raises(Exception):  # BudgetExceededError
        budget.enforce(new_profile)


def test_complexity_profile_with_memory_estimate():
    """Test ComplexityProfile with memory estimate field."""
    profile = ComplexityProfile(
        operator_count=10,
        estimated_cost=50.0,
        memory_estimate=128.0,  # 128 MB
    )

    assert profile.memory_estimate == 128.0

    # Serialize and deserialize
    serialized = profile.to_dict()
    assert serialized["memory_estimate"] == 128.0

    restored = ComplexityProfile.from_dict(serialized)
    assert restored.memory_estimate == 128.0


def test_budget_with_memory_constraint():
    """Test budget enforcement with memory constraint."""
    budget = ComplexityBudget(
        max_cost=100.0,
        max_memory_mb=100.0,
        strict=True,
    )

    # Profile within memory budget
    profile_ok = ComplexityProfile(
        estimated_cost=50.0,
        metadata={"memory_estimate_mb": 80.0},
    )
    assert budget.is_within_budget(profile_ok) is True

    # Profile exceeding memory budget
    profile_exceed = ComplexityProfile(
        estimated_cost=50.0,
        metadata={"memory_estimate_mb": 150.0},
    )
    assert budget.is_within_budget(profile_exceed) is False


def test_complexity_estimator_chain():
    """Test chaining complexity estimation through mutations."""
    estimator = ComplexityEstimator()

    # Start with base factor
    base = ComplexityProfile(
        operator_count=3,
        lookback_periods=20,
        estimated_cost=10.0,
    )

    # Apply sequence of mutations
    step1 = estimator.estimate_mutation_delta(base, "window_adjust", {"new_window": 40})
    step2 = estimator.estimate_mutation_delta(step1, "linear_combination", {})
    step3 = estimator.estimate_mutation_delta(step2, "parameter_tune", {})

    # Each step should track metadata
    assert step1.metadata["mutation_type"] == "window_adjust"
    assert step2.metadata["mutation_type"] == "linear_combination"
    assert step3.metadata["mutation_type"] == "parameter_tune"

    # Complexity should generally increase
    assert step3.estimated_cost >= base.estimated_cost


def test_tracker_violation_details():
    """Test that tracker captures detailed violation information."""
    budget = ComplexityBudget(
        max_cost=50.0,
        max_operator_count=5,
        strict=False,
    )
    tracker = BudgetTracker(budget=budget)

    profile = ComplexityProfile(
        operator_count=10,
        estimated_cost=100.0,
    )

    tracker.record(profile, "violated_candidate")

    assert len(tracker.violations) == 1
    violation = tracker.violations[0]

    assert violation["candidate_id"] == "violated_candidate"
    assert violation["profile"] == profile
    assert "failed_checks" in violation
    assert violation["failed_checks"]["cost"] is False
    assert violation["failed_checks"]["operator_count"] is False
