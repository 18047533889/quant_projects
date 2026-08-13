"""Tests for ComplexityBudget and BudgetTracker."""

import pytest
from factor_optimizer.complexity.budget import (
    ComplexityBudget,
    BudgetTracker,
    create_default_budget,
)
from factor_optimizer.complexity.profile import ComplexityProfile
from factor_optimizer.errors import BudgetExceededError


def test_complexity_budget_creation():
    """Test ComplexityBudget creation."""
    budget = ComplexityBudget(
        max_cost=100.0,
        max_operator_count=10,
        max_lookback_periods=252,
    )

    assert budget.max_cost == 100.0
    assert budget.max_operator_count == 10
    assert budget.max_lookback_periods == 252
    assert budget.strict is True


def test_complexity_budget_check_within():
    """Test budget check for profile within budget."""
    budget = ComplexityBudget(
        max_cost=100.0,
        max_operator_count=10,
        max_lookback_periods=252,
    )

    profile = ComplexityProfile(
        operator_count=5,
        lookback_periods=20,
        estimated_cost=50.0,
    )

    results = budget.check(profile)
    assert results["cost"] is True
    assert results["operator_count"] is True
    assert results["lookback_periods"] is True
    assert budget.is_within_budget(profile) is True


def test_complexity_budget_check_exceeded():
    """Test budget check for profile exceeding budget."""
    budget = ComplexityBudget(
        max_cost=50.0,
        max_operator_count=5,
    )

    profile = ComplexityProfile(
        operator_count=10,
        estimated_cost=100.0,
    )

    results = budget.check(profile)
    assert results["cost"] is False
    assert results["operator_count"] is False
    assert budget.is_within_budget(profile) is False


def test_complexity_budget_partial_constraints():
    """Test budget with only some constraints specified."""
    budget = ComplexityBudget(max_cost=100.0)

    profile = ComplexityProfile(
        operator_count=1000,  # No constraint on this
        estimated_cost=50.0,
    )

    results = budget.check(profile)
    assert "cost" in results
    assert "operator_count" not in results
    assert budget.is_within_budget(profile) is True


def test_complexity_budget_enforce_strict():
    """Test budget enforcement in strict mode."""
    budget = ComplexityBudget(max_cost=50.0, strict=True)

    profile = ComplexityProfile(estimated_cost=100.0)

    with pytest.raises(BudgetExceededError, match="Complexity budget exceeded"):
        budget.enforce(profile)


def test_complexity_budget_enforce_non_strict():
    """Test budget enforcement in non-strict mode."""
    budget = ComplexityBudget(max_cost=50.0, strict=False)

    profile = ComplexityProfile(estimated_cost=100.0)

    # Should not raise in non-strict mode
    budget.enforce(profile)


def test_complexity_budget_memory_check():
    """Test budget check with memory estimate."""
    budget = ComplexityBudget(max_memory_mb=1000.0)

    profile = ComplexityProfile(
        estimated_cost=50.0,
        metadata={"memory_estimate_mb": 500.0},
    )

    results = budget.check(profile)
    assert results["memory"] is True
    assert budget.is_within_budget(profile) is True

    # Exceed memory budget
    profile_large = ComplexityProfile(
        estimated_cost=50.0,
        metadata={"memory_estimate_mb": 2000.0},
    )

    results_large = budget.check(profile_large)
    assert results_large["memory"] is False


def test_budget_tracker_record():
    """Test BudgetTracker recording."""
    budget = ComplexityBudget(max_cost=100.0, strict=False)
    tracker = BudgetTracker(budget=budget)

    profile1 = ComplexityProfile(estimated_cost=50.0)
    profile2 = ComplexityProfile(estimated_cost=150.0)

    assert tracker.record(profile1, "candidate_1") is True
    assert tracker.record(profile2, "candidate_2") is False

    assert len(tracker.history) == 2
    assert len(tracker.violations) == 1
    assert tracker.violations[0]["candidate_id"] == "candidate_2"


def test_budget_tracker_enforce_strict():
    """Test BudgetTracker enforcement in strict mode."""
    budget = ComplexityBudget(max_cost=100.0, strict=True)
    tracker = BudgetTracker(budget=budget)

    profile_ok = ComplexityProfile(estimated_cost=50.0)
    profile_exceed = ComplexityProfile(estimated_cost=150.0)

    # First one should pass
    tracker.enforce(profile_ok, "candidate_1")

    # Second should raise
    with pytest.raises(BudgetExceededError):
        tracker.enforce(profile_exceed, "candidate_2")

    # History should still record both attempts
    assert len(tracker.history) == 2


def test_budget_tracker_stats_empty():
    """Test BudgetTracker stats with no history."""
    budget = ComplexityBudget(max_cost=100.0)
    tracker = BudgetTracker(budget=budget)

    stats = tracker.stats()
    assert stats["total_evaluated"] == 0
    assert stats["violations_count"] == 0
    assert stats["violation_rate"] == 0.0


def test_budget_tracker_stats_with_data():
    """Test BudgetTracker stats with recorded profiles."""
    budget = ComplexityBudget(
        max_cost=100.0,
        max_operator_count=10,
        strict=False,
    )
    tracker = BudgetTracker(budget=budget)

    profiles = [
        ComplexityProfile(operator_count=5, estimated_cost=50.0),
        ComplexityProfile(operator_count=8, estimated_cost=80.0),
        ComplexityProfile(operator_count=15, estimated_cost=150.0),  # Exceeds both
    ]

    for i, profile in enumerate(profiles):
        tracker.record(profile, f"candidate_{i}")

    stats = tracker.stats()
    assert stats["total_evaluated"] == 3
    assert stats["violations_count"] == 1
    assert stats["violation_rate"] == pytest.approx(1.0 / 3.0)
    assert stats["max_observed_cost"] == 150.0
    assert stats["max_observed_operators"] == 15
    assert stats["budget_cost_utilization"] == pytest.approx(1.5)
    assert stats["budget_operators_utilization"] == pytest.approx(1.5)


def test_budget_tracker_stats_utilization():
    """Test BudgetTracker utilization metrics."""
    budget = ComplexityBudget(max_cost=200.0, max_operator_count=20)
    tracker = BudgetTracker(budget=budget)

    # All within budget
    profiles = [
        ComplexityProfile(operator_count=10, estimated_cost=100.0),
        ComplexityProfile(operator_count=15, estimated_cost=150.0),
    ]

    for profile in profiles:
        tracker.record(profile)

    stats = tracker.stats()
    assert stats["budget_cost_utilization"] == pytest.approx(0.75)
    assert stats["budget_operators_utilization"] == pytest.approx(0.75)


def test_create_default_budget():
    """Test default budget creation."""
    budget = create_default_budget()

    assert budget.max_cost == 100.0
    assert budget.max_operator_count == 20
    assert budget.max_lookback_periods == 252
    assert budget.max_depth == 10
    assert budget.strict is True


def test_create_default_budget_with_multipliers():
    """Test default budget creation with multipliers."""
    budget = create_default_budget(
        cost_multiplier=2.0,
        operator_multiplier=0.5,
        strict=False,
    )

    assert budget.max_cost == 200.0
    assert budget.max_operator_count == 10
    assert budget.strict is False


def test_budget_check_all_dimensions():
    """Test budget checking across all dimensions."""
    budget = ComplexityBudget(
        max_cost=100.0,
        max_operator_count=10,
        max_lookback_periods=100,
        max_depth=5,
        max_stateful_operators=3,
    )

    profile = ComplexityProfile(
        operator_count=5,
        max_depth=3,
        lookback_periods=50,
        stateful_operators=2,
        estimated_cost=50.0,
    )

    results = budget.check(profile)
    assert all(results.values())


def test_budget_multiple_violations():
    """Test budget with multiple dimension violations."""
    budget = ComplexityBudget(
        max_cost=50.0,
        max_operator_count=5,
        max_lookback_periods=20,
        strict=True,
    )

    profile = ComplexityProfile(
        operator_count=10,
        lookback_periods=50,
        estimated_cost=100.0,
    )

    results = budget.check(profile)
    violations = [dim for dim, ok in results.items() if not ok]

    assert len(violations) == 3
    assert "cost" in violations
    assert "operator_count" in violations
    assert "lookback_periods" in violations

    with pytest.raises(BudgetExceededError) as exc_info:
        budget.enforce(profile)

    # Check all violations mentioned
    error_msg = str(exc_info.value)
    assert "cost" in error_msg
    assert "operator_count" in error_msg
    assert "lookback_periods" in error_msg


def test_budget_tracker_with_no_violations():
    """Test BudgetTracker with all profiles within budget."""
    budget = ComplexityBudget(max_cost=200.0, strict=True)
    tracker = BudgetTracker(budget=budget)

    profiles = [
        ComplexityProfile(estimated_cost=50.0),
        ComplexityProfile(estimated_cost=100.0),
        ComplexityProfile(estimated_cost=150.0),
    ]

    for profile in profiles:
        tracker.enforce(profile)

    assert len(tracker.history) == 3
    assert len(tracker.violations) == 0

    stats = tracker.stats()
    assert stats["violation_rate"] == 0.0
