"""Tests for SearchBudget and BudgetTracker."""

import pytest
from factor_optimizer.contracts.search_budget import SearchBudget, BudgetTracker


def test_search_budget_creation():
    """Test SearchBudget creation with defaults."""
    budget = SearchBudget()
    assert budget.max_trials == 100
    assert budget.max_evaluations == 50
    assert budget.max_cost_units == 1000.0
    assert budget.max_llm_calls is None


def test_search_budget_custom():
    """Test SearchBudget with custom values."""
    budget = SearchBudget(max_trials=10, max_evaluations=5, max_cost_units=100.0, max_llm_calls=20)
    assert budget.max_trials == 10
    assert budget.max_evaluations == 5
    assert budget.max_cost_units == 100.0
    assert budget.max_llm_calls == 20


def test_search_budget_validation():
    """Test SearchBudget validates constraints."""
    with pytest.raises(ValueError, match="max_trials must be >= 1"):
        SearchBudget(max_trials=0)

    with pytest.raises(ValueError, match="max_evaluations must be >= 1"):
        SearchBudget(max_evaluations=0)

    with pytest.raises(ValueError, match="max_cost_units must be > 0"):
        SearchBudget(max_cost_units=0.0)


def test_budget_tracker_initialization():
    """Test BudgetTracker initialization."""
    budget = SearchBudget(max_trials=10, max_evaluations=5)
    tracker = BudgetTracker(budget)

    assert tracker.trials_used == 0
    assert tracker.evaluations_used == 0
    assert tracker.cost_used == 0.0
    assert tracker.llm_calls_used == 0


def test_budget_tracker_trial_recording():
    """Test recording trials."""
    budget = SearchBudget(max_trials=3)
    tracker = BudgetTracker(budget)

    assert tracker.can_propose_trial()
    tracker.record_trial()
    assert tracker.trials_used == 1

    tracker.record_trial()
    tracker.record_trial()
    assert tracker.trials_used == 3
    assert not tracker.can_propose_trial()


def test_budget_tracker_evaluation_recording():
    """Test recording evaluations."""
    budget = SearchBudget(max_evaluations=2, max_cost_units=10.0)
    tracker = BudgetTracker(budget)

    assert tracker.can_evaluate()
    tracker.record_evaluation(cost=3.0)
    assert tracker.evaluations_used == 1
    assert tracker.cost_used == 3.0

    tracker.record_evaluation(cost=2.5)
    assert tracker.evaluations_used == 2
    assert not tracker.can_evaluate()


def test_budget_tracker_cost_limits():
    """Test cost budget enforcement."""
    budget = SearchBudget(max_cost_units=10.0, max_evaluations=100)
    tracker = BudgetTracker(budget)

    assert tracker.can_spend(5.0)
    tracker.record_evaluation(cost=5.0)

    assert tracker.can_spend(5.0)
    assert not tracker.can_spend(6.0)

    tracker.record_evaluation(cost=5.0)
    assert not tracker.can_spend(1.0)


def test_budget_tracker_exhaustion():
    """Test budget exhaustion detection."""
    budget = SearchBudget(max_trials=2, max_evaluations=2, max_cost_units=10.0)
    tracker = BudgetTracker(budget)

    assert not tracker.is_exhausted()

    tracker.record_trial()
    tracker.record_trial()
    assert tracker.is_exhausted()  # Trials exhausted


def test_budget_tracker_remaining():
    """Test remaining budget calculations."""
    budget = SearchBudget(max_trials=10, max_evaluations=5)
    tracker = BudgetTracker(budget)

    tracker.record_trial()
    tracker.record_trial()
    assert tracker.remaining_trials() == 8

    tracker.record_evaluation()
    assert tracker.remaining_evaluations() == 4


def test_budget_tracker_utilization_report():
    """Test utilization report generation."""
    budget = SearchBudget(max_trials=10, max_evaluations=5, max_cost_units=100.0)
    tracker = BudgetTracker(budget)

    tracker.record_trial()
    tracker.record_evaluation(cost=10.0)

    report = tracker.utilization_report()
    assert report["trials"]["used"] == 1
    assert report["trials"]["max"] == 10
    assert report["evaluations"]["used"] == 1
    assert report["evaluations"]["max"] == 5
    assert report["cost"]["used"] == 10.0
    assert report["cost"]["max"] == 100.0
    assert report["exhausted"] is False


def test_budget_tracker_llm_calls():
    """Test LLM call tracking."""
    budget = SearchBudget(max_llm_calls=3)
    tracker = BudgetTracker(budget)

    assert tracker.can_call_llm()
    tracker.record_llm_call()
    tracker.record_llm_call()
    tracker.record_llm_call()
    assert not tracker.can_call_llm()
    assert tracker.llm_calls_used == 3


def test_budget_tracker_no_llm_limit():
    """Test when LLM limit is not set."""
    budget = SearchBudget()  # No max_llm_calls
    tracker = BudgetTracker(budget)

    assert not tracker.can_call_llm()  # Should return False when unlimited
