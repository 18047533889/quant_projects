"""
Tests for computation budget tracking.
"""

import pytest
import time

from quant_evaluator.runtime.budgets import (
    ComputationBudget,
    ResourceUsage,
    BudgetTracker,
)


class TestComputationBudget:
    def test_budget_creation(self):
        budget = ComputationBudget(
            max_memory_mb=1024.0,
            max_time_seconds=60.0,
            max_operations=10000,
        )

        assert budget.max_memory_mb == 1024.0
        assert budget.max_time_seconds == 60.0
        assert budget.max_operations == 10000

    def test_has_limit(self):
        budget1 = ComputationBudget()
        assert not budget1.has_limit()

        budget2 = ComputationBudget(max_memory_mb=1024.0)
        assert budget2.has_limit()

        budget3 = ComputationBudget(max_time_seconds=60.0)
        assert budget3.has_limit()

        budget4 = ComputationBudget(max_operations=1000)
        assert budget4.has_limit()


class TestResourceUsage:
    def test_usage_creation(self):
        usage = ResourceUsage(
            memory_mb=512.0,
            elapsed_seconds=30.0,
            operations_count=5000,
        )

        assert usage.memory_mb == 512.0
        assert usage.elapsed_seconds == 30.0
        assert usage.operations_count == 5000

    def test_to_dict(self):
        usage = ResourceUsage(
            memory_mb=100.0,
            elapsed_seconds=10.0,
            operations_count=1000,
            peak_memory_mb=150.0,
        )

        data = usage.to_dict()

        assert data["memory_mb"] == 100.0
        assert data["elapsed_seconds"] == 10.0
        assert data["operations_count"] == 1000
        assert data["peak_memory_mb"] == 150.0
        assert "timestamp" in data


class TestBudgetTracker:
    def test_tracker_creation(self):
        budget = ComputationBudget(max_memory_mb=1024.0)
        tracker = BudgetTracker(budget)

        assert tracker.budget == budget

    def test_record_operation(self):
        budget = ComputationBudget()
        tracker = BudgetTracker(budget)

        tracker.record_operation(5)
        usage = tracker.get_current_usage()

        assert usage.operations_count == 5

        tracker.record_operation(3)
        usage = tracker.get_current_usage()

        assert usage.operations_count == 8

    def test_get_current_usage(self):
        budget = ComputationBudget()
        tracker = BudgetTracker(budget)

        time.sleep(0.1)
        usage = tracker.get_current_usage()

        assert usage.memory_mb >= 0
        assert usage.elapsed_seconds >= 0.1
        assert usage.operations_count == 0

    def test_peak_memory_tracking(self):
        budget = ComputationBudget()
        tracker = BudgetTracker(budget)

        usage1 = tracker.get_current_usage()
        peak1 = usage1.peak_memory_mb

        # Peak should be at least current memory
        assert peak1 >= usage1.memory_mb

    def test_check_budget_no_limits(self):
        budget = ComputationBudget()
        tracker = BudgetTracker(budget)

        tracker.record_operation(1000000)

        within_budget, reason = tracker.check_budget(raise_on_exceed=False)

        assert within_budget is True
        assert reason is None

    def test_check_budget_time_exceeded(self):
        budget = ComputationBudget(max_time_seconds=0.05)
        tracker = BudgetTracker(budget)

        time.sleep(0.1)

        within_budget, reason = tracker.check_budget(raise_on_exceed=False)

        assert within_budget is False
        assert reason is not None
        assert "Time budget exceeded" in reason

    def test_check_budget_time_exceeded_raises(self):
        budget = ComputationBudget(max_time_seconds=0.05, allow_overflow=False)
        tracker = BudgetTracker(budget)

        time.sleep(0.1)

        with pytest.raises(RuntimeError, match="Time budget exceeded"):
            tracker.check_budget(raise_on_exceed=True)

    def test_check_budget_operations_exceeded(self):
        budget = ComputationBudget(max_operations=100)
        tracker = BudgetTracker(budget)

        tracker.record_operation(150)

        within_budget, reason = tracker.check_budget(raise_on_exceed=False)

        assert within_budget is False
        assert "Operation budget exceeded" in reason

    def test_check_budget_operations_exceeded_raises(self):
        budget = ComputationBudget(max_operations=100, allow_overflow=False)
        tracker = BudgetTracker(budget)

        tracker.record_operation(150)

        with pytest.raises(RuntimeError, match="Operation budget exceeded"):
            tracker.check_budget(raise_on_exceed=True)

    def test_check_budget_allow_overflow(self):
        budget = ComputationBudget(
            max_operations=100,
            allow_overflow=True,
        )
        tracker = BudgetTracker(budget)

        tracker.record_operation(150)

        # Should not raise even with raise_on_exceed=True
        within_budget, reason = tracker.check_budget(raise_on_exceed=True)

        assert within_budget is False
        assert reason is not None

    def test_warning_threshold(self):
        budget = ComputationBudget(
            max_operations=100,
            warn_threshold=0.8,
        )
        tracker = BudgetTracker(budget)

        # At 85% of budget
        tracker.record_operation(85)

        within_budget, reason = tracker.check_budget(raise_on_exceed=False)

        assert within_budget is True
        assert reason is not None
        assert "approaching limit" in reason

    def test_warning_only_once(self):
        budget = ComputationBudget(
            max_operations=100,
            warn_threshold=0.8,
        )
        tracker = BudgetTracker(budget)

        tracker.record_operation(85)

        # First check should warn
        within_budget1, reason1 = tracker.check_budget(raise_on_exceed=False)
        assert reason1 is not None

        # Second check should not warn again
        within_budget2, reason2 = tracker.check_budget(raise_on_exceed=False)
        assert reason2 is None

    def test_get_remaining_budget(self):
        budget = ComputationBudget(
            max_memory_mb=1000.0,
            max_time_seconds=100.0,
            max_operations=10000,
        )
        tracker = BudgetTracker(budget)

        tracker.record_operation(3000)
        time.sleep(0.1)

        remaining = tracker.get_remaining_budget()

        assert "memory_mb" in remaining
        assert "memory_pct" in remaining
        assert "time_seconds" in remaining
        assert "time_pct" in remaining
        assert "operations" in remaining
        assert "operations_pct" in remaining

        assert remaining["operations"] == 7000
        assert remaining["operations_pct"] == 0.7

    def test_reset(self):
        budget = ComputationBudget(max_operations=1000)
        tracker = BudgetTracker(budget)

        tracker.record_operation(500)
        time.sleep(0.1)

        usage_before = tracker.get_current_usage()
        assert usage_before.operations_count == 500
        assert usage_before.elapsed_seconds > 0

        tracker.reset()

        usage_after = tracker.get_current_usage()
        assert usage_after.operations_count == 0
        assert usage_after.elapsed_seconds < usage_before.elapsed_seconds

    def test_format_usage_report(self):
        budget = ComputationBudget(
            max_memory_mb=1000.0,
            max_time_seconds=60.0,
            max_operations=10000,
        )
        tracker = BudgetTracker(budget)

        tracker.record_operation(5000)
        time.sleep(0.05)

        report = tracker.format_usage_report()

        assert "Memory:" in report
        assert "Time:" in report
        assert "Operations:" in report
        assert "Budget Status" in report
        assert "5,000" in report

    def test_format_usage_report_no_budget(self):
        budget = ComputationBudget()
        tracker = BudgetTracker(budget)

        tracker.record_operation(100)

        report = tracker.format_usage_report()

        assert "Memory:" in report
        assert "Time:" in report
        assert "Operations:" in report
        # Should not have budget status section
        assert "Budget Status" not in report
