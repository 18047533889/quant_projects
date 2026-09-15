import pytest

from factor_optimizer.contracts.search_budget import BudgetTracker, SearchBudget


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("operation", ["reserve", "actual", "reserved", "release"])
def test_boolean_cost_operations_reject_without_mutation(value, operation):
    tracker = BudgetTracker(SearchBudget(3, 3, 10.0))
    amount = int(value)
    if operation != "reserve":
        assert tracker.reserve_evaluation(amount, "attempt")
    before = tracker.to_dict()
    with pytest.raises(TypeError, match="cost"):
        if operation == "reserve":
            tracker.reserve_evaluation(value, "attempt")
        elif operation == "actual":
            tracker.commit_evaluation(amount, value, "attempt")
        elif operation == "reserved":
            tracker.commit_evaluation(value, 0.0, "attempt")
        else:
            tracker.release_evaluation(value, "attempt")
    assert tracker.to_dict() == before


@pytest.mark.parametrize("value", [False, True])
def test_can_spend_does_not_accept_boolean_cost(value):
    assert not BudgetTracker(SearchBudget()).can_spend(value)
