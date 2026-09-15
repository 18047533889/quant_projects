import pytest

from factor_optimizer.contracts.campaign_store import (
    SQLiteCampaignStore, DurableBudgetTracker, CampaignStateError,
)
from factor_optimizer.contracts.search_budget import SearchBudget


@pytest.mark.parametrize("operation", ["commit", "release"])
@pytest.mark.parametrize("amount", [9.0, True, float("nan"), None])
def test_wrong_reserved_amount_cannot_change_ledger(tmp_path, operation, amount):
    store = SQLiteCampaignStore(tmp_path / "amount.sqlite3")
    tracker = DurableBudgetTracker(store, "c", SearchBudget(3, 3, 10.0))
    assert tracker.reserve_evaluation(1.0, "a")
    if operation == "commit":
        tracker.start_evaluation("a")
    before = store.budget_state("c")
    with pytest.raises((ValueError, TypeError, CampaignStateError)):
        if operation == "commit":
            tracker.commit_evaluation(amount, 0.5, "a")
        else:
            tracker.release_evaluation(amount, "a")
    assert store.budget_state("c") == before
    assert tracker.has_reservation("a")
    if operation == "commit":
        tracker.commit_evaluation(1.0, 0.5, "a")
        assert tracker.cost_used == 0.5
    else:
        tracker.release_evaluation(1.0, "a")
    assert not tracker.has_reservation("a")
