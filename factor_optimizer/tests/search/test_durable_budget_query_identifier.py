import pytest

from factor_optimizer.contracts.campaign_store import SQLiteCampaignStore, DurableBudgetTracker
from factor_optimizer.contracts.search_budget import SearchBudget


@pytest.mark.parametrize("invalid", [1, True])
@pytest.mark.parametrize("query", ["budget_state", "has_attempt", "has_campaign"])
def test_query_cannot_alias_numeric_identifier(tmp_path, invalid, query):
    store = SQLiteCampaignStore(tmp_path / "query.sqlite3")
    tracker = DurableBudgetTracker(store, "1", SearchBudget(3, 3, 10.0))
    assert tracker.reserve_evaluation(1.0, "1")
    before = store.budget_state("1")
    if query == "has_campaign":
        tracker.campaign_id = invalid
    with pytest.raises((ValueError, TypeError)):
        if query == "budget_state":
            store.budget_state(invalid)
        else:
            tracker.has_reservation(invalid if query == "has_attempt" else "1")
    assert store.budget_state("1") == before


def test_valid_and_missing_reservation_queries(tmp_path):
    store = SQLiteCampaignStore(tmp_path / "valid.sqlite3")
    tracker = DurableBudgetTracker(store, "1", SearchBudget(3, 3, 10.0))
    assert not tracker.has_reservation()
    assert not tracker.has_reservation("")
    assert not tracker.has_reservation("missing")
    assert tracker.reserve_evaluation(1.0, "1")
    assert tracker.has_reservation("1")
    tracker.start_evaluation("1")
    assert tracker.has_reservation("1")
    tracker.commit_evaluation(1.0, 0.5, "1")
    assert not tracker.has_reservation("1")
