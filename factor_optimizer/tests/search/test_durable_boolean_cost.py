import pytest

from factor_optimizer.contracts.campaign_store import SQLiteCampaignStore


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("operation", ["reserve", "settle"])
def test_boolean_cost_rejected_without_persistent_mutation(tmp_path, value, operation):
    path = tmp_path / "boolean-cost.sqlite3"
    store = SQLiteCampaignStore(path)
    store.create_campaign("campaign", max_evaluations=3, max_cost=10.0)
    if operation == "settle":
        assert store.reserve("campaign", "attempt", 1.0)
        store.start("campaign", "attempt")
    before = store.budget_state("campaign")
    with pytest.raises(TypeError, match="cost"):
        if operation == "reserve":
            store.reserve("campaign", "attempt", value)
        else:
            store.settle("campaign", "attempt", value)
    assert SQLiteCampaignStore(path).budget_state("campaign") == before
