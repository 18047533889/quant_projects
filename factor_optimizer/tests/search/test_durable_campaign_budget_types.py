import pytest

from factor_optimizer.contracts.campaign_store import SQLiteCampaignStore


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("field,value", [
    ("max_evaluations", True), ("max_evaluations", 1.0),
    ("max_evaluations", 1.5), ("max_evaluations", float("inf")),
    ("max_evaluations", float("nan")), ("max_cost", True),
])
def test_invalid_budget_types_cannot_create_or_reopen_campaign(tmp_path, existing, field, value):
    path = tmp_path / "budget.sqlite3"
    store = SQLiteCampaignStore(path)
    if existing:
        store.create_campaign("c", max_evaluations=1, max_cost=1.0)
    with store._connect() as db:
        before = [tuple(row) for row in db.execute("SELECT * FROM campaigns")]
    budget = {"max_evaluations": 1, "max_cost": 1.0, field: value}
    with pytest.raises((TypeError, ValueError)):
        store.create_campaign("c", **budget)
    reopened = SQLiteCampaignStore(path)
    with reopened._connect() as db:
        after = [tuple(row) for row in db.execute("SELECT * FROM campaigns")]
    assert after == before


def test_valid_budget_reopen_and_limit_enforcement(tmp_path):
    path = tmp_path / "valid.sqlite3"
    store = SQLiteCampaignStore(path)
    store.create_campaign("c", max_evaluations=1, max_cost=1)
    store = SQLiteCampaignStore(path)
    store.create_campaign("c", max_evaluations=1, max_cost=1.0)
    assert store.reserve("c", "a", 0.5)
    store.start("c", "a")
    store.settle("c", "a", 0.5)
    assert not store.reserve("c", "b", 0.5)
