import pytest

from factor_optimizer.contracts.campaign_store import SQLiteCampaignStore


@pytest.mark.parametrize("operation", ["reserve", "start", "release", "settle"])
@pytest.mark.parametrize("field", ["campaign_id", "attempt_id"])
@pytest.mark.parametrize("invalid", [1, True])
def test_numeric_identifier_cannot_alias_existing_budget(tmp_path, operation, field, invalid):
    path = tmp_path / "ids.sqlite3"
    store = SQLiteCampaignStore(path)
    store.create_campaign("1", max_evaluations=3, max_cost=10.0)
    assert store.reserve("1", "1", 1.0)
    if operation == "settle":
        store.start("1", "1")
    with store._connect() as db:
        before = [tuple(row) for row in db.execute("SELECT * FROM reservations")]
    budget_before = store.budget_state("1")
    args = {"campaign_id": "1", "attempt_id": "1", field: invalid}
    if operation == "reserve":
        args["estimated_cost"] = 1.0
    if operation == "settle":
        args["actual_cost"] = 0.5
    with pytest.raises((TypeError, ValueError)):
        getattr(store, operation)(**args)
    reopened = SQLiteCampaignStore(path)
    assert reopened.budget_state("1") == budget_before
    with reopened._connect() as db:
        after = [tuple(row) for row in db.execute("SELECT * FROM reservations")]
    assert after == before


@pytest.mark.parametrize("invalid", [1, True, " ", b"1"])
def test_create_requires_nonblank_string_campaign_id(tmp_path, invalid):
    store = SQLiteCampaignStore(tmp_path / "create.sqlite3")
    with pytest.raises((TypeError, ValueError)):
        store.create_campaign(invalid, max_evaluations=1, max_cost=1.0)
    with store._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM campaigns").fetchone()[0] == 0
