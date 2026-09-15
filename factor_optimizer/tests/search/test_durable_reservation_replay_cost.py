import pytest

from factor_optimizer.contracts.campaign_store import SQLiteCampaignStore, CampaignStateError


@pytest.mark.parametrize("state", ["RESERVED", "STARTED", "SETTLED", "RELEASED"])
def test_replay_rejects_changed_cost_without_mutating_ledger(tmp_path, state):
    path = tmp_path / "replay.sqlite3"
    store = SQLiteCampaignStore(path)
    store.create_campaign("c", max_evaluations=3, max_cost=10.0)
    assert store.reserve("c", "a", 1.0)
    if state in ("STARTED", "SETTLED"):
        store.start("c", "a")
    if state == "SETTLED":
        store.settle("c", "a", 0.5)
    if state == "RELEASED":
        store.release("c", "a")
    before = store.budget_state("c")
    with pytest.raises(CampaignStateError, match="cost"):
        store.reserve("c", "a", 9.0)
    restarted = SQLiteCampaignStore(path)
    assert restarted.budget_state("c") == before
    with restarted._connect() as db:
        row = db.execute("SELECT state,estimated_cost FROM reservations WHERE campaign_id='c' AND attempt_id='a'").fetchone()
    assert (row["state"], row["estimated_cost"]) == (state, 1.0)
    # Identical replay keeps the existing idempotent return contract.
    assert restarted.reserve("c", "a", 1.0) is (state != "RELEASED")
    assert restarted.budget_state("c") == before
