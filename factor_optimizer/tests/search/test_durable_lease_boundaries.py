import pytest

from factor_optimizer.contracts import campaign_store
from factor_optimizer.contracts.campaign_store import SQLiteCampaignStore


@pytest.mark.parametrize("lease", [True, False, float("nan"), float("inf"), -float("inf"), 0, -1, None, "60"])
def test_invalid_lease_rejected_without_ledger_changes(tmp_path, monkeypatch, lease):
    monkeypatch.setattr(campaign_store.time, "time", lambda: 100.0)
    path = tmp_path / "lease.sqlite3"
    store = SQLiteCampaignStore(path)
    store.create_campaign("c", max_evaluations=3, max_cost=10.0)
    assert store.reserve("c", "existing", 1.0, lease_seconds=1)
    monkeypatch.setattr(campaign_store.time, "time", lambda: 102.0)
    before = store.budget_state("c")
    with pytest.raises((ValueError, TypeError)):
        store.reserve("c", "invalid", 1.0, lease_seconds=lease)
    reopened = SQLiteCampaignStore(path)
    assert reopened.budget_state("c") == before
    with reopened._connect() as db:
        rows = db.execute("SELECT attempt_id,state FROM reservations").fetchall()
    assert [(r["attempt_id"], r["state"]) for r in rows] == [("existing", "RESERVED")]


@pytest.mark.parametrize("lease", [1, 0.5])
def test_positive_finite_lease_expires_normally(tmp_path, monkeypatch, lease):
    monkeypatch.setattr(campaign_store.time, "time", lambda: 100.0)
    store = SQLiteCampaignStore(tmp_path / "valid.sqlite3")
    store.create_campaign("c", max_evaluations=1, max_cost=1.0)
    assert store.reserve("c", "a", 1.0, lease_seconds=lease)
    assert not store.reserve("c", "b", 1.0)
    monkeypatch.setattr(campaign_store.time, "time", lambda: 100.0 + lease)
    assert store.reserve("c", "b", 1.0)
