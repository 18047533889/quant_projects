import pytest

from factor_optimizer.contracts.campaign_store import SQLiteCampaignStore


@pytest.mark.parametrize("invalid", [1, True, 1.0, b"1", "", " "])
def test_invalid_spec_identity_cannot_enter_ledger(tmp_path, invalid):
    path = tmp_path / "spec.sqlite3"
    store = SQLiteCampaignStore(path)
    store.record_hypothesis_attempt(campaign_id="c", proposal_id="p1",
        evaluation_intent_hash="i", horizon=1, effective_spec_hash="1")
    before = store.hypothesis_family_summary("c")
    with pytest.raises((TypeError, ValueError)):
        store.record_hypothesis_attempt(campaign_id="c", proposal_id="p2",
            evaluation_intent_hash="i", horizon=1, effective_spec_hash=invalid)
    assert SQLiteCampaignStore(path).hypothesis_family_summary("c") == before


def test_optional_and_valid_spec_identity_replay(tmp_path):
    store = SQLiteCampaignStore(tmp_path / "valid.sqlite3")
    for _ in range(2):
        for proposal, spec in [("p1", None), ("p2", "1"), ("p3", "1"), ("p4", "2")]:
            store.record_hypothesis_attempt(campaign_id="c", proposal_id=proposal,
                evaluation_intent_hash="i", horizon=1, effective_spec_hash=spec)
    summary = store.hypothesis_family_summary("c")
    assert summary["proposal_count"] == 4
    assert summary["unique_effective_spec_count"] == 2
