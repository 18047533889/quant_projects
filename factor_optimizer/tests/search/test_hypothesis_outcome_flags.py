import pytest

from factor_optimizer.contracts.campaign_store import SQLiteCampaignStore


@pytest.mark.parametrize("field", ["executed", "has_pvalue"])
@pytest.mark.parametrize("value", [2, -1, 1, "0", 0.5, None])
def test_outcome_flags_cannot_distort_family_counts(tmp_path, field, value):
    path = tmp_path / "flags.sqlite3"
    store = SQLiteCampaignStore(path)
    before = store.hypothesis_family_summary("c")
    flags = {"executed": True, "has_pvalue": False, field: value}
    with pytest.raises((TypeError, ValueError)):
        store.record_hypothesis_attempt(
            campaign_id="c", proposal_id="p", evaluation_intent_hash="i",
            horizon=1, **flags,
        )
    assert SQLiteCampaignStore(path).hypothesis_family_summary("c") == before


@pytest.mark.parametrize("executed,has_pvalue", [(False, False), (True, False), (True, True)])
def test_boolean_outcomes_keep_exact_counts_and_replay(tmp_path, executed, has_pvalue):
    store = SQLiteCampaignStore(tmp_path / "valid.sqlite3")
    for _ in range(2):
        store.record_hypothesis_attempt(
            campaign_id="c", proposal_id="p", evaluation_intent_hash="i",
            horizon=1, executed=executed, has_pvalue=has_pvalue,
        )
    summary = store.hypothesis_family_summary("c")
    assert summary["proposal_count"] == 1
    assert summary["executed_trial_count"] == int(executed)
    assert summary["pvalue_count"] == int(has_pvalue)
