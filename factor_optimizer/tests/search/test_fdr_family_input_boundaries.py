import pytest

from factor_optimizer.contracts.campaign_store import SQLiteCampaignStore, CampaignStateError


def populated_store(tmp_path):
    store = SQLiteCampaignStore(tmp_path / "fdr.sqlite3")
    store.record_hypothesis_attempt(campaign_id="1", proposal_id="p",
        evaluation_intent_hash="i", horizon=1, executed=True, has_pvalue=True)
    return store


@pytest.mark.parametrize("identifier", [1, True])
@pytest.mark.parametrize("operation", ["summary", "complete"])
def test_family_queries_reject_alias_identifiers(tmp_path, identifier, operation):
    store = populated_store(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        if operation == "summary":
            store.hypothesis_family_summary(identifier)
        else:
            store.require_complete_fdr_family(identifier, 1)


@pytest.mark.parametrize("count", [True, 1.0])
def test_complete_family_requires_integer_count(tmp_path, count):
    store = populated_store(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        store.require_complete_fdr_family("1", count)


def test_valid_family_count_and_separate_campaign(tmp_path):
    store = populated_store(tmp_path)
    store.require_complete_fdr_family("1", 1)
    assert store.hypothesis_family_summary("2")["pvalue_count"] == 0
    for campaign, count in [("2", 1), ("1", 0), ("1", 2)]:
        with pytest.raises(CampaignStateError):
            store.require_complete_fdr_family(campaign, count)
