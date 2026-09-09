import pytest

from factor_optimizer.contracts.campaign_store import CampaignStateError, SQLiteCampaignStore
from factor_optimizer.contracts.statistical_governance import (
    SplitEvidenceRef, SplitPurpose, diagnose_horizon_curve, retention_evidence,
)


def test_campaign_family_counts_aliases_and_horizons_without_reset(tmp_path):
    path = tmp_path / "family.sqlite3"
    store = SQLiteCampaignStore(path)
    for proposal, spec, horizon in (
        ("name-a", "same-spec", 3), ("alias-b", "same-spec", 3),
        ("h10", "same-spec", 10), ("parse-failed", None, 10),
    ):
        store.record_hypothesis_attempt(
            campaign_id="parent-campaign", proposal_id=proposal,
            effective_spec_hash=spec, evaluation_intent_hash="intent", horizon=horizon,
            executed=spec is not None, has_pvalue=spec is not None,
        )
    summary = SQLiteCampaignStore(path).hypothesis_family_summary("parent-campaign")
    assert summary == {
        "proposal_count": 4, "executed_trial_count": 3,
        "unique_effective_spec_count": 2, "pvalue_count": 3,
        "horizon_hypothesis_count": 2,
    }
    with pytest.raises(CampaignStateError, match="complete recorded"):
        store.require_complete_fdr_family("parent-campaign", submitted_pvalue_count=1)
    store.require_complete_fdr_family("parent-campaign", submitted_pvalue_count=3)


def test_diagnostic_or_winner_only_split_cannot_claim_production_oos():
    with pytest.raises(ValueError, match="diagnostic"):
        SplitEvidenceRef("pbo", SplitPurpose.DIAGNOSTIC, False, True).require_production_oos()
    with pytest.raises(ValueError, match="winner-only"):
        SplitEvidenceRef("forward", SplitPurpose.PRODUCTION_OOS, True, False).require_production_oos()
    SplitEvidenceRef("forward", SplitPurpose.PRODUCTION_OOS, True, True).require_production_oos()


def test_retention_near_zero_and_sign_reversal_are_not_ratios():
    tiny = retention_evidence(0.0001, 0.0002)
    assert not tiny.ratio_applicable and tiny.retention_ratio is None
    reversal = retention_evidence(0.02, -0.01)
    assert reversal.sign_reversal and reversal.retention_ratio is None
    normal = retention_evidence(0.02, 0.01)
    assert normal.ratio_applicable and normal.retention_ratio == 0.5


def test_horizon_sign_change_has_no_fabricated_half_life_and_counts_family():
    result = diagnose_horizon_curve({1: -0.02, 5: -0.01, 10: 0.03})
    assert result.fit_status == "SIGN_CHANGE"
    assert result.half_life is None
    assert result.hypothesis_count == 3
