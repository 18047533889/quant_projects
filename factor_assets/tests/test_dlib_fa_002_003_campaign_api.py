"""DLIB-FA-002/003 campaign governance + public API authority tests.

- FA's campaign is a research-campaign governance-metadata record only; it
  does NOT own optimization search control (max_evaluations / cost_budget /
  plateau_patience / target_metric / should_stop) — those belong to FO
  SearchSession/Budget or QuantPlatform Workflow.
- The top-level package prioritizes the canonical artifacts (FactorSetArtifact,
  FactorMembership, FactorAdmissionArtifact, SimilarityArtifact,
  TreatmentSelectionArtifact).  FactorSet stays a deprecated alias/view.
"""

import pytest


def test_research_campaign_governance_carries_no_search_control():
    """ResearchCampaignGovernance carries governance metadata only — no
    max_evaluations / cost_budget / plateau_patience / target_metric /
    should_stop (DLIB-FA-002)."""
    from factor_assets.campaigns import ResearchCampaignGovernance

    governance = ResearchCampaignGovernance(
        campaign_id="camp-1",
        generator="search",
        factor_asset_refs=("F1", "F2"),
        admission_outcome_refs=("ad1",),
        library_outcome_refs=("lib1",),
        statistics_refs=("stat1",),
    )
    assert governance.campaign_id == "camp-1"
    assert governance.generator == "search"
    assert governance.factor_asset_refs == ("F1", "F2")
    for forbidden in (
        "max_evaluations",
        "cost_budget",
        "plateau_patience",
        "target_metric",
        "should_stop",
    ):
        assert not hasattr(governance, forbidden), (
            f"ResearchCampaignGovernance must not carry {forbidden} — it "
            "belongs to FO SearchSession/Budget"
        )
    d = governance.to_dict()
    assert d["campaign_id"] == "camp-1"


def test_campaign_coordinator_is_research_only():
    """The legacy CampaignCoordinator / budget types are RESEARCH_ONLY
    retained for compatibility (§108)."""
    from factor_assets.campaigns import (
        CampaignCoordinator,
        CampaignConfig,
        CampaignBudget,
        Campaign,
    )
    from factor_assets.campaigns.campaign_coordinator import CampaignState

    coordinator = CampaignCoordinator()
    config = CampaignConfig(campaign_id="camp-2", objective="rank_ic")
    campaign = coordinator.create_campaign(config)
    assert isinstance(campaign, Campaign)
    assert campaign.state is CampaignState.CREATED
    assert isinstance(campaign.budget, CampaignBudget)


def test_top_level_exports_prioritize_canonical_artifacts():
    """DLIB-FA-003: the top-level package exports the canonical artifacts;
    FactorSet stays a deprecated alias (legacy compatibility view)."""
    import factor_assets

    # Canonical artifacts are exported at top level.
    for name in (
        "FactorSetArtifact",
        "FactorMembership",
        "FactorAdmissionArtifact",
        "SimilarityArtifact",
        "TreatmentSelectionArtifact",
    ):
        assert hasattr(factor_assets, name), f"top-level must export {name}"
        assert name in factor_assets.__all__

    # Legacy FactorSet is retained as a deprecated compatibility view.
    assert hasattr(factor_assets, "FactorSet")
    assert "FactorSet" in factor_assets.__all__


def test_legacy_factor_set_should_be_view_not_authority():
    """FactorSetArtifact.to_legacy_view() produces the legacy FactorSet as a
    projection; the artifact is the canonical authority (DLIB-FA-003)."""
    from factor_assets.contracts.factor_set import (
        FactorSetArtifact,
        FactorMembership,
        FactorSetSpec,
    )

    member = FactorMembership(factor_id="F1")
    spec = FactorSetSpec(
        set_id="s1",
        name="S1",
        selection_policy="manual",
        universe_ref="u1",
        data_snapshot_ref="snap1",
        split_ref="split1",
    )
    artifact = FactorSetArtifact(
        set_id="s1",
        name="S1",
        members=(member,),
        created_at="2024-01-01T00:00:00Z",
        policy_hash="p" * 64,
        assembly_hash="a" * 64,
        snapshot_ref="snap1",
        universe_ref="u1",
        split_ref="split1",
        spec=spec,
    )
    legacy = artifact.to_legacy_view()
    assert legacy.factor_ids == ("F1",)
    assert legacy.assembly_hash == artifact.assembly_hash
    # The legacy view is a projection, not a separate authority.
    assert type(legacy).__name__ == "FactorSet"


def test_ledger_seal_is_logical_not_safety_boundary():
    """seal_test_splits records a LOGICAL seal (projection/audit only). Test
    authority is FO TestAuthorityBroker (DLIB-FA-003 / §43)."""
    from factor_assets.campaigns.ledger_adapter import LedgerAdapter

    class _FakeLedger:
        def __init__(self):
            self.usage = []

        def record_usage(self, split_id, candidate_id, usage_type, metadata):
            self.usage.append(
                {"split_id": split_id, "candidate_id": candidate_id,
                 "usage_type": usage_type, "metadata": metadata}
            )

        def detect_contamination(self, candidate_id, test_split_id):
            return None

        def get_usage_history(self, candidate_id):
            return [r for r in self.usage if r["candidate_id"] == candidate_id]

        def get_split_usage(self, split_id):
            return [r for r in self.usage if r["split_id"] == split_id]

        def list_contamination(self):
            return []

    ledger = _FakeLedger()
    adapter = LedgerAdapter(ledger)
    adapter.seal_test_splits("camp-1", ["test-split-1"])
    assert ledger.usage[0]["usage_type"] == "seal"
    assert ledger.usage[0]["metadata"]["sealed"] is True
    # It is a logical record — no enforcement capability.
    assert ledger.usage[0]["candidate_id"].startswith("__seal_")
