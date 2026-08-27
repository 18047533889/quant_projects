"""Campaign coordination and ledger integration.

Only symbols implemented in this distribution are exported. Split-ledger
storage is not shipped here, so this namespace remains usable with an injected
ledger while advertising that the complete campaign surface is research-only.

DLIB-FA-002: FA's campaign is a RESEARCH-campaign governance-metadata record
(:class:`ResearchCampaignGovernance`) — campaign id / generator / factor
assets / admission outcomes / library outcomes / statistics refs. It does NOT
own optimization search control (max_evaluations / cost_budget /
plateau_patience / target_metric / should_stop), which belong to FO
SearchSession/Budget or QuantPlatform Workflow. The legacy
``CampaignCoordinator`` / ``CampaignConfig`` / ``CampaignBudget`` / ``Campaign``
types are retained for compatibility and are RESEARCH_ONLY.
"""

from factor_assets.campaigns.campaign_coordinator import (
    CampaignCoordinator,
    CampaignState,
    CampaignConfig,
    CampaignBudget,
    Campaign,
    ResearchCampaignGovernance,
)
from factor_assets.campaigns.ledger_adapter import LedgerAdapter

RESEARCH_ONLY = True

# Compatibility aliases for names used by the pre-extraction contract.
CampaignSpec = CampaignConfig
CampaignStatus = CampaignState
CampaignLedgerAdapter = LedgerAdapter

__all__ = [
    "CampaignCoordinator",
    "CampaignConfig",
    "CampaignSpec",
    "CampaignState",
    "CampaignStatus",
    "CampaignBudget",
    "Campaign",
    "ResearchCampaignGovernance",
    "LedgerAdapter",
    "CampaignLedgerAdapter",
    "RESEARCH_ONLY",
]
