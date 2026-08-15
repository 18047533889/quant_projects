"""Campaign coordination and ledger integration.

Only symbols implemented in this distribution are exported. Split-ledger
storage is not shipped here, so this namespace remains usable with an injected
ledger while advertising that the complete campaign surface is research-only.
"""

from factor_assets.campaigns.campaign_coordinator import (
    CampaignCoordinator,
    CampaignState,
    CampaignConfig,
    CampaignBudget,
    Campaign,
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
    "LedgerAdapter",
    "CampaignLedgerAdapter",
    "RESEARCH_ONLY",
]
