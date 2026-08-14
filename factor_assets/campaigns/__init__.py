"""
Factor Assets Campaigns: Optimization campaign coordination and ledger integration.

Coordinates with research_control for campaign lifecycle tracking.
"""

from factor_assets.campaigns.campaign_coordinator import (
    CampaignCoordinator,
    CampaignSpec,
    CampaignStatus,
    CampaignState,
)
from factor_assets.campaigns.ledger_adapter import (
    CampaignLedgerAdapter,
    TrialRecord,
)
from factor_assets.campaigns.split_ledger import (
    SplitContaminationLedger,
)

__all__ = [
    "CampaignCoordinator",
    "CampaignSpec",
    "CampaignStatus",
    "CampaignState",
    "CampaignLedgerAdapter",
    "TrialRecord",
    "SplitContaminationLedger",
]
