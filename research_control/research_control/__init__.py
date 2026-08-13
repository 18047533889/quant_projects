"""Research control ledger for tracking campaigns, trials, and decisions."""

from .events import CampaignEvent, TrialEvent, DecisionEvent

# Legacy in-memory implementations
from .ledger_legacy import EventLedger
from .sync_legacy import IdempotentSync

# New SQLite-backed ledger components
from .ledger.campaign import CampaignLedger
from .ledger.trial import TrialLedger
from .ledger.query import LedgerQuery
from .sync.idempotency import IdempotentSyncEngine

__all__ = [
    "CampaignEvent",
    "TrialEvent",
    "DecisionEvent",
    "EventLedger",
    "IdempotentSync",
    "CampaignLedger",
    "TrialLedger",
    "LedgerQuery",
    "IdempotentSyncEngine",
]
