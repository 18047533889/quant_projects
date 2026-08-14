"""
Example of proper SQLite connection management in research_control.

Demonstrates leak-free patterns for using ledgers.
"""

from research_control.ledger.campaign import CampaignLedger
from research_control.ledger.trial import TrialLedger
from datetime import datetime


# Pattern 1: File-based ledger (uses context managers internally)
def use_file_ledger():
    """File-based ledgers automatically manage connections."""
    ledger = CampaignLedger(db_path="campaign.db")

    # Use normally - connections are managed per operation
    ledger.append(
        event_id="evt_001",
        campaign_id="camp_001",
        state="created",
        timestamp=datetime.now()
    )

    # No explicit close needed for file-based ledgers


# Pattern 2: In-memory ledger (persistent connection)
def use_memory_ledger():
    """In-memory ledgers need explicit close()."""
    ledger = CampaignLedger(db_path=":memory:")

    try:
        # Use ledger
        ledger.append(
            event_id="evt_002",
            campaign_id="camp_002",
            state="created",
            timestamp=datetime.now()
        )

        # Query operations
        history = ledger.get_campaign_history("camp_002")

    finally:
        # Always close in-memory ledgers
        ledger.close()


# Pattern 3: Context manager pattern (recommended)
def use_with_context_manager():
    """Best practice: use context manager for automatic cleanup."""

    # For file-based
    ledger = CampaignLedger(db_path="campaign.db")
    # File-based don't need context manager but it's harmless

    # For in-memory, wrap in try/finally
    mem_ledger = CampaignLedger(db_path=":memory:")
    try:
        mem_ledger.append(
            event_id="evt_003",
            campaign_id="camp_003",
            state="created",
            timestamp=datetime.now()
        )
    finally:
        mem_ledger.close()


# Pattern 4: Long-running service
class LedgerService:
    """Service that manages ledger lifecycle."""

    def __init__(self):
        self.campaign_ledger = CampaignLedger(db_path=":memory:")
        self.trial_ledger = TrialLedger(db_path=":memory:")

    def shutdown(self):
        """Clean up resources on shutdown."""
        self.campaign_ledger.close()
        self.trial_ledger.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


if __name__ == "__main__":
    print("Examples of proper SQLite usage")

    # Example 1: File-based
    print("\n1. File-based ledger:")
    use_file_ledger()

    # Example 2: In-memory with cleanup
    print("\n2. In-memory ledger:")
    use_memory_ledger()

    # Example 3: Service with context manager
    print("\n3. Service with context manager:")
    with LedgerService() as service:
        service.campaign_ledger.append(
            event_id="evt_004",
            campaign_id="camp_004",
            state="created",
            timestamp=datetime.now()
        )

    print("\nAll examples completed without leaks!")
