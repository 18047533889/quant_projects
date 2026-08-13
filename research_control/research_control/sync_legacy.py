"""Idempotent synchronization utilities for event ledger."""

from typing import List, Set

from .events import CampaignEvent, DecisionEvent, TrialEvent
from .ledger_legacy import EventLedger, EventType


class IdempotentSync:
    """Provides idempotent batch sync operations for event ledger."""

    def __init__(self, ledger: EventLedger):
        self._ledger = ledger

    def sync_events(self, events: List[EventType]) -> dict:
        """
        Sync a batch of events to the ledger.

        Returns summary with counts of added, duplicate, and failed events.
        """
        added = 0
        duplicates = 0
        failed = 0
        failed_events = []

        for event in events:
            try:
                if self._ledger.append(event):
                    added += 1
                else:
                    duplicates += 1
            except Exception as e:
                failed += 1
                failed_events.append({"event_id": event.event_id, "error": str(e)})

        return {
            "added": added,
            "duplicates": duplicates,
            "failed": failed,
            "failed_events": failed_events,
            "total_processed": len(events),
        }

    def get_missing_event_ids(self, event_ids: List[str]) -> Set[str]:
        """
        Return set of event IDs that are not yet in the ledger.

        Useful for determining which events need to be fetched/synced.
        """
        missing = set()
        for event_id in event_ids:
            if self._ledger.get_by_event_id(event_id) is None:
                missing.add(event_id)
        return missing

    def verify_campaign_consistency(self, campaign_id: str) -> dict:
        """
        Verify consistency of campaign events.

        Returns dict with validation results and any anomalies detected.
        """
        events = self._ledger.get_by_campaign(campaign_id)

        campaign_events = [e for e in events if isinstance(e, CampaignEvent)]
        trial_events = [e for e in events if isinstance(e, TrialEvent)]
        decision_events = [e for e in events if isinstance(e, DecisionEvent)]

        # Check for campaign lifecycle ordering
        campaign_states = [e.event_type for e in campaign_events]
        anomalies = []

        # Check if campaign was created
        if campaign_events and campaign_states[0] != "created":
            anomalies.append(f"Campaign {campaign_id} first event is '{campaign_states[0]}', not 'created'")

        # Check for invalid state transitions
        if "started" in campaign_states and "created" not in campaign_states:
            anomalies.append(f"Campaign {campaign_id} started without being created")

        if "completed" in campaign_states or "cancelled" in campaign_states:
            final_states = [s for s in campaign_states if s in {"completed", "cancelled"}]
            if len(final_states) > 1:
                anomalies.append(f"Campaign {campaign_id} has multiple terminal states: {final_states}")

        return {
            "campaign_id": campaign_id,
            "total_events": len(events),
            "campaign_events": len(campaign_events),
            "trial_events": len(trial_events),
            "decision_events": len(decision_events),
            "campaign_states": campaign_states,
            "anomalies": anomalies,
            "is_consistent": len(anomalies) == 0,
        }
