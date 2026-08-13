"""In-memory append-only event ledger with query capabilities."""

from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Union

from .events import CampaignEvent, DecisionEvent, TrialEvent

EventType = Union[CampaignEvent, TrialEvent, DecisionEvent]


class EventLedger:
    """Thread-safe append-only ledger for research control events."""

    def __init__(self):
        self._events: List[EventType] = []
        self._event_ids = set()
        self._campaign_index = defaultdict(list)
        self._trial_index = defaultdict(list)
        self._decision_index = defaultdict(list)

    def append(self, event: EventType) -> bool:
        """
        Append event to ledger. Returns True if added, False if duplicate.

        Idempotent: duplicate event_id is silently ignored.
        """
        if event.event_id in self._event_ids:
            return False

        self._events.append(event)
        self._event_ids.add(event.event_id)

        # Update indices
        if isinstance(event, CampaignEvent):
            self._campaign_index[event.campaign_id].append(len(self._events) - 1)
        elif isinstance(event, TrialEvent):
            self._campaign_index[event.campaign_id].append(len(self._events) - 1)
            self._trial_index[event.trial_id].append(len(self._events) - 1)
        elif isinstance(event, DecisionEvent):
            self._campaign_index[event.campaign_id].append(len(self._events) - 1)
            self._decision_index[event.decision_id].append(len(self._events) - 1)

        return True

    def get_all(self) -> List[EventType]:
        """Return all events in append order."""
        return list(self._events)

    def get_by_campaign(self, campaign_id: str) -> List[EventType]:
        """Return all events for a campaign in append order."""
        indices = self._campaign_index.get(campaign_id, [])
        return [self._events[i] for i in indices]

    def get_by_trial(self, trial_id: str) -> List[EventType]:
        """Return all events for a trial in append order."""
        indices = self._trial_index.get(trial_id, [])
        return [self._events[i] for i in indices]

    def get_by_decision(self, decision_id: str) -> List[EventType]:
        """Return all events for a decision in append order."""
        indices = self._decision_index.get(decision_id, [])
        return [self._events[i] for i in indices]

    def get_by_time_range(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[EventType]:
        """Return events within time range (inclusive)."""
        result = []
        for event in self._events:
            if start and event.timestamp < start:
                continue
            if end and event.timestamp > end:
                continue
            result.append(event)
        return result

    def get_by_event_id(self, event_id: str) -> Optional[EventType]:
        """Return event by ID, or None if not found."""
        if event_id not in self._event_ids:
            return None
        for event in self._events:
            if event.event_id == event_id:
                return event
        return None

    def count(self) -> int:
        """Return total event count."""
        return len(self._events)

    def count_by_campaign(self, campaign_id: str) -> int:
        """Return event count for a campaign."""
        return len(self._campaign_index.get(campaign_id, []))
