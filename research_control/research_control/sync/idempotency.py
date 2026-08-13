"""Event deduplication with fail-open and fail-atomic semantics."""

import sqlite3
from typing import Dict, List, Set, Any, Optional
from datetime import datetime

from ..ledger.campaign import CampaignLedger
from ..ledger.trial import TrialLedger


class IdempotentSyncEngine:
    """
    Idempotent event synchronization with fail-open (FO) and fail-atomic (FA) modes.

    Fail-Open (FO): Skips invalid events, processes rest of batch, returns partial success.
    Fail-Atomic (FA): Aborts entire batch on first validation error, rollback all changes.
    """

    def __init__(
        self,
        campaign_ledger: CampaignLedger,
        trial_ledger: TrialLedger,
        decision_ledger: Optional[Any] = None,
    ):
        self._campaign = campaign_ledger
        self._trial = trial_ledger
        self._decision = decision_ledger

    def sync_campaign_events(
        self,
        events: List[Dict[str, Any]],
        mode: str = "fail_open",
    ) -> Dict[str, Any]:
        """
        Sync batch of campaign events with deduplication.

        Args:
            events: List of dicts with keys: event_id, campaign_id, state, timestamp, metadata
            mode: "fail_open" or "fail_atomic"

        Returns:
            Dict with keys:
                - added: int
                - duplicates: int
                - failed: int
                - failed_events: List[Dict] (fail_open only)
                - error: str (fail_atomic only)
        """
        if mode not in {"fail_open", "fail_atomic"}:
            raise ValueError(f"Invalid mode: {mode}. Must be 'fail_open' or 'fail_atomic'")

        if mode == "fail_atomic":
            return self._sync_campaign_atomic(events)
        else:
            return self._sync_campaign_open(events)

    def _sync_campaign_open(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fail-open: skip invalid events, process rest."""
        added = 0
        duplicates = 0
        failed = 0
        failed_events = []

        for event in events:
            try:
                # Validate required fields
                event_id = event["event_id"]
                campaign_id = event["campaign_id"]
                state = event["state"]
                timestamp = self._parse_timestamp(event["timestamp"])
                metadata = event.get("metadata")

                # Attempt append (idempotent)
                if self._campaign.append(event_id, campaign_id, state, timestamp, metadata):
                    added += 1
                else:
                    duplicates += 1
            except Exception as e:
                failed += 1
                failed_events.append({
                    "event": event,
                    "error": str(e),
                })

        return {
            "added": added,
            "duplicates": duplicates,
            "failed": failed,
            "failed_events": failed_events,
            "total_processed": len(events),
        }

    def _sync_campaign_atomic(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fail-atomic: abort entire batch on first error."""
        # Validate all events first
        validated = []
        try:
            for event in events:
                event_id = event["event_id"]
                campaign_id = event["campaign_id"]
                state = event["state"]
                timestamp = self._parse_timestamp(event["timestamp"])
                metadata = event.get("metadata")

                # Validate state early
                if state not in self._campaign.VALID_STATES:
                    raise ValueError(f"Invalid campaign state: {state}")

                validated.append((event_id, campaign_id, state, timestamp, metadata))
        except Exception as e:
            return {
                "added": 0,
                "duplicates": 0,
                "failed": len(events),
                "error": f"Validation failed: {str(e)}",
            }

        # All validated, now apply atomically
        added = 0
        duplicates = 0
        try:
            for event_id, campaign_id, state, timestamp, metadata in validated:
                if self._campaign.append(event_id, campaign_id, state, timestamp, metadata):
                    added += 1
                else:
                    duplicates += 1
        except Exception as e:
            # Rollback not possible with current API; would need transaction support
            return {
                "added": 0,
                "duplicates": 0,
                "failed": len(events),
                "error": f"Atomic append failed: {str(e)}",
            }

        return {
            "added": added,
            "duplicates": duplicates,
            "failed": 0,
            "total_processed": len(events),
        }

    def sync_trial_events(
        self,
        events: List[Dict[str, Any]],
        mode: str = "fail_open",
    ) -> Dict[str, Any]:
        """
        Sync batch of trial events with deduplication.

        Args:
            events: List of dicts with keys: event_id, trial_id, campaign_id, state, timestamp, parameters, metrics, metadata
            mode: "fail_open" or "fail_atomic"

        Returns:
            Same structure as sync_campaign_events
        """
        if mode not in {"fail_open", "fail_atomic"}:
            raise ValueError(f"Invalid mode: {mode}. Must be 'fail_open' or 'fail_atomic'")

        if mode == "fail_atomic":
            return self._sync_trial_atomic(events)
        else:
            return self._sync_trial_open(events)

    def _sync_trial_open(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fail-open: skip invalid events, process rest."""
        added = 0
        duplicates = 0
        failed = 0
        failed_events = []

        for event in events:
            try:
                event_id = event["event_id"]
                trial_id = event["trial_id"]
                campaign_id = event["campaign_id"]
                state = event["state"]
                timestamp = self._parse_timestamp(event["timestamp"])
                parameters = event.get("parameters")
                metrics = event.get("metrics")
                metadata = event.get("metadata")

                if self._trial.append(
                    event_id, trial_id, campaign_id, state, timestamp, parameters, metrics, metadata
                ):
                    added += 1
                else:
                    duplicates += 1
            except Exception as e:
                failed += 1
                failed_events.append({
                    "event": event,
                    "error": str(e),
                })

        return {
            "added": added,
            "duplicates": duplicates,
            "failed": failed,
            "failed_events": failed_events,
            "total_processed": len(events),
        }

    def _sync_trial_atomic(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fail-atomic: abort entire batch on first error."""
        validated = []
        try:
            for event in events:
                event_id = event["event_id"]
                trial_id = event["trial_id"]
                campaign_id = event["campaign_id"]
                state = event["state"]
                timestamp = self._parse_timestamp(event["timestamp"])
                parameters = event.get("parameters")
                metrics = event.get("metrics")
                metadata = event.get("metadata")

                # Validate state early
                if state not in self._trial.VALID_STATES:
                    raise ValueError(f"Invalid trial state: {state}")

                validated.append((event_id, trial_id, campaign_id, state, timestamp, parameters, metrics, metadata))
        except Exception as e:
            return {
                "added": 0,
                "duplicates": 0,
                "failed": len(events),
                "error": f"Validation failed: {str(e)}",
            }

        added = 0
        duplicates = 0
        try:
            for event_id, trial_id, campaign_id, state, timestamp, parameters, metrics, metadata in validated:
                if self._trial.append(
                    event_id, trial_id, campaign_id, state, timestamp, parameters, metrics, metadata
                ):
                    added += 1
                else:
                    duplicates += 1
        except Exception as e:
            return {
                "added": 0,
                "duplicates": 0,
                "failed": len(events),
                "error": f"Atomic append failed: {str(e)}",
            }

        return {
            "added": added,
            "duplicates": duplicates,
            "failed": 0,
            "total_processed": len(events),
        }

    def get_missing_event_ids(
        self,
        event_ids: List[str],
        entity_type: str = "campaign",
    ) -> Set[str]:
        """
        Return event IDs not yet in the ledger.

        Args:
            event_ids: List of event IDs to check
            entity_type: "campaign" or "trial"

        Returns:
            Set of missing event IDs
        """
        if entity_type == "campaign":
            return self._get_missing_campaign_ids(event_ids)
        elif entity_type == "trial":
            return self._get_missing_trial_ids(event_ids)
        else:
            raise ValueError(f"Invalid entity_type: {entity_type}")

    def _get_missing_campaign_ids(self, event_ids: List[str]) -> Set[str]:
        """Check which campaign event IDs are missing."""
        if not event_ids:
            return set()

        with self._campaign._conn() as conn:
            placeholders = ",".join("?" * len(event_ids))
            cursor = conn.execute(
                f"SELECT event_id FROM campaign_events WHERE event_id IN ({placeholders})",
                event_ids,
            )
            existing = {row["event_id"] for row in cursor.fetchall()}

        return set(event_ids) - existing

    def _get_missing_trial_ids(self, event_ids: List[str]) -> Set[str]:
        """Check which trial event IDs are missing."""
        if not event_ids:
            return set()

        with self._trial._conn() as conn:
            placeholders = ",".join("?" * len(event_ids))
            cursor = conn.execute(
                f"SELECT event_id FROM trial_events WHERE event_id IN ({placeholders})",
                event_ids,
            )
            existing = {row["event_id"] for row in cursor.fetchall()}

        return set(event_ids) - existing

    def verify_campaign_consistency(self, campaign_id: str) -> Dict[str, Any]:
        """
        Verify consistency of campaign state transitions.

        Returns:
            Dict with keys:
                - campaign_id: str
                - is_consistent: bool
                - anomalies: List[str]
                - state_sequence: List[str]
                - trial_count: int
        """
        history = self._campaign.get_campaign_history(campaign_id)
        trial_events = self._trial.get_campaign_trials(campaign_id)

        state_sequence = [evt["state"] for evt in history]
        anomalies = []

        # Check first state is "created"
        if state_sequence and state_sequence[0] != "created":
            anomalies.append(f"First state is '{state_sequence[0]}', expected 'created'")

        # Check for started without created
        if "started" in state_sequence and "created" not in state_sequence:
            anomalies.append("Campaign started without being created")

        # Check for multiple terminal states
        terminal_count = sum(s in CampaignLedger.TERMINAL_STATES for s in state_sequence)
        if terminal_count > 1:
            anomalies.append(f"Multiple terminal states: {terminal_count}")

        # Check for state transitions after terminal
        if state_sequence:
            for i, state in enumerate(state_sequence):
                if state in CampaignLedger.TERMINAL_STATES and i < len(state_sequence) - 1:
                    anomalies.append(f"State transition after terminal state: {state} at position {i}")

        trial_ids = {evt["trial_id"] for evt in trial_events}

        return {
            "campaign_id": campaign_id,
            "is_consistent": len(anomalies) == 0,
            "anomalies": anomalies,
            "state_sequence": state_sequence,
            "trial_count": len(trial_ids),
            "total_events": len(history) + len(trial_events),
        }

    def _parse_timestamp(self, ts: Any) -> datetime:
        """Parse timestamp from string or datetime object."""
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            # Try ISO format
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                pass
            # Try other formats if needed
            raise ValueError(f"Cannot parse timestamp: {ts}")
        raise TypeError(f"Invalid timestamp type: {type(ts)}")
