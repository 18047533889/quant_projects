"""Temporal queries and lineage reconstruction for ledger."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from .campaign import CampaignLedger
from .trial import TrialLedger


class LedgerQuery:
    """
    Temporal query interface for campaign and trial ledgers.

    Provides time-range queries, state reconstruction, and full lineage tracking
    across campaigns, trials, and decisions.
    """

    def __init__(
        self,
        campaign_ledger: CampaignLedger,
        trial_ledger: TrialLedger,
        decision_ledger: Optional[Any] = None,
    ):
        self._campaign = campaign_ledger
        self._trial = trial_ledger
        self._decision = decision_ledger  # Optional DecisionLedger

    @contextmanager
    def _conn(self):
        """Context manager for direct database access (campaign ledger)."""
        # Use the campaign ledger's connection strategy
        if self._campaign._persistent_conn:
            yield self._campaign._persistent_conn
        else:
            conn = sqlite3.connect(self._campaign.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    @contextmanager
    def _trial_conn(self):
        """Context manager for trial database access."""
        # Use the trial ledger's connection strategy
        if self._trial._persistent_conn:
            yield self._trial._persistent_conn
        else:
            conn = sqlite3.connect(self._trial.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def query_time_range(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        entity_type: str = "all",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query events across all ledgers within a time range.

        Args:
            start: Inclusive start time (None = no lower bound)
            end: Inclusive end time (None = no upper bound)
            entity_type: "campaign", "trial", "decision", or "all"

        Returns:
            Dict with keys "campaigns", "trials", "decisions" containing matching events.
        """
        result = {"campaigns": [], "trials": [], "decisions": []}

        start_str = start.isoformat() if start else None
        end_str = end.isoformat() if end else None

        # Query campaign ledger
        if entity_type in ("all", "campaign"):
            with self._conn() as conn:
                if start_str and end_str:
                    cursor = conn.execute(
                        """
                        SELECT * FROM campaign_events
                        WHERE timestamp >= ? AND timestamp <= ?
                        ORDER BY timestamp ASC, id ASC
                        """,
                        (start_str, end_str),
                    )
                elif start_str:
                    cursor = conn.execute(
                        """
                        SELECT * FROM campaign_events
                        WHERE timestamp >= ?
                        ORDER BY timestamp ASC, id ASC
                        """,
                        (start_str,),
                    )
                elif end_str:
                    cursor = conn.execute(
                        """
                        SELECT * FROM campaign_events
                        WHERE timestamp <= ?
                        ORDER BY timestamp ASC, id ASC
                        """,
                        (end_str,),
                    )
                else:
                    cursor = conn.execute(
                        "SELECT * FROM campaign_events ORDER BY timestamp ASC, id ASC"
                    )
                result["campaigns"] = [dict(row) for row in cursor.fetchall()]

        # Query trial ledger
        if entity_type in ("all", "trial"):
            with self._trial_conn() as tconn:
                if start_str and end_str:
                    cursor = tconn.execute(
                        """
                        SELECT * FROM trial_events
                        WHERE timestamp >= ? AND timestamp <= ?
                        ORDER BY timestamp ASC, id ASC
                        """,
                        (start_str, end_str),
                    )
                elif start_str:
                    cursor = tconn.execute(
                        """
                        SELECT * FROM trial_events
                        WHERE timestamp >= ?
                        ORDER BY timestamp ASC, id ASC
                        """,
                        (start_str,),
                    )
                elif end_str:
                    cursor = tconn.execute(
                        """
                        SELECT * FROM trial_events
                        WHERE timestamp <= ?
                        ORDER BY timestamp ASC, id ASC
                        """,
                        (end_str,),
                    )
                else:
                    cursor = tconn.execute(
                        "SELECT * FROM trial_events ORDER BY timestamp ASC, id ASC"
                    )
                result["trials"] = [dict(row) for row in cursor.fetchall()]

        # Query decision ledger if available
        if self._decision and entity_type in ("all", "decision"):
            # DecisionLedger has its own db_path; defer to its query interface
            result["decisions"] = self._query_decision_range(start_str, end_str)

        return result

    def _query_decision_range(
        self, start_str: Optional[str], end_str: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Query decision ledger time range (if attached)."""
        if not self._decision:
            return []
        with sqlite3.connect(self._decision.db_path) as dconn:
            dconn.row_factory = sqlite3.Row
            if start_str and end_str:
                cursor = dconn.execute(
                    "SELECT * FROM decision_events WHERE timestamp >= ? AND timestamp <= ?",
                    (start_str, end_str),
                )
            elif start_str:
                cursor = dconn.execute(
                    "SELECT * FROM decision_events WHERE timestamp >= ?",
                    (start_str,),
                )
            elif end_str:
                cursor = dconn.execute(
                    "SELECT * FROM decision_events WHERE timestamp <= ?",
                    (end_str,),
                )
            else:
                cursor = dconn.execute("SELECT * FROM decision_events")
            return [dict(row) for row in cursor.fetchall()]

    def reconstruct_lineage(self, campaign_id: str) -> Dict[str, Any]:
        """
        Reconstruct full lineage for a campaign: states, trials, decisions.

        Returns:
            Dict with keys:
                - campaign_id: str
                - state_history: List of campaign state transitions
                - current_state: Optional[str]
                - trials: Dict[trial_id, List[events]]
                - decisions: Dict[decision_id, List[events]] (if decision ledger attached)
                - timeline: Merged chronological event list with type tags
        """
        # Campaign state history
        state_history = self._campaign.get_campaign_history(campaign_id)
        current_state = self._campaign.get_current_state(campaign_id)

        # Trial history grouped by trial_id
        trial_events = self._trial.get_campaign_trials(campaign_id)
        trials: Dict[str, List[Dict[str, Any]]] = {}
        for event in trial_events:
            trials.setdefault(event["trial_id"], []).append(event)

        # Decision history grouped by decision_id (if available)
        decisions: Dict[str, List[Dict[str, Any]]] = {}
        if self._decision:
            decisions = self._decision.get_campaign_decisions(campaign_id)  # type: ignore[attr-defined]

        # Build unified timeline
        timeline = self._merge_timeline(state_history, trial_events, decisions)

        return {
            "campaign_id": campaign_id,
            "state_history": state_history,
            "current_state": current_state,
            "trials": trials,
            "decisions": decisions,
            "timeline": timeline,
        }

    def _merge_timeline(
        self,
        state_history: List[Dict[str, Any]],
        trial_events: List[Dict[str, Any]],
        decisions: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Merge campaign/trial/decision events into a single chronological timeline."""
        merged: List[Dict[str, Any]] = []

        for evt in state_history:
            merged.append({
                "timestamp": evt["timestamp"],
                "type": "campaign_state",
                "entity_id": evt["campaign_id"],
                "state": evt["state"],
                "event_id": evt["event_id"],
            })

        for evt in trial_events:
            merged.append({
                "timestamp": evt["timestamp"],
                "type": "trial_event",
                "entity_id": evt["trial_id"],
                "state": evt["state"],
                "campaign_id": evt["campaign_id"],
                "event_id": evt["event_id"],
            })

        for dec_id, dec_events in decisions.items():
            for evt in dec_events:
                merged.append({
                    "timestamp": evt["timestamp"],
                    "type": "decision_event",
                    "entity_id": dec_id,
                    "state": evt.get("decision_type") or evt.get("outcome"),
                    "event_id": evt["event_id"],
                })

        # Sort by timestamp; stable sort preserves insertion order for ties
        merged.sort(key=lambda e: (e["timestamp"], e.get("event_id", "")))
        return merged

    def get_state_at(
        self,
        campaign_id: str,
        timestamp: datetime,
    ) -> Optional[str]:
        """
        Return the campaign state that was active at a given timestamp.

        Returns None if the campaign did not exist at that time.
        """
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT state
                FROM campaign_events
                WHERE campaign_id = ? AND timestamp <= ?
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
                """,
                (campaign_id, timestamp.isoformat()),
            )
            row = cursor.fetchone()
            return row["state"] if row else None

    def get_history_in_range(
        self,
        campaign_id: str,
        start: datetime,
        end: datetime,
    ) -> List[Dict[str, Any]]:
        """Return campaign events within a time range."""
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM campaign_events
                WHERE campaign_id = ?
                  AND timestamp >= ?
                  AND timestamp <= ?
                ORDER BY timestamp ASC, id ASC
                """,
                (campaign_id, start.isoformat(), end.isoformat()),
            )
            return [dict(row) for row in cursor.fetchall()]

    def find_campaigns_active_during(
        self,
        start: datetime,
        end: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Find campaigns that were active (not yet terminal) during a time window.

        A campaign is considered active during [start, end] if it was created
        on or before start and reached a terminal state on or after end.
        """
        with self._conn() as conn:
            cursor = conn.execute(
                """
                WITH campaign_states AS (
                    SELECT
                        campaign_id,
                        state,
                        timestamp,
                        ROW_NUMBER() OVER (PARTITION BY campaign_id ORDER BY timestamp DESC, id DESC) as rn
                    FROM campaign_events
                ),
                latest AS (
                    SELECT campaign_id, state as final_state
                    FROM campaign_states
                    WHERE rn = 1
                ),
                created_at AS (
                    SELECT campaign_id, MIN(timestamp) as created_ts
                    FROM campaign_events
                    GROUP BY campaign_id
                ),
                terminal_at AS (
                    SELECT campaign_id, MIN(timestamp) as terminal_ts
                    FROM campaign_events
                    WHERE state IN ('completed', 'cancelled')
                    GROUP BY campaign_id
                )
                SELECT c.campaign_id, c.created_ts, t.terminal_ts, l.final_state
                FROM created_at c
                JOIN latest l ON c.campaign_id = l.campaign_id
                LEFT JOIN terminal_at t ON c.campaign_id = t.campaign_id
                WHERE c.created_ts <= ?
                  AND (t.terminal_ts IS NULL OR t.terminal_ts >= ?)
                """,
                (end.isoformat(), start.isoformat()),
            )
            return [dict(row) for row in cursor.fetchall()]

    def count_events_in_range(
        self,
        start: datetime,
        end: datetime,
    ) -> Dict[str, int]:
        """Count events per entity type within a time range."""
        counts = {"campaigns": 0, "trials": 0, "decisions": 0}

        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) as cnt
                FROM campaign_events
                WHERE timestamp >= ? AND timestamp <= ?
                """,
                (start.isoformat(), end.isoformat()),
            ).fetchone()
            counts["campaigns"] = row["cnt"]

        with self._trial_conn() as tconn:
            row = tconn.execute(
                """
                SELECT COUNT(*) as cnt
                FROM trial_events
                WHERE timestamp >= ? AND timestamp <= ?
                """,
                (start.isoformat(), end.isoformat()),
            ).fetchone()
            counts["trials"] = row["cnt"]

        if self._decision:
            with sqlite3.connect(self._decision.db_path) as dconn:
                dconn.row_factory = sqlite3.Row
                row = dconn.execute(
                    """
                    SELECT COUNT(*) as cnt
                    FROM decision_events
                    WHERE timestamp >= ? AND timestamp <= ?
                    """,
                    (start.isoformat(), end.isoformat()),
                ).fetchone()
                counts["decisions"] = row["cnt"]

        return counts