"""Campaign lifecycle tracking with SQLite append-only ledger."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict


class CampaignLedger:
    """
    Append-only ledger for campaign lifecycle events.

    Tracks campaign state transitions: created → started → paused/resumed → completed/cancelled.
    All operations are append-only; no updates or deletes permitted.
    """

    VALID_STATES = {"created", "started", "paused", "resumed", "completed", "cancelled"}
    TERMINAL_STATES = {"completed", "cancelled"}

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._persistent_conn = None

        # For in-memory databases, maintain a persistent connection
        if db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(db_path)
            self._persistent_conn.row_factory = sqlite3.Row

        self._init_schema()

    @contextmanager
    def _conn(self):
        """Context manager for database connections."""
        if self._persistent_conn:
            # For in-memory, use persistent connection
            try:
                yield self._persistent_conn
                self._persistent_conn.commit()
            except Exception:
                self._persistent_conn.rollback()
                raise
        else:
            # For file-based, create new connection each time
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_schema(self):
        """Initialize campaign ledger schema."""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS campaign_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    campaign_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_campaign_events_campaign_id
                ON campaign_events(campaign_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_campaign_events_timestamp
                ON campaign_events(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_campaign_events_state
                ON campaign_events(state)
            """)

    def append(
        self,
        event_id: str,
        campaign_id: str,
        state: str,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Append campaign state transition event.

        Returns True if added, False if duplicate event_id (idempotent).
        Raises ValueError if state is invalid.
        """
        if state not in self.VALID_STATES:
            raise ValueError(f"Invalid campaign state: {state}. Must be one of {self.VALID_STATES}")

        import json
        metadata_json = json.dumps(metadata) if metadata else None

        with self._conn() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO campaign_events (event_id, campaign_id, state, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (event_id, campaign_id, state, timestamp.isoformat(), metadata_json),
                )
                return True
            except sqlite3.IntegrityError:
                # Duplicate event_id - idempotent behavior
                return False

    def get_campaign_history(self, campaign_id: str) -> List[Dict[str, Any]]:
        """Return all state transitions for a campaign in chronological order."""
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT event_id, campaign_id, state, timestamp, metadata, recorded_at
                FROM campaign_events
                WHERE campaign_id = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (campaign_id,),
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_current_state(self, campaign_id: str) -> Optional[str]:
        """Return the most recent state for a campaign, or None if not found."""
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT state
                FROM campaign_events
                WHERE campaign_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
                """,
                (campaign_id,),
            )
            row = cursor.fetchone()
            return row["state"] if row else None

    def get_all_campaigns(self) -> List[str]:
        """Return list of all campaign IDs with events."""
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT DISTINCT campaign_id
                FROM campaign_events
                ORDER BY campaign_id
                """
            )
            return [row["campaign_id"] for row in cursor.fetchall()]

    def get_active_campaigns(self) -> List[str]:
        """Return campaigns not in terminal state (completed/cancelled)."""
        with self._conn() as conn:
            cursor = conn.execute(
                """
                WITH latest_states AS (
                    SELECT
                        campaign_id,
                        state,
                        ROW_NUMBER() OVER (PARTITION BY campaign_id ORDER BY timestamp DESC, id DESC) as rn
                    FROM campaign_events
                )
                SELECT campaign_id
                FROM latest_states
                WHERE rn = 1 AND state NOT IN ('completed', 'cancelled')
                ORDER BY campaign_id
                """
            )
            return [row["campaign_id"] for row in cursor.fetchall()]

    def count_events(self, campaign_id: Optional[str] = None) -> int:
        """Count total events, optionally filtered by campaign_id."""
        with self._conn() as conn:
            if campaign_id:
                cursor = conn.execute(
                    "SELECT COUNT(*) as cnt FROM campaign_events WHERE campaign_id = ?",
                    (campaign_id,),
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) as cnt FROM campaign_events")
            return cursor.fetchone()["cnt"]

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dict with JSON metadata parsing."""
        import json
        result = dict(row)
        if result.get("metadata"):
            result["metadata"] = json.loads(result["metadata"])
        return result
