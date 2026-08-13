"""Trial event recording with SQLite append-only ledger."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any


class TrialLedger:
    """
    Append-only ledger for trial execution events.

    Tracks trial lifecycle: submitted → running → completed/failed/timeout.
    Records parameters, metrics, and execution metadata.
    """

    VALID_STATES = {"submitted", "running", "completed", "failed", "timeout"}
    TERMINAL_STATES = {"completed", "failed", "timeout"}

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
        """Initialize trial ledger schema."""
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trial_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    trial_id TEXT NOT NULL,
                    campaign_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    parameters TEXT,
                    metrics TEXT,
                    metadata TEXT,
                    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trial_events_trial_id
                ON trial_events(trial_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trial_events_campaign_id
                ON trial_events(campaign_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trial_events_timestamp
                ON trial_events(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trial_events_state
                ON trial_events(state)
            """)

    def append(
        self,
        event_id: str,
        trial_id: str,
        campaign_id: str,
        state: str,
        timestamp: datetime,
        parameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Append trial event.

        Returns True if added, False if duplicate event_id (idempotent).
        Raises ValueError if state is invalid.
        """
        if state not in self.VALID_STATES:
            raise ValueError(f"Invalid trial state: {state}. Must be one of {self.VALID_STATES}")

        import json
        parameters_json = json.dumps(parameters) if parameters else None
        metrics_json = json.dumps(metrics) if metrics else None
        metadata_json = json.dumps(metadata) if metadata else None

        with self._conn() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO trial_events
                    (event_id, trial_id, campaign_id, state, timestamp, parameters, metrics, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        trial_id,
                        campaign_id,
                        state,
                        timestamp.isoformat(),
                        parameters_json,
                        metrics_json,
                        metadata_json,
                    ),
                )
                return True
            except sqlite3.IntegrityError:
                # Duplicate event_id - idempotent behavior
                return False

    def get_trial_history(self, trial_id: str) -> List[Dict[str, Any]]:
        """Return all events for a trial in chronological order."""
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT event_id, trial_id, campaign_id, state, timestamp,
                       parameters, metrics, metadata, recorded_at
                FROM trial_events
                WHERE trial_id = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (trial_id,),
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_campaign_trials(self, campaign_id: str) -> List[Dict[str, Any]]:
        """Return all trial events for a campaign in chronological order."""
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT event_id, trial_id, campaign_id, state, timestamp,
                       parameters, metrics, metadata, recorded_at
                FROM trial_events
                WHERE campaign_id = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (campaign_id,),
            )
            return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_current_state(self, trial_id: str) -> Optional[str]:
        """Return the most recent state for a trial, or None if not found."""
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT state
                FROM trial_events
                WHERE trial_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
                """,
                (trial_id,),
            )
            row = cursor.fetchone()
            return row["state"] if row else None

    def get_all_trials(self, campaign_id: Optional[str] = None) -> List[str]:
        """Return list of all trial IDs, optionally filtered by campaign."""
        with self._conn() as conn:
            if campaign_id:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT trial_id
                    FROM trial_events
                    WHERE campaign_id = ?
                    ORDER BY trial_id
                    """,
                    (campaign_id,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT DISTINCT trial_id
                    FROM trial_events
                    ORDER BY trial_id
                    """
                )
            return [row["trial_id"] for row in cursor.fetchall()]

    def get_completed_trials(self, campaign_id: str) -> List[str]:
        """Return trial IDs in completed state for a campaign."""
        with self._conn() as conn:
            cursor = conn.execute(
                """
                WITH latest_states AS (
                    SELECT
                        trial_id,
                        state,
                        ROW_NUMBER() OVER (PARTITION BY trial_id ORDER BY timestamp DESC, id DESC) as rn
                    FROM trial_events
                    WHERE campaign_id = ?
                )
                SELECT trial_id
                FROM latest_states
                WHERE rn = 1 AND state = 'completed'
                ORDER BY trial_id
                """,
                (campaign_id,),
            )
            return [row["trial_id"] for row in cursor.fetchall()]

    def get_trial_metrics(self, trial_id: str) -> Optional[Dict[str, float]]:
        """Return metrics from the most recent event with metrics for a trial."""
        with self._conn() as conn:
            cursor = conn.execute(
                """
                SELECT metrics
                FROM trial_events
                WHERE trial_id = ? AND metrics IS NOT NULL
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
                """,
                (trial_id,),
            )
            row = cursor.fetchone()
            if row and row["metrics"]:
                import json
                return json.loads(row["metrics"])
            return None

    def count_events(self, trial_id: Optional[str] = None, campaign_id: Optional[str] = None) -> int:
        """Count total events, optionally filtered by trial_id or campaign_id."""
        with self._conn() as conn:
            if trial_id:
                cursor = conn.execute(
                    "SELECT COUNT(*) as cnt FROM trial_events WHERE trial_id = ?",
                    (trial_id,),
                )
            elif campaign_id:
                cursor = conn.execute(
                    "SELECT COUNT(*) as cnt FROM trial_events WHERE campaign_id = ?",
                    (campaign_id,),
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) as cnt FROM trial_events")
            return cursor.fetchone()["cnt"]

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dict with JSON field parsing."""
        import json
        result = dict(row)
        for field in ["parameters", "metrics", "metadata"]:
            if result.get(field):
                result[field] = json.loads(result[field])
        return result
