"""
Persistent seen history tracking with SQLite backend.

Extends in-memory SeenIndex with durable storage for production use.
"""

import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from factor_assets.seen_index.exact import SeenRecord


class PersistentSeenIndex:
    """
    Persistent seen index backed by SQLite.

    Provides exact seen history tracking with durable storage.
    All operations are transactional and crash-safe.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        busy_timeout_ms: int = 1_000,
        max_busy_retries: int = 3,
    ):
        """
        Initialize persistent seen index.

        Args:
            db_path: Path to SQLite database file (":memory:" for in-memory)
            busy_timeout_ms: SQLite lock wait per attempt in milliseconds
            max_busy_retries: Additional retries for transient lock contention
        """
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be non-negative")
        if max_busy_retries < 0:
            raise ValueError("max_busy_retries must be non-negative")

        self.db_path = db_path
        self.busy_timeout_ms = busy_timeout_ms
        self.max_busy_retries = max_busy_retries
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize database schema."""
        self._conn = sqlite3.connect(
            self.db_path,
            timeout=self.busy_timeout_ms / 1_000,
            check_same_thread=False,
        )
        self._conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS seen_factors (
                canonical_hash TEXT PRIMARY KEY,
                factor_id TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                origin TEXT NOT NULL,
                origin_ref TEXT,
                structural_hash TEXT
            )
        """)

        self._conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_factor_id ON seen_factors(factor_id)
        """)

        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_first_seen ON seen_factors(first_seen_at)
        """)

        self._conn.commit()

    def record(
        self,
        canonical_hash: str,
        factor_id: str,
        origin: str = "manual",
        origin_ref: Optional[str] = None,
        structural_hash: Optional[str] = None,
    ) -> SeenRecord:
        """
        Record a factor as seen with persistent storage.

        Args:
            canonical_hash: Canonical hash from FE
            factor_id: Factor identifier
            origin: Origin type
            origin_ref: Optional origin reference
            structural_hash: Optional structural hash

        Returns:
            SeenRecord (existing if already seen, new otherwise)

        Raises:
            ValueError: If canonical_hash or factor_id is empty
        """
        if not canonical_hash:
            raise ValueError("canonical_hash is required")
        if not factor_id:
            raise ValueError("factor_id is required")

        now = datetime.now(timezone.utc).isoformat()
        for attempt in range(self.max_busy_retries + 1):
            try:
                with self._lock, self._conn:
                    self._conn.execute(
                        """
                        INSERT INTO seen_factors
                        (canonical_hash, factor_id, first_seen_at, origin, origin_ref, structural_hash)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(canonical_hash) DO NOTHING
                        """,
                        (canonical_hash, factor_id, now, origin, origin_ref, structural_hash),
                    )
                    row = self._conn.execute(
                        "SELECT * FROM seen_factors WHERE canonical_hash = ?",
                        (canonical_hash,),
                    ).fetchone()
                return self._row_to_record(row)
            except sqlite3.OperationalError as exc:
                if not self._is_busy_error(exc) or attempt == self.max_busy_retries:
                    raise
                time.sleep(min(0.01 * (2 ** attempt), 0.1))

        raise RuntimeError("unreachable")

    @staticmethod
    def _is_busy_error(exc: sqlite3.OperationalError) -> bool:
        message = str(exc).lower()
        return "locked" in message or "busy" in message

    @staticmethod
    def _row_to_record(row: tuple) -> SeenRecord:
        return SeenRecord(
            canonical_hash=row[0],
            factor_id=row[1],
            first_seen_at=row[2],
            origin=row[3],
            origin_ref=row[4],
            structural_hash=row[5],
        )

    def is_seen(self, canonical_hash: str) -> bool:
        """Check if a factor has been seen before."""
        cursor = self._conn.execute(
            "SELECT 1 FROM seen_factors WHERE canonical_hash = ?",
            (canonical_hash,)
        )
        return cursor.fetchone() is not None

    def get(self, canonical_hash: str) -> Optional[SeenRecord]:
        """
        Get seen record by canonical hash.

        Args:
            canonical_hash: Canonical hash to look up

        Returns:
            SeenRecord if found, None otherwise
        """
        cursor = self._conn.execute(
            "SELECT * FROM seen_factors WHERE canonical_hash = ?",
            (canonical_hash,)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def get_all(self) -> list[SeenRecord]:
        """Get all seen records."""
        cursor = self._conn.execute("SELECT * FROM seen_factors ORDER BY first_seen_at")
        rows = cursor.fetchall()

        return [self._row_to_record(row) for row in rows]

    def count(self) -> int:
        """Get total number of seen factors."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM seen_factors")
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
