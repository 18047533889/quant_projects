"""
Persistent seen history tracking with SQLite backend.

Extends in-memory SeenIndex with durable storage for production use.
"""

import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from factor_assets.seen_index.exact import SeenRecord


class PersistentSeenIndex:
    """
    Persistent seen index backed by SQLite.

    Provides exact seen history tracking with durable storage.
    All operations are transactional and crash-safe.
    """

    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize persistent seen index.

        Args:
            db_path: Path to SQLite database file (":memory:" for in-memory)
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize database schema."""
        self._conn = sqlite3.connect(self.db_path)
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
            CREATE INDEX IF NOT EXISTS idx_factor_id ON seen_factors(factor_id)
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

        # Check if already exists
        cursor = self._conn.execute(
            "SELECT * FROM seen_factors WHERE canonical_hash = ?",
            (canonical_hash,)
        )
        row = cursor.fetchone()

        if row:
            # Return existing record
            return SeenRecord(
                canonical_hash=row[0],
                factor_id=row[1],
                first_seen_at=row[2],
                origin=row[3],
                origin_ref=row[4],
                structural_hash=row[5],
            )

        # Create new record
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO seen_factors
            (canonical_hash, factor_id, first_seen_at, origin, origin_ref, structural_hash)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (canonical_hash, factor_id, now, origin, origin_ref, structural_hash)
        )
        self._conn.commit()

        return SeenRecord(
            canonical_hash=canonical_hash,
            factor_id=factor_id,
            first_seen_at=now,
            origin=origin,
            origin_ref=origin_ref,
            structural_hash=structural_hash,
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

        return SeenRecord(
            canonical_hash=row[0],
            factor_id=row[1],
            first_seen_at=row[2],
            origin=row[3],
            origin_ref=row[4],
            structural_hash=row[5],
        )

    def get_all(self) -> list[SeenRecord]:
        """Get all seen records."""
        cursor = self._conn.execute("SELECT * FROM seen_factors ORDER BY first_seen_at")
        rows = cursor.fetchall()

        return [
            SeenRecord(
                canonical_hash=row[0],
                factor_id=row[1],
                first_seen_at=row[2],
                origin=row[3],
                origin_ref=row[4],
                structural_hash=row[5],
            )
            for row in rows
        ]

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
