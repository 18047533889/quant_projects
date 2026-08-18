"""
Exact seen history tracking.

Records first-seen timestamps and canonical identity for deduplication.
No raw factor values stored.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass(frozen=True)
class SeenRecord:
    """
    Record of a factor's first appearance.

    Tracks when and how a factor was first seen by the system.
    """
    canonical_hash: str
    factor_id: str
    first_seen_at: str  # ISO 8601
    origin: str  # "manual", "llm", "search", "corpus", "migration"
    origin_ref: Optional[str] = None
    structural_hash: Optional[str] = None

    def __post_init__(self):
        if not self.canonical_hash:
            raise ValueError("canonical_hash is required")
        if not self.factor_id:
            raise ValueError("factor_id is required")
        if not self.first_seen_at:
            raise ValueError("first_seen_at is required")


class SeenIndex:
    """
    In-memory index of seen factor identities.

    Provides exact seen history tracking for deduplication.
    """

    def __init__(self):
        self._seen: dict[str, SeenRecord] = {}

    def record(
        self,
        canonical_hash: str,
        factor_id: str,
        origin: str = "manual",
        origin_ref: Optional[str] = None,
        structural_hash: Optional[str] = None,
    ) -> SeenRecord:
        """
        Record a factor as seen.

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

        # Return existing record if already seen
        if canonical_hash in self._seen:
            return self._seen[canonical_hash]

        # A factor ID is an immutable identity, so a different canonical hash
        # cannot be recorded under an already-seen ID.
        existing = next(
            (record for record in self._seen.values() if record.factor_id == factor_id),
            None,
        )
        if existing is not None:
            raise ValueError(
                f"factor_id already recorded with canonical_hash {existing.canonical_hash}"
            )

        now = datetime.now(timezone.utc).isoformat()

        record = SeenRecord(
            canonical_hash=canonical_hash,
            factor_id=factor_id,
            first_seen_at=now,
            origin=origin,
            origin_ref=origin_ref,
            structural_hash=structural_hash,
        )

        self._seen[canonical_hash] = record
        return record

    def is_seen(self, canonical_hash: str) -> bool:
        """Check if a factor has been seen before."""
        return canonical_hash in self._seen

    def get(self, canonical_hash: str) -> Optional[SeenRecord]:
        """
        Get seen record by canonical hash.

        Args:
            canonical_hash: Canonical hash to look up

        Returns:
            SeenRecord if found, None otherwise
        """
        return self._seen.get(canonical_hash)

    def get_all(self) -> list[SeenRecord]:
        """Get all seen records."""
        return list(self._seen.values())

    def count(self) -> int:
        """Get total number of seen factors."""
        return len(self._seen)
