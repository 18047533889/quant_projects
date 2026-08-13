"""SeenCache: track previously evaluated factors to avoid duplicates."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Set


@dataclass
class SeenRecord:
    """
    Record of a previously seen factor.

    Attributes:
        canonical_hash: FE canonical identity hash
        first_seen_at: When first encountered
        trial_id: Trial ID from first evaluation
        factor_id: Factor ID if registered
        metadata: Additional metadata
    """

    canonical_hash: str
    first_seen_at: datetime
    trial_id: str
    factor_id: Optional[str] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "canonical_hash": self.canonical_hash,
            "first_seen_at": self.first_seen_at.isoformat(),
            "trial_id": self.trial_id,
            "factor_id": self.factor_id,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SeenRecord":
        """Deserialize from dictionary."""
        data = dict(data)
        if isinstance(data.get("first_seen_at"), str):
            data["first_seen_at"] = datetime.fromisoformat(data["first_seen_at"])
        return cls(**data)


class SeenCache:
    """
    Cache of previously evaluated factors by canonical identity.

    Uses FE canonical identity hash to detect duplicates. Does NOT reimplement
    FE's canonicalization logic; relies on FE adapter.
    """

    def __init__(self, fe_adapter=None):
        """
        Initialize seen cache.

        Args:
            fe_adapter: Optional FE adapter for canonical identity computation
        """
        self.fe_adapter = fe_adapter
        self._seen: Dict[str, SeenRecord] = {}
        self._by_trial: Dict[str, str] = {}  # trial_id -> canonical_hash

    def mark_seen(
        self, canonical_hash: str, trial_id: str, factor_id: Optional[str] = None, metadata: Optional[Dict] = None
    ) -> SeenRecord:
        """
        Mark a factor as seen.

        Args:
            canonical_hash: FE canonical identity hash
            trial_id: Trial ID
            factor_id: Optional registered factor ID
            metadata: Optional metadata

        Returns:
            SeenRecord (existing or newly created)
        """
        if canonical_hash in self._seen:
            # Already seen, update if needed
            record = self._seen[canonical_hash]
            if factor_id and not record.factor_id:
                # Update with factor_id if newly registered
                record.factor_id = factor_id
            return record

        # New record
        record = SeenRecord(
            canonical_hash=canonical_hash,
            first_seen_at=datetime.now(),
            trial_id=trial_id,
            factor_id=factor_id,
            metadata=metadata,
        )
        self._seen[canonical_hash] = record
        self._by_trial[trial_id] = canonical_hash

        return record

    def is_seen(self, canonical_hash: str) -> bool:
        """Check if a canonical hash has been seen."""
        return canonical_hash in self._seen

    def get_record(self, canonical_hash: str) -> Optional[SeenRecord]:
        """Get seen record by canonical hash."""
        return self._seen.get(canonical_hash)

    def get_by_trial(self, trial_id: str) -> Optional[SeenRecord]:
        """Get seen record by trial ID."""
        canonical_hash = self._by_trial.get(trial_id)
        if canonical_hash:
            return self._seen.get(canonical_hash)
        return None

    def compute_canonical_hash(self, factor_definition) -> str:
        """
        Compute canonical hash through FE adapter.

        Args:
            factor_definition: Factor definition (format depends on FE adapter)

        Returns:
            Canonical identity hash

        Raises:
            RuntimeError: If FE adapter not available
        """
        if self.fe_adapter is None:
            raise RuntimeError("FE adapter required for canonical hash computation")

        if not hasattr(self.fe_adapter, "compute_canonical_hash"):
            raise RuntimeError("FE adapter does not support canonical hash computation")

        return self.fe_adapter.compute_canonical_hash(factor_definition)

    def check_and_mark(self, factor_definition, trial_id: str) -> tuple[bool, Optional[SeenRecord]]:
        """
        Check if factor was seen and mark if new.

        Args:
            factor_definition: Factor definition
            trial_id: Current trial ID

        Returns:
            (was_seen, seen_record)
        """
        canonical_hash = self.compute_canonical_hash(factor_definition)
        was_seen = self.is_seen(canonical_hash)

        if was_seen:
            record = self.get_record(canonical_hash)
        else:
            record = self.mark_seen(canonical_hash, trial_id)

        return was_seen, record

    def size(self) -> int:
        """Return number of unique factors seen."""
        return len(self._seen)

    def clear(self) -> None:
        """Clear all seen records."""
        self._seen.clear()
        self._by_trial.clear()

    def export_records(self) -> list[Dict]:
        """Export all seen records as dictionaries."""
        return [record.to_dict() for record in self._seen.values()]

    def import_records(self, records: list[Dict]) -> None:
        """Import seen records from dictionaries."""
        for data in records:
            record = SeenRecord.from_dict(data)
            self._seen[record.canonical_hash] = record
            self._by_trial[record.trial_id] = record.canonical_hash
