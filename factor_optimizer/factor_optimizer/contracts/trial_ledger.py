"""Append-only TrialLedger for every proposal attempt (FO-P0-04).

The search loop burns budget for every proposal.  When a proposal raises, is a
non-``Trial``, collides with a seen id, or is otherwise pruned, the historical
runner recorded the failure either silently or as a ``DUPLICATE``-labelled
entry.  FO-P0-04 requires that EVERY proposal attempt leave a record in a
single append-only ledger, with the burned-budget outcomes distinguished from
true duplicate/illegal outcomes.

The ledger is append-only: entries are immutable once written, and ``from_dict``
validates against reordered/dropped entries so a checkpoint cannot silently
lose a burned-budget record.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from factor_optimizer.contracts.trial import Trial, TrialStatus

# The unified status vocabulary of every proposal attempt (FO-P0-04).
TRIAL_OUTCOMES = (
    "PROPOSED",
    "PROPOSAL_FAILED",
    "INVALID_PROPOSAL",
    "DUPLICATE",
    "ILLEGAL",
    "EVALUATION_FAILED",
    "PRUNED",
    "EVALUATED",
    "SELECTED",
)

TRIAL_STATUS_OUTCOMES = frozenset(TRIAL_OUTCOMES)


@dataclass(frozen=True)
class LedgerEntry:
    """An immutable, append-only record of a single proposal attempt."""

    sequence: int
    status: TrialStatus
    trial_id: Optional[str] = None
    failure_reason: Optional[str] = None
    recorded_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise ValueError("ledger entry sequence must be an integer")
        if self.sequence < 1:
            raise ValueError("ledger entry sequence must be >= 1")
        if not isinstance(self.status, (TrialStatus, str)):
            raise TypeError("ledger entry status must be a TrialStatus or string")
        if self.trial_id is not None and not isinstance(self.trial_id, str):
            raise TypeError("ledger entry trial_id must be a string or None")
        if self.failure_reason is not None and not isinstance(self.failure_reason, str):
            raise TypeError("ledger entry failure_reason must be a string or None")
        if not isinstance(self.recorded_at, datetime):
            raise TypeError("ledger entry recorded_at must be a datetime")

    @property
    def status_value(self) -> str:
        if isinstance(self.status, TrialStatus):
            return self.status.value
        return self.status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "status": self.status_value,
            "trial_id": self.trial_id,
            "failure_reason": self.failure_reason,
            "recorded_at": self.recorded_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LedgerEntry":
        if not isinstance(data, dict):
            raise TypeError("LedgerEntry.from_dict requires a dict")
        values = dict(data)
        recorded = values.get("recorded_at")
        if isinstance(recorded, str):
            values["recorded_at"] = datetime.fromisoformat(recorded)
        elif recorded is None:
            values["recorded_at"] = datetime.now()
        return cls(**values)

    def __str__(self) -> str:
        return (
            f"[{self.sequence}] {self.status_value}"
            + (f" trial={self.trial_id}" if self.trial_id else "")
            + (f" {self.failure_reason}" if self.failure_reason else "")
        )


class TrialLedger:
    """Append-only, order-validated record of every proposal attempt."""

    def __init__(self) -> None:
        self._entries: Tuple[LedgerEntry, ...] = ()
        self._closed: bool = False

    def append(
        self,
        status,
        trial_id: Optional[str] = None,
        failure_reason: Optional[str] = None,
        recorded_at: Optional[datetime] = None,
    ) -> LedgerEntry:
        """Append an immutable entry and return it."""
        if self._closed:
            raise ValueError("ledger is sealed and no longer accepts entries")
        status_value = self._coerce_sequence(status)
        seq = len(self._entries) + 1
        entry = LedgerEntry(
            sequence=seq,
            status=status_value,
            trial_id=trial_id,
            failure_reason=failure_reason,
            recorded_at=recorded_at or datetime.now(),
        )
        self._entries = self._entries + (entry,)
        return entry

    def append_proposal_failed(self, failure_reason: str) -> LedgerEntry:
        return self.append("PROPOSAL_FAILED", failure_reason=failure_reason)

    def append_invalid_proposal(self, failure_reason: str) -> LedgerEntry:
        return self.append("INVALID_PROPOSAL", failure_reason=failure_reason)

    def append_trial(self, status, trial_id, failure_reason=None) -> LedgerEntry:
        return self.append(status, trial_id=trial_id, failure_reason=failure_reason)

    @property
    def entries(self) -> Tuple[LedgerEntry, ...]:
        return self._entries

    def close(self) -> None:
        self._closed = True

    def __len__(self) -> int:
        return len(self._entries)

    def status_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in self._entries:
            key = entry.status_value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self._entries],
            "next_sequence": len(self._entries) + 1,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrialLedger":
        if not isinstance(data, dict) or "entries" not in data:
            raise ValueError("ledger checkpoint missing 'entries'")
        raw = data["entries"]
        if not isinstance(raw, list):
            raise ValueError("ledger entries must be a list")
        entries = [LedgerEntry.from_dict(item) for item in raw]
        # Validate append-only invariants fail-closed: contiguous sequences,
        # monotonic timestamps (>= previous), no reordering/dedup.
        seen = set()
        for i, entry in enumerate(entries):
            if entry.sequence != i + 1:
                raise ValueError(
                    f"ledger entry {i} has non-contiguous sequence "
                    f"{entry.sequence} (expected {i + 1})"
                )
            if entry.status not in TRIAL_STATUS_OUTCOMES:
                raise ValueError(f"ledger entry {i} has unknown status {entry.status!r}")
            if i > 0 and entries[i - 1].recorded_at > entry.recorded_at:
                raise ValueError(
                    f"ledger entry {i} timestamp precedes the previous entry; "
                    "the ledger was reordered or tampered"
                )
        ledger = cls()
        ledger._entries = tuple(entries)
        ledger._closed = False
        return ledger

    def _coerce_sequence(self, status) -> str:
        if isinstance(status, TrialStatus):
            status = status.value
        if not isinstance(status, str):
            raise TypeError("status must be a TrialStatus or string")
        if status not in TRIAL_STATUS_OUTCOMES:
            raise ValueError(
                f"unknown trial outcome status: {status!r}; must be one of "
                f"{sorted(TRIAL_STATUS_OUTCOMES)}"
            )
        return status


__all__ = [
    "LedgerEntry",
    "TrialLedger",
    "TRIAL_STATUS_OUTCOMES",
]
