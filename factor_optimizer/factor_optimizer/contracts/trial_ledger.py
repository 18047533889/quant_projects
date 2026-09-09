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
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    """An immutable, append-only record of a single proposal attempt.

    P0-FO-004: every entry carries ``previous_entry_hash`` and ``entry_hash`` so
    a truncated or reordered ledger is detected by re-verifying the chain.  The
    hashes cover the entry's own semantic fields (sequence / status / trial_id /
    failure_reason / recorded_at) PLUS the previous entry's hash, so deleting
    any tail entry breaks every subsequent hash.
    """

    sequence: int
    status: TrialStatus
    trial_id: Optional[str] = None
    failure_reason: Optional[str] = None
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    previous_entry_hash: str = "0" * 64
    entry_hash: str = ""
    hash_encoding_version: int = 2

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
        if self.entry_hash:
            expected = self.compute_entry_hash()
            if self.entry_hash != expected:
                raise ValueError("LedgerEntry.entry_hash does not match its content")
        else:
            object.__setattr__(self, "entry_hash", self.compute_entry_hash())

    def _hash_payload(self) -> bytes:
        if self.hash_encoding_version == 1:
            return (
                f"{self.sequence}|{self.status_value}|{self.trial_id}|"
                f"{self.failure_reason}|{self.recorded_at.isoformat()}|"
                f"{self.previous_entry_hash}"
            ).encode("utf-8")
        if self.hash_encoding_version != 2:
            raise ValueError("unsupported ledger hash encoding version")
        return json.dumps(
            {
                "sequence": self.sequence,
                "status": self.status_value,
                "trial_id": self.trial_id,
                "failure_reason": self.failure_reason,
                "recorded_at": self.recorded_at.isoformat(),
                "previous_entry_hash": self.previous_entry_hash,
                "hash_encoding_version": self.hash_encoding_version,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    def compute_entry_hash(self) -> str:
        return hashlib.sha256(self._hash_payload()).hexdigest()

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
            "previous_entry_hash": self.previous_entry_hash,
            "entry_hash": self.entry_hash,
            "hash_encoding_version": self.hash_encoding_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LedgerEntry":
        if not isinstance(data, dict):
            raise TypeError("LedgerEntry.from_dict requires a dict")
        values = dict(data)
        values.setdefault("hash_encoding_version", 1)
        recorded = values.get("recorded_at")
        if isinstance(recorded, str):
            values["recorded_at"] = datetime.fromisoformat(recorded)
        elif recorded is None:
            values["recorded_at"] = datetime.now(timezone.utc)
        entry = cls(**values)
        return entry

    def __str__(self) -> str:
        return (
            f"[{self.sequence}] {self.status_value}"
            + (f" trial={self.trial_id}" if self.trial_id else "")
            + (f" {self.failure_reason}" if self.failure_reason else "")
        )


class TrialLedger:
    """Append-only, hash-chained, order-validated record of every proposal attempt.

    P0-FO-004: the ledger is a hash chain — every entry's ``entry_hash`` covers
    the previous entry's hash, so a truncated tail or a reordered entry breaks
    re-verification.  ``seal()`` closes the ledger and records the head hash /
    count; a sealed ledger rejects further appends.
    """

    def __init__(self) -> None:
        self._entries: Tuple[LedgerEntry, ...] = ()
        self._closed: bool = False
        self._sealed_head_hash: Optional[str] = None
        self._sealed_entry_count: Optional[int] = None
        self._sealed_at: Optional[datetime] = None
        self._prior_seals: Tuple[Dict[str, Any], ...] = ()

    def append(
        self,
        status,
        trial_id: Optional[str] = None,
        failure_reason: Optional[str] = None,
        recorded_at: Optional[datetime] = None,
    ) -> LedgerEntry:
        """Append an immutable, hash-chained entry and return it."""
        if self._closed:
            raise ValueError("ledger is sealed and no longer accepts entries")
        status_value = self._coerce_sequence(status)
        seq = len(self._entries) + 1
        prev_hash = self._entries[-1].entry_hash if self._entries else "0" * 64
        entry = LedgerEntry(
            sequence=seq,
            status=status_value,
            trial_id=trial_id,
            failure_reason=failure_reason,
            recorded_at=recorded_at or datetime.now(timezone.utc),
            previous_entry_hash=prev_hash,
            entry_hash="",  # derived in __post_init__
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

    def seal(self) -> None:
        """Close the ledger and record the head hash / entry count.

        After ``seal()`` the ledger is immutable: no further appends and the
        sealed identity (``sealed_head_hash`` / ``sealed_entry_count`` /
        ``sealed_at``) can be bound by an OptimizationResult so a tampered or
        truncated ledger is caught on checkpoint reload.
        """
        if self._sealed_head_hash is not None:
            raise ValueError("ledger is already sealed")
        if self._sealed_entry_count is not None:
            raise ValueError("ledger is already sealed")
        self._closed = True
        self._sealed_head_hash = self._entries[-1].entry_hash if self._entries else "0" * 64
        self._sealed_entry_count = len(self._entries)
        self._sealed_at = datetime.now(timezone.utc)

    def begin_authorized_extension(self) -> None:
        """Open a new append epoch while retaining the prior immutable seal."""
        if not self.sealed:
            raise ValueError("only a sealed ledger can be extended")
        self.verify_chain()
        self._prior_seals = self._prior_seals + ({
            "head_hash": self._sealed_head_hash,
            "entry_count": self._sealed_entry_count,
            "sealed_at": self._sealed_at.isoformat() if self._sealed_at else None,
        },)
        self._sealed_head_hash = None
        self._sealed_entry_count = None
        self._sealed_at = None
        self._closed = False

    @property
    def sealed(self) -> bool:
        return self._sealed_head_hash is not None

    @property
    def sealed_head_hash(self) -> Optional[str]:
        return self._sealed_head_hash

    @property
    def sealed_entry_count(self) -> Optional[int]:
        return self._sealed_entry_count

    @property
    def sealed_at(self) -> Optional[datetime]:
        return self._sealed_at

    def verify_chain(self) -> None:
        """Re-verify the full hash chain, fail-closed on any tamper/truncation.

        Checks, in order: contiguous sequence, monotonic timestamps, the
        chained hashes (each entry's ``entry_hash`` matches its recomputation
        and covers the previous entry's hash), and — when sealed — that the
        head hash and entry count match the recorded seal.
        """
        prev_hash = "0" * 64
        for i, entry in enumerate(self._entries):
            if entry.sequence != i + 1:
                raise ValueError(
                    f"ledger entry {i} has non-contiguous sequence "
                    f"{entry.sequence} (expected {i + 1})"
                )
            if entry.previous_entry_hash != prev_hash:
                raise ValueError(
                    f"ledger entry {i} breaks the hash chain (previous hash "
                    f"mismatch); the ledger was reordered or truncated"
                )
            if entry.entry_hash != entry.compute_entry_hash():
                raise ValueError(
                    f"ledger entry {i} content hash mismatch; the ledger was "
                    "tampered"
                )
            if i > 0 and self._entries[i - 1].recorded_at > entry.recorded_at:
                raise ValueError(
                    f"ledger entry {i} timestamp precedes the previous entry; "
                    "the ledger was reordered or tampered"
                )
            prev_hash = entry.entry_hash
        if self._sealed_head_hash is not None:
            if self._sealed_entry_count != len(self._entries):
                raise ValueError(
                    f"sealed ledger entry count mismatch: expected "
                    f"{self._sealed_entry_count}, got {len(self._entries)}; the "
                    "ledger tail was truncated"
                )
            if prev_hash != self._sealed_head_hash:
                raise ValueError(
                    "sealed ledger head hash mismatch; the ledger was truncated "
                    "or tampered after sealing"
                )

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
            "sealed": self._sealed_head_hash is not None,
            "closed": self._closed,
            "sealed_head_hash": self._sealed_head_hash,
            "sealed_entry_count": self._sealed_entry_count,
            "sealed_at": self._sealed_at.isoformat() if self._sealed_at else None,
            "prior_seals": [dict(item) for item in self._prior_seals],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrialLedger":
        if not isinstance(data, dict) or "entries" not in data:
            raise ValueError("ledger checkpoint missing 'entries'")
        raw = data["entries"]
        if not isinstance(raw, list):
            raise ValueError("ledger entries must be a list")
        entries = [LedgerEntry.from_dict(item) for item in raw]
        ledger = cls()
        ledger._entries = tuple(entries)
        ledger._prior_seals = tuple(dict(item) for item in data.get("prior_seals", ()))
        # Re-seal when the checkpoint recorded a sealed ledger so the loaded
        # state rejects further appends and re-verifies the chain.
        if data.get("sealed"):
            sealed_at = data.get("sealed_at")
            if isinstance(sealed_at, str):
                sealed_at = datetime.fromisoformat(sealed_at)
            ledger._sealed_head_hash = data.get("sealed_head_hash")
            ledger._sealed_entry_count = data.get("sealed_entry_count")
            ledger._sealed_at = sealed_at
            ledger._closed = True
        else:
            ledger._closed = bool(data.get("closed", False))
        ledger.verify_chain()
        expected_next = len(entries) + 1
        if data.get("next_sequence", expected_next) != expected_next:
            raise ValueError("ledger checkpoint next_sequence is inconsistent")
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
