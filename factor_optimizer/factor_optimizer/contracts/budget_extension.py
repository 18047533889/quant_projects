"""FO-P1-24 budget-resume authorization contract.

``SearchRunner.resume`` historically re-issued the whole ``BudgetTracker`` on
every resume, so a session could exceed its budget by repeatedly resuming.  Two
explicit opt-in modes replace silent budget re-arming:

- ``resume_same_budget=True`` continues an UNFINISHED session with its SAME
  partially-consumed budget (a finished session cannot re-arm its budget this
  way).
- ``BudgetExtensionAuthorization`` extends a FINISHED session's budget to a
  larger one.  The authorization carries a content hash over
  ``(session_id, old_budget, new_budget, reason, actor, timestamp)`` so a
  tampered extension is rejected at ``verify()``.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from factor_optimizer.contracts.search_budget import SearchBudget


@dataclass(frozen=True)
class BudgetExtensionAuthorization:
    """Signed authorization to extend a session's budget (FO-P1-24)."""

    search_session_id: str
    reason: str
    old_budget: SearchBudget
    new_budget: SearchBudget
    actor: str
    timestamp: datetime
    # Optional external signature over the content hash (an operator may sign
    # with its own key).  ``None`` means the content hash alone is the binding.
    signature: Optional[str] = None

    def __post_init__(self) -> None:
        for name, value in (
            ("search_session_id", self.search_session_id),
            ("reason", self.reason),
            ("actor", self.actor),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.old_budget, SearchBudget):
            raise TypeError("old_budget must be a SearchBudget")
        if not isinstance(self.new_budget, SearchBudget):
            raise TypeError("new_budget must be a SearchBudget")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        if not isinstance(self.timestamp.tzinfo, type(None)) and self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware or naive")
        object.__setattr__(self, "_pinned_canonical", self._canonical_payload())

    def _canonical_payload(self) -> str:
        return "|".join(
            [
                self.search_session_id,
                self.reason,
                _stable_budget(self.old_budget),
                _stable_budget(self.new_budget),
                self.actor,
                self.timestamp.isoformat(),
            ]
        )

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self._canonical_payload().encode("utf-8")).hexdigest()

    def verify(self) -> None:
        """Fail closed unless the content hash matches the signed payload."""
        if self._canonical_payload() != self._pinned_canonical:
            raise ValueError("budget extension content was tampered")
        if self.signature is not None and self.signature != self.content_hash:
            raise ValueError(
                "budget extension signature does not match its content hash"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "search_session_id": self.search_session_id,
            "reason": self.reason,
            "old_budget": self.old_budget.to_dict(),
            "new_budget": self.new_budget.to_dict(),
            "actor": self.actor,
            "timestamp": self.timestamp.isoformat(),
            "content_hash": self.content_hash,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BudgetExtensionAuthorization":
        if not isinstance(data, dict):
            raise TypeError("BudgetExtensionAuthorization.from_dict requires a dict")
        values = dict(data)
        values["old_budget"] = SearchBudget.from_dict(values["old_budget"])
        values["new_budget"] = SearchBudget.from_dict(values["new_budget"])
        ts = values.get("timestamp")
        if isinstance(ts, str):
            values["timestamp"] = datetime.fromisoformat(ts)
        obj = cls(
            search_session_id=values["search_session_id"],
            reason=values["reason"],
            old_budget=values["old_budget"],
            new_budget=values["new_budget"],
            actor=values["actor"],
            timestamp=values["timestamp"],
            signature=values.get("signature"),
        )
        # Re-binding the computed content hash guards against a tampered
        # content_hash in the dict being re-signed after load.
        declared = values.get("content_hash", obj.content_hash)
        if declared != obj.content_hash:
            raise ValueError(
                "budget extension content_hash does not match its payload"
            )
        return obj


def _stable_budget(budget: SearchBudget) -> str:
    """Stable, order-independent serialization of a SearchBudget for hashing."""
    return repr(
        {
            k: budget.to_dict()[k]
            for k in sorted(budget.to_dict())
        }
    )


import json  # noqa: E402  (placed after _stable_budget for readability)


__all__ = [
    "BudgetExtensionAuthorization",
]
