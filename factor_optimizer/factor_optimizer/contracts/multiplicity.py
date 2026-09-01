"""MultiplicityArtifact: full-proposal-process multiplicity tracking (DLIB-FO-006).

The historical search loop only recorded the trials that reached evaluation
("400 valid evaluated").  DLIB-FO-006 requires that the multiplicity artifact
track the FULL proposal process — every LLM proposal, whether it parse-failed,
was a duplicate, was valid-and-evaluated, or failed evaluation — so the
multiple-testing correction reflects the true number of hypotheses tested, not
just the survivors.

This artifact is deep-immutable and content-hashed.  It records the counts of
every proposal outcome plus the total number of proposals, so a downstream
multiple-testing correction can use the true hypothesis count.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MultiplicityArtifact:
    """Tracks the full proposal process for multiple-testing correction.

    Attributes:
        search_session_id: The search session this multiplicity belongs to.
        total_proposals: Total number of LLM proposals submitted.
        parse_failures: Proposals that failed to parse / validate.
        duplicates: Proposals that were duplicates of an already-seen trial.
        valid_evaluated: Proposals that were valid and successfully evaluated.
        failed_evaluations: Proposals that were valid but failed evaluation.
        illegal: Proposals rejected by the legality/grammar validator.
        created_at: Creation timestamp.
    """

    search_session_id: str
    total_proposals: int
    parse_failures: int
    duplicates: int
    valid_evaluated: int
    failed_evaluations: int
    illegal: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not isinstance(self.search_session_id, str) or not self.search_session_id.strip():
            raise ValueError("search_session_id must be a non-empty string")
        for name in (
            "total_proposals",
            "parse_failures",
            "duplicates",
            "valid_evaluated",
            "failed_evaluations",
            "illegal",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if not isinstance(self.created_at, datetime):
            raise TypeError("created_at must be a datetime")
        # The outcome counts must not exceed the total.
        accounted = (
            self.parse_failures
            + self.duplicates
            + self.valid_evaluated
            + self.failed_evaluations
            + self.illegal
        )
        if accounted > self.total_proposals:
            raise ValueError(
                "outcome counts exceed total_proposals; the multiplicity "
                "artifact cannot account for more proposals than were made"
            )
        object.__setattr__(self, "_canonical", self._canonical_payload())

    def _canonical_payload(self) -> str:
        return json.dumps(
            {
                "search_session_id": self.search_session_id,
                "total_proposals": self.total_proposals,
                "parse_failures": self.parse_failures,
                "duplicates": self.duplicates,
                "valid_evaluated": self.valid_evaluated,
                "failed_evaluations": self.failed_evaluations,
                "illegal": self.illegal,
                "created_at": self.created_at.isoformat(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self._canonical_payload().encode("utf-8")).hexdigest()

    def verify(self) -> None:
        """Fail closed if the artifact was altered after construction."""
        if self._canonical_payload() != self._canonical:
            raise ValueError("multiplicity artifact content was tampered")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "search_session_id": self.search_session_id,
            "total_proposals": self.total_proposals,
            "parse_failures": self.parse_failures,
            "duplicates": self.duplicates,
            "valid_evaluated": self.valid_evaluated,
            "failed_evaluations": self.failed_evaluations,
            "illegal": self.illegal,
            "created_at": self.created_at.isoformat(),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiplicityArtifact":
        if not isinstance(data, dict):
            raise TypeError("MultiplicityArtifact.from_dict requires a dict")
        created = data["created_at"]
        if isinstance(created, str):
            created = datetime.fromisoformat(created)
        obj = cls(
            search_session_id=data["search_session_id"],
            total_proposals=data["total_proposals"],
            parse_failures=data["parse_failures"],
            duplicates=data["duplicates"],
            valid_evaluated=data["valid_evaluated"],
            failed_evaluations=data["failed_evaluations"],
            illegal=data.get("illegal", 0),
            created_at=created,
        )
        if data.get("content_hash") not in (None, obj.content_hash):
            raise ValueError("multiplicity artifact content_hash does not match its payload")
        return obj


__all__ = ["MultiplicityArtifact"]
