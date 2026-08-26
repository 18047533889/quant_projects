"""TimingContract + EvidenceStatus.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.13/§4.14 (spec
§39, §40). PURE stdlib frozen dataclass + enum. A-share timing semantics come
from DataAccess/execution contracts, never guessed by page or QE.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime

__all__ = ["TimingContract", "EvidenceStatus"]


@dataclass(frozen=True)
class TimingContract:
    """Unified timing fields for evaluation/materialization/feature-set (spec §39)."""

    decision_time: datetime | None = None
    signal_available_time: datetime | None = None
    first_executable_time: datetime | None = None
    label_start_time: datetime | None = None
    label_end_time: datetime | None = None


class EvidenceStatus(enum.Enum):
    """Evidence availability status (spec §40). Never zero-filled."""

    NOT_COMPUTED = "NOT_COMPUTED"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID_EVIDENCE = "INVALID_EVIDENCE"
    LABEL_NOT_MATURE = "LABEL_NOT_MATURE"
