"""Typed, value-independent missingness provenance authority.

NaN alone cannot distinguish absence, exclusion, staleness, or an operator
failure.  Producers must provide explicit reason and semantic masks; consumers
carry this plane without re-inferring reasons from numeric values.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class MissingReason(str, Enum):
    OBSERVED = "observed"
    RAW_MISSING = "raw_missing"
    NOT_LISTED = "not_listed"
    SOURCE_OUTAGE = "source_outage"
    ARITHMETIC_INVALID = "arithmetic_invalid"
    MASK_EXCLUDED = "mask_excluded"
    STALE_EXPIRED = "stale_expired"
    BUDGET_NOT_RUN = "budget_not_run"
    LABEL_NOT_YET_MATURE = "label_not_yet_mature"


@dataclass(frozen=True, eq=False)
class MissingReasonPlane:
    """Aligned reason/mask/age evidence; no reason is inferred from a value."""

    reasons: Any
    original_missing: Any
    filled: Any
    usable: Any
    age: Any

    def __post_init__(self) -> None:
        reasons = np.asarray(self.reasons, dtype=str)
        original = np.asarray(self.original_missing)
        filled = np.asarray(self.filled)
        usable = np.asarray(self.usable)
        age = np.asarray(self.age, dtype=np.float64)
        if not reasons.shape or any(x.shape != reasons.shape for x in (original, filled, usable, age)):
            raise ValueError("missingness arrays must have one identical non-scalar shape")
        if any(x.dtype.kind != "b" for x in (original, filled, usable)):
            raise TypeError("original_missing, filled and usable must be boolean arrays")
        allowed = {reason.value for reason in MissingReason}
        unknown = sorted(set(reasons.reshape(-1)) - allowed)
        if unknown:
            raise ValueError(f"unknown missing reasons: {unknown}")
        if np.any(filled & ~original):
            raise ValueError("filled must remain a subset of original_missing")
        if np.any(original & usable & ~filled):
            raise ValueError("originally missing observations are usable only when explicitly filled")
        if np.any(np.isfinite(age) & (age < 0)):
            raise ValueError("missing age must be non-negative or NaN")
        if np.any(filled & ~np.isfinite(age)):
            raise ValueError("filled observations require a finite missing age")
        if np.any((reasons == MissingReason.OBSERVED.value) & original):
            raise ValueError("original_missing observations require an explicit non-observed reason")
        if np.any((reasons != MissingReason.OBSERVED.value) & ~original & usable):
            raise ValueError("excluded/invalid/non-run observations cannot be marked usable")
        for name, value in (
            ("reasons", reasons), ("original_missing", original),
            ("filled", filled), ("usable", usable), ("age", age),
        ):
            frozen = np.array(value, copy=True)
            frozen.flags.writeable = False
            object.__setattr__(self, name, frozen)

    def coverage(self) -> dict[str, int]:
        return {
            "total": int(self.reasons.size),
            "observed": int((~self.original_missing).sum()),
            "usable": int(self.usable.sum()),
            "filled": int(self.filled.sum()),
            "original_missing": int(self.original_missing.sum()),
        }


__all__ = ["MissingReason", "MissingReasonPlane"]
