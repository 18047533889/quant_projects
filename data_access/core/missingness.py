"""Typed, value-independent missingness provenance authority.

NaN alone cannot distinguish absence, exclusion, staleness, or an operator
failure.  Producers must provide explicit reason and semantic masks; consumers
carry this plane without re-inferring reasons from numeric values.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import json
import math

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


# Only absent observations can be imputed under this observation identity.
# A model estimate replacing an invalid calculation, an expired observation,
# or an unavailable label needs a distinct output role, not a usable original.
FILLABLE_REASONS = frozenset({
    MissingReason.RAW_MISSING.value,
    MissingReason.SOURCE_OUTAGE.value,
})
STRUCTURALLY_UNAVAILABLE_REASONS = frozenset({
    MissingReason.NOT_LISTED.value,
    MissingReason.MASK_EXCLUDED.value,
    MissingReason.BUDGET_NOT_RUN.value,
    MissingReason.LABEL_NOT_YET_MATURE.value,
})


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
        if np.any(filled & ~np.isin(reasons, tuple(FILLABLE_REASONS))):
            raise ValueError("this missing reason cannot be filled under the original output identity")
        if np.any(usable & np.isin(reasons, tuple(STRUCTURALLY_UNAVAILABLE_REASONS))):
            raise ValueError("structurally unavailable observations cannot be marked usable")
        if np.any(original & usable & ~filled):
            raise ValueError("originally missing observations are usable only when explicitly filled")
        if np.any(np.isinf(age) | (age < 0)):
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
            contiguous = np.ascontiguousarray(value)
            # A read-only owning ndarray can be made writeable again.  A view
            # backed by immutable bytes cannot have its evidence changed.
            frozen = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(value.shape)
            object.__setattr__(self, name, frozen)

    def to_arrow(self):
        """Transport the existing authority without inferring reasons from values.

        Axes/snapshot identity remain owned by the containing read or feature
        contract.  This table preserves the aligned plane shape and masks.
        """
        import pyarrow as pa

        fields = ("reasons", "original_missing", "filled", "usable", "age")
        table = pa.table({name: getattr(self, name).reshape(-1) for name in fields})
        metadata = {"schema": "data-access-missingness-v1", "shape": list(self.reasons.shape)}
        return table.replace_schema_metadata({
            b"data_access.missingness": json.dumps(metadata, sort_keys=True).encode("utf-8")
        })

    @classmethod
    def from_arrow(cls, table):
        """Revalidate authority after Arrow/Parquet transport; never trust a mask."""
        import pyarrow as pa

        if not isinstance(table, pa.Table):
            raise TypeError("missingness transport requires a pyarrow.Table")
        fields = ("reasons", "original_missing", "filled", "usable", "age")
        if table.column_names != list(fields):
            raise ValueError("missingness transport has unexpected fields or field order")
        try:
            metadata = json.loads((table.schema.metadata or {})[b"data_access.missingness"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("missingness transport requires valid schema metadata") from exc
        if not isinstance(metadata, dict) or metadata.get("schema") != "data-access-missingness-v1":
            raise ValueError("unsupported missingness transport schema")
        shape = metadata.get("shape")
        if (not isinstance(shape, list) or not shape or len(shape) > 32
                or any(type(n) is not int or n < 0 for n in shape)
                or math.prod(shape) != table.num_rows):
            raise ValueError("missingness transport shape does not match its rows")
        expected_types = (pa.string(), pa.bool_(), pa.bool_(), pa.bool_(), pa.float64())
        for name, expected in zip(fields, expected_types):
            if table.schema.field(name).type != expected or table[name].null_count:
                raise ValueError(f"missingness transport field {name!r} has invalid type or nulls")
        values = {
            name: table[name].to_numpy(zero_copy_only=False).reshape(tuple(shape))
            for name in fields
        }
        # Construct this authority, not a subclass that can override validation.
        return MissingReasonPlane(**values)

    def coverage(self) -> dict[str, int]:
        return {
            "total": int(self.reasons.size),
            "observed": int((self.reasons == MissingReason.OBSERVED.value).sum()),
            "usable": int(self.usable.sum()),
            "filled": int(self.filled.sum()),
            "original_missing": int(self.original_missing.sum()),
        }


__all__ = [
    "MissingReason", "MissingReasonPlane", "FILLABLE_REASONS",
    "STRUCTURALLY_UNAVAILABLE_REASONS",
]
