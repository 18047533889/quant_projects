"""Immutable, axis-bound QE quantile memberships for safe aggregation reuse."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from quant_evaluator.contracts.factor_batch import AxisRef
from quant_evaluator.contracts.metric_artifacts import _freeze_array
from quant_evaluator.contracts.quantile_policy import (
    QuantileTiePolicy,
    validate_tie_policy,
)


@dataclass(frozen=True)
class QuantileAssignmentBatch:
    """Owned immutable memberships aligned to scoring time and asset axes.

    ``assignments`` is always ``(T, N, F)`` int32.  ``-1`` is the sole
    missing-membership sentinel; valid IDs are in ``[0, n_quantiles)``.
    Construction copies into QE's immutable-byte-backed array storage and
    never changes the caller array's writeable flag.
    """

    assignments: np.ndarray
    time_axis: AxisRef
    asset_axis: AxisRef
    factor_ids: Tuple[str, ...]
    n_quantiles: int
    tie_policy: QuantileTiePolicy = QuantileTiePolicy.MAX

    def __post_init__(self) -> None:
        raw = self.assignments
        if not isinstance(raw, np.ndarray):
            raise TypeError("assignments must be a numpy ndarray")
        if raw.ndim != 3:
            raise ValueError("assignments must have shape (T, N, F), including F=1")
        if raw.dtype != np.dtype(np.int32):
            raise TypeError("assignments must have dtype int32")
        if isinstance(self.n_quantiles, (bool, np.bool_)) or not isinstance(
            self.n_quantiles, (int, np.integer)
        ) or self.n_quantiles < 1:
            raise ValueError("n_quantiles must be a positive integer")
        if not isinstance(self.time_axis, AxisRef) or not isinstance(self.asset_axis, AxisRef):
            raise TypeError("time_axis and asset_axis must be AxisRef instances")
        if self.time_axis.values is None or self.asset_axis.values is None:
            raise ValueError("assignment axes require explicit coordinates")
        factor_ids = tuple(self.factor_ids)
        if not factor_ids or any(not isinstance(x, str) or not x.strip() for x in factor_ids):
            raise ValueError("factor_ids must contain non-empty names")
        if len(set(factor_ids)) != len(factor_ids):
            raise ValueError("factor_ids must be unique")
        expected = (self.time_axis.size, self.asset_axis.size, len(factor_ids))
        if raw.shape != expected:
            raise ValueError(f"assignment shape must be {expected}, got {raw.shape}")
        policy = validate_tie_policy(self.tie_policy)
        if np.any((raw < -1) | (raw >= self.n_quantiles)):
            raise ValueError("assignment IDs must be -1 or in [0, n_quantiles)")

        object.__setattr__(self, "assignments", _freeze_array(raw, "quantile assignments"))
        object.__setattr__(self, "factor_ids", factor_ids)
        object.__setattr__(self, "n_quantiles", int(self.n_quantiles))
        object.__setattr__(self, "tie_policy", policy)
