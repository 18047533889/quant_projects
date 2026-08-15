"""
Core factor batch contract.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class AxisRef:
    """Reference to a data axis (time or asset)."""
    name: str
    dtype: str
    size: int
    values: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.values is not None and len(self.values) != self.size:
            raise ValueError(f"Axis {self.name} size mismatch: values has {len(self.values)}, declared size {self.size}")


@dataclass(frozen=True)
class FactorBatch:
    """
    Batch of factor values with explicit axes and validity.

    This is the primary input contract for evaluation. A single factor is represented
    as a batch with one factor_id. Batch-first is mandatory.
    """
    factor_ids: Tuple[str, ...] = field(default_factory=tuple)
    time_axis: AxisRef = field(default=None)
    asset_axis: AxisRef = field(default=None)
    values: np.ndarray = field(default=None)
    validity: Optional[np.ndarray] = field(default=None)
    layout: str = "wide"
    dtype: str = "float64"
    context_refs: Dict[str, Any] = field(default_factory=dict)
    value_hash: Optional[str] = None

    def __post_init__(self):
        if not self.factor_ids:
            raise ValueError("factor_ids cannot be empty")
        if self.time_axis is None or self.asset_axis is None:
            raise ValueError("time_axis and asset_axis are required")
        if self.values is None:
            raise ValueError("values cannot be None")

        expected_shape = (self.time_axis.size, self.asset_axis.size, len(self.factor_ids))
        if self.layout != "wide":
            raise ValueError(f"Unsupported FactorBatch layout: {self.layout}; only 'wide' is accepted")
        if self.values.ndim != 3 or self.values.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for layout={self.layout}: "
                f"expected {expected_shape}, got {self.values.shape}"
            )

        if self.validity is not None and self.validity.shape != self.values.shape:
            raise ValueError(
                f"Validity shape {self.validity.shape} must match values shape {self.values.shape}"
            )

    @property
    def num_factors(self) -> int:
        return len(self.factor_ids)

    @property
    def num_times(self) -> int:
        return self.time_axis.size

    @property
    def num_assets(self) -> int:
        return self.asset_axis.size

    def is_valid(self, time_idx: int, asset_idx: int, factor_idx: int) -> bool:
        """Check if a specific value is valid."""
        if self.validity is None:
            return not (np.isnan(self.values[time_idx, asset_idx, factor_idx]) or
                       np.isinf(self.values[time_idx, asset_idx, factor_idx]))
        return bool(self.validity[time_idx, asset_idx, factor_idx])
