"""
Core factor batch contract.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import datetime
import decimal


@dataclass(frozen=True)
class AxisRef:
    """Reference to a data axis (time or asset)."""
    name: str
    dtype: str
    size: int
    values: Optional[np.ndarray] = None

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if name == 'values' and isinstance(value, np.ndarray) and value.dtype.hasobject:
            # Numpy object buffers cannot be backed by immutable bytes. Detach
            # public reads so re-enabling WRITEABLE cannot mutate owned axes.
            value = value.copy()
            value.flags.writeable = False
        return value

    def __post_init__(self):
        if isinstance(self.size, (bool, np.bool_)) or not isinstance(self.size, (int, np.integer)) or self.size < 0:
            raise ValueError("Axis size must be a nonnegative integer")
        if self.values is not None and len(self.values) != self.size:
            raise ValueError(f"Axis {self.name} size mismatch: values has {len(self.values)}, declared size {self.size}")
        if self.values is not None:
            coordinates = np.array(self.values, copy=True)
            if coordinates.ndim != 1:
                raise ValueError("Axis coordinates must be one-dimensional")
            if np.issubdtype(coordinates.dtype, np.number) and not np.isfinite(coordinates).all():
                raise ValueError("Axis coordinates must be finite")
            if np.issubdtype(coordinates.dtype, np.datetime64) and np.isnat(coordinates).any():
                raise ValueError("Axis coordinates cannot contain NaT")
            if coordinates.dtype.kind in ('U', 'S') and any(not str(v).strip() for v in coordinates):
                raise ValueError("Axis coordinates cannot be empty")
            if self.name.lower() in ('time', 'date', 'timestamp') and len(coordinates) > 1:
                if not all(coordinates[i] < coordinates[i+1] for i in range(len(coordinates)-1)):
                    raise ValueError("time coordinates must be strictly increasing")
            if coordinates.dtype.hasobject:
                scalar_types = (str, bytes, int, float, bool, decimal.Decimal,
                                datetime.datetime, datetime.date, datetime.timedelta,
                                np.generic)
                if any(not isinstance(v, scalar_types) for v in coordinates):
                    raise ValueError('Axis coordinates must be immutable scalar values')
                if any(isinstance(v, np.generic) and (v.dtype.hasobject or v.dtype.fields is not None)
                       for v in coordinates):
                    raise ValueError('Axis coordinates require explicit immutable scalars')
            if len(set(coordinates.tolist())) != self.size:
                raise ValueError("Axis coordinates must be unique")
            from .metric_artifacts import _freeze_array
            if coordinates.dtype.hasobject:
                # Scalar-only objects are safe behind this class's detached
                # public reads; generic artifact metadata has no such boundary.
                coordinates.flags.writeable = False
            else:
                coordinates = _freeze_array(coordinates, "axis coordinates")
            object.__setattr__(self, "values", coordinates)


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
        object.__setattr__(self, "factor_ids", tuple(self.factor_ids))
        if not self.factor_ids:
            raise ValueError("factor_ids cannot be empty")
        if any(not isinstance(fid, str) or not fid.strip() for fid in self.factor_ids):
            raise ValueError("factor_ids must contain nonempty strings")
        if len(set(self.factor_ids)) != len(self.factor_ids):
            raise ValueError("factor_ids must be unique")
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
        if self.validity is not None and self.validity.dtype != np.bool_:
            raise ValueError("Factor validity must have boolean dtype")
        from .metric_artifacts import _freeze_array, FrozenMapping
        if not np.issubdtype(self.values.dtype, np.number):
            raise ValueError("Factor values must have a numeric dtype")
        if np.issubdtype(self.values.dtype, np.complexfloating):
            raise ValueError("Factor values must be real")
        # Normalize the descriptive dtype to actual immutable storage. Never
        # advertise FP64 evidence for an FP32 buffer.
        object.__setattr__(self, "dtype", str(self.values.dtype))
        object.__setattr__(self, "values", _freeze_array(self.values, "factor values"))
        if self.validity is not None:
            object.__setattr__(self, "validity", _freeze_array(self.validity, "factor validity"))
        object.__setattr__(self, "context_refs", FrozenMapping(self.context_refs))

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
        return bool(self.validity[time_idx, asset_idx, factor_idx] and
                    np.isfinite(self.values[time_idx, asset_idx, factor_idx]))
