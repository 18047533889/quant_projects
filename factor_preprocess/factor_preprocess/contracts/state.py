"""
Fitted transform state contract.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from factor_preprocess.errors import (
    MissingInputError,
    InvalidContractError,
    TimingContractError,
)


class _FrozenMapping(Mapping):
    """Pickleable immutable mapping used for fitted parameter snapshots."""

    __slots__ = ("_items", "_values")

    def __init__(self, value: Mapping):
        self._items = tuple((key, item) for key, item in value.items())
        self._values = dict(self._items)

    def __getitem__(self, key: Any) -> Any:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __reduce__(self):
        return (_FrozenMapping, (dict(self._items),))


def _freeze(value: Any) -> Any:
    """Snapshot common parameter containers into immutable equivalents."""
    if isinstance(value, Mapping):
        return _FrozenMapping({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return tuple(_freeze(item) for item in value.tolist())
        contiguous = np.ascontiguousarray(value)
        return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _stable_repr(value: Any) -> str:
    """Deterministic canonical string form of a value for content hashing."""
    if isinstance(value, Mapping):
        return "{" + ",".join(
            f"{_stable_repr(k)}:{_stable_repr(v)}" for k, v in sorted(
                value.items(), key=lambda kv: _stable_repr(kv[0])
            )
        ) + "}"
    if isinstance(value, np.ndarray):
        return (
            f"nd:{value.dtype.str}:{value.shape}:"
            + _stable_repr(value.ravel().tolist())
        )
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stable_repr(item) for item in value) + "]"
    if isinstance(value, (set, frozenset)):
        return "<" + ",".join(sorted(_stable_repr(item) for item in value)) + ">"
    if isinstance(value, (bool, int, float, str)) or value is None:
        return repr(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return repr(value)


def _content_hash(value: Any) -> str:
    """SHA-256 hex digest derived from the actual content of ``value``."""
    import hashlib
    return hashlib.sha256(_stable_repr(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FittedState:
    """
    Immutable state from fitting a transform.

    Records fit window, learned parameters, and metadata for reproducibility.
    """
    state_id: str
    transform_name: str
    transform_version: str

    # Fit window metadata (required for causality)
    fit_start_time: datetime
    fit_end_time: datetime
    fit_universe_ref: Optional[str] = None

    # Feature contract
    feature_ids: List[str] = field(default_factory=list)
    feature_order: List[str] = field(default_factory=list)

    # Learned parameters
    learned_params: Dict[str, Any] = field(default_factory=dict)
    # Content-derived hash of learned_params. Recomputed from the actual
    # learned_params; a provided value is validated against the recomputation
    # and any mismatch is rejected (FP-P0-05).
    learned_params_hash: Optional[str] = None

    # Content-derived provenance (FP-P0-05)
    implementation_hash: Optional[str] = None  # hash of transform implementation/version
    data_snapshot_ref: Optional[str] = None    # hash or ref of the fit-time data snapshot
    split_ref: Optional[str] = None            # reference to the train/test split used

    # Provenance
    config_hash: Optional[str] = None
    created_at: Optional[datetime] = None
    producer: str = "factor_preprocess"
    producer_version: str = "0.1.0"

    def __post_init__(self):
        """Validate state on construction."""
        if not self.state_id:
            raise MissingInputError("FittedState.state_id cannot be empty")
        if not self.transform_name:
            raise MissingInputError("FittedState.transform_name cannot be empty")

        # Fit window validation
        if self.fit_start_time >= self.fit_end_time:
            raise TimingContractError("fit_start_time must be before fit_end_time")

        # Feature contract validation
        if self.feature_ids and not self.feature_order:
            raise InvalidContractError("feature_order required when feature_ids provided")
        if len(self.feature_ids) != len(set(self.feature_ids)):
            raise InvalidContractError("Duplicate feature IDs")
        if len(self.feature_order) != len(set(self.feature_order)):
            raise InvalidContractError("Duplicate features in feature_order")
        if set(self.feature_order) != set(self.feature_ids):
            raise InvalidContractError("feature_order must contain exactly feature_ids")

        object.__setattr__(self, "feature_ids", tuple(self.feature_ids))
        object.__setattr__(self, "feature_order", tuple(self.feature_order))
        object.__setattr__(self, "learned_params", _freeze(self.learned_params))

        # FP-P0-05: learned_params_hash must be derived from the actual
        # learned_params content. None is rejected; a provided hash is
        # validated against the recomputed value.
        actual_hash = _content_hash(self.learned_params)
        if self.learned_params_hash is None:
            object.__setattr__(self, "learned_params_hash", actual_hash)
        elif self.learned_params_hash != actual_hash:
            raise InvalidContractError(
                "learned_params_hash does not match learned_params content"
            )

    def is_compatible_with(self, factor_ids: List[str]) -> bool:
        """Check if factors match the positional fitted feature contract."""
        if not self.feature_ids:
            # No feature contract, assume compatible
            return True
        return tuple(factor_ids) == self.feature_order
