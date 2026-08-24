"""
Fitted transform state contract.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional

import base64

import numpy as np

from factor_preprocess.errors import (
    MissingInputError,
    InvalidContractError,
    TimingContractError,
)


class StateKind(Enum):
    """Whether a transform requires fitted state."""

    STATELESS = "stateless"
    FITTED = "fitted"


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
    """Deterministic canonical string form of a value for content hashing.

    Fail-closed: unsupported object types raise ``TypeError`` instead of
    falling back to ``repr`` (which may embed memory addresses and is not
    cross-process stable) — FP-P0-13.
    """
    if isinstance(value, Mapping):
        return "{" + ",".join(
            f"{_stable_repr(k)}:{_stable_repr(v)}" for k, v in sorted(
                value.items(), key=lambda kv: _stable_repr(kv[0])
            )
        ) + "}"
    if isinstance(value, np.ndarray):
        # Object-dtype arrays are not content-stable (elements hash via
        # identity) -> fail closed.
        if value.dtype.hasobject:
            raise TypeError(
                "_stable_repr does not support object-dtype arrays "
                f"(shape={value.shape}, dtype={value.dtype})"
            )
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
    if isinstance(value, bytes):
        # Base64 is content-stable and cross-process deterministic.
        return "b64:" + base64.b64encode(value).decode("ascii")
    raise TypeError(
        "_stable_repr does not support type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _content_hash(value: Any) -> str:
    """SHA-256 hex digest derived from the actual content of ``value``."""
    import hashlib
    return hashlib.sha256(_stable_repr(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FittedState:
    """
    Immutable state from fitting a transform.

    Records fit window, learned parameters, and metadata for reproducibility.

    FP-P0-04: in production mode (``production=True``) the ``state_id`` is
    content-derived from the full provenance surface and caller-supplied
    ``state_id`` values are rejected (fail-closed). Production also requires
    the provenance fields (implementation_hash, data_snapshot_ref, split_ref,
    universe_ref, calendar_ref, fit_coordinate_hash, policy_hash).

    FP-P0-05: a ``StateKind.FITTED`` transform must carry a non-empty feature
    contract. Empty ``feature_ids`` must NOT fail-open for stateful
    transforms; ``is_compatible_with`` raises for a FITTED state with no
    feature contract.
    """
    transform_name: str
    transform_version: str
    fit_start_time: datetime
    fit_end_time: datetime
    state_id: str = ""

    state_kind: StateKind = StateKind.STATELESS
    fit_universe_ref: Optional[str] = None

    # Feature contract
    feature_ids: List[str] = field(default_factory=list)
    feature_order: List[str] = field(default_factory=list)

    # Learned parameters
    learned_params: Dict[str, Any] = field(default_factory=dict)
    # Content-derived hash of learned_params. Recomputed from the actual
    # learned_params; a provided value is validated against the recomputation
    # and any mismatch is rejected.
    learned_params_hash: Optional[str] = None

    # Content-derived provenance
    implementation_hash: Optional[str] = None  # hash of transform implementation/version
    data_snapshot_ref: Optional[str] = None    # hash or ref of the fit-time data snapshot
    split_ref: Optional[str] = None            # reference to the train/test split used
    universe_ref: Optional[str] = None         # universe used at fit time
    calendar_ref: Optional[str] = None         # calendar used at fit time
    fit_coordinate_hash: Optional[str] = None  # hash of the fit coordinates
    policy_hash: Optional[str] = None          # hash of the governing policy

    # Provenance
    config_hash: Optional[str] = None
    created_at: Optional[datetime] = None
    producer: str = "factor_preprocess"
    producer_version: str = "0.1.0"

    # FP-P0-04: when True, state_id is content-derived and provenance fields
    # are required (fail-closed).
    production: bool = False

    def __post_init__(self):
        """Validate state on construction."""
        if not self.transform_name:
            raise MissingInputError("FittedState.transform_name cannot be empty")

        # Fit window validation
        if self.fit_start_time >= self.fit_end_time:
            raise TimingContractError("fit_start_time must be before fit_end_time")

        # FP-P0-13: strictly normalize state_kind BEFORE the FITTED feature
        # contract check so a caller passing the STRING "fitted" cannot bypass
        # the fail-closed FITTED contract.
        if isinstance(self.state_kind, StateKind):
            state_kind = self.state_kind
        elif isinstance(self.state_kind, str):
            try:
                state_kind = StateKind(self.state_kind)
            except ValueError:
                raise InvalidContractError(
                    f"Invalid state_kind string: {self.state_kind!r}. "
                    f"Valid values: {[k.value for k in StateKind]}"
                )
        else:
            raise InvalidContractError(
                "state_kind must be a StateKind or its value string, got "
                f"{type(self.state_kind).__module__}."
                f"{type(self.state_kind).__qualname__}"
            )
        object.__setattr__(self, "state_kind", state_kind)

        # Feature contract validation
        if self.feature_ids and not self.feature_order:
            raise InvalidContractError("feature_order required when feature_ids provided")
        if len(self.feature_ids) != len(set(self.feature_ids)):
            raise InvalidContractError("Duplicate feature IDs")
        if len(self.feature_order) != len(set(self.feature_order)):
            raise InvalidContractError("Duplicate features in feature_order")
        if set(self.feature_order) != set(self.feature_ids):
            raise InvalidContractError("feature_order must contain exactly feature_ids")

        # FP-P0-05: a FITTED transform must carry a non-empty feature contract.
        # Empty feature_ids must NOT fail-open for stateful transforms.
        if state_kind == StateKind.FITTED and not self.feature_ids:
            raise InvalidContractError(
                "FITTED state requires a non-empty feature contract (fail-closed)"
            )

        object.__setattr__(self, "feature_ids", tuple(self.feature_ids))
        object.__setattr__(self, "feature_order", tuple(self.feature_order))
        object.__setattr__(self, "learned_params", _freeze(self.learned_params))

        # learned_params_hash must be derived from the actual learned_params
        # content. None is rejected; a provided hash is validated against the
        # recomputed value.
        actual_hash = _content_hash(self.learned_params)
        if self.learned_params_hash is None:
            object.__setattr__(self, "learned_params_hash", actual_hash)
        elif self.learned_params_hash != actual_hash:
            raise InvalidContractError(
                "learned_params_hash does not match learned_params content"
            )

        # FP-P0-04: production provenance + content-derived state_id.
        if self.production:
            required = {
                "implementation_hash": self.implementation_hash,
                "data_snapshot_ref": self.data_snapshot_ref,
                "split_ref": self.split_ref,
                "universe_ref": self.universe_ref,
                "calendar_ref": self.calendar_ref,
                "fit_coordinate_hash": self.fit_coordinate_hash,
                "policy_hash": self.policy_hash,
            }
            missing = [k for k, v in required.items() if not v]
            if missing:
                raise InvalidContractError(
                    f"production FittedState requires provenance fields: {missing}"
                )
            if self.state_id:
                raise InvalidContractError(
                    "state_id is content-derived in production; "
                    "caller-supplied state_id is rejected"
                )
            object.__setattr__(self, "state_id", self._derive_state_id())
        elif not self.state_id:
            raise MissingInputError("FittedState.state_id cannot be empty")

    def _derive_state_id(self) -> str:
        """Content-derived identity over the full provenance surface."""
        components = {
            "transform_name": self.transform_name,
            "transform_version": self.transform_version,
            "implementation_hash": self.implementation_hash,
            "learned_params_hash": self.learned_params_hash,
            "data_snapshot_ref": self.data_snapshot_ref,
            "split_ref": self.split_ref,
            "universe_ref": self.universe_ref,
            "calendar_ref": self.calendar_ref,
            "fit_coordinate_hash": self.fit_coordinate_hash,
            "feature_order": tuple(self.feature_order),
            "policy_hash": self.policy_hash,
        }
        return _content_hash(components)

    def is_compatible_with(self, factor_ids: List[str]) -> bool:
        """Check if factors match the positional fitted feature contract."""
        if self.state_kind == StateKind.FITTED:
            # Fail-closed: a FITTED state must have a feature contract.
            if not self.feature_ids:
                raise InvalidContractError(
                    "FITTED state has no feature contract (fail-closed)"
                )
            return tuple(factor_ids) == self.feature_order
        # STATELESS: no feature contract required.
        if not self.feature_ids:
            return True
        return tuple(factor_ids) == self.feature_order
