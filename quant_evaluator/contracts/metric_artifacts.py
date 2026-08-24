"""Typed metric artifact contracts.

A :class:`MetricArtifact` is a frozen, self-describing container for one
computed metric: which metric produced it (``metric_id``), which domain it
belongs to, what kind of payload it carries (scalar / series / vector /
matrix / distribution), and where it came from (``provenance``,
``created_from``).

Shape conventions (F = number of factors, always the LAST axis):
    - ScalarMetricArtifact:       values (F,)
    - SeriesMetricArtifact:       values (T, F) with ``time_index`` (T,)
    - VectorMetricArtifact:       values (K, F)
    - MatrixMetricArtifact:       values (K, K, F)
    - DistributionMetricArtifact: samples (B, F) with ``stat_names``

QE-P0-01 (deep immutability): payload arrays are copied and marked read-only
on construction, and ``provenance`` is stored as a recursively-frozen
:class:`FrozenMapping` (a ``MappingProxyType``-wrapped dict) so nested
mutation fails.  ``created_from`` / ``time_index`` / ``stat_names`` are
tuples.  ``__hash__`` is stable across processes and unchanged under any
attempted mutation (mutation raises).

QE-P0-02 (array ownership): :func:`_freeze_array` copies the caller's buffer
(``np.array(value, copy=True, order="C")``) by default, so mutating the
original ndarray afterwards never changes the artifact.  A zero-copy escape is
available only through the explicit :class:`ImmutableBufferRef` ownership
capability.

QE-P0-03 (artifact_kind class authority): ``artifact_kind`` is derived from
the subclass (``ClassVar``), so a ScalarMetricArtifact cannot be constructed
as ``"vector"`` (``ValueError``).

QE-P0-04 (lossless serialization): ``to_dict`` / ``from_dict`` use a canonical
ndarray codec (dtype + shape + base64 payload bytes) instead of the lossy
``tolist()`` / ``np.asarray`` round-trip, preserving dtype, shape, endianness,
NaN/Inf bit patterns, and datetime64 axes exactly.
"""

import datetime
import decimal
import enum
import hashlib

from dataclasses import dataclass, field, fields as dataclass_fields
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from quant_evaluator.contracts._hashutil import stable_content_hex, stable_hash
from quant_evaluator.contracts._ndarray_codec import decode_value, encode_value
from quant_evaluator.contracts.axis_refs import (
    FactorAxisRef,
    MatrixAxisRefs,
    QuantileAxisRef,
    TimeAxisRef,
    axis_ref_from_dict,
)
from quant_evaluator.contracts.errors import InvalidContractError

__all__ = [
    "MetricArtifact",
    "ScalarMetricArtifact",
    "SeriesMetricArtifact",
    "VectorMetricArtifact",
    "MatrixMetricArtifact",
    "DistributionMetricArtifact",
    "FrozenMapping",
    "ImmutableBufferRef",
]


class FrozenMapping(Mapping):
    """A recursively-immutable read-only mapping.

    Wraps a plain dict in a :class:`types.MappingProxyType` so item assignment
    and deletion raise ``TypeError``.  Nested dicts / lists / sets are
    recursively frozen to tuples / frozensets / nested :class:`FrozenMapping`
    so no reachable value can be mutated through the artifact.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping):
        frozen = {k: _freeze_value(v) for k, v in data.items()}
        object.__setattr__(self, "_data", MappingProxyType(frozen))

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenMapping({dict(self._data)!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenMapping):
            return dict(self._data) == dict(other._data)
        if isinstance(other, Mapping):
            return dict(self._data) == dict(other)
        return NotImplemented

    def __hash__(self) -> int:
        # Stable across processes (unlike builtin hash(), which is salted by
        # PYTHONHASHSEED) and correct for ndarray values.  canonicalize is
        # fail-closed (QE-P0-06): unsupported object types raise instead of
        # producing an unstable repr-derived digest.
        return stable_hash(
            stable_content_hex(tag="FrozenMapping", fields=dict(self._data))
        )

    def __getstate__(self) -> dict:
        # Pickle as a plain dict; __setstate__ restores the read-only wrapper.
        return dict(self._data)

    def __setstate__(self, state: dict) -> None:
        frozen = {k: _freeze_value(v) for k, v in state.items()}
        object.__setattr__(self, "_data", MappingProxyType(frozen))


def _freeze_value(value: Any) -> Any:
    """Recursively freeze a value into an immutable form.

    dict -> FrozenMapping; list/tuple -> tuple; set/frozenset -> frozenset;
    ndarray -> read-only copy; immutable scalars/strings/bytes/None/Enum/
    numpy-scalar/datetime pass through unchanged; anything else (a mutable
    object) raises :class:`InvalidContractError` fail-closed.
    """
    if isinstance(value, Mapping):
        return FrozenMapping({k: _freeze_value(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(v) for v in value)
    if isinstance(value, np.ndarray):
        arr = np.array(value, copy=True, order="C")
        arr.flags.writeable = False
        return arr
    if (
        isinstance(value, (str, bytes, int, float, bool))
        or value is None
        or isinstance(value, (enum.Enum, np.generic, decimal.Decimal,
                              datetime.datetime, datetime.date,
                              datetime.timedelta))
    ):
        return value
    raise InvalidContractError(
        "_freeze_value: unsupported mutable value of type "
        f"{type(value).__name__} (fail-closed)"
    )


class ImmutableBufferRef:
    """Explicit zero-copy ownership capability for a caller ndarray.

    Wrapping an ndarray in an :class:`ImmutableBufferRef` tells
    :func:`_freeze_array` to adopt the caller's buffer WITHOUT copying it.
    The buffer is still marked read-only, so the artifact never hands out a
    mutable alias; the caller is asserting it will not mutate the buffer
    afterwards.  This is the only way to opt out of the default copy-on-write
    ownership (QE-P0-02).

    Real ownership (P1-14): the ref captures ``content_sha256`` = SHA-256 of
    the buffer's raw bytes at construction, so mutation-after-adoption by the
    caller is detectable.  Only a WRITABLE ndarray may be handed over — a
    read-only buffer is refused because the caller cannot be the owner of
    something already frozen.
    """

    __slots__ = ("array", "content_sha256")

    def __init__(self, array: np.ndarray):
        if not isinstance(array, np.ndarray):
            raise InvalidContractError(
                f"ImmutableBufferRef requires an ndarray, got {type(array).__name__}"
            )
        if not array.flags.writeable:
            raise InvalidContractError(
                "ImmutableBufferRef requires a WRITABLE ndarray so the caller "
                "can hand over real ownership; got a read-only buffer"
            )
        object.__setattr__(self, "array", array)
        object.__setattr__(self, "content_sha256", _buffer_sha256(array))

    def verify_untouched(self) -> None:
        """Raise :class:`InvalidContractError` if the buffer mutated since adoption.

        Recomputes SHA-256 over the CURRENT buffer bytes and compares with the
        hash captured at construction, detecting a caller mutating the buffer
        after the artifact adopted it (which would silently corrupt a
        supposedly-immutable artifact).
        """
        current = _buffer_sha256(self.array)
        if current != self.content_sha256:
            raise InvalidContractError(
                "ImmutableBufferRef buffer was mutated after adoption "
                f"(content_sha256 changed: expected {self.content_sha256[:12]}..., "
                f"got {current[:12]}...)"
            )


def _base_field_names() -> frozenset:
    return frozenset(f.name for f in dataclass_fields(MetricArtifact))


def _check_metadata_string(name: str, value: Any) -> None:
    """Validate that a metadata string field is a non-empty plain string.

    "Finite" in the sense of the artifact contract: a real (non-empty,
    non-whitespace, control-character-free) string, not None/NaN-castable
    or numeric garbage.
    """
    if not isinstance(value, str):
        raise InvalidContractError(
            f"MetricArtifact.{name} must be a string, got {type(value).__name__}"
        )
    if not value.strip():
        raise InvalidContractError(f"MetricArtifact.{name} must be non-empty")
    if any(ord(ch) < 32 for ch in value):
        raise InvalidContractError(
            f"MetricArtifact.{name} must not contain control characters"
        )


def _check_string_tuple(name: str, value: Any) -> None:
    if not isinstance(value, tuple):
        raise InvalidContractError(
            f"MetricArtifact.{name} must be a tuple, got {type(value).__name__}"
        )
    for item in value:
        _check_metadata_string(name, item)


def _buffer_sha256(array: np.ndarray) -> str:
    """Canonical SHA-256 hex digest of an ndarray's raw buffer bytes."""
    return hashlib.sha256(array.tobytes()).hexdigest()


def _freeze_array(value: Any, name: str) -> np.ndarray:
    """Validate ``value`` and return a read-only ndarray the artifact owns.

    Default (QE-P0-02): the caller's buffer is COPIED (``copy=True``,
    C-contiguous) so mutating the original afterwards never changes the
    artifact.  A zero-copy escape is available only by passing an
    :class:`ImmutableBufferRef` (the caller asserts it will not mutate the
    buffer; it is still marked read-only).
    """
    if isinstance(value, ImmutableBufferRef):
        array = value.array
        if not isinstance(array, np.ndarray):
            raise InvalidContractError(
                f"MetricArtifact.{name} ImmutableBufferRef must wrap an ndarray, "
                f"got {type(array).__name__}"
            )
        try:
            # Adopt the caller's buffer read-only (zero-copy, QE-P0-02).
            array.flags.writeable = False
        except ValueError:
            array = array.copy()
            array.flags.writeable = False
        # P1-14 real ownership: if the caller mutated the buffer between
        # handing over the ref and adoption, the ref's captured hash no longer
        # matches the current bytes -- catch that before returning.
        value.verify_untouched()
        return array
    if not isinstance(value, (np.ndarray, list, tuple)):
        raise InvalidContractError(
            f"MetricArtifact.{name} must be a numpy ndarray, "
            f"got {type(value).__name__}"
        )
    array = np.array(value, copy=True, order="C")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class MetricArtifact:
    """Base class for typed metric artifacts.

    Attributes:
        metric_id: Non-empty identifier of the metric that produced this
            artifact (registry name or canonical dotted alias).
        domain: Non-empty domain tag (e.g. "ic", "coverage", "quantile").
        artifact_kind: Payload kind tag ("scalar", "series", "vector",
            "matrix", "distribution").  Derived from the subclass (QE-P0-03).
        provenance: Mapping of provenance metadata (inputs, config hash,
            ...).  Stored as a recursively-immutable :class:`FrozenMapping`.
        created_from: Tuple of input names this artifact was derived from.
    """

    metric_id: str
    domain: str
    artifact_kind: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)
    created_from: Tuple[str, ...] = ()
    factor_axis: Any = None
    production: bool = False

    def __post_init__(self) -> None:
        _check_metadata_string("metric_id", self.metric_id)
        _check_metadata_string("domain", self.domain)
        # QE-P0-03: artifact_kind is class-authoritative.  The subclass's
        # __artifact_kind__ is the single source of truth; a caller-supplied
        # artifact_kind must match it exactly (ScalarMetricArtifact cannot be
        # constructed as "vector").  An omitted/empty artifact_kind defaults to
        # the class kind.
        expected = type(self).__artifact_kind__
        if self.artifact_kind == "":
            object.__setattr__(self, "artifact_kind", expected)
        elif self.artifact_kind != expected:
            raise ValueError(
                f"{type(self).__name__}.artifact_kind must be {expected!r}, "
                f"got {self.artifact_kind!r}"
            )
        if not isinstance(self.provenance, Mapping):
            raise InvalidContractError(
                "MetricArtifact.provenance must be a Mapping, got "
                f"{type(self.provenance).__name__}"
            )
        for key in self.provenance:
            _check_metadata_string("provenance key", key)
        # Deep-immutable provenance: caller-supplied mappings must not be
        # mutable aliases into the frozen artifact, and nested dicts/lists/sets
        # are recursively frozen so no reachable value can be mutated.
        object.__setattr__(self, "provenance", FrozenMapping(self.provenance))
        object.__setattr__(self, "created_from", tuple(self.created_from))
        _check_string_tuple("created_from", self.created_from)
        # QE-P0-05: factor_axis must be a FactorAxisRef or None.  The concrete
        # F-dimension match is validated by each subclass against its payload.
        if self.factor_axis is not None and not isinstance(self.factor_axis, FactorAxisRef):
            raise InvalidContractError(
                "MetricArtifact.factor_axis must be a FactorAxisRef or None, got "
                f"{type(self.factor_axis).__name__}"
            )
        # P1-14: a production artifact MUST identify its factor axis with a real
        # FactorAxisRef; research/backtest artifacts may omit it (None stays
        # allowed for back-compat).
        if self.production and not isinstance(self.factor_axis, FactorAxisRef):
            raise InvalidContractError(
                "MetricArtifact.production=True requires factor_axis to be a "
                "FactorAxisRef, got "
                f"{'None' if self.factor_axis is None else type(self.factor_axis).__name__}"
            )

    def _validate_f_axis(self, f: int) -> None:
        """QE-P0-05: validate factor_axis against the payload's F dimension."""
        if self.factor_axis is not None and self.factor_axis.num_factors != f:
            raise InvalidContractError(
                f"{type(self).__name__}.factor_axis declares "
                f"{self.factor_axis.num_factors} factors but the payload has F={f}"
            )

    def __eq__(self, other: object) -> bool:
        """Value equality with elementwise ndarray comparison."""
        if type(self) is not type(other):
            return NotImplemented if not isinstance(other, MetricArtifact) else False
        for f in dataclass_fields(type(self)):
            if f.name in _base_field_names():
                continue
            mine = getattr(self, f.name, None)
            theirs = getattr(other, f.name, None)
            if isinstance(mine, np.ndarray) or isinstance(theirs, np.ndarray):
                mine = np.asarray(mine)
                theirs = np.asarray(theirs)
                if mine.shape != theirs.shape or not np.array_equal(
                    mine, theirs, equal_nan=True
                ):
                    return False
            elif mine != theirs:
                return False
        return (
            self.metric_id == other.metric_id
            and self.domain == other.domain
            and self.artifact_kind == other.artifact_kind
            and dict(self.provenance) == dict(other.provenance)
            and tuple(self.created_from) == tuple(other.created_from)
            and self.factor_axis == other.factor_axis
        )

    def __hash__(self) -> int:
        # Stable across processes (unlike builtin hash(), which is salted by
        # PYTHONHASHSEED). Includes every dataclass payload field (arrays hashed
        # by canonical bytes) so value-equal instances hash identically.
        fields_dict: Dict[str, Any] = {
            "metric_id": self.metric_id,
            "domain": self.domain,
            "artifact_kind": self.artifact_kind,
            "provenance": dict(self.provenance),
            "created_from": tuple(self.created_from),
            "factor_axis": (
                None if self.factor_axis is None else self.factor_axis.to_dict()
            ),
        }
        for f in dataclass_fields(type(self)):
            if f.name in _base_field_names():
                continue
            value = getattr(self, f.name, None)
            fields_dict[f.name] = self._axis_to_dict(f.name, value)
        return stable_hash(
            stable_content_hex(tag=type(self).__name__, fields=fields_dict)
        )

    def _axis_to_dict(self, name: str, value: Any) -> Any:
        """Serialize an axis-ref field for hashing (None stays None)."""
        if value is None:
            return None
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        return value

    def _payload_dict(self) -> Dict[str, Any]:
        return {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly plain dict (arrays via lossless codec)."""
        payload = {
            "metric_id": self.metric_id,
            "domain": self.domain,
            "artifact_kind": self.artifact_kind,
            "provenance": encode_value(dict(self.provenance)),
            "created_from": list(self.created_from),
            "factor_axis": self._axis_dict("factor_axis", self.factor_axis),
        }
        payload.update(self._payload_dict())
        return payload

    def _axis_dict(self, name: str, value: Any) -> Any:
        """Return the to_dict payload of an axis-ref field (None stays None)."""
        if value is None:
            return None
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        return value

    @staticmethod
    def _axis_from_dict(data: Any) -> Any:
        """Deserialize an axis ref payload via axis_ref_from_dict (None-safe)."""
        return axis_ref_from_dict(data)

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "MetricArtifact":
        """Deserialize an artifact from its ``to_dict`` payload."""
        kind = data.get("artifact_kind")
        for cls in (
            ScalarMetricArtifact,
            SeriesMetricArtifact,
            VectorMetricArtifact,
            MatrixMetricArtifact,
            DistributionMetricArtifact,
        ):
            if kind == cls.__artifact_kind__:
                return cls._from_dict(data)
        raise InvalidContractError(f"Unknown artifact_kind: {kind!r}")

    @classmethod
    def _from_dict(cls, data: Mapping[str, Any]) -> "MetricArtifact":
        kwargs: Dict[str, Any] = {
            "metric_id": data["metric_id"],
            "domain": data["domain"],
            "artifact_kind": data["artifact_kind"],
            "provenance": decode_value(data.get("provenance", {})),
            "created_from": tuple(data.get("created_from", ())),
            "factor_axis": axis_ref_from_dict(data.get("factor_axis")),
        }
        kwargs.update(cls._payload_from_dict(data))
        return cls(**kwargs)

    @staticmethod
    def _payload_from_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
        return {}


@dataclass(frozen=True, eq=False)
class ScalarMetricArtifact(MetricArtifact):
    """One scalar value per factor: values shape (F,)."""

    __artifact_kind__ = "scalar"

    values: np.ndarray = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        values = _freeze_array(self.values, "values")
        if values.ndim != 1:
            raise InvalidContractError(
                f"ScalarMetricArtifact.values must be (F,), got shape {values.shape}"
            )
        object.__setattr__(self, "values", values)
        self._validate_f_axis(values.shape[0])

    def _payload_dict(self) -> Dict[str, Any]:
        return {"values": encode_value(self.values)}

    @staticmethod
    def _payload_from_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
        return {"values": decode_value(data["values"])}


@dataclass(frozen=True, eq=False)
class SeriesMetricArtifact(MetricArtifact):
    """A time series per factor: values shape (T, F) plus a time index."""

    __artifact_kind__ = "series"

    values: np.ndarray = ()
    time_index: Tuple[Any, ...] = ()
    time_axis: Any = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.time_axis is not None and not isinstance(self.time_axis, TimeAxisRef):
            raise InvalidContractError(
                "SeriesMetricArtifact.time_axis must be a TimeAxisRef or None, got "
                f"{type(self.time_axis).__name__}"
            )
        values = _freeze_array(self.values, "values")
        if values.ndim != 2:
            raise InvalidContractError(
                f"SeriesMetricArtifact.values must be (T, F), got shape {values.shape}"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "time_index", tuple(self.time_index))
        if len(self.time_index) != values.shape[0]:
            raise InvalidContractError(
                f"SeriesMetricArtifact.time_index length {len(self.time_index)} "
                f"does not match T={values.shape[0]}"
            )
        if self.time_axis is not None and self.time_axis.num_times != values.shape[0]:
            raise InvalidContractError(
                f"SeriesMetricArtifact.time_axis declares {self.time_axis.num_times} "
                f"times but the payload has T={values.shape[0]}"
            )
        self._validate_f_axis(values.shape[1])

    def _payload_dict(self) -> Dict[str, Any]:
        return {
            "values": encode_value(self.values),
            "time_index": encode_value(self.time_index),
            "time_axis": self._axis_dict("time_axis", self.time_axis),
        }

    @staticmethod
    def _payload_from_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "values": decode_value(data["values"]),
            "time_index": tuple(decode_value(data.get("time_index", ()))),
            "time_axis": axis_ref_from_dict(data.get("time_axis")),
        }


@dataclass(frozen=True, eq=False)
class VectorMetricArtifact(MetricArtifact):
    """A K-vector per factor (e.g. per-quantile returns): values (K, F)."""

    __artifact_kind__ = "vector"

    values: np.ndarray = ()
    quantile_axis: Any = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.quantile_axis is not None and not isinstance(self.quantile_axis, QuantileAxisRef):
            raise InvalidContractError(
                "VectorMetricArtifact.quantile_axis must be a QuantileAxisRef or None, got "
                f"{type(self.quantile_axis).__name__}"
            )
        values = _freeze_array(self.values, "values")
        if values.ndim != 2:
            raise InvalidContractError(
                f"VectorMetricArtifact.values must be (K, F), got shape {values.shape}"
            )
        object.__setattr__(self, "values", values)
        if self.quantile_axis is not None and self.quantile_axis.num_quantiles != values.shape[0]:
            raise InvalidContractError(
                f"VectorMetricArtifact.quantile_axis declares "
                f"{self.quantile_axis.num_quantiles} quantiles but the payload has K={values.shape[0]}"
            )
        self._validate_f_axis(values.shape[1])

    def _payload_dict(self) -> Dict[str, Any]:
        return {
            "values": encode_value(self.values),
            "quantile_axis": self._axis_dict("quantile_axis", self.quantile_axis),
        }

    @staticmethod
    def _payload_from_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "values": decode_value(data["values"]),
            "quantile_axis": axis_ref_from_dict(data.get("quantile_axis")),
        }


@dataclass(frozen=True, eq=False)
class MatrixMetricArtifact(MetricArtifact):
    """A (K, K) matrix per factor (e.g. transition matrices): values (K, K, F)."""

    __artifact_kind__ = "matrix"

    values: np.ndarray = ()
    matrix_axis: Any = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.matrix_axis is not None and not isinstance(self.matrix_axis, MatrixAxisRefs):
            raise InvalidContractError(
                "MatrixMetricArtifact.matrix_axis must be a MatrixAxisRefs or None, got "
                f"{type(self.matrix_axis).__name__}"
            )
        values = _freeze_array(self.values, "values")
        if values.ndim != 3 or values.shape[0] != values.shape[1]:
            raise InvalidContractError(
                f"MatrixMetricArtifact.values must be (K, K, F), got shape {values.shape}"
            )
        object.__setattr__(self, "values", values)
        if self.matrix_axis is not None and self.matrix_axis.num_quantiles != values.shape[0]:
            raise InvalidContractError(
                f"MatrixMetricArtifact.matrix_axis declares "
                f"{self.matrix_axis.num_quantiles} quantiles but the payload has K={values.shape[0]}"
            )
        self._validate_f_axis(values.shape[2])

    def _payload_dict(self) -> Dict[str, Any]:
        return {
            "values": encode_value(self.values),
            "matrix_axis": self._axis_dict("matrix_axis", self.matrix_axis),
        }

    @staticmethod
    def _payload_from_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "values": decode_value(data["values"]),
            "matrix_axis": axis_ref_from_dict(data.get("matrix_axis")),
        }


@dataclass(frozen=True, eq=False)
class DistributionMetricArtifact(MetricArtifact):
    """Bootstrap/simulation samples per factor: samples (B, F) + stat names."""

    __artifact_kind__ = "distribution"

    samples: np.ndarray = ()
    stat_names: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        samples = _freeze_array(self.samples, "samples")
        if samples.ndim != 2:
            raise InvalidContractError(
                f"DistributionMetricArtifact.samples must be (B, F), "
                f"got shape {samples.shape}"
            )
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "stat_names", tuple(self.stat_names))
        _check_string_tuple("stat_names", self.stat_names)
        self._validate_f_axis(samples.shape[1])

    def _payload_dict(self) -> Dict[str, Any]:
        return {
            "samples": encode_value(self.samples),
            "stat_names": list(self.stat_names),
        }

    @staticmethod
    def _payload_from_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "samples": decode_value(data["samples"]),
            "stat_names": tuple(data.get("stat_names", ())),
        }
