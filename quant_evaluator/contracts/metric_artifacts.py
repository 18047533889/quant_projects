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

Payload arrays are marked read-only on construction where numpy allows it,
so artifacts cannot be mutated in place after validation.
"""

from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from quant_evaluator.contracts._hashutil import stable_content_hex, stable_hash
from quant_evaluator.contracts.errors import InvalidContractError

__all__ = [
    "MetricArtifact",
    "ScalarMetricArtifact",
    "SeriesMetricArtifact",
    "VectorMetricArtifact",
    "MatrixMetricArtifact",
    "DistributionMetricArtifact",
]


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


def _freeze_array(value: Any, name: str) -> np.ndarray:
    """Validate that ``value`` is an ndarray-compatible array and return it
    read-only (lists/tuples are accepted as constructors of convenience)."""
    if not isinstance(value, (np.ndarray, list, tuple)):
        raise InvalidContractError(
            f"MetricArtifact.{name} must be a numpy ndarray, "
            f"got {type(value).__name__}"
        )
    array = np.asarray(value)
    try:
        array.flags.writeable = False
    except ValueError:
        # Some views (e.g. broadcast arrays) cannot be made read-only;
        # the artifact still must not hand out a mutable alias, so copy.
        array = array.copy()
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
            "matrix", "distribution").
        provenance: Mapping of provenance metadata (inputs, config hash,
            ...). Stored as an immutable plain-dict copy.
        created_from: Tuple of input names this artifact was derived from.
    """

    metric_id: str
    domain: str
    artifact_kind: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    created_from: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _check_metadata_string("metric_id", self.metric_id)
        _check_metadata_string("domain", self.domain)
        _check_metadata_string("artifact_kind", self.artifact_kind)
        if not isinstance(self.provenance, Mapping):
            raise InvalidContractError(
                "MetricArtifact.provenance must be a Mapping, got "
                f"{type(self.provenance).__name__}"
            )
        for key in self.provenance:
            _check_metadata_string("provenance key", key)
        # Immutable plain-dict copy: caller-supplied mappings must not be
        # mutable aliases into the frozen artifact.
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "created_from", tuple(self.created_from))
        _check_string_tuple("created_from", self.created_from)

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
        }
        for f in dataclass_fields(type(self)):
            if f.name in _base_field_names():
                continue
            fields_dict[f.name] = getattr(self, f.name, None)
        return stable_hash(
            stable_content_hex(tag=type(self).__name__, fields=fields_dict)
        )

    def _payload_dict(self) -> Dict[str, Any]:
        return {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly plain dict (arrays become lists)."""
        payload = {
            "metric_id": self.metric_id,
            "domain": self.domain,
            "artifact_kind": self.artifact_kind,
            "provenance": dict(self.provenance),
            "created_from": list(self.created_from),
        }
        payload.update(self._payload_dict())
        return payload

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
            "provenance": dict(data.get("provenance", {})),
            "created_from": tuple(data.get("created_from", ())),
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

    def _payload_dict(self) -> Dict[str, Any]:
        return {"values": self.values.tolist()}

    @staticmethod
    def _payload_from_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
        return {"values": np.asarray(data["values"])}


@dataclass(frozen=True, eq=False)
class SeriesMetricArtifact(MetricArtifact):
    """A time series per factor: values shape (T, F) plus a time index."""

    __artifact_kind__ = "series"

    values: np.ndarray = ()
    time_index: Tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
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

    def _payload_dict(self) -> Dict[str, Any]:
        return {"values": self.values.tolist(), "time_index": list(self.time_index)}

    @staticmethod
    def _payload_from_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "values": np.asarray(data["values"]),
            "time_index": tuple(data.get("time_index", ())),
        }


@dataclass(frozen=True, eq=False)
class VectorMetricArtifact(MetricArtifact):
    """A K-vector per factor (e.g. per-quantile returns): values (K, F)."""

    __artifact_kind__ = "vector"

    values: np.ndarray = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        values = _freeze_array(self.values, "values")
        if values.ndim != 2:
            raise InvalidContractError(
                f"VectorMetricArtifact.values must be (K, F), got shape {values.shape}"
            )
        object.__setattr__(self, "values", values)

    def _payload_dict(self) -> Dict[str, Any]:
        return {"values": self.values.tolist()}

    @staticmethod
    def _payload_from_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
        return {"values": np.asarray(data["values"])}


@dataclass(frozen=True, eq=False)
class MatrixMetricArtifact(MetricArtifact):
    """A (K, K) matrix per factor (e.g. transition matrices): values (K, K, F)."""

    __artifact_kind__ = "matrix"

    values: np.ndarray = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        values = _freeze_array(self.values, "values")
        if values.ndim != 3 or values.shape[0] != values.shape[1]:
            raise InvalidContractError(
                f"MatrixMetricArtifact.values must be (K, K, F), got shape {values.shape}"
            )
        object.__setattr__(self, "values", values)

    def _payload_dict(self) -> Dict[str, Any]:
        return {"values": self.values.tolist()}

    @staticmethod
    def _payload_from_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
        return {"values": np.asarray(data["values"])}


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

    def _payload_dict(self) -> Dict[str, Any]:
        return {"samples": self.samples.tolist(), "stat_names": list(self.stat_names)}

    @staticmethod
    def _payload_from_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "samples": np.asarray(data["samples"]),
            "stat_names": tuple(data.get("stat_names", ())),
        }
