"""Explicit axis references for metric artifacts (QE-P0-05).

Every :class:`~quant_evaluator.contracts.metric_artifacts.MetricArtifact`
whose payload's LAST dimension is F (the number of factors) must be able to
declare the axis identity of that factor dimension as a *first-class* field
— never hidden in free-form ``provenance``.  This module defines the typed,
frozen, serializable axis-reference contracts:

- :class:`FactorAxisRef`   — the F axis: factor ids, optional per-factor
  versions, and a stable content hash of the ordered id list.  ``factor_ids``
  are validated unique and non-empty; ``factor_order_hash`` must equal the
  canonical SHA-256 digest of the id tuple (deterministic across processes,
  independent of ``PYTHONHASHSEED``).
- :class:`TimeAxisRef`     — the T axis of a series payload: the time index
  and an optional IANA time zone.
- :class:`QuantileAxisRef` — the K axis of a quantile vector/matrix payload:
  human-readable quantile labels.
- :class:`MatrixAxisRefs`  — both axes of a (K, K, F) matrix payload.

Every axis ref is a frozen dataclass with ``to_dict`` / ``from_dict`` and
lossless pickle support (plain ``tuple``/``str``/``None`` fields), so axis
identity survives serialization and process boundaries.

Validation contract:

- ``FactorAxisRef``: ``factor_ids`` must be a non-empty tuple of non-empty
  strings with no duplicates; ``factor_versions`` (when provided) must be a
  tuple of the same length whose items are ``None`` or non-empty strings;
  ``factor_order_hash`` must be a non-empty hex string that exactly equals
  the canonical digest of ``factor_ids`` (computed via
  :func:`quant_evaluator.contracts._hashutil.stable_content_hex`).
- ``TimeAxisRef``: ``time_index`` (when provided) must be a tuple; items may
  be any orderable scalar (str/date/datetime/timestamp).  ``time_zone`` when
  provided must be a non-empty string.
- ``QuantileAxisRef``: ``quantile_labels`` must be a non-empty tuple of
  non-empty strings with no duplicates.
- ``MatrixAxisRefs``: both ``row`` and ``col`` must be :class:`QuantileAxisRef`
  instances; a matrix whose row and column axes are structurally identical
  (same labels) is expected and validated at construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from quant_evaluator.contracts._hashutil import stable_content_hex
from quant_evaluator.contracts.errors import InvalidContractError

__all__ = [
    "FactorAxisRef",
    "TimeAxisRef",
    "QuantileAxisRef",
    "MatrixAxisRefs",
    "axis_ref_from_dict",
]


def _check_plain_str(name: str, value: Any) -> None:
    """A non-empty, whitespace-free, control-character-free plain string."""
    if not isinstance(value, str):
        raise InvalidContractError(f"{name} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise InvalidContractError(f"{name} must be non-empty")
    if any(ord(ch) < 32 for ch in value):
        raise InvalidContractError(f"{name} must not contain control characters")


def _check_str_tuple(name: str, value: Any, *, allow_empty: bool = False) -> None:
    if not isinstance(value, tuple):
        raise InvalidContractError(f"{name} must be a tuple, got {type(value).__name__}")
    if not allow_empty and len(value) == 0:
        raise InvalidContractError(f"{name} must be non-empty")
    for item in value:
        _check_plain_str(name, item)


def _factor_order_hash(factor_ids: Tuple[str, ...]) -> str:
    """Canonical content digest of an ordered factor-id tuple."""
    return stable_content_hex(tag="FactorAxisRef.factor_ids", fields={"ids": list(factor_ids)})


@dataclass(frozen=True, init=True)
class FactorAxisRef:
    """Identity of the F (factor) axis of a metric-artifact payload.

    Attributes:
        factor_ids: Ordered tuple of factor identifiers.  Unique (a factor id
            appearing twice would make the axis ambiguous) and non-empty.
        factor_versions: Optional tuple aligned with ``factor_ids``; each
            entry is ``None`` (no version) or a non-empty version string.
        factor_order_hash: Stable SHA-256 hex digest of the ordered id tuple,
            validated to equal the canonical digest at construction.
    """

    factor_ids: Tuple[str, ...]
    factor_versions: Optional[Tuple[Optional[str], ...]] = None
    factor_order_hash: str = ""

    def __post_init__(self) -> None:
        _check_str_tuple("FactorAxisRef.factor_ids", self.factor_ids)
        if len(set(self.factor_ids)) != len(self.factor_ids):
            raise InvalidContractError(
                "FactorAxisRef.factor_ids must be unique; got duplicates: "
                f"{[i for i in self.factor_ids if self.factor_ids.count(i) > 1]}"
            )
        if self.factor_versions is not None:
            if not isinstance(self.factor_versions, tuple):
                raise InvalidContractError(
                    "FactorAxisRef.factor_versions must be a tuple or None, got "
                    f"{type(self.factor_versions).__name__}"
                )
            if len(self.factor_versions) != len(self.factor_ids):
                raise InvalidContractError(
                    "FactorAxisRef.factor_versions length "
                    f"{len(self.factor_versions)} must match factor_ids length "
                    f"{len(self.factor_ids)}"
                )
            for i, ver in enumerate(self.factor_versions):
                if ver is None:
                    continue
                _check_plain_str(
                    f"FactorAxisRef.factor_versions[{i}]", ver
                )
        expected = _factor_order_hash(self.factor_ids)
        if not self.factor_order_hash:
            object.__setattr__(self, "factor_order_hash", expected)
        elif self.factor_order_hash != expected:
            raise InvalidContractError(
                "FactorAxisRef.factor_order_hash does not match the canonical "
                "digest of factor_ids; got "
                f"{self.factor_order_hash!r}, expected {expected!r}"
            )

    @property
    def num_factors(self) -> int:
        return len(self.factor_ids)

    @property
    def order_hash(self) -> str:
        return self.factor_order_hash

    def to_dict(self) -> dict:
        return {
            "kind": "factor",
            "factor_ids": list(self.factor_ids),
            "factor_versions": (
                None if self.factor_versions is None else list(self.factor_versions)
            ),
            "factor_order_hash": self.factor_order_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FactorAxisRef":
        return cls(
            factor_ids=tuple(data["factor_ids"]),
            factor_versions=(
                None
                if data.get("factor_versions") is None
                else tuple(data["factor_versions"])
            ),
            factor_order_hash=data.get("factor_order_hash", ""),
        )


@dataclass(frozen=True, init=True)
class TimeAxisRef:
    """Identity of the T (time) axis of a series artifact payload.

    Attributes:
        time_index: Ordered tuple of time coordinates (one per row).  May be
            empty when only the length matters; when provided, consumers
            validate it against the payload's T length.
        time_zone: Optional IANA time-zone name (e.g. ``"Asia/Shanghai"``).
    """

    time_index: Optional[Tuple[Any, ...]] = None
    time_zone: Optional[str] = None

    def __post_init__(self) -> None:
        if self.time_index is not None and not isinstance(self.time_index, tuple):
            raise InvalidContractError(
                "TimeAxisRef.time_index must be a tuple or None, got "
                f"{type(self.time_index).__name__}"
            )
        if self.time_zone is not None:
            _check_plain_str("TimeAxisRef.time_zone", self.time_zone)

    @property
    def num_times(self) -> int:
        return 0 if self.time_index is None else len(self.time_index)

    def to_dict(self) -> dict:
        from quant_evaluator.contracts._ndarray_codec import encode_value
        return {
            "kind": "time",
            "time_index": None if self.time_index is None else encode_value(self.time_index),
            "time_zone": self.time_zone,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TimeAxisRef":
        from quant_evaluator.contracts._ndarray_codec import decode_value
        time_index = decode_value(data.get("time_index"))
        return cls(
            time_index=None if time_index is None else tuple(time_index),
            time_zone=data.get("time_zone"),
        )


@dataclass(frozen=True, init=True)
class QuantileAxisRef:
    """Identity of the K (quantile bucket) axis of a vector/matrix payload.

    Attributes:
        quantile_labels: Ordered tuple of human-readable quantile labels
            (e.g. ``("Q1", "Q2", "Q3", "Q4", "Q5")``).  Unique and non-empty.
    """

    quantile_labels: Tuple[str, ...]

    def __post_init__(self) -> None:
        _check_str_tuple("QuantileAxisRef.quantile_labels", self.quantile_labels)
        if len(set(self.quantile_labels)) != len(self.quantile_labels):
            raise InvalidContractError(
                "QuantileAxisRef.quantile_labels must be unique; got duplicates: "
                f"{[i for i in self.quantile_labels if self.quantile_labels.count(i) > 1]}"
            )

    @property
    def num_quantiles(self) -> int:
        return len(self.quantile_labels)

    def to_dict(self) -> dict:
        return {"kind": "quantile", "quantile_labels": list(self.quantile_labels)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QuantileAxisRef":
        return cls(quantile_labels=tuple(data["quantile_labels"]))


@dataclass(frozen=True, init=True)
class MatrixAxisRefs:
    """Identity of both K axes of a (K, K, F) matrix artifact payload.

    Attributes:
        row: Axis ref of the matrix row dimension (must be QuantileAxisRef).
        col: Axis ref of the matrix column dimension (must be QuantileAxisRef).
    """

    row: QuantileAxisRef
    col: QuantileAxisRef

    def __post_init__(self) -> None:
        if not isinstance(self.row, QuantileAxisRef):
            raise InvalidContractError(
                "MatrixAxisRefs.row must be a QuantileAxisRef, got "
                f"{type(self.row).__name__}"
            )
        if not isinstance(self.col, QuantileAxisRef):
            raise InvalidContractError(
                "MatrixAxisRefs.col must be a QuantileAxisRef, got "
                f"{type(self.col).__name__}"
            )
        if self.row.num_quantiles != self.col.num_quantiles:
            raise InvalidContractError(
                "MatrixAxisRefs.row and col must have the same length; got "
                f"{self.row.num_quantiles} vs {self.col.num_quantiles}"
            )

    @property
    def num_quantiles(self) -> int:
        return self.row.num_quantiles

    def to_dict(self) -> dict:
        return {"kind": "matrix", "row": self.row.to_dict(), "col": self.col.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MatrixAxisRefs":
        return cls(
            row=QuantileAxisRef.from_dict(data["row"]),
            col=QuantileAxisRef.from_dict(data["col"]),
        )


#: Registered axis-ref kinds for ``axis_ref_from_dict`` (``None`` means a
#: bare dict — used by ``to_dict`` payloads).
_AXIS_REF_KINDS = {
    "factor": FactorAxisRef,
    "time": TimeAxisRef,
    "quantile": QuantileAxisRef,
    "matrix": MatrixAxisRefs,
}


def axis_ref_from_dict(data: Mapping[str, Any]) -> Any:
    """Deserialize an axis ref from a ``to_dict`` payload.

    The payload carries its own ``kind`` tag (produced by the axis refs'
    ``to_dict``).  ``None`` and missing values are returned unchanged.
    """
    if data is None:
        return None
    if not isinstance(data, Mapping):
        raise InvalidContractError(
            f"axis ref payload must be a Mapping or None, got {type(data).__name__}"
        )
    kind = data.get("kind")
    if kind is None:
        raise InvalidContractError("axis ref payload is missing its 'kind' tag")
    cls = _AXIS_REF_KINDS.get(kind)
    if cls is None:
        raise InvalidContractError(f"Unknown axis ref kind {kind!r}")
    return cls.from_dict(data)
