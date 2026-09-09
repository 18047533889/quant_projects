"""Reference contracts for evaluation requests (DLIB-QE-003).

An :class:`EvaluationRequest` must be serializable as a strict round-trip
without carrying large raw arrays (factor value panels / label panels).  The
production-preferred shape is that a request *references* the durable
artifacts it evaluates against rather than embedding their payloads:

- :class:`FactorValueRef` — a reference to a factor-value artifact (the
  factor panel / batch) by stable identity.
- :class:`LabelBundleRef` — a reference to a label bundle (the forward-return
  definition) by stable identity.

Both are frozen, hashable, JSON-serializable reference values.  They carry
only identity + provenance, never the underlying arrays, so
``EvaluationRequest.to_dict()`` / ``from_dict()`` can round-trip losslessly
while the raw ``batch_or_factor_ids`` / ``label_bundle`` payloads remain
runtime-only (deliberately not serialized — they are large arrays).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional
from quant_evaluator.contracts.metric_artifacts import FrozenMapping

__all__ = ["FactorValueRef", "LabelBundleRef"]


@dataclass(frozen=True)
class FactorValueRef:
    """Reference to a factor-value artifact (the factor panel / batch).

    Attributes:
        factor_value_id: Stable identity of the factor-value artifact.
        factor_ids:      Factor ids the artifact covers (provenance only).
        source_ref:      Optional upstream source / snapshot reference.
        metadata:        Optional free-form provenance (deep-frozen).
    """

    factor_value_id: str
    factor_ids: tuple = ()
    source_ref: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.factor_value_id, str) or not self.factor_value_id.strip():
            raise ValueError(
                "FactorValueRef.factor_value_id must be a non-empty string, got "
                f"{self.factor_value_id!r}"
            )
        object.__setattr__(self, "factor_ids", tuple(self.factor_ids))
        object.__setattr__(self, "metadata", FrozenMapping(self.metadata or {}))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain, JSON-friendly dict."""
        payload: Dict[str, Any] = {
            "factor_value_id": self.factor_value_id,
            "factor_ids": list(self.factor_ids),
        }
        if self.source_ref is not None:
            payload["source_ref"] = self.source_ref
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FactorValueRef":
        """Rebuild from a :meth:`to_dict` payload."""
        if not isinstance(data, Mapping):
            raise TypeError("FactorValueRef.from_dict requires a mapping")
        return cls(
            factor_value_id=data["factor_value_id"],
            factor_ids=tuple(data.get("factor_ids", ())),
            source_ref=data.get("source_ref"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class LabelBundleRef:
    """Reference to a label bundle (the forward-return definition).

    Attributes:
        label_bundle_id: Stable identity of the label bundle.
        target_id:       The label target id (provenance only).
        horizon:         Forward-return horizon (provenance only).
        source_ref:      Optional upstream source / snapshot reference.
        metadata:        Optional free-form provenance (deep-frozen).
    """

    label_bundle_id: str
    target_id: Optional[str] = None
    horizon: Optional[int] = None
    source_ref: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.label_bundle_id, str) or not self.label_bundle_id.strip():
            raise ValueError(
                "LabelBundleRef.label_bundle_id must be a non-empty string, got "
                f"{self.label_bundle_id!r}"
            )
        if self.horizon is not None and (
            not isinstance(self.horizon, int) or self.horizon <= 0
        ):
            raise ValueError(
                f"LabelBundleRef.horizon must be a positive int or None, got "
                f"{self.horizon!r}"
            )
        object.__setattr__(self, "metadata", FrozenMapping(self.metadata or {}))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain, JSON-friendly dict."""
        payload: Dict[str, Any] = {"label_bundle_id": self.label_bundle_id}
        if self.target_id is not None:
            payload["target_id"] = self.target_id
        if self.horizon is not None:
            payload["horizon"] = self.horizon
        if self.source_ref is not None:
            payload["source_ref"] = self.source_ref
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LabelBundleRef":
        """Rebuild from a :meth:`to_dict` payload."""
        if not isinstance(data, Mapping):
            raise TypeError("LabelBundleRef.from_dict requires a mapping")
        return cls(
            label_bundle_id=data["label_bundle_id"],
            target_id=data.get("target_id"),
            horizon=data.get("horizon"),
            source_ref=data.get("source_ref"),
            metadata=data.get("metadata", {}),
        )
