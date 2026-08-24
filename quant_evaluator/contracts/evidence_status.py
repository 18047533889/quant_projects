"""Evidence status contract for computed metric evidence.

A :class:`MetricEvidence` bundles the canonical :class:`MetricArtifact`
(``quant_evaluator.contracts.metric_artifacts``) with a machine-readable
:class:`EvidenceStatus` and the reason the status was set, so reporting and
downstream consumers can distinguish "computed" from "not computed" /
"failed" / "unsupported" / "insufficient data" — instead of silently
fabricating a 0.0.

Design notes:
    - ``EvidenceStatus`` is a plain :class:`enum.Enum` (single source of
      truth).  ``reason_code`` is a *string* enum of known reason codes, and
      a free-form ``reason_detail`` may carry the full message.
    - ``MetricEvidence`` is a frozen, hashable dataclass so it can ride
      alongside artifacts, live in dicts/sets, and be serialized losslessly
      via ``to_dict`` / ``from_dict``.
    - The ``artifact`` field is the canonical contract artifact (or ``None``
      when nothing was computed).  A non-COMPUTED status with a non-None
      artifact is allowed (e.g. a degraded partial result); consumers decide
      how to render based on ``status`` first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional

from quant_evaluator.contracts.metric_artifacts import MetricArtifact


class EvidenceStatus(Enum):
    """Whether evidence for a metric exists and can be trusted."""

    COMPUTED = "computed"
    NOT_COMPUTED = "not_computed"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT_DATA = "insufficient_data"
    FAILED = "failed"

    @classmethod
    def from_value(cls, value: object) -> "EvidenceStatus":
        """Resolve a str/enum value to an EvidenceStatus (fail closed)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError:
                raise ValueError(
                    f"Unknown EvidenceStatus value {value!r}; expected one of "
                    f"{[m.value for m in cls]}"
                )
        raise TypeError(
            f"EvidenceStatus.from_value expects str or EvidenceStatus, got "
            f"{type(value).__name__}"
        )


class EvidenceReasonCode(str, Enum):
    """Known machine-readable reason codes for an EvidenceStatus."""

    OK = "ok"
    SOURCE_ARTIFACT_MISSING = "source_artifact_missing"
    NOT_YET_COMPUTED = "not_yet_computed"
    UNSUPPORTED_METRIC = "unsupported_metric"
    UNSUPPORTED_AXIS = "unsupported_axis"
    UNSUPPORTED_OUTPUT_TYPE = "unsupported_output_type"
    MIN_PERIODS_NOT_MET = "min_periods_not_met"
    OBSERVATIONS_TOO_FEW = "observations_too_few"
    EMPTY_INPUT = "empty_input"
    KERNEL_FAILURE = "kernel_failure"
    NUMERICAL_FAILURE = "numerical_failure"
    SERIALIZATION_FAILURE = "serialization_failure"
    NOT_IMPLEMENTED = "not_implemented"

    @classmethod
    def from_value(cls, value: object) -> "EvidenceReasonCode":
        """Resolve a str/enum value to an EvidenceReasonCode (fail closed)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value)
            except ValueError:
                raise ValueError(
                    f"Unknown EvidenceReasonCode value {value!r}; expected one of "
                    f"{[m.value for m in cls]}"
                )
        raise TypeError(
            f"EvidenceReasonCode.from_value expects str or EvidenceReasonCode, got "
            f"{type(value).__name__}"
        )


@dataclass(frozen=True)
class MetricEvidence:
    """Evidence bundle for one metric: status + optional canonical artifact.

    Attributes:
        status: :class:`EvidenceStatus` — COMPUTED only when real evidence
            exists.  Anything else must render as a placeholder, never 0.0.
        reason_code: Machine-readable :class:`EvidenceReasonCode` (str-typed).
        observations: Number of observations that backed the computation
            (may be 0 when nothing was computed).
        minimum_required: Minimum observations required for COMPUTED status
            (None when not applicable).
        artifact: The canonical :class:`MetricArtifact` (or None).
        reason_detail: Free-form human-readable detail string.
    """

    status: EvidenceStatus
    reason_code: EvidenceReasonCode = EvidenceReasonCode.OK
    observations: int = 0
    minimum_required: Optional[int] = None
    artifact: Optional[MetricArtifact] = None
    reason_detail: str = ""

    def __post_init__(self) -> None:
        # Normalize enums (accept plain strings too) -- fail closed.
        object.__setattr__(self, "status", EvidenceStatus.from_value(self.status))
        object.__setattr__(
            self, "reason_code", EvidenceReasonCode.from_value(self.reason_code)
        )
        if not isinstance(self.observations, int) or self.observations < 0:
            raise ValueError(
                f"MetricEvidence.observations must be a non-negative int, got "
                f"{self.observations!r}"
            )
        if self.minimum_required is not None and (
            not isinstance(self.minimum_required, int) or self.minimum_required < 0
        ):
            raise ValueError(
                f"MetricEvidence.minimum_required must be a non-negative int or "
                f"None, got {self.minimum_required!r}"
            )
        if self.artifact is not None and not isinstance(self.artifact, MetricArtifact):
            raise TypeError(
                f"MetricEvidence.artifact must be a MetricArtifact or None, got "
                f"{type(self.artifact).__name__}"
            )

    @property
    def computed(self) -> bool:
        """True only when the status is exactly COMPUTED."""
        return self.status is EvidenceStatus.COMPUTED

    @property
    def artifact_kind(self) -> str:
        """Return the artifact kind of the wrapped artifact ('' when absent)."""
        return "" if self.artifact is None else self.artifact.artifact_kind

    @property
    def metric_id(self) -> str:
        """Return the artifact's metric_id ('' when no artifact is attached)."""
        return "" if self.artifact is None else self.artifact.metric_id

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly plain dict."""
        return {
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "observations": self.observations,
            "minimum_required": self.minimum_required,
            "artifact": None if self.artifact is None else self.artifact.to_dict(),
            "reason_detail": self.reason_detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricEvidence":
        """Deserialize from a ``to_dict`` payload."""
        artifact = data.get("artifact")
        return cls(
            status=EvidenceStatus.from_value(data.get("status", "not_computed")),
            reason_code=EvidenceReasonCode.from_value(
                data.get("reason_code", "not_yet_computed")
            ),
            observations=data.get("observations", 0),
            minimum_required=data.get("minimum_required"),
            artifact=(
                None if artifact is None else MetricArtifact.from_dict(artifact)
            ),
            reason_detail=data.get("reason_detail", ""),
        )


__all__ = [
    "EvidenceStatus",
    "EvidenceReasonCode",
    "MetricEvidence",
    "evidence_for_computed",
]


def evidence_for_computed(
    artifact: MetricArtifact,
    observations: int = 0,
    minimum_required: Optional[int] = None,
    reason_detail: str = "computed",
) -> MetricEvidence:
    """Build a COMPUTED evidence bundle around a canonical artifact."""
    return MetricEvidence(
        status=EvidenceStatus.COMPUTED,
        reason_code=EvidenceReasonCode.OK,
        observations=observations,
        minimum_required=minimum_required,
        artifact=artifact,
        reason_detail=reason_detail,
    )


def evidence_for_not_computed(
    reason_code: str = "not_yet_computed",
    *,
    artifact: Optional[MetricArtifact] = None,
    observations: int = 0,
    minimum_required: Optional[int] = None,
    reason_detail: str = "",
) -> MetricEvidence:
    """Build a NOT_COMPUTED evidence bundle (the default for absent data)."""
    return MetricEvidence(
        status=EvidenceStatus.NOT_COMPUTED,
        reason_code=EvidenceReasonCode.from_value(reason_code),
        observations=observations,
        minimum_required=minimum_required,
        artifact=artifact,
        reason_detail=reason_detail,
    )
