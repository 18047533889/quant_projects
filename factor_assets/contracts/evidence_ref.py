"""
Evidence reference contracts.

References to QE evaluation results without duplicating metric truth.
FA stores only references and selected summary fingerprints, not raw evidence.
"""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class EvidenceInvalidationPlan:
    """Read-only impact plan for metric semantic-version changes."""

    stale_evidence_ids: tuple[str, ...]
    affected_factor_ids: tuple[str, ...]
    unaffected_evidence_ids: tuple[str, ...]
    required_metric_versions: tuple[tuple[str, str], ...]


def plan_metric_version_invalidation(
    evidence_refs: Sequence["EvidenceRef"],
    required_metric_versions: Mapping[str, str],
) -> EvidenceInvalidationPlan:
    """Identify old metric evidence without mutating assets or repositories."""
    required = dict(required_metric_versions)
    if any(not key or not value for key, value in required.items()):
        raise ValueError("metric ids and required versions must be non-empty")
    stale = tuple(
        ref.evidence_id for ref in evidence_refs
        if ref.metric_name in required
        and ref.metric_version != required[ref.metric_name]
    )
    stale_set = set(stale)
    return EvidenceInvalidationPlan(
        stale_evidence_ids=stale,
        affected_factor_ids=tuple(sorted({
            ref.factor_id for ref in evidence_refs if ref.evidence_id in stale_set
        })),
        unaffected_evidence_ids=tuple(
            ref.evidence_id for ref in evidence_refs if ref.evidence_id not in stale_set
        ),
        required_metric_versions=tuple(sorted(required.items())),
    )


@dataclass(frozen=True)
class EvidenceRef:
    """
    Reference to a single evaluation evidence result.

    Points to QE-owned evidence without duplicating metric values.
    FA may cache bounded summary statistics only.
    """
    evidence_id: str
    evaluation_run_id: str
    metric_name: str
    metric_version: str
    timestamp: str  # ISO 8601
    factor_id: str
    universe_ref: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    # Optional bounded summary (not full metric values)
    summary_value: Optional[float] = None
    summary_context: Optional[str] = None

    def __post_init__(self):
        if not self.evidence_id:
            raise ValueError("evidence_id is required")
        if not self.evaluation_run_id:
            raise ValueError("evaluation_run_id is required")
        if not self.metric_name:
            raise ValueError("metric_name is required")
        if not self.metric_version:
            raise ValueError("metric_version is required")
        if not self.factor_id:
            raise ValueError("factor_id is required")


@dataclass(frozen=True)
class EvidenceBundleRef:
    """
    Reference to a complete QE EvaluationBundle.

    Bundle-level reference for a factor evaluation containing multiple metrics.
    """
    bundle_id: str
    evaluation_run_id: str
    factor_ids: tuple[str, ...]
    timestamp: str  # ISO 8601
    qe_version: str
    universe_ref: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    label_ref: Optional[str] = None
    config_hash: Optional[str] = None
    # Optional bounded summary
    primary_metric: Optional[str] = None
    primary_value: Optional[float] = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.bundle_id:
            raise ValueError("bundle_id is required")
        if not self.evaluation_run_id:
            raise ValueError("evaluation_run_id is required")
        if not self.factor_ids:
            raise ValueError("factor_ids is required")
        if not self.qe_version:
            raise ValueError("qe_version is required")

    @property
    def has_warnings(self) -> bool:
        """Check if bundle has warnings."""
        return len(self.warnings) > 0


def evidence_bundle_event_id(ref: EvidenceBundleRef) -> str:
    """Return the validated, namespaced event identifier for a bundle reference.

    Lifecycle policy keys such as ``evaluation_bundle_ref`` describe a required
    evidence kind; they are not bundle identities.  Only a typed bundle reference
    can be adapted to the identifier persisted in a lifecycle event.
    """
    if not isinstance(ref, EvidenceBundleRef):
        raise TypeError("ref must be an EvidenceBundleRef")
    return f"qe-bundle:{ref.bundle_id}"
