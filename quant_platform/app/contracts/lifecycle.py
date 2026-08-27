"""LifecycleState + QRPPipelineStage + HealthState — three distinct enums.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.5 (spec §9).
PURE stdlib enums.

LIFECYCLE SEPARATION (§5.7):

- ``LifecycleState`` — the factor **ASSET** governance state machine (FA owns):
  REGISTERED → EVALUATED → APPROVED → PRODUCTION_READY, etc. It is deliberately
  NOT expanded with pipeline-progress states like MATERIALIZING / RAW_EVALUATING /
  TREATMENT_SEARCHING — those belong to the pipeline/job domain.
- ``QRPPipelineStage`` — the end-to-end Factor pipeline progress, distinct from
  the asset state machine and from ``JobStatus``. Owns MATERIALIZING,
  RAW_EVALUATING, TREATMENT_SEARCHING, etc.
- ``HealthState`` — operational health of a live factor asset.

Invariant (spec §9, §5.7): ``QRPPipelineStage != LifecycleState != JobStatus !=
HealthState`` — four independent enums, never conflated.
"""

from __future__ import annotations

import enum

__all__ = ["LifecycleState", "HealthState", "QRPPipelineStage"]


class LifecycleState(enum.Enum):
    """Factor ASSET governance state machine (FA owns, spec §9).

    Deliberately does NOT contain MATERIALIZING / RAW_EVALUATING /
    TREATMENT_SEARCHING — those are pipeline-progress states, not asset states.
    """

    REGISTERED = "REGISTERED"
    EVALUATED = "EVALUATED"
    APPROVED = "APPROVED"
    PRODUCTION_READY = "PRODUCTION_READY"
    PRODUCTION = "PRODUCTION"
    DEGRADED = "DEGRADED"
    RETIRED = "RETIRED"
    QUARANTINED = "QUARANTINED"


class QRPPipelineStage(enum.Enum):
    """End-to-end Factor pipeline progress (§5.7).

    Belongs to the pipeline/job domain, distinct from the asset ``LifecycleState``
    and from ``JobStatus``.
    """

    DISCOVERED = "DISCOVERED"
    VALIDATING = "VALIDATING"
    COMPILING = "COMPILING"
    MATERIALIZING = "MATERIALIZING"
    RAW_EVALUATING = "RAW_EVALUATING"
    TREATMENT_SEARCHING = "TREATMENT_SEARCHING"
    TREATED_EVALUATING = "TREATED_EVALUATING"
    ADMISSION = "ADMISSION"
    CLUSTERING = "CLUSTERING"
    LIBRARY = "LIBRARY"
    PRODUCTION = "PRODUCTION"


class HealthState(enum.Enum):
    """Health state (spec §9). Distinct from LifecycleState, JobStatus,
    QRPPipelineStage."""

    UNKNOWN = "UNKNOWN"
    HEALTHY = "HEALTHY"
    WATCH = "WATCH"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    STALE = "STALE"
