"""LifecycleState + HealthState enums.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.5 (spec §9).
PURE stdlib enums. Invariant: ``LifecycleState != JobStatus != HealthState`` —
three independent enums, never conflated.
"""

from __future__ import annotations

import enum

__all__ = ["LifecycleState", "HealthState"]


class LifecycleState(enum.Enum):
    """Factor lifecycle state machine (spec §9)."""

    DISCOVERED = "DISCOVERED"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    COMPILED = "COMPILED"
    MATERIALIZING = "MATERIALIZING"
    MATERIALIZED = "MATERIALIZED"
    RAW_EVALUATING = "RAW_EVALUATING"
    RAW_EVALUATED = "RAW_EVALUATED"
    TREATMENT_SEARCHING = "TREATMENT_SEARCHING"
    TREATMENT_SELECTED = "TREATMENT_SELECTED"
    TREATED_EVALUATING = "TREATED_EVALUATING"
    TREATED_EVALUATED = "TREATED_EVALUATED"
    NOVELTY_CHECKING = "NOVELTY_CHECKING"
    ADMISSION_REVIEW = "ADMISSION_REVIEW"
    CLUSTER_PENDING = "CLUSTER_PENDING"
    CLUSTERED = "CLUSTERED"
    LIBRARY_CANDIDATE = "LIBRARY_CANDIDATE"
    SHADOW = "SHADOW"
    PRODUCTION_ELIGIBLE = "PRODUCTION_ELIGIBLE"
    PRODUCTION = "PRODUCTION"
    DEGRADED = "DEGRADED"
    RETIRED = "RETIRED"
    QUARANTINED = "QUARANTINED"


class HealthState(enum.Enum):
    """Health state (spec §9). Distinct from LifecycleState and JobStatus."""

    UNKNOWN = "UNKNOWN"
    HEALTHY = "HEALTHY"
    WATCH = "WATCH"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    STALE = "STALE"
