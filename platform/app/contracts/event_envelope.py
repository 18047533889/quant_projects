"""EventEnvelope — unified platform event envelope DTO.

DRAFT. Implements ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` §4.2 (spec §11).
PURE stdlib frozen dataclass. Events are emitted after DB commit via the
Transactional Outbox (spec §11.1); the platform never double-writes events and
state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

__all__ = [
    "EventEnvelope",
    "EVENT_TYPE_FACTOR_CANDIDATE_DISCOVERED",
    "EVENT_TYPE_FACTOR_CANDIDATE_VALIDATED",
    "EVENT_TYPE_FACTOR_DEFINITION_REGISTERED",
    "EVENT_TYPE_FACTOR_MATERIALIZATION_REQUESTED",
    "EVENT_TYPE_FACTOR_MATERIALIZED",
    "EVENT_TYPE_RAW_EVALUATION_COMPLETED",
    "EVENT_TYPE_TREATMENT_SEARCH_COMPLETED",
    "EVENT_TYPE_TREATED_EVALUATION_COMPLETED",
    "EVENT_TYPE_FACTOR_ADMISSION_DECIDED",
    "EVENT_TYPE_FACTOR_CLUSTER_ASSIGNED",
    "EVENT_TYPE_CLUSTER_VERSION_PUBLISHED",
    "EVENT_TYPE_FACTOR_LIBRARY_CANDIDATE_CREATED",
    "EVENT_TYPE_FACTOR_LIBRARY_PROMOTED",
    "EVENT_TYPE_FEATURE_SET_CREATED",
    "EVENT_TYPE_FEATURE_SET_SEMANTIC_CHANGED",
    "EVENT_TYPE_MODEL_RETRAIN_REQUIRED",
    "EVENT_TYPE_FACTOR_HEALTH_CHANGED",
    "EVENT_TYPE_ARTIFACT_PUBLISHED",
    "EVENT_TYPE_JOB_FAILED",
    "EVENT_TYPES",
]

# spec §11 core event types
EVENT_TYPE_FACTOR_CANDIDATE_DISCOVERED = "FactorCandidateDiscovered"
EVENT_TYPE_FACTOR_CANDIDATE_VALIDATED = "FactorCandidateValidated"
EVENT_TYPE_FACTOR_DEFINITION_REGISTERED = "FactorDefinitionRegistered"
EVENT_TYPE_FACTOR_MATERIALIZATION_REQUESTED = "FactorMaterializationRequested"
EVENT_TYPE_FACTOR_MATERIALIZED = "FactorMaterialized"
EVENT_TYPE_RAW_EVALUATION_COMPLETED = "RawEvaluationCompleted"
EVENT_TYPE_TREATMENT_SEARCH_COMPLETED = "TreatmentSearchCompleted"
EVENT_TYPE_TREATED_EVALUATION_COMPLETED = "TreatedEvaluationCompleted"
EVENT_TYPE_FACTOR_ADMISSION_DECIDED = "FactorAdmissionDecided"
EVENT_TYPE_FACTOR_CLUSTER_ASSIGNED = "FactorClusterAssigned"
EVENT_TYPE_CLUSTER_VERSION_PUBLISHED = "ClusterVersionPublished"
EVENT_TYPE_FACTOR_LIBRARY_CANDIDATE_CREATED = "FactorLibraryCandidateCreated"
EVENT_TYPE_FACTOR_LIBRARY_PROMOTED = "FactorLibraryPromoted"
EVENT_TYPE_FEATURE_SET_CREATED = "FeatureSetCreated"
EVENT_TYPE_FEATURE_SET_SEMANTIC_CHANGED = "FeatureSetSemanticChanged"
EVENT_TYPE_MODEL_RETRAIN_REQUIRED = "ModelRetrainRequired"
EVENT_TYPE_FACTOR_HEALTH_CHANGED = "FactorHealthChanged"
EVENT_TYPE_ARTIFACT_PUBLISHED = "ArtifactPublished"
EVENT_TYPE_JOB_FAILED = "JobFailed"

EVENT_TYPES: frozenset[str] = frozenset(
    {
        EVENT_TYPE_FACTOR_CANDIDATE_DISCOVERED,
        EVENT_TYPE_FACTOR_CANDIDATE_VALIDATED,
        EVENT_TYPE_FACTOR_DEFINITION_REGISTERED,
        EVENT_TYPE_FACTOR_MATERIALIZATION_REQUESTED,
        EVENT_TYPE_FACTOR_MATERIALIZED,
        EVENT_TYPE_RAW_EVALUATION_COMPLETED,
        EVENT_TYPE_TREATMENT_SEARCH_COMPLETED,
        EVENT_TYPE_TREATED_EVALUATION_COMPLETED,
        EVENT_TYPE_FACTOR_ADMISSION_DECIDED,
        EVENT_TYPE_FACTOR_CLUSTER_ASSIGNED,
        EVENT_TYPE_CLUSTER_VERSION_PUBLISHED,
        EVENT_TYPE_FACTOR_LIBRARY_CANDIDATE_CREATED,
        EVENT_TYPE_FACTOR_LIBRARY_PROMOTED,
        EVENT_TYPE_FEATURE_SET_CREATED,
        EVENT_TYPE_FEATURE_SET_SEMANTIC_CHANGED,
        EVENT_TYPE_MODEL_RETRAIN_REQUIRED,
        EVENT_TYPE_FACTOR_HEALTH_CHANGED,
        EVENT_TYPE_ARTIFACT_PUBLISHED,
        EVENT_TYPE_JOB_FAILED,
    }
)


@dataclass(frozen=True)
class EventEnvelope:
    """Unified event envelope (spec §11)."""

    event_id: str
    event_type: str
    schema_version: str
    occurred_at: datetime
    producer: str
    aggregate_type: str
    aggregate_id: str
    correlation_id: str
    causation_id: str | None = None
    payload: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id is required")
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event_type: {self.event_type!r}")
        if not self.schema_version:
            raise ValueError("schema_version is required")
        if not self.producer:
            raise ValueError("producer is required")
        if not self.aggregate_type:
            raise ValueError("aggregate_type is required")
        if not self.aggregate_id:
            raise ValueError("aggregate_id is required")
        if not self.correlation_id:
            raise ValueError("correlation_id is required")
        if self.payload is None:
            object.__setattr__(self, "payload", {})
