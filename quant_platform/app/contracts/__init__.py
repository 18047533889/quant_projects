"""Platform contracts DTO layer.

DRAFT. PURE-DTO rule: stdlib dataclasses only (frozen) + ``typing.Protocol``.
NO fastapi / sqlalchemy / pydantic / third-party imports. This layer is the
platform-facing ref / adapter boundary (spec §6); it does not replace
domain-native artifacts, and no domain package imports it.

See ``platform/docs/PLATFORM_CONTRACTS_DRAFT.md`` for the full field tables.
"""

from __future__ import annotations

from ._contenthash import canonical_str, content_hash
from .artifact_ref import (
    ARTIFACT_TYPES,
    ARTIFACT_TYPE_BACKTEST,
    ARTIFACT_TYPE_CLUSTER_ASSIGNMENT,
    ARTIFACT_TYPE_CLUSTER_VERSION,
    ARTIFACT_TYPE_EVALUATION_BUNDLE,
    ARTIFACT_TYPE_FACTOR_CANDIDATE,
    ARTIFACT_TYPE_FACTOR_DEFINITION,
    ARTIFACT_TYPE_FACTOR_LIBRARY_VERSION,
    ARTIFACT_TYPE_FACTOR_VALUE,
    ARTIFACT_TYPE_FEATURE_SET,
    ARTIFACT_TYPE_MODEL,
    ARTIFACT_TYPE_MODEL_DATASET,
    ARTIFACT_TYPE_REPORT_EXPORT,
    ARTIFACT_TYPE_SIMILARITY_FINGERPRINT,
    ARTIFACT_TYPE_SIMILARITY_GRAPH,
    ARTIFACT_TYPE_TREATMENT_SELECTION,
    ArtifactRef,
)
from .backtest import (
    BacktestArtifactRef,
    BacktestProvider,
    BacktestRequest,
    ModelArtifactRef,
    ModelDatasetRequest,
    ModelTrainingRequest,
)
from .candidate import FactorCandidateManifest, READY_MARKER_NAME, is_ready_marker
from .cluster_library import (
    ClusterConversionType,
    ClusterLineageEdge,
    ClusterMembership,
    ClusterSetVersion,
    ClusterVersion,
    FactorLibraryVersion,
    LibraryMembership,
    LogicalCluster,
    SimilarityGraphVersion,
)
from .event_envelope import (
    EVENT_TYPES,
    EVENT_TYPE_ARTIFACT_PUBLISHED,
    EVENT_TYPE_CLUSTER_VERSION_PUBLISHED,
    EVENT_TYPE_FACTOR_ADMISSION_DECIDED,
    EVENT_TYPE_FACTOR_CANDIDATE_DISCOVERED,
    EVENT_TYPE_FACTOR_CANDIDATE_VALIDATED,
    EVENT_TYPE_FACTOR_CLUSTER_ASSIGNED,
    EVENT_TYPE_FACTOR_DEFINITION_REGISTERED,
    EVENT_TYPE_FACTOR_HEALTH_CHANGED,
    EVENT_TYPE_FACTOR_LIBRARY_CANDIDATE_CREATED,
    EVENT_TYPE_FACTOR_LIBRARY_PROMOTED,
    EVENT_TYPE_FACTOR_MATERIALIZATION_REQUESTED,
    EVENT_TYPE_FACTOR_MATERIALIZED,
    EVENT_TYPE_FEATURE_SET_CREATED,
    EVENT_TYPE_FEATURE_SET_SEMANTIC_CHANGED,
    EVENT_TYPE_JOB_FAILED,
    EVENT_TYPE_MODEL_RETRAIN_REQUIRED,
    EVENT_TYPE_RAW_EVALUATION_COMPLETED,
    EVENT_TYPE_TREATED_EVALUATION_COMPLETED,
    EVENT_TYPE_TREATMENT_SEARCH_COMPLETED,
    EventEnvelope,
)
from .feature_set import (
    FeatureMemberRef,
    FeatureSetArtifact,
    FeatureSetDiffCategory,
    FeatureSetVersion,
    ModelRetrainRequiredEvent,
    retrain_required_for_diff,
)
from .identities import (
    EvaluationIdentity,
    FactorDefinitionIdentity,
    FactorValueIdentity,
    Identity,
    TreatmentIdentity,
    canonicalize,
)
from .jobs import (
    ErrorClass,
    JobAttempt,
    JobRecord,
    JobResult,
    JobSpec,
    JobStatus,
    idempotency_key,
    materialization_idempotency_key,
    qe_idempotency_key,
    treatment_idempotency_key,
)
from .lifecycle import HealthState, LifecycleState, QRPPipelineStage
from .rbac import (
    PERMISSION_MIN_CLASSIFICATION,
    ROLE_DEFAULT_CLASSIFICATION,
    ROLE_PERMISSIONS,
    HumanPrincipal,
    Permission,
    ResourceScope,
    Role,
    SecurityClassification,
    Team,
    WorkloadPrincipal,
)
from .storage import (
    ArtifactPublisher,
    ArtifactResolver,
    ArtifactStoragePort,
    CacheEvictionPolicy,
    LocalArtifactCache,
    ObjectMetadata,
    ObjectStore,
)
from .timing import EvidenceStatus, TimingContract
from .workflow import (
    WorkflowBackend,
    WorkflowRun,
    WorkflowSignal,
    WorkflowSpec,
    WorkflowStatus,
)

__all__ = [
    # contenthash
    "canonical_str",
    "content_hash",
    # artifact_ref
    "ArtifactRef",
    "ARTIFACT_TYPES",
    "ARTIFACT_TYPE_FACTOR_CANDIDATE",
    "ARTIFACT_TYPE_FACTOR_DEFINITION",
    "ARTIFACT_TYPE_FACTOR_VALUE",
    "ARTIFACT_TYPE_EVALUATION_BUNDLE",
    "ARTIFACT_TYPE_TREATMENT_SELECTION",
    "ARTIFACT_TYPE_SIMILARITY_FINGERPRINT",
    "ARTIFACT_TYPE_SIMILARITY_GRAPH",
    "ARTIFACT_TYPE_CLUSTER_VERSION",
    "ARTIFACT_TYPE_CLUSTER_ASSIGNMENT",
    "ARTIFACT_TYPE_FACTOR_LIBRARY_VERSION",
    "ARTIFACT_TYPE_FEATURE_SET",
    "ARTIFACT_TYPE_MODEL_DATASET",
    "ARTIFACT_TYPE_MODEL",
    "ARTIFACT_TYPE_BACKTEST",
    "ARTIFACT_TYPE_REPORT_EXPORT",
    # event_envelope
    "EventEnvelope",
    "EVENT_TYPES",
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
    # jobs
    "JobStatus",
    "ErrorClass",
    "JobSpec",
    "JobRecord",
    "JobAttempt",
    "JobResult",
    "idempotency_key",
    "materialization_idempotency_key",
    "qe_idempotency_key",
    "treatment_idempotency_key",
    # candidate
    "FactorCandidateManifest",
    "READY_MARKER_NAME",
    "is_ready_marker",
    # lifecycle
    "LifecycleState",
    "HealthState",
    "QRPPipelineStage",
    # identities
    "Identity",
    "FactorDefinitionIdentity",
    "FactorValueIdentity",
    "EvaluationIdentity",
    "TreatmentIdentity",
    "canonicalize",
    # cluster_library
    "ClusterConversionType",
    "ClusterVersion",
    "ClusterSetVersion",
    "ClusterLineageEdge",
    "ClusterMembership",
    "LogicalCluster",
    "SimilarityGraphVersion",
    "FactorLibraryVersion",
    "LibraryMembership",
    # feature_set
    "FeatureSetArtifact",
    "FeatureSetVersion",
    "FeatureMemberRef",
    "FeatureSetDiffCategory",
    "ModelRetrainRequiredEvent",
    "retrain_required_for_diff",
    # rbac
    "Permission",
    "Role",
    "Team",
    "SecurityClassification",
    "HumanPrincipal",
    "WorkloadPrincipal",
    "ResourceScope",
    "ROLE_PERMISSIONS",
    "PERMISSION_MIN_CLASSIFICATION",
    "ROLE_DEFAULT_CLASSIFICATION",
    # workflow
    "WorkflowBackend",
    "WorkflowSignal",
    "WorkflowSpec",
    "WorkflowRun",
    "WorkflowStatus",
    # storage
    "ArtifactStoragePort",
    "ArtifactPublisher",
    "ArtifactResolver",
    "ObjectStore",
    "ObjectMetadata",
    "LocalArtifactCache",
    "CacheEvictionPolicy",
    # timing
    "TimingContract",
    "EvidenceStatus",
    # backtest
    "BacktestProvider",
    "BacktestRequest",
    "BacktestArtifactRef",
    "ModelDatasetRequest",
    "ModelTrainingRequest",
    "ModelArtifactRef",
]
