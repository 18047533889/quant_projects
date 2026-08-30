/**
 * Enums from quant_platform/app/contracts — THIN mirror, no logic.
 *
 * Source of truth: `quant_platform/docs/PLATFORM_CONTRACTS_DRAFT.md` (§4.5, §4.3,
 * §4.14, §4.10) and `quant_platform/app/contracts/*.py`. Field names and string
 * values are copied verbatim so the frontend renders exactly what the API returns.
 */

/** LifecycleState (contracts/lifecycle.py) — factor ASSET governance state. */
export const LIFECYCLE_STATES = [
  'REGISTERED',
  'EVALUATED',
  'APPROVED',
  'PRODUCTION_READY',
  'PRODUCTION',
  'DEGRADED',
  'RETIRED',
  'QUARANTINED',
] as const;
export type LifecycleState = (typeof LIFECYCLE_STATES)[number];

/** QRPPipelineStage (contracts/lifecycle.py) — end-to-end pipeline progress. */
export const QRP_PIPELINE_STAGES = [
  'DISCOVERED',
  'VALIDATING',
  'COMPILING',
  'MATERIALIZING',
  'RAW_EVALUATING',
  'TREATMENT_SEARCHING',
  'TREATED_EVALUATING',
  'ADMISSION',
  'CLUSTERING',
  'LIBRARY',
  'PRODUCTION',
] as const;
export type QRPPipelineStage = (typeof QRP_PIPELINE_STAGES)[number];

/** HealthState (contracts/lifecycle.py) — operational health. */
export const HEALTH_STATES = [
  'UNKNOWN',
  'HEALTHY',
  'WATCH',
  'DEGRADED',
  'CRITICAL',
  'STALE',
] as const;
export type HealthState = (typeof HEALTH_STATES)[number];

/** JobStatus (contracts/jobs.py). */
export const JOB_STATUSES = [
  'PENDING',
  'RUNNING',
  'SUCCEEDED',
  'FAILED_RETRYABLE',
  'FAILED_TERMINAL',
  'CANCEL_REQUESTED',
  'CANCELLED',
  'BLOCKED_DATA',
  'BLOCKED_DEPENDENCY',
  'TIMED_OUT',
] as const;
export type JobStatus = (typeof JOB_STATUSES)[number];

/** EvidenceStatus (contracts/timing.py) — NEVER zero-filled on the frontend. */
export const EVIDENCE_STATUSES = [
  'NOT_COMPUTED',
  'UNAVAILABLE',
  'INVALID_EVIDENCE',
  'LABEL_NOT_MATURE',
] as const;
export type EvidenceStatus = (typeof EVIDENCE_STATUSES)[number];

/** ClusterConversionType (contracts/cluster_library.py). */
export const CLUSTER_CONVERSION_TYPES = [
  'UNCHANGED',
  'MIGRATED',
  'SPLIT',
  'MERGED',
  'NEW',
  'DISSOLVED',
] as const;
export type ClusterConversionType = (typeof CLUSTER_CONVERSION_TYPES)[number];

/** LibraryStatus (contracts/factor_library.py). */
export const LIBRARY_STATUSES = ['CANDIDATE', 'SHADOW', 'APPROVED', 'PRODUCTION'] as const;
export type LibraryStatus = (typeof LIBRARY_STATUSES)[number];

/** SecurityClassification (contracts/security.py). */
export const SECURITY_CLASSIFICATIONS = [
  'PUBLIC_METADATA',
  'INTERNAL_RESEARCH',
  'CONFIDENTIAL_ALPHA',
  'RESTRICTED_RAW_VALUES',
  'PRODUCTION_ONLY',
] as const;
export type SecurityClassification = (typeof SECURITY_CLASSIFICATIONS)[number];

/** Role (contracts/security.py). */
export const ROLES = ['MEMBER', 'LEAD', 'CORE', 'ADMIN', 'SERVICE'] as const;
export type Role = (typeof ROLES)[number];

/** Team (contracts/security.py). */
export const TEAMS = [
  'FACTOR_TEAM',
  'MODEL_TEAM',
  'PRODUCTION_TEAM',
  'EXECUTIVE',
  'PLATFORM_ADMIN',
] as const;
export type Team = (typeof TEAMS)[number];

/** ARTIFACT_TYPE_* constants (contracts/artifact_ref.py, spec §7.2). */
export const ARTIFACT_TYPES = [
  'FACTOR_CANDIDATE',
  'FACTOR_DEFINITION',
  'FACTOR_VALUE',
  'EVALUATION_BUNDLE',
  'TREATMENT_SELECTION',
  'SIMILARITY_FINGERPRINT',
  'SIMILARITY_GRAPH',
  'CLUSTER_VERSION',
  'CLUSTER_ASSIGNMENT',
  'FACTOR_LIBRARY_VERSION',
  'FEATURE_SET',
  'MODEL_DATASET',
  'MODEL',
  'BACKTEST',
  'REPORT_EXPORT',
] as const;
export type ArtifactType = (typeof ARTIFACT_TYPES)[number];

/** EVENT_TYPE_* constants (contracts/event_envelope.py, spec §11). */
export const EVENT_TYPES = [
  'FactorCandidateDiscovered',
  'FactorCandidateValidated',
  'FactorDefinitionRegistered',
  'FactorMaterializationRequested',
  'FactorMaterialized',
  'RawEvaluationCompleted',
  'TreatmentSearchCompleted',
  'TreatedEvaluationCompleted',
  'FactorAdmissionDecided',
  'FactorClusterAssigned',
  'ClusterVersionPublished',
  'FactorLibraryCandidateCreated',
  'FactorLibraryPromoted',
  'FeatureSetCreated',
  'FeatureSetSemanticChanged',
  'ModelRetrainRequired',
  'FactorHealthChanged',
  'ArtifactPublished',
  'JobFailed',
] as const;
export type EventType = (typeof EVENT_TYPES)[number];

/** FeatureSetDiffCategory (contracts/feature_set.py, spec §18). */
export const FEATURE_SET_DIFF_CATEGORIES = [
  'METADATA_ONLY',
  'EVIDENCE_ONLY',
  'FEATURE_MEMBERSHIP_CHANGE',
  'FEATURE_TRANSFORM_CHANGE',
  'FEATURE_ORIENTATION_CHANGE',
  'FEATURE_SCHEMA_CHANGE',
  'LABEL_CHANGE',
  'DATA_REVISION',
] as const;
export type FeatureSetDiffCategory = (typeof FEATURE_SET_DIFF_CATEGORIES)[number];
