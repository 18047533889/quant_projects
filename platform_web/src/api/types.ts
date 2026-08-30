import type {
  ArtifactType,
  ClusterConversionType,
  EvidenceStatus,
  HealthState,
  JobStatus,
  LifecycleState,
  LibraryStatus,
  QRPPipelineStage,
  SecurityClassification,
} from './enums';

/**
 * TypeScript interfaces mirroring `quant_platform/app/contracts/*` DTOs.
 *
 * Source: `quant_platform/docs/PLATFORM_CONTRACTS_DRAFT.md` (§4.1–§4.14) and the
 * contract modules in `quant_platform/app/contracts/`. These are THIN mirrors of
 * the platform-facing refs / DTOs. No domain logic lives here. The platform
 * FastAPI service (`platform_web/README.md` "API contract") will return these
 * shapes verbatim; when the OpenAPI-generated client lands it replaces `client.ts`.
 */

// ---------------------------------------------------------------------------
// §4.1 ArtifactRef (contracts/artifact_ref.py)
// ---------------------------------------------------------------------------

export interface ArtifactRef {
  artifact_id: string;
  artifact_type: ArtifactType;
  schema_version: string;
  content_hash: string; // 64-char lowercase hex sha256, reported by the publisher
  storage_uri: string; // COS / signed object uri carrying a scheme
  size_bytes: number; // >= 0
  created_at: string; // UTC ISO-8601
  producer_type: string; // e.g. FACTOR_ENGINE
  producer_version: string;
  semantic_hash?: string;
  media_type?: string;
  producer_source_ref?: string | null;
  snapshot_ref?: string | null;
  universe_ref?: string | null;
  security_classification?: SecurityClassification | null;
}

// ---------------------------------------------------------------------------
// §4.3 JobSpec / JobRecord / JobAttempt / JobResult (contracts/jobs.py)
// ---------------------------------------------------------------------------

export interface JobSpec {
  job_type: string;
  idempotency_key: string;
  inputs: string[];
  activity_kind?: string | null;
  priority?: number;
  max_retries?: number;
  timeout_seconds?: number | null;
  heartbeat_seconds?: number | null;
  resource_class?: string;
  estimated_factor_count?: number;
  estimated_row_count?: number;
  input_artifact_refs: string[];
  created_at?: string | null;
}

export interface JobAttempt {
  worker_id: string;
  heartbeat?: string | null;
  error_class?: string | null;
  logs_ref?: string | null;
}

export interface JobRecord {
  status: JobStatus;
  current_stage?: string | null;
  progress?: number; // [0, 1]
  created_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  attempt_count?: number;
  attempts: JobAttempt[];
  error_class?: string | null;
}

export interface JobResult {
  output_artifact_refs: string[];
  summary: string;
}

// ---------------------------------------------------------------------------
// §4.2 EventEnvelope (contracts/event_envelope.py)
// ---------------------------------------------------------------------------

export interface EventEnvelope {
  event_id: string;
  event_type: string;
  schema_version: string;
  occurred_at: string; // UTC ISO-8601
  producer: string;
  aggregate_type: string;
  aggregate_id: string;
  correlation_id: string;
  causation_id?: string | null;
  payload: Record<string, unknown>;
  trace_id?: string | null;
  actor_principal_id?: string | null;
  idempotency_key?: string | null;
  event_version?: number;
}

// ---------------------------------------------------------------------------
// §4.4 FactorCandidateManifest (contracts/candidate.py)
// ---------------------------------------------------------------------------

export interface FactorCandidateManifest {
  schema_version: string;
  candidate_id: string;
  submitted_at: string; // ISO-8601
  submitted_by: string;
  generator_type: string;
  generator_version: string;
  market: string;
  frequency: string;
  formula_language: string;
  factor_spec_uri: string;
  factor_spec_sha256: string;
  parent_factor_ids: string[];
  required_fields: string[];
  semantic_family_hint?: string | null;
  campaign_id?: string | null;
  attempt_id?: string | null;
}

// ---------------------------------------------------------------------------
// §4.5 Lifecycle / Health / Pipeline stage (contracts/lifecycle.py)
// ---------------------------------------------------------------------------

export interface FactorAsset {
  factor_definition_id: string; // domain-owned carried digest (64-hex)
  name: string;
  family: string;
  generator: string;
  campaign_id: string | null;
  lifecycle: LifecycleState;
  health: HealthState;
  pipeline_stage: QRPPipelineStage;
  grade: string | null;
  rank_ic: number | null;
  icir: number | null;
  coverage: number | null;
  turnover: number | null;
  selected_treatment_id: string | null;
  cluster_id: string | null;
  libraries: string[];
  created_at: string;
  last_evaluated_at: string | null;
  evidence_status: EvidenceStatus; // LABEL_NOT_MATURE — never zero-filled
}

// ---------------------------------------------------------------------------
// §4.6 Clusters (contracts/cluster_library.py + contracts/cluster.py)
// ---------------------------------------------------------------------------

export interface SimilarityGraphVersion {
  graph_version_id: string;
  graph_ref: string;
  policy_hash?: string;
  created_at?: string | null;
}

export interface ClusterSetVersion {
  cluster_set_version_id: string;
  similarity_graph_version: SimilarityGraphVersion;
  algorithm: string;
  backend?: string;
  seed?: number;
  resolution?: number | null;
  policy_hash?: string;
  clustering_run_ref?: string;
  created_at?: string | null;
}

export interface ClusterVersion {
  cluster_version_id: string;
  logical_cluster_id: string; // stable, e.g. CL_PV_MOM_0017
  cluster_set_version_id: string;
  algorithm_cluster_label: string; // per-run, e.g. "cluster 18"
  member_factor_ids: string[];
  policy_hash?: string;
  conversion_type?: ClusterConversionType | null;
  created_at?: string | null;
}

export interface ClusterMembership {
  factor_definition_id: string;
  logical_cluster_id: string;
  created_at?: string | null;
}

export interface ClusterLineageEdge {
  old_version_id: string;
  new_version_id: string;
  transition: ClusterConversionType;
  created_at?: string | null;
}

// ---------------------------------------------------------------------------
// §4.7 FactorLibraryVersion + LibraryMembership (contracts/factor_library.py)
// ---------------------------------------------------------------------------

export interface LibraryMembership {
  factor_definition_id: string;
  selected_treatment_id?: string | null;
  orientation?: string | null;
  cluster_id?: string | null;
  representative_of?: string | null;
  similarity_ref?: string | null;
  health_state_ref?: string | null;
  evidence_ref?: string | null;
  assembly_score?: number | null;
  selection_rank?: number | null;
}

export interface FactorLibraryVersion {
  library_version_id: string;
  logical_library_id: string; // e.g. CORE_LOW_REDUNDANCY
  cluster_set_version_id: string; // global clustering run, NOT a single cluster
  members: LibraryMembership[];
  policy_hash?: string;
  evidence_snapshot?: string;
  status: LibraryStatus; // CANDIDATE / SHADOW / APPROVED / PRODUCTION
  created_at?: string | null;
}

export interface LibraryPromotionRecord {
  logical_library_id: string;
  version_id: string;
  from_status: LibraryStatus;
  to_status: LibraryStatus;
  actor_principal_id?: string;
  reason?: string;
  created_at?: string | null;
}

// ---------------------------------------------------------------------------
// §4.8 FeatureSet (contracts/feature_set.py)
// ---------------------------------------------------------------------------

export interface FeatureMemberRef {
  position: number;
  feature_name: string;
  factor_definition_ref: string;
  raw_value_ref?: string | null;
  treatment_selection_ref?: string | null;
  treated_feature_ref?: string | null;
  orientation?: string | null;
  dtype?: string | null;
  channel?: string | null;
  timing_ref?: string | null;
  source_artifact_id?: string | null;
  availability_semantics?: string | null;
  security_classification?: SecurityClassification | null;
  metadata?: Record<string, unknown>;
}

export interface FeatureSetVersion {
  feature_set_id: string;
  version: string;
  ordered_members: FeatureMemberRef[];
  consumer_profile?: string | null;
  source_library_versions?: string[];
  label_definition_ref?: string | null;
  data_revision_ref?: string | null;
  schema_hash?: string;
  semantic_hash?: string;
  created_at?: string | null;
}

export interface FeatureSetArtifact {
  feature_set_id: string;
  feature_set_version: string;
  source_library_versions?: string[];
  ordered_feature_manifest?: string[];
  created_at?: string | null;
  content_hash?: string;
}

// ---------------------------------------------------------------------------
// §4.9 Carried identity refs (contracts/identities.py)
// ---------------------------------------------------------------------------

export interface FactorDefinitionRef {
  hash: string; // 64-char lowercase hex — carried, never recomputed
  factor_version: string;
  descriptor?: Record<string, unknown>;
}

export interface FactorValueRef {
  hash: string;
  factor_definition_ref: string;
}

export interface EvaluationRef {
  hash: string;
  factor_value_ref: string;
}

export interface TreatmentRef {
  hash: string;
  source_factor_value_ref: string;
}

// ---------------------------------------------------------------------------
// §4.9b Admission (contracts/admission.py)
// ---------------------------------------------------------------------------

export interface AdmissionRequest {
  candidate_ref: string;
  content_hash: string;
  factor_definition_ref?: string;
  semantic_ref?: string;
  evaluation?: unknown;
  library_snapshot_ref?: string;
  context?: Record<string, unknown>;
}

export interface AdmissionVerdict {
  decision: 'APPROVED' | 'REJECTED' | 'SHADOWED';
  reason_codes?: string[];
  content_hash?: string;
  policy_ref?: string;
  authority?: string;
  detail?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// §4.12 Backtest / Model DTOs (contracts/backtest.py)
// ---------------------------------------------------------------------------

export interface BacktestRequest {
  request_id: string;
  idempotency_key: string;
  signal_artifact_ref: ArtifactRef;
  execution_policy_ref: string;
  universe_id?: string | null;
  start_time?: string | null;
  end_time?: string | null;
}

export interface BacktestArtifactRef {
  artifact: ArtifactRef;
  backtest_id: string;
  strategy_ref: string;
  request_ref: string;
  result_summary_uri: string;
  qe_reevaluation_status: EvidenceStatus;
}

export interface ModelDatasetRequest {
  dataset_request_id: string;
  feature_set_artifacts: string[];
  label_policy_ref?: string;
  output_uri?: string | null;
}

export interface ModelTrainingRequest {
  training_request_id: string;
  idempotency_key: string;
  model_dataset_ref: ArtifactRef;
  model_spec_ref: string;
  training_params?: Record<string, unknown>;
}

export interface ModelArtifactRef {
  artifact: ArtifactRef;
  model_id: string;
  model_version: string;
  feature_set_ref: string;
  train_dataset_ref: string;
  eval_metrics_uri?: string;
}

// ---------------------------------------------------------------------------
// §4.13 Timing / §4.14 Evidence (contracts/timing.py)
// ---------------------------------------------------------------------------

export interface TimingContract {
  decision_time?: string | null;
  signal_available_time?: string | null;
  first_executable_time?: string | null;
  label_start_time?: string | null;
  label_end_time?: string | null;
}

// ---------------------------------------------------------------------------
// Versioning chain (contracts/versioning.py, contracts/model_version.py)
// ---------------------------------------------------------------------------

export interface ModelWeight {
  feature_name: string;
  position: number;
  weight: number;
}

export interface LabelDefinition {
  label_definition_id: string;
  label_name: string;
  horizon: string;
  return_basis?: string; // "vwap" — global caliber
  aggregation?: string;
  universe_id?: string;
}

export interface DataSnapshot {
  snapshot_id: string;
  start_time?: string | null;
  end_time?: string | null;
  universe_id?: string;
  data_revision_ref?: string;
}

export interface SplitPlan {
  split_plan_id: string;
  train_ratio: number;
  validation_ratio: number;
  test_ratio: number;
  seed: number;
}

export interface ModelVersion {
  model_id: string;
  model_version: string;
  model_architecture: string;
  feature_set_version_ref: string;
  label_definition: LabelDefinition;
  data_snapshot: DataSnapshot;
  split_plan: SplitPlan;
  training_params?: Record<string, unknown>;
  weights: ModelWeight[];
  training_run_ref?: string;
  eval_metrics_uri?: string;
  created_at?: string | null;
}

// ---------------------------------------------------------------------------
// Daily FeatureSet snapshot (contracts/daily_snapshot.py)
// ---------------------------------------------------------------------------

export interface SnapshotFeatureRef {
  position: number;
  feature_name: string;
  factor_definition_id: string;
  treatment_id?: string | null;
  orientation?: string | null;
  dtype?: string | null;
}

export interface SnapshotManifest {
  feature_set_version_id: string;
  feature_set_content_hash: string;
  ordered_features: string[];
  schema_hash: string;
  data_content_hash: string;
  data_snapshot_id: string;
  universe_id: string;
  trade_date: string;
  row_count: number;
  column_count: number;
  created_at: string;
  manifest_hash?: string;
}

// ---------------------------------------------------------------------------
// Workflow (contracts/workflow.py)
// ---------------------------------------------------------------------------

export interface WorkflowSpec {
  workflow_type: string;
  idempotency_key: string;
  jobs: JobSpec[];
  inputs?: Record<string, unknown>;
  policy_ref?: string | null;
  created_at?: string | null;
}

export interface WorkflowRun {
  workflow_id: string;
  status: 'PENDING' | 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'CANCELED' | 'TIMED_OUT';
  job_records: JobRecord[];
  created_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
}

// ---------------------------------------------------------------------------
// RBAC (contracts/permission.py, contracts/principal.py)
// ---------------------------------------------------------------------------

export interface HumanPrincipal {
  principal_id: string;
  display_name: string;
  team: string;
  role: string;
  classification: SecurityClassification;
  principal_type: 'HUMAN';
}

export interface WorkloadPrincipal {
  principal_id: string;
  service_name: string;
  team: string;
  role: string;
  classification: SecurityClassification;
  principal_type: 'SERVICE';
}

// ---------------------------------------------------------------------------
// §26 Dashboard KPI (spec §26) — derived ONLY from API responses, never
// computed in the frontend.
// ---------------------------------------------------------------------------

export interface DashboardKpis {
  total_factors: number;
  new_today: number;
  new_7d: number;
  processing: number;
  validated: number;
  shadow: number;
  production_eligible: number;
  production: number;
  degraded: number;
  quarantined: number;
}

export interface PipelineFunnel {
  generated: number;
  valid: number;
  compiled: number;
  materialized: number;
  qe_passed: number;
  novel: number;
  clustered: number;
  library_candidate: number;
  promoted: number;
}

export interface OperationalSignals {
  job_backlog: number;
  failure_rate: number; // ratio [0, 1]
  factor_throughput: number; // factors / day
  cos_ingest_lag_seconds: number;
  cluster_drift: number;
  library_version_changes: number;
  recent_promotions: number;
  health_alerts: number;
}

export interface DashboardSummary {
  kpis: DashboardKpis;
  funnel: PipelineFunnel;
  signals: OperationalSignals;
}
