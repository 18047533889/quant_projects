/**
 * Typed client for the platform FastAPI service.
 *
 * The platform API surface is defined in `quant_platform/docs/QRP_CAPABILITY_MATRIX.md`
 * (capability #2/#23 — API) and the DTOs in `quant_platform/app/contracts/*`.
 * The frontend renders ONLY what these endpoints return (spec §49: no domain
 * logic in the frontend). Endpoints are stubbed via `NOT_IMPLEMENTED` and the
 * typed request helpers return the canonical shapes — ready to be pointed at the
 * real service in CI / local dev.
 *
 * PERMISSION NOTE: formula / raw-value endpoints are 403 for principals without
 * `factor:read_formula` / `factor:read_raw_values` (spec §24.2). The client
 * propagates the 403 so pages can render the permission-denied state.
 */

import type {
  AdmissionVerdict,
  ArtifactRef,
  ClusterLineageEdge,
  ClusterMembership,
  ClusterSetVersion,
  DashboardSummary,
  EventEnvelope,
  FactorAsset,
  FactorCandidateManifest,
  FactorLibraryVersion,
  FeatureSetVersion,
  JobRecord,
  JobSpec,
  ModelVersion,
} from './types';
import type { JobStatus } from './enums';

export const DEFAULT_API_BASE_URL = '/api/v1';

/** Base URL of the platform FastAPI service (override via VITE_PLATFORM_API_BASE_URL). */
export const API_BASE_URL: string =
  import.meta.env?.VITE_PLATFORM_API_BASE_URL ?? DEFAULT_API_BASE_URL;

/** Canonical error envelope (spec §50). */
export interface ApiError {
  error: {
    code: string; // e.g. "FACTOR_NOT_FOUND"
    message: string;
    request_id: string;
    retryable: boolean;
  };
}

export const API_ERROR_CODES = {
  PERMISSION_DENIED: 'PERMISSION_DENIED',
} as const;

export class PlatformApiError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly status: number,
    public readonly requestId?: string,
    public readonly retryable = false,
  ) {
    super(message);
    this.name = 'PlatformApiError';
  }
}

/** Resource envelope: the backend's registry rows carry evidence/state fields. */
export interface RegistryRow<T> {
  ref: string;
  entity: T;
  lifecycle?: string;
  health?: string;
  evidence_status?: string;
  created_at?: string;
}

/**
 * Typed request helper. Sends a JSON request to the platform API and parses
 * either the domain payload or the canonical error envelope (spec §50). A 403
 * is surfaced as `PlatformApiError` with code PERMISSION_DENIED so pages render
 * the permission-denied state instead of an empty table.
 */
export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { Accept: 'application/json', ...(init?.headers ?? {}) },
    ...init,
  });
  if (res.status === 204) {
    return undefined as T;
  }
  if (!res.ok) {
    let code = `HTTP_${res.status}`;
    let message = `Request failed with status ${res.status}`;
    let requestId: string | undefined;
    let retryable = false;
    try {
      const body = (await res.json()) as ApiError;
      code = body?.error?.code ?? code;
      message = body?.error?.message ?? message;
      requestId = body?.error?.request_id;
      retryable = body?.error?.retryable ?? false;
    } catch {
      // Non-JSON error body — keep the default code/message.
    }
    throw new PlatformApiError(code, message, res.status, requestId, retryable);
  }
  return (await res.json()) as T;
}

/** Query params — object values joined as &key=value. */
export type QueryParams = Record<string, string | number | boolean | undefined>;

function withQuery(path: string, params?: QueryParams): string {
  if (!params) return path;
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined) qs.append(k, String(v));
  }
  const s = qs.toString();
  return s ? `${path}?${s}` : path;
}

/** Every path declared here exists in the platform API contract. */
export const platformApi = {
  dashboard: () =>
    request<DashboardSummary>(withQuery('/dashboard', { include: 'kpis,funnel,signals' })),

  listFactors: (params?: QueryParams) => request<FactorAsset[]>(withQuery('/factors', params)),
  getFactor: (id: string) => request<FactorAsset>(`/factors/${encodeURIComponent(id)}`),

  listCandidates: () => request<RegistryRow<FactorCandidateManifest>[]>('/candidates'),
  listCampaigns: () =>
    request<RegistryRow<{ campaign_id: string; name: string; status: string }>[]>(
      '/campaigns',
    ),
  listTreatments: () =>
    request<
      RegistryRow<{
        treatment_id: string;
        factor_definition_id: string;
        policy_hash: string;
        created_at?: string | null;
      }>[]
    >('/treatments'),

  listClusters: () => request<RegistryRow<ClusterSetVersion>[]>('/clusters'),
  listClusterVersions: () => request<ClusterVersionRef[]>('/clusters/versions'),
  listClusterLineage: () => request<ClusterLineageEdge[]>('/clusters/lineage'),
  listClusterMemberships: () => request<ClusterMembership[]>('/clusters/memberships'),

  listLibraries: () => request<FactorLibraryVersion[]>('/libraries'),
  listLibraryPromotions: () => request<LibraryPromotionRow[]>('/libraries/promotions'),

  listFeatureSets: () => request<FeatureSetVersion[]>('/feature_sets'),
  listModels: () => request<ModelVersion[]>('/models'),

  listBacktests: () => request<BacktestRefRow[]>('/backtests'),
  listOptimizers: () => request<OptimizerRow[]>('/optimizers'),

  listLiveHealth: () => request<RegistryRow<FactorAsset>[]>('/production/live-health'),
  listStreaming: () => request<StreamingStatusRow[]>('/production/streaming'),
  listAlerts: () => request<AlertRow[]>('/production/alerts'),

  listJobs: (status?: JobStatus) =>
    request<RegistryRow<JobRecord>[]>(withQuery('/jobs', status ? { status } : undefined)),
  getJob: (id: string) => request<JobRecord>(`/jobs/${encodeURIComponent(id)}`),

  listDataSnapshots: () => request<DataSnapshotRow[]>('/platform/data-snapshots'),
  listArtifacts: () => request<ArtifactRef[]>('/platform/artifacts'),
  listStandards: () => request<StandardRow[]>('/platform/standards'),
  listExports: () => request<ExportRow[]>('/platform/exports'),
  listAudit: () => request<AuditLogRow[]>('/platform/audit'),
  listUsers: () => request<UserRow[]>('/platform/users'),

  events: () => request<EventEnvelope[]>(withQuery('/events', { limit: 100 })),
};

// ---------------------------------------------------------------------------
// List-item row types (thin, platform-returned shapes).
// ---------------------------------------------------------------------------

export interface ClusterVersionRef {
  cluster_version_id: string;
  logical_cluster_id: string;
  cluster_set_version_id: string;
  algorithm_cluster_label: string;
  member_count: number;
  created_at?: string | null;
}

export interface LibraryPromotionRow {
  logical_library_id: string;
  version_id: string;
  from_status: string;
  to_status: string;
  actor_principal_id?: string;
  reason?: string;
  created_at?: string | null;
}

export interface BacktestRefRow {
  backtest_id: string;
  artifact: ArtifactRef;
  strategy_ref: string;
  result_summary_uri: string;
  qe_reevaluation_status?: string;
  created_at?: string | null;
}

export interface OptimizerRow {
  optimizer_id: string;
  search_policy_ref: string;
  status: string;
  created_at?: string | null;
}

export interface StreamingStatusRow {
  stream_name: string;
  status: string;
  last_updated_at?: string | null;
}

export interface AlertRow {
  alert_id: string;
  severity: string;
  message: string;
  factor_id?: string | null;
  job_id?: string | null;
  created_at: string;
}

export interface DataSnapshotRow {
  snapshot_id: string;
  universe_id?: string;
  start_time?: string | null;
  end_time?: string | null;
  status: string;
  created_at?: string | null;
}

export interface StandardRow {
  standard_id: string;
  name: string;
  version: string;
  status: string;
  updated_at?: string | null;
}

export interface ExportRow {
  export_id: string;
  artifact_type: string;
  status: string;
  created_at?: string | null;
}

export interface AuditLogRow {
  audit_id: string;
  actor_principal_id: string;
  action: string;
  resource: string;
  result: string; // "ALLOW" | "DENY"
  timestamp: string;
  request_id?: string | null;
}

export interface UserRow {
  principal_id: string;
  display_name: string;
  team: string;
  role: string;
  classification: string;
  principal_type: string;
}
