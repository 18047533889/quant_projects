# platform_web — Quant Research Platform Web UI

QRP-P9 (task #43). React + TypeScript + Vite frontend for the Quant Research
Platform (spec §25 information architecture, §26 dashboard, §49 frontend
requirements). This is the **source scaffold** — the real typecheck + build runs
in CI because this development environment has **no node/npm** (see "Build" below).

## What this is

- A single-page app with the spec §25 1-level nav: **Overview / Research
  (Factors, Campaigns, Treatments, Clusters, Libraries) / Models (Feature Sets,
  Models) / Portfolio (Backtests, Optimizers) / Production (Live Health,
  Streaming, Alerts) / Platform (Jobs, Data Snapshots, Artifacts, Standards,
  Exports, Audit, Users & Roles)** — 20 pages.
- The spec §26 dashboard: KPI cards, pipeline funnel, operational signals —
  rendered **only from API responses**, never computed in the frontend.
- A typed API client (`src/api/client.ts` + `src/api/types.ts`) that mirrors the
  platform contract DTOs in `quant_platform/app/contracts/*` (see "Types").
- TanStack Query hooks (`src/hooks`) that feed every page through the **six
  required states** (spec §49) via `src/components/StatePanel.tsx`:
  loading / empty / permission-denied / partial-evidence / error / stale.
  An empty list is rendered as **"No … data yet."** — never as a zero-filled
  table; `LABEL_NOT_MATURE` evidence is rendered as partial evidence, never 0.

## Stack (spec §49)

| Concern | Library |
|---|---|
| UI | React 18 |
| Language | TypeScript (strict) |
| Build | Vite 5 |
| Data fetching / cache | TanStack Query |
| Tables | TanStack Table |
| Routing | React Router 6 |
| Charts | ECharts (declared; not yet used) |
| Tests | Vitest |

`vite.config.ts` uses `vitest/config` so the same config powers dev/build/test.

## Types <-> platform contracts

`src/api/types.ts` is a **thin mirror** of the DTOs in
`quant_platform/app/contracts/` and `quant_platform/docs/PLATFORM_CONTRACTS_DRAFT.md`:

| Frontend type | Platform contract |
|---|---|
| `ArtifactRef` | `contracts/artifact_ref.py` (§4.1) |
| `JobSpec / JobRecord / JobAttempt / JobResult` | `contracts/jobs.py` (§4.3) |
| `EventEnvelope` | `contracts/event_envelope.py` (§4.2) |
| `FactorCandidateManifest` | `contracts/candidate.py` (§4.4) |
| `LifecycleState / HealthState / QRPPipelineStage` | `contracts/lifecycle.py` (§4.5) |
| `ClusterVersion / ClusterSetVersion / ClusterMembership / ClusterLineageEdge` | `contracts/cluster_library.py`, `contracts/cluster.py` (§4.6) |
| `FactorLibraryVersion / LibraryMembership` | `contracts/factor_library.py` (§4.7) |
| `FeatureSetVersion / FeatureSetArtifact / FeatureMemberRef` | `contracts/feature_set.py` (§4.8) |
| `FactorDefinitionRef / FactorValueRef / EvaluationRef / TreatmentRef` | `contracts/identities.py` (§4.9) |
| `AdmissionRequest / AdmissionVerdict` | `contracts/admission.py` (§4.9b) |
| `BacktestArtifactRef / BacktestRequest / ModelArtifactRef` | `contracts/backtest.py` (§4.12) |
| `TimingContract / EvidenceStatus` | `contracts/timing.py` (§4.13/§4.14) |
| `ModelVersion / ModelWeight / LabelDefinition` | `contracts/model_version.py`, `contracts/versioning.py` |
| `SnapshotManifest / SnapshotFeatureRef` | `contracts/daily_snapshot.py` |
| `WorkflowSpec / WorkflowRun` | `contracts/workflow.py` |
| `HumanPrincipal / WorkloadPrincipal / roles / teams / classifications` | `contracts/principal.py`, `contracts/rbac.py`, `contracts/security.py` |

Field names and enum string values are copied verbatim so the frontend renders
exactly what the platform API returns. When the platform FastAPI service lands
with an OpenAPI schema, `src/api/client.ts` / `src/api/types.ts` are the two
files to regenerate (spec §49: "TypeScript 类型应从 OpenAPI 自动生成或至少 CI
校验" — until then, CI typechecks these mirrors).

## API base URL

The client defaults to `/api/v1`. Override with the `VITE_PLATFORM_API_BASE_URL`
env var (see `src/vite-env.d.ts`). Endpoints are stubbed in `src/api/client.ts`
(`NOT_IMPLEMENTED`) so the app typechecks and builds before the backend is wired;
each hook is ready to point at the real service.

## Pages

Every page renders only what the API returns and handles all six states:

| Group | Pages |
|---|---|
| Overview | Dashboard |
| Research | Factors, Campaigns, Treatments, Clusters, Libraries |
| Models | Feature Sets, Models |
| Portfolio | Backtests, Optimizers |
| Production | Live Health, Streaming, Alerts |
| Platform | Jobs, Data Snapshots, Artifacts, Standards, Exports, Audit, Users & Roles |

## Scripts

```sh
npm run dev        # vite dev server
npm run typecheck  # tsc --noEmit
npm test           # vitest run (pure helpers in src/lib)
npm run build      # tsc --noEmit && vite build (emits dist/)
```

## Build (CI)

**This development environment has no node/npm — the build runs in GitHub
Actions.** `.github/workflows/platform-web.yml` (job `platform-web-build`)
checks out the repo, `actions/setup-node@v4` + `npm ci`, then runs
`npm run typecheck`, `npm test` and `npm run build`, and uploads `dist/`.
The `cache` for `actions/setup-node` is keyed to `platform_web/package-lock.json`
— generate it (`npm install` locally or in CI) so the cache works.

## Notes / constraints

- Pure frontend scaffold: **no domain logic in the frontend**. All numbers are
  rendered as returned; the dashboard never recomputes metrics (spec §28).
- `platform_web` does not touch `quant_platform/app/` (backend) and contains no
  Python.
- "No data" is never rendered as 0 (spec §49). Missing values render as `—`;
  an absent label-mature value renders the `LABEL_NOT_MATURE` partial-evidence
  state (spec §40).
