# GAP_ANALYSIS — QRP repo vs master spec at HEAD 26b4c02

- **HEAD:** `26b4c02c58d70410c670da4d57e346e7d6a41dcc`
- **Date:** 2026-08-27
- **Spec:** `QUANT_RESEARCH_PLATFORM_MASTER_IMPLEMENTATION_SPEC_20260826.md`
- **Mode:** READ-ONLY. Working tree is truth. Contract layer is being corrected by another agent — referenced by intended shape, not final on-disk state.
- **Disposition legend:** REUSE_AS_IS / REUSE_WITH_ADAPTER / EXTEND_EXISTING / ADD_NEW / BLOCKED.

This is a delta re-audit. The 5 prior audits (`quant_platform/docs/audit/`, stamped `d58eae`) were re-validated at HEAD; only `factor_engine`, `factor_preprocess`, `factor_optimizer`, `factor_assets`, `quant_platform` changed. Findings below are current.

---

## 1. Platform layer gaps (spec §3/§4/§56 Phase 1+)

| # | Requirement | Current at HEAD | Gap | Disposition | Priority |
|---|---|---|---|---|---|
| G1 | Platform backend package | `quant_platform/` = contracts + docs + pyproject + smoke test only | No `app/api`, `app/auth`, `app/persistence`, `app/storage`, `app/workers`, `app/services`, `app/events`, `app/observability`, `app/settings` | **ADD_NEW** | P0 |
| G2 | Platform API (spec §23) | Two domain FastAPI services exist: FE `service/app.py` (port 8088), DA `service/app.py` (port 8765) | No unified platform API; no `app/api/main.py` | **EXTEND_EXISTING** (reuse FE/DA FastAPI+Starlette patterns) | P0 |
| G3 | PostgreSQL metadata schema (spec §19) | No sqlalchemy/alembic/psycopg in lock or `.venv` (verified: 0 hits in `requirements-production.lock`) | DB layer absent | **ADD_NEW** | P0 |
| G4 | Alembic migrations (spec §47) | No `migrations/` dir | No migration framework | **ADD_NEW** | P0 |
| G5 | Auth (spec §24.5) | FE `service/security.py` `resolve_principal` (API-key→principal mapping; JWT/mTLS/reverse-proxy are *sources*, not implemented); DA `service/app.py` `hmac.compare_digest` API-key | No JWT/session/OAuth; no argon2/bcrypt crypto in any domain package | **EXTEND_EXISTING** (build on `resolve_principal` + platform `rbac.py` DRAFT) | P0 |
| G6 | RBAC + audit log (spec §24) | `quant_platform/app/contracts/rbac.py` DRAFT (Permission/Role/Team/SecurityClassification/ROLE_PERMISSIONS) | Contract only; no enforcement middleware, no audit-log store | **REUSE_WITH_ADAPTER** (contract) + **ADD_NEW** (enforcement) | P0 |
| G7 | Web plane / React (spec §3, §49) | No `platform_web/`, no node/npm in env | Frontend absent | **ADD_NEW** | P1 |
| G8 | Workflow runtime / Temporal (spec §12, §51) | `quant_platform/app/contracts/workflow.py` `WorkflowBackend` Protocol (DRAFT); FE in-memory `BoundedJobQueue` (`service/queue.py:96`); no temporalio importable | No control plane | **REUSE_WITH_ADAPTER** (Protocol) + **ADD_NEW** (impl) | P1 |
| G9 | Transactional outbox (spec §11.1) | `EventEnvelope` contract only; no outbox table/worker | Events can be lost on crash | **ADD_NEW** | P2 |
| G10 | Deployment compose (spec §51) | Only `data_access/docker-compose.yml`; root `Dockerfile` = FactorEngine service (CMD 8766, but service defaults 8088 — **mismatch**) | No platform-api/web/postgres/worker compose | **ADD_NEW** | P1 |
| G11 | Observability (spec §45) | FE/DA have `telemetry/prometheus_exporter.py` + `opentelemetry_bridge.py`; FE service `/metrics` route | Domain-local only; no platform-level SLO/observability | **REUSE_AS_IS** (domain telemetry) + **EXTEND_EXISTING** (platform) | P2 |
| G12 | CI completeness (spec §52) | `.github/workflows/ci.yml` gate_runner verifiers (source-authority, supply-chain, evidence-current, submodule-reachability, no-mirror-packages, fresh-wheel-matrix, gate matrix) | No ruff/formatter, no mypy/pyright, no frontend lint/tsc, no migration test, no OpenAPI-compat check, no full `--require-hashes` clean-venv install | **EXTEND_EXISTING** | P1 |
| G13 | `quant_platform` independent wheel | `pyproject.toml` (untracked) builds `quant-platform-0.1.0` wheel; contracts import (`__all__` len 107) | Wheel not published/installed anywhere; not in `requirements-production.lock` | **REUSE_AS_IS** (build verified) + **ADD_NEW** (publish/lock) | P1 |

---

## 2. Domain-package gaps (delta-validated)

### 2.1 quant_evaluator (unchanged at HEAD — old findings hold)
| # | Requirement | Current | Gap | Disposition | Priority |
|---|---|---|---|---|---|
| G14 | Single metric registry authority (spec §33.2) | **TWO populated catalogs**: `registry/metrics.py` module `_METRIC_CATALOG` (register_metric, ~20 core metrics) AND `metrics/catalog.py` catalog-scoped `_REGISTRY`/`CATALOG` (~10-domain specs). Overlap on `pearson_ic`, `rank_ic`, `quantile_spread` with divergent `ic_method`/domain/tier. | Dual authority — the exact smell spec §33.2 forbids | **EXTEND_EXISTING** (unify to one canonical registry; make `metrics/catalog.py` a read-only view of the same instance) | P0 |
| G15 | `EvaluationArtifact` single truth (spec §33.1) | Only `EvaluationBundle` (`api/requests.py:96`); no `EvaluationArtifact` class anywhere | Missing canonical artifact | **ADD_NEW** | P0 |
| G16 | Evidence status set (spec §40) | QE `EvidenceStatus` = COMPUTED/NOT_COMPUTED/UNAVAILABLE/UNSUPPORTED/INSUFFICIENT_DATA/FAILED | Missing `INVALID_EVIDENCE`, `LABEL_NOT_MATURE` (platform `timing.py` provides them as DRAFT) | **EXTEND_EXISTING** (add 2 enum members) | P0 |
| G17 | Large series via ArtifactRef (spec §33.6) | QE has no `ArtifactRef`; series in-memory ndarray / `series_refs` strings | No ArtifactRef in QE | **REUSE_WITH_ADAPTER** (platform `artifact_ref.py`) | P1 |
| G18 | Grade / health evidence (spec §33) | No `Grade`/`Health` subsystem; only `diagnosis/factor.py` warnings | Missing | **ADD_NEW** | P1 |
| G19 | Report export | `reporting/__init__.py` stubs `generate_library_report`/`compare_libraries`/`generate_adversarial_report` as `NotImplementedError` | Library/adversarial reports not implemented | **EXTEND_EXISTING** | P1 |

### 2.2 factor_assets (2 files changed at HEAD — treatment_selection hardening; rest holds)
| # | Requirement | Current | Gap | Disposition | Priority |
|---|---|---|---|---|---|
| G20 | `EvaluationIdentity` (spec §8.3) | Missing in FA (only `FactorValueIdentity` + `ResultIdentity`) | Platform `identities.py` provides DRAFT `EvaluationIdentity` | **REUSE_WITH_ADAPTER** (reconcile FA ↔ platform) | HIGH |
| G21 | Full lifecycle state machine (spec §9) | FA has 6 states (`contracts/lifecycle.py:27`); platform `lifecycle.py` DRAFT has 23 | FA state machine incomplete vs spec | **EXTEND_EXISTING** (FA) + **REUSE_WITH_ADAPTER** (platform enum) | HIGH |
| G22 | Cluster version / logical id / incremental assignment (spec §14, §36, Stage 11) | FA `ClusterArtifact` stores raw algorithm label only; no logical_cluster_id, no ClusterVersionArtifact, no incremental states | **P6 CRITICAL** — platform `cluster_library.py` DRAFT provides the layered model (SimilarityGraphVersion/ClusterSetVersion/LogicalCluster/ClusterVersion/ClusterLineageEdge) | **REUSE_WITH_ADAPTER** (platform contracts) + **ADD_NEW** (FA impl) | P6 |
| G23 | Factor Library governance (spec §15/§16) | Missing in FA | Platform `cluster_library.py` DRAFT provides `FactorLibraryVersion`/`LibraryMembership` | **REUSE_WITH_ADAPTER** + **ADD_NEW** | HIGH |
| G24 | FeatureSet (spec §17/§18) | Missing in FA | Platform `feature_set.py` DRAFT provides `FeatureSetArtifact`/`FeatureSetDiffCategory`/`ModelRetrainRequiredEvent` | **REUSE_WITH_ADAPTER** + **ADD_NEW** | HIGH |
| G25 | Similarity fingerprint / multi-view (spec §10) | `SimilarityArtifact` views lack pearson/quantile_overlap/regime_conditional; no `SimilarityFingerprintArtifact` | Missing views + fingerprint | **EXTEND_EXISTING** | MED |
| G26 | Assembly composite scores computed in FA | `family_robust`/`quality`/`objectives` read from external metadata, not computed in FA | Ownership decision needed (compute vs consume) | **EXTEND_EXISTING** | MED |

### 2.3 factor_preprocess (2 files changed at HEAD — KAMA `use_current`; rest holds)
| # | Requirement | Current | Gap | Disposition | Priority |
|---|---|---|---|---|---|
| G27 | `TreatmentRecipe` (spec §34, Stage 7) | Missing everywhere; only free-form `winner_recipe: Mapping` in `factor_assets/contracts/treatment_selection.py:250` | No typed winning-treatment contract | **ADD_NEW** | P0 |
| G28 | Causality certification module (spec §34) | `causal_safe`/`admission` flags + `CausalityClass` enum; no dedicated certification record | No per-transform CausalityCertificate / audit trail | **ADD_NEW** | P0 |
| G29 | Asset-isolation property test | Isolation by construction; no standalone test file | Missing dedicated test | **ADD_NEW** | P2 |

### 2.4 factor_optimizer (2 files changed at HEAD — winner-selector validation; rest holds)
| # | Requirement | Current | Gap | Disposition | Priority |
|---|---|---|---|---|---|
| G30 | Multiple-testing correction consumed (spec §35, Stage 7) | Ledger `status_counts` counts failures; corrections in `quant_evaluator/metrics/multiple_testing.py` not imported by FO | No correction applied in FO | **REUSE_WITH_ADAPTER** (QE) + **EXTEND_EXISTING** (FO) | P1 |
| G31 | FO emits `TreatmentSelectionArtifact` | Runner returns `SearchSession` only; artifact consumed by `factor_assets/assembly/engine.py:667` | Search orchestration doesn't produce Stage-7 output artifact | **ADD_NEW** | P1 |
| G32 | Treatment idempotency key (spec §42) | Content hashes exist; platform `jobs.py` provides `treatment_idempotency_key` | No composite key / CACHE_HIT path in FP/FO | **REUSE_WITH_ADAPTER** (platform) + **ADD_NEW** | P2 |

### 2.5 factor_engine (38 files changed at HEAD — new runtime modules; rest holds)
| # | Requirement | Current | Gap | Disposition | Priority |
|---|---|---|---|---|---|
| G33 | §10.3 COS Object Event ingestion | `EventEnvelope` contract only; no COS event source → ingestion | No ingestion service | **ADD_NEW** | P0 |
| G34 | §10.3 reconciliation scanner | Local complete markers only (`data_access/cos_storage_runtime.py:70`) | No periodic scanner / cursor / etag | **ADD_NEW** | P0 |
| G35 | §10.1 candidate artifact lake writer | `FactorCandidateManifest` contract only | No producer writes manifest.json + `_READY` to COS | **ADD_NEW** | P0 |
| G36 | §21 canonical factor-major values on COS | Factor lake local-only (`workspace_paths.py:69`) | No `factor_values/market/freq/factor_bucket/factor_id/snapshot` writer | **EXTEND_EXISTING** (write_targets pattern) | P1 |
| G37 | §21 feature-block derived cache | `materialize_matrix` local only | No `feature_blocks/` writer | **ADD_NEW** | P1 |
| G38 | §39 five timing fields consumable | `TimingContract` DRAFT + `decision_time_policy` in FE identity; DA has `decision_time` only | Other 4 fields not computed/consumed in runtime | **EXTEND_EXISTING** | P1 |
| G39 | §39 A-share evidence as unified contract | Scattered (`ashare_intraday.py`, PIT clocks) | No single A-share timing evidence object | **ADD_NEW** | P1 |
| G40 | §11.1 transactional outbox | `EventEnvelope` contract only; FE queue in-memory | No outbox table/worker | **ADD_NEW** | P2 |
| G41 | §37 Artifact metadata return | Job-level `job.artifacts` + `FactorArtifactMetadata` | No platform `ArtifactRef` publish registry in FE | **REUSE_WITH_ADAPTER** (platform `artifact_ref.py`) | P2 |

### 2.6 data_access (unchanged at HEAD — old findings hold)
| # | Requirement | Current | Gap | Disposition | Priority |
|---|---|---|---|---|---|
| G42 | §38 named `DataSnapshotRef`/`UniverseRef`/`FieldAvailability`/`DataFreshness` | `DataSnapshot`/`UniverseSnapshot`/`SourceColumnFreshness` exist; exact spec names absent | Adapter contract names differ | **REUSE_WITH_ADAPTER** (alias/export adapter-facing refs) | P1 |
| G43 | §22 `LocalArtifactCache` | `LocalObjectStore` exists; no `LocalArtifactCache` class | Spec cache absent (platform `storage.py` DRAFT provides Protocol) | **REUSE_WITH_ADAPTER** (platform Protocol) + **ADD_NEW** (impl) | P1 |
| G44 | COS multipart real vs simulated | **At HEAD:** `COSObjectStore` multipart is a no-op delegating to `put_object`. **Working tree (uncommitted):** a real boto3 multipart impl exists (`begin_multipart`→`create_multipart_upload`, `_flush_one`→`upload_part`, `complete_multipart`), but `boto3` is NOT importable in `.venv` | Real multipart is uncommitted + unverified | **BLOCKED** (needs boto3 dep + commit + test) | P1 |

---

## 3. research_platform — legacy dual artifact authority (spec §32)

`research_platform/` is a self-contained legacy layer with **15 `*Artifact` classes** (`artifacts.py`: `Artifact`, `DataSnapshotArtifact`, `FactorDefinitionArtifact`, `FactorValueArtifact`, `EvaluationBundle`, `SimilarityArtifact`, `AdmissionDecisionArtifact`, `FactorSetArtifact`, `FeatureBundle`, `ModelDatasetArtifact`, `ModelTrainingArtifact`, `BacktestArtifact`, `PortfolioArtifact`, `RiskSnapshotArtifact`, `ExpectedReturnArtifact`) plus lineage graph, campaign ledger, health, firewall, scorecard.

**This is a second artifact authority** — the exact smell the QRP directive forbids. It must migrate, not be deleted (it holds valuable lineage/invalidation/firewall/health/scorecard patterns). The full per-module migration matrix is in `quant_platform/docs/RESEARCH_PLATFORM_MIGRATION_MATRIX.md` (written at C3, still accurate at HEAD — `research_platform` is unchanged).

| research_platform module | Canonical owner | Disposition |
|---|---|---|
| `artifacts.py` (15 `*Artifact`) | `quant_platform` `ArtifactRef` + domain owners | **MIGRATE** / **COMPAT_READ_ONLY** |
| `graph.py` (lineage/invalidation) | PostgreSQL `artifact_lineage` + `EvidenceInvalidationService` | **EXTEND_EXISTING** |
| `campaign.py` (campaign ledger) | `factor_assets/campaigns/` + `factor_optimizer/contracts/trial_ledger.py` | **EXTEND_EXISTING** / **REUSE_WITH_ADAPTER** |
| `health.py` | `factor_assets/lifecycle/state_machine.py` | **EXTEND_EXISTING** |
| `firewall.py` (10-stage heuristic chain) | `CandidatePreflightService` (real FE/DA checks) | **EXTEND_EXISTING** / **REUSE_AS_IS** |
| `scorecard.py` | Generator/Campaign Analytics Read Model | **MIGRATE** |

**FORBIDDEN:** new code must not write `research_platform` artifact types; must produce the canonical owner type and persist via a platform `ArtifactRef`.

---

## 4. Cross-cutting gaps

| # | Requirement | Current | Gap | Disposition | Priority |
|---|---|---|---|---|---|
| G45 | Spec §2 exact model contracts | `ModelDatasetRequest`/`ModelTrainingRequest`/`ModelArtifactRef` NOT in `modeling/`; provided as DRAFT in `quant_platform/app/contracts/backtest.py` | Ownership ambiguity: platform DTO vs modeling-native | **REUSE_WITH_ADAPTER** (reconcile) | P0 |
| G46 | Spec §5 dependency rules | Each domain pkg own pyproject; adapters as optional extras; `riskfolio_qs` hard-depends on `data_access` | No CI-level dependency gate for the forbidden edges (e.g. `factor_assets → FastAPI`) | **EXTEND_EXISTING** (add to `check_no_mirrors.py` or new gate) | P1 |
| G47 | Spec §7.4 two-phase artifact publish | `ArtifactRef` DRAFT + `storage.py` Protocol; no publisher | No two-phase publish registry | **ADD_NEW** | P1 |
| G48 | Spec §19.1 DB stores summary + ArtifactRef, not large matrices | No DB at all | DB layer absent | **ADD_NEW** | P0 |
| G49 | Spec §21 COS data-lake layout | COS holds clean-dataset mirrors (`cos/mirror.py:34`), not canonical factor layout | Layout mismatch | **EXTEND_EXISTING** | P1 |
| G50 | Spec §48 report standards registry | QE `reporting/` reads artifacts only (good); no platform Standards registry | Standards registry absent | **ADD_NEW** | P1 |

---

## 5. BLOCKED / UNVERIFIED items

- **BLOCKED (G44):** real COS multipart is uncommitted in the working tree and `boto3` is not importable in `.venv` — cannot be verified or relied on at HEAD.
- **UNVERIFIED:** full FE test suite not run (collection broken per prior audit; ParamRole pollution being fixed by another agent). Module existence verified by source inspection.
- **UNVERIFIED:** end-to-end PIT wiring (FE four-layer gate ↔ DA contracts) verified at contract level only.
- **UNVERIFIED:** `quant_platform` wheel install into a clean venv (ensurepip unavailable in this env); build + import in repo `.venv` verified.

_End of GAP_ANALYSIS._
