# QRP_CAPABILITY_MATRIX — current state at HEAD 26b4c02

- **HEAD:** `26b4c02c58d70410c670da4d57e346e7d6a41dcc`
- **Date:** 2026-08-27
- **Legend:** MISSING = absent / PARTIAL = present but does not meet spec / PRESENT = usable.
- **Mode:** READ-ONLY. Contract layer being corrected by another agent — referenced by intended shape.

For each major QRP capability: current state, owning package, and the key evidence file.

| # | QRP capability | State | Owning package | Evidence (verified at HEAD) |
|---|---|---|---|---|
| 1 | **Contract layer** (spec §6) | **PRESENT (DRAFT)** | `quant_platform/app/contracts/` | 15 pure-stdlib DTO modules; `__all__` len 107; imports clean. `ArtifactRef`, `EventEnvelope`, `JobSpec/Record/Result`, `FactorCandidateManifest`, `LifecycleState`/`HealthState`, identities, cluster/library, feature_set, RBAC, storage, timing, workflow, backtest/model DTOs. |
| 2 | **API** (spec §23) | **PARTIAL** | `factor_engine/service/app.py` (port 8088), `data_access/service/app.py` (port 8765) | Two domain FastAPI services; **no unified platform API** (`quant_platform/app/api/` absent). |
| 3 | **Postgres metadata** (spec §19) | **MISSING** | — | No sqlalchemy/alembic/psycopg in lock or `.venv` (0 hits in `requirements-production.lock`). |
| 4 | **Auth** (spec §24.5) | **PARTIAL** | `factor_engine/service/security.py`, `data_access/service/app.py` | API-key→principal mapping (`resolve_principal`); `hmac.compare_digest` in DA. No JWT/session/OAuth; no argon2/bcrypt crypto. |
| 5 | **RBAC** (spec §24.1) | **PARTIAL (contract only)** | `quant_platform/app/contracts/rbac.py` | `Permission`/`Role`/`Team`/`SecurityClassification`/`ROLE_PERMISSIONS` DRAFT; no enforcement middleware. |
| 6 | **Artifact registry / COS publisher** (spec §7, §21) | **MISSING** | — | `ArtifactRef` + `storage.py` Protocol DRAFT only; no publisher, no two-phase publish registry. Factor values local-only. |
| 7 | **Candidate ingestion + reconciliation** (spec §10) | **MISSING** | — | `FactorCandidateManifest` + `_READY` contract only; no COS event ingestion, no reconciliation scanner. |
| 8 | **Workflow / workers** (spec §12, §51) | **MISSING** | — | `WorkflowBackend` Protocol DRAFT; FE in-memory `BoundedJobQueue`; no Temporal, no workers. |
| 9 | **FE pipeline** (spec §37) | **PRESENT** | `factor_engine/` | `FactorEngine.compile` (`runtime/engine.py:1041`), `ParquetMaterializer` (`storage/materialize/materializer.py:242`), capability registry, FastAPI service. New runtime modules at HEAD: auto_memory_budget, execution_cohort, feature_block, global_factor_manifest, global_subexpression_index, materialization_tier, resource_broker. |
| 10 | **DA pipeline** (spec §38) | **PRESENT** | `data_access/` | `PITContract`, `MarketCalendar`, `DataSnapshot`, `UniverseSnapshot`, `SemanticFieldCatalog`, `ObjectStore`/`COSObjectStore`. |
| 11 | **QE pipeline** (spec §33) | **PARTIAL** | `quant_evaluator/` | `EvaluationBundle`, `MetricArtifact` family, `MetricSpec`/`MetricRegistry`. **Dual metric catalog** (registry/metrics.py + metrics/catalog.py); `EvaluationArtifact` MISSING; `EvidenceStatus` missing 2 states. |
| 12 | **FP pipeline** (spec §34) | **PARTIAL** | `factor_preprocess/` | Single policy authority `registry/policies.py`; `FittedState`, `FeatureBundle`. `TreatmentRecipe` MISSING; no causality certification module. |
| 13 | **FO pipeline** (spec §35) | **PRESENT** | `factor_optimizer/` | `SearchRunner`, `TrialLedger`, sealed-test, purge+embargo, multi-fidelity. Multiple-testing correction not wired; `TreatmentSelectionArtifact` not emitted. |
| 14 | **FA pipeline** (spec §36) | **PARTIAL** | `factor_assets/` | Identity, admission/novelty, Leiden clustering, `FactorSetArtifact`. Cluster-version/logical-id/incremental-assignment, Factor Library, FeatureSet MISSING. |
| 15 | **Similarity / clustering** (spec §10, §14) | **PARTIAL** | `factor_assets/` | ANN (Faiss/Annoy) + exact + sparse graph + Leiden. No orchestrated incremental pipeline, no fingerprint artifact, no cluster versioning. |
| 16 | **Factor library** (spec §15/§16) | **MISSING** | — | No `FactorLibraryDefinition`/`Version`/`Membership` in FA; platform `cluster_library.py` DRAFT provides `FactorLibraryVersion`/`LibraryMembership`. |
| 17 | **Feature set** (spec §17/§18) | **MISSING** | — | No `FeatureSetArtifact` in FA; platform `feature_set.py` DRAFT provides it + `ModelRetrainRequiredEvent`. |
| 18 | **Report export** (spec §48) | **PARTIAL** | `quant_evaluator/reporting/` | `tear_sheet.py` renders artifacts only (good); `generate_library_report`/`compare_libraries`/`generate_adversarial_report` stubbed `NotImplementedError`. No platform Standards registry. |
| 19 | **Observability** (spec §45) | **PARTIAL** | `factor_engine/telemetry/`, `data_access/telemetry/` | Prometheus exporter + OpenTelemetry bridge in both; FE `/metrics` route. Domain-local only; no platform SLO. |
| 20 | **CI** (spec §52) | **PARTIAL** | `.github/workflows/ci.yml` | gate_runner verifiers (source-authority, supply-chain, evidence-current, submodule-reachability, no-mirror-packages, fresh-wheel-matrix, gate matrix). No ruff/mypy/frontend/migration/OpenAPI jobs. |
| 21 | **Packaging** (spec §0.3) | **PRESENT** | root + each domain pyproject | Multi-package monorepo; each domain pkg own pyproject; `quant_platform` wheel builds (`quant-platform-0.1.0`). `quant_platform` not yet published/installed. |
| 22 | **Model boundary** (spec §2) | **PARTIAL (contract only)** | `quant_platform/app/contracts/backtest.py` | `ModelDatasetRequest`/`ModelTrainingRequest`/`ModelArtifactRef`/`BacktestRequest`/`BacktestArtifactRef` DRAFT. Not in `modeling/`. |
| 23 | **Backtest** (spec §41) | **PRESENT** | `vectorbt_qs/` | `BacktestRequest`/`BacktestArtifact` (`contracts/backtest.py:29/57`); `BacktestProvider` Protocol DRAFT in platform. |
| 24 | **Portfolio optimization** | **PRESENT** | `riskfolio_qs/` | `riskfolio-qs` 0.3.0, depends on `data_access`. |
| 25 | **Modeling** | **PRESENT** | `modeling/` | `FeatureSchema`/`PanelDataset`, `Predictor`, `ModelArtifactCatalog`. |

---

## Summary counts

- **PRESENT:** 9 (contract layer, FE, DA, FO, packaging, backtest, portfolio, modeling, + FE/DA telemetry as observability partial)
- **PARTIAL:** 9 (API, auth, RBAC, QE, FP, FA, similarity/clustering, report export, observability, CI)
- **MISSING:** 7 (Postgres, artifact registry/COS publisher, candidate ingestion+reconciliation, workflow/workers, factor library, feature set, + platform API as part of API partial)

The platform layer (`quant_platform`) is a **contracts-only skeleton**: every runtime capability (API, Postgres, auth enforcement, artifact registry, ingestion, workflow, workers) is MISSING and must be built in Phase 1+ (spec §56).

_End of QRP_CAPABILITY_MATRIX._
