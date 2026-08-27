# PLATFORM_CONTRACTS — reconciled contract status

**status: DRAFT — pending cross-package reconciliation | NOT V1_FROZEN**

- **HEAD:** `26b4c02c58d70410c670da4d57e346e7d6a41dcc`
- **Date:** 2026-08-27
- **Mode:** READ-ONLY reconciliation deliverable. The contract source files (`quant_platform/app/contracts/*.py`) are being corrected by a separate agent right now; this doc aggregates the *reconciled status* and does not assert final on-disk field lists.
- **This is the reconciliation deliverable that later becomes frozen.** It is explicitly **NOT V1_FROZEN** — every contract below remains DRAFT pending cross-package reconciliation against the domain-native types.

---

## 1. Contract layer scope (spec §6)

`quant_platform/app/contracts/` is the only thin integration DTO layer on the platform side. Responsibilities limited to:
- external API DTOs;
- workflow requests;
- `ArtifactRef`;
- `EventEnvelope`;
- domain adapter protocols (`WorkflowBackend`, `ObjectStore`, `BacktestProvider`, …).

It does **not** replace domain-native artifacts (`TreatmentSelectionArtifact` in `factor_assets`, `EvaluationBundle` in `quant_evaluator`, `BacktestArtifact` in `vectorbt_qs`, …). The worker adapter layer translates `Platform DTO ↔ Domain Native Contract`.

**PURE-DTO rule:** stdlib dataclasses only (frozen) + `typing.Protocol`; NO fastapi/sqlalchemy/pydantic/third-party imports. Verified: `quant_platform.app.contracts` imports cleanly with `__all__` length 107; `tests/test_smoke.py` asserts no third-party modules load.

---

## 2. Contract inventory and reconciliation status

| Contract module | Spec anchor | Status | Reconcile against | Reconciliation notes |
|---|---|---|---|---|
| `artifact_ref.py` — `ArtifactRef` + `ARTIFACT_TYPES` (15 types) | §7.2 | **DRAFT** | domain-native refs | `ArtifactRef` is a REFERENCE, not a validator; object-bytes hash verified at storage boundary, not by this DTO. `[RECONCILE]` vs `factor_assets/contracts/treatment_selection.py`, `vectorbt_qs/contracts/backtest.py`. |
| `event_envelope.py` — `EventEnvelope` + 19 `EVENT_TYPE_*` | §11 | **DRAFT** | — | Deep-frozen payload. Transactional outbox impl MISSING (contract only). |
| `jobs.py` — `JobSpec`/`JobRecord`/`JobAttempt`/`JobResult` + `JobStatus`/`ErrorClass` + idempotency keys | §9/§12.2/§42/§43 | **DRAFT** | FE `service/jobstore.py` | Job layer split: `JobSpec` (start, no outputs) vs `JobResult` (terminal, carries `output_artifact_refs`). `JobStatus` distinct from `LifecycleState`/`HealthState`. |
| `workflow.py` — `WorkflowBackend` Protocol + `WorkflowSpec`/`WorkflowRun`/`WorkflowStatus` | §12 | **DRAFT** | — | Workflow vs Job two layers. No Temporal impl. |
| `candidate.py` — `FactorCandidateManifest` + `_READY` | §10.1/§10.2 | **DRAFT** | — | Manifest fields match spec §10.2. No ingestion service/scanner. |
| `lifecycle.py` — `LifecycleState` (23) + `HealthState` (6) | §9 | **DRAFT** | `factor_assets/contracts/lifecycle.py` (6 states) | Platform enum has the full 23-state machine; FA has only 6. Reconciliation: FA must extend to match, or platform enum is the target. |
| `identities.py` — `Identity` + `FactorDefinitionIdentity`/`FactorValueIdentity`/`EvaluationIdentity`/`TreatmentIdentity` | §8.1–§8.4 | **DRAFT** | `factor_assets/identity/identity.py`, `quant_evaluator/contracts/` | `[RECONCILE]` markers in source. `EvaluationIdentity` provided here (missing in FA). |
| `cluster_library.py` — `SimilarityGraphVersion`/`ClusterSetVersion`/`LogicalCluster`/`ClusterVersion`/`ClusterMembership`/`ClusterLineageEdge`/`FactorLibraryVersion`/`LibraryMembership` | §8.5/§8.6/§14/§15/§16 | **DRAFT** | `factor_assets/clustering/` | Layered model: library binds to a global `ClusterSetVersion`, NOT a single cluster version (correction applied). |
| `feature_set.py` — `FeatureSetArtifact`/`FeatureSetVersion`/`FeatureMemberRef` + `FeatureSetDiffCategory` + `ModelRetrainRequiredEvent` | §17/§18 | **DRAFT** | `factor_assets/contracts/factor_set.py` | `content_hash` IS verifiable here (semantic hash over local fields), unlike `ArtifactRef`'s object-bytes hash. |
| `rbac.py` — `Permission`/`Role`/`Team`/`SecurityClassification`/`HumanPrincipal`/`WorkloadPrincipal`/`ResourceScope` + `ROLE_PERMISSIONS` | §24.1 | **DRAFT** | — | Capability-based, not page-based. Authorization = Role × Team × ResourceScope × SecurityClassification. |
| `storage.py` — `ObjectStore`/`LocalArtifactCache` Protocols + `CacheEvictionPolicy` | §22 | **DRAFT** | `data_access/read/object_store.py` | Protocols only; real impl in `data_access`. |
| `timing.py` — `TimingContract` (5 fields) + `EvidenceStatus` (4 states incl. `INVALID_EVIDENCE`/`LABEL_NOT_MATURE`) | §39/§40 | **DRAFT** | `quant_evaluator/contracts/evidence_status.py` | QE `EvidenceStatus` missing the 2 states; platform provides them. |
| `backtest.py` — `BacktestProvider` Protocol + `BacktestRequest`/`BacktestArtifactRef`/`ModelDatasetRequest`/`ModelTrainingRequest`/`ModelArtifactRef` | §41/§2 | **DRAFT** | `vectorbt_qs/contracts/backtest.py`, `modeling/` | Spec §2 exact model contract names provided here (NOT in `modeling/`). |
| `_contenthash.py` — canonical content-hash codec | §2 | **DRAFT** | — | sha256 over sorted length-prefixed semantic fields; rejects NaN/Inf/naive-datetime; no `str()` fallback. |

---

## 3. Reconciliation decisions required before freeze

| # | Decision | Packages involved | Current state |
|---|---|---|---|
| R1 | `EvaluationIdentity` ownership | `quant_platform/identities.py` vs `factor_assets/identity/` | Platform DRAFT provides it; FA missing. Decide: FA implements natively, platform DTO reconciles. |
| R2 | Lifecycle state machine | `quant_platform/lifecycle.py` (23 states) vs `factor_assets/contracts/lifecycle.py` (6 states) | Divergent. Decide canonical state set. |
| R3 | `EvidenceStatus` states | `quant_platform/timing.py` (4 states) vs `quant_evaluator/contracts/evidence_status.py` (6 states, missing 2) | QE must add `INVALID_EVIDENCE`/`LABEL_NOT_MATURE`. |
| R4 | Metric registry single authority | `quant_evaluator/registry/metrics.py` vs `quant_evaluator/metrics/catalog.py` | Dual catalog must unify (spec §33.2). |
| R5 | Model contract names | `quant_platform/backtest.py` vs `modeling/` | Spec §2 names provided in platform DTO; decide if modeling exposes natively. |
| R6 | Cluster/library layered model | `quant_platform/cluster_library.py` vs `factor_assets/clustering/` | Platform DRAFT has the layered model; FA must implement. |
| R7 | `ArtifactRef` vs domain-native refs | `quant_platform/artifact_ref.py` vs `factor_assets`/`vectorbt_qs`/`quant_evaluator` | Reconcile ref semantics (semantic_hash vs content_hash). |
| R8 | `TreatmentRecipe` | `factor_preprocess`/`factor_optimizer`/`factor_assets` | MISSING everywhere; must be added before treatment contracts freeze. |

---

## 4. Freeze gate

Per spec §56 Phase 0, the contract set freezes only after:
1. All `[RECONCILE]` markers resolved against domain-native types.
2. The dual QE metric catalog unified.
3. `TreatmentRecipe` defined.
4. Cross-package contract review (spec §57) passes — reviewer must read code + tests, not just markdown.

Until then, **all contract docs remain DRAFT — pending cross-package reconciliation** (NOT V1_FROZEN).

_End of PLATFORM_CONTRACTS._
