# OWNERSHIP_MATRIX — canonical domain ownership at HEAD 26b4c02

- **HEAD:** `26b4c02c58d70410c670da4d57e346e7d6a41dcc`
- **Date:** 2026-08-27
- **Mode:** READ-ONLY. Working tree is truth. Contract layer being corrected by another agent — referenced by intended shape.
- **Governance base:** `ownership.yaml` (R49) + `scripts/check_no_mirrors.py` (enforces no file duplicated across ownership boundaries).

This is the canonical domain ownership table: each responsibility → single authoritative package, and which packages must NOT own it. It is the reconciliation target for the platform contracts.

---

## 1. Canonical ownership table

| Responsibility | Single authoritative package | Verified evidence | Must NOT own it |
|---|---|---|---|
| Field registry / PIT / snapshot / universe / calendar | `data_access/` | `PITContract` (`contract/runtime_contract.py:45`), `MarketCalendar` (`read/session_calendar.py:340`), `DataSnapshot` (`read/read_contract.py:139`), `UniverseSnapshot` (`r30/universe_snapshot.py:57`), `SemanticFieldCatalog` (`read/semantic_catalog.py:702`) | `quant_platform` (must not be a second storage authority); `research_platform` (legacy `DataSnapshotArtifact` is COMPAT_READ_ONLY) |
| COS ObjectStore / local-disk policy | `data_access/` | `ObjectStore`/`LocalObjectStore`/`COSObjectStore` (`read/object_store.py`), `LocalDiskPolicy` (`read/local_disk_policy.py`) | `quant_platform` (its `storage.py` is a Protocol/adapter boundary only, not an impl) |
| DSL / AST / operator / compile / calc / materialization | `factor_engine/` | `FactorEngine.compile` (`runtime/engine.py:1041`), `ParquetMaterializer` (`storage/materialize/materializer.py:242`), capability registry (`backend/operator_capability.py`), new runtime modules (auto_memory_budget, execution_cohort, feature_block, global_factor_manifest, global_subexpression_index, materialization_tier, resource_broker) | `quant_platform` (must NOT reimplement quant logic / compute RankIC); `research_platform` (legacy `FactorValueArtifact` COMPAT_READ_ONLY) |
| Operator role / mining semantic authority | `factor_engine/` | `mining/operator_catalog.py` + `mining/direct_use.py` (R63 DirectUse verdict = single semantic role authority) | any other package |
| MetricSpec / metric registry / EvaluationBundle | `quant_evaluator/` | `MetricSpec`/`MetricRegistry` (`registry/metrics.py:116/224`), `EvaluationBundle` (`api/requests.py:96`), `MetricArtifact` family (`contracts/metric_artifacts.py`) | `quant_platform` (must NOT recompute RankIC); `research_platform` (legacy `EvaluationBundle` COMPAT_READ_ONLY) |
| Treatment / transform / FittedState / FeatureBundle | `factor_preprocess/` | `FittedState` (`contracts/state.py:114`), `FeatureBundle` (`contracts/feature_bundle.py:390`), single policy authority `registry/policies.py` | `quant_platform`; `research_platform` (legacy `FeatureBundle` COMPAT_READ_ONLY) |
| Trial / TrialLedger / search / sealed test | `factor_optimizer/` | `SearchRunner` (`search/runner.py:881`), `TrialLedger` (`contracts/trial_ledger.py:97`), sealed-test + purge/embargo | `quant_platform`; `research_platform` (legacy `TrialLedger` REUSE_WITH_ADAPTER) |
| Asset identity / admission / novelty / similarity / cluster / library governance | `factor_assets/` | `FactorIdentity` (`identity/canonical.py:61`), `FactorAdmissionArtifact` (`contracts/admission.py`), Leiden clustering (`clustering/families.py:418`), `FactorSetArtifact` (`contracts/factor_set.py:104`) | `quant_platform` (its `cluster_library.py`/`feature_set.py` are DRAFT adapter DTOs, not the FA impl); `research_platform` (legacy `SimilarityArtifact`/`AdmissionDecisionArtifact`/`FactorSetArtifact` COMPAT_READ_ONLY) |
| Model dataset / model training / model artifact | `modeling/` | `FeatureSchema`/`PanelDataset` (`dataset.py`), `Predictor` (`predictor.py`), `ModelArtifactCatalog` (`model_catalog.py`) | `quant_platform` (its `backtest.py` model DTOs are the platform-facing boundary, not the modeling impl); `research_platform` (legacy `ModelDatasetArtifact`/`ModelTrainingArtifact` COMPAT_READ_ONLY) |
| Backtest execution | `vectorbt_qs/` | `BacktestRequest`/`BacktestArtifact` (`contracts/backtest.py:29/57`) | `quant_platform` (its `BacktestProvider` is a Protocol adapter boundary); `research_platform` (legacy `BacktestArtifact` COMPAT_READ_ONLY) |
| Portfolio optimization | `riskfolio_qs/` | `riskfolio-qs` 0.3.0 | `quant_platform` |
| Web / API / auth / RBAC / metadata-projection / ArtifactRef / orchestration / workflow / event / audit | `quant_platform/` | `app/contracts/` (ArtifactRef, EventEnvelope, JobSpec, RBAC, workflow, storage Protocols) | **must NOT reimplement quant logic** — must not compute RankIC, must not materialize factors, must not be a second storage authority |
| CI / gates / evidence / docs / infra | `scripts/`, `evidence/`, `config/`, `.github/` | `gate_runner.py`, `check_no_mirrors.py`, `ci.yml` | domain packages |

---

## 2. Explicit "must NOT own" rules (the anti-patterns the QRP directive forbids)

1. **`quant_platform` must NOT compute RankIC** (spec §0.3). The platform API must not `import quant_evaluator.metrics.xxx` and recompute evaluation metrics. It consumes `EvaluationBundle`/`MetricArtifact` produced by QE via `ArtifactRef`.
2. **`quant_platform` must NOT be a second storage authority.** Its `storage.py` defines `ObjectStore`/`LocalArtifactCache` as **Protocols** (adapter boundary). The real COS/object-store implementation lives in `data_access/read/object_store.py`. The platform must not reimplement COS.
3. **`quant_platform` must NOT reimplement factor materialization.** It orchestrates via `JobSpec`/`WorkflowSpec` and references results via `ArtifactRef`; the actual materialization is `factor_engine`.
4. **`research_platform` must NOT be a second artifact authority.** Its 15 `*Artifact` classes (`artifacts.py`) are legacy; new code must write the canonical owner type and persist via a platform `ArtifactRef`. `research_platform` is COMPAT_READ_ONLY (see `RESEARCH_PLATFORM_MIGRATION_MATRIX.md`).
5. **`factor_assets` must NOT strongly depend on FastAPI / React / Temporal** (spec §0.3). Its adapters to siblings are optional extras only.
6. **`factor_preprocess` must NOT depend on FastAPI; `factor_optimizer` must NOT depend on Temporal** (spec §5).
7. **Domain packages must NOT import `quant_platform`** (spec §6). The platform DTO layer is the platform-facing boundary; domain packages keep their own native contracts.
8. **`quant_evaluator` must NOT have two metric-catalog authorities** (spec §33.2). `registry/metrics.py` and `metrics/catalog.py` must unify to one canonical registry; `metrics/catalog.py` becomes a read-only view of the same instance.
9. **`factor_engine` must NOT be a second PIT/field authority** — it consumes `data_access` PIT contracts via its four-layer gate (`factor_engine/pit_contract.py:79`), it does not redefine them.
10. **`quant_platform` must NOT be a second identity authority** — its `identities.py` DRAFT DTOs must reconcile to `factor_assets/identity/` and `quant_evaluator/contracts/` (marked `[RECONCILE]` in the source).

---

## 3. Reconciliation status (contract layer, being corrected)

The platform DRAFT contracts carry `[RECONCILE]` markers against their domain-native counterparts. Reconciliation targets:

| Platform DRAFT contract | Reconcile against | Status |
|---|---|---|
| `artifact_ref.py` `ArtifactRef` | domain-native refs (`factor_assets/contracts/treatment_selection.py`, `vectorbt_qs/contracts/backtest.py`, `quant_evaluator` series) | DRAFT — pending |
| `identities.py` | `factor_assets/identity/identity.py`, `quant_evaluator/contracts/` | DRAFT — pending |
| `cluster_library.py` | `factor_assets/clustering/` | DRAFT — pending |
| `feature_set.py` | `factor_assets/contracts/factor_set.py` | DRAFT — pending |
| `backtest.py` | `vectorbt_qs/contracts/backtest.py` | DRAFT — pending |
| `timing.py` `EvidenceStatus` | `quant_evaluator/contracts/evidence_status.py` (QE missing 2 states) | DRAFT — pending |
| `lifecycle.py` | `factor_assets/contracts/lifecycle.py` (FA has 6 states vs platform 23) | DRAFT — pending |

All contract docs remain **DRAFT — pending cross-package reconciliation** (NOT V1_FROZEN). See `PLATFORM_CONTRACTS.md`.

_End of OWNERSHIP_MATRIX._
