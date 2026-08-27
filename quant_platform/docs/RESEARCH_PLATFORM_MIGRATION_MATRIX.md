# Research Platform Migration Matrix (QRP-P0R-C3)

**Status:** DOCUMENTATION ONLY — no source changes made.
**Date:** 2026-08-27
**Scope:** `quant_projects/research_platform/` → canonical domain owners.

---

## 1. Mandate (from the QRP spec)

`research_platform/` is a legacy, self-contained "pure-stdlib research
governance / artifact layer" that **overlaps heavily with the canonical domain
packages**. Per the QRP platform contract, it:

- MUST **NOT** remain as a parallel artifact authority.
- MUST **NOT** be deleted (it holds valuable lineage / invalidation / firewall /
  health / scorecard patterns and legacy read-only history).
- MUST be superseded: **new code is FORBIDDEN from writing the artifact types it
  defines**. New code must write the canonical domain artifact via the owning
  package, and reference it through the platform's `ArtifactRef`.

### 1.1 The single-authority rule

The platform layer (`quant_platform`) holds **only** `ArtifactRef` (an immutable
reference DTO: `artifact_id`, `artifact_type`, `schema_version`, `content_hash`,
`storage_uri`, `size_bytes`, `created_at`, `producer_type`, `producer_version`,
`snapshot_id`). It **NEVER redefines domain artifacts**. The set of accepted
`artifact_type` strings is a closed `ARTIFACT_TYPES` frozenset in
`quant_platform/app/contracts/artifact_ref.py`, and `ArtifactRef.__post_init__`
rejects any type outside it.

Canonical domain types live in the owning packages listed in §2 and §3. The
platform ref is the *adapter boundary only*; no domain package imports
`quant_platform`.

---

## 2. Canonical owner map (authorities per the QRP spec)

| Domain type family | Canonical owner package | Confirmed evidence |
|---|---|---|
| DataSnapshot / Universe / field registry | `data_access/` | `data_access/read/read_contract.py:139` (`class DataSnapshot`); `data_access/registry/` (field/schema registry, `schema_validation.py`, `dataset_boundary.py`). |
| FactorDefinition / FactorValue / raw materialization | `factor_engine/` + `factor_assets/` identity | `factor_engine/api/` (`factor.py`, `operator_registry.py`, `dsl_parser.py`, `source_ref.py`); `factor_assets/identity/` (`canonical.py`, `adapters.py`); `factor_assets/contracts/lineage.py`. |
| EvaluationBundle / MetricArtifact / metric registry | `quant_evaluator/` | `quant_evaluator/contracts/metric_artifacts.py` (typed `MetricArtifact`); `quant_evaluator/registry/`; `quant_evaluator/contracts/evidence_status.py`. |
| Treatment / FittedState / FeatureBundle / FeatureManifest | `factor_preprocess/` + `factor_assets/` | `factor_preprocess/factor_preprocess/contracts/feature_bundle.py`, `contracts/state.py` (FittedState), `schemas/feature_manifest.schema.json`; `factor_assets/contracts/treatment_selection.py` (`TreatmentSelectionArtifact`). |
| Trial / SearchSession / SearchBudget / SplitPlan | `factor_optimizer/` | `factor_optimizer/factor_optimizer/contracts/trial.py`, `trial_ledger.py`, `search_budget.py`, `splits.py`, `objective.py`. |
| Similarity / Admission / FactorSet / Cluster / Library / Asset lifecycle | `factor_assets/` | `factor_assets/contracts/similarity.py`, `admission.py`, `factor_set.py`, `cluster_library` (platform ref type), `lifecycle/state_machine.py`, `registry/`; `factor_assets/campaigns/` (`campaign_coordinator.py`, `ledger_adapter.py`). |
| ModelDataset / Model | `modeling/` | `modeling/dataset.py` (`PanelDataset`, `FeatureSchema`), `modeling/predictor.py` (`Predictor`), `modeling/model_catalog.py` (`ModelArtifactCatalog`). |
| Backtest | backtest engine `vectorbt_qs/` | `vectorbt_qs/contracts/backtest.py`, `mvp/engine/`; platform `quant_platform/app/contracts/backtest.py` (`BacktestRequest` / `BacktestArtifactRef`). |

**Ownership governance:** `quant_projects/ownership.yaml` already lists every
top-level path (factor_engine, data_access, modeling, quant_evaluator,
factor_optimizer, factor_assets, factor_preprocess, vectorbt_qs) with a single
owning team, and `scripts/check_no_mirrors.py` enforces that no file is
duplicated across ownership boundaries. `research_platform` is owned by
`research_team` for legacy read-only workflows only.

---

## 3. Migration matrix — by `research_platform` module

### 3.1 `artifacts.py` — unified artifact model

Base `Artifact` (identity + provenance + SHA-256 `content_hash` + JSON
round-trip) is superseded by the platform `ArtifactRef` (identity +
`content_hash`) **plus** the canonical domain type. The `content_hash`
semantic-only-hash idea is valuable and is already reproduced by the platform's
content-hash rule (`quant_platform/app/contracts/_contenthash.py`).

| research_platform type | Canonical owner | Disposition |
|---|---|---|
| `Artifact` (base) | `quant_platform/app/contracts/artifact_ref.py` (`ArtifactRef`) | **MIGRATE** — platform ref supersedes it as the reference layer. |
| `DataSnapshotArtifact` | `data_access/read/read_contract.py` (`DataSnapshot`) | **COMPAT_READ_ONLY** |
| `FactorDefinitionArtifact` | `factor_engine/` (`factor.py`) + `factor_assets/identity/` | **COMPAT_READ_ONLY** |
| `FactorValueArtifact` | `factor_engine/` materialization + `factor_assets/identity/` (`FactorValueIdentity`) | **COMPAT_READ_ONLY** |
| `EvaluationBundle` | `quant_evaluator/contracts/metric_artifacts.py` (`MetricArtifact`) | **COMPAT_READ_ONLY** |
| `SimilarityArtifact` | `factor_assets/contracts/similarity.py` | **COMPAT_READ_ONLY** |
| `AdmissionDecisionArtifact` | `factor_assets/contracts/admission.py` (`FactorAdmissionArtifact` / `AdmissionDecision`) | **COMPAT_READ_ONLY** |
| `FactorSetArtifact` | `factor_assets/contracts/factor_set.py` | **COMPAT_READ_ONLY** |
| `FeatureBundle` | `factor_preprocess/.../contracts/feature_bundle.py` | **COMPAT_READ_ONLY** |
| `ModelDatasetArtifact` | `modeling/dataset.py` (`PanelDataset`) | **COMPAT_READ_ONLY** |
| `ModelTrainingArtifact` | `modeling/` (`Predictor`, `ModelArtifactCatalog`) | **COMPAT_READ_ONLY** |
| `BacktestArtifact` | `vectorbt_qs/` + `quant_platform/.../backtest.py` (`BacktestArtifactRef`) | **COMPAT_READ_ONLY** |
| `PortfolioArtifact` / `RiskSnapshotArtifact` / `ExpectedReturnArtifact` | `modeling/` + risk/backtest layer | **DEPRECATE** — no canonical counterpart yet; read-only legacy. |

**FORBIDDEN:** new code must not import or instantiate these research_platform
artifact classes to *write* artifacts. It must produce the canonical owner type
and persist via a platform `ArtifactRef`.

### 3.2 `graph.py` — lineage & invalidation

`ArtifactGraph` + `DataInvalidationGraph` are pure in-memory DAGs. They are the
single most valuable pattern in the package.

| research_platform type | Canonical owner | Disposition |
|---|---|---|
| `ArtifactGraph` (lineage) | PostgreSQL `artifact_lineage` table (long-term target) | **EXTEND_EXISTING** → migrate to persistent lineage (see §5). |
| `DataInvalidationGraph` | `EvidenceInvalidationService` (see §5) | **EXTEND_EXISTING** → port the reverse-lookup invalidation into the canonical invalidation service. |

**FORBIDDEN:** new code must not use in-memory `research_platform.graph`
graphs as the source of truth for lineage or invalidation; use the persistent
lineage store / invalidation service.

### 3.3 `campaign.py` — campaign bookkeeping & multiple-testing ledger

`ResearchCampaignArtifact` + `TrialLedger` duplicate
`factor_optimizer/contracts/trial_ledger.py` and
`factor_assets/campaigns/ledger_adapter.py`.

| research_platform type | Canonical owner | Disposition |
|---|---|---|
| `ResearchCampaignArtifact` | `factor_assets/campaigns/` (`campaign_coordinator.py`) | **EXTEND_EXISTING** |
| `TrialLedger` | `factor_optimizer/contracts/trial_ledger.py` | **REUSE_WITH_ADAPTER** — map trial counts into the canonical ledger. |

### 3.4 `health.py` — factor health state machine

`FactorHealthState` enum + `FactorHealthMonitor` (IC-decay / drift / stale /
revive / failure thresholds). The canonical lifecycle/health state is
`factor_assets/lifecycle/state_machine.py` (`LifecycleState`, `StateEvent`).

| research_platform type | Canonical owner | Disposition |
|---|---|---|
| `FactorHealthState` / `FactorHealthMonitor` | `factor_assets/lifecycle/state_machine.py` + `factor_assets/contracts/lifecycle.py` | **EXTEND_EXISTING** — fold the IC-threshold heuristics into the lifecycle state machine as health inputs. Platform emits `EVENT_TYPE_FACTOR_HEALTH_CHANGED` (`quant_platform/app/contracts/event_envelope.py`). |

### 3.5 `firewall.py` — candidate validation chain

`FactorCandidateArtifact` (provenance-only) + `AlphaGenerationFirewall`
(10-stage chain). Several stages are **documented heuristics** (see README
"Scope notes"), which is exactly the gap the canonical `CandidatePreflightService`
fills with real FE/DA checks.

| research_platform type | Canonical owner | Disposition |
|---|---|---|
| `FactorCandidateArtifact` | `quant_platform/app/contracts/candidate.py` (`FactorCandidateManifest`) | **REUSE_AS_IS** (as candidate manifest / ready-marker) |
| `AlphaGenerationFirewall` (stages) | `CandidatePreflightService` (real FE/DA checks) | **EXTEND_EXISTING** — see §5. |
| — firewall heuristic stages | → replace with real checks; heuristic fallback labeled `RESEARCH_ONLY` | **MIGRATE** |

**FORBIDDEN:** new code must not rely on the heuristic stages
(`pit_future_leakage_static`, `type_dimension`, `cheap_fingerprint_novelty`,
`fe_compilation` defaulted-to-True) as production gates; they are `RESEARCH_ONLY`.

### 3.6 `scorecard.py` — generator scorecard

`GeneratorScorecard.record(...)` / `.summary()` aggregates generator pass /
duplicate / novelty / IC / cost / diversity. This is read/analytics concern.

| research_platform type | Canonical owner | Disposition |
|---|---|---|
| `GeneratorScorecard` | Generator/Campaign **Analytics Read Model** (see §5) | **MIGRATE** — move the aggregation into a read model over canonical event streams (`quant_platform/app/contracts/event_envelope.py`). |

---

## 4. What new code must do (forbidden list)

1. **FORBIDDEN** to import `research_platform.artifacts.*` and instantiate
   `DataSnapshotArtifact`, `FactorValueArtifact`, `EvaluationBundle`,
   `FactorSetArtifact`, `SimilarityArtifact`, `AdmissionDecisionArtifact`,
   `FeatureBundle`, `ModelDatasetArtifact`, `ModelTrainingArtifact`,
   `BacktestArtifact`, `PortfolioArtifact`, `RiskSnapshotArtifact`,
   `ExpectedReturnArtifact` to *write* artifacts.
2. **FORBIDDEN** to use `research_platform.graph.ArtifactGraph` /
   `DataInvalidationGraph` as the lineage/invalidation source of truth.
3. **FORBIDDEN** to use `research_platform.firewall` heuristic stages as
   production gates.
4. **FORBIDDEN** to add a new parallel artifact definition anywhere outside the
   owning domain package. New artifacts are created by the canonical owner and
   referenced by `quant_platform` `ArtifactRef`.

New code path: **canonical owner produces domain type → platform persists an
`ArtifactRef` (with derived `content_hash`) → events via
`event_envelope.py` → lineage rows in `artifact_lineage` → invalidation via
`EvidenceInvalidationService`.**

---

## 5. Valuable patterns to PRESERVE and their long-term home

| research_platform pattern | Preserve as | Long-term home |
|---|---|---|
| Lineage DAG (`ArtifactGraph`) | Persistent lineage edges (source → dependent, snapshot index) | PostgreSQL **`artifact_lineage`** table |
| Reverse-lookup invalidation (`DataInvalidationGraph`) | Snapshot-replacement → affected-artifact recompute queries | **`EvidenceInvalidationService`** |
| Firewall heuristics (10-stage chain) | Real FE/DA preflight; heuristics kept but clearly labeled | **`CandidatePreflightService`** (real `factor_engine` compile + `data_access` field/PIT checks; heuristic fallback tagged `RESEARCH_ONLY`) |
| Generator scorecard aggregation | Event-driven analytics over canonical event stream | Generator/Campaign **Analytics Read Model** |

---

## 6. Ownership summary table

| research_platform module | Owner package | Disposition |
|---|---|---|
| `artifacts.py` | `quant_platform` (`ArtifactRef`) + domain owners | **MIGRATE** / **COMPAT_READ_ONLY** |
| `graph.py` | PostgreSQL `artifact_lineage` + `EvidenceInvalidationService` | **EXTEND_EXISTING** |
| `campaign.py` | `factor_assets/campaigns/` + `factor_optimizer/contracts/trial_ledger.py` | **EXTEND_EXISTING** / **REUSE_WITH_ADAPTER** |
| `health.py` | `factor_assets/lifecycle/state_machine.py` | **EXTEND_EXISTING** |
| `firewall.py` | `CandidatePreflightService` (+ `quant_platform` candidate manifest) | **EXTEND_EXISTING** / **REUSE_AS_IS** |
| `scorecard.py` | Generator/Campaign Analytics Read Model | **MIGRATE** |

`research_platform/` remains on disk as a **read-only legacy / compatibility**
module owned by `research_team` (`ownership.yaml`); it is not an authority, and
no new writes target its types.
