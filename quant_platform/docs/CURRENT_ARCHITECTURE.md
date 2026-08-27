# CURRENT_ARCHITECTURE — Quant Research Platform repo at HEAD 26b4c02

- **HEAD:** `26b4c02c58d70410c670da4d57e346e7d6a41dcc` (R52: add auto memory budget and execution cohort modules)
- **Date:** 2026-08-27
- **Mode:** READ-ONLY delta re-audit. No source modified. Working tree is truth.
- **Delta vs prior audits:** the 5 audits in `quant_platform/docs/audit/` were stamped `d58eae0821da2c71182e200751af97a577e1a616`. This doc re-validates every claim at HEAD and records what changed.
- **Contract layer note:** `quant_platform/app/contracts/*` is being corrected by a separate agent right now (11 files show working-tree modifications vs HEAD). This doc describes the contract layer by its *intended post-correction shape* and does not assert final on-disk field lists. All contract docs remain **DRAFT — pending cross-package reconciliation** (NOT V1_FROZEN).

---

## 1. Repo topology at HEAD

Multi-package monorepo. Each domain package has its own `pyproject.toml`; the root `pyproject.toml` (`quant-projects` 0.4.0) is an umbrella wheel that also ships `factor_engine.*`, `data_access.*`, `vectorbt_qs.*`, `modeling*`, `evidence*` by source path. The separately-built domain packages (`factor_preprocess`, `factor_optimizer`, `factor_assets`, `quant_evaluator`, `riskfolio_qs`) are NOT in the root find-include — they are independent distributions.

| Top-level package | Type | Authoritative responsibility (verified) |
|---|---|---|
| `data_access/` | domain (own pyproject, `data-access` 0.10.2) | Field registry, PIT semantics, calendar, snapshot, universe, COS ObjectStore, local-disk policy. Single `DataAccessStore` (`data_access/store.py:234`); `get_store()` installs COS runtime (`data_access/__init__.py:58`). |
| `factor_engine/` | domain (own pyproject, `factor-engine` 0.3.1) | DSL/AST/operator/compile/calc/materialization. `FactorEngine.compile` (`runtime/engine.py:1041`), `ParquetMaterializer` (`storage/materialize/materializer.py:242`), capability registry (`backend/operator_capability.py`), FastAPI service (`service/app.py`, port **8088**). |
| `quant_evaluator/` | domain (own pyproject, `quant_evaluator` 0.0.1a1) | MetricSpec/registry/EvaluationBundle. `MetricSpec`/`MetricRegistry` (`registry/metrics.py:116/224`), `EvaluationBundle` (`api/requests.py:96`), `MetricArtifact` family (`contracts/metric_artifacts.py`). |
| `factor_preprocess/` | domain (own pyproject, `factor-preprocess` 0.1.0) | Treatment/transform/FittedState/FeatureBundle. Single policy authority `registry/policies.py`; `FittedState` (`contracts/state.py:114`), `FeatureBundle` (`contracts/feature_bundle.py:390`). |
| `factor_optimizer/` | domain (own pyproject, `factor-optimizer` 0.1.0, hatchling) | Trial/TrialLedger/search. `SearchRunner` (`search/runner.py:881`), `TrialLedger` (`contracts/trial_ledger.py:97`), sealed-test + purge/embargo isolation. |
| `factor_assets/` | domain (own pyproject, `factor_assets` 0.1.0) | Asset identity/admission/novelty/similarity/cluster/library governance. `FactorIdentity` (`identity/canonical.py:61`), `FactorAdmissionArtifact` (`contracts/admission.py`), Leiden clustering (`clustering/families.py:418`), `FactorSetArtifact` (`contracts/factor_set.py:104`). |
| `modeling/` | root umbrella | Alpha prediction/training/evaluation. `FeatureSchema`/`PanelDataset` (`dataset.py`), `Predictor` (`predictor.py`), `ModelArtifactCatalog` (`model_catalog.py`). |
| `vectorbt_qs/` | root umbrella | vectorbt-based backtesting. `BacktestRequest`/`BacktestArtifact` (`contracts/backtest.py:29/57`). |
| `riskfolio_qs/` | domain (own pyproject, `riskfolio-qs` 0.3.0) | Portfolio optimization (depends on `data_access`). |
| `research_platform/` | legacy (no own pyproject) | **LEGACY dual artifact authority** — 15 `*Artifact` classes (`artifacts.py`), lineage graph, campaign ledger, health, firewall, scorecard. Must migrate (see GAP_ANALYSIS §RESEARCH_PLATFORM_MIGRATION_MATRIX). |
| `lightgbm_qs/` | root umbrella | LightGBM training scripts + snapshots. |
| `quant_platform/` | platform (own pyproject, `quant-platform` 0.1.0, **untracked**) | Web/API/auth/RBAC/metadata-projection/ArtifactRef/orchestration/workflow/event/audit — **NOT reimplementing quant logic**. Currently only the thin integration DTO layer (`app/contracts/`) + docs + smoke test. |
| `quant_platform/app/contracts/` | platform DTO layer | Pure-stdlib frozen dataclasses + `typing.Protocol` (spec §6). ArtifactRef, EventEnvelope, JobSpec/JobRecord/JobResult, FactorCandidateManifest, LifecycleState/HealthState, identities, cluster/library, feature_set, RBAC, storage (ObjectStore/LocalArtifactCache), timing, workflow, backtest/model DTOs. |

Supporting: `scripts/` (CI gates, `gate_runner.py`, `check_no_mirrors.py`), `tests/` (root suites), `evidence/`, `config/` (gates.json), `ownership.yaml`, `.github/workflows/ci.yml`, `jobs/`, `runs/`, `data/`, `weekly_backtest_output/`, `量化研究与生产报告中心/`.

---

## 2. The three planes (spec §3) at HEAD

| Plane | Current state | Owning package |
|---|---|---|
| **Control Plane** (when/who/where/retry/promotion/deps/state machine) | **MISSING as a platform plane.** No Temporal, no workflow runtime, no outbox worker. FE has an in-memory `BoundedJobQueue` (`factor_engine/service/queue.py:96`) + SQLite `JobStore` (`service/jobstore.py:173`); DA has a FastAPI service. `quant_platform/app/contracts/workflow.py` defines a `WorkflowBackend` Protocol (DRAFT) but no implementation. | platform (to build) |
| **Metadata Plane** (factor catalog, versions, cluster, library, artifact location, health, job, audit) | **MISSING as a platform plane.** No PostgreSQL, no metadata registry. FE has its own SQLite `FactorCatalog` (`factor_engine/storage/catalog.py:544`); FA has `AssetRepository`/`SQLiteLifecycleRepository` (`factor_assets/registry/`). These are domain-local, not a unified platform metadata registry. | platform (to build) |
| **Data Plane** (factor values, large series, feature matrices, backtest series, model binaries, logs, chart data) | **PARTIAL.** Factor values are materialized **local-only** to the factor lake (`factor_engine/util/workspace_paths.py:69`); COS holds clean-dataset mirrors (`data_access/cos/mirror.py:34`), NOT the spec §21 canonical `factor_values/` layout. `data_access/read/object_store.py` provides `ObjectStore`/`LocalObjectStore`/`COSObjectStore`. | data_access + factor_engine (extend) |

---

## 3. Domain-package delta audit (d58eae → HEAD 26b4c02)

Only `factor_engine`, `factor_preprocess`, `factor_optimizer`, `factor_assets` changed between the two HEADs (plus `quant_platform` itself and `lightgbm_qs`). `data_access`, `quant_evaluator`, `modeling`, `vectorbt_qs`, `riskfolio_qs`, `research_platform` are **unchanged** — their old-audit findings still hold verbatim.

### 3.1 factor_engine — 38 files changed, +3783/−258 (biggest delta)
New runtime modules added at HEAD (all verified present):
- `runtime/auto_memory_budget.py` — startup + resample memory budget authority.
- `runtime/execution_cohort.py` — execution cohort grouping + dynamic cohort size + cohort executor.
- `runtime/feature_block.py` — FeatureBlock (100k factors ≠ 100k files).
- `runtime/global_factor_manifest.py` — lightweight fingerprint registry (anti-mega-DAG).
- `runtime/global_subexpression_index.py` — global shared-subexpression index (cross-cohort CSE).
- `runtime/materialization_tier.py` — not all factors materialize full history.
- `runtime/resource_broker.py` — live headroom + CPU/RAM/IO/spill token admission.

Mining role authority hardened: `mining/operator_catalog.py` + `mining/direct_use.py` now treat the DirectUse verdict as the single semantic authority for operator role (R63): `DIRECT_RECIPE`/`DIRECT_INTERMEDIATE`/`DIRECT_CONTROL_FLOW` are `RECIPE_INTERNAL` (never ALPHA); `MOVE_INTERNAL`/`RESEARCH_TOOL` are explicit non-alpha; full-history-replay indicators (KAMA/DEMA/TEMA/PSAR/Supertrend) stay `ALPHA_HIGH_COST`. This is a **role-authority clarification, not a new authority** — the old audit's "single authority" finding is strengthened, not contradicted.

**Old-audit finding that CHANGED:** the old audit said FE service default port is 8088 (not 8766). **Still holds** — `service/app.py:1521` reads `FACTOR_ENGINE_SERVICE_PORT` default `8088`. The root `Dockerfile` CMD uses 8766, which is a **mismatch** (Dockerfile exposes 8766 but the service defaults to 8088). Flagged as a real drift.

### 3.2 factor_preprocess — 2 files changed
- `transforms/smoothing.py` — KAMA gained a `use_current` parameter (default `False`, preserving the strict causal `shift(1)` contract; `True` only when an external clock like `DecisionClock`/`available_at` has already decided usability). Causal contract preserved.
- `tests/transforms/test_smoothing_kama_oracle.py` — new KAMA oracle test.
- **Old-audit finding unchanged:** single policy authority `registry/policies.py`; `contracts/policy.py` is deprecated compat. `TreatmentRecipe` still MISSING.

### 3.3 factor_optimizer — 2 files changed
- `search/winner_selector.py` — added `_validate_desirabilities_and_scores`: fail-closed validation that every dimension desirability and robustness/complexity score is finite AND in `[0,1]`; NaN/Inf/out-of-range raise `ValueError` (never clamped). Hardens the winner selector against poisoned inputs.
- `tests/search/test_winner_selector.py` — new tests.
- **Old-audit finding unchanged:** sealed test, purge+embargo, trial ledger, multi-fidelity all present. `TreatmentSelectionArtifact` still lives in `factor_assets` and is not emitted by the FO runner. Multiple-testing correction still not wired in FO.

### 3.4 factor_assets — 2 files changed
- `contracts/treatment_selection.py` — added recursive deep-freeze (`_freeze`) and canonical structural hash (`_canonical`) so nested `winner_recipe` mappings are provably immutable and structurally-equal recipes hash identically. Strengthens the derived-only content_hash contract.
- `tests/contracts/test_treatment_selection.py` — new tests.
- **Old-audit finding unchanged:** `EvaluationIdentity` still missing in FA (now provided by `quant_platform/app/contracts/identities.py` as a DRAFT platform DTO); cluster-version/logical-id/incremental-assignment layer still missing in FA (now provided as DRAFT platform contracts in `cluster_library.py`); Factor Library / FeatureSet / SimilarityFingerprint still missing in FA.

### 3.5 quant_platform — created at HEAD (R52)
The entire `quant_platform/` tree was added at HEAD. It contains:
- `app/contracts/` — 15 pure-stdlib DTO modules (see §4).
- `docs/PLATFORM_CONTRACTS_DRAFT.md`, `docs/audit/*` (5 audits), `docs/RESEARCH_PLATFORM_MIGRATION_MATRIX.md` (untracked).
- `pyproject.toml` (untracked) — `quant-platform` 0.1.0, zero runtime deps, PURE-DTO rule.
- `tests/test_smoke.py` (untracked) — verifies contracts import, pure-stdlib, no `platform` shadowing.

**Verified:** `quant_platform` wheel **builds successfully** (`python -m build --wheel` → `quant_platform-0.1.0-py3-none-any.whl`), and `quant_platform.app.contracts` imports cleanly in the repo `.venv` (`__all__` length 107). So the "independently installable wheel" claim is **PRESENT** (pyproject + build verified), though the wheel is not yet published/installed anywhere.

---

## 4. Contract layer (being corrected) — intended shape

`quant_platform/app/contracts/` is the only thin integration DTO layer (spec §6). It is **DRAFT — pending cross-package reconciliation**. The 15 modules and their spec anchors:

| Module | Spec | Intended responsibility |
|---|---|---|
| `artifact_ref.py` | §7.2 | `ArtifactRef` (immutable reference DTO) + closed `ARTIFACT_TYPES` frozenset (15 types). |
| `event_envelope.py` | §11 | `EventEnvelope` + 19 `EVENT_TYPE_*` constants; deep-frozen payload. |
| `jobs.py` | §9/§12.2/§42/§43 | `JobSpec`/`JobRecord`/`JobAttempt`/`JobResult` + `JobStatus`/`ErrorClass` + idempotency-key helpers. |
| `workflow.py` | §12 | `WorkflowBackend` Protocol + `WorkflowSpec`/`WorkflowRun`/`WorkflowStatus`. |
| `candidate.py` | §10.1/§10.2 | `FactorCandidateManifest` + `_READY` marker protocol. |
| `lifecycle.py` | §9 | `LifecycleState` (23 states) + `HealthState` (6 states) — three independent enums (Lifecycle ≠ JobStatus ≠ Health). |
| `identities.py` | §8.1–§8.4 | `Identity` base + `FactorDefinitionIdentity`/`FactorValueIdentity`/`EvaluationIdentity`/`TreatmentIdentity` (canonicalize-then-hash). |
| `cluster_library.py` | §8.5/§8.6/§14/§15/§16 | `SimilarityGraphVersion`/`ClusterSetVersion`/`LogicalCluster`/`ClusterVersion`/`ClusterMembership`/`ClusterLineageEdge`/`FactorLibraryVersion`/`LibraryMembership`. |
| `feature_set.py` | §17/§18 | `FeatureSetArtifact`/`FeatureSetVersion`/`FeatureMemberRef` + `FeatureSetDiffCategory` + `ModelRetrainRequiredEvent`. |
| `rbac.py` | §24.1 | `Permission`/`Role`/`Team`/`SecurityClassification`/`HumanPrincipal`/`WorkloadPrincipal`/`ResourceScope` + `ROLE_PERMISSIONS`. |
| `storage.py` | §22 | `ObjectStore`/`LocalArtifactCache` Protocols + `CacheEvictionPolicy`. |
| `timing.py` | §39/§40 | `TimingContract` (5 timing fields) + `EvidenceStatus` (incl. `INVALID_EVIDENCE`/`LABEL_NOT_MATURE`). |
| `backtest.py` | §41/§2 | `BacktestProvider` Protocol + `BacktestRequest`/`BacktestArtifactRef`/`ModelDatasetRequest`/`ModelTrainingRequest`/`ModelArtifactRef`. |
| `_contenthash.py` | §2 | Canonical content-hash codec (sha256 over sorted length-prefixed semantic fields; rejects NaN/Inf/naive-datetime; no `str()` fallback). |

**Key reconciliation note:** the spec §2 exact contract names `ModelDatasetRequest`/`ModelTrainingRequest`/`ModelArtifactRef` are **NOT in `modeling/`** — they live in `quant_platform/app/contracts/backtest.py` (DRAFT). The old audit flagged these as MISSING in modeling; the platform DTO layer now provides them as the platform-facing boundary. Whether modeling should also expose them natively is a reconciliation decision (see GAP_ANALYSIS).

---

## 5. What the platform layer does NOT yet have (verified)

`quant_platform/` currently contains **only** `app/contracts/` + `docs/` + `pyproject.toml` + `tests/test_smoke.py`. There is **no**:
- `app/api/` (no FastAPI app, no routes, no `main.py`)
- `app/auth/` (no auth middleware; auth is API-key only in FE/DA services)
- `app/persistence/` (no SQLAlchemy/Postgres)
- `app/storage/` (no COS publisher; `storage.py` is a Protocol only)
- `app/workers/` (no worker processes)
- `app/registry/`, `app/domain_views/`, `app/services/`, `app/events/`, `app/observability/`, `app/settings/`
- `migrations/`, `deployment/`, `platform_web/` (no React anywhere)

The platform is a **contracts-only skeleton** at HEAD. All of Phase 1+ (spec §56) is unbuilt.

---

## 6. Key file anchors (verified at HEAD)

- Root pyproject: `pyproject.toml` (name `quant-projects` 0.4.0, custom `_build_backend`).
- `ownership.yaml` — R49 domain ownership (each top-level path → one team; `scripts/check_no_mirrors.py` enforces no cross-boundary duplication).
- FE service: `factor_engine/service/app.py:1521` (port 8088); `factor_engine/service/security.py` (`resolve_principal`, API-key→principal mapping, JWT/mTLS/reverse-proxy as *sources*, not implemented).
- DA service: `data_access/service/app.py` (port 8765, `hmac.compare_digest` API-key check).
- COS: `data_access/cos/remote.py` (DuckDB httpfs S3), `data_access/read/object_store.py` (`ObjectStore`/`LocalObjectStore`/`COSObjectStore`).
- QE: `quant_evaluator/registry/metrics.py` + `quant_evaluator/metrics/catalog.py` (dual catalog — see GAP_ANALYSIS).
- research_platform: `research_platform/artifacts.py` (15 `*Artifact` classes).
- CI: `.github/workflows/ci.yml` (gate_runner verifiers; no ruff/mypy/frontend/migration/OpenAPI jobs).

---

## 7. Delta summary — old-audit findings that CHANGED vs HELD

| Old-audit finding (d58eae) | Status at HEAD 26b4c02 |
|---|---|
| FE service default port 8088 (not 8766) | **HELD** — still 8088; root Dockerfile CMD 8766 is a real mismatch. |
| FE capability registry single authority | **HELD + STRENGTHENED** — R63 DirectUse verdict is now the single semantic role authority. |
| FP single policy authority; `TreatmentRecipe` MISSING | **HELD** — policy authority unchanged; TreatmentRecipe still MISSING. |
| FO sealed-test/purge/ledger present; TreatmentSelectionArtifact not emitted | **HELD** — plus new fail-closed winner-selector validation. |
| FA `EvaluationIdentity`/cluster-version/library/featureset MISSING | **HELD in FA** — but now provided as DRAFT platform DTOs (`identities.py`, `cluster_library.py`, `feature_set.py`). |
| QE dual metric catalog | **HELD** — both `registry/metrics.py` module catalog and `metrics/catalog.py` `_REGISTRY` still populated and divergent. |
| QE `EvaluationArtifact` MISSING; `EvidenceStatus` missing INVALID_EVIDENCE/LABEL_NOT_MATURE | **HELD in QE** — platform `timing.py` provides the full 4-state `EvidenceStatus` (DRAFT). |
| COS multipart simulated (no-op) | **HELD at HEAD** — committed `COSObjectStore` multipart is no-op; a real boto3 multipart impl exists only in the **working tree** (uncommitted, boto3 not importable in `.venv`). |
| No platform_web / no Postgres / no auth crypto / no migrations | **HELD** — all still absent. |
| `quant_platform` wheel not buildable | **CHANGED** — `pyproject.toml` (untracked) now builds a clean `quant-platform` wheel; contracts import. |
| research_platform legacy dual artifact authority | **HELD** — 15 `*Artifact` classes still present; migration matrix written but not executed. |

_End of CURRENT_ARCHITECTURE._
