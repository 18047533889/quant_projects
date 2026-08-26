# Platform Contracts — v1 DRAFT

`status: DRAFT — pending cross-package reconciliation | contract_set_version: "v1-draft" | date: 2026-08-26`

| Field | Value |
|---|---|
| `contract_set_version` | `"v1-draft"` |
| `status` | `DRAFT — pending cross-package reconciliation` |
| `date` | `2026-08-26` |
| Spec source | `QUANT_RESEARCH_PLATFORM_MASTER_IMPLEMENTATION_SPEC_20260826.md` |
| Implementation home | `platform/app/contracts/` |
| Import rule | **PURE**: stdlib dataclasses only (frozen); NO fastapi/sqlalchemy/pydantic/third-party imports |

## Purpose

`platform/app/contracts/` is the only thin integration DTO layer on the platform side
(spec §6). Its responsibilities are limited to:

- external API DTOs;
- workflow requests;
- `ArtifactRef`;
- `EventEnvelope`;
- domain adapter protocols (`WorkflowBackend`, `ObjectStore`, `BacktestProvider`, …).

It does **not** replace domain-native artifacts (`TreatmentSelectionArtifact` in
`factor_assets`, `EvaluationBundle` in `quant_evaluator`, `BacktestArtifact` in
`vectorbt_qs`, …). Those remain the domain packages' own contracts; the platform
DTOs are the *platform-facing refs / adapter boundary*. The worker adapter layer
translates:

```text
Platform DTO  <->  Domain Native Contract
```

Realization of this DTO layer is `platform/app/contracts/*` (frozen dataclasses +
`typing.Protocol`; stdlib only).

## 1. Immutability Rule (spec §7.3)

Every data payload in this contract set is declared `@dataclass(frozen=True)`.

- The tuple of object identified by `(artifact_id, content_hash)` is append-only.
- An update means producing a **new** artifact (a new `artifact_id` / `content_hash`).
- In-place overwrite of the same `(artifact_id, content_hash)` is forbidden.

## 2. Content-Hash Rule

`content_hash` is derived, not self-reported:

- computed as `sha256` over a canonical, sorted, length-prefixed tuple of every
  semantic field (all fields in the artifact `to_dict()` except the hash field
  itself, the `created_at` provenance timestamp, and the `id` fields);
- a caller-supplied `content_hash` that does **not** equal the recomputed value
  fails closed (`ValueError`);
- the hash must be stable across processes (sort dict keys, fix float repr via
  a canonical string form, snapshot datetimes to ISO-8601 UTC).

See `platform/app/contracts/_contenthash.py`.

## 3. Two-Phase Publish Rule (spec §7.4)

Artifact registration is a two-phase publish; the DB must never be written
`READY` before bytes on storage are final and verified:

```text
write temp object
→ checksum
→ upload / rename / finalize COS object
→ verify object metadata/hash
→ insert artifact metadata in DB
→ publish event
```

`ArtifactRef` is only assigned to a *published* artifact. A ref whose object is
still transient is not allowed to appear in events, workflows, libraries, or
FeatureSets.

## 4. Per-Contract Field Tables

### 4.1 ArtifactRef (spec §7.2) `[RECONCILE]`

| field | type | notes |
|---|---|---|
| `artifact_id` | `str` | identity of the artifact object |
| `artifact_type` | `str` | one of `ARTIFACT_TYPE_*` constants |
| `schema_version` | `str` | JSON schema version of the payload |
| `content_hash` | `str` | sha256 over semantic fields (see §2 above) |
| `storage_uri` | `str` | COS / signed object uri; never a domain-internal path |
| `size_bytes` | `int` | byte size of the stored object |
| `created_at` | `datetime` | UTC ISO-8601 creation timestamp |
| `producer_type` | `str` | e.g. `FACTOR_ENGINE`, `QUANT_EVALUATOR`, `FACTOR_ASSETS`, `LLM_AGENT` |
| `producer_version` | `str` | version string of the producer |
| `snapshot_id` | `str \| None` | data snapshot id when the artifact is snapshot-bound |

`ARTIFACT_TYPE_*` constants (spec §7.2):

```text
FACTOR_CANDIDATE       FACTOR_DEFINITION       FACTOR_VALUE
EVALUATION_BUNDLE      TREATMENT_SELECTION     SIMILARITY_FINGERPRINT
SIMILARITY_GRAPH       CLUSTER_VERSION         CLUSTER_ASSIGNMENT
FACTOR_LIBRARY_VERSION FEATURE_SET             MODEL_DATASET
MODEL                  BACKTEST                REPORT_EXPORT
```

### 4.2 EventEnvelope (spec §11)

| field | type | notes |
|---|---|---|
| `event_id` | `str` | uuid-ish, globally unique |
| `event_type` | `str` | one of `EVENT_TYPE_*` constants |
| `schema_version` | `str` | envelope schema version |
| `occurred_at` | `datetime` | UTC ISO-8601 when the event is raised |
| `producer` | `str` | producing component (e.g. `platform.ingest.v1`) |
| `aggregate_type` | `str` | aggregate the event belongs to (e.g. `FACTOR`) |
| `aggregate_id` | `str` | id of that aggregate |
| `correlation_id` | `str` | root workflow / top-level request correlation |
| `causation_id` | `str \| None` | immediate cause; `None` for root events |
| `payload` | `dict` | event-specific fields |

Core `EVENT_TYPE_*` constants (spec §11):

```text
FactorCandidateDiscovered     FactorCandidateValidated      FactorDefinitionRegistered
FactorMaterializationRequested FactorMaterialized           RawEvaluationCompleted
TreatmentSearchCompleted      TreatedEvaluationCompleted    FactorAdmissionDecided
FactorClusterAssigned         ClusterVersionPublished       FactorLibraryCandidateCreated
FactorLibraryPromoted         FeatureSetCreated             FeatureSetSemanticChanged
ModelRetrainRequired          FactorHealthChanged           ArtifactPublished
JobFailed
```

Events are emitted **after** DB commit via the Transactional Outbox (spec §11.1):
DB state update + outbox insert happen in one transaction; an outbox publisher
then delivers. The platform never double-writes events and state.

### 4.3 JobSpec, JobStatus (spec §9, §12.2, §42, §43)

`JobStatus` enum values (spec §9 "执行状态"):

```text
PENDING  RUNNING  SUCCEEDED  FAILED_RETRYABLE  FAILED_TERMINAL
CANCEL_REQUESTED  CANCELLED  BLOCKED_DATA  BLOCKED_DEPENDENCY  TIMED_OUT
```

`JobSpec` fields — every activity is declared with (spec §12.2):

| field | type | notes |
|---|---|---|
| `job_id` | `str` | platform job identity |
| `job_type` | `str` | e.g. `MATERIALIZE_FACTOR`, `TREATMENT_SEARCH`, `LIBRARY_PROMOTION` |
| `idempotency_key` | `str` | sha256 of canonical tuple (§42) |
| `activity_kind` | `str \| None` | logical activity name in the workflow |
| `priority` | `int` | higher = more urgent |
| `max_retries` | `int` | retry budget; `0` = no retry allowed |
| `timeout_seconds` | `float \| None` | timeout bound |
| `heartbeat_seconds` | `float \| None` | liveness heartbeat interval |
| `resource_class` | `str` | light/io_heavy/cpu_heavy/memory_heavy/long_running |
| `estimated_factor_count` | `int = 0` | §44 scheduling hint |
| `estimated_row_count` | `int = 0` | §44 scheduling hint |
| `input_artifact_refs` | `tuple[str, ...]` | artifact ids consumed |
| `output_artifact_refs` | `tuple[str, ...]` | artifact ids produced |
| `created_at` | `datetime` | UTC |

Idempotency-key rules (spec §42):

```text
materialization: sha256(factor_definition_id, data_snapshot_id, universe_id, calculation_spec_id)
qe evaluation:   sha256(factor_value_id, evaluation_policy_id, label_id, profile_id)
treatment:       sha256(source_evidence, search_policy, split_plan)
```

If a legal artifact already exists for an idempotency key, the stage returns
`CACHE_HIT` instead of recomputing. Retry semantics per error class (§43):

```text
RetryableInfrastructureError | RetryableStorageError | RetryableDatabaseError  -> retry (exponential + jitter), within budget
DataUnavailableError          -> retryable, keyed on data availability
InvalidInputError | SemanticContractError | CapabilityError | NumericalFailure
  | ResourceExceededError     -> terminal, do NOT retry
CancellationError             -> user-initiated cancel -> CANCELLED
```

`ErrorClass` enum in `platform/app/contracts/jobs.py` carries this taxonomy.

### 4.4 FactorCandidateManifest (spec §10.2) + `_READY` protocol (spec §10.1)

`FactorCandidateManifest` fields (mirror `manifest.json`):

| field | type |
|---|---|
| `schema_version` | `str` |
| `candidate_id` | `str` |
| `submitted_at` | `str` (ISO-8601) |
| `submitted_by` | `str` |
| `generator_type` | `str` |
| `generator_version` | `str` |
| `market` | `str` |
| `frequency` | `str` |
| `formula_language` | `str` |
| `factor_spec_uri` | `str` |
| `factor_spec_sha256` | `str` |
| `parent_factor_ids` | `tuple[str, ...]` |
| `required_fields` | `tuple[str, ...]` |
| `semantic_family_hint` | `str \| None` |
| `campaign_id` | `str \| None` |
| `attempt_id` | `str \| None` |

`_READY` protocol (spec §10.1): the COS object directory

```text
/candidates/ market=.../ source=.../ date=.../ candidate_id=FC_xxx/
  manifest.json  factor_spec.json  lineage.json?  evidence.json?  _READY
```

is only *accepted* by the Ingestion Service when `_READY` exists **and** the
manifest checksum verifies. Object *naming convention* for readiness objects:
the marker file must be named exactly `_READY` and be empty (0 bytes); any other
suffix (e.g. `_PENDING`) is not a readiness marker. A manifest without `_READY`
is registered as `PENDING`, never `READY`.

### 4.5 LifecycleState + HealthState (spec §9)

`LifecycleState`:

```text
DISCOVERED VALIDATING VALIDATED COMPILED MATERIALIZING MATERIALIZED
RAW_EVALUATING RAW_EVALUATED TREATMENT_SEARCHING TREATMENT_SELECTED
TREATED_EVALUATING TREATED_EVALUATED NOVELTY_CHECKING ADMISSION_REVIEW
CLUSTER_PENDING CLUSTERED LIBRARY_CANDIDATE SHADOW PRODUCTION_ELIGIBLE
PRODUCTION DEGRADED RETIRED QUARANTINED
```

`HealthState`:

```text
UNKNOWN HEALTHY WATCH DEGRADED CRITICAL STALE
```

Invariant (spec §9): `LifecycleState != JobStatus != HealthState` — three
independent enums; never conflated.

### 4.6 ClusterVersion + ClusterVersionId + logical_cluster_id (spec §8.5, §14) `[RECONCILE]`

`logical_cluster_id` (`CL_PV_MOM_0017` style) is stable across versions, while
`algorithm_cluster_label` (`cluster 18`) changes per run. Every global clustering
re-matches against the previous version, producing cluster-lineage conversion:

```text
UNCHANGED MIGRATED SPLIT MERGED NEW DISSOLVED
```

`ClusterVersionId` canonical input (spec §8.5):

```text
member universe snapshot + similarity graph version + clustering algorithm/version
+ parameters + seed + policy hash
```

`ClusterVersion`:

| field | type | notes |
|---|---|---|
| `cluster_version_id` | `str` | identity; content-addressed |
| `logical_cluster_id` | `str` | stable logical id, e.g. `CL_PV_MOM_0017` |
| `algorithm_cluster_label` | `str` | per-run label, e.g. `cluster 18` |
| `conversion_type` | `ClusterConversionType \| None` | UNCHANGED/MIGRATED/SPLIT/MERGED/NEW/DISSOLVED |
| `member_factor_ids` | `tuple[str, ...]` | members of this version |
| `policy_hash` | `str` | clustering policy hash |
| `created_at` | `datetime` | UTC |

`ClusterVersion` is immutable (§14: "ClusterVersion 不可原地更新"); a refresh
always produces a new version and records `v43 -> v44` lineage.

### 4.7 FactorLibraryVersion + membership (spec §8.6, §14, §15, §16) `[RECONCILE]`

`FactorLibraryVersionId` canonical input (spec §8.6):

```text
logical_library_id + ordered/normalized factor membership identities
+ selection policy + cluster_version + evidence_snapshot
```

`FactorLibraryVersion`:

| field | type | notes |
|---|---|---|
| `library_version_id` | `str` | identity |
| `logical_library_id` | `str` | logical library id (CORE_LOW_REDUNDANCY …) |
| `cluster_version_id` | `str` | referenced cluster version |
| `members` | `tuple[LibraryMembership, ...]` | ordered membership list |
| `policy_hash` | `str` | selection-policy hash |
| `evidence_snapshot` | `str` | evidence snapshot ref |
| `created_at` | `datetime` | UTC |
| `status` | `str` | CANDIDATE/SHADOW/APPROVED/PRODUCTION |

`LibraryMembership` (spec §15 "member 不只是 factor_id"):

| field | type |
|---|---|
| `factor_definition_id` | `str` |
| `selected_treatment_id` | `str \| None` |
| `orientation` | `str \| None` |
| `cluster_id` | `str \| None` |
| `representative_of` | `str \| None` |
| `similarity_ref` | `str \| None` |
| `health_state_ref` | `str \| None` |
| `assembly_score` | `float \| None` |
| `selection_rank` | `int \| None` |

Production library is always immutable; promotion flows go through
CANDIDATE → SHADOW → APPROVED → PRODUCTION with explicit rollback (§16).

### 4.8 FeatureSetArtifact + FeatureSetDiff + ModelRetrainRequired (spec §17, §18) `[RECONCILE]`

`FeatureSetArtifact` (spec §17):

| field | type | notes |
|---|---|---|
| `feature_set_id` | `str` | identity |
| `feature_set_version` | `str` | version string |
| `source_library_versions` | `tuple[str, ...]` | source FactorLibraryVersion ids |
| `ordered_feature_manifest` | `tuple[str, ...]` | **ordered** feature identities |
| `content_hash` | `str` | sha256 over semantics incl. ordering |
| `created_at` | `datetime` | UTC |

Each feature in the manifest binds (spec §8.7): `factor_definition_id`,
`factor_value semantics`, `selected treatment`, `orientation`,
`feature channel name`, `required timing`.

`FeatureSetDiff` categories enum (spec §18):

```text
METADATA_ONLY  EVIDENCE_ONLY  FEATURE_MEMBERSHIP_CHANGE
FEATURE_TRANSFORM_CHANGE  FEATURE_ORIENTATION_CHANGE
FEATURE_SCHEMA_CHANGE  LABEL_CHANGE  DATA_REVISION
```

Default retrain decision: `METADATA_ONLY` / `EVIDENCE_ONLY` → no retrain;
any `FEATURE_*`, `LABEL_CHANGE`, major `DATA_REVISION` → emit
`ModelRetrainRequired` event. The model layer consumes the **event**, never a
platform import of model-training code.

### 4.9 Identity canonicalization inputs (spec §8.1–8.4) `[RECONCILE]`

All identities are canonicalize-then-hash (sha256 of a sorted, length-prefixed
tuple). Display names never enter the canonical input.

| identity | canonical input |
|---|---|
| `FactorDefinitionIdentity` | formula/AST, operator semantics version, frequency, market, input schema requirements, parameter values, calculation semantics |
| `FactorValueIdentity` | FactorDefinitionIdentity + DataSnapshotIdentity + UniverseIdentity + CalculationSpecIdentity + TimingSemanticsIdentity |
| `EvaluationIdentity` | FactorValueIdentity + EvaluationPolicyIdentity + LabelDefinitionIdentity + EvaluationProfileIdentity |
| `TreatmentIdentity` | source_factor_value_id + ordered preprocessing recipe + fit boundary + fit state content hash + neutralization schema |

Helpers live in `platform/app/contracts/identities.py` as pure dict-normalizing +
hash functions and are unit-testable (`Identity.hash` round-trip).

### 4.10 RBAC permission vocabulary (spec §24.1)

Permissions (semantics —— not page-based):

```text
factor:read_summary       factor:read_evidence     factor:read_formula
factor:read_values        factor:download_values   factor:reprocess
cluster:read              library:promote          job:retry
user:manage               standards:edit
```

Roles:

```text
VIEWER  RESEARCHER  CORE_RESEARCHER  OPS  ADMIN
```

Sensitive paths are enforced server-side (§24.2): without `factor:read_formula`,
`GET /factors/{id}/formula` → 403, and the summary response omits the formula.
Raw factor values use the signed-URL flow (§24.3): permission check → audit →
short-lived signed URL.

### 4.11 WorkflowBackend / ObjectStore / LocalArtifactCache / BacktestProvider (spec §12, §22, §41)

`WorkflowBackend` Protocol (spec §12):

```text
start(job_spec) -> job_id
signal(job_id, signal_name, payload)
cancel(job_id)
status(job_id) -> JobStatus
```

`ObjectStore` Protocol (spec §22) — minimal surface:

```text
put(uri, bytes)               # upload; two-phase publish semantics
get(uri) -> bytes             # download full object
head(uri) -> metadata         # content_hash / size / etag
delete(uri)                   # explicit deletion; never for published artifacts
```

`LocalArtifactCache` requirements (spec §22): `max_bytes`, LRU eviction,
checksum validation, partial-download temp suffix, file lock, cache metrics.
Worker flow: `ArtifactRef → cache lookup by content_hash → on miss download →
verify hash → use`. Local cache is disposable and rebuildable at any time.

`BacktestProvider` Protocol (spec §41):

```text
run(signal_artifact, request, execution_policy) -> BacktestArtifactRef
```

`BacktestRequest` / `BacktestArtifactRef` / `ModelDatasetRequest` /
`ModelTrainingRequest` / `ModelArtifactRef` are defined in §4.12/§4.13. Cheap
`qe.probe.*` is strictly distinct from formal `bt.*` execution backtests.

### 4.12 Backtest / Model DTOs (spec §41, §2) `[RECONCILE]`

`BacktestRequest`:

| field | type |
|---|---|
| `request_id` | `str` |
| `idempotency_key` | `str` (canonical tuple hash) |
| `signal_artifact_ref` | `ArtifactRef` |
| `execution_policy_ref` | `str` |
| `universe_id` | `str \| None` |
| `start_time` / `end_time` | `datetime \| None` |

`BacktestArtifactRef` — `ArtifactRef` with `artifact_type=BACKTEST` plus
`backtest_id`, `strategy_ref`, `request_ref`, `result_summary_uri`,
`qe_reevaluation_status` (see §4.14 evidence status).

`ModelDatasetRequest`:

| field | type |
|---|---|
| `dataset_request_id` | `str` |
| `feature_set_artifacts` | `tuple[str, ...]` (FeatureSetArtifact ids) |
| `label_policy_ref` | `str` |
| `output_uri` | `str \| None` (optional override) |

`ModelTrainingRequest`:

| field | type |
|---|---|
| `training_request_id` | `str` |
| `idempotency_key` | `str` |
| `model_dataset_ref` | `ArtifactRef` |
| `model_spec_ref` | `str` |
| `training_params` | `dict` |

`ModelArtifactRef` — `ArtifactRef` with `artifact_type=MODEL` plus `model_id`,
`model_version`, `feature_set_ref`, `train_dataset_ref`, `eval_metrics_uri`.

### 4.13 Timing contract (spec §39)

All evaluation / materialization / feature-set DTOs that carry time semantics
use these five fields:

```text
decision_time          # time at which the decision is taken
signal_available_time  # when signal facts are observable
first_executable_time  # earliest executable trade/fill time
label_start_time       # label observation window start
label_end_time         # label observation window end
```

A-share semantics (ST/停牌/涨跌停/IPO/T+1/收盘后信号/公告 revision/PIT) come from
`DataAccess`/execution contracts, never guessed by page or QE.

### 4.14 Evidence status (spec §40)

```text
NOT_COMPUTED  UNAVAILABLE  INVALID_EVIDENCE  LABEL_NOT_MATURE
```

When a label is not mature, the field is `LABEL_NOT_MATURE` — **never** zero-filled.
Incremental updates distinguish *signal-available clock* vs *label-matured clock*
(two clocks, spec §40).

---

## REMAINING_RECONCILIATION_ITEMS

The following contracts are likely duplicated by an existing domain package's
native artifact. Before this set can be frozen to `v1`, each must be reconciled
(adapter boundary vs. replacement) against the domain-native type. All are
`[RECONCILE]` — none are confirmed frozen.

| contract | likely domain-native home | status |
|---|---|---|
| `ArtifactRef` | `factor_assets/contracts/` (asset/evidence_ref), `quant_evaluator/contracts/` | `[RECONCILE]` |
| `EvaluationIdentity` | `factor_assets/identity/identity.py` (`EvaluationIdentity`), `quant_evaluator/contracts/` | `[RECONCILE]` |
| `FactorSet` | `factor_assets/contracts/factor_set.py` | `[RECONCILE]` |
| `SimilarityArtifact` | `factor_assets/contracts/similarity.py` | `[RECONCILE]` |
| `ClusterVersion` | `factor_assets/clustering/` (cluster version artifact) | `[RECONCILE]` |
| `LibraryVersion` | `factor_assets/` (library governance) | `[RECONCILE]` |
| `FeatureSet` | `factor_assets/contracts/factor_set.py` | `[RECONCILE]` |
| `FittedState` | `factor_assets/contracts/treatment_selection.py` (fit state content hash) | `[RECONCILE]` |
| `Trial` / ledger | `factor_assets/contracts/treatment_selection.py` (`all_trial_refs`) | `[RECONCILE]` |
| Metric registry | `quant_evaluator/contracts/metric_artifacts.py` | `[RECONCILE]` |
| `BacktestArtifact` | `vectorbt_qs/contracts/backtest.py` (`BacktestArtifact`) | `[RECONCILE]` |
| `TreatmentSelection` | `factor_assets/contracts/treatment_selection.py` (`TreatmentSelectionArtifact`) | `[RECONCILE]` |

Reconciliation rule (spec §6): the platform DTO is the *platform-facing ref /
adapter boundary*; the domain-native artifact remains the source of truth inside
its package. The worker adapter translates between them. No domain package
imports `platform.app.contracts`, and `platform.app.contracts` imports no domain
package.

---

## Version footer

`contract_set_version: "v1-draft"` — `status: DRAFT — pending cross-package reconciliation` — `date: 2026-08-26`

This document is a DRAFT. It becomes frozen `v1` only after the
`REMAINING_RECONCILIATION_ITEMS` above are resolved and the Phase-1 test harness
passes against `platform/app/contracts/`. Changes after freeze require a version
bump (v1 → v2) with a changelog section.