# Contract Freeze Draft

**STATUS: DRAFT / NOT FROZEN / IMPLEMENTATION NOT AUTHORIZED**

This is a Wave 1 design artifact, not an implementation authorization. It is derived from the current Wave 0 reports and `AI_GUIDE/01`, `08`, `12`, `13`, `20`, and `22`. The evidence snapshot is `LOCAL_REPO_SNAPSHOT`, bound to commit `ddb03749b7ff85e63b633770c43dfbbb7562af19` at `2026-08-13T13:48:17Z`. The repository is changing concurrently; this document must not claim current HEAD. A later snapshot delta, rebinding this draft to the then-current tree, is required before any freeze decision.

## 1. Freeze Gate

The following blockers remain open and prevent implementation:

- FactorAssets migration and its identity/lineage/novelty storage boundary are not independently reviewed.
- A consolidated package matrix and machine-checkable boundary matrix do not yet exist.
- Independent architecture, math, leakage, A-share, extraction, license, and corpus review has not signed off these contracts.
- Compatibility and extraction plans are not complete, including isolated installation without monorepo path leakage.
- Shared schema ownership, Research Control placement, ModelInput ownership, and threshold/tolerance policy remain unresolved.
- A snapshot delta is required because the repository changed after the `ddb03749` binding.

Until all blockers close, `schemas/` and package skeletons are design-only. No production package, public export, migration, or compatibility shim may be implemented from this file.

## 2. Non-Negotiable Boundaries

Runtime flow:

`data_access -> factor_engine -> quant_evaluator -> optional factor_optimizer -> factor_assets -> factor_preprocess -> ModelInput`

The dependency is protocol-based and acyclic:

```text
DataAccess -----> FE adapters (optional) -----> QE runtime
     |                                      \-> FO/FA/FP adapters (optional)
     v
  supplied context -------------------------> QE
FactorEngine --------------------------------> FO adapter
QE ------------------------------------------> FO, FA adapters
FA ------------------------------------------> FP adapter
FP ------------------------------------------> ModelInput handoff
Research Control <--------------------------- FO/FA append-only events
```

Core packages must import without DA or FE installed. Adapters are optional and are the only permitted integration imports. No package may create a second PIT engine, calendar, universe resolver, snapshot manager, parquet/COS/S3/DuckDB reader, factor DSL/parser/compiler, operator kernel, materializer, generic cache, distributed DAG, backtest/execution engine, model trainer, or portfolio optimizer.

## 3. Evidence-Verified Existing Capabilities

### FactorEngine

FE is a namespace-style distribution (`api`, `expr`, `ir`, `backend`, `runtime`, etc.), not `factor_engine.*`. New code must use these public top-level imports after independent installation. FE owns `api.Factor`, DSL parsing, `Expr`, canonical expression payloads, typed IR, analyzer history/axis effects, operator governance, backend routing, CSE, batch execution, complexity inputs, identity hashes, and materialization.

There is no unified `FactorDefinition` serialization facade and no public `Factor.to_json()` found. The adapter must compose `Factor`, `expression_payload`/`canonical_expression`, and typed-plan identity. FactorAssets and FactorOptimizer must not invent a second canonical hash or parser. Identity and complexity are **USE_EXISTING through adapters**.

### DataAccess

DA is the `data_access` distribution. `get_store()` is the primary facade returning `DataAccessStore`; there is no root `DataAccess` class. DA owns PIT, calendars, universe semantics, snapshots, governed reads, budgets, and storage. There is no public `get_universe()` or `read_universe()` facade. A universe adapter may wrap `read_joined(..., universe=...)`, `read_factors(..., universe=...)`, or advanced `UniverseSnapshot.from_store`; it must not duplicate membership or tradability logic.

### QuantEvaluator

QE consumes supplied values and context. A label is never inferred from a price column by default. The frozen direction is an explicit `LabelBundle` carrying decision, execution, label start/end, horizon, delay, validity, and data/context references. QE emits evidence and diagnosis, not admission decisions, registry state, or persistence.

### FactorPreprocess

FP distinguishes stateless transforms from fitted transforms. Fitted objects are represented by `FittedState` and must retain fit window, `fit_end_time`, feature IDs/order, transform version/config hash, training-universe reference, learned-parameter hash, and a state reference. Full-sample fitting before a split is forbidden.

### Corpora

GTJA, Week2, cold-start catalogs, LQTP artifacts, raw-data-layer material, EvoAlpha, CogAlpha, and external logs are corpus/test inputs, not runtime dependencies. Unknown-license artifacts remain corpus-only until provenance and terms are cleared. Corpus adapters must retain source IDs, digests, formula/original metadata, unsupported semantics, approximation notes, and classification.

## 4. Versioned Contract Envelope

Every cross-package record uses a repeated envelope, not a runtime `common` package:

```text
schema_version: str
producer: str
producer_version: str
created_at: timestamp
run_id: str
factor_ids: [str]
market: str | null
frequency: str | null
universe_ref: str | null
source_snapshot_ref: str | null
parent_refs: [str]
config_hash: str | null
code_hash: str | null
```

Large values are Arrow/NumPy/Polars or an external `value_ref`; they are never embedded in JSON metadata. Each semantic registry item also has an independent metric/transform/mutation version. A mathematical change requires a semantic version bump even when the Python name is unchanged.

## 5. Core Contracts

### FactorDefinitionRef

```text
factor_id: str
canonical_repr: str | opaque ref
canonical_hash: str
source_definition_ref: str | null
frequency: str
lookback: LookbackMetadata
required_fields: [SemanticFieldRef]
domains: [str]
timing: TimingContract
fe_identity_ref: str | null
fe_compiler_generation: str | null
```

Produced by an FE adapter from a public `api.Factor`/`Expr` and compiled plan. `canonical_hash` is FE-derived; no FA/FO reimplementation is valid.

### FactorBatch

```text
factor_ids: [str]
time_axis: AxisRef
asset_axis: AxisRef
values: ValueRef | Arrow/NumPy/Polars block
validity: ValidityRef
layout: long | wide | block
dtype: str
value_hash: str | null
context_refs: ContextRefs
```

Batch-first is mandatory. A single-factor call is a thin wrapper over the batch path.

### LabelBundle

```text
target_id: str
values: ValueRef
horizon: str | int
execution_delay: str | int
decision_time: TimeAxisRef
execution_time: TimeAxisRef
label_start_time: TimeAxisRef
label_end_time: TimeAxisRef
validity: ValidityRef
source_ref: str | null
calendar_ref: str | null
```

Ambiguous, missing, or contradictory timing fails closed. QE must not silently shift, fill labels, or use file modification time as publication time.

### EvaluationRequest / EvaluationBundle

`EvaluationRequest` contains factor IDs or a `FactorBatch`, an explicit `LabelBundle`, metric IDs/preset, slices/group-bys, context refs, tier, and cost budget. `EvaluationBundle` contains versioned metric results, validity/coverage diagnostics, grouped results, `FactorDiagnosis`, warnings, metric/config versions, data/context hashes, and optional large-series `EvidenceRef`s. It does not contain admission decisions or mutate FA state.

RankIC semantics are average-tie daily cross-sectional Spearman with pairwise finite filtering; constants and insufficient assets produce NaN; aggregate NaNs remain NaN. Annualized ICIR must be explicitly named.

### CandidateMutation

```text
mutation_id: str
mutation_spec_version: str
parent_factor_ids: [str]
parameters: Mapping
mechanism_hypothesis: str
expected_signatures: [str]
complexity_estimate: ComplexityRef
lineage_ref: str
trial_ref: str
```

Only registered, FE-legal mutations are executable. LLM output may propose this structure but cannot authorize arbitrary Python, validate itself, or bypass FE.

### FactorAsset / FactorSet

`FactorAsset` contains identity, definition reference, lineage, evidence references, lifecycle state/events, provenance, and compact derived fingerprints. `FactorSet` contains selected factor references and aggregation specifications. FA stores no raw factor matrices. Exact novelty/neighbor evidence comes through a QE/provider adapter after ANN/shortlist recall.

### PreprocessingPolicy / FittedState / FeatureBundle

`PreprocessingPolicy` declares ordered transforms, versions, stateless/fitted status, fit semantics, exposure requirements, missingness channels, and output channels. `FeatureBundle` contains feature/value refs, source factor IDs, channel types, fitted-state refs, timing and missingness metadata. `FittedState` is immutable once published and is bound to its fit window and training universe.

## 6. Public API Draft

Names are draft and subordinate to the eventual freeze schema.

```python
# quant_evaluator
qe.evaluate(batch, labels=label_bundle, metrics="factor_core", context=context)
qe.evaluate_many(requests, context=context)
qe.metric_catalog()

# factor_optimizer
fo.optimize(definition=parent, diagnosis=bundle.diagnosis,
           executor=fe_adapter, evaluator=qe_adapter, search_budget=budget)
fo.validate_mutation(mutation, definition=parent)

# factor_assets
assets.register(candidate, evidence_ref=bundle.ref)
assets.assess(asset_id, policy="production_daily_equity")
assets.neighbors(asset_id, k=100)
assets.assemble(universe="active_core", strategy="family_robust")

# factor_preprocess
state = fp.fit(policy, train_values, context=train_context)
features = fp.transform(test_values, state, test_context)
features = fp.transform_stateless(values, policy, context)
```

Optional adapters are draft-named:

```text
quant_evaluator.adapters.data_access
quant_evaluator.adapters.factor_engine
factor_optimizer.adapters.factor_engine
factor_optimizer.adapters.quant_evaluator
factor_assets.adapters.quant_evaluator
factor_preprocess.adapters.factor_assets
```

Adapter import failure must be localized to the adapter module and reported as `OptionalDependencyMissing`; core import must continue to work.

## 7. Error Taxonomy

The public taxonomy is a base family with concrete types, not one ambiguous exception:

```text
ContractError
  SchemaVersionError
  MissingInputError
  InvalidContractError
  TimingContractError
  SnapshotMismatchError

CapabilityError
  UnsupportedMetricError
  UnsupportedTransformError
  UnsupportedMutationError
  OptionalDependencyMissing

Data/EvidenceError
  InsufficientObservations
  InvalidValidityMask
  MissingLabelError
  EvidenceUnavailableError
  StaleEvidenceError

ExecutionError
  NumericalFailure
  OverflowOrNonFiniteError
  BudgetExceededError
  CancellationError

GovernanceError
  IllegalMutationError
  LifecycleConflictError
  DuplicateIdentityError
  CollisionError
  ContractChangeRequired
```

Cross-package callers must branch on typed errors and stable error codes, not message strings. DA/FE native errors remain wrapped at adapter boundaries with source type, source message, and source request/run reference.

## 8. Acceptance Before Freeze

Freeze requires: current snapshot delta; consolidated boundary/matrix; FA migration matrix; exact public signatures and schemas; independent review; compatibility/extraction plans; isolated package install/tests; legacy import audit; reference/golden tests; explicit reference/fast parity plan; benchmark baseline plan; corpus provenance and license disposition; and resolution of every unresolved decision below.

## 9. Unresolved Decisions

1. Canonical terminal type: `FeatureBundle` or `ModelInputBundle`.
2. Whether `FactorSetArtifact` is an alias, wrapper, or replacement for `FactorSet`.
3. Repeated envelope implementation without a runtime shared package.
4. Research Control repository path, API owner, and synchronization/idempotency with FO/FA.
5. ModelInput schema owner and downstream handoff.
6. Whether `PRODUCTION_CERTIFIED` is lifecycle state, evidence flag, or both.
7. Versioned thresholds for minimum assets, tolerances, ANN recall, budgets, and benchmark regressions.
8. Implemented-vs-registered metric tier mapping and HAC/robustness levels.
9. Permitted FA derived evidence, size limits, retention, and reversibility.
10. FA aggregation versus FP representation responsibility.
11. Authoritative `EvidenceRef` fields, freshness, and invalidation rules.
12. One vocabulary mapping for FO `L0-L4` versus fast/full/robust.
13. Concrete error-code strings and serialization shape.
14. API compatibility policy for examples that explicitly are not frozen.
15. Server truth versus worktree merge authority.
16. Tie-break authority between lead architect and chief integrator.
17. QE runtime/core/temporal and FP transform/performance handoffs.
18. Auditor finding severity, closure, dispute, and sign-off protocol.
19. Archive location and import-blocking mechanism.
20. Industry duplicate-row policy and caller/context ownership.
21. Snapshot delta procedure and required commit binding field.

**Disposition:** retain as draft; do not implement.
