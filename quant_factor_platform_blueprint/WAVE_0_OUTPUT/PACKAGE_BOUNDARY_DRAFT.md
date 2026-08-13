# Wave 0 Package Boundary Evidence (DRAFT)

**Status:** DRAFT pending the FactorAssets (FA) migration report and independent audit.

**Date:** 2026-08-13

**Scope:** Read-only package-boundary evidence for the four proposed runtime packages and their existing DA/FE foundations. This document is the only requested output. It does not change production code, package exports, schemas, or any other report.

**Decision vocabulary:**

- `USE_EXISTING`: the authoritative capability exists in the named foundation/package and must be called through its public contract.
- `NEED_ADAPTER`: the underlying capability exists, but a small typed, public-contract-only adapter or facade promotion is required; the adapter must not reproduce the semantics.
- `TRUE_GAP`: no suitable authoritative capability was found; implement only in the named owner, after contract freeze and golden tests.

`USE_EXISTING` and `NEED_ADAPTER` are not permission to import private modules. A package may use a capability only through an installed public API, a frozen protocol, or an explicitly approved optional adapter.

## 1. Evidence Base and Governing Boundary

### 1.1 Sources read

This draft was prepared from:

- `AI_GUIDE/00_START_HERE.md`, `01_MASTER_ARCHITECTURE.md`, `03_QUANT_EVALUATOR_SPEC.md`, `04_FACTOR_PREPROCESS_SPEC.md`, `05_FACTOR_OPTIMIZER_SPEC.md`, `06_FACTOR_ASSETS_SPEC.md`, `07_RESEARCH_CONTROL_AND_CORPORA.md`, `08_INTEGRATION_CONTRACTS.md`, `13_DECISIONS_AND_DO_NOT_BUILD.md`, and `23_A_SHARE_SPECIAL_RULES.md`.
- `WAVE_0_OUTPUT/DA_INVENTORY.md` and `FE_INVENTORY.md`.
- `WAVE_0_OUTPUT/QE_LEGACY_MIGRATION.md`, `FO_LEGACY_MIGRATION.md`, and `FP_LEGACY_MIGRATION.md`.
- `WAVE_0_OUTPUT/LEAKAGE_BOUNDARY_AUDIT.md`.
- `WAVE_0_OUTPUT/CORPUS_INVENTORY.md` and `BLUEPRINT_COVERAGE.md`.

The inventories identify `data_access` and `factor_engine` as the current server-tree authorities. DA owns data semantics, PIT, calendar, universe, snapshots, storage and governed I/O. FE owns factor definitions, DSL/AST/IR, operator semantics, causal legality, compute, plan optimization, CSE, cache and materialization. The four new packages are bounded at model input.

### 1.2 Target protocol chain

```text
caller/modeling supplies explicit request, labels and model boundary
        |
        v
DA -> governed source reads, PIT, calendar, universe, snapshots, factor values
        |
        v
FE -> Factor/Expr -> typed IR -> plan -> compute/materialize
        |
        v
QE -> FactorBatch + LabelBundle + EvaluationContext -> EvaluationBundle/Diagnosis
        |                                      \
        |                                       +--> FA evidence/novelty/decisions
        v
FO (optional feedback loop) -> legal CandidateMutation -> FE -> QE
        |
        v
FA -> identity/registry/lineage/relations/lifecycle/selection/FactorSet
        |
        v
FP -> fold-local/stateless representation -> FeatureBundle/ModelInput
        |
        v
caller/modeling (training is out of scope here)
```

The arrows are protocol dependencies, not a mandate that every core package install every other package. QE core is value-first and can run on Arrow/NumPy/Polars/Pandas adapters. FO depends on evaluator/executor protocols. FA consumes evidence references and requests values only through a provider. FP consumes a selected factor set, values and exposure context. No cycle is permitted.

## 2. Capability Boundary Matrix

The following is the Wave 0 proposed capability inventory. Every row has one classification and one accountable owner. “Owner” means the semantic authority, not necessarily the package that writes an adapter.

| Proposed capability | Classification | Owner | Evidence and boundary action |
|---|---|---|---|
| Dataset/source semantics and semantic field catalog | `USE_EXISTING` | DA | Use `data_access.get_semantic_catalog`, `SemanticField`, dataset contracts and governed read APIs. New packages may describe required fields but cannot define a second catalog. |
| PIT event records and as-of reads | `USE_EXISTING` | DA | Use `PITEventIndex`/`PITEventRecord`, `DataAccessStore.read_asof`, `read_joined`, and COS as-of entrypoints. QE consumes already-resolved data or a caller-built label/context contract; it does not join PIT data. |
| Publication/availability timing | `USE_EXISTING` | DA | DA determines whether a source was knowable at a time. FE validates formula causality/history. QE validates supplied timing consistency. FP records fit and transform timing. No package may substitute file/update time for publication time. |
| Label timing and execution convention | `NEED_ADAPTER` | caller/modeling with DA | A typed `LabelBundle`/`LabelProvider` adapter is needed at the public boundary. The caller or DA supplies decision, execution, label start/end, horizon and return convention. QE must reject ambiguous shifts and must not infer `shift(+n)` or `shift(-n)`. |
| Market sessions, breaks, holidays and early closes | `USE_EXISTING` | DA | Use `MarketCalendar`, `MarketSession`, `get_market_calendar`, and `get_market_session`. FE may consume the calendar contract for history/availability but must not create a competing calendar. |
| Session-bar clock and intraday aggregation | `NEED_ADAPTER` | DA | DA owns market session/availability semantics and minute-to-daily aggregation. FE owns factor operator composition. A typed adapter must expose the needed session clock and aggregation result without copying either implementation. |
| Universe membership and time-varying universe | `NEED_ADAPTER` | DA | DA owns membership, PIT and tradability. There is no root `get_universe()` facade; use `read_joined(..., universe=..., time_varying_universe=True)`, `read_factors(..., universe=...)`, or an approved adapter around `UniverseSnapshot.from_store`. Promote a narrow facade if repeated callers need it. |
| A-share tradability context | `NEED_ADAPTER` | DA | DA/caller supplies ST, suspension, actual limit, board, IPO age, buyable/sellable/holdable and benchmark membership context. QE consumes it for raw versus tradable metrics; FO can only use domains certified by DA. |
| Financial statement PIT, announcement and vintage alignment | `USE_EXISTING` | DA | Use DA’s governed temporal joins and semantic contracts. Single-quarter/TTM construction must use knowable cumulative values. FE consumes legal fields; QE does not reconstruct financial history. |
| Immutable source snapshots and pre-execution verification | `USE_EXISTING` | DA | Use `resolve_source_snapshot`, `ResolvedSourceSnapshot`, and `verify_snapshot_before_execute`. `DataSnapshot` is a read-result identity. New packages must not define `SourceSnapshot` compatibility classes. |
| Calendar/universe experiment snapshots | `NEED_ADAPTER` | DA | DA has advanced `CalendarSnapshot` and `UniverseSnapshot` implementations that are not root-public. Expose typed adapters or promote narrowly; do not implement snapshot managers in FA/QE/FP. |
| Storage, data lake, factor lake and parquet reads | `USE_EXISTING` | DA | Use `get_store`, `DataAccessStore.read/read_arrow/read_factors/read_joined`, `ReadHandle`, `ScanHandle`, and governed relations. Direct parquet/CSV loaders in legacy evaluators are quarantine material. |
| COS/S3/mirror and remote object access | `USE_EXISTING` | DA | Use DA COS/mirror helpers and snapshot verification. No new S3, COS, credential, remote materializer or HTTPFS wrapper. Existing hard-coded local paths and physical imports remain deployment leakage to quarantine. |
| DuckDB connection, deadlines and resource governance | `USE_EXISTING` | DA | Use `DuckDBEngine`, shared engine factories, `QueryBudget`, `GlobalResourceGovernor`, and DA read pipelines. New packages must not open unmanaged DuckDB connections. |
| Factor-value reads by factor ID | `USE_EXISTING` | DA | Use `DataAccessStore.read_factors` or governed factor-lake reads, with explicit snapshot/universe/version requirements. FA may request values through a `ValueProvider`; it cannot store a bulk factor matrix. |
| Factor value publication/write and generation atomicity | `USE_EXISTING` | DA | DA write/publish paths own authorization, generation and atomicity. FE materialization invokes the governed boundary. QE/FA/FP must not write factor values directly. |
| Read plans, predicates, ContractIR and physical source binding | `USE_EXISTING` | DA | Use public `DataRequest`, `ReadPlan`, predicate AST/compiler, `ContractIR`, and `build_contract_ir`. Advanced wave/planner APIs are not public and require explicit promotion rather than copying. |
| Read waves, source grouping and schema epochs | `NEED_ADAPTER` | DA | DA already has `BatchDataRequest`, `ReadWavePlanner`, `PhysicalFactorDAG` and schema epoch gates, but they are not facade exports. A narrow adapter/promotion is needed for FE/QE batch planning; no package may reproduce planning or schema checks. |
| DSL definition and factor object | `USE_EXISTING` | FE | Use `api.Factor`, `Expr`, `col`/`field` and FE public constructors. There is no separate `FactorDefinition` class to recreate. |
| DSL parsing and dialect compatibility | `USE_EXISTING` | FE | Use `api.dsl_parser.parse_expr/parse_factor` and explicit surface/dialect/version/budget. Corpus adapters may translate external formulas into FE definitions, but must retain source and translation notes. |
| AST canonicalization and serialization | `USE_EXISTING` | FE | Use `expression_payload`, `canonical_expression` and FE canonical aliases. FA/FO may consume the result through an adapter; lowercase/whitespace normalization is not identity. |
| Typed IR, semantic types, history and axis effects | `USE_EXISTING` | FE | Use `Analyzer`, `IRNode`, `AnalysisResult`, `ir.types`, history contracts and axis-effect contracts. QE does not parse or type formulas. |
| Logical/physical planning and plan hashes | `USE_EXISTING` | FE | Use FE compiler/planner and source, typed IR, logical, optimized and physical plan hashes. New packages must not build another compiler or plan hash. |
| Canonical factor identity | `NEED_ADAPTER` | FE with FA consumer | FE has canonical AST/IR and `runtime.factor_identity` identities, but the FA contract needs a stable `FactorDefinitionRef` for bare `Factor` callers, version binding and source metadata. The adapter selects one authoritative FE semantic digest; FA stores the resulting identity, not a second parser/hash. |
| Factor identity versus data/source identity | `NEED_ADAPTER` | DA + FE, coordinated by caller | DA supplies source dataset/snapshot identity; FE supplies factor-definition/plan identity. A versioned envelope must bind both without inventing a runtime `quant_contracts` package. |
| Operator registry and allowlists | `USE_EXISTING` | FE | Use `build_dsl_allowlist`, authoring/research/production mining allowlists, operator contracts and `MiningOperator`/`DirectUseContract` adapters. FO’s mutation registry references FE operators and runs a consistency audit; it does not copy metadata or kernels. |
| Operator semantic type/domain/causality/lookback | `USE_EXISTING` | FE | Use FE IR contracts, analyzer history and production certification. Unsupported A-share domains remain unavailable until DA catalog/schema updates. |
| Operator complexity facts | `USE_EXISTING` | FE | Use parser budgets, analyzer history and `backend.operator_cost`/`estimate_plan_cost`. Regex call extraction and parenthesis counting are not truth. FO adapts facts into a `ComplexityProfile`. |
| Complexity profile public DTO | `NEED_ADAPTER` | FE -> FO | No single public `CostProfile` exists. Define an FO-facing adapter DTO containing depth, weighted operator count, lookback, stateful/CS/nonlinear counts, interactions, source/domain count and estimated/observed latency, with FE version/hash. |
| Factor compute and backend selection | `USE_EXISTING` | FE | Use `FactorEngine`, injected `DataSource`, `build_backend`, backend router and certified backend boundaries. External packages do not invoke cleaned kernels directly. |
| Batch compute, compile-many and shared execution | `USE_EXISTING` | FE | Use `compile_many`, `run_many`, iterators, parallel/batch services and adaptive scheduler. Single-factor calls are thin wrappers over batch behavior. |
| Factor materialization and incremental materialization | `USE_EXISTING` | FE with DA storage | Use FE `materialize*` APIs and DA governed persistence/write contracts. Legacy assetization and staging materializers are duplicate implementations to quarantine. |
| Factor-value cache and plan cache | `USE_EXISTING` | FE; DA for read cache | FE owns expression/column/plan cache and CSE; DA owns governed data read cache. QE’s shared intermediates are in-memory and task-local in Wave 0, not a third persistent cache platform. |
| CSE, factor DAG and shared intermediates | `USE_EXISTING` | FE for factor execution; QE for metric-local intermediates | FE planner/runtime performs multi-factor CSE and DAG execution. QE may resolve lightweight metric dependencies such as ranks -> IC series -> summaries. No universal distributed DAG or cross-task artifact platform. |
| Factor computation provenance and execution identity | `NEED_ADAPTER` | FE + DA, consumed by FA/QE | Existing FE/DA identities need a stable cross-package execution envelope containing code/catalog hash, data snapshot, run ID and factor IDs. Use schemas/protocols, not private plan objects. |
| Evaluation input `FactorBatch` | `NEED_ADAPTER` | QE | QE owns the value-first batch contract, accepting Arrow/NumPy/Polars/Pandas adapters and standardizing internally. DA/FE adapters populate it; no QE storage reader. |
| Explicit `LabelBundle` | `NEED_ADAPTER` | caller/modeling with QE validation | Contract must carry horizon, label semantics, start/end timing and return convention. QE validates, never guesses or silently shifts. |
| `EvaluationContext` and exposure context | `NEED_ADAPTER` | caller/modeling with DA | DA constructs governed masks/contexts; QE consumes universe, industry, size, liquidity, beta, volatility, status, tradability and benchmark fields. FP consumes exposure context for representation. Duplicate context assembly is forbidden. |
| Coverage/validity/unique/tie/zero/inf diagnostics | `TRUE_GAP` | QE | No canonical complete API was found. Implement versioned `MetricSpec`s, explicit missing/constant semantics, minimum observations and batch/chunk parity. |
| Pearson IC and daily RankIC | `TRUE_GAP` | QE | Legacy reference seeds exist, but no canonical package API. Implement average-rank Spearman, pairwise finite rows, constant/min-assets -> NaN, and no NaN-to-zero aggregation. |
| IC summaries, IR, sign ratio, horizon/decay | `TRUE_GAP` | QE | Implement as Metric x Slice x GroupBy with explicit annualization and label timing. Do not inherit legacy hard-coded schemas or thresholds. |
| Quantile, top/bottom spread and monotonicity | `TRUE_GAP` | QE | Rewrite legacy mechanics around typed batches, average ties, sparse-bin policy and explicit missing-label behavior. Admission policy remains FA. |
| Turnover, costs, probe portfolio and portfolio statistics | `TRUE_GAP` | QE | Implement evidence/diagnostics only. Preserve missingness, weight denominators and actual strategy paths; distinguish set churn from weight turnover. No backtest/execution engine. |
| Exposure dependence and neutralization survival metrics | `NEED_ADAPTER` | QE | QE owns statistics; DA/caller supplies exposure matrices/context. FP owns actual model-input neutralization transforms. No future-return attribution code enters FP. |
| HAC/bootstrap/robustness/multiple testing metrics | `TRUE_GAP` | QE | Legacy offers partial HAC/BH seeds only; block bootstrap, CPCV, DSR/PBO/SPA are not production-ready. Add selectively after core metrics and version each metric. |
| Diagnosis and evidence vector | `TRUE_GAP` | QE | Define `FactorDiagnosis` and `EvaluationBundle` with metric versions, warnings, slices, context/data hashes and optional series refs. QE must not decide admission. |
| Mutation grammar and typed `MutationSpec` | `TRUE_GAP` | FO | Build versioned metadata/validation around legal FE operators, domains, causal flag, fit requirement, bounds, complexity increment and production tier. FE remains semantic authority. |
| Diagnosis-guided repair policy | `TRUE_GAP` | FO | Generate candidates as falsifiable proposals only. Every child goes through FE validation and fresh QE evaluation; diagnosis is not proof. |
| Search levels, budgets and plateau analysis | `TRUE_GAP` | FO | Own L0-L4 mapping, candidate/evaluation/compute/LLM budgets, successive-halving or Pareto search and parameter plateau evidence. Do not build a distributed search platform. |
| FO executor/evaluator adapters | `NEED_ADAPTER` | FO | Implement `FactorExecutorProtocol` and `EvaluatorProtocol` adapters to public FE/QE APIs. Optional dependencies must not break FO core import. |
| FE operator consistency audit for FO | `NEED_ADAPTER` | FE + FO | FO lists required operators by reference; adapter compares against FE authoritative registry and rejects unavailable type/domain/cost combinations. |
| Exact identity and seen history | `NEED_ADAPTER` | FA using FE identity | FA owns repository/seen state; FE supplies canonical factor identity. Do not use legacy raw-string hashes or a second DSL parser. |
| Factor asset registry and append-only lifecycle | `TRUE_GAP` | FA | FA migration evidence is pending. Expected owner is FA: SQLite WAL repository, versioned migrations, state transitions/events, lineage, decisions and no physical delete. Final classification awaits FA report/audit. |
| Factor relations, families, graph and clusters | `TRUE_GAP` | FA | FA owns multi-view relation evidence, compact fingerprints, ANN shortlist, sparse graph and cluster lineage. Exact similarity/novelty math is requested from QE. Final retention boundary awaits FA report. |
| Conditional novelty and exact similarity orchestration | `NEED_ADAPTER` | FA + QE | FA orchestrates shortlist and decisions; QE computes exact residual IC, nonlinear and utility evidence. No full O(K^2) matrix and no `corr > threshold -> delete`. |
| Factor selection/admission and aggregation specifications | `TRUE_GAP` | FA | FA owns hard gates, evidence/Pareto/context policy, selected members and aggregation specs. It must not duplicate QE metrics or store raw values. The FA report must resolve FactorSet naming and FA/FP computation boundary. |
| Raw factor-value persistence in FA | `TRUE_GAP` (forbidden capability) | FA governance | This is deliberately not to be built. Store refs, compact non-reversible fingerprints and `EvidenceRef` only under a frozen retention policy. Values come from DA through a provider. |
| Stateless cross-sectional transforms | `TRUE_GAP` | FP | Extract/golden-test pure rank, z-score, robust z-score, winsor, demean, rank-Gaussian and channels from legacy. Consolidate duplicates; do not import toolkit runtime. |
| Causal rolling/volatility transforms | `TRUE_GAP` | FP | Implement explicit asset/date ordering, trailing closure, warmup, lag and future-poison behavior. Legacy current-observation windows require rewrite and cannot be assumed causal. |
| Neutralization and exposure decomposition | `TRUE_GAP` | FP | Implement typed exposure/design matrix contracts, date-local as-of-safe OLS first, then fold-local fitted transforms. QE supplies survival evidence; FP does not choose admission. |
| Fitted preprocessing state | `TRUE_GAP` | FP | Introduce `FittedTransform` and `FittedState` with fit start/end, feature IDs/order, transform version/config hash, training-universe snapshot and learned-parameter hash/ref. Full-sample fit then split is forbidden. |
| Missingness, freshness/age and multi-channel output | `TRUE_GAP` | FP | No complete legacy implementation was found. Implement explicit missing indicator, freshness/age channel, raw/rank/zscore/residual/exposure channels and auditable imputation policy. Never default fundamental missingness to zero. |
| Feature bundle and model-facing representation policy | `NEED_ADAPTER` | FP + caller/modeling | FP owns `FeatureBundle` and policies (`linear_ready`, `tree_ready`, `neural_ready`) but the canonical terminal name (`FeatureBundle` versus `ModelInputBundle`) and modeling handoff remain unresolved. No training-library imports. |
| Research campaign/trial/event ledger | `TRUE_GAP` | Research Control | Implement lightweight append-only SQLite/JSONL ledger with stable IDs, parent/child, mutation/config, code/data/evaluation/decision refs, prompt/model version, failures and budgets. It is not a worker/orchestration platform. |
| FO/FA ledger synchronization | `NEED_ADAPTER` | Research Control with FO/FA | Freeze source-of-truth fields, idempotency keys and event references. FO owns proposal/search facts, FA owns lifecycle/decision facts, Research Control owns campaign/trial/event history. |
| Corpus ingestion and provenance | `NEED_ADAPTER` | Research Control with FE/QE/FA | Corpus adapters preserve source ID, digest, license, original formula, translation/approximation notes and unsupported semantics. Corpus membership never confers production status. |
| A-share unsupported domains | `USE_EXISTING` | DA/FE governance | Level2/order book, analyst consensus/revisions, full news sentiment and true northbound flow are not production capabilities unless DA catalog/schema is updated first. FO grammar must not expose them. |
| Model training, portfolio optimization, execution and matching | `USE_EXISTING` (out of scope) | caller/modeling | These are intentionally outside the Wave 0 platform. No package may grow a training framework, full portfolio optimizer, execution engine or TCA system. |

### 2.1 Interpretation of `TRUE_GAP`

`TRUE_GAP` means a new bounded package capability is justified; it does not mean the legacy implementation should be copied. The migration reports repeatedly classify large legacy facades, direct I/O, fixed admission thresholds, report pipelines, evaluator orchestration, generated FE-like kernels and unpersisted fitted objects as `REWRITE`, `REFERENCE_ONLY`, `CORPUS_ONLY` or `DISCARD`. Only pure formulas that pass golden tests may seed a new implementation.

## 3. Ownership Rules by Package

### 3.1 DataAccess (DA)

DA is the sole authority for source data semantics and governed access:

- PIT/as-of events, publication availability and temporal joins.
- Market calendars, sessions, holidays, breaks, early closes and session clocks.
- Universe membership, time-varying membership and tradability data.
- Dataset contracts, semantic fields, source snapshots, object verification and schema epochs.
- COS/S3/mirrors, DuckDB, query budgets, credentials, resource admission, factor-value reads and writes.

DA adapters must expose only typed, stable contracts. They must not return private planner state as a cross-package API. The missing public universe facade, snapshot DTO and wave/schema promotion are adapter work, not new DA semantics in QE/FA/FP.

### 3.2 FactorEngine (FE)

FE is the sole authority for factor definition and execution:

- Factor/Expr DSL, parser, AST canonicalization and dialect metadata.
- Typed IR, semantic/history/axis-effect contracts, causal legality and operator certification.
- Operator registry, domain/type/parameter/availability metadata and kernel dispatch.
- Complexity facts, compilation, logical/physical plans, backend routing, CSE, batch execution, cache and materialization.
- Factor definition/plan identity, subject to a thin cross-package identity adapter.

New packages must call the public FE API or a deliberately frozen adapter. They must not import cleaned operators, copy generated alpha tools, create a second parser, or instantiate unmanaged FE/DA storage.

### 3.3 QuantEvaluator (QE)

QE is a pure evidence engine:

- `FactorBatch`, explicit `LabelBundle`, `EvaluationContext`, `EvaluationRequest` and `EvaluationBundle`.
- Coverage, distribution, IC/RankIC, quantile, turnover, probe portfolio, exposure, temporal robustness and selected multiple-testing metrics.
- Metric registry, reference/fast parity, block/chunk execution, diagnoses and warnings.

QE does not build data frames from disk, calculate factor formulas, choose label shifts, assemble PIT universes, mutate FA records or train models. Evidence is returned with context/data hashes and references; decisions belong to FA and research events to Research Control.

### 3.4 FactorOptimizer (FO)

FO owns the optional evidence-guided feedback loop:

- Versioned `MutationSpec`, grammar, parameter/domain validation and FE consistency checks.
- Diagnosis-to-candidate proposal, legal mutation lineage, search levels, budgets, plateaus and Pareto decisions.
- Structured LLM hypothesis/mutation output, never arbitrary production Python.

FO uses FE for all operator semantics/complexity/compute and QE for all evidence. It writes trial records through Research Control and does not become a compiler, metric engine, data reader or research database.

### 3.5 FactorAssets (FA)

FA is the identity and governance engine, pending its migration report:

- Canonical identity references, seen history, lineage, relations, family/graph, clusters and novelty orchestration.
- Evidence references and hard-gate/evidence/Pareto admission policy.
- Append-only registry/lifecycle/decision events, selection and aggregation specifications.
- Compact fingerprints and value-provider requests, never a bulk factor-value store.

FA must not parse DSL, implement factor kernels, recompute QE metrics, own DA snapshots or silently delete history. FA report/audit must freeze lifecycle/certification representation, evidence-ref authority, allowed derived-value retention and FactorSet/FactorSetArtifact naming.

### 3.6 FactorPreprocess (FP)

FP is the representation boundary immediately before modeling:

- Stateless cross-sectional and causal time-series transforms.
- Exposure neutralization and decomposition with explicit context.
- Fold-local fitted transforms with auditable state.
- Missingness/freshness channels and multichannel `FeatureBundle` policies.

FP does not calculate raw factors, evaluate predictive validity, select/admit assets, construct future labels or train models. The caller/modeling boundary must resolve the canonical terminal output type and supplies any training-fold orchestration.

### 3.7 Research Control

Research Control is a ledger, not a fifth runtime platform. It owns append-only campaign/trial/event provenance and review queries. It does not own factor values, metric truth, lifecycle policy, workers, Kafka, distributed scheduling or a generic orchestration framework.

## 4. Forbidden Imports and Runtime Paths

The following imports and access patterns are forbidden in new package production code and in any promoted public facade:

### 4.1 Monorepo and path leakage

- `sys.path.append(...)`, `sys.path.insert(...)`, `PYTHONPATH` dependence, repository-parent imports and relative path bootstraps.
- Imports from physical top-level directories such as `read`, `runtime`, `security`, `service`, `dataaccess`, or `factor_engine` layout names when the installed package contract is `data_access.*` or the approved FE public module surface.
- Any `from /home/shw/quant_projects/...` or hard-coded workstation path.
- Legacy runtime imports from `factor_layer.*`, `AutoFactorEvaluation*`, `alphapurify`, `toolkit.*`, `alpha_tools.*`, `FactorAnalyzer`, `Exposures`, `Database`, or old LQTP/raw-data modules.
- `dataaccess.*` aliases: the packaged DA name is `data_access.*`; do not invent a compatibility alias.

The leakage audit found 769 `sys.path` occurrences and 210 legacy import/reference matches, including unacceptable production/operational path mutation. Test/example bootstraps may remain in corpus/test quarantine, but must never be copied into runtime packages.

### 4.2 Platform duplication imports

New packages must not import or wrap these duplicate trees as alternate authorities:

- `AutoFactorEvaluation-RECONSTRUCT/data_access/**` and its COS, DuckDB, snapshot, parquet and contract implementations.
- `AutoFactorEvaluation-RECONSTRUCT/factor_engine/**`, including duplicate DSL/compiler, calendar, universe, data sources, materializers, caches, DAG and backends.
- `raw_data_layer/**` downloader, fetcher, scheduler, S3/REST, parquet validator or cleaning runtime.
- `ashare_lqtp_kit/**` runtime client/DSL/data path until provenance and license are cleared.
- `gtja191/**` and `week2_pv_factors/**` runtime validators/DSL normalizers; they are corpus adapters only.

### 4.3 Cross-package direction constraints

- QE core must not import FA, FO, FP, DA or FE internals; optional DA/FE adapters are allowed.
- FO core must not import FE operator kernels or QE metric modules; use protocols and optional adapters.
- FA core must not import FE parser internals or persist raw values; identity/value requests go through adapters.
- FP core must not import FA internals, QE decision logic, forward-return/label builders or model-training libraries.
- Research Control must not become a dependency that all package cores require merely to import.
- No package may create a broad runtime `common`, `shared`, `core`, `helpers` or `quant_contracts` dumping ground.

## 5. Duplicate Implementations to Quarantine

These findings are quarantine targets, not instructions to edit them in this Wave 0 document.

### 5.1 DA/PIT/calendar/universe/snapshot/storage duplicates

- `AutoFactorEvaluation-RECONSTRUCT/assetization/scripts/compute_engine.py` independent `asof_join` and atomic materialization.
- `AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/trading_calendar.py`, `runtime/session_calendar.py`, and `backend/universe_spec.py`.
- `AutoFactorEvaluation-RECONSTRUCT/data_access/read/read_contract.py` and `read/stats.py` snapshot models.
- `AutoFactorEvaluation-RECONSTRUCT/data_access/cos/remote.py` and `cos/s3_duckdb.py` wrappers.
- `AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/parquet_source.py` and `kline_parquet_source.py`.
- `raw_data_layer/raw_data_fetching/validate_parquet.py` and all direct downloader/storage paths.

Canonical owners are DA public store/read/snapshot/COS/engine APIs, with FE data-source integration only through approved boundaries.

### 5.2 DSL/compiler/AST/IR duplicates

- `AutoFactorEvaluation-RECONSTRUCT/factor_engine/api/dsl_parser.py`, `planner/compiler_pass.py`, and `runtime/engine.py`.
- `ashare_lqtp_kit/ashare_lqtp/dsl_compat.py`.
- `gtja191/lib/dsl_legacy_ops.py`, `dsl_normalize.py`, and `dsl_validate.py`.
- `week2_pv_factors/lib/dsl_validate.py`.

Canonical owner is current FE `api`, `expr`, `ir`, `planner` and `runtime` public surface. Corpus adapters may retain source/translation evidence but may not validate production legality independently.

### 5.3 Materialization, cache and DAG duplicates

- `AutoFactorEvaluation-RECONSTRUCT/assetization/scripts/compute_engine.py` and `integrations/quant_platform.py` materialization.
- `AutoFactorEvaluation-RECONSTRUCT/factor_engine/storage/materialize/**` and `runtime/materialize_service.py`.
- `gtja191/scripts/run_materialize.py` orchestration.
- `AutoFactorEvaluation-RECONSTRUCT/cache_utils.py`, duplicate `factor_engine/storage/cache.py`, `cache/column_cache.py`, `expression_cache.py`, `panel_cache.py`, `planner/dag.py`, and `rolling_cache.py`.

Canonical owners are FE execution/cache/DAG/materialization and DA read cache/resource governance. QE’s metric intermediates remain bounded, in-memory/task-local caches.

### 5.4 Evaluator, preprocessing and operator duplicates

- `factor_layer/factor_evaluation/FactorAnalyzer.py`, `Exposures.py`, `Database.py`, and pipeline/config/report shells.
- `factor_layer/factor_evaluation_alphapurify/AlphaPurifier.py`, `APr_utils.py`, `Exposures.py` report wrappers.
- `toolkit/cross_sectional.py` dispatcher and `toolkit/alpha_tools/generated_library.py` runtime kernels.
- FE cleaned-operator copies under reconstructed or legacy paths.
- Gateway regex future scanner/operator validator and regex/bracket complexity.
- `factor_admission/catalog.py` admission DB in QE/FO runtime.

Pure numerical formulas are reference/corpus seeds only. QE owns evidence, FP owns representation, FE owns factor kernels, FO references FE metadata, and FA owns admission/lifecycle.

### 5.5 External/unlicensed corpus runtime

GTJA191, Week2, LQTP examples/kit, EvoAlpha archive, CogAlpha report and raw-data trees lack sufficient permissive license provenance in the audit. Keep them as `CORPUS_ONLY`/black-box fixtures with immutable source metadata, not direct runtime dependencies or source copies. Public GPL/AGPL repositories remain reference/corpus by default.

## 6. Dependency DAG and Data-Flow Invariants

### 6.1 Package dependency DAG

```text
                         +----------------------+
                         | caller / modeling   |
                         | labels + contexts   |
                         +----------+-----------+
                                    |
                    +---------------v----------------+
                    | DA: data/snapshot/context      |
                    +-----------+---------------------+
                                |
                    +-----------v---------------------+
                    | FE: definition/IR/compute      |
                    +-----------+---------------------+
                                |
                 +--------------v---------------+
                 | QE: values -> evidence       |
                 +-------+-----------------------+
                         |              \
                         |               \
                 +-------v--------+   +--v----------------+
                 | FO (optional)  |   | FA: identity and  |
                 | proposals      |   | governance        |
                 +-------+--------+   +--+----------------+
                         |               |
                         +-------+-------+
                                 |
                         +-------v--------+
                         | FP: representation |
                         +-------+--------+
                                 |
                         +-------v--------+
                         | caller/modeling |
                         +-----------------+

Research Control is a side ledger connected by append-only protocol edges to
FO trials, QE evaluations and FA decisions; it is not a runtime dependency hub.
```

Allowed edges:

- DA -> FE for governed factor data/source contracts.
- DA -> QE through an optional context/value adapter.
- FE -> QE through a FactorBatch/executor adapter.
- FE -> FO through FactorExecutorProtocol and canonical metadata adapter.
- QE -> FO through EvaluatorProtocol.
- QE -> FA through EvaluationBundle/EvidenceRef.
- DA -> FA/FP only through value/context providers or approved adapters.
- FA -> FP through a frozen FactorSet/FactorSetArtifact and policy contract.
- FP -> caller/modeling through FeatureBundle/ModelInput contract.
- FO/FA/QE -> Research Control through append-only ledger protocols.

Forbidden edges include QE -> legacy evaluator I/O, FO -> FE kernels, FA -> raw factor lake, FP -> future labels, and any reverse import from DA/FE into package cores that makes optional adapters mandatory.

### 6.2 Data-flow invariants

1. A data snapshot and universe identity are pinned before an evaluation run where reproducibility is required.
2. FE proves formula history/causality; DA proves source availability; QE validates supplied evaluation timing; FP proves fold-local fitting.
3. Labels carry explicit start/end/execution semantics. No package guesses shifts.
4. Raw factor values move as Arrow/NumPy/Polars or governed references, never as large JSON metadata and never into FA registry rows.
5. Every evidence bundle carries schema/metric/config/data/context hashes and a run ID.
6. Every FO child has immutable parent/mutation/config lineage, FE validation and new QE evidence.
7. Every FA lifecycle/decision change is append-only; shadow/rejected/retired history is retained.
8. Every fitted FP state carries fit window, feature order, training universe and parameter hash.
9. Every corpus adapter preserves source, digest, license status, original formula and approximation/unsupported notes.
10. Reference -> golden test -> fast kernel is mandatory for new numerical implementations.

## 7. Unresolved Public-Facade and Contract Gaps

These are explicit Wave 1 Contract Freeze inputs. They are not permission to add private imports or duplicate implementations.

### DA facade gaps

1. No root `get_universe()`/`read_universe()` facade. Decide whether a narrow adapter wraps registered universe datasets or promotes `UniverseSnapshot.from_store`.
2. `CalendarSnapshot`, `UniverseSnapshot`, schema epoch gates, wave planner and some source-binding APIs exist but are advanced/not root-public. Define exact promoted interfaces and versioning.
3. No stable root `SourceRef` DTO matching the blueprint terminology. Bind dataset/parameters/snapshot to a versioned contract without inventing a new runtime common package.
4. DA naming is `data_access`, not `dataaccess`; freeze import/package naming and prevent compatibility aliases.
5. DA has both a read-result `DataSnapshot` and resolved source snapshot objects. Define the cross-package distinction and required fields.

### FE facade gaps

1. FE is a namespace-style source tree without a root `factor_engine` package facade. Freeze the supported installed public modules and adapter import paths.
2. No single `FactorDefinition` or `Factor.to_json()` contract. Define `FactorDefinitionRef` around canonical FE serialization and metadata.
3. FE has multiple identity/hash forms. Freeze which semantic identity hash FA uses and which source/plan hashes are evidence/provenance only.
4. No single public complexity profile DTO. Define the FE-to-FO projection and generation/version binding.
5. Confirm the public contract for compiled execution/materialization across optional DA integration without exposing private plan objects.
6. Wave planner/schema epoch promotion must be explicit if QE/FE batch planning needs it.

### QE/FA/FP contract gaps

1. `FeatureBundle` versus `ModelInputBundle`: choose one canonical terminal public type.
2. `FactorSet` versus `FactorSetArtifact`: choose one canonical type or define a wrapper/alias.
3. Define the repeated metadata envelope: schema version, producer, created time, run ID, factor IDs, time/market/frequency/universe, parents, source/data/code/config hashes.
4. Define `EvidenceRef` authoritative fields, freshness, metric-version policy and permitted selected summaries.
5. Define FA raw-value-derived fingerprint retention, size, reversibility and deletion rules.
6. Define where aggregation specified by FA is computed: FA selection/specification versus FE execution/DA value provider versus FP representation.
7. Define Research Control repository path, public protocol, event IDs, idempotency and synchronization with FO/FA.
8. Define ModelInput owner and handoff to caller/modeling; keep training out of scope.
9. Define exact exception taxonomy for schema, timing, missing input, unsupported capability, insufficient observations, numerical failure and optional dependency errors.
10. Freeze metric/transform/mutation tier mapping, minimum observations, tolerances, benchmark gates, search budgets, ANN recall and cache escalation thresholds.
11. Resolve FA certification representation: lifecycle state or certification flag.
12. Resolve QE role overlap for exposure metrics versus FP neutralization and runtime/core/temporal module ownership.

## 8. Wave 0 Exit Assessment

The central boundary is supported by the current inventories:

- DA already supplies the PIT, calendar, universe, snapshot, storage, factor-value and resource platform.
- FE already supplies DSL/AST/IR, canonicalization, operator registry, complexity facts, computation, materialization, cache/CSE/DAG and scheduling.
- QE, FO and FP have clear `TRUE_GAP` areas, but their legacy sources are coupled and require extraction/golden tests rather than direct migration.
- FA’s expected boundary is clear from the architecture guide, but this draft remains pending the requested FA migration report and independent audit.
- The leakage audit identifies extensive unacceptable path mutation, legacy imports and duplicate DA/FE platform trees that must remain quarantined.
- Corpus inventory confirms GTJA/Week2/LQTP/EvoAlpha and related artifacts are regression/reference inputs, not runtime dependencies.

**Wave 0 status is therefore DRAFT, not implementation approval.** Wave 1 must freeze the unresolved public facades, cross-package schemas, file ownership, error taxonomy, retention policies and audit sign-off procedure before any production package implementation begins.
