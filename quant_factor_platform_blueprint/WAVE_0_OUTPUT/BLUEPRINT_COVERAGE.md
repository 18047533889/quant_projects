# Blueprint Coverage Audit

## Scope and Result

This Wave 0 audit read every regular file in the requested documentation surfaces, without modifying production or existing reports:

- `AI_GUIDE/`: 24 files, the complete numbered sequence `00` through `23`.
- `.claude/agents/`: 33 files, including `README.md` and all role definitions.
- `.claude/skills/quant-factor-platform-development/references/`: 5 files.
- `MANIFEST.json`: read and cross-checked against the filesystem.

Total requested documentation coverage: **62 files**, **3,043 lines**. `MANIFEST.json` was additionally read for the inventory check. The project skill and project rules were loaded as governing context.

No production file or existing report was changed. This file is the only artifact created by this audit.

## Files Read

### AI Guide (24/24)

- [x] `AI_GUIDE/00_START_HERE.md`
- [x] `AI_GUIDE/01_MASTER_ARCHITECTURE.md`
- [x] `AI_GUIDE/02_LOCAL_REPO_AUDIT_AND_MIGRATION.md`
- [x] `AI_GUIDE/03_QUANT_EVALUATOR_SPEC.md`
- [x] `AI_GUIDE/04_FACTOR_PREPROCESS_SPEC.md`
- [x] `AI_GUIDE/05_FACTOR_OPTIMIZER_SPEC.md`
- [x] `AI_GUIDE/06_FACTOR_ASSETS_SPEC.md`
- [x] `AI_GUIDE/07_RESEARCH_CONTROL_AND_CORPORA.md`
- [x] `AI_GUIDE/08_INTEGRATION_CONTRACTS.md`
- [x] `AI_GUIDE/09_TESTING_PERFORMANCE_ACCEPTANCE.md`
- [x] `AI_GUIDE/10_CLAUDE_PARALLEL_ORCHESTRATION.md`
- [x] `AI_GUIDE/11_PUBLIC_REPO_LESSONS_AND_LICENSE.md`
- [x] `AI_GUIDE/12_IMPLEMENTATION_ROADMAP.md`
- [x] `AI_GUIDE/13_DECISIONS_AND_DO_NOT_BUILD.md`
- [x] `AI_GUIDE/14_METRIC_CATALOG.md`
- [x] `AI_GUIDE/15_MUTATION_AND_DIAGNOSIS_CATALOG.md`
- [x] `AI_GUIDE/16_FACTOR_ASSET_LIFECYCLE_AND_GRAPH.md`
- [x] `AI_GUIDE/17_PREPROCESS_POLICY_CATALOG.md`
- [x] `AI_GUIDE/18_SERVER_FIRST_RUNBOOK.md`
- [x] `AI_GUIDE/19_DETAILED_METRIC_REGISTRY_BLUEPRINT.md`
- [x] `AI_GUIDE/20_SUBAGENT_FILE_OWNERSHIP_MATRIX.md`
- [x] `AI_GUIDE/21_LEGACY_FUNCTION_MINING_CHECKLIST.md`
- [x] `AI_GUIDE/22_API_EXAMPLES_AND_PSEUDOCODE.md`
- [x] `AI_GUIDE/23_A_SHARE_SPECIAL_RULES.md`

### Agent Definitions (33/33)

- [x] `.claude/agents/README.md`
- [x] `.claude/agents/ashare-semantics-auditor.md`
- [x] `.claude/agents/boundary-auditor.md`
- [x] `.claude/agents/chief-integrator.md`
- [x] `.claude/agents/corpus-regression-auditor.md`
- [x] `.claude/agents/extraction-auditor.md`
- [x] `.claude/agents/fa-aggregation-agent.md`
- [x] `.claude/agents/fa-graph-cluster-agent.md`
- [x] `.claude/agents/fa-legacy-miner.md`
- [x] `.claude/agents/fa-novelty-agent.md`
- [x] `.claude/agents/fa-registry-identity-agent.md`
- [x] `.claude/agents/fo-grammar-agent.md`
- [x] `.claude/agents/fo-legacy-miner.md`
- [x] `.claude/agents/fo-llm-agent.md`
- [x] `.claude/agents/fo-search-agent.md`
- [x] `.claude/agents/fp-legacy-miner.md`
- [x] `.claude/agents/fp-neutralization-agent.md`
- [x] `.claude/agents/fp-performance-agent.md`
- [x] `.claude/agents/fp-representation-agent.md`
- [x] `.claude/agents/fp-transform-agent.md`
- [x] `.claude/agents/lead-architect.md`
- [x] `.claude/agents/leakage-auditor.md`
- [x] `.claude/agents/license-auditor.md`
- [x] `.claude/agents/math-auditor.md`
- [x] `.claude/agents/performance-auditor.md`
- [x] `.claude/agents/platform-boundary-auditor.md`
- [x] `.claude/agents/qe-core-metrics-agent.md`
- [x] `.claude/agents/qe-legacy-miner.md`
- [x] `.claude/agents/qe-performance-agent.md`
- [x] `.claude/agents/qe-runtime-agent.md`
- [x] `.claude/agents/qe-temporal-robustness-agent.md`
- [x] `.claude/agents/research-ledger-agent.md`
- [x] `.claude/agents/simplification-auditor.md`

### Skill References (5/5)

- [x] `.claude/skills/quant-factor-platform-development/references/architecture.md`
- [x] `.claude/skills/quant-factor-platform-development/references/license.md`
- [x] `.claude/skills/quant-factor-platform-development/references/migration.md`
- [x] `.claude/skills/quant-factor-platform-development/references/performance.md`
- [x] `.claude/skills/quant-factor-platform-development/references/testing.md`

### Inventory and Governing Inputs

- [x] `MANIFEST.json`
- [x] `.claude/skills/quant-factor-platform-development/SKILL.md` loaded as the required project skill.
- [x] `CLAUDE.md` loaded as project rules.

## Manifest Cross-Check

The comparison was finalized after creating this report; counts below exclude `WAVE_0_OUTPUT/BLUEPRINT_COVERAGE.md` itself.

| Check | Result |
|---|---|
| Manifest entries | 68 |
| Other regular files under blueprint root | 77 |
| Manifest entries missing on disk | 0 |
| Other files omitted from manifest | 9 |
| Numbered AI guides | Complete `00`–`23`, no gaps |

Other files present on disk but absent from `MANIFEST.json` at final audit time:

1. `.DS_Store`
2. `WAVE_0_OUTPUT/CORPUS_INVENTORY.md`
3. `WAVE_0_OUTPUT/DA_INVENTORY.md`
4. `WAVE_0_OUTPUT/FE_INVENTORY.md`
5. `WAVE_0_OUTPUT/FO_LEGACY_MIGRATION.md`
6. `WAVE_0_OUTPUT/FP_LEGACY_MIGRATION.md`
7. `WAVE_0_OUTPUT/LEAKAGE_BOUNDARY_AUDIT.md`
8. `WAVE_0_OUTPUT/LOCAL_REPO_SNAPSHOT.md`
9. `WAVE_0_OUTPUT/QE_LEGACY_MIGRATION.md`

Therefore, the manifest is **complete for every path it claims**, but it is not an exhaustive inventory of the working tree. The first omission is incidental filesystem metadata; the other eight are substantive Wave 0 reports if generated reports are intended to be tracked. Reports other than `BLUEPRINT_COVERAGE.md` appeared concurrently and were not created, read, or modified by this audit. This requested report is also absent from the pre-existing manifest by definition.

The manifest's declared package lists agree with the blueprint:

- Core new packages: `quant_evaluator`, `factor_optimizer`, `factor_assets`, `factor_preprocess`.
- Existing foundations: `dataaccess`, `factor_engine`.

## Architecture and Package Ownership

Target flow:

`DataAccess -> FactorEngine -> QuantEvaluator -> (optional FactorOptimizer loop) -> FactorAssets -> FactorPreprocess -> ModelInput`

The scope ends at model input. Model training, portfolio optimization, large backtesting, execution, and matching are out of scope.

| Owner | Authoritative responsibility | Explicit exclusions / boundary |
|---|---|---|
| `dataaccess` | Data semantics and access, PIT behavior, calendar, universe, snapshots, storage/I/O | New packages must not duplicate data readers, PIT joins, calendars, universe logic, snapshots, or storage wrappers. |
| `factor_engine` | Factor definitions, DSL/AST/IR, operators, semantic legality, computation, complexity source, materialization | FO must not duplicate operator kernels or compiler/parser logic; FA must not parse formulas; QE must not become a factor executor. |
| `quant_evaluator` | Batch evidence, metrics, slices/grouping, diagnoses, exact statistical evaluation | No data platform, factor registry, mutation search, model training, or portfolio optimizer. |
| `factor_optimizer` | Legal mutation grammar, diagnosis-guided proposal, multi-fidelity search, Pareto/budget decisions | No FE operator kernels, QE metric kernels, arbitrary generated production Python, or portfolio optimization. |
| `factor_assets` | Identity, registry, seen-state, lineage, lifecycle, evidence references, novelty orchestration, sparse graph/clusters, selection, aggregation | Not a bulk factor-value store, metric engine, formula parser, or training system. |
| `factor_preprocess` | Stateless and fitted transforms, neutralization, causal temporal transforms, policy resolution, model-facing representation | No factor admission, final selection governance, full-sample fitting, or model training. |
| Research Control | Append-only campaign/trial/evaluation/decision/event ledger | Not a fifth runtime platform, worker system, generic orchestrator, or message bus. |
| Root `/schemas` | Language-neutral, versioned cross-package contracts | Not a runtime `quant_contracts` package; no generic `common/shared/core/helpers` dumping package. |
| ModelInput | Terminal handoff to downstream modeling | Concrete package/interface ownership is not fully defined. |

Cross-package integration must use public APIs, versioned contracts, protocols, and optional adapters. Dependencies must remain acyclic; optional adapters must not cause core imports to fail.

## Hard Constraints

### Source of Truth and Migration

- The current server working tree is authoritative. GitHub and historical repositories are references only; do not overwrite server changes with an initial `git pull`.
- Before implementation, inventory DataAccess, FactorEngine, and legacy code. DA/FE findings must be classified `USE_EXISTING`, `NEED_ADAPTER`, or `TRUE_GAP`.
- Every legacy symbol must be classified exactly as `REUSE_AFTER_TEST`, `REWRITE`, `REFERENCE_ONLY`, `CORPUS_ONLY`, or `DISCARD`.
- Mining must inspect implementations, tests, and call sites, not infer capability from paths or names.
- Production imports from legacy directories, monorepo parents, and `sys.path.append('../')` are prohibited. Final acceptance requires a zero-legacy-import audit.
- Preserve useful semantics, not obsolete architecture. Known legacy bugs need characterization and explicit `legacy_bug_regression`; incorrect behavior is not preserved as compatibility.
- Legacy reference implementations are archived only after real regression passes; they must not be deleted while they are the sole behavioral reference.

### Numerical and Runtime Semantics

- APIs are batch-first; a single-factor API may only be a thin wrapper over batch execution.
- Numerical development order is `Reference -> Golden Test -> Fast Kernel`; fast paths require semantic parity.
- Production paths may not silently fall back to Pandas. Reference/debug fallback must be explicit.
- Optimize measured hot paths only. A benchmark miss never authorizes semantic changes.
- QE and FP must be matrix/block oriented; avoid per-factor Python loops, repeated groupby, and repeated construction of ranks, masks, and exposure matrices.
- Statistical accumulators default to `float64`.
- QE must not materialize the full `2500 x 5000 x 10000` cube. Typical factor blocks are `64–512`, selected by memory budget.
- RankIC is daily cross-sectional average-rank Spearman with pairwise NaN removal, average tie ranks, constant inputs and insufficient assets producing NaN, and aggregate NaNs never coerced to zero. Annualized ICIR must be explicitly named/parameterized.
- Metric names represent semantics, not duplicated year/universe variants; slicing belongs to `Metric x Slice x GroupBy`. Rolling windows and horizons are parameters, not copied metric names.
- Public API semver, schema versions, and metric/transform/mutation versions are independent. A mathematical semantic change requires a version bump.
- Large values move through Arrow/NumPy/Polars or references, not JSON payloads.

### Time, PIT, and A-Share Rules

- QE cannot infer label shifts. `LabelProvider`/`ExecutionReturnProvider` or the caller must provide explicit decision, execution, and label timing; unclear timing rejects advanced evaluation.
- EOD signal at `t` cannot assume same-close execution. Horizon and execution delay are distinct dimensions; delays must cover at least `T+1/T+2/T+3` where applicable.
- Financial features use actual publication/knowable time, not period end, generic `UpdateTime`, or file modification time. Single-quarter and TTM values must be causally derived from information known then.
- Raw IC and tradable IC are separate. QE context carries suspension, ST/status, actual limits, `can_buy/can_sell/can_hold`, board, and IPO age.
- No fabricated auction, Level2, consensus, news, or northbound data. New data domains require DataAccess semantic catalog/schema updates before FO grammar can use them.
- Every fitted temporal object records fit window and `fit_end_time`; fitting is fold-local. Full-sample scaler/PCA/neutralization followed by splitting is forbidden.
- Future-poison and fold-boundary tests are mandatory. Historical transforms may not use future industry or index membership.
- Realized IC cannot update same-day health before labels mature.

### Governance, Search, and Assets

- Decisions use hard gates, evidence vectors, conditional novelty, and Pareto/budget logic; no universal fixed factor score.
- `corr > threshold -> delete` cannot be the sole deduplication rule. Exact/canonical identity, structural similarity, behavior, and conditional novelty remain distinct.
- ANN/LSH is recall-only shortlisting; final neighbor/novelty decisions require exact QE evidence. Full O(K²) similarity is forbidden.
- FA's scale design uses compact `64–256D` fingerprints, typically top `50–200` neighbors, and a sparse multi-view graph. Graph version and cluster lineage must be retained.
- Factor values are not stored in the FA registry. Compact derived fingerprints may be stored, but raw factor matrices may not.
- Lifecycle, decisions, trials, and parent/child mutations are append-oriented and traceable. Historical factors are not physically deleted.
- Every child mutation must be legal, budget-accounted, lineage-recorded, and re-evaluated by QE. Diagnosis is a proposal prior, not proof.
- LLM output is constrained to structured `CandidateMutation`; LLM explanations cannot validate a mechanism or authorize production code.
- Each campaign explicitly caps candidates, full-history evaluations, T2 robust evaluations, LLM calls, compute time, and cost. Budgets cannot be silently exceeded.
- A mechanism falsification result is evidence-based (`SUPPORTED`, `MIXED`, `REJECTED`), not LLM self-attestation.

### Testing, Extraction, Performance, and License

- Required layers include unit/reference, golden, property, metamorphic, edge-case, batch/chunk equivalence, reference/fast parity, leakage red-team, integration, scale benchmarks, extraction, legacy import, boundary, real-corpus regression, and independent audit.
- Every package must install and test in an isolated venv/copy with monorepo `PYTHONPATH` absent.
- Benchmarks record baselines, environment, time, and memory; fast work follows profiling and semantic freeze.
- Third-party use is classified `IDEA_ONLY`, `CORPUS`, `DIRECT_DEPENDENCY`, or `SOURCE_COPY`, with repository, commit, and license provenance.
- GPL/AGPL source is reference/corpus by default and must not be copied into proprietary runtime without escalation.
- Completion is determined by independent auditor evidence, not the implementation agent's assertion.

## Phase Gates

### Wave Model

| Wave | Required work and exit condition |
|---|---|
| Wave 0 | Read-only server/repository audit; four legacy miners; platform boundary, corpus, license, A-share/architecture red-team work. No production edits. |
| Wave 1 | Lead Architect and Chief Integrator freeze schemas, APIs, dependency DAG, ownership, errors, and migration order. Shared contracts are centrally owned. |
| Wave 2 | Migrate reference semantics and tests before optimization. Golden behavior must exist first. |
| Wave 3 | Package owners implement core components in disjoint files; performance agents cannot change frozen semantics. |
| Wave 4 | Integrate FE–QE, QE–FO, QE–FA, FA–FP, and Research Control on a real corpus sample before scale claims. |
| Wave 5 | Independent math, leakage, A-share, performance, boundary, simplification, extraction, license, and corpus-regression audits issue evidence-backed PASS/FAIL. |

Parallel work must stop if shared contracts are concurrently edited, dependency cycles appear, optimization precedes semantic freeze, migration provenance is missing, GPL runtime code is copied, or DA/FE capability is duplicated. Shared-contract changes require a `CONTRACT_CHANGE_REQUEST` and Lead decision.

### Implementation Roadmap Gates

| Phase | Scope | Hard gate / evidence |
|---|---|---|
| Phase 0 | Audit and contract freeze | `LOCAL_REPO_SNAPSHOT.md`, `MIGRATION_MATRIX.md`, `PACKAGE_BOUNDARY.md`, `CONTRACT_FREEZE.md`, and `FILE_OWNERSHIP.md` must exist before Phase 1. |
| Phase 1 | QE foundation | Standalone install/tests, old-vs-new golden evidence, and 100/1000-factor benchmark. |
| Phase 2 | FP foundation | Stateless/fitted contracts, future-poison, fold-boundary, and reference/fast parity. |
| Phase 3 | FA identity/registry | No raw values stored; exact identity, append-only state/history, and repository tests. |
| Phase 4 | FO grammar/search | Invalid mutation rejection, search-budget enforcement, lineage, and child QE re-evaluation. |
| Phase 5 | Advanced QE evidence | Add selected robust/HAC/bootstrap capabilities only after core semantics; capability registration is not implementation. |
| Phase 6 | FA scale layer | 10k benchmark, 100k synthetic design benchmark, ANN recall checked against a small exact reference. |
| Phase 7 | Advanced FP representations | Fold-local policies and `LinearReady`/`TreeReady`/`NeuralReady` representations without training. |
| Phase 8 | Research Control | Minimal append-only campaign/trial ledger; no orchestration platform. |
| Phase 9 | Integration/corpora | Real A-share window, GTJA/Week2, and approved external-corpus smoke/regression. |
| Phase 10 | Hardening/extraction | Math, leakage, performance, boundary, simplification, license, extraction, zero legacy imports, and real regression before archive. |

Additional scale/search milestones:

- FO levels are `L0` static gate, `L1` short/sub-universe T0, `L2` full-history T1, `L3` walk-forward/robust T2, and `L4` combination/cost/implementability.
- First release should implement roughly 40 core metrics well; `150+` capability registration is allowed only when unimplemented statuses remain explicit.
- The suggested `300–1000` stable ModelInput candidates and `16–24` workers are planning targets, not acceptance thresholds or fixed policy.

## Do-Not-Build List

Do not create a second implementation of:

- Data lake, factor lake, Parquet reader, COS/S3 wrapper, DuckDB wrapper, or generic storage facade.
- PIT join engine, calendar, universe, snapshot manager, publication-time engine, or independent tradability source.
- Factor DSL, AST/IR, parser, compiler, operator engine/kernel catalog, factor executor, or materialization/cache engine.
- Universal cache platform, distributed DAG, generic orchestration framework, worker platform, Kafka bus, or microservice mesh.
- Neo4j or PostgreSQL service architecture before measured SQLite/ANN limits justify escalation.
- Feature Store, model training framework, broad backtester, execution/matching system, full PortfolioOptimizer, or TCA system in this phase.
- Runtime `quant_contracts` package or generic cross-package `common`, `shared`, `core`, or `helpers` dumping ground.

Do not use as a default algorithm or shortcut:

- Universal `Factor Score`, hard-coded acceptance-rate targets, or a single `SuperAlpha`.
- Full O(K²) correlation/similarity, full-factor SHAP, or PBO/SPA/bootstrap across every factor by default.
- Correlation threshold as the only deduplication decision or ANN output as final truth.
- Default RF/GBDT/PCA neutralization, default 100% industry+size neutralization, default `fillna(0)`, or full-sample preprocessing.
- Large JSON factor values, raw factor matrices in FA, or physical lifecycle deletion.
- Production Pandas fallback disguised as a fast path, regex complexity as FE semantic truth, or unsupported A-share data claims.
- Arbitrary LLM-generated production Python or LLM self-validation of a mechanism.
- GTJA, Week2, or public factor collections as production runtime dependencies; they are corpus/regression inputs only.

Complexity may increase only after measurement demonstrates a concrete bottleneck. The placeholder QE duplicate-intermediate threshold `>X%` is not yet a usable trigger and must be frozen before implementation.

## Agent and Audit Roles

### Governance and Integration

| Role | Responsibility |
|---|---|
| `lead-architect` | Cross-package architecture, dependency DAG, shared schemas/APIs, ownership assignment, and Contract Freeze decisions. |
| `chief-integrator` | Shared exports, root schemas, package-version alignment, cross-library integration tests, conflict resolution, and final acceptance assembly. |

Only these roles may change `/schemas/**`, root build/workspace configuration, top-level `pyproject.toml`, final package exports, cross-package fixtures, and version bumps. Their exact tie-break authority still needs clarification.

### Discovery and Migration Miners

| Role | Read-only scope |
|---|---|
| `qe-legacy-miner` | Legacy evaluators, analyzers, exposures, batch metrics, formulas, layouts, NaN/tie/timing behavior, callers, and golden-test needs. |
| `fo-legacy-miner` | Legacy alpha tools, complexity evaluators, candidate metadata, and factor-agent orchestration; separate reusable ideas from FE duplication. |
| `fa-legacy-miner` | Admission catalogs, metadata, dedup/seen logic, lifecycle, thresholds, physical deletion, and decision storage. |
| `fp-legacy-miner` | Purification, transforms, neutralization, causal/fitted classification, label usage, and leakage risks. |

### Package Builders

| Role | Owned surface |
|---|---|
| `qe-runtime-agent` | QE contracts/API, registry, planner, shared-intermediate runtime; no unilateral shared-contract changes. |
| `qe-core-metrics-agent` | Quality/distribution, IC/RankIC, quantile, turnover, and probe-portfolio reference metrics/tests. |
| `qe-temporal-robustness-agent` | Horizon, temporal robustness, exposures, A-share slices, and selected T2 statistics. |
| `qe-performance-agent` | Block/chunk matrix kernels, parity, memory, and benchmarks without changing `MetricSpec` semantics. |
| `fo-grammar-agent` | Versioned `MutationSpec`, registry/validation, data-domain legality, and FE consistency adapter. |
| `fo-search-agent` | Multi-fidelity search, budget/plateau logic, Pareto selection, and trial lineage. |
| `fo-llm-agent` | Structured hypothesis/mutation proposal interface and prompt/model/signature records. |
| `fa-registry-identity-agent` | SQLite WAL repository/migrations, exact identity/seen, append-only state events, canonical-hash adapter, and shadow metadata. |
| `fa-graph-cluster-agent` | Compact fingerprints, ANN abstraction, exact-neighbor interface, sparse graph, clusters/families, lineage, and recall benchmarks. |
| `fa-novelty-agent` | Novelty orchestration, neighbor-shortlist requests, and `EvidenceRef` handling. |
| `fa-aggregation-agent` | Hard-gate/evidence/Pareto selection and robust representative/rank/shrinkage/stability aggregation. |
| `fp-transform-agent` | Stateless and causal transforms with reference/fast behavior. |
| `fp-neutralization-agent` | OLS/ridge/robust neutralization, design matrices, exposure decomposition, and fold-local `FittedState`. |
| `fp-representation-agent` | Policy resolver, multichannel representation, missing/freshness channels, and `FactorSet -> FeatureBundle`. |
| `fp-performance-agent` | Batch rank/winsor/neutralization optimization, parity, and memory benchmarks. |
| `research-ledger-agent` | Lightweight append-only Research Control campaigns, trials, events, and review queries. |

### Independent Auditors

| Role | Evidence responsibility |
|---|---|
| `ashare-semantics-auditor` | A-share timing, publication/PIT, tradability, limits, board/IPO rules, and unsupported-data claims. |
| `math-auditor` | Formulas, metric/statistical semantics, neutralization, aggregation, and reproducible PASS/FAIL evidence. |
| `leakage-auditor` | PIT/future leakage across labels, universes, exposures, fitted states, graph information, and test-window selection. |
| `boundary-auditor` | Package imports, runtime dependencies, cycles, and architectural boundary violations. |
| `platform-boundary-auditor` | DA/FE capability inventory with `USE_EXISTING`, `NEED_ADAPTER`, or `TRUE_GAP` decisions. |
| `performance-auditor` | Cross-package speed/memory claims, benchmark methodology, block/chunk behavior, and regression evidence. |
| `simplification-auditor` | Duplicate helpers, unnecessary abstractions, compatibility shells, and deletion/merging opportunities. |
| `extraction-auditor` | Isolated-copy/clean-venv installation and tests, hidden parent-path dependencies, package data, and optional dependency behavior. |
| `license-auditor` | Repository/commit/license provenance and GPL/AGPL runtime-copy risk. |
| `corpus-regression-auditor` | End-to-end FE/QE/FO/FA/FP smoke, real-corpus regression, scale, and performance differences. |

Auditors and miners are read-only (`Write`/`Edit` disallowed). They report evidence and return failures to owners rather than changing production semantics.

## Contradictions and Ambiguous Requirements

These items require explicit resolution during Contract Freeze or policy definition.

1. **Final output name:** the architecture uses `ModelInputBundle`, while FP and integration contracts use `FeatureBundle`. Select one canonical public type.
2. **FA output name:** FA uses `FactorSetArtifact`; integration contracts use `FactorSet`. Define whether one wraps or aliases the other.
3. **Contract metadata:** root rules require `schema_version`, producer, creation/run IDs, factor IDs, timing/market/frequency/universe metadata, parents, and hashes, but individual contracts do not consistently declare them. Define a repeated envelope or per-contract fields without creating a runtime common package.
4. **Research Control location:** it is centrally owned but not one of the four runtime packages. Repository path, public API owner, deployment boundary, and synchronization with FO/FA records are unspecified.
5. **ModelInput ownership:** the terminal handoff is named but no package, schema owner, or public interface is fully defined.
6. **Certification model:** `PRODUCTION_CERTIFIED` is left as either a lifecycle state or a certification flag. Freeze one representation.
7. **Threshold policy:** `min_assets`, hard gates, ANN recall targets, tolerances, search budgets, benchmark regression limits, and the QE `>X%` cache trigger are unspecified. They must be versioned policy/config, not hidden constants.
8. **Metric scope/tiering:** the roadmap targets about 40 implemented core metrics while the registry permits `150+` registered capabilities. State clearly that registration is not support/default execution. HAC and DSR/PBO/SPA/Reality Check tiers differ across documents (`T0–T2`, `T2/T3`, or `T3`) and need per-metric resolution.
9. **Daily coverage tier:** `quality.daily_coverage.mean/std/min` is labeled `T0/T1` without suffix-level assignment.
10. **FA value boundary:** Phase 3 says no raw values stored, while graph design allows rank sketches and compressed PnL fingerprints. Define permitted derived evidence, retention, size limits, and reversibility.
11. **FA/FP aggregation boundary:** FA owns aggregation specifications/weights while FP consumes selected factors/representations. Define which adapter computes aggregate values and where they reside.
12. **EvidenceRef authority:** selected QE summaries lack an authoritative field list, freshness policy, and version rule, risking duplicated/stale metric truth.
13. **Search-level vocabulary:** FO uses `L0–L4`; campaign budgets say full-history/T2; examples use `fast/full/robust` and pseudocode also says `L1`. Freeze one mapping.
14. **Error taxonomy:** `UnsupportedMetric/Transform/Mutation` could mean three concrete exceptions or one family. Define exact public types.
15. **API examples versus contracts:** document 22 says examples are target UX, not frozen API. Frozen schemas must take precedence, and deviations should be recorded.
16. **Worktree versus server truth:** isolated worktrees are recommended, while the server working tree is authoritative. The merge/integration procedure and conflict authority are not specified.
17. **File ownership precision:** agent definitions and the ownership matrix mostly use package-relative surfaces, not concrete absolute/repo-relative globs. Shared names such as `registry/**` can overlap until `FILE_OWNERSHIP.md` gives exact paths.
18. **Shared authority:** `lead-architect` and `chief-integrator` both own schema/API decisions; tie-breaking authority is not explicit.
19. **QE role overlap:** runtime/core/temporal agents overlap around IC, exposures, neutralization-derived metrics, and robustness. Exact module ownership and handoff are needed.
20. **FP optimization handoff:** the performance agent necessarily touches transform/neutralization code semantically owned by other agents; handoff and review authority are not formalized.
21. **FA neighbor ownership:** graph-cluster owns neighbor interfaces while novelty owns shortlist requests; provider versus consumer API ownership needs freezing.
22. **Ledger authority:** FO trial lineage, FA lifecycle events, and Research Control campaign/trial records overlap. Define source-of-truth records and synchronization/idempotency.
23. **Auditor overlap and closure:** boundary/platform-boundary, package/independent performance, and math/leakage/A-share roles overlap. Report consolidation, severity taxonomy, dispute resolution, sign-off authority, and finding closure are unspecified.
24. **Archive procedure:** retention location, import-blocking mechanism, and timing relative to extraction are not fully specified.
25. **Planning targets versus hard gates:** `300–1000` ModelInput candidates, `16–24` workers, and any descriptive acceptance rate must remain targets, not policy gates.
26. **Industry policy:** the industry taxonomy is caller/context-owned, but handling of duplicate industry records must be explicit before FP; default full neutralization is prohibited.
27. **Reference summaries are intentionally incomplete:** the five skill references omit the full dependency matrix, concrete schemas, tolerance/benchmark policy, approved-license matrix, and audit evidence schema. They cannot replace the corresponding AI guides.
28. **Manifest completeness:** all 68 declared paths exist, but the final comparison found eight other Wave 0 reports omitted from the manifest, plus `.DS_Store`. Concurrent reports were not part of the requested reading scope and were not touched.

## Coverage Checklist

### Required Reading

- [x] All AI guides `00`–`23` read in full.
- [x] All 33 regular files under `.claude/agents/` read in full.
- [x] All 5 regular files under the skill `references/` directory read in full.
- [x] `MANIFEST.json` read and parsed.
- [x] Project skill loaded before report creation.
- [x] Project rules applied.

### Required Analysis

- [x] Files-read inventory recorded.
- [x] Hard constraints consolidated.
- [x] Package ownership table produced.
- [x] Wave and Phase 0–10 gates recorded.
- [x] Do-not-build list consolidated.
- [x] Governance, miner, builder, and independent audit roles cataloged.
- [x] Contradictions and ambiguous requirements identified.
- [x] Manifest entries checked against actual files.
- [x] Numbered guide sequence checked for gaps.

### Write Discipline

- [x] No production files modified.
- [x] No existing reports modified.
- [x] Only `WAVE_0_OUTPUT/BLUEPRINT_COVERAGE.md` created.

## Overall Assessment

The blueprint is internally consistent on its central architecture and governance principles: reuse DataAccess and FactorEngine, build exactly four bounded pre-model runtime libraries, freeze contracts before implementation, establish reference semantics before optimization, preserve causal/PIT behavior, and require independent extraction and regression evidence.

It is not yet implementation-ready without the Phase 0 freeze artifacts. The most important unresolved items are canonical contract names and metadata, exact file ownership, threshold/tolerance policy, metric tier assignments, FA/FP and ledger authority boundaries, Research Control/ModelInput placement, and a machine-checkable independent-audit closure process.
