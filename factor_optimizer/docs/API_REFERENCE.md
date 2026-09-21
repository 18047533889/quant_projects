# factor_optimizer 完整模块与接口索引

先读 [功能与算法手册](FUNCTIONAL_GUIDE.md)，再查本页的具体入口、参数和实现位置。
扫描实际包目录：**77 个 Python 模块、937 个公开函数/类/方法定义**。
收录非下划线开头的顶层定义及类的公开方法，不把所有内部模块都承诺为稳定API；私有辅助算法见功能手册。
参数、类型、默认值直接取自源码语法树，不导入或启动可选后端。类型注解不代表生产可用性。
未写独立说明的入口会明确标记，不凭名称编造功能；算法讲解、约束、完整流程与例子见功能手册。

## 模块目录

| 模块 | 定义数 | 模块说明 |
|---|---:|---|
| [factor_optimizer/__init__.py](../factor_optimizer/__init__.py) | 1 | FactorOptimizer: Evidence-guided factor mutation and search. |
| [factor_optimizer/adapters/__init__.py](../factor_optimizer/adapters/__init__.py) | 0 | Adapter protocols for FE and QE integration. |
| [factor_optimizer/adapters/factor_assets.py](../factor_optimizer/adapters/factor_assets.py) | 18 | FA factor-intelligence provider adapters (R61-FI-014 / plan §20 E6, matrix E6). |
| [factor_optimizer/adapters/factor_engine.py](../factor_optimizer/adapters/factor_engine.py) | 7 | FactorEngineAdapter: protocol for FE integration (optional dependency). |
| [factor_optimizer/adapters/fitness.py](../factor_optimizer/adapters/fitness.py) | 2 | Fitness adapter: dimension/desirability mapping for generalized treatment decisions. |
| [factor_optimizer/adapters/layered_decay.py](../factor_optimizer/adapters/layered_decay.py) | 4 | TRAIN-frozen research plan for observation-origin twenty-layer decay. |
| [factor_optimizer/adapters/preprocessing.py](../factor_optimizer/adapters/preprocessing.py) | 6 | Versioned research bridge from smoothing proposals to real FP kernels. |
| [factor_optimizer/adapters/quant_evaluator.py](../factor_optimizer/adapters/quant_evaluator.py) | 14 | QuantEvaluatorAdapter: protocol for QE integration (optional dependency). |
| [factor_optimizer/adapters/repair_execution.py](../factor_optimizer/adapters/repair_execution.py) | 5 | Versioned, research-only execution plans for value-level repair families. |
| [factor_optimizer/capabilities.py](../factor_optimizer/capabilities.py) | 5 | Truthful runtime capability metadata for factor_optimizer. |
| [factor_optimizer/complexity/__init__.py](../factor_optimizer/complexity/__init__.py) | 0 | Complexity estimation and budget tracking. |
| [factor_optimizer/complexity/budget.py](../factor_optimizer/complexity/budget.py) | 9 | Complexity budget tracking and enforcement. |
| [factor_optimizer/complexity/profile.py](../factor_optimizer/complexity/profile.py) | 8 | ComplexityProfile: adapter projection of FE complexity analysis. |
| [factor_optimizer/contracts/__init__.py](../factor_optimizer/contracts/__init__.py) | 0 | Contracts for mutation proposals, search budgets, and trials. |
| [factor_optimizer/contracts/budget_extension.py](../factor_optimizer/contracts/budget_extension.py) | 5 | FO-P1-24 budget-resume authorization contract. |
| [factor_optimizer/contracts/campaign_store.py](../factor_optimizer/contracts/campaign_store.py) | 43 | Durable SQLite control-plane state for FO campaigns. |
| [factor_optimizer/contracts/candidate_mutation.py](../factor_optimizer/contracts/candidate_mutation.py) | 3 | CandidateMutation: typed mutation proposal with provenance and complexity estimate. |
| [factor_optimizer/contracts/evaluation_artifact.py](../factor_optimizer/contracts/evaluation_artifact.py) | 11 | TrialEvaluationArtifact: typed, content-hashed evaluation output (FO-P1-25). |
| [factor_optimizer/contracts/evidence_value.py](../factor_optimizer/contracts/evidence_value.py) | 16 | Evidence-bound metric values: missing evidence is never a fabricated 0.0 (R61-FI-030). |
| [factor_optimizer/contracts/factor_fitness.py](../factor_optimizer/contracts/factor_fitness.py) | 9 | FactorFitnessSpec + CandidateFitnessArtifact (R61-FI-031 / plan §21). |
| [factor_optimizer/contracts/library_snapshot_ref.py](../factor_optimizer/contracts/library_snapshot_ref.py) | 4 | Library-snapshot reference binding for FO (DLIB-FO-008). |
| [factor_optimizer/contracts/multiplicity.py](../factor_optimizer/contracts/multiplicity.py) | 5 | MultiplicityArtifact: full-proposal-process multiplicity tracking (DLIB-FO-006). |
| [factor_optimizer/contracts/objective.py](../factor_optimizer/contracts/objective.py) | 4 | ObjectiveSpec: first-class, frozen, serializable search objective contract. |
| [factor_optimizer/contracts/search_budget.py](../factor_optimizer/contracts/search_budget.py) | 23 | SearchBudget: tracking for trials, evaluations, and compute costs. |
| [factor_optimizer/contracts/splits.py](../factor_optimizer/contracts/splits.py) | 18 | Split contracts for SearchRunner. |
| [factor_optimizer/contracts/statistical_governance.py](../factor_optimizer/contracts/statistical_governance.py) | 7 | FO-side statistical-search governance contracts. |
| [factor_optimizer/contracts/treatment_integrity.py](../factor_optimizer/contracts/treatment_integrity.py) | 17 | TreatmentIntegrityEvidence: real integrity evidence for a treatment run (R55 P0-9). |
| [factor_optimizer/contracts/treatment_result.py](../factor_optimizer/contracts/treatment_result.py) | 13 | RawTreatmentOutcome + extended TreatmentOptimizationResultArtifact (R61-FI-033). |
| [factor_optimizer/contracts/trial.py](../factor_optimizer/contracts/trial.py) | 7 | Trial: record of a single mutation attempt with status and results. |
| [factor_optimizer/contracts/trial_ledger.py](../factor_optimizer/contracts/trial_ledger.py) | 23 | Append-only TrialLedger for every proposal attempt (FO-P0-04). |
| [factor_optimizer/contracts/validator.py](../factor_optimizer/contracts/validator.py) | 7 | TrialValidator contracts with identity. |
| [factor_optimizer/data_capabilities.py](../factor_optimizer/data_capabilities.py) | 59 | Real data capabilities: authorization-only scope boundaries. |
| [factor_optimizer/data_providers.py](../factor_optimizer/data_providers.py) | 12 | Real scoped data isolation (FO-P0-01). |
| [factor_optimizer/errors.py](../factor_optimizer/errors.py) | 28 | Core error taxonomy for factor_optimizer. |
| [factor_optimizer/grammar/__init__.py](../factor_optimizer/grammar/__init__.py) | 0 | Mutation grammar: versioned MutationSpec registry and validation. |
| [factor_optimizer/grammar/mutation_spec.py](../factor_optimizer/grammar/mutation_spec.py) | 7 | MutationSpec: typed mutation operation with parameter constraints. |
| [factor_optimizer/grammar/registry.py](../factor_optimizer/grammar/registry.py) | 8 | MutationRegistry: catalog of available mutation operations. |
| [factor_optimizer/grammar/validation.py](../factor_optimizer/grammar/validation.py) | 8 | MutationValidator: validate mutation proposals against grammar and FE legality. |
| [factor_optimizer/llm/__init__.py](../factor_optimizer/llm/__init__.py) | 0 | LLM-powered mutation proposal generation. |
| [factor_optimizer/llm/prompts.py](../factor_optimizer/llm/prompts.py) | 11 | Versioned prompt templates for LLM proposal generation. |
| [factor_optimizer/llm/proposal.py](../factor_optimizer/llm/proposal.py) | 9 | Structured LLM proposal generator with schema validation. |
| [factor_optimizer/llm/records.py](../factor_optimizer/llm/records.py) | 15 | LLM call recording for reproducibility and audit. |
| [factor_optimizer/policy/__init__.py](../factor_optimizer/policy/__init__.py) | 0 | Policy and decision logic. |
| [factor_optimizer/policy/decisions.py](../factor_optimizer/policy/decisions.py) | 20 | Admission decision records for mutation proposals. |
| [factor_optimizer/policy/repair.py](../factor_optimizer/policy/repair.py) | 16 | Diagnosis-to-mutation mapping for failure repair strategies. |
| [factor_optimizer/policy/repair_registry.py](../factor_optimizer/policy/repair_registry.py) | 38 | Versioned repair-family registry + diagnosis-specific repair policy (R61-FI-034). |
| [factor_optimizer/ports/__init__.py](../factor_optimizer/ports/__init__.py) | 0 | Ports: narrow consumer-side contracts FO binds other packages through. |
| [factor_optimizer/ports/factor_intelligence.py](../factor_optimizer/ports/factor_intelligence.py) | 29 | Factor Intelligence provider port (R61-FI-014 / plan §20 E6, matrix E6). |
| [factor_optimizer/research_baseline.py](../factor_optimizer/research_baseline.py) | 9 | Research baseline recipes and TRAIN-only substantial-degradation guard. |
| [factor_optimizer/research_batch.py](../factor_optimizer/research_batch.py) | 14 | Bounded research batch optimization with automatic chronological splitting. |
| [factor_optimizer/research_decay.py](../factor_optimizer/research_decay.py) | 1 | TRAIN-only twenty-layer stale-signal decay, not holding-period PnL. |
| [factor_optimizer/research_diagnostics.py](../factor_optimizer/research_diagnostics.py) | 1 | TRAIN-only multi-dimensional research diagnosis using QE metric authorities. |
| [factor_optimizer/research_final_report.py](../factor_optimizer/research_final_report.py) | 3 | Authority-side final research reporting; never called by candidate search. |
| [factor_optimizer/research_fitness.py](../factor_optimizer/research_fitness.py) | 10 | QE-owned research portfolio metrics and joint paired selection policy. |
| [factor_optimizer/research_manifest.py](../factor_optimizer/research_manifest.py) | 2 | Bind declared COS factor values to their exact research landing record. |
| [factor_optimizer/search/__init__.py](../factor_optimizer/search/__init__.py) | 0 | Search orchestration for factor mutation optimization. |
| [factor_optimizer/search/categorical_strategy.py](../factor_optimizer/search/categorical_strategy.py) | 10 | Categorical search strategy (TPE-style) for treatment auto-optimization. |
| [factor_optimizer/search/conditional_search.py](../factor_optimizer/search/conditional_search.py) | 24 | Hierarchical conditional search over (repair-family, parameters) (R61-FI-035). |
| [factor_optimizer/search/desirability.py](../factor_optimizer/search/desirability.py) | 6 | Soft desirability floors for factor optimization. |
| [factor_optimizer/search/desirability_registry.py](../factor_optimizer/search/desirability_registry.py) | 14 | DesirabilityPolicyRegistry (DLIB-FO-003). |
| [factor_optimizer/search/diagnosis_routing.py](../factor_optimizer/search/diagnosis_routing.py) | 10 | Deterministic V5 diagnosis routing with bounded, non-Cartesian trials. |
| [factor_optimizer/search/dimensions.py](../factor_optimizer/search/dimensions.py) | 5 | Dimension aggregation primitives for factor optimization. |
| [factor_optimizer/search/lineage.py](../factor_optimizer/search/lineage.py) | 27 | Mutation lineage tracking for provenance and genealogy analysis. |
| [factor_optimizer/search/multifidelity.py](../factor_optimizer/search/multifidelity.py) | 37 | Multi-fidelity evaluation: Stage0-5 profile-bound tiers (R61-FI-036). |
| [factor_optimizer/search/paired_comparison.py](../factor_optimizer/search/paired_comparison.py) | 5 | V5 paired common-draw factor comparison. |
| [factor_optimizer/search/pareto.py](../factor_optimizer/search/pareto.py) | 26 | Pareto frontier tracking for multi-objective optimization. |
| [factor_optimizer/search/plateau.py](../factor_optimizer/search/plateau.py) | 19 | Plateau detection for search stopping criteria. |
| [factor_optimizer/search/runner.py](../factor_optimizer/search/runner.py) | 36 | SearchRunner: orchestrate mutation search with budget and plateau stopping. |
| [factor_optimizer/search/statistical_consumption.py](../factor_optimizer/search/statistical_consumption.py) | 2 | FO consumption of QE statistical evidence bound to durable campaign state. |
| [factor_optimizer/search/strategies.py](../factor_optimizer/search/strategies.py) | 57 | Search strategies: Random, Grid, Bayesian (GP), and TPE. |
| [factor_optimizer/search/supervised_parameter.py](../factor_optimizer/search/supervised_parameter.py) | 3 | Train-only selection and freezing for supervised repair parameters. |
| [factor_optimizer/search/tiered_evaluation.py](../factor_optimizer/search/tiered_evaluation.py) | 24 | Tiered (funnel) evaluation for factor auto-treatment optimization. |
| [factor_optimizer/search/treatment_decision.py](../factor_optimizer/search/treatment_decision.py) | 12 | Generalized 8-step treatment decision policy (R61-FI-032 / plan §21.2). |
| [factor_optimizer/search/uncertainty_winner.py](../factor_optimizer/search/uncertainty_winner.py) | 5 | Uncertainty-aware winner selection (DLIB-FO-004). |
| [factor_optimizer/search/winner_selector.py](../factor_optimizer/search/winner_selector.py) | 6 | Robust winner selector for factor auto-treatment optimization. |
| [factor_optimizer/seen/__init__.py](../factor_optimizer/seen/__init__.py) | 0 | Seen cache: track previously evaluated factors using FE canonical identity. |
| [factor_optimizer/seen/identity.py](../factor_optimizer/seen/identity.py) | 15 | SeenCache: track previously evaluated factors to avoid duplicates. |

## factor_optimizer/__init__.py

FactorOptimizer: Evidence-guided factor mutation and search.

显式导出（含重导出）：`package_info`、`CapabilityStatus`、`ExecutionMode`、`PRODUCTION_CAPABILITY`、`ProductionCapability`、`require_production_capability`、`TrialValidatorIdentity`、`MutationGrammarValidator`、`FactorOptimizerError`、`ContractError`、`SchemaVersionError`、`MissingInputError`、`InvalidContractError`、`TimingContractError`、`SnapshotMismatchError`、`CapabilityError`、`UnsupportedMutationError`、`OptionalDependencyMissing`、`DataError`、`InsufficientObservations`、`InvalidValidityMask`、`EvidenceUnavailableError`、`StaleEvidenceError`、`ExecutionError`、`NumericalFailure`、`OverflowOrNonFiniteError`、`BudgetExceededError`、`CancellationError`、`GovernanceError`、`IllegalMutationError`、`DuplicateIdentityError`、`CollisionError`、`ContractChangeRequired`、`TreatmentIntegrityError`、`EVIDENCE_SCHEMA_VERSION`、`RAW_TREATMENT_KIND`、`IntegrityCheckResult`、`TreatmentIntegrityEvidence`、`TreatmentIntegrityStatus`、`build_integrity_evidence`、`describe_integrity_problem`、`digest_value`、`require_integrity_evidence`。

### package_info

[实际实现](../factor_optimizer/__init__.py#L70)。

Return package metadata.

参数：`()`。

## factor_optimizer/adapters/__init__.py

Adapter protocols for FE and QE integration.

显式导出（含重导出）：`FactorEngineAdapter`、`EvidenceStore`、`InMemoryEvidenceStore`、`QuantEvaluatorAdapter`、`FEOptionalDependencyMissing`、`QEOptionalDependencyMissing`、`create_fe_adapter`、`create_qe_adapter`、`create_mock_qe_adapter`。

## factor_optimizer/adapters/factor_assets.py

FA factor-intelligence provider adapters (R61-FI-014 / plan §20 E6, matrix E6).

显式导出（含重导出）：`FaFactorIntelligenceProvider`、`InMemoryFactorIntelligenceProvider`、`create_fa_intelligence_provider`、`create_in_memory_factor_intelligence_provider`、`taxonomy_view_from_fa`、`health_view_from_fa`、`diagnosis_view_from_fa`。

### taxonomy_view_from_fa

[实际实现](../factor_optimizer/adapters/factor_assets.py#L96)。

Project an FA ``FactorTaxonomyArtifact`` (or duck-shape) into the view.

参数：`(artifact: Any, *, factor_definition_id: Optional[str]=None)`。

返回类型：`FactorTaxonomyView`。

### health_view_from_fa

[实际实现](../factor_optimizer/adapters/factor_assets.py#L125)。

Project an FA health-card artifact (or duck-shape) into the view.

参数：`(artifact: Any, *, factor_definition_id: Optional[str]=None, evaluation_ref: str='')`。

返回类型：`FactorHealthView`。

### diagnosis_view_from_fa

[实际实现](../factor_optimizer/adapters/factor_assets.py#L184)。

Project one FA diagnosis artifact (or duck-shape) into the view.

参数：`(diagnosis: Any, *, factor_definition_id: Optional[str]=None, health_ref: str='')`。

返回类型：`DiagnosisView`。

### FaFactorIntelligenceProvider

[实际实现](../factor_optimizer/adapters/factor_assets.py#L234)。

``FactorIntelligenceProvider`` backed by the FA ``profiling`` package.

### FaFactorIntelligenceProvider.__init__

[实际实现](../factor_optimizer/adapters/factor_assets.py#L248)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, *, get_fa_taxonomy=None, get_fa_health=None, get_fa_diagnoses=None, fa_profiling: Optional[Any]=None)`。

返回类型：`None`。

### FaFactorIntelligenceProvider.get_taxonomy

[实际实现](../factor_optimizer/adapters/factor_assets.py#L286)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, factor_definition_id: str)`。

返回类型：`FactorTaxonomyView`。

### FaFactorIntelligenceProvider.get_health_card

[实际实现](../factor_optimizer/adapters/factor_assets.py#L306)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, factor_definition_id: str, evaluation_ref: str)`。

返回类型：`FactorHealthView`。

### FaFactorIntelligenceProvider.get_diagnoses

[实际实现](../factor_optimizer/adapters/factor_assets.py#L323)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, factor_definition_id: str, health_ref: str)`。

返回类型：`Sequence[DiagnosisView]`。

### create_fa_intelligence_provider

[实际实现](../factor_optimizer/adapters/factor_assets.py#L357)。

Create a FA-backed :class:`FaFactorIntelligenceProvider`.

参数：`(*, get_fa_taxonomy=None, get_fa_health=None, get_fa_diagnoses=None, fa_profiling=None)`。

返回类型：`FaFactorIntelligenceProvider`。

### InMemoryFactorIntelligenceProvider

[实际实现](../factor_optimizer/adapters/factor_assets.py#L380)。

Process-local ``FactorIntelligenceProvider`` for tests and wiring.

### InMemoryFactorIntelligenceProvider.__init__

[实际实现](../factor_optimizer/adapters/factor_assets.py#L390)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`None`。

### InMemoryFactorIntelligenceProvider.with_taxonomy

[实际实现](../factor_optimizer/adapters/factor_assets.py#L395)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, factor_definition_id: str, value: Any)`。

返回类型：`'InMemoryFactorIntelligenceProvider'`。

### InMemoryFactorIntelligenceProvider.with_health_card

[实际实现](../factor_optimizer/adapters/factor_assets.py#L405)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, factor_definition_id: str, value: Any, evaluation_ref: str='ev-in-memory')`。

返回类型：`'InMemoryFactorIntelligenceProvider'`。

### InMemoryFactorIntelligenceProvider.with_diagnosis

[实际实现](../factor_optimizer/adapters/factor_assets.py#L419)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, factor_definition_id: str, value: Any, health_ref: str='hc-in-memory')`。

返回类型：`'InMemoryFactorIntelligenceProvider'`。

### InMemoryFactorIntelligenceProvider.get_taxonomy

[实际实现](../factor_optimizer/adapters/factor_assets.py#L436)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, factor_definition_id: str)`。

返回类型：`FactorTaxonomyView`。

### InMemoryFactorIntelligenceProvider.get_health_card

[实际实现](../factor_optimizer/adapters/factor_assets.py#L442)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, factor_definition_id: str, evaluation_ref: str)`。

返回类型：`FactorHealthView`。

### InMemoryFactorIntelligenceProvider.get_diagnoses

[实际实现](../factor_optimizer/adapters/factor_assets.py#L455)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, factor_definition_id: str, health_ref: str)`。

返回类型：`Sequence[DiagnosisView]`。

### create_in_memory_factor_intelligence_provider

[实际实现](../factor_optimizer/adapters/factor_assets.py#L463)。

Create a research/test-only in-memory provider.

参数：`()`。

返回类型：`InMemoryFactorIntelligenceProvider`。

## factor_optimizer/adapters/factor_engine.py

FactorEngineAdapter: protocol for FE integration (optional dependency).

### FactorEngineAdapter

[实际实现](../factor_optimizer/adapters/factor_engine.py#L7)。

Protocol for FactorEngine integration.

基类：`Protocol`。

### FactorEngineAdapter.compute_canonical_hash

[实际实现](../factor_optimizer/adapters/factor_engine.py#L21)。

Compute FE canonical identity hash.

参数：`(self, factor_definition: Any)`。

返回类型：`str`。

### FactorEngineAdapter.validate_mutation

[实际实现](../factor_optimizer/adapters/factor_engine.py#L36)。

Validate mutation legality through FE.

参数：`(self, mutation: Any, spec: Any)`。

返回类型：`Dict[str, Any]`。

### FactorEngineAdapter.estimate_complexity

[实际实现](../factor_optimizer/adapters/factor_engine.py#L52)。

Estimate factor complexity through FE analyzer.

参数：`(self, factor_definition: Any)`。

返回类型：`Dict[str, Any]`。

### FactorEngineAdapter.get_operator_metadata

[实际实现](../factor_optimizer/adapters/factor_engine.py#L70)。

Get operator metadata for mutation grammar.

参数：`(self, operator_names: Optional[List[str]]=None)`。

返回类型：`Dict[str, Dict[str, Any]]`。

### OptionalDependencyMissing

[实际实现](../factor_optimizer/adapters/factor_engine.py#L88)。

Raised when optional FE dependency is not available.

基类：`Exception`。

### create_fe_adapter

[实际实现](../factor_optimizer/adapters/factor_engine.py#L153)。

Create FE adapter if factor-engine is installed.

参数：`()`。

返回类型：`FactorEngineAdapter`。

## factor_optimizer/adapters/fitness.py

Fitness adapter: dimension/desirability mapping for generalized treatment decisions.

显式导出（含重导出）：`grade_to_desirability`、`grade_lower_is_better`、`GRADE_ORDER_ASC`、`LOWER_IS_BETTER_DIMENSIONS`。

### grade_to_desirability

[实际实现](../factor_optimizer/adapters/fitness.py#L30)。

Map one FA grade to a unit desirability (0..1), None for NONE/unknown.

参数：`(grade: str)`。

返回类型：`Optional[float]`。

### grade_lower_is_better

[实际实现](../factor_optimizer/adapters/fitness.py#L63)。

True when the FA dimension's raw value is lower-is-better.

参数：`(dimension: str)`。

返回类型：`bool`。

## factor_optimizer/adapters/layered_decay.py

TRAIN-frozen research plan for observation-origin twenty-layer decay.

### LayeredDecayPlan

[实际实现](../factor_optimizer/adapters/layered_decay.py#L13)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `half_lives` | `tuple` | `必填/未声明默认` |
| `training_context_ref` | `str` | `必填/未声明默认` |

### LayeredDecayPlan.parameters

[实际实现](../factor_optimizer/adapters/layered_decay.py#L32)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### LayeredDecayPlan.identity

[实际实现](../factor_optimizer/adapters/layered_decay.py#L36)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### LayeredDecayPlan.execute

[实际实现](../factor_optimizer/adapters/layered_decay.py#L40)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, values, *, allow_research=False)`。

## factor_optimizer/adapters/preprocessing.py

Versioned research bridge from smoothing proposals to real FP kernels.

### IneligibleSmoothingRepair

[实际实现](../factor_optimizer/adapters/preprocessing.py#L17)。

A proposed repair cannot bind to the current FP execution authority.

基类：`ValueError`。

### SmoothingRepairPlan

[实际实现](../factor_optimizer/adapters/preprocessing.py#L22)。

Immutable resolved kernel arguments and training-context provenance.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `family` | `str` | `必填/未声明默认` |
| `transform` | `str` | `必填/未声明默认` |
| `parameters` | `Tuple[Tuple[str, object], ...]` | `必填/未声明默认` |
| `training_context_ref` | `str` | `必填/未声明默认` |
| `natural_time_scale` | `float` | `必填/未声明默认` |
| `mapping_version` | `str` | `'smoothing-repair.v2'` |

### SmoothingRepairPlan.identity

[实际实现](../factor_optimizer/adapters/preprocessing.py#L33)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### SmoothingRepairPlan.execute

[实际实现](../factor_optimizer/adapters/preprocessing.py#L39)。

Delegate a long asset_id/date/value panel to FP; retain its lag/NaNs.

参数：`(self, values, *, allow_research: bool=False)`。

### compile_smoothing_repair

[实际实现](../factor_optimizer/adapters/preprocessing.py#L52)。

Resolve a supported conditional candidate to FP parameters.

参数：`(family: str, parameters: Mapping[str, object], *, natural_time_scale: float, training_context_ref: str)`。

返回类型：`SmoothingRepairPlan`。

### compile_admissible_smoothing_grid

[实际实现](../factor_optimizer/adapters/preprocessing.py#L113)。

Compile a small method-balanced grid inside both FO and FP domains.

参数：`(*, natural_time_scale: float, training_context_ref: str, target_time_scales=(3.0, 5.0, 8.0, 10.0, 13.0, 20.0, 30.0, 60.0))`。

返回类型：`Tuple[SmoothingRepairPlan, ...]`。

## factor_optimizer/adapters/quant_evaluator.py

QuantEvaluatorAdapter: protocol for QE integration (optional dependency).

### EvidenceStore

[实际实现](../factor_optimizer/adapters/quant_evaluator.py#L14)。

Storage boundary for evidence bundles shared across adapter instances.

基类：`Protocol`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `evidence_scope` | `str` | `必填/未声明默认` |

### EvidenceStore.put

[实际实现](../factor_optimizer/adapters/quant_evaluator.py#L19)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, evidence_id: str, evidence: Dict[str, Any])`。

返回类型：`None`。

### EvidenceStore.get

[实际实现](../factor_optimizer/adapters/quant_evaluator.py#L22)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, evidence_id: str)`。

返回类型：`Optional[Dict[str, Any]]`。

### InMemoryEvidenceStore

[实际实现](../factor_optimizer/adapters/quant_evaluator.py#L26)。

Process-local shared store useful for research and tests.

### InMemoryEvidenceStore.__init__

[实际实现](../factor_optimizer/adapters/quant_evaluator.py#L31)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### InMemoryEvidenceStore.put

[实际实现](../factor_optimizer/adapters/quant_evaluator.py#L34)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, evidence_id: str, evidence: Dict[str, Any])`。

返回类型：`None`。

### InMemoryEvidenceStore.get

[实际实现](../factor_optimizer/adapters/quant_evaluator.py#L37)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, evidence_id: str)`。

返回类型：`Optional[Dict[str, Any]]`。

### QuantEvaluatorAdapter

[实际实现](../factor_optimizer/adapters/quant_evaluator.py#L43)。

Protocol for QuantEvaluator integration.

基类：`Protocol`。

### QuantEvaluatorAdapter.evaluate

[实际实现](../factor_optimizer/adapters/quant_evaluator.py#L56)。

Submit evaluation request to QE.

参数：`(self, factor_batch: Any, labels: Any, metrics: Optional[List[str]]=None, context: Optional[Dict[str, Any]]=None, *, split_ref: Any=None, backend: Any=None, gpu_policy: Any=None, tier: Optional[str]=None, cost_budget: Optional[float]=None, metric_parameters: Optional[Dict[str, Dict[str, Any]]]=None, request_metadata: Optional[Dict[str, Any]]=None, metric_instances: Optional[Sequence[Any]]=None, scenario_inputs: Optional[Mapping[str, Any]]=None)`。

返回类型：`Dict[str, Any]`。

### QuantEvaluatorAdapter.get_evidence

[实际实现](../factor_optimizer/adapters/quant_evaluator.py#L91)。

Retrieve full evidence bundle by evaluation ID.

参数：`(self, evaluation_id: str)`。

返回类型：`Dict[str, Any]`。

### QuantEvaluatorAdapter.list_metrics

[实际实现](../factor_optimizer/adapters/quant_evaluator.py#L107)。

List available metrics from QE catalog.

参数：`(self, tier: Optional[str]=None)`。

返回类型：`List[Dict[str, Any]]`。

### OptionalDependencyMissing

[实际实现](../factor_optimizer/adapters/quant_evaluator.py#L127)。

Raised when optional QE dependency is not available.

基类：`Exception`。

### create_qe_adapter

[实际实现](../factor_optimizer/adapters/quant_evaluator.py#L133)。

Create a real QE adapter; production remains fail-closed until implemented.

参数：`(*, execution_mode: ExecutionMode=ExecutionMode.RESEARCH_ONLY, evidence_store: Optional[EvidenceStore]=None)`。

返回类型：`QuantEvaluatorAdapter`。

### create_mock_qe_adapter

[实际实现](../factor_optimizer/adapters/quant_evaluator.py#L418)。

Create the mock adapter only for explicit research use.

参数：`(*, execution_mode: ExecutionMode=ExecutionMode.RESEARCH_ONLY)`。

返回类型：`QuantEvaluatorAdapter`。

## factor_optimizer/adapters/repair_execution.py

Versioned, research-only execution plans for value-level repair families.

### IneligibleValueRepair

[实际实现](../factor_optimizer/adapters/repair_execution.py#L16)。

A valid registry candidate has no exact executable value primitive.

基类：`ValueError`。

### ValueRepairPlan

[实际实现](../factor_optimizer/adapters/repair_execution.py#L34)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `family` | `str` | `必填/未声明默认` |
| `transform` | `str` | `必填/未声明默认` |
| `parameters` | `Tuple[Tuple[str, object], ...]` | `必填/未声明默认` |
| `training_context_ref` | `str` | `必填/未声明默认` |
| `natural_time_scale` | `float` | `必填/未声明默认` |
| `mapping_version` | `str` | `'value-repair.v3'` |

### ValueRepairPlan.identity

[实际实现](../factor_optimizer/adapters/repair_execution.py#L43)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### ValueRepairPlan.execute

[实际实现](../factor_optimizer/adapters/repair_execution.py#L54)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, values, *, allow_research: bool=False)`。

返回类型：`pd.Series`。

### compile_value_repair

[实际实现](../factor_optimizer/adapters/repair_execution.py#L89)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(family: str, parameters: Mapping[str, object], *, natural_time_scale: float, training_context_ref: str)`。

返回类型：`ValueRepairPlan`。

## factor_optimizer/capabilities.py

Truthful runtime capability metadata for factor_optimizer.

显式导出（含重导出）：`ExecutionMode`、`CapabilityStatus`、`ProductionCapability`、`PRODUCTION_CAPABILITY`、`require_production_capability`。

### ExecutionMode

[实际实现](../factor_optimizer/capabilities.py#L8)。

Supported execution intent.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `RESEARCH_ONLY` | `类常量/枚举` | `'research_only'` |
| `PRODUCTION` | `类常量/枚举` | `'production'` |

### CapabilityStatus

[实际实现](../factor_optimizer/capabilities.py#L15)。

Availability state for a capability.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `SUPPORTED` | `类常量/枚举` | `'supported'` |
| `RESEARCH_ONLY` | `类常量/枚举` | `'research_only'` |
| `UNSUPPORTED` | `类常量/枚举` | `'unsupported'` |

### ProductionCapability

[实际实现](../factor_optimizer/capabilities.py#L24)。

Machine-readable package production readiness.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `status` | `CapabilityStatus` | `必填/未声明默认` |
| `supported` | `bool` | `必填/未声明默认` |
| `blockers` | `Tuple[str, ...]` | `必填/未声明默认` |

### ProductionCapability.as_dict

[实际实现](../factor_optimizer/capabilities.py#L31)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, object]`。

### require_production_capability

[实际实现](../factor_optimizer/capabilities.py#L50)。

Fail closed until all upstream production contracts are implemented.

参数：`()`。

返回类型：`None`。

## factor_optimizer/complexity/__init__.py

Complexity estimation and budget tracking.

显式导出（含重导出）：`ComplexityProfile`、`ComplexityEstimator`、`ComplexityBudget`、`BudgetTracker`、`create_default_budget`。

## factor_optimizer/complexity/budget.py

Complexity budget tracking and enforcement.

### ComplexityBudget

[实际实现](../factor_optimizer/complexity/budget.py#L11)。

Complexity budget constraints for optimization runs.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `max_cost` | `Optional[float]` | `None` |
| `max_operator_count` | `Optional[int]` | `None` |
| `max_lookback_periods` | `Optional[int]` | `None` |
| `max_depth` | `Optional[int]` | `None` |
| `max_stateful_operators` | `Optional[int]` | `None` |
| `max_memory_mb` | `Optional[float]` | `None` |
| `strict` | `bool` | `True` |

### ComplexityBudget.check

[实际实现](../factor_optimizer/complexity/budget.py#L36)。

Check if profile is within budget.

参数：`(self, profile: ComplexityProfile)`。

返回类型：`Dict[str, bool]`。

### ComplexityBudget.is_within_budget

[实际实现](../factor_optimizer/complexity/budget.py#L76)。

Check if profile satisfies all budget constraints.

参数：`(self, profile: ComplexityProfile)`。

返回类型：`bool`。

### ComplexityBudget.enforce

[实际实现](../factor_optimizer/complexity/budget.py#L89)。

Enforce budget constraints, raising exception if violated.

参数：`(self, profile: ComplexityProfile)`。

返回类型：`None`。

### BudgetTracker

[实际实现](../factor_optimizer/complexity/budget.py#L110)。

Track complexity budget usage during optimization.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `budget` | `ComplexityBudget` | `必填/未声明默认` |
| `history` | `List[ComplexityProfile]` | `field(default_factory=list)` |
| `violations` | `List[Dict[str, any]]` | `field(default_factory=list)` |

### BudgetTracker.record

[实际实现](../factor_optimizer/complexity/budget.py#L122)。

Record complexity profile and check against budget.

参数：`(self, profile: ComplexityProfile, candidate_id: Optional[str]=None)`。

返回类型：`bool`。

### BudgetTracker.enforce

[实际实现](../factor_optimizer/complexity/budget.py#L148)。

Record and enforce budget constraint.

参数：`(self, profile: ComplexityProfile, candidate_id: Optional[str]=None)`。

返回类型：`None`。

### BudgetTracker.stats

[实际实现](../factor_optimizer/complexity/budget.py#L164)。

Compute statistics about budget utilization.

参数：`(self)`。

返回类型：`Dict[str, any]`。

### create_default_budget

[实际实现](../factor_optimizer/complexity/budget.py#L204)。

Create sensible default budget.

参数：`(cost_multiplier: float=1.0, operator_multiplier: float=1.0, strict: bool=True)`。

返回类型：`ComplexityBudget`。

## factor_optimizer/complexity/profile.py

ComplexityProfile: adapter projection of FE complexity analysis.

### ComplexityProfile

[实际实现](../factor_optimizer/complexity/profile.py#L9)。

Complexity estimate for a factor or mutation.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `operator_count` | `int` | `0` |
| `max_depth` | `int` | `0` |
| `lookback_periods` | `int` | `0` |
| `stateful_operators` | `int` | `0` |
| `cross_sectional_operators` | `int` | `0` |
| `nonlinear_operators` | `int` | `0` |
| `estimated_cost` | `float` | `0.0` |
| `domains` | `list[str]` | `field(default_factory=list)` |
| `sources` | `list[str]` | `field(default_factory=list)` |
| `estimated_latency_ms` | `Optional[float]` | `None` |
| `memory_estimate` | `Optional[float]` | `None` |
| `metadata` | `Dict[str, Any]` | `field(default_factory=dict)` |

### ComplexityProfile.to_dict

[实际实现](../factor_optimizer/complexity/profile.py#L43)。

Serialize to dictionary.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### ComplexityProfile.from_dict

[实际实现](../factor_optimizer/complexity/profile.py#L61)。

Deserialize from dictionary (fail-closed on unknown fields).

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'ComplexityProfile'`。

### ComplexityProfile.is_within_budget

[实际实现](../factor_optimizer/complexity/profile.py#L74)。

Check if complexity is within budget.

参数：`(self, max_cost: float)`。

返回类型：`bool`。

### ComplexityEstimator

[实际实现](../factor_optimizer/complexity/profile.py#L79)。

Estimate complexity through FE adapter.

### ComplexityEstimator.__init__

[实际实现](../factor_optimizer/complexity/profile.py#L87)。

Initialize estimator.

参数：`(self, fe_adapter=None)`。

### ComplexityEstimator.estimate

[实际实现](../factor_optimizer/complexity/profile.py#L96)。

Estimate complexity of a factor definition.

参数：`(self, factor_definition: Any)`。

返回类型：`ComplexityProfile`。

### ComplexityEstimator.estimate_mutation_delta

[实际实现](../factor_optimizer/complexity/profile.py#L144)。

Estimate complexity change from a mutation.

参数：`(self, parent_profile: ComplexityProfile, mutation_type: str, parameters: Dict[str, Any])`。

返回类型：`ComplexityProfile`。

## factor_optimizer/contracts/__init__.py

Contracts for mutation proposals, search budgets, and trials.

显式导出（含重导出）：`CandidateMutation`、`ObjectiveSpec`、`ObjectiveDirection`、`OBJECTIVE_DIRECTIONS`、`DEFAULT_METRIC_NAME`、`SearchBudget`、`BudgetTracker`、`CampaignStateError`、`DurableBudgetTracker`、`Reservation`、`SQLiteCampaignStore`、`HorizonDiagnosis`、`RetentionEvidence`、`SplitEvidenceRef`、`SplitPurpose`、`diagnose_horizon_curve`、`retention_evidence`、`Trial`、`TrialStatus`、`SplitPlan`、`EvaluationProtocol`、`SelectedExecutionSpec`、`SealedTestEvaluationOutcome`、`TrialValidatorIdentity`、`MutationGrammarValidator`、`LibrarySnapshotRef`、`EVIDENCE_SCHEMA_VERSION`、`RAW_TREATMENT_KIND`、`IntegrityCheckResult`、`TreatmentIntegrityEvidence`、`TreatmentIntegrityStatus`、`build_integrity_evidence`、`describe_integrity_problem`、`digest_value`、`require_integrity_evidence`、`EvidenceStatus`、`EvidenceTier`、`EvidenceValue`、`EvidenceSeries`、`EVIDENCE_TIER_ORDER`、`QE_STATUS_TOKENS`、`status_of`、`FactorFitnessSpec`、`CandidateFitnessArtifact`、`MIN_EVIDENCE_TIER_LEVELS`。

## factor_optimizer/contracts/budget_extension.py

FO-P1-24 budget-resume authorization contract.

显式导出（含重导出）：`BudgetExtensionAuthorization`。

### BudgetExtensionAuthorization

[实际实现](../factor_optimizer/contracts/budget_extension.py#L25)。

Signed authorization to extend a session's budget (FO-P1-24).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `search_session_id` | `str` | `必填/未声明默认` |
| `reason` | `str` | `必填/未声明默认` |
| `old_budget` | `SearchBudget` | `必填/未声明默认` |
| `new_budget` | `SearchBudget` | `必填/未声明默认` |
| `actor` | `str` | `必填/未声明默认` |
| `timestamp` | `datetime` | `必填/未声明默认` |
| `signature` | `Optional[str]` | `None` |

### BudgetExtensionAuthorization.content_hash

[实际实现](../factor_optimizer/contracts/budget_extension.py#L69)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### BudgetExtensionAuthorization.verify

[实际实现](../factor_optimizer/contracts/budget_extension.py#L72)。

Fail closed unless the content hash matches the signed payload.

参数：`(self)`。

返回类型：`None`。

### BudgetExtensionAuthorization.to_dict

[实际实现](../factor_optimizer/contracts/budget_extension.py#L81)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### BudgetExtensionAuthorization.from_dict

[实际实现](../factor_optimizer/contracts/budget_extension.py#L94)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'BudgetExtensionAuthorization'`。

## factor_optimizer/contracts/campaign_store.py

Durable SQLite control-plane state for FO campaigns.

### CampaignStateError

[实际实现](../factor_optimizer/contracts/campaign_store.py#L20)。

Raised when durable campaign state rejects an unsafe transition.

基类：`RuntimeError`。

### Reservation

[实际实现](../factor_optimizer/contracts/campaign_store.py#L25)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `campaign_id` | `str` | `必填/未声明默认` |
| `attempt_id` | `str` | `必填/未声明默认` |
| `state` | `str` | `必填/未声明默认` |
| `estimated_cost` | `float` | `必填/未声明默认` |
| `lease_expires_at` | `float` | `必填/未声明默认` |

### SQLiteCampaignStore

[实际实现](../factor_optimizer/contracts/campaign_store.py#L33)。

Transactional campaign budget and sealed-access store.

### SQLiteCampaignStore.__init__

[实际实现](../factor_optimizer/contracts/campaign_store.py#L42)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, path: str \| Path)`。

### SQLiteCampaignStore.create_campaign

[实际实现](../factor_optimizer/contracts/campaign_store.py#L160)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, campaign_id: str, *, max_evaluations: int, max_cost: float)`。

返回类型：`None`。

### SQLiteCampaignStore.reserve

[实际实现](../factor_optimizer/contracts/campaign_store.py#L183)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, campaign_id: str, attempt_id: str, estimated_cost: float, *, lease_seconds: float=60)`。

返回类型：`bool`。

### SQLiteCampaignStore.start

[实际实现](../factor_optimizer/contracts/campaign_store.py#L228)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, campaign_id: str, attempt_id: str)`。

返回类型：`None`。

### SQLiteCampaignStore.release

[实际实现](../factor_optimizer/contracts/campaign_store.py#L238)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, campaign_id: str, attempt_id: str, *, expected_reserved_cost: Optional[float]=None)`。

返回类型：`None`。

### SQLiteCampaignStore.settle

[实际实现](../factor_optimizer/contracts/campaign_store.py#L244)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, campaign_id: str, attempt_id: str, actual_cost: float, *, expected_reserved_cost: Optional[float]=None)`。

返回类型：`None`。

### SQLiteCampaignStore.freeze_candidate_set

[实际实现](../factor_optimizer/contracts/campaign_store.py#L301)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, *, campaign_id: str, candidate_set_hash: str, dataset_identity: str, split_id: str, purpose: str, profile_hash: str)`。

返回类型：`str`。

### SQLiteCampaignStore.begin_test_attempt

[实际实现](../factor_optimizer/contracts/campaign_store.py#L340)。

Start the one logical evaluation, or return its immutable result.

参数：`(self, scope_hash: str)`。

返回类型：`Optional[Mapping[str, Any]]`。

### SQLiteCampaignStore.mark_infrastructure_failure

[实际实现](../factor_optimizer/contracts/campaign_store.py#L362)。

Release only an attempt that failed before physical test exposure.

参数：`(self, scope_hash: str, attempt_count: int)`。

返回类型：`None`。

### SQLiteCampaignStore.mark_test_exposed

[实际实现](../factor_optimizer/contracts/campaign_store.py#L378)。

Atomically burn the sole physical test read before store access.

参数：`(self, scope_hash: str, attempt_count: int)`。

返回类型：`None`。

### SQLiteCampaignStore.complete_test

[实际实现](../factor_optimizer/contracts/campaign_store.py#L394)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, scope_hash: str, attempt_count: int, result_ref: str, result: Mapping[str, Any])`。

返回类型：`None`。

### SQLiteCampaignStore.sealed_state

[实际实现](../factor_optimizer/contracts/campaign_store.py#L421)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, scope_hash: str)`。

返回类型：`Mapping[str, Any]`。

### SQLiteCampaignStore.budget_state

[实际实现](../factor_optimizer/contracts/campaign_store.py#L428)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, campaign_id: str)`。

返回类型：`Mapping[str, Any]`。

### SQLiteCampaignStore.record_repair_failure

[实际实现](../factor_optimizer/contracts/campaign_store.py#L442)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, *, effective_spec_hash: str, evaluation_intent_hash: str, context_hash: str, reason: str, retry_condition: Optional[str]=None, payload: Optional[Mapping[str, Any]]=None)`。

返回类型：`None`。

### SQLiteCampaignStore.repair_failure

[实际实现](../factor_optimizer/contracts/campaign_store.py#L459)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, *, effective_spec_hash: str, evaluation_intent_hash: str, context_hash: str)`。

返回类型：`Optional[Mapping[str, Any]]`。

### SQLiteCampaignStore.freeze_supervised_parameter

[实际实现](../factor_optimizer/contracts/campaign_store.py#L469)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, campaign_id: str, state)`。

返回类型：`None`。

### SQLiteCampaignStore.supervised_parameter

[实际实现](../factor_optimizer/contracts/campaign_store.py#L490)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, campaign_id: str, parent_factor_id: str, repair_family: str)`。

### SQLiteCampaignStore.record_hypothesis_attempt

[实际实现](../factor_optimizer/contracts/campaign_store.py#L503)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, *, campaign_id: str, proposal_id: str, evaluation_intent_hash: str, horizon: int, effective_spec_hash: Optional[str]=None, executed: bool=False, has_pvalue: bool=False)`。

返回类型：`None`。

### SQLiteCampaignStore.hypothesis_family_summary

[实际实现](../factor_optimizer/contracts/campaign_store.py#L533)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, campaign_id: str)`。

返回类型：`Mapping[str, int]`。

### SQLiteCampaignStore.require_complete_fdr_family

[实际实现](../factor_optimizer/contracts/campaign_store.py#L549)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, campaign_id: str, submitted_pvalue_count: int)`。

返回类型：`None`。

### SQLiteCampaignStore.bind_hypothesis_family

[实际实现](../factor_optimizer/contracts/campaign_store.py#L556)。

Persist a QE family only when it exactly resolves a sealed FO ledger.

参数：`(self, family, ledger)`。

返回类型：`str`。

### DurableBudgetTracker

[实际实现](../factor_optimizer/contracts/campaign_store.py#L637)。

BudgetTracker-compatible facade backed by SQLite transactions.

### DurableBudgetTracker.__init__

[实际实现](../factor_optimizer/contracts/campaign_store.py#L640)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, store: SQLiteCampaignStore, campaign_id: str, budget)`。

### DurableBudgetTracker.evaluations_used

[实际实现](../factor_optimizer/contracts/campaign_store.py#L651)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### DurableBudgetTracker.cost_used

[实际实现](../factor_optimizer/contracts/campaign_store.py#L653)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### DurableBudgetTracker.evaluations_reserved

[实际实现](../factor_optimizer/contracts/campaign_store.py#L655)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### DurableBudgetTracker.cost_reserved

[实际实现](../factor_optimizer/contracts/campaign_store.py#L657)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### DurableBudgetTracker.reserve_evaluation

[实际实现](../factor_optimizer/contracts/campaign_store.py#L659)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, cost, attempt_id=None)`。

### DurableBudgetTracker.commit_evaluation

[实际实现](../factor_optimizer/contracts/campaign_store.py#L663)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, reserved_cost, actual_cost, attempt_id=None)`。

### DurableBudgetTracker.start_evaluation

[实际实现](../factor_optimizer/contracts/campaign_store.py#L668)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, attempt_id=None)`。

### DurableBudgetTracker.release_evaluation

[实际实现](../factor_optimizer/contracts/campaign_store.py#L672)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, reserved_cost, attempt_id=None)`。

### DurableBudgetTracker.has_reservation

[实际实现](../factor_optimizer/contracts/campaign_store.py#L677)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, attempt_id=None)`。

### DurableBudgetTracker.remaining_cost

[实际实现](../factor_optimizer/contracts/campaign_store.py#L689)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### DurableBudgetTracker.remaining_evaluations

[实际实现](../factor_optimizer/contracts/campaign_store.py#L693)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### DurableBudgetTracker.can_propose_trial

[实际实现](../factor_optimizer/contracts/campaign_store.py#L697)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### DurableBudgetTracker.record_trial

[实际实现](../factor_optimizer/contracts/campaign_store.py#L698)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### DurableBudgetTracker.can_call_llm

[实际实现](../factor_optimizer/contracts/campaign_store.py#L699)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### DurableBudgetTracker.record_llm_call

[实际实现](../factor_optimizer/contracts/campaign_store.py#L700)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### DurableBudgetTracker.is_exhausted

[实际实现](../factor_optimizer/contracts/campaign_store.py#L701)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### DurableBudgetTracker.to_dict

[实际实现](../factor_optimizer/contracts/campaign_store.py#L703)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

## factor_optimizer/contracts/candidate_mutation.py

CandidateMutation: typed mutation proposal with provenance and complexity estimate.

### CandidateMutation

[实际实现](../factor_optimizer/contracts/candidate_mutation.py#L11)。

A proposed mutation to a parent factor.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `mutation_id` | `str` | `必填/未声明默认` |
| `mutation_spec_version` | `str` | `必填/未声明默认` |
| `parent_factor_ids` | `List[str]` | `必填/未声明默认` |
| `mutation_type` | `str` | `必填/未声明默认` |
| `parameters` | `Dict[str, Any]` | `必填/未声明默认` |
| `mechanism_hypothesis` | `Optional[str]` | `None` |
| `expected_signatures` | `List[str]` | `field(default_factory=list)` |
| `complexity_estimate` | `Optional[Dict[str, Any]]` | `None` |
| `lineage_ref` | `Optional[str]` | `None` |
| `trial_ref` | `Optional[str]` | `None` |
| `created_at` | `Optional[datetime]` | `None` |
| `producer` | `str` | `'factor_optimizer'` |
| `producer_version` | `str` | `'0.1.0'` |

### CandidateMutation.to_dict

[实际实现](../factor_optimizer/contracts/candidate_mutation.py#L60)。

Serialize to dictionary for storage/transmission.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### CandidateMutation.from_dict

[实际实现](../factor_optimizer/contracts/candidate_mutation.py#L79)。

Deserialize from dictionary.

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'CandidateMutation'`。

## factor_optimizer/contracts/evaluation_artifact.py

TrialEvaluationArtifact: typed, content-hashed evaluation output (FO-P1-25).

显式导出（含重导出）：`TrialEvaluationArtifact`、`ObjectiveValue`、`EvaluationStatus`、`normalize_evaluation_result`。

### EvaluationStatus

[实际实现](../factor_optimizer/contracts/evaluation_artifact.py#L27)。

The outcome of a single evaluation (FO-P1-25).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `COMPLETE` | `类常量/枚举` | `'complete'` |
| `FAILED` | `类常量/枚举` | `'failed'` |
| `PRUNED` | `类常量/枚举` | `'pruned'` |

### ObjectiveValue

[实际实现](../factor_optimizer/contracts/evaluation_artifact.py#L51)。

A single objective value with its direction/spec reference.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `name` | `str` | `必填/未声明默认` |
| `value` | `float` | `必填/未声明默认` |
| `objective_spec_ref` | `Optional[str]` | `None` |

### TrialEvaluationArtifact

[实际实现](../factor_optimizer/contracts/evaluation_artifact.py#L68)。

Typed, attributable output of a single evaluation (FO-P1-25).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `trial_id` | `str` | `必填/未声明默认` |
| `objective_values` | `List[Dict[str, Any]]` | `必填/未声明默认` |
| `objective_spec_ref` | `Optional[str]` | `None` |
| `split_ref` | `Optional[str]` | `None` |
| `fidelity` | `int` | `0` |
| `evidence_ref` | `Optional[str]` | `None` |
| `compute_cost` | `float` | `1.0` |
| `promotion_evidence` | `Optional[Dict[str, Any]]` | `None` |
| `status` | `str` | `EvaluationStatus.COMPLETE` |
| `metadata` | `Dict[str, Any]` | `field(default_factory=dict)` |
| `treatment_integrity_evidence` | `Optional[TreatmentIntegrityEvidence]` | `None` |

### TrialEvaluationArtifact.primary_objective_value

[实际实现](../factor_optimizer/contracts/evaluation_artifact.py#L143)。

The primary (first) objective value.

参数：`(self)`。

返回类型：`float`。

### TrialEvaluationArtifact.integrity_evidence

[实际实现](../factor_optimizer/contracts/evaluation_artifact.py#L148)。

The measured treatment-integrity evidence, when attached.

参数：`(self)`。

返回类型：`Optional[TreatmentIntegrityEvidence]`。

### TrialEvaluationArtifact.require_passing_integrity

[实际实现](../factor_optimizer/contracts/evaluation_artifact.py#L152)。

R55 P0-9 fail-closed gate: the artifact's integrity evidence.

参数：`(self)`。

返回类型：`TreatmentIntegrityEvidence`。

### TrialEvaluationArtifact.content_hash

[实际实现](../factor_optimizer/contracts/evaluation_artifact.py#L189)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### TrialEvaluationArtifact.verify

[实际实现](../factor_optimizer/contracts/evaluation_artifact.py#L192)。

Fail closed if the artifact was altered after construction.

参数：`(self)`。

返回类型：`None`。

### TrialEvaluationArtifact.to_dict

[实际实现](../factor_optimizer/contracts/evaluation_artifact.py#L197)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### TrialEvaluationArtifact.from_dict

[实际实现](../factor_optimizer/contracts/evaluation_artifact.py#L218)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'TrialEvaluationArtifact'`。

### normalize_evaluation_result

[实际实现](../factor_optimizer/contracts/evaluation_artifact.py#L252)。

Normalize a legacy magic dict (or an existing artifact) into an artifact.

参数：`(trial_id: str, result: Any, *, objective_name: str='score', objective_spec_ref: Optional[str]=None, split_ref: Optional[str]=None, fidelity: int=0)`。

返回类型：`TrialEvaluationArtifact`。

## factor_optimizer/contracts/evidence_value.py

Evidence-bound metric values: missing evidence is never a fabricated 0.0 (R61-FI-030).

显式导出（含重导出）：`EvidenceStatus`、`EvidenceTier`、`EvidenceValue`、`EvidenceSeries`、`status_of`、`QE_STATUS_TOKENS`、`EVIDENCE_TIER_ORDER`。

### EvidenceStatus

[实际实现](../factor_optimizer/contracts/evidence_value.py#L69)。

Whether numeric evidence for a metric exists and can be trusted.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `COMPUTED` | `类常量/枚举` | `'computed'` |
| `NOT_COMPUTED` | `类常量/枚举` | `'not_computed'` |
| `UNAVAILABLE` | `类常量/枚举` | `'unavailable'` |
| `UNSUPPORTED` | `类常量/枚举` | `'unsupported'` |
| `INSUFFICIENT_DATA` | `类常量/枚举` | `'insufficient_data'` |
| `LABEL_NOT_MATURE` | `类常量/枚举` | `'label_not_mature'` |
| `INVALID_EVIDENCE` | `类常量/枚举` | `'invalid_evidence'` |
| `FAILED` | `类常量/枚举` | `'failed'` |

### EvidenceStatus.from_value

[实际实现](../factor_optimizer/contracts/evidence_value.py#L87)。

Resolve a str/enum value to an EvidenceStatus (fail closed).

参数：`(cls, value: object)`。

返回类型：`'EvidenceStatus'`。

### EvidenceStatus.computed

[实际实现](../factor_optimizer/contracts/evidence_value.py#L105)。

True only when the status is exactly COMPUTED.

参数：`(self)`。

返回类型：`bool`。

### EvidenceTier

[实际实现](../factor_optimizer/contracts/evidence_value.py#L122)。

The evidence-quality ladder a candidate's value stands on (plan §21/E2).

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `POINT_ESTIMATE_ONLY` | `类常量/枚举` | `'POINT_ESTIMATE_ONLY'` |
| `VALIDATION_SERIES` | `类常量/枚举` | `'VALIDATION_SERIES'` |
| `BOOTSTRAP_CONFIDENCE` | `类常量/枚举` | `'BOOTSTRAP_CONFIDENCE'` |
| `MULTIPLE_TESTING_ADJUSTED` | `类常量/枚举` | `'MULTIPLE_TESTING_ADJUSTED'` |
| `SEALED_TEST_CONFIRMED` | `类常量/枚举` | `'SEALED_TEST_CONFIRMED'` |

### EvidenceTier.from_value

[实际实现](../factor_optimizer/contracts/evidence_value.py#L143)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, value: object)`。

返回类型：`'EvidenceTier'`。

### EvidenceTier.rank

[实际实现](../factor_optimizer/contracts/evidence_value.py#L160)。

Ordinal rank (0 weakest .. 4 strongest).

参数：`(self)`。

返回类型：`int`。

### EvidenceTier.meets

[实际实现](../factor_optimizer/contracts/evidence_value.py#L164)。

True when this tier is at least as strong as ``minimum``.

参数：`(self, minimum: 'EvidenceTier')`。

返回类型：`bool`。

### EvidenceValue

[实际实现](../factor_optimizer/contracts/evidence_value.py#L170)。

A single metric value bound to its evidence status.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `metric_id` | `str` | `必填/未声明默认` |
| `value` | `Optional[float]` | `None` |
| `status` | `EvidenceStatus` | `EvidenceStatus.NOT_COMPUTED` |
| `tier` | `EvidenceTier` | `EvidenceTier.POINT_ESTIMATE_ONLY` |
| `source_ref` | `str` | `''` |
| `reason` | `str` | `''` |

### EvidenceValue.present

[实际实现](../factor_optimizer/contracts/evidence_value.py#L230)。

True when a numeric value is available (status COMPUTED).

参数：`(self)`。

返回类型：`bool`。

### EvidenceValue.to_dict

[实际实现](../factor_optimizer/contracts/evidence_value.py#L234)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`dict`。

### EvidenceValue.from_dict

[实际实现](../factor_optimizer/contracts/evidence_value.py#L245)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Mapping)`。

返回类型：`'EvidenceValue'`。

### EvidenceSeries

[实际实现](../factor_optimizer/contracts/evidence_value.py#L259)。

Bootstrap / validation resampled series with an explicit empty state.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `samples` | `Tuple[float, ...]` | `()` |
| `status` | `EvidenceStatus` | `EvidenceStatus.NOT_COMPUTED` |

### EvidenceSeries.present

[实际实现](../factor_optimizer/contracts/evidence_value.py#L300)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`bool`。

### EvidenceSeries.to_dict

[实际实现](../factor_optimizer/contracts/evidence_value.py#L303)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`dict`。

### EvidenceSeries.from_dict

[实际实现](../factor_optimizer/contracts/evidence_value.py#L310)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Mapping)`。

返回类型：`'EvidenceSeries'`。

### status_of

[实际实现](../factor_optimizer/contracts/evidence_value.py#L317)。

Coerce a QE ``EvidenceStatus`` / FA status string into our enum.

参数：`(value: object)`。

返回类型：`EvidenceStatus`。

## factor_optimizer/contracts/factor_fitness.py

FactorFitnessSpec + CandidateFitnessArtifact (R61-FI-031 / plan §21).

显式导出（含重导出）：`FactorFitnessSpec`、`CandidateFitnessArtifact`、`MIN_EVIDENCE_TIER_LEVELS`。

### FactorFitnessSpec

[实际实现](../factor_optimizer/contracts/factor_fitness.py#L91)。

Frozen contract naming every policy id the decision pipeline applies.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `evidence_profile_id` | `str` | `'QE_EVIDENCE_PROFILE'` |
| `required_health_dimensions` | `tuple[str, ...]` | `()` |
| `hard_gate_policy_id` | `str` | `''` |
| `dimension_floor_policy_id` | `str` | `''` |
| `raw_relative_policy_id` | `str` | `''` |
| `uncertainty_policy_id` | `str` | `''` |
| `multiplicity_policy_id` | `str` | `''` |
| `complexity_policy_id` | `str` | `''` |
| `winner_policy_id` | `str` | `''` |
| `minimum_evidence_tier` | `EvidenceTier` | `EvidenceTier.POINT_ESTIMATE_ONLY` |

### FactorFitnessSpec.requires_evidence_tier

[实际实现](../factor_optimizer/contracts/factor_fitness.py#L151)。

True when ``tier`` meets this spec's minimum evidence tier.

参数：`(self, tier: EvidenceTier)`。

返回类型：`bool`。

### FactorFitnessSpec.to_dict

[实际实现](../factor_optimizer/contracts/factor_fitness.py#L155)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`dict`。

### FactorFitnessSpec.from_dict

[实际实现](../factor_optimizer/contracts/factor_fitness.py#L170)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Mapping)`。

返回类型：`'FactorFitnessSpec'`。

### CandidateFitnessArtifact

[实际实现](../factor_optimizer/contracts/factor_fitness.py#L188)。

Immutable per-candidate fitness record the pipeline consumes/produces.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `trial_id` | `str` | `必填/未声明默认` |
| `evaluation_ref` | `str` | `必填/未声明默认` |
| `health_card_ref` | `str` | `必填/未声明默认` |
| `raw_relative_deltas` | `Mapping[str, Optional[float]]` | `field(default_factory=dict)` |
| `evidence_tier` | `EvidenceTier` | `EvidenceTier.POINT_ESTIMATE_ONLY` |
| `status` | `EvidenceStatus` | `EvidenceStatus.COMPUTED` |
| `dimension_scores` | `Mapping[str, float]` | `field(default_factory=dict)` |

### CandidateFitnessArtifact.delta

[实际实现](../factor_optimizer/contracts/factor_fitness.py#L257)。

Raw-relative delta for one dimension/metric (None when missing).

参数：`(self, name: str)`。

返回类型：`Optional[float]`。

### CandidateFitnessArtifact.meets_evidence_tier

[实际实现](../factor_optimizer/contracts/factor_fitness.py#L261)。

True when this artifact's tier meets ``minimum``.

参数：`(self, minimum: EvidenceTier)`。

返回类型：`bool`。

### CandidateFitnessArtifact.to_dict

[实际实现](../factor_optimizer/contracts/factor_fitness.py#L265)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`dict`。

### CandidateFitnessArtifact.from_dict

[实际实现](../factor_optimizer/contracts/factor_fitness.py#L277)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Mapping)`。

返回类型：`'CandidateFitnessArtifact'`。

## factor_optimizer/contracts/library_snapshot_ref.py

Library-snapshot reference binding for FO (DLIB-FO-008).

显式导出（含重导出）：`LibrarySnapshotRef`。

### LibrarySnapshotRef

[实际实现](../factor_optimizer/contracts/library_snapshot_ref.py#L38)。

Minimal, immutable reference to the FA library snapshot a search ran on.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `library_version_ref` | `str` | `必填/未声明默认` |
| `cluster_set_version_ref` | `Optional[str]` | `None` |
| `factor_set_version_ref` | `Optional[str]` | `None` |
| `snapshot_ref` | `Optional[str]` | `None` |
| `universe_ref` | `Optional[str]` | `None` |

### LibrarySnapshotRef.content_hash

[实际实现](../factor_optimizer/contracts/library_snapshot_ref.py#L80)。

sha256 over the ref fields — stable, order-independent.

参数：`(self)`。

返回类型：`str`。

### LibrarySnapshotRef.to_dict

[实际实现](../factor_optimizer/contracts/library_snapshot_ref.py#L95)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`dict`。

### LibrarySnapshotRef.from_dict

[实际实现](../factor_optimizer/contracts/library_snapshot_ref.py#L106)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: dict)`。

返回类型：`'LibrarySnapshotRef'`。

## factor_optimizer/contracts/multiplicity.py

MultiplicityArtifact: full-proposal-process multiplicity tracking (DLIB-FO-006).

显式导出（含重导出）：`MultiplicityArtifact`。

### MultiplicityArtifact

[实际实现](../factor_optimizer/contracts/multiplicity.py#L25)。

Tracks the full proposal process for multiple-testing correction.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `search_session_id` | `str` | `必填/未声明默认` |
| `total_proposals` | `int` | `必填/未声明默认` |
| `parse_failures` | `int` | `必填/未声明默认` |
| `duplicates` | `int` | `必填/未声明默认` |
| `valid_evaluated` | `int` | `必填/未声明默认` |
| `failed_evaluations` | `int` | `必填/未声明默认` |
| `illegal` | `int` | `0` |
| `created_at` | `datetime` | `field(default_factory=lambda: datetime.now(timezone.utc))` |

### MultiplicityArtifact.content_hash

[实际实现](../factor_optimizer/contracts/multiplicity.py#L98)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### MultiplicityArtifact.verify

[实际实现](../factor_optimizer/contracts/multiplicity.py#L101)。

Fail closed if the artifact was altered after construction.

参数：`(self)`。

返回类型：`None`。

### MultiplicityArtifact.to_dict

[实际实现](../factor_optimizer/contracts/multiplicity.py#L106)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### MultiplicityArtifact.from_dict

[实际实现](../factor_optimizer/contracts/multiplicity.py#L120)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'MultiplicityArtifact'`。

## factor_optimizer/contracts/objective.py

ObjectiveSpec: first-class, frozen, serializable search objective contract.

### ObjectiveSpec

[实际实现](../factor_optimizer/contracts/objective.py#L32)。

Frozen, validated contract naming the search target metric and direction.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `metric_name` | `str` | `必填/未声明默认` |
| `direction` | `ObjectiveDirection` | `必填/未声明默认` |

### ObjectiveSpec.to_dict

[实际实现](../factor_optimizer/contracts/objective.py#L57)。

Serialize to a JSON-safe dict.

参数：`(self)`。

返回类型：`Dict[str, str]`。

### ObjectiveSpec.from_dict

[实际实现](../factor_optimizer/contracts/objective.py#L65)。

Deserialize (and re-validate fail-closed) from a dict.

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'ObjectiveSpec'`。

### ObjectiveSpec.from_direction

[实际实现](../factor_optimizer/contracts/objective.py#L81)。

Build the spec for the default metric from a bare direction string.

参数：`(cls, direction: str)`。

返回类型：`'ObjectiveSpec'`。

## factor_optimizer/contracts/search_budget.py

SearchBudget: tracking for trials, evaluations, and compute costs.

### SearchBudget

[实际实现](../factor_optimizer/contracts/search_budget.py#L10)。

Budget limits for a search session.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `max_trials` | `int` | `100` |
| `max_evaluations` | `int` | `50` |
| `max_cost_units` | `float` | `1000.0` |
| `max_llm_calls` | `Optional[int]` | `None` |

### SearchBudget.to_dict

[实际实现](../factor_optimizer/contracts/search_budget.py#L45)。

Serialize budget limits.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### SearchBudget.from_dict

[实际实现](../factor_optimizer/contracts/search_budget.py#L55)。

Deserialize budget limits.

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'SearchBudget'`。

### BudgetTracker

[实际实现](../factor_optimizer/contracts/search_budget.py#L61)。

Runtime budget consumption tracker with atomic evaluation reservations.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `budget` | `SearchBudget` | `必填/未声明默认` |
| `trials_used` | `int` | `0` |
| `evaluations_used` | `int` | `0` |
| `cost_used` | `float` | `0.0` |
| `llm_calls_used` | `int` | `0` |
| `evaluations_reserved` | `int` | `0` |
| `cost_reserved` | `float` | `0.0` |
| `overrun_cost` | `float` | `0.0` |
| `reservations` | `Dict[str, float]` | `field(default_factory=dict)` |

### BudgetTracker.can_propose_trial

[实际实现](../factor_optimizer/contracts/search_budget.py#L75)。

Check if budget allows another trial.

参数：`(self)`。

返回类型：`bool`。

### BudgetTracker.can_evaluate

[实际实现](../factor_optimizer/contracts/search_budget.py#L79)。

Check if budget allows another evaluation.

参数：`(self)`。

返回类型：`bool`。

### BudgetTracker.can_spend

[实际实现](../factor_optimizer/contracts/search_budget.py#L84)。

Check if budget allows spending cost units.

参数：`(self, cost: float)`。

返回类型：`bool`。

### BudgetTracker.reserve_evaluation

[实际实现](../factor_optimizer/contracts/search_budget.py#L91)。

Atomically reserve one evaluation and its maximum expected cost.

参数：`(self, cost: float, attempt_id: Optional[str]=None)`。

返回类型：`bool`。

### BudgetTracker.commit_evaluation

[实际实现](../factor_optimizer/contracts/search_budget.py#L110)。

Commit a reservation and refund any unused reserved cost.

参数：`(self, reserved_cost: float, actual_cost: float, attempt_id: Optional[str]=None)`。

返回类型：`None`。

### BudgetTracker.start_evaluation

[实际实现](../factor_optimizer/contracts/search_budget.py#L127)。

Mark execution started (in-memory reservations need no transition).

参数：`(self, attempt_id: Optional[str]=None)`。

返回类型：`None`。

### BudgetTracker.has_reservation

[实际实现](../factor_optimizer/contracts/search_budget.py#L130)。

Return whether this exact attempt owns an active reservation.

参数：`(self, attempt_id: Optional[str])`。

返回类型：`bool`。

### BudgetTracker.release_evaluation

[实际实现](../factor_optimizer/contracts/search_budget.py#L135)。

Refund a reservation after an evaluation fails.

参数：`(self, reserved_cost: float, attempt_id: Optional[str]=None)`。

返回类型：`None`。

### BudgetTracker.can_call_llm

[实际实现](../factor_optimizer/contracts/search_budget.py#L162)。

Check if budget allows another LLM call.

参数：`(self)`。

返回类型：`bool`。

### BudgetTracker.record_trial

[实际实现](../factor_optimizer/contracts/search_budget.py#L168)。

Record a trial generation.

参数：`(self)`。

返回类型：`None`。

### BudgetTracker.record_evaluation

[实际实现](../factor_optimizer/contracts/search_budget.py#L172)。

Atomically record an evaluation and its cost.

参数：`(self, cost: float=1.0)`。

返回类型：`None`。

### BudgetTracker.record_llm_call

[实际实现](../factor_optimizer/contracts/search_budget.py#L178)。

Record an LLM API call.

参数：`(self)`。

返回类型：`None`。

### BudgetTracker.is_exhausted

[实际实现](../factor_optimizer/contracts/search_budget.py#L182)。

Check if any budget limit is reached or fully reserved.

参数：`(self)`。

返回类型：`bool`。

### BudgetTracker.remaining_trials

[实际实现](../factor_optimizer/contracts/search_budget.py#L195)。

Return remaining trial budget.

参数：`(self)`。

返回类型：`int`。

### BudgetTracker.remaining_evaluations

[实际实现](../factor_optimizer/contracts/search_budget.py#L199)。

Return remaining unreserved evaluation budget.

参数：`(self)`。

返回类型：`int`。

### BudgetTracker.remaining_cost

[实际实现](../factor_optimizer/contracts/search_budget.py#L204)。

Return remaining unreserved cost budget.

参数：`(self)`。

返回类型：`float`。

### BudgetTracker.utilization_report

[实际实现](../factor_optimizer/contracts/search_budget.py#L209)。

Return budget utilization summary.

参数：`(self)`。

返回类型：`dict`。

### BudgetTracker.to_dict

[实际实现](../factor_optimizer/contracts/search_budget.py#L230)。

Serialize budget consumption under the reservation lock.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### BudgetTracker.from_dict

[实际实现](../factor_optimizer/contracts/search_budget.py#L246)。

Deserialize budget consumption, including active reservations.

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'BudgetTracker'`。

## factor_optimizer/contracts/splits.py

Split contracts for SearchRunner.

显式导出（含重导出）：`SplitType`、`SplitPlan`、`LabelBundle`、`EvaluationProtocol`、`SearchEvaluationResult`、`SelectedExecutionSpec`、`SealedTestHandle`、`SealedTestResult`、`validate_split_plan`、`create_split_aware_evaluation_fn`。

### SplitType

[实际实现](../factor_optimizer/contracts/splits.py#L21)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `TRAIN` | `类常量/枚举` | `'train'` |
| `VALIDATION` | `类常量/枚举` | `'validation'` |
| `TEST` | `类常量/枚举` | `'test'` |

### SplitPlan

[实际实现](../factor_optimizer/contracts/splits.py#L28)。

Immutable, externally supplied split boundary description.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `split_id` | `str` | `必填/未声明默认` |
| `train_mask` | `Any` | `必填/未声明默认` |
| `validation_mask` | `Any` | `必填/未声明默认` |
| `test_mask` | `Any` | `必填/未声明默认` |
| `metadata` | `Dict[str, Any]` | `必填/未声明默认` |
| `time_index` | `Optional[Tuple]` | `None` |
| `label_horizon` | `int` | `0` |
| `label_bundle` | `Optional['LabelBundle']` | `None` |
| `purge` | `int` | `0` |
| `embargo` | `int` | `0` |
| `validation_embargo` | `int` | `0` |

### EvaluationProtocol

[实际实现](../factor_optimizer/contracts/splits.py#L65)。

Evaluator plus a validated split plan for SearchRunner safe mode.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `split_plan` | `SplitPlan` | `必填/未声明默认` |
| `evaluator` | `Callable[[Any, int], Dict[str, Any]]` | `必填/未声明默认` |

### EvaluationProtocol.evaluate

[实际实现](../factor_optimizer/contracts/splits.py#L80)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, trial: Any, fidelity: int)`。

返回类型：`Dict[str, Any]`。

### SearchEvaluationResult

[实际实现](../factor_optimizer/contracts/splits.py#L85)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `evaluation_id` | `str` | `必填/未声明默认` |
| `trial_id` | `str` | `必填/未声明默认` |
| `train_metrics` | `Dict[str, float]` | `必填/未声明默认` |
| `validation_metrics` | `Dict[str, float]` | `必填/未声明默认` |
| `diagnostics` | `Dict[str, Any]` | `必填/未声明默认` |

### SearchEvaluationResult.selection_score

[实际实现](../factor_optimizer/contracts/splits.py#L92)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, metric: str='rank_ic')`。

返回类型：`float`。

### LabelBundle

[实际实现](../factor_optimizer/contracts/splits.py#L97)。

Bundle of label information for temporal leakage validation.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `label_start_time` | `Any` | `None` |
| `label_end_time` | `Any` | `None` |
| `label_horizon` | `int` | `0` |
| `sample_ids` | `Any` | `None` |
| `decision_times` | `Any` | `None` |
| `label_start_times` | `Any` | `None` |
| `label_end_times` | `Any` | `None` |
| `label_availability_times` | `Any` | `None` |

### LabelBundle.has_timestamps

[实际实现](../factor_optimizer/contracts/splits.py#L131)。

True when aligned per-sample timestamp vectors are present.

参数：`(self)`。

返回类型：`bool`。

### SelectedExecutionSpec

[实际实现](../factor_optimizer/contracts/splits.py#L164)。

Deeply frozen mathematical object authorized for sealed evaluation.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `trial_id` | `str` | `必填/未声明默认` |
| `mutation_id` | `str` | `必填/未声明默认` |
| `parent_factor_ids` | `Tuple[str, ...]` | `必填/未声明默认` |
| `selection_evaluation_ref` | `str` | `必填/未声明默认` |
| `sealed_split_hash` | `str` | `必填/未声明默认` |
| `metadata` | `Any` | `必填/未声明默认` |
| `spec_hash` | `str` | `''` |

### SelectedExecutionSpec.to_dict

[实际实现](../factor_optimizer/contracts/splits.py#L190)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, *, include_hash: bool=True)`。

返回类型：`Dict[str, Any]`。

### SelectedExecutionSpec.validate_certification_complete

[实际实现](../factor_optimizer/contracts/splits.py#L203)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`None`。

### SelectedExecutionSpec.required_test_metrics

[实际实现](../factor_optimizer/contracts/splits.py#L250)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Tuple[str, ...]`。

### SelectedExecutionSpec.from_dict

[实际实现](../factor_optimizer/contracts/splits.py#L255)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, value: Dict[str, Any])`。

返回类型：`'SelectedExecutionSpec'`。

### SealedTestEvaluationOutcome

[实际实现](../factor_optimizer/contracts/splits.py#L260)。

Typed evidence returned by the certified sealed-test evaluator.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `metrics` | `Mapping[str, float]` | `必填/未声明默认` |
| `execution_spec_hash` | `str` | `必填/未声明默认` |
| `dataset_identity` | `str` | `必填/未声明默认` |
| `split_id` | `str` | `必填/未声明默认` |
| `factor_identity` | `str` | `必填/未声明默认` |
| `time_identity` | `str` | `必填/未声明默认` |
| `cost_identity` | `str` | `必填/未声明默认` |
| `test_evidence_ref` | `str` | `必填/未声明默认` |

### SealedTestHandle

[实际实现](../factor_optimizer/contracts/splits.py#L287)。

Immutable authority for one test-split evaluation of a frozen winner.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `search_session_id` | `str` | `必填/未声明默认` |
| `trial_id` | `str` | `必填/未声明默认` |
| `split_id` | `str` | `必填/未声明默认` |
| `evaluation_ref` | `str` | `必填/未声明默认` |
| `frozen_at` | `datetime` | `必填/未声明默认` |
| `execution_spec_hash` | `str` | `''` |

### SealedTestResult

[实际实现](../factor_optimizer/contracts/splits.py#L309)。

Result produced by consuming a sealed handle through ``SearchSession``.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `trial_id` | `str` | `必填/未声明默认` |
| `test_metrics` | `Mapping[str, float]` | `必填/未声明默认` |
| `frozen_at` | `datetime` | `必填/未声明默认` |
| `search_session_id` | `str` | `必填/未声明默认` |
| `split_id` | `str` | `必填/未声明默认` |
| `evaluation_ref` | `str` | `必填/未声明默认` |
| `selection_evaluation_ref` | `str` | `''` |
| `execution_spec_hash` | `str` | `''` |

### validate_split_plan

[实际实现](../factor_optimizer/contracts/splits.py#L337)。

Validate non-empty, equal-length, pairwise-disjoint boolean masks.

参数：`(split_plan: SplitPlan)`。

返回类型：`Dict[str, Any]`。

### create_split_aware_evaluation_fn

[实际实现](../factor_optimizer/contracts/splits.py#L726)。

Build the safe boundary around an adapter's evaluator.

参数：`(qe_adapter: Any, split_plan: SplitPlan)`。

返回类型：`EvaluationProtocol`。

## factor_optimizer/contracts/statistical_governance.py

FO-side statistical-search governance contracts.

### SplitPurpose

[实际实现](../factor_optimizer/contracts/statistical_governance.py#L9)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `DIAGNOSTIC` | `类常量/枚举` | `'diagnostic'` |
| `PREDICTION_VALIDATION` | `类常量/枚举` | `'prediction_validation'` |
| `PRODUCTION_OOS` | `类常量/枚举` | `'production_oos'` |

### SplitEvidenceRef

[实际实现](../factor_optimizer/contracts/statistical_governance.py#L16)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `split_ref` | `str` | `必填/未声明默认` |
| `purpose` | `SplitPurpose` | `必填/未声明默认` |
| `chronological_forward` | `bool` | `必填/未声明默认` |
| `candidate_universe_complete` | `bool` | `必填/未声明默认` |

### SplitEvidenceRef.require_production_oos

[实际实现](../factor_optimizer/contracts/statistical_governance.py#L22)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`None`。

### RetentionEvidence

[实际实现](../factor_optimizer/contracts/statistical_governance.py#L30)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `train_value` | `float` | `必填/未声明默认` |
| `validation_value` | `float` | `必填/未声明默认` |
| `signed_difference` | `float` | `必填/未声明默认` |
| `retention_ratio` | `Optional[float]` | `必填/未声明默认` |
| `ratio_applicable` | `bool` | `必填/未声明默认` |
| `sign_reversal` | `bool` | `必填/未声明默认` |

### retention_evidence

[实际实现](../factor_optimizer/contracts/statistical_governance.py#L39)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(train: float, validation: float, *, near_zero: float=0.001)`。

返回类型：`RetentionEvidence`。

### HorizonDiagnosis

[实际实现](../factor_optimizer/contracts/statistical_governance.py#L52)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `horizons` | `Tuple[int, ...]` | `必填/未声明默认` |
| `values` | `Tuple[float, ...]` | `必填/未声明默认` |
| `best_horizon` | `Optional[int]` | `必填/未声明默认` |
| `half_life` | `Optional[float]` | `必填/未声明默认` |
| `fit_status` | `str` | `必填/未声明默认` |
| `hypothesis_count` | `int` | `必填/未声明默认` |

### diagnose_horizon_curve

[实际实现](../factor_optimizer/contracts/statistical_governance.py#L61)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(values: Mapping[int, float])`。

返回类型：`HorizonDiagnosis`。

## factor_optimizer/contracts/treatment_integrity.py

TreatmentIntegrityEvidence: real integrity evidence for a treatment run (R55 P0-9).

显式导出（含重导出）：`EVIDENCE_SCHEMA_VERSION`、`RAW_TREATMENT_KIND`、`IntegrityCheckResult`、`TreatmentIntegrityStatus`、`TreatmentIntegrityEvidence`、`digest_value`、`build_integrity_evidence`、`require_integrity_evidence`、`describe_integrity_problem`。

### digest_value

[实际实现](../factor_optimizer/contracts/treatment_integrity.py#L192)。

Content digest of the treated data (sha256 over :func:`_data_envelope`).

参数：`(value: Any)`。

返回类型：`str`。

### IntegrityCheckResult

[实际实现](../factor_optimizer/contracts/treatment_integrity.py#L204)。

One named integrity check with its measured value (R55 P0-9).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `name` | `str` | `必填/未声明默认` |
| `passed` | `bool` | `必填/未声明默认` |
| `expected` | `str` | `''` |
| `measured_value` | `Optional[float]` | `None` |
| `detail` | `str` | `''` |

### IntegrityCheckResult.to_dict

[实际实现](../factor_optimizer/contracts/treatment_integrity.py#L247)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### IntegrityCheckResult.from_dict

[实际实现](../factor_optimizer/contracts/treatment_integrity.py#L257)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Mapping[str, Any])`。

返回类型：`'IntegrityCheckResult'`。

### TreatmentIntegrityEvidence

[实际实现](../factor_optimizer/contracts/treatment_integrity.py#L278)。

Deep-immutable, content-hashed record of a treatment's integrity (R55 P0-9).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `treatment_id` | `str` | `必填/未声明默认` |
| `treatment_kind` | `str` | `必填/未声明默认` |
| `applied_parameters` | `Mapping[str, Any]` | `必填/未声明默认` |
| `integrity_checks` | `Tuple[IntegrityCheckResult, ...]` | `必填/未声明默认` |
| `before_digest` | `str` | `必填/未声明默认` |
| `after_digest` | `str` | `必填/未声明默认` |
| `data_digest_algorithm` | `str` | `'sha256'` |
| `produced_by` | `str` | `'factor_optimizer'` |
| `created_at` | `datetime` | `field(default_factory=lambda: datetime.now(timezone.utc))` |

### TreatmentIntegrityEvidence.overall_status

[实际实现](../factor_optimizer/contracts/treatment_integrity.py#L361)。

PASSED / FAILED / NOT_RUN, derived from the checks.

参数：`(self)`。

返回类型：`'TreatmentIntegrityStatus'`。

### TreatmentIntegrityEvidence.failed_checks

[实际实现](../factor_optimizer/contracts/treatment_integrity.py#L376)。

Names of the checks that did not pass (empty when all passed).

参数：`(self)`。

返回类型：`Tuple[str, ...]`。

### TreatmentIntegrityEvidence.all_checks_passed

[实际实现](../factor_optimizer/contracts/treatment_integrity.py#L383)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`bool`。

### TreatmentIntegrityEvidence.content_hash

[实际实现](../factor_optimizer/contracts/treatment_integrity.py#L409)。

sha256 over every semantic field of the evidence.

参数：`(self)`。

返回类型：`str`。

### TreatmentIntegrityEvidence.verify

[实际实现](../factor_optimizer/contracts/treatment_integrity.py#L418)。

Fail closed if the evidence was altered after construction.

参数：`(self)`。

返回类型：`None`。

### TreatmentIntegrityEvidence.to_dict

[实际实现](../factor_optimizer/contracts/treatment_integrity.py#L428)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### TreatmentIntegrityEvidence.from_dict

[实际实现](../factor_optimizer/contracts/treatment_integrity.py#L445)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Mapping[str, Any])`。

返回类型：`'TreatmentIntegrityEvidence'`。

### TreatmentIntegrityStatus

[实际实现](../factor_optimizer/contracts/treatment_integrity.py#L476)。

Overall status of a treatment's integrity evidence.

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `PASSED` | `类常量/枚举` | `'PASSED'` |
| `FAILED` | `类常量/枚举` | `'FAILED'` |
| `NOT_RUN` | `类常量/枚举` | `'NOT_RUN'` |

### TreatmentIntegrityStatus.from_value

[实际实现](../factor_optimizer/contracts/treatment_integrity.py#L489)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, value: Any)`。

返回类型：`'TreatmentIntegrityStatus'`。

### build_integrity_evidence

[实际实现](../factor_optimizer/contracts/treatment_integrity.py#L506)。

Build evidence by MEASURING the actual before/after data (R55 P0-9).

参数：`(treatment_id: str, treatment_kind: str, applied_parameters: Mapping[str, Any], before: Any, after: Any, *, expected_parameters: Optional[Mapping[str, Any]]=None, extra_checks: Sequence[IntegrityCheckResult]=(), produced_by: str='factor_optimizer', created_at: Optional[datetime]=None)`。

返回类型：`TreatmentIntegrityEvidence`。

### describe_integrity_problem

[实际实现](../factor_optimizer/contracts/treatment_integrity.py#L613)。

Return a human-readable reason the evidence is unacceptable, or None.

参数：`(treatment_id: str, evidence: Optional[TreatmentIntegrityEvidence], *, treatment_kind: Optional[str]=None)`。

返回类型：`Optional[str]`。

### require_integrity_evidence

[实际实现](../factor_optimizer/contracts/treatment_integrity.py#L666)。

Fail-closed integrity gate (R55 P0-9).

参数：`(treatment_id: str, evidence: Optional[TreatmentIntegrityEvidence], *, treatment_kind: Optional[str]=None)`。

返回类型：`TreatmentIntegrityEvidence`。

## factor_optimizer/contracts/treatment_result.py

RawTreatmentOutcome + extended TreatmentOptimizationResultArtifact (R61-FI-033).

显式导出（含重导出）：`RawTreatmentOutcome`、`RAW_OUTCOME_VALUES`、`OUTCOME_IS_RAW_WIN`、`OUTCOME_IS_SUCCESS`、`TreatmentOptimizationResultArtifact`、`treatment_result_from_selection`、`validate_outcome_refs`。

### RawTreatmentOutcome

[实际实现](../factor_optimizer/contracts/treatment_result.py#L64)。

Canonical outcome of a treatment-optimization run (plan §22).

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `IMPROVED` | `类常量/枚举` | `'IMPROVED'` |
| `RAW_SELECTED_NO_IMPROVEMENT` | `类常量/枚举` | `'RAW_SELECTED_NO_IMPROVEMENT'` |
| `RAW_SELECTED_NEAR_EQUIVALENT` | `类常量/枚举` | `'RAW_SELECTED_NEAR_EQUIVALENT'` |
| `RAW_SELECTED_CANDIDATES_FAILED` | `类常量/枚举` | `'RAW_SELECTED_CANDIDATES_FAILED'` |
| `NO_ELIGIBLE_REPAIR` | `类常量/枚举` | `'NO_ELIGIBLE_REPAIR'` |
| `SEARCH_BUDGET_EXHAUSTED_RAW_SELECTED` | `类常量/枚举` | `'SEARCH_BUDGET_EXHAUSTED_RAW_SELECTED'` |
| `FAILED_RAW_VALID` | `类常量/枚举` | `'FAILED_RAW_VALID'` |
| `FAILED_RAW_INVALID` | `类常量/枚举` | `'FAILED_RAW_INVALID'` |

### RawTreatmentOutcome.from_value

[实际实现](../factor_optimizer/contracts/treatment_result.py#L101)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, value: object)`。

返回类型：`'RawTreatmentOutcome'`。

### RawTreatmentOutcome.raw_won

[实际实现](../factor_optimizer/contracts/treatment_result.py#L118)。

True when RAW is the production candidate returned by the run.

参数：`(self)`。

返回类型：`bool`。

### RawTreatmentOutcome.produced_winner

[实际实现](../factor_optimizer/contracts/treatment_result.py#L123)。

True when the run ended with a winner candidate to produce.

参数：`(self)`。

返回类型：`bool`。

### validate_outcome_refs

[实际实现](../factor_optimizer/contracts/treatment_result.py#L170)。

Fail-closed ref validation for a RAW outcome (plan §22 iron law).

参数：`(outcome: RawTreatmentOutcome, *, raw_factor_ref: Optional[str], raw_evaluation_ref: Optional[str], raw_health_ref: Optional[str], winner_factor_ref: Optional[str]=None, winner_evaluation_ref: Optional[str]=None, winner_health_ref: Optional[str]=None, candidate_trial_refs: Tuple[str, ...]=(), pareto_refs: Tuple[str, ...]=(), multiplicity_family_ref: str='', selection_reason: str='', failure_summary: str='')`。

返回类型：`None`。

### TreatmentOptimizationResultArtifact

[实际实现](../factor_optimizer/contracts/treatment_result.py#L304)。

Deep-immutable, content-hashed result of a treatment-optimization search.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `search_session_id` | `str` | `必填/未声明默认` |
| `source_factor_value_ref` | `str` | `必填/未声明默认` |
| `raw_baseline_evidence_ref` | `str` | `必填/未声明默认` |
| `factor_profile_ref` | `str` | `必填/未声明默认` |
| `treatment_search_space_ref` | `str` | `必填/未声明默认` |
| `transform_registry_snapshot_ref` | `str` | `必填/未声明默认` |
| `desirability_policy_ref` | `str` | `必填/未声明默认` |
| `winner_policy_ref` | `str` | `必填/未声明默认` |
| `split_plan_ref` | `str` | `必填/未声明默认` |
| `trial_ledger_ref` | `str` | `必填/未声明默认` |
| `all_trial_refs` | `Tuple[str, ...]` | `必填/未声明默认` |
| `pareto_trial_refs` | `Tuple[str, ...]` | `必填/未声明默认` |
| `multiplicity_ref` | `str` | `必填/未声明默认` |
| `library_snapshot_ref` | `Optional[Any]` | `None` |
| `require_library_snapshot_ref` | `bool` | `False` |
| `selected_trial_ref` | `str` | `''` |
| `uncertainty_evidence_ref` | `str` | `''` |
| `sealed_test_ref` | `Optional[str]` | `None` |
| `created_at` | `datetime` | `field(default_factory=lambda: datetime.now(timezone.utc))` |
| `raw_outcome` | `str` | `'IMPROVED'` |
| `raw_factor_ref` | `str` | `''` |
| `raw_evaluation_ref` | `str` | `''` |
| `raw_health_ref` | `str` | `''` |
| `winner_factor_ref` | `str` | `''` |
| `winner_evaluation_ref` | `str` | `''` |
| `winner_health_ref` | `str` | `''` |
| `candidate_trial_refs` | `Tuple[str, ...]` | `()` |
| `pareto_refs` | `Tuple[str, ...]` | `()` |
| `multiplicity_family_ref` | `str` | `''` |
| `selection_reason` | `str` | `''` |
| `failure_summary` | `str` | `''` |
| `per_field_refs` | `Mapping[str, str]` | `field(default_factory=dict)` |

### TreatmentOptimizationResultArtifact.raw_won

[实际实现](../factor_optimizer/contracts/treatment_result.py#L465)。

True when RAW is the production candidate of this run.

参数：`(self)`。

返回类型：`bool`。

### TreatmentOptimizationResultArtifact.produced_winner

[实际实现](../factor_optimizer/contracts/treatment_result.py#L470)。

True when the run produced a winner candidate.

参数：`(self)`。

返回类型：`bool`。

### TreatmentOptimizationResultArtifact.content_hash

[实际实现](../factor_optimizer/contracts/treatment_result.py#L524)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### TreatmentOptimizationResultArtifact.verify

[实际实现](../factor_optimizer/contracts/treatment_result.py#L527)。

Fail closed if the artifact was altered after construction.

参数：`(self)`。

返回类型：`None`。

### TreatmentOptimizationResultArtifact.to_dict

[实际实现](../factor_optimizer/contracts/treatment_result.py#L534)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### TreatmentOptimizationResultArtifact.from_dict

[实际实现](../factor_optimizer/contracts/treatment_result.py#L580)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'TreatmentOptimizationResultArtifact'`。

### treatment_result_from_selection

[实际实现](../factor_optimizer/contracts/treatment_result.py#L637)。

Build a fully-ref-attributed result artifact from a selection summary.

参数：`(*, search_session_id: str, source_factor_value_ref: str, raw_baseline_evidence_ref: str, factor_profile_ref: str, treatment_search_space_ref: str, transform_registry_snapshot_ref: str, desirability_policy_ref: str, winner_policy_ref: str, split_plan_ref: str, trial_ledger_ref: str, all_trial_refs: Tuple[str, ...], pareto_trial_refs: Tuple[str, ...], multiplicity_ref: str, outcome: RawTreatmentOutcome, raw_factor_ref: str, raw_evaluation_ref: str, raw_health_ref: str, winner_factor_ref: str='', winner_evaluation_ref: str='', winner_health_ref: str='', candidate_trial_refs: Tuple[str, ...]=(), pareto_refs: Tuple[str, ...]=(), multiplicity_family_ref: str='', selection_reason: str='', failure_summary: str='', selected_trial_ref: str='', library_snapshot_ref: Optional[Any]=None, require_library_snapshot_ref: bool=False, uncertainty_evidence_ref: str='', sealed_test_ref: Optional[str]=None, per_field_refs: Optional[Mapping[str, str]]=None)`。

返回类型：`TreatmentOptimizationResultArtifact`。

## factor_optimizer/contracts/trial.py

Trial: record of a single mutation attempt with status and results.

### TrialStatus

[实际实现](../factor_optimizer/contracts/trial.py#L17)。

Status of a mutation trial.

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `PROPOSED` | `类常量/枚举` | `'proposed'` |
| `VALIDATING` | `类常量/枚举` | `'validating'` |
| `ILLEGAL` | `类常量/枚举` | `'illegal'` |
| `LEGAL` | `类常量/枚举` | `'legal'` |
| `EVALUATING` | `类常量/枚举` | `'evaluating'` |
| `EVALUATED` | `类常量/枚举` | `'evaluated'` |
| `FAILED` | `类常量/枚举` | `'failed'` |
| `DUPLICATE` | `类常量/枚举` | `'duplicate'` |
| `PROPOSAL_FAILED` | `类常量/枚举` | `'proposal_failed'` |
| `INVALID_PROPOSAL` | `类常量/枚举` | `'invalid_proposal'` |
| `EVALUATION_FAILED` | `类常量/枚举` | `'evaluation_failed'` |
| `PRUNED` | `类常量/枚举` | `'pruned'` |
| `SELECTED` | `类常量/枚举` | `'selected'` |

### Trial

[实际实现](../factor_optimizer/contracts/trial.py#L44)。

Record of a single mutation trial.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `trial_id` | `str` | `必填/未声明默认` |
| `mutation_id` | `str` | `必填/未声明默认` |
| `status` | `TrialStatus` | `必填/未声明默认` |
| `parent_factor_ids` | `list[str]` | `field(default_factory=list)` |
| `created_at` | `datetime` | `field(default_factory=lambda: datetime.now(timezone.utc))` |
| `updated_at` | `datetime` | `field(default_factory=lambda: datetime.now(timezone.utc))` |
| `legality_check` | `Optional[Dict[str, Any]]` | `None` |
| `evaluation_ref` | `Optional[str]` | `None` |
| `failure_reason` | `Optional[str]` | `None` |
| `metadata` | `Dict[str, Any]` | `field(default_factory=dict)` |

### Trial.update_status

[实际实现](../factor_optimizer/contracts/trial.py#L76)。

Update trial status and timestamp.

参数：`(self, new_status: TrialStatus, **kwargs)`。

返回类型：`None`。

### Trial.is_terminal

[实际实现](../factor_optimizer/contracts/trial.py#L91)。

Check if trial has reached a terminal status.

参数：`(self)`。

返回类型：`bool`。

### Trial.is_successful

[实际实现](../factor_optimizer/contracts/trial.py#L105)。

Check if trial succeeded (has evaluation).

参数：`(self)`。

返回类型：`bool`。

### Trial.to_dict

[实际实现](../factor_optimizer/contracts/trial.py#L109)。

Serialize to dictionary.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### Trial.from_dict

[实际实现](../factor_optimizer/contracts/trial.py#L125)。

Deserialize from dictionary.

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'Trial'`。

## factor_optimizer/contracts/trial_ledger.py

Append-only TrialLedger for every proposal attempt (FO-P0-04).

显式导出（含重导出）：`LedgerEntry`、`TrialLedger`、`TRIAL_STATUS_OUTCOMES`。

### LedgerEntry

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L40)。

An immutable, append-only record of a single proposal attempt.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `sequence` | `int` | `必填/未声明默认` |
| `status` | `TrialStatus` | `必填/未声明默认` |
| `trial_id` | `Optional[str]` | `None` |
| `failure_reason` | `Optional[str]` | `None` |
| `recorded_at` | `datetime` | `field(default_factory=lambda: datetime.now(timezone.utc))` |
| `previous_entry_hash` | `str` | `'0' * 64` |
| `entry_hash` | `str` | `''` |
| `hash_encoding_version` | `int` | `2` |

### LedgerEntry.compute_entry_hash

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L103)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### LedgerEntry.status_value

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L107)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### LedgerEntry.to_dict

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L112)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### LedgerEntry.from_dict

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L125)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'LedgerEntry'`。

### TrialLedger

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L146)。

Append-only, hash-chained, order-validated record of every proposal attempt.

### TrialLedger.__init__

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L155)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`None`。

### TrialLedger.append

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L163)。

Append an immutable, hash-chained entry and return it.

参数：`(self, status, trial_id: Optional[str]=None, failure_reason: Optional[str]=None, recorded_at: Optional[datetime]=None)`。

返回类型：`LedgerEntry`。

### TrialLedger.append_proposal_failed

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L188)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, failure_reason: str)`。

返回类型：`LedgerEntry`。

### TrialLedger.append_invalid_proposal

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L191)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, failure_reason: str)`。

返回类型：`LedgerEntry`。

### TrialLedger.append_trial

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L194)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, status, trial_id, failure_reason=None)`。

返回类型：`LedgerEntry`。

### TrialLedger.entries

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L198)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Tuple[LedgerEntry, ...]`。

### TrialLedger.close

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L201)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`None`。

### TrialLedger.seal

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L204)。

Close the ledger and record the head hash / entry count.

参数：`(self)`。

返回类型：`None`。

### TrialLedger.begin_authorized_extension

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L221)。

Open a new append epoch while retaining the prior immutable seal.

参数：`(self)`。

返回类型：`None`。

### TrialLedger.sealed

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L237)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`bool`。

### TrialLedger.sealed_head_hash

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L241)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Optional[str]`。

### TrialLedger.sealed_entry_count

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L245)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Optional[int]`。

### TrialLedger.sealed_at

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L249)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Optional[datetime]`。

### TrialLedger.verify_chain

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L252)。

Re-verify the full hash chain, fail-closed on any tamper/truncation.

参数：`(self)`。

返回类型：`None`。

### TrialLedger.status_counts

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L299)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, int]`。

### TrialLedger.to_dict

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L306)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### TrialLedger.from_dict

[实际实现](../factor_optimizer/contracts/trial_ledger.py#L319)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'TrialLedger'`。

## factor_optimizer/contracts/validator.py

TrialValidator contracts with identity.

显式导出（含重导出）：`TrialValidatorIdentity`、`ProductionTrialValidator`、`MutationGrammarValidator`、`_specs_canonical_json`。

### TrialValidatorIdentity

[实际实现](../factor_optimizer/contracts/validator.py#L25)。

Immutable identity of a trial validator implementation.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `validator_id` | `str` | `必填/未声明默认` |
| `validator_version` | `str` | `必填/未声明默认` |
| `implementation_hash` | `str` | `必填/未声明默认` |

### TrialValidatorIdentity.as_dict

[实际实现](../factor_optimizer/contracts/validator.py#L49)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, str]`。

### ProductionTrialValidator

[实际实现](../factor_optimizer/contracts/validator.py#L58)。

A trial validator that carries a verifiable identity.

基类：`Protocol`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `identity` | `TrialValidatorIdentity` | `必填/未声明默认` |

### ProductionTrialValidator.validate

[实际实现](../factor_optimizer/contracts/validator.py#L68)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, trial: Any)`。

返回类型：`Dict[str, Any]`。

### MutationGrammarValidator

[实际实现](../factor_optimizer/contracts/validator.py#L112)。

Concrete production validator wrapping the grammar ``MutationValidator``.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `VALIDATOR_ID` | `类常量/枚举` | `'factor_optimizer.mutation_grammar'` |
| `VALIDATOR_VERSION` | `类常量/枚举` | `'0.1.0'` |

### MutationGrammarValidator.__init__

[实际实现](../factor_optimizer/contracts/validator.py#L125)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, registry: Optional[MutationRegistry]=None, implementation_hash: Optional[str]=None)`。

### MutationGrammarValidator.validate

[实际实现](../factor_optimizer/contracts/validator.py#L165)。

Validate a trial and return a production-shaped legality dict.

参数：`(self, trial: Trial)`。

返回类型：`Dict[str, Any]`。

## factor_optimizer/data_capabilities.py

Real data capabilities: authorization-only scope boundaries.

显式导出（含重导出）：`DataScope`、`DataCapability`、`TrainDataCapability`、`ValidationDataCapability`、`TestDataCapability`、`TestAuthorityBroker`、`TestDataProvider`、`TestStoreRef`、`TestDatasetIdentity`、`ScopedEvaluator`、`build_search_capabilities`、`build_data_capabilities`。

### DataScope

[实际实现](../factor_optimizer/data_capabilities.py#L45)。

The only three data scopes a split plan can authorize.

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `TRAIN` | `类常量/枚举` | `'train'` |
| `VALIDATION` | `类常量/枚举` | `'validation'` |
| `TEST` | `类常量/枚举` | `'test'` |

### DataCapability

[实际实现](../factor_optimizer/data_capabilities.py#L82)。

Authorization for evaluating against a single data scope.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `scope` | `DataScope` | `必填/未声明默认` |

### DataCapability.__init__

[实际实现](../factor_optimizer/data_capabilities.py#L93)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, scope: DataScope, allowed_mask, *, capability_id: Optional[str]=None, search_session_id: Optional[str]=None, split_id: Optional[str]=None, dataset_identity: Optional[str]=None, provider_identity: Optional[str]=None, issued_at: Optional[datetime]=None, expiry: Optional[datetime]=None, nonce: Optional[str]=None)`。

### DataCapability.scope

[实际实现](../factor_optimizer/data_capabilities.py#L171)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`DataScope`。

### DataCapability.allowed_mask

[实际实现](../factor_optimizer/data_capabilities.py#L175)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Tuple[bool, ...]`。

### DataCapability.capability_id

[实际实现](../factor_optimizer/data_capabilities.py#L179)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### DataCapability.search_session_id

[实际实现](../factor_optimizer/data_capabilities.py#L183)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### DataCapability.split_id

[实际实现](../factor_optimizer/data_capabilities.py#L187)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### DataCapability.dataset_identity

[实际实现](../factor_optimizer/data_capabilities.py#L191)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### DataCapability.coordinate_hash

[实际实现](../factor_optimizer/data_capabilities.py#L195)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### DataCapability.provider_identity

[实际实现](../factor_optimizer/data_capabilities.py#L199)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### DataCapability.issued_at

[实际实现](../factor_optimizer/data_capabilities.py#L203)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Optional[datetime]`。

### DataCapability.expiry

[实际实现](../factor_optimizer/data_capabilities.py#L207)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Optional[datetime]`。

### DataCapability.nonce

[实际实现](../factor_optimizer/data_capabilities.py#L211)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### DataCapability.is_expired

[实际实现](../factor_optimizer/data_capabilities.py#L214)。

True when the capability has passed its expiry.

参数：`(self, now: Optional[datetime]=None)`。

返回类型：`bool`。

### DataCapability.can_evaluate

[实际实现](../factor_optimizer/data_capabilities.py#L224)。

True iff ``split_plan``'s coordinates for this scope are an EXACT SUBSET of the authorized coordinates.

参数：`(self, split_plan: SplitPlan)`。

返回类型：`bool`。

### DataCapability.verify_identity

[实际实现](../factor_optimizer/data_capabilities.py#L258)。

Re-derive the coordinate hash and reject any tampering.

参数：`(self)`。

返回类型：`None`。

### TrainDataCapability

[实际实现](../factor_optimizer/data_capabilities.py#L292)。

Authorization for the train scope only.

基类：`DataCapability`。

### TrainDataCapability.__init__

[实际实现](../factor_optimizer/data_capabilities.py#L295)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, allowed_mask, **identity)`。

### ValidationDataCapability

[实际实现](../factor_optimizer/data_capabilities.py#L299)。

Authorization for the validation scope only.

基类：`DataCapability`。

### ValidationDataCapability.__init__

[实际实现](../factor_optimizer/data_capabilities.py#L302)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, allowed_mask, **identity)`。

### TestDataCapability

[实际实现](../factor_optimizer/data_capabilities.py#L306)。

Authorization for the test scope only.

基类：`DataCapability`。

### TestDataCapability.__init__

[实际实现](../factor_optimizer/data_capabilities.py#L315)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, allowed_mask, *, attempt_token: int=0, **identity)`。

### TestDataCapability.attempt_token

[实际实现](../factor_optimizer/data_capabilities.py#L329)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`int`。

### TestDataCapability.verify_identity

[实际实现](../factor_optimizer/data_capabilities.py#L332)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`None`。

### TestStoreRef

[实际实现](../factor_optimizer/data_capabilities.py#L357)。

Opaque reference to a test-data store (FO-P0-03).

### TestStoreRef.__init__

[实际实现](../factor_optimizer/data_capabilities.py#L367)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, store, *, dataset_identity: str)`。

### TestStoreRef.dataset_identity

[实际实现](../factor_optimizer/data_capabilities.py#L376)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### TestStoreRef.read

[实际实现](../factor_optimizer/data_capabilities.py#L379)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### TestDatasetIdentity

[实际实现](../factor_optimizer/data_capabilities.py#L389)。

Opaque identity of a test dataset (FO-P0-003).

### TestDatasetIdentity.__init__

[实际实现](../factor_optimizer/data_capabilities.py#L396)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, dataset_identity: str, store_ref: TestStoreRef)`。

### TestDatasetIdentity.dataset_identity

[实际实现](../factor_optimizer/data_capabilities.py#L405)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### TestDatasetIdentity.store_ref

[实际实现](../factor_optimizer/data_capabilities.py#L409)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`TestStoreRef`。

### TestAuthorityBroker

[实际实现](../factor_optimizer/data_capabilities.py#L413)。

Interface seam for issuing TEST capabilities and resolving test data.

### TestAuthorityBroker.__init__

[实际实现](../factor_optimizer/data_capabilities.py#L427)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, *, dataset_identity: str, provider_identity: str, search_session_id: str, split_id: str, ttl_seconds: int=3600, campaign_store=None, campaign_id: Optional[str]=None, candidate_set_hash: Optional[str]=None, purpose: str='sealed_test', profile_hash: Optional[str]=None)`。

### TestAuthorityBroker.store_ref

[实际实现](../factor_optimizer/data_capabilities.py#L483)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Optional[TestStoreRef]`。

### TestAuthorityBroker.dataset_identity

[实际实现](../factor_optimizer/data_capabilities.py#L487)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### TestAuthorityBroker.search_session_id

[实际实现](../factor_optimizer/data_capabilities.py#L491)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### TestAuthorityBroker.split_id

[实际实现](../factor_optimizer/data_capabilities.py#L495)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### TestAuthorityBroker.attach_store_ref

[实际实现](../factor_optimizer/data_capabilities.py#L498)。

Attach the opaque physical store this broker owns.

参数：`(self, store_ref: TestStoreRef)`。

返回类型：`None`。

### TestAuthorityBroker.issue_test_capability

[实际实现](../factor_optimizer/data_capabilities.py#L510)。

Issue a TEST capability bound to this broker's identity.

参数：`(self, test_mask)`。

返回类型：`TestDataCapability`。

### TestAuthorityBroker.cached_test_result

[实际实现](../factor_optimizer/data_capabilities.py#L544)。

Previously completed immutable result, when durable mode found one.

参数：`(self)`。

### TestAuthorityBroker.active_attempt_token

[实际实现](../factor_optimizer/data_capabilities.py#L549)。

Opaque immutable token for the attempt most recently issued.

参数：`(self)`。

返回类型：`int`。

### TestAuthorityBroker.has_durable_authority

[实际实现](../factor_optimizer/data_capabilities.py#L556)。

Whether test exposure is fenced by a cross-process transaction.

参数：`(self)`。

返回类型：`bool`。

### TestAuthorityBroker.durable_binding

[实际实现](../factor_optimizer/data_capabilities.py#L560)。

Return the immutable persisted binding for certification checks.

参数：`(self)`。

返回类型：`Mapping[str, Any]`。

### TestAuthorityBroker.mark_test_infrastructure_failure

[实际实现](../factor_optimizer/data_capabilities.py#L566)。

Permit an audited retry of the same frozen request only.

参数：`(self, attempt_token: int)`。

返回类型：`None`。

### TestAuthorityBroker.complete_test_attempt

[实际实现](../factor_optimizer/data_capabilities.py#L574)。

Persist the immutable outcome of the logical sealed evaluation.

参数：`(self, attempt_token: int, result_ref: str, result)`。

返回类型：`None`。

### TestAuthorityBroker.create_test_provider

[实际实现](../factor_optimizer/data_capabilities.py#L582)。

Create a provider backed only by the broker's attached store.

参数：`(self, attempt_token: Optional[int]=None)`。

返回类型：`'TestDataProvider'`。

### TestDataProvider

[实际实现](../factor_optimizer/data_capabilities.py#L605)。

Resolves test data ONLY for a valid, unexpired TEST capability.

### TestDataProvider.__init__

[实际实现](../factor_optimizer/data_capabilities.py#L616)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, store_ref: Optional[TestStoreRef], *, dataset_identity: str, provider_identity: str, search_session_id: Optional[str]=None, split_id: Optional[str]=None, expected_attempt_token: Optional[int]=None, exposure_authorizer=None)`。

### TestDataProvider.dataset_identity

[实际实现](../factor_optimizer/data_capabilities.py#L644)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### TestDataProvider.provider_identity

[实际实现](../factor_optimizer/data_capabilities.py#L648)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### TestDataProvider.store_ref

[实际实现](../factor_optimizer/data_capabilities.py#L652)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Optional[TestStoreRef]`。

### TestDataProvider.resolve

[实际实现](../factor_optimizer/data_capabilities.py#L655)。

Return the test payload only for a valid TEST capability.

参数：`(self, capability: DataCapability)`。

### ScopedEvaluator

[实际实现](../factor_optimizer/data_capabilities.py#L710)。

Evaluates a trial against data resolved from a capability.

### ScopedEvaluator.__init__

[实际实现](../factor_optimizer/data_capabilities.py#L718)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, provider: TestDataProvider, evaluation_fn)`。

### ScopedEvaluator.evaluate

[实际实现](../factor_optimizer/data_capabilities.py#L726)。

Resolve data from the capability and evaluate the trial.

参数：`(self, capability: DataCapability, trial, fidelity: int)`。

### build_search_capabilities

[实际实现](../factor_optimizer/data_capabilities.py#L737)。

Build ONLY the train and validation capabilities for a search session.

参数：`(split_plan: SplitPlan, *, search_session_id: str, dataset_identity: str, provider_identity: str)`。

返回类型：`Dict[DataScope, DataCapability]`。

### build_data_capabilities

[实际实现](../factor_optimizer/data_capabilities.py#L770)。

Backward-compatible builder returning one capability per scope.

参数：`(split_plan: SplitPlan)`。

返回类型：`Dict[DataScope, DataCapability]`。

## factor_optimizer/data_providers.py

Real scoped data isolation (FO-P0-01).

显式导出（含重导出）：`DataProvider`、`TrainDataProvider`、`ValidationDataProvider`、`TestDataProvider`、`ScopedEvaluator`。

### DataProvider

[实际实现](../factor_optimizer/data_providers.py#L53)。

Resolves only the coordinate rows a capability authorizes.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `scope` | `DataScope` | `必填/未声明默认` |

### DataProvider.__init__

[实际实现](../factor_optimizer/data_providers.py#L65)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, data: Dict[str, Any], *, search_session_id: str, split_id: str, dataset_identity: str, provider_identity: str)`。

### DataProvider.scope

[实际实现](../factor_optimizer/data_providers.py#L91)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`DataScope`。

### DataProvider.dataset_identity

[实际实现](../factor_optimizer/data_providers.py#L95)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### DataProvider.provider_identity

[实际实现](../factor_optimizer/data_providers.py#L99)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### DataProvider.resolve

[实际实现](../factor_optimizer/data_providers.py#L134)。

Return a COPY of the authorized rows, or raise fail-closed.

参数：`(self, capability: DataCapability)`。

返回类型：`Dict[str, Any]`。

### TrainDataProvider

[实际实现](../factor_optimizer/data_providers.py#L141)。

Resolves only the authorized TRAIN rows.

基类：`DataProvider`。

### ValidationDataProvider

[实际实现](../factor_optimizer/data_providers.py#L147)。

Resolves only the authorized VALIDATION rows.

基类：`DataProvider`。

### TestDataProvider

[实际实现](../factor_optimizer/data_providers.py#L153)。

Resolves only the authorized TEST rows (test authority only).

基类：`DataProvider`。

### ScopedEvaluator

[实际实现](../factor_optimizer/data_providers.py#L159)。

Evaluates a trial against ONLY the data a provider resolves.

### ScopedEvaluator.__init__

[实际实现](../factor_optimizer/data_providers.py#L166)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, provider: DataProvider, evaluation_fn: Callable)`。

### ScopedEvaluator.evaluate

[实际实现](../factor_optimizer/data_providers.py#L174)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, capability: DataCapability, trial, fidelity: int)`。

## factor_optimizer/errors.py

Core error taxonomy for factor_optimizer.

显式导出（含重导出）：`FactorOptimizerError`、`ContractError`、`SchemaVersionError`、`MissingInputError`、`InvalidContractError`、`TimingContractError`、`SnapshotMismatchError`、`CapabilityError`、`CapabilityForgeryError`、`UnsupportedMutationError`、`OptionalDependencyMissing`、`DataError`、`InsufficientObservations`、`InvalidValidityMask`、`EvidenceUnavailableError`、`StaleEvidenceError`、`ExecutionError`、`NumericalFailure`、`OverflowOrNonFiniteError`、`BudgetExceededError`、`CancellationError`、`GovernanceError`、`IllegalMutationError`、`DuplicateIdentityError`、`CollisionError`、`ContractChangeRequired`、`TreatmentIntegrityError`。

### FactorOptimizerError

[实际实现](../factor_optimizer/errors.py#L11)。

Base exception for factor_optimizer.

基类：`Exception`。

### ContractError

[实际实现](../factor_optimizer/errors.py#L21)。

Data contract or schema violation.

基类：`FactorOptimizerError`。

### SchemaVersionError

[实际实现](../factor_optimizer/errors.py#L26)。

Schema version mismatch or unsupported version.

基类：`ContractError`。

### MissingInputError

[实际实现](../factor_optimizer/errors.py#L31)。

Required input field is missing.

基类：`ContractError`。

### InvalidContractError

[实际实现](../factor_optimizer/errors.py#L36)。

Input violates a contract constraint.

基类：`ContractError`。

### TimingContractError

[实际实现](../factor_optimizer/errors.py#L41)。

Timing/ordering constraint violated.

基类：`ContractError`。

### SnapshotMismatchError

[实际实现](../factor_optimizer/errors.py#L46)。

Snapshot or context reference mismatch.

基类：`ContractError`。

### CapabilityError

[实际实现](../factor_optimizer/errors.py#L56)。

Requested capability is not available.

基类：`FactorOptimizerError`。

### CapabilityForgeryError

[实际实现](../factor_optimizer/errors.py#L61)。

A capability's identity/authorization fields were forged or altered.

基类：`CapabilityError`。

### UnsupportedMutationError

[实际实现](../factor_optimizer/errors.py#L66)。

Requested mutation type is not implemented.

基类：`CapabilityError`。

### OptionalDependencyMissing

[实际实现](../factor_optimizer/errors.py#L71)。

Optional dependency (FE, QE, etc.) is not installed.

基类：`CapabilityError`。

### OptionalDependencyMissing.__init__

[实际实现](../factor_optimizer/errors.py#L74)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, package_name: str, feature_name: str)`。

### DataError

[实际实现](../factor_optimizer/errors.py#L88)。

Data or evidence quality issue.

基类：`FactorOptimizerError`。

### InsufficientObservations

[实际实现](../factor_optimizer/errors.py#L93)。

Not enough valid observations for optimization.

基类：`DataError`。

### InvalidValidityMask

[实际实现](../factor_optimizer/errors.py#L98)。

Validity mask is malformed or contradictory.

基类：`DataError`。

### EvidenceUnavailableError

[实际实现](../factor_optimizer/errors.py#L103)。

Evidence cannot be computed or retrieved.

基类：`DataError`。

### StaleEvidenceError

[实际实现](../factor_optimizer/errors.py#L108)。

Evidence is outdated or bound to stale state.

基类：`DataError`。

### ExecutionError

[实际实现](../factor_optimizer/errors.py#L118)。

Execution or computation failed.

基类：`FactorOptimizerError`。

### NumericalFailure

[实际实现](../factor_optimizer/errors.py#L123)。

Numerical computation failed (overflow, NaN, Inf, etc.).

基类：`ExecutionError`。

### OverflowOrNonFiniteError

[实际实现](../factor_optimizer/errors.py#L128)。

Result is infinite, NaN, or overflowed.

基类：`NumericalFailure`。

### BudgetExceededError

[实际实现](../factor_optimizer/errors.py#L133)。

Computational budget or resource limit exceeded.

基类：`ExecutionError`。

### CancellationError

[实际实现](../factor_optimizer/errors.py#L138)。

Search or optimization was cancelled.

基类：`ExecutionError`。

### GovernanceError

[实际实现](../factor_optimizer/errors.py#L148)。

Governance policy or constraint violation.

基类：`FactorOptimizerError`。

### IllegalMutationError

[实际实现](../factor_optimizer/errors.py#L153)。

Mutation violates FE legality or grammar constraints.

基类：`GovernanceError`。

### DuplicateIdentityError

[实际实现](../factor_optimizer/errors.py#L158)。

Factor identity already exists.

基类：`GovernanceError`。

### CollisionError

[实际实现](../factor_optimizer/errors.py#L163)。

Hash or identity collision detected.

基类：`GovernanceError`。

### ContractChangeRequired

[实际实现](../factor_optimizer/errors.py#L168)。

Operation requires contract schema change.

基类：`GovernanceError`。

### TreatmentIntegrityError

[实际实现](../factor_optimizer/errors.py#L173)。

Treatment integrity evidence is missing, stale, tampered, or failing.

基类：`GovernanceError`。

## factor_optimizer/grammar/__init__.py

Mutation grammar: versioned MutationSpec registry and validation.

显式导出（含重导出）：`MutationSpec`、`ParameterSpec`、`ParameterKind`、`ParameterRole`、`MutationRegistry`、`get_mutation_registry`、`MutationValidator`。

## factor_optimizer/grammar/mutation_spec.py

MutationSpec: typed mutation operation with parameter constraints.

### ParameterKind

[实际实现](../factor_optimizer/grammar/mutation_spec.py#L8)。

Type of parameter value.

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `INTEGER` | `类常量/枚举` | `'integer'` |
| `FLOAT` | `类常量/枚举` | `'float'` |
| `BOOLEAN` | `类常量/枚举` | `'boolean'` |
| `STRING` | `类常量/枚举` | `'string'` |
| `ENUM` | `类常量/枚举` | `'enum'` |
| `INTEGER_LIST` | `类常量/枚举` | `'integer_list'` |
| `FLOAT_LIST` | `类常量/枚举` | `'float_list'` |
| `STRING_LIST` | `类常量/枚举` | `'string_list'` |

### ParameterRole

[实际实现](../factor_optimizer/grammar/mutation_spec.py#L21)。

Semantic role of parameter in factor computation.

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `WINDOW` | `类常量/枚举` | `'window'` |
| `DECAY` | `类常量/枚举` | `'decay'` |
| `THRESHOLD` | `类常量/枚举` | `'threshold'` |
| `QUANTILE` | `类常量/枚举` | `'quantile'` |
| `SCALAR` | `类常量/枚举` | `'scalar'` |
| `CATEGORICAL` | `类常量/枚举` | `'categorical'` |
| `STRUCTURAL` | `类常量/枚举` | `'structural'` |
| `TIMING` | `类常量/枚举` | `'timing'` |
| `CAUSAL` | `类常量/枚举` | `'causal'` |

### ParameterSpec

[实际实现](../factor_optimizer/grammar/mutation_spec.py#L36)。

Specification for a mutation parameter.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `name` | `str` | `必填/未声明默认` |
| `kind` | `ParameterKind` | `必填/未声明默认` |
| `role` | `ParameterRole` | `必填/未声明默认` |
| `required` | `bool` | `True` |
| `default` | `Optional[Any]` | `None` |
| `min_value` | `Optional[Union[int, float]]` | `None` |
| `max_value` | `Optional[Union[int, float]]` | `None` |
| `allowed_values` | `Optional[List[Any]]` | `None` |
| `description` | `str` | `''` |

### ParameterSpec.validate_value

[实际实现](../factor_optimizer/grammar/mutation_spec.py#L81)。

Validate a parameter value.

参数：`(self, value: Any)`。

返回类型：`tuple[bool, Optional[str]]`。

### MutationSpec

[实际实现](../factor_optimizer/grammar/mutation_spec.py#L144)。

Specification for a mutation operation type.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `mutation_type` | `str` | `必填/未声明默认` |
| `version` | `str` | `必填/未声明默认` |
| `description` | `str` | `必填/未声明默认` |
| `parameters` | `List[ParameterSpec]` | `field(default_factory=list)` |
| `requires_single_parent` | `bool` | `True` |
| `requires_multiple_parents` | `bool` | `False` |
| `domains` | `List[str]` | `field(default_factory=list)` |
| `sources` | `List[str]` | `field(default_factory=list)` |

### MutationSpec.get_parameter

[实际实现](../factor_optimizer/grammar/mutation_spec.py#L182)。

Get parameter spec by name.

参数：`(self, name: str)`。

返回类型：`Optional[ParameterSpec]`。

### MutationSpec.validate_parameters

[实际实现](../factor_optimizer/grammar/mutation_spec.py#L189)。

Validate a parameter dictionary.

参数：`(self, params: Dict[str, Any])`。

返回类型：`tuple[bool, List[str]]`。

## factor_optimizer/grammar/registry.py

MutationRegistry: catalog of available mutation operations.

### MutationRegistry

[实际实现](../factor_optimizer/grammar/registry.py#L7)。

Registry of available mutation operation specifications.

### MutationRegistry.__init__

[实际实现](../factor_optimizer/grammar/registry.py#L10)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### MutationRegistry.register

[实际实现](../factor_optimizer/grammar/registry.py#L14)。

Register a mutation specification.

参数：`(self, spec: MutationSpec)`。

返回类型：`None`。

### MutationRegistry.get

[实际实现](../factor_optimizer/grammar/registry.py#L20)。

Get mutation specification by type.

参数：`(self, mutation_type: str)`。

返回类型：`Optional[MutationSpec]`。

### MutationRegistry.list_mutation_types

[实际实现](../factor_optimizer/grammar/registry.py#L24)。

List all registered mutation types.

参数：`(self)`。

返回类型：`List[str]`。

### MutationRegistry.list_specs

[实际实现](../factor_optimizer/grammar/registry.py#L28)。

List all registered mutation specifications.

参数：`(self)`。

返回类型：`List[MutationSpec]`。

### MutationRegistry.version

[实际实现](../factor_optimizer/grammar/registry.py#L32)。

Get registry version.

参数：`(self)`。

返回类型：`str`。

### get_mutation_registry

[实际实现](../factor_optimizer/grammar/registry.py#L41)。

Get the global mutation registry, creating with defaults if needed.

参数：`()`。

返回类型：`MutationRegistry`。

## factor_optimizer/grammar/validation.py

MutationValidator: validate mutation proposals against grammar and FE legality.

### ValidationResult

[实际实现](../factor_optimizer/grammar/validation.py#L8)。

Result of mutation validation.

### ValidationResult.__init__

[实际实现](../factor_optimizer/grammar/validation.py#L11)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, is_valid: bool, errors: Optional[List[str]]=None, warnings: Optional[List[str]]=None, metadata: Optional[Dict[str, Any]]=None)`。

### ValidationResult.to_dict

[实际实现](../factor_optimizer/grammar/validation.py#L26)。

Serialize to dictionary.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### MutationValidator

[实际实现](../factor_optimizer/grammar/validation.py#L36)。

Validates mutation proposals against grammar and optionally FE legality.

### MutationValidator.__init__

[实际实现](../factor_optimizer/grammar/validation.py#L46)。

Initialize validator.

参数：`(self, registry: Optional[MutationRegistry]=None, fe_adapter=None)`。

### MutationValidator.validate

[实际实现](../factor_optimizer/grammar/validation.py#L57)。

Validate a mutation proposal.

参数：`(self, mutation: CandidateMutation)`。

返回类型：`ValidationResult`。

### MutationValidator.validate_batch

[实际实现](../factor_optimizer/grammar/validation.py#L129)。

Validate multiple mutations.

参数：`(self, mutations: List[CandidateMutation])`。

返回类型：`Dict[str, ValidationResult]`。

### MutationValidator.quick_check

[实际实现](../factor_optimizer/grammar/validation.py#L144)。

Quick parameter validation without full mutation object.

参数：`(self, mutation_type: str, parameters: Dict[str, Any])`。

返回类型：`Tuple[bool, List[str]]`。

## factor_optimizer/llm/__init__.py

LLM-powered mutation proposal generation.

显式导出（含重导出）：`ProposalGenerator`、`ProposalRequest`、`ProposalResponse`、`PromptTemplate`、`PromptRegistry`、`LLMRecord`、`RecordStore`。

## factor_optimizer/llm/prompts.py

Versioned prompt templates for LLM proposal generation.

### PromptTemplate

[实际实现](../factor_optimizer/llm/prompts.py#L8)。

A versioned prompt template with metadata.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `template_id` | `str` | `必填/未声明默认` |
| `version` | `str` | `必填/未声明默认` |
| `system_prompt` | `str` | `必填/未声明默认` |
| `user_prompt_template` | `str` | `必填/未声明默认` |
| `output_schema` | `Dict[str, Any]` | `必填/未声明默认` |
| `model_hints` | `Dict[str, Any]` | `field(default_factory=dict)` |
| `description` | `str` | `''` |

### PromptTemplate.render

[实际实现](../factor_optimizer/llm/prompts.py#L41)。

Render the user prompt with provided variables.

参数：`(self, **kwargs)`。

返回类型：`str`。

### PromptTemplate.get_full_prompt

[实际实现](../factor_optimizer/llm/prompts.py#L59)。

Get both system and rendered user prompts.

参数：`(self, **kwargs)`。

返回类型：`tuple[str, str]`。

### PromptRegistry

[实际实现](../factor_optimizer/llm/prompts.py#L69)。

Registry for managing versioned prompt templates.

### PromptRegistry.__init__

[实际实现](../factor_optimizer/llm/prompts.py#L76)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### PromptRegistry.register

[实际实现](../factor_optimizer/llm/prompts.py#L79)。

Register a prompt template.

参数：`(self, template: PromptTemplate)`。

返回类型：`None`。

### PromptRegistry.get

[实际实现](../factor_optimizer/llm/prompts.py#L100)。

Retrieve a prompt template.

参数：`(self, template_id: str, version: Optional[str]=None)`。

返回类型：`PromptTemplate`。

### PromptRegistry.list_templates

[实际实现](../factor_optimizer/llm/prompts.py#L130)。

List all registered template IDs.

参数：`(self)`。

返回类型：`list[str]`。

### PromptRegistry.list_versions

[实际实现](../factor_optimizer/llm/prompts.py#L134)。

List all versions of a template.

参数：`(self, template_id: str)`。

返回类型：`list[str]`。

### PromptRegistry.clear

[实际实现](../factor_optimizer/llm/prompts.py#L140)。

Clear all registered templates.

参数：`(self)`。

返回类型：`None`。

### create_default_registry

[实际实现](../factor_optimizer/llm/prompts.py#L197)。

Create a registry with default templates.

参数：`()`。

返回类型：`PromptRegistry`。

## factor_optimizer/llm/proposal.py

Structured LLM proposal generator with schema validation.

### ProposalRequest

[实际实现](../factor_optimizer/llm/proposal.py#L15)。

Request for LLM-generated mutation proposals.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `parent_factor_info` | `Dict[str, Any]` | `必填/未声明默认` |
| `performance_metrics` | `Dict[str, Any]` | `必填/未声明默认` |
| `search_objective` | `str` | `必填/未声明默认` |
| `available_mutations` | `List[str]` | `必填/未声明默认` |
| `context` | `Dict[str, Any]` | `field(default_factory=dict)` |

### ProposalRequest.to_dict

[实际实现](../factor_optimizer/llm/proposal.py#L33)。

Serialize to dictionary.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### ProposalResponse

[实际实现](../factor_optimizer/llm/proposal.py#L45)。

Response from LLM proposal generation.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `request_id` | `str` | `必填/未声明默认` |
| `proposals` | `List[CandidateMutation]` | `必填/未声明默认` |
| `llm_record` | `LLMRecord` | `必填/未声明默认` |
| `raw_response` | `str` | `必填/未声明默认` |
| `validation_errors` | `List[str]` | `field(default_factory=list)` |

### ProposalResponse.is_valid

[实际实现](../factor_optimizer/llm/proposal.py#L63)。

Check if response has valid proposals.

参数：`(self)`。

返回类型：`bool`。

### ProposalGenerator

[实际实现](../factor_optimizer/llm/proposal.py#L68)。

Generate structured mutation proposals using LLMs.

### ProposalGenerator.__init__

[实际实现](../factor_optimizer/llm/proposal.py#L76)。

Initialize the proposal generator.

参数：`(self, prompt_registry: Optional[PromptRegistry]=None, record_store: Optional[RecordStore]=None, model: str='mock-model')`。

### ProposalGenerator.generate_proposals

[实际实现](../factor_optimizer/llm/proposal.py#L94)。

Generate mutation proposals for the given request.

参数：`(self, request: ProposalRequest, template_id: str='mutation_proposal', template_version: Optional[str]=None, parent_factor_ids: Optional[List[str]]=None)`。

返回类型：`ProposalResponse`。

### ProposalGenerator.get_records

[实际实现](../factor_optimizer/llm/proposal.py#L245)。

Get all LLM call records.

参数：`(self)`。

返回类型：`List[LLMRecord]`。

### ProposalGenerator.get_token_usage

[实际实现](../factor_optimizer/llm/proposal.py#L249)。

Get total token usage statistics.

参数：`(self)`。

返回类型：`Dict[str, int]`。

## factor_optimizer/llm/records.py

LLM call recording for reproducibility and audit.

### LLMRecord

[实际实现](../factor_optimizer/llm/records.py#L11)。

Record of a single LLM API call for reproducibility.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `record_id` | `str` | `必填/未声明默认` |
| `timestamp` | `datetime` | `必填/未声明默认` |
| `model` | `str` | `必填/未声明默认` |
| `model_version` | `Optional[str]` | `必填/未声明默认` |
| `prompt_template_id` | `str` | `必填/未声明默认` |
| `prompt_template_version` | `str` | `必填/未声明默认` |
| `prompt_hash` | `str` | `必填/未声明默认` |
| `input_tokens` | `Optional[int]` | `必填/未声明默认` |
| `output_tokens` | `Optional[int]` | `必填/未声明默认` |
| `latency_ms` | `Optional[float]` | `必填/未声明默认` |
| `response_hash` | `str` | `必填/未声明默认` |
| `metadata` | `Dict[str, Any]` | `field(default_factory=dict)` |

### LLMRecord.to_dict

[实际实现](../factor_optimizer/llm/records.py#L56)。

Serialize to dictionary.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### LLMRecord.from_dict

[实际实现](../factor_optimizer/llm/records.py#L74)。

Deserialize from dictionary.

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'LLMRecord'`。

### compute_hash

[实际实现](../factor_optimizer/llm/records.py#L82)。

Compute SHA256 hash of content.

参数：`(content: str)`。

返回类型：`str`。

### RecordStore

[实际实现](../factor_optimizer/llm/records.py#L87)。

In-memory store for LLM call records.

### RecordStore.__init__

[实际实现](../factor_optimizer/llm/records.py#L94)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### RecordStore.add

[实际实现](../factor_optimizer/llm/records.py#L98)。

Add a record to the store.

参数：`(self, record: LLMRecord)`。

返回类型：`None`。

### RecordStore.get

[实际实现](../factor_optimizer/llm/records.py#L105)。

Retrieve a record by ID.

参数：`(self, record_id: str)`。

返回类型：`Optional[LLMRecord]`。

### RecordStore.list_all

[实际实现](../factor_optimizer/llm/records.py#L109)。

List all records in chronological order.

参数：`(self)`。

返回类型：`List[LLMRecord]`。

### RecordStore.list_by_model

[实际实现](../factor_optimizer/llm/records.py#L113)。

List records for a specific model.

参数：`(self, model: str)`。

返回类型：`List[LLMRecord]`。

### RecordStore.list_by_template

[实际实现](../factor_optimizer/llm/records.py#L117)。

List records for a specific prompt template.

参数：`(self, template_id: str)`。

返回类型：`List[LLMRecord]`。

### RecordStore.total_tokens

[实际实现](../factor_optimizer/llm/records.py#L121)。

Calculate total token usage.

参数：`(self)`。

返回类型：`Dict[str, int]`。

### RecordStore.clear

[实际实现](../factor_optimizer/llm/records.py#L131)。

Clear all records.

参数：`(self)`。

返回类型：`None`。

### RecordStore.export_json

[实际实现](../factor_optimizer/llm/records.py#L136)。

Export all records as JSON.

参数：`(self)`。

返回类型：`str`。

### RecordStore.import_json

[实际实现](../factor_optimizer/llm/records.py#L140)。

Import records from JSON. Returns count of imported records.

参数：`(self, json_str: str)`。

返回类型：`int`。

## factor_optimizer/policy/__init__.py

Policy and decision logic.

显式导出（含重导出）：`DiagnosisKind`、`DiagnosisRecord`、`RepairMapper`、`RepairProposal`、`RepairStrategy`、`AdmissionCriteria`、`AdmissionDecision`、`AdmissionPolicy`、`AdmissionVerdict`、`RejectionReason`。

## factor_optimizer/policy/decisions.py

Admission decision records for mutation proposals.

### AdmissionVerdict

[实际实现](../factor_optimizer/policy/decisions.py#L15)。

Admission decision outcome.

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `ADMITTED` | `类常量/枚举` | `'admitted'` |
| `REJECTED` | `类常量/枚举` | `'rejected'` |
| `DEFERRED` | `类常量/枚举` | `'deferred'` |
| `CONDITIONAL` | `类常量/枚举` | `'conditional'` |

### RejectionReason

[实际实现](../factor_optimizer/policy/decisions.py#L24)。

Reason for rejection.

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `BUDGET_EXCEEDED` | `类常量/枚举` | `'budget_exceeded'` |
| `DUPLICATE` | `类常量/枚举` | `'duplicate'` |
| `ILLEGAL` | `类常量/枚举` | `'illegal'` |
| `DOMAIN_VIOLATION` | `类常量/枚举` | `'domain_violation'` |
| `TIMING_VIOLATION` | `类常量/枚举` | `'timing_violation'` |
| `LOW_EXPECTED_VALUE` | `类常量/枚举` | `'low_expected_value'` |
| `COMPLEXITY_LIMIT` | `类常量/枚举` | `'complexity_limit'` |
| `PARENT_QUALITY` | `类常量/枚举` | `'parent_quality'` |
| `POLICY_VIOLATION` | `类常量/枚举` | `'policy_violation'` |
| `INTEGRITY_EVIDENCE_MISSING` | `类常量/枚举` | `'integrity_evidence_missing'` |
| `INTEGRITY_EVIDENCE_FAILED` | `类常量/枚举` | `'integrity_evidence_failed'` |
| `INTEGRITY_EVIDENCE_STALE` | `类常量/枚举` | `'integrity_evidence_stale'` |

### AdmissionCriteria

[实际实现](../factor_optimizer/policy/decisions.py#L63)。

Criteria used for admission decision.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `max_complexity_cost` | `float` | `float('inf')` |
| `min_expected_value` | `float` | `0.0` |
| `max_lookback_periods` | `int` | `252` |
| `allowed_domains` | `List[str]` | `field(default_factory=list)` |
| `allowed_sources` | `List[str]` | `field(default_factory=list)` |
| `require_parent_evidence` | `bool` | `False` |
| `min_parent_quality` | `float` | `0.0` |
| `budget_constraints` | `Dict[str, Any]` | `field(default_factory=dict)` |

### AdmissionCriteria.to_dict

[实际实现](../factor_optimizer/policy/decisions.py#L87)。

Serialize to dictionary.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### AdmissionCriteria.from_dict

[实际实现](../factor_optimizer/policy/decisions.py#L101)。

Deserialize from dictionary (fail-closed on unknown fields).

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'AdmissionCriteria'`。

### AdmissionDecision

[实际实现](../factor_optimizer/policy/decisions.py#L123)。

Record of an admission decision for a mutation proposal.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `decision_id` | `str` | `必填/未声明默认` |
| `mutation_id` | `str` | `必填/未声明默认` |
| `trial_id` | `str` | `必填/未声明默认` |
| `verdict` | `AdmissionVerdict` | `必填/未声明默认` |
| `decided_at` | `datetime` | `field(default_factory=lambda: datetime.now(timezone.utc))` |
| `criteria_used` | `Optional[AdmissionCriteria]` | `None` |
| `rejection_reasons` | `List[RejectionReason]` | `field(default_factory=list)` |
| `conditions` | `List[str]` | `field(default_factory=list)` |
| `expected_value` | `Optional[float]` | `None` |
| `complexity_cost` | `Optional[float]` | `None` |
| `decision_metadata` | `Dict[str, Any]` | `field(default_factory=dict)` |
| `decided_by` | `str` | `'factor_optimizer'` |

### AdmissionDecision.is_approved

[实际实现](../factor_optimizer/policy/decisions.py#L155)。

Check if mutation is approved for evaluation.

参数：`(self)`。

返回类型：`bool`。

### AdmissionDecision.is_rejected

[实际实现](../factor_optimizer/policy/decisions.py#L159)。

Check if mutation is rejected.

参数：`(self)`。

返回类型：`bool`。

### AdmissionDecision.is_deferred

[实际实现](../factor_optimizer/policy/decisions.py#L163)。

Check if mutation is deferred.

参数：`(self)`。

返回类型：`bool`。

### AdmissionDecision.primary_rejection_reason

[实际实现](../factor_optimizer/policy/decisions.py#L167)。

Get primary rejection reason.

参数：`(self)`。

返回类型：`Optional[RejectionReason]`。

### AdmissionDecision.to_dict

[实际实现](../factor_optimizer/policy/decisions.py#L171)。

Serialize to dictionary.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### AdmissionDecision.from_dict

[实际实现](../factor_optimizer/policy/decisions.py#L189)。

Deserialize from dictionary.

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'AdmissionDecision'`。

### AdmissionPolicy

[实际实现](../factor_optimizer/policy/decisions.py#L205)。

Policy engine for mutation admission decisions.

### AdmissionPolicy.__init__

[实际实现](../factor_optimizer/policy/decisions.py#L212)。

Initialize admission policy.

参数：`(self, criteria: Optional[AdmissionCriteria]=None)`。

### AdmissionPolicy.decide

[实际实现](../factor_optimizer/policy/decisions.py#L222)。

Make admission decision for a mutation proposal.

参数：`(self, mutation_id: str, trial_id: str, complexity_cost: float, expected_value: float, parent_quality: Optional[float]=None, domains: Optional[List[str]]=None, sources: Optional[List[str]]=None, lookback_periods: Optional[int]=None, metadata: Optional[Dict[str, Any]]=None, integrity_evidence: Optional[TreatmentIntegrityEvidence]=None)`。

返回类型：`AdmissionDecision`。

### AdmissionPolicy.get_decision

[实际实现](../factor_optimizer/policy/decisions.py#L347)。

Retrieve decision by ID.

参数：`(self, decision_id: str)`。

返回类型：`Optional[AdmissionDecision]`。

### AdmissionPolicy.get_decisions_for_trial

[实际实现](../factor_optimizer/policy/decisions.py#L351)。

Get all decisions for a trial.

参数：`(self, trial_id: str)`。

返回类型：`List[AdmissionDecision]`。

### AdmissionPolicy.update_criteria

[实际实现](../factor_optimizer/policy/decisions.py#L355)。

Update default admission criteria.

参数：`(self, criteria: AdmissionCriteria)`。

返回类型：`None`。

### AdmissionPolicy.get_admission_stats

[实际实现](../factor_optimizer/policy/decisions.py#L359)。

Get admission decision statistics.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### AdmissionPolicy.clear_history

[实际实现](../factor_optimizer/policy/decisions.py#L386)。

Clear decision history.

参数：`(self)`。

返回类型：`None`。

## factor_optimizer/policy/repair.py

Diagnosis-to-mutation mapping for failure repair strategies.

### DiagnosisKind

[实际实现](../factor_optimizer/policy/repair.py#L8)。

Kind of failure diagnosis.

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `HIGH_COMPLEXITY` | `类常量/枚举` | `'high_complexity'` |
| `POOR_COVERAGE` | `类常量/枚举` | `'poor_coverage'` |
| `HIGH_VARIANCE` | `类常量/枚举` | `'high_variance'` |
| `LOW_SIGNAL` | `类常量/枚举` | `'low_signal'` |
| `HIGH_TURNOVER` | `类常量/枚举` | `'high_turnover'` |
| `TIMING_VIOLATION` | `类常量/枚举` | `'timing_violation'` |
| `DOMAIN_MISMATCH` | `类常量/枚举` | `'domain_mismatch'` |
| `NUMERICAL_INSTABILITY` | `类常量/枚举` | `'numerical_instability'` |
| `SEMANTIC_DUPLICATE` | `类常量/枚举` | `'semantic_duplicate'` |
| `OVERFITTING` | `类常量/枚举` | `'overfitting'` |
| `INTEGRITY_FAILURE` | `类常量/枚举` | `'integrity_failure'` |
| `PIT_VIOLATION` | `类常量/枚举` | `'pit_violation'` |
| `LABEL_TIMING_VIOLATION` | `类常量/枚举` | `'label_timing_violation'` |
| `DATA_QUALITY_FAILURE` | `类常量/枚举` | `'data_quality_failure'` |
| `STALE_DATA` | `类常量/枚举` | `'stale_data'` |
| `HIGH_TIE_RATIO` | `类常量/枚举` | `'high_tie_ratio'` |
| `SPARSE_FACTOR` | `类常量/枚举` | `'sparse_factor'` |
| `LOW_PREDICTIVE` | `类常量/枚举` | `'low_predictive'` |
| `UNSTABLE_IC` | `类常量/枚举` | `'unstable_ic'` |
| `RECENT_DEGRADATION` | `类常量/枚举` | `'recent_degradation'` |
| `OVERFIT_GENERALIZATION` | `类常量/枚举` | `'overfit_generalization'` |
| `LOW_STATISTICAL_CONFIDENCE` | `类常量/枚举` | `'low_statistical_confidence'` |
| `U_SHAPE` | `类常量/枚举` | `'u_shape'` |
| `INVERTED_U` | `类常量/枚举` | `'inverted_u'` |
| `TOP_TAIL_COLLAPSE` | `类常量/枚举` | `'top_tail_collapse'` |
| `BOTTOM_TAIL_COLLAPSE` | `类常量/枚举` | `'bottom_tail_collapse'` |
| `TAIL_ONLY` | `类常量/枚举` | `'tail_only'` |
| `NONSTATIONARY_SHAPE` | `类常量/枚举` | `'nonstationary_shape'` |
| `HIGH_COST_DRAG` | `类常量/枚举` | `'high_cost_drag'` |
| `LOW_CAPACITY` | `类常量/枚举` | `'low_capacity'` |
| `HIGH_DRAWDOWN` | `类常量/枚举` | `'high_drawdown'` |
| `LONG_UNDERWATER` | `类常量/枚举` | `'long_underwater'` |
| `NEGATIVE_TAIL_RISK` | `类常量/枚举` | `'negative_tail_risk'` |
| `REGIME_DEPENDENT` | `类常量/枚举` | `'regime_dependent'` |
| `SIZE_EXPOSURE` | `类常量/枚举` | `'size_exposure'` |
| `INDUSTRY_EXPOSURE` | `类常量/枚举` | `'industry_exposure'` |
| `BETA_EXPOSURE` | `类常量/枚举` | `'beta_exposure'` |
| `LIQUIDITY_EXPOSURE` | `类常量/枚举` | `'liquidity_exposure'` |
| `VOLATILITY_EXPOSURE` | `类常量/枚举` | `'volatility_exposure'` |
| `MULTI_STYLE_EXPOSURE` | `类常量/枚举` | `'multi_style_exposure'` |
| `VALUE_NEAR_DUPLICATE` | `类常量/枚举` | `'value_near_duplicate'` |
| `LOW_NOVELTY` | `类常量/枚举` | `'low_novelty'` |

### DiagnosisKind.canonical_names

[实际实现](../factor_optimizer/policy/repair.py#L79)。

Upper-snake canonical E2 diagnosis kind names (plan E2/FA C7).

参数：`(cls)`。

返回类型：`tuple[str, ...]`。

### DiagnosisKind.canonical_name

[实际实现](../factor_optimizer/policy/repair.py#L88)。

Upper-snake canonical name (e.g. ``HIGH_TURNOVER``).

参数：`(self)`。

返回类型：`str`。

### DiagnosisCategory

[实际实现](../factor_optimizer/policy/repair.py#L93)。

Functional grouping of diagnosis kinds (plan E2 taxonomy families).

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `INTEGRITY` | `类常量/枚举` | `'integrity'` |
| `DATA_QUALITY` | `类常量/枚举` | `'data_quality'` |
| `PREDICTIVE` | `类常量/枚举` | `'predictive'` |
| `STABILITY` | `类常量/枚举` | `'stability'` |
| `GENERALIZATION` | `类常量/枚举` | `'generalization'` |
| `SHAPE` | `类常量/枚举` | `'shape'` |
| `TRADABILITY` | `类常量/枚举` | `'tradability'` |
| `DRAWDOWN_RISK` | `类常量/枚举` | `'drawdown_risk'` |
| `EXPOSURE` | `类常量/枚举` | `'exposure'` |
| `NOVELTY_REDUNDANCY` | `类常量/枚举` | `'novelty_redundancy'` |
| `COMPLEXITY` | `类常量/枚举` | `'complexity'` |

### RepairStrategy

[实际实现](../factor_optimizer/policy/repair.py#L156)。

Repair mutation strategy.

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `REDUCE_WINDOW` | `类常量/枚举` | `'reduce_window'` |
| `INCREASE_WINDOW` | `类常量/枚举` | `'increase_window'` |
| `INCREASE_HORIZON` | `类常量/枚举` | `'increase_horizon'` |
| `ADJUST_DECAY` | `类常量/枚举` | `'adjust_decay'` |
| `THRESHOLD_TUNE` | `类常量/枚举` | `'threshold_tune'` |
| `OPERATOR_SWAP` | `类常量/枚举` | `'operator_swap'` |
| `ADD_INTERACTION` | `类常量/枚举` | `'add_interaction'` |
| `ADD_REGULARIZATION` | `类常量/枚举` | `'add_regularization'` |
| `CHANGE_NORMALIZATION` | `类常量/枚举` | `'change_normalization'` |
| `FILTER_UNIVERSE` | `类常量/枚举` | `'filter_universe'` |
| `WINSORIZE` | `类常量/枚举` | `'winsorize'` |
| `LAG_CORRECTION` | `类常量/枚举` | `'lag_correction'` |
| `ABANDON` | `类常量/枚举` | `'abandon'` |

### canonical_diagnosis_name

[实际实现](../factor_optimizer/policy/repair.py#L225)。

Upper-snake canonical E2 name of a DiagnosisKind.

参数：`(kind: DiagnosisKind)`。

返回类型：`str`。

### DiagnosisRecord

[实际实现](../factor_optimizer/policy/repair.py#L231)。

Record of a failure diagnosis.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `trial_id` | `str` | `必填/未声明默认` |
| `diagnosis_kind` | `DiagnosisKind` | `必填/未声明默认` |
| `severity` | `float` | `0.5` |
| `evidence` | `Dict[str, Any]` | `field(default_factory=dict)` |
| `secondary_diagnoses` | `List[DiagnosisKind]` | `field(default_factory=list)` |
| `metadata` | `Dict[str, Any]` | `field(default_factory=dict)` |

### DiagnosisRecord.to_dict

[实际实现](../factor_optimizer/policy/repair.py#L256)。

Serialize to dictionary.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### DiagnosisRecord.from_dict

[实际实现](../factor_optimizer/policy/repair.py#L268)。

Deserialize from dictionary.

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'DiagnosisRecord'`。

### RepairProposal

[实际实现](../factor_optimizer/policy/repair.py#L281)。

Proposed repair mutation for a diagnosed failure.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `diagnosis_record` | `DiagnosisRecord` | `必填/未声明默认` |
| `strategy` | `RepairStrategy` | `必填/未声明默认` |
| `mutation_type` | `str` | `必填/未声明默认` |
| `parameters` | `Dict[str, Any]` | `必填/未声明默认` |
| `expected_improvement` | `str` | `''` |
| `confidence` | `float` | `0.5` |
| `fallback_strategies` | `List[RepairStrategy]` | `field(default_factory=list)` |

### RepairProposal.to_dict

[实际实现](../factor_optimizer/policy/repair.py#L308)。

Serialize to dictionary.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### RepairMapper

[实际实现](../factor_optimizer/policy/repair.py#L321)。

Maps diagnosis to repair mutation proposals.

### RepairMapper.__init__

[实际实现](../factor_optimizer/policy/repair.py#L328)。

Initialize repair mapper with default rules.

参数：`(self)`。

### RepairMapper.propose_repairs

[实际实现](../factor_optimizer/policy/repair.py#L387)。

Generate repair proposals for a diagnosis.

参数：`(self, diagnosis: DiagnosisRecord, max_proposals: int=3)`。

返回类型：`List[RepairProposal]`。

### RepairMapper.rank_repairs

[实际实现](../factor_optimizer/policy/repair.py#L525)。

Rank repair proposals by priority.

参数：`(self, proposals: List[RepairProposal])`。

返回类型：`List[RepairProposal]`。

### RepairMapper.generate_mutation_spec

[实际实现](../factor_optimizer/policy/repair.py#L543)。

Generate MutationSpec from repair proposal.

参数：`(self, proposal: RepairProposal)`。

返回类型：`Dict[str, Any]`。

## factor_optimizer/policy/repair_registry.py

Versioned repair-family registry + diagnosis-specific repair policy (R61-FI-034).

### ExecutionDomain

[实际实现](../factor_optimizer/policy/repair_registry.py#L36)。

Owner execution domain of a repair family (plan §20 E3).

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `FE` | `类常量/枚举` | `'FE'` |
| `FP` | `类常量/枚举` | `'FP'` |

### CausalityClass

[实际实现](../factor_optimizer/policy/repair_registry.py#L52)。

Causality classification of a repair family (plan §20 E3 / matrix notes).

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `CAUSAL` | `类常量/枚举` | `'causal'` |
| `STATELESS` | `类常量/枚举` | `'stateless'` |
| `STRUCTURAL` | `类常量/枚举` | `'structural'` |
| `ABANDON` | `类常量/枚举` | `'abandon'` |
| `UNKNOWN` | `类常量/枚举` | `'unknown'` |

### RepairFamily

[实际实现](../factor_optimizer/policy/repair_registry.py#L69)。

The 20 canonical repair families (plan §20 E3).

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `NO_OP_RAW` | `类常量/枚举` | `'no_op_raw'` |
| `SIGN_ORIENTATION` | `类常量/枚举` | `'sign_orientation'` |
| `WINDOW_REFINEMENT` | `类常量/枚举` | `'window_refinement'` |
| `DECAY_REFINEMENT` | `类常量/枚举` | `'decay_refinement'` |
| `CAUSAL_SMOOTHING` | `类常量/枚举` | `'causal_smoothing'` |
| `ROBUST_OUTLIER` | `类常量/枚举` | `'robust_outlier'` |
| `MISSINGNESS_FRESHNESS` | `类常量/枚举` | `'missingness_freshness'` |
| `U_SHAPE_REPAIR` | `类常量/枚举` | `'u_shape_repair'` |
| `INVERTED_U_REPAIR` | `类常量/枚举` | `'inverted_u_repair'` |
| `TAIL_SATURATION` | `类常量/枚举` | `'tail_saturation'` |
| `TAIL_HINGE` | `类常量/枚举` | `'tail_hinge'` |
| `INDUSTRY_NEUTRALIZATION` | `类常量/枚举` | `'industry_neutralization'` |
| `SIZE_NEUTRALIZATION` | `类常量/枚举` | `'size_neutralization'` |
| `STYLE_NEUTRALIZATION` | `类常量/枚举` | `'style_neutralization'` |
| `REPRESENTATION_RANK` | `类常量/枚举` | `'representation_rank'` |
| `REPRESENTATION_ZSCORE` | `类常量/枚举` | `'representation_zscore'` |
| `ROBUST_SCALE` | `类常量/枚举` | `'robust_scale'` |
| `OPERATOR_SWAP` | `类常量/枚举` | `'operator_swap'` |
| `LOW_DOF_INTERACTION` | `类常量/枚举` | `'low_dof_interaction'` |
| `ABANDON` | `类常量/枚举` | `'abandon'` |

### RepairFamily.names

[实际实现](../factor_optimizer/policy/repair_registry.py#L94)。

Upper-snake canonical family names.

参数：`(cls)`。

返回类型：`tuple[str, ...]`。

### ParameterSchema

[实际实现](../factor_optimizer/policy/repair_registry.py#L104)。

Declared parameter schema of a repair family (plan §20 E3).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `params` | `Mapping[str, str]` | `field(default_factory=dict)` |

### ParameterSchema.parameter_names

[实际实现](../factor_optimizer/policy/repair_registry.py#L163)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`tuple[str, ...]`。

### ParameterSchema.to_dict

[实际实现](../factor_optimizer/policy/repair_registry.py#L166)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, str]`。

### ParameterPrior

[实际实现](../factor_optimizer/policy/repair_registry.py#L171)。

Declared parameter prior of a repair family (plan §20 E3 / §24).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `prior` | `Mapping[str, Any]` | `field(default_factory=dict)` |

### ParameterPrior.value_of

[实际实现](../factor_optimizer/policy/repair_registry.py#L189)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, name: str)`。

返回类型：`Optional[Any]`。

### ParameterPrior.to_dict

[实际实现](../factor_optimizer/policy/repair_registry.py#L192)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### RepairFamilyDeclaration

[实际实现](../factor_optimizer/policy/repair_registry.py#L197)。

One versioned repair-family declaration (plan §20 E3 fields).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `family` | `RepairFamily` | `必填/未声明默认` |
| `eligible_diagnoses` | `FrozenSet[str]` | `必填/未声明默认` |
| `owner` | `ExecutionDomain` | `必填/未声明默认` |
| `parameter_schema` | `ParameterSchema` | `field(default_factory=ParameterSchema)` |
| `parameter_prior` | `ParameterPrior` | `field(default_factory=ParameterPrior)` |
| `maximum_candidates` | `int` | `2` |
| `causality_class` | `CausalityClass` | `CausalityClass.UNKNOWN` |
| `complexity_cost` | `int` | `1` |
| `required_evidence_profile` | `str` | `'CHEAP_SCREEN_CN_1D'` |
| `policy_id` | `str` | `'FO_REPAIR_FAMILY'` |
| `policy_version` | `str` | `'1.0.0'` |

### RepairFamilyDeclaration.family_name

[实际实现](../factor_optimizer/policy/repair_registry.py#L285)。

Upper-snake canonical family name (``CAUSAL_SMOOTHING``).

参数：`(self)`。

返回类型：`str`。

### RepairFamilyDeclaration.eligible_for

[实际实现](../factor_optimizer/policy/repair_registry.py#L289)。

True when this family may repair *diagnosis*.

参数：`(self, diagnosis: DiagnosisKind)`。

返回类型：`bool`。

### RepairFamilyDeclaration.to_dict

[实际实现](../factor_optimizer/policy/repair_registry.py#L293)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### RepairFamilyRegistry

[实际实现](../factor_optimizer/policy/repair_registry.py#L650)。

Versioned registry of repair-family declarations (plan §20 E3).

### RepairFamilyRegistry.__init__

[实际实现](../factor_optimizer/policy/repair_registry.py#L660)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, declarations: Sequence[RepairFamilyDeclaration], policy_id: str='FO_REPAIR_FAMILY', policy_version: str='1.0.0')`。

返回类型：`None`。

### RepairFamilyRegistry.default

[实际实现](../factor_optimizer/policy/repair_registry.py#L689)。

The canonical 20-family registry (plan §20 E3).

参数：`(cls)`。

返回类型：`'RepairFamilyRegistry'`。

### RepairFamilyRegistry.policy_id

[实际实现](../factor_optimizer/policy/repair_registry.py#L694)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### RepairFamilyRegistry.policy_version

[实际实现](../factor_optimizer/policy/repair_registry.py#L698)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### RepairFamilyRegistry.family_names

[实际实现](../factor_optimizer/policy/repair_registry.py#L702)。

Upper-snake names of every declared family, sorted.

参数：`(self)`。

返回类型：`tuple[str, ...]`。

### RepairFamilyRegistry.declarations

[实际实现](../factor_optimizer/policy/repair_registry.py#L707)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Tuple[RepairFamilyDeclaration, ...]`。

### RepairFamilyRegistry.get

[实际实现](../factor_optimizer/policy/repair_registry.py#L710)。

Resolve a family declaration (by member or upper-snake name).

参数：`(self, family: RepairFamily \| str)`。

返回类型：`RepairFamilyDeclaration`。

### RepairFamilyRegistry.eligible_families_for

[实际实现](../factor_optimizer/policy/repair_registry.py#L728)。

Declarations eligible for a diagnosis, in registry order.

参数：`(self, diagnosis: DiagnosisKind \| str)`。

返回类型：`Tuple[RepairFamilyDeclaration, ...]`。

### RepairFamilyRegistry.to_dict

[实际实现](../factor_optimizer/policy/repair_registry.py#L747)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### DomainToken

[实际实现](../factor_optimizer/policy/repair_registry.py#L774)。

Taxonomy/evidence tokens a repair rule may condition on.

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `PRICE_VOLUME` | `类常量/枚举` | `'price_volume'` |
| `OUTLIER_EVIDENCE` | `类常量/枚举` | `'outlier_evidence'` |
| `SIZE_CONCENTRATION` | `类常量/枚举` | `'size_concentration'` |
| `STABLE_SHAPE_CONFIDENCE` | `类常量/枚举` | `'stable_shape_confidence'` |
| `ALPHA_AFTER_RESIDUAL` | `类常量/枚举` | `'alpha_after_residual'` |
| `NONE` | `类常量/枚举` | `'none'` |

### classify_evidence_context

[实际实现](../factor_optimizer/policy/repair_registry.py#L792)。

Deterministic, conservative tokenization of a repair evidence context.

参数：`(evidence: Optional[Mapping[str, object]])`。

返回类型：`frozenset[DomainToken]`。

### RepairRule

[实际实现](../factor_optimizer/policy/repair_registry.py#L845)。

One diagnosis→repair-family rule (plan §20 E4).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `rule_id` | `str` | `必填/未声明默认` |
| `diagnosis` | `str` | `必填/未声明默认` |
| `allowed_families` | `Tuple[str, ...]` | `必填/未声明默认` |
| `requires_tokens` | `Tuple[DomainToken, ...]` | `()` |
| `rationale` | `str` | `''` |

### DiagnosisRepairPolicy

[实际实现](../factor_optimizer/policy/repair_registry.py#L881)。

Versioned diagnosis→repair-family rule table (plan §20 E4).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `policy_id` | `str` | `'FO_DIAGNOSIS_REPAIR'` |
| `policy_version` | `str` | `'1.0.0'` |
| `rules` | `Tuple[RepairRule, ...]` | `()` |

### DiagnosisRepairPolicy.families_for

[实际实现](../factor_optimizer/policy/repair_registry.py#L906)。

Ordered repair-family names allowed for a diagnosis.

参数：`(self, diagnosis: DiagnosisKind \| str, evidence: Optional[Mapping[str, object]]=None, domain_token: Optional[DomainToken]=None)`。

返回类型：`Tuple[str, ...]`。

### default_diagnosis_repair_policy

[实际实现](../factor_optimizer/policy/repair_registry.py#L1243)。

The canonical diagnosis-repair policy (plan §20 E4).

参数：`()`。

返回类型：`DiagnosisRepairPolicy`。

### RepairBudgetConfig

[实际实现](../factor_optimizer/policy/repair_registry.py#L1254)。

Configurable search-budget bounds (plan §20 E5).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `max_primary_diagnoses` | `int` | `3` |
| `max_repair_families_per_diagnosis` | `int` | `2` |
| `min_candidates_per_family` | `int` | `2` |
| `max_candidates_per_family` | `int` | `4` |
| `min_total_candidates` | `int` | `6` |
| `max_total_candidates` | `int` | `12` |

### RepairBudgetConfig.candidates_per_family_range

[实际实现](../factor_optimizer/policy/repair_registry.py#L1308)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Tuple[int, int]`。

### RepairCandidateSlot

[实际实现](../factor_optimizer/policy/repair_registry.py#L1313)。

One planned candidate slot in a diagnosis-directed search budget.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `diagnosis` | `str` | `必填/未声明默认` |
| `family` | `str` | `必填/未声明默认` |
| `max_candidates` | `int` | `2` |

### DiagnosisSearchBudget

[实际实现](../factor_optimizer/policy/repair_registry.py#L1342)。

A concrete, bounded candidate plan for one factor (plan §20 E5).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `primary_diagnoses` | `Tuple[str, ...]` | `必填/未声明默认` |
| `slots` | `Tuple[RepairCandidateSlot, ...]` | `必填/未声明默认` |
| `max_total_candidates` | `int` | `12` |
| `min_total_candidates` | `int` | `6` |
| `policy_id` | `str` | `'FO_DIAGNOSIS_REPAIR'` |
| `policy_version` | `str` | `'1.0.0'` |

### DiagnosisSearchBudget.total_candidates_ceiling

[实际实现](../factor_optimizer/policy/repair_registry.py#L1364)。

Sum of every slot's candidate allowance.

参数：`(self)`。

返回类型：`int`。

### DiagnosisSearchBudget.within_soft_budget

[实际实现](../factor_optimizer/policy/repair_registry.py#L1368)。

True when the plan respects the soft 6-12 total candidate budget.

参数：`(self)`。

返回类型：`bool`。

### DiagnosisSearchBudget.to_dict

[实际实现](../factor_optimizer/policy/repair_registry.py#L1378)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### diagnosis_search_budget

[实际实现](../factor_optimizer/policy/repair_registry.py#L1396)。

Generate a concrete candidate plan from diagnoses (plan §20 E5).

参数：`(diagnoses: Sequence[DiagnosisKind \| str], *, policy: Optional[DiagnosisRepairPolicy]=None, registry: Optional[RepairFamilyRegistry]=None, evidence: Optional[Mapping[str, object]]=None, config: Optional[RepairBudgetConfig]=None)`。

返回类型：`DiagnosisSearchBudget`。

## factor_optimizer/ports/__init__.py

Ports: narrow consumer-side contracts FO binds other packages through.

显式导出（含重导出）：`FactorIntelligenceProvider`、`FactorTaxonomyView`、`FactorHealthView`、`DiagnosisView`、`HealthDimension`、`HealthGrade`、`DiagnosisSeverity`、`FactorIntelligenceView`、`FactorIntelligenceUnknownFactorError`。

## factor_optimizer/ports/factor_intelligence.py

Factor Intelligence provider port (R61-FI-014 / plan §20 E6, matrix E6).

显式导出（含重导出）：`HealthGrade`、`HealthDimension`、`DiagnosisSeverity`、`DiagnosisRepairability`、`FactorTaxonomyView`、`FactorHealthView`、`DiagnosisView`、`FactorIntelligenceProvider`、`FactorIntelligenceView`、`FactorIntelligenceUnknownFactorError`、`unknown_factor_view`、`SelectionDecisionRequestView`、`SelectionDecisionReceiptView`、`DecisionProvider`、`require_bound_decision_receipt`。

### HealthGrade

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L35)。

Grade letters with S+ best, D worst (plan §7 C5 / matrix C5).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `S_PLUS` | `类常量/枚举` | `'S+'` |
| `S` | `类常量/枚举` | `'S'` |
| `A_PLUS` | `类常量/枚举` | `'A+'` |
| `A` | `类常量/枚举` | `'A'` |
| `B_PLUS` | `类常量/枚举` | `'B+'` |
| `B` | `类常量/枚举` | `'B'` |
| `C` | `类常量/枚举` | `'C'` |
| `D` | `类常量/枚举` | `'D'` |
| `NONE` | `类常量/枚举` | `'NONE'` |

### HealthGrade.all

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L54)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls)`。

返回类型：`tuple[str, ...]`。

### HealthDimension

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L58)。

Canonical 14 health dimensions (plan §7/§8, matrix C5 '14 dims').

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `PREDICTIVE_POWER` | `类常量/枚举` | `'predictive_power'` |
| `STABILITY` | `类常量/枚举` | `'stability'` |
| `ROBUSTNESS` | `类常量/枚举` | `'robustness'` |
| `TURNOVER` | `类常量/枚举` | `'turnover'` |
| `CAPACITY` | `类常量/枚举` | `'capacity'` |
| `COST_DRAG` | `类常量/枚举` | `'cost_drag'` |
| `DRAWDOWN` | `类常量/枚举` | `'drawdown'` |
| `TAIL_RISK` | `类常量/枚举` | `'tail_risk'` |
| `DATA_COVERAGE` | `类常量/枚举` | `'data_coverage'` |
| `FRESHNESS` | `类常量/枚举` | `'freshness'` |
| `COMPLEXITY` | `类常量/枚举` | `'complexity'` |
| `ECONOMIC_SENSE` | `类常量/枚举` | `'economic_sense'` |
| `SHAPE_QUALITY` | `类常量/枚举` | `'shape_quality'` |
| `REGIME_SENSITIVITY` | `类常量/枚举` | `'regime_sensitivity'` |

### HealthDimension.all

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L83)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls)`。

返回类型：`tuple[str, ...]`。

### DiagnosisSeverity

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L102)。

Diagnosis severity bands (plan §9 C7).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `CRITICAL` | `类常量/枚举` | `'CRITICAL'` |
| `HIGH` | `类常量/枚举` | `'HIGH'` |
| `MEDIUM` | `类常量/枚举` | `'MEDIUM'` |
| `LOW` | `类常量/枚举` | `'LOW'` |
| `INFO` | `类常量/枚举` | `'INFO'` |
| `UNKNOWN` | `类常量/枚举` | `'UNKNOWN'` |

### DiagnosisSeverity.all

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L113)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls)`。

返回类型：`tuple[str, ...]`。

### DiagnosisRepairability

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L117)。

Whether/how a diagnosis can be repaired (plan §9 / FO repair vocabulary).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `REPAIRABLE` | `类常量/枚举` | `'REPAIRABLE'` |
| `PARTIALLY_REPAIRABLE` | `类常量/枚举` | `'PARTIALLY_REPAIRABLE'` |
| `UNREPAIRABLE` | `类常量/枚举` | `'UNREPAIRABLE'` |
| `UNKNOWN` | `类常量/枚举` | `'UNKNOWN'` |

### DiagnosisRepairability.all

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L126)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls)`。

返回类型：`tuple[str, ...]`。

### FactorTaxonomyView

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L136)。

Consumer view of a factor's taxonomy (narrow projection of the FA artifact).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `factor_definition_id` | `str` | `必填/未声明默认` |
| `data_domains` | `tuple[str, ...]` | `()` |
| `display_family` | `str` | `'UNKNOWN'` |
| `mechanism_tags` | `tuple[tuple[str, str, float, tuple[str, ...]], ...]` | `()` |
| `is_unknown` | `bool` | `False` |

### FactorTaxonomyView.mechanism_tag_names

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L177)。

Tag names only (no evidence) for quick checks.

参数：`(self)`。

返回类型：`tuple[str, ...]`。

### FactorHealthView

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L183)。

Consumer view of a factor's health card (plan §7/§8, matrix C5).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `factor_definition_id` | `str` | `必填/未声明默认` |
| `evaluation_ref` | `str` | `''` |
| `dimension_grades` | `dict[str, str]` | `None` |
| `overall_grade` | `str` | `HealthGrade.NONE` |
| `is_unknown` | `bool` | `False` |

### FactorHealthView.grade_of

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L220)。

Grade of one dimension (``HealthGrade.NONE`` when absent/unknown).

参数：`(self, dimension: str)`。

返回类型：`str`。

### FactorHealthView.all_dimensions_graded

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L225)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`bool`。

### DiagnosisView

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L230)。

Consumer view of one diagnosis (plan §9 C7).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `factor_definition_id` | `str` | `必填/未声明默认` |
| `health_ref` | `str` | `''` |
| `tag` | `str` | `''` |
| `severity` | `str` | `DiagnosisSeverity.UNKNOWN` |
| `confidence` | `float` | `0.0` |
| `repairability` | `str` | `DiagnosisRepairability.UNKNOWN` |
| `details` | `tuple[str, ...]` | `()` |

### FactorIntelligenceProvider

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L273)。

Narrow consumer boundary FO binds to FA factor intelligence.

基类：`Protocol`。

### FactorIntelligenceProvider.get_taxonomy

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L283)。

Taxonomy view for a factor definition.

参数：`(self, factor_definition_id: str)`。

返回类型：`FactorTaxonomyView`。

### FactorIntelligenceProvider.get_health_card

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L291)。

Health-card view for a factor definition + evaluation.

参数：`(self, factor_definition_id: str, evaluation_ref: str)`。

返回类型：`FactorHealthView`。

### FactorIntelligenceProvider.get_diagnoses

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L297)。

Diagnoses for a factor definition + health-card ref.

参数：`(self, factor_definition_id: str, health_ref: str)`。

返回类型：`Sequence[DiagnosisView]`。

### SelectionDecisionRequestView

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L305)。

Structural FO view of FA's authoritative decision request.

基类：`Protocol`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `request_id` | `str` | `必填/未声明默认` |
| `policy_id` | `str` | `必填/未声明默认` |
| `policy_content_hash` | `str` | `必填/未声明默认` |
| `comparison_context_hash` | `str` | `必填/未声明默认` |
| `purpose` | `str` | `必填/未声明默认` |
| `decision_level` | `str` | `必填/未声明默认` |
| `baseline_ref` | `Optional[str]` | `必填/未声明默认` |
| `candidates` | `Sequence[Any]` | `必填/未声明默认` |
| `required_final_fidelity` | `str` | `必填/未声明默认` |
| `hypothesis_family_ref` | `str` | `必填/未声明默认` |

### SelectionDecisionRequestView.content_hash

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L320)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### SelectionDecisionRequestView.candidate_set_hash

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L323)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### SelectionDecisionReceiptView

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L327)。

Structural FO view of the immutable FA decision receipt.

基类：`Protocol`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `request_id` | `str` | `必填/未声明默认` |
| `request_hash` | `str` | `必填/未声明默认` |
| `policy_id` | `str` | `必填/未声明默认` |
| `policy_content_hash` | `str` | `必填/未声明默认` |
| `comparison_context_hash` | `str` | `必填/未声明默认` |
| `candidate_set_hash` | `str` | `必填/未声明默认` |
| `decision_id` | `str` | `必填/未声明默认` |
| `content_hash` | `str` | `必填/未声明默认` |
| `status` | `Any` | `必填/未声明默认` |
| `eligibility` | `Mapping[str, bool]` | `必填/未声明默认` |
| `gate_receipts` | `Sequence[Any]` | `必填/未声明默认` |
| `point_utility` | `Mapping[str, Optional[float]]` | `必填/未声明默认` |
| `conservative_utility` | `Mapping[str, Optional[float]]` | `必填/未声明默认` |
| `relationship` | `Mapping[str, Any]` | `必填/未声明默认` |
| `effect_refs` | `Mapping[str, Optional[str]]` | `必填/未声明默认` |
| `qualification_scope` | `Mapping[str, Optional[str]]` | `必填/未声明默认` |
| `reasons` | `Sequence[str]` | `必填/未声明默认` |
| `winner_id` | `Optional[str]` | `必填/未声明默认` |
| `retained_ids` | `Sequence[str]` | `必填/未声明默认` |
| `final_fidelity` | `str` | `必填/未声明默认` |

### DecisionProvider

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L353)。

Sole selection authority consumed by FO; implementations live in FA.

基类：`Protocol`。

### DecisionProvider.decide

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L356)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, request: SelectionDecisionRequestView)`。

返回类型：`SelectionDecisionReceiptView`。

### require_bound_decision_receipt

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L361)。

Fail closed unless an FA receipt is bound to the exact FO request.

参数：`(request: SelectionDecisionRequestView, receipt: SelectionDecisionReceiptView)`。

返回类型：`SelectionDecisionReceiptView`。

### FactorIntelligenceView

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L396)。

Union-of-views sentinel: one unknown id maps to one explicit view.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `taxonomy` | `FactorTaxonomyView` | `必填/未声明默认` |
| `health` | `FactorHealthView` | `必填/未声明默认` |
| `diagnoses` | `tuple[DiagnosisView, ...]` | `必填/未声明默认` |

### FactorIntelligenceUnknownFactorError

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L416)。

Typed error for an unknown factor definition id.

基类：`Exception`。

### FactorIntelligenceUnknownFactorError.__init__

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L424)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, factor_definition_id: str)`。

### unknown_factor_view

[实际实现](../factor_optimizer/ports/factor_intelligence.py#L429)。

Explicit unknown-factor view set for providers that don't raise.

参数：`(factor_definition_id: str)`。

返回类型：`FactorIntelligenceView`。

## factor_optimizer/research_baseline.py

Research baseline recipes and TRAIN-only substantial-degradation guard.

### BaselinePlan

[实际实现](../factor_optimizer/research_baseline.py#L21)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `operations` | `tuple[str, ...]` | `必填/未声明默认` |
| `omissions` | `tuple[str, ...]` | `必填/未声明默认` |
| `exposure_columns` | `tuple[str, ...]` | `必填/未声明默认` |
| `training_context_ref` | `str` | `必填/未声明默认` |
| `minimum_assets` | `int` | `20` |
| `lower_quantile` | `float` | `0.01` |
| `upper_quantile` | `float` | `0.99` |

### BaselinePlan.identity

[实际实现](../factor_optimizer/research_baseline.py#L31)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### BaselinePlan.execute

[实际实现](../factor_optimizer/research_baseline.py#L38)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, values, *, exposures=None, allow_research=False)`。

### BaselineRepairPlan

[实际实现](../factor_optimizer/research_baseline.py#L81)。

Replay the TRAIN-frozen baseline followed by one frozen value repair.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `baseline` | `BaselinePlan` | `必填/未声明默认` |
| `repair` | `object` | `必填/未声明默认` |

### BaselineRepairPlan.family

[实际实现](../factor_optimizer/research_baseline.py#L91)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### BaselineRepairPlan.identity

[实际实现](../factor_optimizer/research_baseline.py#L95)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### BaselineRepairPlan.execute

[实际实现](../factor_optimizer/research_baseline.py#L98)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, values, *, exposures=None, allow_research=False)`。

### compile_baseline

[实际实现](../factor_optimizer/research_baseline.py#L105)。

Frozen winsor -> OLS -> rank recipe; do not repeat historical CS rank.

参数：`(lineage, *, training_context_ref, exposure_columns=(), minimum_assets=20)`。

### assess_baseline_training

[实际实现](../factor_optimizer/research_baseline.py#L143)。

Reject material TRAIN RankIC loss with a paired moving-block upper bound.

参数：`(raw, candidate, batch, labels, split, config, *, maximum_loss=0.01)`。

## factor_optimizer/research_batch.py

Bounded research batch optimization with automatic chronological splitting.

显式导出（含重导出）：`BatchOptimizationConfig`、`AutomaticTimeSplit`、`FactorOptimizationResult`、`BatchOptimizationResult`、`OrientedRepairPlan`、`automatic_time_split`、`optimize_factor_batch`。

### BatchOptimizationConfig

[实际实现](../factor_optimizer/research_batch.py#L21)。

Conservative defaults; callers need not choose calendar cutoffs.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `train_fraction` | `float` | `0.6` |
| `validation_fraction` | `float` | `0.2` |
| `warmup_bars` | `int` | `30` |
| `embargo_bars` | `int` | `1` |
| `minimum_train_days` | `int` | `60` |
| `minimum_validation_days` | `int` | `30` |
| `minimum_test_days` | `int` | `30` |
| `minimum_assets` | `int` | `20` |
| `minimum_coverage` | `float` | `0.9` |
| `minimum_improvement` | `float` | `0.01` |
| `confidence_level` | `float` | `0.95` |
| `bootstrap_draws` | `int` | `499` |
| `block_length` | `int` | `5` |
| `seed` | `int` | `20260921` |
| `natural_time_scale` | `float` | `10.0` |
| `families` | `tuple[str, ...]` | `()` |
| `maximum_candidates` | `int` | `128` |
| `compose_smoothing_sign` | `bool` | `True` |
| `selection_objective` | `str` | `'joint'` |
| `research_cost_rate` | `float` | `0.001` |

### AutomaticTimeSplit

[实际实现](../factor_optimizer/research_batch.py#L85)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `train_indices` | `tuple[int, ...]` | `必填/未声明默认` |
| `validation_indices` | `tuple[int, ...]` | `必填/未声明默认` |
| `test_indices` | `tuple[int, ...]` | `必填/未声明默认` |
| `validation_start` | `int` | `必填/未声明默认` |
| `test_start` | `int` | `必填/未声明默认` |
| `identity` | `str` | `必填/未声明默认` |

### FactorOptimizationResult

[实际实现](../factor_optimizer/research_batch.py#L95)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `factor_id` | `str` | `必填/未声明默认` |
| `status` | `str` | `必填/未声明默认` |
| `selected_family` | `str` | `必填/未声明默认` |
| `plan_identity` | `str` | `必填/未声明默认` |
| `plan` | `Any` | `必填/未声明默认` |
| `train_gain` | `float \| None` | `必填/未声明默认` |
| `validation_lower_bound` | `float \| None` | `必填/未声明默认` |
| `candidates` | `tuple[Mapping[str, Any], ...]` | `必填/未声明默认` |
| `reason` | `str` | `必填/未声明默认` |
| `validation_candidate_identity` | `str \| None` | `None` |
| `validation_coverage` | `float \| None` | `None` |
| `training_diagnostics` | `Mapping[str, Any] \| None` | `None` |
| `baseline_diagnostics` | `Mapping[str, Any] \| None` | `None` |
| `joint_diagnostics` | `Mapping[str, Any] \| None` | `None` |
| `materialization_error` | `str \| None` | `None` |

### OrientedRepairPlan

[实际实现](../factor_optimizer/research_batch.py#L114)。

Frozen sign after a temporal repair; both decisions are chosen on TRAIN.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `base` | `Any` | `必填/未声明默认` |
| `multiplier` | `int` | `-1` |

### OrientedRepairPlan.family

[实际实现](../factor_optimizer/research_batch.py#L124)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### OrientedRepairPlan.identity

[实际实现](../factor_optimizer/research_batch.py#L128)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### OrientedRepairPlan.execute

[实际实现](../factor_optimizer/research_batch.py#L131)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, values, *, allow_research=False)`。

### BatchOptimizationResult

[实际实现](../factor_optimizer/research_batch.py#L136)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `optimized` | `Any` | `必填/未声明默认` |
| `factors` | `Mapping[str, FactorOptimizationResult]` | `必填/未声明默认` |
| `split` | `AutomaticTimeSplit` | `必填/未声明默认` |
| `execution_mode` | `str` | `'research_only'` |
| `test_evaluated` | `bool` | `False` |
| `config` | `BatchOptimizationConfig \| None` | `None` |

### automatic_time_split

[实际实现](../factor_optimizer/research_batch.py#L145)。

60/20/20 in time, warmup, and purge real label windows at boundaries.

参数：`(labels, config: BatchOptimizationConfig \| None=None)`。

### PairICCache

[实际实现](../factor_optimizer/research_batch.py#L179)。

At most two RAW IC references, private to one factor's TRAIN search.

### PairICCache.__init__

[实际实现](../factor_optimizer/research_batch.py#L186)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### PairICCache.get

[实际实现](../factor_optimizer/research_batch.py#L190)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, key)`。

### PairICCache.put

[实际实现](../factor_optimizer/research_batch.py#L196)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, key, value)`。

### optimize_factor_batch

[实际实现](../factor_optimizer/research_batch.py#L320)。

Optimize aligned QE contracts automatically, preserving every input ID.

参数：`(batch, labels, *, config=None, allow_research=False, lineages=None, exposures=None, exposure_columns=(), maximum_baseline_loss=0.01)`。

## factor_optimizer/research_decay.py

TRAIN-only twenty-layer stale-signal decay, not holding-period PnL.

### diagnose_layer_decay

[实际实现](../factor_optimizer/research_decay.py#L7)。

Compare x[t-lag] with the same y[t] and common TRAIN dates at each lag.

参数：`(batch, labels, split, config, factor_index, *, minimum_assets_per_quantile=10)`。

## factor_optimizer/research_diagnostics.py

TRAIN-only multi-dimensional research diagnosis using QE metric authorities.

### diagnose_training_batch

[实际实现](../factor_optimizer/research_diagnostics.py#L15)。

Inspect 20 real quantile bins, IC/ICIR and gross-one spread risk on TRAIN.

参数：`(batch, labels, *, config=None, periods_per_year=252, minimum_assets_per_quantile=10)`。

## factor_optimizer/research_final_report.py

Authority-side final research reporting; never called by candidate search.

### FrozenSelection

[实际实现](../factor_optimizer/research_final_report.py#L40)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `raw` | `object` | `必填/未声明默认` |
| `result` | `object` | `必填/未声明默认` |
| `dataset_identity` | `str` | `必填/未声明默认` |
| `selection_hash` | `str` | `必填/未声明默认` |
| `profile_hash` | `str` | `必填/未声明默认` |
| `split_identity` | `str` | `必填/未声明默认` |

### freeze_selection

[实际实现](../factor_optimizer/research_final_report.py#L57)。

Freeze identities without reading any label; detect later buffer changes.

参数：`(raw, result, *, dataset_identity)`。

### evaluate_frozen

[实际实现](../factor_optimizer/research_final_report.py#L130)。

Read the authority store once; persist TEST and separate description.

参数：`(frozen, broker)`。

## factor_optimizer/research_fitness.py

QE-owned research portfolio metrics and joint paired selection policy.

### portfolio_series

[实际实现](../factor_optimizer/research_fitness.py#L13)。

Top/bottom quintiles, gross-one equal stock weights, signal-only membership.

参数：`(values, returns, *, cost_rate=0.001)`。

### summarize

[实际实现](../factor_optimizer/research_fitness.py#L39)。

Summarize aligned columns [RankIC, net portfolio return, full turnover].

参数：`(series, *, periods_per_year=252)`。

### joint_utility

[实际实现](../factor_optimizer/research_fitness.py#L65)。

Prespecified bounded research utility; never fitted to VALIDATION/TEST.

参数：`(m)`。

### passes_floors

[实际实现](../factor_optimizer/research_fitness.py#L72)。

Hard raw-relative guards prevent one metric buying material damage.

参数：`(raw, candidate)`。

### RawSeriesCache

[实际实现](../factor_optimizer/research_fitness.py#L82)。

One-entry RAW evidence cache; returned arrays never alias stored state.

### RawSeriesCache.__init__

[实际实现](../factor_optimizer/research_fitness.py#L85)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### RawSeriesCache.get

[实际实现](../factor_optimizer/research_fitness.py#L89)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, key)`。

### RawSeriesCache.put

[实际实现](../factor_optimizer/research_fitness.py#L92)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, key, series)`。

### paired_series

[实际实现](../factor_optimizer/research_fitness.py#L96)。

QE metric inputs share signal availability, never ex-post label membership.

参数：`(raw, candidate, batch, labels, indices, *, minimum_assets=20, cost_rate=0.001, raw_cache=None)`。

### compare_joint

[实际实现](../factor_optimizer/research_fitness.py#L141)。

Recompute all nonlinear metrics within each shared moving-block draw.

参数：`(raw_series, candidate_series, config)`。

## factor_optimizer/research_manifest.py

Bind declared COS factor values to their exact research landing record.

### BoundResearchFactor

[实际实现](../factor_optimizer/research_manifest.py#L11)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `factor_id` | `str` | `必填/未声明默认` |
| `factor` | `object` | `必填/未声明默认` |
| `manifest_uri` | `str` | `必填/未声明默认` |
| `manifest_sha256` | `str` | `必填/未声明默认` |
| `expression` | `str` | `必填/未声明默认` |
| `source_status` | `str` | `必填/未声明默认` |
| `contains_cs_rank` | `bool` | `必填/未声明默认` |
| `output_is_cs_rank` | `bool` | `必填/未声明默认` |
| `lineage` | `object` | `必填/未声明默认` |

### read_bound_factor

[实际实现](../factor_optimizer/research_manifest.py#L83)。

Read declared datasets through DataAccess; enforce URI, bytes and SHA256.

参数：`(store, manifest_dataset, factor_dataset, factor_id, *, manifest_params=None, factor_params=None, allow_research=False)`。

## factor_optimizer/search/__init__.py

Search orchestration for factor mutation optimization.

显式导出（含重导出）：`SearchRunner`、`SearchConfig`、`SearchSession`、`FidelityTier`、`FidelitySpec`、`MultiFidelityScheduler`、`PromotionCriteria`、`ParetoPoint`、`ParetoFrontier`、`ParetoArchive`、`PlateauDetector`、`PlateauConfig`、`AdaptivePlateauDetector`、`MultiObjectivePlateauDetector`、`LineageNode`、`LineageTree`、`LineageAnalyzer`、`ParameterSpace`、`SearchSpace`、`SearchStrategy`、`SearchStrategyState`、`RandomSearch`、`GridSearch`、`BayesianSearch`、`TPESearch`。

## factor_optimizer/search/categorical_strategy.py

Categorical search strategy (TPE-style) for treatment auto-optimization.

### CategoricalRecipe

[实际实现](../factor_optimizer/search/categorical_strategy.py#L43)。

One discrete treatment-recipe candidate in the categorical space.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `name` | `str` | `必填/未声明默认` |
| `param_ranges` | `Dict[str, List[Any]]` | `field(default_factory=dict)` |

### CategoricalRecipe.to_dict

[实际实现](../factor_optimizer/search/categorical_strategy.py#L75)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### CategoricalRecipe.from_dict

[实际实现](../factor_optimizer/search/categorical_strategy.py#L79)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'CategoricalRecipe'`。

### CategoricalSearchStrategy

[实际实现](../factor_optimizer/search/categorical_strategy.py#L88)。

TPE-style search over a categorical treatment grammar.

基类：`SearchStrategy`。

### CategoricalSearchStrategy.__init__

[实际实现](../factor_optimizer/search/categorical_strategy.py#L102)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, recipes: Sequence[CategoricalRecipe], n_initial: int=5, gamma: float=0.25, seed: Optional[int]=None, objective_spec: Optional[ObjectiveSpec]=None, context: Optional[StrategyContext]=None)`。

返回类型：`None`。

### CategoricalSearchStrategy.propose_recipe

[实际实现](../factor_optimizer/search/categorical_strategy.py#L204)。

Return the next candidate as a plain dict (no Trial wrapper).

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### CategoricalSearchStrategy.propose

[实际实现](../factor_optimizer/search/categorical_strategy.py#L213)。

Suggest a new trial with categorical params in ``metadata["params"]``.

参数：`(self, trial_id: Optional[str]=None)`。

返回类型：`Trial`。

### CategoricalSearchStrategy.record

[实际实现](../factor_optimizer/search/categorical_strategy.py#L230)。

Record a completed evaluation (the strategy learns from history).

参数：`(self, params: Dict[str, Any], score: float)`。

返回类型：`None`。

### CategoricalSearchStrategy.to_checkpoint_dict

[实际实现](../factor_optimizer/search/categorical_strategy.py#L264)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### CategoricalSearchStrategy.from_checkpoint_dict

[实际实现](../factor_optimizer/search/categorical_strategy.py#L268)。

Deserialize a standalone categorical strategy checkpoint.

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'CategoricalSearchStrategy'`。

## factor_optimizer/search/conditional_search.py

Hierarchical conditional search over (repair-family, parameters) (R61-FI-035).

### ConditionalParameter

[实际实现](../factor_optimizer/search/conditional_search.py#L73)。

One parameter of a repair family with a bounded conditional domain.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `name` | `str` | `必填/未声明默认` |
| `kind` | `str` | `必填/未声明默认` |
| `choices` | `Tuple[Any, ...]` | `()` |
| `low` | `Optional[float]` | `None` |
| `high` | `Optional[float]` | `None` |
| `prior` | `Optional[Any]` | `None` |

### ConditionalParameter.sample

[实际实现](../factor_optimizer/search/conditional_search.py#L143)。

Draw one value from this parameter's conditional domain.

参数：`(self, rng: random.Random)`。

返回类型：`Any`。

### ConditionalParameter.to_dict

[实际实现](../factor_optimizer/search/conditional_search.py#L151)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### ConditionalParameter.from_dict

[实际实现](../factor_optimizer/search/conditional_search.py#L162)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'ConditionalParameter'`。

### ConditionalFamily

[实际实现](../factor_optimizer/search/conditional_search.py#L167)。

A repair family's conditional parameter tree (plan §24).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `family` | `str` | `必填/未声明默认` |
| `parameters` | `Tuple[ConditionalParameter, ...]` | `()` |

### ConditionalFamily.parameter_names

[实际实现](../factor_optimizer/search/conditional_search.py#L195)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`tuple[str, ...]`。

### ConditionalFamily.validate_params

[实际实现](../factor_optimizer/search/conditional_search.py#L198)。

Fail closed when *params* does not exactly match this family's tree.

参数：`(self, params: Mapping[str, Any])`。

返回类型：`None`。

### ConditionalFamily.to_dict

[实际实现](../factor_optimizer/search/conditional_search.py#L245)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### ConditionalFamily.from_dict

[实际实现](../factor_optimizer/search/conditional_search.py#L249)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'ConditionalFamily'`。

### build_conditional_tree

[实际实现](../factor_optimizer/search/conditional_search.py#L292)。

Compile the family declarations into the conditional search tree.

参数：`(families: Sequence[RepairFamily \| str], registry: Optional[RepairFamilyRegistry]=None)`。

返回类型：`Tuple[ConditionalFamily, ...]`。

### ConditionalPriorProvider

[实际实现](../factor_optimizer/search/conditional_search.py#L318)。

Injected prior source for conditional parameter values (plan §24).

基类：`Protocol`。

### ConditionalPriorProvider.prior_for

[实际实现](../factor_optimizer/search/conditional_search.py#L328)。

Return a JSON-safe prior value for *parameter* of *family* (or None).

参数：`(self, family: str, parameter: str)`。

返回类型：`Optional[Any]`。

### StaticPriorProvider

[实际实现](../factor_optimizer/search/conditional_search.py#L334)。

Deterministic stub prior source (plan §24, defaults).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `registry` | `RepairFamilyRegistry` | `field(default_factory=RepairFamilyRegistry.default)` |
| `overrides` | `Mapping[str, Any]` | `field(default_factory=dict)` |

### StaticPriorProvider.prior_for

[实际实现](../factor_optimizer/search/conditional_search.py#L349)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, family: str, parameter: str)`。

返回类型：`Optional[Any]`。

### ConditionalRecord

[实际实现](../factor_optimizer/search/conditional_search.py#L403)。

A recorded conditional observation (family + params + utility).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `family` | `str` | `必填/未声明默认` |
| `params` | `Tuple[Tuple[str, Any], ...]` | `必填/未声明默认` |
| `utility` | `float` | `必填/未声明默认` |

### ConditionalRecord.to_dict

[实际实现](../factor_optimizer/search/conditional_search.py#L416)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### ConditionalRecord.from_dict

[实际实现](../factor_optimizer/search/conditional_search.py#L420)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'ConditionalRecord'`。

### HierarchicalConditionalSearch

[实际实现](../factor_optimizer/search/conditional_search.py#L428)。

TPE-style hierarchical conditional search (plan §24).

基类：`SearchStrategy`。

### HierarchicalConditionalSearch.__init__

[实际实现](../factor_optimizer/search/conditional_search.py#L456)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, tree: Sequence[ConditionalFamily], n_initial: int=3, gamma: float=0.25, n_candidates: int=24, seed: Optional[int]=None, prior_provider: Optional[ConditionalPriorProvider]=None, objective_spec: Optional[ObjectiveSpec]=None, context: Optional[StrategyContext]=None)`。

返回类型：`None`。

### HierarchicalConditionalSearch.propose_recipe

[实际实现](../factor_optimizer/search/conditional_search.py#L642)。

Return the next conditional candidate as a plain dict.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### HierarchicalConditionalSearch.propose

[实际实现](../factor_optimizer/search/conditional_search.py#L648)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, trial_id: Optional[str]=None)`。

返回类型：`Trial`。

### HierarchicalConditionalSearch.record

[实际实现](../factor_optimizer/search/conditional_search.py#L660)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, params: Dict[str, Any], score: float)`。

返回类型：`None`。

### HierarchicalConditionalSearch.to_checkpoint_dict

[实际实现](../factor_optimizer/search/conditional_search.py#L696)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### HierarchicalConditionalSearch.from_checkpoint_dict

[实际实现](../factor_optimizer/search/conditional_search.py#L700)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'HierarchicalConditionalSearch'`。

## factor_optimizer/search/desirability.py

Soft desirability floors for factor optimization.

### desirability_for

[实际实现](../factor_optimizer/search/desirability.py#L19)。

Return a continuous [0,1] desirability for ``value`` given ``anchors``.

参数：`(direction: str, anchors: Sequence[Anchor], value: float)`。

返回类型：`float`。

### catastrophic_floor

[实际实现](../factor_optimizer/search/desirability.py#L69)。

Hard floor for research-integrity-class unacceptable values.

参数：`(value: float, catastrophic_threshold: float=0.01)`。

返回类型：`float`。

### Desirability

[实际实现](../factor_optimizer/search/desirability.py#L150)。

Small catalog of common metric maps.

### Desirability.__init__

[实际实现](../factor_optimizer/search/desirability.py#L153)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, maps: Dict[str, Dict[str, Sequence[Anchor]]] \| None=None)`。

返回类型：`None`。

### Desirability.score

[实际实现](../factor_optimizer/search/desirability.py#L156)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, metric: str, value: float)`。

返回类型：`float`。

### Desirability.keys

[实际实现](../factor_optimizer/search/desirability.py#L162)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

## factor_optimizer/search/desirability_registry.py

DesirabilityPolicyRegistry (DLIB-FO-003).

显式导出（含重导出）：`DesirabilityContext`、`DesirabilityPolicy`、`DesirabilityPolicyRegistry`、`CONTEXT_KEYS`。

### DesirabilityContext

[实际实现](../factor_optimizer/search/desirability_registry.py#L47)。

The market/asset/frequency context that selects a desirability policy.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `market` | `str` | `必填/未声明默认` |
| `asset_type` | `str` | `必填/未声明默认` |
| `frequency` | `str` | `必填/未声明默认` |
| `factor_family` | `str` | `必填/未声明默认` |
| `label_horizon` | `str` | `必填/未声明默认` |
| `universe_profile` | `str` | `必填/未声明默认` |
| `liquidity_tier` | `str` | `必填/未声明默认` |
| `consumer_profile` | `str` | `必填/未声明默认` |

### DesirabilityContext.to_dict

[实际实现](../factor_optimizer/search/desirability_registry.py#L72)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, str]`。

### DesirabilityContext.from_dict

[实际实现](../factor_optimizer/search/desirability_registry.py#L76)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Dict[str, str])`。

返回类型：`'DesirabilityContext'`。

### DesirabilityPolicy

[实际实现](../factor_optimizer/search/desirability_registry.py#L88)。

A frozen set of metric->anchor maps for a specific context.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `policy_id` | `str` | `必填/未声明默认` |
| `context` | `DesirabilityContext` | `必填/未声明默认` |
| `maps` | `Dict[str, Dict[str, Sequence[Anchor]]]` | `必填/未声明默认` |
| `source` | `str` | `'bootstrap_default'` |

### DesirabilityPolicy.score

[实际实现](../factor_optimizer/search/desirability_registry.py#L143)。

Desirability of ``value`` for ``metric`` under this policy.

参数：`(self, metric: str, value: float)`。

返回类型：`float`。

### DesirabilityPolicy.to_dict

[实际实现](../factor_optimizer/search/desirability_registry.py#L153)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, object]`。

### DesirabilityPolicy.from_dict

[实际实现](../factor_optimizer/search/desirability_registry.py#L168)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Dict[str, object])`。

返回类型：`'DesirabilityPolicy'`。

### DesirabilityPolicyRegistry

[实际实现](../factor_optimizer/search/desirability_registry.py#L205)。

Registry of desirability policies keyed by DesirabilityContext.

### DesirabilityPolicyRegistry.__init__

[实际实现](../factor_optimizer/search/desirability_registry.py#L214)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`None`。

### DesirabilityPolicyRegistry.register

[实际实现](../factor_optimizer/search/desirability_registry.py#L218)。

Register (or return) a policy for ``context``.

参数：`(self, context: DesirabilityContext, maps: Optional[Dict[str, Dict[str, Sequence[Anchor]]]]=None, *, source: str='bootstrap_default', policy_id: Optional[str]=None)`。

返回类型：`DesirabilityPolicy`。

### DesirabilityPolicyRegistry.get

[实际实现](../factor_optimizer/search/desirability_registry.py#L250)。

Return the frozen policy with ``policy_id`` (fail-closed).

参数：`(self, policy_id: str)`。

返回类型：`DesirabilityPolicy`。

### DesirabilityPolicyRegistry.get_for_context

[实际实现](../factor_optimizer/search/desirability_registry.py#L256)。

Return the policy for ``context``, or raise if none registered.

参数：`(self, context: DesirabilityContext)`。

返回类型：`DesirabilityPolicy`。

### DesirabilityPolicyRegistry.freeze_session

[实际实现](../factor_optimizer/search/desirability_registry.py#L267)。

Freeze a desirability_policy_id for a search session.

参数：`(self, context: DesirabilityContext)`。

返回类型：`str`。

### DesirabilityPolicyRegistry.suggest_calibration

[实际实现](../factor_optimizer/search/desirability_registry.py#L276)。

Suggest a calibration source for ``context``.

参数：`(self, context: DesirabilityContext, *, historical_admitted: Optional[Sequence[Dict[str, float]]]=None, raw_distribution: Optional[Dict[str, Sequence[float]]]=None, trial_distribution: Optional[Dict[str, Sequence[float]]]=None)`。

返回类型：`Dict[str, str]`。

## factor_optimizer/search/diagnosis_routing.py

Deterministic V5 diagnosis routing with bounded, non-Cartesian trials.

显式导出（含重导出）：`RecipeKind`、`RoutedTrial`、`RoutingPolicy`、`route_diagnoses`、`FrozenPortfolioRecipe`、`compile_portfolio_recipe`、`execute_routed_portfolio_trial`。

### RecipeKind

[实际实现](../factor_optimizer/search/diagnosis_routing.py#L12)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `FACTOR` | `类常量/枚举` | `'FACTOR_RECIPE'` |
| `PORTFOLIO` | `类常量/枚举` | `'PORTFOLIO_RECIPE'` |

### FrozenPortfolioRecipe

[实际实现](../factor_optimizer/search/diagnosis_routing.py#L18)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `treatment` | `str` | `必填/未声明默认` |
| `parameters` | `Tuple[Tuple[str, float], ...]` | `必填/未声明默认` |
| `recipe_version` | `str` | `'portfolio-routing.v1'` |

### FrozenPortfolioRecipe.recipe_ref

[实际实现](../factor_optimizer/search/diagnosis_routing.py#L42)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

### FrozenPortfolioRecipe.execute

[实际实现](../factor_optimizer/search/diagnosis_routing.py#L47)。

Compile the frozen recipe into stateful target inclusion weights.

参数：`(self, signal_ranks: Sequence[Sequence[float]])`。

返回类型：`np.ndarray`。

### compile_portfolio_recipe

[实际实现](../factor_optimizer/search/diagnosis_routing.py#L68)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(treatment: str)`。

返回类型：`FrozenPortfolioRecipe`。

### execute_routed_portfolio_trial

[实际实现](../factor_optimizer/search/diagnosis_routing.py#L77)。

Resolve the frozen ref, execute weights, and pass them to execution/QE.

参数：`(trial, signal_ranks, consumer, fidelity)`。

### RoutedTrial

[实际实现](../factor_optimizer/search/diagnosis_routing.py#L90)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `trial_id` | `str` | `必填/未声明默认` |
| `diagnosis` | `str` | `必填/未声明默认` |
| `treatment` | `str` | `必填/未声明默认` |
| `recipe_kind` | `RecipeKind` | `必填/未声明默认` |
| `donor_family` | `str` | `必填/未声明默认` |
| `is_raw_baseline` | `bool` | `False` |
| `routing_identity` | `str` | `''` |

### RoutedTrial.to_trial

[实际实现](../factor_optimizer/search/diagnosis_routing.py#L99)。

Compile the route into the canonical trial consumed by SearchRunner.

参数：`(self)`。

### RoutingPolicy

[实际实现](../factor_optimizer/search/diagnosis_routing.py#L122)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `max_new_candidates_initial` | `int` | `8` |
| `max_new_candidates_explicit` | `int` | `12` |
| `max_per_donor_family` | `int` | `4` |
| `policy_version` | `str` | `'V5.0'` |

### route_diagnoses

[实际实现](../factor_optimizer/search/diagnosis_routing.py#L158)。

Return raw plus a deterministic union of single-treatment trials.

参数：`(parent_factor_id: str, diagnoses: Sequence[str], *, policy: RoutingPolicy=RoutingPolicy(), explicit_budget: int \| None=None)`。

返回类型：`Tuple[RoutedTrial, ...]`。

## factor_optimizer/search/dimensions.py

Dimension aggregation primitives for factor optimization.

### DimensionName

[实际实现](../factor_optimizer/search/dimensions.py#L17)。

The balanced-quality dimensions a candidate factor is scored on.

基类：`enum.Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `PREDICTIVE` | `类常量/枚举` | `'predictive'` |
| `STABILITY` | `类常量/枚举` | `'stability'` |
| `ROBUSTNESS` | `类常量/枚举` | `'robustness'` |
| `TRADABILITY` | `类常量/枚举` | `'tradability'` |
| `PURITY_EXPOSURE` | `类常量/枚举` | `'purity_exposure'` |
| `DATA_QUALITY` | `类常量/枚举` | `'data_quality'` |
| `COMPLEXITY` | `类常量/枚举` | `'complexity'` |

### DimensionScores

[实际实现](../factor_optimizer/search/dimensions.py#L43)。

Per-dimension desirability plus the raw metrics that produced it.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `dimension` | `DimensionName` | `必填/未声明默认` |
| `desirability` | `float` | `必填/未声明默认` |
| `raw_metrics` | `Dict[str, float]` | `field(default_factory=dict)` |

### bottleneck_min

[实际实现](../factor_optimizer/search/dimensions.py#L51)。

Lowest dimension desirability: the strictest bottleneck.

参数：`(desirabilities: List[float])`。

返回类型：`float`。

### geomean_floor

[实际实现](../factor_optimizer/search/dimensions.py#L58)。

Geometric mean with a min penalty.

参数：`(desirabilities: List[float], floor: float=1e-06, min_penalty: float=0.5)`。

返回类型：`float`。

### aggregate_dimension

[实际实现](../factor_optimizer/search/dimensions.py#L78)。

Aggregate per-dimension desirabilities into a single score.

参数：`(desirabilities: List[float], *, method: str='bottleneck_geomean', low_quantile: float=0.2, min_penalty: float=0.5)`。

返回类型：`float`。

## factor_optimizer/search/lineage.py

Mutation lineage tracking for provenance and genealogy analysis.

### LineageNode

[实际实现](../factor_optimizer/search/lineage.py#L14)。

A node in the mutation lineage tree.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `trial_id` | `str` | `必填/未声明默认` |
| `parent_ids` | `List[str]` | `field(default_factory=list)` |
| `mutation_type` | `Optional[str]` | `None` |
| `generation` | `int` | `0` |
| `score` | `Optional[float]` | `None` |
| `created_at` | `datetime` | `field(default_factory=_utcnow)` |
| `metadata` | `Dict[str, Any]` | `field(default_factory=dict)` |

### LineageNode.is_seed

[实际实现](../factor_optimizer/search/lineage.py#L35)。

Check if this is a seed node (no parents).

参数：`(self)`。

返回类型：`bool`。

### LineageNode.to_dict

[实际实现](../factor_optimizer/search/lineage.py#L39)。

Serialize to dictionary.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### LineageNode.from_dict

[实际实现](../factor_optimizer/search/lineage.py#L52)。

Deserialize from dictionary.

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'LineageNode'`。

### LineageTree

[实际实现](../factor_optimizer/search/lineage.py#L60)。

Tracks mutation lineage as a directed acyclic graph (DAG).

### LineageTree.__init__

[实际实现](../factor_optimizer/search/lineage.py#L67)。

Initialize empty lineage tree.

参数：`(self)`。

### LineageTree.add_node

[实际实现](../factor_optimizer/search/lineage.py#L72)。

Add a node to the lineage tree.

参数：`(self, node: LineageNode)`。

返回类型：`None`。

### LineageTree.get_node

[实际实现](../factor_optimizer/search/lineage.py#L95)。

Retrieve a node by trial ID.

参数：`(self, trial_id: str)`。

返回类型：`Optional[LineageNode]`。

### LineageTree.get_children

[实际实现](../factor_optimizer/search/lineage.py#L99)。

Get all direct children of a node.

参数：`(self, trial_id: str)`。

返回类型：`List[LineageNode]`。

### LineageTree.get_parents

[实际实现](../factor_optimizer/search/lineage.py#L104)。

Get all direct parents of a node.

参数：`(self, trial_id: str)`。

返回类型：`List[LineageNode]`。

### LineageTree.get_ancestors

[实际实现](../factor_optimizer/search/lineage.py#L111)。

Get all ancestors of a node (recursive parents).

参数：`(self, trial_id: str)`。

返回类型：`Set[str]`。

### LineageTree.get_descendants

[实际实现](../factor_optimizer/search/lineage.py#L140)。

Get all descendants of a node (recursive children).

参数：`(self, trial_id: str)`。

返回类型：`Set[str]`。

### LineageTree.get_seeds

[实际实现](../factor_optimizer/search/lineage.py#L164)。

Get all seed nodes (no parents).

参数：`(self)`。

返回类型：`List[LineageNode]`。

### LineageTree.get_leaves

[实际实现](../factor_optimizer/search/lineage.py#L168)。

Get all leaf nodes (no children).

参数：`(self)`。

返回类型：`List[LineageNode]`。

### LineageTree.depth

[实际实现](../factor_optimizer/search/lineage.py#L175)。

Get depth of a node (longest path from any seed).

参数：`(self, trial_id: str)`。

返回类型：`int`。

### LineageTree.path_to_seed

[实际实现](../factor_optimizer/search/lineage.py#L199)。

Get path from node to its earliest seed ancestor.

参数：`(self, trial_id: str)`。

返回类型：`List[str]`。

### LineageTree.best_in_lineage

[实际实现](../factor_optimizer/search/lineage.py#L223)。

Find best scoring node in the lineage of a trial.

参数：`(self, trial_id: str)`。

返回类型：`Optional[LineageNode]`。

### LineageTree.subtree_stats

[实际实现](../factor_optimizer/search/lineage.py#L251)。

Compute statistics for the subtree rooted at a node.

参数：`(self, trial_id: str)`。

返回类型：`Dict[str, Any]`。

### LineageTree.prune_lineage

[实际实现](../factor_optimizer/search/lineage.py#L279)。

Remove a node and optionally its descendants from the tree.

参数：`(self, trial_id: str, keep_ancestors: bool=True)`。

返回类型：`int`。

### LineageTree.to_dict

[实际实现](../factor_optimizer/search/lineage.py#L312)。

Serialize tree to dictionary.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### LineageTree.from_dict

[实际实现](../factor_optimizer/search/lineage.py#L322)。

Deserialize tree from dictionary.

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'LineageTree'`。

### LineageAnalyzer

[实际实现](../factor_optimizer/search/lineage.py#L369)。

Analyzes mutation lineage for insights.

### LineageAnalyzer.__init__

[实际实现](../factor_optimizer/search/lineage.py#L376)。

Initialize analyzer.

参数：`(self, tree: LineageTree)`。

### LineageAnalyzer.success_rate_by_mutation_type

[实际实现](../factor_optimizer/search/lineage.py#L385)。

Compute success rate for each mutation type.

参数：`(self)`。

返回类型：`Dict[str, float]`。

### LineageAnalyzer.generation_statistics

[实际实现](../factor_optimizer/search/lineage.py#L421)。

Compute statistics for each generation.

参数：`(self)`。

返回类型：`Dict[int, Dict[str, Any]]`。

### LineageAnalyzer.most_productive_lineage

[实际实现](../factor_optimizer/search/lineage.py#L456)。

Find seed with the most successful descendants.

参数：`(self)`。

返回类型：`Optional[str]`。

### LineageAnalyzer.diversity_score

[实际实现](../factor_optimizer/search/lineage.py#L486)。

Compute lineage diversity (how spread out across seeds).

参数：`(self)`。

返回类型：`float`。

## factor_optimizer/search/multifidelity.py

Multi-fidelity evaluation: Stage0-5 profile-bound tiers (R61-FI-036).

### FidelityTier

[实际实现](../factor_optimizer/search/multifidelity.py#L38)。

Evaluation fidelity tiers from cheapest to most expensive.

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `L0` | `类常量/枚举` | `0` |
| `L1` | `类常量/枚举` | `1` |
| `L2` | `类常量/枚举` | `2` |
| `L3` | `类常量/枚举` | `3` |
| `L4` | `类常量/枚举` | `4` |

### EvaluationStage

[实际实现](../factor_optimizer/search/multifidelity.py#L60)。

Canonical multi-fidelity stages (plan §31, R61-FI-036).

基类：`Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `STAGE0` | `类常量/枚举` | `0` |
| `STAGE1` | `类常量/枚举` | `1` |
| `STAGE2` | `类常量/枚举` | `2` |
| `STAGE3` | `类常量/枚举` | `3` |
| `STAGE4` | `类常量/枚举` | `4` |
| `STAGE5` | `类常量/枚举` | `5` |

### EvaluationStage.fidelity_tier

[实际实现](../factor_optimizer/search/multifidelity.py#L76)。

Map a stage onto the legacy L0-L4 fidelity integer contract.

参数：`(self)`。

返回类型：`FidelityTier`。

### StageEvaluationProfile

[实际实现](../factor_optimizer/search/multifidelity.py#L109)。

A multi-fidelity stage bound to a named QE EvidenceProfile (plan §31).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `stage` | `EvaluationStage` | `必填/未声明默认` |
| `evidence_profile` | `str` | `''` |
| `policy_id` | `str` | `'FO_EVALUATION_STAGE'` |
| `policy_version` | `str` | `'1.0.0'` |

### StageProfileRegistry

[实际实现](../factor_optimizer/search/multifidelity.py#L188)。

Versioned registry mapping each EvaluationStage to its evidence profile.

### StageProfileRegistry.__init__

[实际实现](../factor_optimizer/search/multifidelity.py#L196)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, profiles: Sequence[StageEvaluationProfile], policy_id: str='FO_EVALUATION_STAGE', policy_version: str='1.0.0')`。

返回类型：`None`。

### StageProfileRegistry.default

[实际实现](../factor_optimizer/search/multifidelity.py#L224)。

The canonical Stage0-5 registry (plan §31).

参数：`(cls)`。

返回类型：`'StageProfileRegistry'`。

### StageProfileRegistry.policy_id

[实际实现](../factor_optimizer/search/multifidelity.py#L229)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### StageProfileRegistry.policy_version

[实际实现](../factor_optimizer/search/multifidelity.py#L233)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### StageProfileRegistry.profile_for

[实际实现](../factor_optimizer/search/multifidelity.py#L236)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, stage: EvaluationStage \| int)`。

返回类型：`StageEvaluationProfile`。

### StageProfileRegistry.stages

[实际实现](../factor_optimizer/search/multifidelity.py#L250)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Tuple[EvaluationStage, ...]`。

### StageProfileRegistry.evidence_profiles

[实际实现](../factor_optimizer/search/multifidelity.py#L253)。

Stage name → evidence-profile id (string reference) mapping.

参数：`(self)`。

返回类型：`Dict[str, str]`。

### StageProfileRegistry.to_dict

[实际实现](../factor_optimizer/search/multifidelity.py#L262)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### Stage2DiagnosisBudget

[实际实现](../factor_optimizer/search/multifidelity.py#L284)。

The diagnosis-directed candidate plan Stage 2 executes (plan §31).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `stage_profile` | `StageEvaluationProfile` | `必填/未声明默认` |
| `plan` | `DiagnosisSearchBudget` | `必填/未声明默认` |
| `budget_config` | `RepairBudgetConfig` | `field(default_factory=RepairBudgetConfig)` |

### Stage2DiagnosisBudget.candidate_ceiling

[实际实现](../factor_optimizer/search/multifidelity.py#L312)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`int`。

### Stage2DiagnosisBudget.to_dict

[实际实现](../factor_optimizer/search/multifidelity.py#L315)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### build_stage2_budget

[实际实现](../factor_optimizer/search/multifidelity.py#L323)。

Build the Stage-2 diagnosis-directed budget for a factor (plan §31).

参数：`(diagnoses: Sequence[DiagnosisKind \| str], *, evidence: Optional[Mapping[str, object]]=None, budget_config: Optional[RepairBudgetConfig]=None, stage_registry: Optional[StageProfileRegistry]=None)`。

返回类型：`Stage2DiagnosisBudget`。

### StageCostTelemetry

[实际实现](../factor_optimizer/search/multifidelity.py#L356)。

Mutable per-stage cost accumulator (plan §31 cost telemetry).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `stage` | `EvaluationStage` | `必填/未声明默认` |
| `evidence_profile` | `str` | `''` |
| `wall_time_s` | `float` | `0.0` |
| `cpu_time_s` | `float` | `0.0` |
| `gpu_time_s` | `float` | `0.0` |
| `h2d_d2h_bytes` | `int` | `0` |
| `peak_vram_bytes` | `int` | `0` |
| `bytes_read` | `int` | `0` |
| `candidate_count` | `int` | `0` |
| `metric_count` | `int` | `0` |
| `factor_value_cache_hits` | `int` | `0` |
| `policy_id` | `str` | `'FO_EVALUATION_STAGE'` |
| `policy_version` | `str` | `'1.0.0'` |

### StageCostTelemetry.record_evaluation

[实际实现](../factor_optimizer/search/multifidelity.py#L426)。

Accumulate one evaluation's costs into this stage telemetry.

参数：`(self, *, candidate_count: int=1, metric_count: int=0, wall_time_s: float=0.0, cpu_time_s: float=0.0, gpu_time_s: float=0.0, h2d_d2h_bytes: int=0, peak_vram_bytes: int=0, bytes_read: int=0, factor_value_cache_hits: int=0)`。

返回类型：`None`。

### StageCostTelemetry.to_dict

[实际实现](../factor_optimizer/search/multifidelity.py#L457)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### StageCostTelemetry.from_dict

[实际实现](../factor_optimizer/search/multifidelity.py#L475)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Mapping[str, Any])`。

返回类型：`'StageCostTelemetry'`。

### MultiStageCostTelemetry

[实际实现](../factor_optimizer/search/multifidelity.py#L482)。

Accumulates one :class:`StageCostTelemetry` per stage (plan §31).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `stage_profiles` | `StageProfileRegistry` | `field(default_factory=StageProfileRegistry.default)` |

### MultiStageCostTelemetry.telemetry_for

[实际实现](../factor_optimizer/search/multifidelity.py#L497)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, stage: EvaluationStage)`。

返回类型：`StageCostTelemetry`。

### MultiStageCostTelemetry.record_stage

[实际实现](../factor_optimizer/search/multifidelity.py#L508)。

Record one evaluation's costs into *stage*'s accumulator.

参数：`(self, stage: EvaluationStage, *, candidate_count: int=1, metric_count: int=0, wall_time_s: float=0.0, cpu_time_s: float=0.0, gpu_time_s: float=0.0, h2d_d2h_bytes: int=0, peak_vram_bytes: int=0, bytes_read: int=0, factor_value_cache_hits: int=0)`。

返回类型：`None`。

### MultiStageCostTelemetry.report

[实际实现](../factor_optimizer/search/multifidelity.py#L535)。

Structured telemetry report (stage -> StageCostTelemetry.to_dict()).

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### MultiStageCostTelemetry.to_dict

[实际实现](../factor_optimizer/search/multifidelity.py#L542)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### FidelitySpec

[实际实现](../factor_optimizer/search/multifidelity.py#L550)。

Specification for evaluation at a specific fidelity tier.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `tier` | `FidelityTier` | `必填/未声明默认` |
| `sample_fraction` | `float` | `必填/未声明默认` |
| `num_metrics` | `int` | `必填/未声明默认` |
| `cross_validate` | `bool` | `False` |
| `compute_robustness` | `bool` | `False` |
| `cost_multiplier` | `float` | `1.0` |
| `typical_duration_ms` | `int` | `100` |

### PromotionCriteria

[实际实现](../factor_optimizer/search/multifidelity.py#L631)。

Criteria for promoting a candidate to higher fidelity.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `min_score` | `Optional[float]` | `None` |
| `top_k_fraction` | `Optional[float]` | `None` |
| `min_improvement` | `Optional[float]` | `None` |
| `require_all` | `bool` | `True` |

### PromotionCriteria.should_promote

[实际实现](../factor_optimizer/search/multifidelity.py#L646)。

Check if candidate meets promotion criteria.

参数：`(self, score: float, rank: int, total: int, baseline_score: Optional[float]=None)`。

返回类型：`bool`。

### MultiFidelityScheduler

[实际实现](../factor_optimizer/search/multifidelity.py#L686)。

Schedules evaluation across fidelity tiers.

### MultiFidelityScheduler.__init__

[实际实现](../factor_optimizer/search/multifidelity.py#L693)。

Initialize scheduler.

参数：`(self, tier_specs: Optional[Dict[FidelityTier, FidelitySpec]]=None, promotion_criteria: Optional[Dict[FidelityTier, PromotionCriteria]]=None)`。

### MultiFidelityScheduler.get_spec

[实际实现](../factor_optimizer/search/multifidelity.py#L717)。

Get specification for a fidelity tier.

参数：`(self, tier: FidelityTier)`。

返回类型：`FidelitySpec`。

### MultiFidelityScheduler.next_tier

[实际实现](../factor_optimizer/search/multifidelity.py#L721)。

Get next higher fidelity tier.

参数：`(self, current: FidelityTier)`。

返回类型：`Optional[FidelityTier]`。

### MultiFidelityScheduler.should_promote

[实际实现](../factor_optimizer/search/multifidelity.py#L727)。

Check if candidate should be promoted to next tier.

参数：`(self, current_tier: FidelityTier, score: float, rank: int, total: int, baseline_score: Optional[float]=None)`。

返回类型：`bool`。

### MultiFidelityScheduler.estimate_cost

[实际实现](../factor_optimizer/search/multifidelity.py#L754)。

Estimate cost for evaluations at given tier.

参数：`(self, tier: FidelityTier, num_evaluations: int=1)`。

返回类型：`float`。

### MultiFidelityScheduler.estimate_duration_ms

[实际实现](../factor_optimizer/search/multifidelity.py#L759)。

Estimate duration for evaluations at given tier.

参数：`(self, tier: FidelityTier, num_evaluations: int=1)`。

返回类型：`int`。

### MultiFidelityScheduler.optimal_tier_for_budget

[实际实现](../factor_optimizer/search/multifidelity.py#L764)。

Determine highest tier affordable with remaining budget.

参数：`(self, remaining_budget: float)`。

返回类型：`FidelityTier`。

## factor_optimizer/search/paired_comparison.py

V5 paired common-draw factor comparison.

显式导出（含重导出）：`ComparisonStatus`、`PairedDraws`、`ComparisonThresholds`、`PairedComparisonResult`、`compare_paired_draws`。

### ComparisonStatus

[实际实现](../factor_optimizer/search/paired_comparison.py#L15)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

基类：`str, Enum`。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `SUPERIOR` | `类常量/枚举` | `'SUPERIOR'` |
| `NON_INFERIOR_CHEAPER` | `类常量/枚举` | `'NON_INFERIOR_CHEAPER'` |
| `EQUIVALENT` | `类常量/枚举` | `'EQUIVALENT'` |
| `TRADE_OFF` | `类常量/枚举` | `'TRADE_OFF'` |
| `INCONCLUSIVE` | `类常量/枚举` | `'INCONCLUSIVE'` |
| `INVALID_CONTEXT` | `类常量/枚举` | `'INVALID_CONTEXT'` |

### PairedDraws

[实际实现](../factor_optimizer/search/paired_comparison.py#L25)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `candidate_id` | `str` | `必填/未声明默认` |
| `baseline_id` | `str` | `必填/未声明默认` |
| `draw_ids` | `Tuple[str, ...]` | `必填/未声明默认` |
| `candidate_metrics` | `Mapping[str, Sequence[float]]` | `必填/未声明默认` |
| `baseline_metrics` | `Mapping[str, Sequence[float]]` | `必填/未声明默认` |
| `context_identity` | `str` | `必填/未声明默认` |
| `candidate_cost` | `Sequence[float]` | `()` |
| `baseline_cost` | `Sequence[float]` | `()` |

### ComparisonThresholds

[实际实现](../factor_optimizer/search/paired_comparison.py#L37)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `minimum_improvement` | `float` | `必填/未声明默认` |
| `maximum_noninferiority_loss` | `float` | `必填/未声明默认` |
| `equivalence_bound` | `float` | `必填/未声明默认` |
| `minimum_cost_improvement` | `float` | `0.0` |
| `confidence_level` | `float` | `0.95` |

### PairedComparisonResult

[实际实现](../factor_optimizer/search/paired_comparison.py#L55)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `status` | `ComparisonStatus` | `必填/未声明默认` |
| `difference_interval` | `Tuple[float, float] \| None` | `必填/未声明默认` |
| `mean_difference` | `float \| None` | `必填/未声明默认` |
| `wins` | `int` | `必填/未声明默认` |
| `losses` | `int` | `必填/未声明默认` |
| `ties` | `int` | `必填/未声明默认` |
| `draw_count` | `int` | `必填/未声明默认` |
| `reason` | `str` | `必填/未声明默认` |

### compare_paired_draws

[实际实现](../factor_optimizer/search/paired_comparison.py#L75)。

Classify B(candidate)-A(baseline) on identical resampling draws.

参数：`(evidence: PairedDraws, thresholds: ComparisonThresholds, *, utility: Callable[[Mapping[str, float]], float], expected_context_identity: str)`。

返回类型：`PairedComparisonResult`。

## factor_optimizer/search/pareto.py

Pareto frontier tracking for multi-objective optimization.

### ParetoPoint

[实际实现](../factor_optimizer/search/pareto.py#L10)。

A point in the Pareto frontier.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `trial_id` | `str` | `必填/未声明默认` |
| `objectives` | `Tuple[float, ...]` | `必填/未声明默认` |
| `metadata` | `Dict[str, Any]` | `field(default_factory=dict)` |

### ParetoPoint.dominates

[实际实现](../factor_optimizer/search/pareto.py#L35)。

Check if this point dominates another.

参数：`(self, other: 'ParetoPoint')`。

返回类型：`bool`。

### ParetoPoint.distance_to

[实际实现](../factor_optimizer/search/pareto.py#L53)。

Euclidean distance to another point.

参数：`(self, other: 'ParetoPoint')`。

返回类型：`float`。

### ParetoFrontier

[实际实现](../factor_optimizer/search/pareto.py#L63)。

Tracks the Pareto frontier for multi-objective optimization.

### ParetoFrontier.__init__

[实际实现](../factor_optimizer/search/pareto.py#L71)。

Initialize Pareto frontier.

参数：`(self, objective_names: Optional[List[str]]=None)`。

### ParetoFrontier.num_dimensions

[实际实现](../factor_optimizer/search/pareto.py#L82)。

Number of objectives.

参数：`(self)`。

返回类型：`int`。

### ParetoFrontier.size

[实际实现](../factor_optimizer/search/pareto.py#L89)。

Number of points on the frontier.

参数：`(self)`。

返回类型：`int`。

### ParetoFrontier.add

[实际实现](../factor_optimizer/search/pareto.py#L93)。

Add a point to the frontier if non-dominated.

参数：`(self, point: ParetoPoint)`。

返回类型：`bool`。

### ParetoFrontier.dominated

[实际实现](../factor_optimizer/search/pareto.py#L106)。

Return the set of trial_ids among ``points`` that are dominated by the frontier.

参数：`(self, points)`。

返回类型：`Set[str]`。

### ParetoFrontier.frontier

[实际实现](../factor_optimizer/search/pareto.py#L123)。

Return the set of non-dominated points currently on the frontier.

参数：`(self)`。

返回类型：`List[ParetoPoint]`。

### ParetoFrontier.add_point

[实际实现](../factor_optimizer/search/pareto.py#L132)。

Add a point to the frontier if non-dominated.

参数：`(self, point: ParetoPoint)`。

返回类型：`bool`。

### ParetoFrontier.is_dominated

[实际实现](../factor_optimizer/search/pareto.py#L154)。

Check if a point is dominated by the frontier.

参数：`(self, point: ParetoPoint)`。

返回类型：`bool`。

### ParetoFrontier.get_point

[实际实现](../factor_optimizer/search/pareto.py#L164)。

Retrieve a point by trial ID.

参数：`(self, trial_id: str)`。

返回类型：`Optional[ParetoPoint]`。

### ParetoFrontier.extremes

[实际实现](../factor_optimizer/search/pareto.py#L171)。

Get extreme points for each objective.

参数：`(self)`。

返回类型：`Dict[int, ParetoPoint]`。

### ParetoFrontier.hypervolume

[实际实现](../factor_optimizer/search/pareto.py#L194)。

Compute hypervolume indicator (approximation for 2D).

参数：`(self, reference_point: Tuple[float, ...])`。

返回类型：`float`。

### ParetoFrontier.coverage

[实际实现](../factor_optimizer/search/pareto.py#L244)。

Compute coverage metric: fraction of other's points dominated by this frontier.

参数：`(self, other: 'ParetoFrontier')`。

返回类型：`float`。

### ParetoFrontier.spacing

[实际实现](../factor_optimizer/search/pareto.py#L264)。

Compute spacing metric: uniformity of point distribution.

参数：`(self)`。

返回类型：`float`。

### ParetoFrontier.to_dict

[实际实现](../factor_optimizer/search/pareto.py#L289)。

Serialize frontier to dictionary.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### ParetoFrontier.from_dict

[实际实现](../factor_optimizer/search/pareto.py#L306)。

Deserialize frontier from dictionary.

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'ParetoFrontier'`。

### ParetoArchive

[实际实现](../factor_optimizer/search/pareto.py#L319)。

Archive tracking multiple Pareto frontiers over time.

### ParetoArchive.__init__

[实际实现](../factor_optimizer/search/pareto.py#L326)。

Initialize archive.

参数：`(self, objective_names: Optional[List[str]]=None)`。

### ParetoArchive.snapshot

[实际实现](../factor_optimizer/search/pareto.py#L337)。

Take a snapshot of the current frontier.

参数：`(self, iteration: int)`。

返回类型：`None`。

### ParetoArchive.add_point

[实际实现](../factor_optimizer/search/pareto.py#L351)。

Add a point to the current frontier.

参数：`(self, point: ParetoPoint)`。

返回类型：`bool`。

### ParetoArchive.get_frontier

[实际实现](../factor_optimizer/search/pareto.py#L355)。

Get frontier at a specific iteration.

参数：`(self, iteration: Optional[int]=None)`。

返回类型：`ParetoFrontier`。

### ParetoArchive.frontier_growth

[实际实现](../factor_optimizer/search/pareto.py#L374)。

Track frontier size over iterations.

参数：`(self)`。

返回类型：`List[Tuple[int, int]]`。

### ParetoArchive.hypervolume_progress

[实际实现](../factor_optimizer/search/pareto.py#L385)。

Track hypervolume over iterations.

参数：`(self, reference_point: Tuple[float, ...])`。

返回类型：`List[Tuple[int, float]]`。

## factor_optimizer/search/plateau.py

Plateau detection for search stopping criteria.

### PlateauConfig

[实际实现](../factor_optimizer/search/plateau.py#L9)。

Configuration for plateau detection.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `window_size` | `int` | `20` |
| `min_relative_improvement` | `float` | `0.001` |
| `min_absolute_improvement` | `Optional[float]` | `None` |
| `require_both` | `bool` | `False` |
| `use_median` | `bool` | `False` |
| `direction` | `str` | `'maximize'` |

### PlateauDetector

[实际实现](../factor_optimizer/search/plateau.py#L38)。

Detects when search progress has plateaued.

### PlateauDetector.__init__

[实际实现](../factor_optimizer/search/plateau.py#L46)。

Initialize detector.

参数：`(self, config: PlateauConfig)`。

### PlateauDetector.add_score

[实际实现](../factor_optimizer/search/plateau.py#L56)。

Add a new score to history.

参数：`(self, score: float)`。

返回类型：`None`。

### PlateauDetector.is_plateau

[实际实现](../factor_optimizer/search/plateau.py#L60)。

Check if current window indicates a plateau.

参数：`(self, scores: Optional[List[float]]=None)`。

返回类型：`bool`。

### PlateauDetector.reset

[实际实现](../factor_optimizer/search/plateau.py#L111)。

Clear score history.

参数：`(self)`。

返回类型：`None`。

### PlateauDetector.plateau_duration

[实际实现](../factor_optimizer/search/plateau.py#L115)。

Return number of evaluations since last significant improvement.

参数：`(self)`。

返回类型：`int`。

### AdaptivePlateauDetector

[实际实现](../factor_optimizer/search/plateau.py#L163)。

Adaptive plateau detection with dynamic thresholds.

### AdaptivePlateauDetector.__init__

[实际实现](../factor_optimizer/search/plateau.py#L170)。

Initialize adaptive detector.

参数：`(self, initial_config: PlateauConfig, early_phase_window: int=50, volatility_window: int=10)`。

### AdaptivePlateauDetector.add_score

[实际实现](../factor_optimizer/search/plateau.py#L189)。

Add a new score to history.

参数：`(self, score: float)`。

返回类型：`None`。

### AdaptivePlateauDetector.is_plateau

[实际实现](../factor_optimizer/search/plateau.py#L193)。

Check if current state indicates a plateau.

参数：`(self)`。

返回类型：`bool`。

### AdaptivePlateauDetector.reset

[实际实现](../factor_optimizer/search/plateau.py#L233)。

Clear history.

参数：`(self)`。

返回类型：`None`。

### MultiObjectivePlateauDetector

[实际实现](../factor_optimizer/search/plateau.py#L238)。

Plateau detection for multi-objective optimization.

### MultiObjectivePlateauDetector.__init__

[实际实现](../factor_optimizer/search/plateau.py#L245)。

Initialize multi-objective detector.

参数：`(self, window_size: int=20, min_new_nondominated: int=1)`。

### MultiObjectivePlateauDetector.add_hypervolume

[实际实现](../factor_optimizer/search/plateau.py#L262)。

Record feasible-frontier quality against one fixed reference point.

参数：`(self, hypervolume: float)`。

返回类型：`None`。

### MultiObjectivePlateauDetector.add_frontier_size

[实际实现](../factor_optimizer/search/plateau.py#L269)。

Record current frontier size.

参数：`(self, size: int)`。

返回类型：`None`。

### MultiObjectivePlateauDetector.is_plateau

[实际实现](../factor_optimizer/search/plateau.py#L273)。

Check if frontier has stopped growing.

参数：`(self)`。

返回类型：`bool`。

### MultiObjectivePlateauDetector.growth_rate

[实际实现](../factor_optimizer/search/plateau.py#L290)。

Compute recent frontier growth rate.

参数：`(self)`。

返回类型：`float`。

### MultiObjectivePlateauDetector.reset

[实际实现](../factor_optimizer/search/plateau.py#L310)。

Clear history.

参数：`(self)`。

返回类型：`None`。

## factor_optimizer/search/runner.py

SearchRunner: orchestrate mutation search with budget and plateau stopping.

### SearchConfig

[实际实现](../factor_optimizer/search/runner.py#L69)。

Configuration for search execution.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `budget` | `SearchBudget` | `必填/未声明默认` |
| `plateau_window` | `int` | `20` |
| `plateau_threshold` | `float` | `0.001` |
| `enable_multifidelity` | `bool` | `True` |
| `max_concurrency` | `int` | `1` |
| `evaluation_cost_units` | `Optional[float]` | `None` |
| `execution_mode` | `ExecutionMode` | `ExecutionMode.RESEARCH_ONLY` |
| `require_evaluation_protocol` | `bool` | `True` |
| `objective_direction` | `Optional[ObjectiveDirection]` | `None` |
| `objective_spec` | `Optional[ObjectiveSpec]` | `None` |
| `tiered_evaluation` | `Optional[TieredEvaluationPolicy]` | `None` |
| `candidate_strategy` | `Optional[SearchStrategy]` | `None` |
| `candidate_strategy_spec` | `Optional['SearchStrategySpec']` | `None` |
| `durable_campaign_store_path` | `Optional[str]` | `None` |
| `screening_only` | `bool` | `True` |

### SearchConfig.to_dict

[实际实现](../factor_optimizer/search/runner.py#L223)。

Serialize search configuration.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### SearchConfig.from_dict

[实际实现](../factor_optimizer/search/runner.py#L254)。

Deserialize search configuration.

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'SearchConfig'`。

### SearchSession

[实际实现](../factor_optimizer/search/runner.py#L285)。

Runtime state for an active search session.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `session_id` | `str` | `必填/未声明默认` |
| `config` | `SearchConfig` | `必填/未声明默认` |
| `budget_tracker` | `BudgetTracker` | `必填/未声明默认` |
| `trials` | `List[Trial]` | `field(default_factory=list)` |
| `duplicate_trials` | `List[Trial]` | `field(default_factory=list)` |
| `best_score` | `Optional[float]` | `None` |
| `best_trial_id` | `Optional[str]` | `None` |
| `started_at` | `datetime` | `field(default_factory=_utcnow)` |
| `finished_at` | `Optional[datetime]` | `None` |
| `stop_reason` | `Optional[str]` | `None` |
| `recent_scores` | `List[float]` | `field(default_factory=list)` |
| `frozen_at` | `Optional[datetime]` | `None` |
| `sealed_trial_id` | `Optional[str]` | `None` |
| `sealed_split_id` | `Optional[str]` | `None` |
| `sealed_test_masks` | `Optional[Dict[str, List[bool]]]` | `None` |
| `sealed_test_consumed` | `bool` | `False` |
| `sealed_test_state` | `str` | `'AVAILABLE'` |
| `sealed_test_reservation_ref` | `Optional[str]` | `None` |
| `sealed_execution_spec` | `Optional[SelectedExecutionSpec]` | `None` |
| `sealed_test_results` | `List[Dict[str, Any]]` | `field(default_factory=list)` |
| `strategy` | `Optional[SearchStrategy]` | `None` |
| `ledger` | `TrialLedger` | `field(default_factory=TrialLedger)` |
| `multiplicity_artifact` | `Optional['MultiplicityArtifact']` | `None` |
| `execution_plan_hash` | `Optional[str]` | `None` |
| `stage_execution_trace` | `List[Dict[str, Any]]` | `field(default_factory=list)` |
| `selection_decision_id` | `Optional[str]` | `None` |
| `selection_decision_hash` | `Optional[str]` | `None` |
| `selection_request_hash` | `Optional[str]` | `None` |

### SearchSession.add_trial

[实际实现](../factor_optimizer/search/runner.py#L355)。

Add a trial to the session, rejecting reused trial IDs.

参数：`(self, trial: Trial)`。

返回类型：`None`。

### SearchSession.has_trial

[实际实现](../factor_optimizer/search/runner.py#L362)。

Return whether a trial ID has already been recorded.

参数：`(self, trial_id: str)`。

返回类型：`bool`。

### SearchSession.update_best

[实际实现](../factor_optimizer/search/runner.py#L366)。

Update best score if improved. Returns True if new best.

参数：`(self, trial_id: str, score: float)`。

返回类型：`bool`。

### SearchSession.successful_trials

[实际实现](../factor_optimizer/search/runner.py#L393)。

Return all successfully evaluated trials.

参数：`(self)`。

返回类型：`List[Trial]`。

### SearchSession.finish

[实际实现](../factor_optimizer/search/runner.py#L397)。

Mark session as finished (idempotent) and seal the multiplicity ledger.

参数：`(self, reason: str)`。

返回类型：`None`。

### SearchSession.freeze_for_sealed_test

[实际实现](../factor_optimizer/search/runner.py#L456)。

Freeze the finished winner and issue its one-shot test authority.

参数：`(self, split_plan: SplitPlan)`。

返回类型：`SealedTestHandle`。

### SearchSession.consume_sealed_test

[实际实现](../factor_optimizer/search/runner.py#L518)。

Validate and consume a handle by evaluating only its bound test plan.

参数：`(self, handle: SealedTestHandle, split_plan: SplitPlan, evaluator: Callable[[Trial, SplitPlan], Dict[str, float]])`。

返回类型：`SealedTestResult`。

### SearchSession.is_finished

[实际实现](../factor_optimizer/search/runner.py#L542)。

Check if session is complete.

参数：`(self)`。

返回类型：`bool`。

### SearchSession.duration_seconds

[实际实现](../factor_optimizer/search/runner.py#L546)。

Return session duration in seconds.

参数：`(self)`。

返回类型：`float`。

### SearchSession.checkpoint

[实际实现](../factor_optimizer/search/runner.py#L553)。

Write a full checkpoint (strategy + RNG + budget) to *path*.

参数：`(self, path: str)`。

返回类型：`None`。

### SearchSession.resume

[实际实现](../factor_optimizer/search/runner.py#L589)。

Load and restore a session checkpoint written by ``checkpoint``.

参数：`(cls, path: str)`。

返回类型：`'SearchSession'`。

### SearchSession.to_dict

[实际实现](../factor_optimizer/search/runner.py#L602)。

Serialize session state for a quiescent checkpoint.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### SearchSession.from_dict

[实际实现](../factor_optimizer/search/runner.py#L653)。

Deserialize and validate a session checkpoint.

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'SearchSession'`。

### SearchRunner

[实际实现](../factor_optimizer/search/runner.py#L1162)。

Orchestrates mutation search with budget and stopping criteria.

### SearchRunner.plan_diagnosis_trials

[实际实现](../factor_optimizer/search/runner.py#L1179)。

Public runner entry for the bounded V5 diagnosis router.

参数：`(parent_factor_id: str, diagnoses, *, explicit_budget: Optional[int]=None)`。

### SearchRunner.for_diagnoses

[实际实现](../factor_optimizer/search/runner.py#L1193)。

Build a real runner whose proposals come from the V5 route plan.

参数：`(cls, config: SearchConfig, parent_factor_id: str, diagnoses, evaluation_fn, *, explicit_budget: Optional[int]=None, portfolio_recipe_context=None, **runner_kwargs)`。

### SearchRunner.__init__

[实际实现](../factor_optimizer/search/runner.py#L1235)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, config: SearchConfig, proposal_fn: Callable[[], Trial], evaluation_fn: Callable[[Trial, int], Dict[str, Any]], plateau_detector: Optional[Callable[[List[float]], bool]]=None, trial_validator: Optional[Callable[[Trial], Dict[str, Any]]]=None, strategy: Optional[SearchStrategy]=None, budget_tracker_factory: Optional[Callable[[str, SearchBudget], Any]]=None, decision_provider: Optional[Any]=None, decision_request_factory: Optional[Callable[[SearchSession], Any]]=None, stage_decision_request_factory: Optional[Callable[..., Any]]=None)`。

### SearchRunner.run

[实际实现](../factor_optimizer/search/runner.py#L1410)。

Execute a new search until budget exhausted or plateau reached.

参数：`(self, session_id: str)`。

返回类型：`SearchSession`。

### SearchRunner.resume

[实际实现](../factor_optimizer/search/runner.py#L1434)。

Resume a quiescent, unfinished session with matching configuration.

参数：`(self, session: SearchSession, *, resume_same_budget: bool=False, budget_extension: Optional['BudgetExtensionAuthorization']=None)`。

返回类型：`SearchSession`。

### SearchRunner.create_train_evaluation_context

[实际实现](../factor_optimizer/search/runner.py#L1552)。

Create an isolated evaluation context for training data.

参数：`(self)`。

返回类型：`'TrainEvaluationContext'`。

### SearchRunner.create_validation_evaluation_context

[实际实现](../factor_optimizer/search/runner.py#L1556)。

Create an isolated evaluation context for validation data.

参数：`(self)`。

返回类型：`'ValidationEvaluationContext'`。

### SearchRunner.create_sealed_test_executor

[实际实现](../factor_optimizer/search/runner.py#L1560)。

Create an isolated executor for sealed test data.

参数：`(self)`。

返回类型：`'SealedTestExecutor'`。

### TrainEvaluationContext

[实际实现](../factor_optimizer/search/runner.py#L2105)。

Isolated evaluation context for training data.

### TrainEvaluationContext.__init__

[实际实现](../factor_optimizer/search/runner.py#L2111)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, runner: SearchRunner)`。

### TrainEvaluationContext.evaluate

[实际实现](../factor_optimizer/search/runner.py#L2120)。

Evaluate a trial using only training data.

参数：`(self, trial: Trial, fidelity: int)`。

返回类型：`Dict[str, Any]`。

### ValidationEvaluationContext

[实际实现](../factor_optimizer/search/runner.py#L2147)。

Isolated evaluation context for validation data.

### ValidationEvaluationContext.__init__

[实际实现](../factor_optimizer/search/runner.py#L2153)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, runner: SearchRunner)`。

### ValidationEvaluationContext.evaluate

[实际实现](../factor_optimizer/search/runner.py#L2164)。

Evaluate a trial using only validation data.

参数：`(self, trial: Trial, fidelity: int)`。

返回类型：`Dict[str, Any]`。

### SealedTestExecutor

[实际实现](../factor_optimizer/search/runner.py#L2189)。

Isolated executor for sealed test data.

### SealedTestExecutor.__init__

[实际实现](../factor_optimizer/search/runner.py#L2199)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, runner: SearchRunner, test_authority_broker: Optional[TestAuthorityBroker]=None)`。

### SealedTestExecutor.has_test_authority

[实际实现](../factor_optimizer/search/runner.py#L2211)。

True only when a test authority broker is attached.

参数：`(self)`。

返回类型：`bool`。

### SealedTestExecutor.evaluate_sealed_test

[实际实现](../factor_optimizer/search/runner.py#L2215)。

Evaluate the sealed test using only test data.

参数：`(self, session: SearchSession, handle: SealedTestHandle, split_plan: SplitPlan)`。

返回类型：`SealedTestResult`。

## factor_optimizer/search/statistical_consumption.py

FO consumption of QE statistical evidence bound to durable campaign state.

### SearchStatisticalEvidence

[实际实现](../factor_optimizer/search/statistical_consumption.py#L26)。

Auditable statistical bundle consumed by an FO search decision.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `campaign_id` | `str` | `必填/未声明默认` |
| `hypothesis_summary` | `Mapping[str, int]` | `必填/未声明默认` |
| `dsr` | `DSREvidence` | `必填/未声明默认` |
| `pbo` | `PBOEvidence` | `必填/未声明默认` |
| `retention` | `RetentionEvidence` | `必填/未声明默认` |
| `horizon_curve` | `HorizonCurveEvidence` | `必填/未声明默认` |
| `regime` | `RegimeEvidence` | `必填/未声明默认` |

### build_search_statistical_evidence

[实际实现](../factor_optimizer/search/statistical_consumption.py#L38)。

Build the statistical bundle from a complete durable search family.

参数：`(store: SQLiteCampaignStore, campaign_id: str, candidate_returns: np.ndarray, family_sharpes: Sequence[float], *, selected_candidate_index: int, trial_ledger_ref: str, common_cost_spec_ref: str, returns_frequency: str, annualization_factor: float, train_series: Sequence[float], validation_series: Sequence[float], train_value: float, validation_value: float, horizon_ic_series: Mapping[int, Sequence[float]], horizon_label_refs: Mapping[int, str], regime_values: Sequence[float], regime_labels: Sequence[int], regime_available_times: Sequence[object], regime_kind: str, regime_state_ref: Optional[str]=None, family_sharpe_scale: str='raw_periodic', pbo_splits: int=6, near_zero: float=0.001, bootstrap_block_length: int=10, bootstrap_repetitions: int=1000, bootstrap_seed: int=0, min_periods: int=30)`。

返回类型：`SearchStatisticalEvidence`。

## factor_optimizer/search/strategies.py

Search strategies: Random, Grid, Bayesian (GP), and TPE.

### SearchStrategySpec

[实际实现](../factor_optimizer/search/strategies.py#L24)。

Semantic spec of a search strategy for checkpoint identity.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `strategy_type` | `str` | `必填/未声明默认` |
| `ctor` | `Dict[str, Any]` | `field(default_factory=dict)` |
| `seed` | `Optional[int]` | `None` |

### SearchStrategySpec.from_strategy

[实际实现](../factor_optimizer/search/strategies.py#L39)。

Capture the semantic spec of a strategy instance.

参数：`(cls, strategy: 'SearchStrategy')`。

返回类型：`'SearchStrategySpec'`。

### SearchStrategySpec.to_dict

[实际实现](../factor_optimizer/search/strategies.py#L47)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### SearchStrategySpec.from_dict

[实际实现](../factor_optimizer/search/strategies.py#L55)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'SearchStrategySpec'`。

### SearchStrategyState

[实际实现](../factor_optimizer/search/strategies.py#L71)。

Snapshot of a search strategy for checkpoint/resume.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `strategy_type` | `str` | `必填/未声明默认` |
| `rng_state` | `Dict[str, Any]` | `必填/未声明默认` |
| `np_rng_state` | `Optional[Dict[str, Any]]` | `None` |
| `history` | `List[Tuple[Dict[str, Any], float]]` | `field(default_factory=list)` |
| `iteration_count` | `int` | `0` |
| `strategy_fields` | `Dict[str, Any]` | `field(default_factory=dict)` |
| `extra_rng_states` | `Dict[str, Dict[str, Any]]` | `field(default_factory=dict)` |

### StrategyContext

[实际实现](../factor_optimizer/search/strategies.py#L100)。

Single direction authority handed to a strategy by the runner.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `objective_spec` | `ObjectiveSpec` | `必填/未声明默认` |

### StrategyContext.direction

[实际实现](../factor_optimizer/search/strategies.py#L112)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`str`。

### StrategyContext.direction_sign

[实际实现](../factor_optimizer/search/strategies.py#L116)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`int`。

### StrategyContext.to_utility

[实际实现](../factor_optimizer/search/strategies.py#L119)。

Map a raw metric to the internal utility convention (maximize).

参数：`(self, raw_metric: float)`。

返回类型：`float`。

### StrategyContext.to_dict

[实际实现](../factor_optimizer/search/strategies.py#L123)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### StrategyContext.from_dict

[实际实现](../factor_optimizer/search/strategies.py#L127)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'StrategyContext'`。

### ParameterSpace

[实际实现](../factor_optimizer/search/strategies.py#L236)。

Describes the search space for a single parameter.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `name` | `str` | `必填/未声明默认` |
| `kind` | `str` | `必填/未声明默认` |
| `low` | `Optional[float]` | `None` |
| `high` | `Optional[float]` | `None` |
| `choices` | `Optional[List[Any]]` | `None` |
| `log_scale` | `bool` | `False` |

### ParameterSpace.validate_for_numeric_kernel

[实际实现](../factor_optimizer/search/strategies.py#L332)。

Reject parameters a numeric GP/TPE kernel cannot represent.

参数：`(self)`。

返回类型：`None`。

### ParameterSpace.sample

[实际实现](../factor_optimizer/search/strategies.py#L349)。

Draw a uniform random sample from this parameter's domain.

参数：`(self, rng: Optional[random.Random]=None)`。

返回类型：`Any`。

### ParameterSpace.to_checkpoint_dict

[实际实现](../factor_optimizer/search/strategies.py#L372)。

Serialize this parameter definition to a JSON-safe dict.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### ParameterSpace.from_checkpoint_dict

[实际实现](../factor_optimizer/search/strategies.py#L384)。

Deserialize a parameter definition captured by ``to_checkpoint_dict``.

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'ParameterSpace'`。

### ParameterSpace.to_grid

[实际实现](../factor_optimizer/search/strategies.py#L388)。

Return a list of evenly-spaced values for grid enumeration.

参数：`(self, n_points: int=10)`。

返回类型：`List[Any]`。

### ParameterSpace.bounds

[实际实现](../factor_optimizer/search/strategies.py#L406)。

Return (low, high) as floats for numeric parameters.

参数：`(self)`。

返回类型：`Tuple[float, float]`。

### SearchSpace

[实际实现](../factor_optimizer/search/strategies.py#L414)。

Ordered collection of parameter definitions.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `parameters` | `List[ParameterSpace]` | `必填/未声明默认` |

### SearchSpace.require_numeric_kernel

[实际实现](../factor_optimizer/search/strategies.py#L432)。

Fail closed if any parameter lacks numeric kernel semantics.

参数：`(self, strategy_name: str)`。

返回类型：`None`。

### SearchSpace.sample

[实际实现](../factor_optimizer/search/strategies.py#L447)。

Draw a random point from the full space.

参数：`(self, rng: Optional[random.Random]=None)`。

返回类型：`Dict[str, Any]`。

### SearchSpace.grid

[实际实现](../factor_optimizer/search/strategies.py#L451)。

Enumerate the Cartesian product of per-parameter grids.

参数：`(self, n_points: int=10)`。

返回类型：`List[Dict[str, Any]]`。

### SearchSpace.validate_point

[实际实现](../factor_optimizer/search/strategies.py#L457)。

Validate a proposed point against this space, fail-closed (FO-P1-03).

参数：`(self, point: Dict[str, Any])`。

返回类型：`None`。

### SearchSpace.to_array

[实际实现](../factor_optimizer/search/strategies.py#L530)。

Convert a named parameter dict to a 1-D numpy array (float64).

参数：`(self, point: Dict[str, Any])`。

返回类型：`np.ndarray`。

### SearchSpace.from_array

[实际实现](../factor_optimizer/search/strategies.py#L543)。

Convert a 1-D numpy array back to a named parameter dict.

参数：`(self, arr: np.ndarray)`。

返回类型：`Dict[str, Any]`。

### SearchSpace.dim

[实际实现](../factor_optimizer/search/strategies.py#L559)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`int`。

### SearchSpace.to_checkpoint_dict

[实际实现](../factor_optimizer/search/strategies.py#L562)。

Serialize this search space to a JSON-safe dict.

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### SearchSpace.from_checkpoint_dict

[实际实现](../factor_optimizer/search/strategies.py#L567)。

Deserialize a search space captured by ``to_checkpoint_dict``.

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'SearchSpace'`。

### SearchStrategy

[实际实现](../factor_optimizer/search/strategies.py#L583)。

Base class for all search strategies.

基类：`ABC`。

### SearchStrategy.__init__

[实际实现](../factor_optimizer/search/strategies.py#L597)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, space: SearchSpace, seed: Optional[int]=None, objective_spec: Optional[ObjectiveSpec]=None, context: Optional[StrategyContext]=None)`。

### SearchStrategy.objective_spec

[实际实现](../factor_optimizer/search/strategies.py#L630)。

The authoritative objective spec driving this strategy's direction.

参数：`(self)`。

返回类型：`ObjectiveSpec`。

### SearchStrategy.direction

[实际实现](../factor_optimizer/search/strategies.py#L635)。

The strategy's objective direction, always derived from the spec.

参数：`(self)`。

返回类型：`str`。

### SearchStrategy.proposal_sequence

[实际实现](../factor_optimizer/search/strategies.py#L640)。

Number of proposals emitted so far (single central counter).

参数：`(self)`。

返回类型：`int`。

### SearchStrategy.record

[实际实现](../factor_optimizer/search/strategies.py#L697)。

Record a completed evaluation for strategies that learn from history.

参数：`(self, params: Dict[str, Any], score: float)`。

返回类型：`None`。

### SearchStrategy.propose

[实际实现](../factor_optimizer/search/strategies.py#L702)。

Suggest a new trial with parameters in ``metadata["params"]``.

参数：`(self, trial_id: Optional[str]=None)`。

返回类型：`Trial`。

### SearchStrategy.proposal_fn

[实际实现](../factor_optimizer/search/strategies.py#L706)。

Callable compatible with ``SearchRunner(proposal_fn=...)``.

参数：`(self)`。

返回类型：`Trial`。

### SearchStrategy.save_state

[实际实现](../factor_optimizer/search/strategies.py#L715)。

Capture this strategy's full stochastic state.

参数：`(self)`。

返回类型：`SearchStrategyState`。

### SearchStrategy.load_state

[实际实现](../factor_optimizer/search/strategies.py#L737)。

Restore this strategy's full stochastic state from ``state``.

参数：`(self, state: SearchStrategyState)`。

返回类型：`None`。

### SearchStrategy.to_checkpoint_dict

[实际实现](../factor_optimizer/search/strategies.py#L780)。

Serialize this strategy to a JSON-safe dict (standalone checkpoint).

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### SearchStrategy.save_checkpoint

[实际实现](../factor_optimizer/search/strategies.py#L784)。

Write this strategy's state to a JSON file at *path*.

参数：`(self, path: str)`。

返回类型：`None`。

### SearchStrategy.from_checkpoint_dict

[实际实现](../factor_optimizer/search/strategies.py#L792)。

Deserialize a standalone strategy checkpoint.

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'SearchStrategy'`。

### SearchStrategy.load_checkpoint

[实际实现](../factor_optimizer/search/strategies.py#L839)。

Load and restore a standalone strategy checkpoint from *path*.

参数：`(cls, path: str)`。

返回类型：`'SearchStrategy'`。

### RandomSearch

[实际实现](../factor_optimizer/search/strategies.py#L852)。

Uniform random sampling over the parameter space.

基类：`SearchStrategy`。

### RandomSearch.__init__

[实际实现](../factor_optimizer/search/strategies.py#L859)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, space: SearchSpace, seed: Optional[int]=None, objective_spec: Optional[ObjectiveSpec]=None, context: Optional[StrategyContext]=None)`。

### RandomSearch.propose

[实际实现](../factor_optimizer/search/strategies.py#L869)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, trial_id: Optional[str]=None)`。

返回类型：`Trial`。

### GridSearch

[实际实现](../factor_optimizer/search/strategies.py#L896)。

Grid enumeration with optional lazy grid reduction (FO-P1-04).

基类：`SearchStrategy`。

### GridSearch.__init__

[实际实现](../factor_optimizer/search/strategies.py#L918)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, space: SearchSpace, grid_points: int=10, max_combinations: Optional[int]=None, seed: Optional[int]=None, objective_spec: Optional[ObjectiveSpec]=None, context: Optional[StrategyContext]=None)`。

### GridSearch.total_combinations

[实际实现](../factor_optimizer/search/strategies.py#L960)。

The theoretical grid size (never materialized).

参数：`(self)`。

返回类型：`int`。

### GridSearch.propose

[实际实现](../factor_optimizer/search/strategies.py#L974)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, trial_id: Optional[str]=None)`。

返回类型：`Trial`。

### BayesianSearch

[实际实现](../factor_optimizer/search/strategies.py#L1009)。

Bayesian optimization with a Gaussian Process surrogate.

基类：`SearchStrategy`。

### BayesianSearch.__init__

[实际实现](../factor_optimizer/search/strategies.py#L1024)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, space: SearchSpace, n_initial: int=5, acquisition: str='ei', ucb_kappa: float=2.576, seed: Optional[int]=None, objective_spec: Optional[ObjectiveSpec]=None, context: Optional[StrategyContext]=None)`。

### BayesianSearch.record

[实际实现](../factor_optimizer/search/strategies.py#L1059)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, params: Dict[str, Any], score: float)`。

返回类型：`None`。

### BayesianSearch.propose

[实际实现](../factor_optimizer/search/strategies.py#L1149)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, trial_id: Optional[str]=None)`。

返回类型：`Trial`。

### TPESearch

[实际实现](../factor_optimizer/search/strategies.py#L1209)。

Tree-structured Parzen Estimator (TPE).

基类：`SearchStrategy`。

### TPESearch.__init__

[实际实现](../factor_optimizer/search/strategies.py#L1233)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, space: SearchSpace, n_initial: int=5, gamma: float=0.25, n_candidates: int=24, seed: Optional[int]=None, maximize: Optional[bool]=None, objective_spec: Optional[ObjectiveSpec]=None, context: Optional[StrategyContext]=None)`。

### TPESearch.record

[实际实现](../factor_optimizer/search/strategies.py#L1297)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, params: Dict[str, Any], score: float)`。

返回类型：`None`。

### TPESearch.propose

[实际实现](../factor_optimizer/search/strategies.py#L1343)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, trial_id: Optional[str]=None)`。

返回类型：`Trial`。

## factor_optimizer/search/supervised_parameter.py

Train-only selection and freezing for supervised repair parameters.

### FrozenSupervisedParameter

[实际实现](../factor_optimizer/search/supervised_parameter.py#L27)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `parent_factor_id` | `str` | `必填/未声明默认` |
| `repair_family` | `str` | `必填/未声明默认` |
| `parameter_name` | `str` | `必填/未声明默认` |
| `value` | `float` | `必填/未声明默认` |
| `candidate_grid` | `Tuple[float, ...]` | `必填/未声明默认` |
| `train_split_ref` | `str` | `必填/未声明默认` |
| `training_evidence_ref` | `str` | `必填/未声明默认` |
| `objective_id` | `str` | `必填/未声明默认` |
| `state_hash` | `str` | `''` |

### FrozenSupervisedParameter.recipe_parameters

[实际实现](../factor_optimizer/search/supervised_parameter.py#L62)。

Apply the frozen choice to validation/test without refitting.

参数：`(self, *, split_role: str)`。

返回类型：`Mapping[str, float]`。

### fit_supervised_parameter

[实际实现](../factor_optimizer/search/supervised_parameter.py#L69)。

Choose from a finite grid using TRAIN evidence only.

参数：`(*, parent_factor_id: str, repair_family: str, parameter_name: str, candidate_grid: Sequence[float], train_scores: Mapping[float, float], split_role: str, train_split_ref: str, training_evidence_ref: str, objective_id: str)`。

返回类型：`FrozenSupervisedParameter`。

## factor_optimizer/search/tiered_evaluation.py

Tiered (funnel) evaluation for factor auto-treatment optimization.

### IssuedTierJob

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L27)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `job_id` | `str` | `必填/未声明默认` |
| `candidate_id` | `str` | `必填/未声明默认` |
| `recipe_version` | `str` | `必填/未声明默认` |
| `tier_name` | `str` | `必填/未声明默认` |
| `attempt` | `int` | `必填/未声明默认` |

### TierEvaluationOutcome

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L43)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `outcome_id` | `str` | `必填/未声明默认` |
| `job_id` | `str` | `必填/未声明默认` |
| `candidate_id` | `str` | `必填/未声明默认` |
| `recipe_version` | `str` | `必填/未声明默认` |
| `tier_name` | `str` | `必填/未声明默认` |
| `attempt` | `int` | `必填/未声明默认` |
| `evaluation_ref` | `str` | `必填/未声明默认` |
| `passed` | `bool` | `必填/未声明默认` |

### EvaluationTier

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L65)。

A single stage in the evaluation funnel.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `name` | `str` | `必填/未声明默认` |
| `cost_multiplier` | `float` | `必填/未声明默认` |
| `fidelity_threshold` | `int` | `必填/未声明默认` |
| `promote_after` | `int` | `0` |

### TieredEvaluationPolicy

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L120)。

Frozen policy describing an ordered evaluation funnel.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `tiers` | `Tuple[EvaluationTier, ...]` | `field(default_factory=lambda: (EvaluationTier('Tier1', 1.0, 0), EvaluationTier('Tier2', 5.0, 1), EvaluationTier('Tier3', 25.0, 3), EvaluationTier('Tier4', 100.0, 4)))` |

### TieredEvaluationPolicy.first_tier

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L172)。

The cheapest screening tier every candidate enters at.

参数：`(self)`。

返回类型：`EvaluationTier`。

### TieredEvaluationPolicy.full_tier

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L177)。

The final tier (full backtest), reached only by survivors.

参数：`(self)`。

返回类型：`EvaluationTier`。

### TieredEvaluationPolicy.tier_names

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L182)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`List[str]`。

### TieredEvaluationPolicy.tier_index

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L185)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, name: str)`。

返回类型：`int`。

### TieredEvaluationPolicy.to_dict

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L191)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### TieredEvaluationPolicy.from_dict

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L197)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'TieredEvaluationPolicy'`。

### TieredEvaluationScheduler

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L208)。

Routes each candidate through the funnel, pruning non-survivors.

### TieredEvaluationScheduler.__init__

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L224)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, policy: TieredEvaluationPolicy)`。

返回类型：`None`。

### TieredEvaluationScheduler.policy

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L243)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`TieredEvaluationPolicy`。

### TieredEvaluationScheduler.initial_tier

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L246)。

The tier every candidate starts at (the cheapest).

参数：`(self)`。

返回类型：`str`。

### TieredEvaluationScheduler.fidelity_for

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L250)。

The fidelity threshold of the tier *candidate_id* currently runs at.

参数：`(self, candidate_id: str)`。

返回类型：`int`。

### TieredEvaluationScheduler.tier_name_for

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L255)。

The name of the tier *candidate_id* currently runs at.

参数：`(self, candidate_id: str)`。

返回类型：`str`。

### TieredEvaluationScheduler.tier_for

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L259)。

The tier *candidate_id* currently occupies.

参数：`(self, candidate_id: str)`。

返回类型：`str`。

### TieredEvaluationScheduler.is_pruned

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L278)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, candidate_id: str)`。

返回类型：`bool`。

### TieredEvaluationScheduler.is_completed

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L281)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, candidate_id: str)`。

返回类型：`bool`。

### TieredEvaluationScheduler.issue

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L284)。

Issue the only outcome-capable job for a candidate's current tier.

参数：`(self, candidate_id: str, recipe_version: str)`。

返回类型：`IssuedTierJob`。

### TieredEvaluationScheduler.advance

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L300)。

Apply an outcome bound to an actually issued job and attempt.

参数：`(self, outcome: TierEvaluationOutcome)`。

返回类型：`Optional[str]`。

### TieredEvaluationScheduler.promote_count

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L369)。

Number of consecutive successful evaluations recorded at a tier.

参数：`(self, candidate_id: str, tier_name: str)`。

返回类型：`int`。

### TieredEvaluationScheduler.to_dict

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L375)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self)`。

返回类型：`Dict[str, Any]`。

### TieredEvaluationScheduler.from_dict

[实际实现](../factor_optimizer/search/tiered_evaluation.py#L389)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(cls, data: Dict[str, Any])`。

返回类型：`'TieredEvaluationScheduler'`。

## factor_optimizer/search/treatment_decision.py

Generalized 8-step treatment decision policy (R61-FI-032 / plan §21.2).

显式导出（含重导出）：`TreatmentMetrics`、`IntegrityGate`、`DesirabilityAnchors`、`DecisionResult`、`HealthDecisionInput`、`TreatmentDecisionPolicy`、`RAW_METRICS`、`BALANCED_DIMENSIONS`、`HEALTH_DIMENSION_IDS`、`grade_to_desirability`、`RAW_TREATMENT_KIND`、`IntegrityCheckResult`、`TreatmentIntegrityEvidence`、`TreatmentIntegrityError`、`build_integrity_evidence`、`describe_integrity_problem`、`digest_value`。

### IntegrityGate

[实际实现](../factor_optimizer/search/treatment_decision.py#L109)。

A single non-negotiable hard-reject gate (STEP 1).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `name` | `str` | `必填/未声明默认` |
| `passed` | `bool` | `必填/未声明默认` |
| `detail` | `str` | `''` |

### TreatmentMetrics

[实际实现](../factor_optimizer/search/treatment_decision.py#L124)。

Raw metrics for a single treatment candidate (legacy scalar path).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `trial_id` | `str` | `必填/未声明默认` |
| `rank_ic` | `float` | `0.0` |
| `icir` | `float` | `0.0` |
| `turnover` | `float` | `0.0` |
| `cost_adjusted_alpha` | `float` | `0.0` |
| `worst_slice` | `float` | `0.0` |
| `exposure` | `float` | `0.0` |
| `coverage` | `float` | `0.0` |
| `stability` | `float` | `0.0` |
| `robustness` | `float` | `0.0` |
| `complexity_score` | `float` | `0.0` |
| `n_transforms` | `int` | `0` |
| `compute_cost` | `float` | `0.0` |
| `bootstrap_samples` | `Dict[str, Sequence[float]]` | `field(default_factory=dict)` |

### TreatmentMetrics.metric

[实际实现](../factor_optimizer/search/treatment_decision.py#L182)。

Return the raw metric value by name.

参数：`(self, name: str)`。

返回类型：`float`。

### HealthDecisionInput

[实际实现](../factor_optimizer/search/treatment_decision.py#L190)。

One candidate's FA health view + decision-local numeric evidence.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `trial_id` | `str` | `必填/未声明默认` |
| `health_view` | `FactorHealthView` | `必填/未声明默认` |
| `robustness` | `float` | `0.5` |
| `complexity_score` | `float` | `0.0` |
| `n_transforms` | `int` | `0` |
| `compute_cost` | `float` | `0.0` |
| `evidence_tier` | `EvidenceTier` | `EvidenceTier.POINT_ESTIMATE_ONLY` |
| `bootstrap_samples` | `Optional[Mapping[str, Sequence[float]]]` | `None` |

### DesirabilityAnchors

[实际实现](../factor_optimizer/search/treatment_decision.py#L259)。

Per-metric desirability anchors for the scalar decision path (STEP 3).

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `maps` | `Dict[str, Dict[str, Sequence[Tuple[float, float]]]]` | `必填/未声明默认` |

### DesirabilityAnchors.score

[实际实现](../factor_optimizer/search/treatment_decision.py#L275)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, metric: str, value: float)`。

返回类型：`float`。

### DecisionResult

[实际实现](../factor_optimizer/search/treatment_decision.py#L283)。

The outcome of the (scalar or health-dimension) decision pipeline.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `winner_trial_id` | `str` | `必填/未声明默认` |
| `step_trace` | `Dict[str, str]` | `field(default_factory=dict)` |
| `pareto_trial_ids` | `List[str]` | `field(default_factory=list)` |
| `statistically_plausible` | `List[str]` | `field(default_factory=list)` |
| `raw_kept` | `bool` | `False` |
| `outcome` | `str` | `'IMPROVED'` |
| `fitness_artifacts` | `Tuple[CandidateFitnessArtifact, ...]` | `()` |
| `authoritative` | `bool` | `False` |
| `purpose` | `str` | `'SCREENING_DIAGNOSTIC'` |

### TreatmentDecisionPolicy

[实际实现](../factor_optimizer/search/treatment_decision.py#L345)。

Treatment-selection facade with one formal and two diagnostic paths.

### TreatmentDecisionPolicy.__init__

[实际实现](../factor_optimizer/search/treatment_decision.py#L365)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, anchors: Optional[DesirabilityAnchors]=None, policy: Optional[WinnerPolicy]=None, uncertainty_config: Optional[UncertaintyConfig]=None, *, fitness_spec: Optional[FactorFitnessSpec]=None, require_integrity_evidence: bool=True, decision_provider: Optional[DecisionProvider]=None)`。

### TreatmentDecisionPolicy.authoritative_decision

[实际实现](../factor_optimizer/search/treatment_decision.py#L404)。

Delegate final selection to FA and verify exact request binding.

参数：`(self, request: SelectionDecisionRequestView)`。

返回类型：`SelectionDecisionReceiptView`。

### TreatmentDecisionPolicy.decide

[实际实现](../factor_optimizer/search/treatment_decision.py#L806)。

Run the local scalar pipeline as an explicit screening diagnostic.

参数：`(self, candidates: Sequence[TreatmentMetrics], *, raw_trial_id: str='RAW', integrity_evidence: Optional[Mapping[str, TreatmentIntegrityEvidence]]=None, screening_only: bool=False)`。

返回类型：`DecisionResult`。

### TreatmentDecisionPolicy.decision

[实际实现](../factor_optimizer/search/treatment_decision.py#L940)。

Run the local health-dimension pipeline as a screening diagnostic.

参数：`(self, candidates: Sequence[HealthDecisionInput], *, raw_trial_id: str='RAW', integrity_evidence: Optional[Mapping[str, TreatmentIntegrityEvidence]]=None, screening_only: bool=False)`。

返回类型：`DecisionResult`。

## factor_optimizer/search/uncertainty_winner.py

Uncertainty-aware winner selection (DLIB-FO-004).

显式导出（含重导出）：`UncertaintyConfig`、`UncertaintyEvidence`、`UncertaintyAwareWinnerSelector`。

### UncertaintyConfig

[实际实现](../factor_optimizer/search/uncertainty_winner.py#L34)。

Controls the uncertainty-aware winner selector.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `minimum_meaningful_improvement` | `float` | `0.01` |
| `equivalence_region` | `float` | `0.5` |
| `confidence_level` | `float` | `0.95` |
| `decisive_probability` | `float` | `0.95` |
| `equivalence_probability_band` | `float` | `0.15` |

### UncertaintyEvidence

[实际实现](../factor_optimizer/search/uncertainty_winner.py#L86)。

Bootstrap uncertainty evidence for a single candidate.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `trial_id` | `str` | `必填/未声明默认` |
| `dimension_samples` | `Dict[str, Sequence[float]]` | `必填/未声明默认` |
| `complexity_score` | `float` | `0.0` |
| `turnover` | `float` | `0.0` |
| `compute_cost` | `float` | `0.0` |
| `n_transforms` | `int` | `0` |
| `bootstrap_plan_ref` | `Optional[str]` | `None` |
| `draw_ids` | `Optional[Sequence[str]]` | `None` |

### UncertaintyAwareWinnerSelector

[实际实现](../factor_optimizer/search/uncertainty_winner.py#L326)。

Select a winner among candidates using statistical uncertainty.

### UncertaintyAwareWinnerSelector.__init__

[实际实现](../factor_optimizer/search/uncertainty_winner.py#L336)。

此入口没有独立文档字符串；结合所属类合同、功能手册及实现链接使用，不推断额外行为。

参数：`(self, policy: WinnerPolicy, config: Optional[UncertaintyConfig]=None)`。

### UncertaintyAwareWinnerSelector.select

[实际实现](../factor_optimizer/search/uncertainty_winner.py#L346)。

Return the winning candidate among ``candidates``.

参数：`(self, candidates: Sequence[UncertaintyEvidence], robustness_scores: Dict[str, float])`。

返回类型：`UncertaintyEvidence`。

## factor_optimizer/search/winner_selector.py

Robust winner selector for factor auto-treatment optimization.

### WinnerPolicy

[实际实现](../factor_optimizer/search/winner_selector.py#L31)。

Versioned policy controlling the robust winner selector.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `alpha` | `float` | `必填/未声明默认` |
| `beta` | `float` | `必填/未声明默认` |
| `gamma` | `float` | `必填/未声明默认` |
| `lambda_` | `float` | `必填/未声明默认` |
| `policy_id` | `str` | `必填/未声明默认` |
| `policy_version` | `str` | `必填/未声明默认` |

### WinnerSetPolicy

[实际实现](../factor_optimizer/search/winner_selector.py#L68)。

Predeclared policy for a small complementary winner set.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `max_winners` | `int` | `3` |
| `max_per_family` | `int` | `1` |
| `utility_epsilon` | `float` | `1e-06` |
| `duplicate_correlation` | `float` | `0.995` |
| `minimum_incremental_value` | `float` | `0.0` |

### RobustBalancedUtility

[实际实现](../factor_optimizer/search/winner_selector.py#L145)。

Compute the robust balanced utility of a candidate recipe.

参数：`(dimension_desirabilities: List[float], robustness_score: float, complexity_score: float, policy: WinnerPolicy)`。

返回类型：`float`。

### augmented_tchebycheff

[实际实现](../factor_optimizer/search/winner_selector.py#L192)。

Alternative winner metric: augmented weighted Tchebycheff scalarization.

参数：`(dimension_desirabilities: List[float], robustness_score: float, complexity_score: float, policy: WinnerPolicy, reference: Optional[Sequence[float]]=None)`。

返回类型：`float`。

### select_winner

[实际实现](../factor_optimizer/search/winner_selector.py#L236)。

Select the single best candidate among the Pareto frontier.

参数：`(pareto_candidates: Iterable[ParetoPoint], robustness_scores: Dict[str, float], complexity_scores: Dict[str, float], policy: WinnerPolicy)`。

返回类型：`ParetoPoint`。

### select_complementary_winners

[实际实现](../factor_optimizer/search/winner_selector.py#L319)。

Select zero or more eligible, incremental, non-duplicate candidates.

参数：`(pareto_candidates: Iterable[ParetoPoint], robustness_scores: Dict[str, float], complexity_scores: Dict[str, float], utility_policy: WinnerPolicy, set_policy: WinnerSetPolicy, *, pairwise_correlations: Optional[Dict[Tuple[str, str], float]]=None)`。

返回类型：`List[ParetoPoint]`。

## factor_optimizer/seen/__init__.py

Seen cache: track previously evaluated factors using FE canonical identity.

显式导出（含重导出）：`SeenCache`、`SeenRecord`。

## factor_optimizer/seen/identity.py

SeenCache: track previously evaluated factors to avoid duplicates.

### SeenRecord

[实际实现](../factor_optimizer/seen/identity.py#L9)。

Record of a previously seen factor.

本类声明字段（继承字段见基类；实际限制仍需合同校验）：

| 字段 | 类型 | 默认值/值 |
|---|---|---|
| `canonical_hash` | `str` | `必填/未声明默认` |
| `first_seen_at` | `datetime` | `必填/未声明默认` |
| `trial_id` | `str` | `必填/未声明默认` |
| `factor_id` | `Optional[str]` | `None` |
| `metadata` | `Optional[Dict]` | `None` |

### SeenRecord.to_dict

[实际实现](../factor_optimizer/seen/identity.py#L27)。

Serialize to dictionary.

参数：`(self)`。

返回类型：`Dict`。

### SeenRecord.from_dict

[实际实现](../factor_optimizer/seen/identity.py#L38)。

Deserialize from dictionary.

参数：`(cls, data: Dict)`。

返回类型：`'SeenRecord'`。

### SeenCache

[实际实现](../factor_optimizer/seen/identity.py#L46)。

Cache of previously evaluated factors by canonical identity.

### SeenCache.__init__

[实际实现](../factor_optimizer/seen/identity.py#L54)。

Initialize seen cache.

参数：`(self, fe_adapter=None)`。

### SeenCache.mark_seen

[实际实现](../factor_optimizer/seen/identity.py#L65)。

Mark a factor as seen.

参数：`(self, canonical_hash: str, trial_id: str, factor_id: Optional[str]=None, metadata: Optional[Dict]=None)`。

返回类型：`SeenRecord`。

### SeenCache.is_seen

[实际实现](../factor_optimizer/seen/identity.py#L101)。

Check if a canonical hash has been seen.

参数：`(self, canonical_hash: str)`。

返回类型：`bool`。

### SeenCache.get_record

[实际实现](../factor_optimizer/seen/identity.py#L105)。

Get seen record by canonical hash.

参数：`(self, canonical_hash: str)`。

返回类型：`Optional[SeenRecord]`。

### SeenCache.get_by_trial

[实际实现](../factor_optimizer/seen/identity.py#L109)。

Get seen record by trial ID.

参数：`(self, trial_id: str)`。

返回类型：`Optional[SeenRecord]`。

### SeenCache.compute_canonical_hash

[实际实现](../factor_optimizer/seen/identity.py#L116)。

Compute canonical hash through FE adapter.

参数：`(self, factor_definition)`。

返回类型：`str`。

### SeenCache.check_and_mark

[实际实现](../factor_optimizer/seen/identity.py#L137)。

Check if factor was seen and mark if new.

参数：`(self, factor_definition, trial_id: str)`。

返回类型：`tuple[bool, Optional[SeenRecord]]`。

### SeenCache.size

[实际实现](../factor_optimizer/seen/identity.py#L158)。

Return number of unique factors seen.

参数：`(self)`。

返回类型：`int`。

### SeenCache.clear

[实际实现](../factor_optimizer/seen/identity.py#L162)。

Clear all seen records.

参数：`(self)`。

返回类型：`None`。

### SeenCache.export_records

[实际实现](../factor_optimizer/seen/identity.py#L167)。

Export all seen records as dictionaries.

参数：`(self)`。

返回类型：`list[Dict]`。

### SeenCache.import_records

[实际实现](../factor_optimizer/seen/identity.py#L171)。

Import seen records from dictionaries.

参数：`(self, records: list[Dict])`。

返回类型：`None`。

## 源码一致性

<details>
<summary>展开模块指纹</summary>

| 模块 | SHA-256 |
|---|---|
| `factor_optimizer/__init__.py` | `c56006f3491a1795bc0dc673f116e214f4f20e2965ba1d468a452c9f0cc1d76d` |
| `factor_optimizer/adapters/__init__.py` | `d2cd050fc93d62119ef592d8d663e985e5218554c6cb289d839e384a49b40a23` |
| `factor_optimizer/adapters/factor_assets.py` | `6a2ad44b43710199fd6eed901de2974fb8a66314f525be016c2f4796bf8af6d4` |
| `factor_optimizer/adapters/factor_engine.py` | `924ecf37167dfba994e803f88c66ab6bac83586b3452c00b4535c35fbf87d2f0` |
| `factor_optimizer/adapters/fitness.py` | `4486884657f730c65064fe6e1aebc7555a515fbc3cba41c71c1bf8c9e4c48ce5` |
| `factor_optimizer/adapters/layered_decay.py` | `86ac91175c82eef473abef116a7a67802872a0da96ce0fa7a6022bf783bb3e26` |
| `factor_optimizer/adapters/preprocessing.py` | `9518e42ad19077f34700d0edb9d2c126974cd687d969defb44d44175d1b61efe` |
| `factor_optimizer/adapters/quant_evaluator.py` | `f91ab4d4514ca1d8e2131842c42fd972ff86433f2d3bdaff57d86d6674aecb17` |
| `factor_optimizer/adapters/repair_execution.py` | `260fad479637ff715f6cbd93e552aab338670727b9600746ec0e1aa7e1481466` |
| `factor_optimizer/capabilities.py` | `efb1fbf1b54b1b14f128ffe63c74f54f7a22f99a409a0761aac67fc1b6e3ed28` |
| `factor_optimizer/complexity/__init__.py` | `79c8daf01ef8071b77eb7cbb8df4ae45fc9e2d4a4349f35d402a12f79936e696` |
| `factor_optimizer/complexity/budget.py` | `d797ada6f52d0ccab3820450fb12e8f84f3f3dbb4e9f1e53f64eb5bb21a3469d` |
| `factor_optimizer/complexity/profile.py` | `f3063dc26e486c33ad3aaefb3c8b5330c9e1f28eacb84bec4518cba793ada23a` |
| `factor_optimizer/contracts/__init__.py` | `14ef3afa8541d37f91e067c43fa07adbcbffc6f263647ff5d4653e08713274c4` |
| `factor_optimizer/contracts/budget_extension.py` | `28ed5293531dc782a1233cbe99ec3b1ad12f3d61a644f4a3742b566508176f2b` |
| `factor_optimizer/contracts/campaign_store.py` | `9919bd28419c1c53b9d65274a91d999c427a50b9d8f83c057c5c913940ce45fd` |
| `factor_optimizer/contracts/candidate_mutation.py` | `e1bc4d9e1603e04ad4d781b64289fb44da0fc666b8203c3d896214f0515a745b` |
| `factor_optimizer/contracts/evaluation_artifact.py` | `6573c2904d7d06d6ebd9838936b3c86d8e4ca3ab123d32571f14bd41e1db64b3` |
| `factor_optimizer/contracts/evidence_value.py` | `baad4e4f12539dd6d0a1f2194d9bcb687c53343af3cae52881b60e0eb87c4ed9` |
| `factor_optimizer/contracts/factor_fitness.py` | `95d31bd48f0fc97ab16d87f0eaaf2e4f428a82b6bd3a6d132ab13f0592101b27` |
| `factor_optimizer/contracts/library_snapshot_ref.py` | `0e532900b99adce38c5b30fd2a0544f393394019ba493b741a963d35e6ec1cda` |
| `factor_optimizer/contracts/multiplicity.py` | `44875c5de62fe684dd1bb253a47a74b27f18b60228c6a4a0815a3ef3e4a48ee3` |
| `factor_optimizer/contracts/objective.py` | `ec8bf1cfacbf547e2cae2ae05175fe9ef38abf67139773c48bfbc46f4bf94704` |
| `factor_optimizer/contracts/search_budget.py` | `0fc7fe2bf7754952ae0c261a1b5e974d4e0b9de9595b834bb8c0fa1622923eaf` |
| `factor_optimizer/contracts/splits.py` | `e77f13e4dd3289058b445c68ded01e2a8a02ab29e522f1ce2723e09a9c564018` |
| `factor_optimizer/contracts/statistical_governance.py` | `a483d1bb19563c2ea22ea5ad9f43f193d9f8b86bb925bb004fdf39e224f11e1a` |
| `factor_optimizer/contracts/treatment_integrity.py` | `010d28bec0ec8deef08821f7938dadd85d1d10a7666185a963df3f8d4877ac88` |
| `factor_optimizer/contracts/treatment_result.py` | `c16cf30f2784ae335c8bc2f469d0e9d5309a530bf34a0ada189de4c99a987111` |
| `factor_optimizer/contracts/trial.py` | `1f446d513cdf9ab261cb979335c988389761f9fd3b396ba34e04883fb36cc325` |
| `factor_optimizer/contracts/trial_ledger.py` | `a62289e7cffe6591b033fa7fda52ef7f29431ec9088a6c3206989dc73a0fc51a` |
| `factor_optimizer/contracts/validator.py` | `87714f5ee74d6d2a0b0cd1e594e14fc0738b76f6c3d643c49c1285b2b5dd081e` |
| `factor_optimizer/data_capabilities.py` | `aca1502b1f57b9c4d4427434c64e18296df60cfd24933ee17b9cf55aa257d2f5` |
| `factor_optimizer/data_providers.py` | `b48e46739d93f4d58ed8f4e467b3fd1ba814258b20206651d18536c2dab5c3e9` |
| `factor_optimizer/errors.py` | `1ca3b4509ed4ef22865a28d2af269e750cd83fb14341ba5159f43904690e56bb` |
| `factor_optimizer/grammar/__init__.py` | `6dcf330abeef99755f874e82841ec9b3c5f98a15f8f97a2b23f6d08f29eddd47` |
| `factor_optimizer/grammar/mutation_spec.py` | `f0e99a12550ceb442e59d92f943144d07c13becf9a90ee57b01421d830e06e92` |
| `factor_optimizer/grammar/registry.py` | `7944c9c61149a5e6a36503c98af168f39d756019d641df4de12b56f8094e4457` |
| `factor_optimizer/grammar/validation.py` | `f1b030934370bbbc556b64fc41d19a7730a6705303ae25825ef56e0ceebece45` |
| `factor_optimizer/llm/__init__.py` | `ecda8ca3a8f5c0531e80bb41a2201058bb7133a605f693a69fbf6b2f8424dae5` |
| `factor_optimizer/llm/prompts.py` | `b0a9f202f8cb611cfbfe2c061b0dce2b900e2ece4b2d8275756b5af061cfc95c` |
| `factor_optimizer/llm/proposal.py` | `5694c083d3360553911c8d5d1440dc6aba80e7896101b0bec3fc5d20e91bc4d5` |
| `factor_optimizer/llm/records.py` | `92cb8ffd2aa99686057fd91fc3cdf415f3c49aadabab09a11d6f65fb19ce7b8e` |
| `factor_optimizer/policy/__init__.py` | `ddba38b9e34581d134c45902fd04db8e8fc540fd5b2438e4a3a46e3753888caf` |
| `factor_optimizer/policy/decisions.py` | `832c06ae174ab078f518a9ee8237efe89cd88849c2f42e48e250dc6b676d16d3` |
| `factor_optimizer/policy/repair.py` | `193d142a9cd970b4e9a055fe60642d09514145868107cb25e039f956a9501570` |
| `factor_optimizer/policy/repair_registry.py` | `1f337900eec0750622901a2f26f0bb19ad5bae248399c62546b3ede85b998545` |
| `factor_optimizer/ports/__init__.py` | `5ae5b84348a74c71b61d1465bf3bb3acc3c77b5b7186436ed5c299adb677c827` |
| `factor_optimizer/ports/factor_intelligence.py` | `d1f8cb9761da774d354cb3b5d6d91f79b54cb5f7c3519d0911ad98e7c79e2821` |
| `factor_optimizer/research_baseline.py` | `3eee6b3b915c7a93df09b61ea7d5b8eaba7b662abf36201b565be048f1c0dfc0` |
| `factor_optimizer/research_batch.py` | `88efcb0cfb4274397a29d7c0603dc1d237aca22de25001ad25bb568a6c4b1ce5` |
| `factor_optimizer/research_decay.py` | `1355b74f2ca32c7f819357880214580502eeff1df9d09593c601ed14d7fb71b0` |
| `factor_optimizer/research_diagnostics.py` | `6353a8fcf0777f751b14f255e77930de89696972a99ae5ba4f63ea66beef104f` |
| `factor_optimizer/research_final_report.py` | `58ec87ceff4f1994388e884b4aa7f32f71120c6eff41c3d9f624203ad6f077c9` |
| `factor_optimizer/research_fitness.py` | `8d1174abfe3cd22c0e76276a88d713fc55fc2a1574169a408c23499bf6c1581f` |
| `factor_optimizer/research_manifest.py` | `5098274c402cf69b3c90cfb0fdb37b50fe945db89ac74066ec757270bd120a17` |
| `factor_optimizer/search/__init__.py` | `bc5887aefa3239ffab88396650aa76b1b060b0e94d916e19923f8fa4e4a53419` |
| `factor_optimizer/search/categorical_strategy.py` | `8dc0559f952cd904f436ec90c49e7fa2ec5e175593881d36128cdcead46506ac` |
| `factor_optimizer/search/conditional_search.py` | `4b6dfeceacd797ad16f2406d0cd8c5cc1a4146c59c65c611f875356ce7251731` |
| `factor_optimizer/search/desirability.py` | `ee476f33f383ef8831341df3f8cbe4110fd0149b0790ddbb5ddad38cfa7f055b` |
| `factor_optimizer/search/desirability_registry.py` | `f2e2c47cdb4dc24cf4621e9b6d2d78331272b650ff0fbf374214f83bc3dcc6b0` |
| `factor_optimizer/search/diagnosis_routing.py` | `997ca9033fddbdb6248d3c06620edb528357b116c879c84c6a3e4d31f20ed229` |
| `factor_optimizer/search/dimensions.py` | `d96278aa024bf275edc88072a5abfe0b749f023047971b94bccc99ae67964d90` |
| `factor_optimizer/search/lineage.py` | `979031c4c987a40cbb1987b360704ffd8fe0749792a21db0352c41497657b25b` |
| `factor_optimizer/search/multifidelity.py` | `6967520ca3f3f78f5648cfccaa397f759514091f2dbb2bc3977506990f1747df` |
| `factor_optimizer/search/paired_comparison.py` | `9bfdabf2b4e3c26b41697553fcdcbbcf49f49aa9ef20222546c69f57dfbce3be` |
| `factor_optimizer/search/pareto.py` | `b368befd9d0d3d46498a43cbd1b399c474d795457de2af27cdd5e9eebbafebe6` |
| `factor_optimizer/search/plateau.py` | `7c096dfd215cc4d2f22d2953a3adf3ff73ac2914345724aac36e542cda3e117d` |
| `factor_optimizer/search/runner.py` | `b7e0a5b19c29005e73d6c3b5083a5552a8fe3cd497c13a23e2562faaeeebc13a` |
| `factor_optimizer/search/statistical_consumption.py` | `a72535f6b06ba50a8996c35c606851980bd7f578e456eaa9b76a4a9bcc3c747b` |
| `factor_optimizer/search/strategies.py` | `9507b9bc3f72c25fcee43a966781d936fcecf25325be7dd572e56cd451972cc2` |
| `factor_optimizer/search/supervised_parameter.py` | `becb5018c82a3c522d8641530ca94ba39c7e3a9d2467ced68e3ae57e4b9a797c` |
| `factor_optimizer/search/tiered_evaluation.py` | `649bce9eef0084762dae3244a894872e7eabcdeb3a277381c480fb17b563a8c8` |
| `factor_optimizer/search/treatment_decision.py` | `7073d47030e31fb8b66848c748960ed24da7dcc8452fb0082750a3aa98a0a29e` |
| `factor_optimizer/search/uncertainty_winner.py` | `102b6c9d9ce32b010876c9f915f8abfd0cae70ad9bc180e75ad9cbdbf5387ca9` |
| `factor_optimizer/search/winner_selector.py` | `3a2ff94402e3eb4b7e1b0a6c373362a96bab475d30c28c05e62db43edc70cb77` |
| `factor_optimizer/seen/__init__.py` | `f37e222be83f53595c4ee7325542f299a1b6d64eeebce400e5296d0f56d33676` |
| `factor_optimizer/seen/identity.py` | `dbf0d6cbbb1cd12f907a12fa2ce0089a9c2f1a25c9ee7da96b1735032163ff10` |

</details>

重新生成：`python scripts/build_api_reference.py`；检查：`python scripts/build_api_reference.py --check`。
