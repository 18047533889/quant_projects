# FactorOptimizer Changelog

All notable changes to FactorOptimizer will be documented in this file.

## [0.1.0] - 2026-08-14

### Added - Core Functionality

#### Contracts
- `CandidateMutation`: Immutable mutation proposal with provenance
- `SearchBudget`: Resource limits (trials, evaluations, cost, LLM calls)
- `BudgetTracker`: Resource usage tracking with can_*/record_* methods
- `Trial`: Trial lifecycle management with status transitions
- `TrialStatus`: Enum (PROPOSED, VALIDATING, LEGAL, ILLEGAL, EVALUATING, EVALUATED, FAILED, DUPLICATE)

#### Grammar
- `MutationSpec`: Versioned mutation specification with parameter types
- `ParameterSpec`: Parameter validation (kind, role, min/max, allowed values)
- `ParameterKind`: Enum (INTEGER, FLOAT, BOOLEAN, STRING, ENUM, *_LIST)
- `ParameterRole`: Enum (WINDOW, DECAY, THRESHOLD, QUANTILE, SCALAR, CATEGORICAL, STRUCTURAL, TIMING, CAUSAL)
- `MutationRegistry`: Centralized mutation catalog
- `MutationValidator`: Legal mutation validation
- 6 default mutations: parameter_tune, window_adjust, operator_swap, decay_adjust, linear_combination, threshold_adjust

#### Seen Cache
- `SeenCache`: In-memory deduplication with FE canonical hash
- `SeenRecord`: First-seen record with trial/factor IDs

#### Complexity Estimation
- `ComplexityProfile`: Operator count, depth, lookback, estimated cost
- `ComplexityEstimator`: Heuristic complexity estimation

#### Admission Policy
- `AdmissionPolicy`: Conservative admission decisions
- `AdmissionCriteria`: Max complexity, min value, allowed domains
- `AdmissionDecision`: Immutable decision record
- `AdmissionVerdict`: Enum (ADMITTED, REJECTED, DEFERRED, CONDITIONAL)
- `RejectionReason`: Enum (BUDGET_EXCEEDED, DUPLICATE, ILLEGAL, etc.)

#### Repair Strategy
- `RepairMapper`: Diagnosis-to-repair mapping
- `DiagnosisRecord`: Problem identification
- `RepairProposal`: Suggested fix with confidence
- `DiagnosisKind`: Enum (HIGH_COMPLEXITY, POOR_COVERAGE, etc.)
- `RepairStrategy`: Enum (REDUCE_WINDOW, ADD_REGULARIZATION, etc.)

#### Search Orchestration
- `SearchRunner`: Main search loop with stopping conditions
- `SearchSession`: Complete search state and history
- `SearchConfig`: Search parameters and stopping criteria

#### Multi-Fidelity
- `MultiFidelityScheduler`: Progressive evaluation tiers
- `FidelityTier`: Enum (L0=5%, L1=15%, L2=50%, L3=100%, L4=100%+CV)
- `FidelitySpec`: Tier specification with cost/duration
- `PromotionCriteria`: Promotion decision logic

#### Pareto Frontier
- `ParetoFrontier`: Multi-objective frontier tracking
- `ParetoPoint`: Point in objective space
- `ParetoArchive`: Frontier evolution history
- Methods: dominates, extremes, hypervolume, coverage, spacing

#### Plateau Detection
- `PlateauDetector`: Convergence detection
- `AdaptivePlateauDetector`: Dynamic threshold adjustment
- `MultiObjectivePlateauDetector`: Frontier-based stopping
- `PlateauConfig`: Detection parameters

#### Lineage Tracking
- `LineageTree`: Trial ancestry DAG
- `LineageNode`: Single trial in tree
- `LineageAnalyzer`: Success rate analysis
- Methods: path_to_seed, best_in_lineage, subtree_stats, prune_lineage

#### LLM Integration (Stubs)
- `PromptTemplate`: Versioned prompt templates
- `PromptRegistry`: Template catalog
- `LLMRecord`: LLM call record for auditing
- `RecordStore`: LLM call history
- `ProposalGenerator`: Mock LLM-based proposal (stub)

#### Adapters
- `FactorEngineAdapter`: Protocol for FE legality checking
- `QuantEvaluatorAdapter`: Protocol for QE evaluation
- `create_fe_adapter()`, `create_qe_adapter()`: Factory functions

### Testing
- 22 test files, comprehensive coverage
- Unit tests for all contracts and grammar
- Search orchestration tests
- Pareto frontier tests
- Lineage tracking tests
- Multi-fidelity tests

### Documentation
- README with scope and boundaries
- Design principles documented
- Protocol-based adapter pattern

### Known Limitations

#### Not Implemented
- LLM proposal generation (stub only)
- Parallel evaluation
- Distributed search
- Advanced repair strategies
- Real-time monitoring

#### Conservative Validation
- May reject valid mutations (fail-closed)
- No arbitrary Python code execution
- Limited mutation types (6 default)

### Dependencies
- `pyyaml>=6.0`, `numpy>=1.24`
- Optional: `factor-engine>=0.3.0`, `quant-evaluator>=0.1.0`

---

## [Unreleased]

### Planned for 0.2.0
- [ ] LLM-based proposal generation
- [ ] Parallel evaluation
- [ ] Advanced repair strategies
- [ ] Real-time search monitoring
- [ ] More mutation types

---

**Last Updated:** 2026-08-14
