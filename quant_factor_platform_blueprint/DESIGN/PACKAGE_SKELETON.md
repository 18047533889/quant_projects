# Package Skeleton

**STATUS: DRAFT / NOT FROZEN / IMPLEMENTATION NOT AUTHORIZED**

This is a proposed repository-relative skeleton only. It does not authorize creating these directories or files. It is based on Wave 0 evidence bound to `LOCAL_REPO_SNAPSHOT` commit `ddb03749b7ff85e63b633770c43dfbbb7562af19`; a later snapshot delta is mandatory because the repository changes concurrently. FA migration, consolidated boundary/matrix, independent review, and compatibility/extraction plans remain blockers.

## 1. Repository Shape

```text
/schemas/                         # language-neutral schemas; lead-only; not a runtime package
quant_evaluator/                  # pure Evidence/Metric/Diagnosis
factor_optimizer/                 # legal mutations/search decisions
factor_assets/                    # identity/registry/relations/lifecycle/selection
factor_preprocess/                # transforms/neutralization/representation
research_control/                 # lightweight append-only ledger; placement unresolved
```

No `quant_contracts`, `common`, `shared`, `core`, or `helpers` dumping package is proposed. Root schemas may be generated into package-local types during builds, but the generated types must not become a fifth runtime dependency.

## 2. QuantEvaluator Skeleton

```text
quant_evaluator/
  pyproject.toml                         # lead-owned final packaging
  __init__.py                             # lead-owned final exports
  api/
    __init__.py
    evaluate.py                           # evaluate/evaluate_many batch facade
    requests.py                           # EvaluationRequest
    results.py                            # EvaluationBundle, EvidenceRef
  contracts/
    __init__.py
    factor_batch.py                       # FactorBatch, axes, validity/value refs
    label_bundle.py                       # explicit LabelBundle
    context.py                             # supplied timing/calendar/universe context refs
    envelope.py                            # package-local envelope projection
  registry/
    __init__.py
    metrics.py                            # MetricSpec/catalog/status, no hidden policy
    presets.py                             # named metric sets and tier declarations
  planner/
    __init__.py
    batch_plan.py                         # block/chunk/intermediate planning
    dependency_plan.py                    # metric dependency graph
  runtime/
    __init__.py
    evaluator.py
    intermediates.py
    budgets.py
    diagnostics.py
  metrics/
    __init__.py
    quality.py
    distribution.py
    ic.py
    ic_summary.py
    quantile.py
    turnover.py
    probe_portfolio.py
    portfolio_stats.py
    exposure.py
    temporal.py
    robustness.py
    multiple_testing.py
  kernels/
    __init__.py                            # performance owner only
    reference_bridge.py
    fast.py
  adapters/
    __init__.py
    data_access.py                         # optional DA get_store/read adapter
    factor_engine.py                       # optional FE result/identity adapter
    pandas.py                              # explicit reference/debug adapter only
  diagnosis/
    __init__.py
    factor.py
    warnings.py
  tests/
    contracts/
    metrics/
    parity/
    leakage/
    integration/
    extraction/
  benchmarks/
    README.md
    baseline.py
```

QE must consume a caller-supplied `LabelBundle`; it cannot infer forward labels, perform PIT joins, or own admission policy. The public facade is batch-first. The pandas adapter is reference/debug and must never be an implicit production fallback.

## 3. FactorOptimizer Skeleton

```text
factor_optimizer/
  pyproject.toml
  __init__.py
  contracts/
    __init__.py
    candidate_mutation.py                 # CandidateMutation
    search_budget.py
    trial.py
  grammar/
    __init__.py
    mutation_spec.py                      # versioned typed MutationSpec
    registry.py
    validation.py
    legality.py                            # calls FE adapter; no FE kernel copy
  complexity/
    __init__.py
    profile.py                            # adapter projection of FE complexity
    budget.py
  search/
    __init__.py
    runner.py
    multifidelity.py                       # L0-L4 mapping after decision
    pareto.py
    plateau.py
    lineage.py
  llm/
    __init__.py
    proposal.py                            # structured proposal only
    prompts.py
    records.py                             # model/prompt/hash/signature records
  adapters/
    __init__.py
    factor_engine.py                       # FE canonical identity/operator legality
    quant_evaluator.py                     # QE evidence requests
  policy/
    __init__.py
    repair.py
    decisions.py
  seen/
    __init__.py
    identity.py                            # uses FE identity adapter
    cache.py
  tests/
    grammar/
    search/
    adapters/
    lineage/
    budgets/
    extraction/
  benchmarks/
    baseline.py
```

FO does not parse formulas, implement operators, compute metrics, or execute arbitrary generated Python. `complexity/profile.py` is an adapter view over FE parser budgets, analyzer history, operator cost, and plan cost. Search decisions require QE evidence and remain optional in the platform DAG.

## 4. FactorAssets Skeleton

```text
factor_assets/
  pyproject.toml
  __init__.py
  contracts/
    __init__.py
    asset.py                            # FactorAsset
    factor_set.py                       # FactorSet / unresolved artifact name
    evidence_ref.py
    lifecycle.py
  registry/
    __init__.py
    repository.py
    migrations.py
    snapshots.py
  identity/
    __init__.py
    canonical.py                        # calls FE adapter; no second hash
    factor_id.py
    lineage.py
  seen_index/
    __init__.py
    exact.py
    structural.py
    records.py
  novelty/
    __init__.py
    orchestrator.py
    provider.py                          # QE/provider protocol boundary
  similarity/
    __init__.py
    fingerprint.py
    ann.py                               # shortlist only
    exact.py                             # QE-backed exact evidence request
  graph/
    __init__.py
    sparse.py
    edges.py
  clustering/
    __init__.py
    families.py
    lineage.py
  selection/
    __init__.py
    gates.py
    pareto.py
    policy.py
  aggregation/
    __init__.py
    specs.py
    representatives.py
  adapters/
    __init__.py
    quant_evaluator.py
    factor_engine.py                    # optional definition identity adapter
    data_access.py                       # optional factor-value/catalog reads
  tests/
    registry/
    identity/
    novelty/
    graph/
    selection/
    lifecycle/
    no_raw_values/
    extraction/
  benchmarks/
    baseline.py
```

FA does not store raw factor matrices, run a second formula parser, or treat `corr > threshold` as its sole duplicate rule. It may store bounded compact derived fingerprints only after unresolved retention/reversibility policy is frozen. DA factor reads remain through `get_store()`/public read APIs; a public universe facade gap is handled by an adapter, not duplicated semantics.

## 5. FactorPreprocess Skeleton

```text
factor_preprocess/
  pyproject.toml
  __init__.py
  contracts/
    __init__.py
    policy.py                           # PreprocessingPolicy
    state.py                            # FittedState
    feature_bundle.py                   # FeatureBundle / ModelInput decision pending
    context.py                           # exposure/timing/fit context
  registry/
    __init__.py
    transforms.py
    policies.py
  transforms/
    __init__.py
    cross_sectional.py
    winsor.py
    rolling.py
    volatility.py
    missingness.py
    freshness.py
  neutralization/
    __init__.py
    design_matrix.py
    ols.py
    regularized.py
    diagnostics.py
  representation/
    __init__.py
    multichannel.py
    linear_ready.py
    tree_ready.py
    neural_ready.py
  kernels/
    __init__.py
    reference_bridge.py
    fast.py
  adapters/
    __init__.py
    factor_assets.py
    data_access.py                       # only if exposure/context retrieval is needed
  tests/
    contracts/
    transforms/
    neutralization/
    representation/
    future_poison/
    fold_boundary/
    parity/
    extraction/
  benchmarks/
    baseline.py
```

FP separates stateless transforms from fitted state. Rolling and volatility transforms must make window closure, lag, warmup, and asset isolation explicit. PCA/ICA, estimated power transforms, and nonlinear regressors remain advanced/opt-in until a real serializable `FittedState` and fold-local evidence exists. No future-return label or model training belongs here.

## 6. Optional Adapter Rules

- Adapter modules may import `data_access`, FE's top-level public modules (`api`, `expr`, `ir`, `runtime`, etc.), QE, or FA only where the contract says so.
- Core package imports must not import adapter modules eagerly.
- Adapter construction must fail with a typed `OptionalDependencyMissing` or `CapabilityError`, preserving a usable core import.
- DA adapter calls `get_store()` and public `DataAccessStore` methods; it must not import physical `read`, `runtime`, or private universe resolution modules.
- FE adapter compiles through public FE runtime/planner boundaries and obtains canonical identity/complexity; it must not call cleaned kernels directly.
- Corpus adapters live under tests or package-local `adapters/corpus/` only and are never runtime dependencies.

## 7. Package-Level Public API Minimum

Each package must expose only a small, documented facade from `__init__.py`, plus `package_info()`/capability metadata as appropriate. Internal modules, generated snapshots, benchmarks, and test fixtures are not cross-package APIs. Every public record includes schema version and provenance required by the contract envelope. API, schema, metric, transform, mutation, and package versions are independent.

## 8. Implementation Order After Freeze

1. Rebind evidence to a later snapshot delta and close Phase 0 artifacts.
2. Freeze root schemas, exact signatures, error codes, ownership, and adapter protocols.
3. QE reference contracts and core metric golden tests.
4. FP stateless/fitted contracts and leakage tests.
5. FA identity/registry/append-only lifecycle after its migration and independent review.
6. FO grammar and FE/QE adapters; then bounded search.
7. QE advanced evidence, FA scale, and FP representations.
8. Research Control and real corpus integration.
9. Extraction, boundary, license, performance, leakage, math, A-share, and corpus audits.

No skeleton directory should be created until blockers in `CONTRACT_FREEZE_DRAFT.md` close.

## 9. Unresolved Skeleton Decisions

- Root schema filenames and generated-type mechanism.
- Exact package build layout and whether each package is independently installable from repository root or subdirectories.
- Canonical terminal type (`FeatureBundle` versus `ModelInputBundle`).
- `FactorSet` versus `FactorSetArtifact` relationship.
- Research Control location and ownership.
- Exact cross-package import allowlist and CI checker.
- FA fingerprint dimensions, retention, reversibility, and storage limits.
- QE metric registry status vocabulary and tier mapping.
- FO L0-L4 versus fast/full/robust vocabulary.
- Test fixture ownership where performance agents need semantically owned files.
