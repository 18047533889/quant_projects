# File Ownership

**STATUS: DRAFT / NOT FROZEN / IMPLEMENTATION NOT AUTHORIZED**

This is a proposed exact ownership map for later Wave 1 execution. It is not permission to create or modify package files. Evidence is the Wave 0 `LOCAL_REPO_SNAPSHOT` bound to `ddb03749b7ff85e63b633770c43dfbbb7562af19`; a later snapshot delta is mandatory because the repository changes concurrently. FA migration, consolidated boundary/matrix, independent review, and compatibility/extraction plans remain blockers.

## 1. Ownership Vocabulary

- `OWNER`: may author files in the exact glob after freeze.
- `REVIEW`: must review changes affecting its boundary but does not own the file.
- `LEAD`: lead architect/chief integrator only; shared contracts and final exports.
- `AUDIT`: read-only auditors; never edit production semantics.
- `CORPUS`: fixtures/references only; never imported by runtime.

Globs are repository-relative to `/home/shw/quant_projects`. No absolute filesystem path is used as a source import contract. A path matching more than one owner is a collision and is not implicitly shared.

## 2. Lead-Only and Shared Surfaces

| Owner | Exact repo-relative globs | Status/notes |
|---|---|---|
| `lead-architect` + `chief-integrator` | `quant_factor_platform_blueprint/DESIGN/**` | These draft artifacts only until freeze; do not edit existing Wave 0 reports. |
| `lead-architect` | `schemas/**` | Language-neutral schema source; no runtime package. |
| `chief-integrator` | `quant_evaluator/pyproject.toml`, `quant_evaluator/__init__.py`, `factor_optimizer/pyproject.toml`, `factor_optimizer/__init__.py`, `factor_assets/pyproject.toml`, `factor_assets/__init__.py`, `factor_preprocess/pyproject.toml`, `factor_preprocess/__init__.py` | Final package metadata and exports. Workers may submit a `CONTRACT_CHANGE_REQUEST`, never edit directly. |
| `chief-integrator` | root `pyproject.toml`, `setup.py`, `setup.cfg`, `tox.ini`, `noxfile.py`, `Makefile`, `requirements*.txt`, `.github/workflows/**`, `pytest.ini`, `.pre-commit-config.yaml` | Root build, CI, and cross-package integration. Existing DA/FE files are excluded from new ownership. |
| `chief-integrator` | `tests/cross_package/**`, `tests/extraction/**`, `tests/boundary/**` | Integration/extraction fixtures only. |
| `research-ledger-agent` | `research_control/**` | Proposed location; ownership remains unresolved until ledger decision. |

No new owner may modify `dataaccess/**`, `factor_engine/**`, `factor_layer/**`, `AutoFactorEvaluation-RECONSTRUCT/**`, `toolkit/**`, or corpus trees under this draft. Those are existing foundations, legacy, or corpus surfaces and are governed by their own source-of-truth rules.

## 3. QuantEvaluator Ownership

| Agent | Exact repo-relative globs | Review boundaries |
|---|---|---|
| `qe-runtime-agent` | `quant_evaluator/api/**`, `quant_evaluator/contracts/**`, `quant_evaluator/registry/**`, `quant_evaluator/planner/**`, `quant_evaluator/runtime/**` | QE semantic owner; lead approval for contract changes. |
| `qe-core-metrics-agent` | `quant_evaluator/metrics/quality.py`, `quant_evaluator/metrics/distribution.py`, `quant_evaluator/metrics/ic.py`, `quant_evaluator/metrics/ic_summary.py`, `quant_evaluator/metrics/quantile.py`, `quant_evaluator/metrics/turnover.py`, `quant_evaluator/metrics/probe_portfolio.py`, matching `quant_evaluator/tests/metrics/**` | Must preserve frozen MetricSpec semantics; math auditor review. |
| `qe-temporal-robustness-agent` | `quant_evaluator/metrics/temporal.py`, `quant_evaluator/metrics/exposure.py`, `quant_evaluator/metrics/robustness.py`, `quant_evaluator/metrics/multiple_testing.py`, `quant_evaluator/diagnosis/**`, matching `quant_evaluator/tests/temporal/**`, `quant_evaluator/tests/diagnosis/**` | A-share/leakage/math review required for timing and exposure changes. |
| `qe-performance-agent` | `quant_evaluator/kernels/**`, `quant_evaluator/benchmarks/**`, `quant_evaluator/tests/parity/**`, `quant_evaluator/tests/performance/**` | Cannot change MetricSpec, result schema, NaN/tie/timing semantics. |
| `qe-adapter-owner` | `quant_evaluator/adapters/**` | Optional DA/FE adapters; platform-boundary review. |
| `qe-test-owner` | `quant_evaluator/tests/contracts/**`, `quant_evaluator/tests/leakage/**`, `quant_evaluator/tests/integration/**`, `quant_evaluator/tests/extraction/**` | Lead coordinates cross-package fixtures; auditors remain read-only. |

`LabelBundle` is owned by `qe-runtime-agent` under `quant_evaluator/contracts/label_bundle.py`. QE cannot infer labels or duplicate DA PIT.

## 4. FactorPreprocess Ownership

| Agent | Exact repo-relative globs | Review boundaries |
|---|---|---|
| `fp-transform-agent` | `factor_preprocess/contracts/policy.py`, `factor_preprocess/contracts/context.py`, `factor_preprocess/registry/**`, `factor_preprocess/transforms/**`, `factor_preprocess/tests/transforms/**`, `factor_preprocess/tests/future_poison/**` | Must distinguish stateless/causal/fitted semantics; leakage review. |
| `fp-neutralization-agent` | `factor_preprocess/contracts/state.py`, `factor_preprocess/neutralization/**`, `factor_preprocess/tests/neutralization/**`, `factor_preprocess/tests/fold_boundary/**` | Owns `FittedState` implementation; math/leakage review. |
| `fp-representation-agent` | `factor_preprocess/contracts/feature_bundle.py`, `factor_preprocess/representation/**`, `factor_preprocess/adapters/factor_assets.py`, `factor_preprocess/tests/representation/**` | ModelInput/FeatureBundle decision remains unresolved; lead approval required. |
| `fp-performance-agent` | `factor_preprocess/kernels/**`, `factor_preprocess/benchmarks/**`, `factor_preprocess/tests/parity/**`, `factor_preprocess/tests/performance/**` | Cannot change transform semantics or fitted-state fields. |
| `fp-adapter-owner` | `factor_preprocess/adapters/data_access.py` | Optional context adapter only; no direct storage/PIT implementation. |

The performance agent may propose changes to transform/neutralization files but must not edit semantically owned files. It submits a `CONTRACT_CHANGE_REQUEST` or patch for the owning agent.

## 5. FactorOptimizer Ownership

| Agent | Exact repo-relative globs | Review boundaries |
|---|---|---|
| `fo-grammar-agent` | `factor_optimizer/contracts/candidate_mutation.py`, `factor_optimizer/contracts/search_budget.py`, `factor_optimizer/contracts/trial.py`, `factor_optimizer/grammar/**`, `factor_optimizer/complexity/**`, `factor_optimizer/seen/identity.py`, `factor_optimizer/tests/grammar/**`, `factor_optimizer/tests/adapters/**` | FE adapter is authoritative for legality/identity/complexity; no FE copy. |
| `fo-search-agent` | `factor_optimizer/search/**`, `factor_optimizer/policy/**`, `factor_optimizer/seen/cache.py`, `factor_optimizer/tests/search/**`, `factor_optimizer/tests/budgets/**`, `factor_optimizer/tests/lineage/**` | QE evidence and budget/lineage review; no admission policy. |
| `fo-llm-agent` | `factor_optimizer/llm/**`, `factor_optimizer/tests/llm/**` | Structured proposals only; no arbitrary Python or self-validation. |
| `fo-adapter-owner` | `factor_optimizer/adapters/**` | Optional FE/QE adapters; boundary review. |

FO is optional in the runtime flow. No FO owner may alter QE metric modules or FE operator code.

## 6. FactorAssets Ownership

| Agent | Exact repo-relative globs | Review boundaries |
|---|---|---|
| `fa-registry-identity-agent` | `factor_assets/contracts/asset.py`, `factor_assets/contracts/factor_set.py`, `factor_assets/contracts/evidence_ref.py`, `factor_assets/contracts/lifecycle.py`, `factor_assets/registry/**`, `factor_assets/identity/**`, `factor_assets/seen_index/**`, `factor_assets/tests/registry/**`, `factor_assets/tests/identity/**`, `factor_assets/tests/lifecycle/**`, `factor_assets/tests/no_raw_values/**` | FA migration and independent review are blockers; no raw values. |
| `fa-novelty-agent` | `factor_assets/novelty/**`, `factor_assets/adapters/quant_evaluator.py`, `factor_assets/tests/novelty/**` | QE/provider protocol owner review; no metric reimplementation. |
| `fa-graph-cluster-agent` | `factor_assets/similarity/**`, `factor_assets/graph/**`, `factor_assets/clustering/**`, `factor_assets/tests/graph/**`, `factor_assets/tests/similarity/**`, `factor_assets/benchmarks/**` | ANN is shortlist only; exact evidence through QE. |
| `fa-aggregation-agent` | `factor_assets/selection/**`, `factor_assets/aggregation/**`, `factor_assets/tests/selection/**`, `factor_assets/tests/aggregation/**` | Hard gates/evidence/Pareto; no universal score. |
| `fa-adapter-owner` | `factor_assets/adapters/factor_engine.py`, `factor_assets/adapters/data_access.py` | FE identity and DA public `get_store()`/factor reads only; platform review. |

FA ownership is intentionally marked conditional: no implementation until its migration matrix, value boundary, and independent review close.

## 7. Corpus and Reference Fixtures

| Owner | Exact repo-relative globs | Rule |
|---|---|---|
| `corpus-regression-auditor` | `tests/corpus/**`, `quant_evaluator/tests/corpus/**`, `factor_optimizer/tests/corpus/**`, `factor_assets/tests/corpus/**`, `factor_preprocess/tests/corpus/**` | Read/fixture ownership only; no runtime imports. |
| corpus adapter owners | Package-local `*/adapters/corpus/**` and matching package tests | Preserve source path, digest/commit, source ID, license status, original formula, translation/approximation notes, unsupported semantics. |
| `license-auditor` | `tests/license/**`, `docs/provenance/**` | Audit-only; cannot promote unknown-license material. |

Canonical external corpus directories (`gtja191`, `week2_pv_factors`, `factor_cold_start`, LQTP, raw-data layer, EvoAlpha, CogAlpha, external logs) remain outside runtime ownership. Their current license status is unknown or corpus-only as recorded in `CORPUS_INVENTORY.md`.

## 8. Read-Only Auditors

The following agents may not `Write` or `Edit` any production package or shared contract:

```text
math-auditor
leakage-auditor
ashare-semantics-auditor
boundary-auditor
platform-boundary-auditor
performance-auditor
simplification-auditor
extraction-auditor
license-auditor
corpus-regression-auditor
```

They issue evidence-backed findings with file, symbol, severity, reproduction, expected/observed behavior, and closure requirement. They do not self-close findings.

## 9. Collision Protocol

A collision exists when two agents need the same exact file, when a requested edit crosses an ownership glob, or when a worker discovers a required shared-contract change.

1. Stop editing the colliding file immediately; do not use `git checkout`, `restore`, `stash`, `clean`, or overwrite another worker's changes.
2. Record `CONTRACT_CHANGE_REQUEST` in the agent message/coordination channel, not a new report file. Include: requested path/symbol, current owner, reason, contract impact, affected packages/tests, compatibility/migration plan, and evidence refs.
3. The lead architect determines whether the change is a contract change, a handoff, or an adapter-local fix. The chief integrator records the decision in the next approved design revision.
4. If a contract changes, all affected owners re-read the revised schema and re-run their boundary/parity tests before continuing.
5. Shared files remain lead-owned; no “temporary” direct edits are allowed.
6. If a generated file is involved, edit its authoritative source and regenerate under lead control; never hand-edit generated snapshots.

## 10. Handoff Protocol

A patch handoff must include: exact globs changed, public symbols changed, schema/metric/transform/mutation versions, tests run, benchmark impact, legacy import scan, optional dependency behavior, and unresolved findings. A worker may not claim completion while a required independent audit is pending.

## 11. Explicit Exclusions

This map grants no ownership over DA, FE, legacy runtime, settings, `CLAUDE.md`, or existing Wave 0 reports. In particular, new code must not import legacy packages, use `sys.path` parent injection, call FE cleaned kernels directly, infer DA universe/PIT behavior, or make corpus material a runtime dependency.

## 12. Unresolved Ownership Decisions

- Final tie-break authority between lead architect and chief integrator.
- Whether `research_control/**` belongs in this repository or another controlled project.
- Owner of the final ModelInput contract and terminal handoff.
- Whether `FeatureBundle` or `ModelInputBundle` is canonical.
- Owner of generated package-local schema types.
- Exact ownership of shared cross-package integration fixtures if they need package-specific additions.
- Whether `factor_assets/benchmarks/**` is graph-cluster-owned or a separate performance owner.
- Exact CI checker and allowlisted imports.
- FA migration owner and independent-review schedule.

**Disposition:** draft only; no implementation is authorized.
