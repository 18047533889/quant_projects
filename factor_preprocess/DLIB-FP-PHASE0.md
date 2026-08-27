# DLIB-FP Phase 0 standards — factor_preprocess library layer
**DLIB-FP Phase0 delivery: bounded mediation pathway + standards-locked conventions, with the following expectations made explicit so the factory stage does not over-reach.**

## Boundaries honored (factory-expectation guardrails)
1. Elasticity: FP set to MODERATE. Bounded-response allowed. Not strict boutique-deploy.
2. Registry return_shape: TransformRegistry CAN return high-dimensional meta bursts (metadata dashes swapped per legal use).
3. Compose harmonics: FP per-library rules only — no cross-package FE orchestrations.
4. Priority codings: catalog-backed ordering + depth-first. Hashes remain canonical on the `_FE_DSL_SEMANTIC_MAP` composure without formula ID churn.
5. Factor intake ordering: pinned null ordering to fixed scores.
6. Prefetch/depth: in-package latency fairness + bounded prefetch depth.
7. Compactness/granularity: fine-medium grain, catalog gold record. Dependency shape matches `.claude/skills`.
8. Parameterized aliases stay frozen-dataclass; authorized budget bookkeeping from `NeutralizationSpec`.

## Canonical public API (DLIB-FP-009)
Top-level `factor_preprocess` now exposes the deep-immutable, registry-based contracts. `PreprocessingPolicy`/`TransformSpec` remain a *deprecated compatibility view*.

Key exported names: `TreatmentRecipe`, `RecipeStep`, `FitBoundary`, `TransformLineage`, `TransformSemanticID`, `TransformStage`, `ExistingTreatmentSignature`/`Status`, `build_signature_from_lineage`, `PolicyPreset`/`PolicyRegistry`/`PolicyLevel`/`TransformStep`, `TransformRegistry`/`TransformMetadata`/`TransformCategory`, `FactorProfileArtifact`, `NeutralizationDiagnostics`, `NeutralizationSpec`, `FeatureBundle`, `FittedState`.

## Single-transform-authority (DLIB-FP-002)
- The eligibility engine has **no second catalog**. It consults `EligibilityRuleRegistry` for family -> allowed *treatment families*, maps families to canonical transform ids, and validates every proposal against `TransformRegistry`.
- `industry_neutral`/`size_neutral`/`dual_neutral` bind to the `ols_neutralize` kernel (single canonical neutralization implementation). `event_decay` and `freshness_aware_fill` are real executable transforms registered in the catalog.
- No dangling transform id can be returned: a proposal that does not resolve in the canonical registry is dropped.

## Deep immutability (DLIB-FP-026 / FP-003)
`_deep_freeze` normalizes dict/list/set into `_FrozenDict`/tuple/frozenset (deep-copy + pickle-safe). Applied to:
- `TransformStep` parameters, `TransformLineage` steps, `RecipeStep` parameters, `FactorProfileArtifact` nested dicts, `TransformMetadata` semantic surface, `TreatmentSearchSpace` transform/param map, `PolicyPreset.steps`/`tags`.
- `TreatmentRecipe` is fully content-addressed and deeply hash-safe.

## Lineage fail-safe (DLIB-FP-004)
`build_signature_from_lineage`: any semantic step the FP map does not recognize => `status = UNKNOWN` => `is_unknown_or_incomplete` blocks silent "untreated" routing.

## Rich profile descriptive fields (DLIB-FP-006)
`FactorProfileArtifact` gained non-leaking descriptive fields (`raw_turnover`, `half_life`, `sparsity`, `missingness`, `freshness`, `outlier_rate`, `cross_section_cardinality`, `sign_stability`, `scale_drift`, `industry_exposure`, `size_exposure`, `tail_concentration`, `event_semantics`, `preexisting_treatments`) — all participate in the content hash.

## Self-describing registry metadata (DLIB-FP-015)
`TransformRegistry.register` accepts an extended semantic surface; `TransformRegistry.enrich(name, **fields)` attaches semantics post-registration without changing signature/implementation/numeric-policy hashes. The semantic surface (`semantic_id`, `stage`, `family_tags`, `causality_class`, `requires_fit`, `parameter_domain`, `numeric_policy`, `cost_class`, `output_channels`) is populated for every catalog entry.

## Neutralization audit artifact (DLIB-FP-020/021)
`NeutralizationDiagnostics` — deep-immutable, content-hashed audit record (effective_n, rank, condition_number, R², residual_variance, missing_exposure_count, industry_coverage, solver, regularization, warnings, rank_deficient_resolution). `NeutralizationSpec` remains the semantic recipe, kernels canonicalized on `ols_neutralize`.

## Registry entry points
- `factor_preprocess/registry/transforms.py`: `TransformRegistry`, `TransformMetadata`, `TransformCategory`, `ALL_FAMILY_TAGS`, `create_default_registry`, `get_default_registry`, `TransformRegistry.enrich()`.
- `factor_preprocess/registry/policies.py`: `PolicyPreset`, `PolicyRegistry`, `PolicyLevel`, `TransformStep` (deep-frozen, content-hashable).
- `factor_preprocess/eligibility/rules.py`: `EligibilityRuleRegistry`, `FamilyRule`, `TreatmentFamily`, `FactorFamily`, `create_default_eligibility_rules`.
- `factor_preprocess/eligibility/engine.py`: `TreatmentEligibilityEngine` (single-transform-authority), `TreatmentSearchSpace` (deep-frozen).
- `factor_preprocess/contracts/treatment_recipe.py`: `TreatmentRecipe`, `RecipeStep`, `FitBoundary`.
- `factor_preprocess/contracts/treatment_lineage.py`: `TransformStage`, `TransformSemanticID`, `TransformLineage`, `build_signature_from_lineage`, `ExistingTreatmentSignature`.
- `factor_preprocess/contracts/_deep_freeze.py`: `deep_freeze`, `_FrozenDict`, `as_plain`.
- `factor_preprocess/contracts/factor_profile.py`: `FactorProfileArtifact` (rich descriptive surface).
- `factor_preprocess/neutralization/diagnostics_artifact.py`: `NeutralizationDiagnostics`.
- `factor_preprocess/transforms/event_decay.py`, `factor_preprocess/transforms/treatment_variants.py`.