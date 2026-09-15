# FA / FO second-pass bug hunt — 2026-09-12

Base HEAD: `e94ac507d670fd1c16b1d6a63fc5d6286daa5970`  
Working tree: `/home/sunhaiwei/quant_projects`  
Scope: `factor_assets/**`, `factor_optimizer/**` only.

No branch, worktree, repository copy, commit, push, deployment, production publication, or formal-data deletion was performed. Existing edits were preserved.

## FA-policy-02 — mutable diagnosis policy singleton

Before the fix, `DiagnosisPolicy.__post_init__` copied its three rule mappings into ordinary dictionaries. The frozen dataclass therefore remained mutable. A direct counterexample changed the global `INTEGRITY_FAILURE` severity from `CRITICAL` to `UNKNOWN` under the same policy id/version.

Fix: copy and wrap severity, repairability and confidence mappings in `MappingProxyType`; normalize confidence values to finite policy scalars before exposure.

Code: `factor_assets/profiling/policies.py::DiagnosisPolicy.__post_init__`  
Regression: `factor_assets/tests/profiling/test_policies.py::TestDiagnosisPolicyRegistry::test_policy_mappings_are_deeply_immutable_and_detached`

## FO-checkpoint-02 — shared config alias bypassed execution-plan binding

Before the fix, `SearchRunner.resume` compared the session and runner config dictionaries but did not compare the session config with its saved `execution_plan_hash`. When session and runner shared the same mutable `SearchConfig` object, mutating it after session creation changed both sides of the equality and resume accepted the altered plan. The counterexample changed `plateau_threshold` to `0.5` and resumed successfully while the stored plan hash still described the old plan.

Fix: resume now recomputes the session execution-plan hash at its first boundary and rejects any drift before budget, strategy or evaluation work.

Code: `factor_optimizer/factor_optimizer/search/runner.py::SearchRunner.resume`  
Regression: `factor_optimizer/tests/search/test_execution_plan_checkpoint_v3.py::test_resume_rejects_mutated_config_shared_by_session_and_runner`

## FA-policy-03 — mutable taxonomy display policy

Before the fix, `TaxonomyPolicy` normalized its main vocabularies but left `display_family_tokens` and `display_family_label_groups` caller-owned and mutable. The process-global current policy accepted an in-place rewrite of the `PRICE` group to `UNKNOWN` without changing policy id/version.

Fix: detach and tuple-normalize display tokens, detach every nested group into a tuple, and expose the mapping through `MappingProxyType`.

Code: `factor_assets/profiling/policies.py::TaxonomyPolicy.__post_init__`  
Regression: `factor_assets/tests/profiling/test_policies.py::test_taxonomy_policy_display_groups_are_deeply_immutable_and_detached`

## FO-evidence-03 — evidence lookup trusted an unverified store payload

Before the fix, the QE adapter computed `content_identity` on write but `get_evidence` returned whatever mapping the store supplied. A corrupted or mis-keyed durable store could change metric contents while retaining the old identity, or return another evaluation under the requested key.

Fix: centralize canonical content-identity calculation; on read require a mapping, exact requested evaluation-id binding, and a recomputed identity match.

Code: `factor_optimizer/factor_optimizer/adapters/quant_evaluator.py::ConcreteQEAdapter._content_identity/get_evidence`  
Regression: `factor_optimizer/tests/search/test_qe_adapter_v3.py::test_evidence_lookup_rejects_tampered_content_and_cross_key_payloads`

## FA-policy-04 — mutable module policy registries

Before the fix, all three policy catalogs were ordinary dictionaries despite being documented as read-only frozen catalogs. A caller could replace a current version tuple and redirect later `get_*_policy` calls process-wide.

Fix: expose taxonomy, health and diagnosis catalogs through `MappingProxyType`; version collections were already tuples.

Code: `factor_assets/profiling/policies.py::{TAXONOMY_POLICIES,FACTOR_HEALTH_POLICIES,DIAGNOSIS_POLICIES}`  
Regression: `factor_assets/tests/profiling/test_policies.py::test_module_policy_registries_are_read_only`

## Verification

- Focused combined regression: `102 passed in 1.02s`.
- Individual policy suite after fix: `32 passed in 0.12s`.
- Individual QE adapter suite after fix: `9 passed in 1.39s`.
- Checkpoint plus budget-resume suites after fix: `13 passed in 0.19s`.
- Full FactorOptimizer suite: `887 passed, 12 warnings in 18.73s`.
- FactorAssets suite excluding the three known FE-bootstrap-dependent test files: `1490 passed, 44 warnings in 4.93s`.
- Taxonomy plus policy regression after the additional freeze: `79 passed in 0.16s`.
- Policy, taxonomy and diagnosis suites after registry freeze: `126 passed in 0.18s`.
- `git diff --check` over owned files and this evidence: pass.

## Remaining boundaries

- FO production capability remains truthfully `research_only`; no production status was changed.
- The adapter verifies the evidence snapshot returned by its store, but a production store still requires its own durability, access-control and atomicity guarantees.
- SearchConfig remains mutable for compatibility, but checkpoint and resume boundaries now detect execution-plan drift. In-process callers that never cross either boundary remain responsible for not mutating an active runner configuration; converting the public configuration graph to immutable values is a larger compatibility migration.
