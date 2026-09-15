# FA / FO V2 recheck evidence — 2026-09-12

## Scope and execution identity

- Repository: `/home/sunhaiwei/quant_projects` (the sole server-c working tree)
- Base HEAD tested: `e94ac507d670fd1c16b1d6a63fc5d6286daa5970`
- Python: `.venv/bin/python`
- Disk before work: 777 GiB available
- No branch, worktree, repository copy, commit, push, deployment, production publication, or formal-data deletion was performed.
- Existing FactorEngine changes were preserved. The combined first run was affected by an in-flight FactorEngine registry edit; see Remaining blockers.

## Delta found and fixed

FA-03 still had a material content-identity hole: `FactorHealthPolicy.policy_hash` included ids and a partial use-case view, but omitted the actual metric thresholds, dimension aggregation rules, default admission floors, hard gates, display bands, status, and several scope fields. Thus a threshold or hard-gate change could retain the same hash.

The hash payload now covers the full executable grading/admission contract with canonical JSON ordering. Construction also copies every caller-owned sequence into immutable tuples, including nested display-band rows and all three anchor tables, so later mutation of input lists cannot change an existing policy. Regressions prove both content sensitivity and external-container isolation.

Changed code: `factor_assets/profiling/policies.py` (`FactorHealthPolicy.policy_hash`).

Changed tests: `factor_assets/tests/profiling/test_policies.py::test_health_policy_hash_binds_actual_thresholds_and_admission_contract` and `test_health_policy_copies_all_caller_owned_sequence_containers`.

## Per-item current evidence

| Item | Current conclusion | Code evidence | Current executed test evidence | Remaining authority / limitation |
|---|---|---|---|---|
| FA-01 | VERIFIED code contract | `profiling/health_card.py::_integrity_results`, `build_health_card` | `test_health_card.py`, focused suite pass | Production trust still depends on resolver-backed upstream evidence; a string alone is not claimed as proof. |
| FA-02 | VERIFIED code contract | `health_card.py::FactorHealthCardArtifact.__post_init__`, `selection/decision.py::DecisionProvider` | health-card and V8 selection/integration tests pass | No production release executed. |
| FA-03 | FIXED_LOCAL and focused VERIFIED | `profiling/policies.py::FactorHealthPolicy.__post_init__`, `policy_hash` | content-hash and external-list-alias regressions pass | Historical cards need policy-hash migration/re-evaluation by release authority. |
| FA-04 | VERIFIED code contract | `profiling/dimensions.py::build_dimension_grade` | `test_dimensions.py` pass | Real unavailable metrics remain UNKNOWN/PARTIAL; they are not certified. |
| FA-05 | VERIFIED code contract | `dimensions.py::build_dimension_grade(shape_family=...)` | U/monotonic branch tests pass | QE shape evidence quality remains QE-owned. |
| FA-06 | VERIFIED fail-closed integration contract | `profiling/taxonomy.py`, `adapters/production_taxonomy.py` | taxonomy unit tests pass; FE-crossing tests blocked in combined run | Canonical field catalog and PIT authority remain DA/FE owned. |
| FA-07 | VERIFIED legacy fail-closed contract | `library/promotion_gate.py`, `selection/decision.py` | promotion and V8 decision tests pass | Production admission is resolver-backed DecisionProvider; legacy gate is not presented as final production authority. |
| FA-08 | VERIFIED code contract | `clustering/incremental.py` final-cluster evidence rebinding | incremental assignment tests pass | No production ClusterSet published. |
| FA-09 | VERIFIED code contract | `PairwiseEvidenceStatus`, `CertifiedPairwiseEvidence` | FA09 and incremental tests pass | ANN recall alone remains approximate and cannot certify uniqueness. |
| FA-10 | VERIFIED bounded implementation | `clustering/incremental.py` batch ANN and cluster support fields | incremental and cluster-quality tests pass | Persistent production index/refresh requires runtime store and real corpus benchmark. |
| FA-11 | VERIFIED code contract | `optimizer/pareto.py::compute_frontier`; FO canonical Pareto path | FA/FO Pareto tests pass | Deprecated FA optimizer remains compatibility-only; FO is selection math authority. |
| FO-01 | VERIFIED typed batch adapter contract | `adapters/quant_evaluator.py::ConcreteQEAdapter.evaluate` grouped evidence | `test_qe_adapter_v3.py`, V8 integration pass | No private-data QE batch was run. |
| FO-02 | VERIFIED parameter forwarding contract | same adapter; EvaluationRequest and evidence snapshot | adapter and V8 integration tests pass | Durable production evidence store and trusted split authority not configured here. |
| FO-03 | BLOCKED as designed | `capabilities.py::PRODUCTION_CAPABILITY` | production fail-closed tests pass | Capability remains `research_only`; blockers are truthful and were not toggled. |
| FO-04 | VERIFIED deterministic staged contract | `search/runner.py`, `search/multifidelity.py` | stage trace, multi-fidelity, V8 integration tests pass | `max_concurrency > 1` remains explicitly unsupported; no real GPU/private-data benchmark. |
| FO-05 | VERIFIED checkpoint binding contract | `search/runner.py::SearchConfig/SearchSession.checkpoint/from_dict` | execution-plan checkpoint and remaining-gap tests pass | Campaign durability is locally tested only; no external distributed reservation authority exercised. |
| V-02 | PARTIAL / fail-closed | FA taxonomy/health builders and V8 integration refresh artifacts | relevant focused tests pass | No live candidate-ingestion service was exercised, so per-arrival production regeneration is not certified. |
| V-03 | VERIFIED code policy layers | metric grading → dimension grading → use-case floors → DecisionProvider | profiling, decision and V8 integration tests pass | Policy remains calibration-required; not a production performance certification. |
| V-04 | VERIFIED bounded repair contracts | FO diagnosis routing/repair registry and FA lineage artifacts | existing routing/lineage tests pass in package suite | No real non-mock parent/child small-batch search. |
| V-05 | VERIFIED code guard | `tests/search/test_v05_cheap_prefilter.py` and routing policy | focused test pass | Statistical usefulness on real sparse/nonlinear factors not claimed. |
| V-09 | VERIFIED bounded routing/budget contract | diagnosis routing, conditional search, SearchBudget ledger | package and focused tests pass | Atomic multi-process reservation remains outside current serialized runner. |
| V-10 | VERIFIED identity separation contract | tiered evaluation artifacts and stage execution trace | stage trace and V8 integration tests pass | No claimed GPU final evidence. |
| V-11 | VERIFIED local durable governance contract | campaign store, TestAuthorityBroker, sealed state and multiplicity ledger | sealed/campaign/V8 tests pass in package suite | Production external authority/storage integration not exercised. |
| V-12 | PARTIAL | FO split/label interval contracts reject overlaps and test fitting | package split and label tests pass | Real DA PIT snapshots, revised filings and calendars were not available to this scope. |
| V-14 | VERIFIED layered code contract | exact identity, ANN candidate retrieval, certified pairwise evidence | similarity, FA09 and clustering tests pass | Full-corpus recall/false-negative bound requires real corpus. |
| V-15 | VERIFIED versioned lifecycle contract | cluster lineage and refresh service | lifecycle/lineage/cluster tests pass | No production global refresh or model retrain. |
| V-16 | VERIFIED selection contract | FA `selection/decision.py`, FO winner-set/treatment decision | V8 integration and package selection tests pass | OOS incremental utility requires trusted QE/model evidence. |
| V-17 | VERIFIED asset identity contract | `aggregation/specs.py`, `aggregation/composite.py` | aggregation/composite tests pass | No production composite published. |
| V-18 | VERIFIED FA/FO identity chain portion | evidence refs, treatment results, selection bindings | V8 cross-layer/integration tests pass | Full FE/DA/QE/model chain remains cross-owner work. |
| V-21 | PARTIAL | health/evaluation identities are immutable and historical refs retained | profiling/registry tests pass | No live matured post-release sample monitor was exercised. |
| V-22 | VERIFIED fail-closed proposal boundary | FO grammar validation/repair whitelist/budget and research-only capability | proposal/strategy/production-validator package tests pass | Agent suggestions remain non-authoritative by design. |

## Commands and observed results

1. Combined package run before the local delta: `2374 passed, 17 failed, 56 warnings in 34.62s`. All 17 failures had the same external FactorEngine bootstrap cause: an in-flight `factor_engine/cleaned_operators/registry.py` change rejected `cs_huber_resid` with `_calculate_series has no certifiable implementation identity`. The failing FA tests were FE adapter/taxonomy integration tests; no FA/FO-owned assertion failed.
2. Focused acceptance run after the delta: `187 passed, 3 warnings in 1.73s`.
3. Full FA+FO rerun after the delta: `2375 passed, 17 failed, 56 warnings in 38.92s`; the same 17 FE-bootstrap-dependent tests failed and all FO tests plus all non-FE-dependent FA tests passed.
4. `git diff --check -- factor_assets factor_optimizer`: pass.
5. FA-03 policy regression after deep-freeze fix: `31 passed in 0.11s`.

Changed-file SHA-256 after testing: `factor_assets/profiling/policies.py` = `e8c3fefe8c223bc9766e35d10d5ff794b02bd51d48f63fb730aef5d4c2a11903`; `factor_assets/tests/profiling/test_policies.py` = `c190ceebbe0cf1f36e0cfa085a7ae1fa08efe62f4b454d556f3117a8b78424e5`.

Focused command covered policy, health-card, dimensions, legacy promotion, incremental clustering, FA Pareto, QE adapter, checkpoint identity, stage trace, cheap prefilter, V8 FA/FO/QE integration, and remaining acceptance gaps.

## Remaining blockers

- FO production capability intentionally remains fail-closed (`research_only`). No private market data, production split authority, durable production evidence store, GPU benchmark, model validation, or publication authority was available in this task.
- The combined all-package green run must be repeated after the independent FactorEngine registry edit stabilizes. The initial red result must not be represented as an FA/FO production pass.
- Policy hash semantics changed. Any persisted artifact keyed by the prior incomplete hash needs an explicit migration/invalidation decision; this task did not rewrite historical artifacts.
- V-02, V-12, V-21 and the production portions of the other V items require live DA/QE/platform/model authority and real snapshots. They remain partial or blocked rather than being overstated.
