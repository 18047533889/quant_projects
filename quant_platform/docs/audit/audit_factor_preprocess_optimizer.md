# Audit: factor_preprocess + factor_optimizer (Phase-0)

- Date: 2026-08-26
- git HEAD: `d58eae0821da2c71182e200751af97a577e1a616`
- Status: COMPLETE
- Mode: READ-ONLY audit (no source modified, no install)
- Baselines (as provided): FP `cd factor_preprocess && ../.venv/bin/python -m pytest tests -q` = 103 passed 1 xfailed; FO `cd factor_optimizer && ../.venv/bin/python -m pytest tests -q` = 505 passed
- Master spec: `QUANT_RESEARCH_PLATFORM_MASTER_IMPLEMENTATION_SPEC_20260826.md`
  - §34 FactorPreprocess 改造要求 = line 2324
  - §35 FactorOptimizer 改造要求 = line 2367
  - §13 / Stage 7 Treatment Search = line 1112
  - §42 幂等性设计 = line 2628
- Repo root: `/home/sunhaiwei/quant_projects`

---

## 1. FP Policy Authority — single authority confirmed

**Canonical (single authority):** `factor_preprocess/factor_preprocess/registry/policies.py`
- `TransformStep` (line 32) — carries `step_id`; `__post_init__` defaults `step_id=name` for back-compat (lines 44–49).
- `PolicyPreset` (line 53) — `validate()` enforces `Duplicate step_id in pipeline` (lines 147–149) and production ⇒ `causal_safe=True` (lines 141–142).
- `PolicyRegistry` (line 154), `create_default_policies()` (line 212), `get_default_policy_registry()` (line 415).
- `policy_identity` (lines 78–104) — content-derived sha256 over ordered step_ids + transform impl/numeric hashes + params + skip policy (§42 idempotency component for policy).

**Deprecated compat view (NOT authority):** `factor_preprocess/factor_preprocess/contracts/policy.py`
- Module docstring (lines 1–9) states verbatim: `PreprocessingPolicy` / `TransformSpec` are retained as a **deprecated compatibility view**; single policy authority is `registry.policies`.
- `TransformSpec` (line 30), `PreprocessingPolicy` (line 61) — frozen dataclasses; no `step_id` field, no duplicate-name enforcement (comment lines 88–90 defers to canonical view).

**Verdict:** Authority conflict is resolved; old schema is compat-only. Tests lock this: `tests/contracts/test_policy_unification.py:18` `test_duplicate_step_ids_rejected`, `:25` `test_step_id_uniqueness_enforced`.

---

## 2. FP Contracts inventory — canonical vs compat / missing

| Class / artifact | Location (file:line) | Status |
|---|---|---|
| `FittedState` | `contracts/state.py:114` | canonical |
| `FeatureBundle` | `contracts/feature_bundle.py:390` | canonical |
| `FeatureManifest` | `contracts/feature_bundle.py:82` | canonical |
| `AxisRef` / `ChannelRef` | `contracts/feature_bundle.py:38 / 66` | canonical |
| `FactorProfileArtifact` (content_hash, with_lineage) | `contracts/factor_profile.py:50 / 123 / 152 / 172` | canonical |
| `TransformStage` / `TransformSemanticID` / `TransformLineage` / `ExistingTreatmentSignature` / `map_fe_dsl_to_semantic` | `contracts/treatment_lineage.py:17 / 34 / 98 / 142 / 203` | canonical |
| `TransformSpec` / `PreprocessingPolicy` | `contracts/policy.py:30 / 61` | deprecated compat only |
| `TreatmentRecipe` | **NOT FOUND** in `factor_preprocess/factor_preprocess/` (grep across repo also returns nothing; closest analog is `winner_recipe: Mapping` in `factor_assets/contracts/treatment_selection.py:250`) | **MISSING** |
| Causality certification module (e.g. `CausalityCertificate`) | **NOT FOUND** — no dedicated module in `contracts/`, `transforms/`, or `adapters/` | **MISSING** (see §3 for partial mechanisms) |
| Adapters | `adapters/factor_assets.py`, `adapters/data_access.py` (+ `_data_access_impl.py`) — optional, Protocol-based | canonical, no TreatmentRecipe/Causality wiring |

---

## 3. FP Causal vs train-fitted classification

**Transform registry admission control** — `registry/transforms.py`:
- `TransformRegistry.register` (line 157) validates `admission ∈ {PRODUCTION, OFFLINE_ONLY, RESEARCH_ONLY}` (lines 197–198) and rejects `admission=="PRODUCTION" and not causal_safe` (lines 199–200). `TransformMetadata.causal_safe` (line 63).
- `list_causal_safe()` (line 279), production validation (line 260).

**Full-sample / non-causal transforms fail closed — hp_filter is OFFLINE_ONLY:**
- `transforms/decomposition/trend.py:2–11` — header states `STATUS: OFFLINE_ONLY / RESEARCH ONLY`, HP filter "over a prefix does not reproduce the first outputs".
- `registry/transforms.py:583–607` — comment (583–589) documents HP filter solves the HP objective over the WHOLE lagged sample (`spsolve`), is NOT prefix-invariant, NOT production-causal; loop registers `bandpass_filter`, `extract_cycle`, `christiano_fitzgerald_filter`, `wavelet_decompose`, `wavelet_smooth`, `wavelet_denoise`, `hp_filter`, `hp_decompose` all with `admission="OFFLINE_ONLY"`, `causal_safe=False`.
- Other fail-closed registrations: `impute_with_fallback` RESEARCH_ONLY / causal_safe=False (lines 517–524); `detect_correlation_regime` RESEARCH_ONLY (lines 528–535).

**Causal smoothing family** — `transforms/smoothing.py`:
- Module docstring (lines 1–15): all smoothers operate on `<= t-1` info via `shift(1)`, per-asset isolated, prefix-invariant, NaN warmup.
- `trailing_sma` (30), `trailing_median` (85), `robust_ewma` (140), `kama` (207), `one_sided_iir_lowpass` (341), `kalman_local_level` (403). No in-module registry table; all registered causal_safe=True in `registry/transforms.py` (e.g. `trailing_sma` 455, `robust_ewma` 438, `ewma` 415, rolling family 394/401/408).

**Eligibility / search-space rules** — `eligibility/engine.py`:
- `TreatmentSearchSpace` (80), `TreatmentEligibilityEngine.build_search_space` (112) — family-driven (PRICE_VOLUME/HIGH_TURNOVER/FUNDAMENTAL/SPARSE_UPDATE/EVENT/BINARY/DISCRETE, lines 26–32), smoothing budget halved when already smoothed (127–132), low-turnover trims smoothing (177–184), already-industry-neutral prunes neutralization (186–190), RAW always a candidate (124–125).
- Transform catalog constants at `eligibility/engine.py:38,48,53,58,63,66` (`SMOOTHING_TRANSFORMS`, `EVENT_DECAY_TRANSFORMS`, `FRESHNESS_FILL_TRANSFORMS`, `WINSOR`, `RANK`, `ZSCORE`) — a second, separate transform catalog from `registry/transforms.py` (see §10).

**Grammar / certified orderings** — `grammar/search_grammar.py`:
- `Stage` (22), `STAGE_ORDER` (34), `CausalityClass` (42: CAUSAL / CROSS_SECTIONAL / NO_OP), `OrderTemplate` (51), `CERTIFIED_TEMPLATES` (74), `is_certified_template` (123), `validate_stage_order` (131).

**Neutralization** — `neutralization/spec.py`:
- `NeutralizationSpec` (84), `is_production_admissible` (160), `is_research_only` (165), `kernel_name` (169). Methods (19–57) split into `_PRODUCTION_METHODS` / `_RESEARCH_METHODS`.

---

## 4. FP step_id uniqueness + prefix-invariance property tests

- **step_id uniqueness rule:** `registry/policies.py:147–149` — duplicate `step_id` rejected at `PolicyPreset.validate`; `TransformStep.__post_init__` (44–49) back-compat default `step_id=name` so old name-only callers still pass while uniqueness is enforced on `step_id`.
- **Tests:** `tests/contracts/test_policy_unification.py:18` (duplicate step_ids rejected), `:25` (uniqueness enforced).
- **Prefix-invariance property tests:**
  - `tests/transforms/test_hp_prefix_invariance.py` — `test_trailing_sma_is_prefix_invariant` (50, PASS); hp_filter NOT prefix-invariant (failing RED test, module docstring 1–16; this is the xfailed test in the 103 passed 1 xfailed baseline).
  - `tests/contracts/test_leakage_properties.py` — `test_causal_prefix_invariance` (67), `test_split_fit_isolation_per_row` (96), `test_split_fit_isolation_windowed` (112), `test_past_only_invariance` (156).
  - `tests/transforms/test_smoothing_kama_oracle.py` — KAMA oracle.
- Fit/Apply boundary: exercised via `tests/regime/test_regime_production.py` (`test_serialize_into_canonical_fitted_state` 46, `test_supervised_weight_must_not_be_used_in_plain_preprocess_policy` 122). Asset Isolation is covered structurally by per-asset grouping in smoothing/rolling (module docstrings) rather than a dedicated test file; no standalone `test_asset_isolation.py` exists.

---

## 5. FO Treatment search orchestration

- `search/runner.py` — `SearchConfig` (54, `require_evaluation_protocol: bool = True` at 71, fail-closed default; deprecated generic `evaluation_fn` escape hatch at 69, 933–941 warns DeprecationWarning), `SearchSession` (212, append-only `ledger: TrialLedger` at 236), `SearchRunner` (881), `_bind_capabilities` (1010), `run` (1029), `resume` (1040, with signed `BudgetExtensionAuthorization` growth checks 1066, 1136), `_run_session` (1169), `_validate_trial` (1430), `_check_plateau` (1495), `_evaluate_with_capability` (1519).
- There is no separate `search/search_runner.py` module; `search/runner.py` is the search runner (`search/` contents: `categorical_strategy.py`, `desirability.py`, `dimensions.py`, `lineage.py`, `multifidelity.py`, `pareto.py`, `plateau.py`, `runner.py`, `strategies.py`, `tiered_evaluation.py`, `winner_selector.py`).
- `contracts/trial_ledger.py` — `LedgerEntry` (39), `TrialLedger` (97), `append_proposal_failed` (126), `append_invalid_proposal` (129), `append_trial` (132), `status_counts` (145). Runner writes every outcome: PROPOSAL_FAILED (1201), INVALID_PROPOSAL (1210), DUPLICATE (1221), PROPOSED (1224), ILLEGAL (1230), EVALUATION_FAILED (1258–1259, 1300–1301, 1338).
- `contracts/splits.py` — `SplitPlan` (25: `purge`, `embargo`, `validation_embargo` at 53–55), `EvaluationProtocol` (62), `LabelBundle` (94), `SealedTestHandle` (140), `SealedTestResult` (159), `validate_split_plan` (180, purge/embargo/label-horizon leakage check 407–506), `create_split_aware_evaluation_fn` (548).
- **Capability-based DataProvider isolation:** `data_capabilities.py` — `DataCapability` (81, real-identity verification `verify_identity` 257, `can_evaluate` exact-subset 223), `TrainDataCapability` (291), `ValidationDataCapability` (298), `TestDataCapability` (305), `TestStoreRef` (328), `TestDatasetIdentity` (360), `TestAuthorityBroker` (384), `build_search_capabilities` (589). `data_providers.py` — `DataProvider` (53), `TrainDataProvider` (141), `ValidationDataProvider` (147), `TestDataProvider` (153), `ScopedEvaluator` (159). Runner search worker holds ONLY train+validation capabilities, no TEST capability (runner.py:1010–1028; tests `test_real_data_isolation.py:174`, `test_adversarial_isolation.py:103/110/122/130`).

---

## 6. FO sealed test / no test object / purge+embargo / multi-fidelity / failed-trial / multiple-testing

- **Sealed test:** `SearchSession.freeze_for_sealed_test` (runner.py:306, pins immutable masks 333–343), `consume_sealed_test` (352), `SealedTestExecutor` (1645, docstring: search-runner-created executor has NO `TestAuthorityBroker` so cannot resolve test data), `evaluate_sealed_test` (1671 — requires broker, issues TEST capability, provider gated on capability, replaces old direct `session.consume_sealed_test(...)` path per comment 1687–1690). Tests: `test_sealed_test_hardening.py`, `test_split_safety.py`, `test_test_authority_boundary.py`, `test_scoped_isolation.py`, `test_adversarial_isolation.py`.
- **No test object in callback:** `_evaluate_with_capability` (1519) verifies capability identity + exact-subset authorization before invoking the callback; sealed test route only via broker (1680–1683); test-data path never handed to search callback. Tests `test_real_data_isolation.py:117/174`, `test_adversarial_isolation.py:103–240` (forgery/altered-mask rejected).
- **Purge + embargo:** `contracts/splits.py:53–55` fields; fail-closed validation at 407–506 (`_validate_temporal_leakage`, `_count_foreign_inside` 366). Tests `test_split_safety.py`, `test_label_bundle_intervals.py`.
- **Multi-fidelity / tiered eval:** `search/multifidelity.py` (`FidelityTier` 8, `FidelitySpec` 26, `PromotionCriteria` 107, `MultiFidelityScheduler` 162); `search/tiered_evaluation.py` (`EvaluationTier` 26, `TieredEvaluationPolicy` 81, `TieredEvaluationScheduler` 169); runner wires `TieredEvaluationScheduler` when `config.tiered_evaluation` set (runner.py:92, 950–954). Tests `test_multifidelity.py`, `test_tiered_evaluation.py`.
- **Failed trial logging:** runner marks `TrialStatus.FAILED`/`EVALUATION_FAILED` with failure_reason and appends ledger entries (runner.py:1257–1259, 1299–1301, 1336–1338); `TrialStatus` (contracts/trial.py:17, FAILED 33, PROPOSAL_FAILED 36, EVALUATION_FAILED 38).
- **Multiple-testing counts:** **PARTIAL.** Ledger records every trial incl. failures via `status_counts` (trial_ledger.py:145–150), satisfying Stage 7 "failed trials 也记入" at the counting level; but no correction is applied inside FO — the Bonferroni/BH/Holm/Sidak implementations live in `quant_evaluator/metrics/multiple_testing.py` (bonferroni 67, BH 114, holm 195, sidak 278) and are NOT referenced anywhere in `factor_optimizer/factor_optimizer/` (grep: zero hits). No `multiple_testing` symbol in FO source or tests.

---

## 7. FO Desirability / dimension aggregation / winner selector + pareto (inventory)

- **Desirability soft floors:** `search/desirability.py` — `desirability_for` (19), `catastrophic_floor` (69, hard integrity floor distinct from soft floor), `DEFAULT_MAPS` anchor tables per metric (79+: rank_ic, icir, turnover, …), `Desirability` catalog (150).
- **Dimension aggregation:** `search/dimensions.py` — `bottleneck_min` (51), `geomean_floor` (58, geomean blended with minimum via `min_penalty`), `aggregate_dimension` (78, methods `min`/`geomean`/`bottleneck_geomean`/`low_quantile`).
- **Winner selector:** `search/winner_selector.py` — `WinnerPolicy` (31), `RobustBalancedUtility` (125: `U = alpha*min + beta*geomean + gamma*robustness - lambda*complexity`, docstring line 12), `augmented_tchebycheff` (172, present but NOT used by select_winner), `select_winner` (216: max utility, complexity tie-break 284, determinism via trial_id).
- **Pareto:** `search/pareto.py` — `ParetoPoint` (9), `ParetoFrontier` (62, dominates 34, hypervolume 201, coverage 251, spacing 271), `ParetoArchive` (326). Tests `test_desirability.py`, `test_pareto.py`, `test_winner_selector.py`.
- **Selection artifact:** `TreatmentSelectionArtifact` lives in `factor_assets/contracts/treatment_selection.py:216` (NOT in FO). Fields: `raw_baseline_evidence_ref`, `factor_profile_ref`, `eligibility_policy_ref`, `search_space_ref`, `all_trial_refs`, `pareto_candidate_refs`, `winner_recipe`, `winner_policy_identity`, `absolute/delta_metric_refs`, `dimension_scores`, `hard_gate_results`, `soft_floor_results`, `robustness_evidence`, `complexity_score`, `content_hash` (263). FO's `SearchRunner`/`select_winner` do NOT emit this artifact (grep: no reference from `factor_optimizer/`); it is consumed by `factor_assets/assembly/engine.py` (lazy import 667–680). `factor_assets/contracts/treatment_policy.py:24` `TreatmentPolicyRef`.

---

## 8. Mapping to spec

### §34 FactorPreprocess 改造要求 (spec line 2324)

| Spec item | Status | Location |
|---|---|---|
| TreatmentRecipe | **MISSING** | no class in FP or FO; only free-form `winner_recipe: Mapping` in factor_assets/contracts/treatment_selection.py:250 |
| FittedState | EXISTS | contracts/state.py:114 (state_id derive 257, is_compatible_with 274) |
| FeatureBundle | EXISTS | contracts/feature_bundle.py:390 |
| FeatureManifest | EXISTS | contracts/feature_bundle.py:82 |
| Causality certification | **PARTIAL** | causal_safe + admission gates registry/transforms.py:197–200, OFFLINE_ONLY registrations 590–607; CausalityClass grammar/search_grammar.py:42; no dedicated certification module/class |
| 统一 policy authority（旧 schema 只兼容） | EXISTS | registry/policies.py canonical; contracts/policy.py:1–9 deprecated compat |
| 唯一 step_id | EXISTS | registry/policies.py:147–149; tests/contracts/test_policy_unification.py:18/25 |
| property tests: Prefix Invariance / Asset Isolation / Fit/Apply Boundary | PARTIAL | tests/transforms/test_hp_prefix_invariance.py:50; tests/contracts/test_leakage_properties.py:67/96/112/156; Fit/Apply via tests/regime/test_regime_production.py:46/122; no standalone asset-isolation test file |
| 禁用 full-sample HP/STL/filtfilt/interpolation/未来感知 impute | EXISTS | registry/transforms.py:517–524, 528–535, 590–607 (OFFLINE_ONLY/RESEARCH_ONLY) |

### §35 FactorOptimizer 改造要求 (spec line 2367)

| Spec item | Status | Location |
|---|---|---|
| sealed test | EXISTS | runner.py:306/352/1645/1671; splits.py:140/159; tests/test_sealed_test_hardening.py |
| no test object in search callback | EXISTS | runner.py:1010–1028, 1519–1558, 1678–1683; data_capabilities.py:384+; tests/test_real_data_isolation.py, test_adversarial_isolation.py |
| purge + embargo | EXISTS | contracts/splits.py:53–55, 407–506 |
| trial ledger | EXISTS | contracts/trial_ledger.py:97; runner.py ledger writes 1201–1338 |
| multi-fidelity | EXISTS | search/multifidelity.py; search/tiered_evaluation.py; runner.py:92, 950–954 |
| failed trial logging | EXISTS | runner.py:1257–1301, 1336–1338; trial.py:33–38 |
| multiple testing (failed trials 计入 count) | **PARTIAL** | ledger status_counts trial_ledger.py:145 (counts all incl. failures) BUT no correction wiring in FO; corrections only in quant_evaluator/metrics/multiple_testing.py:67/114/195/278 — not imported by FO |

### §13 / Stage 7 Treatment Search (spec line 1112)

| Spec item | Status | Location |
|---|---|---|
| 可搜索变换（raw/winsor/robust scale/rank/EWMA/causal smoothing/neutralize/residualize） | EXISTS | eligibility/engine.py:38–66, build_search_space 112–196 |
| 硬规则：causal or train-fitted / 禁 full-sample leakage | EXISTS | registry/transforms.py:197–200, 590–607 |
| search trial ledger 全量保留 | EXISTS | trial_ledger.py; runner.py checkpoint round-trip 732–742 |
| failed trials 记入 multiple-testing count | PARTIAL | counted via status_counts but not consumed by any correction |
| 输出 TreatmentSelectionArtifact 记录 raw profile/candidate treatments/trial results/selection policy/winner/runner-ups/failed trials/fit boundary | **PARTIAL** | artifact exists factor_assets/contracts/treatment_selection.py:216 (winner_recipe, all_trial_refs, pareto_candidate_refs, content_hash) but no explicit `runner_ups` / `failed_trials` / `fit_boundary` fields; not emitted by FO runner |

### §42 Idempotency (spec line 2628)

| Spec item | Status | Location |
|---|---|---|
| Treatment idempotency key = hash(source_evidence, search_policy, split_plan) | **PARTIAL** | components exist: FactorProfileArtifact.content_hash (factor_profile.py:123/152), PolicyPreset.policy_identity (policies.py:78), SplitPlan.split_id (splits.py:25), TreatmentSelectionArtifact.content_hash (treatment_selection.py:263); no explicit composite key / CACHE_HIT semantics in FP or FO |

---

## 9. GAP list

| # | Requirement | Current (files) | Gap | Action | Reuse/Modify/Add | Priority |
|---|---|---|---|---|---|---|
| 1 | TreatmentRecipe class (§34/Stage 7) | Missing everywhere; only free-form `winner_recipe` dict (factor_assets/contracts/treatment_selection.py:250) | No typed contract for the winning treatment recipe (steps + step_ids + params + semantic ids) | Define canonical TreatmentRecipe (reuse TransformStep semantics); wire to eligibility + search space + artifact | Add | P0 |
| 2 | Causality certification module (§34) | causal_safe/admission flags (registry/transforms.py:63,197–200) + CausalityClass enum (grammar/search_grammar.py:42) | No explicit per-transform CausalityCertificate / certification record, no audit trail of WHY a transform is causal | Add certification contract (causal reason class: one-sided/rolling/fitted; evidence ref); certify smoothing + fitted transforms | Add | P0 |
| 3 | Multiple-testing count consumed (§35/Stage 7) | Ledger status_counts counts failures (trial_ledger.py:145); corrections in quant_evaluator/metrics/multiple_testing.py:67+ | FO never applies correction; count not exposed on search result | Feed trial/ledger counts into multiple_testing correction; surface adjusted verdict in artifact | Modify (FO) + Reuse (QE) | P1 |
| 4 | TreatmentSelectionArtifact fields runner-ups / failed trials / fit boundary (Stage 7) | artifact has all_trial_refs/pareto_candidate_refs/winner_recipe (treatment_selection.py:248–251) | Explicit runner-ups, failed-trial list, fit boundary not recorded | Add fields to TreatmentSelectionArtifact; populate from ledger + split_plan | Modify | P1 |
| 5 | FO emits TreatmentSelectionArtifact | Runner returns SearchSession only (runner.py:1428); artifact consumed by factor_assets/assembly/engine.py:667 | Search orchestration does not produce the Stage-7 output artifact | Add an artifact builder after winner selection (runner or new module) | Add | P1 |
| 6 | Idempotency key + CACHE_HIT for treatment (§42) | Content hashes exist (factor_profile.py:123, policies.py:78, splits.py:25, treatment_selection.py:263) | No composite treatment idempotency key; no cache-hit path | Compose hash(source_evidence, search_policy_identity, split_plan) and short-circuit CACHE_HIT | Add | P2 |
| 7 | Standalone Asset Isolation property test (§34) | Asset isolation by construction (smoothing/rolling per-asset groupby); leakage tests test_leakage_properties.py | No dedicated asset-isolation test file | Add property test that cross-asset recompute is identity | Add | P2 |
| 8 | Fit/Apply boundary contract test for fitted transforms | FittedState (state.py:114) + regime tests | No generic fit→apply boundary property test across fitted transforms | Add generic fit/apply test harness | Add | P2 |
| 9 | augmented_tchebycheff dead code | winner_selector.py:172 defined, never used by select_winner (216) | Second utility not wired | Either wire as alternative policy or remove/mark research-only | Modify | P3 |
| 10 | Multi-fidelity duplication | multifidelity.py (FidelityTier) vs tiered_evaluation.py (EvaluationTier) | Two tiered-eval stacks; runner uses only tiered_evaluation | Declare one canonical tiered-eval path; keep other as compat/inventory | Modify | P3 |

---

## 10. Second-authority duplicates (documented honestly, not resolved)

1. **FP policy schemas:** `PolicyPreset/TransformStep/PolicyRegistry` (`registry/policies.py:32/53/154`) vs `TransformSpec/PreprocessingPolicy` (`contracts/policy.py:30/61`) — resolved: deprecated compat view documented in contracts/policy.py:1–9; still two importable schemas.
2. **`TransformStep` name collision:** `registry/policies.py:32` (pipeline step with step_id) vs `contracts/treatment_lineage.py:74` (semantic lineage step with semantic_id) — same class name, different contracts in different modules; easy to import the wrong one.
3. **Transform catalog duplicated:** `eligibility/engine.py:38–66` constants (SMOOTHING/WINSOR/RANK/ZSCORE/FRESHNESS dicts) vs `registry/transforms.py` registered transforms — two sources of "which transforms exist/are searchable"; risk of drift.
4. **Sealed-test execution paths:** `SearchSession.consume_sealed_test` (runner.py:352) vs `SealedTestExecutor.evaluate_sealed_test` (runner.py:1671) — old direct path documented as replaced (comment 1687–1690) but still present in code.
5. **Multi-fidelity stacks:** `search/multifidelity.py` (FidelityTier/FidelitySpec/MultiFidelityScheduler) vs `search/tiered_evaluation.py` (EvaluationTier/TieredEvaluationPolicy/TieredEvaluationScheduler) — both complete, only tiered_evaluation wired into runner.
6. **Winner utility functions:** `RobustBalancedUtility` (winner_selector.py:125) vs `augmented_tchebycheff` (winner_selector.py:172) — only the former used by `select_winner`.
7. **Generic evaluation_fn escape hatch:** runner.py:69/933–941 — `require_evaluation_protocol=False` still permits a bare callable bypassing `EvaluationProtocol`/SplitPlan (deprecated, warned, but a second evaluation path when enabled).

---

## Appendix: test baselines (as provided, not re-run)

- FP: 103 passed 1 xfailed (`tests/transforms/test_hp_prefix_invariance.py` hp_filter non-prefix-invariant test is the xfail).
- FO: 505 passed across `tests/search/` (26 test modules, incl. sealed-test, adversarial isolation, split safety, real-data isolation, tiered/multifidelity, desirability, pareto, winner_selector, plateau, lineage, strategy, runner).
