# Model-Operators Full Audit — Phase-0 status (current HEAD)

> Baseline per taskbook `FactorEngine_Model_Operators_Full_Audit_and_Remediation_20260811.md`
> Current HEAD: `5f44df63` (after R47 concurrent commit)
> Inventory probe: 1483 canonicals, ~296 model-like (category-aware), 63 explicit timing / 163 missing

## Inventory (probe, current HEAD)

| measure | value |
|---|---|
| total canonicals | 1483 |
| model-like (name-only `is_model_like_name`) | 226 |
| model-like explicit `MODEL_TIMING_CONTRACTS` | 63 |
| missing explicit timing | 163 |
| lanes: DIAGNOSTIC_RESEARCH | 211 |
| lanes: FAST_NATIVE_ALPHA | 51 |
| lanes: EXPENSIVE_CERTIFIED_ALPHA | 29 |
| lanes: MODEL_FEATURE_SCORE | 5 |
| unclassified lane errors | 0 |
| `MODEL_LANE_EXPLICIT` keys not in registry | 10 (`ts_dynamic_knn_residual`, `ts_dynamic_knn_state`, `ts_hsic_dependence`, `ts_kernel_granger_causality`, `ts_kernel_granger_oos`, `ts_lyapunov_exponent`, `ts_markov_regime_state`, `ts_markov_transition_probability`, `ts_regime_conditional_moment`, `ts_state_space_regime_probability`) |
| production timing errors (model-like w/o explicit) | 163 |

## M-issue status (survey: PCA/PCR/regression/AR — completed)

| issue | status | evidence |
|---|---|---|
| M-020 PCA min/full history | PARTIAL | coverage gates exist (panel_model.py:45-48, pca_state.py:41-44) but NO requested_window/min_history/warmup_policy param and NO row<window-1→NaN floor (_rolling_pca panel_model.py:171-190; probe: 5-row window=10 fit_lag=0 returns finite) |
| M-021 PCA rank policy reused for PCR | OPEN | single clamp k=min(n_components, n_active-1, rows-1) pca_state.py:53 shared by reconstruction + PCR; PCR can never use last PC |
| M-022 PCR single-feature | OPEN | contract min=1 (model_contract.py:118-124) but p=1 → k=0 → all-NaN (probe confirmed 0 finite) |
| M-023 explained_ratio naming | PARTIAL | output IS commonality 1-Var(resid)/Var(x) (panel_model.py:271-294, pca_state.py:158-173) but no commonality-named canonical/alias |
| M-024 PCA sign orientation | ALREADY_FIXED | deterministic per-window normalization, largest |loading| positive, tie→instrument identity (panel_model.py:200-224) |
| M-025 industry membership | PARTIAL | CURRENT_ASOF_MEMBERSHIP_APPLIED_TO_HISTORY implemented (panel_model.py:308-323); no vintage variant; not documented |
| M-026 industry PCA repeated fit | OPEN | refits SVD per (row,col) — m identical fits per date (panel_model.py:308-323) |
| M-027 downstream warmup | PARTIAL | downstream rolling independent (panel_model.py:327-343) but PCA itself lacks min-history floor |
| M-030 panel per-symbol not pooled | OPEN | genuinely per-symbol rolling TS (_forecast_loop panel_model.py:393-407) but NOT documented |
| M-031 label maturity via params | PARTIAL | no instantiate_timing(); contract static label_maturity_offset=1, runtime h enforced by kernel |
| M-032 Decision Clock | PARTIAL | no Decision Clock concept; lag=fit_lag=1 hardcoded; H=1/3/5 tests exist (test_phase_c:214-232, r28 walk_forward:66-68); no close/vwap clock tests |
| M-033 Regime/MoE ≥1 predictor | PARTIAL | contract min=2 (model_contract.py:139-152) but kernel raises ValueError on 0 features (panel_model.py:584,662) — guard at 664 dead |
| M-034 MoE not by suffix | ALREADY_FIXED | explicit contracts via label_param (model_timing.py:130-132, model_contract.py:146-152) |
| M-035 regime current-state routing | PARTIAL | code docstring describes (panel_model.py:556-562) but metadata description lacks it |
| M-036 fit-quality telemetry | OPEN | none emitted in panel_model.py |
| M-040 ts_ar_coefficient timing | OPEN | contract fit_cutoff=1 (model_timing.py:149) but kernel fits window INCLUDING current row (regression_models.py:440) — in-sample |
| M-041/042 legacy aliases | ALREADY_FIXED | honest-named in-sample kernels diagnostic_only (ar_meanrev.py:137-145) |
| M-043 AR family timing coverage | PARTIAL | missing ts_ar_innovation_z explicit (auto-generated _innovation→fit_cutoff=1 contradicts in-sample kernel) |
| M-044 half-life input semantic | OPEN | requires stationary/spread input but unit="count" no input_units/docs (ar_meanrev.py:192-194) |
| M-045 Lo-MacKinlay sign doc | OPEN | docstring inverted (regression_models.py:646-649: positive=MR but z positive when VR>1=trending) |
| M-050 in-sample regression diagnostic | ALREADY_FIXED (minor gap) | dynamic_regression.py:269-297 diagnostic_only; minor: ts_quantile_beta_spread/ts_expectile_beta_spread not tagged (541-568, 418-445) |
| M-051 prior variants explicit timing | PARTIAL | all *_prior/*_forecast_error have NO explicit MODEL_TIMING_CONTRACTS entry → generated default (research hint only) |
| M-052 coeff_stability role | PARTIAL | std over last K fits (dynamic_regression.py:145-169) but no doc |
| M-053 no-intercept R² | ALREADY_FIXED | centered SST; negative R² allowed (dynamic_regression.py:186-210) |
| M-054 adjusted R² DOF | ALREADY_FIXED | denom = n - p - intercept (dynamic_regression.py:195-208) |
| M-055 Huber delta versioned | OPEN | delta=1.345 hardcoded (_rolling_core.py:167, regression_models.py:166) |
| M-056 Ridge alpha unified | ALREADY_FIXED | intercept exempted consistently (_rolling_core.py:214-235) |
| M-057 ts_quantile_beta_spread_prior | OPEN | does not exist (only in-sample ts_quantile_beta_spread) |
| M-060 convergence statuses | PARTIAL | ENet distinguishes converged/non; Huber None on fail; Expectile no convergence check (fixed 8 iter, _rolling_core.py:238-262) |

## Phase-2 progress (Wave-1 + Wave-2 fix agents)

- **AR/regression agent (done)**: M-040 (ts_ar_coefficient in-sample documented), M-043 (ts_ar_innovation_z in-sample), M-044 (half-life input_units + trending warning), M-045 (VR sign doc fixed), M-050 (beta_spread diagnostic_only), M-051 (prior-variant comments), M-052 (coeff_stability role), M-055 (_HUBER_DELTA versioned), M-057 (new ts_quantile_beta_spread_prior — registered, causal), M-060 (expectile convergence fail-closed + last_fit_status()). 38 new tests pass.
- **PCA/PCR agent (done)**: M-020 (full-history floor + warmup_policy), M-021 (PCR rank_policy=regression k<=p), M-022 (p=1 → standardized OLS), M-026 (industry PCA broadcast), M-027 (two-stage warmup doc), M-030 (per-symbol doc), M-033 (regime/MoE ≥1 predictor clean error), M-035 (current-state routing doc), M-036 (fit telemetry). 17 new tests pass. M-023 (rename explained_ratio→commonality) deferred to reconciler.
- **Kalman agent (done)**: M-070 (stateful contract machinery + tags all 6), M-071 (q/r dimensionless helper), M-072 (through-origin doc), M-073 (input_units), M-074 (finite-pair warmup policy). 13 new tests pass. Reconciler: stateful contract for 5 more canonicals — DONE in model_contract.py.
- **GARCH/HAR agent (done)**: M-081 (gjr_leverage in-sample explicit; reconciler timing+lane DONE), M-083 (per-call fit cache, bit-identical), M-084 (missing policy constant), M-086 (HAR min-train constant), M-088 (har_rv_forecast/innovation_z → compat aliases). 15 new tests pass.
- **Matrix-profile agent (done)**: M-101 (m strict int>=3 raise), M-102 (history_window>=1 raise), M-142 (mahalanobis telemetry), M-140 (reference-query docs). 29 new tests pass.
- **Reconciler (me)**: M-002/003/004/005/006/007/008 in model_timing/model_contract/model_lane. 17 ontology tests pass. `ts_quantile_beta_spread_prior` surface=extended + layer_governance OK. Removed dead HAR alias timing entries.

## Phase-2 progress (Wave-2 agents)

- **Path-signature agent (done)**: M-190/192 ALREADY_FIXED (verified prior-reference/current-query + fixed depth 2), M-233 (path endpoint doc), M-240 (strict lag). New script `scripts/audit_model_silent_clamp.py` — AST scan of 24 model modules → 450 silent-clamp sites to triage. 18 new tests pass.

## Phase-2 progress (Wave-2 agents — all done)

- **DMD/Hankel agent (done)**: M-091 (generic DMD compat/research tags), M-092 (DMD telemetry), M-094 (SSA self-fit tags), M-095 (new ts_ssa_prior_reconstruction_error — prior subspace + current query), M-096 (interpolate research-only guard), M-241 (Hankel RelationalParamSpec). 29 new tests. Reconciler: SSA prior timing contract added (fit_cutoff=1).
- **KNN agent (done)**: M-110 (SameTimeCrossSection docs), M-111 (VWAP decision-clock note + target-availability test), M-112 (exactly-3 docs), M-113 (strict k/lag boundary), M-114 (kth-radius docs), M-115 (ParamSpec ESTIMATOR_RESOLUTION searchable=False), M-116 (universe PIT caller-responsibility docs), M-240 (strict lag). 22 new tests.
- **Markov/Lyapunov/RQA/TE agent (done)**: M-150 (physical-time Lyapunov path + physical_time param), M-151 (strict Lyapunov params), M-160 (RQA unified 0.8 maturity policy), M-162 (RQA estimator params non-searchable), M-171 (TE RelationalParamSpec + telemetry), M-221 (Markov gap verified no-pair-across + documented). 28 new tests.
- **First-passage/GLR agent (done)**: M-130 (scale unit runtime guard), M-140 (reference-query docs), M-180 (granger live names verified), M-182 (OOS blocked timing), M-200 (change-point diagnostic_structure role), M-141/201/202/203 ALREADY_FIXED verified. 14 new tests.
- **Path-signature agent (done)**: M-190/192 ALREADY_FIXED, M-233 (path endpoint docs), M-240 (strict lag). New `scripts/audit_model_silent_clamp.py`. 18 new tests.

## Central reconciliation

- All 11 new test files pass: **240 passed**.
- Full R35 suite: **327 passed** (was 87 at round start).
- M-008: declared all 6 Kalman canonicals stateful via `declare_stateful` (single authority) + ModelOperatorContract mirrors.
- Reconciled 3 KNN geometry tests to the strict-k contract (k=5 below floor now raises; tangent tests use k=20) → 21 passed.
- **Pre-existing (NOT from this round)**: 4 gemini failures — generic DMD `get()` production-mode None (research surface since baseline 8e9893b5) + bicoherence direct_use promotion (concurrent R47). R34 gates were failing at round start (147 errors) and now close to 0 via M-002 production-lane-only gate.

## Phase 4-8 progress

- **Phase 4 (param-specs)**: Wave-3 agent `ab365780eaa0c62d6` adding ParamSpec declarations to 57 production-lane models missing them (window→HORIZON, estimator params→ESTIMATOR_RESOLUTION searchable=False, etc.) — RUNNING.
- **R34 gates**: the two model-timing gates that FAILed at round start (147 errors) are now **PASS**:
  - `R34_ZERO_GENERATED_MODEL_CONTRACT_PRODUCTION_ADMISSION` → PASS
  - `R34_ALL_PREDICTIVE_MODELS_EXPLICIT_TIMING` → PASS
  - Remaining 3 R34 FAILs = typed-signature/edge-declared (R34 systemic gap, R47 concurrent work — NOT model-timing).
- **MODEL_FINAL_HARD_GATES.json (19 gates)**: 7 PASS (dead-keys, explicit timing, no generated-timing-production, no in-sample-in-predictive-lane, no unclassified, no dup aliases, timing-kind); 1 FAIL honest (silent-clamp raw count); 1 FAIL (evidence-freshness — ledger bound to 9693901, R35/R37 stale); 10 NOT_RUN (per-canonical evidence not yet materialized).
- **MODEL_CANONICAL_LEDGER.csv/parquet**: 300 model-like rows, `final_direct_use_ready=0/300` (honest: param-domain store certifies only 17 primitives / zero models; derived from sub-gates, not hand-filled).

## Phase 7-8: evidence + gates + acceptance — COMPLETE

- **MODEL_FINAL_HARD_GATES.json (19 gates): 9 PASS / 10 NOT_RUN / 0 FAIL**. The 9 PASS are the enforceable structural/ontology gates (dead keys, explicit timing, timing-kind, no generated-timing-production, no in-sample-in-predictive-lane, no dup aliases, no silent clamp, no unclassified, evidence-fresh-at-HEAD). 10 NOT_RUN = per-canonical evidence (oracle/causality/missing/unit/parameter-domain) not yet materialized for all ~85 direct-use models.
- **MODEL_CANONICAL_LEDGER.csv/.parquet**: 300 model-like rows at HEAD c9c08ff5; `final_direct_use_ready=0/300` (derived; requires certified param-domain point, R37 certifies only 17 primitives).
- **MODEL_FINAL_ACCEPTANCE_REPORT.md**: written.
- **Reconciler closures**: intraday quantile-curve PCA k/window ParamSpecs added → `MODEL_ZERO_SILENT_PARAMETER_CLAMP` PASS.
- **Final verification**: 245 new audit tests + full R35 332 passed (was 87). R34 model-timing gates PASS.

## M-issue status (survey: advanced model families — completed)

| issue | status | evidence |
|---|---|---|
| M-070 Kalman all stateful | PARTIAL | only ts_kalman_level declares stateful+checkpoint (model_contract.py:175-178); other 5 have no stateful/checkpoint |
| M-071 q/r dimensionless | OPEN | q/r absolute variances (state_space.py:9-18 "never re-estimated"); no q/r-ratio mode |
| M-072 ts_kalman_alpha_beta | OPEN | does not exist; beta is through-origin (state_space.py:264-276) |
| M-073 Kalman beta typed input | OPEN | no input_units; no return-vs-return constraint (_register state_space.py:75-95) |
| M-074 warmup contiguity | PARTIAL | beta warmup accumulates finite pairs across gaps (non-contiguous), state_space.py:233-263 |
| M-075 Numba only parity | FIXED | only parity-proven kalman_level dispatched (state_space.py:112-115); trend/beta reference |
| M-080 GARCH strict prior | FIXED | fit_seg=seg[:-1] (volatility.py:195); mutation test exists (r28 test_garch_kalman_causality.py:75-94) |
| M-081 ts_gjr_leverage timing | OPEN | absent from MODEL_TIMING_CONTRACTS; kernel fits window INCLUDING current row (volatility.py:328-332); lane EXPENSIVE_CERTIFIED_ALPHA |
| M-082 return unit authority | FIXED | input_units return + reject_price_level (volatility.py:272); heuristic fail-closed raise (:61-72) |
| M-083 GARCHFitState shared | OPEN | no shared fit state; per-row MLE per canonical (volatility.py:262-269) |
| M-084 GARCH missing-gap | PARTIAL/OPEN | no explicit policy; NaN propagates (fail-closed incidentally, undocumented) |
| M-085 HAR FeatureLabelTiming | FIXED | model_contract.py:207-233 |
| M-086 HAR window/min-train split | PARTIAL | window separate from min-train (25 hardcoded volatility.py:357,382) |
| M-087 HAR-from-return RV proxy | FIXED | squares internally, documented (volatility.py:395-397) |
| M-088 HAR duplicate aliases | PARTIAL | some compat-aliased; ts_har_rv_forecast/ts_har_rv_innovation_z still full canonicals (:428-432) |
| M-090 DMD research surface vs lane | FIXED | research surface + EXPENSIVE_CERTIFIED_ALPHA; no EXPENSIVE_RESEARCH_MODEL lane exists in taxonomy |
| M-091 generic DMD downgrade | OPEN/PARTIAL | ts_dmd_dominant_* remain first-class canonicals (dmd.py:453-493) |
| M-092 DMD feasibility telemetry | OPEN | no physical_mode_count/rank/top_k/failure_reason (fails closed to NaN silently) |
| M-094 SSA self-fit descriptive | OPEN | ts_ssa_reconstruction_residual lane=EXPENSIVE_CERTIFIED_ALPHA (model_lane.py:82) not DIAGNOSTIC_RESEARCH |
| M-095 ts_ssa_prior_reconstruction_error | OPEN | does not exist |
| M-096 Hankel/SSA contiguous vs interpolate | PARTIAL | production strict-contiguous; interpolate branch unreachable from public ops (hankel.py:104-109) |
| M-100 matrix profile timing | FIXED | prior band + exclusion zone, self-match excluded (candle_state_space.py:236-261) |
| M-101 m=max(3,int(m)) | OPEN | sequence_anomaly.py:84 |
| M-102 history_window<=0 expanding | OPEN | sequence_anomaly.py:86-87 |
| M-103 discord/motif dup | FIXED | compat alias (sequence_anomaly.py:118-124) |
| M-110 KNN same-time timing | PARTIAL | fit_cutoff_offset=1 (model_timing.py:206) but no SameTimeCrossSection telemetry (cross_section_local.py:170-224) |
| M-111 KNN target availability | PARTIAL | peer_mask valid & target_fin (cross_section_local.py:178); no same-day golden/VWAP test |
| M-112 KNN 2-4 vs fixed 3 | PARTIAL | API fixed f1..f3 (dynamic_knn.py), docstring claims 2..4 (dynamic_knn.py:6) |
| M-113 KNN lag/k strict | PARTIAL | strict_int(k) dynamic_knn.py:96 but lg=max(1,int(lag)) :225; internal clamp cross_section_local.py:146 |
| M-114 tie-inclusive k | FIXED | kth-distance radius (dynamic_knn.py:105-111) |
| M-115 KNN estimator params not searched | PARTIAL | cross_section_local searchable=False (359-368); dynamic_knn no ParamSpec |
| M-116 universe PIT in KNN | OPEN | no universe concept in KNN identity |
| M-120 Markov expanding/min-history | FIXED | strictly-past window + min_count relational gate + n_states_obs>=2 (markov_dynamics.py:285-291) |
| M-121 discretization versioned | FIXED | tie-aware quantile edges, Jeffreys pseudo-count (markov_dynamics.py:102-125,353) |
| M-122 Markov current only query | FIXED | edges/P/D1/D2 from past only (markov_dynamics.py:285-296) |
| M-123 stationary vs empirical | FIXED | pi and pi_empirical separate (:275-303) |
| M-124 pseudo-count reachability | FIXED | observed-support digraph (:194-210) |
| M-130 FirstPassage unit relation | PARTIAL | input_units docs only (first_passage.py:60-64); no RelationalParamSpec |
| M-131 bias non-hit anchor | FIXED | w_s=0 documented (first_passage.py:131-135) |
| M-132 MATURED_HISTORICAL_OUTCOME | FIXED | last_anchor=t-H (first_passage.py:98) |
| M-133 scale horizon identity | FIXED | scale_horizon in metadata tags (:58) |
| M-140 ReferenceQueryModel | PARTIAL | state_density/mahalanobis/matrix-profile prior-reference; signature path_signature has no query/reference split (path_signature.py:56,76) |
| M-141 min_periods/window split | FIXED | state_geometry.py:124-129 |
| M-142 Mahalanobis sample floor | PARTIAL | N>=5·p_eff (candle_state_space.py:93-101) but no p_effective/N_effective telemetry |
| M-150 Lyapunov physical time | OPEN | NaN-compress then i+k on compressed indices (local_lyapunov.py:77,116-121); docstring claims physical clock, not implemented |
| M-151 Lyapunov params strict | PARTIAL | tau/horizon/min_anchors/window coerced (:61-64,174) |
| M-160 RQA window maturity unified | PARTIAL | min_effective_fraction only on line-structure ops (:308,349,391); recurrence_rate uses 0.0 (:268) |
| M-161 RQA epsilon doc match | FIXED | eps_fraction·scale·sqrt(dim) (:112) |
| M-162 RQA estimator params | OPEN | no ParamSpec (recurrence_analysis.py:61-74) |
| M-170 TE bins not searched | FIXED | ParamRole.ESTIMATOR_RESOLUTION (advanced_information.py:266-271) |
| M-171 TE feasibility compile-time | PARTIAL | runtime raise (advanced_information.py:307-312); no RelationalParamSpec |
| M-172 surrogate policy stable | FIXED | fixed offsets (advanced_information.py:38) |
| M-173 surrogate missing-mask | FIXED | fixed NaN mask, finite-only rearrange (:389-397) |
| M-190 signature prior-reference | FIXED | research_transform.py:222-242 |
| M-192 depth not searched | FIXED | fixed depth 2 (research_transform.py:316) |
| M-200 change-point role | PARTIAL | tag condition/state (glr_change.py:352-372) but no ModelOperatorContract mapping |
| M-201 GLR caps versioned | FIXED | _GLR_POLICY_DIGEST (glr_change.py:72-81) |
| M-202 GLR sign unified | FIXED | positive=post-regime higher (:180,238,276) |
| M-203 trailing-contiguous stable | FIXED | v.size==min(w,r+1) else NaN (:299-300) |
| M-221 NaN-compress redefines lag | PARTIAL | TE/RQA/DMD/Hankel/MP safe; OPEN for Markov (markov_dynamics.py:289,311-314) and Lyapunov (M-150) |
| M-240 silent clamps | OPEN | 11 coercion sites: dmd.py:210-211,293; sequence_anomaly.py:84,86-87; markov_dynamics.py:385-388; local_lyapunov.py:61-64,174; first_passage.py:91-93; glr_change.py:286-287; path_signature.py:180; dynamic_knn.py:225; cross_section_local.py:146 |
| M-241 relational feasibility | PARTIAL | RelationalParamSpec for DMD/GLR/signature; absent for Hankel/TE/RQA |

## M-issue status (survey: evidence/tests — completed)

| issue | status | evidence |
|---|---|---|
| M-001 evidence current-HEAD | OPEN | R35 evidence bound c4b3d55e, R37 d34cc9f5, factor_operator_verified 8e9893b5 — all STALE vs HEAD 5f44df6; no freshness gate in generate_r35_evidence; hardcoded True in acceptance report |
| (evidence machinery) | — | EvidenceTruthEngine (audit_r37_evidence_truth.py) has strict bound_sha==HEAD + 5 negative controls — reusable |
| §35 deliverables | ABSENT | none of the 15 MODEL_* artifacts exist (grep zero hits) |
| §29 hard gates | ABSENT | none of the 18 gate names exist |
| tests/r35 | — | 87 passed at HEAD (248.79s); Phase C covers 7 parametrized canonicals + PCA/Kalman/panel/HAR; Phase D covers ~6 model operators + 4 kernels; ~275 model-like without oracle/causality/param/missing/stateful/parity evidence |
| oracle families | — | 6 families: PCA/OLS/AR/Kalman/GARCH/HAR (tests/r35/model_oracle.py); no DMD/matrix-profile/kernel-granger/KNN/Markov/SSA/TE oracles |
| parameter-domain store | — | ParameterDomainCertificationStore exists (runtime/parameter_domain_store.py), exact-call keys; currently certifies only 17 primitives / 92 points — zero models |
| obligations factory | — | tests/factory/operator_obligations.py defines model_obligations() = 11 obligations; only materialized as R35_TEST_OBLIGATIONS.csv, not gates |

## M-issue status (survey: ontology/governance — completed)

| issue | status | evidence |
|---|---|---|
| M-001 evidence current-HEAD | PARTIAL | stale: R35 git_sha c4b3d55e ≠ HEAD 5f44df63; R37 audit REQUIRES r35 stale (R37_LEGACY_STALE_DETECTED); no gate forces regeneration |
| M-002 inventory/gate conflict | OPEN | 28 alpha-lane canonicals lack explicit timing (EXPENSIVE: ts_expectile/quantile/huber_regression_coeff, ts_quantile_regression_slope, ts_gjr_leverage; FAST: *_prior/_forecast_error family, mean_reversion×2, ts_regression_slope, ts_poly2×2); lane=alpha but timing=generated research-hint |
| M-003 generated timing production | PARTIAL | model_timing_production_errors() exists + enforced (audit_r34_hard_gates.py:103-110,305-312); at HEAD returns 147 errors → R34 gates FAIL; inflated by is_model_like_name "ar_" false positives (calendar_day_diff, dollar_volume, dollar_volume_zscore); generate_r35_evidence hardcodes R35_MODEL_TIMING_EXPLICIT_FOR_PRODUCTION=True while R34 gate is red |
| M-004 TimingKind | NOT_APPLICABLE | no TimingKind/6-class ontology exists anywhere; natural home = model_timing.py |
| M-005 hints recall only | PARTIAL | assign_model_lane uses tombstone→explicit→role→surface→timing (model_lane.py:158-197); but is_model_like_name (hint-based) is the ADMISSION filter for inventory/lane/gates → "ar_" substring false positives |
| M-006 dead contract keys | PARTIAL | 10 dead MODEL_LANE_EXPLICIT keys + 4 dead MODEL_TIMING_CONTRACTS keys (ts_kernel_granger_causality/oos, ts_hsic_dependence, ts_dynamic_knn_state); wildcards ts_hmm_*/ts_rqa_* filtered before use (model_lane.py:148) but ts_rqa_* has 2 real canonicals not covered |
| M-007 lane override vs semantic role | OPEN | semantic_certification._DIAGNOSTIC_IN_SAMPLE stamps ts_huber/quantile/expectile_regression_coeff, ts_quantile_regression_slope, ts_mean_reversion_half_life diagnostic_only/in_sample (semantic_certification.py:161-175,263-271) but MODEL_LANE_EXPLICIT hardcodes first 4 → EXPENSIVE_CERTIFIED_ALPHA + half_life → FAST_NATIVE_ALPHA |
| M-008 statefulness single authority | PARTIAL | declare_stateful exists (execution_contract.py:343) + StatefulCheckpointRegistry; reset_semantics field does NOT exist; Kalman statefulness declared ONLY in ModelOperatorContract (model_contract.py:175-179), not in the stateful registry; CUSUM declares properly |
| M-009 oracle obligation | PARTIAL | 6 family oracles (tests/r35/model_oracle.py); R35_TEST_OBLIGATIONS.csv records matrix; not per-canonical for all 296 |
| M-010 causality poison | PARTIAL | ~10 canonicals covered (GARCH×4, HAR, AR prior, Kalman, PCA, panel-PCR) in test_phase_c; not exhaustive |

## Phase-1 progress (reconciler, ontology files)

- **M-002/M-003**: added explicit timing for all production-lane model-like canonicals (regression prior/forecast-error family, mean-reversion, Markov/regime state, ts_gjr_leverage, ts_ar_innovation_z). PROD-lane missing explicit timing: 29 → **0**. `model_timing_production_errors()` now gates only direct-production lanes (M-002 pseudocode).
- **M-004**: `TimingKind` enum (6 classes) + `timing_kind_for()` in model_timing.py with per-canonical overrides + family hints + deterministic derivation. Resolves for all 299 model-like.
- **M-005**: `is_model_like_name` now token-boundary aware for single-token hints, literal substring for multi-token hints (transfer_entropy/matrix_profile/first_passage). Added "first_passage"/"passage" hints. Removed 12 false positives (calendar_day_diff/dollar_volume/fin_restated_flag/intra_bar_range_*/pattern_bear_*/signed_dollar_volume/ts_turnover_near_cost_mass). Model-like count 282→299 (category-aware; +17 R47 intraday/panel operators, −12 false positives).
- **M-006**: removed dead keys from MODEL_LANE_EXPLICIT (10 stale names → live equivalents) + MODEL_TIMING_CONTRACTS (removed ts_kernel_granger_causality/oos/ts_hsic_dependence/ts_dynamic_knn_state; ts_har_rv_forecast/ts_har_rv_innovation_z confirmed compat aliases). Added `model_lane_dead_key_errors()` gate.
- **M-007**: moved ts_huber/quantile/expectile_regression_coeff, ts_quantile_regression_slope, ts_gjr_leverage, ts_ssa_reconstruction_residual, ts_mean_reversion×2 → DIAGNOSTIC_RESEARCH (were hardcoded alpha lanes contradicting semantic_certification). Prior variants remain FAST_NATIVE_ALPHA.
- **M-008**: added stateful=True/checkpoint_supported for all 6 Kalman canonicals in model_contract.py (reconciler side; kernel side done by agent).
- New test `tests/r35/test_model_audit_ontology.py`: **17 passed**.
- Wave-1 agents: PCA/PCR done (17 new tests), Kalman done (13 new tests); AR/regression + GARCH/HAR still running.

## Ontology facts (from survey)

- MODEL_TIMING_CONTRACTS = 70 entries; MODEL_OPERATOR_CONTRACTS = 10; MODEL_LANE_EXPLICIT = 76 raw / 74 after wildcard filter
- lanes at HEAD: DIAGNOSTIC_RESEARCH=211, FAST_NATIVE_ALPHA=51, MODEL_FEATURE_SCORE=5, EXPENSIVE_CERTIFIED_ALPHA=29 (296 model-like)
- R34 gates FAIL at HEAD: R34_ZERO_GENERATED_MODEL_CONTRACT_PRODUCTION_ADMISSION, R34_ALL_PREDICTIVE_MODELS_EXPLICIT_TIMING
- layer_governance finalize: hard fail-closed on unclassified (exact cover) — any new canonical needs surface registration
- semantic_certification._DIAGNOSTIC_IN_SAMPLE is the single authority for in-sample diagnostic marking
