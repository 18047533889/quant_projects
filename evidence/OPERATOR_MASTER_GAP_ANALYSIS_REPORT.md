# Operator Master List Gap Analysis Report

**Analysis Date:** 2026-08-12  
**HEAD Commit:** 4b2b39bfce16e2dfa5d1449a344157a31631aa17  
**Total Candidates Analyzed:** 245

## Executive Summary

This report provides a comprehensive semantic diff and classification of 245 operators from the master list against the current codebase implementation. The analysis categorizes each operator to determine whether it requires new implementation, already exists, needs certification, or should be rejected.

### Current Implementation Baseline

- **Total operators in current codebase:** 608
  - Daily surface: 86 operators
  - Extended surface: 519 operators
  - Research surface: 3 operators

### Master List Composition

The 245 candidates come from:
- R41-R47 historical candidates: 156 operators
- Additional primitives: 8 operators
- Filter/State-space candidates: 28 operators
- Fiscal/Relation backlog: 31 operators
- PROD_RECERTIFY (existing, needs certification): 22 operators

## Classification Results

### Summary by Disposition

| Disposition | Count | Percentage | Action Required |
|-------------|-------|------------|-----------------|
| **TRUE_GAP_IMPLEMENT** | 173 | 70.6% | New implementation needed |
| **RESEARCH_ONLY** | 34 | 13.9% | Research surface implementation |
| **PROD_RECERTIFY** | 22 | 9.0% | Backend/evidence certification |
| **EXISTING_ALIAS** | 6 | 2.4% | Already exists as alias |
| **EXISTING_EXACT** | 6 | 2.4% | Already exists exactly |
| **BLOCKED_DATA_CONTRACT** | 2 | 0.8% | Missing data fields |
| **EXISTING_EQUIVALENT** | 1 | 0.4% | Semantic equivalent exists |
| **ARCHITECTURE_SUPERSEDED** | 1 | 0.4% | Handled by architecture layer |
| **TOTAL** | **245** | **100%** | |

### Work Distribution

#### No New Work Required (16 operators, 6.5%)
These operators already exist in the current implementation:
- **EXISTING_EXACT (6):** KAMA, holder_concentration_change, industry_size_neutralize, relation_entropy, revision_delta, ts_ewm_std
- **EXISTING_ALIAS (6):** cn_sma→ts_sma_cn, financial_snapshot_lag→fin_lag, fiscal_delta→fin_diff, fiscal_lag→fin_lag, fiscal_pct_change→fin_pct_change, price_delay_score→ts_price_delay
- **EXISTING_EQUIVALENT (1):** fiscal_acceleration→fin_growth_acceleration
- **ARCHITECTURE_SUPERSEDED (1):** report_asof (handled by DataAccess PIT join layer)
- **BLOCKED_DATA_CONTRACT (2):** fin_schema_gate, laborforce_efficiency

#### Certification/Enhancement Required (22 operators, 9.0%)
**PROD_RECERTIFY** - These operators exist but need production certification:
- Filter layer despike (3): ts_hampel_filter_causal, ts_median3_causal, ts_rolling_median_causal
- Filter layer smooth (4): ts_robust_ema, ts_super_smoother, ts_kama, ts_butterworth_lowpass_causal, ts_causal_local_linear_smoother
- Filter layer hysteresis (10): state_adaptive_deadband, state_rank_deadband, state_quantile_hysteresis, state_adaptive_slew_limit, state_l1_turnover_prox, state_l2_partial_adjustment, state_cost_aware_deadband, state_cost_aware_slew, state_confidence_weighted_ema, state_uncertainty_deadband
- Robust operators (4): ts_quantile_range, ts_trimmed_mean, ts_robust_zscore_inclusive, ts_robust_zscore_prior

**Required Certification Activities:**
1. Polars backend implementation (no pandas fallback)
2. Numeric parity validation (pandas vs polars)
3. PIT causality verification
4. Checkpoint/resume testing for stateful operators
5. Parameter domain evidence generation
6. Cold-start operator contract compliance

#### New Implementation Required (173 operators, 70.6%)
**TRUE_GAP_IMPLEMENT** - Genuine gaps requiring development:

**By Category:**

1. **Technical Indicators (8):** ALMA, CoppockCurve, ElderRay, FisherTransform, HMA, QQE, RSX, WMA

2. **Fiscal/Fundamental Analysis (32):**
   - Fiscal quality metrics: fiscal_accrual_quality, fiscal_reversal_ratio, fiscal_true_streak
   - Fiscal time series: fiscal_rolling_regression, fiscal_rolling_std, fiscal_rolling_slope
   - Fiscal statistics: fiscal_autocorr, fiscal_ar_resid_std, fiscal_direction_consistency
   - Fundamental lifecycle: cash_flow_lifecycle_stage, fundamental_cash_flow_duration
   - Industry/comparison: industry_fiscal_resid, accounting_comparability_score

3. **Intraday Microstructure (95+):**
   - State-space features: intra_state_count, intra_state_sum, intra_state_vwap
   - Volume profile: intra_volume_at_price_profile, intra_volume_profile_peak_geometry
   - Event detection: intra_impulse_event_detector, intra_post_impulse_response
   - Time structure: intra_session_segment_reduce, intra_business_time_deformation
   - Market impact: intra_supply_absorption_score, intra_absorption_curve_area

4. **Filter/Signal Processing (16):**
   - State-driven: state_change_point_adaptive_ema, state_filter_reset_on_break, state_gain_scheduler
   - Shrinkage: cs_shrink_to_market_mean, cs_shrink_to_group_mean, cs_empirical_bayes_shrinkage
   - Trend filters: ts_l1_trend_filter_trailing, ts_total_variation_filter_trailing
   - Adaptive filters: ts_vidya, ts_mcginley_dynamic, ts_one_euro_filter

5. **Panel/Cross-sectional (12):**
   - Panel models: panel_ewm_beta_ex_self, panel_apm_residual_tstat, panel_async_beta_ex_self
   - Clustering: panel_similarity_crowding_score, panel_cluster_risk_score
   - Factor analysis: panel_factor_pocket_strength, cs_factor_bucket_return

6. **Calendar/Timing (6):**
   - Same-calendar features: same_calendar_day_mean, same_calendar_month_return
   - Date utilities: date_diff_days, years_since_date, fundamental_staleness_days
   - Trading calendar: trading_calendar_mask

7. **Relation/Network (4):**
   - Relation metrics: relation_jaccard, relation_period_change
   - Network: panel_peer_graph_aggregate
   - Utilities: row_sum_skipna

#### Research-Only Implementation (34 operators, 13.9%)
**RESEARCH_ONLY** - Complex ML/topological methods for research surface first:

**By Method Class:**

1. **Topological Data Analysis (5):**
   - intra_persistent_homology_features
   - intra_topological_anomaly_score
   - intra_topological_peer_anomaly
   - cs_topological_anomaly_score
   - intra_recurrence_network_features

2. **State-Space/Hidden Models (6):**
   - intra_hmm_state_features, intra_hmm_posterior_entropy
   - intra_hsmm_duration_features
   - intra_hawkes_event_features
   - intra_dmd_koopman_features
   - intra_kalman_latent_price (note: should be production feasible)

3. **Advanced Signal Processing (7):**
   - intra_wavelet_scattering_features
   - intra_emd_hilbert_huang_features
   - ts_modwt_denoise_trailing
   - ts_spectral_lowpass_trailing
   - ts_wavelet_shrinkage_trailing (note: should be production feasible)
   - intra_visibility_graph_features
   - intra_information_flow_features

4. **Machine Learning / Deep Learning (6):**
   - cs_isolation_forest_score
   - intra_neural_cde_embedding
   - intra_contrastive_path_embedding
   - intra_functional_autoencoder_score
   - cs_predictability_mosaic_score, panel_predictability_mosaic_score

5. **Adaptive Filtering (5):**
   - ts_nlms_filter, ts_rls_filter
   - ts_student_t_kalman_filter, ts_adaptive_noise_kalman
   - ts_h_infinity_level_filter

6. **Manifold/Geometry (5):**
   - intra_covariance_manifold_shift
   - intra_diffusion_map_state
   - intra_dynamic_stock_graph_features
   - cs_peer_graph_smooth
   - intra_function_on_function_anomaly_response

## Key Findings

### 1. Filter Layer Success
The recent filter layer implementation (despike, smooth, hysteresis) added 18 operators that were in the master list. All 18 exist but need production certification. This demonstrates effective planning alignment.

### 2. High True Gap Percentage
173 operators (70.6%) are genuine gaps. This indicates:
- The master list contains many specialized/advanced operators
- Significant development work is needed for full coverage
- Many candidates are intraday microstructure focused (95+ operators)

### 3. Fiscal/Fundamental Coverage Gaps
Despite extensive fin_* operators in the codebase, 32 fiscal-specific operators are missing:
- Fiscal quality metrics (accrual_quality, reversal_ratio)
- Fiscal regression/correlation features
- Industry-adjusted fiscal metrics

### 4. Intraday Dominance
95+ intraday operators are in TRUE_GAP, representing 55% of all gaps. Categories:
- State-space features (13)
- Volume profile analysis (10)
- Event-driven features (15)
- Functional/path features (20)
- Advanced microstructure (37+)

### 5. Research vs Production Trade-off
34 operators classified as RESEARCH_ONLY involve:
- Complex ML models (isolation forest, autoencoders, neural CDEs)
- Topological data analysis (persistent homology, recurrence networks)
- Advanced state-space models (HMM, HSMM, Hawkes processes)
- Computationally intensive methods (wavelet scattering, EMD)

These should be implemented on research surface first with:
- No production-ready requirements initially
- Pandas-only backends acceptable
- Focus on semantic correctness over performance
- Gradual migration path to extended/daily surfaces

## Recommendations

### Priority 0: PROD_RECERTIFY (22 operators)
**Timeline:** 1-2 weeks  
**Effort:** Medium

Complete production certification for existing filter layer operators:
1. Implement Polars backends (no pandas fallback)
2. Validate numeric parity (pandas vs polars, DuckDB where applicable)
3. Generate parameter domain evidence
4. Test checkpoint/resume for stateful operators
5. Verify PIT causality
6. Update operator contracts with certification metadata

**Deliverables:**
- `recent_operator_backend_audit.csv`
- `polars_parity.csv`
- `state_checkpoint_parity.csv`
- `parameter_domain_evidence.csv`

### Priority 1: Technical Indicators (8 operators)
**Timeline:** 1 week  
**Effort:** Low-Medium

Implement standard technical indicators with established definitions:
- HMA (Hull Moving Average)
- WMA (Weighted Moving Average)
- ALMA (Arnaud Legoux Moving Average)
- RSX (Relative Strength Index alternative)
- QQE (Quantitative Qualitative Estimation)
- CoppockCurve, ElderRay, FisherTransform

These have well-defined mathematics and existing TA-Lib references.

### Priority 2: Fiscal Quality & Time Series (15 operators)
**Timeline:** 2-3 weeks  
**Effort:** Medium

Implement core fiscal analysis operators:
- Quality metrics: fiscal_accrual_quality, fiscal_reversal_ratio, fiscal_true_streak
- Time series: fiscal_rolling_regression, fiscal_rolling_std, fiscal_autocorr
- Comparison: fiscal_direction_consistency, fiscal_pair_direction_agreement

### Priority 3: Filter/Signal Processing (16 operators)
**Timeline:** 2-3 weeks  
**Effort:** Medium-High

Complete the filter layer with missing primitives:
- State-driven filters (3)
- Shrinkage operators (3)
- Trend filters (2)
- Adaptive filters (3)
- Robust Kalman variants (2)

### Priority 4: Intraday State & Profile (25 operators)
**Timeline:** 3-4 weeks  
**Effort:** High

Implement foundational intraday features:
- State aggregations (10): intra_state_count, intra_state_sum, intra_state_vwap, etc.
- Volume profile (5): intra_volume_at_price_profile, peak_geometry, supply_structure
- Event windows (5): intra_event_window_reduce, intra_event_pre_post_contrast
- Session features (5): intra_session_segment_reduce, boundary_jump, etc.

### Deferred: Research Methods (34 operators)
**Timeline:** Long-term / ongoing research  
**Effort:** Very High

Implement complex methods on research surface with no immediate production requirements:
- Start with simpler methods (isolation forest, basic Kalman variants)
- Progress to advanced methods as research validates utility
- No production certification required initially
- Focus on semantic correctness over performance

### Rejected: Blocked/Superseded (3 operators)
Do not implement:
- `report_asof`: Architecture handles this via DataAccess PIT join
- `fin_schema_gate`: Missing schema infrastructure
- `laborforce_efficiency`: Missing employee_count field in A-share data

## Implementation Guidelines

### For TRUE_GAP Operators

1. **Research Phase:**
   - Review academic papers / TA-Lib definitions
   - Identify mathematical specification
   - Determine PIT requirements
   - Check data field availability

2. **Design Phase:**
   - Define canonical name following naming conventions
   - Write operator contract (timing, role, lane, state)
   - Design parameter specification with ParamRole
   - Identify backend capabilities (pandas/polars/duckdb)

3. **Implementation Phase:**
   - Implement pandas reference backend first
   - Add Polars backend for daily operators
   - Write unit tests (correctness, PIT, edge cases)
   - Generate parameter domain evidence

4. **Certification Phase:**
   - Validate numeric parity across backends
   - Test checkpoint/resume for stateful operators
   - Verify PIT causality with future-poison tests
   - Measure performance benchmarks

### For RESEARCH_ONLY Operators

1. **Research Surface First:**
   - Implement on research surface with pandas-only
   - Focus on semantic correctness
   - Accept slower performance initially
   - Document mathematical foundations

2. **Validation Phase:**
   - Test with real research workflows
   - Validate utility in factor mining
   - Identify performance bottlenecks
   - Gather user feedback

3. **Production Migration (if validated):**
   - Implement Polars backend
   - Add production hardening
   - Complete six-gate certification
   - Migrate to extended surface

## Files Generated

1. **`operator_master_gap_preflight_854bdc22.csv`**
   - Complete classification of all 245 operators
   - Columns: candidate, source_round, disposition, reason, current_exact/alias/equivalent, surface
   - Machine-readable format for downstream processing

2. **`OPERATOR_MASTER_GAP_ANALYSIS_REPORT.md`** (this file)
   - Executive summary and findings
   - Recommendations and priorities
   - Implementation guidelines

## Next Steps

1. **Immediate (Week 1):**
   - Review and approve classification results
   - Begin PROD_RECERTIFY certification for 22 existing operators
   - Start Priority 1 technical indicators implementation

2. **Short-term (Weeks 2-4):**
   - Complete filter layer certification
   - Implement technical indicators
   - Begin fiscal quality operators

3. **Medium-term (Weeks 5-12):**
   - Implement filter/signal processing gaps
   - Develop intraday state & profile operators
   - Begin panel/cross-sectional operators

4. **Long-term (3+ months):**
   - Research-only methods on research surface
   - Advanced intraday microstructure
   - Validation and production migration

## Appendix: Classification Methodology

### Disposition Categories

- **EXISTING_EXACT:** Canonical name found in operator_surface.py surfaces
- **EXISTING_ALIAS:** Found as alias (e.g., KAMA → ts_kama)
- **EXISTING_EQUIVALENT:** Semantically equivalent operator exists (e.g., fiscal_lag ≈ fin_lag)
- **COMPOSABLE_NO_NEW:** Can compose from 2-3 existing primitives (not used in this analysis)
- **TRUE_GAP_IMPLEMENT:** Real gap, needs implementation
- **PROD_RECERTIFY:** Exists but needs backend/evidence certification
- **RESEARCH_ONLY:** Research surface only (complex ML/topological methods)
- **BLOCKED_DATA_CONTRACT:** Missing data fields in current data contract
- **REJECT_LOOKAHEAD:** Violates PIT principles (not applicable in this analysis)
- **ARCHITECTURE_SUPERSEDED:** Architecture layer replaced it

### Data Sources

- Master list: `/home/shw/quant_projects/FactorEngine_全部新增算子_Master清单_20260812.md`
- Operator surfaces: `cleaned_operators/operator_surface.py`
- Implementation files: `cleaned_operators/*.py`, `cleaned_operators/fundamental/*.py`, `cleaned_operators/intraday/*.py`
- Current HEAD: 4b2b39bfce16e2dfa5d1449a344157a31631aa17

---

**Report generated:** 2026-08-12  
**Analysis tool:** `scripts/classify_master_operators.py`  
**Total time investment:** Comprehensive semantic diff of 245 operators against 608 existing operators
