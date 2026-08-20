# R23 Fundamental Operator Audit

Fundamental canonicals: 160

| canonical | final_status | accepted_flow_semantics | blockers |
|---|---|---|---|
| altman_z_score | SUPPORTING_ONLY | not-declared |  |
| cash_flow_lifecycle_stage | SUPPORTING_ONLY | not-declared |  |
| date_diff_days | SUPPORTING_ONLY | not-declared |  |
| fin_accrual_ratio | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_acquisition_cash_intensity | SUPPORTING_ONLY | not-declared |  |
| fin_actual_expectation_divergence | CERTIFIED_CONTEXTUAL | not-declared | PIT11_EXPECTATION_POST_EVENT_LEAK |
| fin_average_balance | SUPPORTING_ONLY | not-declared |  |
| fin_beat_streak | CERTIFIED_CONTEXTUAL | not-declared | PIT11_EXPECTATION_POST_EVENT_LEAK |
| fin_borrowing_intensity | SUPPORTING_ONLY | not-declared |  |
| fin_cagr | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_capex_growth | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_capex_intensity | SUPPORTING_ONLY | not-declared |  |
| fin_cash_burn_runway | SUPPORTING_ONLY | not-declared |  |
| fin_cash_conversion | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_cash_earnings_gap | SUPPORTING_ONLY | not-declared |  |
| fin_cash_sales_divergence | SUPPORTING_ONLY | not-declared |  |
| fin_cashflow_persistence | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_common_size | SUPPORTING_ONLY | not-declared |  |
| fin_component_score | SUPPORTING_ONLY | not-declared |  |
| fin_comprehensive_income_gap | SUPPORTING_ONLY | not-declared |  |
| fin_contract_asset_growth | SUPPORTING_ONLY | flow_type:Stock |  |
| fin_contract_asset_intensity | SUPPORTING_ONLY | not-declared |  |
| fin_contract_asset_liability_gap | SUPPORTING_ONLY | not-declared |  |
| fin_contract_liability_growth | SUPPORTING_ONLY | flow_type:Stock |  |
| fin_contract_liability_intensity | SUPPORTING_ONLY | not-declared |  |
| fin_core_earnings_ratio | SUPPORTING_ONLY | not-declared |  |
| fin_cv | SUPPORTING_ONLY | not-declared |  |
| fin_days_since_expectation_revision | CERTIFIED_CONTEXTUAL | not-declared | PIT18_REVISION_EVENT_UNPROVEN |
| fin_days_since_update | CERTIFIED_CONTEXTUAL | not-declared | PIT18_REVISION_EVENT_UNPROVEN |
| fin_debt_repayment_intensity | SUPPORTING_ONLY | not-declared |  |
| fin_debt_service_coverage_proxy | SUPPORTING_ONLY | not-declared |  |
| fin_deferred_tax_gap | SUPPORTING_ONLY | not-declared |  |
| fin_delta_noa | SUPPORTING_ONLY | flow_type:Stock |  |
| fin_diff | SUPPORTING_ONLY | not-declared |  |
| fin_discontinued_operation_ratio | SUPPORTING_ONLY | not-declared |  |
| fin_divergence | SUPPORTING_ONLY | not-declared |  |
| fin_earnings_cash_gap_volatility | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_earnings_persistence | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_earnings_smoothness | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_equity_capital_growth | SUPPORTING_ONLY | flow_type:Stock |  |
| fin_expectation_dispersion | SUPPORTING_ONLY | not-declared |  |
| fin_expectation_revision | CERTIFIED_CONTEXTUAL | not-declared | PIT18_REVISION_EVENT_UNPROVEN |
| fin_expectation_revision_count | CERTIFIED_CONTEXTUAL | not-declared | PIT18_REVISION_EVENT_UNPROVEN |
| fin_expectation_revision_magnitude | CERTIFIED_CONTEXTUAL | not-declared | PIT18_REVISION_EVENT_UNPROVEN |
| fin_expectation_revision_pct | CERTIFIED_CONTEXTUAL | not-declared | PIT18_REVISION_EVENT_UNPROVEN |
| fin_expectation_revision_speed | CERTIFIED_CONTEXTUAL | not-declared | PIT18_REVISION_EVENT_UNPROVEN |
| fin_expense_sales_divergence | SUPPORTING_ONLY | not-declared |  |
| fin_fair_value_income_dependence | SUPPORTING_ONLY | not-declared |  |
| fin_financing_gap | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_fundamental_strength_coverage | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_fundamental_strength_score | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_goodwill_intensity | SUPPORTING_ONLY | not-declared |  |
| fin_goodwill_risk_score | SUPPORTING_ONLY | flow_type:Stock |  |
| fin_growth | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_growth_acceleration | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_growth_change | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_growth_persistence | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_growth_stability | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_growth_volatility | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_impairment_intensity | SUPPORTING_ONLY | not-declared |  |
| fin_interest_coverage_proxy | SUPPORTING_ONLY | not-declared |  |
| fin_inventory_sales_divergence | SUPPORTING_ONLY | not-declared |  |
| fin_investment_income_dependence | SUPPORTING_ONLY | not-declared |  |
| fin_lag | SUPPORTING_ONLY | not-declared |  |
| fin_lease_asset_liability_gap | SUPPORTING_ONLY | not-declared |  |
| fin_lease_intensity | SUPPORTING_ONLY | not-declared |  |
| fin_log_change | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_mad | SUPPORTING_ONLY | not-declared |  |
| fin_margin_persistence | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_mean_abs_deviation | SUPPORTING_ONLY | not-declared |  |
| fin_median_abs_deviation | SUPPORTING_ONLY | not-declared |  |
| fin_minority_profit_share | SUPPORTING_ONLY | not-declared |  |
| fin_miss_streak | CERTIFIED_CONTEXTUAL | not-declared | PIT11_EXPECTATION_POST_EVENT_LEAK |
| fin_monotonicity | SUPPORTING_ONLY | not-declared |  |
| fin_negative_streak | SUPPORTING_ONLY | not-declared |  |
| fin_net_borrowing_cashflow | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_net_debt_issuance | SUPPORTING_ONLY | flow_type:Stock |  |
| fin_noncore_income_ratio | SUPPORTING_ONLY | not-declared |  |
| fin_oci_to_equity | SUPPORTING_ONLY | not-declared |  |
| fin_other_earnings_dependence | SUPPORTING_ONLY | not-declared |  |
| fin_pct_change | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_percentile_history | SUPPORTING_ONLY | not-declared |  |
| fin_percentile_vs_prior_history | SUPPORTING_ONLY | not-declared |  |
| fin_period_restated | SUPPORTING_ONLY | not-declared |  |
| fin_period_revision_age | SUPPORTING_ONLY | not-declared |  |
| fin_period_revision_count | SUPPORTING_ONLY | not-declared |  |
| fin_positive_streak | SUPPORTING_ONLY | not-declared |  |
| fin_qoq | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_quarter_from_cumulative | SUPPORTING_ONLY | not-declared |  |
| fin_range | SUPPORTING_ONLY | not-declared |  |
| fin_ratio | SUPPORTING_ONLY | not-declared |  |
| fin_rd_capitalization_ratio | SUPPORTING_ONLY | not-declared |  |
| fin_rd_total_intensity | SUPPORTING_ONLY | not-declared |  |
| fin_receivable_sales_divergence | SUPPORTING_ONLY | not-declared |  |
| fin_restated_flag | CERTIFIED_CONTEXTUAL | not-declared | PIT18_REVISION_EVENT_UNPROVEN |
| fin_revision_count | CERTIFIED_CONTEXTUAL | not-declared | PIT18_REVISION_EVENT_UNPROVEN |
| fin_revision_delta | CERTIFIED_CONTEXTUAL | not-declared | PIT18_REVISION_EVENT_UNPROVEN |
| fin_revision_direction | CERTIFIED_CONTEXTUAL | not-declared | PIT18_REVISION_EVENT_UNPROVEN |
| fin_revision_magnitude | CERTIFIED_CONTEXTUAL | not-declared | PIT18_REVISION_EVENT_UNPROVEN |
| fin_revision_pct | CERTIFIED_CONTEXTUAL | not-declared | PIT18_REVISION_EVENT_UNPROVEN |
| fin_roe_cash_gap | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_seasonal_percentile | SUPPORTING_ONLY | not-declared |  |
| fin_seasonal_zscore | SUPPORTING_ONLY | not-declared |  |
| fin_sign_change_count | SUPPORTING_ONLY | not-declared |  |
| fin_stability | SUPPORTING_ONLY | not-declared |  |
| fin_staleness | CERTIFIED_CONTEXTUAL | not-declared | PIT18_REVISION_EVENT_UNPROVEN |
| fin_std | SUPPORTING_ONLY | not-declared |  |
| fin_surprise | CERTIFIED_CONTEXTUAL | not-declared | PIT11_EXPECTATION_POST_EVENT_LEAK |
| fin_surprise_event_percentile | CERTIFIED_CONTEXTUAL | not-declared | PIT11_EXPECTATION_POST_EVENT_LEAK |
| fin_surprise_event_zscore | CERTIFIED_CONTEXTUAL | not-declared | PIT11_EXPECTATION_POST_EVENT_LEAK |
| fin_surprise_zscore | CERTIFIED_CONTEXTUAL | not-declared | PIT11_EXPECTATION_POST_EVENT_LEAK |
| fin_total_operating_accruals | SUPPORTING_ONLY | flow_type:Stock |  |
| fin_trend_acceleration | SUPPORTING_ONLY | not-declared |  |
| fin_trend_r2 | SUPPORTING_ONLY | not-declared |  |
| fin_trend_slope | SUPPORTING_ONLY | not-declared |  |
| fin_trend_tstat | SUPPORTING_ONLY | not-declared |  |
| fin_ttm_cumulative | SUPPORTING_ONLY | not-declared |  |
| fin_ttm_quarterly | SUPPORTING_ONLY | not-declared |  |
| fin_turnover | SUPPORTING_ONLY | not-declared |  |
| fin_working_capital_accruals | SUPPORTING_ONLY | flow_type:Stock |  |
| fin_working_capital_change | SUPPORTING_ONLY | not-declared |  |
| fin_yoy | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| fin_zscore_history | SUPPORTING_ONLY | not-declared |  |
| fin_zscore_vs_prior_history | SUPPORTING_ONLY | not-declared |  |
| fiscal_accrual_quality | SUPPORTING_ONLY | not-declared |  |
| fiscal_ar_resid_std | SUPPORTING_ONLY | not-declared |  |
| fiscal_asymmetric_elasticity | SUPPORTING_ONLY | not-declared |  |
| fiscal_autocorr | SUPPORTING_ONLY | not-declared |  |
| fiscal_change_direction_agreement | SUPPORTING_ONLY | not-declared |  |
| fiscal_direction_consistency | SUPPORTING_ONLY | not-declared |  |
| fiscal_pair_direction_agreement | SUPPORTING_ONLY | not-declared |  |
| fiscal_perpetual_inventory | SUPPORTING_ONLY | not-declared |  |
| fiscal_regression_resid_std | SUPPORTING_ONLY | not-declared |  |
| fiscal_reversal_ratio | SUPPORTING_ONLY | not-declared |  |
| fiscal_sign_agreement | SUPPORTING_ONLY | not-declared |  |
| fiscal_sign_consistency | SUPPORTING_ONLY | not-declared |  |
| fiscal_standardized_surprise | SUPPORTING_ONLY | not-declared |  |
| fiscal_true_streak | SUPPORTING_ONLY | not-declared |  |
| fundamental_staleness | SUPPORTING_ONLY | not-declared |  |
| period_average | SUPPORTING_ONLY | not-declared |  |
| period_cagr | SUPPORTING_ONLY | not-declared |  |
| period_change | SUPPORTING_ONLY | not-declared |  |
| period_lag | SUPPORTING_ONLY | not-declared |  |
| period_stability | SUPPORTING_ONLY | not-declared |  |
| piotroski_f_score | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| piotroski_f_score_tolerant | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| piotroski_observed_count | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| piotroski_partial_score | SUPPORTING_ONLY | flow_type:SinglePeriodFlow |  |
| quarter_from_cumulative | SUPPORTING_ONLY | not-declared |  |
| report_change_breadth | SUPPORTING_ONLY | not-declared |  |
| report_change_coherence | SUPPORTING_ONLY | not-declared |  |
| report_filing_delay_surprise | SUPPORTING_ONLY | not-declared |  |
| report_revision_magnitude | SUPPORTING_ONLY | not-declared |  |
| report_rolling_mean | SUPPORTING_ONLY | not-declared |  |
| report_yoy_lag | SUPPORTING_ONLY | not-declared |  |
| revision_delta | SUPPORTING_ONLY | not-declared |  |
| ttm_from_cumulative | SUPPORTING_ONLY | not-declared |  |
| ttm_from_quarterly | SUPPORTING_ONLY | not-declared |  |
| yoy_by_period | SUPPORTING_ONLY | not-declared |  |
| zmijewski_score | SUPPORTING_ONLY | not-declared |  |
