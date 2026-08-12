# FactorEngine 全部新增/候选算子 Master 清单
## R41–R47 156 候选 + 后续补充 + Filter/State-space + Fiscal/Relation Backlog

**目标仓库：** `18047533889/quant_projects`  
**本次整理日期：** 2026-08-12  
**本次核验时最新可见 main HEAD：** `854bdc2278678e3db03c894c059cd5fdbb7dac1b`  

> 这是一份 **Master Preflight Universe**，不是“把下面 223 个全部重新实现”的指令。
> 旧 R41–R47 总规范本身就明确：156 个是 future-operator recommendations / 候选池，执行时必须先对 current main 做 exact / alias / semantic-equivalent / composable / true-gap 分类。
> 当前 main 已经持续新增算子，因此 AI 必须以执行时最新 HEAD 为代码真值。

## 0. 数量口径

- R41–R47 历史候选：**156**
- 旧总规范额外补充 primitive：**8**
- R47 后新增 Filter / State-space / Shrinkage 候选：**28**
- 财务 PIT / 关系型补充 backlog：**31**（与历史候选有少量语义重叠，需 preflight）
- exact-name 去重后的“待检查候选 universe”：**223**
- 另有当前已实现、但要求重新生产化/认证的近期算子：**22**
- 本文件总 review universe：**245 个 canonical/proposed canonical 名称**

## 1. AI 执行时的状态分类

每个名称只能归入以下之一：

```text
EXISTING_EXACT
EXISTING_ALIAS
EXISTING_EQUIVALENT
COMPOSABLE_NO_NEW_OPERATOR
TRUE_GAP_IMPLEMENT
PROD_RECERTIFY
RESEARCH_ONLY_IMPLEMENT
BLOCKED_BY_DATA_CONTRACT
REJECT_FIELD_MISMATCH
REJECT_LOOKAHEAD
ARCHITECTURE_SUPERSEDED
```

**只有 `TRUE_GAP_IMPLEMENT` 和明确批准的 `RESEARCH_ONLY_IMPLEMENT` 才能新增 runtime canonical。**

---

# 2.1 R41 历史候选（31）

| # | candidate | preflight default |
|---:|---|---|
| 1 | `fiscal_acceleration` | CURRENT_HEAD_RECHECK |
| 2 | `fiscal_delta` | CURRENT_HEAD_RECHECK |
| 3 | `fiscal_lag` | CURRENT_HEAD_RECHECK |
| 4 | `fiscal_pct_change` | CURRENT_HEAD_RECHECK |
| 5 | `intra_market_profile_corr_ex_self` | CURRENT_HEAD_RECHECK |
| 6 | `report_asof` | ARCHITECTURE_SUPERSEDED 倾向 |
| 7 | `same_calendar_month_return` | CURRENT_HEAD_RECHECK |
| 8 | `same_clock_lag` | CURRENT_HEAD_RECHECK |
| 9 | `cn_sma` | CURRENT_HEAD_RECHECK |
| 10 | `cs_isolation_forest_score` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 11 | `event_window_return_asof` | BLOCKED/CONDITIONAL |
| 12 | `financial_snapshot_lag` | EXISTING_EQUIVALENT/ALIAS 倾向 |
| 13 | `fiscal_capital_stock` | CURRENT_HEAD_RECHECK |
| 14 | `fiscal_rolling_regression` | CURRENT_HEAD_RECHECK |
| 15 | `fiscal_rolling_std` | CURRENT_HEAD_RECHECK |
| 16 | `intraday_value_at_extreme_state` | CURRENT_HEAD_RECHECK |
| 17 | `panel_peer_graph_aggregate` | CURRENT_HEAD_RECHECK |
| 18 | `price_delay_score` | CURRENT_HEAD_RECHECK |
| 19 | `winsorized_ratio` | CURRENT_HEAD_RECHECK |
| 20 | `HMA` | CURRENT_HEAD_RECHECK |
| 21 | `KAMA` | CURRENT_HEAD_RECHECK |
| 22 | `QQE` | CURRENT_HEAD_RECHECK |
| 23 | `RSX` | CURRENT_HEAD_RECHECK |
| 24 | `WMA` | CURRENT_HEAD_RECHECK |
| 25 | `cs_factor_bucket_return` | CURRENT_HEAD_RECHECK |
| 26 | `pastor_stambaugh_beta` | CURRENT_HEAD_RECHECK |
| 27 | `same_calendar_day_mean` | CURRENT_HEAD_RECHECK |
| 28 | `ALMA` | CURRENT_HEAD_RECHECK |
| 29 | `CoppockCurve` | CURRENT_HEAD_RECHECK |
| 30 | `ElderRay` | CURRENT_HEAD_RECHECK |
| 31 | `FisherTransform` | CURRENT_HEAD_RECHECK |

# 2.2 R42 历史候选（25）

| # | candidate | preflight default |
|---:|---|---|
| 32 | `intra_same_slot_zscore` | CURRENT_HEAD_RECHECK |
| 33 | `intra_neighbor_event_class` | CURRENT_HEAD_RECHECK |
| 34 | `intra_state_count` | CURRENT_HEAD_RECHECK |
| 35 | `intra_state_sum` | CURRENT_HEAD_RECHECK |
| 36 | `intra_state_vwap` | CURRENT_HEAD_RECHECK |
| 37 | `intra_state_interval_moment` | CURRENT_HEAD_RECHECK |
| 38 | `intra_state_follow_ratio` | CURRENT_HEAD_RECHECK |
| 39 | `intra_state_follow_beta` | CURRENT_HEAD_RECHECK |
| 40 | `intra_state_follow_corr` | CURRENT_HEAD_RECHECK |
| 41 | `intra_state_pair_same_slot_corr` | CURRENT_HEAD_RECHECK |
| 42 | `intra_range_gap_flag` | CURRENT_HEAD_RECHECK |
| 43 | `intra_volume_peak_ridge_valley_state` | CURRENT_HEAD_RECHECK |
| 44 | `intra_price_peak_ridge_valley_state` | CURRENT_HEAD_RECHECK |
| 45 | `intra_smart_money_vwap_ratio` | CURRENT_HEAD_RECHECK |
| 46 | `intra_smart_money_fcm_score` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 47 | `panel_apm_residual_tstat` | CURRENT_HEAD_RECHECK |
| 48 | `panel_day_night_beta_gap` | CURRENT_HEAD_RECHECK |
| 49 | `panel_async_beta_ex_self` | CURRENT_HEAD_RECHECK |
| 50 | `intra_jump_wavelet_morphology` | CURRENT_HEAD_RECHECK |
| 51 | `intra_cojump_breadth_ex_self` | CURRENT_HEAD_RECHECK |
| 52 | `cs_topological_anomaly_score` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 53 | `panel_predictability_mosaic_score` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 54 | `panel_factor_pocket_strength` | CURRENT_HEAD_RECHECK |
| 55 | `intra_price_shape_cosine_match` | CURRENT_HEAD_RECHECK |
| 56 | `intra_distribution_moment` | CURRENT_HEAD_RECHECK |

# 2.3 R43 历史候选（11）

| # | candidate | preflight default |
|---:|---|---|
| 57 | `intra_slice_mask_reduce` | CURRENT_HEAD_RECHECK |
| 58 | `intra_slice_mask_pair_reduce` | CURRENT_HEAD_RECHECK |
| 59 | `intra_multiresolution_resample_reduce` | CURRENT_HEAD_RECHECK |
| 60 | `panel_intraday_ordered_reduce` | CURRENT_HEAD_RECHECK |
| 61 | `ts_ewm_std` | CURRENT_HEAD_RECHECK |
| 62 | `panel_ewm_beta_ex_self` | CURRENT_HEAD_RECHECK |
| 63 | `panel_cmra_ex_self` | CURRENT_HEAD_RECHECK |
| 64 | `intra_idiosyncratic_semivariance_balance_ex_self` | CURRENT_HEAD_RECHECK |
| 65 | `panel_similarity_crowding_score` | CURRENT_HEAD_RECHECK |
| 66 | `panel_cluster_risk_score` | CURRENT_HEAD_RECHECK |
| 67 | `intra_functional_beta_profile_ex_self` | CURRENT_HEAD_RECHECK |

# 2.4 R44 历史候选（9）

| # | candidate | preflight default |
|---:|---|---|
| 68 | `fiscal_cost_stickiness_score` | CURRENT_HEAD_RECHECK |
| 69 | `fundamental_cash_flow_duration` | CURRENT_HEAD_RECHECK |
| 70 | `intra_round_price_clustering_share` | CURRENT_HEAD_RECHECK |
| 71 | `intra_round_price_barrier_response` | CURRENT_HEAD_RECHECK |
| 72 | `intra_session_segment_reduce` | CURRENT_HEAD_RECHECK |
| 73 | `intra_session_boundary_jump` | CURRENT_HEAD_RECHECK |
| 74 | `trading_calendar_mask` | CURRENT_HEAD_RECHECK |
| 75 | `suspension_restart_response` | CURRENT_HEAD_RECHECK |
| 76 | `intra_limit_pre_hit_pressure_profile` | CURRENT_HEAD_RECHECK |

# 2.5 R45 历史候选（53）

| # | candidate | preflight default |
|---:|---|---|
| 77 | `intra_impulse_event_detector` | CURRENT_HEAD_RECHECK |
| 78 | `intra_post_impulse_response` | CURRENT_HEAD_RECHECK |
| 79 | `intra_probe_outcome_score` | CURRENT_HEAD_RECHECK |
| 80 | `intra_supply_absorption_score` | CURRENT_HEAD_RECHECK |
| 81 | `intra_consolidation_quality` | CURRENT_HEAD_RECHECK |
| 82 | `intra_response_curve_features` | CURRENT_HEAD_RECHECK |
| 83 | `intra_volume_at_price_profile` | CURRENT_HEAD_RECHECK |
| 84 | `intra_volume_profile_peak_geometry` | CURRENT_HEAD_RECHECK |
| 85 | `intra_volume_profile_supply_structure` | CURRENT_HEAD_RECHECK |
| 86 | `turnover_chip_distribution` | CURRENT_HEAD_RECHECK |
| 87 | `turnover_chip_distribution_transport` | CURRENT_HEAD_RECHECK |
| 88 | `turnover_chip_age_cost_surface` | CURRENT_HEAD_RECHECK |
| 89 | `intra_log_signature_features` | CURRENT_HEAD_RECHECK |
| 90 | `intra_functional_pca_shape` | CURRENT_HEAD_RECHECK |
| 91 | `intra_functional_motif_score` | CURRENT_HEAD_RECHECK |
| 92 | `intra_shapelet_match` | CURRENT_HEAD_RECHECK |
| 93 | `intra_rqa_features` | CURRENT_HEAD_RECHECK |
| 94 | `intra_persistent_homology_features` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 95 | `intra_topological_anomaly_score` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 96 | `intra_optimal_transport_profile_shift` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 97 | `intra_dmd_koopman_features` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 98 | `intra_kalman_latent_price` | CURRENT_HEAD_RECHECK |
| 99 | `intra_state_space_volume_components` | CURRENT_HEAD_RECHECK |
| 100 | `intra_hmm_state_features` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 101 | `intra_hsmm_duration_features` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 102 | `intra_change_point_sequence_features` | CURRENT_HEAD_RECHECK |
| 103 | `intra_hawkes_event_features` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 104 | `intra_multifractal_spectrum` | CURRENT_HEAD_RECHECK |
| 105 | `intra_wavelet_scattering_features` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 106 | `intra_emd_hilbert_huang_features` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 107 | `intra_visibility_graph_features` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 108 | `intra_information_flow_features` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 109 | `intra_covariance_manifold_shift` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 110 | `intra_diffusion_map_state` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 111 | `intra_common_trading_intensity` | CURRENT_HEAD_RECHECK |
| 112 | `intra_price_efficiency_state_space` | CURRENT_HEAD_RECHECK |
| 113 | `intra_neural_cde_embedding` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 114 | `intra_contrastive_path_embedding` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 115 | `intra_matrix_profile_session_features` | CURRENT_HEAD_RECHECK |
| 116 | `intra_dtw_archetype_features` | CURRENT_HEAD_RECHECK |
| 117 | `intra_local_conditional_entropy` | CURRENT_HEAD_RECHECK |
| 118 | `intra_business_time_deformation` | CURRENT_HEAD_RECHECK |
| 119 | `intra_quantile_dependence_features` | CURRENT_HEAD_RECHECK |
| 120 | `intra_kramers_moyal_dynamics` | CURRENT_HEAD_RECHECK |
| 121 | `intra_extreme_event_interval_memory` | CURRENT_HEAD_RECHECK |
| 122 | `intra_signature_lead_lag_network` | CURRENT_HEAD_RECHECK |
| 123 | `intra_validated_lead_lag_network` | CURRENT_HEAD_RECHECK |
| 124 | `intra_dynamic_stock_graph_features` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 125 | `intra_tensor_common_mode` | CURRENT_HEAD_RECHECK |
| 126 | `intra_topological_peer_anomaly` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 127 | `intra_critical_transition_score` | CURRENT_HEAD_RECHECK |
| 128 | `intra_realized_measure_state_vector` | CURRENT_HEAD_RECHECK |
| 129 | `intra_symbolic_dynamics_features` | CURRENT_HEAD_RECHECK |

# 2.6 R46 历史候选（20）

| # | candidate | preflight default |
|---:|---|---|
| 130 | `intra_eod_reversal_decomposition` | CURRENT_HEAD_RECHECK |
| 131 | `intra_volume_shock_state` | CURRENT_HEAD_RECHECK |
| 132 | `intra_comovement_curve_ex_self` | CURRENT_HEAD_RECHECK |
| 133 | `intra_price_volume_cross_wavelet` | CURRENT_HEAD_RECHECK |
| 134 | `intra_bicoherence_features` | CURRENT_HEAD_RECHECK |
| 135 | `intra_frequency_granger_price_volume` | CURRENT_HEAD_RECHECK |
| 136 | `intra_range_competition_profile` | CURRENT_HEAD_RECHECK |
| 137 | `intra_absorption_curve_area` | CURRENT_HEAD_RECHECK |
| 138 | `intra_volume_profile_value_area` | CURRENT_HEAD_RECHECK |
| 139 | `turnover_chip_overhang_surface` | CURRENT_HEAD_RECHECK |
| 140 | `fundamental_latent_balance_sheet_factor` | CURRENT_HEAD_RECHECK |
| 141 | `fundamental_working_capital_financing_state` | CURRENT_HEAD_RECHECK |
| 142 | `fundamental_cost_stickiness_panel` | CURRENT_HEAD_RECHECK |
| 143 | `intra_multiscale_state_residence` | CURRENT_HEAD_RECHECK |
| 144 | `intra_visibility_motif_transition` | CURRENT_HEAD_RECHECK |
| 145 | `intra_recurrence_network_features` | CURRENT_HEAD_RECHECK |
| 146 | `intra_transfer_operator_metastability` | CURRENT_HEAD_RECHECK |
| 147 | `intra_local_drift_diffusion_surface` | CURRENT_HEAD_RECHECK |
| 148 | `intra_session_ot_map` | CURRENT_HEAD_RECHECK |
| 149 | `panel_intraday_low_rank_residual_ex_self` | CURRENT_HEAD_RECHECK |

# 2.7 R47 历史候选（7）

| # | candidate | preflight default |
|---:|---|---|
| 150 | `fin_schema_gate` | BLOCKED_BY_DATA_CONTRACT |
| 151 | `ts_lagged_predictability_score` | CURRENT_HEAD_RECHECK |
| 152 | `cs_predictability_mosaic_score` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 153 | `intra_functional_autoencoder_score` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 154 | `intra_hmm_posterior_entropy` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |
| 155 | `intra_liquidity_resilience_curve_fit` | CURRENT_HEAD_RECHECK |
| 156 | `intra_function_on_function_anomaly_response` | RESEARCH_FIRST / CURRENT_HEAD_RECHECK |

# 3. 旧总规范后补的 8 个 generic primitives

- `intra_event_window_reduce` — **CURRENT_HEAD_RECHECK**
- `intra_event_pre_post_contrast` — **CURRENT_HEAD_RECHECK**
- `ts_event_decay_kernel` — **CURRENT_HEAD_RECHECK**
- `intra_state_transition_entropy` — **CURRENT_HEAD_RECHECK**
- `intra_state_dwell_stats` — **CURRENT_HEAD_RECHECK**
- `cs_robust_mahalanobis_score` — **CURRENT_HEAD_RECHECK**
- `intra_piecewise_linear_path_features` — **CURRENT_HEAD_RECHECK**
- `ts_online_change_point_score` — **RESEARCH_FIRST**

# 4. R47 以后新补的 Filter / State-space / Shrinkage 候选（28）

这部分是后续针对低相关因子、信号稳健性、成本可交易性、状态切换和横截面去噪新增的建议。

| candidate | priority | disposition default |
|---|---|---|
| `state_change_point_adaptive_ema` | P0 | TRUE_GAP_RECHECK |
| `state_filter_reset_on_break` | P0 | TRUE_GAP_RECHECK |
| `state_gain_scheduler` | P0 | TRUE_GAP_RECHECK |
| `state_elastic_turnover_prox` | P0 | TRUE_GAP_RECHECK |
| `cs_shrink_to_market_mean` | P0 | TRUE_GAP_RECHECK |
| `cs_shrink_to_group_mean` | P0 | TRUE_GAP_RECHECK |
| `cs_empirical_bayes_shrinkage` | P0 | TRUE_GAP_RECHECK |
| `ts_l1_trend_filter_trailing` | P0 | TRUE_GAP_RECHECK |
| `ts_total_variation_filter_trailing` | P0 | TRUE_GAP_RECHECK |
| `ts_robust_kalman_level` | P0 | TRUE_GAP_RECHECK |
| `ts_one_euro_filter` | P1 | TRUE_GAP_RECHECK |
| `ts_alpha_beta_filter` | P1 | TRUE_GAP_RECHECK |
| `ts_bessel_lowpass_causal` | P1 | TRUE_GAP_RECHECK |
| `ts_fir_lowpass_causal` | P1 | TRUE_GAP_RECHECK |
| `ts_causal_savgol_endpoint` | P1 | TRUE_GAP_RECHECK |
| `ts_vidya` | P1 | TRUE_GAP_RECHECK |
| `ts_mcginley_dynamic` | P1 | TRUE_GAP_RECHECK |
| `ts_nlms_filter` | Research/P1-P2 | RESEARCH_ONLY_RECHECK |
| `ts_rls_filter` | Research/P1-P2 | RESEARCH_ONLY_RECHECK |
| `ts_student_t_kalman_filter` | Research/P1-P2 | RESEARCH_ONLY_RECHECK |
| `ts_adaptive_noise_kalman` | Research/P1-P2 | RESEARCH_ONLY_RECHECK |
| `ts_wavelet_shrinkage_trailing` | P1 | TRUE_GAP_RECHECK |
| `ts_ssa_denoise_trailing` | P1 | TRUE_GAP_RECHECK |
| `cs_peer_graph_smooth` | Research/P1-P2 | RESEARCH_ONLY_RECHECK |
| `ts_h_infinity_level_filter` | Research/P1-P2 | RESEARCH_ONLY_RECHECK |
| `ts_particle_filter_level` | Research/P1-P2 | PROPOSED_NAME / RESEARCH_ONLY_RECHECK |
| `ts_modwt_denoise_trailing` | Research/P1-P2 | RESEARCH_ONLY_RECHECK |
| `ts_spectral_lowpass_trailing` | Research/P1-P2 | RESEARCH_ONLY_RECHECK |

# 5. 财务 PIT / 关系型 / 横截面补充 backlog

这些来自此前基本面/关系型 operator contract 与后续补库需求。最新 main 已经出现 relation/robust 相关提交，因此**必须重新检查，不得按旧状态直接实现**。

| candidate | special note |
|---|---|
| `row_sum_skipna` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `industry_size_neutralize` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `fiscal_perpetual_inventory` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `fiscal_standardized_surprise` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `fiscal_direction_consistency` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `fiscal_pair_direction_agreement` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `fiscal_autocorr` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `fiscal_ar_resid_std` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `fiscal_rolling_regression_resid` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `fiscal_asymmetric_elasticity` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `industry_fiscal_resid` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `fiscal_accrual_quality` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `accounting_comparability_score` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `fiscal_asymmetric_timeliness` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `fiscal_reversal_ratio` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `fiscal_logit_score` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `cash_flow_lifecycle_stage` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `date_diff_days` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `fin_seasonal_zscore` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `fin_seasonal_percentile` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `fiscal_true_streak` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `group_cs_resid` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `holder_concentration_change` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `laborforce_efficiency` | BLOCKED_BY_DATA_CONTRACT：当前 A 股字段字典缺 employee_count。 |
| `relation_entropy` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `relation_jaccard` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `relation_period_change` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `revision_delta` | BLOCKED_BY_DATA_CONTRACT / RECHECK：必须保留真实 vintage/revision identity。 |
| `years_since_date` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `fundamental_staleness_days` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |
| `fiscal_rolling_slope` | CURRENT_HEAD_RECHECK；若已有 exact/equivalent 则跳过。 |

# 6. 已经存在，但必须做生产化/重新认证的近期算子（22）

下面这些**不是新增 canonical backlog**。它们已经在近期 Filter/Robust 层出现；任务是检查真实 Polars backend、numeric parity、PIT、checkpoint/resume、parameter evidence 和 production no-pandas-fallback。

- `ts_hampel_filter_causal` — **PROD_RECERTIFY，不要重复造轮子**
- `ts_median3_causal` — **PROD_RECERTIFY，不要重复造轮子**
- `ts_rolling_median_causal` — **PROD_RECERTIFY，不要重复造轮子**
- `ts_robust_ema` — **PROD_RECERTIFY，不要重复造轮子**
- `ts_super_smoother` — **PROD_RECERTIFY，不要重复造轮子**
- `ts_kama` — **PROD_RECERTIFY，不要重复造轮子**
- `ts_butterworth_lowpass_causal` — **PROD_RECERTIFY，不要重复造轮子**
- `ts_causal_local_linear_smoother` — **PROD_RECERTIFY，不要重复造轮子**
- `state_adaptive_deadband` — **PROD_RECERTIFY，不要重复造轮子**
- `state_rank_deadband` — **PROD_RECERTIFY，不要重复造轮子**
- `state_quantile_hysteresis` — **PROD_RECERTIFY，不要重复造轮子**
- `state_adaptive_slew_limit` — **PROD_RECERTIFY，不要重复造轮子**
- `state_l1_turnover_prox` — **PROD_RECERTIFY，不要重复造轮子**
- `state_l2_partial_adjustment` — **PROD_RECERTIFY，不要重复造轮子**
- `state_cost_aware_deadband` — **PROD_RECERTIFY，不要重复造轮子**
- `state_cost_aware_slew` — **PROD_RECERTIFY，不要重复造轮子**
- `state_confidence_weighted_ema` — **PROD_RECERTIFY，不要重复造轮子**
- `state_uncertainty_deadband` — **PROD_RECERTIFY，不要重复造轮子**
- `ts_quantile_range` — **PROD_RECERTIFY，不要重复造轮子**
- `ts_trimmed_mean` — **PROD_RECERTIFY，不要重复造轮子**
- `ts_robust_zscore_inclusive` — **PROD_RECERTIFY，不要重复造轮子**
- `ts_robust_zscore_prior` — **PROD_RECERTIFY，不要重复造轮子**

# 7. 已知容易重复/已被架构替代的名称

- `KAMA`：旧候选名；current Filter Layer 已有 `ts_kama`，优先做 semantic mapping。
- `price_delay_score`：current main 已出现 `ts_price_delay` 等相关能力，必须先比对数学语义。
- `report_asof`：PIT join 原则上应由 DataAccess 负责，通常倾向 `ARCHITECTURE_SUPERSEDED`。
- `financial_snapshot_lag`：优先映射真实 report-period/fiscal lag，不允许再写一套 daily-row shift。
- `fiscal_lag` / `fiscal_delta` / `fiscal_pct_change` / `fiscal_acceleration`：current main 已有大量 `period_*` / `fin_*` / fiscal family，必须 semantic preflight。
- `HMA` / `QQE` / `RSX` / `ALMA` / `CoppockCurve` / `ElderRay` / `FisherTransform`：当前仓库已有 technical.new_indicators 等模块，先 exact lookup。
- 股东、指数、涨跌停、Markov、first-passage、spectral、topology、intraday 等旧候选中，current main 已经实现了大量能力，不能按 2026-08-11 snapshot 重复开发。

# 8. 明确不要新增成 operator 的组合因子

下面即使有研究价值，也优先保留为 DSL factor expression，而不是新 canonical：

- `fundamental_disclosure_latency`
- `index_rebalance_pressure`
- `holder_persistence_weighted_concentration`
- `valuation_quality_combo`
- `growth_value_combo`
- `fundamental_market_confirmation`
- `market_breadth_alpha`
- `filter_disagreement_alpha`
- `filter_consensus_alpha`
- `cost_survivability_alpha`
- `topology_migration_alpha`

此外禁止：

- 只改 window / lag / threshold / quantile / decay 的复制算子；
- 只给已有 operator 换名字；
- 需要当前 A 股不存在的 Level2 / Tick / orderbook / news / analyst fields 的算子；
- centered rolling、negative shift、bfill、full-sample smoothing 等未来函数。

# 9. 对全部 223 个候选统一做 current-main preflight

AI 必须生成：

`factor_engine/evidence/operator_master_gap_preflight_<HEAD>.csv`

至少包含：

```text
candidate
source_round
current_exact
current_alias
semantic_equivalent
composable
replacement
current_surface
pandas_backend
polars_backend
duckdb_backend
field_legal
pit_legal
stateful
checkpointable
parameter_evidence
disposition
reason
```

# 10. 实现/认证底线

对最终确认为 `TRUE_GAP_IMPLEMENT` 的算子：

1. Pandas/NumPy 可作为清晰 reference semantics；
2. Daily production 原则上必须有**真实 Polars path**；
3. 禁止 `Polars -> pandas -> operator -> Polars` 假 backend；
4. DuckDB 只实现自然适合 SQL 的算子；
5. 所有时序算子做 future-poison / prefix causality；
6. 财务算子严格走 `PubDate` knowledge-time +真实 fiscal period identity；
7. 分钟全日信号默认 EOD 完成、最早下一交易日使用；
8. ex-self 必须真正扣除 self；
9. stateful 算子必须 batch = chunk = checkpoint/resume；
10. 不能把代码存在等同于 cold-start certified。

# 11. 建议的认证等级

```text
REGISTERED
FUNCTIONAL
EXTENDED_CERTIFIED
POLARS_CERTIFIED
INCREMENTAL_CERTIFIED
COLDSTART_CERTIFIED
```

只有最终 `COLDSTART_CERTIFIED` 才进入新的冷启动 operator contract。

# 12. 最终交付物

```text
operator_master_gap_preflight_<HEAD>.csv
implemented_true_gaps.csv
skipped_existing_equivalent.csv
blocked_by_data_contract.csv
research_only_operators.csv
recent_operator_backend_audit.csv
polars_parity.csv
state_checkpoint_parity.csv
parameter_domain_evidence.csv
CURRENT_HEAD_COLDSTART_OPERATOR_CONTRACT.csv
CURRENT_HEAD_COLDSTART_OPERATOR_CALLSHAPES.csv
CURRENT_HEAD_OPERATOR_RECERTIFICATION.json
IMPLEMENTATION_REPORT.md
```

# 13. 一句话执行指令

> **重新拉取最新 main → 对本文件 223 个候选 + 22 个已实现待认证算子做全量 preflight → 只实现 TRUE_GAP → 已有算子只修/认证不重复造 → 补 Polars/PIT/causality/checkpoint/evidence → 重生 current-head cold-start operator contract。**
