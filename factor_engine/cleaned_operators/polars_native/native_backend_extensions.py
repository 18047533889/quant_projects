"""Central registration of pandas-free Polars backend extensions.

Keep this load-order catalog separate from the operator implementations and
install it before the registry freezes and evidence overlays are applied.
"""
from __future__ import annotations


def register_polars_native_extensions() -> None:
    """Register validated extensions in their existing deterministic order."""
    from factor_engine.cleaned_operators.polars_native.orderflow_signed_native_20260928 import register_orderflow_signed_native_20260928
    register_orderflow_signed_native_20260928()

    from factor_engine.cleaned_operators.polars_native.listing_age_expr_20260928 import register_listing_age_expr_20260928
    register_listing_age_expr_20260928()

    from factor_engine.cleaned_operators.polars_native.piotroski_native_batch_a_20260928 import register_piotroski_native_batch_a_20260928
    register_piotroski_native_batch_a_20260928()

    from factor_engine.cleaned_operators.polars_native.rolling_beta_native_20260928 import register_rolling_beta_native_20260928
    register_rolling_beta_native_20260928()

    from factor_engine.cleaned_operators.polars_native.fiscal_growth_batch_a_20260928 import register_fiscal_growth_batch_a_20260928
    register_fiscal_growth_batch_a_20260928()

    from factor_engine.cleaned_operators.polars_native.fiscal_persistence_batch_a_20260928 import register_fiscal_persistence_batch_a_20260928
    register_fiscal_persistence_batch_a_20260928()

    from factor_engine.cleaned_operators.polars_native.liquidity_decay_native_20260928 import register_liquidity_decay_native_20260928
    register_liquidity_decay_native_20260928()

    from factor_engine.cleaned_operators.polars_native.report_state_batch_a_20260928 import register_report_state_batch_a_20260928
    register_report_state_batch_a_20260928()

    from factor_engine.cleaned_operators.polars_native.volregime_native_20260928 import register_volregime_native_20260928
    register_volregime_native_20260928()

    from factor_engine.cleaned_operators.polars_native.fiscal_history_stats_batch_a_20260928 import register_fiscal_history_stats_batch_a_20260928
    register_fiscal_history_stats_batch_a_20260928()

    from factor_engine.cleaned_operators.polars_native.earnings_dynamics_native_20260928 import register_earnings_dynamics_native_20260928
    register_earnings_dynamics_native_20260928()

    from factor_engine.cleaned_operators.polars_native.fiscal_delta_slope_native_20260928 import register_fiscal_delta_slope_native_20260928
    register_fiscal_delta_slope_native_20260928()

    from factor_engine.cleaned_operators.polars_native.group_feature_stats_native_20260928 import register_group_feature_stats_native_20260928
    register_group_feature_stats_native_20260928()

    from factor_engine.cleaned_operators.polars_native.kama_distance_native_20260928 import register_kama_distance_native_20260928
    register_kama_distance_native_20260928()

    from factor_engine.cleaned_operators.polars_native.psar_derivatives_native_20260928 import register_psar_derivatives_native_20260928
    register_psar_derivatives_native_20260928()

    from factor_engine.cleaned_operators.polars_native.supertrend_derivatives_native_20260928 import register_supertrend_derivatives_native_20260928
    register_supertrend_derivatives_native_20260928()

    from factor_engine.cleaned_operators.polars_native.financial_revision_ledger_native_20260928 import register_financial_revision_ledger_native_20260928
    register_financial_revision_ledger_native_20260928()

    from factor_engine.cleaned_operators.polars_native.technical_rolling_native_20260928 import register_technical_rolling_native_20260928
    register_technical_rolling_native_20260928()

    from factor_engine.cleaned_operators.polars_native.klinger_stochrsi_native_20260928 import register_klinger_stochrsi_native_20260928
    register_klinger_stochrsi_native_20260928()

    from factor_engine.cleaned_operators.polars_native.aroon_native_20260928 import register_aroon_native_20260928
    register_aroon_native_20260928()

    from factor_engine.cleaned_operators.polars_native.fiscal_regression_quality_native_20260928 import register_fiscal_regression_quality_native_20260928
    register_fiscal_regression_quality_native_20260928()

    from factor_engine.cleaned_operators.polars_native.operating_accruals_native_20260928 import register_operating_accruals_native_20260928
    register_operating_accruals_native_20260928()

    from factor_engine.cleaned_operators.polars_native.earnings_state_native_20260928 import register_earnings_state_native_20260928
    register_earnings_state_native_20260928()

    from factor_engine.cleaned_operators.polars_native.valuation_state_native_20260928 import register_valuation_state_native_20260928
    register_valuation_state_native_20260928()

    from factor_engine.cleaned_operators.polars_native.report_asof_native_20260928 import register_report_asof_native_20260928
    register_report_asof_native_20260928()

    from factor_engine.cleaned_operators.polars_native.time_semantics import register_time_semantic_expr_backends
    register_time_semantic_expr_backends()


    from factor_engine.cleaned_operators.polars_native.group_policy import register_group_policy_native
    register_group_policy_native()

    from factor_engine.cleaned_operators.polars_native.calendar_seasonal_native_20260929 import register_calendar_seasonal_native_20260929
    register_calendar_seasonal_native_20260929()

    from factor_engine.cleaned_operators.polars_native.tech_osc_native_20260929 import register_tech_osc_native_20260929
    register_tech_osc_native_20260929()

    from factor_engine.cleaned_operators.polars_native.event_window_asof_native_20260929 import register_event_window_asof_native_20260929
    register_event_window_asof_native_20260929()

    from factor_engine.cleaned_operators.polars_native.group_peer_diffusion_20260929 import register_group_peer_diffusion_native
    register_group_peer_diffusion_native()

    from factor_engine.cleaned_operators.polars_native.cs_isolation_forest_polars_20260930 import register_cs_isolation_forest_polars_20260930
    register_cs_isolation_forest_polars_20260930()

    from factor_engine.cleaned_operators.polars_native.aq1_accrual_stability_numpy_20260930 import register_aq1_accrual_stability_numpy_20260930
    register_aq1_accrual_stability_numpy_20260930()

    from factor_engine.cleaned_operators.polars_native.group_return_dispersion_expr_20260929 import register_group_return_dispersion_expr_native
    register_group_return_dispersion_expr_native()

    from factor_engine.cleaned_operators.polars_native.credit_scores_wide_20260929 import register_credit_scores_wide
    register_credit_scores_wide()

    from factor_engine.cleaned_operators.polars_native.valuation_rank_expr_20260929 import register_valuation_rank_expr_native
    register_valuation_rank_expr_native()

    from factor_engine.cleaned_operators.polars_native.group_leader_laggard_expr_20260929 import register_group_leader_laggard_expr_native
    register_group_leader_laggard_expr_native()

    from factor_engine.cleaned_operators.polars_native.exself_mad_native_20260929 import register_exself_mad_native_20260929
    register_exself_mad_native_20260929()

    from factor_engine.cleaned_operators.polars_native.same_calendar_month_expr_20260929 import register_same_calendar_month_expr_native
    register_same_calendar_month_expr_native()

    from factor_engine.cleaned_operators.polars_native.ts_regression_residual_expr_20260929 import register_ts_regression_residual_expr_native
    register_ts_regression_residual_expr_native()

    from factor_engine.cleaned_operators.polars_native.m1_ranked_fast_expr_20260929 import register_m1_ranked_fast_native
    register_m1_ranked_fast_native()

    from factor_engine.cleaned_operators.polars_native.ewm_pair_expr_20260929 import register_ewm_pair_expr_native
    register_ewm_pair_expr_native()

    from factor_engine.cleaned_operators.polars_native.m1_dispersion_native_20260929 import register_m1_cs_momentum_dispersion_native
    register_m1_cs_momentum_dispersion_native()

    from factor_engine.cleaned_operators.polars_native.technical_vr_chunked_20260929 import register_vr_chunked_native
    register_vr_chunked_native()

    from factor_engine.cleaned_operators.polars_native.spectral_lowpass_expr_native_20260929 import register_spectral_lowpass_expr_native_20260929
    register_spectral_lowpass_expr_native_20260929()

    from factor_engine.cleaned_operators.polars_native.cs_rotation_native_20260929 import register_cs_rotation_native_20260929
    register_cs_rotation_native_20260929()

    from factor_engine.cleaned_operators.polars_native.weekday_effect_strength_expr_20260929 import register_weekday_effect_strength_expr_20260929
    register_weekday_effect_strength_expr_20260929()

    from factor_engine.cleaned_operators.polars_native.group_ts_decay_expr_20260929 import register_group_ts_decay_expr_native
    register_group_ts_decay_expr_native()

    from factor_engine.cleaned_operators.polars_native.ewm_corr_expr_20260929 import register_ewm_corr_native
    register_ewm_corr_native()

    from factor_engine.cleaned_operators.polars_native.cs_rank_combined_expr_20260929 import register_cs_rank_combined_churn_native_20260929
    register_cs_rank_combined_churn_native_20260929()

    from factor_engine.cleaned_operators.polars_native.cs_bucket_shrink_native_20260928 import register_cs_bucket_shrink_native_20260928
    register_cs_bucket_shrink_native_20260928(("cs_factor_bucket_return",))

    from factor_engine.cleaned_operators.polars_native.group_signal_attraction_expr_20260929 import register_group_signal_attraction_expr_native
    register_group_signal_attraction_expr_native()

    from factor_engine.cleaned_operators.polars_native.peer import register_peer_group_expr
    register_peer_group_expr(("relative_strength_group_pct",))


    from factor_engine.cleaned_operators.polars_native.intraday_next_stage_long_expr_20261001 import register_intraday_next_stage_long_expr_20261001
    register_intraday_next_stage_long_expr_20261001()
    from factor_engine.cleaned_operators.polars_native.intraday_lagcorr_long_expr_20260929 import register_intraday_lagcorr_long_expr_20260929
    register_intraday_lagcorr_long_expr_20260929()


    from factor_engine.cleaned_operators.polars_native.r69_native_batchB import register_r69_batch_b
    register_r69_batch_b()
