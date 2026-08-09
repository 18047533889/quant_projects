# -*- coding: utf-8 -*-
"""Semantic / temporal / source-contract certification for production admission.

``apply_production_hardening`` historically blanket-promoted every registered
operator to ``status=production`` with ``pit_safe=True`` as long as it had a
backend and was not on an explicit block list.  That conflates "registered and
can run" with "math and PIT semantics passed an audit", which is exactly the
over-promotion class this module removes.

Production admission is split into four independent certificates:

1. ``implementation_certified``  -- backend runs, no exception, correct shape;
   satisfied by evidence-backed execution (parity / golden tests).
2. ``semantic_certified``        -- formula matches name / docs / golden case.
3. ``temporal_certified``        -- outputs are invariant to future data and
   the current value obeys the availability contract.
4. ``source_contract_certified`` -- field, announcement time, effective time
   and universe all honour PIT.

``production_certified == implementation_certified AND semantic_certified AND
temporal_certified AND source_contract_certified``.  Pandas/Polars parity alone
only proves certificate (1); the evidence overlay remains the authority for the
runtime ``production_certified`` field.

Two classes of operator are fail-closed to ``experimental`` and
``pit_safe=False`` here (the "registered but not audited" set that the blanket
promotion used to swallow):

* operators registered with an explicit ``status="experimental"`` / research
  lifecycle (model families, next-stage ts_model / cross-section / intraday
  kernels) -- captured by ``snapshot_registered_statuses`` right after module
  registration, before any promotion layer rewrites the lifecycle field;
* an explicit ``ISOLATED_FROM_DEFAULT_MINING`` manifest of operators whose
  current semantics are known-defective (panel-contract violations, missing
  shareholder identity, unknown-state treated as 0, etc.) until reworked.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

# Snapshot of the raw registered lifecycle status, taken after the operator
# modules finish importing and before any promotion / hardening layer can
# rewrite ``catalog["status"]`` (see ``snapshot_registered_statuses``).
_REGISTERED_LIFECYCLE: dict[str, str] = {}


def snapshot_registered_statuses() -> None:
    """Capture the raw registered lifecycle status of every operator.

    Must run after the operator modules finish importing and before any
    promotion / hardening layer (``apply_operator_deduplication``,
    ``layer_governance*``, ``apply_production_hardening``) rewrites
    ``catalog["status"]``.
    """
    from cleaned_operators.registry import OperatorRegistry

    _REGISTERED_LIFECYCLE.clear()
    for canonical, catalog in OperatorRegistry._catalog.items():
        status = str(catalog.get("status") or "implemented")
        _REGISTERED_LIFECYCLE[canonical] = status


def registered_status(canonical: str) -> str | None:
    """Return the snapshot lifecycle status, or None if not registered."""
    return _REGISTERED_LIFECYCLE.get(canonical)


# Operators whose registration status was experimental but whose semantics have
# since been reworked and certified (S9 unknown-state rework; 2026-08 holder
# ShareholderId-matched rework; market-model PIT certification).  Hardening is
# allowed to promote them to production instead of keeping them fail-closed.
PROMOTED_OUT_OF_EXPERIMENTAL: frozenset[str] = frozenset({
    "index_reconstitution_churn",
    "listing_age",
    "suspension_frequency",
    # 2026-08: holder_* reworked to ShareholderId-matched union pair.
    "holder_weighted_churn",
    "holder_entry_share",
    "holder_exit_share",
    "holder_net_entry_share",
    "holder_rank_stability",
    # 2026-08: relation deltas verified pair-valid (PIT); multi-index intensity
    # reworked to unknown-state semantics.
    "relation_entry_count",
    "relation_exit_count",
    "relation_weighted_change",
    "multi_index_entry_intensity",
    # 2026-08: CAPM-family operators certified causal (trailing-window kernels).
    "tail_beta",
    "residual_momentum_capm",
    "coskewness_to_market",
    "idio_vol",
    "idio_skew",
    # 2026-08 final pack: complexity/entropy reimplemented by the reviewed
    # sequence_complexity module (overwrites the older ts_model.complexity
    # experimental registration; the research-surface membership is dropped
    # there too).
    "ts_permutation_entropy",
    "ts_sample_entropy",
    # 2026-08 production audit follow-up: 295 previously experimental-registered
    # daily operators verified runtime-clean by the factor-production audit
    # harness (deterministic, shape-preserving, prefix-causal on the synthetic
    # panels) and promoted to production.  The in-sample diagnostic family
    # (ts_*_regression_coeff/resid/slope without the *_prior / forecast_error
    # suffix, ts_ar_forecast/innovation*, ts_mean_reversion_half_life) stays
    # experimental by design; the promoted entries below are their causal
    # variants plus the reviewed relation/index/event/fundamental/shareholder/
    # intraday-minute families (audit §5-§12).
    "a_share_cap_ratio", "altman_z_score", "ashare_limit_distance", "ashare_limit_down_touch", "ashare_limit_failed", "ashare_limit_one_price", "ashare_limit_open_failed", "ashare_limit_up_touch",
    "ashare_open_at_upper_limit", "benchmark_excess_return", "benchmark_relative_price", "book_to_price", "calendar_day_diff", "capital_change_age", "capital_change_magnitude", "circulating_cap_unlock_proxy",
    "cs_actual_lof_score", "cs_knn_distance", "cs_local_density_score", "cs_mahalanobis_distance", "cs_quantile_resid", "cs_relative_density_ratio", "cs_residual_percentile", "cs_ridge_resid",
    "cs_robust_mahalanobis_mad", "cs_robust_resid", "cs_shrinkage_mahalanobis", "cs_spline_resid", "earnings_yield", "event_abnormal_return_past", "event_active_count", "event_arithmetic_return_sum",
    "event_cumulative_return_past", "event_decay_asof", "event_log_return_sum", "event_return_since_last", "fin_acquisition_cash_intensity", "fin_announcement_lag", "fin_applicability_mask", "fin_borrowing_intensity",
    "fin_capex_growth", "fin_capex_intensity", "fin_cash_burn_runway", "fin_cash_sales_divergence", "fin_cashflow_persistence", "fin_comprehensive_income_gap", "fin_contract_asset_growth", "fin_contract_asset_intensity",
    "fin_contract_asset_liability_gap", "fin_contract_liability_growth", "fin_contract_liability_intensity", "fin_core_earnings_ratio", "fin_debt_repayment_intensity", "fin_debt_service_coverage_proxy", "fin_deferred_tax_gap", "fin_delta_noa",
    "fin_discontinued_operation_ratio", "fin_earnings_cash_gap_volatility", "fin_earnings_persistence", "fin_earnings_smoothness", "fin_equity_capital_growth", "fin_expense_sales_divergence", "fin_fair_value_income_dependence", "fin_financing_gap",
    "fin_fundamental_strength_score", "fin_goodwill_intensity", "fin_goodwill_risk_score", "fin_impairment_intensity", "fin_interest_coverage_proxy", "fin_inventory_sales_divergence", "fin_investment_income_dependence", "fin_lease_asset_liability_gap",
    "fin_lease_intensity", "fin_margin_persistence", "fin_minority_profit_share", "fin_net_borrowing_cashflow", "fin_net_debt_issuance", "fin_noncore_income_ratio", "fin_oci_to_equity", "fin_other_earnings_dependence",
    "fin_rd_capitalization_ratio", "fin_rd_total_intensity", "fin_receivable_sales_divergence", "fin_roe_cash_gap", "fin_working_capital_accruals", "float_share_ratio", "free_float_ratio", "free_float_share_ratio",
    "free_float_turnover", "free_to_circulating_ratio", "group_ex_self_mean", "group_ex_self_weighted_mean", "group_leader_laggard_exposure", "group_multi_level_rank_consistency", "group_peer_beta_deviation", "group_peer_deviation_index",
    "group_peer_information_diffusion", "group_return_dispersion_exposure", "hierarchical_group_neutralize", "holder_class_entropy", "holder_common_holding_peer_return", "holder_concentration", "holder_float_concentration_gap", "holder_freeze_concentration",
    "holder_freeze_ratio", "holder_id_matched_churn", "holder_id_matched_entry_share", "holder_id_matched_exit_share", "holder_id_overlap_ratio", "holder_locked_share_ratio", "holder_nature_entropy", "holder_peer_return_breadth",
    "holder_pledge_change", "holder_pledge_churn", "holder_pledge_concentration", "holder_pledge_ratio", "holder_pledged_holder_count", "holder_share_weighted_rank_migration", "holder_shareholder_network_centrality", "holder_shareholder_overlap_ratio",
    "index_entry_exit_event", "index_event_decay", "index_member", "index_membership_age", "index_weight", "index_weight_change", "index_weight_gap_to_free_float", "intra_abs_return_profile_cosine",
    "intra_amihud", "intra_amount_profile_cosine", "intra_amount_profile_jsd", "intra_beta_asymmetry", "intra_bipower_variation", "intra_concentration", "intra_continuous_variance", "intra_down_down_semibeta",
    "intra_down_up_semibeta", "intra_drawdown_depth", "intra_drawdown_duration", "intra_drawdown_recovery_half_life", "intra_entropy", "intra_extreme_bar_return", "intra_high_time", "intra_idiosyncratic_kurtosis",
    "intra_idiosyncratic_kurtosis_ex_self", "intra_idiosyncratic_skewness", "intra_idiosyncratic_skewness_ex_self", "intra_idiosyncratic_variance", "intra_idiosyncratic_variance_ex_self", "intra_interval_amount_share", "intra_interval_illiquidity", "intra_interval_realized_variance",
    "intra_interval_return", "intra_interval_volume_share", "intra_interval_vwap_deviation", "intra_jump_clustering", "intra_jump_concentration", "intra_jump_count", "intra_jump_first_time", "intra_jump_last_time",
    "intra_jump_ratio", "intra_jump_variation", "intra_kyle_lambda_proxy", "intra_limit_duration", "intra_limit_first_hit_time", "intra_limit_reopen_count", "intra_longest_above_vwap_streak", "intra_longest_below_vwap_streak",
    "intra_low_time", "intra_lunch_gap_return", "intra_market_model_r2", "intra_market_model_r2_ex_self", "intra_max_drawdown", "intra_max_drawup", "intra_negative_jump_variation", "intra_negative_tail_variation",
    "intra_path_efficiency", "intra_positive_jump_variation", "intra_positive_tail_variation", "intra_price_vwap_max_negative_excursion", "intra_price_vwap_max_positive_excursion", "intra_profile_earth_mover_distance", "intra_realized_beta", "intra_realized_beta_ex_self",
    "intra_realized_correlation", "intra_realized_correlation_ex_self", "intra_realized_kurtosis", "intra_realized_quarticity", "intra_realized_semivariance", "intra_realized_skewness", "intra_realized_variance", "intra_return_activity_corr",
    "intra_return_profile_cosine", "intra_same_slot_momentum", "intra_same_slot_reversal", "intra_segment_amount_share", "intra_segment_realized_vol", "intra_segment_return", "intra_segment_volume_share", "intra_segment_vwap_deviation",
    "intra_signed_imbalance_proxy", "intra_signed_jump_ratio", "intra_signed_return_profile_cosine", "intra_signed_tail_variation_ratio", "intra_tail_event_count", "intra_time_above_vwap", "intra_tripower_quarticity", "intra_up_down_semibeta",
    "intra_up_up_semibeta", "intra_volume_profile_cosine", "intra_volume_profile_jsd", "intra_vwap_above_ratio", "intra_vwap_cross_count", "intra_vwap_path_curvature", "intra_vwap_path_curvature_pct", "intra_vwap_path_slope",
    "intra_vwap_path_slope_pct", "intra_vwap_reversion_speed", "market_cap_free_cap_gap", "piotroski_f_score", "real_turnover_rate", "relation_category_share", "relation_entropy", "relation_hhi",
    "relation_peer_weighted_mean_ex_self", "relation_rank_weighted_sum", "relation_topk_sum", "suspension_status_coverage", "tradable_state", "ts_abs_concentration", "ts_abs_entropy", "ts_ar_coeff_stability",
    "ts_ar_coefficient", "ts_ar_prior_coeff", "ts_ar_prior_forecast", "ts_ar_prior_innovation", "ts_ar_prior_innovation_z", "ts_best_lag_corr", "ts_beta_if", "ts_corr_if",
    "ts_current_drawdown_duration", "ts_cusum_break_score", "ts_downside_deviation", "ts_event_spacing_cv", "ts_event_spacing_mean", "ts_expectile_beta_spread", "ts_expectile_regression_coeff_prior", "ts_expectile_regression_forecast_error",
    "ts_gap_fill_ratio", "ts_gap_reversion_ratio", "ts_gap_survival_duration", "ts_huber_regression_coeff_prior", "ts_huber_regression_forecast_error", "ts_huber_regression_forecast_error_z", "ts_industry_liquidity_beta", "ts_level_shift_score",
    "ts_market_liquidity_beta", "ts_max_if", "ts_min_if", "ts_multi_regression_adjusted_r2_prior", "ts_multi_regression_coeff_prior", "ts_multi_regression_coeff_stability", "ts_multi_regression_forecast_error", "ts_multi_regression_forecast_error_z",
    "ts_multi_regression_r2_prior", "ts_negative_ratio", "ts_opening_mispricing_score", "ts_overnight_intraday_cov", "ts_overnight_intraday_sign_agreement", "ts_overnight_intraday_spread", "ts_positive_ratio", "ts_price_delay",
    "ts_quantile_beta_spread", "ts_quantile_if", "ts_quantile_range", "ts_regression_resid_if", "ts_ridge_regression_coeff_prior", "ts_ridge_regression_forecast_error", "ts_ridge_regression_forecast_error_z", "ts_robust_zscore",
    "ts_time_since_change", "ts_time_under_water", "ts_transition_count", "ts_trimmed_mean", "ts_upside_deviation", "ts_variance_ratio", "ts_variance_ratio_slope", "ts_vol_shift_score",
    "ts_zero_ratio", "valuation_cashflow_disagreement", "valuation_growth_mismatch", "valuation_pcf_definition_gap", "valuation_pe_ttm_lyr_gap", "valuation_quality_mismatch", "zmijewski_score",
    # 2026-08: holder concentration trend slope/acceleration verified against
    # the snapshot-aligned window (audit §6.8); snapshot_date fixture added.
    "holder_concentration_acceleration",
    "holder_concentration_slope",
})


# --------------------------------------------------------------------------
# Honesty metadata for legacy in-sample / over-named variants (audit §10.1,
# §9.3-§9.5).  Stamped at load time without changing surface or lifecycle: these
# operators stay experimental and fail-closed for production; the metadata makes
# the registry advertise that they are diagnostic/benchmark-only and names the
# preferred replacements.
# --------------------------------------------------------------------------
_DIAGNOSTIC_IN_SAMPLE = frozenset({
    "ts_multi_regression_coeff", "ts_multi_regression_resid",
    "ts_multi_regression_resid_z", "ts_multi_regression_r2",
    "ts_huber_regression_coeff", "ts_huber_regression_resid",
    "ts_huber_regression_resid_z", "ts_ridge_regression_coeff",
    "ts_ridge_regression_resid", "ts_ridge_regression_resid_z",
    "ts_ar_forecast", "ts_ar_innovation", "ts_ar_innovation_z",
    "ts_mean_reversion_half_life",
    # Expectile / quantile regression families are fitted on the full look-back
    # window (in-sample), so their coeff/resid/slope are diagnostics; the causal
    # surface uses the *_prior / *_forecast_error variants.
    "ts_expectile_regression_coeff", "ts_expectile_regression_resid",
    "ts_quantile_regression_coeff", "ts_quantile_regression_resid",
    "ts_quantile_regression_slope",
})
_IN_SAMPLE_REPLACEMENTS = {
    "ts_expectile_regression_coeff": "ts_expectile_regression_coeff_prior",
    "ts_multi_regression_coeff": "ts_multi_regression_coeff_prior",
    "ts_multi_regression_resid": "ts_multi_regression_forecast_error",
    "ts_multi_regression_resid_z": "ts_multi_regression_forecast_error_z",
    "ts_multi_regression_r2": "ts_multi_regression_r2_prior",
    "ts_huber_regression_coeff": "ts_huber_regression_coeff_prior",
    "ts_huber_regression_resid": "ts_huber_regression_forecast_error",
    "ts_huber_regression_resid_z": "ts_huber_regression_forecast_error_z",
    "ts_ridge_regression_coeff": "ts_ridge_regression_coeff_prior",
    "ts_ridge_regression_resid": "ts_ridge_regression_forecast_error",
    "ts_ridge_regression_resid_z": "ts_ridge_regression_forecast_error_z",
    "ts_ar_forecast": "ts_ar_prior_forecast",
    "ts_ar_innovation": "ts_ar_prior_innovation",
    "ts_ar_innovation_z": "ts_ar_prior_innovation_z",
}
_JUMP_ALIAS_DEPRECATED = frozenset({
    "intra_positive_jump_variation", "intra_negative_jump_variation",
    "intra_signed_jump_ratio", "intra_return_profile_cosine",
})
_JUMP_REPLACEMENTS = {
    "intra_positive_jump_variation": "intra_positive_tail_variation",
    "intra_negative_jump_variation": "intra_negative_tail_variation",
    "intra_signed_jump_ratio": "intra_signed_tail_variation_ratio",
    "intra_return_profile_cosine": "intra_signed_return_profile_cosine",
}
_BENCHMARK_ONLY = frozenset({
    "intra_realized_beta", "intra_realized_correlation",
    "intra_idiosyncratic_variance",
})
# Legacy order-flow proxies whose names over-claim what they compute
# (audit 2026-08).  They stay registered and experimental for historical
# reproducibility; the metadata advertises that they are NOT the strict
# quantities their names suggest and names the real replacements.
_LEGACY_PROXY_CANONICALS = frozenset({
    "micro_vpin",
    "micro_kyle_lambda",
})
_LEGACY_PROXY_REPLACEMENTS = {
    "micro_vpin": ["micro_bvc_vpin", "intraday_bvc_imbalance"],
    "micro_kyle_lambda": ["intraday_impact_beta", "intraday_impact_asymmetry"],
}
# Shareholder rank-slot naming consolidation (audit §6.1/§6.2).  The historic
# holder_weighted_churn family was reworked 2026-08 to the ShareholderId-matched
# union pair, so those legacy names are functionally the ID-matched
# implementations.  Stamping preferred_replacements consolidates mining/tooling
# onto the explicit holder_id_matched_* canonical names without demoting the
# reworked production operators.  holder_pledge_churn remains genuinely
# rank-slot (no ShareholderId input) and is flagged for a dedicated ID-matched
# pledge operator; the deprecated concentration/count-change row metrics point
# to the snapshot-change relation primitive.
_RANK_SLOT_ALIAS_REPLACEMENTS = {
    "holder_weighted_churn": ("holder_id_matched_churn",),
    "holder_entry_share": ("holder_id_matched_entry_share",),
    "holder_exit_share": ("holder_id_matched_exit_share",),
    "holder_net_entry_share": (
        "holder_id_matched_entry_share", "holder_id_matched_exit_share",
    ),
    "holder_rank_stability": ("holder_share_weighted_rank_migration",),
}
_RANK_SLOT_COMPATIBILITY_ONLY = frozenset({
    "holder_pledge_churn",
    "holder_concentration_change",
    "holder_count_change_rate",
})


def stamp_compatibility_metadata() -> None:
    """Stamp honest diagnostic/benchmark metadata onto legacy registry entries."""
    from cleaned_operators.registry import OperatorRegistry

    for canon in _DIAGNOSTIC_IN_SAMPLE:
        entry = OperatorRegistry._catalog.get(canon)
        if entry is None:
            continue
        entry["in_sample"] = True
        entry["diagnostic_only"] = True
        entry["hidden_from_default_mining"] = True
        replacement = _IN_SAMPLE_REPLACEMENTS.get(canon)
        if replacement is not None:
            entry["preferred_replacements"] = [replacement]
        entry.setdefault("semantic_note", "训练窗口含当前样本(旧 in-sample);默认挖掘与生产应使用 *_prior / *_forecast_error")
    for canon in _JUMP_ALIAS_DEPRECATED:
        entry = OperatorRegistry._catalog.get(canon)
        if entry is None:
            continue
        entry["compatibility_only"] = True
        replacement = _JUMP_REPLACEMENTS.get(canon)
        if replacement is not None:
            entry["preferred_replacements"] = [replacement]
        entry.setdefault("semantic_note", "旧别名(阈值尾部法/含义模糊);使用显式 *_tail_variation / *_signed_return_profile_cosine")
    for canon in _BENCHMARK_ONLY:
        entry = OperatorRegistry._catalog.get(canon)
        if entry is None:
            continue
        entry["benchmark_only"] = True
        entry.setdefault("semantic_note", "非 ex-self 市场模型仅作 benchmark/legacy;默认搜索使用 *_ex_self 版本")
    for canon in _LEGACY_PROXY_CANONICALS:
        entry = OperatorRegistry._catalog.get(canon)
        if entry is None:
            continue
        entry["legacy_proxy"] = True
        entry["hidden_from_default_mining"] = True
        replacement = _LEGACY_PROXY_REPLACEMENTS.get(canon)
        if replacement:
            entry["preferred_replacements"] = list(replacement)
        entry.setdefault(
            "semantic_note",
            "legacy 代理: 数学语义不是其名字所指的严格测度(非 BVC VPIN / 非 signed-flow Kyle λ);"
            "保留以维持历史复现, 新研究请用 preferred_replacements",
        )
    for canon, replacements in _RANK_SLOT_ALIAS_REPLACEMENTS.items():
        entry = OperatorRegistry._catalog.get(canon)
        if entry is None:
            continue
        entry["preferred_replacements"] = list(replacements)
        entry.setdefault(
            "semantic_note",
            "名次槽位命名遗留；实现已重写为股东 ID 匹配，规范名见 preferred_replacements",
        )
    for canon in _RANK_SLOT_COMPATIBILITY_ONLY:
        entry = OperatorRegistry._catalog.get(canon)
        if entry is None:
            continue
        entry["compatibility_only"] = True
        entry.setdefault(
            "semantic_note",
            "名次槽位/行数口径（无股东 ID 输入）；需专用 ID 匹配算子或 relation 快照指标",
        )


def is_intentionally_experimental(canonical: str) -> bool:
    """True when the operator was registered as experimental/research.

    These operators may have full backends and run correctly, but they were
    explicitly marked non-production at registration time.  Hardening must not
    silently upgrade them to ``status=production`` / ``pit_safe=True``.
    """
    if canonical in PROMOTED_OUT_OF_EXPERIMENTAL:
        return False
    return _REGISTERED_LIFECYCLE.get(canonical) in {"experimental", "research"}


# --------------------------------------------------------------------------
# Explicit isolation manifest (audit S18): operators whose current semantics are
# known-defective and must not be default production targets until reworked.
# Kept registered so explicit recipes can still reference them and fail loudly,
# but they are no longer blanket-promoted or marked pit-safe.
# --------------------------------------------------------------------------
ISOLATED_FROM_DEFAULT_MINING: frozenset[str] = frozenset({
    # Panel-contract violation: per-row entity counts broadcast across all
    # instrument columns; must become a source aggregation or a dedicated type.
    # They are relation-domain tools, not factor-panel targets, so they stay off
    # the daily surface.
    "relation_distinct_count",
    "relation_overlap_ratio",
    # Legacy ambiguous TTM/period names retained only to fail with a migration
    # error; never default production (prefer fin_ttm_quarterly / fin_ttm_cumulative).
    "fin_ttm",
    "ttm",
    "quarter",
    "yoy",
    # Conflation of NaN/±Inf/zero and unknown-data in one sink.
    "nan_to_num",
    "fillna",
    "protected_div",
    "causal_linear_extrapolate",
    # Revision operators cannot be PIT-certified without historical revision
    # vintages (which publication version was knowable at each past date).
    # Until a real revision-vintage source exists they stay experimental and
    # ``source_pit_passed=False`` (review §5.7).
    "fin_revision_delta",
    "fin_revision_count",
    "fin_restated_flag",
    "fin_revision_magnitude",
    "fin_revision_pct",
    "fin_revision_direction",
})


def is_isolated_from_default_mining(canonical: str) -> bool:
    return canonical in ISOLATED_FROM_DEFAULT_MINING


def should_fail_closed(canonical: str) -> bool:
    """True when this canonical must not be promoted / marked pit-safe."""
    return (
        is_intentionally_experimental(canonical)
        or is_isolated_from_default_mining(canonical)
    )


@dataclass(frozen=True)
class SemanticCert:
    """A single canonical's certification record (six independent gates).

    The first four gates are the semantic/PIT certificates; ``edge_case_passed``
    and ``backend_passed`` are the additional production-admission dimensions
    (see review §11.7).  ``operator_certification`` is the six-gate AND used as
    the strict production-admission authority; ``production_certified`` keeps the
    historical four-gate semantics for backward compatibility with the current
    evidence overlay.
    """

    canonical: str
    implementation_certified: bool
    semantic_certified: bool
    temporal_certified: bool
    source_contract_certified: bool
    pit_safe: bool
    edge_case_passed: bool = False
    backend_passed: bool = False
    notes: tuple[str, ...] = ()

    @property
    def production_certified(self) -> bool:
        return bool(
            self.implementation_certified
            and self.semantic_certified
            and self.temporal_certified
            and self.source_contract_certified
        )

    @property
    def operator_certification(self) -> bool:
        """Six-gate production certification (review §2 / §11.7)."""
        return bool(
            self.implementation_certified
            and self.semantic_certified
            and self.temporal_certified
            and self.source_contract_certified
            and self.edge_case_passed
            and self.backend_passed
        )

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperatorCertification:
    """Six-gate production certification record (review §2).

    ``production_certified`` requires every gate to pass.  Only operators with
    ``production_certified=True`` may enter the daily DSL allowlist, the cold-start
    operator pool, and the AlphaProbe/AlphaMiner search space.
    """

    implementation_passed: bool
    semantic_passed: bool
    temporal_passed: bool
    source_pit_passed: bool
    edge_case_passed: bool
    backend_passed: bool

    @property
    def production_certified(self) -> bool:
        return all(
            (
                self.implementation_passed,
                self.semantic_passed,
                self.temporal_passed,
                self.source_pit_passed,
                self.edge_case_passed,
                self.backend_passed,
            )
        )

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def operator_certification_for(
    canonical: str,
    catalog: dict[str, Any] | None = None,
) -> OperatorCertification:
    """Compose the six-gate certification record from the semantic certificates."""
    cert = semantic_cert(canonical, catalog)
    return OperatorCertification(
        implementation_passed=cert.implementation_certified,
        semantic_passed=cert.semantic_certified,
        temporal_passed=cert.temporal_certified,
        source_pit_passed=cert.source_contract_certified,
        edge_case_passed=cert.edge_case_passed,
        backend_passed=cert.backend_passed,
    )


def semantic_cert(
    canonical: str,
    catalog: dict[str, Any] | None = None,
    *,
    evidence_production_certified: bool | None = None,
    semantic_golden_verified: bool | None = None,
    temporal_prefix_verified: bool | None = None,
    source_contract_verified: bool | None = None,
) -> SemanticCert:
    """Compose the four-certificate record for ``canonical``.

    Fail-closed (review P0-A01): an operator is NEVER certified by absence.
    ``semantic_ok`` / ``temporal_ok`` / ``source_ok`` require the canonical to
    be bound to a verified implementation artifact
    (``evidence_production_certified=True``) AND to sit outside the
    experimental/isolation manifest.  The ``None`` path (pre-overlay hardening,
    before ``apply_evidence_certification_overlay`` binds per-operator evidence)
    grants no certificate — candidate status only, never certification.

    Per-gate independence (review P0-18): each of the semantic / temporal /
    source-contract gates requires its OWN verified evidence record
    (``semantic_golden_verified`` / ``temporal_prefix_verified`` /
    ``source_contract_verified``).  Implementation evidence proves the backend
    ran; it never back-stops the other gates.  Absent per-gate records default
    to False (fail-closed) and are sourced from the catalog when the caller does
    not supply them explicitly.
    """
    notes: list[str] = []

    # Resolve each gate's OWN independent evidence.  Implementation evidence must
    # never grant the semantic/temporal/source certificates (review P0-18); an
    # operator with no per-gate record fails closed on that gate.
    if semantic_golden_verified is None:
        semantic_golden_verified = bool((catalog or {}).get("semantic_golden_verified", False))
    if temporal_prefix_verified is None:
        temporal_prefix_verified = bool((catalog or {}).get("temporal_prefix_verified", False))
    if source_contract_verified is None:
        source_contract_verified = bool((catalog or {}).get("source_contract_verified", False))

    # (1) Implementation: evidence overlay result is authoritative when given.
    if evidence_production_certified is True:
        implementation = True
    elif evidence_production_certified is False:
        implementation = False
    else:
        implementation = not should_fail_closed(canonical)

    # (2) Semantic / (3) temporal: each gate is granted only when the operator
    # has BOTH a verified implementation artifact AND its own per-gate evidence
    # record.  ``not should_fail_closed`` alone never certifies.
    semantic_ok = bool(
        evidence_production_certified is True
        and semantic_golden_verified
        and not should_fail_closed(canonical)
    )
    temporal_ok = bool(
        evidence_production_certified is True
        and temporal_prefix_verified
        and not should_fail_closed(canonical)
    )

    # (4) Source contract: evidence-backed AND own source/PIT record AND not
    # source-blocked AND not research-marked.
    from cleaned_operators.production_hardening import SOURCE_BLOCKED_CANONICALS

    source_ok = bool(
        evidence_production_certified is True
        and source_contract_verified
        and canonical not in SOURCE_BLOCKED_CANONICALS
        and not should_fail_closed(canonical)
    )

    if should_fail_closed(canonical):
        notes.append("registered experimental/research or in isolation manifest")
    if not source_ok:
        notes.append("source contract not certified")

    # (5) Edge-case evidence: declared/verified edge dimensions must be complete.
    edge_ok = False
    try:
        from cleaned_operators.edge_requirements import (
            production_edge_evidence_complete,
        )

        edge_ok = bool(production_edge_evidence_complete(canonical))
    except Exception:
        edge_ok = False
    if not edge_ok:
        notes.append("edge-case evidence incomplete")

    # (6) Backend evidence: at least one independently certified production backend.
    backend_ok = False
    try:
        from backend.operator_capability import production_eligible_backends

        backend_ok = bool(production_eligible_backends(canonical))
    except Exception:
        backend_ok = False
    if not backend_ok:
        notes.append("no evidence-backed production backend")

    pit_safe = bool(implementation and semantic_ok and temporal_ok and source_ok)
    return SemanticCert(
        canonical=canonical,
        implementation_certified=bool(implementation),
        semantic_certified=bool(semantic_ok),
        temporal_certified=bool(temporal_ok),
        source_contract_certified=bool(source_ok),
        pit_safe=pit_safe,
        edge_case_passed=edge_ok,
        backend_passed=backend_ok,
        notes=tuple(notes),
    )


def attach_four_certificates(canonical: str, catalog: dict[str, Any]) -> SemanticCert:
    """Write the certificate fields + ``production_certified`` onto catalog.

    Returns the composed record so callers can gate status/pit_safe on it.
    Also writes the six-gate fields (``edge_case_passed`` / ``backend_passed`` /
    ``operator_certification``) so the strict production-admission composite is
    visible per operator.

    The **six-gate** composite is the single production-certification authority
    (review §2.1): ``catalog["production_certified"]`` now carries
    ``operator_certification`` (implementation + semantic + temporal + source
    contract + edge-case + backend).  The historical four-gate AND is retained
    under ``semantic_pit_review_passed`` for diagnostics only; it no longer
    grants production admission.
    """
    cert = semantic_cert(canonical, catalog)
    catalog["implementation_certified"] = cert.implementation_certified
    catalog["semantic_certified"] = cert.semantic_certified
    catalog["temporal_certified"] = cert.temporal_certified
    catalog["source_contract_certified"] = cert.source_contract_certified
    # Historical four-gate review record — NOT a production authority.
    catalog["semantic_pit_review_passed"] = cert.production_certified
    catalog["production_certified"] = cert.operator_certification
    catalog["edge_case_passed"] = cert.edge_case_passed
    catalog["backend_passed"] = cert.backend_passed
    catalog["operator_certification"] = cert.operator_certification
    catalog["certification_notes"] = list(cert.notes)
    return cert


DEFAULT_CERTIFICATION = {
    "implementation_passed": False,
    "semantic_passed": False,
    "temporal_passed": False,
    "source_pit_passed": False,
    "edge_case_passed": False,
    "backend_passed": False,
}


def reconcile_operator_certification(
    canonical: str,
    catalog: dict[str, Any] | None = None,
) -> OperatorCertification:
    """Converge the six-gate production certification from final evidence.

    Runs after the evidence overlay has bound physical backend certification to
    the immutable artifacts.  This is the **single authority** for
    ``status`` / ``lifecycle_status`` / ``pit_safe`` / ``production_certified``:
    an operator is production only when all six gates pass (review §2.3, §2.4).

    ``status == "production"`` therefore means "fully certified", not merely
    "a reviewed target".  Operators whose evidence is stale, absent, or whose
    edge/backend gates fail are downgraded to ``experimental`` and fail-closed.
    """
    from cleaned_operators.registry import OperatorRegistry

    catalog = (
        catalog
        if catalog is not None
        else OperatorRegistry._catalog.get(canonical, {})
    )
    cert = semantic_cert(canonical, catalog)

    # (1) Implementation gate: evidence-bound backend certification is the only
    # authority.  ``should_fail_closed`` alone must never grant it (review §2.2).
    meta = ((catalog.get("backend_meta") or {}).get("pandas_numpy") or {})
    implementation_passed = bool(meta.get("production_certified")) or bool(
        cert.backend_passed
    )

    # Recompute the semantic/temporal/source certificates with the per-operator
    # evidence binding so they are genuinely evidence-driven (review P0-A01):
    # an operator not bound to a verified artifact gets all-negative
    # certificates, never default trust.  Each gate also requires its OWN
    # per-gate evidence record (review P0-18) — implementation evidence alone
    # never grants the semantic/temporal/source certificates.
    cert = semantic_cert(
        canonical,
        catalog,
        evidence_production_certified=bool(implementation_passed),
        semantic_golden_verified=bool(
            catalog.get("semantic_golden_verified", False)
        ),
        temporal_prefix_verified=bool(
            catalog.get("temporal_prefix_verified", False)
        ),
        source_contract_verified=bool(
            catalog.get("source_contract_verified", False)
        ),
    )

    # (2)-(4) Semantic / temporal / source-PIT gates: an operator is never
    # certified by *absence* — the four-gate review record must ALSO be backed
    # by a version-bound evidence record (``implementation_passed``).  The
    # experimental/isolated set stays all-negative regardless (review §2.2).
    semantic_passed = bool(
        implementation_passed and cert.semantic_certified
    )
    temporal_passed = bool(
        implementation_passed and cert.temporal_certified
    )
    source_pit_passed = bool(
        implementation_passed and cert.source_contract_certified
    )

    # (5) Edge-case evidence is INDEPENDENT (review P0-A02): implementation
    # evidence never back-stops it.  ``production_edge_evidence_complete``
    # reports whether every declared NaN/Inf edge dimension has a verified edge
    # case; an operator whose edge dimensions are genuinely unverified fails
    # closed here even if its backend runs.
    edge_case_passed = bool(cert.edge_case_passed)
    # (6) At least one evidence-backed production backend must exist.
    backend_passed = cert.backend_passed

    six = OperatorCertification(
        implementation_passed=bool(implementation_passed),
        semantic_passed=bool(semantic_passed),
        temporal_passed=bool(temporal_passed),
        source_pit_passed=bool(source_pit_passed),
        edge_case_passed=bool(edge_case_passed),
        backend_passed=bool(backend_passed),
    )
    certified = six.production_certified

    catalog["implementation_certified"] = six.implementation_passed
    catalog["semantic_certified"] = six.semantic_passed
    catalog["temporal_certified"] = six.temporal_passed
    catalog["source_contract_certified"] = six.source_pit_passed
    catalog["edge_case_passed"] = six.edge_case_passed
    catalog["backend_passed"] = six.backend_passed
    catalog["semantic_pit_review_passed"] = all(
        (
            six.implementation_passed,
            six.semantic_passed,
            six.temporal_passed,
            six.source_pit_passed,
        )
    )
    catalog["operator_certification"] = certified
    catalog["production_certified"] = certified
    catalog["pit_safe"] = certified
    catalog["status"] = "production" if certified else "experimental"
    catalog["lifecycle_status"] = "production" if certified else "experimental"
    catalog["certification_notes"] = list(cert.notes)

    # Keep the operator-policy table consistent with the reconciled lifecycle.
    # ``apply_production_hardening`` sealed every target candidate/false before
    # the evidence overlay ran; the final authority must re-open pit_safe for
    # operators that are genuinely certified so ``infer_operator_policy`` agrees
    # with ``catalog["pit_safe"]`` (review P0-A03: no blanket, reconcile wins).
    #
    # ``pit_safe`` in ``_EXPLICIT_POLICIES`` is a *structural* property: an
    # elementwise / cross-sectional / group / trailing-window operator is causal
    # by construction and its policy stays True regardless of evidence state.
    # Certification gates production *admission* (``catalog["pit_safe"]``), not
    # structural causality.  So reconcile only re-opens pit_safe for genuinely
    # certified operators; it never downgrades an explicit structural True to
    # False (that would deadlock the primitive/factor certifier bootstrap,
    # whose convergence stage runs before the artifact is written).
    #
    # Exception: an intentionally-experimental / isolated canonical (the
    # in-sample diagnostic family — ts_ar_forecast/innovation*, *_resid without
    # the *_prior / forecast_error suffix, etc.) must stay pit_safe=False even
    # though it carries an explicit structural policy entry: those operators are
    # reviewed *not to be admitted*, so ``should_fail_closed`` wins over the
    # structural table.  ``infer_operator_policy`` then agrees with the
    # fail-closed lifecycle and the manifest/catalog surfaces stay consistent.
    try:
        import cleaned_operators.operator_policy as _operator_policy_mod
    except ImportError:
        # P0-34: only a genuinely unimportable policy module is benign (nothing
        # to sync).  Any OTHER failure — including a missing ``_EXPLICIT_POLICIES``
        # attribute (code drift) or an exception raised DURING the sync — must
        # propagate so certification/finalization fails rather than leaving the
        # catalog and policy table divergent.
        _operator_policy_mod = None
    if _operator_policy_mod is not None:
        _EXPLICIT_POLICIES = _operator_policy_mod._EXPLICIT_POLICIES
        existing = dict(_EXPLICIT_POLICIES.get(canonical) or {})
        if should_fail_closed(canonical):
            existing["pit_safe"] = False
        else:
            existing["pit_safe"] = existing.get("pit_safe", False) or bool(certified)
        _EXPLICIT_POLICIES[canonical] = existing
    return six
