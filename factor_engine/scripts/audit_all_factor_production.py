#!/usr/bin/env python3
"""Audit every retained Factor DSL canonical against production invariants.

Runtime-only mode bootstraps semantic evidence by executing the Pandas reference,
checking axes, determinism and prefix causality. Strict mode additionally
requires OperatorSpec admission and at least one evidence-backed physical
backend. Public parameters are generated from the final Registry contract;
unknown parameters fail closed instead of receiving an arbitrary panel.
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FE_ROOT = Path(__file__).resolve().parents[1]
for path in (str(FE_ROOT.parent), str(FE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

_PANEL_PARAMETERS = frozenset({
    "x", "y", "z", "a", "b", "w", "g", "left", "right",
    "numerator", "denominator", "ret", "returns", "benchmark_ret",
    "market_ret", "benchmark", "market", "open", "high", "low", "close",
    "price", "volume", "amount", "vwap", "turnover", "weight", "weights",
    "signal", "fallback", "condition", "group", "industry", "sector",
    "fiscal_quarter", "period_id", "target_period_id", "quarter", "revision_id",
    "decision_time", "available_time", "available_at", "exposure", "exposures",
    "control", "controls", "factor", "target", "mask", "event", "value",
    "values", "v1", "v2", "sort_col", "float_shares", "flow", "balance",
    "earnings", "cashflow", "assets", "working_capital", "base",
    "dollar_volume", "scale_base", "fundamental_x", "fundamental_y",
    "fundamental_scale", "actual", "expected", "expected_std", "expected_mean",
    "current_assets", "current_liabilities", "inventory", "total_debt",
    "total_equity", "short_debt", "long_debt", "cash", "total_assets",
    "operating_income", "revenue", "gross_profit", "net_income",
    "operating_cash_flow", "research_development", "capex", "invested_capital",
    "nopat", "receivables", "cost_of_goods_sold", "interest_expense",
    "market_cap", "one", "pe", "pb", "total_shares", "free_float_shares",
    "upper_limit", "lower_limit", "top_holder_shares", "listed", "suspended", "limit_up", "limit_down", "benchmark_price",
    "holder_count", "concentration", "signal_x", "signal_y", "x1", "x2",
    "cost", "activity", "operating", "investing", "financing", "accrual",
    "period_end",
    # Operator expansion (2026-08) panel inputs.
    "pre_close", "subgroup", "member", "index_weight", "date1", "date2",
    "period_end_date", "pub_date", "state", "event", "entity_id",
    "snapshot_id", "snapshot_date", "previous_snapshot_date",
    "high_limit", "low_limit", "stock_return",
    "benchmark_return", "category", "day_vwap", "current_snapshot",
    "previous_snapshot", "weights2", "s1", "s2", "s3",
    # Minute/relation integration (2026-08).
    "entity_ids", "current_ids", "previous_ids", "components",
    "high_limit", "low_limit", "activity",
    # valuation / fundamental / holder / ts_model / panel inputs (2026-08).
    "a_cap", "account_receivable", "asset_impairment_loss", "borrowing_repayment",
    "capex_cash", "capitalized_dev_increase", "cash_equivalents", "cash_from_borrowing",
    "change_date", "circulating_capital", "component", "contract_assets",
    "contract_liability", "debt_repayment", "deferred_tax_assets",
    "discontinued_operation_profit", "earnings_yield", "entry_event", "entry_index_a",
    "fair_value_income", "free_cap", "free_float_weight", "free_market_cap",
    "freeze_shares", "goods_sale_cash", "goodwill", "inventories", "investment_income",
    "is_suspend", "listing_date", "locked_shares", "minority_profit",
    "net_cash_from_subcompany", "net_profit", "ocf", "operating_profit",
    "other_comprehensive_income", "other_earnings", "own_return", "paidin_capital",
    "pcf_ratio", "peer_return", "pe_ratio", "period_expense", "pledge_ratio",
    "pledge_shares", "rd_expense", "retained_earnings", "roa", "shared_holders",
    "short_term_debt", "short_term_loan", "total_capital", "total_composite_income",
    "total_liabilities", "usufruct_assets", "valuation", "group", "breadth",
    # 2026-08 concurrent expansion: fundamental-quality / valuation / holder /
    # ts_model / cross-sectional-robust / panel-model series inputs.
    "asset_deal_income", "asset_turnover", "avg_assets", "avg_equity",
    "bonds_payable", "capital_reserve", "capitalization", "cash_from_bonds_issue",
    "credit_impairment_loss", "current_ratio", "deferred_tax_liability",
    "dividend_interest_payment", "entry_index_b", "entry_index_c",
    "goodwill_prev", "gross_margin", "industry_dispersion", "industry_liquidity",
    "interest_cost", "inventory_turnover", "leader_return", "lease_liability",
    "leverage", "long_term_debt", "long_term_loan", "market_liquidity",
    "market_state", "non_operating_expense", "non_operating_revenue", "ocf_yield",
    "operating_revenue", "overlap", "pcf_ratio2", "pe_ratio_lyr", "profit_growth",
    "quality", "receivable_turnover", "revenue_growth", "tax_payable",
    "circulating_cap", "top10_concentration", "top10_float_concentration",
    "total", "total_holders", "total_profit", "...", "d1", "d2", "d3", "d4", "d5",
    "group1", "group2", "group3", "x3", "x4", "f1", "f2", "f3", "f4",
    "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10",
    "s4", "s5", "s6", "s7", "s8", "s9", "s10",
    # shareholder churn/network (2026-08): current/prev holder-id slots are
    # per-instrument series, not scalars.
    "sid1", "sid2", "sid3", "sid4", "sid5", "sid6", "sid7", "sid8", "sid9", "sid10",
    "psid1", "psid2", "psid3", "psid4", "psid5", "psid6", "psid7", "psid8", "psid9", "psid10",
    # cross-section robust (2026-08): residual/distance panels.
    "resid", "score",
    # 2026-08 final pack: A-share limit/suspension state machine event panels,
    # relation concentration/entropy panels and rank slots.
    "valid_trade", "limit_up_event", "limit_down_event", "up_event", "down_event",
    "known_status", "is_suspend", "hhi", "entropy",
    "rank1", "rank2", "rank3", "rank4", "rank5",
    "rank6", "rank7", "rank8", "rank9", "rank10",
    # 2026-08 stateful rule/episode pack condition panels.
    "set_condition", "reset_condition", "update_condition",
    # 2026-08 advanced operators: transfer-entropy source panel; holder-class
    # previous-disclosure share slots for the JS-shift operator.
    "source", "ps1", "ps2", "ps3", "ps4", "ps5",
    # 2026-08 V2/V3 dynamics pack: event-response target, report timing delay,
    # previous published value for revision magnitude.
    "response", "delay", "prev_x",
    # 2026-08 concurrent envelope family: envelope mid band panel.
    "mid",
    # 2026-08-08 market-language pack: update-clock true-update marker and
    # marked-event magnitude panels.
    "mark", "update_event",
})

# Params whose generic names also appear in _SCALAR_VALUES but are PANEL series
# inputs for these specific operators.  Checked before scalar fallback.
_PANEL_FORCE: frozenset[tuple[str, str]] = frozenset({
    ("ts_har_rv_forecast", "rv"),
    ("ts_har_rv_innovation_z", "rv"),
    ("ts_har_rv_next_forecast", "rv"),
    ("ts_har_rv_forecast_error_z", "rv"),
    ("fin_margin_persistence", "margin"),
    ("group_peer_beta_deviation", "beta"),
    # ``scale`` is the per-row volatility panel for the first-passage barrier
    # (not the scalar intra_amihud ``scale``).
    ("ts_first_passage_bias", "scale"),
    ("ts_first_passage_hit_probability", "scale"),
    ("ts_first_passage_conditional_time", "scale"),
    # 2026-08 concurrent episode / envelope families: ``scale`` and envelope
    # band panels collide with scalar names.
    ("state_episode_excursion_balance", "scale"),
    ("state_episode_mae", "scale"),
    ("state_episode_mfe", "scale"),
    ("ts_envelope_boundary_dwell", "upper"),
    ("ts_envelope_boundary_dwell", "lower"),
    ("ts_envelope_pressure", "upper"),
    ("ts_envelope_pressure", "lower"),
    ("ts_envelope_compression", "upper"),
    ("ts_envelope_compression", "lower"),
    # 2026-08-08 directional-change: ``scale`` is the per-row volatility/ATR
    # panel (not the scalar 1e8 intra_amihud scale).
    ("ts_dc_overshoot_ratio", "scale"),
    ("ts_dc_event_rate", "scale"),
    ("ts_dc_duration_asymmetry", "scale"),
    ("ts_dc_overshoot_asymmetry", "scale"),
})

_SCALAR_VALUES: dict[str, Any] = {
    "window": 20,
    "d": 20,
    "n": 3,
    "m": 2,
    "span": 20,
    "period": 20,
    "periods": 8,
    "lag": 1,
    "lags": 1,
    "k": 3,
    "q": 0.2,
    "quantile": 0.2,
    "threshold": 0.01,
    "run": 2,
    "hump": 0.02,
    "min_periods": 5,
    "min_pairs": 3,
    "min_bin_count": 3,
    "min_peers": 5,
    "min_reference_days": 5,
    "min_transitions": 30,
    "target_q": 0.1,
    "source_q": 0.1,
    "level": 3,
    "sampling": 5,
    "n_slots": 32,
    "ridge": 1e-3,
    "steps": 2,
    "max_interval": 16,
    "event_lag": 1,
    "n_updates": 5,
    "min_history": 4,
    "lookback_periods": 8,
    "seasonal_lag": 4,
    "min_train": 6,
    "ar_lag": 1,
    "min_obs_per_regime": 3,
    "min_obs": 8,
    "ddof": 1,
    "ann_factor": 252.0,
    "decimals": 2,
    "to": 1.0,
    "lower": -2.0,
    "upper": 2.0,
    "eps": 1e-8,
    "epsilon": 1e-8,
    "alpha": 0.2,
    "shrinkage": 0.1,
    "breaks": [-1.0, 0.0, 1.0],
    "quantiles": (0.2, 0.4, 0.6, 0.8),
    "fallback_policy": "nan",
    "fast": 12,
    "slow": 26,
    "fast_period": 12,
    "slow_period": 26,
    "fast_window": 12,
    "slow_window": 26,
    "signal_span": 9,
    "signal_window": 9,
    "signal_period": 9,
    "side": "lower",
    "order": "largest",
    "add_intercept": True,
    "clip": 3.0,
    "limit": 3,
    "max_gap": 3,
    "max_periods": 8,
    "max_lookback": 60,
    "power": 2.0,
    "exponent": 2.0,
    "p": 0.5,
    "c": 1.0,
    "fraction": 0.5,
    "buckets": 5,
    "top": 3,
    "asc": True,
    "annualization": 252.0,
    "annualization_factor": 252.0,
    "periods_per_year": 4,
    "years": 5,
    "depreciation": 0.15,
    "warmup_periods": 8,
    "method": "std",
    "mode": "absolute",
    "interpolation": "linear",
    "center": True,
    "ascending": True,
    "inclusive": True,
    "offset": 0,
    "require_consecutive": True,
    "trim_pct": 0.1,
    "sign_policy": "strict",
    "denominator_policy": "signed",
    "aggr_func": "sum",
    "lo": -2.0,
    "hi": 2.0,
    "left_window": 3,
    "right_window": 3,
    "history_window": 60,
    "max_lag": 5,
    "drift": 0.5,
    "cooldown": 5,
    "half_life": 10,
    "initial_state": 0,
    "min_episode": 2,
    "min_completed_runs": 20,
    "max_age": 60,
    "direction": "up",
    "band": 0.05,
    "points": 3,
    "std_dev": 2.0,
    "skipna": True,
    "revision_policy": "latest_available",
    "missing_group_policy": "raise",
    "short_window": 7,
    "medium_window": 14,
    "long_window": 28,
    "ema_window": 20,
    "atr_window": 14,
    "tenkan_window": 9,
    "kijun_window": 26,
    "senkou_b_window": 52,
    "er_window": 10,
    "vol_window": 20,
    "baseline_window": 40,
    "price_window": 20,
    "volume_window": 20,
    "turnover_window": 20,
    "adl_window": 60,
    "impulse_window": 20,
    "flag_window": 10,
    "pennant_window": 12,
    "cup_window": 80,
    "handle_window": 15,
    "growth_periods": 4,
    "compare_periods": 1,
    "window_periods": 8,
    "average_periods": 2,
    "short_periods": 3,
    "long_periods": 4,
    "window_days": 60,
    "max_days": 252,
    "max_wait": 10,
    "body_window": 10,
    "shadow_window": 10,
    "tolerance": 0.03,
    "tick_tolerance": 0.005,
    "min_depth": 0.05,
    "min_spacing": 5,
    "max_spacing": 40,
    "shoulder_tolerance": 0.05,
    "head_min_prominence": 0.05,
    "max_neckline_slope": 0.02,
    "slope_threshold": 0.001,
    "parallel_tolerance": 0.001,
    "min_impulse": 0.08,
    "min_swing": 0.05,
    "min_fit": 0.01,
    "max_retracement": 0.5,
    "max_handle_retracement": 0.5,
    "max_width": 0.12,
    "max_edge_diff": 0.10,
    "volume_decay_threshold": 0.0,
    "multiplier": 2.0,
    "volume_scale": 1_000_000.0,
    "acceleration": 0.02,
    "maximum": 0.2,
    "penetration": 0.3,
    "short_weight": 4.0,
    "medium_weight": 2.0,
    "long_weight": 1.0,
    "pattern": "high_wave",
    "body_factor": 1.0,
    "shadow_factor": 1.0,
    # Operator expansion (2026-08) scalar parameters.
    "q_low": 0.25,
    "q_high": 0.75,
    "trim_ratio": 0.1,
    "center": "median",
    "scale": "mad",
    "normalize": True,
    "target": 0.0,
    "max_lag": 3,
    "min_events": 2,
    "min_count": 3,
    # Minute/relation integration (2026-08).
    "segment": "morning",
    "transition": "open",
    "absolute_return": False,
    "method": "jaccard",
    "component_directions": "up",
    "normalize_weight": True,
    "scale": 1e8,
    "half_life": 20.0,
    "score_weights": 1.0,
    # 2026-08 alpha-language expansion params.
    "max_run": 5,
    "cap": 60,
    "split": 0.5,
    "morning_cutoff": "11:30",
    "afternoon_start": "13:00",
    "event_effective_lag": 1,
    "missing_policy": "break",
    # valuation / fundamental / holder / ts_model / panel (2026-08) scalars.
    "bins": 10,
    "degree": 2,
    "knots": 4,
    "n_components": 3,
    "threshold_scale": 1.0,
    "q_level": 1e-4,
    "q_trend": 1e-5,
    "margin": 0.1,
    "top": 3,
    "start_minute": 570,
    "end_minute": 900,
    "rv": 0.1,
    "beta": 0.5,
    "s": 0.1,
    "r": 1.0,
    "f": 0.5,
    "d": 20,
    "x": 0.5,
    "y": 0.5,
    "coefficient_index": 0,
    "decay": 0.9,
    "l1_ratio": 0.5,
    "n_experts": 3,
    "n_regimes": 2,
    "l": 5,
    # 2026-08 final pack: robust tail / complexity / intraday scalars.
    "tail_quantile": 0.75,
    "edge_minutes": 30,
    "mid_start": 660,
    "mid_end": 810,
    "tail_minutes": 30,
    "estimator": "quantile_hist",
    "delay": 1,
    "outer": 0.05,
    "inner": 0.95,
    "q_mid": 0.5,
    "use_abs": True,
    "k_max": 10,
    "n_scales": 10,
    "embedding_dim": 2,
    "tolerance_scale": 0.2,
    "min_patterns": 5,
    "min_valid_lags": 3,
    "normalized": True,
    # 2026-08 turnover-survival / behavioural / order-flow family scalars.
    "preset": "bmw2016",
    "bucket_count": 20,
    # 2026-08 advanced operators: topology / PCA / barrier scalars.
    "tau": 1,
    "directions": 16,
    "lookback": 5,
    # 2026-08 V2/V3 dynamics pack scalars.
    "horizon": 5,
    "barrier": 1.0,
    "bandwidth": 1.0,
    "history_length": 1,
    "tail_fraction": 0.2,
    "min_tail_count": 10,
    "min_anchors": 3,
    "grid": 8,
    "residual_fraction": 0.25,
    # 2026-08 concurrent recurrence-analysis / spectral-transport packs.
    "dim": 3,
    "eps_fraction": 0.1,
    "scales": (2, 4, 8, 16),
    "subsequence_length": 10,
    "prominence": 0.1,
    "confirmation": 3,
    "split_quantile": 0.5,
    "block": 5,
    "match_lag": 1,
    "history": 20,
    "fd": 0.5,
    "cutoff": 0.1,
}

_SPECIAL_SCALARS: dict[tuple[str, str], Any] = {
    ("cs_tail_retention", "side"): "top",
    ("state_since_reduce", "mode"): "sum",
    ("cs_rank_gaussian", "method"): "blom",
    ("cs_regression", "mode"): 0,
    ("cs_quantile", "p"): 0.5,
    ("group_percentile", "p"): 0.5,
    ("group_percentile", "side"): "top",
    ("group_percentile", "missing_group_policy"): "raise",
    ("revision_delta", "mode"): "absolute",
    ("period_change", "mode"): "absolute",
    ("period_lag", "revision_policy"): "latest_available",
    ("period_stability", "method"): "std",
    ("ts_nth_value", "order"): "largest",
    ("ts_permutation_entropy", "order"): 4,
    # 2026-08 alpha-language state ops require 0 <= lower < upper; override the
    # signed global upper/lower for them.
    ("ts_hysteresis_age", "lower"): 0.0,
    ("ts_hysteresis_age", "upper"): 1.0,
    ("ts_hysteresis_state", "lower"): 0.0,
    ("ts_hysteresis_state", "upper"): 1.0,
    ("ts_state_entry_strength", "lower"): 0.0,
    ("ts_state_entry_strength", "upper"): 1.0,
    ("ts_state_integral", "lower"): 0.0,
    ("ts_state_integral", "upper"): 1.0,
    ("ts_ar_forecast", "order"): 3,
    ("ts_ar_innovation", "order"): 3,
    ("ts_ar_innovation_z", "order"): 3,
    ("ts_ar_prior_forecast", "order"): 3,
    ("ts_ar_prior_innovation", "order"): 3,
    ("ts_ar_prior_innovation_z", "order"): 3,
    ("ts_ar_prior_coeff", "order"): 3,
    ("ts_ar_coeff_stability", "order"): 3,
    ("panel_rolling_pca_loading", "component"): 0,
    ("industry_rolling_pca_loading", "component"): 0,
    ("ts_mad", "scale"): 1.0,
    ("ts_product", "skipna"): True,
    ("MACD_line", "signal"): 9,
    ("MACD_signal", "signal"): 9,
    ("MACD_hist", "signal"): 9,
    ("fillna_const", "value"): 0.0,
    # ``delay`` / ``composition_policy`` are scalar for these operators even
    # though ``delay`` also names a panel input elsewhere in the audit.
    ("ts_permutation_entropy", "delay"): 1,
    ("ts_weighted_permutation_entropy", "delay"): 1,
    ("ts_permutation_transition_entropy", "delay"): 1,
    ("ts_ordinal_irreversibility", "delay"): 1,
    ("group_spd_feature_structure_shift", "composition_policy"): "current",
    ("group_winsorize", "a"): 0.05,
    ("winsorize", "lower"): 0.05,
    ("winsorize", "upper"): 0.95,
    ("ts_regression_slope", "retval"): "slope",
    ("fiscal_asymmetric_elasticity", "mode"): "down_minus_up",
    ("fiscal_sign_consistency", "min_periods"): 3,
    ("fiscal_sign_agreement", "min_periods"): 3,
    ("fiscal_change_direction_agreement", "min_periods"): 3,
    ("fiscal_direction_consistency", "min_periods"): 3,
    ("fiscal_pair_direction_agreement", "min_periods"): 3,
    ("yoy_by_period", "denominator"): "signed",
    ("ts_downside_deviation", "target"): 0.0,
    ("ts_upside_deviation", "target"): 0.0,
    ("ts_weighted_semivariance", "target"): 0.0,
    ("ts_weighted_expected_shortfall", "side"): "lower",
    ("intra_limit_first_hit_time", "side"): "up",
    ("intra_limit_duration", "side"): "up",
    ("intra_limit_reopen_count", "side"): "up",
    ("intra_realized_semivariance", "side"): "down",
    ("intra_extreme_bar_return", "side"): "max",
    ("intra_return_activity_corr", "absolute_return"): False,
    ("ashare_limit_one_price", "side"): "up",
    # 2026-08 final pack: partial-moment ``order`` is a numeric power, not the
    # generic "largest"/"smallest" tail selector; permutation ``order`` is the
    # embedding order; weighted-permutation ``weight`` is a scheme name.
    ("ts_lower_partial_moment", "order"): 2.0,
    ("ts_upper_partial_moment", "order"): 2.0,
    ("ts_weighted_permutation_entropy", "order"): 4,
    ("ts_permutation_transition_entropy", "order"): 4,
    ("ts_weighted_permutation_entropy", "weight"): "variance",
    # A-share state machine ``side`` is up/down (not the generic lower/upper).
    ("ashare_limit_touch_count", "side"): "up",
    ("ashare_failed_limit_count", "side"): "up",
    ("ashare_one_price_limit_streak", "side"): "up",
    # DFA scale grid is positive integers; the generic min_/max_ fallbacks
    # (0.01 / 0.5) are degenerate scales.
    ("ts_hurst_dfa", "min_scale"): 2,
    ("ts_hurst_dfa", "max_scale"): 50,
    # Quantile-kurtosis ``outer``/``inner`` are 2-tuples of quantiles, not scalars.
    ("ts_quantile_kurtosis", "outer"): (0.025, 0.975),
    ("ts_quantile_kurtosis", "inner"): (0.25, 0.75),
    # 2026-08 advanced operators: per-op scalars.  Transfer entropy uses a small
    # ``bins``; Kramers-Moyal a moderate one; Bures/Student-t need non-overlapping
    # non-trivial windows; PCA ops use component k=1; SW/SPD use a long reference.
    ("ts_transfer_entropy", "bins"): 3,
    ("ts_effective_transfer_entropy", "bins"): 3,
    ("ts_kramers_moyal_drift", "bins"): 5,
    ("ts_kramers_moyal_diffusion", "bins"): 5,
    ("ts_bures_corr_shift", "recent_window"): 10,
    ("ts_bures_corr_shift", "prior_window"): 60,
    ("ts_fisher_information_shift", "recent_window"): 60,
    ("ts_fisher_information_shift", "prior_window"): 120,
    ("cs_sliced_wasserstein_copula_shift", "window"): 60,
    ("group_spd_feature_structure_shift", "reference_window"): 60,
    ("intraday_quantile_curve_pca_score", "window"): 60,
    ("intraday_quantile_curve_pca_score", "k"): 1,
    ("intraday_quantile_curve_pca_residual", "window"): 60,
    ("intraday_quantile_curve_pca_residual", "k"): 1,
    ("ts_kramers_moyal_drift", "min_bin_count"): 0,
    ("ts_kramers_moyal_diffusion", "min_bin_count"): 0,
    ("group_spd_feature_structure_shift", "min_peers"): 5,
    ("group_spd_feature_structure_shift", "min_reference_days"): 5,
    ("ts_transfer_entropy", "min_transitions"): 30,
    ("ts_effective_transfer_entropy", "min_transitions"): 30,
    ("ts_bures_corr_shift", "min_pairs"): 5,
    ("ts_betti_1_max_persistence", "window"): 60,
    ("ts_betti_1_max_persistence", "tau"): 1,
    ("ts_persistence_diagram_shift", "window"): 60,
    ("ts_persistence_diagram_shift", "tau"): 1,
    # 2026-08-08 market-language pack: expectile tau in (0,1); conditional TE
    # keeps a small bin count; local-KNN needs k >= 4; feature rotation/break
    # need recent+prior <= window; signature slope a power-of-two span.
    ("ts_expectile", "tau"): 0.1,
    ("ts_expectile_beta", "tau"): 0.1,
    ("ts_conditional_transfer_entropy", "bins"): 3,
    ("cs_knn_local_linear_residual", "k"): 5,
    ("cs_knn_local_gradient_norm", "k"): 5,
    ("cs_knn_tangent_residual", "k"): 5,
    ("ts_feature_subspace_rotation", "window"): 120,
    ("ts_feature_subspace_rotation", "recent_window"): 30,
    ("ts_feature_subspace_rotation", "prior_window"): 90,
    ("ts_beta_break_score", "window"): 120,
    ("ts_beta_break_score", "recent_window"): 30,
    ("ts_beta_break_score", "prior_window"): 90,
    # 2026-08 V2/V3 dynamics pack: small state counts / fixed order / side
    # selectors.  A-share state-machine ``side`` remains up/down; these are
    # upper/lower tail selectors.
    ("ts_markov_persistence", "bins"): 3,
    ("ts_markov_state_entropy", "bins"): 3,
    ("ts_markov_transition_surprisal", "bins"): 3,
    ("ts_markov_entropy_production", "bins"): 3,
    ("ts_active_information_storage", "bins"): 3,
    ("ts_kramers_moyal_local_stability", "bins"): 5,
    ("ts_ordinal_irreversibility", "order"): 3,
    ("ts_multiscale_permutation_entropy_slope", "order"): 3,
    ("event_historical_response_mean", "mode"): "sum",
    ("ts_hill_tail_index", "side"): "upper",
    ("ts_local_lyapunov_exponent", "tau"): 1,
    ("ts_local_lyapunov_exponent", "embedding_dim"): 3,
    # 2026-08 concurrent deepening: first-passage side selectors + TE peak bins
    # (small, stays in the 2..8 joint-state space).
    ("ts_first_passage_hit_probability", "side"): "upper",
    ("ts_first_passage_conditional_time", "side"): "upper",
    ("ts_transfer_entropy_peak_lag", "bins"): 3,
    ("ts_transfer_entropy_peak_strength", "bins"): 3,
    # 2026-08 concurrent SSA / structural-level / rough-vol packs.
    ("ts_ssa_reconstruction_residual", "n_components"): 1,
    ("ts_structural_level_density", "prominence"): 0.1,
    ("ts_structural_level_strength", "prominence"): 0.1,
    ("ts_vol_pvariation_roughness", "scales"): (2, 4, 8, 16),
    ("ts_extrema_divergence_strength", "confirmation"): 3,
    ("ts_extrema_divergence_strength", "side"): "peak",
    ("ts_nearest_structural_level_distance", "confirmation"): 3,
    ("ts_nearest_structural_level_distance", "direction"): "any",
    ("ts_fractional_difference", "cutoff"): 1,
    ("ts_structural_level_density", "confirmation"): 3,
    ("ts_structural_level_strength", "confirmation"): 3,
    ("ts_forbidden_ordinal_pattern_ratio", "order"): 3,
    ("ts_extrema_confirmation_rate", "side"): "peak",
    ("ts_interval_nesting_depth", "mode"): "inside",
    ("ts_multiscale_trend_consensus", "scales"): (2, 4, 8, 16),
    ("ts_multiscale_trend_curvature", "scales"): (2, 4, 8, 16),
    ("ts_multiscale_trend_dispersion", "scales"): (2, 4, 8, 16),
    # ``delay`` doubles as a panel (report_filing_delay_surprise) and as an
    # ordinal/recurrence embedding lag; force the embedding-lag ops to scalar 1.
    ("ts_delay_intrinsic_dimension", "delay"): 1,
    ("ts_forbidden_ordinal_pattern_ratio", "delay"): 1,
    ("ts_ordinal_irreversibility", "delay"): 1,
    ("ts_permutation_entropy", "delay"): 1,
    ("ts_permutation_transition_entropy", "delay"): 1,
    ("ts_weighted_permutation_entropy", "delay"): 1,
    ("ts_recurrence_diagonal_entropy", "delay"): 1,
    ("ts_recurrence_divergence", "delay"): 1,
    ("ts_recurrence_rate", "delay"): 1,
    ("ts_recurrence_trapping_time", "delay"): 1,
}
_SPECIAL_POSITIONAL = {
    "cs_multi_resid": ("target", "exposure", "control"),
    "cs_neutralize": ("target", "exposure", "control"),
    "row_sum_skipna": ("x", "y"),
    # Variadic ranked-panel relation operators: feed exactly 3 panels.
    "relation_hhi": ("s1", "s2", "s3"),
    "relation_entropy": ("s1", "s2", "s3"),
    "relation_topk_sum": ("s1", "s2", "s3"),
    "relation_rank_weighted_sum": ("s1", "s2", "s3"),
}
_SPECIAL_KWARGS = {
    "cs_multi_resid": {"add_intercept": True, "min_obs": 8},
    "cs_neutralize": {"add_intercept": True, "min_obs": 8},
    "fin_component_score": {"component_directions": "up", "weights": None},
}
_SPECIAL_POSITIONAL = {
    **_SPECIAL_POSITIONAL,
    "fin_component_score": ("components",),
}


# ---------------------------------------------------------------------------
# Minute-source fixture
#
# Minute-source ``intra_*`` operators consume minute-frequency panels (row
# index is a minute timestamp, columns are instruments) and return one scalar
# per (TradeDate, Symbol).  The daily audit fixture cannot exercise a minute
# kernel, so these panels are built here and the audit feeds them to operators
# carrying the ``minute`` tag (plus the two pre-existing ``intraday_*``
# operators that predate the tag convention).
# ---------------------------------------------------------------------------

_MINUTES_PER_DAY = 240
_MINUTE_PREFIX_DAYS = 20

# Operators named ``intraday_*`` are daily-frequency (rolling on daily OHLC /
# session-aware but frequency-preserving) rather than minute→daily aggregators,
# so they are NOT minute-source and resolve against the daily panels.
_MINUTE_SOURCE_EXTRA: frozenset[str] = frozenset()

# Operator parameter name -> minute panel key.  These names also exist as daily
# panels, so the minute-source dispatch in ``_value`` must win for minute ops.
_MINUTE_PANEL_PARAMS: dict[str, str] = {
    "close": "minute_close",
    "open": "minute_open",
    "high": "minute_high",
    "low": "minute_low",
    "price": "minute_price",
    "amount": "minute_amount",
    "volume": "minute_volume",
    "value": "minute_value",
    "activity": "minute_activity",
    "vwap": "minute_vwap",
    "return": "minute_ret",
    "returns": "minute_ret",
    "x": "minute_close",
    # 2026-08 order-flow family inputs.
    "flow": "minute_ret",
    "locked": "minute_zero",
    # 2026-08 advanced intraday: pair-distribution second series.
    "y": "minute_ret",
    # 2026-08 V2/V3 session-recovery: minute shock-indicator event panel.
    "event": "minute_shock",
}

# Minute-source operators also consume a few daily panels that are broadcast
# onto the minute grid by the operator itself (realized beta's market-cap
# weights, limit prices).
_MINUTE_DAILY_PARAMS: frozenset[str] = frozenset(
    {"free_market_cap", "high_limit", "low_limit",
     "benchmark_ret", "market_ret", "market", "benchmark"}
)


def _minute_source(canonical: str) -> bool:
    """Return True for minute-frequency-source canonicals."""
    if canonical.startswith("intra_") or canonical in _MINUTE_SOURCE_EXTRA:
        return True
    from cleaned_operators.registry import OperatorRegistry

    operator = OperatorRegistry.get(canonical, "pandas_numpy")
    if operator is None:
        return False
    tags = tuple(getattr(operator.metadata, "tags", ()) or ())
    return "minute" in tags


def _minute_index(dates: pd.DatetimeIndex, minutes_per_day: int = _MINUTES_PER_DAY):
    """Naive per-minute DatetimeIndex across the same business days as ``dates``.

    Minute-of-day spans 571..810 (morning 09:31..11:30, afternoon 13:01..15:00
    approximate), so ``minute_of_day``-based segment masks see real session
    structure instead of a flat 00:00..03:59 block.
    """
    offset = 571
    return pd.DatetimeIndex(
        [
            d + pd.Timedelta(minutes=offset + m)
            for d in dates
            for m in range(minutes_per_day)
        ]
    )


def _minute_panels(dates, assets, rows):
    """Deterministic synthetic minute-frequency panels (same days as daily)."""
    per = _MINUTES_PER_DAY
    n_days = int(rows)
    columns = len(assets)
    days = np.arange(n_days, dtype=float)[:, None, None]
    bars = np.arange(per, dtype=float)[None, :, None]
    assets_vec = np.arange(columns, dtype=float)[None, None, :]
    rng = np.random.default_rng(7)
    shock = rng.normal(0.0, 0.002, size=(n_days, per, columns))
    season = 0.0006 * np.sin(bars / 20.0 + assets_vec)
    drift = 0.00005 * days
    ret3 = shock + season + drift  # (days, bars, assets)
    log_p = np.cumsum(ret3, axis=1)
    base = 50.0 + 0.15 * days + 0.7 * assets_vec
    close3 = np.exp(log_p) * base
    open3 = np.concatenate([close3[:, :1] * (1.0 + ret3[:, :1]), close3[:, :-1]], axis=1)
    high3 = np.maximum(open3, close3) * 1.004
    low3 = np.minimum(open3, close3) * 0.996
    vol3 = 200_000.0 + 3_000.0 * bars + 12_000.0 * assets_vec + 50_000.0 * np.abs(shock)
    amt3 = vol3 * close3
    price3 = (high3 + low3 + close3) / 3.0
    ret4 = ret3  # per-bar log return
    absret3 = np.abs(ret3)

    def frame(arr):
        return pd.DataFrame(arr.reshape(n_days * per, columns), index=_minute_index(dates), columns=assets)

    minute_ret = frame(ret4)
    return {
        "minute_close": frame(close3),
        "minute_open": frame(open3),
        "minute_high": frame(high3),
        "minute_low": frame(low3),
        "minute_price": frame(price3),
        "minute_volume": frame(vol3),
        "minute_amount": frame(amt3),
        "minute_value": frame(vol3 * (1.0 + 0.5 * np.abs(shock))),
        "minute_activity": frame(vol3 * (1.0 + np.abs(shock))),
        "minute_abs_return": frame(absret3),
        "minute_ret": minute_ret,
        "minute_vwap": frame(amt3 / vol3),
        # 2026-08 order-flow family: no limit-locked bars in the audit fixture,
        # so the ``locked`` mask is all zeros (neutral override not exercised).
        "minute_zero": frame(np.zeros((n_days * per, columns))),
        # 2026-08 session-recovery: deterministic minute shock indicator
        # (abs return > ~2 sigma of the synthetic per-minute noise).
        "minute_shock": frame((absret3 > 0.004).astype(float)),
    }


def _is_minute_frame(value: Any) -> bool:
    """True if a DataFrame/Series index is minute-frequency (>=2 bars on day 0)."""
    if not isinstance(value, (pd.DataFrame, pd.Series)) or len(value) < 2:
        return False
    idx = value.index
    if not isinstance(idx, pd.DatetimeIndex):
        return False
    norm = idx.normalize()
    return bool(norm[0] == norm[1])


def _slice_minute(value: Any, days: int) -> Any:
    """Slice to the first ``days`` complete trading days (minute-aware)."""
    if _is_minute_frame(value):
        return value.iloc[: days * _MINUTES_PER_DAY]
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.iloc[:days]
    return value


def _panels(rows: int = 220, columns: int = 6) -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    assets = [f"A{index}" for index in range(columns)]
    time = np.arange(rows, dtype=float)[:, None]
    asset = np.arange(columns, dtype=float)[None, :]
    close = pd.DataFrame(
        50.0 + 0.15 * time + 0.7 * asset + np.sin(time / 5.0 + asset / 3.0),
        index=dates,
        columns=assets,
    )
    open_ = close * (1.0 + 0.002 * np.cos(time / 4.0 + asset))
    high = pd.DataFrame(
        np.maximum(open_, close) * 1.01, index=dates, columns=assets
    )
    low = pd.DataFrame(
        np.minimum(open_, close) * 0.99, index=dates, columns=assets
    )
    volume = pd.DataFrame(
        1_000_000.0 + 5_000.0 * time + 10_000.0 * asset,
        index=dates,
        columns=assets,
    )
    amount = volume * close
    returns = close.pct_change().fillna(0.0)
    market = pd.DataFrame(
        np.repeat(returns.mean(axis=1).to_numpy()[:, None], columns, axis=1),
        index=dates,
        columns=assets,
    )
    groups = np.array(["G0", "G1", "G2", "G0", "G1", "G2"], dtype=object)
    group = pd.DataFrame(
        np.tile(groups[:columns], (rows, 1)), index=dates, columns=assets
    )
    condition = volume.gt(volume.rolling(5, min_periods=1).mean())

    report_number = np.arange(rows) // 10
    period = pd.DataFrame(
        np.repeat(report_number[:, None], columns, axis=1),
        index=dates,
        columns=assets,
    )
    target_period = pd.DataFrame(
        np.repeat((report_number + 1)[:, None], columns, axis=1),
        index=dates,
        columns=assets,
    )
    fiscal_quarter = pd.DataFrame(
        np.repeat((report_number % 4 + 1)[:, None], columns, axis=1),
        index=dates,
        columns=assets,
    )
    revision = pd.DataFrame(
        np.repeat((np.arange(rows) // 5)[:, None], columns, axis=1),
        index=dates,
        columns=assets,
    )
    decision = pd.DataFrame(
        np.repeat(dates.to_numpy()[:, None], columns, axis=1),
        index=dates,
        columns=assets,
    )
    available = decision - pd.Timedelta(days=3)
    weights = volume.div(volume.sum(axis=1), axis=0)
    control = volume.pct_change().fillna(0.0)
    zero = pd.DataFrame(0.0, index=dates, columns=assets)
    float_shares = pd.DataFrame(
        np.repeat(
            (100_000_000.0 + np.arange(rows, dtype=float)[:, None] * 10_000.0),
            columns,
            axis=1,
        ),
        index=dates,
        columns=assets,
    )
    turnover = volume / float_shares

    panels: dict[str, pd.DataFrame] = {
        "x": close,
        "y": open_,
        "z": control,
        "a": close,
        "b": open_,
        "w": weights,
        "g": group,
        "left": close,
        "right": open_,
        "numerator": close,
        "denominator": open_.abs() + 1.0,
        "ret": returns,
        "returns": returns,
        "benchmark_ret": market,
        "market_ret": market,
        "benchmark": market,
        "market": market,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "price": (high + low + close) / 3.0,
        "volume": volume,
        "amount": amount,
        "vwap": amount / volume,
        "turnover": turnover,
        "weight": weights,
        "weights": weights,
        "signal": returns,
        "fallback": zero,
        "condition": condition,
        "mask": condition,
        "event": condition,
        "group": group,
        "industry": group,
        "sector": group,
        "fiscal_quarter": fiscal_quarter,
        "period_id": period,
        "target_period_id": target_period,
        "quarter": fiscal_quarter,
        "revision_id": revision,
        "decision_time": decision,
        "available_time": available,
        "available_at": available,
        "snapshot_date": decision,
        "previous_snapshot_date": available,
        "exposure": market,
        "exposures": market,
        "control": control,
        "controls": control,
        "factor": market,
        "target": returns,
        "sort_col": volume,
        "value": close,
        "values": close,
        "v1": close,
        "v2": open_,
        "float_shares": float_shares,
        "flow": close,
        "balance": open_.abs() + 10.0,
        "earnings": close,
        "cashflow": open_,
        "assets": close.abs() + 100.0,
        "working_capital": close,
        "base": close.abs() + 100.0,
        "dollar_volume": amount,
        "scale_base": close.abs() + 100.0,
        "actual": close * 1.02,
        "expected": close,
        "expected_std": close.abs() * 0.1 + 1.0,
        "expected_mean": close,
        # 2026-08 stratified-mean-spread sorter input (volume-sorted recipes).
        "sorter": volume,
        # 2026-08 V2/V3 report-timing: sawtooth filing delay (days) and the
        # previous published value for revision magnitude.
        "delay": pd.DataFrame(
            np.tile(((np.arange(rows) // 3) % 10 + 1)[:, None], (1, columns)),
            index=dates,
            columns=assets,
        ),
        "prev_x": close.shift(1).fillna(close),
    }
    panels.update(_minute_panels(dates, assets, rows))
    for name in sorted(_PANEL_PARAMETERS):
        if name in panels:
            continue
        panels[name] = close * (1.0 + 0.001 * (len(panels) % 17)) + 10.0
    return panels


def _value(canonical: str, name: str, panels: dict[str, pd.DataFrame]) -> Any:
    key = str(name)
    special = _SPECIAL_SCALARS.get((canonical, key))
    if special is not None:
        return special
    if _minute_source(canonical):
        if key in _MINUTE_PANEL_PARAMS:
            return panels[_MINUTE_PANEL_PARAMS[key]]
        if key in _MINUTE_DAILY_PARAMS:
            return panels[key]
    if key in panels:
        return panels[key]
    if (canonical, key) in _PANEL_FORCE:
        return panels["x"]
    if key in _SCALAR_VALUES:
        return _SCALAR_VALUES[key]
    if key.endswith("_window") or key.startswith("window"):
        return 20
    if key.endswith("_periods"):
        return 4
    if key.endswith("_days"):
        return 60
    if key.startswith("min_"):
        return 0.01
    if key.startswith("max_"):
        return 0.5
    if key.endswith("_id"):
        return panels["period_id"]
    if key in _PANEL_PARAMETERS:
        return panels["x"]
    # Generic scalar fallbacks for concurrent expansion families (recurrence /
    # transport / MMD / chord).  Additive and deterministic: unknown parameters
    # fail closed to a runnable scalar instead of aborting the whole audit.
    if key in {"dim", "tau", "order", "degree", "k", "n_components"}:
        return 3
    if key.endswith("_fraction") or key.startswith("fraction"):
        return 0.1
    if key.endswith("_dim"):
        return 3
    if key.startswith("eps"):
        return 0.1
    raise KeyError(f"unclassified public parameter {canonical}.{key}")


def _build_call(canonical: str, operator: Any, panels: dict[str, pd.DataFrame]):
    from cleaned_operators.registry import OperatorRegistry

    names = tuple(
        str(value)
        for value in (
            OperatorRegistry._catalog.get(canonical, {}).get("param_names") or ()
        )
    ) or ("x",)
    if canonical in _SPECIAL_POSITIONAL:
        arguments = [
            _value(canonical, name, panels)
            for name in _SPECIAL_POSITIONAL[canonical]
        ]
        return arguments, dict(_SPECIAL_KWARGS.get(canonical, {}))
    return [
        _value(canonical, name, panels) for name in names if name != "..."
    ], dict(_SPECIAL_KWARGS.get(canonical, {}))


def _slice(value: Any, rows: int) -> Any:
    return value.iloc[:rows] if isinstance(value, (pd.DataFrame, pd.Series)) else value


def _to_frame(value: Any, template: pd.DataFrame) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        if value.index.equals(template.index):
            return pd.DataFrame(
                np.repeat(value.to_numpy()[:, None], template.shape[1], axis=1),
                index=template.index,
                columns=template.columns,
            )
        raise TypeError("Series result does not share the time index")
    array = np.asarray(value)
    if array.shape == template.shape:
        return pd.DataFrame(array, index=template.index, columns=template.columns)
    if np.isscalar(value):
        return pd.DataFrame(value, index=template.index, columns=template.columns)
    raise TypeError(
        f"unsupported result shape/type: {type(value).__name__} {array.shape}"
    )


def _equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if (
        left.shape != right.shape
        or not left.index.equals(right.index)
        or not left.columns.equals(right.columns)
    ):
        return False
    try:
        return bool(
            np.allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                equal_nan=True,
                rtol=1e-6,
                atol=1e-8,
            )
        )
    except (TypeError, ValueError):
        return left.astype(object).where(pd.notna(left), None).equals(
            right.astype(object).where(pd.notna(right), None)
        )


def _numeric_delta(left: pd.DataFrame, right: pd.DataFrame) -> float | None:
    try:
        left_values = left.to_numpy(dtype=float)
        right_values = right.to_numpy(dtype=float)
    except (TypeError, ValueError):
        return None
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(left_values[finite] - right_values[finite])))


def _implementation_label(operator: Any) -> str:
    cls = type(operator)
    source = inspect.getsourcefile(cls)
    try:
        source = str(Path(source).resolve().relative_to(FE_ROOT)) if source else "?"
    except Exception:
        source = str(source or "?")
    return f"{cls.__module__}.{cls.__name__}@{source}"


def audit(*, require_admission: bool = True) -> list[str]:
    from backend.operator_capability import production_eligible_backends
    from cleaned_operators import load_all
    from cleaned_operators.operator_policy import infer_operator_policy
    from cleaned_operators.operator_spec import build_operator_spec
    from cleaned_operators.production_hardening import factor_production_targets
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    errors: list[str] = []
    panels = _panels()
    template = panels["x"]
    prefix_rows = 160

    from cleaned_operators.semantic_certification import should_fail_closed

    for canonical in sorted(factor_production_targets()):
        if should_fail_closed(canonical):
            # Registered experimental/research or isolated: hardening keeps them
            # experimental and non-PIT-safe by design; not a production defect.
            continue
        operator = OperatorRegistry.get(canonical, "pandas_numpy")
        if operator is None:
            errors.append(f"{canonical}: no pandas semantic reference")
            continue
        catalog = OperatorRegistry._catalog.get(canonical, {})
        policy = infer_operator_policy(operator, canonical=canonical)
        # Six-gate certification is the single production authority (review
        # §2.1/§2.4): status may only be "production" when the evidence overlay
        # has certified the operator.  During a pre-evidence runtime audit an
        # uncertified target is correctly experimental (fail-closed), so this
        # check enforces the invariant production ⟺ certified rather than
        # demanding every reviewed target be promoted regardless of evidence.
        certified = catalog.get("production_certified") is True
        if (str(catalog.get("status")) == "production") != certified:
            errors.append(
                f"{canonical}: status must be production iff six-gate certified"
            )
        if not policy.pit_safe or policy.lag < 0:
            errors.append(f"{canonical}: PIT policy is not causal")
        if not policy.shape_preserving:
            errors.append(f"{canonical}: shape_preserving=False")
        if require_admission:
            specification = build_operator_spec(canonical)
            if specification is None or not specification.allow_in_production:
                errors.append(f"{canonical}: OperatorSpec.allow_in_production=False")
                continue
            if not production_eligible_backends(canonical):
                errors.append(f"{canonical}: no production eligible backend")
                continue
        try:
            arguments, keyword_arguments = _build_call(canonical, operator, panels)
            first = _to_frame(
                operator.calculate(*arguments, **keyword_arguments), template
            )
            second = _to_frame(
                operator.calculate(*arguments, **keyword_arguments), template
            )
            if not first.index.equals(template.index) or not first.columns.equals(
                template.columns
            ):
                errors.append(
                    f"{canonical}: output axes changed [{_implementation_label(operator)}]"
                )
                continue
            if not _equal(first, second):
                errors.append(
                    f"{canonical}: non-deterministic repeated evaluation "
                    f"[{_implementation_label(operator)}]"
                )
                continue
            if _minute_source(canonical):
                # Minute kernels aggregate per complete trading day, so the
                # prefix check slices inputs to whole days and compares the
                # same number of daily output rows.
                prefix_args = [_slice_minute(value, _MINUTE_PREFIX_DAYS) for value in arguments]
                prefix_kwargs = {
                    key: _slice_minute(value, _MINUTE_PREFIX_DAYS)
                    for key, value in keyword_arguments.items()
                }
                prefix_template = template.iloc[:_MINUTE_PREFIX_DAYS]
                prefix = _to_frame(
                    operator.calculate(*prefix_args, **prefix_kwargs),
                    prefix_template,
                )
                historical = first.iloc[:_MINUTE_PREFIX_DAYS]
            else:
                prefix_args = [_slice(value, prefix_rows) for value in arguments]
                prefix_kwargs = {
                    key: _slice(value, prefix_rows)
                    for key, value in keyword_arguments.items()
                }
                prefix_template = template.iloc[:prefix_rows]
                prefix = _to_frame(
                    operator.calculate(*prefix_args, **prefix_kwargs),
                    prefix_template,
                )
                historical = first.iloc[:prefix_rows]
            if not _equal(historical, prefix):
                delta = _numeric_delta(historical, prefix)
                details = (
                    f" (max_abs_delta={delta:.6g})" if delta is not None else ""
                )
                errors.append(
                    f"{canonical}: prefix invariance / causality violation{details} "
                    f"[{_implementation_label(operator)}]"
                )
        except Exception as error:
            errors.append(
                f"{canonical}: {type(error).__name__}: {error} "
                f"[{_implementation_label(operator)}]"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="skip admission evidence and audit semantic runtime only",
    )
    arguments = parser.parse_args()
    errors = audit(require_admission=not arguments.runtime_only)
    if errors:
        print(
            f"factor production audit FAILED ({len(errors)} issues)",
            file=sys.stderr,
        )
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    from cleaned_operators.production_hardening import factor_production_targets

    phase = "runtime" if arguments.runtime_only else "admission"
    print(
        f"factor production {phase} audit passed "
        f"({len(factor_production_targets())} canonicals)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
