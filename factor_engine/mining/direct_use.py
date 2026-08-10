# -*- coding: utf-8 -*-
"""R18 "direct usability" layer — the single authority for what a miner may use.

R18's thesis: "registered / implemented / daily-surface / production-certified"
is NOT the same as "an automated miner can use it today".  Every public
canonical must have a final ``DirectUseStatus`` and a legal AST role, or it is
moved/deleted.  This module provides:

* :class:`DirectUseStatus` — the R18 §1 machine-readable verdict vocabulary.
* :func:`resolve_direct_use_status` — deterministic, fail-closed classification
  that combines an *explicit* family override table (R18-009..019 and friends,
  which the old surface rule wrongly promoted to ALPHA) with the existing
  ``MiningRole`` authority.
* :func:`get_direct_use_mining_operators` — the ONE entry point automated mining
  (AlphaProbe / AlphaMiner / cold-start) may read.  R18-001: the legacy
  ``backend.fastpath_allowlists`` only expresses backend execution capability;
  it never decides what a miner may see.
* :func:`TERMINAL_ROLES` / ``terminal_allowed`` — a single *positive* authority
  (R18-003).  A role is terminal only if it is listed; nothing falls through.
* Input-slot contracts (R18-029/002): ``data_inputs`` / ``scalar_parameters`` /
  ``context_inputs`` / ``group_inputs`` / ``event_inputs`` are separate; an
  ``inputs`` key, where kept, is exactly the data-input list.
* ``economic_effect_family`` / ``semantic_redundancy_group`` (R18-068/069),
  ``monotonic_transform_class`` (R18-021), ``default_input_recipe`` (R18-070),
  effective-sample contracts (R18-047), retention metadata (R18-091..094).

Everything is registry-level and read-only over the operator implementations —
no operator kernel is modified here, and no production-certification decision is
re-litigated.  The R18 DirectUse Matrix / manifests are generated from this one
module, so the artifacts can never drift from the code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

from mining.operator_catalog import (
    MiningRole,
    RoleSource,
    _ROLE_AST_POSITIONS,
    assign_mining_role_ex,
    cost_contract_declared,
    cost_tier,
    market_support,
    mining_eligible,
    source_status,
)

# ---------------------------------------------------------------------------
# R18 §1: DirectUseStatus vocabulary
# ---------------------------------------------------------------------------


class DirectUseStatus(str, Enum):
    """R18 §1 machine verdict for every registered canonical.

    The public factor/mining layer must never contain ``UNRESOLVED`` /
    ``UNKNOWN`` / ``PENDING_FOREVER`` / ``MAYBE_ALPHA`` — every retained
    operator gets one of the DIRECT_* statuses (or is moved/deleted).
    """

    DIRECT_ALPHA = "direct_alpha"
    DIRECT_ALPHA_HIGH_COST = "direct_alpha_high_cost"
    DIRECT_STATE = "direct_state"
    DIRECT_CONDITION = "direct_condition"
    DIRECT_EVENT = "direct_event"
    DIRECT_GROUP_STATE = "direct_group_state"
    DIRECT_GLOBAL_STATE = "direct_global_state"
    DIRECT_INTERMEDIATE = "direct_intermediate"
    DIRECT_SOURCE_TRANSFORM = "direct_source_transform"
    DIRECT_RECIPE = "direct_recipe"
    DIRECT_CONTROL_FLOW = "direct_control_flow"

    # supporting / diagnostic layers
    RESEARCH_TOOL = "research_tool"
    MOVE_INTERNAL = "move_internal"

    # R18 §1 delete classes
    DELETE_DUPLICATE = "delete_duplicate"
    DELETE_NO_DATA = "delete_no_data"
    DELETE_NONCAUSAL = "delete_noncausal"
    DELETE_MATH_DEFECT = "delete_math_defect"
    DELETE_USELESS = "delete_useless"
    DELETE_OBSOLETE = "delete_obsolete"


# The ONLY statuses that may ever stand alone as a factor terminal
# (R18-003: positive authority — nothing is terminal by exclusion).
_DIRECT_TERMINAL_STATUSES = frozenset(
    {
        DirectUseStatus.DIRECT_ALPHA,
        DirectUseStatus.DIRECT_ALPHA_HIGH_COST,
        DirectUseStatus.DIRECT_RECIPE,
    }
)

# R18-080 search lanes.  SOURCE_TRANSFORM never consumes alpha tree depth.
_DIRECT_STATUS_LANES: dict[DirectUseStatus, str] = {
    DirectUseStatus.DIRECT_ALPHA: "alpha_direct",
    DirectUseStatus.DIRECT_ALPHA_HIGH_COST: "alpha_high_cost",
    DirectUseStatus.DIRECT_STATE: "state",
    DirectUseStatus.DIRECT_CONDITION: "condition",
    DirectUseStatus.DIRECT_EVENT: "event",
    DirectUseStatus.DIRECT_GROUP_STATE: "group_state",
    DirectUseStatus.DIRECT_GLOBAL_STATE: "global_state",
    DirectUseStatus.DIRECT_INTERMEDIATE: "intermediate",
    DirectUseStatus.DIRECT_SOURCE_TRANSFORM: "source_transform",
    DirectUseStatus.DIRECT_RECIPE: "alpha_direct",
    DirectUseStatus.DIRECT_CONTROL_FLOW: "control_flow",
}

_AST_POSITIONS_BY_STATUS: dict[DirectUseStatus, tuple[str, ...]] = {
    DirectUseStatus.DIRECT_ALPHA: ("intermediate", "terminal"),
    DirectUseStatus.DIRECT_ALPHA_HIGH_COST: ("intermediate", "terminal"),
    DirectUseStatus.DIRECT_STATE: ("gate", "interaction"),
    DirectUseStatus.DIRECT_CONDITION: ("condition",),
    DirectUseStatus.DIRECT_EVENT: ("condition", "mask"),
    DirectUseStatus.DIRECT_GROUP_STATE: ("gate", "interaction"),
    DirectUseStatus.DIRECT_GLOBAL_STATE: ("gate", "interaction"),
    DirectUseStatus.DIRECT_INTERMEDIATE: ("intermediate",),
    DirectUseStatus.DIRECT_SOURCE_TRANSFORM: ("data_processing",),
    DirectUseStatus.DIRECT_RECIPE: ("intermediate", "terminal"),
    DirectUseStatus.DIRECT_CONTROL_FLOW: ("control_flow",),
    DirectUseStatus.RESEARCH_TOOL: (),
    DirectUseStatus.MOVE_INTERNAL: (),
    DirectUseStatus.DELETE_DUPLICATE: (),
    DirectUseStatus.DELETE_NO_DATA: (),
    DirectUseStatus.DELETE_NONCAUSAL: (),
    DirectUseStatus.DELETE_MATH_DEFECT: (),
    DirectUseStatus.DELETE_USELESS: (),
    DirectUseStatus.DELETE_OBSOLETE: (),
}

# ---------------------------------------------------------------------------
# Explicit family overrides (R18-009..019 and friends).
#
# The old surface rule (`surface == daily/extended -> ALPHA`) promoted these to
# ALPHA.  R18 fixes the role so a miner never treats a Bool, a category id, a
# global broadcast, a price level, or a missing-fill as a standalone alpha
# terminal.  Key: canonical -> (status, terminal_allowed, retention_reason).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectUseContract:
    status: DirectUseStatus
    terminal_allowed: bool
    retention_reason: str
    ast_positions: tuple[str, ...] = ()
    replacement: str = ""
    delete_reason: str = ""


def _contract(
    status: DirectUseStatus,
    reason: str,
    *,
    terminal: bool | None = None,
    replacement: str = "",
    delete_reason: str = "",
) -> DirectUseContract:
    if terminal is None:
        terminal = status in _DIRECT_TERMINAL_STATUSES
    pos = _AST_POSITIONS_BY_STATUS[status]
    return DirectUseContract(
        status=status,
        terminal_allowed=terminal,
        retention_reason=reason,
        ast_positions=pos,
        replacement=replacement,
        delete_reason=delete_reason,
    )


# R18-009: Boolean / comparison / missingness are CONDITIONs, not numeric alphas.
_BOOL_CONDITION_OPS = frozenset(
    {
        "and_", "or_", "not_",
        "eq", "ne", "gt", "ge", "lt", "le",
        "is_finite", "is_infinite", "is_null", "is_not_null", "is_nan",
        "eq_zero", "is_zero",
    }
)

# R18-010: cross-sectional *aggregates* broadcast the same value to every stock
# that day — a GLOBAL_STATE, never a stock-level alpha terminal.  Per-stock
# relative transforms (cs_rank / cs_zscore / cs_percentile) stay ALPHA.
_CS_GLOBAL_STATE_OPS = frozenset(
    {
        "cs_count", "cs_mean", "cs_std", "cs_sum", "cs_mad", "cs_quantile",
        "cs_valid_count", "cs_coverage_ratio", "cs_hartigan_dip",
        "cs_median", "cs_skew", "cs_kurtosis", "cs_min", "cs_max",
        "cs_spread", "cs_entropy", "cs_hhi", "cs_lorenz",
    }
)

# R18-011: group *aggregates* broadcast within the group -> GROUP_STATE.  The
# per-stock relative forms (group_rank / group_zscore / group_neutralize /
# group_percentile) are ALPHA and are NOT listed here.
_GROUP_STATE_OPS = frozenset(
    {
        "group_count", "group_max", "group_mean", "group_min", "group_std",
        "group_sum", "group_weighted_mean", "group_valid_count",
        "group_distribution_js_divergence", "group_feature_spectral_gap",
        "group_feature_second_mode_localization",
        "group_median", "group_skew", "group_kurtosis", "group_mad",
        "group_quantile", "group_entropy",
    }
)

# R18-012: bucket / categorical outputs are STATE (category condition /
# interaction), never continuous-strength ranks.
_CATEGORY_STATE_OPS = frozenset(
    {
        "cs_bucket", "cs_bucket_fixed", "cs_bucket_historical",
        "candlestick_pattern", "bin", "quantile_bucket",
    }
)

# R18-013: K-line binary patterns / breakout indicators.  Output is 0/1/NaN (or
# {-1,0,1}) -> EVENT.  A continuous strength variant is a separate ALPHA op.
_CDL_EVENT_OPS = frozenset(
    {
        "cdl_doji", "cdl_hammer", "cdl_inverted_hammer", "cdl_shooting_star",
        "cdl_marubozu", "cdl_spinning_top", "cdl_engulfing", "cdl_inside_bar",
        "cdl_outside_bar", "cdl_dragonfly_doji", "cdl_gravestone_doji",
        "cdl_hanging_man", "cdl_harami", "cdl_harami_cross", "cdl_piercing",
        "cdl_dark_cloud_cover", "cdl_morning_star", "cdl_evening_star",
        "cdl_three_white_soldiers", "cdl_three_black_crows",
        "cdl_tweezer_top", "cdl_tweezer_bottom",
    }
)
_CDL_EVENT_LANE = "candlestick_event"

# R18-013: breakout / new-high indicators whose base form is a 0/1 event mask.
# Continuous *strength* variants (e.g. breakout strength over the barrier) are
# separate ALPHA ops and are not listed.
_BREAKOUT_EVENT_OPS = frozenset(
    {
        "ts_breakout_high", "ts_breakdown_low", "ts_new_high", "ts_new_low",
        "ts_resistance_break", "ts_support_break",
    }
)

# R18-014: A-share limit family — Boolean touch/close/failed/open-limit are
# EVENT/STATE; continuous statistics (distance / density / asymmetry /
# days-since / streak) are ALPHA/STATE statistics and stay terminal.
_LIMIT_EVENT_OPS = frozenset(
    {
        "limit_up_close", "limit_down_close", "limit_up_open", "limit_down_open",
        "ashare_limit_touch", "ashare_limit_one_price", "ashare_limit_failed",
        "ashare_limit_open_failed", "ashare_open_at_upper_limit",
        "ashare_limit_touch_count",
    }
)

# R18-015: explicit per-operator role for the fin/index mask family.
_FIN_INDEX_ROLE = {
    "fin_applicability_mask": _contract(DirectUseStatus.DIRECT_CONDITION, "applicability gate — condition slot only"),
    "index_member": _contract(DirectUseStatus.DIRECT_STATE, "index membership — discrete state gate"),
    "index_entry_exit_event": _contract(DirectUseStatus.DIRECT_EVENT, "index entry/exit — event mask"),
}

# R18-016: price-level raw building blocks are INTERMEDIATE — not scale-invariant,
# never default terminal.  The normalized/relative counterparts (R18-017) are
# ALPHA and are NOT listed here.
_PRICE_LEVEL_INTERMEDIATE_OPS = frozenset(
    {
        "rolling_vwap", "true_range", "ATR_WILDER",
        "donchian_upper", "donchian_lower", "donchian_mid",
        "KeltnerMid", "KeltnerUpper", "KeltnerLower",
        "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a",
        "ichimoku_senkou_b",
        "KAMA", "DEMA", "TEMA", "PSAR", "Supertrend",
        "ts_prev_high", "ts_prev_low",
        "ts_last_pivot_high", "ts_last_pivot_low",
        "ts_nth_pivot_high", "ts_nth_pivot_low",
        "ts_resistance_level", "ts_support_level",
        "candle_body", "candle_abs_body", "candle_range",
        "candle_upper_shadow", "candle_lower_shadow", "candle_gap",
        "ts_swing_amplitude", "ts_channel_width", "ts_consolidation_width",
        "ts_ema", "ts_sma", "ts_wma", "ts_hma", "ts_dema",
        "bollinger_upper", "bollinger_lower", "bollinger_mid",
    }
)

# R18-019: missing-value / source transforms are data-processing, never alpha
# search branches.
_SOURCE_TRANSFORM_OPS = frozenset(
    {
        "fillna_const", "coalesce",
        "cs_fill_mean", "cs_fill_median",
        "cs_impute_mean", "cs_impute_median",
        "group_impute_median",
        "ffill_limit", "ts_ffill_limited",
    }
)

# R18-020: arbitrary math — keep only what has a real finance/stat use.
#  * trig: cyclic/noise in the raw form -> research tool (phase use is a recipe)
#  * floor/ceil/round/fix: discrete jumps -> intermediate (bucketing building block)
#  * monotonic powers/roots: intermediate normalizers (R18-021 family)
_ARBITRARY_MATH_STATUS: dict[str, DirectUseContract] = {
    "sin": _contract(DirectUseStatus.RESEARCH_TOOL, "cyclic transform — noise in raw form; used only inside a phase/seasonality recipe"),
    "cos": _contract(DirectUseStatus.RESEARCH_TOOL, "cyclic transform — noise in raw form; used only inside a phase/seasonality recipe"),
    "acos": _contract(DirectUseStatus.RESEARCH_TOOL, "cyclic transform — domain-narrow; recipe-only"),
    "asin": _contract(DirectUseStatus.RESEARCH_TOOL, "cyclic transform — domain-narrow; recipe-only"),
    "atan": _contract(DirectUseStatus.DIRECT_INTERMEDIATE, "phase extraction building block (atan2 preferred)"),
    "atan2": _contract(DirectUseStatus.DIRECT_INTERMEDIATE, "phase/angle extraction building block"),
    "floor": _contract(DirectUseStatus.DIRECT_INTERMEDIATE, "discrete-jump bucketing building block"),
    "ceil": _contract(DirectUseStatus.DIRECT_INTERMEDIATE, "discrete-jump bucketing building block"),
    "round": _contract(DirectUseStatus.DIRECT_INTERMEDIATE, "discrete-jump quantization building block"),
    "fix": _contract(DirectUseStatus.DIRECT_INTERMEDIATE, "discrete-jump quantization building block"),
    "cbrt": _contract(DirectUseStatus.DIRECT_INTERMEDIATE, "monotonic root — normalizer building block"),
    "sqrt_abs": _contract(DirectUseStatus.DIRECT_INTERMEDIATE, "monotonic root — normalizer building block"),
    "square": _contract(DirectUseStatus.DIRECT_INTERMEDIATE, "monotonic power — normalizer building block"),
    "signed_power": _contract(DirectUseStatus.DIRECT_INTERMEDIATE, "monotonic power — normalizer building block"),
    "power": _contract(DirectUseStatus.DIRECT_INTERMEDIATE, "monotonic power — normalizer building block"),
    "lerp": _contract(DirectUseStatus.DIRECT_CONTROL_FLOW, "linear interpolation — control-flow building block"),
}

# R18-061: flex_max / flex_min are exact semantic duplicates of maximum/minimum
# (only the input flexibility differs).  One canonical + backend policy, no
# second search branch.
_FLEX_DUP_REPLACEMENT = {
    "flex_max": "maximum",
    "flex_min": "minimum",
}

# R18-021: monotonic transform class — Rank-IC / search dedups strict-monotonic
# equivalents so x / exp(x) / sigmoid(x) / rank(x) are ONE capability family.
MONOTONIC_TRANSFORM_CLASS: dict[str, str] = {
    "exp": "monotonic_increasing",
    "log": "monotonic_increasing",
    "sqrt": "monotonic_increasing",
    "signed_log": "monotonic_increasing",
    "sigmoid": "monotonic_increasing",
    "tanh": "monotonic_increasing",
    "rank": "monotonic_equiv",
    "scale": "affine_equiv",
    "normalize": "monotonic_equiv",
    "unitize": "monotonic_equiv",
    "square": "monotonic_piecewise",
    "cbrt": "monotonic_increasing",
    "sqrt_abs": "monotonic_piecewise",
    "signed_power": "monotonic_piecewise",
}

# R18-068: economic effect family (for cold-start diversity, NOT IC-based).
_ECONOMIC_EFFECT_FAMILY: dict[str, str] = {
    # trend / momentum
    "ts_mean": "trend", "ts_sma": "trend", "ts_ema": "trend", "ts_dema": "trend",
    "ts_slope": "trend", "ts_regression_slope": "trend", "ts_lin_reg": "trend",
    "ts_momentum": "trend", "ts_return": "trend", "ts_log_return": "trend",
    "MACD_line": "trend", "MACD_signal": "trend", "MACD_hist": "trend",
    "PPO": "trend", "PPO_signal": "trend", "PPO_hist": "trend",
    "TSI": "trend", "TSI_signal": "trend",
    "DMI_plus": "trend", "DMI_minus": "trend", "DX": "trend",
    "ts_breakout_high": "breakout", "ts_breakdown_low": "breakout",
    "ts_new_high": "breakout", "ts_new_low": "breakout",
    "ts_resistance_break": "breakout", "ts_support_break": "breakout",
    # mean reversion
    "ts_zscore": "mean_reversion", "ts_bollinger_zscore": "mean_reversion",
    "bollinger_pct_b": "mean_reversion", "bollinger_width": "volatility",
    "ts_mean_reversion_half_life": "mean_reversion",
    "ts_autocorr": "autocorrelation", "ts_hurst_exponent": "mean_reversion",
    "ts_penetration": "mean_reversion",
    # volatility
    "ts_volatility": "volatility", "ts_std": "volatility",
    "parkinson_vol": "volatility", "garman_klass_vol": "volatility",
    "rogers_satchell_vol": "volatility", "yang_zhang_vol": "volatility",
    "ATR_WILDER": "volatility", "NATR": "volatility", "true_range": "volatility",
    "ts_vol_pvariation_roughness": "volatility",
    # liquidity / volume
    "dollar_volume": "liquidity", "ts_average_volume": "volume",
    "adv": "volume", "amihud_illiquidity": "liquidity",
    "rolling_obv": "volume", "rolling_pvt": "volume",
    "turnover": "volume", "ts_turnover": "volume",
    # price-volume interaction
    "ts_volume_ratio": "price_volume_interaction",
    "vwap_deviation": "price_volume_interaction",
    "rolling_vwap": "price_volume_interaction",
    "volume_clock": "price_volume_interaction",
    # range / gap
    "ts_range": "range", "candle_range": "range",
    "candle_gap": "gap", "candle_gap_pct": "gap", "candle_gap_atr": "gap",
    "ts_gap": "gap",
    # tail
    "ts_skew": "distribution_shape", "ts_kurtosis": "distribution_shape",
    "ts_max_drawdown": "tail", "ts_var": "tail", "ts_cvar": "tail",
    "ts_cvar_deviation": "tail", "ts_expected_shortfall": "tail",
    "ts_gpd_shape_pwm": "tail", "ts_extremal_index": "tail",
    # fundamental
    "pe": "fundamental_value", "pb": "fundamental_value",
    "ps": "fundamental_value", "pe_ttm": "fundamental_value",
    "earnings_yield": "fundamental_value", "dividend_yield": "fundamental_value",
    "roe": "fundamental_quality", "roa": "fundamental_quality",
    "operating_margin": "fundamental_quality",
    "debt_to_equity": "fundamental_quality", "current_ratio": "fundamental_quality",
    "fin_revision_*": "fundamental_revision", "fin_expectation_*": "fundamental_revision",
    # ownership / index flow
    "holder_*": "ownership", "index_*": "index_flow",
    # intraday microstructure
    "intra_*": "intraday_microstructure",
    "intraday_vwap_deviation": "intraday_microstructure",
    # event
    "event_*": "event", "cross_event": "event",
    "cdl_*": "pattern", "candlestick_pattern": "pattern",
    "state_*": "regime", "directional_change_state": "regime",
    # nonlinear dependence / spectral / geometry
    "ts_markov_*": "nonlinear_dependence",
    "ts_transfer_entropy_*": "nonlinear_dependence",
    "ts_hsic": "nonlinear_dependence",
    "ts_conditional_mutual_information": "nonlinear_dependence",
    "ts_mmd_rbf_shift": "nonlinear_dependence",
    "ts_kernel_granger_score": "nonlinear_dependence",
    "ts_spectral_*": "spectral", "ts_hankel_*": "spectral",
    "ts_ssa_*": "spectral", "ts_multifractal_*": "spectral",
    "ts_hvg_*": "geometry", "ts_matrix_profile_*": "geometry",
    "ts_delay_intrinsic_dimension": "geometry",
}

# R18-069: semantic redundancy group — high-correlation but NOT duplicate
# operators share a group and a family budget quota instead of being deleted.
_SEMANTIC_REDUNDANCY_GROUP: dict[str, str] = {
    "exp": "monotonic_transform", "log": "monotonic_transform",
    "sqrt": "monotonic_transform", "sigmoid": "monotonic_transform",
    "tanh": "monotonic_transform", "rank": "monotonic_transform",
    "normalize": "monotonic_transform", "unitize": "monotonic_transform",
    "scale": "monotonic_transform",
    "parkinson_vol": "range_vol", "garman_klass_vol": "range_vol",
    "rogers_satchell_vol": "range_vol", "yang_zhang_vol": "range_vol",
    "ts_volatility": "range_vol", "ts_std": "range_vol",
    "bollinger_pct_b": "oscillator", "bollinger_width": "oscillator",
    "MACD_hist": "oscillator", "PPO_hist": "oscillator",
    "TSI": "oscillator", "CCI": "oscillator",
    "KAMA": "smoothed_price", "DEMA": "smoothed_price",
    "TEMA": "smoothed_price", "ts_dema": "smoothed_price",
    "ts_ema": "smoothed_price", "ts_sma": "smoothed_price",
    "rolling_vwap": "smoothed_price", "donchian_mid": "smoothed_price",
    "ts_markov_*": "nonlinear_stat", "ts_km_*": "nonlinear_stat",
    "ts_hvg_*": "nonlinear_stat", "ts_recurrence_*": "nonlinear_stat",
    "ts_spectral_*": "nonlinear_stat",
    "ts_hankel_*": "spectral_family", "ts_ssa_*": "spectral_family",
    "ts_multifractal_*": "spectral_family",
    "cdl_*": "candlestick_event", "candlestick_pattern": "candlestick_event",
    "dollar_volume": "size_liquidity", "ts_average_volume": "size_liquidity",
    "adv": "size_liquidity",
}

# R18-047: effective-sample contracts for advanced statistical families.
_MIN_EFFECTIVE_SAMPLE: dict[str, dict[str, int]] = {
    "ts_markov_*": {"min_effective_samples": 50, "min_state_occupancy": 2},
    "ts_km_*": {"min_effective_samples": 30},
    "ts_first_passage_*": {"min_event_count": 1},
    "ts_extremal_index": {"min_event_count": 20},
    "ts_mean_excess_slope": {"min_effective_samples": 100},
    "ts_gpd_shape_pwm": {"min_effective_samples": 100},
    "ts_transfer_entropy_*": {"min_effective_samples": 60, "min_event_count": 5},
    "ts_mmd_rbf_shift": {"min_effective_samples": 60},
    "ts_hsic": {"min_effective_samples": 60},
    "ts_conditional_mutual_information": {"min_effective_samples": 60},
    "ts_recurrence_*": {"min_effective_samples": 30},
    "ts_hvg_*": {"min_effective_samples": 10},
    "ts_matrix_profile_*": {"min_effective_samples": 32},
    "ts_spectral_*": {"min_effective_samples": 50},
    "ts_hankel_*": {"min_effective_samples": 50},
    "ts_ssa_*": {"min_effective_samples": 50},
    "ts_generalized_hurst_*": {"min_effective_samples": 100},
    "ts_multifractal_*": {"min_effective_samples": 100},
    "ts_delay_intrinsic_dimension": {"min_effective_samples": 50},
    "ts_persistence_entropy_*": {"min_effective_samples": 30},
    "ts_hartigan_dip": {"min_unique_values": 5},
    "cs_hartigan_dip": {"min_unique_values": 5},
}

# R18-070: default input binding recipes (operator -> slot -> field recipe).
# The actual market field resolution stays in the R17 provider resolver; this
# is the operator-level binding contract only.
_DEFAULT_INPUT_RECIPE: dict[str, dict[str, str]] = {
    "ts_log_return": {"x": "continuous_close"},
    "parkinson_vol": {"high": "continuous_high", "low": "continuous_low"},
    "amihud_illiquidity": {"ret": "return_decimal", "amount": "amount_local"},
    "ts_average_volume": {"volume": "continuous_volume"},
    "dollar_volume": {"close": "continuous_close", "volume": "continuous_volume"},
    "true_range": {"high": "continuous_high", "low": "continuous_low", "close": "continuous_close"},
    "ATR_WILDER": {"high": "continuous_high", "low": "continuous_low", "close": "continuous_close"},
    "NATR": {"high": "continuous_high", "low": "continuous_low", "close": "continuous_close"},
    "tail_beta": {"ret": "return_decimal", "benchmark_ret": "benchmark_return", "window": 60},
    "residual_momentum_capm": {"ret": "return_decimal", "benchmark_ret": "benchmark_return", "window": 60},
    "coskewness_to_market": {"ret": "return_decimal", "benchmark_ret": "benchmark_return", "window": 60},
    "idio_vol": {"ret": "return_decimal", "benchmark_ret": "benchmark_return", "window": 60},
    "idio_skew": {"ret": "return_decimal", "benchmark_ret": "benchmark_return", "window": 60},
}

# R18-055: *_if operators carry a condition slot that only accepts
# Condition/Event/State Bool.
_CONDITION_SLOT_OPS = frozenset(
    {
        "ts_mean_if", "ts_sum_if", "ts_std_if", "ts_min_if", "ts_max_if",
        "ts_quantile_if", "ts_corr_if", "ts_beta_if",
        "ts_regression_resid_if", "ts_cov_if", "ts_rank_if",
    }
)

# R18-067: activity / size-sensitive outputs — usable as alpha but the miner
# should prefer rank/zscore/relative composition (never treat as scale-neutral).
_SIZE_SENSITIVE_OPS = frozenset(
    {
        "dollar_volume", "ts_average_volume", "adv", "rolling_obv", "rolling_pvt",
    }
)

# R18-041: rank_corr is a genuine cross-sectional dependency alpha.
_R18_041_STATUS = {
    "rank_corr": _contract(DirectUseStatus.DIRECT_ALPHA, "cross-sectional rank correlation — dependency alpha"),
    "ts_poly2_coeff": _contract(DirectUseStatus.RESEARCH_TOOL, "in-sample polynomial fit coefficient — diagnostic; use ts_poly2_resid prior forms for causality"),
    "ts_poly2_resid": _contract(DirectUseStatus.RESEARCH_TOOL, "in-sample polynomial residual — diagnostic; causal counterpart preferred"),
}

# R18-040: fundamental period transforms are legitimate DIRECT_RECIPE building
# blocks (they are the standard form, not an obsolete second branch).
_FUNDAMENTAL_TRANSFORM_RECIPES = frozenset(
    {
        "quarter_from_cumulative", "ttm_from_cumulative", "ttm_from_quarterly",
        "yoy_by_period", "period_average",
    }
)

# R18-038: benchmark-required risk factors are high-value; keep them DIRECT_ALPHA
# with an explicit benchmark source (never a permanent deny).
_BENCHMARK_REQUIRED_ALPHA = frozenset(
    {
        "tail_beta", "residual_momentum_capm", "coskewness_to_market",
        "idio_vol", "idio_skew",
    }
)

# R18-042/043/044: market/data-context verdicts.
_MARKET_CONTEXT_STATUS = {
    # A has a certified minute source -> A direct; US has none -> US blocked.
    "intraday_vwap_deviation": _contract(
        DirectUseStatus.DIRECT_ALPHA,
        "A-share direct with certified minute source; US blocked (no minute source) — market-context direct use",
    ),
    # R17 has no reliable depreciation/amortization provider in either market.
    "fin_total_operating_accruals": _contract(
        DirectUseStatus.DELETE_NO_DATA,
        "no reliable depreciation/amortization source in A or US after R17 — keep as backlog/docs, not public mining",
        delete_reason="required concepts (depreciation, amortization) have no provider in either market",
    ),
}

# R18-044: fin_revision_* / fin_expectation_* / fin_surprise* — each must prove
# a current provider; both markets without one -> DELETE_NO_DATA.  The
# A-share revision/surprise family IS registered with a provider path, so it
# stays direct; the no-data verdict applies only to families with no provider.
_FIN_REVISION_NO_DATA = frozenset(
    {
        # e.g. "fin_revision_count" has no provider yet -> listed here.
    }
)

# R18-037: retired names that must NOT be visible in surface/manifest.
RETIRED_GHOST_CANONICALS = frozenset(
    {
        "state_since_reduce",  # split into state_since_sum/mean/count/last
    }
)

# R18-046: expanding_rank has sample-start dependence — bounded rolling
# alternative recommended; keep only as high-cost full-history replay with a
# fixed anchor.
_EXPANDING_RANK_VERDICT = _contract(
    DirectUseStatus.DIRECT_ALPHA_HIGH_COST,
    "full-history expanding rank — sample-start dependent; requires fixed anchor in factor identity",
)


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def _fnmatch_any(canonical: str, patterns: Iterable[str]) -> bool:
    return any(re.fullmatch(p.replace("*", ".*"), canonical) for p in patterns)


def _resolve_explicit(canonical: str, catalog: dict[str, Any]) -> DirectUseContract | None:
    """Explicit family overrides first — these win over every surface rule."""
    if canonical in RETIRED_GHOST_CANONICALS:
        return _contract(
            DirectUseStatus.DELETE_OBSOLETE,
            "retired and split into successor canonical(s); must not be visible in surface",
            replacement="state_since_sum/state_since_mean/state_since_count/state_since_last",
        )
    if canonical in _BOOL_CONDITION_OPS:
        return _contract(DirectUseStatus.DIRECT_CONDITION, "Boolean/comparison/missingness — condition slot only")
    if canonical in _CS_GLOBAL_STATE_OPS:
        return _contract(DirectUseStatus.DIRECT_GLOBAL_STATE, "cross-sectional aggregate — same value across stocks; global context only")
    if canonical in _GROUP_STATE_OPS:
        return _contract(DirectUseStatus.DIRECT_GROUP_STATE, "group aggregate — same value within group; group context only")
    if canonical in _CATEGORY_STATE_OPS:
        return _contract(DirectUseStatus.DIRECT_STATE, "categorical output — category condition/interaction, not continuous strength")
    if canonical in _CDL_EVENT_OPS:
        return _contract(DirectUseStatus.DIRECT_EVENT, "candlestick binary pattern — event mask")
    if canonical in _BREAKOUT_EVENT_OPS:
        return _contract(DirectUseStatus.DIRECT_EVENT, "breakout/new-high binary indicator — event mask")
    if canonical in _LIMIT_EVENT_OPS:
        return _contract(DirectUseStatus.DIRECT_EVENT, "A-share limit Boolean — event mask")
    if canonical in _FIN_INDEX_ROLE:
        return _FIN_INDEX_ROLE[canonical]
    if canonical in _PRICE_LEVEL_INTERMEDIATE_OPS:
        return _contract(DirectUseStatus.DIRECT_INTERMEDIATE, "price-level building block — not scale-invariant, composition only")
    if canonical in _SOURCE_TRANSFORM_OPS:
        return _contract(DirectUseStatus.DIRECT_SOURCE_TRANSFORM, "missing-value / source transform — data-processing slot only")
    if canonical in _ARBITRARY_MATH_STATUS:
        return _ARBITRARY_MATH_STATUS[canonical]
    if canonical in _FLEX_DUP_REPLACEMENT:
        repl = _FLEX_DUP_REPLACEMENT[canonical]
        return _contract(
            DirectUseStatus.DELETE_DUPLICATE,
            "exact semantic duplicate of " + repl + " (only input flexibility differs)",
            replacement=repl,
        )
    if canonical in _R18_041_STATUS:
        return _R18_041_STATUS[canonical]
    if canonical in _FUNDAMENTAL_TRANSFORM_RECIPES:
        return _contract(DirectUseStatus.DIRECT_RECIPE, "fundamental period transform — standard recipe building block")
    if canonical in _BENCHMARK_REQUIRED_ALPHA:
        return _contract(DirectUseStatus.DIRECT_ALPHA, "benchmark-required risk factor — direct usable with explicit benchmark source")
    if canonical in _MARKET_CONTEXT_STATUS:
        return _MARKET_CONTEXT_STATUS[canonical]
    if canonical in _FIN_REVISION_NO_DATA:
        return _contract(
            DirectUseStatus.DELETE_NO_DATA,
            "no provider for required concepts in either market",
            delete_reason="no provider",
        )
    if canonical == "expanding_rank":
        return _EXPANDING_RANK_VERDICT
    # family patterns
    if canonical.startswith("cdl_"):
        return _contract(DirectUseStatus.DIRECT_EVENT, "candlestick binary pattern — event mask")
    if canonical.startswith("fin_revision_") or canonical.startswith("fin_expectation_") or "surprise" in canonical:
        if canonical.startswith("fin_surprise") or canonical.startswith("fin_revision") or canonical.startswith("fin_expectation"):
            return _contract(DirectUseStatus.DIRECT_ALPHA, "fundamental revision/expectation/surprise statistic — alpha (PIT lane)")
    if canonical in _CONDITION_SLOT_OPS:
        return _contract(DirectUseStatus.DIRECT_ALPHA, "conditional statistic — numeric alpha with a Condition/Event/State Bool condition slot")
    return None


def _default_from_role(canonical: str, catalog: dict[str, Any]) -> DirectUseContract:
    """Fall back to the verified MiningRole (the pre-R18 authority)."""
    role, _src = assign_mining_role_ex(canonical, catalog)
    if role in (MiningRole.ALPHA, MiningRole.INTRADAY_EOD):
        if catalog.get("lifecycle_status") == "experimental" or not catalog.get("production_certified"):
            return _contract(DirectUseStatus.DIRECT_ALPHA, "stock-level alpha (verified role)", terminal=False)
        return _contract(DirectUseStatus.DIRECT_ALPHA, "stock-level alpha (verified role)")
    if role is MiningRole.ALPHA_HIGH_COST:
        return _contract(DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "full-history / high-cost alpha (verified role)")
    if role is MiningRole.STATE:
        return _contract(DirectUseStatus.DIRECT_STATE, "discrete state machine (verified role)")
    if role is MiningRole.CONDITION:
        return _contract(DirectUseStatus.DIRECT_CONDITION, "condition (verified role)")
    if role is MiningRole.EVENT:
        return _contract(DirectUseStatus.DIRECT_EVENT, "event mask (verified role)")
    if role is MiningRole.GROUP_STATE:
        return _contract(DirectUseStatus.DIRECT_GROUP_STATE, "group state (verified role)")
    if role is MiningRole.GLOBAL_STATE:
        return _contract(DirectUseStatus.DIRECT_GLOBAL_STATE, "global state (verified role)")
    if role is MiningRole.FUNDAMENTAL_PIT:
        return _contract(DirectUseStatus.DIRECT_ALPHA, "fundamental PIT alpha (verified role)")
    if role is MiningRole.RECIPE_INTERNAL:
        return _contract(DirectUseStatus.DIRECT_RECIPE, "recipe-internal building block")
    if role is MiningRole.SOURCE_TRANSFORM:
        return _contract(DirectUseStatus.DIRECT_SOURCE_TRANSFORM, "source transform (verified role)")
    if role is MiningRole.DIAGNOSTIC:
        return _contract(DirectUseStatus.RESEARCH_TOOL, "diagnostic / in-sample metric — research tool only")
    if role is MiningRole.RESEARCH:
        return _contract(DirectUseStatus.RESEARCH_TOOL, "research-tier operator — research tool only")
    if role is MiningRole.INTERNAL:
        return _contract(DirectUseStatus.MOVE_INTERNAL, "internal supporting layer — not public mining")
    if role is MiningRole.LEGACY:
        return _contract(DirectUseStatus.DELETE_OBSOLETE, "legacy surface — obsolete or aliased")
    if role is MiningRole.DENIED:
        return _contract(
            DirectUseStatus.DELETE_NONCAUSAL,
            "permanently-forbidden / non-causal primitive",
        )
    # UNRESOLVED must never survive into the retained set — surface it.
    return _contract(
        DirectUseStatus.DELETE_USELESS,
        "no verified role — unresolved canonical must be deleted or explicitly classified",
    )


def resolve_direct_use(canonical: str, catalog: dict[str, Any] | None = None) -> DirectUseContract:
    """R18-124: one final verdict per canonical, never SKIP/UNRESOLVED."""
    from cleaned_operators.registry import OperatorRegistry

    catalog = dict(catalog) if catalog is not None else (OperatorRegistry._catalog.get(canonical) or {})
    explicit = _resolve_explicit(canonical, catalog)
    return explicit if explicit is not None else _default_from_role(canonical, catalog)


def resolve_direct_use_status(canonical: str, catalog: dict[str, Any] | None = None) -> DirectUseStatus:
    """R18-096: the status alone (for matrix rows / manifests)."""
    return resolve_direct_use(canonical, catalog).status


def terminal_allowed_for(canonical: str, catalog: dict[str, Any] | None = None) -> bool:
    """R18-003: positive authority — terminal iff the resolved status allows it."""
    return resolve_direct_use(canonical, catalog).terminal_allowed


# ---------------------------------------------------------------------------
# R18-029: input slot contract
# ---------------------------------------------------------------------------

# context / group / event slot names per operator family (explicit).
_CONTEXT_SLOTS: dict[str, tuple[str, ...]] = {
    "cs_rank": ("group",),
    "cs_zscore": ("group",),
    "cs_percentile": ("group",),
    "group_rank": ("group",),
    "group_zscore": ("group",),
    "group_neutralize": ("group",),
    "group_percentile": ("group",),
    "cs_bucket": ("group",),
    "size_neutralize": ("size",),
    "industry_neutralize": ("group",),
}
# condition slots map to the panel param that carries the condition.
_CONDITION_SLOT_NAMES: dict[str, str] = {
    "ts_mean_if": "condition",
    "ts_sum_if": "condition",
    "ts_std_if": "condition",
    "ts_min_if": "condition",
    "ts_max_if": "condition",
    "ts_quantile_if": "condition",
    "ts_corr_if": "condition",
    "ts_beta_if": "condition",
    "ts_regression_resid_if": "condition",
    "ts_cov_if": "condition",
    "ts_rank_if": "condition",
}
_EVENT_SLOTS: dict[str, tuple[str, ...]] = {
    "event_interval_mark_coupling": ("event",),
    "event_mark_autocorr": ("event",),
    "event_count": ("event",),
    "cross_event": ("a", "b"),
}

PANEL_PARAM_SET = frozenset(
    {
        "x", "y", "z", "a", "b", "c", "v", "u", "w", "p", "q",
        "price", "event", "x_df", "y_df", "returns", "return", "values",
        "group", "mask", "market", "weights", "weight",
        "open", "high", "low", "close", "volume", "amount", "vwap",
        "open_p", "high_p", "low_p", "close_p", "volume_p", "amount_p",
        "source", "target", "a_", "b_", "condition", "benchmark_ret",
        "member", "ret", "factor",
    }
)


@dataclass(frozen=True)
class InputSlotSpec:
    """R18-029: one positional slot — parameter name + semantic contract."""

    parameter: str
    allowed_semantic_kinds: tuple[str, ...] = ("field",)
    allowed_units: tuple[str, ...] = ()
    allowed_roles: tuple[str, ...] = ("data",)
    cardinality: str = "panel"  # panel | broadcast | scalar | condition | event | group
    axis_semantics: str = "per_instrument"  # per_instrument | cross_section | group | session | global
    context: str = "data"  # data | context | group | event | condition | scalar


def input_slot_specs(canonical: str, catalog: dict[str, Any]) -> tuple[InputSlotSpec, ...]:
    """Build the R18-029 input-slot contract for a canonical.

    ``data_inputs`` are the panel data fields; ``scalar_parameters`` are the
    numeric knobs; ``context_inputs`` / ``group_inputs`` / ``event_inputs`` are
    explicit context slots.  An ``inputs`` key, where kept, equals exactly the
    data-input list (R18-002).
    """
    params = list(catalog.get("param_names") or ())
    slots: list[InputSlotSpec] = []
    condition_name = _CONDITION_SLOT_NAMES.get(canonical)
    event_names = _EVENT_SLOTS.get(canonical, ())
    context_names = _CONTEXT_SLOTS.get(canonical, ())
    for name in params:
        if condition_name and name == condition_name:
            slots.append(
                InputSlotSpec(
                    parameter=name,
                    allowed_semantic_kinds=("condition", "event", "state"),
                    allowed_units=(),
                    allowed_roles=("gate",),
                    cardinality="condition",
                    axis_semantics="per_instrument",
                    context="condition",
                )
            )
            continue
        if name in event_names:
            slots.append(
                InputSlotSpec(
                    parameter=name,
                    allowed_semantic_kinds=("event",),
                    allowed_units=(),
                    allowed_roles=("event",),
                    cardinality="event",
                    axis_semantics="per_instrument",
                    context="event",
                )
            )
            continue
        if name in context_names:
            slots.append(
                InputSlotSpec(
                    parameter=name,
                    allowed_semantic_kinds=("context", "group", "broadcast"),
                    allowed_units=(),
                    allowed_roles=("context",),
                    cardinality="broadcast",
                    axis_semantics="cross_section" if name == "group" else "global",
                    context="group" if name == "group" else "context",
                )
            )
            continue
        if name in PANEL_PARAM_SET:
            slots.append(
                InputSlotSpec(
                    parameter=name,
                    allowed_semantic_kinds=("field",),
                    allowed_units=(),
                    allowed_roles=("data",),
                    cardinality="panel",
                    axis_semantics="per_instrument",
                    context="data",
                )
            )
            continue
        # everything else is a scalar knob
        slots.append(
            InputSlotSpec(
                parameter=name,
                allowed_semantic_kinds=("scalar",),
                allowed_units=(),
                allowed_roles=("parameter",),
                cardinality="scalar",
                axis_semantics="none",
                context="scalar",
            )
        )
    return tuple(slots)


def split_input_slots(
    canonical: str, catalog: dict[str, Any]
) -> dict[str, list[str]]:
    """R18-002: data_inputs / scalar_parameters / context_inputs / group_inputs /
    event_inputs split.  ``inputs`` (when needed) == data_inputs."""
    data: list[str] = []
    scalar: list[str] = []
    context: list[str] = []
    group: list[str] = []
    event: list[str] = []
    for slot in input_slot_specs(canonical, catalog):
        if slot.context == "data":
            data.append(slot.parameter)
        elif slot.context == "scalar":
            scalar.append(slot.parameter)
        elif slot.context == "group":
            group.append(slot.parameter)
        elif slot.context == "event":
            event.append(slot.parameter)
        elif slot.context == "condition":
            context.append(slot.parameter)
        else:
            context.append(slot.parameter)
    return {
        "data_inputs": data,
        "scalar_parameters": scalar,
        "context_inputs": context,
        "group_inputs": group,
        "event_inputs": event,
    }


# ---------------------------------------------------------------------------
# metadata helpers
# ---------------------------------------------------------------------------

def economic_effect_family(canonical: str) -> str:
    """R18-068: coarse economic family for cold-start diversity."""
    for pattern, family in _ECONOMIC_EFFECT_FAMILY.items():
        if pattern.endswith("*"):
            if _fnmatch_any(canonical, (pattern,)):
                return family
        elif canonical == pattern:
            return family
    return "unknown"


def semantic_redundancy_group(canonical: str) -> str:
    """R18-069: redundancy group for family-budget quotas."""
    for pattern, group in _SEMANTIC_REDUNDANCY_GROUP.items():
        if pattern.endswith("*"):
            if _fnmatch_any(canonical, (pattern,)):
                return group
        elif canonical == pattern:
            return group
    return ""


def monotonic_transform_class(canonical: str) -> str:
    """R18-021: monotonic-equivalent class (dedup x/exp(x)/sigmoid(x)/rank(x))."""
    return MONOTONIC_TRANSFORM_CLASS.get(canonical, "")


def min_effective_sample_contract(canonical: str) -> dict[str, int]:
    """R18-047: effective-sample floors for advanced statistical families."""
    for pattern, contract in _MIN_EFFECTIVE_SAMPLE.items():
        if _fnmatch_any(canonical, (pattern,)):
            return dict(contract)
    return {}


def default_input_recipe(canonical: str) -> dict[str, str]:
    """R18-070: operator-level default input binding recipe."""
    return dict(_DEFAULT_INPUT_RECIPE.get(canonical, {}))


def scale_sensitive_flags(canonical: str) -> dict[str, bool]:
    """R18-067: activity/size-sensitive output exposure."""
    if canonical in _SIZE_SENSITIVE_OPS:
        return {
            "scale_sensitive": True,
            "size_sensitive": True,
            "liquidity_exposure": True,
        }
    return {"scale_sensitive": False, "size_sensitive": False, "liquidity_exposure": False}


def condition_slot(canonical: str) -> str | None:
    """R18-055: the condition slot name ('' if none)."""
    return _CONDITION_SLOT_NAMES.get(canonical)


# ---------------------------------------------------------------------------
# R18-001: the ONE authority for automated mining
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectUseOperator:
    """R18-096 matrix row — every retained public canonical gets one."""

    canonical: str
    aliases: tuple[str, ...]
    module: str
    class_name: str
    authoring_tier: str

    direct_use_status: DirectUseStatus
    mining_role: str
    terminal_allowed: bool
    allowed_ast_positions: tuple[str, ...]
    mining_lane: str

    retention_reason: str
    delete_reason: str
    replacement: str

    economic_effect_family: str
    semantic_redundancy_group: str
    monotonic_transform_class: str

    input_slots: tuple[InputSlotSpec, ...]
    data_inputs: tuple[str, ...]
    scalar_parameters: tuple[str, ...]
    context_inputs: tuple[str, ...]
    group_inputs: tuple[str, ...]
    event_inputs: tuple[str, ...]
    condition_slot: str

    output_semantic_kind: str
    output_unit: str | None
    output_cardinality: str
    output_value_domain: str

    default_input_recipe: dict[str, str]
    smoke_recipe_ids: tuple[str, ...]

    supported_markets: tuple[str, ...]
    source_recipes: tuple[str, ...]

    default_params: tuple[str, ...]
    searchable_params: tuple[str, ...]
    search_grade_by_param: dict[str, str]
    parameter_injectivity_passed: bool

    stateful: bool
    execution_model: str
    checkpoint_supported: bool
    full_history_replay_allowed: bool

    runtime_cost: int
    memory_cost: int
    preferred_backend: str
    reference_backend: str

    production_certified: bool
    directly_usable: bool

    min_effective_samples: dict[str, int]
    scale_sensitive: dict[str, bool]
    retention_usage: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if k == "input_slots":
                d[k] = [
                    {
                        "parameter": s.parameter,
                        "allowed_semantic_kinds": list(s.allowed_semantic_kinds),
                        "allowed_units": list(s.allowed_units),
                        "allowed_roles": list(s.allowed_roles),
                        "cardinality": s.cardinality,
                        "axis_semantics": s.axis_semantics,
                        "context": s.context,
                    }
                    for s in v
                ]
            elif isinstance(v, DirectUseStatus):
                d[k] = v.value
            elif isinstance(v, tuple):
                d[k] = list(v)
            elif isinstance(v, dict):
                d[k] = dict(v)
            else:
                d[k] = v
        return d


@dataclass(frozen=True)
class DirectUseContext:
    """Environment the miner declares — R18-031 contextual direct usability."""

    available_sources: tuple[str, ...] = ()
    target_frequency: str | None = None
    market: str | None = None
    max_cost: int | None = None


def _alias_map() -> dict[str, str]:
    try:
        from cleaned_operators.registry import OperatorRegistry

        return dict(OperatorRegistry._aliases)
    except Exception:
        return {}


def _op_class_name(canonical: str) -> str:
    try:
        from cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(canonical, "pandas_numpy") or OperatorRegistry.get(canonical)
        return type(op).__name__
    except Exception:
        return ""


def _op_module(canonical: str) -> str:
    try:
        from cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(canonical, "pandas_numpy") or OperatorRegistry.get(canonical)
        return getattr(type(op), "__module__", "") or ""
    except Exception:
        return ""


def _search_grades(canonical: str) -> dict[str, str]:
    """R18-028: full/coarse/fixed/excluded per searchable param."""
    grades: dict[str, str] = {}
    try:
        from cleaned_operators.base import param_search_grade
        from cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(canonical, "pandas_numpy") or OperatorRegistry.get(canonical)
        meta = getattr(op, "metadata", None)
        if meta is None:
            return grades
        for name in getattr(meta, "param_names", None) or ():
            grade = param_search_grade(meta, name)
            if grade:
                grades[name] = str(grade)
    except Exception:
        pass
    return grades


def _backend_names(canonical: str) -> tuple[str, str]:
    """(preferred, reference) backend.  Pandas-numpy is always the reference."""
    reference = "pandas_numpy"
    preferred = "pandas_numpy"
    try:
        from cleaned_operators.registry import OperatorRegistry

        for be in ("polars_udf", "polars", "sql", "pandas_numpy"):
            if OperatorRegistry.get(canonical, be) is not None:
                preferred = be
                break
    except Exception:
        pass
    return preferred, reference


def build_direct_use_operator(canonical: str, catalog: dict[str, Any]) -> DirectUseOperator:
    """Build one R18-096 matrix row (registry-level, no operator execution)."""
    contract = resolve_direct_use(canonical, catalog)
    role, role_src = assign_mining_role_ex(canonical, catalog)
    split = split_input_slots(canonical, catalog)
    from cleaned_operators.operator_surface import classify_canonical

    tier = classify_canonical(canonical)
    try:
        from cleaned_operators.base import searchable_param_names

        op = None
        try:
            from cleaned_operators.registry import OperatorRegistry

            op = OperatorRegistry.get(canonical, "pandas_numpy") or OperatorRegistry.get(canonical)
        except Exception:
            op = None
        meta = getattr(op, "metadata", None)
        if meta is not None:
            grades = searchable_param_names(meta)
            searchable = tuple(
                sorted(set(grades.get("full", ())) | set(grades.get("coarse", ())))
            )
        else:
            searchable = tuple(split["scalar_parameters"])
    except Exception:
        searchable = tuple(split["scalar_parameters"])
    try:
        from runtime.execution_contract import execution_contract

        ec = execution_contract(canonical)
        state_model = str(getattr(ec, "state_model", "stateless"))
        chunking = str(getattr(ec, "chunking", "independent"))
        stateful = state_model not in ("stateless", "unknown")
        checkpoint = chunking == "checkpoint"
        full_replay = chunking in ("required_full_history", "unknown")
        exec_model = "checkpoint" if checkpoint else ("full_history" if full_replay else "independent_with_warmup")
    except Exception:
        stateful, checkpoint, full_replay, exec_model = False, False, False, "independent_with_warmup"

    market = market_support(canonical)
    source_reqs = source_status(canonical, catalog, None).required
    preferred, reference = _backend_names(canonical)
    scale_sens = scale_sensitive_flags(canonical)
    eff_sample = min_effective_sample_contract(canonical)
    directly_usable = (
        contract.status in _DIRECT_TERMINAL_STATUSES
        and bool(catalog.get("production_certified"))
        and cost_contract_declared(canonical, catalog)
        and not source_status(canonical, catalog, None).missing
    )
    # for market-context ops, directly_usable is only meaningful with a context
    if contract.status is DirectUseStatus.DIRECT_ALPHA and canonical in _MARKET_CONTEXT_STATUS:
        directly_usable = directly_usable and "ashare" in market
    usage = {
        "terminal": contract.terminal_allowed,
        "intermediate": "intermediate" in contract.ast_positions,
        "condition": "condition" in contract.ast_positions,
        "interaction": "interaction" in contract.ast_positions,
        "state_gate": contract.status in (DirectUseStatus.DIRECT_STATE,),
        "event_gate": contract.status in (DirectUseStatus.DIRECT_EVENT,),
        "group_context": contract.status is DirectUseStatus.DIRECT_GROUP_STATE,
        "global_context": contract.status is DirectUseStatus.DIRECT_GLOBAL_STATE,
    }
    return DirectUseOperator(
        canonical=canonical,
        aliases=_aliases_of(canonical),
        module=_op_module(canonical),
        class_name=_op_class_name(canonical),
        authoring_tier=tier,
        direct_use_status=contract.status,
        mining_role=role.value,
        terminal_allowed=contract.terminal_allowed,
        allowed_ast_positions=contract.ast_positions,
        mining_lane=_DIRECT_STATUS_LANES.get(contract.status, "supporting"),
        retention_reason=contract.retention_reason,
        delete_reason=contract.delete_reason,
        replacement=contract.replacement,
        economic_effect_family=economic_effect_family(canonical),
        semantic_redundancy_group=semantic_redundancy_group(canonical),
        monotonic_transform_class=monotonic_transform_class(canonical),
        input_slots=input_slot_specs(canonical, catalog),
        data_inputs=tuple(split["data_inputs"]),
        scalar_parameters=tuple(split["scalar_parameters"]),
        context_inputs=tuple(split["context_inputs"]),
        group_inputs=tuple(split["group_inputs"]),
        event_inputs=tuple(split["event_inputs"]),
        condition_slot=condition_slot(canonical) or "",
        output_semantic_kind=str(catalog.get("output_semantic_kind") or catalog.get("return_type") or "series"),
        output_unit=catalog.get("output_unit"),
        output_cardinality="panel" if role not in (MiningRole.GROUP_STATE, MiningRole.GLOBAL_STATE) else "broadcast",
        output_value_domain="",
        default_input_recipe=default_input_recipe(canonical),
        smoke_recipe_ids=(),
        supported_markets=market,
        source_recipes=source_reqs,
        default_params=tuple(catalog.get("param_names") or ()),
        searchable_params=searchable,
        search_grade_by_param=_search_grades(canonical),
        parameter_injectivity_passed=False,
        stateful=stateful,
        execution_model=exec_model,
        checkpoint_supported=checkpoint,
        full_history_replay_allowed=full_replay or not stateful,
        runtime_cost=cost_tier(canonical, catalog),
        memory_cost=cost_tier(canonical, catalog),
        preferred_backend=preferred,
        reference_backend=reference,
        production_certified=bool(catalog.get("production_certified")),
        directly_usable=directly_usable,
        min_effective_samples=eff_sample,
        scale_sensitive=scale_sens,
        retention_usage=usage,
    )


def _aliases_of(canonical: str) -> tuple[str, ...]:
    try:
        from cleaned_operators.registry import OperatorRegistry

        return tuple(sorted(a for a, t in OperatorRegistry._aliases.items() if t == canonical))
    except Exception:
        return ()


def get_direct_use_mining_operators(
    context: DirectUseContext | None = None,
    *,
    admission: str = "eligible",
) -> list[DirectUseOperator]:
    """R18-001: the ONE authority for automated mining.

    Only DIRECT_* statuses are returned.  ``admission`` mirrors the mining
    layer: ``eligible`` (production_certified + role-admissible + sources +
    cost contract) or ``all`` (every retained DIRECT_* row, including not-yet-
    certified — used by the DirectUse Matrix / manifests).
    """
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    ctx = context or DirectUseContext()
    mode = str(admission or "eligible").strip().lower()
    if mode not in {"eligible", "all"}:
        raise ValueError("admission must be eligible or all")
    out: list[DirectUseOperator] = []
    for canonical in sorted(OperatorRegistry._catalog):
        catalog = OperatorRegistry._catalog[canonical]
        row = build_direct_use_operator(canonical, catalog)
        if row.direct_use_status.value.startswith("delete_") or row.direct_use_status in (
            DirectUseStatus.RESEARCH_TOOL,
            DirectUseStatus.MOVE_INTERNAL,
        ):
            continue
        if ctx.market is not None and ctx.market not in row.supported_markets:
            continue
        if mode == "eligible":
            if not row.production_certified:
                continue
            if not row.terminal_allowed and not row.allowed_ast_positions:
                continue
            if ctx.max_cost is not None and row.runtime_cost > ctx.max_cost:
                continue
            if ctx.available_sources:
                missing = [
                    s for s in row.source_recipes if s not in ctx.available_sources
                ]
                if missing:
                    continue
            if ctx.target_frequency is not None:
                if ctx.target_frequency == "minute" and row.output_semantic_kind == "daily":
                    continue
        out.append(row)
    out.sort(key=lambda r: (r.direct_use_status.value, r.canonical))
    return out


def direct_use_matrix_rows() -> list[DirectUseOperator]:
    """R18-096: every registered canonical -> a DirectUseOperator row."""
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    rows: list[DirectUseOperator] = []
    for canonical in sorted(OperatorRegistry._catalog):
        rows.append(build_direct_use_operator(canonical, OperatorRegistry._catalog[canonical]))
    return rows


def direct_use_status_counts(rows: Sequence[DirectUseOperator] | None = None) -> dict[str, int]:
    """R18-097: before/after status histogram."""
    from collections import Counter

    if rows is None:
        rows = direct_use_matrix_rows()
    counts: dict[str, int] = Counter()
    for row in rows:
        counts[row.direct_use_status.value] += 1
    return dict(sorted(counts.items()))


def retained_direct_rows(rows: Sequence[DirectUseOperator] | None = None) -> list[DirectUseOperator]:
    """The DIRECT_* subset (public mining surface)."""
    if rows is None:
        rows = direct_use_matrix_rows()
    return [r for r in rows if r.direct_use_status.value.startswith("direct_")]
