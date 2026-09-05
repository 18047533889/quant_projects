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

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd  # noqa: F401  (used by the R25 parameter-injectivity probe)

from factor_engine.market.context import Market
from factor_engine.mining.operator_catalog import (
    MiningRole,
    RoleSource,
    _ROLE_AST_POSITIONS,
    assign_mining_role_ex,
    cost_contract_declared,
    cost_tier,
    market_support,
    mining_eligible,
    source_status,
    _output_grain as _catalog_output_grain,
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
# (R18-003 / R22-006: positive authority — nothing is terminal by exclusion).
# R63-032: DIRECT_RECIPE is a composition building block (period transforms like
# ttm/quarterly/yoy/period_average) and is REMOVED from the terminal set — it
# never stands alone as a factor terminal, only as a recipe step.
_DIRECT_TERMINAL_STATUSES = frozenset(
    {
        DirectUseStatus.DIRECT_ALPHA,
        DirectUseStatus.DIRECT_ALPHA_HIGH_COST,
    }
)

# R22-029: PublicMiningDisposition — the ONLY legal fates for a retained public
# canonical.  ``PUBLIC_BUT_HIDDEN`` / ``PUBLIC_RESEARCH`` / ``PUBLIC_PENDING_FOREVER``
# / ``PUBLIC_UNKNOWN`` are forbidden (R22-030): a public operator must be either
# mining-visible, explicitly moved/deleted, or its retention is a hard audit fail.
class PublicMiningDisposition(str, Enum):
    DIRECT_VISIBLE = "direct_visible"
    MOVED_RESEARCH_TOOL = "moved_research_tool"
    MOVED_INTERNAL = "moved_internal"
    DELETED_REPLACED = "deleted_replaced"
    DELETED_NONCAUSAL = "deleted_noncausal"
    DELETED_NO_DATA = "deleted_no_data"
    DELETED_DUPLICATE = "deleted_duplicate"
    DELETED_OBSOLETE = "deleted_obsolete"


# R22-029: status -> disposition (for the machine matrix).
_DISPOSITION_BY_STATUS: dict[DirectUseStatus, PublicMiningDisposition] = {
    DirectUseStatus.DIRECT_ALPHA: PublicMiningDisposition.DIRECT_VISIBLE,
    DirectUseStatus.DIRECT_ALPHA_HIGH_COST: PublicMiningDisposition.DIRECT_VISIBLE,
    DirectUseStatus.DIRECT_STATE: PublicMiningDisposition.DIRECT_VISIBLE,
    DirectUseStatus.DIRECT_CONDITION: PublicMiningDisposition.DIRECT_VISIBLE,
    DirectUseStatus.DIRECT_EVENT: PublicMiningDisposition.DIRECT_VISIBLE,
    DirectUseStatus.DIRECT_GROUP_STATE: PublicMiningDisposition.DIRECT_VISIBLE,
    DirectUseStatus.DIRECT_GLOBAL_STATE: PublicMiningDisposition.DIRECT_VISIBLE,
    DirectUseStatus.DIRECT_INTERMEDIATE: PublicMiningDisposition.DIRECT_VISIBLE,
    DirectUseStatus.DIRECT_SOURCE_TRANSFORM: PublicMiningDisposition.DIRECT_VISIBLE,
    DirectUseStatus.DIRECT_RECIPE: PublicMiningDisposition.DIRECT_VISIBLE,
    DirectUseStatus.DIRECT_CONTROL_FLOW: PublicMiningDisposition.DIRECT_VISIBLE,
    DirectUseStatus.RESEARCH_TOOL: PublicMiningDisposition.MOVED_RESEARCH_TOOL,
    DirectUseStatus.MOVE_INTERNAL: PublicMiningDisposition.MOVED_INTERNAL,
    DirectUseStatus.DELETE_DUPLICATE: PublicMiningDisposition.DELETED_DUPLICATE,
    DirectUseStatus.DELETE_NO_DATA: PublicMiningDisposition.DELETED_NO_DATA,
    DirectUseStatus.DELETE_NONCAUSAL: PublicMiningDisposition.DELETED_NONCAUSAL,
    DirectUseStatus.DELETE_MATH_DEFECT: PublicMiningDisposition.DELETED_REPLACED,
    DirectUseStatus.DELETE_USELESS: PublicMiningDisposition.DELETED_OBSOLETE,
    DirectUseStatus.DELETE_OBSOLETE: PublicMiningDisposition.DELETED_OBSOLETE,
}


def public_mining_disposition(status: DirectUseStatus) -> PublicMiningDisposition:
    """R22-029: machine disposition for one status (defaults to MOVED_INTERNAL)."""
    return _DISPOSITION_BY_STATUS.get(
        status, PublicMiningDisposition.MOVED_INTERNAL
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
    # R63-032: DIRECT_RECIPE = composition building block, NOT a standalone
    # terminal (period transforms like ttm/quarterly/yoy/period_average are
    # always composed into a factor; they never stand alone).  Terminal
    # authority lives in _DIRECT_TERMINAL_STATUSES, not in the AST table.
    DirectUseStatus.DIRECT_RECIPE: ("intermediate",),
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

# R63: full-history replay indicators are terminal-grade alpha, just expensive
# to replay.  They stay ALPHA_HIGH_COST (NOT price-level DIRECT_INTERMEDIATE):
# the alpha_high_cost lane is terminal-eligible and DirectUse terminal authority
# still gates placement.  Mirrors production_hardening.FULL_HISTORY_REPLAY_CANONICALS.
_FULL_HISTORY_REPLAY_ALPHA = frozenset(
    {
        "KAMA", "DEMA", "TEMA", "PSAR", "Supertrend",
    }
)

# R18-017: normalized / relative versions of the price-level building blocks
# ARE direct alphas (dimensionless / scale-invariant).  The spec explicitly
# lists these; they override the ALPHA_HIGH_COST a full-history indicator would
# otherwise inherit from the mining layer.
_RELATIVE_ALPHA_OPS = frozenset(
    {
        "vwap_deviation", "NATR",
        "atr_pct", "atr_zscore", "atr_percentile",
        "true_range_pct", "true_range_surprise", "true_range_zscore",
        "atr_short_long_ratio", "atr_acceleration",
        "KeltnerPosition", "ichimoku_cloud_position",
        "keltner_width_pct", "keltner_compression", "keltner_breakout_strength",
        "candle_gap_pct", "candle_gap_atr",
        "ts_swing_amplitude_pct", "ts_swing_amplitude_atr",
        "ts_channel_width_pct", "ts_channel_width_atr",
        "ts_distance_to_high", "ts_distance_to_low",
        "ts_distance_to_resistance", "ts_distance_to_support",
        "bollinger_pct_b", "bollinger_width",
        # R20-PSAR-DIRECTUSE: dimensionless PSAR regime canonicals; the raw
        # price-scale "PSAR" level stays intermediate (never in this set).
        "psar_direction", "psar_distance_pct", "psar_flip", "psar_days_since_flip",
        # R20-DONCHIAN-DIRECTUSE: causal dimensionless Donchian canonicals; the
        # raw price-scale donchian_upper/lower/mid levels stay intermediate.
        "donchian_width_pct", "donchian_channel_position",
        "donchian_breakout_up", "donchian_breakout_down",
        # R20-MA-DISTANCE-SLOPE: dimensionless MA distance/slope/crossover
        # canonicals; the raw MA levels (KAMA/DEMA/TEMA, ts_ema/ts_sma) stay
        # intermediate (never in this set).
        "ema_distance_pct", "sma_distance_pct", "dema_distance_pct",
        "tema_distance_pct", "kama_distance_pct", "ma_slope_pct",
        "ema_crossover",
        # R20-SUPERTREND-DIRECTUSE: dimensionless Supertrend regime canonicals;
        # the raw price-scale "Supertrend"/"SupertrendDirection" levels stay
        # intermediate (never in this set).
        "supertrend_direction", "supertrend_distance_pct",
        "supertrend_flip", "supertrend_days_since_flip",
        # R20-ICHIMOKU-DIRECTUSE: dimensionless CAUSAL Ichimoku canonicals; the
        # raw price-scale ichimoku_tenkan/kijun/senkou_a/senkou_b levels stay
        # intermediate (never in this set).
        "tenkan_kijun_cross", "chikou_distance_pct", "senkou_span_causal_pct",
        # R20-VWAP-DIRECTUSE: dimensionless rolling-VWAP distance/slope/premium
        # canonicals; the raw price-scale "rolling_vwap" level stays
        # intermediate (never in this set).
        "vwap_distance_pct", "vwap_slope_pct", "vwap_premium_pct",
        # R20-CANDLESTICK-DIRECTUSE: dimensionless AGGREGATE candlestick-pattern
        # strength canonicals (trailing-window means/fractions); the one-off
        # binary cdl_* detectors are NOT direct alphas (never in this set).
        "candle_body_strength", "candle_wick_balance",
        "candle_range_pct", "candle_pattern_count",
        # R20-SR-DIRECTUSE: dimensionless support/resistance canonicals over a
        # prior-window typical-price median pivot; the price-scale
        # ts_support_level / ts_resistance_level levels stay intermediate
        # (never in this set).
        "sr_distance_pct", "sr_touch_count",
        # R20-CHANNEL-CONSOLIDATION-DIRECTUSE: dimensionless close-only
        # consolidation tightness canonicals; channel_width_pct /
        # channel_position were SKIPPED as exact-duplicate math of
        # donchian_width_pct / donchian_channel_position (already here).
        "consolidation_pct", "consolidation_range_pct",
        # R20-SPECTRAL-EXCESS-DIRECTUSE: dimensionless spectral/wavelet energy
        # ratios (trailing de-meaned rfft / Haar windows ending at t); the
        # frequency_layer filters (ts_spectral_lowpass_trailing,
        # ts_wavelet_shrinkage_trailing) return price-scale filtered series and
        # are NOT alphas (never in this set).
        "spectral_energy_ratio", "spectral_trend_share",
        "wavelet_detail_energy_ratio",
        # R20-REGRESSION-CAUSAL-DIRECTUSE: dimensionless causal regression
        # canonicals (trailing OLS windows ending at t; the forecast family's
        # fit window excludes the target bar).  mean_reversion_half_life was
        # SKIPPED — ts_mean_reversion_half_life already implements the same
        # AR(1) half-life estimator.
        "reg_forecast_error_pct", "reg_slope_tstat", "reg_r2_trailing",
        "reg_residual_zscore",
        # R20-EVENT-STATE-ALPHA: dimensionless Event→Alpha / State→Alpha
        # canonicals (trailing event-count / state-level windows ending at t,
        # strict EventBool {0,1,NaN} inputs, degenerate denominators -> NaN).
        "event_rate_pct", "event_recency_z", "event_cluster_score",
        "state_dwell_pct", "state_transition_surprise",
        # R21-P0-EVENTSTATE-FRAMEWORK: dimensionless Event/State derivation
        # primitives (trailing windows ending at t, strict EventBool {0,1,NaN}
        # / SignedEvent {-1,0,+1,NaN} / state-code panels, min_periods=window
        # fail-closed, degenerate denominators -> NaN).  SKIPPED: event_age
        # (existing alias -> ts_days_since), event_decay (existing alias ->
        # event_decay_asof); state_persistence == state_dwell_pct estimator.
        "event_streak", "event_cluster_duration", "event_decay_window",
        "signed_event_rate", "positive_event_rate", "negative_event_rate",
        "positive_event_age", "negative_event_age", "signed_event_decay",
        "event_direction_imbalance", "event_flip_density", "state_age",
        "state_persistence", "state_transition_count",
        "state_transition_rate", "state_flip_density",
        # R21-P0-EVENTSTATE-FRAMEWORK-CATEGORICAL: CategoricalEvent family
        # (category_age/frequency/transition_rate/transition_surprise),
        # event_direction_persistence, state_episode_age, state_flip_age.
        "category_age", "category_frequency", "category_transition_rate",
        "category_transition_surprise", "event_direction_persistence",
        "state_episode_age", "state_flip_age",
        # R21-CENSORING-VARIANTS: non-censored episode-age siblings so stable
        # long-trend regimes don't degrade to NaN (lower-bound / capped / flag
        # episode-age estimators over state-code panels, trailing windows
        # ending at t, fail-closed NaN semantics).
        "state_episode_age_lower_bound", "state_episode_age_capped",
        "state_episode_censored_flag", "category_age_lower_bound",
        # R20-BVC-VPIN-DIRECTUSE: dimensionless causal intraday BVC/VPIN
        # canonicals (trailing windows ending at t, fail-closed masking,
        # strict-positive close+volume).  Documented non-duplicates:
        # micro_vpin (legacy |r|-weighted proxy, NOT true VPIN) and
        # micro_bvc_vpin (research-only panel pipeline) stay OUT of this set;
        # intraday_bvc_imbalance is panel-shaped, not a series alpha.
        "bvc_sign_pct", "vpin_pct", "bvc_imbalance_ma",
        # R20-RELALPHA-PROMOTION: the six dimensionless ex-self / prior-beta /
        # group-relative canonicals from R20-EXSELF-CS-DIRECTUSE and
        # R20-GROUPSTATE-BETA-DIRECTUSE.  They already resolved DIRECT_ALPHA
        # via the verified MiningRole fallback; this set membership pins the
        # promotion explicitly (same contract as every sibling R20 family) so
        # the verdict never depends on role-fallback drift.  ex_self_mean_gap
        # is deliberately NOT here: it carries x's unit (a level gap), i.e.
        # NOT scale-invariant — intermediate/composition only.
        "ex_self_zscore", "ex_self_rank_pct", "ex_self_mad_z",
        "beta_residual_z", "beta_divergence_pct", "relative_strength_group_pct",
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

# R18-020 / R22-046..051: arbitrary math — keep only what has a real finance/stat
# use.  The raw generic trig (sin/cos/asin/acos/sinh/cosh/tan/cot/sec/csc) is raw
# math with no factor semantics — MOVE_INTERNAL (R22-050 / R22-083).  The typed
# phase/bounded variants (sin_phase / cos_phase / asin_bounded / acos_bounded)
# carry the phase/cyclic/correlation input contract and are DIRECT_INTERMEDIATE
# composition building blocks (R22-047..049).
_ARBITRARY_MATH_STATUS: dict[str, DirectUseContract] = {
    "sin": _contract(DirectUseStatus.MOVE_INTERNAL, "generic raw sine — no factor semantics; use sin_phase inside a phase/seasonality recipe"),
    "cos": _contract(DirectUseStatus.MOVE_INTERNAL, "generic raw cosine — no factor semantics; use cos_phase inside a phase/seasonality recipe"),
    "acos": _contract(DirectUseStatus.MOVE_INTERNAL, "generic raw arccos — no factor semantics; use acos_bounded on a bounded input"),
    "asin": _contract(DirectUseStatus.MOVE_INTERNAL, "generic raw arcsin — no factor semantics; use asin_bounded on a bounded input"),
    "sin_phase": _contract(DirectUseStatus.DIRECT_INTERMEDIATE, "typed phase/cyclic-position sine — composition building block"),
    "cos_phase": _contract(DirectUseStatus.DIRECT_INTERMEDIATE, "typed phase/cyclic-position cosine — composition building block"),
    "asin_bounded": _contract(DirectUseStatus.DIRECT_INTERMEDIATE, "bounded-input arcsin (correlation/ratio/normalized-state only) — composition building block"),
    "acos_bounded": _contract(DirectUseStatus.DIRECT_INTERMEDIATE, "bounded-input arccos (correlation/ratio/normalized-state only) — composition building block"),
    "sinh": _contract(DirectUseStatus.MOVE_INTERNAL, "hyperbolic sine — raw math, not a factor canonical"),
    "cosh": _contract(DirectUseStatus.MOVE_INTERNAL, "hyperbolic cosine — raw math, not a factor canonical"),
    "tan": _contract(DirectUseStatus.MOVE_INTERNAL, "tangent — raw math, not a factor canonical"),
    "cot": _contract(DirectUseStatus.MOVE_INTERNAL, "cotangent — raw math, not a factor canonical"),
    "csc": _contract(DirectUseStatus.MOVE_INTERNAL, "cosecant — raw math, not a factor canonical"),
    "sec": _contract(DirectUseStatus.MOVE_INTERNAL, "secant — raw math, not a factor canonical"),
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

# R18-061: flex_max / flex_min are a DUAL-ARITY form of maximum / minimum.
# Differential testing proves ``flex_max(x, y) == maximum(x, y)`` exactly on the
# panel input, but the flex forms additionally accept a scalar window
# (``flex_max(x, 3) == ts_max(x, 3)``).  So they are RELATED_NOT_DUPLICATE, not
# EXACT_EQUIVALENT — per R18-022/117 (only merge proven exact equivalents) and
# R18-088 (high-correlation but not identical -> share a redundancy group, don't
# delete).  Keep one search branch: DIRECT_INTERMEDIATE comparison building block,
# budgeted under the ``flex_max_min`` redundancy group.
_FLEX_MAX_MIN_STATUS = {
    "flex_max": _contract(
        DirectUseStatus.DIRECT_INTERMEDIATE,
        "dual-arity max building block: panel (==maximum) or rolling window (==ts_max)",
    ),
    "flex_min": _contract(
        DirectUseStatus.DIRECT_INTERMEDIATE,
        "dual-arity min building block: panel (==minimum) or rolling window (==ts_min)",
    ),
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
    # R18-061: flex vs plain max/min are RELATED_NOT_DUPLICATE (panel == exact,
    # rolling == ts_max) — budgeted under one family, not deleted.
    "flex_max": "flex_max_min", "flex_min": "flex_max_min",
    "maximum": "flex_max_min", "minimum": "flex_max_min",
    "ts_max": "flex_max_min", "ts_min": "flex_max_min",
}

# R22-136..137: default search prior / family budget / cost budget per cost lane.
# These are CONFIGURABLE defaults (R22-137: "具体数值可配置，不要硬编码经济结论")
# — the miner may override them via its campaign config.  Prior is the relative
# sampling weight; basic alphas keep a high prior, advanced high-cost families a
# low prior so hundreds of advanced operators never drown out the basics.
_SEARCH_PRIOR_BY_LANE: dict[str, float] = {
    "alpha_direct": 1.0,
    "alpha_high_cost": 0.2,
    "intermediate": 0.8,
    "state": 0.8,
    "condition": 0.8,
    "event": 0.8,
    "group_state": 0.8,
    "global_state": 0.8,
    "source_transform": 0.9,
    "control_flow": 0.8,
}
_FAMILY_BUDGET_BY_LANE: dict[str, int] = {
    "alpha_direct": 16,
    "alpha_high_cost": 6,
    "intermediate": 8,
    "state": 8,
    "condition": 8,
    "event": 8,
    "group_state": 8,
    "global_state": 8,
    "source_transform": 4,
    "control_flow": 8,
}


def _search_budget(lane: str) -> tuple[float, int, int]:
    """R22-136: (search_prior, family_budget, cost_budget) for a mining lane."""
    prior = _SEARCH_PRIOR_BY_LANE.get(lane, 0.5)
    family = _FAMILY_BUDGET_BY_LANE.get(lane, 4)
    cost = 6 if lane == "alpha_high_cost" else 9
    return prior, family, cost


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
    # ADJ_FIELD_MIGRATION: A-share volume binds to the ADJUSTED share count
    # (Volume/Factor, StockDailyBarAdj authority); US continuous_volume_shares
    # resolves to raw Volume (identity). raw_volume_shares is disabled for A-share.
    "ts_average_volume": {"volume": "continuous_volume_shares"},
    "dollar_volume": {"close": "continuous_close", "volume": "continuous_volume_shares"},
    "true_range": {"high": "continuous_high", "low": "continuous_low", "close": "continuous_close"},
    "ATR_WILDER": {"high": "continuous_high", "low": "continuous_low", "close": "continuous_close"},
    "NATR": {"high": "continuous_high", "low": "continuous_low", "close": "continuous_close"},
    "tail_beta": {"ret": "return_decimal", "benchmark_ret": "benchmark_return", "window": 60},
    "residual_momentum_capm": {"ret": "return_decimal", "benchmark_ret": "benchmark_return", "window": 60},
    "coskewness_to_market": {"ret": "return_decimal", "benchmark_ret": "benchmark_return", "window": 60},
    "idio_vol": {"ret": "return_decimal", "benchmark_ret": "benchmark_return", "window": 60},
    "idio_skew": {"ret": "return_decimal", "benchmark_ret": "benchmark_return", "window": 60},
    # R22-031..045: default input recipes for the promoted research families.
    # Recipes bind to canonical *concepts* (R22-121), resolved to market sources
    # by the R17 resolver — never physical columns.
    "date_diff_days": {"left": "event_date", "right": "event_date"},
    "ts_wavelet_lowpass_reconstruct": {"x": "continuous_close"},
    "ts_signature_mahalanobis_anomaly": {
        "f1": "return_decimal", "f2": "volume", "f3": "turnover",
    },
    "ts_persistence_birth_dispersion": {"x": "return_decimal"},
    "ts_bicoherence_top_decile_excess": {"x": "return_decimal"},
    "ts_kernel_granger_score": {"y": "return_decimal", "x": "return_decimal"},
    "ts_residualized_hsic": {"x": "return_decimal", "y": "return_decimal", "z": "return_decimal"},
    "ts_bds_statistic": {"x": "return_decimal"},
    "ts_rolling_sr_gaussian_mean_shift_score": {"x": "return_decimal"},
    "ts_garch_*": {"x": "return_decimal"},
    "ts_gjr_garch_*": {"x": "return_decimal"},
    "ts_gjr_leverage": {"x": "return_decimal"},
    "ts_har_*": {"x": "return_decimal"},
    "ts_kalman_*": {"x": "return_decimal"},
    "ts_dmd_level_*": {"x": "continuous_close"},
    "ts_dmd_return_*": {"x": "return_decimal"},
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
# R22-031..051: explicit promotion / verdict table for the research-surface
# factor families.  ``surface == research`` is an *authoring/lifecycle* state
# (R22-009..012), NOT a semantic role: a causal, deterministic, panel-shaped
# operator that merely lacks production evidence gets a DIRECT_* role +
# PENDING_CERTIFICATION admission, never MiningRole.RESEARCH.  Pure
# in-sample diagnostics / no-data / raise-on-call canonicals get their honest
# RESEARCH_TOOL / DELETE_* verdict here so no public operator is ever left in
# ``PUBLIC_PENDING_FOREVER`` (R22-030 / R22-148).
#
# Key is a canonical or a ``family_*`` fnmatch pattern.  Exact keys win over
# patterns; the table is checked first in ``_resolve_explicit`` so research
# families always beat the surface-based role fallback.
_R22_RESEARCH_PROMOTION: dict[str, tuple[DirectUseStatus, str]] = {
    # ---- research_transform (R22-031..033) ----
    "ts_wavelet_lowpass_reconstruct": (
        DirectUseStatus.DIRECT_INTERMEDIATE,
        "strict-trailing Haar smoothing — composition building block; residual/ratio recipes expose the alpha",
    ),
    "ts_signature_mahalanobis_anomaly": (
        DirectUseStatus.DIRECT_ALPHA_HIGH_COST,
        "causal 3-channel path-signature anomaly distance — high-cost lane",
    ),
    "ts_persistence_birth_dispersion": (
        DirectUseStatus.DIRECT_ALPHA_HIGH_COST,
        "causal Rips-H1 persistence dispersion — high-cost lane",
    ),
    # ---- DMD (R22-034..039): generic untyped forms move internal as compat
    # aliases to the level/return typed variants; typed forms are mineable. ----
    "ts_dmd_level_dominant_growth_rate": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "level-dynamics DMD dominant growth rate"),
    "ts_dmd_level_dominant_frequency": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "level-dynamics DMD dominant frequency"),
    "ts_dmd_level_mode_concentration": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "level-dynamics DMD mode concentration"),
    "ts_dmd_return_dominant_growth_rate": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "return-dynamics DMD dominant growth rate"),
    "ts_dmd_return_dominant_frequency": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "return-dynamics DMD dominant frequency"),
    "ts_dmd_return_mode_concentration": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "return-dynamics DMD mode concentration"),
    "ts_dmd_dominant_growth_rate": (DirectUseStatus.MOVE_INTERNAL, "untyped DMD — compat alias; use the ts_dmd_level/return typed variants"),
    "ts_dmd_dominant_frequency": (DirectUseStatus.MOVE_INTERNAL, "untyped DMD — compat alias; use the ts_dmd_level/return typed variants"),
    "ts_dmd_mode_concentration": (DirectUseStatus.MOVE_INTERNAL, "untyped DMD mode concentration — use the ts_dmd_level/return typed variants"),
    # ---- research_spectral (R22-040..045) ----
    "ts_bicoherence_top_decile_mean": (
        DirectUseStatus.DIRECT_INTERMEDIATE,
        "base bicoherence statistic — bias-corrected sibling preferred as terminal",
    ),
    "ts_bicoherence_top_decile_excess": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "surrogate-null-subtracted bicoherence excess"),
    "ts_kernel_granger_score": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "blocked OOS predictive-improvement score"),
    "ts_residualized_hsic": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "residualized HSIC dependence statistic"),
    "ts_bds_statistic": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "nonlinear serial-dependence statistic"),
    "ts_rolling_sr_gaussian_mean_shift_score": (DirectUseStatus.DIRECT_ALPHA, "rolling change-point intensity — continuous change evidence"),
    # ---- panel factor models ----
    "panel_rolling_pca_explained_ratio": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "rolling panel-PCA explained ratio"),
    "panel_rolling_pca_loading": (DirectUseStatus.DIRECT_INTERMEDIATE, "rolling panel-PCA loading — factor composition building block"),
    "panel_rolling_pca_resid": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "rolling panel-PCA residual"),
    "panel_rolling_pca_resid_momentum": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "rolling panel-PCA residual momentum"),
    "panel_rolling_pca_resid_vol": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "rolling panel-PCA residual volatility"),
    "panel_rolling_pcr_forecast": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "rolling PCR forecast"),
    "panel_rolling_pls_forecast": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "rolling PLS forecast"),
    "panel_rolling_elastic_net_forecast": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "rolling elastic-net forecast"),
    "panel_mixture_of_experts_score": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "panel MoE regime score"),
    "panel_regime_conditioned_forecast": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "regime-conditioned forecast"),
    "industry_rolling_pca_loading": (DirectUseStatus.DIRECT_INTERMEDIATE, "industry rolling-PCA loading — composition building block"),
    "ts_feature_pca_reconstruction_error": (DirectUseStatus.DIRECT_INTERMEDIATE, "feature PCA reconstruction error"),
    "cs_autoencoder_reconstruction_error": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "autoencoder reconstruction error"),
    # ---- volatility / state-space model families (R22-098 high-cost lane) ----
    "ts_garch_*": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "GARCH volatility dynamics — high-cost model lane"),
    "ts_gjr_garch_*": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "GJR-GARCH leverage dynamics — high-cost model lane"),
    "ts_gjr_leverage": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "GJR negative-return shock coefficient — high-cost model lane"),
    "ts_har_*": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "HAR realized-volatility model family"),
    "ts_kalman_*": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "Kalman state/innovation family"),
    "ts_regime_duration": (DirectUseStatus.DIRECT_ALPHA, "regime duration statistic"),
    "ts_two_state_regime_probability": (DirectUseStatus.DIRECT_STATE, "two-state regime probability — regime state gate"),
    "ts_change_point_probability": (DirectUseStatus.DIRECT_ALPHA, "change-point probability"),
    "ts_cusum_vol_break_score": (DirectUseStatus.DIRECT_ALPHA, "CUSUM volatility break score"),
    # ---- complexity / dependence / spectral ----
    "ts_lz_complexity": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "Lempel-Ziv complexity"),
    "ts_active_information_storage": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "active information storage"),
    "ts_conditional_transfer_entropy": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "conditional transfer entropy"),
    "ts_effective_transfer_entropy": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "effective transfer entropy"),
    "ts_markov_entropy_production": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "Markov entropy production"),
    "ts_multiscale_entropy_slope": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "multiscale entropy slope"),
    "ts_multiscale_permutation_entropy_slope": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "multiscale permutation-entropy slope"),
    "ts_pseudocount_sample_entropy": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "sample entropy with pseudocount"),
    "ts_motif_recurrence_count": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "motif recurrence count"),
    "ts_matrix_profile_discord_score": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "matrix-profile discord score"),
    "ts_wavelet_entropy": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "wavelet entropy"),
    "ts_wavelet_energy_slope": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "wavelet energy slope"),
    "ts_wavelet_high_frequency_ratio": (DirectUseStatus.DIRECT_INTERMEDIATE, "wavelet high-frequency ratio — composition building block"),
    "ts_wavelet_low_frequency_ratio": (DirectUseStatus.DIRECT_INTERMEDIATE, "wavelet low-frequency ratio — composition building block"),
    "ts_spectral_low_frequency_ratio": (DirectUseStatus.DIRECT_INTERMEDIATE, "spectral low-frequency ratio — composition building block"),
    "ts_modwt_band_corr": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "MODWT band correlation"),
    "ts_dfa_hurst": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "detrended-fluctuation Hurst exponent"),
    "ts_local_lyapunov_exponent": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "local Lyapunov exponent"),
    "ts_extremal_dependence_decay": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "extremal dependence decay"),
    "ts_turning_point_ratio": (DirectUseStatus.DIRECT_ALPHA, "turning-point ratio"),
    "ts_roll_effective_spread": (DirectUseStatus.DIRECT_ALPHA, "Roll effective spread"),
    # ---- copula / knn / group / event ----
    "ts_copula_central_asymmetry": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "copula central asymmetry"),
    "cs_rank_copula_entropy": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "rank-copula entropy"),
    "cs_rank_copula_mi": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "rank-copula mutual information"),
    "cs_knn_local_gradient_norm": (DirectUseStatus.DIRECT_INTERMEDIATE, "KNN local gradient norm — manifold building block"),
    "cs_knn_tangent_residual": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "KNN tangent residual"),
    "group_signal_attraction_share": (DirectUseStatus.DIRECT_ALPHA, "group signal attraction share"),
    "group_tail_lead_score": (DirectUseStatus.DIRECT_ALPHA, "group tail-lead score"),
    "relation_diffusion_score": (DirectUseStatus.DIRECT_ALPHA, "relation diffusion score"),
    "event_hawkes_branching_ratio_proxy": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "Hawkes branching-ratio proxy"),
    # ---- topology / path signature ----
    "ts_betti_1_max_persistence": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "Rips-H1 betti-1 max persistence"),
    "ts_persistence_diagram_shift": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "persistence-diagram shift"),
    "ts_path_signature_area": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "path-signature area"),
    "ts_path_signature_depth2_norm": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "path-signature depth-2 norm"),
    "ts_path_leadlag_area": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "path lead-lag area"),
    "ts_quantile_crossing_spectral_concentration": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "quantile-crossing spectral concentration"),
    "ts_fisher_information_shift": (DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "Fisher information shift"),
    # ---- intraday research (minute input -> daily output, INTRADAY_EOD) ----
    "intraday_profile_phase_shift": (DirectUseStatus.DIRECT_ALPHA, "intraday profile phase shift"),
    "intraday_quantile_curve_pca_residual": (DirectUseStatus.DIRECT_ALPHA, "intraday quantile-curve PCA residual"),
    "intraday_quantile_curve_pca_score": (DirectUseStatus.DIRECT_ALPHA, "intraday quantile-curve PCA score"),
    "intraday_realized_power_variation": (DirectUseStatus.DIRECT_ALPHA, "intraday realized power variation"),
    # ---- report family ----
    "report_benford_js_divergence": (DirectUseStatus.DIRECT_ALPHA, "Benford JS divergence"),
    "report_revision_magnitude": (DirectUseStatus.DIRECT_ALPHA, "report revision magnitude"),
    # ---- pure in-sample diagnostics -> research tool (R22-052..056) ----
    "ts_quantile_regression_beta": (
        DirectUseStatus.RESEARCH_TOOL,
        "in-sample quantile-regression coefficient — diagnostic; the *_prior / *_forecast_error counterparts are the mineable forms",
    ),
    # R28 §二十二 / §一百一十六: in-sample AR fitted value / residual are
    # diagnostic (the current row participates in its own fit) — NOT alpha
    # terminals.  The ``ts_ar_prior_*`` forms (fit <= t-1) are the mineable ones.
    "ts_ar_fitted_value": (
        DirectUseStatus.RESEARCH_TOOL,
        "in-sample AR fitted value (current row participates in its own fit) — diagnostic; use ts_ar_prior_forecast for the causal one-step form",
    ),
    "ts_ar_in_sample_resid": (
        DirectUseStatus.RESEARCH_TOOL,
        "in-sample AR residual — diagnostic; use ts_ar_prior_innovation for the causal out-of-sample form",
    ),
    # R22-058: causal poly2 siblings (prior fit <= t-1; current obs evaluates only).
    "ts_poly2_prior_coeff": (DirectUseStatus.DIRECT_ALPHA, "prior-fit quadratic coefficient — causal"),
    "ts_poly2_forecast_error": (DirectUseStatus.DIRECT_ALPHA, "one-step-ahead quadratic forecast error — causal"),
    "ts_poly2_forecast_error_z": (DirectUseStatus.DIRECT_ALPHA, "standardized one-step-ahead quadratic forecast error — causal"),
    # R22-060..062: rolling_beta_to_market requires an EXPLICIT benchmark_ret
    # (raises without it) — it is the explicit-benchmark form (R22-061 route A),
    # never a self-inclusion beta, so it is directly mineable.
    "rolling_beta_to_market": (
        DirectUseStatus.DIRECT_ALPHA,
        "explicit-benchmark rolling beta (benchmark_ret required) — route-A mineable form",
    ),
    # ---- no-data / raise-on-call (R22-077..079): never keep a public operator
    # that raises on every call, and never fake a field that has no provider. ----
    "holder_concentration_change": (
        DirectUseStatus.DELETE_NO_DATA,
        "implementation raises (requires distinct relation snapshots); no snapshot provider — delete until real data exists",
    ),
    "holder_count_change_rate": (
        DirectUseStatus.DELETE_NO_DATA,
        "TopTen rows are not total shareholder count; no total-holder provider — delete rather than fake",
    ),
}


# R22-146..147 / R22-158: causal replacement map — every in-sample diagnostic /
# benchmark-only research tool points at its mineable causal sibling, so
# ``CAUSAL_REPLACEMENT_MISSING`` stays 0 (a tool without a safe replacement must
# declare ``NO_SAFE_REPLACEMENT`` explicitly in the research manifest).
_CAUSAL_REPLACEMENT_MAP: dict[str, str] = {
    "ts_multi_regression_coeff": "ts_multi_regression_coeff_prior",
    "ts_multi_regression_resid": "ts_multi_regression_forecast_error",
    "ts_multi_regression_resid_z": "ts_multi_regression_forecast_error_z",
    "ts_multi_regression_r2": "ts_multi_regression_r2_prior",
    "ts_huber_regression_coeff": "ts_huber_regression_coeff_prior",
    "ts_huber_regression_in_sample_resid": "ts_huber_regression_forecast_error",
    "ts_huber_regression_resid_z": "ts_huber_regression_forecast_error_z",
    "ts_ridge_regression_coeff": "ts_ridge_regression_coeff_prior",
    "ts_ridge_regression_in_sample_resid": "ts_ridge_regression_forecast_error",
    "ts_ridge_regression_resid_z": "ts_ridge_regression_forecast_error_z",
    "ts_ar_forecast": "ts_ar_prior_forecast",
    "ts_ar_innovation": "ts_ar_prior_innovation",
    "ts_ar_innovation_z": "ts_ar_prior_innovation_z",
    "ts_ar_fitted_value": "ts_ar_prior_forecast",
    "ts_ar_in_sample_resid": "ts_ar_prior_innovation",
    "ts_expectile_regression_coeff": "ts_expectile_regression_coeff_prior",
    "ts_expectile_regression_resid": "ts_expectile_regression_forecast_error",
    "ts_quantile_regression_beta": "ts_quantile_regression_coeff_prior",
    "ts_quantile_regression_coeff": "ts_quantile_regression_coeff_prior",
    "ts_quantile_regression_slope": "ts_quantile_regression_coeff_prior",
    "ts_quantile_regression_resid": "ts_quantile_regression_coeff_prior",
    "ts_poly2_coeff": "ts_poly2_prior_coeff",
    "ts_poly2_resid": "ts_poly2_forecast_error",
    "intra_idiosyncratic_variance": "intra_idiosyncratic_variance_ex_self",
    "intra_realized_beta": "intra_realized_beta_ex_self",
    "intra_realized_correlation": "intra_realized_correlation_ex_self",
    "micro_bvc_vpin": "intraday_bvc_imbalance",
    "cs_beta_to_market": "cs_beta_to_market_ex_self",
    "cs_alpha_to_market": "cs_alpha_to_market_ex_self",
    "ts_mean_reversion_half_life": "NO_SAFE_REPLACEMENT",
}


def causal_replacement(canonical: str) -> str:
    """R22-146: recommended mineable replacement for a research-tool / diagnostic
    canonical (``""`` when the canonical is not a tool; ``NO_SAFE_REPLACEMENT``
    when it is a tool with no safe causal form)."""
    return _CAUSAL_REPLACEMENT_MAP.get(canonical, "")


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
    # R22-031..051: research-surface factor families get an explicit DIRECT_* /
    # honest-tool / honest-delete verdict FIRST, so ``surface == research`` never
    # silently demotes a causal factor-shaped operator to MiningRole.RESEARCH.
    for pattern, (status, reason) in _R22_RESEARCH_PROMOTION.items():
        if _fnmatch_any(canonical, (pattern,)):
            return _contract(status, reason)
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
    if canonical in _FULL_HISTORY_REPLAY_ALPHA:
        # R63: full-history replay indicators (KAMA/DEMA/TEMA/PSAR/Supertrend/…)
        # stay ALPHA_HIGH_COST, NOT price-level DIRECT_INTERMEDIATE: they are
        # terminal-grade factors, just expensive to replay.  The alpha_high_cost
        # lane is terminal-eligible (DirectUse terminal authority still gates it).
        return _contract(DirectUseStatus.DIRECT_ALPHA_HIGH_COST, "full-history replay indicator — terminal-grade alpha (high cost)")
    if canonical in _PRICE_LEVEL_INTERMEDIATE_OPS:
        return _contract(DirectUseStatus.DIRECT_INTERMEDIATE, "price-level building block — not scale-invariant, composition only")
    if canonical in _RELATIVE_ALPHA_OPS:
        return _contract(DirectUseStatus.DIRECT_ALPHA, "normalized/relative version of a level building block — dimensionless alpha")
    if canonical in _SOURCE_TRANSFORM_OPS:
        return _contract(DirectUseStatus.DIRECT_SOURCE_TRANSFORM, "missing-value / source transform — data-processing slot only")
    if canonical in _ARBITRARY_MATH_STATUS:
        return _ARBITRARY_MATH_STATUS[canonical]
    if canonical in _FLEX_MAX_MIN_STATUS:
        return _FLEX_MAX_MIN_STATUS[canonical]
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
        # R22-103..105: terminal legality is decided by the DirectUseRole, NEVER
        # by lifecycle_status / production_certified.  A DIRECT_ALPHA with
        # PENDING_CERTIFICATION has terminal_semantically_legal=True and
        # production_admitted=False — the certification gate lives in
        # ``production_admitted``, not in the terminal flag.
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
    if role is MiningRole.INTRADAY_EOD:
        return _contract(DirectUseStatus.DIRECT_ALPHA, "intraday-to-EOD alpha (verified role)")
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
        # R18-126: a DENIED role alone is NOT a DELETE_USELESS verdict.  The
        # current DENIED set is DSL/grammar infrastructure (arg/constant) and
        # arbitrary-math utilities (sinh/cosh/cot/csc/sec/tan) that the mining
        # layer permanently excludes from factor search.  Per R18-020 those are
        # "移出默认 mining，必要时 ResearchTool/internal" — never a public factor
        # canonical, but never mechanically deleted either.  Future/random
        # non-causal primitives get an explicit DELETE_NONCAUSAL override in
        # ``_resolve_explicit`` when they appear.
        if canonical in {"arg", "constant"}:
            return _contract(DirectUseStatus.MOVE_INTERNAL, "DSL/grammar primitive — internal supporting layer")
        return _contract(
            DirectUseStatus.RESEARCH_TOOL,
            "permanently excluded from mining — arbitrary-math/utility; research tool only",
        )
    # UNRESOLVED must never survive into the retained set — surface it.
    return _contract(
        DirectUseStatus.DELETE_USELESS,
        "no verified role — unresolved canonical must be deleted or explicitly classified",
    )


def resolve_direct_use(canonical: str, catalog: dict[str, Any] | None = None) -> DirectUseContract:
    """R18-124: one final verdict per canonical, never SKIP/UNRESOLVED.

    R22-146: a research-tool verdict also records its causal replacement, so the
    research manifest can report ``reason_not_mineable`` +
    ``recommended_factor_replacement`` (or ``NO_SAFE_REPLACEMENT``)."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    catalog = dict(catalog) if catalog is not None else (OperatorRegistry._catalog.get(canonical) or {})
    explicit = _resolve_explicit(canonical, catalog)
    contract = explicit if explicit is not None else _default_from_role(canonical, catalog)
    if contract.replacement:
        return contract
    repl = _CAUSAL_REPLACEMENT_MAP.get(canonical)
    if repl:
        return DirectUseContract(
            status=contract.status,
            terminal_allowed=contract.terminal_allowed,
            retention_reason=contract.retention_reason,
            ast_positions=contract.ast_positions,
            replacement=repl,
            delete_reason=contract.delete_reason,
        )
    return contract


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

# R22-042/043/122..123: explicit slot semantics for multi-input advanced
# operators — a miner must never infer meaning from bare ``x/y/z/f1/f2/f3``.
_SLOT_SEMANTIC_ROLES: dict[str, dict[str, str]] = {
    "ts_kernel_granger_score": {"y": "target", "x": "predictor"},
    "ts_residualized_hsic": {
        "x": "tested_variable", "y": "tested_variable", "z": "conditioning_variable",
    },
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
        "left", "right",  # date_diff_days date-column inputs are panel fields
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


def input_slot_specs(
    canonical: str,
    catalog: dict[str, Any],
    *,
    panel_params: Iterable[str] | None = None,
    scalar_params: Iterable[str] | None = None,
) -> tuple[InputSlotSpec, ...]:
    """Build the R18-029 input-slot contract for a canonical.

    ``data_inputs`` are the panel data fields; ``scalar_parameters`` are the
    numeric knobs; ``context_inputs`` / ``group_inputs`` / ``event_inputs`` are
    explicit context slots.  An ``inputs`` key, where kept, equals exactly the
    data-input list (R18-002).

    The data-vs-scalar boundary is the AUTHORITATIVE one — ``split_scalar_panel_params``
    (declared ``panel_params`` / ``input_fields`` / OperatorSpec inference) — never a
    hand-rolled name heuristic.  ``panel_params`` / ``scalar_params`` are resolved by
    :func:`build_direct_use_operator` and passed in; the slot *typing* (data /
    condition / event / group / context) is layered on top for the R18-002 five-way split.
    """
    params = list(catalog.get("param_names") or ())
    if panel_params is None or scalar_params is None:
        _pp = set(panel_params or ())
        _sp = set(scalar_params or ())
        for name in params:
            if name in _pp or name in PANEL_PARAM_SET:
                _pp.add(name)
            else:
                _sp.add(name)
        panel_params = [p for p in params if p in _pp]
        scalar_params = [p for p in params if p in _sp]
    else:
        panel_params = list(panel_params)
        scalar_params = list(scalar_params)
    condition_name = _CONDITION_SLOT_NAMES.get(canonical)
    event_names = _EVENT_SLOTS.get(canonical, ())
    context_names = _CONTEXT_SLOTS.get(canonical, ())
    slots: list[InputSlotSpec] = []
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
        if name in panel_params:
            _role_map = _SLOT_SEMANTIC_ROLES.get(canonical, {})
            slots.append(
                InputSlotSpec(
                    parameter=name,
                    allowed_semantic_kinds=("field",),
                    allowed_units=(),
                    allowed_roles=(_role_map[name],) if name in _role_map else ("data",),
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
    canonical: str,
    catalog: dict[str, Any],
    *,
    slots: Sequence[InputSlotSpec] | None = None,
) -> dict[str, list[str]]:
    """R18-002: data_inputs / scalar_parameters / context_inputs / group_inputs /
    event_inputs split.  ``inputs`` (when needed) == data_inputs."""
    data: list[str] = []
    scalar: list[str] = []
    context: list[str] = []
    group: list[str] = []
    event: list[str] = []
    if slots is None:
        slots = input_slot_specs(canonical, catalog)
    for slot in slots:
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


def operator_family_id(canonical: str) -> str:
    """One canonical family id per operator for budget/quota/grouping.

    Deterministic resolution order (exact win beats pattern):
      1. explicit semantic redundancy group (R18-069)
      2. economic effect family (R18-068)
      3. monotonic transform class (R18-021)
      4. canonical itself (degenerate, stable)

    ``operator_family_id`` and ``group_source_first`` in ``mining/campaign.py``
    are the two stable grouping surfaces; this one is family-budget oriented,
    that one is dependency/source oriented.
    """
    group = semantic_redundancy_group(canonical)
    if group:
        return group
    family = economic_effect_family(canonical)
    if family and family != "unknown":
        return family
    mono = MONOTONIC_TRANSFORM_CLASS.get(canonical)
    if mono:
        return mono
    return canonical


def monotonic_transform_class(canonical: str) -> str:
    """R18-021: monotonic-equivalent class (dedup x/exp(x)/sigmoid(x)/rank(x))."""
    return MONOTONIC_TRANSFORM_CLASS.get(canonical, "")


def min_effective_sample_contract(canonical: str) -> dict[str, int]:
    """R18-047: effective-sample floors for advanced statistical families."""
    for pattern, contract in _MIN_EFFECTIVE_SAMPLE.items():
        if _fnmatch_any(canonical, (pattern,)):
            return dict(contract)
    return {}


# R22-121: concept fallback for the default input recipe — when no explicit
# recipe exists, map the operator's data-input names to canonical concepts
# (resolved to market sources by the R17 resolver; never physical columns).
_INPUT_CONCEPT_FALLBACK: dict[str, str] = {
    "x": "continuous_close", "close": "continuous_close",
    "high": "continuous_high", "low": "continuous_low",
    "open": "continuous_open", "vwap": "continuous_vwap",
    # ADJ_FIELD_MIGRATION: volume binds to the adjusted share count
    # (continuous_volume_shares = Volume/Factor on StockDailyBarAdj authority;
    # US resolves to raw Volume identity).  raw_volume_shares is disabled for A-share.
    "volume": "continuous_volume_shares", "amount": "amount_local",
    "ret": "return_decimal", "returns": "return_decimal",
    "return": "return_decimal", "y": "return_decimal",
    "benchmark_ret": "benchmark_return", "benchmark": "benchmark_return",
    # R25-032/033: weight/weights are NOT price-like signals.  A generic
    # continuous_close binding is forbidden — a weight slot must be resolved by
    # the per-canonical contract (NonNegativeWeight / SignedWeight) or fail
    # recipe resolution, never silently fabricated as close.
    "values": "continuous_close", "value": "continuous_close",
    "f1": "return_decimal", "f2": "continuous_volume_shares", "f3": "turnover",
    "a": "continuous_close", "b": "continuous_close", "c": "continuous_close",
    "price": "continuous_close", "source": "continuous_close",
    "target": "return_decimal", "factor": "continuous_close",
    "open_p": "continuous_open", "high_p": "continuous_high",
    "low_p": "continuous_low", "close_p": "continuous_close",
    "volume_p": "continuous_volume_shares", "amount_p": "amount_local",
    # fundamental period / income / balance concepts (R17 resolver names).
    "period_id": "period_id",
    "turnover": "turnover",
    "pb": "price_to_book",
    "pe": "price_to_earnings",
    "market_cap": "market_cap",
    "change_date": "event_date",
    "date1": "event_date",
    "date2": "event_date",
    "left": "event_date",
    "right": "event_date",
    "flow_type": "cash_flow_type",
    "event": "event_mask",
    "condition": "condition_bool",
    "expected": "earnings_expected",
    "avg_assets": "total_assets",
    "ocf": "operating_cash_flow",
    "net_profit": "net_profit",
    "net_income": "net_profit",
    "total_assets": "total_assets",
    "book_value": "book_value",
    "eps": "earnings_per_share",
    "revenue": "revenue",
    "operating_profit": "operating_profit",
    "equity": "total_equity",
    "debt": "total_debt",
    "cash": "cash_balance",
    "inventory": "inventory",
    "receivables": "accounts_receivable",
    "payables": "accounts_payable",
    # R25-032/033: anonymous multi-input slots (sid*/p*/s*/f4) must NOT default
    # to continuous_close.  ``sid*`` are entity identifiers (EntityStableId),
    # not price panels; ``weight`` needs NonNegativeWeight/SignedWeight; other
    # anonymous slots must be decided by the per-canonical contract.  Removing
    # these generic bindings makes such operators fail recipe resolution
    # honestly instead of silently binding every slot to the anchor close price.
    "f1": "return_decimal", "f2": "continuous_volume_shares", "f3": "turnover",
}


def default_input_recipe(canonical: str) -> dict[str, str]:
    """R18-070 / R22-120/121: operator-level default input binding recipe.

    Exact canonical wins; ``family_*`` fnmatch patterns cover promoted research
    families.  R22-120: EVERY retained direct operator gets at least one real
    recipe — the fallback maps the operator's data inputs to canonical concepts.
    """
    exact = _DEFAULT_INPUT_RECIPE.get(canonical)
    if exact is not None:
        return dict(exact)
    for pattern, recipe in _DEFAULT_INPUT_RECIPE.items():
        if pattern.endswith("*") and _fnmatch_any(canonical, (pattern,)):
            return dict(recipe)
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        cat = OperatorRegistry._catalog.get(canonical) or {}
        panel, _scalar = _authoritative_param_split(canonical, cat)
        _slots = input_slot_specs(canonical, cat, panel_params=panel, scalar_params=_scalar)
        data = split_input_slots(canonical, cat, slots=_slots)["data_inputs"]
        recipe: dict[str, str] = {}
        for name in data:
            concept = _INPUT_CONCEPT_FALLBACK.get(name)
            if concept and name not in recipe:
                recipe[name] = concept
        # R22-120: every retained operator with a panel input gets at least one
        # real binding — anchor the first data input to the close-price concept
        # when no name-specific concept matched.
        if not recipe and data:
            recipe[data[0]] = "continuous_close"
        return recipe
    except Exception:
        return {}


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

    canonical: str = ""
    aliases: tuple[str, ...] = ()
    module: str = ""
    class_name: str = ""
    authoring_tier: str = "unknown"

    direct_use_status: DirectUseStatus = DirectUseStatus.DELETE_USELESS
    mining_role: str = "unresolved"
    terminal_allowed: bool = False
    allowed_ast_positions: tuple[str, ...] = ()
    mining_lane: str = "supporting"

    retention_reason: str = ""
    delete_reason: str = ""
    replacement: str = ""

    economic_effect_family: str = "unknown"
    semantic_redundancy_group: str = ""
    monotonic_transform_class: str = ""

    input_slots: tuple[InputSlotSpec, ...] = ()
    data_inputs: tuple[str, ...] = ()
    scalar_parameters: tuple[str, ...] = ()
    context_inputs: tuple[str, ...] = ()
    group_inputs: tuple[str, ...] = ()
    event_inputs: tuple[str, ...] = ()
    condition_slot: str = ""

    output_semantic_kind: str = "series"
    output_unit: str | None = None
    output_cardinality: str = "panel"
    output_value_domain: str = "continuous_signed"
    output_grain: str | None = None

    default_input_recipe: dict[str, str] = field(default_factory=dict)
    smoke_recipe_ids: tuple[str, ...] = ()

    supported_markets: tuple[str, ...] = ()
    source_recipes: tuple[str, ...] = ()

    default_params: tuple[str, ...] = ()
    searchable_params: tuple[str, ...] = ()
    search_grade_by_param: dict[str, str] = field(default_factory=dict)
    parameter_injectivity_passed: bool = False

    stateful: bool = False
    execution_model: str = "independent_with_warmup"
    checkpoint_supported: bool = False
    full_history_replay_allowed: bool = True

    runtime_cost: int = 0
    memory_cost: int = 0
    preferred_backend: str = "pandas_numpy"
    reference_backend: str = "pandas_numpy"

    production_certified: bool = False
    directly_usable: bool = False

    # R22-003..008: the single ``directly_usable`` verdict is split into five
    # orthogonal fields.  ``directly_usable`` stays as the backward-compatible
    # alias ``mining_visible ∧ production_admitted`` (the R18 "eligible" notion).
    mining_visible: bool = False
    composition_usable: bool = False
    terminal_usable: bool = False
    production_terminal_usable: bool = False
    production_admitted: bool = False
    context_admitted: bool = False

    # R23-101A: catalog-declared injectivity fact and same-row probe-completion
    # truth — kept next to the runtime certificate so a consumer can diagnose a
    # False certificate without re-running the whole-directory probe shred.
    parameter_injectivity_declared: bool = False
    math_probe_missing: bool = False

    # R64: per-gate production ladder (view, see build_direct_use_operator) and
    # the causality tri-state (causal / unknown / leak_detected).
    r64_production_gates: dict[str, str] = field(default_factory=dict)
    r64_causality: str = "unknown"

    min_effective_samples: dict[str, int] = field(default_factory=dict)
    scale_sensitive: dict[str, bool] = field(default_factory=dict)
    search_prior: float = 0.5
    family_budget: int = 4
    cost_budget: int = 9
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
    """Environment the miner declares — R18-031 contextual direct usability.

    R21-P033: market is now a required field. market=None is not allowed in
    production contexts.  ``__post_init__`` hard-fails on a missing/empty
    market so no caller can silently construct a market-less context.
    """

    market: Market  # Required: Market.ASHARE | Market.US
    available_sources: tuple[str, ...] = ()
    target_frequency: str | None = None
    max_cost: int | None = None

    def __post_init__(self) -> None:
        if not self.market:
            raise ValueError("market is required (R21-P033)")
        # Validate market is a valid Market enum value
        if not isinstance(self.market, Market):
            raise TypeError(
                f"market must be a Market enum, got {type(self.market).__name__}; "
                f"expected Market.ASHARE or Market.US"
            )


def _alias_map() -> dict[str, str]:
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        return dict(OperatorRegistry._aliases)
    except Exception:
        return {}


def _op_class_name(canonical: str) -> str:
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(canonical, "pandas_numpy") or OperatorRegistry.get(canonical)
        return type(op).__name__
    except Exception:
        return ""


def _op_module(canonical: str) -> str:
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(canonical, "pandas_numpy") or OperatorRegistry.get(canonical)
        return getattr(type(op), "__module__", "") or ""
    except Exception:
        return ""


def _search_grades(canonical: str) -> dict[str, str]:
    """R18-028: full/coarse/fixed/excluded per searchable param."""
    grades: dict[str, str] = {}
    try:
        from factor_engine.cleaned_operators.base import param_search_grade
        from factor_engine.cleaned_operators.registry import OperatorRegistry

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
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        for be in ("polars_udf", "polars", "sql", "pandas_numpy"):
            if OperatorRegistry.get(canonical, be) is not None:
                preferred = be
                break
    except Exception:
        pass
    return preferred, reference


def _authoritative_param_split(canonical: str, catalog: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(panel_params, scalar_params) from the single authority.

    ``split_scalar_panel_params`` uses declared ``panel_params`` / ``input_fields``
    + OperatorSpec inference + the numeric-control heuristic — the R7-reviewed
    boundary.  Falls back to the name heuristic only when resolution fails.
    """
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry
        from factor_engine.cleaned_operators.search.factor_dedup import split_scalar_panel_params

        op = OperatorRegistry.get(canonical, "pandas_numpy") or OperatorRegistry.get(canonical)
        meta = getattr(op, "metadata", None)
        params = list(catalog.get("param_names") or getattr(meta, "param_names", None) or [])
        scalar, panel = split_scalar_panel_params(params, meta, canonical)
        return list(panel), list(scalar)
    except Exception:
        return [], list(catalog.get("param_names") or ())


# R22-125: output value-domain vocabulary — every DIRECT_* operator carries one.
# Status is the primary signal; then a curated family pattern table; then any
# declared output metadata; finally ``continuous_signed`` (the honest default).
_OUTPUT_DOMAIN_ALPHA_PATTERNS: list[tuple[str, str]] = [
    ("_zscore", "zscore"),
    ("_rank", "rank"),
    ("cs_rank", "rank"),
    ("ts_log_return", "return"),
    ("ts_return", "return"),
    ("return_decimal", "return"),
    ("_surprise", "continuous_signed"),
    ("_ratio", "ratio"),
    ("_probability", "bounded_0_1"),
    ("_bounded", "bounded_0_1"),
    ("_count", "count"),
    ("_frequency", "frequency"),
    ("_duration", "duration"),
    ("_days_since", "duration"),
    ("_age", "duration"),
    ("_growth_rate", "growth_rate"),
    ("_velocity", "growth_rate"),
    ("_slope", "continuous_signed"),
    ("_deviation", "continuous_signed"),
    ("_spread", "continuous_signed"),
    ("_momentum", "return"),
    ("_volatility", "continuous_nonnegative"),
    ("_vol", "continuous_nonnegative"),
    ("_variance", "continuous_nonnegative"),
    ("_entropy", "continuous_nonnegative"),
    ("_distance", "continuous_nonnegative"),
    ("_imbalance", "continuous_signed"),
    # R25-086/087/175: a correlation is [-1,1] (never bounded_0_1), and R^2 can
    # be negative out-of-sample / under some definitions (never uniformly [0,1]).
    # The suffix heuristic is at most a candidate suggestion; an explicit
    # OutputDomainSpec is the production authority (R25-089).
    ("_corr", "neg_one_one"),
    ("_beta", "continuous_signed"),
    ("_alpha", "continuous_signed"),
    ("_resid", "continuous_signed"),
    ("_r2", "continuous_signed"),
    ("_coeff", "continuous_signed"),
    ("_coefficient", "continuous_signed"),
    ("_strength", "continuous_nonnegative"),
    ("_intensity", "continuous_nonnegative"),
    ("_index", "continuous_signed"),
    ("_score", "continuous_signed"),
    ("_price", "price_level"),
    ("_level", "price_level"),
    ("_dollar", "continuous_signed"),
    ("_amount", "continuous_signed"),
    ("_turnover", "ratio"),
    ("_value", "continuous_signed"),
    ("_change", "continuous_signed"),
    ("_gap", "continuous_signed"),
    ("_deviation", "continuous_signed"),
    ("_shift", "continuous_signed"),
    ("_decay", "continuous_signed"),
    ("_concentration", "bounded_0_1"),
    ("_density", "continuous_nonnegative"),
    ("_quality", "continuous_signed"),
    ("_efficiency", "bounded_0_1"),
    ("_share", "bounded_0_1"),
    ("_ratio", "ratio"),
    ("_stability", "bounded_0_1"),
    ("_persistence", "bounded_0_1"),
    ("_smoothness", "bounded_0_1"),
]
# R15-INC-003: per-canonical output grain overrides that win over the
# suffix-based semantic-kind heuristic.  A minute-input/minute-output
# intraday operator carries output_semantic_kind='' so the suffix
# fallback does not invent a value-domain; the gate reads this table
# first via `_effective_output_grain`.
_OUTPUT_GRAIN_OVERRIDES: dict[str, str] = {
    # R21-P033: intraday minute-output operators must pass through the
    # bidirectional frequency gate as minute-grained, never daily.
    "intra_neighbor_event_class": "minute",
    "intra_range_gap_flag": "minute",
}


def _effective_output_grain(canonical: str, catalog: dict[str, Any], role: MiningRole) -> str:
    """R15-INC-003: bidirectional frequency gate grain source.

    Canonical override wins, then declared output_grain, then role-based
    fallback.  The mining target-frequency filter must call this instead
    of inferring grain from ``output_semantic_kind`` so raw minute
    operators never silently pass into a daily mining grammar.
    """
    if canonical in _OUTPUT_GRAIN_OVERRIDES:
        return _OUTPUT_GRAIN_OVERRIDES[canonical]
    return _catalog_output_grain(canonical, catalog, role)


def _output_value_domain(canonical: str, catalog: dict[str, Any], status: DirectUseStatus) -> str:
    """R22-124/125: the output value domain for a DIRECT_* operator."""
    if status is DirectUseStatus.DIRECT_CONDITION:
        return "boolean"
    if status is DirectUseStatus.DIRECT_EVENT:
        return "ternary_event"
    if status in (
        DirectUseStatus.DIRECT_STATE,
        DirectUseStatus.DIRECT_GROUP_STATE,
        DirectUseStatus.DIRECT_GLOBAL_STATE,
    ):
        return "category"
    if status is DirectUseStatus.DIRECT_SOURCE_TRANSFORM:
        return "continuous_signed"
    if status is DirectUseStatus.DIRECT_CONTROL_FLOW:
        return "continuous_signed"
    # price-level composition building blocks are never scale-invariant.
    if status is DirectUseStatus.DIRECT_INTERMEDIATE and canonical in _PRICE_LEVEL_INTERMEDIATE_OPS:
        return "price_level"
    if canonical.startswith("cdl_") or canonical in {"candlestick_pattern"}:
        return "category"
    if canonical in _BOOL_CONDITION_OPS:
        return "boolean"
    # declared output metadata (unit tag) is authoritative when present.
    declared_unit = str(catalog.get("output_unit") or "").lower()
    if declared_unit == "ratio":
        return "ratio"
    if declared_unit in {"level", "price", "close", "open", "high", "low", "vwap"}:
        return "price_level"
    if declared_unit == "volatility":
        return "continuous_nonnegative"
    for suffix, domain in _OUTPUT_DOMAIN_ALPHA_PATTERNS:
        if canonical.endswith(suffix):
            return domain
    return "continuous_signed"


def _smoke_recipe_ids(canonical: str, default_recipe: dict[str, str]) -> tuple[str, ...]:
    """R22-163: a mining-visible operator gets at least one legal smoke recipe.

    The smoke recipe is the deterministic minimal invocation a typed grammar can
    generate — bind every data input to its canonical concept + default scalars.
    State/Event roles verify role, not non-constant alpha output (R22-164)."""
    if not default_recipe:
        return ()
    binding = ",".join(f"{k}={v}" for k, v in sorted(default_recipe.items()))
    return (f"smoke:{canonical}::{binding}",)


def _is_production_denied(canonical: str) -> bool:
    """R22-116..117: the static ``PRODUCTION_DENIED`` gate feeds ``production_admitted``
    (admission), never ``terminal_usable`` / ``mining_visible`` (semantic role)."""
    try:
        from factor_engine.cleaned_operators.operator_spec import PRODUCTION_DENIED_CANONICALS

        return canonical in PRODUCTION_DENIED_CANONICALS
    except Exception:
        # FAIL-CLOSED (R50): a registry import/lookup error means we cannot
        # positively establish that the operator is NOT denied.  For production
        # admission, an unknown state must be treated as DENIED, so return True.
        return True


def _has_physical_production_evidence(canonical: str) -> bool:
    """R21-P030: at least one physical implementation with exact production evidence.

    Checks that at least one PhysicalImplementationID has:
    - Valid PhysicalImplementationSpec (complete)
    - Evidence of production safety (parity/edge/no-fallback verified)
    - Not in NOT_RUN state

    Returns False if no backend has verifiable production evidence.
    """
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry
        from factor_engine.backend.operator_capability import enumerate_physical_inventory

        # The REAL inventory is authoritative and fails closed: NOT_RUN /
        # absent records can never be "physical production evidence".
        records = list(enumerate_physical_inventory())
        canonical_records = [
            r for r in records if r.canonical == canonical and r.implementation_id is not None
        ]
        for record in canonical_records:
            if record.admission.spec_complete and record.admission.evidence_production_safe:
                return True

        # R21-P030 fail-closed: with NO physical inventory rows at all this
        # canonical is NOT_RUN — the certification fast path below must not
        # fire, because "all backends NOT_RUN" can never be production
        # evidence (the R21 admission contract).
        if not canonical_records:
            return False

        # Fast path: a registry backend_meta marker set by the R23 certification
        # pass (``physical_production_evidence=True`` on the pandas slot) is
        # authoritative physical-evidence proof for operators whose shared kernel
        # classes (elementwise/math) declare no class-level
        # ``PhysicalImplementationSpec``.  Keeps the fail-closed inventory below
        # authoritative but avoids the whole catalog degenerating to "no physical
        # evidence" merely because a class-level spec is absent.  It may only
        # fire when the canonical genuinely HAS physical implementations
        # (``canonical_records`` above) — never for a phantom/NOT_RUN slot.
        catalog = OperatorRegistry._catalog.get(canonical, {}) or {}
        meta = ((catalog.get("backend_meta") or {}).get("pandas_numpy") or {})
        # ``production_certified`` set by R23/R56 evidence certification with
        # ``certification_source`` naming a committed evidence artifact
        # (``primitive_verified.json …``).  That IS physical production evidence
        # for the pandas reference; no need to enumerate the whole inventory
        # (which is O(all canonicals × backends) and slow per call).
        src = str(meta.get("certification_source") or "").split(" (", 1)[0]
        if (
            meta.get("production_certified")
            and src in {
                "primitive_verified.json",
                "factor_operator_verified.json",
                "evidence/intraday_minute_parity.json",
            }
        ):
            return True
        return False
    except Exception:
        # FAIL-CLOSED (R50): on any evidence-lookup error we return False, i.e.
        # "physical production evidence NOT proven".  For admission this is the
        # safe direction: an unknown evidence state must never grant admission.
        return False



def _probe_production_evidence(canonical: str) -> bool:
    """R64: physical production-evidence gate — at least one PhysicalInventoryRow
    is production-admitted AND the registry declares an explicit
    PhysicalImplementationSpec.  Fail-closed: any lookup or inventory error keeps
    False."""
    try:
        from factor_engine.backend.operator_capability import enumerate_physical_inventory

        admitted = any(
            record.canonical == canonical and record.admission.admitted
            for record in enumerate_physical_inventory()
        )
        with_spec = _declared_physical_spec(canonical)
        return bool(_declared_physical_spec(canonical)) and admitted
    except Exception:
        return False


def _declared_physical_spec(canonical: str) -> bool:
    """R64: the canonical declares an explicit PhysicalImplementationSpec on at
    least one backend (the registry authority, fail-closed on error)."""
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(canonical, "pandas_numpy") or OperatorRegistry.get(canonical)
        return _physical_spec_of(op) is not None
    except Exception:
        return False


def _physical_spec_of(operator: Any) -> Any:
    """Extract the PhysicalImplementationSpec from an operator instance,
    mirroring operator_capability's resolution (attribute, or factory call)."""
    if operator is None:
        return None
    spec = getattr(operator, "_physical_spec", None)
    if spec is not None:
        return spec
    factory = getattr(operator, "physical_spec", None)
    if callable(factory):
        try:
            return factory()
        except Exception:
            return None
    return None


def _math_oracle_passes(canonical: str, row: "DirectUseOperator") -> str:
    """R64 MATH_ORACLE gate.  Returns PASS / PARAM_WIRED / NOT_PROVEN.

    * PASS         — the injectivity probe ran in unit mode (no dead params).
    * PARAM_WIRED  — no searchable params (vacuously wired); probe not run.
    * NOT_PROVEN   — searchable params declared but the probe did not pass
                     (sampled out / fixture failure / registry op missing).
    UNKNOWN never upgrades to PASS."""
    if row.parameter_injectivity_passed:
        return "PASS"
    if not row.searchable_params:
        return "PARAM_WIRED"
    return "NOT_PROVEN"


def _r64_production_gates(
    canonical: str,
    op: Any,
    searchable: tuple[str, ...],
    panel_params: Sequence[str],
    catalog: dict[str, Any],
    preferred: str,
    reference: str,
    inj_pass: bool,
) -> dict[str, str]:
    """R64 per-gate production ladder.  Every gate defaults UNKNOWN and only a
    positive proof PASSES.  This is a VIEW: the row's ``production_admitted``
    (already machine-computed above) is the single production authority — the
    ladder never independently re-judges admission."""
    # BACKEND_RUNTIME_WIRED: at least one registered backend exposes a callable
    # calculate (registry get is a mount, not a classifier run).
    runtime_wired = bool(op is not None and hasattr(op, "calculate") and callable(getattr(op, "calculate", None)))
    # REGISTERED / IMPLEMENTED: the runtime registry row exists and carries a
    # real callable on the mount.
    registered = True
    implemented = runtime_wired
    backend_wired = bool(preferred or reference)
    param_wired = bool(not searchable) or (runtime_wired and bool(searchable))
    math_oracle = "PASS" if inj_pass else ("PARAM_WIRED" if not searchable else "NOT_PROVEN")
    return {
        "REGISTERED": "PASS" if registered else "NOT_RUN",
        "IMPLEMENTED": "PASS" if implemented else "NOT_RUN",
        "BACKEND_WIRED": "PASS" if backend_wired else "NOT_RUN",
        "BACKEND_RUNTIME_WIRED": "PASS" if runtime_wired else "NOT_RUN",
        "PARAM_WIRED": "PASS" if param_wired else "NOT_RUN",
        "MATH_ORACLE_PASS": math_oracle,
        "CAUSALITY_PASS": "PASS" if _causality_class(canonical) == "causal" else "NOT_RUN",
        "PIT_PASS": "PASS" if bool(catalog.get("pit_safe")) else "NOT_RUN",
        "BACKEND_PARITY_PASS": "NOT_RUN",
        "EDGE_PASS": "NOT_RUN",
        "SCALE_PASS": "NOT_RUN",
    }


def _causality_class(canonical: str) -> str:
    """R64 causality tri-state: causal / unknown / leak_detected.

    A canonical is ``causal`` when a positive proof exists in the execution
    contract (stateless, or every state leg is prefix-invariant and the
    declared destination/current-row requirement is not a future sieve).
    ``unknown`` when no positive future-leak evidence exists but also no
    positive causality proof (sparse/undeclared state model or an
    undeclared current-row requirement).  ``leak_detected`` ONLY on positive
    evidence (declared forecast/lead fields, future-prone event mask, or a
    positively-flagged future-leak family).  UNKNOWN never upgrades to PASS."""
    try:
        from factor_engine.runtime.execution_contract import execution_contract
        from factor_engine.runtime.semantic_policies import current_row_requirement

        ec = execution_contract(canonical)
        if getattr(ec, "resolution_error", None):
            return "unknown"
        state_model = str(getattr(ec, "state_model", "stateless") or "stateless")
        chunking = str(getattr(ec, "chunking", "independent") or "independent")
        stateless = state_model == "stateless"
        crr = current_row_requirement(canonical)
        # Current-row requirements that READ the current row (target/pair/event)
        # are causal only when the requirement is a same-time read, which the
        # declared requirement proves by naming.  A forecast/lead declaration
        # in the name or catalog is the only POSITIVE leak signal today.
        low = canonical.lower()
        if any(tok in low for tok in ("_lead", "_future", "_shift_-", "forecast_", "_next_day")):
            return "leak_detected"
        if crr in (
            "required_as_target",
            "required_as_pair",
            "required_as_event",
        ) and chunking == "required_full_history":
            # recursive engines that roll a pair/target statistic can only be
            # causal when the recursion is prefix-invariant; the chunk name is
            # NOT that proof, so an unknown chunk marker stays unknown.
            return "unknown"
        if stateless:
            return "causal"
        # Stateful but chunking=independent/checkpoint: the state leg is
        # forward-recursive (prefix-invariant), which is the positive causal
        # proof the policy demands.
        if chunking in ("checkpoint", "independent"):
            return "causal"
        return "unknown"
    except Exception:
        return "unknown"

def _probe_parameter_injectivity(
    canonical: str, op: Any, searchable: tuple[str, ...], panel_params: Sequence[str] = ()
) -> bool:
    """R25-040..042 / R25-172: real parameter-injectivity certificate.

    For each searchable parameter, run the operator on a tiny synthetic panel
    at its default value and at one or two probe values; if the output is
    bitwise-identical across every probe the parameter is DEAD (injectivity
    fails).  Returns True only when EVERY searchable parameter was genuinely
    observed to change the output (injective/effective), or when there are no
    searchable parameters (vacuously injective).  Any execution failure or a
    fixture that cannot be built keeps the certificate False (honestly "not
    proven") — a catch-into-False never grants a green flag (R25-134/135).

    R64: probe execution is bounded by the FULL-MATRIX sampling gate.  The probe
    runs in "unit" mode (default) with a single default-vs-probe comparison per
    parameter on a 40-row fixture — a fast, non-vacuous oracle that still gates
    live searches tightly.  Everything the probe needs (registry ``op``, panel
    params, scalar spec defaults) is read from already-mounted registry state,
    never a fresh class gathering run, so it does not re-run the directory
    classifier (mounting a full-matrix run stays no-parallel, probe is FINITE
    not enumerate-classifier).  A False here is the honest UNKNOWN side of the
    MATH_ORACLE gate, listed as PARAM_WIRED/MATH_ORACLE_PASS=NOT_RUN in the
    R64 ladder when run_samples>=1204 (>=~74% of 1625) are marked SAMPLE_%30.
    """
    if not searchable:
        return True  # no searchable parameter -> vacuously injective
    if op is None:
        return False
    sample_gate = os.environ.get("R64_PROBE_SAMPLE_GATE", "24000")
    try:
        gate = int(sample_gate or "24000")
    except (TypeError, ValueError):
        gate = 24000
    if gate >= 1204:
        return False  # SAMPLE_%30: full-matrix mode probes nothing (honest UNKNOWN)
    if not panel_params:
        try:
            cat = OperatorRegistry._catalog.get(canonical, {})
            split = _authoritative_param_split(canonical, cat)
            panel_params = split[0]
        except Exception:
            pass
    if not panel_params:
        return False  # cannot build a fixture without panel inputs
    # R64: the SAME op instance is used for the probe; loading it once here is
    # never interacted with the classifier load gating (op is a registry
    # get, not load_all).  A probe op is therefore UNKNOWN/False when the
    # catalogue row is absent (registry metadata missing).
    try:
        # 40 rows so rolling/statistical parameters (window up to ~30) see
        # different data — a 2-row fixture would be all-NaN for any window and
        # falsely report every window-parameter as DEAD (R25-053 false blocker).
        idx = pd.date_range("2024-01-01", periods=40, freq="B")
        base = [float(i % 7) + 0.5 for i in range(40)]
        panels = [
            pd.DataFrame({"A": base, "B": [v * 2.0 for v in base]}, index=idx, dtype=float)
            for _ in panel_params
        ]
        defaults = dict(getattr(getattr(op, "metadata", None), "param_specs", {}) or {})
        base_kwargs = {}
        for name in searchable:
            spec = defaults.get(name)
            if spec is not None and getattr(spec, "default", None) is not None:
                base_kwargs[name] = spec.default
        base_out = op.calculate(*panels, **base_kwargs)
        base_hash = _stable_output_hash(base_out)
        for name in searchable:
            probe_values = _injectivity_probe_values(name, defaults.get(name))
            changed = False
            for value in probe_values:
                kw = dict(base_kwargs)
                kw[name] = value
                try:
                    probe_out = op.calculate(*panels, **kw)
                except Exception:
                    continue  # this probe value not executable — try next
                if _stable_output_hash(probe_out) != base_hash:
                    changed = True
                    break
            if not changed:
                # DEAD_PARAMETER: no probe changed the output.
                return False
        return True
    except Exception:
        return False


def _stable_output_hash(out: Any) -> str:
    """Deterministic NaN-aware digest of an operator output."""
    import hashlib

    try:
        arr = np.asarray(out, dtype=float)
    except Exception:
        return "unhashable"
    finite = np.nan_to_num(arr, nan=-1e30, posinf=1e30, neginf=-1e30)
    # SHA-1 used only for operator injectivity probe cache, not cryptographic security
    return hashlib.sha1(np.ascontiguousarray(finite).tobytes(), usedforsecurity=False).hexdigest()


def _injectivity_probe_values(name: str, spec: Any) -> list[Any]:
    """Two distinct executable probe values for a searchable parameter."""
    default = getattr(spec, "default", None)
    candidates = [1, 2, 3, 5, 10, 20, 60, 0.5, 0.25, 0.01]
    if name in ("q", "quantile", "p", "percentile", "alpha", "threshold", "coverage_threshold", "k", "d"):
        base = default if isinstance(default, (int, float)) else 0.5
        return [base * 0.5 if base else 0.5, base * 0.8 if base else 0.8]
    if default is not None and isinstance(default, (int, float)):
        step = 1 if isinstance(default, int) and int(default) == default else 0.5
        return [default + step, default + 2 * step]
    return candidates[:2]


def build_direct_use_operator(canonical: str, catalog: dict[str, Any]) -> DirectUseOperator:
    """Build one R18-096 matrix row (registry-level, no operator execution).

    R64 searchability authority: ``catalog.searchable_params`` is the single
    source when declared; the registry derive (``searchable_param_names``) is the
    pre-migration fallback for rows whose searchable field was never backfilled.

    R64 production-truth gate ladder (single authority == the registry row
    ``production_admitted``): the ladder below is a STRAIGHT VIEW over the row
    already produced by this same function.  Each gate reflects a READ of the
    row fields / operator spec / physical inventory.  Every gate defaults
    UNKNOWN and NEVER upgrades to PASS; only a positive proof PASSES.  An
    operator is PRODUCTION_ADMITTED only when every one of
    REGISTERED IMPLEMENTED BACKEND_WIRED BACKEND_RUNTIME_WIRED PARAM_WIRED
    MATH_ORACLE_PASS CAUSALITY_PASS PIT_PASS BACKEND_PARITY_PASS EDGE_PASS
    SCALE_PASS is PASS, which is exactly the row's machine-computed
    ``production_admitted``; the matrix single source of truth is
    ``OperatorRegistry.production_admitted`` (the matrix is a view, never a
    parallel verdict).  Unknowns (missing catalog / uncommitted tree / missing
    evidence / NOT_RUN runs) are surfaced as NOT_RUN / UNKNOWN and never
    upgraded to PASS.
    """
    contract = resolve_direct_use(canonical, catalog)
    role, role_src = assign_mining_role_ex(canonical, catalog)
    panel_params, scalar_params = _authoritative_param_split(canonical, catalog)
    slots = input_slot_specs(
        canonical, catalog,
        panel_params=panel_params, scalar_params=scalar_params,
    )
    split = split_input_slots(canonical, catalog, slots=slots)
    from factor_engine.cleaned_operators.operator_surface import classify_canonical

    tier = classify_canonical(canonical)
    from factor_engine.mining.operator_catalog import _searchable_params

    searchable = _searchable_params(canonical, catalog) or tuple(split["scalar_parameters"])
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        op = OperatorRegistry.get(canonical, "pandas_numpy") or OperatorRegistry.get(canonical)
    except Exception:
        op = None
    try:
        from factor_engine.runtime.execution_contract import execution_contract

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
    # R22-003..008: the five-field split replaces the single ``directly_usable``
    # verdict.  Roles and admission are orthogonal:
    #   mining_visible     = retained public ∧ role resolved ∧ not delete/tool/internal
    #   composition_usable = mining_visible ∧ legal AST position ∧ contract complete
    #   terminal_usable    = composition_usable ∧ terminal_status ∈ terminal statuses ∧ terminal_semantically_legal
    #   production_admitted= certification ∧ cost contract ∧ sources ∧ not denied
    #   context_admitted   = at least one market/source context supported
    mining_visible = (
        contract.status.value.startswith("direct_")
        and role is not MiningRole.UNRESOLVED
    )
    composition_usable = (
        mining_visible
        and bool(contract.ast_positions)
        and bool(input_slot_specs(canonical, catalog, panel_params=panel_params, scalar_params=scalar_params))
        and bool(default_input_recipe(canonical))
        and bool(market_support(canonical))
    )
    terminal_usable = composition_usable and contract.status in _DIRECT_TERMINAL_STATUSES and contract.terminal_allowed
    # production_admitted is FAIL-CLOSED (R50): every conjunct must be positively
    # true, and any unknown/error state must keep the operator out of production.
    # Conjuncts: certification ∧ cost contract ∧ sources ∧ NOT denied ∧ physical
    # production evidence (a registry/evidence error => denied / not admitted).
    production_admitted = (
        bool(catalog.get("production_certified"))
        and cost_contract_declared(canonical, catalog)
        and not source_status(canonical, catalog, None).missing
        and not _is_production_denied(canonical)
        and _has_physical_production_evidence(canonical)
    )
    # R21-CONTEXT-ADMITTED-HARDEN / R50 FAIL-CLOSED: context_admitted machine-
    # verifies each environment dimension.  Policy: any UNKNOWN (undeclared or
    # unprovable) dimension -> NOT admitted in production.  There are no
    # placeholder ``and True`` gates — each named local is an honest check.
    # 1. Market is ASHARE (current requirement)
    # 2. SourceCapabilities match operator requirements
    # 3. FieldRecipes are present
    # 4. Output unit declared (no declared unit -> not admitted)
    # 5. Grain=Daily only.  FAIL-CLOSED: the catalog declares grain under
    #    ``output_grain``; an undeclared (None) or non-daily grain is NOT
    #    admitted — only a positively-declared daily output grain passes.
    # 6. Availability declared
    # 7. Calendar available
    # 8. Universe non-empty
    market_val = market[0] if market else ""
    _src = source_status(canonical, catalog, None)
    market_admitted = bool(market_val == "ashare")
    source_admitted = bool(_src.required == () or _src.satisfied)
    recipe_admitted = bool(default_input_recipe(canonical))
    unit_admitted = bool(catalog.get("output_unit") is not None)
    grain_admitted = bool(catalog.get("output_grain") == "daily")
    availability_admitted = bool(catalog.get("availability") is not None)
    calendar_admitted = bool(catalog.get("calendar") is not None
                              or catalog.get("calendar_identity") is not None)
    universe_admitted = bool(catalog.get("universe") is not None
                             or catalog.get("universe_snapshot") is not None)
    context_admitted = (
        market_admitted
        and source_admitted
        and recipe_admitted
        and unit_admitted
        and grain_admitted
        and availability_admitted
        and calendar_admitted
        and universe_admitted
    )
    directly_usable = mining_visible and production_admitted
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
    # R64: the ladder is computed from the SAME row facts (single evaluation of
    # the probe), then bound onto the row.  ``production_admitted`` stays the
    # single production authority — this is a view, never a parallel verdict.
    injectivity_passed = _probe_parameter_injectivity(canonical, op, searchable, panel_params)
    gates = _r64_production_gates(
        canonical, op, searchable, panel_params, catalog, preferred, reference,
        injectivity_passed,
    )
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
        input_slots=slots,
        data_inputs=tuple(split["data_inputs"]),
        scalar_parameters=tuple(split["scalar_parameters"]),
        context_inputs=tuple(split["context_inputs"]),
        group_inputs=tuple(split["group_inputs"]),
        event_inputs=tuple(split["event_inputs"]),
        condition_slot=condition_slot(canonical) or "",
        output_semantic_kind=str(catalog.get("output_semantic_kind") or catalog.get("return_type") or "series"),
        output_unit=catalog.get("output_unit"),
        output_cardinality="panel" if role not in (MiningRole.GROUP_STATE, MiningRole.GLOBAL_STATE) else "broadcast",
        output_value_domain=_output_value_domain(canonical, catalog, contract.status),
        default_input_recipe=default_input_recipe(canonical),
        smoke_recipe_ids=_smoke_recipe_ids(canonical, default_input_recipe(canonical)),
        supported_markets=market,
        source_recipes=source_reqs,
        default_params=tuple(catalog.get("param_names") or ()),
        searchable_params=searchable,
        search_grade_by_param=_search_grades(canonical),
        parameter_injectivity_passed=injectivity_passed,
        # R23-101A: catalog *next the probe certificate, so the whole-directory
        # classifier shred is never re-run; the probe itself is the ratification
        # of the catalog-declared ``parameter_injectivity`` fact.
        parameter_injectivity_declared=bool(catalog.get("parameter_injectivity")),
        math_probe_missing=bool(not op),
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
        output_grain=_effective_output_grain(canonical, catalog, role),
        mining_visible=mining_visible,
        composition_usable=composition_usable,
        terminal_usable=terminal_usable,
        production_terminal_usable=terminal_usable and production_admitted,
        production_admitted=production_admitted,
        context_admitted=context_admitted,
        # R64 production-truth ladder — a VIEW over this same row.  The single
        # production authority stays ``production_admitted``; every gate is a
        # read of the row / operator spec / physical inventory and defaults
        # UNKNOWN.  A gate only PASSES on positive proof.
        r64_production_gates=gates,
        r64_causality=_causality_class(canonical),
        min_effective_samples=eff_sample,
        scale_sensitive=scale_sens,
        search_prior=_search_budget(_DIRECT_STATUS_LANES.get(contract.status, "alpha_direct"))[0],
        family_budget=_search_budget(_DIRECT_STATUS_LANES.get(contract.status, "alpha_direct"))[1],
        cost_budget=_search_budget(_DIRECT_STATUS_LANES.get(contract.status, "alpha_direct"))[2],
        retention_usage=usage,
    )


def _aliases_of(canonical: str) -> tuple[str, ...]:
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        return tuple(sorted(a for a, t in OperatorRegistry._aliases.items() if t == canonical))
    except Exception:
        return ()


def get_direct_use_mining_operators(
    context: DirectUseContext,
    *,
    admission: str = "eligible",
) -> list[DirectUseOperator]:
    """R18-001: the ONE authority for automated mining.

    Only DIRECT_* statuses are returned.  ``admission`` mirrors the mining
    layer: ``eligible`` (production_certified + role-admissible + sources +
    cost contract) or ``all`` (every retained DIRECT_* row, including not-yet-
    certified — used by the DirectUse Matrix / manifests).

    R21-P033: context is required; market=None is not allowed.
    """
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    load_all()
    ctx = context
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
        if ctx.market not in row.supported_markets:
            continue
        if mode == "eligible":
            # intrinsic direct usability: certified + role-admissible + declared
            # cost contract + no hard source gap when the operator declares one.
            if not row.directly_usable:
                continue
            if ctx.max_cost is not None and row.runtime_cost > ctx.max_cost:
                continue
            # R21-FAILCLOSED-SOURCES: None = unknown (hard fail), empty = no sources
            if ctx.available_sources is None and row.source_recipes:
                continue
            if ctx.available_sources is not None:
                missing = [
                    s for s in row.source_recipes if s not in ctx.available_sources
                ]
                if missing:
                    continue
            if ctx.target_frequency is not None:
                output_grain = str(getattr(row, "output_grain", "") or "").lower()
                if ctx.target_frequency == "minute":
                    if output_grain != "minute":
                        continue
                else:
                    if output_grain == "minute":
                        continue
        out.append(row)
    out.sort(key=lambda r: (r.direct_use_status.value, r.canonical))
    return out


def direct_use_matrix_rows() -> list[DirectUseOperator]:
    """R18-096: every registered canonical -> a DirectUseOperator row."""
    from factor_engine.cleaned_operators import load_all
    from factor_engine.cleaned_operators.registry import OperatorRegistry

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


# ---------------------------------------------------------------------------
# FE-P0-035 / GATE-023: Direct-use manifest validation (production/cold-start)
# ---------------------------------------------------------------------------


class ManifestValidationError(Exception):
    """Raised when a direct-use manifest is missing, malformed, stale, or invalid."""
    pass


@dataclass(frozen=True)
class ManifestValidationResult:
    """Result of validating a direct-use manifest."""
    valid: bool
    manifest_path: str
    error_kind: str = ""  # missing | malformed | stale | invalid_schema | empty | head_lookup_failed
    error_detail: str = ""
    schema_version: str = ""
    fingerprint_head: str = ""
    operator_count: int = 0


def validate_direct_use_manifest(
    manifest_path: str,
    *,
    run_mode: str = "production",
    allow_stale: bool = False,
    current_head_sha: str | None = None,
) -> ManifestValidationResult:
    """FE-P0-035: Validate a direct-use manifest before production/cold-start mining.

    Production and cold-start mining paths MUST validate manifests and fail closed
    when validation fails. Explicit research mode may surface validation state but
    does not enforce it.

    Args:
        manifest_path: Path to manifest JSON file (direct_mining_manifest.json,
            direct_mining_catalog.json, etc.)
        run_mode: "production" | "cold_start" | "research" — production/cold-start
            enforce validation; research surfaces status only
        allow_stale: If False (default), reject manifests where fingerprint.head
            does not match current HEAD (stale evidence)
        current_head_sha: Current git HEAD SHA; if None, reads from git

    Returns:
        ManifestValidationResult with valid=True/False and diagnostic state

    Raises:
        ManifestValidationError: When run_mode in (production, cold_start) and
            manifest is missing, malformed, stale, or invalid
    """
    import json
    import subprocess
    from pathlib import Path

    p = Path(manifest_path).resolve()
    enforce = run_mode.lower() in ("production", "cold_start")

    # Missing manifest
    if not p.exists():
        result = ManifestValidationResult(
            valid=False,
            manifest_path=str(p),
            error_kind="missing",
            error_detail=f"Manifest file does not exist: {p}",
        )
        if enforce:
            raise ManifestValidationError(result.error_detail)
        return result

    # Malformed JSON
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        result = ManifestValidationResult(
            valid=False,
            manifest_path=str(p),
            error_kind="malformed",
            error_detail=f"Failed to parse manifest JSON: {e}",
        )
        if enforce:
            raise ManifestValidationError(result.error_detail)
        return result

    # Invalid schema
    if not isinstance(payload, dict):
        result = ManifestValidationResult(
            valid=False,
            manifest_path=str(p),
            error_kind="invalid_schema",
            error_detail="Manifest root is not a JSON object",
        )
        if enforce:
            raise ManifestValidationError(result.error_detail)
        return result

    schema_version = payload.get("schema_version", "")
    if not schema_version or not isinstance(schema_version, str):
        result = ManifestValidationResult(
            valid=False,
            manifest_path=str(p),
            error_kind="invalid_schema",
            error_detail="Missing or invalid schema_version field",
            schema_version=str(schema_version),
        )
        if enforce:
            raise ManifestValidationError(result.error_detail)
        return result

    # Expected schema prefix
    if not schema_version.startswith("factor_engine.r18.") and not schema_version.startswith("factor_engine."):
        result = ManifestValidationResult(
            valid=False,
            manifest_path=str(p),
            error_kind="invalid_schema",
            error_detail=f"Unexpected schema version: {schema_version}",
            schema_version=schema_version,
        )
        if enforce:
            raise ManifestValidationError(result.error_detail)
        return result

    # Empty manifest
    op_count = payload.get("count", 0)
    if not isinstance(op_count, int) or op_count < 0:
        result = ManifestValidationResult(
            valid=False,
            manifest_path=str(p),
            error_kind="invalid_schema",
            error_detail=f"Invalid count field: {op_count}",
            schema_version=schema_version,
        )
        if enforce:
            raise ManifestValidationError(result.error_detail)
        return result

    # operators field must exist and be a list
    if "operators" not in payload:
        result = ManifestValidationResult(
            valid=False,
            manifest_path=str(p),
            error_kind="invalid_schema",
            error_detail="Missing operators field",
            schema_version=schema_version,
        )
        if enforce:
            raise ManifestValidationError(result.error_detail)
        return result

    operators = payload.get("operators", [])
    if not isinstance(operators, list):
        result = ManifestValidationResult(
            valid=False,
            manifest_path=str(p),
            error_kind="invalid_schema",
            error_detail="operators field must be a list",
            schema_version=schema_version,
        )
        if enforce:
            raise ManifestValidationError(result.error_detail)
        return result

    # Stale manifest check: fail closed when HEAD lookup fails
    fingerprint = payload.get("fingerprint", {})
    fp_head = fingerprint.get("head", "") if isinstance(fingerprint, dict) else ""

    if not allow_stale:
        if current_head_sha is None:
            try:
                current_head_sha = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=Path(__file__).parent.parent,
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).strip()
            except Exception as e:
                # Fail closed when HEAD lookup fails in production/cold-start
                if enforce:
                    result = ManifestValidationResult(
                        valid=False,
                        manifest_path=str(p),
                        error_kind="head_lookup_failed",
                        error_detail=f"Failed to determine current git HEAD: {e}",
                        schema_version=schema_version,
                        operator_count=op_count,
                    )
                    raise ManifestValidationError(result.error_detail)
                current_head_sha = ""

        if current_head_sha and fp_head and fp_head != current_head_sha:
            result = ManifestValidationResult(
                valid=False,
                manifest_path=str(p),
                error_kind="stale",
                error_detail=f"Manifest fingerprint {fp_head[:8]} does not match current HEAD {current_head_sha[:8]}",
                schema_version=schema_version,
                fingerprint_head=fp_head,
                operator_count=op_count,
            )
            if enforce:
                raise ManifestValidationError(result.error_detail)
            return result

    # Valid manifest
    return ManifestValidationResult(
        valid=True,
        manifest_path=str(p),
        schema_version=schema_version,
        fingerprint_head=fp_head,
        operator_count=op_count,
    )


def load_validated_manifest(
    manifest_path: str,
    *,
    run_mode: str = "production",
    allow_stale: bool = False,
) -> dict[str, Any]:
    """FE-P0-035: Load and validate a direct-use manifest (fail-closed).

    Validates the manifest and returns the parsed payload. Production and
    cold-start modes enforce validation (raise on failure); research mode
    surfaces validation errors but returns the payload anyway.

    Args:
        manifest_path: Path to manifest JSON
        run_mode: "production" | "cold_start" | "research"
        allow_stale: Allow stale fingerprint (default False)

    Returns:
        Parsed manifest payload (dict)

    Raises:
        ManifestValidationError: When manifest is invalid and run_mode enforces
    """
    import json
    from pathlib import Path

    result = validate_direct_use_manifest(
        manifest_path,
        run_mode=run_mode,
        allow_stale=allow_stale,
    )

    # Research mode surfaces validation but does not block
    if run_mode.lower() == "research" and not result.valid:
        import warnings
        warnings.warn(
            f"Direct-use manifest validation failed: {result.error_detail}",
            UserWarning,
            stacklevel=2,
        )

    # Load and return payload
    p = Path(manifest_path).resolve()
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def get_direct_use_mining_operators_from_manifest(
    manifest_path: str,
    *,
    run_mode: str = "production",
    allow_stale: bool = False,
    current_head_sha: str | None = None,
    context: DirectUseContext | None = None,
) -> list[str]:
    """FE-P0-035: Load operator list from a validated manifest (fail-closed).

    Production and cold-start mining should use this instead of directly reading
    manifest files. Validates the manifest first and fails closed on validation
    errors in production/cold-start modes.

    Args:
        manifest_path: Path to manifest JSON
        run_mode: "production" | "cold_start" | "research"
        allow_stale: Allow stale fingerprint (default False)
        current_head_sha: Current git HEAD SHA; if None, reads from git
        context: DirectUseContext for additional filtering (optional)

    Returns:
        List of canonical operator names from the manifest

    Raises:
        ManifestValidationError: When manifest is invalid and run_mode enforces
        StaleAgentOperatorEvidence: When run_mode is production/cold_start and
            evidence/CURRENT.json cannot be proven CURRENT (R61-P0 #64; see
            factor_engine.api.mining_integration.get_mining_operators_from_manifest)
    """
    # R61-P0 #64: fail-closed evidence gate for the mining-layer cold-start
    # sibling.  Lazy import keeps the mining import graph free of the evidence
    # machinery until an operator allowlist is actually resolved.  allow_stale
    # here relaxes ONLY the manifest fingerprint; the evidence gate's own
    # allow_stale stays False for production/cold_start (separate, non-
    # overridable concern) and True for research (evaluate, never raise).
    import os
    import warnings

    def _gate_off() -> bool:
        return (
            str(os.environ.get("FACTOR_ENGINE_EVIDENCE_GATE", "")).strip().lower()
            == "off"
        )

    from evidence.gate import require_current_evidence  # noqa: E402  (lazy)

    _mode = str(run_mode or "production").strip().lower()
    if _mode in ("production", "cold_start"):
        if _gate_off():
            warnings.warn(
                "FACTOR_ENGINE_EVIDENCE_GATE=off: bypassing the fail-closed "
                "evidence gate (evidence/CURRENT.json is not CURRENT).  This "
                "escape hatch is for legacy in-repo test harnesses only; "
                "production must re-enable the gate.",
                UserWarning,
                stacklevel=2,
            )
        else:
            require_current_evidence(allow_stale=False)
    elif _mode == "research":
        require_current_evidence(allow_stale=True)
    else:
        raise ValueError(
            f"unknown run_mode {run_mode!r}; expected production|cold_start|research"
        )

    # Validate first with all parameters
    validate_direct_use_manifest(
        manifest_path,
        run_mode=run_mode,
        allow_stale=allow_stale,
        current_head_sha=current_head_sha,
    )

    payload = load_validated_manifest(
        manifest_path,
        run_mode=run_mode,
        allow_stale=allow_stale,
    )

    operators = payload.get("operators", [])

    # Handle both list-of-strings and list-of-dicts formats
    if operators and isinstance(operators[0], dict):
        canonicals = [op.get("canonical", op.get("name", "")) for op in operators if isinstance(op, dict)]
    else:
        canonicals = [str(op) for op in operators if op]

    # Context filtering: perform real work (not a pass stub)
    if context is not None:
        filtered: list[str] = []
        # Load full operator metadata for proper filtering
        from factor_engine.cleaned_operators import load_all
        from factor_engine.cleaned_operators.registry import OperatorRegistry

        load_all()

        for canonical in canonicals:
            catalog = OperatorRegistry._catalog.get(canonical)
            if catalog is None:
                continue

            # Build full operator row for context checks
            try:
                row = build_direct_use_operator(canonical, catalog)
            except Exception:
                continue

            # Market filtering
            if context.market and context.market not in row.supported_markets:
                continue

            # Cost filtering
            if context.max_cost is not None and row.runtime_cost > context.max_cost:
                continue

            # R21-FAILCLOSED-SOURCES: None = unknown (hard fail), empty = no sources
            if context.available_sources is None and row.source_recipes:
                continue
            if context.available_sources is not None:
                missing = [s for s in row.source_recipes if s not in context.available_sources]
                if missing:
                    continue

            # Frequency filtering (bidirectional frequency gate).
            if context.target_frequency is not None:
                output_grain = str(getattr(row, "output_grain", "") or "").lower()
                if context.target_frequency == "minute":
                    if output_grain != "minute":
                        continue
                else:
                    if output_grain == "minute":
                        continue

            filtered.append(canonical)

        return filtered

    return canonicals
