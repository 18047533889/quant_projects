# -*- coding: utf-8 -*-
"""R30 §25 (P1-024): migrate production recursive kernels off the legacy
``_STATEFUL_CANONICALS`` name-seed fallback onto explicit self-declarations.

``execution_contract()`` keeps ``_STATEFUL_CANONICALS`` ONLY as a migration
marker (``legacy_seed_fallback=True``).  R30 requires every production operator
to declare its own statefulness / chunking / history so the hard gates
``PRODUCTION_LEGACY_STATEFUL_SEED_FALLBACK_ZERO`` and
``PRODUCTION_HISTORY_NAME_GUESS_FALLBACK_ZERO`` can be closed.  This module
declares the recursive technical kernels that previously relied on the name
seed; after this runs they resolve via ``_DECLARED_STATEFUL`` (the documented
single authority), never the legacy set.
"""
from __future__ import annotations

from factor_engine.runtime.execution_contract import declare_stateful

# Checkpoint-capable recursive kernels (segmented execution supported) — these
# are already backed by runtime.stateful_incremental checkpoint restore.
_CHECKPOINT_RECURSIVE: tuple[str, ...] = (
    "ts_ema", "ts_ewm_std", "ts_ewm_var", "ts_ewm_cov", "ts_ewm_corr",
    "RSI_WILDER", "ATR_WILDER", "ADX",
    "MACD_line", "MACD_signal", "MACD_hist",
)

# Recursive kernels WITHOUT a checkpoint restore: they must re-read full history
# for an incremental segment (fail-closed chunking).
_FULL_HISTORY_RECURSIVE: tuple[str, ...] = (
    "ts_sma_cn", "MACD", "KAMA", "Supertrend", "SupertrendDirection", "PSAR",
    "DMI_plus", "DMI_minus", "DX", "NATR",
    "PPO", "PPO_signal", "PPO_hist", "PVO", "PVO_signal", "PVO_hist",
    "KeltnerMid", "KeltnerUpper", "KeltnerLower", "KeltnerPosition",
    "TSI", "TSI_signal", "DEMA", "TEMA", "ChaikinOscillator", "ForceIndex",
    "expanding_rank", "hump_decay",
    "state_latch", "state_hold", "state_slew_limit", "state_deadband",
    "state_ewm_if",
    "event_refractory", "directional_change_state", "directional_change_extent",
    "state_since_trend_tstat",
    "ts_threshold_cycle_period", "ts_threshold_cycle_asymmetry",
    "state_episode_mfe", "state_episode_mae", "state_episode_efficiency",
    "state_episode_retrace_ratio", "state_episode_excursion_balance",
    "ts_interval_nesting_depth", "candle_gap_atr",
)

# R20 execution_contract statefulness extension (collective follow-up from
# R20-MA-DISTANCE-SLOPE / R20-SUPERTREND / R20-BVC-VPIN / R20-KELTNER): the
# derivative canonicals below sit on recursive parents already declared above
# (Wilder ATR / EMA-spread / Supertrend state machine / PSAR / KAMA), so their
# runtime contract is the SAME recursive full-history replay — but they were
# only tagged ``stateful``/``full_replay`` in technical/indicators_v2.py's
# ``_RECURSIVE_EWM`` registration tags, which ``execution_contract()`` never
# consults.  Without this tuple they resolved ``stateless`` while genuinely
# depending on full history (a chunked/incremental execution would silently
# drop the recursive warmup).
_R20_DERIVATIVE_RECURSIVE: tuple[str, ...] = (
    # Wilder-ATR derivatives (R20-KELTNER + indicators_v2 atr_* family)
    "atr_pct", "atr_zscore", "atr_percentile", "atr_short_long_ratio",
    "atr_acceleration",
    "keltner_width_pct", "keltner_compression", "keltner_breakout_strength",
    # EMA/DEMA/TEMA/KAMA-recursive distance/slope/crossover (R20-MA-DISTANCE)
    "ema_distance_pct", "dema_distance_pct", "tema_distance_pct",
    "ma_slope_pct", "ema_crossover",
    # Supertrend state-machine derivatives (R20-SUPERTREND)
    "supertrend_direction", "supertrend_distance_pct",
    "supertrend_flip", "supertrend_days_since_flip",
    # EMA-spread signed-flow imbalance (R20-BVC-VPIN)
    "bvc_imbalance_ma",
    # PSAR / KAMA recursive derivatives
    "psar_direction", "psar_distance_pct", "psar_flip",
    "psar_days_since_flip", "kama_distance_pct",
)

_APPLIED = False


def apply_stateful_contract_migration() -> None:
    """Declare explicit execution contracts for legacy recursive kernels."""
    global _APPLIED
    if _APPLIED:
        return
    for name in _CHECKPOINT_RECURSIVE:
        declare_stateful(name, state_model="recursive", chunking="checkpoint")
    for name in _FULL_HISTORY_RECURSIVE:
        declare_stateful(name, state_model="recursive", chunking="required_full_history")
    for name in _R20_DERIVATIVE_RECURSIVE:
        declare_stateful(name, state_model="recursive", chunking="required_full_history")
    _APPLIED = True
