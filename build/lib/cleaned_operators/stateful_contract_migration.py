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

from runtime.execution_contract import declare_stateful

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
    _APPLIED = True
