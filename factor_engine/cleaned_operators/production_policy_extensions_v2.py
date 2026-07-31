# -*- coding: utf-8 -*-
"""Production-policy extensions for v2 factor operators.

Imported during registry bootstrap before ``apply_production_hardening``.
"""
from __future__ import annotations

import cleaned_operators.production_hardening as ph

_STATEFUL = {
    "KAMA": ("last_value", "last_timestamp"),
    "Supertrend": ("final_upper", "final_lower", "direction", "last_close", "last_timestamp"),
    "SupertrendDirection": ("final_upper", "final_lower", "direction", "last_close", "last_timestamp"),
    "PSAR": ("sar", "direction", "extreme_point", "acceleration_factor", "prev_high", "prev_low", "last_timestamp"),
}
ph.FULL_HISTORY_REPLAY_CANONICALS = frozenset(set(ph.FULL_HISTORY_REPLAY_CANONICALS) | set(_STATEFUL))
ph.STATEFUL_CHECKPOINTS.update(_STATEFUL)

# Fundamental-v2 names have reporting-period semantics, not trading-row periods.
from cleaned_operators.operator_surface import _FUNDAMENTAL_V2_CANONICALS, _STRUCTURE_V2_CANONICALS
for _name in _FUNDAMENTAL_V2_CANONICALS:
    ph._SCOPE_OVERRIDES[_name] = "fundamental_period"
for _name in _STRUCTURE_V2_CANONICALS:
    ph._SCOPE_OVERRIDES[_name] = "ts"

# Explicitly classify the new technical/liquidity families.  This is mainly for
# names without the ts_ prefix (PPO, Keltner, ADL, ...).
from cleaned_operators.operator_surface import _LIQUIDITY_V2_CANONICALS, _TECHNICAL_V2_CANONICALS
for _name in _LIQUIDITY_V2_CANONICALS | _TECHNICAL_V2_CANONICALS:
    ph._SCOPE_OVERRIDES[_name] = "ts"

ph._FUNDAMENTAL_PERIOD_CANONICALS = frozenset(set(ph._FUNDAMENTAL_PERIOD_CANONICALS) | set(_FUNDAMENTAL_V2_CANONICALS))
