# -*- coding: utf-8 -*-
"""Production-policy extensions for reviewed v2 factor operators."""
from __future__ import annotations

import cleaned_operators.production_hardening as ph
from cleaned_operators import operator_surface as surface

_STATEFUL = {
    "KAMA": ("last_value", "last_timestamp"),
    "Supertrend": (
        "final_upper",
        "final_lower",
        "direction",
        "last_close",
        "last_timestamp",
    ),
    "SupertrendDirection": (
        "final_upper",
        "final_lower",
        "direction",
        "last_close",
        "last_timestamp",
    ),
    "PSAR": (
        "sar",
        "direction",
        "extreme_point",
        "acceleration_factor",
        "prev_high",
        "prev_low",
        "last_timestamp",
    ),
}
ph.FULL_HISTORY_REPLAY_CANONICALS = frozenset(
    set(ph.FULL_HISTORY_REPLAY_CANONICALS) | set(_STATEFUL)
)
ph.STATEFUL_CHECKPOINTS.update(_STATEFUL)

for _name in surface._FUNDAMENTAL_V2_CANONICALS:
    ph._SCOPE_OVERRIDES[_name] = "fundamental_period"
for _name in surface._STRUCTURE_V2_CANONICALS:
    ph._SCOPE_OVERRIDES[_name] = "ts"
for _name in surface._LIQUIDITY_V2_CANONICALS | surface._TECHNICAL_V2_CANONICALS:
    ph._SCOPE_OVERRIDES[_name] = "ts"
ph._FUNDAMENTAL_PERIOD_CANONICALS = frozenset(
    set(ph._FUNDAMENTAL_PERIOD_CANONICALS)
    | set(surface._FUNDAMENTAL_V2_CANONICALS)
)

# Refresh surface-derived candidate tiers after every reviewed module has
# registered.  Membership means signature/evidence-gated candidate, never
# physical production admission by itself.
import cleaned_operators.production_tiers as pt

_all_extended = set(surface.EXTENDED_ONLY_CANONICALS)
pt.PANDAS_FIRST_PRODUCTION_CANONICALS = frozenset(
    set(pt.PANDAS_FIRST_PRODUCTION_CANONICALS) | _all_extended
)
pt.PANDAS_FIRST_SIGNATURE_GATED = frozenset(
    set(pt.PANDAS_FIRST_SIGNATURE_GATED) | _all_extended
)
pt.PANDAS_FIRST_BACKEND_PINNED = frozenset(
    set(pt.PANDAS_FIRST_BACKEND_PINNED) | _all_extended
)

# Generalize parameter classes for the new public contracts without weakening
# fail-closed validation. Dynamic values are rejected for every scalar.
import backend.pandas_first_signature as ps

ps._PANEL_NAMES = frozenset(
    set(ps._PANEL_NAMES)
    | {
        "actual",
        "expected",
        "expected_std",
        "expected_mean",
        "target_period_id",
        "fundamental_x",
        "fundamental_y",
        "fundamental_scale",
    }
)
ps._WINDOW_NAMES = frozenset(
    set(ps._WINDOW_NAMES)
    | {
        "cup_window",
        "handle_window",
        "pennant_window",
        "body_window",
        "shadow_window",
        "window_days",
        "max_days",
        "max_wait",
    }
)

# Analyzer is imported before registry bootstrap in common parser paths. Extend
# its semantic sets here after expectation operators are registered, avoiding an
# import-order-dependent warmup contract.
import ir.analyzer as analyzer

analyzer._FIN_REPORT_PERIOD_CANONICALS = frozenset(
    set(analyzer._FIN_REPORT_PERIOD_CANONICALS)
    | {
        "fin_surprise_event_zscore",
        "fin_surprise_event_percentile",
    }
)
analyzer._FIN_DAILY_WINDOW_CANONICALS = frozenset(
    set(analyzer._FIN_DAILY_WINDOW_CANONICALS)
    | {
        "fin_expectation_revision_count",
        "fin_expectation_revision_magnitude",
        "fin_days_since_expectation_revision",
    }
)
