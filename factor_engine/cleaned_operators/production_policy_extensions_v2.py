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
ph.STATEFUL_CHECKPOINTS.update(_STATEFUL)
_unrestored_stateful = set(ph.STATEFUL_CHECKPOINTS).difference(
    ph.SEGMENTED_EXECUTION_CANONICALS
)
ph.FULL_HISTORY_REPLAY_CANONICALS = frozenset(
    set(ph.FULL_HISTORY_REPLAY_CANONICALS) | _unrestored_stateful
)

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

# ``production_tiers`` owns the candidate set:
# ``PANDAS_FIRST_PRODUCTION_CANONICALS = frozenset(EXTENDED_ONLY_CANONICALS)``.
# It is deliberately NOT re-assigned here; a previous union-with-self rewrite
# was redundant and hid the real owner.

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

# WS-D #257: the analyzer must read the SAME history authority as warmup and
# composite lowering.  Register every canonical in the stateful seed into
# ``ir.analyzer``'s canonical history-requirement registry, so the analyzer's
# lookback derives from ``HistoryRequirement`` instead of a name-guessing tuple.
# The ``FULL_HISTORY_LOOKBACK_SENTINEL`` integer survives ONLY as the
# serialized-analysis encoding (``analysis.lookback``) for round-trip
# compatibility; it is not consulted by warmup/chunking anymore.
from runtime.execution_contract import _STATEFUL_CANONICALS, history_requirement

for _stateful_canon in _STATEFUL_CANONICALS:
    if _stateful_canon in analyzer._HISTORY_REQUIREMENT_FUNCS:
        continue

    def _make_history_fn(canon: str):
        def _history_fn(params):
            return history_requirement(canon, params).rows

        return _history_fn

    analyzer.register_history_requirement(
        _stateful_canon, _make_history_fn(_stateful_canon)
    )

FULL_HISTORY_LOOKBACK_SENTINEL = 1_000_000_000
analyzer.FULL_HISTORY_LOOKBACK_SENTINEL = FULL_HISTORY_LOOKBACK_SENTINEL
if not getattr(analyzer.AnalysisResult, "_full_history_init_installed", False):
    _analysis_result_init = analyzer.AnalysisResult.__init__

    def _init_with_full_history_sentinel(
        self,
        ir,
        lookback,
        has_ts_op,
        has_cs_op,
        referenced_columns,
        requires_full_history=False,
        referenced_fields=None,
        column_schemas=None,
        **kwargs,
    ):
        encoded = (
            max(int(lookback), FULL_HISTORY_LOOKBACK_SENTINEL)
            if requires_full_history
            else int(lookback)
        )
        _analysis_result_init(
            self,
            ir,
            encoded,
            has_ts_op,
            has_cs_op,
            referenced_columns,
            requires_full_history,
            referenced_fields or {},
            column_schemas or {},
        )

    analyzer.AnalysisResult.__init__ = _init_with_full_history_sentinel
    analyzer.AnalysisResult._full_history_init_installed = True
