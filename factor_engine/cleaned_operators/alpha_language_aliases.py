# -*- coding: utf-8 -*-
"""Alpha-language DSL aliases onto existing canonicals (2026-08).

Semantic-diff outcome: these requested operator names already have an exact or
near-exact existing canonical.  Per the "don't duplicate" rule we register them
as DSL aliases instead of new implementations.  This module must load AFTER the
canonical targets exist (it sits at the end of ``_REVIEWED_EXTENSIONS``, after
``fundamental/transforms_v2``).
"""
from __future__ import annotations

from cleaned_operators.registry import OperatorRegistry

_ALIASES = {
    # generic event language -> existing generic/domain primitives
    "event_age": "ts_days_since",                 # days since condition last true
    "event_decay": "event_decay_asof",            # causal half-life exponential decay
    "event_interval_mean": "ts_event_spacing_mean",
    "event_interval_cv": "ts_event_spacing_cv",
    # fundamental report sequence -> audited fin_* kernels (PIT/revision safe)
    "report_lag": "fin_lag",
    "report_age": "fin_staleness",
    "report_single_quarter": "fin_quarter_from_cumulative",
    "report_rolling_std": "fin_std",
    "report_rank": "fin_percentile_history",
    "report_surprise_to_trend": "fin_surprise_zscore",
    # intraday signed jump balance == intra_signed_jump_ratio
    # ((posJV-negJV)/(posJV+negJV)) — exact duplicate, registered as an alias.
    "intraday_signed_jump_balance": "intra_signed_jump_ratio",
}


def _register_aliases() -> None:
    for alias, canonical in _ALIASES.items():
        try:
            OperatorRegistry.register_alias(alias, canonical)
        except KeyError as exc:  # target not registered -> surface loudly
            raise RuntimeError(f"alias {alias}->{canonical}: {exc}") from exc


_register_aliases()
