# -*- coding: utf-8 -*-
"""Next-stage intraday factor operators (2026-08 expansion).

Minute-frequency panels in, one scalar per (TradeDate, Symbol) out.  All
kernels are causal and PIT-safe: they consume only the day's own minute data
(plus that day's daily limit / weight panels) and form at that day's close,
defaulting to next-trading-day availability.

This package is intentionally separate from ``microstructure`` so the two
evolution tracks do not share file-level edit conflicts.
"""
from __future__ import annotations

# PERF-2: bind the vectorized daily_agg kernel whitelist so the dispatch
# points in ``_core.daily_agg{,_two,_three}`` route the three harness-proven
# TRUE_GAP kernels to the (day, bar, inst) 3-D fast path.  Importing this
# package (as every intraday operator test does) activates the whitelist.
from factor_engine.cleaned_operators.intraday import perf_vec_kernels as _pvk

try:
    _pvk.bind_whitelist()
except Exception:
    # Optional perf-kernel whitelist: smart_money / vwap_path are Polars-gated.
    # When `polars` is absent the whole bind is skipped so intraday operator
    # modules can still register their pandas backends (time_structure etc.).
    pass
del _pvk

__all__ = [
    "higher_moments",
    "realized_beta",
    "time_structure",
    "vwap_path",
    "overnight",
    "topology_manifold",
    "state_space",
    "pattern_recognition",
]
