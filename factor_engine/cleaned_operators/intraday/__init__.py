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
# points in ``_core.daily_agg{,_two,_three}`` route the harness-proven
# TRUE_GAP / smart_money / vwap_path / higher_moments / sufficient-stats
# kernels to the (day, bar, inst) 3-D fast path.  Importing this package (as
# every intraday operator test does) activates the whitelist.
from factor_engine.cleaned_operators.intraday import perf_vec_kernels as _pvk

# PERF-2 (100k GO §6.3): vector binding must NOT be silently swallowed — an
# import/bind/feature-rename failure in the fast path would otherwise silently
# fall back to the scalar path (and silently time scalar under a "vec"
# benchmark / hide a slow path under production load).  Fail loudly here so a
# broken whitelist is caught at import time.
_pvk.bind_whitelist()
del _pvk

# R61-P1 #57: the shared sufficient-statistics layer + its operator family are
# imported here so importing the package (as every intraday operator test and
# the coverage generator do) registers the 16 new ``intra_ts_*`` operators and
# exposes the sufficient_stats module.  See sufficient_stats.py /
# sufficient_stats_ops.py.
from factor_engine.cleaned_operators.intraday import sufficient_stats as _ss  # noqa: E402
from factor_engine.cleaned_operators.intraday import sufficient_stats_ops as _sso  # noqa: E402
del _ss, _sso

__all__ = [
    "higher_moments",
    "realized_beta",
    "time_structure",
    "vwap_path",
    "overnight",
    "topology_manifold",
    "state_space",
    "pattern_recognition",
    "sufficient_stats",
    "sufficient_stats_ops",
]
