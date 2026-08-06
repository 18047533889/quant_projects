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

__all__ = [
    "higher_moments",
    "realized_beta",
    "time_structure",
    "vwap_path",
    "overnight",
]
