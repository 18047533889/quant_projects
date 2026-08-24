# -*- coding: utf-8 -*-
"""Polars backend for panel_batch1 operators (2026-08-13).

Polars implementations for the 5 TRUE_GAP panel operators using native polars
expressions where feasible, falling back to UDF delegation for complex kernels.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like


# ---------------------------------------------------------------------------
# Polars backend registrations (delegate to pandas via UDF for now)
# ---------------------------------------------------------------------------
def _register_polars_backends() -> None:
    """Register polars backends for panel_batch1 operators.

    These operators use complex multi-column rolling regressions and event
    tracking that don't map cleanly to polars expressions, so we delegate to
    the certified pandas_numpy reference via polars UDF.
    """
    if pl is None:
        return

    from cleaned_operators.registry import OperatorRegistry
    from cleaned_operators.rolling_pack import register_polars_udf

    canonicals = [
        "panel_day_night_beta_gap",
        "pastor_stambaugh_beta",
        "price_delay_score",
        "report_asof",
        "event_window_return_asof",
    ]

    for canonical in canonicals:
        if canonical in OperatorRegistry.list_canonical():
            register_polars_udf(canonical)


_register_polars_backends()
