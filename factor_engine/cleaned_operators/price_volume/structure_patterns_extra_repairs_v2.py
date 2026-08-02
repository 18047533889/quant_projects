# -*- coding: utf-8 -*-
"""Production repairs for advanced bounded chart patterns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.price_volume.structure_patterns_extra_v2 import _pf, _pi, _similar
from cleaned_operators.price_volume.structure_patterns_v2 import (
    ts_nth_pivot_high,
    ts_nth_pivot_low,
    ts_pivot_high_spacing,
    ts_pivot_low_spacing,
)


def _between(frame: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    """Elementwise inclusive between for DataFrame; Series.between is unavailable."""
    return frame.ge(lower) & frame.le(upper)


def pattern_triple_top(
    high,
    low,
    left_window,
    right_window,
    history_window,
    tolerance,
    min_depth,
    min_spacing,
    max_spacing,
):
    tolerance = _pf(tolerance, "tolerance", 1e-12)
    highs = [
        ts_nth_pivot_high(
            high, left_window, right_window, history_window, rank
        )
        for rank in (1, 2, 3)
    ]
    trough = ts_nth_pivot_low(
        low, left_window, right_window, history_window, 1
    )
    mean_high = sum(highs) / 3.0
    depth = (mean_high / trough.replace(0, np.nan) - 1.0).ge(
        _pf(min_depth, "min_depth", 0)
    )
    spacing = ts_pivot_high_spacing(
        high, left_window, right_window, history_window
    )
    valid_spacing = _between(
        spacing,
        _pi(min_spacing, "min_spacing"),
        _pi(max_spacing, "max_spacing"),
    )
    return _similar(highs, tolerance) * depth.astype(float) * valid_spacing.astype(float)


def pattern_triple_bottom(
    high,
    low,
    left_window,
    right_window,
    history_window,
    tolerance,
    min_depth,
    min_spacing,
    max_spacing,
):
    tolerance = _pf(tolerance, "tolerance", 1e-12)
    lows = [
        ts_nth_pivot_low(
            low, left_window, right_window, history_window, rank
        )
        for rank in (1, 2, 3)
    ]
    peak = ts_nth_pivot_high(
        high, left_window, right_window, history_window, 1
    )
    mean_low = sum(lows) / 3.0
    depth = (peak / mean_low.replace(0, np.nan) - 1.0).ge(
        _pf(min_depth, "min_depth", 0)
    )
    spacing = ts_pivot_low_spacing(
        low, left_window, right_window, history_window
    )
    valid_spacing = _between(
        spacing,
        _pi(min_spacing, "min_spacing"),
        _pi(max_spacing, "max_spacing"),
    )
    return _similar(lows, tolerance) * depth.astype(float) * valid_spacing.astype(float)


def _register(name, function):
    metadata = OperatorMetadata(
        name=name,
        category="chart_pattern",
        description=f"Repaired causal {name.replace('_', ' ')}.",
        param_names=[
            "high",
            "low",
            "left_window",
            "right_window",
            "history_window",
            "tolerance",
            "min_depth",
            "min_spacing",
            "max_spacing",
        ],
        return_type="series",
        tags=[
            "pit_safe",
            "causal",
            "bounded_history",
            "production_repair",
        ],
    )

    def calculate(self, *args, **kwargs):
        return function(*args, **kwargs)

    operator = type(
        f"ChartPatternRepair_{name}",
        (SeriesOperator,),
        {
            "metadata": metadata,
            "_calculate_series": calculate,
            "__module__": __name__,
        },
    )
    register_operator(
        name=name,
        category="chart_pattern",
        business_category="technical_structure",
        canonical=name,
        source="structure_patterns_extra_repairs_v2",
        backend="pandas_numpy",
        status="production",
    )(operator)


_register("pattern_triple_top", pattern_triple_top)
_register("pattern_triple_bottom", pattern_triple_bottom)
