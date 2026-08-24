# -*- coding: utf-8 -*-
"""Production repairs for advanced bounded chart patterns."""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.price_volume.structure_patterns_extra_v2 import _pf, _pi, _similar
from factor_engine.cleaned_operators.price_volume.structure_patterns_v2 import _seq_features


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
    # Time-topology gate (audit 43): trailing pivots must literally alternate
    # high -> low -> high -> low -> high, not just supply 3 highs + 1 low.
    ok, prices, positions = _seq_features(
        high, low, left_window, right_window, history_window, 5,
        (True, False, True, False, True),
    )
    highs = [prices[0], prices[2], prices[4]]
    trough = prices[3]
    mean_high = sum(highs) / 3.0
    depth = (mean_high / trough.replace(0, np.nan) - 1.0).ge(
        _pf(min_depth, "min_depth", 0)
    )
    spacing = positions[4] - positions[2]
    valid_spacing = _between(
        spacing,
        _pi(min_spacing, "min_spacing"),
        _pi(max_spacing, "max_spacing"),
    )
    return (
        _similar(highs, tolerance) * depth.astype(float) * valid_spacing.astype(float)
    ) * ok.astype(float)


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
    # Time-topology gate (audit 44): trailing pivots must literally alternate
    # low -> high -> low -> high -> low, not just supply 3 lows + 1 high.
    ok, prices, positions = _seq_features(
        high, low, left_window, right_window, history_window, 5,
        (False, True, False, True, False),
    )
    lows = [prices[0], prices[2], prices[4]]
    peak = prices[3]
    mean_low = sum(lows) / 3.0
    depth = (peak / mean_low.replace(0, np.nan) - 1.0).ge(
        _pf(min_depth, "min_depth", 0)
    )
    spacing = positions[4] - positions[2]
    valid_spacing = _between(
        spacing,
        _pi(min_spacing, "min_spacing"),
        _pi(max_spacing, "max_spacing"),
    )
    return (
        _similar(lows, tolerance) * depth.astype(float) * valid_spacing.astype(float)
    ) * ok.astype(float)


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
