# -*- coding: utf-8 -*-
"""
Chart Pattern Operators - Polars Native

Implements chart pattern recognition using pure Polars/numpy.
27 pattern_* operators for technical analysis pattern detection.
"""
from __future__ import annotations
import polars as pl
import numpy as np
from factor_engine.cleaned_operators.price_volume.structure_extra_contracts import extra_contract
from factor_engine.cleaned_operators.base_polars import (
    SeriesOperator,
    OperatorMetadata,
    register_operator,
    PANEL_SKIP_COLUMNS,
)


def _resolved_pattern_kernel(operator):
    from factor_engine.cleaned_operators.price_volume import polars_structure
    return getattr(polars_structure, operator.metadata.name)


def _result_df(data_dict: dict, template_df: pl.DataFrame) -> pl.DataFrame:
    """Create result DataFrame with metadata columns from template."""
    result = pl.DataFrame(data_dict)
    for meta_col in PANEL_SKIP_COLUMNS:
        if meta_col in template_df.columns:
            result = result.with_columns([template_df[meta_col]])
    return result


def _find_pivots(data: np.ndarray, left: int = 2, right: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Find pivot highs and lows in price data."""
    n = len(data)
    pivot_highs = np.zeros(n, dtype=bool)
    pivot_lows = np.zeros(n, dtype=bool)
    
    for i in range(left, n - right):
        # Pivot high: higher than left and right neighbors
        if np.all(data[i] >= data[i-left:i]) and np.all(data[i] >= data[i+1:i+right+1]):
            pivot_highs[i] = True
        # Pivot low: lower than left and right neighbors
        if np.all(data[i] <= data[i-left:i]) and np.all(data[i] <= data[i+1:i+right+1]):
            pivot_lows[i] = True
    
    return pivot_highs, pivot_lows


# ==================== Chart Pattern Operators (27 total) ====================

@register_operator(
    name="pattern_123_bear",
    category="chart_pattern",
    canonical="pattern_123_bear",
    source="polars_native_patterns")
class Pattern123Bear(SeriesOperator):
    """Canonical continuous pattern score; sequential Polars/NumPy kernel."""
    metadata = OperatorMetadata(
        name="pattern_123_bear", category="chart_pattern",
        description="Canonical pattern_123_bear with matching confirmed-history semantics.",
        param_names=["high","low","left_window","right_window","history_window","min_swing"],
        **extra_contract("pattern_123_bear", ["high","low","left_window","right_window","history_window","min_swing"]),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, min_swing: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_123_bear
        return pattern_123_bear(high, low, left_window, right_window, history_window, min_swing)


@register_operator(
    name="pattern_123_bull",
    category="chart_pattern",
    canonical="pattern_123_bull",
    source="polars_native_patterns")
class Pattern123Bull(SeriesOperator):
    """Canonical continuous pattern score; sequential Polars/NumPy kernel."""
    metadata = OperatorMetadata(
        name="pattern_123_bull", category="chart_pattern",
        description="Canonical pattern_123_bull with matching confirmed-history semantics.",
        param_names=["high","low","left_window","right_window","history_window","min_swing"],
        **extra_contract("pattern_123_bull", ["high","low","left_window","right_window","history_window","min_swing"]),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, min_swing: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_123_bull
        return pattern_123_bull(high, low, left_window, right_window, history_window, min_swing)


@register_operator(
    name="pattern_ascending_triangle",
    category="chart_pattern",
    canonical="pattern_ascending_triangle",
    source="polars_native_patterns")
@register_operator(
    name="pattern_ascending_triangle", category="chart_pattern", canonical="pattern_ascending_triangle",
    source="polars_native_patterns")
class PatternAscendingTriangle(SeriesOperator):
    """Canonical conversion wrapper with the authoritative pattern contract."""
    metadata = OperatorMetadata(
        name="pattern_ascending_triangle", category="chart_pattern",
        description="Canonical pattern_ascending_triangle with matching confirmed-history semantics.",
        param_names=['high', 'low', 'left_window', 'right_window', 'history_window', 'points', 'slope_threshold'],
        **extra_contract("pattern_ascending_triangle", ['high', 'low', 'left_window', 'right_window', 'history_window', 'points', 'slope_threshold']),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, points: int, slope_threshold: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_ascending_triangle
        return pattern_ascending_triangle(high, low, left_window, right_window, history_window, points, slope_threshold)


@register_operator(
    name="pattern_bear_flag",
    category="chart_pattern",
    canonical="pattern_bear_flag",
    source="polars_native_patterns")
@register_operator(
    name="pattern_bear_flag", category="chart_pattern", canonical="pattern_bear_flag",
    source="polars_native_patterns")
class PatternBearFlag(SeriesOperator):
    """Canonical conversion wrapper with the authoritative pattern contract."""
    metadata = OperatorMetadata(
        name="pattern_bear_flag", category="chart_pattern",
        description="Canonical pattern_bear_flag with matching confirmed-history semantics.",
        param_names=['close', 'high', 'low', 'volume', 'impulse_window', 'flag_window', 'min_impulse', 'max_retracement', 'max_width', 'volume_decay_threshold'],
        **extra_contract("pattern_bear_flag", ['close', 'high', 'low', 'volume', 'impulse_window', 'flag_window', 'min_impulse', 'max_retracement', 'max_width', 'volume_decay_threshold']),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, close: pl.DataFrame, high: pl.DataFrame, low: pl.DataFrame, volume: pl.DataFrame, impulse_window: int, flag_window: int, min_impulse: float, max_retracement: float, max_width: float, volume_decay_threshold: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_bear_flag
        return pattern_bear_flag(close, high, low, volume, impulse_window, flag_window, min_impulse, max_retracement, max_width, volume_decay_threshold)


@register_operator(
    name="pattern_bear_pennant",
    category="chart_pattern",
    canonical="pattern_bear_pennant",
    source="polars_native_patterns")
class PatternBearPennant(SeriesOperator):
    """Canonical continuous pattern score; sequential Polars/NumPy kernel."""
    metadata = OperatorMetadata(
        name="pattern_bear_pennant", category="chart_pattern",
        description="Canonical pattern_bear_pennant with matching confirmed-history semantics.",
        param_names=["close","high","low","volume","impulse_window","pennant_window","min_impulse","max_width","volume_decay_threshold"],
        **extra_contract("pattern_bear_pennant", ["close","high","low","volume","impulse_window","pennant_window","min_impulse","max_width","volume_decay_threshold"]),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, close: pl.DataFrame, high: pl.DataFrame, low: pl.DataFrame, volume: pl.DataFrame, impulse_window: int, pennant_window: int, min_impulse: float, max_width: float, volume_decay_threshold: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_bear_pennant
        return pattern_bear_pennant(close, high, low, volume, impulse_window, pennant_window, min_impulse, max_width, volume_decay_threshold)


@register_operator(
    name="pattern_breakdown_retest",
    category="chart_pattern",
    canonical="pattern_breakdown_retest",
    source="polars_native_patterns")
class PatternBreakdownRetest(SeriesOperator):
    """Canonical continuous pattern score; sequential Polars/NumPy kernel."""
    metadata = OperatorMetadata(
        name="pattern_breakdown_retest", category="chart_pattern",
        description="Canonical pattern_breakdown_retest with matching confirmed-history semantics.",
        param_names=["close","window","max_wait","tolerance"],
        **extra_contract("pattern_breakdown_retest", ["close","window","max_wait","tolerance"]),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, close: pl.DataFrame, window: int, max_wait: int, tolerance: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_breakdown_retest
        return pattern_breakdown_retest(close, window, max_wait, tolerance)


@register_operator(
    name="pattern_breakout_retest",
    category="chart_pattern",
    canonical="pattern_breakout_retest",
    source="polars_native_patterns")
class PatternBreakoutRetest(SeriesOperator):
    """Canonical continuous pattern score; sequential Polars/NumPy kernel."""
    metadata = OperatorMetadata(
        name="pattern_breakout_retest", category="chart_pattern",
        description="Canonical pattern_breakout_retest with matching confirmed-history semantics.",
        param_names=["close","window","max_wait","tolerance"],
        **extra_contract("pattern_breakout_retest", ["close","window","max_wait","tolerance"]),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, close: pl.DataFrame, window: int, max_wait: int, tolerance: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_breakout_retest
        return pattern_breakout_retest(close, window, max_wait, tolerance)


@register_operator(
    name="pattern_broadening",
    category="chart_pattern",
    canonical="pattern_broadening",
    source="polars_native_patterns")
@register_operator(
    name="pattern_broadening", category="chart_pattern", canonical="pattern_broadening",
    source="polars_native_patterns")
class PatternBroadening(SeriesOperator):
    """Canonical conversion wrapper with the authoritative pattern contract."""
    metadata = OperatorMetadata(
        name="pattern_broadening", category="chart_pattern",
        description="Canonical pattern_broadening with matching confirmed-history semantics.",
        param_names=['high', 'low', 'left_window', 'right_window', 'history_window', 'points', 'slope_threshold'],
        **extra_contract("pattern_broadening", ['high', 'low', 'left_window', 'right_window', 'history_window', 'points', 'slope_threshold']),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, points: int, slope_threshold: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_broadening
        return pattern_broadening(high, low, left_window, right_window, history_window, points, slope_threshold)


@register_operator(
    name="pattern_bull_flag",
    category="chart_pattern",
    canonical="pattern_bull_flag",
    source="polars_native_patterns")
@register_operator(
    name="pattern_bull_flag", category="chart_pattern", canonical="pattern_bull_flag",
    source="polars_native_patterns")
class PatternBullFlag(SeriesOperator):
    """Canonical conversion wrapper with the authoritative pattern contract."""
    metadata = OperatorMetadata(
        name="pattern_bull_flag", category="chart_pattern",
        description="Canonical pattern_bull_flag with matching confirmed-history semantics.",
        param_names=['close', 'high', 'low', 'volume', 'impulse_window', 'flag_window', 'min_impulse', 'max_retracement', 'max_width', 'volume_decay_threshold'],
        **extra_contract("pattern_bull_flag", ['close', 'high', 'low', 'volume', 'impulse_window', 'flag_window', 'min_impulse', 'max_retracement', 'max_width', 'volume_decay_threshold']),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, close: pl.DataFrame, high: pl.DataFrame, low: pl.DataFrame, volume: pl.DataFrame, impulse_window: int, flag_window: int, min_impulse: float, max_retracement: float, max_width: float, volume_decay_threshold: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_bull_flag
        return pattern_bull_flag(close, high, low, volume, impulse_window, flag_window, min_impulse, max_retracement, max_width, volume_decay_threshold)


@register_operator(
    name="pattern_bull_pennant",
    category="chart_pattern",
    canonical="pattern_bull_pennant",
    source="polars_native_patterns")
class PatternBullPennant(SeriesOperator):
    """Canonical continuous pattern score; sequential Polars/NumPy kernel."""
    metadata = OperatorMetadata(
        name="pattern_bull_pennant", category="chart_pattern",
        description="Canonical pattern_bull_pennant with matching confirmed-history semantics.",
        param_names=["close","high","low","volume","impulse_window","pennant_window","min_impulse","max_width","volume_decay_threshold"],
        **extra_contract("pattern_bull_pennant", ["close","high","low","volume","impulse_window","pennant_window","min_impulse","max_width","volume_decay_threshold"]),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, close: pl.DataFrame, high: pl.DataFrame, low: pl.DataFrame, volume: pl.DataFrame, impulse_window: int, pennant_window: int, min_impulse: float, max_width: float, volume_decay_threshold: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_bull_pennant
        return pattern_bull_pennant(close, high, low, volume, impulse_window, pennant_window, min_impulse, max_width, volume_decay_threshold)


@register_operator(
    name="pattern_cup",
    category="chart_pattern",
    canonical="pattern_cup",
    source="polars_native_patterns")
class PatternCup(SeriesOperator):
    """Canonical continuous pattern score; sequential Polars/NumPy kernel."""
    metadata = OperatorMetadata(
        name="pattern_cup", category="chart_pattern",
        description="Canonical pattern_cup with matching confirmed-history semantics.",
        param_names=["close","window","min_depth","max_edge_diff","min_fit"],
        **extra_contract("pattern_cup", ["close","window","min_depth","max_edge_diff","min_fit"]),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, close: pl.DataFrame, window: int, min_depth: float, max_edge_diff: float, min_fit: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_cup
        return pattern_cup(close, window, min_depth, max_edge_diff, min_fit)


@register_operator(
    name="pattern_cup_handle",
    category="chart_pattern",
    canonical="pattern_cup_handle",
    source="polars_native_patterns")
class PatternCupHandle(SeriesOperator):
    """Canonical continuous pattern score; sequential Polars/NumPy kernel."""
    metadata = OperatorMetadata(
        name="pattern_cup_handle", category="chart_pattern",
        description="Canonical pattern_cup_handle with matching confirmed-history semantics.",
        param_names=["close","high","low","cup_window","handle_window","min_depth","max_edge_diff","min_fit","max_handle_retracement"],
        **extra_contract("pattern_cup_handle", ["close","high","low","cup_window","handle_window","min_depth","max_edge_diff","min_fit","max_handle_retracement"]),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, close: pl.DataFrame, high: pl.DataFrame, low: pl.DataFrame, cup_window: int, handle_window: int, min_depth: float, max_edge_diff: float, min_fit: float, max_handle_retracement: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_cup_handle
        return pattern_cup_handle(close, high, low, cup_window, handle_window, min_depth, max_edge_diff, min_fit, max_handle_retracement)


@register_operator(
    name="pattern_descending_triangle",
    category="chart_pattern",
    canonical="pattern_descending_triangle",
    source="polars_native_patterns")
@register_operator(
    name="pattern_descending_triangle", category="chart_pattern", canonical="pattern_descending_triangle",
    source="polars_native_patterns")
class PatternDescendingTriangle(SeriesOperator):
    """Canonical conversion wrapper with the authoritative pattern contract."""
    metadata = OperatorMetadata(
        name="pattern_descending_triangle", category="chart_pattern",
        description="Canonical pattern_descending_triangle with matching confirmed-history semantics.",
        param_names=['high', 'low', 'left_window', 'right_window', 'history_window', 'points', 'slope_threshold'],
        **extra_contract("pattern_descending_triangle", ['high', 'low', 'left_window', 'right_window', 'history_window', 'points', 'slope_threshold']),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, points: int, slope_threshold: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_descending_triangle
        return pattern_descending_triangle(high, low, left_window, right_window, history_window, points, slope_threshold)


@register_operator(
    name="pattern_double_bottom",
    category="chart_pattern",
    canonical="pattern_double_bottom",
    source="polars_native_patterns")
@register_operator(
    name="pattern_double_bottom", category="chart_pattern", canonical="pattern_double_bottom",
    source="polars_native_patterns")
class PatternDoubleBottom(SeriesOperator):
    """Canonical conversion wrapper with the authoritative pattern contract."""
    metadata = OperatorMetadata(
        name="pattern_double_bottom", category="chart_pattern",
        description="Canonical pattern_double_bottom with matching confirmed-history semantics.",
        param_names=['high', 'low', 'left_window', 'right_window', 'history_window', 'tolerance', 'min_depth', 'min_spacing', 'max_spacing'],
        **extra_contract("pattern_double_bottom", ['high', 'low', 'left_window', 'right_window', 'history_window', 'tolerance', 'min_depth', 'min_spacing', 'max_spacing']),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, tolerance: float, min_depth: float, min_spacing: int, max_spacing: int, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_double_bottom
        return pattern_double_bottom(high, low, left_window, right_window, history_window, tolerance, min_depth, min_spacing, max_spacing)


@register_operator(
    name="pattern_double_top",
    category="chart_pattern",
    canonical="pattern_double_top",
    source="polars_native_patterns")
@register_operator(
    name="pattern_double_top", category="chart_pattern", canonical="pattern_double_top",
    source="polars_native_patterns")
class PatternDoubleTop(SeriesOperator):
    """Canonical conversion wrapper with the authoritative pattern contract."""
    metadata = OperatorMetadata(
        name="pattern_double_top", category="chart_pattern",
        description="Canonical pattern_double_top with matching confirmed-history semantics.",
        param_names=['high', 'low', 'left_window', 'right_window', 'history_window', 'tolerance', 'min_depth', 'min_spacing', 'max_spacing'],
        **extra_contract("pattern_double_top", ['high', 'low', 'left_window', 'right_window', 'history_window', 'tolerance', 'min_depth', 'min_spacing', 'max_spacing']),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, tolerance: float, min_depth: float, min_spacing: int, max_spacing: int, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_double_top
        return pattern_double_top(high, low, left_window, right_window, history_window, tolerance, min_depth, min_spacing, max_spacing)


@register_operator(
    name="pattern_falling_channel",
    category="chart_pattern",
    canonical="pattern_falling_channel",
    source="polars_native_patterns")
@register_operator(
    name="pattern_falling_channel", category="chart_pattern", canonical="pattern_falling_channel",
    source="polars_native_patterns")
class PatternFallingChannel(SeriesOperator):
    """Canonical conversion wrapper with the authoritative pattern contract."""
    metadata = OperatorMetadata(
        name="pattern_falling_channel", category="chart_pattern",
        description="Canonical pattern_falling_channel with matching confirmed-history semantics.",
        param_names=['high', 'low', 'left_window', 'right_window', 'history_window', 'points', 'slope_threshold', 'parallel_tolerance'],
        **extra_contract("pattern_falling_channel", ['high', 'low', 'left_window', 'right_window', 'history_window', 'points', 'slope_threshold', 'parallel_tolerance']),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, points: int, slope_threshold: float, parallel_tolerance: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_falling_channel
        return pattern_falling_channel(high, low, left_window, right_window, history_window, points, slope_threshold, parallel_tolerance)


@register_operator(
    name="pattern_falling_wedge",
    category="chart_pattern",
    canonical="pattern_falling_wedge",
    source="polars_native_patterns")
@register_operator(
    name="pattern_falling_wedge", category="chart_pattern", canonical="pattern_falling_wedge",
    source="polars_native_patterns")
class PatternFallingWedge(SeriesOperator):
    """Canonical conversion wrapper with the authoritative pattern contract."""
    metadata = OperatorMetadata(
        name="pattern_falling_wedge", category="chart_pattern",
        description="Canonical pattern_falling_wedge with matching confirmed-history semantics.",
        param_names=['high', 'low', 'left_window', 'right_window', 'history_window', 'points', 'slope_threshold'],
        **extra_contract("pattern_falling_wedge", ['high', 'low', 'left_window', 'right_window', 'history_window', 'points', 'slope_threshold']),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, points: int, slope_threshold: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_falling_wedge
        return pattern_falling_wedge(high, low, left_window, right_window, history_window, points, slope_threshold)


@register_operator(
    name="pattern_head_shoulders",
    category="chart_pattern",
    canonical="pattern_head_shoulders",
    source="polars_native_patterns")
@register_operator(
    name="pattern_head_shoulders", category="chart_pattern", canonical="pattern_head_shoulders",
    source="polars_native_patterns")
class PatternHeadShoulders(SeriesOperator):
    """Canonical conversion wrapper with the authoritative pattern contract."""
    metadata = OperatorMetadata(
        name="pattern_head_shoulders", category="chart_pattern",
        description="Canonical pattern_head_shoulders with matching confirmed-history semantics.",
        param_names=['high', 'low', 'left_window', 'right_window', 'history_window', 'shoulder_tolerance', 'head_min_prominence', 'max_neckline_slope'],
        **extra_contract("pattern_head_shoulders", ['high', 'low', 'left_window', 'right_window', 'history_window', 'shoulder_tolerance', 'head_min_prominence', 'max_neckline_slope']),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, shoulder_tolerance: float, head_min_prominence: float, max_neckline_slope: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_head_shoulders
        return pattern_head_shoulders(high, low, left_window, right_window, history_window, shoulder_tolerance, head_min_prominence, max_neckline_slope)


@register_operator(
    name="pattern_inverse_head_shoulders",
    category="chart_pattern",
    canonical="pattern_inverse_head_shoulders",
    source="polars_native_patterns")
@register_operator(
    name="pattern_inverse_head_shoulders", category="chart_pattern", canonical="pattern_inverse_head_shoulders",
    source="polars_native_patterns")
class PatternInverseHeadShoulders(SeriesOperator):
    """Canonical conversion wrapper with the authoritative pattern contract."""
    metadata = OperatorMetadata(
        name="pattern_inverse_head_shoulders", category="chart_pattern",
        description="Canonical pattern_inverse_head_shoulders with matching confirmed-history semantics.",
        param_names=['high', 'low', 'left_window', 'right_window', 'history_window', 'shoulder_tolerance', 'head_min_prominence', 'max_neckline_slope'],
        **extra_contract("pattern_inverse_head_shoulders", ['high', 'low', 'left_window', 'right_window', 'history_window', 'shoulder_tolerance', 'head_min_prominence', 'max_neckline_slope']),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, shoulder_tolerance: float, head_min_prominence: float, max_neckline_slope: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_inverse_head_shoulders
        return pattern_inverse_head_shoulders(high, low, left_window, right_window, history_window, shoulder_tolerance, head_min_prominence, max_neckline_slope)


@register_operator(
    name="pattern_rectangle",
    category="chart_pattern",
    canonical="pattern_rectangle",
    source="polars_native_patterns")
@register_operator(
    name="pattern_rectangle", category="chart_pattern", canonical="pattern_rectangle",
    source="polars_native_patterns")
class PatternRectangle(SeriesOperator):
    """Canonical conversion wrapper with the authoritative pattern contract."""
    metadata = OperatorMetadata(
        name="pattern_rectangle", category="chart_pattern",
        description="Canonical pattern_rectangle with matching confirmed-history semantics.",
        param_names=['high', 'low', 'left_window', 'right_window', 'history_window', 'points', 'slope_threshold'],
        **extra_contract("pattern_rectangle", ['high', 'low', 'left_window', 'right_window', 'history_window', 'points', 'slope_threshold']),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, points: int, slope_threshold: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_rectangle
        return pattern_rectangle(high, low, left_window, right_window, history_window, points, slope_threshold)


@register_operator(
    name="pattern_rising_channel",
    category="chart_pattern",
    canonical="pattern_rising_channel",
    source="polars_native_patterns")
@register_operator(
    name="pattern_rising_channel", category="chart_pattern", canonical="pattern_rising_channel",
    source="polars_native_patterns")
class PatternRisingChannel(SeriesOperator):
    """Canonical conversion wrapper with the authoritative pattern contract."""
    metadata = OperatorMetadata(
        name="pattern_rising_channel", category="chart_pattern",
        description="Canonical pattern_rising_channel with matching confirmed-history semantics.",
        param_names=['high', 'low', 'left_window', 'right_window', 'history_window', 'points', 'slope_threshold', 'parallel_tolerance'],
        **extra_contract("pattern_rising_channel", ['high', 'low', 'left_window', 'right_window', 'history_window', 'points', 'slope_threshold', 'parallel_tolerance']),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, points: int, slope_threshold: float, parallel_tolerance: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_rising_channel
        return pattern_rising_channel(high, low, left_window, right_window, history_window, points, slope_threshold, parallel_tolerance)


@register_operator(
    name="pattern_rising_wedge",
    category="chart_pattern",
    canonical="pattern_rising_wedge",
    source="polars_native_patterns")
@register_operator(
    name="pattern_rising_wedge", category="chart_pattern", canonical="pattern_rising_wedge",
    source="polars_native_patterns")
class PatternRisingWedge(SeriesOperator):
    """Canonical conversion wrapper with the authoritative pattern contract."""
    metadata = OperatorMetadata(
        name="pattern_rising_wedge", category="chart_pattern",
        description="Canonical pattern_rising_wedge with matching confirmed-history semantics.",
        param_names=['high', 'low', 'left_window', 'right_window', 'history_window', 'points', 'slope_threshold'],
        **extra_contract("pattern_rising_wedge", ['high', 'low', 'left_window', 'right_window', 'history_window', 'points', 'slope_threshold']),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, points: int, slope_threshold: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_rising_wedge
        return pattern_rising_wedge(high, low, left_window, right_window, history_window, points, slope_threshold)


@register_operator(
    name="pattern_rounding_bottom",
    category="chart_pattern",
    canonical="pattern_rounding_bottom",
    source="polars_native_patterns")
class PatternRoundingBottom(SeriesOperator):
    """Canonical continuous pattern score; sequential Polars/NumPy kernel."""
    metadata = OperatorMetadata(
        name="pattern_rounding_bottom", category="chart_pattern",
        description="Canonical pattern_rounding_bottom with matching confirmed-history semantics.",
        param_names=["close","window","min_fit"],
        **extra_contract("pattern_rounding_bottom", ["close","window","min_fit"]),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, close: pl.DataFrame, window: int, min_fit: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_rounding_bottom
        return pattern_rounding_bottom(close, window, min_fit)


@register_operator(
    name="pattern_rounding_top",
    category="chart_pattern",
    canonical="pattern_rounding_top",
    source="polars_native_patterns")
class PatternRoundingTop(SeriesOperator):
    """Canonical continuous pattern score; sequential Polars/NumPy kernel."""
    metadata = OperatorMetadata(
        name="pattern_rounding_top", category="chart_pattern",
        description="Canonical pattern_rounding_top with matching confirmed-history semantics.",
        param_names=["close","window","min_fit"],
        **extra_contract("pattern_rounding_top", ["close","window","min_fit"]),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, close: pl.DataFrame, window: int, min_fit: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_rounding_top
        return pattern_rounding_top(close, window, min_fit)


@register_operator(
    name="pattern_sym_triangle",
    category="chart_pattern",
    canonical="pattern_sym_triangle",
    source="polars_native_patterns")
@register_operator(
    name="pattern_sym_triangle", category="chart_pattern", canonical="pattern_sym_triangle",
    source="polars_native_patterns")
class PatternSymTriangle(SeriesOperator):
    """Canonical conversion wrapper with the authoritative pattern contract."""
    metadata = OperatorMetadata(
        name="pattern_sym_triangle", category="chart_pattern",
        description="Canonical pattern_sym_triangle with matching confirmed-history semantics.",
        param_names=['high', 'low', 'left_window', 'right_window', 'history_window', 'points', 'slope_threshold'],
        **extra_contract("pattern_sym_triangle", ['high', 'low', 'left_window', 'right_window', 'history_window', 'points', 'slope_threshold']),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, points: int, slope_threshold: float, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_sym_triangle
        return pattern_sym_triangle(high, low, left_window, right_window, history_window, points, slope_threshold)


@register_operator(
    name="pattern_triple_bottom",
    category="chart_pattern",
    canonical="pattern_triple_bottom",
    source="polars_native_patterns")
class PatternTripleBottom(SeriesOperator):
    """Canonical continuous pattern score; sequential Polars/NumPy kernel."""
    metadata = OperatorMetadata(
        name="pattern_triple_bottom", category="chart_pattern",
        description="Canonical pattern_triple_bottom with matching confirmed-history semantics.",
        param_names=["high","low","left_window","right_window","history_window","tolerance","min_depth","min_spacing","max_spacing"],
        **extra_contract("pattern_triple_bottom", ["high","low","left_window","right_window","history_window","tolerance","min_depth","min_spacing","max_spacing"]),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, tolerance: float, min_depth: float, min_spacing: int, max_spacing: int, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_triple_bottom
        return pattern_triple_bottom(high, low, left_window, right_window, history_window, tolerance, min_depth, min_spacing, max_spacing)


@register_operator(
    name="pattern_triple_top",
    category="chart_pattern",
    canonical="pattern_triple_top",
    source="polars_native_patterns")
class PatternTripleTop(SeriesOperator):
    """Canonical continuous pattern score; sequential Polars/NumPy kernel."""
    metadata = OperatorMetadata(
        name="pattern_triple_top", category="chart_pattern",
        description="Canonical pattern_triple_top with matching confirmed-history semantics.",
        param_names=["high","low","left_window","right_window","history_window","tolerance","min_depth","min_spacing","max_spacing"],
        **extra_contract("pattern_triple_top", ["high","low","left_window","right_window","history_window","tolerance","min_depth","min_spacing","max_spacing"]),
    )

    _contract_callable = property(_resolved_pattern_kernel)

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, left_window: int, right_window: int, history_window: int, tolerance: float, min_depth: float, min_spacing: int, max_spacing: int, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.price_volume.polars_structure import pattern_triple_top
        return pattern_triple_top(high, low, left_window, right_window, history_window, tolerance, min_depth, min_spacing, max_spacing)
