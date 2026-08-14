# -*- coding: utf-8 -*-
"""
Chart Pattern Operators - Polars Native

Implements chart pattern recognition using pure Polars/numpy.
27 pattern_* operators for technical analysis pattern detection.
"""
from __future__ import annotations
import polars as pl
import numpy as np
from cleaned_operators.base_polars import (
    SeriesOperator,
    OperatorMetadata,
    register_operator,
    PANEL_SKIP_COLUMNS,
)


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
    """Bearish 1-2-3 pattern: high, higher high, lower high."""

    metadata = OperatorMetadata(
        name="pattern_123_bear",
        category="chart_pattern",
        description="Bearish 1-2-3 reversal pattern",
        param_names=["high", "window"],
        param_types={"high": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, high: pl.DataFrame, window: int = 10, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        h_vals = high.select(cols).to_numpy()
        result = np.zeros_like(h_vals)

        for i in range(window, len(h_vals)):
            window_data = h_vals[i-window:i+1]
            pivots, _ = _find_pivots(window_data)
            pivot_indices = np.where(pivots)[0]

            # Need at least 3 pivot highs
            if len(pivot_indices) >= 3:
                p1, p2, p3 = pivot_indices[-3:]
                h1, h2, h3 = window_data[p1], window_data[p2], window_data[p3]
                # 1-2-3 bear: h2 > h1 and h3 < h2
                if h2 > h1 and h3 < h2:
                    result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="pattern_123_bull",
    category="chart_pattern",
    canonical="pattern_123_bull",
    source="polars_native_patterns")
class Pattern123Bull(SeriesOperator):
    """Bullish 1-2-3 pattern: low, lower low, higher low."""

    metadata = OperatorMetadata(
        name="pattern_123_bull",
        category="chart_pattern",
        description="Bullish 1-2-3 reversal pattern",
        param_names=["low", "window"],
        param_types={"low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, low: pl.DataFrame, window: int = 10, **kwargs) -> pl.DataFrame:
        cols = [c for c in low.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return low

        l_vals = low.select(cols).to_numpy()
        result = np.zeros_like(l_vals)

        for i in range(window, len(l_vals)):
            window_data = l_vals[i-window:i+1]
            _, pivots = _find_pivots(window_data)
            pivot_indices = np.where(pivots)[0]

            if len(pivot_indices) >= 3:
                p1, p2, p3 = pivot_indices[-3:]
                l1, l2, l3 = window_data[p1], window_data[p2], window_data[p3]
                # 1-2-3 bull: l2 < l1 and l3 > l2
                if l2 < l1 and l3 > l2:
                    result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, low)


@register_operator(
    name="pattern_ascending_triangle",
    category="chart_pattern",
    canonical="pattern_ascending_triangle",
    source="polars_native_patterns")
class PatternAscendingTriangle(SeriesOperator):
    """Ascending triangle: flat resistance, rising support."""

    metadata = OperatorMetadata(
        name="pattern_ascending_triangle",
        category="chart_pattern",
        description="Ascending triangle consolidation pattern",
        param_names=["high", "low", "window"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        h_vals = high.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        result = np.zeros_like(h_vals)

        for i in range(window, len(h_vals)):
            h_window = h_vals[i-window:i+1]
            l_window = l_vals[i-window:i+1]
            
            # Flat resistance: highs clustered around max
            resistance = np.max(h_window)
            h_std = np.std(h_window[-5:])  # Recent highs
            
            # Rising support: positive slope in lows
            x = np.arange(len(l_window))
            l_slope = np.polyfit(x, l_window, 1)[0] if len(x) > 1 else 0
            
            if h_std < 0.01 * resistance and l_slope > 0:
                result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="pattern_bear_flag",
    category="chart_pattern",
    canonical="pattern_bear_flag",
    source="polars_native_patterns")
class PatternBearFlag(SeriesOperator):
    """Bear flag: downtrend with upward consolidation."""

    metadata = OperatorMetadata(
        name="pattern_bear_flag",
        category="chart_pattern",
        description="Bearish flag continuation pattern",
        param_names=["close", "window"],
        param_types={"close": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, close: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in close.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return close

        c_vals = close.select(cols).to_numpy()
        result = np.zeros_like(c_vals)

        for i in range(window, len(c_vals)):
            # Flagpole: sharp decline before flag
            pole = c_vals[i-window:i-window//2]
            pole_slope = np.polyfit(np.arange(len(pole)), pole, 1)[0] if len(pole) > 1 else 0
            
            # Flag: slight upward drift
            flag = c_vals[i-window//2:i+1]
            flag_slope = np.polyfit(np.arange(len(flag)), flag, 1)[0] if len(flag) > 1 else 0
            
            if pole_slope < -0.01 and 0 < flag_slope < 0.005:
                result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, close)


@register_operator(
    name="pattern_bear_pennant",
    category="chart_pattern",
    canonical="pattern_bear_pennant",
    source="polars_native_patterns")
class PatternBearPennant(SeriesOperator):
    """Bear pennant: downtrend with converging consolidation."""

    metadata = OperatorMetadata(
        name="pattern_bear_pennant",
        category="chart_pattern",
        description="Bearish pennant continuation pattern",
        param_names=["high", "low", "window"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        h_vals = high.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        result = np.zeros_like(h_vals)

        for i in range(window, len(h_vals)):
            # Pennant: converging range
            ranges = h_vals[i-window//2:i+1] - l_vals[i-window//2:i+1]
            if len(ranges) > 2 and ranges[-1] < ranges[0] * 0.5:
                result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="pattern_breakdown_retest",
    category="chart_pattern",
    canonical="pattern_breakdown_retest",
    source="polars_native_patterns")
class PatternBreakdownRetest(SeriesOperator):
    """Price breaks support then retests it from below."""

    metadata = OperatorMetadata(
        name="pattern_breakdown_retest",
        category="chart_pattern",
        description="Breakdown followed by retest of support",
        param_names=["close", "low", "window"],
        param_types={"close": pl.DataFrame, "low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, close: pl.DataFrame, low: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in close.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return close

        c_vals = close.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        result = np.zeros_like(c_vals)

        for i in range(window + 5, len(c_vals)):
            # Support from earlier window
            support = np.min(l_vals[i-window:i-5])
            # Recent breakdown and retest
            recent_low = np.min(l_vals[i-5:i])
            if recent_low < support * 0.99 and c_vals[i] > support * 0.98:
                result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, close)


@register_operator(
    name="pattern_breakout_retest",
    category="chart_pattern",
    canonical="pattern_breakout_retest",
    source="polars_native_patterns")
class PatternBreakoutRetest(SeriesOperator):
    """Price breaks resistance then retests it from above."""

    metadata = OperatorMetadata(
        name="pattern_breakout_retest",
        category="chart_pattern",
        description="Breakout followed by retest of resistance",
        param_names=["close", "high", "window"],
        param_types={"close": pl.DataFrame, "high": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, close: pl.DataFrame, high: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in close.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return close

        c_vals = close.select(cols).to_numpy()
        h_vals = high.select(cols).to_numpy()
        result = np.zeros_like(c_vals)

        for i in range(window + 5, len(c_vals)):
            resistance = np.max(h_vals[i-window:i-5])
            recent_high = np.max(h_vals[i-5:i])
            if recent_high > resistance * 1.01 and c_vals[i] < resistance * 1.02:
                result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, close)


@register_operator(
    name="pattern_broadening",
    category="chart_pattern",
    canonical="pattern_broadening",
    source="polars_native_patterns")
class PatternBroadening(SeriesOperator):
    """Broadening pattern: expanding volatility range."""

    metadata = OperatorMetadata(
        name="pattern_broadening",
        category="chart_pattern",
        description="Broadening/megaphone pattern with expanding range",
        param_names=["high", "low", "window"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        h_vals = high.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        result = np.zeros_like(h_vals)

        for i in range(window, len(h_vals)):
            ranges = h_vals[i-window:i+1] - l_vals[i-window:i+1]
            # Expanding range: recent > early
            if len(ranges) > 2 and ranges[-1] > ranges[0] * 1.5:
                result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="pattern_bull_flag",
    category="chart_pattern",
    canonical="pattern_bull_flag",
    source="polars_native_patterns")
class PatternBullFlag(SeriesOperator):
    """Bull flag: uptrend with downward consolidation."""

    metadata = OperatorMetadata(
        name="pattern_bull_flag",
        category="chart_pattern",
        description="Bullish flag continuation pattern",
        param_names=["close", "window"],
        param_types={"close": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, close: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in close.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return close

        c_vals = close.select(cols).to_numpy()
        result = np.zeros_like(c_vals)

        for i in range(window, len(c_vals)):
            pole = c_vals[i-window:i-window//2]
            pole_slope = np.polyfit(np.arange(len(pole)), pole, 1)[0] if len(pole) > 1 else 0
            
            flag = c_vals[i-window//2:i+1]
            flag_slope = np.polyfit(np.arange(len(flag)), flag, 1)[0] if len(flag) > 1 else 0
            
            if pole_slope > 0.01 and -0.005 < flag_slope < 0:
                result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, close)


@register_operator(
    name="pattern_bull_pennant",
    category="chart_pattern",
    canonical="pattern_bull_pennant",
    source="polars_native_patterns")
class PatternBullPennant(SeriesOperator):
    """Bull pennant: uptrend with converging consolidation."""

    metadata = OperatorMetadata(
        name="pattern_bull_pennant",
        category="chart_pattern",
        description="Bullish pennant continuation pattern",
        param_names=["high", "low", "window"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        h_vals = high.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        result = np.zeros_like(h_vals)

        for i in range(window, len(h_vals)):
            ranges = h_vals[i-window//2:i+1] - l_vals[i-window//2:i+1]
            if len(ranges) > 2 and ranges[-1] < ranges[0] * 0.5:
                result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="pattern_cup",
    category="chart_pattern",
    canonical="pattern_cup",
    source="polars_native_patterns")
class PatternCup(SeriesOperator):
    """Cup pattern: U-shaped recovery."""

    metadata = OperatorMetadata(
        name="pattern_cup",
        category="chart_pattern",
        description="Cup pattern (U-shaped bottom)",
        param_names=["close", "window"],
        param_types={"close": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, close: pl.DataFrame, window: int = 30, **kwargs) -> pl.DataFrame:
        cols = [c for c in close.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return close

        c_vals = close.select(cols).to_numpy()
        result = np.zeros_like(c_vals)

        for i in range(window, len(c_vals)):
            window_data = c_vals[i-window:i+1]
            mid = len(window_data) // 2
            # U-shape: edges higher than middle
            left_avg = np.mean(window_data[:window//4])
            bottom_avg = np.mean(window_data[mid-2:mid+2])
            right_avg = np.mean(window_data[-window//4:])
            
            if bottom_avg < left_avg * 0.95 and right_avg > bottom_avg * 1.05:
                result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, close)


@register_operator(
    name="pattern_cup_handle",
    category="chart_pattern",
    canonical="pattern_cup_handle",
    source="polars_native_patterns")
class PatternCupHandle(SeriesOperator):
    """Cup and handle: U-shaped recovery with small dip."""

    metadata = OperatorMetadata(
        name="pattern_cup_handle",
        category="chart_pattern",
        description="Cup and handle bullish continuation",
        param_names=["close", "window"],
        param_types={"close": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, close: pl.DataFrame, window: int = 40, **kwargs) -> pl.DataFrame:
        cols = [c for c in close.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return close

        c_vals = close.select(cols).to_numpy()
        result = np.zeros_like(c_vals)

        # TODO: Implement cup detection followed by handle (small pullback)
        # For now, simplified version
        for i in range(window, len(c_vals)):
            window_data = c_vals[i-window:i+1]
            # Detect cup in first 75%, handle in last 25%
            cup_end = int(len(window_data) * 0.75)
            handle = window_data[cup_end:]
            if len(handle) > 2 and handle[-1] < handle[0] * 0.98:
                result[i] = 0.5  # Partial confidence

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, close)


@register_operator(
    name="pattern_descending_triangle",
    category="chart_pattern",
    canonical="pattern_descending_triangle",
    source="polars_native_patterns")
class PatternDescendingTriangle(SeriesOperator):
    """Descending triangle: flat support, falling resistance."""

    metadata = OperatorMetadata(
        name="pattern_descending_triangle",
        category="chart_pattern",
        description="Descending triangle consolidation pattern",
        param_names=["high", "low", "window"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        h_vals = high.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        result = np.zeros_like(h_vals)

        for i in range(window, len(h_vals)):
            h_window = h_vals[i-window:i+1]
            l_window = l_vals[i-window:i+1]
            
            # Flat support
            support = np.min(l_window)
            l_std = np.std(l_window[-5:])
            
            # Falling resistance
            x = np.arange(len(h_window))
            h_slope = np.polyfit(x, h_window, 1)[0] if len(x) > 1 else 0
            
            if l_std < 0.01 * support and h_slope < 0:
                result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="pattern_double_bottom",
    category="chart_pattern",
    canonical="pattern_double_bottom",
    source="polars_native_patterns")
class PatternDoubleBottom(SeriesOperator):
    """Double bottom: two similar lows (W pattern)."""

    metadata = OperatorMetadata(
        name="pattern_double_bottom",
        category="chart_pattern",
        description="Double bottom bullish reversal",
        param_names=["low", "window"],
        param_types={"low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, low: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in low.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return low

        l_vals = low.select(cols).to_numpy()
        result = np.zeros_like(l_vals)

        for i in range(window, len(l_vals)):
            window_data = l_vals[i-window:i+1]
            _, pivots = _find_pivots(window_data)
            pivot_indices = np.where(pivots)[0]

            if len(pivot_indices) >= 2:
                l1, l2 = window_data[pivot_indices[-2]], window_data[pivot_indices[-1]]
                # Similar lows
                if abs(l1 - l2) < 0.02 * l1:
                    result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, low)


@register_operator(
    name="pattern_double_top",
    category="chart_pattern",
    canonical="pattern_double_top",
    source="polars_native_patterns")
class PatternDoubleTop(SeriesOperator):
    """Double top: two similar highs (M pattern)."""

    metadata = OperatorMetadata(
        name="pattern_double_top",
        category="chart_pattern",
        description="Double top bearish reversal",
        param_names=["high", "window"],
        param_types={"high": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, high: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        h_vals = high.select(cols).to_numpy()
        result = np.zeros_like(h_vals)

        for i in range(window, len(h_vals)):
            window_data = h_vals[i-window:i+1]
            pivots, _ = _find_pivots(window_data)
            pivot_indices = np.where(pivots)[0]

            if len(pivot_indices) >= 2:
                h1, h2 = window_data[pivot_indices[-2]], window_data[pivot_indices[-1]]
                if abs(h1 - h2) < 0.02 * h1:
                    result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="pattern_falling_channel",
    category="chart_pattern",
    canonical="pattern_falling_channel",
    source="polars_native_patterns")
class PatternFallingChannel(SeriesOperator):
    """Falling channel: parallel downtrend lines."""

    metadata = OperatorMetadata(
        name="pattern_falling_channel",
        category="chart_pattern",
        description="Falling channel with parallel support/resistance",
        param_names=["high", "low", "window"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        h_vals = high.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        result = np.zeros_like(h_vals)

        for i in range(window, len(h_vals)):
            h_window = h_vals[i-window:i+1]
            l_window = l_vals[i-window:i+1]
            x = np.arange(len(h_window))
            
            h_slope = np.polyfit(x, h_window, 1)[0] if len(x) > 1 else 0
            l_slope = np.polyfit(x, l_window, 1)[0] if len(x) > 1 else 0
            
            # Both slopes negative and similar
            if h_slope < -0.001 and abs(h_slope - l_slope) < 0.001:
                result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="pattern_falling_wedge",
    category="chart_pattern",
    canonical="pattern_falling_wedge",
    source="polars_native_patterns")
class PatternFallingWedge(SeriesOperator):
    """Falling wedge: converging downtrend (bullish reversal)."""

    metadata = OperatorMetadata(
        name="pattern_falling_wedge",
        category="chart_pattern",
        description="Falling wedge bullish reversal pattern",
        param_names=["high", "low", "window"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        h_vals = high.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        result = np.zeros_like(h_vals)

        for i in range(window, len(h_vals)):
            h_window = h_vals[i-window:i+1]
            l_window = l_vals[i-window:i+1]
            x = np.arange(len(h_window))
            
            h_slope = np.polyfit(x, h_window, 1)[0] if len(x) > 1 else 0
            l_slope = np.polyfit(x, l_window, 1)[0] if len(x) > 1 else 0
            
            # Both slopes negative, but converging (range shrinking)
            ranges = h_window - l_window
            if h_slope < 0 and l_slope < 0 and ranges[-1] < ranges[0] * 0.6:
                result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="pattern_head_shoulders",
    category="chart_pattern",
    canonical="pattern_head_shoulders",
    source="polars_native_patterns")
class PatternHeadShoulders(SeriesOperator):
    """Head and shoulders: bearish reversal with 3 peaks."""

    metadata = OperatorMetadata(
        name="pattern_head_shoulders",
        category="chart_pattern",
        description="Head and shoulders bearish reversal",
        param_names=["high", "window"],
        param_types={"high": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, high: pl.DataFrame, window: int = 30, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        h_vals = high.select(cols).to_numpy()
        result = np.zeros_like(h_vals)

        for i in range(window, len(h_vals)):
            window_data = h_vals[i-window:i+1]
            pivots, _ = _find_pivots(window_data)
            pivot_indices = np.where(pivots)[0]

            if len(pivot_indices) >= 3:
                left, head, right = pivot_indices[-3:]
                h_left = window_data[left]
                h_head = window_data[head]
                h_right = window_data[right]
                
                # Head higher than shoulders, shoulders similar height
                if h_head > h_left and h_head > h_right and abs(h_left - h_right) < 0.05 * h_head:
                    result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="pattern_inverse_head_shoulders",
    category="chart_pattern",
    canonical="pattern_inverse_head_shoulders",
    source="polars_native_patterns")
class PatternInverseHeadShoulders(SeriesOperator):
    """Inverse head and shoulders: bullish reversal with 3 troughs."""

    metadata = OperatorMetadata(
        name="pattern_inverse_head_shoulders",
        category="chart_pattern",
        description="Inverse head and shoulders bullish reversal",
        param_names=["low", "window"],
        param_types={"low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, low: pl.DataFrame, window: int = 30, **kwargs) -> pl.DataFrame:
        cols = [c for c in low.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return low

        l_vals = low.select(cols).to_numpy()
        result = np.zeros_like(l_vals)

        for i in range(window, len(l_vals)):
            window_data = l_vals[i-window:i+1]
            _, pivots = _find_pivots(window_data)
            pivot_indices = np.where(pivots)[0]

            if len(pivot_indices) >= 3:
                left, head, right = pivot_indices[-3:]
                l_left = window_data[left]
                l_head = window_data[head]
                l_right = window_data[right]
                
                # Head lower than shoulders, shoulders similar depth
                if l_head < l_left and l_head < l_right and abs(l_left - l_right) < 0.05 * l_head:
                    result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, low)


@register_operator(
    name="pattern_rectangle",
    category="chart_pattern",
    canonical="pattern_rectangle",
    source="polars_native_patterns")
class PatternRectangle(SeriesOperator):
    """Rectangle: horizontal support and resistance."""

    metadata = OperatorMetadata(
        name="pattern_rectangle",
        category="chart_pattern",
        description="Rectangle consolidation pattern",
        param_names=["high", "low", "window"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        h_vals = high.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        result = np.zeros_like(h_vals)

        for i in range(window, len(h_vals)):
            h_window = h_vals[i-window:i+1]
            l_window = l_vals[i-window:i+1]
            
            # Flat highs and lows
            h_std = np.std(h_window[-10:]) / np.mean(h_window[-10:])
            l_std = np.std(l_window[-10:]) / np.mean(l_window[-10:])
            
            if h_std < 0.02 and l_std < 0.02:
                result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="pattern_rising_channel",
    category="chart_pattern",
    canonical="pattern_rising_channel",
    source="polars_native_patterns")
class PatternRisingChannel(SeriesOperator):
    """Rising channel: parallel uptrend lines."""

    metadata = OperatorMetadata(
        name="pattern_rising_channel",
        category="chart_pattern",
        description="Rising channel with parallel support/resistance",
        param_names=["high", "low", "window"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        h_vals = high.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        result = np.zeros_like(h_vals)

        for i in range(window, len(h_vals)):
            h_window = h_vals[i-window:i+1]
            l_window = l_vals[i-window:i+1]
            x = np.arange(len(h_window))
            
            h_slope = np.polyfit(x, h_window, 1)[0] if len(x) > 1 else 0
            l_slope = np.polyfit(x, l_window, 1)[0] if len(x) > 1 else 0
            
            # Both slopes positive and similar
            if h_slope > 0.001 and abs(h_slope - l_slope) < 0.001:
                result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="pattern_rising_wedge",
    category="chart_pattern",
    canonical="pattern_rising_wedge",
    source="polars_native_patterns")
class PatternRisingWedge(SeriesOperator):
    """Rising wedge: converging uptrend (bearish reversal)."""

    metadata = OperatorMetadata(
        name="pattern_rising_wedge",
        category="chart_pattern",
        description="Rising wedge bearish reversal pattern",
        param_names=["high", "low", "window"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        h_vals = high.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        result = np.zeros_like(h_vals)

        for i in range(window, len(h_vals)):
            h_window = h_vals[i-window:i+1]
            l_window = l_vals[i-window:i+1]
            x = np.arange(len(h_window))
            
            h_slope = np.polyfit(x, h_window, 1)[0] if len(x) > 1 else 0
            l_slope = np.polyfit(x, l_window, 1)[0] if len(x) > 1 else 0
            
            # Both slopes positive, but converging
            ranges = h_window - l_window
            if h_slope > 0 and l_slope > 0 and ranges[-1] < ranges[0] * 0.6:
                result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="pattern_rounding_bottom",
    category="chart_pattern",
    canonical="pattern_rounding_bottom",
    source="polars_native_patterns")
class PatternRoundingBottom(SeriesOperator):
    """Rounding bottom: gradual U-shaped recovery."""

    metadata = OperatorMetadata(
        name="pattern_rounding_bottom",
        category="chart_pattern",
        description="Rounding bottom bullish reversal",
        param_names=["close", "window"],
        param_types={"close": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, close: pl.DataFrame, window: int = 30, **kwargs) -> pl.DataFrame:
        cols = [c for c in close.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return close

        c_vals = close.select(cols).to_numpy()
        result = np.zeros_like(c_vals)

        for i in range(window, len(c_vals)):
            window_data = c_vals[i-window:i+1]
            x = np.arange(len(window_data))
            
            # Fit quadratic: positive coefficient = U-shape
            try:
                coeffs = np.polyfit(x, window_data, 2)
                if coeffs[0] > 0:  # Upward opening parabola
                    result[i] = 1.0
            except:
                pass

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, close)


@register_operator(
    name="pattern_rounding_top",
    category="chart_pattern",
    canonical="pattern_rounding_top",
    source="polars_native_patterns")
class PatternRoundingTop(SeriesOperator):
    """Rounding top: gradual inverted U-shaped decline."""

    metadata = OperatorMetadata(
        name="pattern_rounding_top",
        category="chart_pattern",
        description="Rounding top bearish reversal",
        param_names=["close", "window"],
        param_types={"close": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, close: pl.DataFrame, window: int = 30, **kwargs) -> pl.DataFrame:
        cols = [c for c in close.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return close

        c_vals = close.select(cols).to_numpy()
        result = np.zeros_like(c_vals)

        for i in range(window, len(c_vals)):
            window_data = c_vals[i-window:i+1]
            x = np.arange(len(window_data))
            
            try:
                coeffs = np.polyfit(x, window_data, 2)
                if coeffs[0] < 0:  # Downward opening parabola
                    result[i] = 1.0
            except:
                pass

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, close)


@register_operator(
    name="pattern_sym_triangle",
    category="chart_pattern",
    canonical="pattern_sym_triangle",
    source="polars_native_patterns")
class PatternSymTriangle(SeriesOperator):
    """Symmetric triangle: converging highs and lows."""

    metadata = OperatorMetadata(
        name="pattern_sym_triangle",
        category="chart_pattern",
        description="Symmetric triangle consolidation pattern",
        param_names=["high", "low", "window"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        h_vals = high.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        result = np.zeros_like(h_vals)

        for i in range(window, len(h_vals)):
            h_window = h_vals[i-window:i+1]
            l_window = l_vals[i-window:i+1]
            x = np.arange(len(h_window))
            
            h_slope = np.polyfit(x, h_window, 1)[0] if len(x) > 1 else 0
            l_slope = np.polyfit(x, l_window, 1)[0] if len(x) > 1 else 0
            
            # Converging: highs falling, lows rising
            if h_slope < 0 and l_slope > 0:
                result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="pattern_triple_bottom",
    category="chart_pattern",
    canonical="pattern_triple_bottom",
    source="polars_native_patterns")
class PatternTripleBottom(SeriesOperator):
    """Triple bottom: three similar lows."""

    metadata = OperatorMetadata(
        name="pattern_triple_bottom",
        category="chart_pattern",
        description="Triple bottom bullish reversal",
        param_names=["low", "window"],
        param_types={"low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, low: pl.DataFrame, window: int = 30, **kwargs) -> pl.DataFrame:
        cols = [c for c in low.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return low

        l_vals = low.select(cols).to_numpy()
        result = np.zeros_like(l_vals)

        for i in range(window, len(l_vals)):
            window_data = l_vals[i-window:i+1]
            _, pivots = _find_pivots(window_data)
            pivot_indices = np.where(pivots)[0]

            if len(pivot_indices) >= 3:
                l1, l2, l3 = window_data[pivot_indices[-3]], window_data[pivot_indices[-2]], window_data[pivot_indices[-1]]
                # Three similar lows
                avg_low = (l1 + l2 + l3) / 3
                if abs(l1 - avg_low) < 0.02 * avg_low and abs(l2 - avg_low) < 0.02 * avg_low and abs(l3 - avg_low) < 0.02 * avg_low:
                    result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, low)


@register_operator(
    name="pattern_triple_top",
    category="chart_pattern",
    canonical="pattern_triple_top",
    source="polars_native_patterns")
class PatternTripleTop(SeriesOperator):
    """Triple top: three similar highs."""

    metadata = OperatorMetadata(
        name="pattern_triple_top",
        category="chart_pattern",
        description="Triple top bearish reversal",
        param_names=["high", "window"],
        param_types={"high": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, high: pl.DataFrame, window: int = 30, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        h_vals = high.select(cols).to_numpy()
        result = np.zeros_like(h_vals)

        for i in range(window, len(h_vals)):
            window_data = h_vals[i-window:i+1]
            pivots, _ = _find_pivots(window_data)
            pivot_indices = np.where(pivots)[0]

            if len(pivot_indices) >= 3:
                h1, h2, h3 = window_data[pivot_indices[-3]], window_data[pivot_indices[-2]], window_data[pivot_indices[-1]]
                avg_high = (h1 + h2 + h3) / 3
                if abs(h1 - avg_high) < 0.02 * avg_high and abs(h2 - avg_high) < 0.02 * avg_high and abs(h3 - avg_high) < 0.02 * avg_high:
                    result[i] = 1.0

        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, high)
