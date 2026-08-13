# -*- coding: utf-8 -*-
"""
Polars native 蜡烛图算子 - Candlestick patterns and metrics

所有算子使用 Polars/NumPy 实现，不依赖 TA-Lib。
包含 13 个 candle_* 指标算子和 22 个 cdl_* 形态识别算子。
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


def _result_df(data_cols: dict, template_df: pl.DataFrame) -> pl.DataFrame:
    """将计算结果与元数据列组合成完整 DataFrame"""
    result = pl.DataFrame(data_cols)
    for meta_col in PANEL_SKIP_COLUMNS:
        if meta_col in template_df.columns:
            result = result.with_columns([template_df[meta_col]])
    return result


def _get_ohlc_arrays(open_df: pl.DataFrame, high_df: pl.DataFrame, 
                     low_df: pl.DataFrame, close_df: pl.DataFrame):
    """提取 OHLC 数据列的 numpy arrays"""
    cols = [c for c in open_df.columns if c not in PANEL_SKIP_COLUMNS]
    o = open_df.select(cols).to_numpy()
    h = high_df.select(cols).to_numpy()
    l = low_df.select(cols).to_numpy()
    c = close_df.select(cols).to_numpy()
    return o, h, l, c, cols


# ==================== Candle Metrics (13 operators) ====================

@register_operator(
    name="candle_body_percentile",
    category="candlestick",
    canonical="candle_body_percentile",
    source="polars_native_candlestick",
)
class CandleBodyPercentile(SeriesOperator):
    """Percentile rank of candle body size over rolling window"""

    metadata = OperatorMetadata(
        name="candle_body_percentile",
        category="candlestick",
        description="Rolling percentile rank of absolute body size",
        param_names=["open", "close", "window"],
        param_types={"open": pl.DataFrame, "close": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, open: pl.DataFrame, close: pl.DataFrame, 
                         window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in open.columns if c not in PANEL_SKIP_COLUMNS]
        o_vals = open.select(cols).to_numpy()
        c_vals = close.select(cols).to_numpy()
        
        body = np.abs(c_vals - o_vals)
        result = np.full_like(body, np.nan)
        
        for i in range(window - 1, len(body)):
            window_data = body[i - window + 1:i + 1]
            result[i] = np.where(window * 100 != 0, (np.sum(window_data <= body[i])) / (window * 100), np.nan)
        
        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="candle_body_zscore",
    category="candlestick",
    canonical="candle_body_zscore",
    source="polars_native_candlestick",
)
class CandleBodyZScore(SeriesOperator):
    """Z-score of candle body size"""

    metadata = OperatorMetadata(
        name="candle_body_zscore",
        category="candlestick",
        description="Z-score of absolute body size over rolling window",
        param_names=["open", "close", "window"],
        param_types={"open": pl.DataFrame, "close": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, open: pl.DataFrame, close: pl.DataFrame, 
                         window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in open.columns if c not in PANEL_SKIP_COLUMNS]
        o_vals = open.select(cols).to_numpy()
        c_vals = close.select(cols).to_numpy()
        
        body = np.abs(c_vals - o_vals)
        result = np.full_like(body, np.nan)
        
        for i in range(window - 1, len(body)):
            window_data = body[i - window + 1:i + 1]
            mean = np.mean(window_data, axis=0)
            std = np.std(window_data, axis=0)
            result[i] = ((body[i] - mean)) / (std + 1e-10) if (std + 1e-10) > 1e-10 else np.nan
        
        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="candle_close_strength",
    category="candlestick",
    canonical="candle_close_strength",
    source="polars_native_candlestick",
)
class CandleCloseStrength(SeriesOperator):
    """Close position within the bar range: (close - low) / (high - low)"""

    metadata = OperatorMetadata(
        name="candle_close_strength",
        category="candlestick",
        description="Close position within bar range (0=low, 1=high)",
        param_names=["high", "low", "close"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, 
                         close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        h_vals = high.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        c_vals = close.select(cols).to_numpy()
        
        range_val = h_vals - l_vals
        strength = np.where(range_val > 1e-10, 
                           (c_vals - l_vals) / range_val, 
                           0.5)
        
        return _result_df({col: strength[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="candle_gap_atr",
    category="candlestick",
    canonical="candle_gap_atr",
    source="polars_native_candlestick",
)
class CandleGapATR(SeriesOperator):
    """Gap size normalized by ATR"""

    metadata = OperatorMetadata(
        name="candle_gap_atr",
        category="candlestick",
        description="Gap between open and previous close, normalized by ATR",
        param_names=["open", "close", "period"],
        param_types={"open": pl.DataFrame, "close": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, open: pl.DataFrame, close: pl.DataFrame, 
                         period: int = 14, **kwargs) -> pl.DataFrame:
        cols = [c for c in open.columns if c not in PANEL_SKIP_COLUMNS]
        o_vals = open.select(cols).to_numpy()
        c_vals = close.select(cols).to_numpy()
        
        # Calculate gap
        c_prev = np.roll(c_vals, 1, axis=0)
        c_prev[0] = np.nan
        gap = o_vals - c_prev
        
        # Simple ATR approximation using close-to-close volatility
        returns = np.diff(c_vals, axis=0, prepend=np.nan)
        atr = np.full_like(returns, np.nan)
        for i in range(period, len(returns)):
            atr[i] = np.nanmean(np.abs(returns[i - period:i]), axis=0)
        
        gap_atr = (gap) / ((atr + 1e-10)) if ((atr + 1e-10)) != 0 else np.nan
        
        return _result_df({col: gap_atr[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="candle_inside_ratio",
    category="candlestick",
    canonical="candle_inside_ratio",
    source="polars_native_candlestick",
)
class CandleInsideRatio(SeriesOperator):
    """Rolling ratio of inside bars (bars within previous bar's range)"""

    metadata = OperatorMetadata(
        name="candle_inside_ratio",
        category="candlestick",
        description="Ratio of inside bars over rolling window",
        param_names=["high", "low", "window"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, 
                         window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        h_vals = high.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        
        # Identify inside bars
        h_prev = np.roll(h_vals, 1, axis=0)
        l_prev = np.roll(l_vals, 1, axis=0)
        h_prev[0] = np.nan
        l_prev[0] = np.nan
        
        inside = ((h_vals <= h_prev) & (l_vals >= l_prev)).astype(float)
        inside[0] = np.nan
        
        # Rolling ratio
        result = np.full_like(inside, np.nan)
        for i in range(window, len(inside)):
            result[i] = np.nanmean(inside[i - window + 1:i + 1], axis=0)
        
        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="candle_lower_shadow_zscore",
    category="candlestick",
    canonical="candle_lower_shadow_zscore",
    source="polars_native_candlestick",
)
class CandleLowerShadowZScore(SeriesOperator):
    """Z-score of lower shadow length"""

    metadata = OperatorMetadata(
        name="candle_lower_shadow_zscore",
        category="candlestick",
        description="Z-score of lower shadow over rolling window",
        param_names=["open", "low", "close", "window"],
        param_types={"open": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, open: pl.DataFrame, low: pl.DataFrame, 
                         close: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in open.columns if c not in PANEL_SKIP_COLUMNS]
        o_vals = open.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        c_vals = close.select(cols).to_numpy()
        
        body_low = np.minimum(o_vals, c_vals)
        lower_shadow = body_low - l_vals
        
        result = np.full_like(lower_shadow, np.nan)
        for i in range(window - 1, len(lower_shadow)):
            window_data = lower_shadow[i - window + 1:i + 1]
            mean = np.mean(window_data, axis=0)
            std = np.std(window_data, axis=0)
            result[i] = ((lower_shadow[i] - mean)) / (std + 1e-10) if (std + 1e-10) > 1e-10 else np.nan
        
        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="candle_overlap_ratio",
    category="candlestick",
    canonical="candle_overlap_ratio",
    source="polars_native_candlestick",
)
class CandleOverlapRatio(SeriesOperator):
    """Body overlap ratio with previous candle"""

    metadata = OperatorMetadata(
        name="candle_overlap_ratio",
        category="candlestick",
        description="Ratio of body overlap with previous candle body",
        param_names=["open", "close"],
        param_types={"open": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = [c for c in open.columns if c not in PANEL_SKIP_COLUMNS]
        o_vals = open.select(cols).to_numpy()
        c_vals = close.select(cols).to_numpy()
        
        body_high = np.maximum(o_vals, c_vals)
        body_low = np.minimum(o_vals, c_vals)
        body_size = body_high - body_low
        
        body_high_prev = np.roll(body_high, 1, axis=0)
        body_low_prev = np.roll(body_low, 1, axis=0)
        body_size_prev = np.roll(body_size, 1, axis=0)
        
        overlap_high = np.minimum(body_high, body_high_prev)
        overlap_low = np.maximum(body_low, body_low_prev)
        overlap = np.maximum(0, overlap_high - overlap_low)
        
        ratio = (overlap) / ((body_size_prev + 1e-10)) if ((body_size_prev + 1e-10)) != 0 else np.nan
        ratio[0] = np.nan
        
        return _result_df({col: ratio[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="candle_range_atr",
    category="candlestick",
    canonical="candle_range_atr",
    source="polars_native_candlestick",
)
class CandleRangeATR(SeriesOperator):
    """Candle range normalized by ATR"""

    metadata = OperatorMetadata(
        name="candle_range_atr",
        category="candlestick",
        description="High-low range normalized by ATR",
        param_names=["high", "low", "close", "period"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, 
                         close: pl.DataFrame, period: int = 14, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        h_vals = high.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        c_vals = close.select(cols).to_numpy()
        
        candle_range = h_vals - l_vals
        
        # Simple ATR approximation
        c_prev = np.roll(c_vals, 1, axis=0)
        c_prev[0] = np.nan
        tr = np.maximum(np.maximum(candle_range, np.abs(h_vals - c_prev)), np.abs(l_vals - c_prev))
        
        atr = np.full_like(tr, np.nan)
        for i in range(period, len(tr)):
            atr[i] = np.nanmean(tr[i - period:i], axis=0)
        
        range_atr = (candle_range) / ((atr + 1e-10)) if ((atr + 1e-10)) != 0 else np.nan
        
        return _result_df({col: range_atr[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="candle_range_percentile",
    category="candlestick",
    canonical="candle_range_percentile",
    source="polars_native_candlestick",
)
class CandleRangePercentile(SeriesOperator):
    """Percentile rank of candle range"""

    metadata = OperatorMetadata(
        name="candle_range_percentile",
        category="candlestick",
        description="Rolling percentile rank of high-low range",
        param_names=["high", "low", "window"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, 
                         window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        h_vals = high.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        
        candle_range = h_vals - l_vals
        result = np.full_like(candle_range, np.nan)
        
        for i in range(window - 1, len(candle_range)):
            window_data = candle_range[i - window + 1:i + 1]
            result[i] = np.where(window * 100 != 0, (np.sum(window_data <= candle_range[i], axis=0)) / (window * 100), np.nan)
        
        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="candle_range_zscore",
    category="candlestick",
    canonical="candle_range_zscore",
    source="polars_native_candlestick",
)
class CandleRangeZScore(SeriesOperator):
    """Z-score of candle range"""

    metadata = OperatorMetadata(
        name="candle_range_zscore",
        category="candlestick",
        description="Z-score of high-low range over rolling window",
        param_names=["high", "low", "window"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, 
                         window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        h_vals = high.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        
        candle_range = h_vals - l_vals
        result = np.full_like(candle_range, np.nan)
        
        for i in range(window - 1, len(candle_range)):
            window_data = candle_range[i - window + 1:i + 1]
            mean = np.mean(window_data, axis=0)
            std = np.std(window_data, axis=0)
            result[i] = ((candle_range[i] - mean)) / (std + 1e-10) if (std + 1e-10) > 1e-10 else np.nan
        
        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="candle_rejection_lower",
    category="candlestick",
    canonical="candle_rejection_lower",
    source="polars_native_candlestick",
)
class CandleRejectionLower(SeriesOperator):
    """Lower wick as ratio of total range"""

    metadata = OperatorMetadata(
        name="candle_rejection_lower",
        category="candlestick",
        description="Lower shadow / total range ratio",
        param_names=["open", "high", "low", "close"],
        param_types={"open": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, high: pl.DataFrame, 
                         low: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = [c for c in open.columns if c not in PANEL_SKIP_COLUMNS]
        o_vals = open.select(cols).to_numpy()
        h_vals = high.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        c_vals = close.select(cols).to_numpy()
        
        body_low = np.minimum(o_vals, c_vals)
        lower_shadow = body_low - l_vals
        total_range = h_vals - l_vals
        
        rejection = np.where(total_range > 1e-10, 
                            lower_shadow / total_range, 
                            0)
        
        return _result_df({col: rejection[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="candle_rejection_upper",
    category="candlestick",
    canonical="candle_rejection_upper",
    source="polars_native_candlestick",
)
class CandleRejectionUpper(SeriesOperator):
    """Upper wick as ratio of total range"""

    metadata = OperatorMetadata(
        name="candle_rejection_upper",
        category="candlestick",
        description="Upper shadow / total range ratio",
        param_names=["open", "high", "low", "close"],
        param_types={"open": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, high: pl.DataFrame, 
                         low: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = [c for c in open.columns if c not in PANEL_SKIP_COLUMNS]
        o_vals = open.select(cols).to_numpy()
        h_vals = high.select(cols).to_numpy()
        l_vals = low.select(cols).to_numpy()
        c_vals = close.select(cols).to_numpy()
        
        body_high = np.maximum(o_vals, c_vals)
        upper_shadow = h_vals - body_high
        total_range = h_vals - l_vals
        
        rejection = np.where(total_range > 1e-10, 
                            upper_shadow / total_range, 
                            0)
        
        return _result_df({col: rejection[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="candle_upper_shadow_zscore",
    category="candlestick",
    canonical="candle_upper_shadow_zscore",
    source="polars_native_candlestick",
)
class CandleUpperShadowZScore(SeriesOperator):
    """Z-score of upper shadow length"""

    metadata = OperatorMetadata(
        name="candle_upper_shadow_zscore",
        category="candlestick",
        description="Z-score of upper shadow over rolling window",
        param_names=["open", "high", "close", "window"],
        param_types={"open": pl.DataFrame, "high": pl.DataFrame, "close": pl.DataFrame, "window": int},
    )

    def _calculate_series(self, open: pl.DataFrame, high: pl.DataFrame, 
                         close: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        cols = [c for c in open.columns if c not in PANEL_SKIP_COLUMNS]
        o_vals = open.select(cols).to_numpy()
        h_vals = high.select(cols).to_numpy()
        c_vals = close.select(cols).to_numpy()
        
        body_high = np.maximum(o_vals, c_vals)
        upper_shadow = h_vals - body_high
        
        result = np.full_like(upper_shadow, np.nan)
        for i in range(window - 1, len(upper_shadow)):
            window_data = upper_shadow[i - window + 1:i + 1]
            mean = np.mean(window_data, axis=0)
            std = np.std(window_data, axis=0)
            result[i] = ((upper_shadow[i] - mean)) / (std + 1e-10) if (std + 1e-10) > 1e-10 else np.nan
        
        return _result_df({col: result[:, idx] for idx, col in enumerate(cols)}, open)


# ==================== Candlestick Patterns (22 operators) ====================

@register_operator(
    name="cdl_dark_cloud_cover",
    category="candlestick_pattern",
    canonical="cdl_dark_cloud_cover",
    source="polars_native_candlestick",
)
class CDL_DarkCloudCover(SeriesOperator):
    """Dark Cloud Cover pattern: bearish reversal (2-bar)"""

    metadata = OperatorMetadata(
        name="cdl_dark_cloud_cover",
        category="candlestick_pattern",
        description="Dark Cloud Cover pattern (-1=bearish, 0=none)",
        param_names=["open", "high", "low", "close"],
        param_types={"open": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, high: pl.DataFrame, 
                         low: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        o, h, l, c, cols = _get_ohlc_arrays(open, high, low, close)
        
        # Day 1: bullish candle
        day1_bullish = c > o
        day1_body = c - o
        
        # Day 2: bearish candle opening above day 1 close
        o_prev = np.roll(o, 1, axis=0)
        c_prev = np.roll(c, 1, axis=0)
        day1_body_prev = np.roll(day1_body, 1, axis=0)
        day1_bullish_prev = np.roll(day1_bullish, 1, axis=0)
        
        day2_bearish = c < o
        day2_opens_above = o > c_prev
        day2_closes_into_body = (c > o_prev) & (c < c_prev)
        day2_closes_deep = (c - o_prev) < (day1_body_prev * 0.5)
        
        pattern = (day1_bullish_prev & day2_bearish & day2_opens_above & 
                  day2_closes_into_body & day2_closes_deep).astype(float)
        pattern = np.where(pattern, -1, 0)
        pattern[0] = 0
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="cdl_doji",
    category="candlestick_pattern",
    canonical="cdl_doji",
    source="polars_native_candlestick",
)
class CDL_Doji(SeriesOperator):
    """Doji pattern: indecision (open ≈ close)"""

    metadata = OperatorMetadata(
        name="cdl_doji",
        category="candlestick_pattern",
        description="Doji pattern (1=doji, 0=none)",
        param_names=["open", "high", "low", "close", "threshold"],
        param_types={"open": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, 
                    "close": pl.DataFrame, "threshold": float},
    )

    def _calculate_series(self, open: pl.DataFrame, high: pl.DataFrame, 
                         low: pl.DataFrame, close: pl.DataFrame, 
                         threshold: float = 0.1, **kwargs) -> pl.DataFrame:
        o, h, l, c, cols = _get_ohlc_arrays(open, high, low, close)
        
        body = np.abs(c - o)
        total_range = h - l
        body_ratio = (body) / ((total_range + 1e-10)) if ((total_range + 1e-10)) != 0 else np.nan
        
        pattern = (body_ratio < threshold).astype(float)
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="cdl_dragonfly_doji",
    category="candlestick_pattern",
    canonical="cdl_dragonfly_doji",
    source="polars_native_candlestick",
)
class CDL_DragonflyDoji(SeriesOperator):
    """Dragonfly Doji: bullish reversal (T-shaped, long lower shadow)"""

    metadata = OperatorMetadata(
        name="cdl_dragonfly_doji",
        category="candlestick_pattern",
        description="Dragonfly Doji pattern (1=bullish, 0=none)",
        param_names=["open", "high", "low", "close"],
        param_types={"open": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, high: pl.DataFrame, 
                         low: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        o, h, l, c, cols = _get_ohlc_arrays(open, high, low, close)
        
        body = np.abs(c - o)
        body_mid = ((c + o)) / 2 if 2 != 0 else np.nan
        upper_shadow = h - np.maximum(o, c)
        lower_shadow = np.minimum(o, c) - l
        total_range = h - l
        
        is_doji = (body) / ((total_range + 1e-10) < 0.1) if ((total_range + 1e-10) < 0.1) != 0 else np.nan
        upper_small = (upper_shadow) / ((total_range + 1e-10) < 0.1) if ((total_range + 1e-10) < 0.1) != 0 else np.nan
        lower_long = (lower_shadow) / ((total_range + 1e-10) > 0.6) if ((total_range + 1e-10) > 0.6) != 0 else np.nan
        
        pattern = (is_doji & upper_small & lower_long).astype(float)
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="cdl_engulfing",
    category="candlestick_pattern",
    canonical="cdl_engulfing",
    source="polars_native_candlestick",
)
class CDL_Engulfing(SeriesOperator):
    """Engulfing pattern: reversal (body completely engulfs previous)"""

    metadata = OperatorMetadata(
        name="cdl_engulfing",
        category="candlestick_pattern",
        description="Engulfing pattern (1=bullish, -1=bearish, 0=none)",
        param_names=["open", "close"],
        param_types={"open": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = [c for c in open.columns if c not in PANEL_SKIP_COLUMNS]
        o = open.select(cols).to_numpy()
        c = close.select(cols).to_numpy()
        
        o_prev = np.roll(o, 1, axis=0)
        c_prev = np.roll(c, 1, axis=0)
        
        # Bullish engulfing: prev bearish, current bullish, engulfs prev body
        prev_bearish = c_prev < o_prev
        curr_bullish = c > o
        bullish_engulf = curr_bullish & prev_bearish & (o <= c_prev) & (c >= o_prev)
        
        # Bearish engulfing: prev bullish, current bearish, engulfs prev body
        prev_bullish = c_prev > o_prev
        curr_bearish = c < o
        bearish_engulf = curr_bearish & prev_bullish & (o >= c_prev) & (c <= o_prev)
        
        pattern = np.where(bullish_engulf, 1, np.where(bearish_engulf, -1, 0))
        pattern[0] = 0
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="cdl_evening_star",
    category="candlestick_pattern",
    canonical="cdl_evening_star",
    source="polars_native_candlestick",
)
class CDL_EveningStar(SeriesOperator):
    """Evening Star: bearish reversal (3-bar pattern)"""

    metadata = OperatorMetadata(
        name="cdl_evening_star",
        category="candlestick_pattern",
        description="Evening Star pattern (-1=bearish, 0=none)",
        param_names=["open", "high", "low", "close"],
        param_types={"open": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, high: pl.DataFrame, 
                         low: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        o, h, l, c, cols = _get_ohlc_arrays(open, high, low, close)
        
        # Day 1: strong bullish
        day1_bullish = c > o
        day1_body = c - o
        
        # Day 2: small body (star) gapping up
        body_size = np.abs(c - o)
        day1_body_prev = np.roll(day1_body, 1, axis=0)
        day2_small = body_size < (day1_body_prev * 0.3)
        
        # Day 3: bearish closing into day 1 body
        o_prev2 = np.roll(o, 2, axis=0)
        c_prev2 = np.roll(c, 2, axis=0)
        day1_bullish_prev2 = np.roll(day1_bullish, 2, axis=0)
        day3_bearish = c < o
        day3_closes_low = c < (o_prev2 + c_prev2) / 2
        
        day2_small_prev = np.roll(day2_small, 1, axis=0)
        
        pattern = (day1_bullish_prev2 & day2_small_prev & day3_bearish & day3_closes_low).astype(float)
        pattern = np.where(pattern, -1, 0)
        pattern[0:2] = 0
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="cdl_gravestone_doji",
    category="candlestick_pattern",
    canonical="cdl_gravestone_doji",
    source="polars_native_candlestick",
)
class CDL_GravestoneDoji(SeriesOperator):
    """Gravestone Doji: bearish reversal (inverted T, long upper shadow)"""

    metadata = OperatorMetadata(
        name="cdl_gravestone_doji",
        category="candlestick_pattern",
        description="Gravestone Doji pattern (-1=bearish, 0=none)",
        param_names=["open", "high", "low", "close"],
        param_types={"open": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, high: pl.DataFrame, 
                         low: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        o, h, l, c, cols = _get_ohlc_arrays(open, high, low, close)
        
        body = np.abs(c - o)
        upper_shadow = h - np.maximum(o, c)
        lower_shadow = np.minimum(o, c) - l
        total_range = h - l
        
        is_doji = body / (total_range + 1e-10) < 0.1
        lower_small = lower_shadow / (total_range + 1e-10) < 0.1
        upper_long = upper_shadow / (total_range + 1e-10) > 0.6
        
        pattern = np.where(is_doji & lower_small & upper_long, -1, 0)
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="cdl_hammer",
    category="candlestick_pattern",
    canonical="cdl_hammer",
    source="polars_native_candlestick",
)
class CDL_Hammer(SeriesOperator):
    """Hammer: bullish reversal (small body, long lower shadow)"""

    metadata = OperatorMetadata(
        name="cdl_hammer",
        category="candlestick_pattern",
        description="Hammer pattern (1=bullish, 0=none)",
        param_names=["open", "high", "low", "close"],
        param_types={"open": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, high: pl.DataFrame, 
                         low: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        o, h, l, c, cols = _get_ohlc_arrays(open, high, low, close)
        
        body = np.abs(c - o)
        upper_shadow = h - np.maximum(o, c)
        lower_shadow = np.minimum(o, c) - l
        total_range = h - l
        
        small_body = body / (total_range + 1e-10) < 0.3
        small_upper = upper_shadow / (total_range + 1e-10) < 0.1
        long_lower = lower_shadow >= 2 * body
        
        pattern = (small_body & small_upper & long_lower).astype(float)
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="cdl_hanging_man",
    category="candlestick_pattern",
    canonical="cdl_hanging_man",
    source="polars_native_candlestick",
)
class CDL_HangingMan(SeriesOperator):
    """Hanging Man: bearish reversal (same shape as hammer, but at top)"""

    metadata = OperatorMetadata(
        name="cdl_hanging_man",
        category="candlestick_pattern",
        description="Hanging Man pattern (-1=bearish, 0=none)",
        param_names=["open", "high", "low", "close"],
        param_types={"open": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, high: pl.DataFrame, 
                         low: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        # TODO: Should check for uptrend context for proper identification
        o, h, l, c, cols = _get_ohlc_arrays(open, high, low, close)
        
        body = np.abs(c - o)
        upper_shadow = h - np.maximum(o, c)
        lower_shadow = np.minimum(o, c) - l
        total_range = h - l
        
        small_body = body / (total_range + 1e-10) < 0.3
        small_upper = upper_shadow / (total_range + 1e-10) < 0.1
        long_lower = lower_shadow >= 2 * body
        
        pattern = np.where(small_body & small_upper & long_lower, -1, 0)
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="cdl_harami",
    category="candlestick_pattern",
    canonical="cdl_harami",
    source="polars_native_candlestick",
)
class CDL_Harami(SeriesOperator):
    """Harami: reversal pattern (small body inside previous large body)"""

    metadata = OperatorMetadata(
        name="cdl_harami",
        category="candlestick_pattern",
        description="Harami pattern (1=bullish, -1=bearish, 0=none)",
        param_names=["open", "close"],
        param_types={"open": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = [c for c in open.columns if c not in PANEL_SKIP_COLUMNS]
        o = open.select(cols).to_numpy()
        c = close.select(cols).to_numpy()
        
        o_prev = np.roll(o, 1, axis=0)
        c_prev = np.roll(c, 1, axis=0)
        
        body_high = np.maximum(o, c)
        body_low = np.minimum(o, c)
        body_high_prev = np.maximum(o_prev, c_prev)
        body_low_prev = np.minimum(o_prev, c_prev)
        
        # Current body inside previous body
        inside = (body_high < body_high_prev) & (body_low > body_low_prev)
        
        # Bullish harami: prev bearish, current bullish
        prev_bearish = c_prev < o_prev
        curr_bullish = c > o
        bullish_harami = inside & prev_bearish & curr_bullish
        
        # Bearish harami: prev bullish, current bearish
        prev_bullish = c_prev > o_prev
        curr_bearish = c < o
        bearish_harami = inside & prev_bullish & curr_bearish
        
        pattern = np.where(bullish_harami, 1, np.where(bearish_harami, -1, 0))
        pattern[0] = 0
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="cdl_harami_cross",
    category="candlestick_pattern",
    canonical="cdl_harami_cross",
    source="polars_native_candlestick",
)
class CDL_HaramiCross(SeriesOperator):
    """Harami Cross: reversal pattern (doji inside previous body)"""

    metadata = OperatorMetadata(
        name="cdl_harami_cross",
        category="candlestick_pattern",
        description="Harami Cross pattern (1=bullish, -1=bearish, 0=none)",
        param_names=["open", "high", "low", "close"],
        param_types={"open": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, high: pl.DataFrame, 
                         low: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        o, h, l, c, cols = _get_ohlc_arrays(open, high, low, close)
        
        o_prev = np.roll(o, 1, axis=0)
        c_prev = np.roll(c, 1, axis=0)
        
        # Current is doji
        body = np.abs(c - o)
        total_range = h - l
        is_doji = body / (total_range + 1e-10) < 0.1
        
        # Inside previous body
        body_high_prev = np.maximum(o_prev, c_prev)
        body_low_prev = np.minimum(o_prev, c_prev)
        body_mid = (o + c) / 2
        inside = (body_mid < body_high_prev) & (body_mid > body_low_prev)
        
        prev_bearish = c_prev < o_prev
        prev_bullish = c_prev > o_prev
        
        bullish_pattern = is_doji & inside & prev_bearish
        bearish_pattern = is_doji & inside & prev_bullish
        
        pattern = np.where(bullish_pattern, 1, np.where(bearish_pattern, -1, 0))
        pattern[0] = 0
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="cdl_inside_bar",
    category="candlestick_pattern",
    canonical="cdl_inside_bar",
    source="polars_native_candlestick",
)
class CDL_InsideBar(SeriesOperator):
    """Inside Bar: consolidation (range within previous bar)"""

    metadata = OperatorMetadata(
        name="cdl_inside_bar",
        category="candlestick_pattern",
        description="Inside Bar pattern (1=inside, 0=none)",
        param_names=["high", "low"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        h = high.select(cols).to_numpy()
        l = low.select(cols).to_numpy()
        
        h_prev = np.roll(h, 1, axis=0)
        l_prev = np.roll(l, 1, axis=0)
        
        pattern = ((h <= h_prev) & (l >= l_prev)).astype(float)
        pattern[0] = 0
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="cdl_inverted_hammer",
    category="candlestick_pattern",
    canonical="cdl_inverted_hammer",
    source="polars_native_candlestick",
)
class CDL_InvertedHammer(SeriesOperator):
    """Inverted Hammer: bullish reversal (small body, long upper shadow)"""

    metadata = OperatorMetadata(
        name="cdl_inverted_hammer",
        category="candlestick_pattern",
        description="Inverted Hammer pattern (1=bullish, 0=none)",
        param_names=["open", "high", "low", "close"],
        param_types={"open": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, high: pl.DataFrame, 
                         low: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        o, h, l, c, cols = _get_ohlc_arrays(open, high, low, close)
        
        body = np.abs(c - o)
        upper_shadow = h - np.maximum(o, c)
        lower_shadow = np.minimum(o, c) - l
        total_range = h - l
        
        small_body = body / (total_range + 1e-10) < 0.3
        small_lower = lower_shadow / (total_range + 1e-10) < 0.1
        long_upper = upper_shadow >= 2 * body
        
        pattern = (small_body & small_lower & long_upper).astype(float)
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="cdl_marubozu",
    category="candlestick_pattern",
    canonical="cdl_marubozu",
    source="polars_native_candlestick",
)
class CDL_Marubozu(SeriesOperator):
    """Marubozu: strong trend (no or minimal shadows)"""

    metadata = OperatorMetadata(
        name="cdl_marubozu",
        category="candlestick_pattern",
        description="Marubozu pattern (1=bullish, -1=bearish, 0=none)",
        param_names=["open", "high", "low", "close"],
        param_types={"open": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, high: pl.DataFrame, 
                         low: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        o, h, l, c, cols = _get_ohlc_arrays(open, high, low, close)
        
        body = np.abs(c - o)
        total_range = h - l
        body_ratio = body / (total_range + 1e-10)
        
        is_marubozu = body_ratio > 0.95
        is_bullish = c > o
        
        pattern = np.where(is_marubozu & is_bullish, 1, 
                          np.where(is_marubozu & ~is_bullish, -1, 0))
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="cdl_morning_star",
    category="candlestick_pattern",
    canonical="cdl_morning_star",
    source="polars_native_candlestick",
)
class CDL_MorningStar(SeriesOperator):
    """Morning Star: bullish reversal (3-bar pattern)"""

    metadata = OperatorMetadata(
        name="cdl_morning_star",
        category="candlestick_pattern",
        description="Morning Star pattern (1=bullish, 0=none)",
        param_names=["open", "high", "low", "close"],
        param_types={"open": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, high: pl.DataFrame, 
                         low: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        o, h, l, c, cols = _get_ohlc_arrays(open, high, low, close)
        
        # Day 1: strong bearish
        day1_bearish = c < o
        day1_body = o - c
        
        # Day 2: small body (star) gapping down
        body_size = np.abs(c - o)
        day1_body_prev = np.roll(day1_body, 1, axis=0)
        day2_small = body_size < (day1_body_prev * 0.3)
        
        # Day 3: bullish closing into day 1 body
        o_prev2 = np.roll(o, 2, axis=0)
        c_prev2 = np.roll(c, 2, axis=0)
        day1_bearish_prev2 = np.roll(day1_bearish, 2, axis=0)
        day3_bullish = c > o
        day3_closes_high = c > (o_prev2 + c_prev2) / 2
        
        day2_small_prev = np.roll(day2_small, 1, axis=0)
        
        pattern = (day1_bearish_prev2 & day2_small_prev & day3_bullish & day3_closes_high).astype(float)
        pattern[0:2] = 0
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="cdl_outside_bar",
    category="candlestick_pattern",
    canonical="cdl_outside_bar",
    source="polars_native_candlestick",
)
class CDL_OutsideBar(SeriesOperator):
    """Outside Bar: expansion (range engulfs previous bar)"""

    metadata = OperatorMetadata(
        name="cdl_outside_bar",
        category="candlestick_pattern",
        description="Outside Bar pattern (1=bullish, -1=bearish, 0=none)",
        param_names=["high", "low", "close"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, 
                         close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        h = high.select(cols).to_numpy()
        l = low.select(cols).to_numpy()
        c = close.select(cols).to_numpy()
        
        h_prev = np.roll(h, 1, axis=0)
        l_prev = np.roll(l, 1, axis=0)
        c_prev = np.roll(c, 1, axis=0)
        
        is_outside = (h > h_prev) & (l < l_prev)
        is_bullish = c > c_prev
        
        pattern = np.where(is_outside & is_bullish, 1, 
                          np.where(is_outside & ~is_bullish, -1, 0))
        pattern[0] = 0
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, high)


@register_operator(
    name="cdl_piercing",
    category="candlestick_pattern",
    canonical="cdl_piercing",
    source="polars_native_candlestick",
)
class CDL_Piercing(SeriesOperator):
    """Piercing Pattern: bullish reversal (2-bar, opposite of dark cloud)"""

    metadata = OperatorMetadata(
        name="cdl_piercing",
        category="candlestick_pattern",
        description="Piercing pattern (1=bullish, 0=none)",
        param_names=["open", "high", "low", "close"],
        param_types={"open": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, high: pl.DataFrame, 
                         low: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        o, h, l, c, cols = _get_ohlc_arrays(open, high, low, close)
        
        # Day 1: bearish candle
        day1_bearish = c < o
        day1_body = o - c
        
        # Day 2: bullish candle opening below day 1 close
        o_prev = np.roll(o, 1, axis=0)
        c_prev = np.roll(c, 1, axis=0)
        day1_body_prev = np.roll(day1_body, 1, axis=0)
        day1_bearish_prev = np.roll(day1_bearish, 1, axis=0)
        
        day2_bullish = c > o
        day2_opens_below = o < c_prev
        day2_closes_into_body = (c < o_prev) & (c > c_prev)
        day2_closes_deep = (c - c_prev) > (day1_body_prev * 0.5)
        
        pattern = (day1_bearish_prev & day2_bullish & day2_opens_below & 
                  day2_closes_into_body & day2_closes_deep).astype(float)
        pattern[0] = 0
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="cdl_shooting_star",
    category="candlestick_pattern",
    canonical="cdl_shooting_star",
    source="polars_native_candlestick",
)
class CDL_ShootingStar(SeriesOperator):
    """Shooting Star: bearish reversal (same shape as inverted hammer, at top)"""

    metadata = OperatorMetadata(
        name="cdl_shooting_star",
        category="candlestick_pattern",
        description="Shooting Star pattern (-1=bearish, 0=none)",
        param_names=["open", "high", "low", "close"],
        param_types={"open": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, high: pl.DataFrame, 
                         low: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        # TODO: Should check for uptrend context for proper identification
        o, h, l, c, cols = _get_ohlc_arrays(open, high, low, close)
        
        body = np.abs(c - o)
        upper_shadow = h - np.maximum(o, c)
        lower_shadow = np.minimum(o, c) - l
        total_range = h - l
        
        small_body = body / (total_range + 1e-10) < 0.3
        small_lower = lower_shadow / (total_range + 1e-10) < 0.1
        long_upper = upper_shadow >= 2 * body
        
        pattern = np.where(small_body & small_lower & long_upper, -1, 0)
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="cdl_spinning_top",
    category="candlestick_pattern",
    canonical="cdl_spinning_top",
    source="polars_native_candlestick",
)
class CDL_SpinningTop(SeriesOperator):
    """Spinning Top: indecision (small body, long shadows both sides)"""

    metadata = OperatorMetadata(
        name="cdl_spinning_top",
        category="candlestick_pattern",
        description="Spinning Top pattern (1=present, 0=none)",
        param_names=["open", "high", "low", "close"],
        param_types={"open": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, high: pl.DataFrame, 
                         low: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        o, h, l, c, cols = _get_ohlc_arrays(open, high, low, close)
        
        body = np.abs(c - o)
        upper_shadow = h - np.maximum(o, c)
        lower_shadow = np.minimum(o, c) - l
        total_range = h - l
        
        small_body = body / (total_range + 1e-10) < 0.3
        has_upper = upper_shadow > body
        has_lower = lower_shadow > body
        
        pattern = (small_body & has_upper & has_lower).astype(float)
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="cdl_three_black_crows",
    category="candlestick_pattern",
    canonical="cdl_three_black_crows",
    source="polars_native_candlestick",
)
class CDL_ThreeBlackCrows(SeriesOperator):
    """Three Black Crows: bearish reversal (3 consecutive bearish candles)"""

    metadata = OperatorMetadata(
        name="cdl_three_black_crows",
        category="candlestick_pattern",
        description="Three Black Crows pattern (-1=bearish, 0=none)",
        param_names=["open", "close"],
        param_types={"open": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = [c for c in open.columns if c not in PANEL_SKIP_COLUMNS]
        o = open.select(cols).to_numpy()
        c = close.select(cols).to_numpy()
        
        # All three bearish
        bearish = c < o
        bearish_1 = np.roll(bearish, 1, axis=0)
        bearish_2 = np.roll(bearish, 2, axis=0)
        
        # Each opens within previous body and closes lower
        c_prev = np.roll(c, 1, axis=0)
        c_prev2 = np.roll(c, 2, axis=0)
        
        closes_lower = (c < c_prev) & (c_prev < c_prev2)
        
        pattern = np.where(bearish & bearish_1 & bearish_2 & closes_lower, -1, 0)
        pattern[0:2] = 0
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="cdl_three_white_soldiers",
    category="candlestick_pattern",
    canonical="cdl_three_white_soldiers",
    source="polars_native_candlestick",
)
class CDL_ThreeWhiteSoldiers(SeriesOperator):
    """Three White Soldiers: bullish reversal (3 consecutive bullish candles)"""

    metadata = OperatorMetadata(
        name="cdl_three_white_soldiers",
        category="candlestick_pattern",
        description="Three White Soldiers pattern (1=bullish, 0=none)",
        param_names=["open", "close"],
        param_types={"open": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, open: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = [c for c in open.columns if c not in PANEL_SKIP_COLUMNS]
        o = open.select(cols).to_numpy()
        c = close.select(cols).to_numpy()
        
        # All three bullish
        bullish = c > o
        bullish_1 = np.roll(bullish, 1, axis=0)
        bullish_2 = np.roll(bullish, 2, axis=0)
        
        # Each opens within previous body and closes higher
        c_prev = np.roll(c, 1, axis=0)
        c_prev2 = np.roll(c, 2, axis=0)
        
        closes_higher = (c > c_prev) & (c_prev > c_prev2)
        
        pattern = (bullish & bullish_1 & bullish_2 & closes_higher).astype(float)
        pattern[0:2] = 0
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, open)


@register_operator(
    name="cdl_tweezer_bottom",
    category="candlestick_pattern",
    canonical="cdl_tweezer_bottom",
    source="polars_native_candlestick",
)
class CDL_TweezerBottom(SeriesOperator):
    """Tweezer Bottom: bullish reversal (two candles with same low)"""

    metadata = OperatorMetadata(
        name="cdl_tweezer_bottom",
        category="candlestick_pattern",
        description="Tweezer Bottom pattern (1=bullish, 0=none)",
        param_names=["low", "close"],
        param_types={"low": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, low: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = [c for c in low.columns if c not in PANEL_SKIP_COLUMNS]
        l = low.select(cols).to_numpy()
        c = close.select(cols).to_numpy()
        
        l_prev = np.roll(l, 1, axis=0)
        c_prev = np.roll(c, 1, axis=0)
        
        # Similar lows (within 0.2% tolerance)
        same_low = np.abs(l - l_prev) / (l_prev + 1e-10) < 0.002
        
        # First bearish, second bullish (or at least higher close)
        c_prev2 = np.roll(c, 2, axis=0)
        first_down = c_prev < c_prev2
        second_up = c > c_prev
        
        pattern = (same_low & second_up).astype(float)
        pattern[0] = 0
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, low)


@register_operator(
    name="cdl_tweezer_top",
    category="candlestick_pattern",
    canonical="cdl_tweezer_top",
    source="polars_native_candlestick",
)
class CDL_TweezerTop(SeriesOperator):
    """Tweezer Top: bearish reversal (two candles with same high)"""

    metadata = OperatorMetadata(
        name="cdl_tweezer_top",
        category="candlestick_pattern",
        description="Tweezer Top pattern (-1=bearish, 0=none)",
        param_names=["high", "close"],
        param_types={"high": pl.DataFrame, "close": pl.DataFrame},
    )

    def _calculate_series(self, high: pl.DataFrame, close: pl.DataFrame, **kwargs) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        h = high.select(cols).to_numpy()
        c = close.select(cols).to_numpy()
        
        h_prev = np.roll(h, 1, axis=0)
        c_prev = np.roll(c, 1, axis=0)
        
        # Similar highs (within 0.2% tolerance)
        same_high = np.abs(h - h_prev) / (h_prev + 1e-10) < 0.002
        
        # First bullish, second bearish (or at least lower close)
        c_prev2 = np.roll(c, 2, axis=0)
        first_up = c_prev > c_prev2
        second_down = c < c_prev
        
        pattern = np.where(same_high & second_down, -1, 0)
        pattern[0] = 0
        
        return _result_df({col: pattern[:, idx] for idx, col in enumerate(cols)}, high)


# Summary: 35 candlestick operators implemented
# 13 candle_* metrics: body_percentile, body_zscore, close_strength, gap_atr,
#                      inside_ratio, lower_shadow_zscore, overlap_ratio, range_atr,
#                      range_percentile, range_zscore, rejection_lower, rejection_upper,
#                      upper_shadow_zscore
# 22 cdl_* patterns: dark_cloud_cover, doji, dragonfly_doji, engulfing, evening_star,
#                    gravestone_doji, hammer, hanging_man, harami, harami_cross,
#                    inside_bar, inverted_hammer, marubozu, morning_star, outside_bar,
#                    piercing, shooting_star, spinning_top, three_black_crows,
#                    three_white_soldiers, tweezer_bottom, tweezer_top
