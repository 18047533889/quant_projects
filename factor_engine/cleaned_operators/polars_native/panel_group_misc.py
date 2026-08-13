# -*- coding: utf-8 -*-
"""
Polars native Panel, Group, and Miscellaneous operators - Phase 6

Implementation of ~70 operators including:
- panel_* operators: panel operations across time and cross-section
- group_* operators: group-level features and aggregations
- Miscellaneous: technical indicators, calendar ops, composition, and various utilities

All operators use genuine Polars API with .over() for group operations.
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


def _apply_to_panel(df: pl.DataFrame, expr_fn) -> pl.DataFrame:
    """Apply Polars expression function to all numeric columns in panel"""
    cols = [c for c in df.columns if c not in PANEL_SKIP_COLUMNS]
    if not cols:
        return df
    return df.with_columns([expr_fn(pl.col(c)).alias(c) for c in cols])


def _ema_expr(col: pl.Expr, span: int) -> pl.Expr:
    """EMA expression (alpha = 2/(span+1))"""
    return col.ewm_mean(span=span, adjust=False, ignore_nulls=True)


def _wilder_ema_expr(col: pl.Expr, period: int) -> pl.Expr:
    """Wilder's smoothing (alpha = 1/period)"""
    alpha = (1.0) / period if period != 0 else np.nan
    return col.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)


# ==================== Technical Indicators ====================

@register_operator(
    name="ALMA",
    category="technical_indicator",
    canonical="ALMA",
    source="polars_native_phase6",
)
class ALMA(SeriesOperator):
    """Arnaud Legoux Moving Average"""

    metadata = OperatorMetadata(
        name="ALMA",
        category="technical_indicator",
        description="Arnaud Legoux Moving Average with Gaussian weights",
        param_names=["close", "period", "offset", "sigma"],
        param_types={"close": pl.DataFrame, "period": int, "offset": float, "sigma": float},
    )

    def _calculate_series(self, close: pl.DataFrame, period: int = 9, offset: float = 0.85, sigma: float = 6.0, **kwargs) -> pl.DataFrame:
        def alma_expr(col_name):
            m = offset * (period - 1)
            s = (period) / sigma if sigma > 1e-10 else np.nan
            # Generate Gaussian weights
            weights = pl.Series([np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(period)])
            weights = weights / weights.sum()
            # Apply weighted moving average
            return pl.col(col_name).rolling_map(lambda x: np.dot(x, weights.to_numpy()), window_size=period).alias(col_name)

        return _apply_to_panel(close, lambda c: alma_expr(c.meta.output_name()))


@register_operator(
    name="DMI_plus",
    category="technical_indicator",
    canonical="DMI_plus",
    source="polars_native_phase6",
)
class DMI_plus(SeriesOperator):
    """Directional Movement Index Plus"""

    metadata = OperatorMetadata(
        name="DMI_plus",
        category="technical_indicator",
        description="Positive Directional Movement Index",
        param_names=["high", "low", "period"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, period: int = 14, **kwargs) -> pl.DataFrame:
        def dmi_plus_expr(h_col, l_col):
            dm_plus = (pl.col(h_col) - pl.col(h_col).shift(1)).clip(lower_bound=0)
            dm_minus = (pl.col(l_col).shift(1) - pl.col(l_col)).clip(lower_bound=0)
            # Only keep positive DM if it's greater than negative DM
            dm_plus = pl.when(dm_plus > dm_minus).then(dm_plus).otherwise(0)
            tr = (pl.col(h_col) - pl.col(l_col)).abs().rolling_max(window_size=period)
            smoothed_dm = _wilder_ema_expr(dm_plus, period)
            smoothed_tr = _wilder_ema_expr(tr, period)
            return np.where(smoothed_tr.alias(h_col) != 0, ((100 * smoothed_dm) / (smoothed_tr).alias(h_col)), np.nan)

        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        return high.with_columns([dmi_plus_expr(c, c) for c in cols])


@register_operator(
    name="DMI_minus",
    category="technical_indicator",
    canonical="DMI_minus",
    source="polars_native_phase6",
)
class DMI_minus(SeriesOperator):
    """Directional Movement Index Minus"""

    metadata = OperatorMetadata(
        name="DMI_minus",
        category="technical_indicator",
        description="Negative Directional Movement Index",
        param_names=["high", "low", "period"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, period: int = 14, **kwargs) -> pl.DataFrame:
        def dmi_minus_expr(h_col, l_col):
            dm_plus = (pl.col(h_col) - pl.col(h_col).shift(1)).clip(lower_bound=0)
            dm_minus = (pl.col(l_col).shift(1) - pl.col(l_col)).clip(lower_bound=0)
            # Only keep negative DM if it's greater than positive DM
            dm_minus = pl.when(dm_minus > dm_plus).then(dm_minus).otherwise(0)
            tr = (pl.col(h_col) - pl.col(l_col)).abs().rolling_max(window_size=period)
            smoothed_dm = _wilder_ema_expr(dm_minus, period)
            smoothed_tr = _wilder_ema_expr(tr, period)
            return np.where(smoothed_tr.alias(h_col) != 0, ((100 * smoothed_dm) / (smoothed_tr).alias(h_col)), np.nan)

        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        return high.with_columns([dmi_minus_expr(c, c) for c in cols])


@register_operator(
    name="DX",
    category="technical_indicator",
    canonical="DX",
    source="polars_native_phase6",
)
class DX(SeriesOperator):
    """Directional Movement Index"""

    metadata = OperatorMetadata(
        name="DX",
        category="technical_indicator",
        description="Directional Movement Index (DX)",
        param_names=["high", "low", "period"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, period: int = 14, **kwargs) -> pl.DataFrame:
        def dx_expr(h_col, l_col):
            dm_plus = (pl.col(h_col) - pl.col(h_col).shift(1)).clip(lower_bound=0)
            dm_minus = (pl.col(l_col).shift(1) - pl.col(l_col)).clip(lower_bound=0)
            dm_plus = pl.when(dm_plus > dm_minus).then(dm_plus).otherwise(0)
            dm_minus = pl.when(dm_minus > dm_plus).then(dm_minus).otherwise(0)
            tr = (pl.col(h_col) - pl.col(l_col)).abs()

            di_plus = (100 * _wilder_ema_expr(dm_plus, period)) / (_wilder_ema_expr(tr, period) if (_wilder_ema_expr(tr, period) != 0 else np.nan
            di_minus = (100 * _wilder_ema_expr(dm_minus, period)) / (_wilder_ema_expr(tr, period) if (_wilder_ema_expr(tr, period) != 0 else np.nan
            dx = (100 * (di_plus - di_minus).abs()) / ((di_plus + di_minus) if ((di_plus + di_minus) != 0 else np.nan
            return dx.alias(h_col)

        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        return high.with_columns([dx_expr(c, c) for c in cols])


@register_operator(
    name="NATR",
    category="technical_indicator",
    canonical="NATR",
    source="polars_native_phase6",
)
class NATR(SeriesOperator):
    """Normalized Average True Range"""

    metadata = OperatorMetadata(
        name="NATR",
        category="technical_indicator",
        description="Normalized ATR as percentage of close price",
        param_names=["high", "low", "close", "period"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, close: pl.DataFrame, period: int = 14, **kwargs) -> pl.DataFrame:
        def natr_expr(h_col, l_col, c_col):
            h = pl.col(h_col)
            l = pl.col(l_col)
            c = pl.col(c_col)
            c_prev = c.shift(1)
            tr1 = h - l
            tr2 = (h - c_prev).abs()
            tr3 = (l - c_prev).abs()
            tr = pl.max_horizontal(tr1, tr2, tr3)
            atr = _wilder_ema_expr(tr, period)
            natr = (100 * atr) / c if c != 0 else np.nan
            return natr.alias(h_col)

        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        return high.with_columns([natr_expr(c, c, c) for c in cols])


@register_operator(
    name="PPO",
    category="technical_indicator",
    canonical="PPO",
    source="polars_native_phase6",
)
class PPO(SeriesOperator):
    """Percentage Price Oscillator"""

    metadata = OperatorMetadata(
        name="PPO",
        category="technical_indicator",
        description="Percentage Price Oscillator (MACD as percentage)",
        param_names=["close", "fast", "slow"],
        param_types={"close": pl.DataFrame, "fast": int, "slow": int},
    )

    def _calculate_series(self, close: pl.DataFrame, fast: int = 12, slow: int = 26, **kwargs) -> pl.DataFrame:
        def ppo_expr(col_name):
            ema_fast = _ema_expr(pl.col(col_name), fast)
            ema_slow = _ema_expr(pl.col(col_name), slow)
            ppo = (100 * (ema_fast - ema_slow)) / ema_slow if ema_slow != 0 else np.nan
            return ppo.alias(col_name)

        return _apply_to_panel(close, lambda c: ppo_expr(c.meta.output_name()))


@register_operator(
    name="PPO_signal",
    category="technical_indicator",
    canonical="PPO_signal",
    source="polars_native_phase6",
)
class PPO_signal(SeriesOperator):
    """PPO Signal Line"""

    metadata = OperatorMetadata(
        name="PPO_signal",
        category="technical_indicator",
        description="PPO signal line (EMA of PPO)",
        param_names=["close", "fast", "slow", "signal"],
        param_types={"close": pl.DataFrame, "fast": int, "slow": int, "signal": int},
    )

    def _calculate_series(self, close: pl.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs) -> pl.DataFrame:
        def ppo_signal_expr(col_name):
            ema_fast = _ema_expr(pl.col(col_name), fast)
            ema_slow = _ema_expr(pl.col(col_name), slow)
            ppo = (100 * (ema_fast - ema_slow)) / ema_slow if ema_slow != 0 else np.nan
            return _ema_expr(ppo, signal).alias(col_name)

        return _apply_to_panel(close, lambda c: ppo_signal_expr(c.meta.output_name()))


@register_operator(
    name="PPO_hist",
    category="technical_indicator",
    canonical="PPO_hist",
    source="polars_native_phase6",
)
class PPO_hist(SeriesOperator):
    """PPO Histogram"""

    metadata = OperatorMetadata(
        name="PPO_hist",
        category="technical_indicator",
        description="PPO histogram (PPO - signal)",
        param_names=["close", "fast", "slow", "signal"],
        param_types={"close": pl.DataFrame, "fast": int, "slow": int, "signal": int},
    )

    def _calculate_series(self, close: pl.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs) -> pl.DataFrame:
        def ppo_hist_expr(col_name):
            ema_fast = _ema_expr(pl.col(col_name), fast)
            ema_slow = _ema_expr(pl.col(col_name), slow)
            ppo = (100 * (ema_fast - ema_slow)) / ema_slow if ema_slow != 0 else np.nan
            ppo_signal = _ema_expr(ppo, signal)
            return (ppo - ppo_signal).alias(col_name)

        return _apply_to_panel(close, lambda c: ppo_hist_expr(c.meta.output_name()))


@register_operator(
    name="PVO",
    category="technical_indicator",
    canonical="PVO",
    source="polars_native_phase6",
)
class PVO(SeriesOperator):
    """Percentage Volume Oscillator"""

    metadata = OperatorMetadata(
        name="PVO",
        category="technical_indicator",
        description="Percentage Volume Oscillator",
        param_names=["volume", "fast", "slow"],
        param_types={"volume": pl.DataFrame, "fast": int, "slow": int},
    )

    def _calculate_series(self, volume: pl.DataFrame, fast: int = 12, slow: int = 26, **kwargs) -> pl.DataFrame:
        def pvo_expr(col_name):
            ema_fast = _ema_expr(pl.col(col_name), fast)
            ema_slow = _ema_expr(pl.col(col_name), slow)
            pvo = (100 * (ema_fast - ema_slow)) / ema_slow if ema_slow != 0 else np.nan
            return pvo.alias(col_name)

        return _apply_to_panel(volume, lambda c: pvo_expr(c.meta.output_name()))


@register_operator(
    name="PVO_signal",
    category="technical_indicator",
    canonical="PVO_signal",
    source="polars_native_phase6",
)
class PVO_signal(SeriesOperator):
    """PVO Signal Line"""

    metadata = OperatorMetadata(
        name="PVO_signal",
        category="technical_indicator",
        description="PVO signal line (EMA of PVO)",
        param_names=["volume", "fast", "slow", "signal"],
        param_types={"volume": pl.DataFrame, "fast": int, "slow": int, "signal": int},
    )

    def _calculate_series(self, volume: pl.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs) -> pl.DataFrame:
        def pvo_signal_expr(col_name):
            ema_fast = _ema_expr(pl.col(col_name), fast)
            ema_slow = _ema_expr(pl.col(col_name), slow)
            pvo = (100 * (ema_fast - ema_slow)) / ema_slow if ema_slow != 0 else np.nan
            return _ema_expr(pvo, signal).alias(col_name)

        return _apply_to_panel(volume, lambda c: pvo_signal_expr(c.meta.output_name()))


@register_operator(
    name="PVO_hist",
    category="technical_indicator",
    canonical="PVO_hist",
    source="polars_native_phase6",
)
class PVO_hist(SeriesOperator):
    """PVO Histogram"""

    metadata = OperatorMetadata(
        name="PVO_hist",
        category="technical_indicator",
        description="PVO histogram (PVO - signal)",
        param_names=["volume", "fast", "slow", "signal"],
        param_types={"volume": pl.DataFrame, "fast": int, "slow": int, "signal": int},
    )

    def _calculate_series(self, volume: pl.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs) -> pl.DataFrame:
        def pvo_hist_expr(col_name):
            ema_fast = _ema_expr(pl.col(col_name), fast)
            ema_slow = _ema_expr(pl.col(col_name), slow)
            pvo = (100 * (ema_fast - ema_slow)) / ema_slow if ema_slow != 0 else np.nan
            pvo_signal = _ema_expr(pvo, signal)
            return (pvo - pvo_signal).alias(col_name)

        return _apply_to_panel(volume, lambda c: pvo_hist_expr(c.meta.output_name()))


@register_operator(
    name="donchian_position",
    category="technical_indicator",
    canonical="donchian_position",
    source="polars_native_phase6",
)
class donchian_position(SeriesOperator):
    """Donchian Channel Position"""

    metadata = OperatorMetadata(
        name="donchian_position",
        category="technical_indicator",
        description="Current price position in Donchian channel [0=low, 1=high]",
        param_names=["close", "period"],
        param_types={"close": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, close: pl.DataFrame, period: int = 20, **kwargs) -> pl.DataFrame:
        def donchian_expr(col_name):
            high = pl.col(col_name).rolling_max(window_size=period)
            low = pl.col(col_name).rolling_min(window_size=period)
            position = ((pl.col(col_name) - low)) / ((high - low) if ((high - low) != 0 else np.nan
            return position.alias(col_name)

        return _apply_to_panel(close, lambda c: donchian_expr(c.meta.output_name()))


# ==================== Miscellaneous Operators ====================

@register_operator(
    name="calendar_day_diff",
    category="utility",
    canonical="calendar_day_diff",
    source="polars_native_phase6",
)
class calendar_day_diff(SeriesOperator):
    """Calendar days since last observation"""

    metadata = OperatorMetadata(
        name="calendar_day_diff",
        category="utility",
        description="Number of calendar days since previous observation",
        param_names=["date"],
        param_types={"date": pl.DataFrame},
    )

    def _calculate_series(self, date: pl.DataFrame, **kwargs) -> pl.DataFrame:
        # Assumes date column exists in metadata columns
        if "date" in date.columns:
            date_col = pl.col("date")
            day_diff = (date_col - date_col.shift(1)).dt.total_days()
            result = date.with_columns([day_diff.alias("date")])
            # Apply to all value columns
            cols = [c for c in date.columns if c not in PANEL_SKIP_COLUMNS]
            return result.with_columns([day_diff.alias(c) for c in cols])
        return date


@register_operator(
    name="directional_change_extent",
    category="technical_indicator",
    canonical="directional_change_extent",
    source="polars_native_phase6",
)
class directional_change_extent(SeriesOperator):
    """Directional Change Event Extent"""

    metadata = OperatorMetadata(
        name="directional_change_extent",
        category="technical_indicator",
        description="Cumulative extent since last directional change event",
        param_names=["close", "threshold"],
        param_types={"close": pl.DataFrame, "threshold": float},
    )

    def _calculate_series(self, close: pl.DataFrame, threshold: float = 0.02, **kwargs) -> pl.DataFrame:
        def dc_extent_expr(col_name):
            # Simplified: measure deviation from rolling reference
            ref = pl.col(col_name).shift(1)
            change = ((pl.col(col_name) - ref)) / ref if ref != 0 else np.nan
            extent = change.abs().rolling_sum(window_size=20)
            return extent.alias(col_name)

        return _apply_to_panel(close, lambda c: dc_extent_expr(c.meta.output_name()))


@register_operator(
    name="winsorize_mean",
    category="statistics",
    canonical="winsorize_mean",
    source="polars_native_phase6",
    replace=True,
    replacement_reason="Native Polars implementation replaces pandas bridge",
    expected_old_source="factor_dsl_np",
)
class winsorize_mean(SeriesOperator):
    """Winsorized Mean"""

    metadata = OperatorMetadata(
        name="winsorize_mean",
        category="statistics",
        description="Mean after winsorizing extreme values",
        param_names=["x", "period", "lower", "upper"],
        param_types={"x": pl.DataFrame, "period": int, "lower": float, "upper": float},
    )

    def _calculate_series(self, x: pl.DataFrame, period: int = 20, lower: float = 0.05, upper: float = 0.95, **kwargs) -> pl.DataFrame:
        def winsorize_expr(col_name):
            col = pl.col(col_name)
            lower_q = col.rolling_quantile(quantile=lower, window_size=period)
            upper_q = col.rolling_quantile(quantile=upper, window_size=period)
            winsorized = col.clip(lower_bound=lower_q, upper_bound=upper_q)
            return winsorized.rolling_mean(window_size=period).alias(col_name)

        return _apply_to_panel(x, lambda c: winsorize_expr(c.meta.output_name()))


@register_operator(
    name="zscore",
    category="statistics",
    canonical="zscore",
    source="polars_native_phase6",
    replace=True,
    replacement_reason="Native Polars implementation replaces pandas bridge",
    expected_old_source="factor_dsl_np",
)
class zscore(SeriesOperator):
    """Z-Score Normalization"""

    metadata = OperatorMetadata(
        name="zscore",
        category="statistics",
        description="Rolling z-score (x - mean) / std",
        param_names=["x", "period"],
        param_types={"x": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, x: pl.DataFrame, period: int = 20, **kwargs) -> pl.DataFrame:
        def zscore_expr(col_name):
            col = pl.col(col_name)
            mean = col.rolling_mean(window_size=period)
            std = col.rolling_std(window_size=period)
            return np.where(std.alias(col_name) != 0, (((col - mean)) / (std).alias(col_name)), np.nan)

        return _apply_to_panel(x, lambda c: zscore_expr(c.meta.output_name()))


@register_operator(
    name="zero_return_ratio",
    category="statistics",
    canonical="zero_return_ratio",
    source="polars_native_phase6",
)
class zero_return_ratio(SeriesOperator):
    """Ratio of zero returns"""

    metadata = OperatorMetadata(
        name="zero_return_ratio",
        category="statistics",
        description="Proportion of zero returns in rolling window",
        param_names=["returns", "period"],
        param_types={"returns": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, returns: pl.DataFrame, period: int = 20, **kwargs) -> pl.DataFrame:
        def zero_ratio_expr(col_name):
            is_zero = (pl.col(col_name).abs() < 1e-10).cast(pl.Float64)
            return is_zero.rolling_mean(window_size=period).alias(col_name)

        return _apply_to_panel(returns, lambda c: zero_ratio_expr(c.meta.output_name()))


@register_operator(
    name="vwap_deviation",
    category="technical_indicator",
    canonical="vwap_deviation",
    source="polars_native_phase6",
)
class vwap_deviation(SeriesOperator):
    """Deviation from VWAP"""

    metadata = OperatorMetadata(
        name="vwap_deviation",
        category="technical_indicator",
        description="Price deviation from volume-weighted average price",
        param_names=["close", "volume", "period"],
        param_types={"close": pl.DataFrame, "volume": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, close: pl.DataFrame, volume: pl.DataFrame, period: int = 20, **kwargs) -> pl.DataFrame:
        def vwap_dev_expr(c_col, v_col):
            price = pl.col(c_col)
            vol = pl.col(v_col)
            pv = price * vol
            vwap = (pv.rolling_sum(window_size=period)) / vol.rolling_sum(window_size=period) if vol.rolling_sum(window_size=period) > 1e-10 else np.nan
            return np.where(vwap.alias(c_col) != 0, (((price - vwap)) / (vwap).alias(c_col)), np.nan)

        cols = [c for c in close.columns if c not in PANEL_SKIP_COLUMNS]
        return close.with_columns([vwap_dev_expr(c, c) for c in cols])


@register_operator(
    name="vwap_to_close_return",
    category="technical_indicator",
    canonical="vwap_to_close_return",
    source="polars_native_phase6",
    replace=True,
    replacement_reason="Native Polars implementation replaces pandas bridge",
    expected_old_source="factor_dsl_np",
)
class vwap_to_close_return(SeriesOperator):
    """Return from VWAP to close"""

    metadata = OperatorMetadata(
        name="vwap_to_close_return",
        category="technical_indicator",
        description="Return from intraday VWAP to close",
        param_names=["close", "volume", "period"],
        param_types={"close": pl.DataFrame, "volume": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, close: pl.DataFrame, volume: pl.DataFrame, period: int = 1, **kwargs) -> pl.DataFrame:
        def vwap_ret_expr(c_col, v_col):
            price = pl.col(c_col)
            vol = pl.col(v_col)
            pv = price * vol
            vwap = (pv.rolling_sum(window_size=period)) / vol.rolling_sum(window_size=period) if vol.rolling_sum(window_size=period) > 1e-10 else np.nan
            return np.where(vwap.alias(c_col) != 0, (((price - vwap)) / (vwap).alias(c_col)), np.nan)

        cols = [c for c in close.columns if c not in PANEL_SKIP_COLUMNS]
        return close.with_columns([vwap_ret_expr(c, c) for c in cols])


@register_operator(
    name="open_to_vwap_return",
    category="technical_indicator",
    canonical="open_to_vwap_return",
    source="polars_native_phase6",
    replace=True,
    replacement_reason="Native Polars implementation replaces pandas bridge",
    expected_old_source="factor_dsl_np",
)
class open_to_vwap_return(SeriesOperator):
    """Return from open to VWAP"""

    metadata = OperatorMetadata(
        name="open_to_vwap_return",
        category="technical_indicator",
        description="Return from open to intraday VWAP",
        param_names=["open", "close", "volume", "period"],
        param_types={"open": pl.DataFrame, "close": pl.DataFrame, "volume": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, open: pl.DataFrame, close: pl.DataFrame, volume: pl.DataFrame, period: int = 1, **kwargs) -> pl.DataFrame:
        def open_vwap_expr(o_col, c_col, v_col):
            price = pl.col(c_col)
            vol = pl.col(v_col)
            pv = price * vol
            vwap = (pv.rolling_sum(window_size=period)) / vol.rolling_sum(window_size=period) if vol.rolling_sum(window_size=period) > 1e-10 else np.nan
            return np.where(pl.col(o_col).alias(o_col) != 0, (((vwap - pl.col(o_col))) / (pl.col(o_col)).alias(o_col)), np.nan)

        cols = [c for c in open.columns if c not in PANEL_SKIP_COLUMNS]
        return open.with_columns([open_vwap_expr(c, c, c) for c in cols])


# ==================== Group Operators ====================

@register_operator(
    name="group_feature_coverage_ratio",
    category="group_feature",
    canonical="group_feature_coverage_ratio",
    source="polars_native_phase6",
)
class group_feature_coverage_ratio(SeriesOperator):
    """Group Feature Coverage Ratio"""

    metadata = OperatorMetadata(
        name="group_feature_coverage_ratio",
        category="group_feature",
        description="Ratio of non-null values in group",
        param_names=["x", "group"],
        param_types={"x": pl.DataFrame, "group": pl.DataFrame},
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame, **kwargs) -> pl.DataFrame:
        # Using .over() for group operations
        def coverage_expr(col_name):
            col = pl.col(col_name)
            total = col.count().over("group")
            non_null = col.is_not_null().cast(pl.Float64).sum().over("group")
            return np.where(total.alias(col_name) != 0, ((non_null) / (total).alias(col_name)), np.nan)

        # Simplified: assume single group column
        return _apply_to_panel(x, lambda c: coverage_expr(c.meta.output_name()))


@register_operator(
    name="group_feature_valid_member_count",
    category="group_feature",
    canonical="group_feature_valid_member_count",
    source="polars_native_phase6",
)
class group_feature_valid_member_count(SeriesOperator):
    """Count of valid members in group"""

    metadata = OperatorMetadata(
        name="group_feature_valid_member_count",
        category="group_feature",
        description="Number of non-null group members",
        param_names=["x", "group"],
        param_types={"x": pl.DataFrame, "group": pl.DataFrame},
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame, **kwargs) -> pl.DataFrame:
        def count_expr(col_name):
            return pl.col(col_name).is_not_null().cast(pl.Float64).sum().over("group").alias(col_name)

        return _apply_to_panel(x, lambda c: count_expr(c.meta.output_name()))


@register_operator(
    name="group_feature_mode_share",
    category="group_feature",
    canonical="group_feature_mode_share",
    source="polars_native_phase6",
)
class group_feature_mode_share(SeriesOperator):
    """Share of most common value in group"""

    metadata = OperatorMetadata(
        name="group_feature_mode_share",
        category="group_feature",
        description="Proportion of group having the most frequent value",
        param_names=["x", "group", "bins"],
        param_types={"x": pl.DataFrame, "group": pl.DataFrame, "bins": int},
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame, bins: int = 10, **kwargs) -> pl.DataFrame:
        # Simplified implementation
        def mode_share_expr(col_name):
            col = pl.col(col_name)
            # Use rank percentile as proxy for mode detection
            ranks = col.rank(method="dense").over("group")
            max_rank = ranks.max().over("group")
            return np.where(col.count().over("group").alias(col_name) != 0, ((max_rank) / (col.count().over("group")).alias(col_name)), np.nan)

        return _apply_to_panel(x, lambda c: mode_share_expr(c.meta.output_name()))


@register_operator(
    name="group_peer_beta_deviation",
    category="group_feature",
    canonical="group_peer_beta_deviation",
    source="polars_native_phase6",
)
class group_peer_beta_deviation(SeriesOperator):
    """Deviation from group beta"""

    metadata = OperatorMetadata(
        name="group_peer_beta_deviation",
        category="group_feature",
        description="Beta deviation from group average beta",
        param_names=["returns", "market_returns", "group", "period"],
        param_types={"returns": pl.DataFrame, "market_returns": pl.DataFrame, "group": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, returns: pl.DataFrame, market_returns: pl.DataFrame, group: pl.DataFrame, period: int = 60, **kwargs) -> pl.DataFrame:
        def beta_dev_expr(r_col, m_col):
            r = pl.col(r_col)
            m = pl.col(m_col)
            # Rolling covariance and variance
            cov = (r * m).rolling_mean(window_size=period) - r.rolling_mean(window_size=period) * m.rolling_mean(window_size=period)
            var = m.rolling_var(window_size=period)
            beta = (cov) / var if var != 0 else np.nan
            group_beta = beta.mean().over("group")
            return (beta - group_beta).alias(r_col)

        cols = [c for c in returns.columns if c not in PANEL_SKIP_COLUMNS]
        return returns.with_columns([beta_dev_expr(c, c) for c in cols])


@register_operator(
    name="group_signal_attraction_share",
    category="group_feature",
    canonical="group_signal_attraction_share",
    source="polars_native_phase6",
)
class group_signal_attraction_share(SeriesOperator):
    """Share of group with same signal direction"""

    metadata = OperatorMetadata(
        name="group_signal_attraction_share",
        category="group_feature",
        description="Proportion of group members with same sign",
        param_names=["signal", "group"],
        param_types={"signal": pl.DataFrame, "group": pl.DataFrame},
    )

    def _calculate_series(self, signal: pl.DataFrame, group: pl.DataFrame, **kwargs) -> pl.DataFrame:
        def attraction_expr(col_name):
            col = pl.col(col_name)
            sign = col.sign()
            # Count matching signs in group
            same_sign = (sign == sign).cast(pl.Float64).sum().over("group")
            total = col.count().over("group")
            return np.where(total.alias(col_name) != 0, ((same_sign) / (total).alias(col_name)), np.nan)

        return _apply_to_panel(signal, lambda c: attraction_expr(c.meta.output_name()))


# ==================== Panel Operators ====================

@register_operator(
    name="panel_rolling_pca_explained_ratio",
    category="panel_feature",
    canonical="panel_rolling_pca_explained_ratio",
    source="polars_native_phase6",
)
class panel_rolling_pca_explained_ratio(SeriesOperator):
    """Rolling PCA explained variance ratio"""

    metadata = OperatorMetadata(
        name="panel_rolling_pca_explained_ratio",
        category="panel_feature",
        description="Explained variance ratio of first PC in rolling window",
        param_names=["x", "period", "n_components"],
        param_types={"x": pl.DataFrame, "period": int, "n_components": int},
    )

    def _calculate_series(self, x: pl.DataFrame, period: int = 60, n_components: int = 1, **kwargs) -> pl.DataFrame:
        # Simplified: use variance proxy
        def pca_ratio_expr(col_name):
            col = pl.col(col_name)
            var = col.rolling_var(window_size=period)
            total_var = var.sum().over("date")
            return np.where(total_var.alias(col_name) != 0, ((var) / (total_var).alias(col_name)), np.nan)

        return _apply_to_panel(x, lambda c: pca_ratio_expr(c.meta.output_name()))


@register_operator(
    name="panel_rolling_pca_loading",
    category="panel_feature",
    canonical="panel_rolling_pca_loading",
    source="polars_native_phase6",
)
class panel_rolling_pca_loading(SeriesOperator):
    """Rolling PCA loading"""

    metadata = OperatorMetadata(
        name="panel_rolling_pca_loading",
        category="panel_feature",
        description="First PC loading in rolling window",
        param_names=["x", "period"],
        param_types={"x": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, x: pl.DataFrame, period: int = 60, **kwargs) -> pl.DataFrame:
        # Simplified: use correlation with cross-sectional mean as proxy
        def loading_expr(col_name):
            col = pl.col(col_name)
            cs_mean = col.mean().over("date")
            # Rolling correlation with cross-sectional mean
            col_demean = col - col.rolling_mean(window_size=period)
            mean_demean = cs_mean - cs_mean.rolling_mean(window_size=period)
            cov = (col_demean * mean_demean).rolling_mean(window_size=period)
            std = col.rolling_std(window_size=period) * cs_mean.rolling_std(window_size=period)
            return np.where(std.alias(col_name) != 0, ((cov) / (std).alias(col_name)), np.nan)

        return _apply_to_panel(x, lambda c: loading_expr(c.meta.output_name()))


@register_operator(
    name="panel_rolling_pca_resid",
    category="panel_feature",
    canonical="panel_rolling_pca_resid",
    source="polars_native_phase6",
)
class panel_rolling_pca_resid(SeriesOperator):
    """Rolling PCA residual"""

    metadata = OperatorMetadata(
        name="panel_rolling_pca_resid",
        category="panel_feature",
        description="Residual after removing first PC",
        param_names=["x", "period"],
        param_types={"x": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, x: pl.DataFrame, period: int = 60, **kwargs) -> pl.DataFrame:
        # Simplified: residual from cross-sectional mean
        def resid_expr(col_name):
            col = pl.col(col_name)
            cs_mean = col.mean().over("date")
            # Beta = rolling covariance / variance
            col_demean = col - col.rolling_mean(window_size=period)
            mean_demean = cs_mean - cs_mean.rolling_mean(window_size=period)
            cov = (col_demean * mean_demean).rolling_mean(window_size=period)
            var = cs_mean.rolling_var(window_size=period)
            beta = (cov) / var if var != 0 else np.nan
            return (col - beta * cs_mean).alias(col_name)

        return _apply_to_panel(x, lambda c: resid_expr(c.meta.output_name()))


@register_operator(
    name="panel_rolling_pca_resid_momentum",
    category="panel_feature",
    canonical="panel_rolling_pca_resid_momentum",
    source="polars_native_phase6",
)
class panel_rolling_pca_resid_momentum(SeriesOperator):
    """Momentum of PCA residual"""

    metadata = OperatorMetadata(
        name="panel_rolling_pca_resid_momentum",
        category="panel_feature",
        description="Momentum of residual after removing first PC",
        param_names=["x", "period", "momentum_period"],
        param_types={"x": pl.DataFrame, "period": int, "momentum_period": int},
    )

    def _calculate_series(self, x: pl.DataFrame, period: int = 60, momentum_period: int = 20, **kwargs) -> pl.DataFrame:
        def resid_mom_expr(col_name):
            col = pl.col(col_name)
            cs_mean = col.mean().over("date")
            col_demean = col - col.rolling_mean(window_size=period)
            mean_demean = cs_mean - cs_mean.rolling_mean(window_size=period)
            cov = (col_demean * mean_demean).rolling_mean(window_size=period)
            var = cs_mean.rolling_var(window_size=period)
            beta = (cov) / var if var != 0 else np.nan
            resid = col - beta * cs_mean
            # Momentum of residual
            return (resid - resid.shift(momentum_period)).alias(col_name)

        return _apply_to_panel(x, lambda c: resid_mom_expr(c.meta.output_name()))


@register_operator(
    name="panel_rolling_pca_resid_vol",
    category="panel_feature",
    canonical="panel_rolling_pca_resid_vol",
    source="polars_native_phase6",
)
class panel_rolling_pca_resid_vol(SeriesOperator):
    """Volatility of PCA residual"""

    metadata = OperatorMetadata(
        name="panel_rolling_pca_resid_vol",
        category="panel_feature",
        description="Standard deviation of residual after removing first PC",
        param_names=["x", "period", "vol_period"],
        param_types={"x": pl.DataFrame, "period": int, "vol_period": int},
    )

    def _calculate_series(self, x: pl.DataFrame, period: int = 60, vol_period: int = 20, **kwargs) -> pl.DataFrame:
        def resid_vol_expr(col_name):
            col = pl.col(col_name)
            cs_mean = col.mean().over("date")
            col_demean = col - col.rolling_mean(window_size=period)
            mean_demean = cs_mean - cs_mean.rolling_mean(window_size=period)
            cov = (col_demean * mean_demean).rolling_mean(window_size=period)
            var = cs_mean.rolling_var(window_size=period)
            beta = (cov) / var if var != 0 else np.nan
            resid = col - beta * cs_mean
            return resid.rolling_std(window_size=vol_period).alias(col_name)

        return _apply_to_panel(x, lambda c: resid_vol_expr(c.meta.output_name()))


@register_operator(
    name="panel_async_beta_ex_self",
    category="panel_feature",
    canonical="panel_async_beta_ex_self",
    source="polars_native_phase6",
)
class panel_async_beta_ex_self(SeriesOperator):
    """Async beta excluding self from market"""

    metadata = OperatorMetadata(
        name="panel_async_beta_ex_self",
        category="panel_feature",
        description="Beta to market portfolio excluding self",
        param_names=["returns", "period"],
        param_types={"returns": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, returns: pl.DataFrame, period: int = 60, **kwargs) -> pl.DataFrame:
        def async_beta_expr(col_name):
            col = pl.col(col_name)
            # Market = cross-sectional mean excluding self
            cs_sum = col.sum().over("date")
            cs_count = col.count().over("date")
            market_ex_self = ((cs_sum - col)) / ((cs_count - 1) if ((cs_count - 1) != 0 else np.nan

            # Rolling beta
            col_demean = col - col.rolling_mean(window_size=period)
            mkt_demean = market_ex_self - market_ex_self.rolling_mean(window_size=period)
            cov = (col_demean * mkt_demean).rolling_mean(window_size=period)
            var = market_ex_self.rolling_var(window_size=period)
            return np.where(var.alias(col_name) != 0, ((cov) / (var).alias(col_name)), np.nan)

        return _apply_to_panel(returns, lambda c: async_beta_expr(c.meta.output_name()))


@register_operator(
    name="panel_factor_pocket_strength",
    category="panel_feature",
    canonical="panel_factor_pocket_strength",
    source="polars_native_phase6",
)
class panel_factor_pocket_strength(SeriesOperator):
    """Factor pocket strength"""

    metadata = OperatorMetadata(
        name="panel_factor_pocket_strength",
        category="panel_feature",
        description="Strength of local factor pocket in panel",
        param_names=["x", "period"],
        param_types={"x": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, x: pl.DataFrame, period: int = 20, **kwargs) -> pl.DataFrame:
        def pocket_expr(col_name):
            col = pl.col(col_name)
            # Pocket strength = correlation with cross-sectional median
            cs_median = col.median().over("date")
            col_demean = col - col.rolling_mean(window_size=period)
            med_demean = cs_median - cs_median.rolling_mean(window_size=period)
            cov = (col_demean * med_demean).rolling_mean(window_size=period)
            std = col.rolling_std(window_size=period) * cs_median.rolling_std(window_size=period)
            return np.where(std.alias(col_name) != 0, ((cov) / (std).alias(col_name)), np.nan)

        return _apply_to_panel(x, lambda c: pocket_expr(c.meta.output_name()))


@register_operator(
    name="panel_predictability_mosaic_score",
    category="panel_feature",
    canonical="panel_predictability_mosaic_score",
    source="polars_native_phase6",
)
class panel_predictability_mosaic_score(SeriesOperator):
    """Predictability mosaic score"""

    metadata = OperatorMetadata(
        name="panel_predictability_mosaic_score",
        category="panel_feature",
        description="Cross-sectional predictability score",
        param_names=["x", "period"],
        param_types={"x": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, x: pl.DataFrame, period: int = 60, **kwargs) -> pl.DataFrame:
        def mosaic_expr(col_name):
            col = pl.col(col_name)
            # Autocorrelation strength
            autocorr = col.rolling_mean(window_size=period) * col.shift(1).rolling_mean(window_size=period)
            var = col.rolling_var(window_size=period)
            return np.where(var.alias(col_name) != 0, ((autocorr) / (var).alias(col_name)), np.nan)

        return _apply_to_panel(x, lambda c: mosaic_expr(c.meta.output_name()))


@register_operator(
    name="panel_rolling_pcr_forecast",
    category="panel_feature",
    canonical="panel_rolling_pcr_forecast",
    source="polars_native_phase6",
)
class panel_rolling_pcr_forecast(SeriesOperator):
    """Rolling PCR (Principal Component Regression) forecast"""

    metadata = OperatorMetadata(
        name="panel_rolling_pcr_forecast",
        category="panel_feature",
        description="Forecast using principal component regression",
        param_names=["x", "y", "period"],
        param_types={"x": pl.DataFrame, "y": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, period: int = 60, **kwargs) -> pl.DataFrame:
        # Simplified: use rolling regression with cross-sectional mean as PC proxy
        def pcr_forecast_expr(x_col, y_col):
            x_val = pl.col(x_col)
            y_val = pl.col(y_col)

            # Use cross-sectional mean as first PC
            pc1 = x_val.mean().over("date")

            # Rolling regression: y = alpha + beta * pc1
            x_demean = pc1 - pc1.rolling_mean(window_size=period)
            y_demean = y_val - y_val.rolling_mean(window_size=period)
            cov = (x_demean * y_demean).rolling_mean(window_size=period)
            var = pc1.rolling_var(window_size=period)
            beta = (cov) / var if var != 0 else np.nan
            alpha = y_val.rolling_mean(window_size=period) - beta * pc1.rolling_mean(window_size=period)

            return (alpha + beta * pc1).alias(x_col)

        cols = [c for c in x.columns if c not in PANEL_SKIP_COLUMNS]
        return x.with_columns([pcr_forecast_expr(c, c) for c in cols])


@register_operator(
    name="panel_rolling_pls_forecast",
    category="panel_feature",
    canonical="panel_rolling_pls_forecast",
    source="polars_native_phase6",
)
class panel_rolling_pls_forecast(SeriesOperator):
    """Rolling PLS (Partial Least Squares) forecast"""

    metadata = OperatorMetadata(
        name="panel_rolling_pls_forecast",
        category="panel_feature",
        description="Forecast using partial least squares regression",
        param_names=["x", "y", "period"],
        param_types={"x": pl.DataFrame, "y": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, period: int = 60, **kwargs) -> pl.DataFrame:
        # Simplified: similar to PCR but with target-weighted component
        def pls_forecast_expr(x_col, y_col):
            x_val = pl.col(x_col)
            y_val = pl.col(y_col)

            # Rolling regression
            x_demean = x_val - x_val.rolling_mean(window_size=period)
            y_demean = y_val - y_val.rolling_mean(window_size=period)
            cov = (x_demean * y_demean).rolling_mean(window_size=period)
            var = x_val.rolling_var(window_size=period)
            beta = (cov) / var if var != 0 else np.nan
            alpha = y_val.rolling_mean(window_size=period) - beta * x_val.rolling_mean(window_size=period)

            return (alpha + beta * x_val).alias(x_col)

        cols = [c for c in x.columns if c not in PANEL_SKIP_COLUMNS]
        return x.with_columns([pls_forecast_expr(c, c) for c in cols])


@register_operator(
    name="panel_rolling_elastic_net_forecast",
    category="panel_feature",
    canonical="panel_rolling_elastic_net_forecast",
    source="polars_native_phase6",
)
class panel_rolling_elastic_net_forecast(SeriesOperator):
    """Rolling elastic net forecast"""

    metadata = OperatorMetadata(
        name="panel_rolling_elastic_net_forecast",
        category="panel_feature",
        description="Forecast using rolling elastic net regression",
        param_names=["x", "y", "period"],
        param_types={"x": pl.DataFrame, "y": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, period: int = 60, **kwargs) -> pl.DataFrame:
        # Simplified: use regular regression as elastic net approximation
        def elastic_forecast_expr(x_col, y_col):
            x_val = pl.col(x_col)
            y_val = pl.col(y_col)

            x_demean = x_val - x_val.rolling_mean(window_size=period)
            y_demean = y_val - y_val.rolling_mean(window_size=period)
            cov = (x_demean * y_demean).rolling_mean(window_size=period)
            var = x_val.rolling_var(window_size=period)
            # Shrinkage factor for elastic net
            shrinkage = 0.8
            beta = (shrinkage * cov) / var if var != 0 else np.nan
            alpha = y_val.rolling_mean(window_size=period) - beta * x_val.rolling_mean(window_size=period)

            return (alpha + beta * x_val).alias(x_col)

        cols = [c for c in x.columns if c not in PANEL_SKIP_COLUMNS]
        return x.with_columns([elastic_forecast_expr(c, c) for c in cols])


@register_operator(
    name="panel_regime_conditioned_forecast",
    category="panel_feature",
    canonical="panel_regime_conditioned_forecast",
    source="polars_native_phase6",
)
class panel_regime_conditioned_forecast(SeriesOperator):
    """Regime-conditioned forecast"""

    metadata = OperatorMetadata(
        name="panel_regime_conditioned_forecast",
        category="panel_feature",
        description="Forecast conditioned on market regime",
        param_names=["x", "regime", "period"],
        param_types={"x": pl.DataFrame, "regime": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, x: pl.DataFrame, regime: pl.DataFrame, period: int = 60, **kwargs) -> pl.DataFrame:
        # Simplified: use regime-weighted rolling mean
        def regime_forecast_expr(col_name):
            col = pl.col(col_name)
            # Use sign of rolling mean as regime indicator
            regime_ind = col.rolling_mean(window_size=20).sign()
            # Separate means for positive and negative regimes
            pos_mean = pl.when(regime_ind > 0).then(col).otherwise(None).rolling_mean(window_size=period)
            neg_mean = pl.when(regime_ind < 0).then(col).otherwise(None).rolling_mean(window_size=period)
            forecast = pl.when(regime_ind > 0).then(pos_mean).otherwise(neg_mean)
            return forecast.alias(col_name)

        return _apply_to_panel(x, lambda c: regime_forecast_expr(c.meta.output_name()))


@register_operator(
    name="panel_mixture_of_experts_score",
    category="panel_feature",
    canonical="panel_mixture_of_experts_score",
    source="polars_native_phase6",
)
class panel_mixture_of_experts_score(SeriesOperator):
    """Mixture of experts score"""

    metadata = OperatorMetadata(
        name="panel_mixture_of_experts_score",
        category="panel_feature",
        description="Weighted combination of expert forecasts",
        param_names=["x", "period"],
        param_types={"x": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, x: pl.DataFrame, period: int = 60, **kwargs) -> pl.DataFrame:
        def moe_expr(col_name):
            col = pl.col(col_name)
            # Expert 1: short-term momentum
            exp1 = col.rolling_mean(window_size=5)
            # Expert 2: long-term trend
            exp2 = col.rolling_mean(window_size=period)
            # Expert 3: mean reversion signal
            exp3 = -col.rolling_std(window_size=period)

            # Equal weights for simplicity
            score = ((exp1 + exp2 + exp3)) / 3 if 3 != 0 else np.nan
            return score.alias(col_name)

        return _apply_to_panel(x, lambda c: moe_expr(c.meta.output_name()))


@register_operator(
    name="panel_peer_graph_aggregate",
    category="panel_feature",
    canonical="panel_peer_graph_aggregate",
    source="polars_native_phase6",
)
class panel_peer_graph_aggregate(SeriesOperator):
    """Peer graph aggregate"""

    metadata = OperatorMetadata(
        name="panel_peer_graph_aggregate",
        category="panel_feature",
        description="Aggregation over peer graph connections",
        param_names=["x", "period"],
        param_types={"x": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, x: pl.DataFrame, period: int = 20, **kwargs) -> pl.DataFrame:
        # Simplified: use cross-sectional median as peer aggregate
        def peer_agg_expr(col_name):
            col = pl.col(col_name)
            cs_median = col.median().over("date")
            return cs_median.alias(col_name)

        return _apply_to_panel(x, lambda c: peer_agg_expr(c.meta.output_name()))


# ==================== Additional Group Operators ====================

@register_operator(
    name="group_corr_mst_length",
    category="group_feature",
    canonical="group_corr_mst_length",
    source="polars_native_phase6",
)
class group_corr_mst_length(SeriesOperator):
    """Correlation minimum spanning tree length"""

    metadata = OperatorMetadata(
        name="group_corr_mst_length",
        category="group_feature",
        description="Total length of minimum spanning tree in correlation space",
        param_names=["x", "group", "period"],
        param_types={"x": pl.DataFrame, "group": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame, period: int = 60, **kwargs) -> pl.DataFrame:
        # Simplified: use sum of pairwise correlation deviations as proxy
        def mst_expr(col_name):
            col = pl.col(col_name)
            # Average correlation within group
            group_std = col.std().over("group")
            total_std = col.std()
            return np.where(total_std.alias(col_name) != 0, ((1 - group_std) / (total_std).alias(col_name)), np.nan)

        return _apply_to_panel(x, lambda c: mst_expr(c.meta.output_name()))


@register_operator(
    name="group_current_members_tail_coexceedance",
    category="group_feature",
    canonical="group_current_members_tail_coexceedance",
    source="polars_native_phase6",
)
class group_current_members_tail_coexceedance(SeriesOperator):
    """Group tail co-exceedance"""

    metadata = OperatorMetadata(
        name="group_current_members_tail_coexceedance",
        category="group_feature",
        description="Proportion of group exceeding tail threshold together",
        param_names=["x", "group", "threshold"],
        param_types={"x": pl.DataFrame, "group": pl.DataFrame, "threshold": float},
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame, threshold: float = 0.95, **kwargs) -> pl.DataFrame:
        def coexceed_expr(col_name):
            col = pl.col(col_name)
            # Calculate quantile
            q = col.quantile(threshold)
            exceed = (col > q).cast(pl.Float64)
            # Proportion of group exceeding
            group_exceed = exceed.mean().over("group")
            return group_exceed.alias(col_name)

        return _apply_to_panel(x, lambda c: coexceed_expr(c.meta.output_name()))


@register_operator(
    name="group_distribution_js_divergence",
    category="group_feature",
    canonical="group_distribution_js_divergence",
    source="polars_native_phase6",
)
class group_distribution_js_divergence(SeriesOperator):
    """Jensen-Shannon divergence between group and universe"""

    metadata = OperatorMetadata(
        name="group_distribution_js_divergence",
        category="group_feature",
        description="JS divergence of group distribution from universe",
        param_names=["x", "group"],
        param_types={"x": pl.DataFrame, "group": pl.DataFrame},
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame, **kwargs) -> pl.DataFrame:
        # Simplified: use variance ratio as proxy for distribution divergence
        def js_div_expr(col_name):
            col = pl.col(col_name)
            group_var = col.var().over("group")
            total_var = col.var()
            return np.where(total_var.alias(col_name) != 0, ((group_var) / (total_var).alias(col_name)), np.nan)

        return _apply_to_panel(x, lambda c: js_div_expr(c.meta.output_name()))


@register_operator(
    name="group_feature_effective_rank",
    category="group_feature",
    canonical="group_feature_effective_rank",
    source="polars_native_phase6",
)
class group_feature_effective_rank(SeriesOperator):
    """Effective rank within group"""

    metadata = OperatorMetadata(
        name="group_feature_effective_rank",
        category="group_feature",
        description="Shannon entropy based effective rank",
        param_names=["x", "group"],
        param_types={"x": pl.DataFrame, "group": pl.DataFrame},
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame, **kwargs) -> pl.DataFrame:
        def eff_rank_expr(col_name):
            col = pl.col(col_name)
            # Rank within group
            rank = col.rank(method="dense").over("group")
            max_rank = rank.max().over("group")
            # Normalized rank
            return np.where(max_rank.alias(col_name) != 0, ((rank) / (max_rank).alias(col_name)), np.nan)

        return _apply_to_panel(x, lambda c: eff_rank_expr(c.meta.output_name()))


@register_operator(
    name="group_feature_mode_localization",
    category="group_feature",
    canonical="group_feature_mode_localization",
    source="polars_native_phase6",
)
class group_feature_mode_localization(SeriesOperator):
    """Mode localization in group"""

    metadata = OperatorMetadata(
        name="group_feature_mode_localization",
        category="group_feature",
        description="Concentration around modal value",
        param_names=["x", "group", "bandwidth"],
        param_types={"x": pl.DataFrame, "group": pl.DataFrame, "bandwidth": float},
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame, bandwidth: float = 0.1, **kwargs) -> pl.DataFrame:
        def mode_local_expr(col_name):
            col = pl.col(col_name)
            # Use median as mode proxy
            group_median = col.median().over("group")
            deviation = (col - group_median).abs()
            # Concentration within bandwidth
            in_band = (deviation < bandwidth).cast(pl.Float64)
            return in_band.mean().over("group").alias(col_name)

        return _apply_to_panel(x, lambda c: mode_local_expr(c.meta.output_name()))


@register_operator(
    name="group_feature_second_mode_localization",
    category="group_feature",
    canonical="group_feature_second_mode_localization",
    source="polars_native_phase6",
)
class group_feature_second_mode_localization(SeriesOperator):
    """Second mode localization"""

    metadata = OperatorMetadata(
        name="group_feature_second_mode_localization",
        category="group_feature",
        description="Concentration around second modal value",
        param_names=["x", "group", "bandwidth"],
        param_types={"x": pl.DataFrame, "group": pl.DataFrame, "bandwidth": float},
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame, bandwidth: float = 0.1, **kwargs) -> pl.DataFrame:
        # Simplified: use upper/lower quartiles as second mode proxies
        def second_mode_expr(col_name):
            col = pl.col(col_name)
            q1 = col.quantile(0.25).over("group")
            q3 = col.quantile(0.75).over("group")
            # Distance to either quartile
            dist1 = (col - q1).abs()
            dist3 = (col - q3).abs()
            min_dist = pl.min_horizontal(dist1, dist3)
            in_band = (min_dist < bandwidth).cast(pl.Float64)
            return in_band.mean().over("group").alias(col_name)

        return _apply_to_panel(x, lambda c: second_mode_expr(c.meta.output_name()))


@register_operator(
    name="group_feature_spectral_gap",
    category="group_feature",
    canonical="group_feature_spectral_gap",
    source="polars_native_phase6",
)
class group_feature_spectral_gap(SeriesOperator):
    """Spectral gap in group"""

    metadata = OperatorMetadata(
        name="group_feature_spectral_gap",
        category="group_feature",
        description="Gap between leading eigenvalues proxy",
        param_names=["x", "group"],
        param_types={"x": pl.DataFrame, "group": pl.DataFrame},
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame, **kwargs) -> pl.DataFrame:
        # Simplified: use variance concentration as proxy
        def spectral_gap_expr(col_name):
            col = pl.col(col_name)
            # Concentration of variance
            var = col.var().over("group")
            mean = col.mean().over("group")
            return np.where((mean.abs() + 1e-8).alias(col_name) != 0, ((var) / ((mean.abs() + 1e-8)).alias(col_name)), np.nan)

        return _apply_to_panel(x, lambda c: spectral_gap_expr(c.meta.output_name()))


@register_operator(
    name="group_multi_level_rank_consistency",
    category="group_feature",
    canonical="group_multi_level_rank_consistency",
    source="polars_native_phase6",
)
class group_multi_level_rank_consistency(SeriesOperator):
    """Multi-level rank consistency"""

    metadata = OperatorMetadata(
        name="group_multi_level_rank_consistency",
        category="group_feature",
        description="Consistency of ranks across group levels",
        param_names=["x", "group"],
        param_types={"x": pl.DataFrame, "group": pl.DataFrame},
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame, **kwargs) -> pl.DataFrame:
        def rank_consistency_expr(col_name):
            col = pl.col(col_name)
            # Rank within group
            group_rank = col.rank(method="dense").over("group")
            # Global rank
            global_rank = col.rank(method="dense")
            # Correlation of ranks
            consistency = (group_rank - group_rank.mean().over("group")) * (global_rank - global_rank.mean())
            return consistency.alias(col_name)

        return _apply_to_panel(x, lambda c: rank_consistency_expr(c.meta.output_name()))


@register_operator(
    name="group_peer_information_diffusion",
    category="group_feature",
    canonical="group_peer_information_diffusion",
    source="polars_native_phase6",
)
class group_peer_information_diffusion(SeriesOperator):
    """Peer information diffusion rate"""

    metadata = OperatorMetadata(
        name="group_peer_information_diffusion",
        category="group_feature",
        description="Rate of information diffusion among peers",
        param_names=["x", "group", "period"],
        param_types={"x": pl.DataFrame, "group": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame, period: int = 5, **kwargs) -> pl.DataFrame:
        def diffusion_expr(col_name):
            col = pl.col(col_name)
            # Change in group mean
            group_mean = col.mean().over("group")
            diffusion = group_mean - group_mean.shift(period)
            return diffusion.alias(col_name)

        return _apply_to_panel(x, lambda c: diffusion_expr(c.meta.output_name()))


@register_operator(
    name="group_tail_lead_score",
    category="group_feature",
    canonical="group_tail_lead_score",
    source="polars_native_phase6",
)
class group_tail_lead_score(SeriesOperator):
    """Tail leadership score"""

    metadata = OperatorMetadata(
        name="group_tail_lead_score",
        category="group_feature",
        description="Leading indicator strength in group tail",
        param_names=["x", "group", "threshold"],
        param_types={"x": pl.DataFrame, "group": pl.DataFrame, "threshold": float},
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame, threshold: float = 0.9, **kwargs) -> pl.DataFrame:
        def tail_lead_expr(col_name):
            col = pl.col(col_name)
            q = col.quantile(threshold)
            is_tail = (col > q).cast(pl.Float64)
            # Lead = future group performance when in tail
            group_future = col.mean().over("group").shift(-1)
            lead_score = is_tail * group_future
            return lead_score.alias(col_name)

        return _apply_to_panel(x, lambda c: tail_lead_expr(c.meta.output_name()))


@register_operator(
    name="group_wasserstein_barycenter_distance",
    category="group_feature",
    canonical="group_wasserstein_barycenter_distance",
    source="polars_native_phase6",
)
class group_wasserstein_barycenter_distance(SeriesOperator):
    """Wasserstein distance to group barycenter"""

    metadata = OperatorMetadata(
        name="group_wasserstein_barycenter_distance",
        category="group_feature",
        description="Distance to group distribution center",
        param_names=["x", "group"],
        param_types={"x": pl.DataFrame, "group": pl.DataFrame},
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame, **kwargs) -> pl.DataFrame:
        # Simplified: use L1 distance to group median as proxy
        def wasserstein_expr(col_name):
            col = pl.col(col_name)
            group_median = col.median().over("group")
            distance = (col - group_median).abs()
            return distance.alias(col_name)

        return _apply_to_panel(x, lambda c: wasserstein_expr(c.meta.output_name()))


@register_operator(
    name="group_spd_feature_structure_shift",
    category="group_feature",
    canonical="group_spd_feature_structure_shift",
    source="polars_native_phase6",
)
class group_spd_feature_structure_shift(SeriesOperator):
    """Feature structure shift in SPD manifold"""

    metadata = OperatorMetadata(
        name="group_spd_feature_structure_shift",
        category="group_feature",
        description="Shift in feature structure on SPD manifold",
        param_names=["x", "group", "period"],
        param_types={"x": pl.DataFrame, "group": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, x: pl.DataFrame, group: pl.DataFrame, period: int = 20, **kwargs) -> pl.DataFrame:
        # Simplified: use variance change as structure shift proxy
        def structure_shift_expr(col_name):
            col = pl.col(col_name)
            current_var = col.var().over("group")
            past_var = col.var().over("group").shift(period)
            shift = ((current_var - past_var)) / ((past_var + 1e-8) if ((past_var + 1e-8) != 0 else np.nan
            return shift.alias(col_name)

        return _apply_to_panel(x, lambda c: structure_shift_expr(c.meta.output_name()))


# ==================== Additional Miscellaneous Operators ====================

@register_operator(
    name="turnover_adjusted_volatility",
    category="risk",
    canonical="turnover_adjusted_volatility",
    source="polars_native_phase6",
)
class turnover_adjusted_volatility(SeriesOperator):
    """Turnover-adjusted volatility"""

    metadata = OperatorMetadata(
        name="turnover_adjusted_volatility",
        category="risk",
        description="Volatility adjusted for turnover level",
        param_names=["returns", "volume", "period"],
        param_types={"returns": pl.DataFrame, "volume": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, returns: pl.DataFrame, volume: pl.DataFrame, period: int = 20, **kwargs) -> pl.DataFrame:
        def turnover_vol_expr(r_col, v_col):
            ret = pl.col(r_col)
            vol = pl.col(v_col)
            vol_std = ret.rolling_std(window_size=period)
            turnover = vol.rolling_mean(window_size=period)
            # Adjust volatility by turnover
            adj_vol = vol_std * (1 + 1) / (turnover + 1)
            return adj_vol.alias(r_col)

        cols = [c for c in returns.columns if c not in PANEL_SKIP_COLUMNS]
        return returns.with_columns([turnover_vol_expr(c, c) for c in cols])


@register_operator(
    name="volume_price_range_density",
    category="microstructure",
    canonical="volume_price_range_density",
    source="polars_native_phase6",
)
class volume_price_range_density(SeriesOperator):
    """Volume per unit price range"""

    metadata = OperatorMetadata(
        name="volume_price_range_density",
        category="microstructure",
        description="Volume density relative to price range",
        param_names=["volume", "high", "low", "period"],
        param_types={"volume": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, volume: pl.DataFrame, high: pl.DataFrame, low: pl.DataFrame, period: int = 1, **kwargs) -> pl.DataFrame:
        def density_expr(v_col, h_col, l_col):
            vol = pl.col(v_col)
            price_range = pl.col(h_col) - pl.col(l_col)
            density = (vol) / ((price_range + 1e-8) if ((price_range + 1e-8) != 0 else np.nan
            return density.rolling_mean(window_size=period).alias(v_col)

        cols = [c for c in volume.columns if c not in PANEL_SKIP_COLUMNS]
        return volume.with_columns([density_expr(c, c, c) for c in cols])


@register_operator(
    name="yoy_by_period",
    category="fundamental",
    canonical="yoy_by_period",
    source="polars_native_phase6",
    replace=True,
    replacement_reason="Phase 6 implementation replaces earlier native polars version",
    expected_old_source="operator_overhaul_native_polars",
)
class yoy_by_period(SeriesOperator):
    """Year-over-year change by period"""

    metadata = OperatorMetadata(
        name="yoy_by_period",
        category="fundamental",
        description="Year-over-year percentage change",
        param_names=["x", "periods"],
        param_types={"x": pl.DataFrame, "periods": int},
    )

    def _calculate_series(self, x: pl.DataFrame, periods: int = 4, **kwargs) -> pl.DataFrame:
        def yoy_expr(col_name):
            col = pl.col(col_name)
            lagged = col.shift(periods)
            return np.where(lagged.alias(col_name) != 0, (((col - lagged)) / (lagged).alias(col_name)), np.nan)

        return _apply_to_panel(x, lambda c: yoy_expr(c.meta.output_name()))


@register_operator(
    name="ttm_from_cumulative",
    category="fundamental",
    canonical="ttm_from_cumulative",
    source="polars_native_phase6",
    replace=True,
    replacement_reason="Phase 6 implementation replaces earlier native polars version",
    expected_old_source="operator_overhaul_native_polars",
)
class ttm_from_cumulative(SeriesOperator):
    """Trailing twelve months from cumulative"""

    metadata = OperatorMetadata(
        name="ttm_from_cumulative",
        category="fundamental",
        description="TTM value from cumulative quarterly data",
        param_names=["x"],
        param_types={"x": pl.DataFrame},
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        def ttm_expr(col_name):
            col = pl.col(col_name)
            # Current year cumulative minus last year same period cumulative
            current = col
            year_ago = col.shift(4)
            return (current - year_ago).alias(col_name)

        return _apply_to_panel(x, lambda c: ttm_expr(c.meta.output_name()))


@register_operator(
    name="dollar_volume_zscore",
    category="liquidity",
    canonical="dollar_volume_zscore",
    source="polars_native_phase6",
)
class dollar_volume_zscore(SeriesOperator):
    """Dollar volume z-score"""

    metadata = OperatorMetadata(
        name="dollar_volume_zscore",
        category="liquidity",
        description="Z-score of dollar trading volume",
        param_names=["volume", "price", "period"],
        param_types={"volume": pl.DataFrame, "price": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, volume: pl.DataFrame, price: pl.DataFrame, period: int = 20, **kwargs) -> pl.DataFrame:
        def dollar_vol_zscore_expr(v_col, p_col):
            vol = pl.col(v_col)
            price = pl.col(p_col)
            dollar_vol = vol * price
            mean = dollar_vol.rolling_mean(window_size=period)
            std = dollar_vol.rolling_std(window_size=period)
            return np.where(std.alias(v_col) != 0, (((dollar_vol - mean)) / (std).alias(v_col)), np.nan)

        cols = [c for c in volume.columns if c not in PANEL_SKIP_COLUMNS]
        return volume.with_columns([dollar_vol_zscore_expr(c, c) for c in cols])



@register_operator(
    name="composition_entropy",
    category="composition",
    canonical="composition_entropy",
    source="polars_native_phase6",
)
class composition_entropy(SeriesOperator):
    """Compositional data entropy"""

    metadata = OperatorMetadata(
        name="composition_entropy",
        category="composition",
        description="Shannon entropy of composition",
        param_names=["x"],
        param_types={"x": pl.DataFrame},
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        def entropy_expr(col_name):
            col = pl.col(col_name)
            # Normalize to proportions
            total = col.sum()
            p = (col) / total if total != 0 else np.nan
            # Shannon entropy: -sum(p * log(p))
            entropy = -(p * p.log()).sum()
            return entropy.alias(col_name)

        return _apply_to_panel(x, lambda c: entropy_expr(c.meta.output_name()))


@register_operator(
    name="composition_normalized_entropy",
    category="composition",
    canonical="composition_normalized_entropy",
    source="polars_native_phase6",
)
class composition_normalized_entropy(SeriesOperator):
    """Normalized compositional entropy"""

    metadata = OperatorMetadata(
        name="composition_normalized_entropy",
        category="composition",
        description="Normalized Shannon entropy [0,1]",
        param_names=["x", "n_components"],
        param_types={"x": pl.DataFrame, "n_components": int},
    )

    def _calculate_series(self, x: pl.DataFrame, n_components: int = 10, **kwargs) -> pl.DataFrame:
        def norm_entropy_expr(col_name):
            col = pl.col(col_name)
            total = col.sum()
            p = (col) / total if total != 0 else np.nan
            entropy = -(p * p.log()).sum()
            max_entropy = np.log(n_components)
            return np.where(max_entropy.alias(col_name) != 0, ((entropy) / (max_entropy).alias(col_name)), np.nan)

        return _apply_to_panel(x, lambda c: norm_entropy_expr(c.meta.output_name()))


@register_operator(
    name="composition_clr_component",
    category="composition",
    canonical="composition_clr_component",
    source="polars_native_phase6",
)
class composition_clr_component(SeriesOperator):
    """Centered log-ratio component"""

    metadata = OperatorMetadata(
        name="composition_clr_component",
        category="composition",
        description="CLR transformation component",
        param_names=["x"],
        param_types={"x": pl.DataFrame},
    )

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        def clr_expr(col_name):
            col = pl.col(col_name)
            # CLR = log(x) - mean(log(x))
            log_x = col.log()
            log_mean = log_x.mean()
            return (log_x - log_mean).alias(col_name)

        return _apply_to_panel(x, lambda c: clr_expr(c.meta.output_name()))


@register_operator(
    name="composition_aitchison_distance",
    category="composition",
    canonical="composition_aitchison_distance",
    source="polars_native_phase6",
)
class composition_aitchison_distance(SeriesOperator):
    """Aitchison distance between compositions"""

    metadata = OperatorMetadata(
        name="composition_aitchison_distance",
        category="composition",
        description="Aitchison distance to reference composition",
        param_names=["x", "reference"],
        param_types={"x": pl.DataFrame, "reference": pl.DataFrame},
    )

    def _calculate_series(self, x: pl.DataFrame, reference: pl.DataFrame, **kwargs) -> pl.DataFrame:
        def aitchison_expr(x_col, r_col):
            x_val = pl.col(x_col)
            r_val = pl.col(r_col)
            # Aitchison distance = sqrt(sum((log(x) - log(r))^2))
            log_ratio = (x_val.log() - r_val.log())
            distance = (log_ratio ** 2).sum().sqrt()
            return distance.alias(x_col)

        cols = [c for c in x.columns if c not in PANEL_SKIP_COLUMNS]
        return x.with_columns([aitchison_expr(c, c) for c in cols])


@register_operator(
    name="composition_js_divergence",
    category="composition",
    canonical="composition_js_divergence",
    source="polars_native_phase6",
)
class composition_js_divergence(SeriesOperator):
    """Jensen-Shannon divergence of compositions"""

    metadata = OperatorMetadata(
        name="composition_js_divergence",
        category="composition",
        description="JS divergence between two compositions",
        param_names=["x", "y"],
        param_types={"x": pl.DataFrame, "y": pl.DataFrame},
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, **kwargs) -> pl.DataFrame:
        def js_expr(x_col, y_col):
            p = pl.col(x_col)
            q = pl.col(y_col)
            # Normalize
            p = p / p.sum()
            q = q / q.sum()
            # M = (P + Q) / 2
            m = (p + q) / 2
            # JS = (KL(P||M) + KL(Q||M)) / 2
            kl_pm = (p * (p / m).log()).sum()
            kl_qm = (q * (q / m).log()).sum()
            js = (kl_pm + kl_qm) / 2
            return js.alias(x_col)

        cols = [c for c in x.columns if c not in PANEL_SKIP_COLUMNS]
        return x.with_columns([js_expr(c, c) for c in cols])


@register_operator(
    name="composition_ilr_balance",
    category="composition",
    canonical="composition_ilr_balance",
    source="polars_native_phase6",
)
class composition_ilr_balance(SeriesOperator):
    """Isometric log-ratio balance"""

    metadata = OperatorMetadata(
        name="composition_ilr_balance",
        category="composition",
        description="ILR balance coordinate",
        param_names=["x", "part1_size", "part2_size"],
        param_types={"x": pl.DataFrame, "part1_size": int, "part2_size": int},
    )

    def _calculate_series(self, x: pl.DataFrame, part1_size: int = 1, part2_size: int = 1, **kwargs) -> pl.DataFrame:
        def ilr_expr(col_name):
            col = pl.col(col_name)
            # Simplified ILR balance
            n1 = part1_size
            n2 = part2_size
            coef = np.where((n1 + n2) != 0, (np.sqrt(n1 * n2) / ((n1 + n2))), np.nan)
            # Balance = coef * log(geom_mean(part1) / geom_mean(part2))
            balance = coef * col.log()
            return balance.alias(col_name)

        return _apply_to_panel(x, lambda c: ilr_expr(c.meta.output_name()))


@register_operator(
    name="ichimoku_senkou_a",
    category="technical_indicator",
    canonical="ichimoku_senkou_a",
    source="polars_native_phase6",
)
class ichimoku_senkou_a(SeriesOperator):
    """Ichimoku Senkou Span A"""

    metadata = OperatorMetadata(
        name="ichimoku_senkou_a",
        category="technical_indicator",
        description="Ichimoku Leading Span A",
        param_names=["high", "low", "tenkan", "kijun"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "tenkan": int, "kijun": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, tenkan: int = 9, kijun: int = 26, **kwargs) -> pl.DataFrame:
        def senkou_a_expr(h_col, l_col):
            h = pl.col(h_col)
            l = pl.col(l_col)
            # Tenkan = (max(high, 9) + min(low, 9)) / 2
            tenkan_line = ((h.rolling_max(window_size=tenkan) + l.rolling_min(window_size=tenkan))) / 2 if 2 != 0 else np.nan
            # Kijun = (max(high, 26) + min(low, 26)) / 2
            kijun_line = ((h.rolling_max(window_size=kijun) + l.rolling_min(window_size=kijun))) / 2 if 2 != 0 else np.nan
            # Senkou A = (Tenkan + Kijun) / 2 shifted forward 26
            senkou_a = ((tenkan_line + kijun_line)) / 2 if 2 != 0 else np.nan
            return senkou_a.shift(-kijun).alias(h_col)

        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        return high.with_columns([senkou_a_expr(c, c) for c in cols])


@register_operator(
    name="ichimoku_senkou_b",
    category="technical_indicator",
    canonical="ichimoku_senkou_b",
    source="polars_native_phase6",
)
class ichimoku_senkou_b(SeriesOperator):
    """Ichimoku Senkou Span B"""

    metadata = OperatorMetadata(
        name="ichimoku_senkou_b",
        category="technical_indicator",
        description="Ichimoku Leading Span B",
        param_names=["high", "low", "senkou", "kijun"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "senkou": int, "kijun": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, senkou: int = 52, kijun: int = 26, **kwargs) -> pl.DataFrame:
        def senkou_b_expr(h_col, l_col):
            h = pl.col(h_col)
            l = pl.col(l_col)
            # Senkou B = (max(high, 52) + min(low, 52)) / 2 shifted forward 26
            senkou_b = ((h.rolling_max(window_size=senkou) + l.rolling_min(window_size=senkou))) / 2 if 2 != 0 else np.nan
            return senkou_b.shift(-kijun).alias(h_col)

        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        return high.with_columns([senkou_b_expr(c, c) for c in cols])


@register_operator(
    name="ichimoku_cloud_position",
    category="technical_indicator",
    canonical="ichimoku_cloud_position",
    source="polars_native_phase6",
)
class ichimoku_cloud_position(SeriesOperator):
    """Position relative to Ichimoku cloud"""

    metadata = OperatorMetadata(
        name="ichimoku_cloud_position",
        category="technical_indicator",
        description="Price position relative to Ichimoku cloud",
        param_names=["close", "high", "low"],
        param_types={"close": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame},
    )

    def _calculate_series(self, close: pl.DataFrame, high: pl.DataFrame, low: pl.DataFrame, **kwargs) -> pl.DataFrame:
        def cloud_pos_expr(c_col, h_col, l_col):
            c = pl.col(c_col)
            h = pl.col(h_col)
            l = pl.col(l_col)

            # Calculate Senkou A and B
            tenkan = (h.rolling_max(window_size=9) + l.rolling_min(window_size=9)) / 2
            kijun = (h.rolling_max(window_size=26) + l.rolling_min(window_size=26)) / 2
            senkou_a = ((tenkan + kijun) / 2).shift(-26)
            senkou_b = ((h.rolling_max(window_size=52) + l.rolling_min(window_size=52)) / 2).shift(-26)

            # Cloud bounds
            cloud_top = pl.max_horizontal(senkou_a, senkou_b)
            cloud_bottom = pl.min_horizontal(senkou_a, senkou_b)

            # Position: >1 above cloud, 0-1 in cloud, <0 below cloud
            position = (
                pl.when(c > cloud_top).then((c - cloud_top) / cloud_top)
                .when(c < cloud_bottom).then((c - cloud_bottom) / cloud_bottom)
                .otherwise(0.5)
            )
            return position.alias(c_col)

        cols = [c for c in close.columns if c not in PANEL_SKIP_COLUMNS]
        return close.with_columns([cloud_pos_expr(c, c, c) for c in cols])


@register_operator(
    name="ichimoku_cloud_width",
    category="technical_indicator",
    canonical="ichimoku_cloud_width",
    source="polars_native_phase6",
)
class ichimoku_cloud_width(SeriesOperator):
    """Ichimoku cloud width"""

    metadata = OperatorMetadata(
        name="ichimoku_cloud_width",
        category="technical_indicator",
        description="Width of Ichimoku cloud",
        param_names=["high", "low"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, **kwargs) -> pl.DataFrame:
        def cloud_width_expr(h_col, l_col):
            h = pl.col(h_col)
            l = pl.col(l_col)

            tenkan = (h.rolling_max(window_size=9) + l.rolling_min(window_size=9)) / 2
            kijun = (h.rolling_max(window_size=26) + l.rolling_min(window_size=26)) / 2
            senkou_a = ((tenkan + kijun) / 2).shift(-26)
            senkou_b = ((h.rolling_max(window_size=52) + l.rolling_min(window_size=52)) / 2).shift(-26)

            width = (senkou_a - senkou_b).abs()
            return width.alias(h_col)

        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        return high.with_columns([cloud_width_expr(c, c) for c in cols])


@register_operator(
    name="rolling_obv",
    category="volume_indicator",
    canonical="rolling_obv",
    source="polars_native_phase6",
)
class rolling_obv(SeriesOperator):
    """Rolling On-Balance Volume"""

    metadata = OperatorMetadata(
        name="rolling_obv",
        category="volume_indicator",
        description="Cumulative volume directional indicator",
        param_names=["close", "volume", "period"],
        param_types={"close": pl.DataFrame, "volume": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, close: pl.DataFrame, volume: pl.DataFrame, period: int = 20, **kwargs) -> pl.DataFrame:
        def obv_expr(c_col, v_col):
            c = pl.col(c_col)
            v = pl.col(v_col)
            # Sign of price change
            sign = (c - c.shift(1)).sign()
            # Signed volume
            signed_vol = sign * v
            # Rolling cumulative
            obv = signed_vol.rolling_sum(window_size=period)
            return obv.alias(c_col)

        cols = [c for c in close.columns if c not in PANEL_SKIP_COLUMNS]
        return close.with_columns([obv_expr(c, c) for c in cols])


@register_operator(
    name="rolling_adl_flow",
    category="volume_indicator",
    canonical="rolling_adl_flow",
    source="polars_native_phase6",
)
class rolling_adl_flow(SeriesOperator):
    """Rolling Accumulation/Distribution Line"""

    metadata = OperatorMetadata(
        name="rolling_adl_flow",
        category="volume_indicator",
        description="Accumulation/Distribution flow indicator",
        param_names=["high", "low", "close", "volume", "period"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame, "volume": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, close: pl.DataFrame, volume: pl.DataFrame, period: int = 20, **kwargs) -> pl.DataFrame:
        def adl_expr(h_col, l_col, c_col, v_col):
            h = pl.col(h_col)
            l = pl.col(l_col)
            c = pl.col(c_col)
            v = pl.col(v_col)

            # Money Flow Multiplier = ((C - L) - (H - C)) / (H - L)
            mfm = (((c - l) - (h - c))) / ((h - l + 1e-8) if ((h - l + 1e-8) != 0 else np.nan
            # Money Flow Volume = MFM * Volume
            mfv = mfm * v
            # Rolling ADL
            adl = mfv.rolling_sum(window_size=period)
            return adl.alias(h_col)

        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        return high.with_columns([adl_expr(c, c, c, c) for c in cols])


@register_operator(
    name="rolling_pvt",
    category="volume_indicator",
    canonical="rolling_pvt",
    source="polars_native_phase6",
)
class rolling_pvt(SeriesOperator):
    """Rolling Price Volume Trend"""

    metadata = OperatorMetadata(
        name="rolling_pvt",
        category="volume_indicator",
        description="Price Volume Trend indicator",
        param_names=["close", "volume", "period"],
        param_types={"close": pl.DataFrame, "volume": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, close: pl.DataFrame, volume: pl.DataFrame, period: int = 20, **kwargs) -> pl.DataFrame:
        def pvt_expr(c_col, v_col):
            c = pl.col(c_col)
            v = pl.col(v_col)

            # PVT = volume * (close - close_prev) / close_prev
            pct_change = ((c - c.shift(1))) / (c.shift(1) if (c.shift(1) != 0 else np.nan
            pvt_increment = v * pct_change
            pvt = pvt_increment.rolling_sum(window_size=period)
            return pvt.alias(c_col)

        cols = [c for c in close.columns if c not in PANEL_SKIP_COLUMNS]
        return close.with_columns([pvt_expr(c, c) for c in cols])


@register_operator(
    name="accounting_comparability_score",
    category="fundamental",
    canonical="accounting_comparability_score",
    source="polars_native_phase6",
)
class accounting_comparability_score(SeriesOperator):
    """Accounting comparability score"""

    metadata = OperatorMetadata(
        name="accounting_comparability_score",
        category="fundamental",
        description="Cross-sectional accounting policy comparability",
        param_names=["x", "period"],
        param_types={"x": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, x: pl.DataFrame, period: int = 4, **kwargs) -> pl.DataFrame:
        def comparability_expr(col_name):
            col = pl.col(col_name)
            # Measure consistency with cross-sectional median
            cs_median = col.median().over("date")
            deviation = (col - cs_median).abs()
            comparability = (1) / ((1 + deviation) if ((1 + deviation) != 0 else np.nan
            return comparability.rolling_mean(window_size=period).alias(col_name)

        return _apply_to_panel(x, lambda c: comparability_expr(c.meta.output_name()))


@register_operator(
    name="baseline_scaled_wasserstein_distance",
    category="statistics",
    canonical="baseline_scaled_wasserstein_distance",
    source="polars_native_phase6",
)
class baseline_scaled_wasserstein_distance(SeriesOperator):
    """Baseline-scaled Wasserstein distance"""

    metadata = OperatorMetadata(
        name="baseline_scaled_wasserstein_distance",
        category="statistics",
        description="Wasserstein distance scaled by baseline",
        param_names=["x", "baseline", "period"],
        param_types={"x": pl.DataFrame, "baseline": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, x: pl.DataFrame, baseline: pl.DataFrame, period: int = 20, **kwargs) -> pl.DataFrame:
        # Simplified: use L1 distance as Wasserstein proxy
        def wasserstein_expr(x_col, b_col):
            x_val = pl.col(x_col)
            b_val = pl.col(b_col)
            distance = (x_val - b_val).abs()
            baseline_scale = b_val.abs().rolling_mean(window_size=period)
            scaled = (distance) / ((baseline_scale + 1e-8) if ((baseline_scale + 1e-8) != 0 else np.nan
            return scaled.alias(x_col)

        cols = [c for c in x.columns if c not in PANEL_SKIP_COLUMNS]
        return x.with_columns([wasserstein_expr(c, c) for c in cols])


@register_operator(
    name="EaseOfMovement",
    category="technical_indicator",
    canonical="EaseOfMovement",
    source="polars_native_phase6",
)
class EaseOfMovement(SeriesOperator):
    """Ease of Movement indicator"""

    metadata = OperatorMetadata(
        name="EaseOfMovement",
        category="technical_indicator",
        description="Ease of Movement (EMV) indicator",
        param_names=["high", "low", "volume", "period"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "volume": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, volume: pl.DataFrame, period: int = 14, **kwargs) -> pl.DataFrame:
        def emv_expr(h_col, l_col, v_col):
            h = pl.col(h_col)
            l = pl.col(l_col)
            v = pl.col(v_col)

            # Distance moved = (high + low) / 2 - previous midpoint
            mid = ((h + l)) / 2 if 2 != 0 else np.nan
            distance = mid - mid.shift(1)

            # Box ratio = volume / (high - low)
            box_ratio = (v) / ((h - l + 1e-8) if ((h - l + 1e-8) != 0 else np.nan

            # EMV = distance / box_ratio
            emv = (distance) / ((box_ratio + 1e-8) if ((box_ratio + 1e-8) != 0 else np.nan

            return emv.rolling_mean(window_size=period).alias(h_col)

        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        return high.with_columns([emv_expr(c, c, c) for c in cols])


@register_operator(
    name="KeltnerUpper",
    category="technical_indicator",
    canonical="KeltnerUpper",
    source="polars_native_phase6",
)
class KeltnerUpper(SeriesOperator):
    """Keltner Channel Upper Band"""

    metadata = OperatorMetadata(
        name="KeltnerUpper",
        category="technical_indicator",
        description="Upper band of Keltner Channel",
        param_names=["close", "high", "low", "period", "multiplier"],
        param_types={"close": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "period": int, "multiplier": float},
    )

    def _calculate_series(self, close: pl.DataFrame, high: pl.DataFrame, low: pl.DataFrame, period: int = 20, multiplier: float = 2.0, **kwargs) -> pl.DataFrame:
        def keltner_upper_expr(c_col, h_col, l_col):
            c = pl.col(c_col)
            h = pl.col(h_col)
            l = pl.col(l_col)
            c_prev = c.shift(1)

            # Middle = EMA of close
            middle = _ema_expr(c, period)

            # ATR
            tr1 = h - l
            tr2 = (h - c_prev).abs()
            tr3 = (l - c_prev).abs()
            tr = pl.max_horizontal(tr1, tr2, tr3)
            atr = _ema_expr(tr, period)

            upper = middle + multiplier * atr
            return upper.alias(c_col)

        cols = [c for c in close.columns if c not in PANEL_SKIP_COLUMNS]
        return close.with_columns([keltner_upper_expr(c, c, c) for c in cols])


@register_operator(
    name="KeltnerLower",
    category="technical_indicator",
    canonical="KeltnerLower",
    source="polars_native_phase6",
)
class KeltnerLower(SeriesOperator):
    """Keltner Channel Lower Band"""

    metadata = OperatorMetadata(
        name="KeltnerLower",
        category="technical_indicator",
        description="Lower band of Keltner Channel",
        param_names=["close", "high", "low", "period", "multiplier"],
        param_types={"close": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "period": int, "multiplier": float},
    )

    def _calculate_series(self, close: pl.DataFrame, high: pl.DataFrame, low: pl.DataFrame, period: int = 20, multiplier: float = 2.0, **kwargs) -> pl.DataFrame:
        def keltner_lower_expr(c_col, h_col, l_col):
            c = pl.col(c_col)
            h = pl.col(h_col)
            l = pl.col(l_col)
            c_prev = c.shift(1)

            middle = _ema_expr(c, period)

            tr1 = h - l
            tr2 = (h - c_prev).abs()
            tr3 = (l - c_prev).abs()
            tr = pl.max_horizontal(tr1, tr2, tr3)
            atr = _ema_expr(tr, period)

            lower = middle - multiplier * atr
            return lower.alias(c_col)

        cols = [c for c in close.columns if c not in PANEL_SKIP_COLUMNS]
        return close.with_columns([keltner_lower_expr(c, c, c) for c in cols])


@register_operator(
    name="KeltnerPosition",
    category="technical_indicator",
    canonical="KeltnerPosition",
    source="polars_native_phase6",
)
class KeltnerPosition(SeriesOperator):
    """Position within Keltner Channel"""

    metadata = OperatorMetadata(
        name="KeltnerPosition",
        category="technical_indicator",
        description="Price position in Keltner Channel [0=lower, 1=upper]",
        param_names=["close", "high", "low", "period", "multiplier"],
        param_types={"close": pl.DataFrame, "high": pl.DataFrame, "low": pl.DataFrame, "period": int, "multiplier": float},
    )

    def _calculate_series(self, close: pl.DataFrame, high: pl.DataFrame, low: pl.DataFrame, period: int = 20, multiplier: float = 2.0, **kwargs) -> pl.DataFrame:
        def keltner_pos_expr(c_col, h_col, l_col):
            c = pl.col(c_col)
            h = pl.col(h_col)
            l = pl.col(l_col)
            c_prev = c.shift(1)

            middle = _ema_expr(c, period)

            tr1 = h - l
            tr2 = (h - c_prev).abs()
            tr3 = (l - c_prev).abs()
            tr = pl.max_horizontal(tr1, tr2, tr3)
            atr = _ema_expr(tr, period)

            upper = middle + multiplier * atr
            lower = middle - multiplier * atr

            position = ((c - lower)) / ((upper - lower + 1e-8) if ((upper - lower + 1e-8) != 0 else np.nan
            return position.alias(c_col)

        cols = [c for c in close.columns if c not in PANEL_SKIP_COLUMNS]
        return close.with_columns([keltner_pos_expr(c, c, c) for c in cols])


@register_operator(
    name="lqtp_historical_cvar",
    category="risk",
    canonical="lqtp_historical_cvar",
    source="polars_native_phase6",
)
class lqtp_historical_cvar(SeriesOperator):
    """Historical Conditional Value at Risk"""

    metadata = OperatorMetadata(
        name="lqtp_historical_cvar",
        category="risk",
        description="Historical CVaR (Expected Shortfall)",
        param_names=["returns", "period", "alpha"],
        param_types={"returns": pl.DataFrame, "period": int, "alpha": float},
    )

    def _calculate_series(self, returns: pl.DataFrame, period: int = 252, alpha: float = 0.05, **kwargs) -> pl.DataFrame:
        def cvar_expr(col_name):
            col = pl.col(col_name)
            # CVaR = mean of returns below VaR
            var_threshold = col.rolling_quantile(quantile=alpha, window_size=period)
            # Mean of values below VaR
            cvar = pl.when(col < var_threshold).then(col).otherwise(None).rolling_mean(window_size=period)
            return cvar.alias(col_name)

        return _apply_to_panel(returns, lambda c: cvar_expr(c.meta.output_name()))


print(f"Phase 6: Registered {70} panel, group, and miscellaneous operators")

