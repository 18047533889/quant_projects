# -*- coding: utf-8 -*-
"""
Polars native 技术指标算子 - Final Batch

实现 10 个高级技术指标：
ElderRay, FisherTransform, PSAR, QQE, RSX, SupertrendDirection, TSI, TSI_signal, VortexMinus, VortexPlus
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
    """对 panel 所有数值列应用 Polars 表达式函数"""
    cols = [c for c in df.columns if c not in PANEL_SKIP_COLUMNS]
    if not cols:
        return df
    return df.with_columns([expr_fn(pl.col(c)).alias(c) for c in cols])


def _ema_expr(col: pl.Expr, span: int) -> pl.Expr:
    """EMA 表达式（alpha = 2/(span+1)）"""
    return col.ewm_mean(span=span, adjust=False, ignore_nulls=True)


def _wilder_ema_expr(col: pl.Expr, period: int) -> pl.Expr:
    """Wilder's smoothing (alpha = 1/period)"""
    alpha = 1.0 / period if period > 0 else 0.0
    return col.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)


# ==================== ElderRay ====================
@register_operator(
    name="ElderRay",
    category="technical_indicator",
    canonical="ElderRay",
    source="polars_native_technical_final",
    backend="polars",
)
class ElderRay(SeriesOperator):
    """Elder Ray Index (Bull Power + Bear Power combined indicator)"""

    metadata = OperatorMetadata(
        name="ElderRay",
        category="technical_indicator",
        description="Elder Ray Index: measures buying and selling pressure",
        param_names=["high", "low", "close", "period"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame, "period": int},
    )

    def _calculate_series(
        self, high: pl.DataFrame, low: pl.DataFrame, close: pl.DataFrame, period: int = 13, **kwargs
    ) -> pl.DataFrame:
        """Bull Power - Bear Power"""
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        result_data = {}
        for col in cols:
            h_val = high[col]
            l_val = low[col]
            c_val = close[col]
            ema_val = _ema_expr(c_val, period)
            # Bull Power - Bear Power
            bull_power = h_val - ema_val
            bear_power = l_val - ema_val
            elder_ray = bull_power - bear_power
            result_data[col] = elder_ray

        result_df = pl.DataFrame(result_data)
        for meta_col in PANEL_SKIP_COLUMNS:
            if meta_col in high.columns:
                result_df = result_df.with_columns([high[meta_col]])
        return result_df


# ==================== FisherTransform ====================
@register_operator(
    name="FisherTransform",
    category="technical_indicator",
    canonical="FisherTransform",
    source="polars_native_technical_final",
    backend="polars",
)
class FisherTransform(SeriesOperator):
    """Fisher Transform: converts prices to Gaussian normal distribution"""

    metadata = OperatorMetadata(
        name="FisherTransform",
        category="technical_indicator",
        description="Fisher Transform for turning price into Gaussian distribution",
        param_names=["close", "period"],
        param_types={"close": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, close: pl.DataFrame, period: int = 10, **kwargs) -> pl.DataFrame:
        def fisher_expr(col_name):
            c = pl.col(col_name)
            # Normalize to [-1, 1]
            low_n = c.rolling_min(period)
            high_n = c.rolling_max(period)
            range_n = high_n - low_n
            # Avoid division by zero
            normalized = pl.when(range_n > 0).then(
                (2 * (c - low_n) / range_n - 1).clip(-0.999, 0.999)
            ).otherwise(0.0)
            # Fisher transform: 0.5 * ln((1+x)/(1-x))
            fisher = 0.5 * ((1 + normalized) / (1 - normalized)).log()
            # Apply EMA smoothing
            fisher_smooth = _ema_expr(fisher, 3)
            return fisher_smooth.alias(col_name)

        return _apply_to_panel(close, lambda c: fisher_expr(c.meta.output_name()))


# ==================== PSAR (Parabolic SAR) ====================
@register_operator(
    name="PSAR",
    category="technical_indicator",
    canonical="PSAR",
    source="polars_native_technical_final",
    backend="polars",
)
class PSAR(SeriesOperator):
    """Parabolic SAR: Stop and Reverse indicator"""

    metadata = OperatorMetadata(
        name="PSAR",
        category="technical_indicator",
        description="Parabolic SAR (Stop and Reverse)",
        param_names=["high", "low", "close", "af_start", "af_increment", "af_max"],
        param_types={
            "high": pl.DataFrame,
            "low": pl.DataFrame,
            "close": pl.DataFrame,
            "af_start": float,
            "af_increment": float,
            "af_max": float,
        },
    )

    def _calculate_series(
        self,
        high: pl.DataFrame,
        low: pl.DataFrame,
        close: pl.DataFrame,
        af_start: float = 0.02,
        af_increment: float = 0.02,
        af_max: float = 0.20,
        **kwargs,
    ) -> pl.DataFrame:
        """
        Parabolic SAR calculation - stateful algorithm requiring loop
        TODO: Full implementation requires tracking trend direction, EP, AF per bar
        """
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        result_data = {}
        for col in cols:
            h = high[col].to_numpy()
            l = low[col].to_numpy()
            c = close[col].to_numpy()
            n = len(h)

            sar = np.full(n, np.nan)
            if n < 2:
                result_data[col] = sar
                continue

            # Initialize
            is_uptrend = c[1] > c[0]
            sar[0] = l[0] if is_uptrend else h[0]
            ep = h[0] if is_uptrend else l[0]
            af = af_start

            for i in range(1, n):
                # Calculate SAR
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

                # Check for reversal
                if is_uptrend:
                    if l[i] < sar[i]:
                        # Reverse to downtrend
                        is_uptrend = False
                        sar[i] = ep
                        ep = l[i]
                        af = af_start
                    else:
                        # Continue uptrend
                        if h[i] > ep:
                            ep = h[i]
                            af = min(af + af_increment, af_max)
                        sar[i] = min(sar[i], l[i - 1], l[i])
                else:
                    if h[i] > sar[i]:
                        # Reverse to uptrend
                        is_uptrend = True
                        sar[i] = ep
                        ep = h[i]
                        af = af_start
                    else:
                        # Continue downtrend
                        if l[i] < ep:
                            ep = l[i]
                            af = min(af + af_increment, af_max)
                        sar[i] = max(sar[i], h[i - 1], h[i])

            result_data[col] = sar

        result_df = pl.DataFrame(result_data)
        for meta_col in PANEL_SKIP_COLUMNS:
            if meta_col in high.columns:
                result_df = result_df.with_columns([high[meta_col]])
        return result_df


# ==================== QQE ====================
@register_operator(
    name="QQE",
    category="technical_indicator",
    canonical="QQE",
    source="polars_native_technical_final",
    backend="polars",
)
class QQE(SeriesOperator):
    """Quantitative Qualitative Estimation: smoothed RSI with ATR bands"""

    metadata = OperatorMetadata(
        name="QQE",
        category="technical_indicator",
        description="QQE indicator based on smoothed RSI",
        param_names=["close", "rsi_period", "smoothing"],
        param_types={"close": pl.DataFrame, "rsi_period": int, "smoothing": int},
    )

    def _calculate_series(
        self, close: pl.DataFrame, rsi_period: int = 14, smoothing: int = 5, **kwargs
    ) -> pl.DataFrame:
        def qqe_expr(col_name):
            c = pl.col(col_name)
            # Calculate RSI
            delta = c.diff()
            gain = delta.clip(lower_bound=0)
            loss = (-delta).clip(lower_bound=0)
            avg_gain = _wilder_ema_expr(gain, rsi_period)
            avg_loss = _wilder_ema_expr(loss, rsi_period)
            rs = avg_gain / avg_loss
            rsi = 100 - 100 / (1 + rs)
            # Smooth RSI
            rsi_smooth = _ema_expr(rsi, smoothing)
            # Calculate ATR of RSI
            rsi_delta = rsi_smooth.diff().abs()
            atr_rsi = _wilder_ema_expr(rsi_delta, 2 * rsi_period - 1)
            # QQE line
            qqe = _ema_expr(rsi_smooth, smoothing)
            return qqe.alias(col_name)

        return _apply_to_panel(close, lambda c: qqe_expr(c.meta.output_name()))


# ==================== RSX ====================
@register_operator(
    name="RSX",
    category="technical_indicator",
    canonical="RSX",
    source="polars_native_technical_final",
    backend="polars",
)
class RSX(SeriesOperator):
    """Relative Strength Xtra: noise-free RSI using Jurik-style smoothing"""

    metadata = OperatorMetadata(
        name="RSX",
        category="technical_indicator",
        description="RSX - smoother version of RSI with reduced noise",
        param_names=["close", "period"],
        param_types={"close": pl.DataFrame, "period": int},
    )

    def _calculate_series(self, close: pl.DataFrame, period: int = 14, **kwargs) -> pl.DataFrame:
        """
        RSX approximation using triple exponential smoothing
        Note: Full Jurik smoothing is proprietary; this is a reasonable approximation
        """

        def rsx_expr(col_name):
            c = pl.col(col_name)
            # Calculate momentum
            delta = c.diff()
            gain = delta.clip(lower_bound=0)
            loss = (-delta).clip(lower_bound=0)

            # Triple smoothing for both gain and loss
            alpha = 2.0 / (period + 1)
            gain_smooth1 = gain.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            gain_smooth2 = gain_smooth1.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            gain_smooth3 = gain_smooth2.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)

            loss_smooth1 = loss.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            loss_smooth2 = loss_smooth1.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            loss_smooth3 = loss_smooth2.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)

            # Calculate RSX
            rs = gain_smooth3 / loss_smooth3
            rsx = 100 - 100 / (1 + rs)
            return rsx.alias(col_name)

        return _apply_to_panel(close, lambda c: rsx_expr(c.meta.output_name()))


# ==================== SupertrendDirection ====================
@register_operator(
    name="SupertrendDirection",
    category="technical_indicator",
    canonical="SupertrendDirection",
    source="polars_native_technical_final",
    backend="polars",
)
class SupertrendDirection(SeriesOperator):
    """Supertrend Direction: +1 for uptrend, -1 for downtrend"""

    metadata = OperatorMetadata(
        name="SupertrendDirection",
        category="technical_indicator",
        description="Supertrend direction indicator based on ATR bands",
        param_names=["high", "low", "close", "period", "multiplier"],
        param_types={
            "high": pl.DataFrame,
            "low": pl.DataFrame,
            "close": pl.DataFrame,
            "period": int,
            "multiplier": float,
        },
    )

    def _calculate_series(
        self,
        high: pl.DataFrame,
        low: pl.DataFrame,
        close: pl.DataFrame,
        period: int = 10,
        multiplier: float = 3.0,
        **kwargs,
    ) -> pl.DataFrame:
        """
        Supertrend direction calculation
        TODO: Full implementation requires stateful tracking of trend changes
        """
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        result_data = {}
        for col in cols:
            h = high[col].to_numpy()
            l = low[col].to_numpy()
            c = close[col].to_numpy()
            n = len(h)

            # Calculate ATR
            hl = h - l
            hc = np.abs(h - np.roll(c, 1))
            lc = np.abs(l - np.roll(c, 1))
            tr = np.maximum(np.maximum(hl, hc), lc)
            tr[0] = hl[0]  # First bar

            # Wilder's ATR
            alpha = 1.0 / period
            atr = np.full(n, np.nan)
            atr[0] = tr[0]
            for i in range(1, n):
                atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

            # Basic bands
            hl_avg = (h + l) / 2
            upper_band = hl_avg + multiplier * atr
            lower_band = hl_avg - multiplier * atr

            # Determine direction (simplified: not fully stateful)
            direction = np.where(c > upper_band, 1, np.where(c < lower_band, -1, 0))

            # Forward fill direction
            for i in range(1, n):
                if direction[i] == 0:
                    direction[i] = direction[i - 1]

            result_data[col] = direction

        result_df = pl.DataFrame(result_data)
        for meta_col in PANEL_SKIP_COLUMNS:
            if meta_col in high.columns:
                result_df = result_df.with_columns([high[meta_col]])
        return result_df


# ==================== TSI ====================
@register_operator(
    name="TSI",
    category="technical_indicator",
    canonical="TSI",
    source="polars_native_technical_final",
    backend="polars",
)
class TSI(SeriesOperator):
    """True Strength Index: double-smoothed momentum oscillator"""

    metadata = OperatorMetadata(
        name="TSI",
        category="technical_indicator",
        description="True Strength Index - double smoothed momentum",
        param_names=["close", "long", "short"],
        param_types={"close": pl.DataFrame, "long": int, "short": int},
    )

    def _calculate_series(self, close: pl.DataFrame, long: int = 25, short: int = 13, **kwargs) -> pl.DataFrame:
        def tsi_expr(col_name):
            c = pl.col(col_name)
            # Price momentum
            momentum = c.diff()
            abs_momentum = momentum.abs()

            # Double smooth both momentum and absolute momentum
            momentum_smooth1 = _ema_expr(momentum, long)
            momentum_smooth2 = _ema_expr(momentum_smooth1, short)

            abs_momentum_smooth1 = _ema_expr(abs_momentum, long)
            abs_momentum_smooth2 = _ema_expr(abs_momentum_smooth1, short)

            # TSI = 100 * (double smoothed momentum / double smoothed absolute momentum)
            tsi = 100 * momentum_smooth2 / abs_momentum_smooth2
            return tsi.alias(col_name)

        return _apply_to_panel(close, lambda c: tsi_expr(c.meta.output_name()))


# ==================== TSI_signal ====================
@register_operator(
    name="TSI_signal",
    category="technical_indicator",
    canonical="TSI_signal",
    source="polars_native_technical_final",
    backend="polars",
)
class TSI_signal(SeriesOperator):
    """TSI signal line: EMA of TSI"""

    metadata = OperatorMetadata(
        name="TSI_signal",
        category="technical_indicator",
        description="TSI signal line - EMA smoothing of TSI",
        param_names=["close", "long", "short", "signal"],
        param_types={"close": pl.DataFrame, "long": int, "short": int, "signal": int},
    )

    def _calculate_series(
        self, close: pl.DataFrame, long: int = 25, short: int = 13, signal: int = 7, **kwargs
    ) -> pl.DataFrame:
        def tsi_signal_expr(col_name):
            c = pl.col(col_name)
            # Calculate TSI first
            momentum = c.diff()
            abs_momentum = momentum.abs()

            momentum_smooth1 = _ema_expr(momentum, long)
            momentum_smooth2 = _ema_expr(momentum_smooth1, short)

            abs_momentum_smooth1 = _ema_expr(abs_momentum, long)
            abs_momentum_smooth2 = _ema_expr(abs_momentum_smooth1, short)

            tsi = 100 * momentum_smooth2 / abs_momentum_smooth2

            # Apply signal smoothing
            tsi_signal = _ema_expr(tsi, signal)
            return tsi_signal.alias(col_name)

        return _apply_to_panel(close, lambda c: tsi_signal_expr(c.meta.output_name()))


# ==================== VortexMinus ====================
@register_operator(
    name="VortexMinus",
    category="technical_indicator",
    canonical="VortexMinus",
    source="polars_native_technical_final",
    backend="polars",
)
class VortexMinus(SeriesOperator):
    """Vortex Indicator Minus (VI-)"""

    metadata = OperatorMetadata(
        name="VortexMinus",
        category="technical_indicator",
        description="Vortex Indicator Minus component",
        param_names=["high", "low", "close", "period"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame, "period": int},
    )

    def _calculate_series(
        self, high: pl.DataFrame, low: pl.DataFrame, close: pl.DataFrame, period: int = 14, **kwargs
    ) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        result_data = {}
        for col in cols:
            h = high[col]
            l = low[col]
            c = close[col]

            # Vortex Movement Minus: |Low[i] - High[i-1]|
            vm_minus = (l - h.shift(1)).abs()

            # True Range
            hl = h - l
            hc = (h - c.shift(1)).abs()
            lc = (l - c.shift(1)).abs()
            tr = pl.max_horizontal(hl, hc, lc)

            # Sum over period
            vm_minus_sum = vm_minus.rolling_sum(period)
            tr_sum = tr.rolling_sum(period)

            # VI- = Sum(VM-) / Sum(TR)
            vi_minus = vm_minus_sum / tr_sum
            result_data[col] = vi_minus

        result_df = pl.DataFrame(result_data)
        for meta_col in PANEL_SKIP_COLUMNS:
            if meta_col in high.columns:
                result_df = result_df.with_columns([high[meta_col]])
        return result_df


# ==================== VortexPlus ====================
@register_operator(
    name="VortexPlus",
    category="technical_indicator",
    canonical="VortexPlus",
    source="polars_native_technical_final",
    backend="polars",
)
class VortexPlus(SeriesOperator):
    """Vortex Indicator Plus (VI+)"""

    metadata = OperatorMetadata(
        name="VortexPlus",
        category="technical_indicator",
        description="Vortex Indicator Plus component",
        param_names=["high", "low", "close", "period"],
        param_types={"high": pl.DataFrame, "low": pl.DataFrame, "close": pl.DataFrame, "period": int},
    )

    def _calculate_series(
        self, high: pl.DataFrame, low: pl.DataFrame, close: pl.DataFrame, period: int = 14, **kwargs
    ) -> pl.DataFrame:
        cols = [c for c in high.columns if c not in PANEL_SKIP_COLUMNS]
        if not cols:
            return high

        result_data = {}
        for col in cols:
            h = high[col]
            l = low[col]
            c = close[col]

            # Vortex Movement Plus: |High[i] - Low[i-1]|
            vm_plus = (h - l.shift(1)).abs()

            # True Range
            hl = h - l
            hc = (h - c.shift(1)).abs()
            lc = (l - c.shift(1)).abs()
            tr = pl.max_horizontal(hl, hc, lc)

            # Sum over period
            vm_plus_sum = vm_plus.rolling_sum(period)
            tr_sum = tr.rolling_sum(period)

            # VI+ = Sum(VM+) / Sum(TR)
            vi_plus = vm_plus_sum / tr_sum
            result_data[col] = vi_plus

        result_df = pl.DataFrame(result_data)
        for meta_col in PANEL_SKIP_COLUMNS:
            if meta_col in high.columns:
                result_df = result_df.with_columns([high[meta_col]])
        return result_df


# Summary: 10 technical indicators implemented
# ElderRay, FisherTransform, PSAR, QQE, RSX
# SupertrendDirection, TSI, TSI_signal, VortexMinus, VortexPlus
