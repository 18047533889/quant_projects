# -*- coding: utf-8 -*-
"""Technical indicators - Polars native implementations (Phase 2, Module 7).

All operators use TRUE Polars expressions only - no pandas fallback, no NumPy.
Follows polars_daily_native.py pattern with _numeric_cols, _with_meta, strict validation.
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

from factor_engine.cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.base import ParamRole, ParamSpec

_SKIP = frozenset({"date", "stock_code"})
_SRC = "factor_dsl_polars_native"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _with_meta(result: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    if "date" in source.columns and "date" not in result.columns:
        result = result.with_columns(source["date"])
    return result


# ---------------------------------------------------------------------------
# Weighted moving averages
# ---------------------------------------------------------------------------


@register_operator(
    name="WMA",
    category="time_series",
    business_category="technical_indicators",
    canonical="WMA",
    source=_SRC,
    backend="polars")
class WMANative(SeriesOperator):
    """Weighted Moving Average: linear decreasing weights."""

    metadata = OperatorMetadata(
        name="WMA",
        category="time_series",
        description="加权移动平均",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(x)

        # WMA = sum(x_i * (n-i+1)) / sum(n-i+1) where i=0..n-1
        # weight_sum = n*(n+1)/2
        weight_sum = (w * (w + 1)) / 2.0

        exprs = []
        for c in cols:
            # Compute weighted sum using rolling window
            weighted_sum = pl.lit(0.0)
            for i in range(w):
                weight = w - i
                weighted_sum = weighted_sum + pl.col(c).shift(i) * weight
            exprs.append((weighted_sum / weight_sum).alias(c))

        return x.with_columns(exprs)


@register_operator(
    name="DEMA",
    category="time_series",
    business_category="technical_indicators",
    canonical="DEMA",
    source=_SRC,
    backend="polars")
class DEMANative(SeriesOperator):
    """Double Exponential Moving Average: 2*EMA - EMA(EMA)."""

    metadata = OperatorMetadata(
        name="DEMA",
        category="time_series",
        description="双指数移动平均",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(x)
        alpha = 2.0 / (w + 1)

        exprs = []
        for c in cols:
            ema1 = pl.col(c).ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            ema2 = ema1.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            exprs.append((2 * ema1 - ema2).alias(c))

        return x.with_columns(exprs)


@register_operator(
    name="TEMA",
    category="time_series",
    business_category="technical_indicators",
    canonical="TEMA",
    source=_SRC,
    backend="polars")
class TEMANative(SeriesOperator):
    """Triple Exponential Moving Average: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))."""

    metadata = OperatorMetadata(
        name="TEMA",
        category="time_series",
        description="三重指数移动平均",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(x)
        alpha = 2.0 / (w + 1)

        exprs = []
        for c in cols:
            ema1 = pl.col(c).ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            ema2 = ema1.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            ema3 = ema2.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            exprs.append((3 * ema1 - 3 * ema2 + ema3).alias(c))

        return x.with_columns(exprs)


@register_operator(
    name="HMA",
    category="time_series",
    business_category="technical_indicators",
    canonical="HMA",
    source=_SRC,
    backend="polars")
class HMANative(SeriesOperator):
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""

    metadata = OperatorMetadata(
        name="HMA",
        category="time_series",
        description="Hull移动平均",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)
        half_w = max(1, w // 2)
        sqrt_w = max(1, int(w ** 0.5))

        def _wma(col_expr: pl.Expr, period: int) -> pl.Expr:
            weight_sum = (period * (period + 1)) / 2.0
            weighted = pl.lit(0.0)
            for i in range(period):
                weighted = weighted + col_expr.shift(i) * (period - i)
            return weighted / weight_sum

        exprs = []
        for c in cols:
            wma_half = _wma(pl.col(c), half_w)
            wma_full = _wma(pl.col(c), w)
            raw_hma = 2 * wma_half - wma_full
            # Apply WMA to the result
            hma = _wma(raw_hma, sqrt_w)
            exprs.append(hma.alias(c))

        return x.with_columns(exprs)


# ---------------------------------------------------------------------------
# Momentum oscillators
# ---------------------------------------------------------------------------


@register_operator(
    name="RSI_WILDER",
    category="time_series",
    business_category="technical_indicators",
    canonical="RSI_WILDER",
    source=_SRC,
    backend="polars")
class RSIWilderNative(SeriesOperator):
    """RSI using Wilder's smoothing: 100 - 100/(1 + RS), RS = EMA(gain)/EMA(loss)."""

    metadata = OperatorMetadata(
        name="RSI_WILDER",
        category="time_series",
        description="Wilder RSI",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 14, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(x)
        alpha = 1.0 / w  # Wilder's alpha

        exprs = []
        for c in cols:
            delta = pl.col(c) - pl.col(c).shift(1)
            gain = pl.when(delta > 0).then(delta).otherwise(0.0)
            loss = pl.when(delta < 0).then(-delta).otherwise(0.0)

            avg_gain = gain.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=False)
            avg_loss = loss.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=False)

            rs = pl.when(avg_loss != 0).then(avg_gain / avg_loss).otherwise(None)
            rsi = pl.when(avg_loss == 0).then(100.0).otherwise(100.0 - 100.0 / (1.0 + rs))
            exprs.append(rsi.alias(c))

        return x.with_columns(exprs)


@register_operator(
    name="CMO",
    category="time_series",
    business_category="technical_indicators",
    canonical="CMO",
    source=_SRC,
    backend="polars")
class CMONative(SeriesOperator):
    """Chande Momentum Oscillator: 100 * (sum_up - sum_down) / (sum_up + sum_down)."""

    metadata = OperatorMetadata(
        name="CMO",
        category="time_series",
        description="Chande动量震荡",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 14, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(x)

        exprs = []
        for c in cols:
            delta = pl.col(c) - pl.col(c).shift(1)
            gain = pl.when(delta > 0).then(delta).otherwise(0.0)
            loss = pl.when(delta < 0).then(-delta).otherwise(0.0)

            sum_gain = gain.rolling_sum(window_size=w, min_samples=1)
            sum_loss = loss.rolling_sum(window_size=w, min_samples=1)
            total = sum_gain + sum_loss

            cmo = pl.when(total == 0).then(None).otherwise(100.0 * (sum_gain - sum_loss) / total)
            exprs.append(cmo.alias(c))

        return x.with_columns(exprs)


@register_operator(
    name="MACD_line",
    category="time_series",
    business_category="technical_indicators",
    canonical="MACD_line",
    source=_SRC,
    backend="polars")
class MACDLineNative(SeriesOperator):
    """MACD line: EMA(fast) - EMA(slow)."""

    metadata = OperatorMetadata(
        name="MACD_line",
        category="time_series",
        description="MACD主线",
        param_names=["x", "fast_period", "slow_period"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, x: pl.DataFrame, fast_period: int = 12, slow_period: int = 26, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        fast = strict_integer(fast_period, "fast_period", minimum=1)
        slow = strict_integer(slow_period, "slow_period", minimum=1)
        cols = _numeric_cols(x)

        alpha_fast = 2.0 / (fast + 1)
        alpha_slow = 2.0 / (slow + 1)

        exprs = []
        for c in cols:
            # R21-DEF2: ewm_mean with ignore_nulls=True publishes from first
            # observation (one row early vs pandas min_periods=window). Fix:
            # count non-null observations and mask results where count < window.
            ema_fast_raw = pl.col(c).ewm_mean(alpha=alpha_fast, adjust=False, ignore_nulls=True)
            ema_fast_count = pl.col(c).is_not_null().cast(pl.Int64).cum_sum()
            ema_fast = pl.when(ema_fast_count >= fast).then(ema_fast_raw).otherwise(None)
            ema_slow_raw = pl.col(c).ewm_mean(alpha=alpha_slow, adjust=False, ignore_nulls=True)
            ema_slow_count = pl.col(c).is_not_null().cast(pl.Int64).cum_sum()
            ema_slow = pl.when(ema_slow_count >= slow).then(ema_slow_raw).otherwise(None)
            exprs.append((ema_fast - ema_slow).alias(c))

        return x.with_columns(exprs)


@register_operator(
    name="MACD_signal",
    category="time_series",
    business_category="technical_indicators",
    canonical="MACD_signal",
    source=_SRC,
    backend="polars")
class MACDSignalNative(SeriesOperator):
    """MACD signal: EMA of MACD line."""

    metadata = OperatorMetadata(
        name="MACD_signal",
        category="time_series",
        description="MACD信号线",
        param_names=["x", "fast_period", "slow_period", "signal_period"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, x: pl.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        fast = strict_integer(fast_period, "fast_period", minimum=1)
        slow = strict_integer(slow_period, "slow_period", minimum=1)
        signal = strict_integer(signal_period, "signal_period", minimum=1)
        cols = _numeric_cols(x)

        alpha_fast = 2.0 / (fast + 1)
        alpha_slow = 2.0 / (slow + 1)
        alpha_signal = 2.0 / (signal + 1)

        exprs = []
        for c in cols:
            # R21-DEF2: ewm_mean with ignore_nulls=True publishes from first
            # observation (one row early vs pandas min_periods=window). Fix:
            # count non-null observations and mask results where count < window.
            ema_fast_raw = pl.col(c).ewm_mean(alpha=alpha_fast, adjust=False, ignore_nulls=True)
            ema_fast_count = pl.col(c).is_not_null().cast(pl.Int64).cum_sum()
            ema_fast = pl.when(ema_fast_count >= fast).then(ema_fast_raw).otherwise(None)
            ema_slow_raw = pl.col(c).ewm_mean(alpha=alpha_slow, adjust=False, ignore_nulls=True)
            ema_slow_count = pl.col(c).is_not_null().cast(pl.Int64).cum_sum()
            ema_slow = pl.when(ema_slow_count >= slow).then(ema_slow_raw).otherwise(None)
            macd_line = ema_fast - ema_slow
            signal_line_raw = macd_line.ewm_mean(alpha=alpha_signal, adjust=False, ignore_nulls=True)
            signal_line_count = macd_line.is_not_null().cast(pl.Int64).cum_sum()
            signal_line = pl.when(signal_line_count >= signal).then(signal_line_raw).otherwise(None)
            exprs.append(signal_line.alias(c))

        return x.with_columns(exprs)


@register_operator(
    name="MACD_hist",
    category="time_series",
    business_category="technical_indicators",
    canonical="MACD_hist",
    source=_SRC,
    backend="polars")
class MACDHistNative(SeriesOperator):
    """MACD histogram: MACD line - signal."""

    metadata = OperatorMetadata(
        name="MACD_hist",
        category="time_series",
        description="MACD柱状图",
        param_names=["x", "fast_period", "slow_period", "signal_period"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, x: pl.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        fast = strict_integer(fast_period, "fast_period", minimum=1)
        slow = strict_integer(slow_period, "slow_period", minimum=1)
        signal = strict_integer(signal_period, "signal_period", minimum=1)
        cols = _numeric_cols(x)

        alpha_fast = 2.0 / (fast + 1)
        alpha_slow = 2.0 / (slow + 1)
        alpha_signal = 2.0 / (signal + 1)

        exprs = []
        for c in cols:
            ema_fast = pl.col(c).ewm_mean(alpha=alpha_fast, adjust=False, ignore_nulls=True)
            ema_slow = pl.col(c).ewm_mean(alpha=alpha_slow, adjust=False, ignore_nulls=True)
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm_mean(alpha=alpha_signal, adjust=False, ignore_nulls=True)
            exprs.append((macd_line - signal_line).alias(c))

        return x.with_columns(exprs)


# ---------------------------------------------------------------------------
# Volatility & Trend Strength
# ---------------------------------------------------------------------------


@register_operator(
    name="ATR_WILDER",
    category="time_series",
    business_category="technical_indicators",
    canonical="ATR_WILDER",
    source=_SRC,
    backend="polars")
class ATRWilderNative(SeriesOperator):
    """Average True Range using Wilder's smoothing."""

    metadata = OperatorMetadata(
        name="ATR_WILDER",
        category="time_series",
        description="Wilder平均真实波幅",
        param_names=["high", "low", "close", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame | None = None,
                         close: pl.DataFrame | None = None, window: int = 14, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(high)
        alpha = 1.0 / w

        if low is None or close is None:
            raise ValueError("ATR_WILDER requires high, low, close")

        exprs = []
        for c in cols:
            h = pl.col(c)
            l_col = low[c] if c in low.columns else pl.lit(None)
            c_prev = close[c].shift(1) if c in close.columns else pl.lit(None)

            tr1 = h - l_col
            tr2 = (h - c_prev).abs()
            tr3 = (l_col - c_prev).abs()
            true_range = pl.max_horizontal(tr1, tr2, tr3)

            atr = true_range.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=False)
            exprs.append(atr.alias(c))

        return high.with_columns(exprs)


@register_operator(
    name="ADX",
    category="time_series",
    business_category="technical_indicators",
    canonical="ADX",
    source=_SRC,
    backend="polars")
class ADXNative(SeriesOperator):
    """Average Directional Index."""

    metadata = OperatorMetadata(
        name="ADX",
        category="time_series",
        description="平均趋向指数",
        param_names=["high", "low", "close", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame | None = None,
                         close: pl.DataFrame | None = None, window: int = 14, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(high)
        alpha = 1.0 / w

        if low is None or close is None:
            raise ValueError("ADX requires high, low, close")

        exprs = []
        for c in cols:
            h = pl.col(c)
            l_col = low[c] if c in low.columns else pl.lit(None)
            c_col = close[c] if c in close.columns else pl.lit(None)
            c_prev = c_col.shift(1)

            # True Range
            tr1 = h - l_col
            tr2 = (h - c_prev).abs()
            tr3 = (l_col - c_prev).abs()
            tr = pl.max_horizontal(tr1, tr2, tr3)

            # Directional Movement
            up_move = h - h.shift(1)
            down_move = l_col.shift(1) - l_col

            plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0.0)
            minus_dm = pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0.0)

            # Smoothed values
            atr = tr.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=False)
            plus_di = pl.when(atr != 0).then(100.0 * plus_dm.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=False) / atr).otherwise(None)
            minus_di = pl.when(atr != 0).then(100.0 * minus_dm.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=False) / atr).otherwise(None)

            # DX and ADX
            dx = pl.when((plus_di + minus_di) != 0).then(100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)).otherwise(None)
            adx = dx.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=False)

            exprs.append(adx.alias(c))

        return high.with_columns(exprs)


# ---------------------------------------------------------------------------
# Bollinger & Keltner Bands
# ---------------------------------------------------------------------------


@register_operator(
    name="bollinger_pct_b",
    category="time_series",
    business_category="technical_indicators",
    canonical="bollinger_pct_b",
    source=_SRC,
    backend="polars")
class BollingerPctBNative(SeriesOperator):
    """Bollinger %B: (price - lower_band) / (upper_band - lower_band)."""

    metadata = OperatorMetadata(
        name="bollinger_pct_b",
        category="time_series",
        description="布林带百分比B",
        param_names=["x", "window", "num_std"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, num_std: float = 2.0, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer, strict_finite_scalar

        w = strict_integer(window, "window", minimum=2)
        k = strict_finite_scalar(num_std, "num_std", minimum=0.0)
        cols = _numeric_cols(x)

        exprs = []
        for c in cols:
            ma = pl.col(c).rolling_mean(window_size=w, min_samples=1)
            std = pl.col(c).rolling_std(window_size=w, min_samples=2)
            upper = ma + k * std
            lower = ma - k * std
            bandwidth = upper - lower

            pct_b = pl.when(bandwidth != 0).then((pl.col(c) - lower) / bandwidth).otherwise(None)
            exprs.append(pct_b.alias(c))

        return x.with_columns(exprs)


@register_operator(
    name="bollinger_width",
    category="time_series",
    business_category="technical_indicators",
    canonical="bollinger_width",
    source=_SRC,
    backend="polars")
class BollingerWidthNative(SeriesOperator):
    """Bollinger Band Width: (upper - lower) / middle."""

    metadata = OperatorMetadata(
        name="bollinger_width",
        category="time_series",
        description="布林带宽度",
        param_names=["x", "window", "num_std"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, num_std: float = 2.0, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer, strict_finite_scalar

        w = strict_integer(window, "window", minimum=2)
        k = strict_finite_scalar(num_std, "num_std", minimum=0.0)
        cols = _numeric_cols(x)

        exprs = []
        for c in cols:
            ma = pl.col(c).rolling_mean(window_size=w, min_samples=1)
            std = pl.col(c).rolling_std(window_size=w, min_samples=2)
            width = pl.when(ma == 0).then(None).otherwise(2 * k * std / ma)
            exprs.append(width.alias(c))

        return x.with_columns(exprs)


@register_operator(
    name="KeltnerMid",
    category="time_series",
    business_category="technical_indicators",
    canonical="KeltnerMid",
    source=_SRC,
    backend="polars")
class KeltnerMidNative(SeriesOperator):
    """Keltner Channel middle line: EMA of typical price."""

    metadata = OperatorMetadata(
        name="KeltnerMid",
        category="time_series",
        description="Keltner通道中线",
        param_names=["high", "low", "close", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame | None = None,
                         close: pl.DataFrame | None = None, window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(high)
        alpha = 2.0 / (w + 1)

        if low is None or close is None:
            raise ValueError("KeltnerMid requires high, low, close")

        exprs = []
        for c in cols:
            h = pl.col(c)
            l_col = low[c] if c in low.columns else pl.lit(None)
            c_col = close[c] if c in close.columns else pl.lit(None)

            typical = (h + l_col + c_col) / 3.0
            mid = typical.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            exprs.append(mid.alias(c))

        return high.with_columns(exprs)


# ---------------------------------------------------------------------------
# Donchian & Ichimoku
# ---------------------------------------------------------------------------


@register_operator(
    name="donchian_lower",
    category="time_series",
    business_category="technical_indicators",
    canonical="donchian_lower",
    source=_SRC,
    backend="polars")
class DonchianLowerNative(SeriesOperator):
    """Donchian Channel lower: rolling min of low."""

    metadata = OperatorMetadata(
        name="donchian_lower",
        category="time_series",
        description="唐奇安通道下轨",
        param_names=["low", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, low: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(low)

        return low.with_columns([
            pl.col(c).rolling_min(window_size=w, min_samples=1).alias(c)
            for c in cols
        ])


@register_operator(
    name="donchian_mid",
    category="time_series",
    business_category="technical_indicators",
    canonical="donchian_mid",
    source=_SRC,
    backend="polars")
class DonchianMidNative(SeriesOperator):
    """Donchian Channel mid: (rolling_max(high) + rolling_min(low)) / 2."""

    metadata = OperatorMetadata(
        name="donchian_mid",
        category="time_series",
        description="唐奇安通道中线",
        param_names=["high", "low", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame | None = None,
                         window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(high)

        if low is None:
            raise ValueError("donchian_mid requires high and low")

        exprs = []
        for c in cols:
            h_max = pl.col(c).rolling_max(window_size=w, min_samples=1)
            l_min = low[c].rolling_min(window_size=w, min_samples=1) if c in low.columns else pl.lit(None)
            exprs.append(((h_max + l_min) / 2.0).alias(c))

        return high.with_columns(exprs)


@register_operator(
    name="donchian_upper",
    category="time_series",
    business_category="technical_indicators",
    canonical="donchian_upper",
    source=_SRC,
    backend="polars")
class DonchianUpperNative(SeriesOperator):
    """Donchian Channel upper: rolling max of high."""

    metadata = OperatorMetadata(
        name="donchian_upper",
        category="time_series",
        description="唐奇安通道上轨",
        param_names=["high", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, high: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(high)

        return high.with_columns([
            pl.col(c).rolling_max(window_size=w, min_samples=1).alias(c)
            for c in cols
        ])


@register_operator(
    name="ichimoku_tenkan",
    category="time_series",
    business_category="technical_indicators",
    canonical="ichimoku_tenkan",
    source=_SRC,
    backend="polars")
class IchimokuTenkanNative(SeriesOperator):
    """Ichimoku Tenkan-sen (conversion line): (9-period high + 9-period low) / 2."""

    metadata = OperatorMetadata(
        name="ichimoku_tenkan",
        category="time_series",
        description="一目均衡表转折线",
        param_names=["high", "low", "tenkan_window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame | None = None,
                         tenkan_window: int = 9, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(tenkan_window, "tenkan_window", minimum=1)
        cols = _numeric_cols(high)

        if low is None:
            raise ValueError("ichimoku_tenkan requires high and low")

        exprs = []
        for c in cols:
            h_max = pl.col(c).rolling_max(window_size=w, min_samples=1)
            l_min = low[c].rolling_min(window_size=w, min_samples=1) if c in low.columns else pl.lit(None)
            exprs.append(((h_max + l_min) / 2.0).alias(c))

        return high.with_columns(exprs)


@register_operator(
    name="ichimoku_kijun",
    category="time_series",
    business_category="technical_indicators",
    canonical="ichimoku_kijun",
    source=_SRC,
    backend="polars")
class IchimokuKijunNative(SeriesOperator):
    """Ichimoku Kijun-sen (base line): (26-period high + 26-period low) / 2."""

    metadata = OperatorMetadata(
        name="ichimoku_kijun",
        category="time_series",
        description="一目均衡表基准线",
        param_names=["high", "low", "kijun_window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame | None = None,
                         kijun_window: int = 26, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(kijun_window, "kijun_window", minimum=1)
        cols = _numeric_cols(high)

        if low is None:
            raise ValueError("ichimoku_kijun requires high and low")

        exprs = []
        for c in cols:
            h_max = pl.col(c).rolling_max(window_size=w, min_samples=1)
            l_min = low[c].rolling_min(window_size=w, min_samples=1) if c in low.columns else pl.lit(None)
            exprs.append(((h_max + l_min) / 2.0).alias(c))

        return high.with_columns(exprs)


# ---------------------------------------------------------------------------
# Volume-based indicators
# ---------------------------------------------------------------------------


@register_operator(
    name="MFI",
    category="time_series",
    business_category="technical_indicators",
    canonical="MFI",
    source=_SRC,
    backend="polars")
class MFINative(SeriesOperator):
    """Money Flow Index: volume-weighted RSI."""

    metadata = OperatorMetadata(
        name="MFI",
        category="time_series",
        description="资金流量指标",
        param_names=["high", "low", "close", "volume", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical", "volume"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame | None = None,
                         close: pl.DataFrame | None = None, volume: pl.DataFrame | None = None,
                         window: int = 14, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(high)

        if low is None or close is None or volume is None:
            raise ValueError("MFI requires high, low, close, volume")

        exprs = []
        for c in cols:
            h = pl.col(c)
            l_col = low[c] if c in low.columns else pl.lit(None)
            c_col = close[c] if c in close.columns else pl.lit(None)
            v_col = volume[c] if c in volume.columns else pl.lit(None)

            typical = (h + l_col + c_col) / 3.0
            raw_money_flow = typical * v_col

            flow_delta = typical - typical.shift(1)
            valid = raw_money_flow.is_not_null() & flow_delta.is_not_null()
            positive_flow = pl.when(valid).then(
                pl.when(flow_delta > 0).then(raw_money_flow).otherwise(0.0)
            ).otherwise(None)
            negative_flow = pl.when(valid).then(
                pl.when(flow_delta < 0).then(raw_money_flow).otherwise(0.0)
            ).otherwise(None)

            # Match the pandas authority: missing bars are excluded from both
            # cohorts, and a complete window is required before emitting MFI.
            pos_sum = positive_flow.rolling_sum(window_size=w, min_samples=w)
            neg_sum = negative_flow.rolling_sum(window_size=w, min_samples=w)

            money_ratio = pl.when(neg_sum != 0).then(pos_sum / neg_sum).otherwise(None)
            mfi = pl.when(neg_sum.is_null() | pos_sum.is_null()).then(None)
            mfi = mfi.when((neg_sum == 0) & (pos_sum > 0)).then(100.0)
            mfi = mfi.when((neg_sum == 0) & (pos_sum == 0)).then(50.0)
            mfi = mfi.when(neg_sum != 0).then(
                100.0 - 100.0 / (1.0 + money_ratio)
            ).otherwise(mfi)
            exprs.append(mfi.alias(c))

        return high.with_columns(exprs)


@register_operator(
    name="CMF",
    category="time_series",
    business_category="technical_indicators",
    canonical="CMF",
    source=_SRC,
    backend="polars")
class CMFNative(SeriesOperator):
    """Chaikin Money Flow."""

    metadata = OperatorMetadata(
        name="CMF",
        category="time_series",
        description="蔡金资金流",
        param_names=["high", "low", "close", "volume", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical", "volume"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame | None = None,
                         close: pl.DataFrame | None = None, volume: pl.DataFrame | None = None,
                         window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(high)

        if low is None or close is None or volume is None:
            raise ValueError("CMF requires high, low, close, volume")

        exprs = []
        for c in cols:
            h = pl.col(c)
            l_col = low[c] if c in low.columns else pl.lit(None)
            c_col = close[c] if c in close.columns else pl.lit(None)
            v_col = volume[c] if c in volume.columns else pl.lit(None)

            clv = pl.when((h - l_col) == 0).then(0.0).otherwise(
                ((c_col - l_col) - (h - c_col)) / (h - l_col)
            )
            money_flow_volume = clv * v_col

            vol_sum = v_col.rolling_sum(window_size=w, min_samples=1)
            cmf = pl.when(vol_sum != 0).then(money_flow_volume.rolling_sum(window_size=w, min_samples=1) / vol_sum).otherwise(None)
            exprs.append(cmf.alias(c))

        return high.with_columns(exprs)


@register_operator(
    name="ForceIndex",
    category="time_series",
    business_category="technical_indicators",
    canonical="ForceIndex",
    source=_SRC,
    backend="polars")
class ForceIndexNative(SeriesOperator):
    """Force Index: (close - close_prev) * volume, then EMA smoothed."""

    metadata = OperatorMetadata(
        name="ForceIndex",
        category="time_series",
        description="强力指标",
        param_names=["close", "volume", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical", "volume"],
    )

    def _calculate_series(self, close: pl.DataFrame, volume: pl.DataFrame | None = None,
                         window: int = 13, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(close)
        alpha = 2.0 / (w + 1)

        if volume is None:
            raise ValueError("ForceIndex requires close and volume")

        exprs = []
        for c in cols:
            c_col = pl.col(c)
            v_col = volume[c] if c in volume.columns else pl.lit(None)

            force = (c_col - c_col.shift(1)) * v_col
            force_ema = force.ewm_mean(alpha=alpha, adjust=False, ignore_nulls=True)
            exprs.append(force_ema.alias(c))

        return close.with_columns(exprs)


@register_operator(
    name="ChaikinOscillator",
    category="time_series",
    business_category="technical_indicators",
    canonical="ChaikinOscillator",
    source=_SRC,
    backend="polars")
class ChaikinOscillatorNative(SeriesOperator):
    """Chaikin Oscillator: EMA(ADL, fast) - EMA(ADL, slow)."""

    metadata = OperatorMetadata(
        name="ChaikinOscillator",
        category="time_series",
        description="蔡金震荡指标",
        param_names=["high", "low", "close", "volume", "fast_period", "slow_period"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical", "volume"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame | None = None,
                         close: pl.DataFrame | None = None, volume: pl.DataFrame | None = None,
                         fast_period: int = 3, slow_period: int = 10, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        fast = strict_integer(fast_period, "fast_period", minimum=1)
        slow = strict_integer(slow_period, "slow_period", minimum=1)
        cols = _numeric_cols(high)

        if low is None or close is None or volume is None:
            raise ValueError("ChaikinOscillator requires high, low, close, volume")

        alpha_fast = 2.0 / (fast + 1)
        alpha_slow = 2.0 / (slow + 1)

        exprs = []
        for c in cols:
            h = pl.col(c)
            l_col = low[c] if c in low.columns else pl.lit(None)
            c_col = close[c] if c in close.columns else pl.lit(None)
            v_col = volume[c] if c in volume.columns else pl.lit(None)

            # Accumulation/Distribution Line
            clv = pl.when((h - l_col) == 0).then(0.0).otherwise(
                ((c_col - l_col) - (h - c_col)) / (h - l_col)
            )
            adl = (clv * v_col).cum_sum()

            fast_ema = adl.ewm_mean(alpha=alpha_fast, adjust=False, ignore_nulls=True)
            slow_ema = adl.ewm_mean(alpha=alpha_slow, adjust=False, ignore_nulls=True)

            co = fast_ema - slow_ema
            exprs.append(co.alias(c))

        return high.with_columns(exprs)


@register_operator(
    name="rolling_vwap",
    category="time_series",
    business_category="technical_indicators",
    canonical="rolling_vwap",
    source=_SRC,
    backend="polars")
class RollingVWAPNative(SeriesOperator):
    """Rolling Volume-Weighted Average Price."""

    metadata = OperatorMetadata(
        name="rolling_vwap",
        category="time_series",
        description="滚动成交量加权均价",
        param_names=["price", "volume", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "volume"],
    )

    def _calculate_series(self, price: pl.DataFrame, volume: pl.DataFrame | None = None,
                         window: int = 20, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=1)
        cols = _numeric_cols(price)

        if volume is None:
            raise ValueError("rolling_vwap requires price and volume")

        exprs = []
        for c in cols:
            p = pl.col(c)
            v = volume[c] if c in volume.columns else pl.lit(None)

            pv_sum = (p * v).rolling_sum(window_size=w, min_samples=1)
            v_sum = v.rolling_sum(window_size=w, min_samples=1)

            vwap = pl.when(v_sum == 0).then(None).otherwise(pv_sum / v_sum)
            exprs.append(vwap.alias(c))

        return price.with_columns(exprs)


@register_operator(
    name="CoppockCurve",
    category="time_series",
    business_category="technical_indicators",
    canonical="CoppockCurve",
    source=_SRC,
    backend="polars")
class CoppockCurveNative(SeriesOperator):
    """Coppock Curve: WMA of (ROC14 + ROC11)."""

    metadata = OperatorMetadata(
        name="CoppockCurve",
        category="time_series",
        description="库克曲线",
        param_names=["x", "short_roc", "long_roc", "wma_period"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, x: pl.DataFrame, short_roc: int = 11, long_roc: int = 14,
                         wma_period: int = 10, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        s = strict_integer(short_roc, "short_roc", minimum=1)
        l = strict_integer(long_roc, "long_roc", minimum=1)
        w = strict_integer(wma_period, "wma_period", minimum=1)
        cols = _numeric_cols(x)

        exprs = []
        for c in cols:
            # ROC = (x / x_lag - 1) * 100
            shifted_s = pl.col(c).shift(s)
            shifted_l = pl.col(c).shift(l)
            roc_s = pl.when(shifted_s != 0).then((pl.col(c) / shifted_s - 1) * 100).otherwise(None)
            roc_l = pl.when(shifted_l != 0).then((pl.col(c) / shifted_l - 1) * 100).otherwise(None)
            roc_sum = roc_s + roc_l

            # WMA
            weight_sum = (w * (w + 1)) / 2.0
            weighted = pl.lit(0.0)
            for i in range(w):
                weighted = weighted + roc_sum.shift(i) * (w - i)
            curve = weighted / weight_sum

            exprs.append(curve.alias(c))

        return x.with_columns(exprs)


# ---------------------------------------------------------------------------
# Efficiency & Oscillators
# ---------------------------------------------------------------------------


@register_operator(
    name="efficiency_ratio",
    category="time_series",
    business_category="technical_indicators",
    canonical="efficiency_ratio",
    source=_SRC,
    backend="polars")
class EfficiencyRatioNative(SeriesOperator):
    """Kaufman Efficiency Ratio: net_change / sum(abs(change))."""

    metadata = OperatorMetadata(
        name="efficiency_ratio",
        category="time_series",
        description="效率比率",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 10, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(x)

        exprs = []
        for c in cols:
            net_change = (pl.col(c) - pl.col(c).shift(w - 1)).abs()
            delta = (pl.col(c) - pl.col(c).shift(1)).abs()
            volatility = delta.rolling_sum(window_size=w, min_samples=1)

            er = pl.when(volatility == 0).then(None).otherwise(net_change / volatility)
            exprs.append(er.alias(c))

        return x.with_columns(exprs)


@register_operator(
    name="choppiness_index",
    category="time_series",
    business_category="technical_indicators",
    canonical="choppiness_index",
    source=_SRC,
    backend="polars")
class ChoppinessIndexNative(SeriesOperator):
    """Choppiness Index: 100 * log10(sum(ATR) / (max - min)) / log10(n)."""

    metadata = OperatorMetadata(
        name="choppiness_index",
        category="time_series",
        description="震荡指标",
        param_names=["high", "low", "close", "window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame | None = None,
                         close: pl.DataFrame | None = None, window: int = 14, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer
        import math

        w = strict_integer(window, "window", minimum=2)
        cols = _numeric_cols(high)

        if low is None or close is None:
            raise ValueError("choppiness_index requires high, low, close")

        log10_n = math.log10(w)

        exprs = []
        for c in cols:
            h = pl.col(c)
            l_col = low[c] if c in low.columns else pl.lit(None)
            c_col = close[c] if c in close.columns else pl.lit(None)
            c_prev = c_col.shift(1)

            # True Range
            tr1 = h - l_col
            tr2 = (h - c_prev).abs()
            tr3 = (l_col - c_prev).abs()
            tr = pl.max_horizontal(tr1, tr2, tr3)

            # R21-DEF2: rolling operations with min_samples=1 publish from first
            # observation (one row early vs pandas min_periods=window). Fix:
            # use min_samples=window for rolling operations.
            sum_tr = tr.rolling_sum(window_size=w, min_samples=w)
            h_max = h.rolling_max(window_size=w, min_samples=w)
            l_min = l_col.rolling_min(window_size=w, min_samples=w)

            range_val = h_max - l_min
            ci = pl.when(range_val == 0).then(None).otherwise(
                100.0 * (sum_tr / range_val).log10() / log10_n
            )
            exprs.append(ci.alias(c))

        return high.with_columns(exprs)


@register_operator(
    name="UltimateOscillator",
    category="time_series",
    business_category="technical_indicators",
    canonical="UltimateOscillator",
    source=_SRC,
    backend="polars")
class UltimateOscillatorNative(SeriesOperator):
    """Ultimate Oscillator: weighted average of three time periods."""

    metadata = OperatorMetadata(
        name="UltimateOscillator",
        category="time_series",
        description="终极震荡指标",
        param_names=["high", "low", "close", "short_window", "medium_window", "long_window"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame | None = None,
                         close: pl.DataFrame | None = None, short_window: int = 7,
                         medium_window: int = 14, long_window: int = 28, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        s = strict_integer(short_window, "short_window", minimum=1)
        m = strict_integer(medium_window, "medium_window", minimum=1)
        l = strict_integer(long_window, "long_window", minimum=1)
        cols = _numeric_cols(high)

        if low is None or close is None:
            raise ValueError("UltimateOscillator requires high, low, close")

        exprs = []
        for c in cols:
            h = pl.col(c)
            l_col = low[c] if c in low.columns else pl.lit(None)
            c_col = close[c] if c in close.columns else pl.lit(None)
            c_prev = c_col.shift(1)

            bp = c_col - pl.min_horizontal(l_col, c_prev)
            tr = pl.max_horizontal(h, c_prev) - pl.min_horizontal(l_col, c_prev)

            # R21-DEF2: rolling_sum with min_samples=1 publishes from first
            # observation (one row early vs pandas min_periods=window). Fix:
            # use min_samples=window for rolling operations.
            bp_s = bp.rolling_sum(window_size=s, min_samples=s)
            tr_s = tr.rolling_sum(window_size=s, min_samples=s)
            avg_s = pl.when(tr_s != 0).then(bp_s / tr_s).otherwise(None)

            bp_m = bp.rolling_sum(window_size=m, min_samples=m)
            tr_m = tr.rolling_sum(window_size=m, min_samples=m)
            avg_m = pl.when(tr_m != 0).then(bp_m / tr_m).otherwise(None)

            bp_l = bp.rolling_sum(window_size=l, min_samples=l)
            tr_l = tr.rolling_sum(window_size=l, min_samples=l)
            avg_l = pl.when(tr_l != 0).then(bp_l / tr_l).otherwise(None)

            uo = 100.0 * (4 * avg_s + 2 * avg_m + avg_l) / 7.0
            exprs.append(uo.alias(c))

        return high.with_columns(exprs)


# ---------------------------------------------------------------------------
# Supertrend & PSAR
# ---------------------------------------------------------------------------

# R21-DEF1-SUPERTRND: honest ExecutionKind for the stateful Supertrend kernel.
# The band ratchet/flip recursion is a per-row NumPy kernel over Polars column
# arrays — it is NOT a pure native Polars expression and must never be
# classified as one (no per-row Python loop is ever claimed native).
try:  # pragma: no cover - import guard mirrors other backend-aware modules
    from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec

    _SupertrendSpec = PhysicalImplementationSpec(
        canonical="Supertrend",
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,
        supports_lazy=False,
        supports_streaming=False,
        stateful=True,
        materializes_full_panel=True,
        requires_sorted=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=False,
        notes=("R21-DEF1 fix: true Supertrend ratchet state machine (Wilder ATR, "
               "band ratchet, close-cross flip); semantics match the pandas "
               "reference technical/indicators_v2.Supertrend. NumPy kernel over "
               "polars column arrays, NOT a native polars expression."),
        implementation_source_hash="cleaned_operators.common.polars_technical_indicators:SupertrendNative:v2-r21-def1",
        kernel_identity="numpy:supertrend_ratchet_wilder_atr:v2-r21-def1",
        parameter_domain_hash="Supertrend.high:panel,low:panel,close:panel,window:int:min=1:default=10,multiplier:float:min=0:default=3.0",
        semantic_contract_hash="Supertrend:active_band_level:ratchet_flip:v2-r21-def1",
    )
except Exception:  # pragma: no cover - registry-only environments
    _SupertrendSpec = None


@register_operator(
    name="Supertrend",
    category="time_series",
    business_category="technical_indicators",
    canonical="Supertrend",
    source=_SRC,
    backend="polars")
class SupertrendNative(SeriesOperator):
    """Supertrend indicator value.

    R21-DEF1-SUPERTRND (P1 fix): the pre-fix kernel returned exactly
    (high+low)/2 on every row — the m*ATR bands cancelled algebraically in
    (upper+lower)/2, so ``window`` and ``multiplier`` were dead parameters and
    there was zero trend/ratchet state (also: simple rolling mean of TR, not
    Wilder ATR).  This is now the true Supertrend ratchet state machine with
    semantics identical to the audited pandas reference
    (``cleaned_operators/technical/indicators_v2.py::Supertrend``):
    basic bands = (h+l)/2 ± multiplier*WilderATR(window); final bands ratchet
    (upper only falls, lower only rises, each gated by the prior close); the
    trend flips when close crosses the active final band; the output is the
    ACTIVE BAND LEVEL (lower band in an uptrend, upper band in a downtrend).

    The ratchet/flip recursion is inherently per-row stateful and cannot be
    expressed as a pure native Polars expression; it is implemented as a
    NumPy kernel over Polars column arrays.  Classified honestly via
    ``_physical_spec`` as ``ExecutionKind.POLARS_NUMPY_KERNEL`` (NOT
    POLARS_NATIVE_EXPR; no per-row Python loop is ever claimed native).
    """

    _physical_spec = _SupertrendSpec

    metadata = OperatorMetadata(
        name="Supertrend",
        category="time_series",
        description="超级趋势指标",
        param_names=["high", "low", "close", "window", "multiplier"],
        return_type="series",
        tags=["time_series", "polars", "stateful", "full_replay", "technical"],
    )

    @staticmethod
    def _wilder_atr_1d(h, l, c, w: int):
        """Wilder ATR (alpha=1/w, adjust=False, min_periods=w) on one column.

        Matches the pandas reference ``_wilder(_tr(...), w)`` including TR NaN
        propagation on any missing component (np.maximum.reduce semantics).
        """
        import numpy as np

        n = len(c)
        prev = np.full(n, np.nan, dtype=float)
        prev[1:] = c[:-1]
        with np.errstate(invalid="ignore"):
            cand = np.vstack([
                h - l,
                np.abs(h - prev),
                np.abs(l - prev),
            ])
            tr = np.where(np.all(np.isfinite(cand), axis=0), np.nanmax(cand, axis=0), np.nan)
        alpha = 1.0 / float(w)
        atr = np.full(n, np.nan, dtype=float)
        state = np.nan   # ewm state (last blended value); NaN until seeded
        seen = 0         # finite TR observations fed into the recursion
        gap = 0          # consecutive NaN rows since the last finite observation
        for t in range(n):
            if not np.isfinite(tr[t]):
                gap += 1
                # pandas ewm ignore_nulls: NaN rows hold the previous state
                # forward (NaN output only while min_periods is unmet).
                if seen >= w:
                    atr[t] = state
                continue
            if not np.isfinite(state):
                state = tr[t]
            elif gap == 0:
                state = alpha * tr[t] + (1.0 - alpha) * state
            else:
                # pandas ewm(adjust=False) across gaps: the new observation
                # re-blends with the held state at effective weight ratio
                # (1-alpha)^(gap+1)/alpha (derived and verified against pandas
                # ewm on hand series across alpha in {1/14, .25, .5, .75} and
                # gap in {1,2,3}): state = (x + ratio*s)/(1+ratio).
                decay = (1.0 - alpha) ** (gap + 1) / alpha
                state = (tr[t] + decay * state) / (1.0 + decay)
            gap = 0
            seen += 1
            if seen >= w:
                atr[t] = state
        return atr

    @staticmethod
    def _supertrend_1d(h, l, c, atr, m: float):
        """Supertrend ratchet state machine on one column (pandas parity).

        Mirrors ``indicators_v2.Supertrend``: post-gap re-seed enters UNKNOWN
        (trend=0, NaN output) and the next valid bar re-asserts direction from
        the close vs the band midline; final bands ratchet with prior-close
        gates; output = active band level.
        """
        import numpy as np

        n = len(c)
        out = np.full(n, np.nan, dtype=float)
        with np.errstate(invalid="ignore"):
            mid = 0.5 * (h + l)
        basic_u = mid + m * atr
        basic_l = mid - m * atr
        final_u = basic_u.copy()
        final_l = basic_l.copy()
        trend = np.ones(n, dtype=int)
        post_gap = False
        for t in range(1, n):
            if not (np.isfinite(h[t]) and np.isfinite(l[t]) and np.isfinite(c[t])
                    and np.isfinite(basic_u[t]) and np.isfinite(basic_l[t])):
                trend[t] = 0  # sentinel: state broken, next valid bar re-seeds
                post_gap = True
                continue
            if post_gap:
                # First valid bar after a break publishes UNKNOWN (no
                # manufactured direction); the next bar re-asserts below.
                trend[t] = 0
                final_u[t] = basic_u[t]
                final_l[t] = basic_l[t]
                post_gap = False
                out[t] = np.nan
                continue
            if np.isfinite(final_u[t - 1]) and (basic_u[t] >= final_u[t - 1] and c[t - 1] <= final_u[t - 1]):
                final_u[t] = final_u[t - 1]
            if np.isfinite(final_l[t - 1]) and (basic_l[t] <= final_l[t - 1] and c[t - 1] >= final_l[t - 1]):
                final_l[t] = final_l[t - 1]
            if trend[t - 1] == 0:
                mid_t = 0.5 * (basic_u[t] + basic_l[t])
                trend[t] = 1 if c[t] >= mid_t else -1
            elif trend[t - 1] > 0 and c[t] < final_l[t]:
                trend[t] = -1
            elif trend[t - 1] < 0 and c[t] > final_u[t]:
                trend[t] = 1
            else:
                trend[t] = trend[t - 1]
            if trend[t] > 0:
                out[t] = final_l[t]
            elif trend[t] < 0:
                out[t] = final_u[t]
            else:
                out[t] = np.nan
        return out

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame | None = None,
                         close: pl.DataFrame | None = None, window: int = 10,
                         multiplier: float = 3.0, **kwargs) -> pl.DataFrame:
        import numpy as np

        from factor_engine.cleaned_operators.parameter_validation import strict_integer, strict_finite_scalar

        w = strict_integer(window, "window", minimum=1)
        m = strict_finite_scalar(multiplier, "multiplier", minimum=0.0)
        cols = _numeric_cols(high)

        if low is None or close is None:
            raise ValueError("Supertrend requires high, low, close")

        out_cols: dict[str, "np.ndarray"] = {}  # np is numpy (imported below)
        for c in cols:
            h = high[c].to_numpy().astype(float, copy=False)
            l = low[c].to_numpy().astype(float, copy=False) if c in low.columns else np.full(high.height, np.nan)
            cl = close[c].to_numpy().astype(float, copy=False) if c in close.columns else np.full(high.height, np.nan)
            atr = self._wilder_atr_1d(h, l, cl, w)
            out_cols[c] = self._supertrend_1d(h, l, cl, atr, float(m))

        result = pl.DataFrame({c: out_cols[c] for c in cols})
        return _with_meta(result, high)


@register_operator(
    name="KAMA",
    category="time_series",
    business_category="technical_indicators",
    canonical="KAMA",
    source=_SRC,
    backend="polars")
class KAMANative(SeriesOperator):
    """Kaufman Adaptive Moving Average."""

    metadata = OperatorMetadata(
        name="KAMA",
        category="time_series",
        description="卡夫曼自适应移动平均",
        param_names=["x", "window", "fast_period", "slow_period"],
        return_type="series",
        tags=["time_series", "polars", "native", "technical"],
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 10, fast_period: int = 2,
                         slow_period: int = 30, **kwargs) -> pl.DataFrame:
        from factor_engine.cleaned_operators.parameter_validation import strict_integer

        w = strict_integer(window, "window", minimum=2)
        fast = strict_integer(fast_period, "fast_period", minimum=1)
        slow = strict_integer(slow_period, "slow_period", minimum=1)
        cols = _numeric_cols(x)

        fastest = 2.0 / (fast + 1)
        slowest = 2.0 / (slow + 1)

        exprs = []
        for c in cols:
            # Efficiency Ratio
            net_change = (pl.col(c) - pl.col(c).shift(w - 1)).abs()
            delta = (pl.col(c) - pl.col(c).shift(1)).abs()
            # R21-DEF2: rolling_sum with min_samples=1 publishes from first
            # observation (one row early vs pandas min_periods=window). Fix:
            # use min_samples=window for rolling operations.
            volatility = delta.rolling_sum(window_size=w, min_samples=w)
            er = pl.when(volatility == 0).then(0.0).otherwise(net_change / volatility)

            # Smoothing Constant
            sc = ((er * (fastest - slowest) + slowest) ** 2)

            # R21-DEF2: ewm_mean with ignore_nulls=True publishes from first
            # observation (one row early vs pandas min_periods=window). Fix:
            # count non-null observations and mask results where count < window.
            kama_raw = pl.col(c).ewm_mean(alpha=sc, adjust=False, ignore_nulls=True)
            kama_count = pl.col(c).is_not_null().cast(pl.Int64).cum_sum()
            kama = pl.when(kama_count >= w).then(kama_raw).otherwise(None)
            exprs.append(kama.alias(c))

        return x.with_columns(exprs)


# Complete module with remaining operators can be added in next chunks

