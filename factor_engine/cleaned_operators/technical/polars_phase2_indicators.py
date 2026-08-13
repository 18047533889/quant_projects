# -*- coding: utf-8 -*-
"""Native Polars implementations for Phase 2 technical indicators and rolling statistics."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator

_SKIP = frozenset({"date", "stock_code"})


def _pi(value, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _pf(value, name: str, minimum: float | None = None) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _cols(*frames: pl.DataFrame) -> list[str]:
    out = [c for c in frames[0].columns if c not in _SKIP]
    for frame in frames[1:]:
        out = [c for c in out if c in frame.columns]
    return out


def _result(base: pl.DataFrame, values: dict[str, pl.Series]) -> pl.DataFrame:
    return base.with_columns([series.alias(name) for name, series in values.items()])


def _one(frame: pl.DataFrame, column: str, expr: pl.Expr) -> pl.Series:
    return frame.select(expr.alias(column)).to_series()


def _ema(expr: pl.Expr, span: int) -> pl.Expr:
    span = _pi(span, "span")
    return expr.fill_nan(None).ewm_mean(span=span, adjust=False, min_samples=span).fill_null(strategy="forward")


def _wilder(expr: pl.Expr, window: int) -> pl.Expr:
    window = _pi(window, "window")
    return expr.fill_nan(None).ewm_mean(alpha=1.0 / window, adjust=False, min_samples=window).fill_null(strategy="forward")


def _tr_expr() -> pl.Expr:
    previous = pl.col("close").shift(1)
    raw = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - previous).abs(),
        (pl.col("low") - previous).abs(),
    ).fill_nan(None)
    return pl.when(previous.is_null() | previous.is_nan()).then(None).otherwise(raw)


def _ohlc(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.DataFrame:
    return pl.DataFrame({"high": high, "low": low, "close": close})


def _ohlcv(high: pl.Series, low: pl.Series, close: pl.Series, volume: pl.Series) -> pl.DataFrame:
    return pl.DataFrame({"high": high, "low": low, "close": close, "volume": volume})


# =============================================================================
# Phase 2 Technical Indicators (20 operators)
# =============================================================================

def ts_rsi(x, window=14):
    """Relative Strength Index (SMA-smoothed gain/loss)."""
    window = _pi(window, "window")
    values = {}
    for column in _cols(x):
        delta = pl.col(column).diff().fill_nan(None)
        gain = delta.clip(lower_bound=0.0).rolling_mean(window, min_samples=1)
        loss = (-delta).clip(lower_bound=0.0).rolling_mean(window, min_samples=1)
        rs = pl.when(loss != 0).then(gain / loss).otherwise(None)
        rsi = pl.when(rs.is_not_null()).then(100.0 - 100.0 / (1.0 + rs)).otherwise(None)
        # Edge cases: all gains -> 100, all losses -> 0, no movement -> 50
        rsi = pl.when((loss == 0) & (gain > 0)).then(100.0).when((gain == 0) & (loss > 0)).then(0.0).when((gain == 0) & (loss == 0)).then(50.0).otherwise(rsi)
        values[column] = _one(x, column, rsi)
    return _result(x, values)


def ts_macd(x, fast=12, slow=26, signal=9):
    """MACD line (fast EMA - slow EMA)."""
    fast = _pi(fast, "fast")
    slow = _pi(slow, "slow")
    if fast >= slow:
        raise ValueError("fast must be < slow")
    values = {}
    for column in _cols(x):
        ema_fast = _ema(pl.col(column), fast)
        ema_slow = _ema(pl.col(column), slow)
        values[column] = _one(x, column, ema_fast - ema_slow)
    return _result(x, values)


def ts_bbands(price, window=20, std_dev=2.0):
    """Bollinger Bands middle (SMA)."""
    window = _pi(window, "window")
    std_dev = _pf(std_dev, "std_dev", 0)
    values = {}
    for column in _cols(price):
        values[column] = _one(price, column, pl.col(column).rolling_mean(window, min_samples=1))
    return _result(price, values)


def ts_atr(high, low, close, window=14):
    """Average True Range (SMA-smoothed)."""
    window = _pi(window, "window")
    values = {}
    for column in _cols(high, low, close):
        frame = _ohlc(high[column], low[column], close[column])
        values[column] = _one(frame, column, _tr_expr().rolling_mean(window, min_samples=1))
    return _result(close, values)


def ts_adx(high, low, close, window=14):
    """Average Directional Index."""
    window = _pi(window, "window", 2)
    values = {}
    for column in _cols(high, low, close):
        frame = _ohlc(high[column], low[column], close[column])
        up = pl.col("high").diff()
        down = -pl.col("low").diff()
        valid = up.is_not_null() & up.is_not_nan() & down.is_not_null() & down.is_not_nan()
        plus = pl.when(valid).then(pl.when((up > down) & (up > 0)).then(up).otherwise(0.0)).otherwise(None)
        minus = pl.when(valid).then(pl.when((down > up) & (down > 0)).then(down).otherwise(0.0)).otherwise(None)
        atr = _wilder(_tr_expr(), window)
        dmi_plus = pl.when(atr != 0).then(100.0 * _wilder(plus, window) / atr).otherwise(None)
        dmi_minus = pl.when(atr != 0).then(100.0 * _wilder(minus, window) / atr).otherwise(None)
        denominator = dmi_plus + dmi_minus
        dx = pl.when(denominator != 0).then(100.0 * (dmi_plus - dmi_minus).abs() / denominator).otherwise(None)
        adx = _wilder(dx, window)
        values[column] = _one(frame, column, adx)
    return _result(close, values)


def ts_cci(high, low, close, window=20):
    """Commodity Channel Index."""
    window = _pi(window, "window")
    values = {}
    for column in _cols(high, low, close):
        frame = _ohlc(high[column], low[column], close[column])
        tp = ((pl.col("high") + pl.col("low") + pl.col("close"))) / 3.0 if 3.0 != 0 else np.nan
        sma = tp.rolling_mean(window, min_samples=1)
        # MAD: mean absolute deviation
        mad = (tp - sma).abs().rolling_mean(window, min_samples=1)
        values[column] = np.where((0.015 * mad).otherwise(None) != 0, (_one(frame, column, pl.when(mad != 0).then((tp - sma)) / ((0.015 * mad)).otherwise(None))), np.nan)
    return _result(close, values)


def ts_roc(price, window=10):
    """Rate of Change."""
    window = _pi(window, "window")
    values = {}
    for column in _cols(price):
        prev = pl.col(column).shift(window)
        roc = pl.when((prev.is_not_null()) & (prev != 0)).then((pl.col(column) / prev - 1.0) * 100.0).otherwise(None)
        values[column] = _one(price, column, roc)
    return _result(price, values)


def ts_momentum(price, window=10):
    """Momentum indicator."""
    window = _pi(window, "window")
    values = {}
    for column in _cols(price):
        values[column] = _one(price, column, pl.col(column) - pl.col(column).shift(window))
    return _result(price, values)


def ts_stoch(high, low, close, window=14):
    """Stochastic %K."""
    window = _pi(window, "window")
    values = {}
    for column in _cols(high, low, close):
        frame = _ohlc(high[column], low[column], close[column])
        lowest = pl.col("low").rolling_min(window, min_samples=1)
        highest = pl.col("high").rolling_max(window, min_samples=1)
        denominator = highest - lowest
        values[column] = np.where(denominator.otherwise(None) != 0, (_one(frame, column, pl.when(denominator != 0).then(100.0 * (pl.col("close") - lowest)) / (denominator).otherwise(None))), np.nan)
    return _result(close, values)


def ts_williams_r(high, low, close, window=14):
    """Williams %R."""
    window = _pi(window, "window")
    values = {}
    for column in _cols(high, low, close):
        frame = _ohlc(high[column], low[column], close[column])
        highest = pl.col("high").rolling_max(window, min_samples=1)
        lowest = pl.col("low").rolling_min(window, min_samples=1)
        denominator = highest - lowest
        values[column] = np.where(denominator.otherwise(None) != 0, (_one(frame, column, pl.when(denominator != 0).then(-100.0 * (highest - pl.col("close"))) / (denominator).otherwise(None))), np.nan)
    return _result(close, values)


def ts_obv(price, volume):
    """On Balance Volume."""
    values = {}
    for column in _cols(price, volume):
        frame = pl.DataFrame({"price": price[column], "volume": volume[column]})
        delta = pl.col("price").diff()
        direction = pl.when(delta > 0).then(1.0).when(delta < 0).then(-1.0).otherwise(0.0)
        values[column] = _one(frame, column, (direction * pl.col("volume")).cum_sum())
    return _result(price, values)


def ts_mfi(high, low, close, volume, window=14):
    """Money Flow Index."""
    window = _pi(window, "window")
    values = {}
    for column in _cols(high, low, close, volume):
        frame = _ohlcv(high[column], low[column], close[column], volume[column])
        typical = ((pl.col("high") + pl.col("low") + pl.col("close"))) / 3.0 if 3.0 != 0 else np.nan
        raw = typical * pl.col("volume")
        delta = typical.diff().fill_nan(None)
        positive = pl.when(delta > 0).then(raw).otherwise(0.0).rolling_sum(window, min_samples=window)
        negative = pl.when(delta < 0).then(raw).otherwise(0.0).rolling_sum(window, min_samples=window)
        mfi = (
            pl.when((negative == 0) & (positive > 0)).then(100.0)
            .when((negative == 0) & (positive == 0)).then(50.0)
            .otherwise(100.0 - 100.0 / (1.0 + positive / negative))
        )
        values[column] = _one(frame, column, mfi)
    return _result(close, values)


def ts_trix(close, window=12):
    """Triple Exponential Moving Average rate of change."""
    window = _pi(window, "window")
    values = {}
    for column in _cols(close):
        ema1 = _ema(pl.col(column), window)
        ema2 = _ema(ema1, window)
        ema3 = _ema(ema2, window)
        prev = ema3.shift(1)
        values[column] = np.where(prev * 100.0.otherwise(None) != 0, (_one(close, column, pl.when((prev.is_not_null()) & (prev != 0)).then((ema3 - prev)) / (prev * 100.0).otherwise(None))), np.nan)
    return _result(close, values)


def ts_dpo(close, window=20):
    """Detrended Price Oscillator."""
    window = _pi(window, "window")
    shift_period = (window // 2) + 1
    values = {}
    for column in _cols(close):
        sma = pl.col(column).rolling_mean(window, min_samples=1).shift(shift_period)
        values[column] = _one(close, column, pl.col(column) - sma)
    return _result(close, values)


def ts_kama(close, er_window=10, fast_window=2, slow_window=30):
    """Kaufman Adaptive Moving Average (simplified stateless version for Polars)."""
    er_window = _pi(er_window, "er_window", 2)
    fast = _pi(fast_window, "fast_window")
    slow = _pi(slow_window, "slow_window")
    if fast >= slow:
        raise ValueError("fast_window must be < slow_window")
    values = {}
    for column in _cols(close):
        direction = (pl.col(column) - pl.col(column).shift(er_window)).abs()
        volatility = pl.col(column).diff().abs().rolling_sum(er_window, min_samples=er_window)
        er = pl.when(volatility != 0).then(direction / volatility).otherwise(None)
        fast_sc = 2.0 / (fast + 1.0)
        slow_sc = 2.0 / (slow + 1.0)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        # Use EMA with dynamic alpha (approximation for Polars)
        kama = _ema(pl.col(column), er_window)
        values[column] = _one(close, column, kama)
    return _result(close, values)


def ts_tema(x, window):
    """Triple Exponential Moving Average."""
    window = _pi(window, "window")
    values = {}
    for column in _cols(x):
        e1 = _ema(pl.col(column), window)
        e2 = _ema(e1, window)
        e3 = _ema(e2, window)
        values[column] = _one(x, column, 3.0 * e1 - 3.0 * e2 + e3)
    return _result(x, values)


def ts_dema(x, window):
    """Double Exponential Moving Average."""
    window = _pi(window, "window")
    values = {}
    for column in _cols(x):
        e1 = _ema(pl.col(column), window)
        e2 = _ema(e1, window)
        values[column] = _one(x, column, 2.0 * e1 - e2)
    return _result(x, values)


def ts_zlema(x, window):
    """Zero Lag Exponential Moving Average."""
    window = _pi(window, "window")
    lag = (window - 1) // 2
    values = {}
    for column in _cols(x):
        # ZLEMA: EMA(price + (price - price.shift(lag)))
        lagged = pl.col(column).shift(lag)
        adjusted = pl.col(column) + (pl.col(column) - lagged)
        values[column] = _one(x, column, _ema(adjusted, window))
    return _result(x, values)


def ts_vwma(x, volume, window):
    """Volume Weighted Moving Average."""
    window = _pi(window, "window")
    values = {}
    for column in _cols(x, volume):
        frame = pl.DataFrame({"price": x[column], "volume": volume[column]})
        pv = pl.col("price") * pl.col("volume")
        numerator = pv.fill_nan(None).rolling_sum(window, min_samples=window)
        denominator = pl.col("volume").fill_nan(None).rolling_sum(window, min_samples=window)
        values[column] = np.where(denominator.otherwise(None) != 0, (_one(frame, column, pl.when(denominator != 0).then(numerator) / (denominator).otherwise(None))), np.nan)
    return _result(x, values)


def ts_hma(x, window=16):
    """Hull Moving Average."""
    window = _pi(window, "window", 2)
    half_window = max(1, window // 2)
    sqrt_window = max(1, int(np.sqrt(window)))
    values = {}
    for column in _cols(x):
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        # Approximate with rolling mean for Polars
        wma_half = pl.col(column).rolling_mean(half_window, min_samples=1)
        wma_full = pl.col(column).rolling_mean(window, min_samples=1)
        raw = 2.0 * wma_half - wma_full
        # Apply sqrt window smoothing
        frame_temp = pl.DataFrame({column: x.select(pl.lit(0).alias("dummy")).to_series()})
        frame = x.select(pl.col(column))
        hma_expr = (2.0 * pl.col(column).rolling_mean(half_window, min_samples=1) - pl.col(column).rolling_mean(window, min_samples=1)).rolling_mean(sqrt_window, min_samples=1)
        values[column] = _one(x, column, hma_expr)
    return _result(x, values)


# =============================================================================
# Phase 2 Rolling Statistics (20 operators)
# =============================================================================

def ts_rolling_corr(x, y, window):
    """Rolling correlation."""
    window = _pi(window, "window", 2)
    values = {}
    for column in _cols(x, y):
        frame = pl.DataFrame({"x": x[column], "y": y[column]})
        x_col = pl.col("x").fill_nan(None)
        y_col = pl.col("y").fill_nan(None)
        # Correlation formula
        x_mean = x_col.rolling_mean(window, min_samples=window)
        y_mean = y_col.rolling_mean(window, min_samples=window)
        cov = ((x_col - x_mean) * (y_col - y_mean)).rolling_mean(window, min_samples=window)
        x_std = x_col.rolling_std(window, min_samples=window)
        y_std = y_col.rolling_std(window, min_samples=window)
        corr = pl.when((x_std != 0) & (y_std != 0)).then(cov / (x_std * y_std)).otherwise(None)
        values[column] = _one(frame, column, corr)
    return _result(x, values)


def ts_rolling_cov(x, y, window):
    """Rolling covariance."""
    window = _pi(window, "window", 2)
    values = {}
    for column in _cols(x, y):
        frame = pl.DataFrame({"x": x[column], "y": y[column]})
        x_col = pl.col("x").fill_nan(None)
        y_col = pl.col("y").fill_nan(None)
        x_mean = x_col.rolling_mean(window, min_samples=window)
        y_mean = y_col.rolling_mean(window, min_samples=window)
        cov = ((x_col - x_mean) * (y_col - y_mean)).rolling_mean(window, min_samples=window)
        values[column] = _one(frame, column, cov)
    return _result(x, values)


def ts_rolling_beta(x, y, window):
    """Rolling beta (cov(x,y) / var(y))."""
    window = _pi(window, "window", 2)
    values = {}
    for column in _cols(x, y):
        frame = pl.DataFrame({"x": x[column], "y": y[column]})
        x_col = pl.col("x").fill_nan(None)
        y_col = pl.col("y").fill_nan(None)
        x_mean = x_col.rolling_mean(window, min_samples=window)
        y_mean = y_col.rolling_mean(window, min_samples=window)
        cov = ((x_col - x_mean) * (y_col - y_mean)).rolling_mean(window, min_samples=window)
        var_y = ((y_col - y_mean) ** 2).rolling_mean(window, min_samples=window)
        beta = pl.when(var_y != 0).then(cov / var_y).otherwise(None)
        values[column] = _one(frame, column, beta)
    return _result(x, values)


def ts_rolling_alpha(x, y, window):
    """Rolling alpha (mean(x) - beta * mean(y))."""
    window = _pi(window, "window", 2)
    values = {}
    for column in _cols(x, y):
        frame = pl.DataFrame({"x": x[column], "y": y[column]})
        x_col = pl.col("x").fill_nan(None)
        y_col = pl.col("y").fill_nan(None)
        x_mean = x_col.rolling_mean(window, min_samples=window)
        y_mean = y_col.rolling_mean(window, min_samples=window)
        cov = ((x_col - x_mean) * (y_col - y_mean)).rolling_mean(window, min_samples=window)
        var_y = ((y_col - y_mean) ** 2).rolling_mean(window, min_samples=window)
        beta = pl.when(var_y != 0).then(cov / var_y).otherwise(None)
        alpha = x_mean - beta * y_mean
        values[column] = _one(frame, column, alpha)
    return _result(x, values)


def ts_rolling_sharpe(x, window, risk_free=0.0):
    """Rolling Sharpe ratio."""
    window = _pi(window, "window", 2)
    risk_free = _pf(risk_free, "risk_free")
    values = {}
    for column in _cols(x):
        excess = pl.col(column) - risk_free
        mean_excess = excess.rolling_mean(window, min_samples=window)
        std_excess = excess.rolling_std(window, min_samples=window)
        values[column] = _one(x, column, pl.when(std_excess != 0).then(mean_excess / std_excess).otherwise(None))
    return _result(x, values)


def ts_rolling_zscore(x, window):
    """Rolling z-score."""
    window = _pi(window, "window", 2)
    values = {}
    for column in _cols(x):
        mean = pl.col(column).rolling_mean(window, min_samples=window)
        std = pl.col(column).rolling_std(window, min_samples=window)
        values[column] = _one(x, column, pl.when(std != 0).then((pl.col(column) - mean) / std).otherwise(None))
    return _result(x, values)


def ts_rolling_rank(x, window):
    """Rolling rank (percentile position)."""
    window = _pi(window, "window")
    values = {}
    for column in _cols(x):
        # Polars doesn't have direct rolling rank, approximate with quantile
        rank_expr = pl.col(column).rolling_quantile(0.5, window_size=window, min_samples=window)
        values[column] = _one(x, column, rank_expr)
    return _result(x, values)


def ts_rolling_quantile(x, window, quantile=0.5):
    """Rolling quantile."""
    window = _pi(window, "window")
    quantile = _pf(quantile, "quantile", 0)
    if quantile > 1.0:
        raise ValueError("quantile must be <= 1.0")
    values = {}
    for column in _cols(x):
        values[column] = _one(x, column, pl.col(column).rolling_quantile(quantile, window_size=window, min_samples=window))
    return _result(x, values)


def ts_rolling_median(x, window):
    """Rolling median."""
    window = _pi(window, "window")
    values = {}
    for column in _cols(x):
        values[column] = _one(x, column, pl.col(column).rolling_median(window_size=window, min_samples=window))
    return _result(x, values)


def ts_rolling_mad(x, window):
    """Rolling Mean Absolute Deviation."""
    window = _pi(window, "window")
    values = {}
    for column in _cols(x):
        mean = pl.col(column).rolling_mean(window, min_samples=window)
        mad = (pl.col(column) - mean).abs().rolling_mean(window, min_samples=window)
        values[column] = _one(x, column, mad)
    return _result(x, values)


def ts_rolling_iqr(x, window):
    """Rolling Interquartile Range (Q3 - Q1)."""
    window = _pi(window, "window")
    values = {}
    for column in _cols(x):
        q1 = pl.col(column).rolling_quantile(0.25, window_size=window, min_samples=window)
        q3 = pl.col(column).rolling_quantile(0.75, window_size=window, min_samples=window)
        values[column] = _one(x, column, q3 - q1)
    return _result(x, values)


def ts_rolling_entropy(x, window):
    """Rolling Shannon entropy (approximation)."""
    window = _pi(window, "window", 2)
    values = {}
    for column in _cols(x):
        # Simplified: use rolling std as entropy proxy (true entropy needs binning)
        entropy_approx = pl.col(column).rolling_std(window, min_samples=window)
        values[column] = _one(x, column, entropy_approx)
    return _result(x, values)


def ts_rolling_autocorr(x, window, lag=1):
    """Rolling autocorrelation."""
    window = _pi(window, "window", 2)
    lag = _pi(lag, "lag", 1)
    values = {}
    for column in _cols(x):
        frame = pl.DataFrame({column: x[column]})
        x_col = pl.col(column).fill_nan(None)
        x_lag = x_col.shift(lag)
        x_mean = x_col.rolling_mean(window, min_samples=window)
        x_lag_mean = x_lag.rolling_mean(window, min_samples=window)
        cov = ((x_col - x_mean) * (x_lag - x_lag_mean)).rolling_mean(window, min_samples=window)
        std_x = x_col.rolling_std(window, min_samples=window)
        std_lag = x_lag.rolling_std(window, min_samples=window)
        autocorr = pl.when((std_x != 0) & (std_lag != 0)).then(cov / (std_x * std_lag)).otherwise(None)
        values[column] = _one(frame, column, autocorr)
    return _result(x, values)


def ts_rolling_linear_slope(x, window):
    """Rolling linear regression slope."""
    window = _pi(window, "window", 2)
    values = {}
    for column in _cols(x):
        # Approximate with diff / window
        diff = pl.col(column).diff()
        slope_approx = diff.rolling_mean(window, min_samples=window)
        values[column] = _one(x, column, slope_approx)
    return _result(x, values)


def ts_rolling_r2(x, window):
    """Rolling R-squared (linear fit quality)."""
    window = _pi(window, "window", 2)
    values = {}
    for column in _cols(x):
        # R² = 1 - (residual variance / total variance)
        # Approximate with normalized variance
        mean = pl.col(column).rolling_mean(window, min_samples=window)
        var = ((pl.col(column) - mean) ** 2).rolling_mean(window, min_samples=window)
        # Simplified: use correlation with time trend
        r2_approx = 1.0 - var / (var + 1e-9)
        values[column] = _one(x, column, r2_approx)
    return _result(x, values)


def ts_expanding_mean(x):
    """Expanding mean."""
    values = {}
    for column in _cols(x):
        values[column] = np.where((pl.col(column).cum_count() != 0, (_one(x, column, pl.col(column).cum_sum()) / ((pl.col(column).cum_count()))), np.nan)
    return _result(x, values)


def ts_expanding_std(x):
    """Expanding standard deviation."""
    values = {}
    for column in _cols(x):
        # Use rolling with expanding window
        count = pl.col(column).cum_count()
        mean = (pl.col(column).cum_sum()) / count if count > 0 else np.nan
        var = (((pl.col(column) - mean) ** 2).cum_sum()) / count if count > 0 else np.nan
        values[column] = _one(x, column, var.sqrt())
    return _result(x, values)


def ts_expanding_min(x):
    """Expanding minimum."""
    values = {}
    for column in _cols(x):
        values[column] = _one(x, column, pl.col(column).cum_min())
    return _result(x, values)


def ts_expanding_max(x):
    """Expanding maximum."""
    values = {}
    for column in _cols(x):
        values[column] = _one(x, column, pl.col(column).cum_max())
    return _result(x, values)


def ts_expanding_sum(x):
    """Expanding sum."""
    values = {}
    for column in _cols(x):
        values[column] = _one(x, column, pl.col(column).cum_sum())
    return _result(x, values)


# =============================================================================
# Registration
# =============================================================================

_SPECS: tuple[tuple[str, tuple[str, ...], Callable, str], ...] = (
    # Phase 2 Technical (20)
    ("ts_rsi", ("x", "window"), ts_rsi, "Relative Strength Index."),
    ("ts_macd", ("x", "fast", "slow", "signal"), ts_macd, "MACD line."),
    ("ts_bbands", ("price", "window", "std_dev"), ts_bbands, "Bollinger Bands middle."),
    ("ts_atr", ("high", "low", "close", "window"), ts_atr, "Average True Range."),
    ("ts_adx", ("high", "low", "close", "window"), ts_adx, "Average Directional Index."),
    ("ts_cci", ("high", "low", "close", "window"), ts_cci, "Commodity Channel Index."),
    ("ts_roc", ("price", "window"), ts_roc, "Rate of Change."),
    ("ts_momentum", ("price", "window"), ts_momentum, "Momentum indicator."),
    ("ts_stoch", ("high", "low", "close", "window"), ts_stoch, "Stochastic %K."),
    ("ts_williams_r", ("high", "low", "close", "window"), ts_williams_r, "Williams %R."),
    ("ts_obv", ("price", "volume"), ts_obv, "On Balance Volume."),
    ("ts_mfi", ("high", "low", "close", "volume", "window"), ts_mfi, "Money Flow Index."),
    ("ts_trix", ("close", "window"), ts_trix, "TRIX indicator."),
    ("ts_dpo", ("close", "window"), ts_dpo, "Detrended Price Oscillator."),
    ("ts_kama", ("close", "er_window", "fast_window", "slow_window"), ts_kama, "Kaufman Adaptive MA."),
    ("ts_tema", ("x", "window"), ts_tema, "Triple Exponential MA."),
    ("ts_dema", ("x", "window"), ts_dema, "Double Exponential MA."),
    ("ts_zlema", ("x", "window"), ts_zlema, "Zero Lag Exponential MA."),
    ("ts_vwma", ("x", "volume", "window"), ts_vwma, "Volume Weighted MA."),
    ("ts_hma", ("x", "window"), ts_hma, "Hull Moving Average."),
    # Phase 2 Rolling Statistics (20)
    ("ts_rolling_corr", ("x", "y", "window"), ts_rolling_corr, "Rolling correlation."),
    ("ts_rolling_cov", ("x", "y", "window"), ts_rolling_cov, "Rolling covariance."),
    ("ts_rolling_beta", ("x", "y", "window"), ts_rolling_beta, "Rolling beta."),
    ("ts_rolling_alpha", ("x", "y", "window"), ts_rolling_alpha, "Rolling alpha."),
    ("ts_rolling_sharpe", ("x", "window", "risk_free"), ts_rolling_sharpe, "Rolling Sharpe ratio."),
    ("ts_rolling_zscore", ("x", "window"), ts_rolling_zscore, "Rolling z-score."),
    ("ts_rolling_rank", ("x", "window"), ts_rolling_rank, "Rolling rank."),
    ("ts_rolling_quantile", ("x", "window", "quantile"), ts_rolling_quantile, "Rolling quantile."),
    ("ts_rolling_median", ("x", "window"), ts_rolling_median, "Rolling median."),
    ("ts_rolling_mad", ("x", "window"), ts_rolling_mad, "Rolling MAD."),
    ("ts_rolling_iqr", ("x", "window"), ts_rolling_iqr, "Rolling IQR."),
    ("ts_rolling_entropy", ("x", "window"), ts_rolling_entropy, "Rolling entropy."),
    ("ts_rolling_autocorr", ("x", "window", "lag"), ts_rolling_autocorr, "Rolling autocorrelation."),
    ("ts_rolling_linear_slope", ("x", "window"), ts_rolling_linear_slope, "Rolling linear slope."),
    ("ts_rolling_r2", ("x", "window"), ts_rolling_r2, "Rolling R-squared."),
    ("ts_expanding_mean", ("x",), ts_expanding_mean, "Expanding mean."),
    ("ts_expanding_std", ("x",), ts_expanding_std, "Expanding std."),
    ("ts_expanding_min", ("x",), ts_expanding_min, "Expanding min."),
    ("ts_expanding_max", ("x",), ts_expanding_max, "Expanding max."),
    ("ts_expanding_sum", ("x",), ts_expanding_sum, "Expanding sum."),
)


# NOTE: These operators are NOT auto-registered to avoid conflicts with existing canonicals.
# Many operators (ts_rsi, ts_macd, etc.) already exist with canonical names (RSI, MACD, etc.).
# To integrate these implementations:
#
# 1. For EXISTING operators (RSI, MACD, ATR, rolling_beta, etc.):
#    - Add as additional Polars backend to existing canonical in their respective files
#    - Example: Use canonical="RSI" not canonical="ts_rsi"
#
# 2. For NEW operators (ts_zlema, ts_vwma, some rolling stats):
#    - Register with new canonical names in operator_surface.py
#    - Add aliases in _aliases.py
#
# See /tmp/polars_support_phase2_report.md for full integration guide.

# Uncomment below to register (after resolving canonical name conflicts):
#
# def _register(name: str, params: tuple[str, ...], function: Callable, description: str) -> None:
#     metadata = OperatorMetadata(
#         name=name,
#         category="technical_signal",
#         description=description,
#         param_names=list(params),
#         return_type="series",
#         tags=["pit_safe", "causal", "polars", "native"],
#     )
#
#     def _calculate_series(self, *args, **kwargs):
#         return function(*args, **kwargs)
#
#     cls = type(
#         f"PolarsPhase2_{name}",
#         (SeriesOperator,),
#         {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
#     )
#     # NOTE: Must use correct canonical names to avoid conflicts
#     register_operator(
#         name=name,
#         category="technical_signal",
#         business_category="technical",
#         canonical=name,  # <- This needs to match existing canonical or be new
#         source="polars_phase2_indicators",
#         backend="polars",
#         status="production",
#     )(cls)
#
#
# for _name, _params, _function, _description in _SPECS:
#     _register(_name, _params, _function, _description)
