"""Auto-generated active alpha tools.

This file is maintained by scripts/update_alpha_tools.py. Manual edits may be
overwritten by the next accepted tool update.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# --- alpha_tool: directional_volatility_ratio
def directional_volatility_ratio(df, window=20, min_periods=5):
    """Compute the ratio of average up-range to average down-range.

    Economic meaning:
        Measures asymmetry in volatility: a high ratio indicates that on up days,
        the price range is larger than on down days, suggesting bullish momentum
        or positive skewness in volatility.
    Args:
        df: Single-stock OHLCV DataFrame.
        window: Rolling window length.
        min_periods: Minimum number of observations required.
    Returns:
        pd.Series aligned to df.index, ratio = up_avg_range / down_avg_range.
    Required columns:
        high, low, close.
    Leakage notes:
        Uses shifted close to determine direction, so only past and present
        observations are used.
    """
    up_close = df['close'] > df['close'].shift(1)
    down_close = ~up_close
    range_ = df['high'] - df['low']
    up_sum = range_.where(up_close, 0).rolling(window, min_periods=min_periods).sum()
    down_sum = range_.where(down_close, 0).rolling(window, min_periods=min_periods).sum()
    up_count = up_close.rolling(window, min_periods=min_periods).sum()
    down_count = down_close.rolling(window, min_periods=min_periods).sum()
    up_avg = up_sum / up_count.replace(0, np.nan)
    down_avg = down_sum / down_count.replace(0, np.nan)
    ratio = up_avg / down_avg.replace(0, np.nan)
    return ratio


# --- alpha_tool: asymmetric_volatility_ratio
def asymmetric_volatility_ratio(df, window=20, min_periods=10):
    """Compute log ratio of downside to upside volatility in returns.

    Economic meaning:
        Measures asymmetry in return volatility. A high positive value indicates
        that downside volatility is larger than upside volatility, often associated
        with crash risk or the leverage effect. A negative value indicates the
        opposite.
    Args:
        df: Single-stock OHLCV DataFrame.
        window: Rolling window length.
        min_periods: Minimum number of observations required.
    Returns:
        pd.Series aligned to df.index, log(neg_vol / pos_vol).
    Required columns:
        close.
    Leakage notes:
        Uses trailing rolling statistics only; no future data.
    """
    close = df['close']
    returns = close.pct_change()
    pos_returns = returns.where(returns > 0, 0.0)
    neg_returns = returns.where(returns < 0, 0.0).abs()
    pos_vol = pos_returns.rolling(window, min_periods=min_periods).std()
    neg_vol = neg_returns.rolling(window, min_periods=min_periods).std()
    ratio = neg_vol / pos_vol.replace(0, np.nan)
    log_ratio = np.log(ratio.clip(lower=1e-12))
    return log_ratio


# --- alpha_tool: volume_variability
def volume_variability(df, median_window=20, std_window=15):
    """Compute rolling standard deviation of volume relative to its trailing median.

    Economic meaning:
        Measures the time-series variability of volume relative to its median level.
        High values indicate erratic volume patterns, which may signal informed trading
        or regime instability.
    Args:
        df: Single-stock OHLCV DataFrame.
        median_window: Trailing window for median of volume.
        std_window: Trailing window for standard deviation of volume/median ratio.
    Returns:
        pd.Series aligned to df.index, representing volume variability.
    Required columns:
        volume.
    Leakage notes:
        Uses only past observations via trailing rolling windows.
    """
    vol = df['volume']
    median_vol = vol.rolling(median_window, min_periods=median_window).median()
    ratio = vol / median_vol.replace(0, np.nan)
    variability = ratio.rolling(std_window, min_periods=std_window).std()
    return variability.rename('volume_variability')


# --- alpha_tool: price_volume_correlation
def price_volume_correlation(df, window=20, min_periods=10):
    """Compute rolling correlation between daily returns and volume percent change.

    Economic meaning:
        Measures the degree to which price and volume move together over a trailing window.
        Positive correlation indicates volume confirms price trends; negative may indicate divergence.
    Args:
        df: Single-stock OHLCV DataFrame including 'close' and 'volume'.
        window: Rolling window length for correlation.
        min_periods: Minimum observations required.
    Returns:
        pd.Series of rolling correlation values, aligned to df.index.
    Required columns:
        close, volume.
    Leakage notes:
        Uses only past observations via rolling windows; no future data.
    """
    ret = df['close'].pct_change()
    vol_pct = df['volume'].pct_change()
    corr = ret.rolling(window, min_periods=min_periods).corr(vol_pct)
    return corr.rename('price_volume_correlation')


# --- alpha_tool: price_volume_direction_agreement
def price_volume_direction_agreement(intraday_ret, volume_diff, window=10):
    """Compute rolling proportion of days where intraday return sign matches volume change sign.

    Economic meaning:
        Measures the degree of directional confirmation between price and volume.
        High values indicate consistent agreement, suggesting informed trading or trend continuation.
        Low values indicate divergence, which may signal weakness or reversal.
    Args:
        intraday_ret: pd.Series of intraday returns (close/open - 1) for one stock.
        volume_diff: pd.Series of daily volume changes (volume.diff()) for one stock.
        window: Rolling window length for averaging agreement.
    Returns:
        pd.Series of agreement proportion, between 0 and 1, aligned to input index.
    Required columns:
        None (both inputs are precomputed series).
    Leakage notes:
        Uses only past observations via rolling window. No future data.
    """
    price_dir = np.sign(intraday_ret)
    vol_dir = np.sign(volume_diff)
    agreement = (price_dir == vol_dir).astype(float)
    avg_agreement = agreement.rolling(window, min_periods=window).mean()
    return avg_agreement.rename('price_volume_direction_agreement')

# --- alpha_tool: robust_zscore
def robust_zscore(series, window=20, clip=3.0):
    """Compute robust z-score of series relative to trailing median and MAD.

    Economic meaning:
        Measures how many MADs the current value deviates from the trailing median,
        using robust normalization that resists outliers. High absolute values
        indicate unusual activity, which may signal informed trading or regime shifts.
    Args:
        series: pd.Series of daily values for one stock.
        window: Trailing lookback window for median and MAD calculation.
        clip: Max absolute value for clipping; values beyond are capped.
    Returns:
        pd.Series of robust z-scores, aligned to the input index.
    Required columns:
        None (input is a series).
    Leakage notes:
        Uses only past observations via rolling median and MAD; no future data.
    """
    rolling_median = series.rolling(window, min_periods=window).median()
    mad = (series - rolling_median).abs().rolling(window, min_periods=window).median()
    mad = mad.replace(0, np.nan)
    z = (series - rolling_median) / mad
    z = z.clip(-clip, clip)
    return z

# --- alpha_tool: average_true_range
def average_true_range(df, window=20):
    """Compute the average true range (ATR) over a trailing window.

    Economic meaning:
        Measures the trailing average of true range, providing a volatility estimate
        based on price extremes. Higher values indicate greater volatility.
    Args:
        df: Single-stock OHLCV DataFrame.
        window: Rolling window length for averaging true range.
    Returns:
        pd.Series of ATR values, aligned to df.index.
    Required columns:
        high, low, close.
    Leakage notes:
        Uses only past observations via rolling window. Uses shifted close to compute
        true range, which is a lagged value, so no future data.
    """
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': (df['high'] - df['close'].shift()).abs(),
        'lc': (df['low'] - df['close'].shift()).abs()
    }).max(axis=1)
    atr = tr.rolling(window, min_periods=window).mean()
    return atr.rename('average_true_range')

# --- alpha_tool: intraday_close_location
def intraday_close_location(df):
    """Compute the relative position of close within the day's high-low range.

    Economic meaning:
        Measures where the closing price falls within the day's high-low range,
        from 0 (close equals low) to 1 (close equals high). Values near 1 indicate
        bullish intraday pressure; near 0 indicate bearish pressure.
    Args:
        df: Single-stock OHLCV DataFrame.
    Returns:
        pd.Series aligned to df.index, with values between 0 and 1.
    Required columns:
        high, low, close.
    Leakage notes:
        Uses only same-day high, low, and close; no future data.
    """
    range_ = df['high'] - df['low']
    range_ = range_.replace(0, np.nan)
    close_loc = (df['close'] - df['low']) / range_
    return close_loc.rename('intraday_close_location')

# --- alpha_tool: intraday_return_ratio
def intraday_return_ratio(df):
    """Compute the open-to-close return normalized by the daily high-low range.

    Economic meaning:
        Measures the direction and magnitude of intraday price movement relative to the day's volatility range. Positive values indicate bullish pressure (close above open), negative values indicate bearish pressure (close below open). The ratio is bounded between -1 and 1 for days where close stays within the range, but can exceed these bounds if close is outside the range (e.g., gap moves).
    Args:
        df: Single-stock OHLCV DataFrame.
    Returns:
        pd.Series of intraday return ratios, aligned to df.index.
    Required columns:
        open, high, low, close.
    Leakage notes:
        Uses only same-day open, high, low, close; no future data.
    """
    range_ = df['high'] - df['low']
    range_ = range_.replace(0, np.nan)
    ratio = (df['close'] - df['open']) / range_
    return ratio.rename('intraday_return_ratio')

# --- alpha_tool: range_ratio
def range_ratio(df, short_window=5, long_window=30):
    """Compute ratio of short-term average range to long-term average range.

    Economic meaning:
        Measures the trailing ratio of short-term volatility (average daily range)
        to long-term volatility. Values > 1 indicate short-term volatility expansion,
        values < 1 indicate contraction. Useful for regime detection.
    Args:
        df: Single-stock OHLCV DataFrame.
        short_window: Window for short-term average range.
        long_window: Window for long-term average range.
    Returns:
        pd.Series of range ratios, aligned to df.index.
    Required columns:
        high, low.
    Leakage notes:
        Uses only past observations via rolling windows; no future data.
    """
    range_ = df['high'] - df['low']
    short_avg = range_.rolling(short_window, min_periods=short_window).mean()
    long_avg = range_.rolling(long_window, min_periods=long_window).mean()
    ratio = short_avg / long_avg.replace(0, np.nan)
    return ratio.rename('range_ratio')

# --- alpha_tool: drawdown_depth
def drawdown_depth(df, window=252):
    """Compute trailing drawdown depth from peak close.

    Economic meaning:
        Measures how far the current close has fallen from its trailing maximum close,
        expressed as a fraction of the peak. A higher value indicates a deeper drawdown,
        signaling increased risk or mean reversion potential.
    Args:
        df: Single-stock OHLCV DataFrame.
        window: Trailing lookback window for rolling maximum.
    Returns:
        pd.Series of drawdown depth (positive values between 0 and 1), aligned to df.index.
    Required columns:
        close.
    Leakage notes:
        Uses only trailing rolling maximum; no future data.
    """
    rolling_max = df['close'].rolling(window, min_periods=window).max()
    depth = (rolling_max - df['close']) / rolling_max.replace(0, np.nan)
    return depth.rename('drawdown_depth')
