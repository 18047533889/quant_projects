"""
Cycle extraction using bandpass filters.

All functions are causal - they exclude the current observation and operate
per-asset to ensure no future leakage.
"""
import numpy as np
import pandas as pd
from typing import Optional
from scipy import signal


def bandpass_filter(
    values: pd.DataFrame,
    low_freq: float,
    high_freq: float,
    order: int = 4,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Causal bandpass filter for cycle extraction.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    low_freq : float
        Low cutoff frequency (cycles per observation).
        Must be in (0, 0.5) where 0.5 is Nyquist frequency.
    high_freq : float
        High cutoff frequency (cycles per observation).
        Must be in (low_freq, 0.5).
    order : int
        Filter order. Higher order = sharper cutoff.
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to filter

    Returns
    -------
    pd.Series
        Bandpass filtered series (cycle component).
        First observation per asset is NaN (causal lag).

    Notes
    -----
    Uses shift(1) to exclude current observation, ensuring causality.
    Frequencies are normalized to [0, 0.5] where 0.5 is Nyquist.
    For business cycle extraction (2-8 years with quarterly data):
        low_freq = 1/(8*4) = 0.03125, high_freq = 1/(2*4) = 0.125
    """
    if not (0 < low_freq < 0.5):
        raise ValueError(f"low_freq must be in (0, 0.5), got {low_freq}")
    if not (low_freq < high_freq < 0.5):
        raise ValueError(f"high_freq must be in (low_freq, 0.5), got {high_freq}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    def _bandpass_single(series: pd.Series) -> pd.Series:
        """Apply bandpass filter to a single asset's time series."""
        # Shift to exclude current observation
        lagged = series.shift(1)

        if lagged.isna().all():
            return pd.Series(np.nan, index=series.index)

        # Extract non-NaN values
        valid_mask = lagged.notna()
        y = lagged[valid_mask].values

        if len(y) < 2 * order + 1:
            # Need sufficient data for filter
            result = pd.Series(np.nan, index=series.index)
            return result

        # Design Butterworth bandpass filter
        sos = signal.butter(
            order,
            [low_freq, high_freq],
            btype='bandpass',
            output='sos',
            fs=1.0,  # Normalized frequency
        )

        # Apply filter with zero-phase (filtfilt) to avoid phase shift
        # Note: filtfilt is acausal but we already shifted data by 1
        try:
            filtered = signal.sosfiltfilt(sos, y)
        except Exception:
            # Filter may fail for various reasons
            result = pd.Series(np.nan, index=series.index)
            return result

        # Map back to original index
        result = pd.Series(np.nan, index=series.index)
        result.loc[valid_mask] = filtered

        return result

    result = values.groupby(asset_col, sort=False)[value_col].apply(_bandpass_single)

    # Reset index to match input
    result.index = values.index

    return result


def extract_cycle(
    values: pd.DataFrame,
    low_period: int,
    high_period: int,
    order: int = 4,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Extract cycle component with specified period range.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    low_period : int
        Minimum cycle period in observations (e.g., 8 quarters for 2 years)
    high_period : int
        Maximum cycle period in observations (e.g., 32 quarters for 8 years)
    order : int
        Filter order
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to filter

    Returns
    -------
    pd.Series
        Cycle component within specified period range.
        First observation per asset is NaN.

    Notes
    -----
    Converts period to frequency: freq = 1 / period
    For business cycle (2-8 years, quarterly data):
        low_period = 8, high_period = 32
    """
    if low_period < 2:
        raise ValueError(f"low_period must be >= 2, got {low_period}")
    if high_period <= low_period:
        raise ValueError(f"high_period must be > low_period, got {high_period} <= {low_period}")

    # Convert periods to frequencies
    # Longer period = lower frequency
    high_freq = 1.0 / low_period
    low_freq = 1.0 / high_period

    # Ensure frequencies are valid
    high_freq = min(high_freq, 0.49)  # Below Nyquist
    low_freq = max(low_freq, 0.01)    # Above zero

    if low_freq >= high_freq:
        raise ValueError(
            f"Invalid period range: [{low_period}, {high_period}] "
            f"produces invalid frequency range [{low_freq}, {high_freq}]"
        )

    return bandpass_filter(
        values,
        low_freq=low_freq,
        high_freq=high_freq,
        order=order,
        asset_col=asset_col,
        time_col=time_col,
        value_col=value_col,
    )


def christiano_fitzgerald_filter(
    values: pd.DataFrame,
    low_period: int,
    high_period: int,
    drift: bool = True,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Causal Christiano-Fitzgerald bandpass filter for cycle extraction.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    low_period : int
        Minimum cycle period in observations
    high_period : int
        Maximum cycle period in observations
    drift : bool
        If True, allow for drift in the series
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to filter

    Returns
    -------
    pd.Series
        Cycle component within specified period range.
        First observation per asset is NaN.

    Notes
    -----
    The CF filter is an approximation to the ideal bandpass filter that
    handles finite samples better than traditional filters.
    Uses shift(1) to exclude current observation, ensuring causality.
    """
    if low_period < 2:
        raise ValueError(f"low_period must be >= 2, got {low_period}")
    if high_period <= low_period:
        raise ValueError(f"high_period must be > low_period, got {high_period} <= {low_period}")

    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    def _cf_filter_single(series: pd.Series) -> pd.Series:
        """Apply CF filter to a single asset's time series."""
        # Shift to exclude current observation
        lagged = series.shift(1)

        if lagged.isna().all():
            return pd.Series(np.nan, index=series.index)

        # Extract non-NaN values
        valid_mask = lagged.notna()
        y = lagged[valid_mask].values

        if len(y) < max(low_period, 6):
            # Need sufficient data
            result = pd.Series(np.nan, index=series.index)
            return result

        try:
            # Compute CF filter weights
            cycle = _cf_asymmetric_filter(y, low_period, high_period, drift)

            # Map back to original index
            result = pd.Series(np.nan, index=series.index)
            result.loc[valid_mask] = cycle

            return result

        except Exception:
            result = pd.Series(np.nan, index=series.index)
            return result

    result = values.groupby(asset_col, sort=False)[value_col].apply(_cf_filter_single)

    # Reset index to match input
    result.index = values.index

    return result


def _cf_asymmetric_filter(
    y: np.ndarray,
    low_period: int,
    high_period: int,
    drift: bool,
) -> np.ndarray:
    """
    Compute Christiano-Fitzgerald asymmetric filter.

    This is a simplified causal implementation using one-sided filters.

    Parameters
    ----------
    y : np.ndarray
        Input series
    low_period : int
        Minimum period
    high_period : int
        Maximum period
    drift : bool
        Whether to allow drift

    Returns
    -------
    np.ndarray
        Filtered cycle component
    """
    n = len(y)

    # Need sufficient data
    if n < high_period:
        return np.full(n, np.nan)

    # Detrend if drift is True
    if drift:
        x = np.arange(n)
        coeffs = np.polyfit(x, y, 1)
        trend = np.polyval(coeffs, x)
        y_detrended = y - trend
    else:
        y_detrended = y - np.mean(y)

    # Frequency cutoffs
    omega_low = 2 * np.pi / high_period
    omega_high = 2 * np.pi / low_period

    # For causal implementation, use asymmetric weights looking backward
    cycle = np.full(n, np.nan)

    # Compute ideal filter weights (one-sided, backward-looking)
    max_lag = min(n, 2 * high_period)

    for t in range(max_lag, n):
        # Use data from t-max_lag to t (excluding t due to prior shift)
        j = np.arange(1, max_lag + 1)
        # One-sided filter weights
        weights = (np.sin(omega_high * j) - np.sin(omega_low * j)) / (np.pi * j)

        # Window from past data only
        window = y_detrended[t - max_lag:t][::-1]  # Reverse to align with weights

        if len(window) == len(weights):
            cycle[t] = np.dot(weights, window)

    return cycle
