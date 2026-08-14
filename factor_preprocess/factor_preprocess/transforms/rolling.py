"""
Rolling (time-series) transforms with explicit causality.

All rolling transforms:
- Exclude current observation (strict lag)
- Operate per-asset (isolated groups)
- Have explicit warmup periods
- Preserve NaN propagation
"""
import numpy as np
import pandas as pd
from typing import Optional


def _groupwise_lagged_stat(values, window, min_periods, asset_col, value_col, operation):
    """Apply a lagged window operation independently for each asset."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for positions in values.groupby(asset_col, sort=False).indices.values():
        positions = list(positions)
        lagged = values.iloc[positions][value_col].shift(1)
        result.iloc[positions] = operation(lagged, window, min_periods)
    return result


def rolling_mean(
    values: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Causal rolling mean per asset.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    window : int
        Number of periods to look back (excluding current)
    min_periods : int, optional
        Minimum observations required. Defaults to window.
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to transform

    Returns
    -------
    pd.Series
        Rolling mean, aligned with input index.
        First `window` observations per asset are NaN.

    Notes
    -----
    Uses shift(1) to exclude current observation, ensuring no future leakage.
    """
    if min_periods is None:
        min_periods = window

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    # Shift to exclude current observation, then rolling
    result = _groupwise_lagged_stat(
        values, window, min_periods, asset_col, value_col,
        lambda series, w, mp: series.rolling(window=w, min_periods=mp).mean().to_numpy(),
    )

    return result


def rolling_std(
    values: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    ddof: int = 1,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Causal rolling standard deviation per asset.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    window : int
        Number of periods to look back (excluding current)
    min_periods : int, optional
        Minimum observations required. Defaults to window.
    ddof : int
        Delta degrees of freedom
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to transform

    Returns
    -------
    pd.Series
        Rolling std, aligned with input index.
        First `window` observations per asset are NaN.
    """
    if min_periods is None:
        min_periods = window

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    # Shift to exclude current observation, then rolling
    result = _groupwise_lagged_stat(
        values, window, min_periods, asset_col, value_col,
        lambda series, w, mp: series.rolling(window=w, min_periods=mp).std(ddof=ddof).to_numpy(),
    )

    return result


def rolling_zscore(
    values: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    ddof: int = 1,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Causal rolling z-score normalization per asset.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    window : int
        Number of periods to look back (excluding current)
    min_periods : int, optional
        Minimum observations required. Defaults to window.
    ddof : int
        Delta degrees of freedom
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to transform

    Returns
    -------
    pd.Series
        Rolling z-score: (current - rolling_mean) / rolling_std.
        First `window` observations per asset are NaN.
        Zero std produces NaN.

    Notes
    -----
    Current observation is normalized against lagged statistics.
    """
    if min_periods is None:
        min_periods = window

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    grouped = values.groupby(asset_col, sort=False)[value_col]

    # Compute lagged statistics (excluding current)
    lagged = grouped.shift(1)
    rolling_mean_val = _groupwise_lagged_stat(
        values, window, min_periods, asset_col, value_col,
        lambda series, w, mp: series.rolling(window=w, min_periods=mp).mean().to_numpy(),
    )
    rolling_std_val = _groupwise_lagged_stat(
        values, window, min_periods, asset_col, value_col,
        lambda series, w, mp: series.rolling(window=w, min_periods=mp).std(ddof=ddof).to_numpy(),
    )

    # Z-score: (current - mean) / std
    current = values[value_col]
    result = (current - rolling_mean_val) / rolling_std_val

    return result


def ewma(
    values: pd.DataFrame,
    halflife: float,
    min_periods: int = 1,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Causal exponentially-weighted moving average per asset.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    halflife : float
        Halflife in units of observations
    min_periods : int
        Minimum observations required
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to transform

    Returns
    -------
    pd.Series
        EWMA, aligned with input index.

    Notes
    -----
    Uses shift(1) to exclude current observation.
    """
    if halflife <= 0:
        raise ValueError(f"halflife must be > 0, got {halflife}")

    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    # Shift to exclude current observation, then EWMA
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for positions in values.groupby(asset_col, sort=False).indices.values():
        positions = list(positions)
        result.iloc[positions] = (
            values.iloc[positions][value_col]
            .shift(1)
            .ewm(halflife=halflife, min_periods=min_periods, adjust=False)
            .mean()
            .to_numpy()
        )

    return result
