"""
Missingness transforms for handling missing data.

All transforms maintain temporal causality:
- Forward fill uses only past observations
- Missing indicators reflect current state
- Max lag prevents unbounded forward fill
"""
import numpy as np
import pandas as pd
from typing import Optional


def forward_fill(
    values: pd.DataFrame,
    max_lag: Optional[int] = None,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Forward fill missing values with bounded lag.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    max_lag : int, optional
        Maximum number of periods to forward fill.
        If None, forward fill without limit.
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to forward fill

    Returns
    -------
    pd.Series
        Forward-filled values.
        NaN if no prior value exists or max_lag exceeded.

    Notes
    -----
    Forward fill propagates the last valid observation forward.
    With max_lag, stops filling after max_lag consecutive NaNs.
    """
    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    grouped = values.groupby(asset_col, sort=False)[value_col]

    if max_lag is None:
        # Unbounded forward fill
        result = grouped.ffill()
    else:
        # Bounded forward fill
        if max_lag < 1:
            raise ValueError(f"max_lag must be >= 1, got {max_lag}")
        result = grouped.ffill(limit=max_lag)

    return result


def missing_indicator(
    values: pd.DataFrame,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Create binary indicator for missing values.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to check for missingness

    Returns
    -------
    pd.Series
        Binary indicator: 1.0 if missing, 0.0 if present.

    Notes
    -----
    Useful for creating features that capture missingness patterns.
    """
    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    result = pd.isna(values[value_col]).astype(np.float64)
    return result


def missing_run_length(
    values: pd.DataFrame,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Count consecutive missing observations up to current time.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to check for missingness

    Returns
    -------
    pd.Series
        Count of consecutive missing observations.
        0.0 if current value is present.

    Notes
    -----
    Tracks how long data has been missing.
    Resets to 0 when valid observation appears.
    """
    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    def compute_run_length(series: pd.Series) -> pd.Series:
        """Compute run length for a single asset."""
        is_missing = pd.isna(series)

        # Create groups of consecutive missing/not-missing
        # Each time is_missing changes, start a new group
        groups = (is_missing != is_missing.shift()).cumsum()

        # Within each group, compute cumulative count
        run_length = is_missing.groupby(groups).cumsum()

        # Convert to float
        return run_length.astype(np.float64)

    result = values.groupby(asset_col, sort=False)[value_col].transform(compute_run_length)
    return result


def missing_rate(
    values: pd.DataFrame,
    window: int,
    min_periods: int = 1,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Compute rolling missing rate from lagged observations.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    window : int
        Number of periods to look back (excluding current)
    min_periods : int
        Minimum observations required
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to check for missingness

    Returns
    -------
    pd.Series
        Fraction of missing values in rolling window [0, 1].
        Computed from lagged observations (excluding current).

    Notes
    -----
    Uses shift(1) to exclude current observation.
    Measures historical missingness, not current state.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    # Shift and roll inside each asset so interleaved rows cannot contaminate
    # another asset's history. transform preserves the input index and order.
    is_missing = pd.isna(values[value_col]).astype(np.float64)
    result = is_missing.groupby(values[asset_col], sort=False).transform(
        lambda series: series.shift(1).rolling(
            window=window,
            min_periods=min_periods,
        ).mean()
    )

    return result


def linear_interpolate(
    values: pd.DataFrame,
    max_gap: Optional[int] = None,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Linear interpolation with bounded gap size.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    max_gap : int, optional
        Maximum gap size to interpolate.
        If None, interpolate all gaps.
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to interpolate

    Returns
    -------
    pd.Series
        Linearly interpolated values.
        Gaps larger than max_gap remain NaN.

    Notes
    -----
    Interpolates linearly between valid observations.
    With max_gap, only fills gaps of max_gap or fewer consecutive NaNs.
    Does not extrapolate beyond first/last valid observation.
    """
    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    def interpolate_group(series: pd.Series) -> pd.Series:
        """Interpolate a single asset's series."""
        if max_gap is None:
            # Interpolate all gaps
            return series.interpolate(method='linear', limit_area='inside')
        else:
            # Interpolate with gap limit
            if max_gap < 1:
                raise ValueError(f"max_gap must be >= 1, got {max_gap}")
            return series.interpolate(method='linear', limit=max_gap, limit_area='inside')

    result = values.groupby(asset_col, sort=False)[value_col].transform(interpolate_group)
    result.name = value_col
    return result


def time_weighted_interpolate(
    values: pd.DataFrame,
    max_gap: Optional[int] = None,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Time-weighted interpolation respecting actual time distances.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    max_gap : int, optional
        Maximum gap size (in number of observations) to interpolate.
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (must be datetime)
    value_col : str
        Value column to interpolate

    Returns
    -------
    pd.Series
        Time-weighted interpolated values.
        Gaps larger than max_gap remain NaN.

    Notes
    -----
    Uses actual time distances for interpolation weights.
    More accurate than linear interpolation when time spacing is irregular.
    """
    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    def interpolate_group(group: pd.DataFrame) -> pd.DataFrame:
        """Time-weighted interpolation for a single asset."""
        if max_gap is not None and max_gap < 1:
            raise ValueError(f"max_gap must be >= 1, got {max_gap}")

        # Extract the value and time columns
        val_series = group[value_col] if value_col in group.columns else group
        time_series = group[time_col] if time_col in group.columns else group.index

        # Create temporary dataframe with time index
        temp_df = pd.DataFrame({value_col: val_series.values}, index=time_series.values)

        # Interpolate using time index
        if max_gap is None:
            interpolated = temp_df[value_col].interpolate(method='time', limit_area='inside')
        else:
            interpolated = temp_df[value_col].interpolate(method='time', limit=max_gap, limit_area='inside')

        # Return DataFrame to preserve structure
        return pd.DataFrame({value_col: interpolated.values}, index=group.index)

    result = values.groupby(asset_col, sort=False, group_keys=False).apply(interpolate_group)

    # Extract the value column
    result = result[value_col]
    result.name = value_col
    return result


def impute_with_fallback(
    values: pd.DataFrame,
    fallback_strategy: str = "forward_fill",
    max_lag: Optional[int] = None,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Impute missing values with fallback strategy hierarchy.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    fallback_strategy : str
        Strategy for remaining NaNs after forward fill.
        Options: "forward_fill", "zero", "median", "mean"
    max_lag : int, optional
        Maximum periods for forward fill. If None, unlimited.
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to impute

    Returns
    -------
    pd.Series
        Imputed values with fallback applied.

    Notes
    -----
    First tries forward fill (with max_lag if specified).
    Then applies fallback strategy to remaining NaNs.
    Median/mean computed per-asset from available data.
    """
    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    # Start with forward fill if max_lag is provided and > 0
    if max_lag is not None and max_lag > 0:
        result = forward_fill(values, max_lag=max_lag, asset_col=asset_col, time_col=time_col, value_col=value_col)
    elif max_lag is None:
        result = forward_fill(values, max_lag=None, asset_col=asset_col, time_col=time_col, value_col=value_col)
    else:
        # max_lag is 0, skip forward fill
        result = values[value_col].copy()

    # Apply fallback strategy to remaining NaNs
    if fallback_strategy == "forward_fill":
        # Already done, no additional fallback
        pass
    elif fallback_strategy == "zero":
        result = result.fillna(0.0)
    elif fallback_strategy == "median":
        # Fill with per-asset median
        medians = values.groupby(asset_col, sort=False)[value_col].transform('median')
        result = result.fillna(medians)
    elif fallback_strategy == "mean":
        # Fill with per-asset mean
        means = values.groupby(asset_col, sort=False)[value_col].transform('mean')
        result = result.fillna(means)
    else:
        raise ValueError(f"Unknown fallback_strategy: {fallback_strategy}. "
                        f"Options: 'forward_fill', 'zero', 'median', 'mean'")

    result.name = value_col
    return result
