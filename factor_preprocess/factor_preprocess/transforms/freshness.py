"""
Data freshness transforms for tracking observation age and staleness.

All transforms maintain temporal causality and track how long since
the last valid observation or how old data is at decision time.
"""
import numpy as np
import pandas as pd
from typing import Optional


def days_since_update(
    values: pd.DataFrame,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Count days since last valid observation for each asset.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (must be datetime)
    value_col : str
        Value column to track freshness

    Returns
    -------
    pd.Series
        Days since last valid observation.
        0.0 if current value is valid.
        NaN if no prior valid observation exists.

    Notes
    -----
    Useful for detecting stale data and data quality issues.
    Counts calendar days, not trading days.
    """
    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    def compute_days_since(group: pd.DataFrame) -> pd.Series:
        """Compute days since last update for a single asset."""
        dates = group[time_col].values
        vals = group[value_col].values

        # Find last valid date for each observation
        result = np.full(len(group), np.nan, dtype=np.float64)
        current_valid_date = None

        for i in range(len(group)):
            if pd.notna(vals[i]):
                current_valid_date = dates[i]
                result[i] = 0.0
            else:
                if current_valid_date is not None:
                    days_diff = (dates[i] - current_valid_date) / np.timedelta64(1, 'D')
                    result[i] = float(days_diff)
                else:
                    result[i] = np.nan

        return pd.Series(result, index=group.index)

    result = values.groupby(asset_col, sort=False, group_keys=False)[value_col].transform(
        lambda x: compute_days_since(values.loc[x.index])
    )
    return result


def observation_age(
    values: pd.DataFrame,
    observation_date_col: str,
    current_date_col: str,
    asset_col: str = "asset_id",
    time_col: str = "date",
) -> pd.Series:
    """
    Compute age of observation relative to decision time.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, observation_date_col, current_date_col]
        Must be sorted by [asset_col, time_col]
    observation_date_col : str
        Column containing the observation/publication date
    current_date_col : str
        Column containing the current decision date
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)

    Returns
    -------
    pd.Series
        Days between observation date and current date.
        Positive values indicate observation is from the past.
        NaN if either date is missing.

    Notes
    -----
    Typical use: track point-in-time lag for fundamental data.
    observation_date might be earnings announcement date,
    current_date is the decision/trading date.
    """
    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    obs_dates = pd.to_datetime(values[observation_date_col])
    curr_dates = pd.to_datetime(values[current_date_col])

    # Compute difference in days
    result = (curr_dates - obs_dates).dt.days.astype(np.float64)

    return result


def freshness_score(
    values: pd.DataFrame,
    halflife_days: float,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Compute exponential freshness score based on data age.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    halflife_days : float
        Halflife for exponential decay (in days)
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (must be datetime)
    value_col : str
        Value column to track freshness

    Returns
    -------
    pd.Series
        Freshness score in [0, 1].
        1.0 for current valid observation.
        Decays exponentially with age: exp(-days * ln(2) / halflife).
        0.0 if no valid observation exists.

    Notes
    -----
    Useful for weighting factors by data freshness.
    Score of 0.5 at halflife_days ago.
    """
    if halflife_days <= 0:
        raise ValueError(f"halflife_days must be > 0, got {halflife_days}")

    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    # Compute days since last update
    days_since = days_since_update(values, asset_col=asset_col, time_col=time_col, value_col=value_col)

    # Exponential decay: exp(-days * ln(2) / halflife)
    decay_rate = np.log(2) / halflife_days
    freshness = np.exp(-days_since * decay_rate)

    # Handle NaN (no prior valid observation) -> 0.0 freshness
    freshness = freshness.fillna(0.0)

    return freshness


def stale_data_indicator(
    values: pd.DataFrame,
    max_days: int,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Binary indicator for stale data.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    max_days : int
        Maximum acceptable age in days
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (must be datetime)
    value_col : str
        Value column to check staleness

    Returns
    -------
    pd.Series
        Binary indicator: 1.0 if data is stale (age > max_days), 0.0 otherwise.
        NaN if no valid observation exists.

    Notes
    -----
    Useful for filtering out stale data or creating data quality flags.
    """
    if max_days < 0:
        raise ValueError(f"max_days must be >= 0, got {max_days}")

    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    # Compute days since last update
    days_since = days_since_update(values, asset_col=asset_col, time_col=time_col, value_col=value_col)

    # Create binary indicator
    result = (days_since > max_days).astype(np.float64)

    # Preserve NaN where days_since is NaN
    result = result.where(pd.notna(days_since), np.nan)

    return result
