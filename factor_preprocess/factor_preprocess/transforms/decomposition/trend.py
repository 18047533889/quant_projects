"""
Trend extraction using Hodrick-Prescott filter.

All functions are causal - they exclude the current observation and operate
per-asset to ensure no future leakage.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy import sparse
from scipy.sparse.linalg import spsolve


def hp_filter(
    values: pd.DataFrame,
    lambda_param: float = 1600.0,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Causal Hodrick-Prescott filter for trend extraction.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    lambda_param : float
        Smoothing parameter. Higher values produce smoother trends.
        - 100 for annual data
        - 1600 for quarterly data (default)
        - 14400 for monthly data
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to decompose

    Returns
    -------
    pd.Series
        Trend component, aligned with input index.
        First observation per asset is NaN (causal lag).

    Notes
    -----
    Uses shift(1) to exclude current observation, ensuring causality.
    The HP filter minimizes: sum((y_t - tau_t)^2) + lambda * sum((tau_t+1 - 2*tau_t + tau_t-1)^2)
    where tau is the trend component.
    """
    if lambda_param <= 0:
        raise ValueError(f"lambda_param must be > 0, got {lambda_param}")

    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    def _hp_filter_single(series: pd.Series) -> pd.Series:
        """Apply HP filter to a single asset's time series."""
        # Shift to exclude current observation
        lagged = series.shift(1)

        if lagged.isna().all():
            return pd.Series(np.nan, index=series.index)

        # Extract non-NaN values for filtering
        valid_mask = lagged.notna()
        y = lagged[valid_mask].values

        if len(y) < 3:
            # Need at least 3 points for second-order differences
            result = pd.Series(np.nan, index=series.index)
            return result

        n = len(y)

        # Build second-order difference matrix
        # D is (n-2) x n matrix for second differences
        diag_data = np.array([
            np.ones(n-2),
            -2 * np.ones(n-2),
            np.ones(n-2)
        ])
        D = sparse.diags(diag_data, [0, 1, 2], shape=(n-2, n))

        # Solve: (I + lambda * D^T * D) * trend = y
        I = sparse.eye(n)
        A = I + lambda_param * D.T @ D
        trend_values = spsolve(A, y)

        # Map back to original index
        result = pd.Series(np.nan, index=series.index)
        result.loc[valid_mask] = trend_values

        return result

    result = values.groupby(asset_col, sort=False)[value_col].apply(_hp_filter_single)

    # Realign by label, not position — see decomposition.cycle.bandpass_filter
    # for why a positional index reassignment mislabels interleaved layouts.
    if isinstance(result.index, pd.MultiIndex):
        result = result.droplevel(0)
    result = result.loc[values.index].copy()
    result.index = values.index

    return result


def hp_decompose(
    values: pd.DataFrame,
    lambda_param: float = 1600.0,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> Tuple[pd.Series, pd.Series]:
    """
    Causal Hodrick-Prescott decomposition into trend and cycle.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    lambda_param : float
        Smoothing parameter. Higher values produce smoother trends.
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to decompose

    Returns
    -------
    trend : pd.Series
        Trend component (low-frequency)
    cycle : pd.Series
        Cyclical component (high-frequency deviation from trend)

    Notes
    -----
    Decomposition: y_t = trend_t + cycle_t
    First observation per asset is NaN for both components.
    """
    trend = hp_filter(
        values,
        lambda_param=lambda_param,
        asset_col=asset_col,
        time_col=time_col,
        value_col=value_col,
    )

    # Cycle is the residual: y - trend
    # Since trend is lagged by 1, we need to compute cycle from lagged y
    lagged_y = values.groupby(asset_col, sort=False)[value_col].shift(1)
    cycle = lagged_y - trend

    return trend, cycle
