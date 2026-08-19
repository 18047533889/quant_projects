"""
Seasonal decomposition using STL (Seasonal and Trend decomposition using Loess).

All functions are causal - they exclude the current observation and operate
per-asset to ensure no future leakage.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, NamedTuple
from statsmodels.tsa.seasonal import STL as STL_statsmodels

# Narrowed catch set for STL's degenerate-but-valid-input failures.
# Programming errors (TypeError, AttributeError, bad column names) must
# propagate instead of being silently NaN'd.
_NUMERICAL_FAILURES = (ValueError, IndexError, np.linalg.LinAlgError, ArithmeticError)


class STLResult(NamedTuple):
    """Container for STL decomposition results."""
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series


def stl_decompose(
    values: pd.DataFrame,
    period: int,
    seasonal: int = 7,
    trend: Optional[int] = None,
    low_pass: Optional[int] = None,
    seasonal_deg: int = 1,
    trend_deg: int = 1,
    low_pass_deg: int = 1,
    robust: bool = False,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Causal STL (Seasonal-Trend decomposition using Loess) decomposition.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    period : int
        Periodicity of the seasonal component (e.g., 12 for monthly data with yearly seasonality)
    seasonal : int
        Length of the seasonal smoother. Must be odd.
    trend : int, optional
        Length of the trend smoother. Must be odd.
        If None, defaults to smallest odd integer >= 1.5 * period / (1 - 1.5/seasonal)
    low_pass : int, optional
        Length of the low-pass filter. Must be odd.
        If None, defaults to smallest odd integer > period
    seasonal_deg : int
        Degree of seasonal LOESS smoothing (0 or 1)
    trend_deg : int
        Degree of trend LOESS smoothing (0 or 1)
    low_pass_deg : int
        Degree of low-pass LOESS smoothing (0 or 1)
    robust : bool
        If True, use robust fitting (resistant to outliers)
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to decompose

    Returns
    -------
    trend : pd.Series
        Trend component
    seasonal : pd.Series
        Seasonal component
    residual : pd.Series
        Residual component

    Notes
    -----
    Uses shift(1) to exclude current observation, ensuring causality.
    Requires at least 2 * period observations per asset.
    First observation per asset is NaN (causal lag).
    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")

    if seasonal % 2 == 0:
        raise ValueError(f"seasonal must be odd, got {seasonal}")

    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    def _stl_single(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Apply STL to a single asset's time series."""
        # Shift to exclude current observation
        lagged = series.shift(1)

        if lagged.isna().all():
            na_series = pd.Series(np.nan, index=series.index)
            return na_series.copy(), na_series.copy(), na_series.copy()

        # Need sufficient data for STL
        valid_count = lagged.notna().sum()
        if valid_count < 2 * period:
            na_series = pd.Series(np.nan, index=series.index)
            return na_series.copy(), na_series.copy(), na_series.copy()

        # STL requires no NaN values - forward fill for gaps
        filled = lagged.ffill()

        if filled.isna().any():
            # Still have NaN at the start
            na_series = pd.Series(np.nan, index=series.index)
            return na_series.copy(), na_series.copy(), na_series.copy()

        try:
            stl = STL_statsmodels(
                filled,
                period=period,
                seasonal=seasonal,
                trend=trend,
                low_pass=low_pass,
                seasonal_deg=seasonal_deg,
                trend_deg=trend_deg,
                low_pass_deg=low_pass_deg,
                robust=robust,
            )
            result = stl.fit()

            trend_out = pd.Series(result.trend, index=series.index)
            seasonal_out = pd.Series(result.seasonal, index=series.index)
            residual_out = pd.Series(result.resid, index=series.index)

            # Restore NaN where original lagged data was NaN
            nan_mask = lagged.isna()
            trend_out[nan_mask] = np.nan
            seasonal_out[nan_mask] = np.nan
            residual_out[nan_mask] = np.nan

            return trend_out, seasonal_out, residual_out

        except _NUMERICAL_FAILURES:
            # STL may fail on degenerate-but-valid input (constant series,
            # insufficient non-NaN observations).  Keep the NaN
            # fail-closed contract for these numerical failures; do not
            # swallow programming errors that indicate a wiring bug.
            na_series = pd.Series(np.nan, index=series.index)
            return na_series.copy(), na_series.copy(), na_series.copy()

    # Apply per asset
    results = []
    for asset_id, group in values.groupby(asset_col, sort=False):
        t, s, r = _stl_single(group[value_col])
        results.append(pd.DataFrame({
            'trend': t,
            'seasonal': s,
            'residual': r,
        }, index=group.index))

    combined = pd.concat(results)

    # Ensure order matches input
    combined = combined.loc[values.index]

    return combined['trend'], combined['seasonal'], combined['residual']


def seasonal_component(
    values: pd.DataFrame,
    period: int,
    seasonal: int = 7,
    trend: Optional[int] = None,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Extract seasonal component using STL.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    period : int
        Periodicity of the seasonal component
    seasonal : int
        Length of the seasonal smoother. Must be odd.
    trend : int, optional
        Length of the trend smoother
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to decompose

    Returns
    -------
    pd.Series
        Seasonal component only
    """
    _, seasonal_comp, _ = stl_decompose(
        values,
        period=period,
        seasonal=seasonal,
        trend=trend,
        asset_col=asset_col,
        time_col=time_col,
        value_col=value_col,
    )
    return seasonal_comp


def trend_component(
    values: pd.DataFrame,
    period: int,
    seasonal: int = 7,
    trend: Optional[int] = None,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Extract trend component using STL.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    period : int
        Periodicity of the seasonal component
    seasonal : int
        Length of the seasonal smoother. Must be odd.
    trend : int, optional
        Length of the trend smoother
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to decompose

    Returns
    -------
    pd.Series
        Trend component only
    """
    trend_comp, _, _ = stl_decompose(
        values,
        period=period,
        seasonal=seasonal,
        trend=trend,
        asset_col=asset_col,
        time_col=time_col,
        value_col=value_col,
    )
    return trend_comp


def residual_component(
    values: pd.DataFrame,
    period: int,
    seasonal: int = 7,
    trend: Optional[int] = None,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Extract residual component using STL.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    period : int
        Periodicity of the seasonal component
    seasonal : int
        Length of the seasonal smoother. Must be odd.
    trend : int, optional
        Length of the trend smoother
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to decompose

    Returns
    -------
    pd.Series
        Residual component only
    """
    _, _, residual_comp = stl_decompose(
        values,
        period=period,
        seasonal=seasonal,
        trend=trend,
        asset_col=asset_col,
        time_col=time_col,
        value_col=value_col,
    )
    return residual_comp
