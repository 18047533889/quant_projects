"""
Volatility scaling transforms with explicit causality.

All transforms exclude current observation to prevent future leakage.
Volatility is computed from lagged returns/values.
"""
import numpy as np
import pandas as pd
from typing import Optional


def volatility_scale(
    values: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    ddof: int = 1,
    target_vol: float = 1.0,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Scale values by lagged volatility to target volatility.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    window : int
        Number of periods to compute volatility (excluding current)
    min_periods : int, optional
        Minimum observations required. Defaults to window.
    ddof : int
        Delta degrees of freedom for std calculation
    target_vol : float
        Target volatility level
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column to transform

    Returns
    -------
    pd.Series
        Volatility-scaled values: value * (target_vol / lagged_vol).
        First `window` observations per asset are NaN.
        Zero volatility produces NaN.

    Notes
    -----
    Volatility is computed from lagged observations (shift(1) + rolling),
    ensuring no future leakage. Current value is scaled by lagged vol.
    """
    if min_periods is None:
        min_periods = window

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    if target_vol <= 0:
        raise ValueError(f"target_vol must be > 0, got {target_vol}")

    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    # Compute lagged volatility independently within each asset. Applying
    # rolling directly to the shifted, interleaved Series would let one
    # asset's observations enter another asset's window.
    lagged_vol = values.groupby(asset_col, sort=False)[value_col].transform(
        lambda series: series.shift(1).rolling(
            window=window,
            min_periods=min_periods,
        ).std(ddof=ddof)
    )

    # The public contract specifies NaN for a zero-volatility history.
    lagged_vol = lagged_vol.mask(lagged_vol == 0.0)
    current = values[value_col]
    result = current * (target_vol / lagged_vol)

    return result


def volatility_scale_returns(
    returns: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    ddof: int = 1,
    target_vol: float = 0.01,
    asset_col: str = "asset_id",
    time_col: str = "date",
    return_col: str = "return",
) -> pd.Series:
    """
    Scale returns by lagged return volatility.

    Parameters
    ----------
    returns : pd.DataFrame
        Must have columns: [asset_col, time_col, return_col]
        Must be sorted by [asset_col, time_col]
    window : int
        Number of periods to compute return volatility (excluding current)
    min_periods : int, optional
        Minimum observations required. Defaults to window.
    ddof : int
        Delta degrees of freedom for std calculation
    target_vol : float
        Target return volatility (e.g., 0.01 for 1% daily)
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    return_col : str
        Return column to transform

    Returns
    -------
    pd.Series
        Volatility-scaled returns: return * (target_vol / lagged_vol).
        First `window` observations per asset are NaN.

    Notes
    -----
    Typical use: normalize factor returns to constant volatility.
    Volatility computed from lagged returns only (no future leakage).
    """
    return volatility_scale(
        returns,
        window=window,
        min_periods=min_periods,
        ddof=ddof,
        target_vol=target_vol,
        asset_col=asset_col,
        time_col=time_col,
        value_col=return_col,
    )


def realized_volatility(
    values: pd.DataFrame,
    window: int,
    min_periods: Optional[int] = None,
    ddof: int = 1,
    annualization_factor: float = 1.0,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Compute realized volatility from lagged observations.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    window : int
        Number of periods to compute volatility (excluding current)
    min_periods : int, optional
        Minimum observations required. Defaults to window.
    ddof : int
        Delta degrees of freedom
    annualization_factor : float
        Scaling factor for annualization (e.g., sqrt(252) for daily returns)
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column (typically returns)

    Returns
    -------
    pd.Series
        Realized volatility computed from lagged observations.
        First `window` observations per asset are NaN.

    Notes
    -----
    Uses shift(1) to exclude current observation.
    Result represents volatility known at time t (computed from t-window to t-1).
    """
    if min_periods is None:
        min_periods = window

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    # Compute volatility from lagged observations within each asset.
    vol = values.groupby(asset_col, sort=False)[value_col].transform(
        lambda series: series.shift(1).rolling(
            window=window,
            min_periods=min_periods,
        ).std(ddof=ddof)
    )

    # Apply annualization factor
    result = vol * annualization_factor

    return result


def ewma_volatility(
    values: pd.DataFrame,
    halflife: float,
    min_periods: int = 1,
    annualization_factor: float = 1.0,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Compute EWMA volatility from lagged observations.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    halflife : float
        Halflife in units of observations for exponential weighting
    min_periods : int
        Minimum observations required
    annualization_factor : float
        Scaling factor for annualization (e.g., sqrt(252) for daily returns)
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column (typically returns or residuals)

    Returns
    -------
    pd.Series
        EWMA volatility computed from lagged observations.

    Notes
    -----
    Computes exponentially weighted standard deviation using lagged observations.
    Uses shift(1) to exclude current observation, preventing future leakage.
    More responsive to recent volatility changes than simple rolling volatility.
    """
    if halflife <= 0:
        raise ValueError(f"halflife must be > 0, got {halflife}")

    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    def ewma_vol_group(series: pd.Series) -> pd.Series:
        """Compute EWMA volatility for a single asset."""
        lagged = series.shift(1)
        vol = lagged.ewm(halflife=halflife, min_periods=min_periods, adjust=False).std()
        return vol * annualization_factor

    result = values.groupby(asset_col, sort=False)[value_col].transform(ewma_vol_group)
    result.name = value_col
    return result


def garch_inspired_volatility(
    values: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
    min_periods: Optional[int] = None,
    ddof: int = 1,
    annualization_factor: float = 1.0,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> pd.Series:
    """
    Compute GARCH-inspired rolling volatility with short-term and long-term components.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    short_window : int
        Window for short-term volatility (captures recent volatility spikes)
    long_window : int
        Window for long-term volatility (captures baseline volatility)
    min_periods : int, optional
        Minimum observations required. Defaults to short_window.
    ddof : int
        Delta degrees of freedom
    annualization_factor : float
        Scaling factor for annualization
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column (typically returns)

    Returns
    -------
    pd.Series
        Combined volatility: 0.7 * short_term + 0.3 * long_term.
        Captures both recent volatility shocks and baseline volatility.

    Notes
    -----
    Inspired by GARCH models which combine recent shocks with long-term variance.
    Uses shift(1) to exclude current observation from all computations.
    The 70/30 weighting emphasizes recent volatility while maintaining stability.
    """
    if min_periods is None:
        min_periods = short_window

    if short_window < 1:
        raise ValueError(f"short_window must be >= 1, got {short_window}")
    if long_window < short_window:
        raise ValueError(f"long_window must be >= short_window, got {long_window} < {short_window}")

    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    grouped = values.groupby(asset_col, sort=False)[value_col]

    # Both components restart at every asset boundary; transform retains the
    # caller's original row index and order, including interleaved assets.
    short_vol = grouped.transform(
        lambda series: series.shift(1).rolling(
            window=short_window, min_periods=min_periods
        ).std(ddof=ddof)
    )
    long_vol = grouped.transform(
        lambda series: series.shift(1).rolling(
            window=long_window, min_periods=long_window
        ).std(ddof=ddof)
    )

    # Combine: weight recent volatility more heavily
    combined_vol = 0.7 * short_vol + 0.3 * long_vol

    # Apply annualization factor
    result = combined_vol * annualization_factor

    return result
