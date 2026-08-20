"""
Online regime detection for volatility and correlation regimes.

All detectors are causal: state at time t uses only data up to t-1.
Regime transitions are based on rolling statistics with explicit lags.
"""
import numpy as np
import pandas as pd
from typing import Optional, Literal, NamedTuple
from dataclasses import dataclass


class RegimeState(NamedTuple):
    """
    Regime classification result.

    Attributes
    ----------
    regime : pd.Series
        Integer regime labels (0, 1, 2, ...) aligned with input index.
        NaN during warmup period.
    regime_strength : pd.Series
        Continuous measure of regime intensity [0, 1].
        Higher = more confident in current regime.
    transition_flag : pd.Series
        Boolean flag indicating regime change (current != previous).
    """
    regime: pd.Series
    regime_strength: pd.Series
    transition_flag: pd.Series


def detect_variance_regime(
    values: pd.DataFrame,
    window: int,
    n_regimes: int = 2,
    percentiles: Optional[list[float]] = None,
    min_periods: Optional[int] = None,
    asset_col: str = "asset_id",
    time_col: str = "date",
    value_col: str = "value",
) -> RegimeState:
    """
    Detect variance regimes using rolling realized volatility.

    Parameters
    ----------
    values : pd.DataFrame
        Must have columns: [asset_col, time_col, value_col]
        Must be sorted by [asset_col, time_col]
    window : int
        Rolling window for volatility estimation (excluding current)
    n_regimes : int
        Number of regimes (2 = low/high, 3 = low/medium/high)
    percentiles : list[float], optional
        Custom percentiles for regime boundaries [0, 1].
        Defaults to equal-sized bins based on n_regimes.
    min_periods : int, optional
        Minimum observations for volatility calculation. Defaults to window.
    asset_col : str
        Asset identifier column
    time_col : str
        Time column (for sorting verification)
    value_col : str
        Value column (typically returns or factor values)

    Returns
    -------
    RegimeState
        Regime classification with integer labels, strength, and transition flags.
        regime=0 is lowest volatility, regime=(n_regimes-1) is highest.

    Notes
    -----
    - Uses shift(1) to exclude current observation from volatility calculation
    - Cross-sectional percentiles computed at each time point
    - Regime strength = distance from nearest boundary, normalized
    - First `window` observations per asset are NaN

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'asset_id': ['A', 'A', 'A', 'B', 'B', 'B'],
    ...     'date': pd.date_range('2020-01-01', periods=3).tolist() * 2,
    ...     'value': [0.01, 0.05, 0.02, 0.02, 0.01, 0.03]
    ... })
    >>> state = detect_variance_regime(df, window=20, n_regimes=2)
    >>> state.regime  # 0 = low vol, 1 = high vol
    """
    if n_regimes < 2:
        raise ValueError(f"n_regimes must be >= 2, got {n_regimes}")

    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    if min_periods is None:
        min_periods = window

    # Verify sort order
    if not values.groupby(asset_col, sort=False)[time_col].is_monotonic_increasing.all():
        raise ValueError("DataFrame must be sorted by [asset_col, time_col]")

    # Default percentiles: equal-sized bins
    if percentiles is None:
        percentiles = [i / n_regimes for i in range(1, n_regimes)]

    if len(percentiles) != n_regimes - 1:
        raise ValueError(
            f"percentiles must have {n_regimes - 1} values for {n_regimes} regimes, "
            f"got {len(percentiles)}"
        )

    # Compute rolling realized volatility (causal)
    # Must apply rolling within each group to avoid cross-asset contamination
    realized_vol = (
        values.groupby(asset_col, sort=False)[value_col]
        .shift(1)  # Exclude current observation
        .groupby(values[asset_col], sort=False)  # Re-group after shift
        .rolling(window=window, min_periods=min_periods)
        .std()
        .reset_index(level=0, drop=True)  # Drop the groupby level from index
    )

    # Build DataFrame for cross-sectional operations
    temp_df = values[[asset_col, time_col]].copy()
    temp_df['realized_vol'] = realized_vol

    # Compute cross-sectional percentile boundaries at each time
    def assign_regime(group):
        """Assign regime labels based on cross-sectional percentiles."""
        vol = group['realized_vol'].values

        # Handle all-NaN case
        if np.all(np.isnan(vol)):
            return pd.DataFrame({
                'regime': np.full(len(vol), np.nan),
                'regime_strength': np.full(len(vol), np.nan),
            }, index=group.index)

        # Compute percentile boundaries (ignoring NaN)
        boundaries = np.nanquantile(vol, percentiles)

        # Assign regime: digitize returns bin indices starting from 1
        # We want 0-indexed, so subtract 1
        regime_labels = np.digitize(vol, boundaries, right=False)

        # Compute regime strength: distance from nearest boundary, normalized
        strength = np.full(len(vol), np.nan)
        for i, v in enumerate(vol):
            if np.isnan(v):
                continue

            # Find nearest boundaries
            if regime_labels[i] == 0:
                # Lowest regime: distance from upper boundary
                upper = boundaries[0]
                strength[i] = max(0.0, (upper - v) / (upper + 1e-9))
            elif regime_labels[i] == n_regimes - 1:
                # Highest regime: distance from lower boundary
                lower = boundaries[-1]
                strength[i] = max(0.0, (v - lower) / (lower + 1e-9))
            else:
                # Middle regime: distance from both boundaries
                lower = boundaries[regime_labels[i] - 1]
                upper = boundaries[regime_labels[i]]
                mid = (lower + upper) / 2
                dist_from_mid = abs(v - mid)
                half_width = (upper - lower) / 2
                strength[i] = max(0.0, 1.0 - dist_from_mid / (half_width + 1e-9))

            # Clip to [0, 1]
            strength[i] = np.clip(strength[i], 0.0, 1.0)

        return pd.DataFrame({
            'regime': regime_labels,
            'regime_strength': strength,
        }, index=group.index)

    result_df = temp_df.groupby(time_col, group_keys=False).apply(assign_regime, include_groups=False)

    # Detect transitions
    transition_flag = (
        result_df.groupby(values[asset_col])['regime']
        .diff()
        .ne(0)
        .fillna(False)
    )

    return RegimeState(
        regime=result_df['regime'],
        regime_strength=result_df['regime_strength'],
        transition_flag=transition_flag,
    )


def detect_correlation_regime(
    values: pd.DataFrame,
    window: int,
    n_regimes: int = 2,
    percentiles: Optional[list[float]] = None,
    min_periods: Optional[int] = None,
    time_col: str = "date",
    value_cols: Optional[list[str]] = None,
) -> RegimeState:
    """
    Detect correlation regimes using rolling average pairwise correlation.

    Parameters
    ----------
    values : pd.DataFrame
        Wide-format DataFrame with time_col and multiple value columns,
        OR long-format with pivot to wide using value_cols.
        Must be sorted by time_col.
    window : int
        Rolling window for correlation estimation (excluding current)
    n_regimes : int
        Number of regimes (2 = low/high correlation, 3 = low/medium/high)
    percentiles : list[float], optional
        Custom percentiles for regime boundaries.
    min_periods : int, optional
        Minimum observations for correlation calculation.
    time_col : str
        Time column (for sorting and grouping)
    value_cols : list[str], optional
        Value columns to compute correlation across.
        If None, uses all numeric columns except time_col.

    Returns
    -------
    RegimeState
        Regime classification at each time point.
        regime=0 is lowest correlation, regime=(n_regimes-1) is highest.

    Notes
    -----
    - Computes average pairwise correlation across all value_cols at each time
    - Uses shift(1) to exclude current observation
    - Regime is constant across all assets/factors at each time point
    - First `window` observations are NaN

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2020-01-01', periods=100),
    ...     'factor1': np.random.randn(100),
    ...     'factor2': np.random.randn(100),
    ...     'factor3': np.random.randn(100),
    ... })
    >>> state = detect_correlation_regime(df, window=20, n_regimes=2)
    >>> state.regime  # 0 = low correlation regime, 1 = high correlation
    """
    if n_regimes < 2:
        raise ValueError(f"n_regimes must be >= 2, got {n_regimes}")

    if window < 2:
        raise ValueError(f"window must be >= 2 for correlation, got {window}")

    if min_periods is None:
        min_periods = window

    # Verify sort order
    if not values[time_col].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by time_col")

    # Select value columns
    if value_cols is None:
        value_cols = [c for c in values.columns if c != time_col and pd.api.types.is_numeric_dtype(values[c])]

    if len(value_cols) < 2:
        raise ValueError(f"Need at least 2 value columns for correlation, got {len(value_cols)}")

    # Default percentiles
    if percentiles is None:
        percentiles = [i / n_regimes for i in range(1, n_regimes)]

    if len(percentiles) != n_regimes - 1:
        raise ValueError(
            f"percentiles must have {n_regimes - 1} values for {n_regimes} regimes"
        )

    # Extract value matrix
    value_matrix = values[value_cols].values

    # Compute rolling average pairwise correlation (causal)
    avg_corr = np.full(len(values), np.nan)

    for i in range(window, len(values)):
        # Window from [i - window, i), excluding current observation at i
        window_data = value_matrix[i - window:i, :]

        # Check sufficient data
        n_valid = np.sum(~np.isnan(window_data), axis=0)
        if np.any(n_valid < min_periods):
            continue

        # Compute correlation matrix
        corr_matrix = np.corrcoef(window_data, rowvar=False)

        # Average off-diagonal elements
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        avg_corr[i] = np.nanmean(corr_matrix[mask])

    # Compute percentile boundaries over time
    boundaries = np.nanquantile(avg_corr, percentiles)

    # Assign regime labels (preserving NaN)
    regime_labels = np.full(len(values), np.nan)
    valid_mask = np.isfinite(avg_corr)
    regime_labels[valid_mask] = np.digitize(avg_corr[valid_mask], boundaries, right=False)

    # Compute regime strength
    strength = np.full(len(values), np.nan)
    for i, v in enumerate(avg_corr):
        if np.isnan(v):
            continue

        if regime_labels[i] == 0:
            upper = boundaries[0]
            strength[i] = max(0.0, (upper - v) / (upper + 1e-9))
        elif regime_labels[i] == n_regimes - 1:
            lower = boundaries[-1]
            strength[i] = max(0.0, (v - lower) / (lower + 1e-9))
        else:
            lower = boundaries[regime_labels[i] - 1]
            upper = boundaries[regime_labels[i]]
            mid = (lower + upper) / 2
            dist_from_mid = abs(v - mid)
            half_width = (upper - lower) / 2
            strength[i] = max(0.0, 1.0 - dist_from_mid / (half_width + 1e-9))

        strength[i] = np.clip(strength[i], 0.0, 1.0)

    # Detect transitions
    regime_series = pd.Series(regime_labels, index=values.index)
    transition_flag = regime_series.diff().ne(0).fillna(False)

    return RegimeState(
        regime=regime_series,
        regime_strength=pd.Series(strength, index=values.index),
        transition_flag=transition_flag,
    )
