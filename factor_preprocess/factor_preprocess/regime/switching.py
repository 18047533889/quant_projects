"""
Regime-based transform switching.

Applies different preprocessing transforms (zscore, rank, winsor, etc.) based on
detected regime. Fits transform parameters per regime and switches causally.
"""
import numpy as np
import pandas as pd
from typing import Optional, Literal, Dict, Callable, Any
from dataclasses import dataclass

from factor_preprocess.errors import (
    InsufficientObservations,
    MissingFittedStateError,
    StaleFittedStateError,
)


@dataclass
class RegimeSwitchingState:
    """
    Fitted regime-specific transform parameters.

    Attributes
    ----------
    regime_params : dict[int, dict[str, Any]]
        Mapping from regime label to transform parameters.
        Parameters depend on transform type.
    transform_type : str
        Transform type: "zscore", "rank", "winsor", "scale", "none".
    factor_name : str
        Factor name being transformed.
    n_regimes : int
        Number of regimes fitted.
    fit_window_start : pd.Timestamp
        Start of fitting window.
    fit_window_end : pd.Timestamp
        End of fitting window.
    """
    regime_params: Dict[int, Dict[str, Any]]
    transform_type: str
    factor_name: str
    n_regimes: int
    fit_window_start: pd.Timestamp
    fit_window_end: pd.Timestamp


def fit_regime_switching(
    values: pd.DataFrame,
    regime_labels: pd.Series,
    transform_type: Literal["zscore", "rank", "winsor", "scale", "none"] = "zscore",
    time_col: str = "date",
    value_col: str = "value",
    min_obs_per_regime: int = 30,
) -> RegimeSwitchingState:
    """
    Fit regime-specific transform parameters on a training window.

    Parameters
    ----------
    values : pd.DataFrame
        Factor values with time_col and value_col.
        Must be sorted by time_col.
    regime_labels : pd.Series
        Regime labels (0, 1, 2, ...) aligned with values.index.
    transform_type : str
        Transform to fit:
        - "zscore": fit mean and std per regime
        - "rank": no parameters (always percentile rank)
        - "winsor": fit quantile boundaries per regime
        - "scale": fit std per regime
        - "none": no transform (identity)
    time_col : str
        Time column
    value_col : str
        Value column to transform
    min_obs_per_regime : int
        Minimum observations per regime.

    Returns
    -------
    RegimeSwitchingState
        Fitted parameters per regime.

    Raises
    ------
    InsufficientObservations
        If any regime has fewer than min_obs_per_regime observations.

    Notes
    -----
    - zscore: stores {mean, std} per regime
    - rank: no parameters stored (always use current cross-section)
    - winsor: stores {lower_bound, upper_bound} at [1%, 99%] per regime
    - scale: stores {std} per regime
    - none: no parameters
    """
    if transform_type not in ["zscore", "rank", "winsor", "scale", "none"]:
        raise ValueError(f"Unknown transform_type: {transform_type}")

    # Verify sort order
    if not values[time_col].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by time_col")

    # Extract time window
    fit_window_start = values[time_col].min()
    fit_window_end = values[time_col].max()

    # Get unique regimes
    unique_regimes = sorted([r for r in regime_labels.dropna().unique() if not np.isnan(r)])
    n_regimes = len(unique_regimes)

    if n_regimes == 0:
        raise InsufficientObservations("No valid regime labels found")

    # Fit parameters per regime
    regime_params = {}

    for regime in unique_regimes:
        mask = (regime_labels == regime) & regime_labels.notna()
        regime_values = values.loc[mask, value_col].values

        n_obs = np.sum(mask)
        if n_obs < min_obs_per_regime:
            raise InsufficientObservations(
                f"Regime {regime} has only {n_obs} observations, "
                f"need at least {min_obs_per_regime}"
            )

        params = {}

        if transform_type == "zscore":
            params["mean"] = np.nanmean(regime_values)
            params["std"] = np.nanstd(regime_values, ddof=1)
            # Avoid division by zero
            if params["std"] <= 0:
                params["std"] = 1.0

        elif transform_type == "rank":
            # No parameters needed
            pass

        elif transform_type == "winsor":
            # Fit 1% and 99% quantiles
            params["lower_bound"] = np.nanquantile(regime_values, 0.01)
            params["upper_bound"] = np.nanquantile(regime_values, 0.99)

        elif transform_type == "scale":
            params["std"] = np.nanstd(regime_values, ddof=1)
            # Avoid division by zero
            if params["std"] <= 0:
                params["std"] = 1.0

        elif transform_type == "none":
            # No parameters
            pass

        regime_params[int(regime)] = params

    return RegimeSwitchingState(
        regime_params=regime_params,
        transform_type=transform_type,
        factor_name=value_col,
        n_regimes=n_regimes,
        fit_window_start=fit_window_start,
        fit_window_end=fit_window_end,
    )


def regime_switching_transform(
    values: pd.DataFrame,
    regime_labels: pd.Series,
    fitted_state: RegimeSwitchingState,
    time_col: str = "date",
    value_col: str = "value",
    check_staleness: bool = True,
) -> pd.Series:
    """
    Apply regime-specific transform to values.

    Parameters
    ----------
    values : pd.DataFrame
        Factor values with time_col and value_col.
        Must be sorted by time_col.
    regime_labels : pd.Series
        Regime labels (0, 1, 2, ...) aligned with values.index.
        NaN regime labels produce NaN output.
    fitted_state : RegimeSwitchingState
        Fitted parameters from fit_regime_switching.
    time_col : str
        Time column
    value_col : str
        Value column to transform
    check_staleness : bool
        If True, raises if values contain dates <= fitted_state.fit_window_end.

    Returns
    -------
    pd.Series
        Transformed values aligned with values.index.

    Raises
    ------
    MissingFittedStateError
        If fitted_state is None.
    StaleFittedStateError
        If check_staleness=True and values overlap with fit window.

    Notes
    -----
    - Transform is applied row-wise based on regime at each time
    - NaN regime labels produce NaN output
    - Unknown regimes produce NaN output (fail-closed)
    - For "rank", uses cross-sectional percentile rank at each time
    """
    if fitted_state is None:
        raise MissingFittedStateError("fitted_state is required")

    # Verify sort order
    if not values[time_col].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by time_col")

    # Check staleness
    if check_staleness:
        min_date = values[time_col].min()
        if min_date <= fitted_state.fit_window_end:
            raise StaleFittedStateError(
                f"Values start at {min_date}, but fit window ends at "
                f"{fitted_state.fit_window_end}. This violates fold-safety."
            )

    # Extract values
    raw_values = values[value_col].values
    result = np.full(len(values), np.nan)

    transform_type = fitted_state.transform_type

    if transform_type == "rank":
        # Special case: rank is cross-sectional, not per-regime
        # But we still switch based on regime presence
        # Group by time and compute percentile rank
        def rank_cs(group):
            """Cross-sectional percentile rank."""
            vals = group[value_col].values
            regimes = regime_labels.loc[group.index].values

            # Only rank where regime is valid
            mask = np.isfinite(vals) & np.isfinite(regimes)

            if not np.any(mask):
                return pd.Series(np.full(len(group), np.nan), index=group.index)

            ranked = np.full(len(group), np.nan)

            # Rank only valid regime observations
            valid_vals = vals[mask]
            if len(valid_vals) > 1:
                from scipy.stats import rankdata
                ranks = rankdata(valid_vals, method="average")
                # Convert to percentile [0, 1]
                pct_ranks = (ranks - 1.0) / (len(ranks) - 1.0)
                ranked[mask] = pct_ranks
            elif len(valid_vals) == 1:
                ranked[mask] = 0.5

            return pd.Series(ranked, index=group.index)

        # Use transform to avoid nested structures
        result_list = []
        for time_val, group in values.groupby(time_col, sort=False):
            result_list.append(rank_cs(group))

        result_series = pd.concat(result_list)
        return result_series

    # For other transforms, apply row-wise
    for i in range(len(values)):
        regime = regime_labels.iloc[i]

        if pd.isna(regime):
            # NaN regime -> NaN output
            continue

        regime_int = int(regime)

        if regime_int not in fitted_state.regime_params:
            # Unknown regime -> NaN output (fail-closed)
            continue

        val = raw_values[i]

        if not np.isfinite(val):
            # NaN input -> NaN output
            continue

        params = fitted_state.regime_params[regime_int]

        if transform_type == "zscore":
            mean = params["mean"]
            std = params["std"]
            result[i] = (val - mean) / std

        elif transform_type == "winsor":
            lower = params["lower_bound"]
            upper = params["upper_bound"]
            result[i] = np.clip(val, lower, upper)

        elif transform_type == "scale":
            std = params["std"]
            result[i] = val / std

        elif transform_type == "none":
            result[i] = val

    return pd.Series(result, index=values.index)
