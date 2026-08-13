"""
Regime-dependent factor weighting.

Fits regime-specific factor weights and applies them causally based on
detected regime state. All operations preserve fold-safety.
"""
import numpy as np
import pandas as pd
from typing import Optional, Literal, Dict
from dataclasses import dataclass

from factor_preprocess.errors import (
    InsufficientObservations,
    MissingFittedStateError,
    StaleFittedStateError,
)


@dataclass
class RegimeWeightState:
    """
    Fitted regime-specific weights.

    Attributes
    ----------
    regime_weights : dict[int, np.ndarray]
        Mapping from regime label to factor weights.
        Each array has shape (n_factors,) and sums to 1.0.
    factor_names : list[str]
        Factor names in weight array order.
    n_regimes : int
        Number of regimes fitted.
    fit_window_start : pd.Timestamp
        Start of fitting window (for staleness checking).
    fit_window_end : pd.Timestamp
        End of fitting window.
    """
    regime_weights: Dict[int, np.ndarray]
    factor_names: list[str]
    n_regimes: int
    fit_window_start: pd.Timestamp
    fit_window_end: pd.Timestamp


def fit_regime_weights(
    factors: pd.DataFrame,
    regime_labels: pd.Series,
    target: Optional[pd.Series] = None,
    method: Literal["equal", "volatility_inverse", "sharpe"] = "equal",
    time_col: str = "date",
    factor_cols: Optional[list[str]] = None,
    min_obs_per_regime: int = 30,
) -> RegimeWeightState:
    """
    Fit regime-specific factor weights on a training window.

    Parameters
    ----------
    factors : pd.DataFrame
        Factor values with time_col and factor columns.
        Must be sorted by time_col.
    regime_labels : pd.Series
        Regime labels (0, 1, 2, ...) aligned with factors.index.
    target : pd.Series, optional
        Target variable for Sharpe-based weighting.
        Required if method="sharpe".
    method : str
        Weighting method:
        - "equal": 1/n_factors per factor
        - "volatility_inverse": inverse volatility weighting
        - "sharpe": Sharpe ratio weighting (requires target)
    time_col : str
        Time column
    factor_cols : list[str], optional
        Factor columns to weight. If None, uses all numeric columns except time_col.
    min_obs_per_regime : int
        Minimum observations per regime. Raises if any regime has fewer.

    Returns
    -------
    RegimeWeightState
        Fitted weights per regime.

    Raises
    ------
    InsufficientObservations
        If any regime has fewer than min_obs_per_regime observations.

    Notes
    -----
    - All methods normalize weights to sum to 1.0
    - volatility_inverse: w_i ∝ 1/σ_i, where σ_i is factor i's std in regime
    - sharpe: w_i ∝ (μ_i / σ_i), where μ_i is correlation with target
    - Negative weights are clipped to zero and renormalized
    """
    if method not in ["equal", "volatility_inverse", "sharpe"]:
        raise ValueError(f"Unknown method: {method}")

    if method == "sharpe" and target is None:
        raise ValueError("target is required for method='sharpe'")

    # Verify sort order
    if not factors[time_col].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by time_col")

    # Select factor columns
    if factor_cols is None:
        factor_cols = [
            c for c in factors.columns
            if c != time_col and pd.api.types.is_numeric_dtype(factors[c])
        ]

    if len(factor_cols) == 0:
        raise ValueError("No factor columns found")

    # Extract time window
    fit_window_start = factors[time_col].min()
    fit_window_end = factors[time_col].max()

    # Get unique regimes
    unique_regimes = sorted([r for r in regime_labels.dropna().unique() if not np.isnan(r)])
    n_regimes = len(unique_regimes)

    if n_regimes == 0:
        raise InsufficientObservations("No valid regime labels found")

    # Fit weights per regime
    regime_weights = {}

    for regime in unique_regimes:
        mask = (regime_labels == regime) & regime_labels.notna()
        regime_factors = factors.loc[mask, factor_cols].values

        n_obs = np.sum(mask)
        if n_obs < min_obs_per_regime:
            raise InsufficientObservations(
                f"Regime {regime} has only {n_obs} observations, "
                f"need at least {min_obs_per_regime}"
            )

        if method == "equal":
            weights = np.ones(len(factor_cols)) / len(factor_cols)

        elif method == "volatility_inverse":
            # Compute std per factor (column-wise)
            stds = np.nanstd(regime_factors, axis=0, ddof=1)

            # Avoid division by zero
            stds = np.where(stds > 0, stds, 1e-9)

            # Inverse volatility
            weights = 1.0 / stds

            # Normalize
            weights = weights / np.sum(weights)

        elif method == "sharpe":
            # Compute correlation with target
            regime_target = target.loc[mask].values

            # Check alignment
            if len(regime_target) != len(regime_factors):
                raise ValueError("Target and factors must be aligned")

            correlations = np.full(len(factor_cols), 0.0)
            stds = np.full(len(factor_cols), 1e-9)

            for i in range(len(factor_cols)):
                factor_vals = regime_factors[:, i]
                valid_mask = np.isfinite(factor_vals) & np.isfinite(regime_target)

                if np.sum(valid_mask) >= 2:
                    correlations[i] = np.corrcoef(
                        factor_vals[valid_mask],
                        regime_target[valid_mask]
                    )[0, 1]
                    stds[i] = np.nanstd(factor_vals[valid_mask], ddof=1)

            # Sharpe-like weights: correlation / std
            stds = np.where(stds > 0, stds, 1e-9)
            weights = correlations / stds

            # Clip negative weights to zero
            weights = np.maximum(weights, 0.0)

            # Normalize (handle all-zero case)
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                # Fall back to equal weights
                weights = np.ones(len(factor_cols)) / len(factor_cols)

        regime_weights[int(regime)] = weights

    return RegimeWeightState(
        regime_weights=regime_weights,
        factor_names=factor_cols,
        n_regimes=n_regimes,
        fit_window_start=fit_window_start,
        fit_window_end=fit_window_end,
    )


def regime_adaptive_weights(
    factors: pd.DataFrame,
    regime_labels: pd.Series,
    fitted_state: RegimeWeightState,
    time_col: str = "date",
    factor_cols: Optional[list[str]] = None,
    check_staleness: bool = True,
) -> pd.DataFrame:
    """
    Apply regime-specific weights to factors.

    Parameters
    ----------
    factors : pd.DataFrame
        Factor values with time_col and factor columns.
        Must be sorted by time_col.
    regime_labels : pd.Series
        Regime labels (0, 1, 2, ...) aligned with factors.index.
        NaN regime labels produce NaN weighted factors.
    fitted_state : RegimeWeightState
        Fitted weights from fit_regime_weights.
    time_col : str
        Time column
    factor_cols : list[str], optional
        Factor columns to weight. If None, uses fitted_state.factor_names.
    check_staleness : bool
        If True, raises if factors contain dates <= fitted_state.fit_window_end.

    Returns
    -------
    pd.DataFrame
        Weighted factors with columns [time_col] + factor_cols.
        Each factor is multiplied by its regime-specific weight.

    Raises
    ------
    MissingFittedStateError
        If fitted_state is None.
    StaleFittedStateError
        If check_staleness=True and factors overlap with fit window.

    Notes
    -----
    - Weights are applied row-wise based on regime at each time
    - NaN regime labels produce NaN weighted factors
    - Original factor values are preserved if regime is not in fitted_state
    """
    if fitted_state is None:
        raise MissingFittedStateError("fitted_state is required")

    # Verify sort order
    if not factors[time_col].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by time_col")

    # Check staleness
    if check_staleness:
        min_date = factors[time_col].min()
        if min_date <= fitted_state.fit_window_end:
            raise StaleFittedStateError(
                f"Factors start at {min_date}, but fit window ends at "
                f"{fitted_state.fit_window_end}. This violates fold-safety."
            )

    # Select factor columns
    if factor_cols is None:
        factor_cols = fitted_state.factor_names

    # Verify factor alignment
    if set(factor_cols) != set(fitted_state.factor_names):
        raise ValueError(
            f"factor_cols {factor_cols} does not match fitted factor names "
            f"{fitted_state.factor_names}"
        )

    # Reorder to match fitted state
    factor_cols_ordered = fitted_state.factor_names

    # Extract factor matrix
    factor_matrix = factors[factor_cols_ordered].values

    # Apply weights row-wise
    weighted_matrix = np.full_like(factor_matrix, np.nan)

    for i in range(len(factors)):
        regime = regime_labels.iloc[i]

        if pd.isna(regime):
            # NaN regime -> NaN output
            continue

        regime_int = int(regime)

        if regime_int not in fitted_state.regime_weights:
            # Unknown regime -> keep original values
            weighted_matrix[i, :] = factor_matrix[i, :]
        else:
            # Apply regime-specific weights
            weights = fitted_state.regime_weights[regime_int]
            weighted_matrix[i, :] = factor_matrix[i, :] * weights

    # Build output DataFrame
    result = factors[[time_col]].copy()
    for j, col in enumerate(factor_cols_ordered):
        result[col] = weighted_matrix[:, j]

    return result
