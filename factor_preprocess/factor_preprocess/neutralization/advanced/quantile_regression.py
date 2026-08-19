"""
Quantile regression neutralization for cross-sectional residuals.

Implements quantile regression for neutralization against exposures
at specified quantiles. Useful for understanding exposure effects
across the factor distribution.
"""
import numpy as np
import pandas as pd
from typing import Optional

from factor_preprocess.neutralization._alignment import (
    add_position_key,
    align_residuals_to_values,
    merged_exposure_cols,
    nan_residuals,
)


def quantile_neutralize(
    values: pd.DataFrame,
    exposures: pd.DataFrame,
    quantile: float = 0.5,
    date_col: str = "date",
    asset_col: str = "asset_id",
    value_col: str = "value",
    min_observations: int = 10,
    add_intercept: bool = True,
    normalize: bool = True,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> pd.Series:
    """
    Cross-sectional quantile regression neutralization.

    Neutralizes factors by regressing on exposures at a specified quantile
    (e.g., median, upper/lower tail). Useful for asymmetric exposure effects.

    Parameters
    ----------
    values : pd.DataFrame
        Factor values to neutralize. Columns: [date_col, asset_col, value_col]
    exposures : pd.DataFrame
        Exposure matrix. Columns: [date_col, asset_col, exposure1, exposure2, ...]
    quantile : float
        Target quantile in [0, 1]. 0.5 = median, 0.95 = upper tail, etc.
    date_col : str
        Date column name
    asset_col : str
        Asset identifier column
    value_col : str
        Factor value column to neutralize
    min_observations : int
        Minimum valid observations per date to fit
    add_intercept : bool
        Whether to add intercept column
    normalize : bool
        Whether to standardize exposures before fitting
    max_iter : int
        Maximum IRLS iterations
    tol : float
        Convergence tolerance for coefficient changes

    Returns
    -------
    pd.Series
        Residuals aligned with values index.
        Dates with insufficient data produce NaN.

    Notes
    -----
    Uses iteratively reweighted least squares (IRLS) with asymmetric weights.
    For quantile tau, loss is: rho(u) = u * (tau - I(u < 0))
    Per-date operation prevents time-series leakage.
    Quantile=0.5 is equivalent to LAD (median regression).
    """
    if not 0 <= quantile <= 1:
        raise ValueError(f"Quantile must be in [0, 1], got {quantile}")

    values_with_key, row_key = add_position_key(values)
    merged = values_with_key.merge(
        exposures,
        on=[date_col, asset_col],
        how="left",
        suffixes=("", "_exp"),
    )

    exposure_cols = merged_exposure_cols(exposures, values, date_col, asset_col)

    results = []

    for date, group in merged.groupby(date_col):
        y = group[value_col].values
        X = group[exposure_cols].values.astype(np.float64, copy=False)

        # Drop rows with any NaN
        valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        n_valid = np.sum(valid_mask)

        if n_valid < min_observations or n_valid <= X.shape[1] + int(add_intercept):
            residuals = nan_residuals(y)
        else:
            y_valid = y[valid_mask]
            X_valid = X[valid_mask]

            # Normalize exposures
            if normalize:
                X_mean = X_valid.mean(axis=0)
                X_std = X_valid.std(axis=0, ddof=1)
                X_std[X_std == 0] = 1.0
                X_valid_norm = (X_valid - X_mean) / X_std
            else:
                X_valid_norm = X_valid
                X_mean = None
                X_std = None

            try:
                # Add intercept if requested
                if add_intercept:
                    X_fit = np.column_stack([np.ones(n_valid), X_valid_norm])
                else:
                    X_fit = X_valid_norm

                # Initialize with OLS
                coef, _, _, _ = np.linalg.lstsq(X_fit, y_valid, rcond=None)

                # IRLS for quantile regression
                tau = quantile
                epsilon = 1e-6

                for iteration in range(max_iter):
                    coef_old = coef.copy()

                    # Compute residuals
                    r = y_valid - X_fit @ coef

                    # Quantile regression weights
                    # w_i = tau / max(r_i, epsilon) if r_i > 0
                    # w_i = (tau - 1) / min(r_i, -epsilon) if r_i < 0
                    weights = np.where(
                        r > 0,
                        tau / np.maximum(r, epsilon),
                        (tau - 1) / np.minimum(r, -epsilon)
                    )
                    weights = np.abs(weights)

                    # Weighted least squares
                    W = np.diag(weights)
                    XtWX = X_fit.T @ W @ X_fit
                    XtWy = X_fit.T @ W @ y_valid
                    coef = np.linalg.solve(XtWX, XtWy)

                    # Check convergence
                    if np.max(np.abs(coef - coef_old)) < tol:
                        break

                # Predict on all data
                if normalize:
                    X_all_norm = (X - X_mean) / X_std
                else:
                    X_all_norm = X

                if add_intercept:
                    X_all_fit = np.column_stack([np.ones(len(y)), X_all_norm])
                else:
                    X_all_fit = X_all_norm

                y_pred = X_all_fit @ coef
                residuals = y - y_pred

                # NaN inputs produce NaN residuals
                residuals[~valid_mask] = np.nan

            except (np.linalg.LinAlgError, ValueError):
                residuals = nan_residuals(y)

        result_df = pd.DataFrame({
            date_col: date,
            asset_col: group[asset_col].values,
            "residual": residuals,
        }, index=group[row_key].to_numpy())

        results.append(result_df)

    return align_residuals_to_values(results, row_key, values.index)
