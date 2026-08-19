"""
Regularized neutralization for cross-sectional residuals.

Implements Ridge, Lasso, and Elastic Net per-date neutralization
with automatic alpha selection and numerical stability checks.
"""
import numpy as np
import pandas as pd
from typing import Optional, Literal

from factor_preprocess.neutralization._alignment import (
    add_position_key,
    align_residuals_to_values,
    merged_exposure_cols,
    nan_residuals,
)


def ridge_neutralize(
    values: pd.DataFrame,
    exposures: pd.DataFrame,
    alpha: float = 1.0,
    date_col: str = "date",
    asset_col: str = "asset_id",
    value_col: str = "value",
    min_observations: int = 10,
    add_intercept: bool = True,
    normalize: bool = True,
) -> pd.Series:
    """
    Cross-sectional Ridge (L2) neutralization.

    Parameters
    ----------
    values : pd.DataFrame
        Factor values to neutralize. Columns: [date_col, asset_col, value_col]
    exposures : pd.DataFrame
        Exposure matrix. Columns: [date_col, asset_col, exposure1, exposure2, ...]
    alpha : float
        Regularization strength (larger = more shrinkage)
    date_col : str
        Date column name
    asset_col : str
        Asset identifier column
    value_col : str
        Factor value column to neutralize
    min_observations : int
        Minimum valid observations per date to fit
    add_intercept : bool
        Whether to add intercept column (intercept not regularized)
    normalize : bool
        Whether to standardize exposures before fitting

    Returns
    -------
    pd.Series
        Residuals aligned with values index.
        Dates with insufficient data produce NaN.

    Notes
    -----
    Uses closed-form Ridge solution: (X'X + alpha*I)^-1 X'y
    Intercept is not penalized when add_intercept=True.
    Per-date operation prevents time-series leakage.
    """
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
            # Insufficient data, or the fit is (near-)determined — an
            # exactly-determined regression "fits" perfectly and silently
            # returns ~0 residuals that look like a fully neutralized factor.
            residuals = nan_residuals(y)
        else:
            y_valid = y[valid_mask]
            X_valid = X[valid_mask]

            # Normalize exposures if requested
            if normalize:
                X_mean = X_valid.mean(axis=0)
                X_std = X_valid.std(axis=0, ddof=1)
                X_std[X_std == 0] = 1.0  # Avoid division by zero
                X_valid_norm = (X_valid - X_mean) / X_std
            else:
                X_valid_norm = X_valid
                X_mean = None
                X_std = None

            try:
                if add_intercept:
                    # Center y, fit Ridge on normalized X
                    y_mean = y_valid.mean()
                    y_centered = y_valid - y_mean

                    # Ridge regression on normalized features
                    XtX = X_valid_norm.T @ X_valid_norm
                    Xty = X_valid_norm.T @ y_centered
                    ridge_matrix = XtX + alpha * np.eye(XtX.shape[0])
                    coef = np.linalg.solve(ridge_matrix, Xty)

                    # Compute intercept
                    intercept = y_mean
                else:
                    # Ridge without intercept
                    XtX = X_valid_norm.T @ X_valid_norm
                    Xty = X_valid_norm.T @ y_valid
                    ridge_matrix = XtX + alpha * np.eye(XtX.shape[0])
                    coef = np.linalg.solve(ridge_matrix, Xty)
                    intercept = 0.0

                # Predict on all data (including NaN rows)
                if normalize:
                    X_all_norm = (X - X_mean) / X_std
                else:
                    X_all_norm = X

                y_pred = X_all_norm @ coef + intercept
                residuals = y - y_pred

                # NaN inputs produce NaN residuals
                residuals[~valid_mask] = np.nan

            except np.linalg.LinAlgError:
                residuals = nan_residuals(y)

        result_df = pd.DataFrame({
            date_col: date,
            asset_col: group[asset_col].values,
            "residual": residuals,
        }, index=group[row_key].to_numpy())

        results.append(result_df)

    return align_residuals_to_values(results, row_key, values.index)


def lasso_neutralize(
    values: pd.DataFrame,
    exposures: pd.DataFrame,
    alpha: float = 1.0,
    date_col: str = "date",
    asset_col: str = "asset_id",
    value_col: str = "value",
    min_observations: int = 10,
    add_intercept: bool = True,
    normalize: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> pd.Series:
    """
    Cross-sectional Lasso (L1) neutralization.

    Parameters
    ----------
    values : pd.DataFrame
        Factor values to neutralize
    exposures : pd.DataFrame
        Exposure matrix
    alpha : float
        Regularization strength
    date_col : str
        Date column name
    asset_col : str
        Asset identifier column
    value_col : str
        Factor value column to neutralize
    min_observations : int
        Minimum valid observations per date
    add_intercept : bool
        Whether to add intercept (not regularized)
    normalize : bool
        Whether to standardize exposures
    max_iter : int
        Maximum coordinate descent iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    pd.Series
        Residuals aligned with values index

    Notes
    -----
    Uses coordinate descent for L1 minimization.
    Implements soft thresholding for feature selection.
    """
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

        valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        n_valid = np.sum(valid_mask)

        if n_valid < min_observations or n_valid <= X.shape[1] + int(add_intercept):
            # Insufficient data, or the fit is (near-)determined — an
            # exactly-determined regression "fits" perfectly and silently
            # returns ~0 residuals that look like a fully neutralized factor.
            residuals = nan_residuals(y)
        else:
            y_valid = y[valid_mask]
            X_valid = X[valid_mask]

            # Normalize
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
                if add_intercept:
                    y_mean = y_valid.mean()
                    y_centered = y_valid - y_mean
                else:
                    y_centered = y_valid
                    y_mean = 0.0

                # Coordinate descent for Lasso
                n_features = X_valid_norm.shape[1]
                coef = np.zeros(n_features)

                for iteration in range(max_iter):
                    coef_old = coef.copy()

                    for j in range(n_features):
                        # Compute partial residual
                        r = y_centered - X_valid_norm @ coef + coef[j] * X_valid_norm[:, j]

                        # Soft thresholding
                        rho = X_valid_norm[:, j] @ r
                        z = np.sum(X_valid_norm[:, j] ** 2)

                        if z > 0:
                            if rho < -alpha:
                                coef[j] = (rho + alpha) / z
                            elif rho > alpha:
                                coef[j] = (rho - alpha) / z
                            else:
                                coef[j] = 0.0

                    # Check convergence
                    if np.max(np.abs(coef - coef_old)) < tol:
                        break

                # Compute intercept
                if add_intercept:
                    intercept = y_mean - (X_valid_norm.mean(axis=0) @ coef)
                else:
                    intercept = 0.0

                # Predict
                if normalize:
                    X_all_norm = (X - X_mean) / X_std
                else:
                    X_all_norm = X

                y_pred = X_all_norm @ coef + intercept
                residuals = y - y_pred
                residuals[~valid_mask] = np.nan

            # MemoryError is deliberately NOT caught: a per-date OOM must
            # propagate, not silently shrink the panel one date at a time.
            except (ValueError, IndexError, np.linalg.LinAlgError, ArithmeticError):
                residuals = nan_residuals(y)

        result_df = pd.DataFrame({
            date_col: date,
            asset_col: group[asset_col].values,
            "residual": residuals,
        }, index=group[row_key].to_numpy())

        results.append(result_df)

    return align_residuals_to_values(results, row_key, values.index)


def elastic_net_neutralize(
    values: pd.DataFrame,
    exposures: pd.DataFrame,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    date_col: str = "date",
    asset_col: str = "asset_id",
    value_col: str = "value",
    min_observations: int = 10,
    add_intercept: bool = True,
    normalize: bool = True,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> pd.Series:
    """
    Cross-sectional Elastic Net (L1 + L2) neutralization.

    Parameters
    ----------
    values : pd.DataFrame
        Factor values to neutralize
    exposures : pd.DataFrame
        Exposure matrix
    alpha : float
        Overall regularization strength
    l1_ratio : float
        Mix of L1 and L2 penalty. 0 = Ridge, 1 = Lasso
    date_col : str
        Date column name
    asset_col : str
        Asset identifier column
    value_col : str
        Factor value column to neutralize
    min_observations : int
        Minimum valid observations per date
    add_intercept : bool
        Whether to add intercept
    normalize : bool
        Whether to standardize exposures
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    pd.Series
        Residuals aligned with values index

    Notes
    -----
    Combines L1 (feature selection) and L2 (stability).
    Uses coordinate descent with elastic net penalty.
    """
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

        valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        n_valid = np.sum(valid_mask)

        if n_valid < min_observations or n_valid <= X.shape[1] + int(add_intercept):
            # Insufficient data, or the fit is (near-)determined — an
            # exactly-determined regression "fits" perfectly and silently
            # returns ~0 residuals that look like a fully neutralized factor.
            residuals = nan_residuals(y)
        else:
            y_valid = y[valid_mask]
            X_valid = X[valid_mask]

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
                if add_intercept:
                    y_mean = y_valid.mean()
                    y_centered = y_valid - y_mean
                else:
                    y_centered = y_valid
                    y_mean = 0.0

                # Elastic net coordinate descent
                n_features = X_valid_norm.shape[1]
                coef = np.zeros(n_features)

                l1_penalty = alpha * l1_ratio
                l2_penalty = alpha * (1 - l1_ratio)

                for iteration in range(max_iter):
                    coef_old = coef.copy()

                    for j in range(n_features):
                        r = y_centered - X_valid_norm @ coef + coef[j] * X_valid_norm[:, j]
                        rho = X_valid_norm[:, j] @ r
                        z = np.sum(X_valid_norm[:, j] ** 2) + l2_penalty

                        if z > 0:
                            if rho < -l1_penalty:
                                coef[j] = (rho + l1_penalty) / z
                            elif rho > l1_penalty:
                                coef[j] = (rho - l1_penalty) / z
                            else:
                                coef[j] = 0.0

                    if np.max(np.abs(coef - coef_old)) < tol:
                        break

                if add_intercept:
                    intercept = y_mean - (X_valid_norm.mean(axis=0) @ coef)
                else:
                    intercept = 0.0

                if normalize:
                    X_all_norm = (X - X_mean) / X_std
                else:
                    X_all_norm = X

                y_pred = X_all_norm @ coef + intercept
                residuals = y - y_pred
                residuals[~valid_mask] = np.nan

            # MemoryError deliberately not caught — see ridge path above.
            except (ValueError, IndexError, np.linalg.LinAlgError, ArithmeticError):
                residuals = nan_residuals(y)

        result_df = pd.DataFrame({
            date_col: date,
            asset_col: group[asset_col].values,
            "residual": residuals,
        }, index=group[row_key].to_numpy())

        results.append(result_df)

    return align_residuals_to_values(results, row_key, values.index)
