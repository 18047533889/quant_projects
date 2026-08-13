"""
OLS neutralization for cross-sectional residuals.

Computes residuals from regressing factors against exposures.
"""
import numpy as np
import pandas as pd
from typing import Optional, Union
from datetime import datetime


def ols_neutralize(
    values: pd.DataFrame,
    exposures: pd.DataFrame,
    date_col: str = "date",
    asset_col: str = "asset_id",
    value_col: str = "value",
    min_observations: int = 10,
    add_intercept: bool = True,
) -> pd.Series:
    """
    Cross-sectional OLS neutralization (per-date residuals).

    Parameters
    ----------
    values : pd.DataFrame
        Factor values to neutralize. Columns: [date_col, asset_col, value_col]
    exposures : pd.DataFrame
        Exposure matrix. Columns: [date_col, asset_col, exposure1, exposure2, ...]
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

    Returns
    -------
    pd.Series
        Residuals aligned with values index.
        Dates with insufficient data produce NaN.

    Notes
    -----
    Uses numpy.linalg.lstsq for stability with rank-deficient matrices.
    Operates per-date (no time-series leakage).
    NaN handling: pairwise deletion per date.
    """
    # Merge values and exposures
    merged = values.merge(
        exposures,
        on=[date_col, asset_col],
        how="left",
        suffixes=("", "_exp"),
    )

    # Get exposure columns (exclude metadata)
    exposure_cols = [c for c in exposures.columns if c not in [date_col, asset_col]]

    if not exposure_cols:
        raise ValueError("No exposure columns found")

    results = []

    for date, group in merged.groupby(date_col):
        # Extract y and X
        y = group[value_col].values
        X = group[exposure_cols].values

        # Drop rows with any NaN in y or X
        valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        n_valid = np.sum(valid_mask)

        if n_valid < min_observations or n_valid <= X.shape[1]:
            # Insufficient data or rank deficient
            residuals = np.full_like(y, np.nan)
        else:
            y_valid = y[valid_mask]
            X_valid = X[valid_mask]

            if add_intercept:
                X_valid = np.column_stack([np.ones(n_valid), X_valid])

            # Fit OLS using lstsq
            try:
                coef, _, rank, _ = np.linalg.lstsq(X_valid, y_valid, rcond=None)

                # Predict and compute residuals
                if add_intercept:
                    X_all = np.column_stack([np.ones(len(y)), X])
                else:
                    X_all = X

                y_pred = X_all @ coef
                residuals = y - y_pred

                # NaN inputs produce NaN residuals
                residuals[~valid_mask] = np.nan

            except np.linalg.LinAlgError:
                # Singular matrix
                residuals = np.full_like(y, np.nan)

        # Build result DataFrame with original index
        result_df = pd.DataFrame({
            date_col: date,
            asset_col: group[asset_col].values,
            "residual": residuals,
        }, index=group.index)

        results.append(result_df)

    # Concatenate and align with original index
    if not results:
        return pd.Series(np.nan, index=values.index)

    all_results = pd.concat(results)
    return all_results["residual"].reindex(values.index)


def compute_exposures(
    residuals: pd.DataFrame,
    exposures: pd.DataFrame,
    date_col: str = "date",
    asset_col: str = "asset_id",
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Compute exposure coefficients from factor-exposure regression.

    Parameters
    ----------
    residuals : pd.DataFrame
        Factor values (or residuals). Columns: [date_col, asset_col, value_col]
    exposures : pd.DataFrame
        Exposure matrix. Columns: [date_col, asset_col, exposure1, exposure2, ...]
    date_col : str
        Date column name
    asset_col : str
        Asset identifier column
    value_col : str
        Factor value column

    Returns
    -------
    pd.DataFrame
        Coefficients per date. Columns: [date_col, exposure1_coef, exposure2_coef, ...]

    Notes
    -----
    Uses OLS per date. Returns NaN for dates with insufficient data.
    """
    merged = residuals.merge(
        exposures,
        on=[date_col, asset_col],
        how="left",
    )

    exposure_cols = [c for c in exposures.columns if c not in [date_col, asset_col]]

    if not exposure_cols:
        raise ValueError("No exposure columns found")

    results = []

    for date, group in merged.groupby(date_col):
        y = group[value_col].values
        X = group[exposure_cols].values

        # Drop NaN
        valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        n_valid = np.sum(valid_mask)

        if n_valid > X.shape[1]:
            y_valid = y[valid_mask]
            X_valid = X[valid_mask]

            # Add intercept
            X_valid = np.column_stack([np.ones(n_valid), X_valid])

            try:
                coef, _, _, _ = np.linalg.lstsq(X_valid, y_valid, rcond=None)
                # coef[0] is intercept, coef[1:] are exposure coefficients
                coef_dict = {date_col: date, "intercept": coef[0]}
                for i, col in enumerate(exposure_cols):
                    coef_dict[f"{col}_coef"] = coef[i + 1]
            except np.linalg.LinAlgError:
                coef_dict = {date_col: date, "intercept": np.nan}
                for col in exposure_cols:
                    coef_dict[f"{col}_coef"] = np.nan
        else:
            coef_dict = {date_col: date, "intercept": np.nan}
            for col in exposure_cols:
                coef_dict[f"{col}_coef"] = np.nan

        results.append(coef_dict)

    return pd.DataFrame(results)
