"""
OLS neutralization for cross-sectional residuals.

Computes residuals from regressing factors against exposures.
"""
import numpy as np
import pandas as pd
from typing import Optional, Union
from datetime import datetime


def _align_exposures(values, exposures, date_col, asset_col, value_col):
    """Join unambiguous exposure columns without changing factor row order."""
    keys = [date_col, asset_col]
    if not values.columns.is_unique or not exposures.columns.is_unique:
        raise ValueError("Input column names must be unique")
    if exposures.duplicated(keys).any():
        raise ValueError("Exposure date/asset keys must be unique")
    exposure_cols = [c for c in exposures.columns if c not in keys]
    if not exposure_cols:
        raise ValueError("No exposure columns found")
    # Use private names on BOTH sides: a user exposure called 'value' must
    # never resolve to the dependent variable after a pandas suffix merge.
    private_cols = []
    used = set(values.columns) | set(exposures.columns)
    for i in range(len(exposure_cols)):
        name = f"__ols_exposure_{i}__"
        while name in used:
            name = "_" + name
        used.add(name)
        private_cols.append(name)
    merged = values.merge(
        exposures.rename(columns=dict(zip(exposure_cols, private_cols))),
        on=keys, how="left", sort=False, validate="many_to_one",
    )
    return merged, exposure_cols, private_cols


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
    # Carry a positional key through the merge because pandas resets the row
    # index during merge; the public result must remain aligned to ``values``.
    row_key = "__ols_row_position__"
    while row_key in values.columns or row_key in exposures.columns:
        row_key = f"_{row_key}"
    values_with_key = values.copy()
    values_with_key[row_key] = np.arange(len(values), dtype=np.intp)
    merged, _, exposure_cols = _align_exposures(
        values_with_key, exposures, date_col, asset_col, value_col)

    results = []

    for date, group in merged.groupby(date_col):
        # Extract y and X
        y = group[value_col].to_numpy(dtype=float, na_value=np.nan)
        X = group[exposure_cols].to_numpy(dtype=float, na_value=np.nan)

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
        }, index=group[row_key].to_numpy())

        results.append(result_df)

    # Concatenate and align with original index
    if not results:
        return pd.Series(np.nan, index=values.index)

    all_results = pd.concat(results)
    residuals = all_results["residual"].reindex(range(len(values)))
    residuals.index = values.index
    residuals.name = "residual"
    return residuals


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
    merged, exposure_cols, private_cols = _align_exposures(
        residuals, exposures, date_col, asset_col, value_col)

    results = []

    for date, group in merged.groupby(date_col):
        y = group[value_col].to_numpy(dtype=float, na_value=np.nan)
        X = group[private_cols].to_numpy(dtype=float, na_value=np.nan)

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
