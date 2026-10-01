"""FE-owned long-panel OLS neutralization composites."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _private_name(base: str, used: set) -> str:
    name = base
    while name in used:
        name = "_" + name
    used.add(name)
    return name


def ols_effective_rank(
    values: pd.DataFrame,
    exposures: pd.DataFrame,
    date_col: str = "date",
    asset_col: str = "asset_id",
    value_col: str = "value",
    min_observations: int = 10,
    add_intercept: bool = True,
) -> pd.Series:
    """Return per-date OLS residuals using effective-rank residual DOF.

    This is the FE-owned counterpart of FP's public ``ols_neutralize`` policy:
    exposure keys are unique and left-joined to values, complete finite rows
    are fit with ``lstsq``, all-zero predictors are removed, and residuals are
    available only when effective rank is strictly below valid observations.
    The generic FE ``cs_neutralize`` operator intentionally retains its stricter
    full-column-rank/condition-number contract.
    """
    if not isinstance(values, pd.DataFrame) or not isinstance(exposures, pd.DataFrame):
        raise TypeError("values and exposures must be pandas DataFrames")
    keys = [date_col, asset_col]
    if not values.columns.is_unique or not exposures.columns.is_unique:
        raise ValueError("Input column names must be unique")
    missing_values = set(keys + [value_col]).difference(values.columns)
    missing_exposures = set(keys).difference(exposures.columns)
    if missing_values:
        raise ValueError(f"values missing required columns: {sorted(missing_values)}")
    if missing_exposures:
        raise ValueError(f"exposures missing key columns: {sorted(missing_exposures)}")
    if exposures.duplicated(keys).any():
        raise ValueError("Exposure date/asset keys must be unique")
    exposure_cols = [column for column in exposures.columns if column not in keys]
    if not exposure_cols:
        raise ValueError("No exposure columns found")

    # Preserve the public input's positional identity, even when its index has
    # duplicate labels or the values table has repeated date/asset rows.
    used = set(values.columns) | set(exposures.columns)
    row_key = _private_name("__ols_row_position__", used)
    private_cols = []
    for index in range(len(exposure_cols)):
        private_cols.append(_private_name(f"__ols_exposure_{index}__", used))
    left = values.copy()
    left[row_key] = np.arange(len(left), dtype=np.intp)
    right = exposures[keys + exposure_cols].rename(
        columns=dict(zip(exposure_cols, private_cols))
    )
    merged = left.merge(
        right, on=keys, how="left", sort=False, validate="many_to_one"
    )

    output = np.full(len(values), np.nan, dtype=float)
    # Match pandas groupby default dropna behavior: null dates are not fitted.
    has_date_groups = False
    for _, group in merged.groupby(date_col):
        has_date_groups = True
        y = group[value_col].to_numpy(dtype=float, na_value=np.nan)
        x = group[private_cols].to_numpy(dtype=float, na_value=np.nan)
        valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
        n_valid = int(valid.sum())
        if n_valid < min_observations:
            continue
        y_valid = y[valid]
        x_valid = x[valid]
        # Empty/absent categories must not alter SVD shape or rank cutoff.
        x_valid = x_valid[:, np.any(x_valid != 0.0, axis=0)]
        design = (np.column_stack((np.ones(n_valid), x_valid))
                  if add_intercept else x_valid)
        try:
            beta, _, rank, _ = np.linalg.lstsq(design, y_valid, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if rank >= n_valid:
            continue
        prediction = design @ beta
        residual = y_valid - prediction
        magnitude = max(float(np.max(np.abs(y_valid))),
                        float(np.max(np.abs(prediction))))
        tolerance = 8 * np.finfo(float).eps * max(design.shape)
        if (np.isfinite(residual).all()
                and (magnitude == 0.0 or
                     float(np.max(np.abs(residual))) / magnitude <= tolerance)):
            residual[:] = 0.0
        positions = group[row_key].to_numpy(dtype=np.intp)[valid]
        output[positions] = residual
    return pd.Series(
        output, index=values.index, name="residual" if has_date_groups else None
    )


__all__ = ["ols_effective_rank"]
