"""
Kernel-based non-parametric neutralization for cross-sectional residuals.

Implements local polynomial regression using kernel weighting for
non-linear neutralization. Useful when exposure-factor relationships
are non-linear or vary across the exposure space.
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


def _gaussian_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Gaussian kernel: K(u) = exp(-u^2 / 2) / sqrt(2*pi)"""
    return np.exp(-0.5 * (distances / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))


def _epanechnikov_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Epanechnikov kernel: K(u) = 0.75 * (1 - u^2) for |u| <= 1"""
    u = distances / bandwidth
    weights = np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0.0)
    return weights / bandwidth


def _tricube_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Tricube kernel: K(u) = (1 - |u|^3)^3 for |u| <= 1"""
    u = distances / bandwidth
    weights = np.where(np.abs(u) <= 1, (1 - np.abs(u) ** 3) ** 3, 0.0)
    return weights / bandwidth


def kernel_neutralize(
    values: pd.DataFrame,
    exposures: pd.DataFrame,
    bandwidth: Optional[float] = None,
    kernel: Literal["gaussian", "epanechnikov", "tricube"] = "gaussian",
    date_col: str = "date",
    asset_col: str = "asset_id",
    value_col: str = "value",
    min_observations: int = 10,
    local_constant: bool = True,
    normalize: bool = True,
) -> pd.Series:
    """
    Cross-sectional kernel regression neutralization.

    Uses local polynomial regression with kernel weighting to neutralize
    factors against exposures in a non-parametric way. Allows for non-linear
    relationships between factors and exposures.

    Parameters
    ----------
    values : pd.DataFrame
        Factor values to neutralize. Columns: [date_col, asset_col, value_col]
    exposures : pd.DataFrame
        Exposure matrix. Columns: [date_col, asset_col, exposure1, exposure2, ...]
    bandwidth : float, optional
        Kernel bandwidth parameter. If None, uses Scott's rule.
        Larger values = smoother fit.
    kernel : {"gaussian", "epanechnikov", "tricube"}
        Kernel function to use.
        - "gaussian": smooth, unbounded support
        - "epanechnikov": compact support, efficient
        - "tricube": compact support, smooth
    date_col : str
        Date column name
    asset_col : str
        Asset identifier column
    value_col : str
        Factor value column to neutralize
    min_observations : int
        Minimum valid observations per date to fit
    local_constant : bool
        If True, fits local constant (Nadaraya-Watson).
        If False, fits local linear (more robust at boundaries).
    normalize : bool
        Whether to standardize exposures before computing distances

    Returns
    -------
    pd.Series
        Residuals aligned with values index.
        Dates with insufficient data produce NaN.

    Notes
    -----
    Per-date operation prevents time-series leakage.
    Computational complexity is O(n^2) per date for distance calculation.
    For high-dimensional exposures, consider using bandwidth_factor > 1
    to avoid curse of dimensionality.
    """
    if kernel not in ["gaussian", "epanechnikov", "tricube"]:
        raise ValueError(f"Unknown kernel: {kernel}")

    kernel_func = {
        "gaussian": _gaussian_kernel,
        "epanechnikov": _epanechnikov_kernel,
        "tricube": _tricube_kernel,
    }[kernel]

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

        if n_valid < min_observations:
            residuals = nan_residuals(y)
        else:
            y_valid = y[valid_mask]
            X_valid = X[valid_mask]

            try:
                # Normalize exposures for distance calculation
                if normalize:
                    X_mean = X_valid.mean(axis=0)
                    X_std = X_valid.std(axis=0, ddof=1)
                    X_std[X_std == 0] = 1.0
                    X_valid_norm = (X_valid - X_mean) / X_std
                else:
                    X_valid_norm = X_valid
                    X_mean = None
                    X_std = None

                # Determine bandwidth using Scott's rule if not provided
                if bandwidth is None:
                    d = X_valid_norm.shape[1]
                    h = n_valid ** (-1 / (d + 4))  # Scott's rule
                else:
                    h = bandwidth

                # Compute pairwise distances (Euclidean)
                # distances[i, j] = ||X_valid_norm[i] - X_valid_norm[j]||
                distances = np.sqrt(
                    np.sum((X_valid_norm[:, np.newaxis, :] - X_valid_norm[np.newaxis, :, :]) ** 2, axis=2)
                )

                # Compute kernel weights for each observation
                # weights[i, j] = K(||X[i] - X[j]|| / h)
                weights = kernel_func(distances, h)

                # Predict using local regression
                y_pred_valid = np.full(n_valid, np.nan)

                for i in range(n_valid):
                    w = weights[i, :]  # Weights for observation i

                    if w.sum() < 1e-10:
                        # Degenerate kernel weights (isolated point / tiny
                        # bandwidth): no local information exists, so the
                        # prediction is undefined. Falling back to the global
                        # mean would silently replace a residual with 0 (y - mean
                        # of nothing meaningful) — fail closed to NaN instead.
                        continue
                    else:
                        if local_constant:
                            # Nadaraya-Watson estimator (local constant)
                            y_pred_valid[i] = np.average(y_valid, weights=w)
                        else:
                            # Local linear regression
                            W_diag = np.diag(w)
                            X_centered = X_valid_norm - X_valid_norm[i]
                            X_fit = np.column_stack([np.ones(n_valid), X_centered])

                            try:
                                XtWX = X_fit.T @ W_diag @ X_fit
                                XtWy = X_fit.T @ W_diag @ y_valid
                                coef = np.linalg.solve(XtWX, XtWy)
                                y_pred_valid[i] = coef[0]  # Intercept is prediction at X[i]
                            except np.linalg.LinAlgError:
                                # Fallback to local constant
                                y_pred_valid[i] = np.average(y_valid, weights=w)

                # Compute residuals for valid observations
                residuals_valid = y_valid - y_pred_valid

                # Assign residuals back to full array
                residuals = nan_residuals(y)
                residuals[valid_mask] = residuals_valid

            # MemoryError is deliberately NOT caught: a per-date OOM must
            # propagate, not silently shrink the panel one date at a time.
            except (np.linalg.LinAlgError, ValueError):
                residuals = nan_residuals(y)

        result_df = pd.DataFrame({
            date_col: date,
            asset_col: group[asset_col].values,
            "residual": residuals,
        }, index=group[row_key].to_numpy())

        results.append(result_df)

    return align_residuals_to_values(results, row_key, values.index)
