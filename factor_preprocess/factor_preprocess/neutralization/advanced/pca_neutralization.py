"""
PCA-based neutralization for cross-sectional residuals.

Uses principal component analysis to extract orthogonal exposure factors,
then neutralizes against the top-k components. Useful when exposures
are highly collinear or when reducing dimensionality.
"""
import numpy as np
import pandas as pd
from typing import Optional, Union


def pca_neutralize(
    values: pd.DataFrame,
    exposures: pd.DataFrame,
    n_components: Optional[int] = None,
    variance_threshold: Optional[float] = None,
    date_col: str = "date",
    asset_col: str = "asset_id",
    value_col: str = "value",
    min_observations: int = 10,
    add_intercept: bool = True,
    center: bool = True,
    scale: bool = True,
) -> pd.Series:
    """
    Cross-sectional PCA neutralization.

    Performs PCA on exposures per date, then neutralizes against
    the top principal components.

    Parameters
    ----------
    values : pd.DataFrame
        Factor values to neutralize. Columns: [date_col, asset_col, value_col]
    exposures : pd.DataFrame
        Exposure matrix. Columns: [date_col, asset_col, exposure1, exposure2, ...]
    n_components : int, optional
        Number of principal components to neutralize against.
        If None, uses variance_threshold instead.
    variance_threshold : float, optional
        Cumulative explained variance threshold (e.g., 0.95 for 95%).
        Only used if n_components is None. Default is 0.95.
    date_col : str
        Date column name
    asset_col : str
        Asset identifier column
    value_col : str
        Factor value column to neutralize
    min_observations : int
        Minimum valid observations per date to fit
    add_intercept : bool
        Whether to add intercept column after PCA projection
    center : bool
        Whether to center exposures before PCA
    scale : bool
        Whether to scale exposures to unit variance before PCA

    Returns
    -------
    pd.Series
        Residuals aligned with values index.
        Dates with insufficient data produce NaN.

    Notes
    -----
    Per-date operation prevents time-series leakage.
    PCA is fit on valid observations only.
    Handles multicollinearity by construction (orthogonal components).
    """
    if n_components is None and variance_threshold is None:
        variance_threshold = 0.95

    merged = values.merge(
        exposures,
        on=[date_col, asset_col],
        how="left",
        suffixes=("", "_exp"),
    )

    exposure_cols = [c for c in exposures.columns if c not in [date_col, asset_col]]

    if not exposure_cols:
        raise ValueError("No exposure columns found")

    results = []

    for date, group in merged.groupby(date_col):
        y = group[value_col].values
        X = group[exposure_cols].values

        # Drop rows with any NaN
        valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        n_valid = np.sum(valid_mask)

        if n_valid < min_observations:
            residuals = np.full_like(y, np.nan)
        else:
            y_valid = y[valid_mask]
            X_valid = X[valid_mask]

            try:
                # Center and scale exposures
                if center:
                    X_mean = X_valid.mean(axis=0)
                    X_centered = X_valid - X_mean
                else:
                    X_centered = X_valid
                    X_mean = np.zeros(X_valid.shape[1])

                if scale:
                    X_std = X_centered.std(axis=0, ddof=1)
                    X_std[X_std == 0] = 1.0
                    X_scaled = X_centered / X_std
                else:
                    X_scaled = X_centered
                    X_std = np.ones(X_centered.shape[1])

                # Perform PCA via SVD
                # X_scaled = U @ S @ Vt
                # Principal components are columns of U @ S or X_scaled @ V
                U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)

                # Determine number of components
                if n_components is not None:
                    k = min(n_components, len(s))
                else:
                    # Use variance threshold
                    explained_var = (s ** 2) / (n_valid - 1)
                    cumsum_var = np.cumsum(explained_var)
                    total_var = cumsum_var[-1]
                    k = np.searchsorted(cumsum_var / total_var, variance_threshold) + 1
                    k = min(k, len(s))

                # Principal component scores for valid observations
                # Z = U @ S = X_scaled @ V.T (scores)
                Z_valid = U[:, :k] * s[:k]

                # Neutralize y against top-k principal components
                if add_intercept:
                    y_mean = y_valid.mean()
                    y_centered = y_valid - y_mean

                    # Regress y_centered on Z_valid
                    ZtZ = Z_valid.T @ Z_valid
                    Zty = Z_valid.T @ y_centered
                    coef_pca = np.linalg.solve(ZtZ, Zty)
                else:
                    y_mean = 0.0
                    ZtZ = Z_valid.T @ Z_valid
                    Zty = Z_valid.T @ y_valid
                    coef_pca = np.linalg.solve(ZtZ, Zty)

                # Project all observations (including NaN) to PC space
                if center:
                    X_all_centered = X - X_mean
                else:
                    X_all_centered = X

                if scale:
                    X_all_scaled = X_all_centered / X_std
                else:
                    X_all_scaled = X_all_centered

                # V is the loading matrix (eigenvectors)
                V = Vt.T
                Z_all = X_all_scaled @ V[:, :k]

                # Predict and compute residuals
                y_pred = Z_all @ coef_pca + y_mean
                residuals = y - y_pred

                # NaN inputs produce NaN residuals
                residuals[~valid_mask] = np.nan

            except (np.linalg.LinAlgError, ValueError):
                residuals = np.full_like(y, np.nan)

        result_df = pd.DataFrame({
            date_col: date,
            asset_col: group[asset_col].values,
            "residual": residuals,
        }, index=group.index)

        results.append(result_df)

    if not results:
        return pd.Series(np.nan, index=values.index)

    all_results = pd.concat(results)
    return all_results["residual"].reindex(values.index)
