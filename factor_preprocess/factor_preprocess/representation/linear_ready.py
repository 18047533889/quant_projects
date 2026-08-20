"""
Linear model ready features: dense feature matrix builder.

Prepares factors for linear models (OLS, Ridge, Lasso, GLM).
"""
import numpy as np
from typing import Optional, Literal
from dataclasses import dataclass


@dataclass(frozen=True)
class LinearReadyConfig:
    """Configuration for linear-ready feature preparation."""

    fill_method: Literal["zero", "mean", "drop"] = "zero"
    add_intercept: bool = False
    standardize: bool = True
    ddof: int = 1

    def __post_init__(self):
        """Validate configuration."""
        valid_fill = {"zero", "mean", "drop"}
        if self.fill_method not in valid_fill:
            raise ValueError(f"fill_method must be one of {valid_fill}")


@dataclass(frozen=True)
class LinearReadyResult:
    """Result of linear-ready transformation."""

    X: np.ndarray
    feature_names: list
    n_samples: int
    n_features: int
    has_intercept: bool
    fill_stats: dict

    def validate(self) -> bool:
        """
        Validate that features are ready for linear models.

        Returns
        -------
        bool
            True if features are valid
        """
        if self.X.size == 0:
            return False

        if not np.all(np.isfinite(self.X)):
            return False

        if self.X.shape != (self.n_samples, self.n_features):
            return False

        return True


def build_linear_ready(
    values: np.ndarray,
    config: LinearReadyConfig,
    feature_names: Optional[list] = None,
) -> LinearReadyResult:
    """
    Build dense feature matrix ready for linear models.

    Parameters
    ----------
    values : np.ndarray
        Feature values. Shape (n_samples, n_features)
    config : LinearReadyConfig
        Configuration for feature preparation
    feature_names : list, optional
        Names for each feature column

    Returns
    -------
    LinearReadyResult
        Dense feature matrix with no missing values

    Notes
    -----
    - fill_method="zero": replace NaN with 0
    - fill_method="mean": replace NaN with column mean
    - fill_method="drop": drop rows with any NaN
    - standardize: standardize each column to mean=0, std=1
    - add_intercept: prepend column of ones

    Raises
    ------
    ValueError
        If input is not 2D or if drop results in empty matrix
    """
    if values.ndim != 2:
        raise ValueError(f"values must be 2D, got shape {values.shape}")

    if values.size == 0:
        raise ValueError("values cannot be empty")

    n_samples, n_features = values.shape
    X = values.copy()

    # Track fill statistics
    fill_stats = {
        "n_missing_before": np.sum(np.isnan(X)),
        "missing_per_column": np.sum(np.isnan(X), axis=0).tolist(),
    }

    # Handle missing values
    if config.fill_method == "zero":
        X = np.nan_to_num(X, nan=0.0)

    elif config.fill_method == "mean":
        col_means = np.nanmean(X, axis=0)
        for col_idx in range(n_features):
            mask = np.isnan(X[:, col_idx])
            if np.any(mask):
                X[mask, col_idx] = col_means[col_idx]
        # Handle columns that are all NaN
        X = np.nan_to_num(X, nan=0.0)

    elif config.fill_method == "drop":
        mask = np.all(np.isfinite(X), axis=1)
        X = X[mask]
        n_samples = X.shape[0]
        if n_samples == 0:
            raise ValueError("All rows contain NaN; cannot build feature matrix")
        fill_stats["n_rows_dropped"] = np.sum(~mask)

    fill_stats["n_missing_after"] = np.sum(np.isnan(X))

    # Standardize
    if config.standardize:
        col_mean = np.mean(X, axis=0)
        col_std = np.std(X, axis=0, ddof=config.ddof)

        # Avoid division by zero for constant columns
        col_std = np.where(col_std > 0, col_std, 1.0)

        X = (X - col_mean) / col_std

        fill_stats["standardize_mean"] = col_mean.tolist()
        fill_stats["standardize_std"] = col_std.tolist()

    # Add intercept
    has_intercept = config.add_intercept
    if config.add_intercept:
        intercept_col = np.ones((n_samples, 1))
        X = np.hstack([intercept_col, X])
        n_features += 1

    # Build feature names
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(values.shape[1])]
    else:
        if len(feature_names) != values.shape[1]:
            raise ValueError(
                f"feature_names length {len(feature_names)} != n_features {values.shape[1]}"
            )

    if config.add_intercept:
        feature_names = ["intercept"] + list(feature_names)

    return LinearReadyResult(
        X=X,
        feature_names=feature_names,
        n_samples=n_samples,
        n_features=n_features,
        has_intercept=has_intercept,
        fill_stats=fill_stats,
    )


def assess_collinearity(X: np.ndarray, threshold: float = 0.99) -> dict:
    """
    Assess collinearity in feature matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    threshold : float
        Correlation threshold for flagging collinearity

    Returns
    -------
    dict
        Collinearity diagnostics

    Notes
    -----
    Simple correlation-based check. For full diagnostics, compute VIF.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    n_samples, n_features = X.shape

    if n_samples < 2 or n_features < 2:
        return {
            "has_collinearity": False,
            "max_correlation": 0.0,
            "high_correlation_pairs": [],
        }

    # Compute correlation matrix
    corr_matrix = np.corrcoef(X, rowvar=False)

    # Find high correlations (excluding diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    high_corr_pairs = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            if np.abs(corr_matrix[i, j]) >= threshold:
                high_corr_pairs.append((i, j, corr_matrix[i, j]))

    max_corr = 0.0
    if mask.any():
        max_corr = np.max(np.abs(corr_matrix[mask]))

    return {
        "has_collinearity": len(high_corr_pairs) > 0,
        "max_correlation": float(max_corr),
        "high_correlation_pairs": high_corr_pairs,
        "n_high_corr_pairs": len(high_corr_pairs),
    }
