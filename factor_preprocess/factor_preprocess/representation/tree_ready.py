"""
Tree model ready features: categorical encoding stubs.

Prepares factors for tree-based models (XGBoost, LightGBM, CatBoost).
"""
import numpy as np
from typing import Optional, Literal, Dict, List
from dataclasses import dataclass


@dataclass(frozen=True)
class TreeReadyConfig:
    """Configuration for tree-ready feature preparation."""

    handle_missing: Literal["keep", "flag", "fill_median"] = "keep"
    add_missing_indicator: bool = False
    categorical_encoding: Literal["ordinal", "onehot", "none"] = "none"
    clip_outliers: bool = False
    outlier_std_threshold: float = 5.0

    def __post_init__(self):
        """Validate configuration."""
        valid_missing = {"keep", "flag", "fill_median"}
        if self.handle_missing not in valid_missing:
            raise ValueError(f"handle_missing must be one of {valid_missing}")

        valid_encoding = {"ordinal", "onehot", "none"}
        if self.categorical_encoding not in valid_encoding:
            raise ValueError(f"categorical_encoding must be one of {valid_encoding}")

        if self.outlier_std_threshold <= 0:
            raise ValueError("outlier_std_threshold must be positive")


@dataclass(frozen=True)
class TreeReadyResult:
    """Result of tree-ready transformation."""

    X: np.ndarray
    feature_names: List[str]
    n_samples: int
    n_features: int
    missing_indicators: Optional[np.ndarray]
    preprocessing_stats: Dict

    def validate(self) -> bool:
        """
        Validate that features are ready for tree models.

        Returns
        -------
        bool
            True if features are valid
        """
        if self.X.size == 0:
            return False

        if self.X.shape != (self.n_samples, self.n_features):
            return False

        return True


def build_tree_ready(
    values: np.ndarray,
    config: TreeReadyConfig,
    feature_names: Optional[List[str]] = None,
    categorical_mask: Optional[np.ndarray] = None,
) -> TreeReadyResult:
    """
    Build feature matrix ready for tree-based models.

    Parameters
    ----------
    values : np.ndarray
        Feature values. Shape (n_samples, n_features)
    config : TreeReadyConfig
        Configuration for feature preparation
    feature_names : list, optional
        Names for each feature column
    categorical_mask : np.ndarray, optional
        Boolean mask indicating categorical features

    Returns
    -------
    TreeReadyResult
        Feature matrix prepared for tree models

    Notes
    -----
    Tree models can handle missing values natively, so this is lighter than
    linear_ready. Focus is on:
    - Optional missing value indicators
    - Categorical encoding (stub implementation)
    - Outlier clipping

    Most tree libraries prefer to keep NaN as-is for native handling.
    """
    if values.ndim != 2:
        raise ValueError(f"values must be 2D, got shape {values.shape}")

    if values.size == 0:
        raise ValueError("values cannot be empty")

    n_samples, n_features = values.shape
    X = values.copy()

    preprocessing_stats = {
        "n_missing_original": int(np.sum(np.isnan(X))),
        "missing_per_column": np.sum(np.isnan(X), axis=0).tolist(),
    }

    # Missing indicators
    missing_indicators = None
    if config.add_missing_indicator:
        missing_indicators = np.isnan(X).astype(np.float64)
        preprocessing_stats["n_missing_indicators_added"] = n_features

    # Handle missing values
    if config.handle_missing == "keep":
        # Keep NaN as-is for native tree handling
        pass

    elif config.handle_missing == "flag":
        # Replace NaN with a special flag value (e.g., -999)
        X = np.where(np.isnan(X), -999.0, X)
        preprocessing_stats["missing_flag_value"] = -999.0

    elif config.handle_missing == "fill_median":
        # An all-NaN column has no median; nan_to_num would silently
        # substitute 0.0 and inject a fake all-zero "signal", so fail
        # closed naming the dead columns.
        all_nan = np.all(np.isnan(X), axis=0)
        if np.any(all_nan):
            dead = [feature_names[i] if feature_names is not None else f"f{i}"
                    for i in np.where(all_nan)[0]]
            raise ValueError(
                f"handle_missing='fill_median' cannot fill all-NaN columns: {dead}"
            )
        col_medians = np.nanmedian(X, axis=0)
        for col_idx in range(n_features):
            mask = np.isnan(X[:, col_idx])
            if np.any(mask):
                X[mask, col_idx] = col_medians[col_idx]
        preprocessing_stats["fill_medians"] = col_medians.tolist()

    # Clip outliers
    if config.clip_outliers:
        n_clipped = 0
        for col_idx in range(n_features):
            col_data = X[:, col_idx]
            finite_mask = np.isfinite(col_data)

            if np.sum(finite_mask) > 0:
                col_mean = np.mean(col_data[finite_mask])
                col_std = np.std(col_data[finite_mask], ddof=1)

                if col_std > 0:
                    lower_bound = col_mean - config.outlier_std_threshold * col_std
                    upper_bound = col_mean + config.outlier_std_threshold * col_std

                    clipped = np.clip(col_data, lower_bound, upper_bound)
                    n_clipped += np.sum(col_data != clipped)
                    X[:, col_idx] = clipped

        preprocessing_stats["n_values_clipped"] = int(n_clipped)
        preprocessing_stats["outlier_threshold_std"] = config.outlier_std_threshold

    # Categorical encoding (stub)
    if config.categorical_encoding != "none":
        if categorical_mask is not None and np.any(categorical_mask):
            raise NotImplementedError(
                f"categorical_encoding='{config.categorical_encoding}' not yet implemented"
            )

    # Build feature names
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    else:
        if len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length {len(feature_names)} != n_features {n_features}"
            )

    preprocessing_stats["n_missing_final"] = int(np.sum(np.isnan(X)))

    return TreeReadyResult(
        X=X,
        feature_names=list(feature_names),
        n_samples=n_samples,
        n_features=n_features,
        missing_indicators=missing_indicators,
        preprocessing_stats=preprocessing_stats,
    )


def suggest_tree_params(X: np.ndarray) -> Dict:
    """
    Suggest initial tree model hyperparameters based on data characteristics.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)

    Returns
    -------
    dict
        Suggested hyperparameters for tree models

    Notes
    -----
    Simple heuristics based on dataset size and feature count.
    These are starting points for tuning, not production values.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    n_samples, n_features = X.shape

    # Suggest max_depth based on sample size
    if n_samples < 1000:
        max_depth = 3
    elif n_samples < 10000:
        max_depth = 5
    else:
        max_depth = 7

    # Suggest learning rate
    if n_samples < 1000:
        learning_rate = 0.1
    else:
        learning_rate = 0.05

    # Suggest min_child_weight / min_samples_leaf
    min_samples_leaf = max(1, int(n_samples * 0.001))

    # Suggest subsample
    subsample = 0.8 if n_samples > 1000 else 1.0

    # Suggest colsample
    colsample_bytree = 0.8 if n_features > 10 else 1.0

    return {
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "min_samples_leaf": min_samples_leaf,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "n_estimators": 100,
        "notes": "These are heuristic starting points. Tune based on validation performance.",
    }
