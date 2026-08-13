"""
Neural network ready features: robust scaling and embedding preparation.

Prepares factors for neural networks (MLP, LSTM, Transformer).
"""
import numpy as np
from typing import Optional, Literal, Dict, List
from dataclasses import dataclass


@dataclass(frozen=True)
class NeuralReadyConfig:
    """Configuration for neural-ready feature preparation."""

    scaling: Literal["robust", "minmax", "standard", "none"] = "robust"
    robust_quantile_range: tuple = (25.0, 75.0)
    minmax_range: tuple = (0.0, 1.0)
    clip_outliers: bool = True
    outlier_quantile_range: tuple = (0.5, 99.5)
    handle_missing: Literal["zero", "mean", "forward_fill"] = "zero"
    add_time_features: bool = False
    normalize_per_sample: bool = False

    def __post_init__(self):
        """Validate configuration."""
        valid_scaling = {"robust", "minmax", "standard", "none"}
        if self.scaling not in valid_scaling:
            raise ValueError(f"scaling must be one of {valid_scaling}")

        valid_missing = {"zero", "mean", "forward_fill"}
        if self.handle_missing not in valid_missing:
            raise ValueError(f"handle_missing must be one of {valid_missing}")

        if len(self.robust_quantile_range) != 2:
            raise ValueError("robust_quantile_range must be (lower, upper)")
        if self.robust_quantile_range[0] >= self.robust_quantile_range[1]:
            raise ValueError("robust_quantile_range lower must be < upper")

        if len(self.minmax_range) != 2:
            raise ValueError("minmax_range must be (min, max)")

        if len(self.outlier_quantile_range) != 2:
            raise ValueError("outlier_quantile_range must be (lower, upper)")
        if self.outlier_quantile_range[0] >= self.outlier_quantile_range[1]:
            raise ValueError("outlier_quantile_range lower must be < upper")


@dataclass(frozen=True)
class NeuralReadyResult:
    """Result of neural-ready transformation."""

    X: np.ndarray
    feature_names: List[str]
    n_samples: int
    n_features: int
    scaling_params: Dict
    preprocessing_stats: Dict

    def validate(self) -> bool:
        """
        Validate that features are ready for neural networks.

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


def build_neural_ready(
    values: np.ndarray,
    config: NeuralReadyConfig,
    feature_names: Optional[List[str]] = None,
) -> NeuralReadyResult:
    """
    Build feature matrix ready for neural networks.

    Parameters
    ----------
    values : np.ndarray
        Feature values. Shape (n_samples, n_features)
    config : NeuralReadyConfig
        Configuration for feature preparation
    feature_names : list, optional
        Names for each feature column

    Returns
    -------
    NeuralReadyResult
        Feature matrix prepared for neural networks

    Notes
    -----
    Neural networks benefit from:
    - Robust scaling: uses median and IQR, less sensitive to outliers
    - Outlier clipping: prevents extreme values from dominating gradients
    - Missing value handling: networks require complete data
    - Optional per-sample normalization: for sequence models

    Raises
    ------
    ValueError
        If input is not 2D or if input is empty
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

    # Handle missing values
    if config.handle_missing == "zero":
        X = np.nan_to_num(X, nan=0.0)

    elif config.handle_missing == "mean":
        col_means = np.nanmean(X, axis=0)
        for col_idx in range(n_features):
            mask = np.isnan(X[:, col_idx])
            if np.any(mask):
                X[mask, col_idx] = col_means[col_idx]
        X = np.nan_to_num(X, nan=0.0)

    elif config.handle_missing == "forward_fill":
        for col_idx in range(n_features):
            col_data = X[:, col_idx]
            mask = np.isnan(col_data)
            if np.any(mask):
                # Forward fill: propagate last valid observation
                valid_indices = np.where(~mask)[0]
                if len(valid_indices) > 0:
                    last_valid_idx = 0
                    for i in range(n_samples):
                        if not mask[i]:
                            last_valid_idx = i
                        else:
                            if i > 0:
                                col_data[i] = col_data[last_valid_idx]
                X[:, col_idx] = col_data
        X = np.nan_to_num(X, nan=0.0)

    preprocessing_stats["n_missing_after_fill"] = int(np.sum(np.isnan(X)))

    # Clip outliers before scaling
    if config.clip_outliers:
        lower_q, upper_q = config.outlier_quantile_range
        n_clipped = 0
        clip_bounds = []

        for col_idx in range(n_features):
            col_data = X[:, col_idx]
            lower_bound = np.percentile(col_data, lower_q)
            upper_bound = np.percentile(col_data, upper_q)

            clipped = np.clip(col_data, lower_bound, upper_bound)
            n_clipped += np.sum(col_data != clipped)
            X[:, col_idx] = clipped

            clip_bounds.append({"lower": float(lower_bound), "upper": float(upper_bound)})

        preprocessing_stats["n_values_clipped"] = int(n_clipped)
        preprocessing_stats["clip_bounds"] = clip_bounds

    # Scaling
    scaling_params = {}

    if config.scaling == "robust":
        # Robust scaling: (X - median) / IQR
        lower_q, upper_q = config.robust_quantile_range
        col_medians = np.median(X, axis=0)
        col_q25 = np.percentile(X, lower_q, axis=0)
        col_q75 = np.percentile(X, upper_q, axis=0)
        col_iqr = col_q75 - col_q25

        # Avoid division by zero for constant columns
        col_iqr = np.where(col_iqr > 0, col_iqr, 1.0)

        X = (X - col_medians) / col_iqr

        scaling_params["method"] = "robust"
        scaling_params["medians"] = col_medians.tolist()
        scaling_params["iqr"] = col_iqr.tolist()
        scaling_params["quantile_range"] = config.robust_quantile_range

    elif config.scaling == "minmax":
        # Min-max scaling: X' = (X - X_min) / (X_max - X_min) * range + min
        col_min = np.min(X, axis=0)
        col_max = np.max(X, axis=0)
        col_range = col_max - col_min

        # Avoid division by zero for constant columns
        col_range = np.where(col_range > 0, col_range, 1.0)

        target_min, target_max = config.minmax_range
        X = (X - col_min) / col_range * (target_max - target_min) + target_min

        scaling_params["method"] = "minmax"
        scaling_params["col_min"] = col_min.tolist()
        scaling_params["col_max"] = col_max.tolist()
        scaling_params["target_range"] = config.minmax_range

    elif config.scaling == "standard":
        # Standard scaling: (X - mean) / std
        col_mean = np.mean(X, axis=0)
        col_std = np.std(X, axis=0, ddof=1)

        # Avoid division by zero for constant columns
        col_std = np.where(col_std > 0, col_std, 1.0)

        X = (X - col_mean) / col_std

        scaling_params["method"] = "standard"
        scaling_params["mean"] = col_mean.tolist()
        scaling_params["std"] = col_std.tolist()

    elif config.scaling == "none":
        scaling_params["method"] = "none"

    # Per-sample normalization (useful for sequence models)
    if config.normalize_per_sample:
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        row_norms = np.where(row_norms > 0, row_norms, 1.0)
        X = X / row_norms
        preprocessing_stats["per_sample_normalized"] = True

    # Build feature names
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]
    else:
        if len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length {len(feature_names)} != n_features {n_features}"
            )

    preprocessing_stats["n_missing_final"] = int(np.sum(np.isnan(X)))

    return NeuralReadyResult(
        X=X,
        feature_names=list(feature_names),
        n_samples=n_samples,
        n_features=n_features,
        scaling_params=scaling_params,
        preprocessing_stats=preprocessing_stats,
    )


def prepare_embeddings(
    categorical_features: np.ndarray,
    embedding_dim: Optional[int] = None,
) -> Dict:
    """
    Prepare categorical features for embedding layers.

    Parameters
    ----------
    categorical_features : np.ndarray
        Categorical feature values. Shape (n_samples, n_categorical_features)
    embedding_dim : int, optional
        Embedding dimension. If None, uses heuristic: min(50, (cardinality + 1) // 2)

    Returns
    -------
    dict
        Embedding configuration with cardinality and suggested dimensions

    Notes
    -----
    For neural networks with categorical features, embeddings are typically
    more effective than one-hot encoding. This function analyzes cardinality
    and suggests embedding dimensions.
    """
    if categorical_features.ndim != 2:
        raise ValueError(f"categorical_features must be 2D, got shape {categorical_features.shape}")

    n_samples, n_categorical = categorical_features.shape

    embedding_configs = []

    for col_idx in range(n_categorical):
        col_data = categorical_features[:, col_idx]

        # Get unique values (excluding NaN)
        finite_mask = np.isfinite(col_data)
        unique_values = np.unique(col_data[finite_mask])
        cardinality = len(unique_values)

        # Heuristic for embedding dimension
        if embedding_dim is None:
            suggested_dim = min(50, max(1, (cardinality + 1) // 2))
        else:
            suggested_dim = embedding_dim

        embedding_configs.append({
            "feature_idx": col_idx,
            "cardinality": int(cardinality),
            "embedding_dim": suggested_dim,
            "unique_values": unique_values.tolist(),
        })

    return {
        "n_categorical_features": n_categorical,
        "embedding_configs": embedding_configs,
        "total_embedding_params": sum(
            cfg["cardinality"] * cfg["embedding_dim"] for cfg in embedding_configs
        ),
    }
