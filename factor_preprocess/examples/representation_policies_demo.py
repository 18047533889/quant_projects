"""
Demonstration of representation policies for different model types.

Shows how to prepare the same factor data for:
1. Linear models (OLS, Ridge, Lasso) - standardization required
2. Tree models (XGBoost, LightGBM) - native missing handling, no scaling
3. Neural networks (MLP, LSTM) - robust scaling, outlier clipping
"""
import numpy as np
from factor_preprocess.representation import (
    build_linear_ready,
    LinearReadyConfig,
    build_tree_ready,
    TreeReadyConfig,
    build_neural_ready,
    NeuralReadyConfig,
    prepare_embeddings,
)


def demo_representation_policies():
    """Demonstrate the three representation policies."""
    print("=" * 70)
    print("Representation Policies Demo")
    print("=" * 70)

    # Create sample factor data with missing values and outliers
    np.random.seed(42)
    n_samples, n_features = 100, 5

    # Generate base data
    X = np.random.randn(n_samples, n_features) * 10 + 50

    # Add some missing values
    missing_mask = np.random.rand(n_samples, n_features) < 0.1
    X[missing_mask] = np.nan

    # Add outliers
    outlier_mask = np.random.rand(n_samples, n_features) < 0.05
    X[outlier_mask] = X[outlier_mask] * 5

    print(f"\nInput data shape: {X.shape}")
    print(f"Missing values: {np.sum(np.isnan(X))}")
    print(f"Data range: [{np.nanmin(X):.2f}, {np.nanmax(X):.2f}]")

    # 1. Linear model preparation
    print("\n" + "-" * 70)
    print("1. LINEAR MODEL READY")
    print("-" * 70)

    linear_config = LinearReadyConfig(
        fill_method="mean",
        standardize=True,
        add_intercept=False,
    )

    linear_result = build_linear_ready(X, linear_config)
    print(f"Output shape: {linear_result.X.shape}")
    print(f"Has NaN: {np.any(np.isnan(linear_result.X))}")
    print(f"Mean: {np.mean(linear_result.X, axis=0)[:3]}")  # Should be ~0
    print(f"Std: {np.std(linear_result.X, axis=0, ddof=1)[:3]}")  # Should be ~1
    print(f"Missing filled: {linear_result.fill_stats['n_missing_before']} -> {linear_result.fill_stats['n_missing_after']}")

    # 2. Tree model preparation
    print("\n" + "-" * 70)
    print("2. TREE MODEL READY")
    print("-" * 70)

    tree_config = TreeReadyConfig(
        handle_missing="keep",  # Trees handle NaN natively
        add_missing_indicator=True,
        clip_outliers=False,  # Trees are robust to outliers
    )

    tree_result = build_tree_ready(X, tree_config)
    print(f"Output shape: {tree_result.X.shape}")
    print(f"Has NaN: {np.any(np.isnan(tree_result.X))}")
    print(f"Missing indicators shape: {tree_result.missing_indicators.shape if tree_result.missing_indicators is not None else None}")
    print(f"Data range: [{np.nanmin(tree_result.X):.2f}, {np.nanmax(tree_result.X):.2f}]")
    print(f"Original missing preserved: {tree_result.preprocessing_stats['n_missing_original']}")

    # 3. Neural network preparation
    print("\n" + "-" * 70)
    print("3. NEURAL NETWORK READY")
    print("-" * 70)

    neural_config = NeuralReadyConfig(
        scaling="robust",  # Robust to outliers
        handle_missing="zero",
        clip_outliers=True,
        outlier_quantile_range=(1.0, 99.0),
    )

    neural_result = build_neural_ready(X, neural_config)
    print(f"Output shape: {neural_result.X.shape}")
    print(f"Has NaN: {np.any(np.isnan(neural_result.X))}")
    print(f"Scaling method: {neural_result.scaling_params['method']}")
    print(f"Data range: [{np.min(neural_result.X):.2f}, {np.max(neural_result.X):.2f}]")
    print(f"Values clipped: {neural_result.preprocessing_stats.get('n_values_clipped', 0)}")

    # 4. Embedding preparation for categorical features
    print("\n" + "-" * 70)
    print("4. EMBEDDING PREPARATION (Neural Networks)")
    print("-" * 70)

    # Create sample categorical features (e.g., sector, industry codes)
    categorical_data = np.random.randint(0, 10, size=(n_samples, 3)).astype(float)

    embedding_info = prepare_embeddings(categorical_data, embedding_dim=None)
    print(f"Categorical features: {embedding_info['n_categorical_features']}")

    for i, config in enumerate(embedding_info['embedding_configs']):
        print(f"  Feature {i}: cardinality={config['cardinality']}, "
              f"embedding_dim={config['embedding_dim']}")

    print(f"Total embedding parameters: {embedding_info['total_embedding_params']}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: Key Differences")
    print("=" * 70)
    print(f"{'Policy':<20} {'Missing':<15} {'Scaling':<15} {'Outliers':<15}")
    print("-" * 70)
    print(f"{'Linear':<20} {'Fill (mean)':<15} {'Standard':<15} {'Not handled':<15}")
    print(f"{'Tree':<20} {'Keep (native)':<15} {'None':<15} {'Robust':<15}")
    print(f"{'Neural':<20} {'Fill (zero)':<15} {'Robust':<15} {'Clip':<15}")
    print("=" * 70)


if __name__ == "__main__":
    demo_representation_policies()
