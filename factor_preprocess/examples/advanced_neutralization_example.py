"""
Example demonstrating all advanced neutralization methods.

Shows how to use PCA, robust regression, quantile regression, and kernel
neutralization on the same dataset for comparison.
"""
import pandas as pd
import numpy as np
from factor_preprocess.neutralization.advanced import (
    pca_neutralize,
    huber_neutralize,
    lad_neutralize,
    quantile_neutralize,
    kernel_neutralize,
)
from factor_preprocess.neutralization import ols_neutralize


def create_sample_data(n_dates=5, n_assets=50, with_outliers=True):
    """Create sample factor and exposure data."""
    dates = pd.date_range("2020-01-01", periods=n_dates)
    assets = [f"ASSET_{i:03d}" for i in range(n_assets)]

    values_list = []
    exposures_list = []

    np.random.seed(42)

    for date in dates:
        for i, asset in enumerate(assets):
            # Base factor value with linear relationship to size
            size_exposure = float(i) / n_assets
            base_value = size_exposure * 2.0 + np.random.randn() * 0.5

            # Add outliers to some observations
            if with_outliers and i >= n_assets - 3:
                base_value += 10.0  # Outlier

            values_list.append({
                "date": date,
                "asset_id": asset,
                "value": base_value,
            })

            # Create correlated exposures
            exposures_list.append({
                "date": date,
                "asset_id": asset,
                "size": size_exposure,
                "value_exp": size_exposure * 0.9 + np.random.randn() * 0.1,
                "momentum": np.random.randn() * 0.3,
            })

    return pd.DataFrame(values_list), pd.DataFrame(exposures_list)


def compare_neutralization_methods():
    """Compare all neutralization methods on the same data."""
    print("Creating sample data with outliers...")
    values_df, exposures_df = create_sample_data(n_dates=5, n_assets=50)

    print(f"\nData shape: {len(values_df)} observations")
    print(f"Original value range: [{values_df['value'].min():.2f}, {values_df['value'].max():.2f}]")
    print(f"Original value std: {values_df['value'].std():.2f}")

    min_obs = 20

    # 1. OLS (baseline, not robust to outliers)
    print("\n1. OLS Neutralization (baseline)...")
    ols_residuals = ols_neutralize(values_df, exposures_df, min_observations=min_obs)
    print(f"   Residuals std: {ols_residuals.std():.4f}")
    print(f"   Valid residuals: {ols_residuals.notna().sum()}/{len(ols_residuals)}")

    # 2. PCA (handles collinearity)
    print("\n2. PCA Neutralization (n_components=2)...")
    pca_residuals = pca_neutralize(
        values_df, exposures_df, n_components=2, min_observations=min_obs
    )
    print(f"   Residuals std: {pca_residuals.std():.4f}")
    print(f"   Valid residuals: {pca_residuals.notna().sum()}/{len(pca_residuals)}")

    # 3. Huber (moderate outlier robustness)
    print("\n3. Huber Regression (delta=1.35)...")
    huber_residuals = huber_neutralize(
        values_df, exposures_df, delta=1.35, min_observations=min_obs
    )
    print(f"   Residuals std: {huber_residuals.std():.4f}")
    print(f"   Valid residuals: {huber_residuals.notna().sum()}/{len(huber_residuals)}")

    # 4. LAD (high outlier robustness)
    print("\n4. LAD Regression (median)...")
    lad_residuals = lad_neutralize(values_df, exposures_df, min_observations=min_obs)
    print(f"   Residuals std: {lad_residuals.std():.4f}")
    print(f"   Valid residuals: {lad_residuals.notna().sum()}/{len(lad_residuals)}")

    # 5. Quantile regression (75th percentile)
    print("\n5. Quantile Regression (tau=0.75)...")
    quantile_residuals = quantile_neutralize(
        values_df, exposures_df, quantile=0.75, min_observations=min_obs
    )
    print(f"   Residuals std: {quantile_residuals.std():.4f}")
    print(f"   Valid residuals: {quantile_residuals.notna().sum()}/{len(quantile_residuals)}")

    # 6. Kernel regression (non-parametric)
    print("\n6. Kernel Regression (Gaussian, auto bandwidth)...")
    kernel_residuals = kernel_neutralize(
        values_df,
        exposures_df,
        kernel="gaussian",
        bandwidth=None,
        min_observations=min_obs,
    )
    print(f"   Residuals std: {kernel_residuals.std():.4f}")
    print(f"   Valid residuals: {kernel_residuals.notna().sum()}/{len(kernel_residuals)}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY: Residual Standard Deviations")
    print("=" * 60)
    print(f"Original values:      {values_df['value'].std():.4f}")
    print(f"OLS:                  {ols_residuals.std():.4f}")
    print(f"PCA (2 components):   {pca_residuals.std():.4f}")
    print(f"Huber:                {huber_residuals.std():.4f}")
    print(f"LAD:                  {lad_residuals.std():.4f}")
    print(f"Quantile (0.75):      {quantile_residuals.std():.4f}")
    print(f"Kernel (Gaussian):    {kernel_residuals.std():.4f}")

    # Check outlier handling
    print("\n" + "=" * 60)
    print("OUTLIER ROBUSTNESS: Max Absolute Residual")
    print("=" * 60)
    print(f"OLS:                  {ols_residuals.abs().max():.4f}")
    print(f"PCA:                  {pca_residuals.abs().max():.4f}")
    print(f"Huber:                {huber_residuals.abs().max():.4f}")
    print(f"LAD:                  {lad_residuals.abs().max():.4f}")
    print(f"Quantile:             {quantile_residuals.abs().max():.4f}")
    print(f"Kernel:               {kernel_residuals.abs().max():.4f}")

    print("\n✓ All methods completed successfully!")


if __name__ == "__main__":
    compare_neutralization_methods()
