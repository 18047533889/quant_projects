"""
Example usage of polars backend for factor preprocessing.

Demonstrates cross-sectional transforms and neutralization with polars.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from factor_preprocess.backends.polars_backend import (
    cs_rank_polars,
    cs_zscore_polars,
    cs_demean_polars,
    cs_winsor_polars,
    ols_neutralize_polars,
)


def example_basic_transforms():
    """Example: Basic cross-sectional transforms."""
    print("=" * 70)
    print("Example: Basic Cross-Sectional Transforms")
    print("=" * 70)

    # Generate sample panel data
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    assets = [f"ASSET_{i}" for i in range(10)]

    df = pd.DataFrame({
        "date": [d for d in dates for _ in assets],
        "asset_id": assets * len(dates),
        "factor_value": np.random.randn(len(dates) * len(assets)) * 10 + 100,
    })

    print(f"\nInput data: {len(df)} rows, {df['date'].nunique()} dates, {df['asset_id'].nunique()} assets")
    print(df.head(10))

    # Apply transforms
    print("\n1. Cross-sectional rank (percentile):")
    df["rank_pct"] = cs_rank_polars(df, value_col="factor_value", group_col="date", pct=True)
    print(df[df["date"] == dates[0]][["asset_id", "factor_value", "rank_pct"]].head())

    print("\n2. Cross-sectional z-score:")
    df["zscore"] = cs_zscore_polars(df, value_col="factor_value", group_col="date")
    print(df[df["date"] == dates[0]][["asset_id", "factor_value", "zscore"]].head())

    print("\n3. Cross-sectional demean:")
    df["demeaned"] = cs_demean_polars(df, value_col="factor_value", group_col="date")
    print(df[df["date"] == dates[0]][["asset_id", "factor_value", "demeaned"]].head())

    print("\n4. Cross-sectional winsorization:")
    df["winsorized"] = cs_winsor_polars(
        df, value_col="factor_value", group_col="date", lower=0.1, upper=0.9
    )
    print(df[df["date"] == dates[0]][["asset_id", "factor_value", "winsorized"]].head())

    print("\n✓ All transforms completed successfully")


def example_neutralization():
    """Example: OLS neutralization against exposures."""
    print("\n" + "=" * 70)
    print("Example: OLS Neutralization")
    print("=" * 70)

    # Generate sample panel data
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    assets = [f"ASSET_{i}" for i in range(50)]

    df_values = pd.DataFrame({
        "date": [d for d in dates for _ in assets],
        "asset_id": assets * len(dates),
        "value": np.random.randn(len(dates) * len(assets)) * 10 + 100,
    })

    # Generate exposures (e.g., market cap, sector)
    np.random.seed(42)
    df_exposures = df_values[["date", "asset_id"]].copy()
    df_exposures["market_cap"] = np.random.lognormal(20, 1, size=len(df_exposures))
    df_exposures["sector"] = np.random.choice([0, 1, 2], size=len(df_exposures))

    print(f"\nInput: {len(df_values)} observations")
    print(f"Exposures: market_cap (continuous), sector (categorical)")

    # Neutralize
    residuals = ols_neutralize_polars(
        df_values,
        df_exposures,
        date_col="date",
        asset_col="asset_id",
        value_col="value",
        add_intercept=True,
    )

    df_values["residual"] = residuals

    # Show sample results
    sample_date = dates[0]
    print(f"\nSample results for {sample_date.date()}:")
    print(df_values[df_values["date"] == sample_date][["asset_id", "value", "residual"]].head(10))

    # Verify orthogonality
    sample_df = df_values[df_values["date"] == sample_date].merge(
        df_exposures[df_exposures["date"] == sample_date],
        on=["date", "asset_id"],
    )

    corr_original = sample_df[["value", "market_cap"]].corr().iloc[0, 1]
    corr_residual = sample_df[["residual", "market_cap"]].corr().iloc[0, 1]

    print(f"\nCorrelation with market_cap:")
    print(f"  Original factor:  {corr_original:.4f}")
    print(f"  Neutralized:      {corr_residual:.4f}")
    print(f"  Reduction:        {abs(corr_original) - abs(corr_residual):.4f}")

    print("\n✓ Neutralization completed successfully")


def example_pipeline():
    """Example: Complete preprocessing pipeline."""
    print("\n" + "=" * 70)
    print("Example: Complete Preprocessing Pipeline")
    print("=" * 70)

    # Generate sample data
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    assets = [f"ASSET_{i}" for i in range(100)]

    df = pd.DataFrame({
        "date": [d for d in dates for _ in assets],
        "asset_id": assets * len(dates),
        "raw_factor": np.random.randn(len(dates) * len(assets)) * 10 + 100,
    })

    # Add some outliers
    outlier_mask = np.random.rand(len(df)) < 0.05
    df.loc[outlier_mask, "raw_factor"] *= 3

    print(f"\nInput: {len(df)} observations with outliers")
    print(f"Raw factor stats: mean={df['raw_factor'].mean():.2f}, std={df['raw_factor'].std():.2f}")

    # Pipeline: winsorize -> demean -> zscore
    print("\nApplying pipeline: Winsorize → Demean → Z-score")

    # Step 1: Winsorize outliers
    df["step1_winsor"] = cs_winsor_polars(
        df, value_col="raw_factor", group_col="date", lower=0.01, upper=0.99
    )

    # Step 2: Demean
    df["step2_demean"] = cs_demean_polars(
        df, value_col="step1_winsor", group_col="date"
    )

    # Step 3: Z-score
    df["step3_zscore"] = cs_zscore_polars(
        df, value_col="step2_demean", group_col="date"
    )

    # Final percentile rank
    df["final_rank"] = cs_rank_polars(
        df, value_col="step3_zscore", group_col="date", pct=True
    )

    # Show sample results
    sample_date = dates[0]
    print(f"\nSample results for {sample_date.date()}:")
    sample = df[df["date"] == sample_date].sort_values("final_rank", ascending=False).head(5)
    print(sample[["asset_id", "raw_factor", "step1_winsor", "step2_demean", "step3_zscore", "final_rank"]])

    # Verify final distribution
    print(f"\nFinal zscore distribution:")
    print(f"  Mean:  {df['step3_zscore'].mean():.6f}")
    print(f"  Std:   {df['step3_zscore'].std():.6f}")
    print(f"  Min:   {df['step3_zscore'].min():.2f}")
    print(f"  Max:   {df['step3_zscore'].max():.2f}")

    print("\n✓ Pipeline completed successfully")


def main():
    """Run all examples."""
    print("Polars Backend Usage Examples")
    print("=" * 70)

    example_basic_transforms()
    example_neutralization()
    example_pipeline()

    print("\n" + "=" * 70)
    print("All examples completed successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()
