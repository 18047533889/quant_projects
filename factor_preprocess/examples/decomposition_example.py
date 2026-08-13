"""
Example usage of time series decomposition transforms.
"""
import pandas as pd
import numpy as np
from factor_preprocess.transforms.decomposition import (
    hp_filter,
    hp_decompose,
    stl_decompose,
    bandpass_filter,
    extract_cycle,
    wavelet_decompose,
    wavelet_smooth,
)


def example_hp_filter():
    """Example: HP filter for trend extraction."""
    # Create sample data with trend + cycle
    n = 200
    t = np.arange(n)
    trend = 0.5 * t
    cycle = 10 * np.sin(2 * np.pi * t / 40)

    df = pd.DataFrame({
        "asset_id": ["AAPL"] * n,
        "date": pd.date_range("2020-01-01", periods=n),
        "value": trend + cycle,
    })

    # Extract smooth trend
    trend_component = hp_filter(df, lambda_param=1600)

    # Full decomposition
    trend_hp, cycle_hp = hp_decompose(df, lambda_param=1600)

    print("HP Filter Example:")
    print(f"  Input shape: {df.shape}")
    print(f"  Trend extracted: {trend_component.notna().sum()} valid observations")
    print(f"  First value (NaN): {pd.isna(trend_component.iloc[0])}")
    print()


def example_stl_decomposition():
    """Example: STL seasonal decomposition."""
    # Create monthly data with trend + seasonality
    n = 120  # 10 years
    t = np.arange(n)
    trend = 0.1 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 12)

    df = pd.DataFrame({
        "asset_id": ["SPY"] * n,
        "date": pd.date_range("2020-01-01", periods=n, freq='MS'),
        "value": trend + seasonal + np.random.randn(n) * 0.5,
    })

    # Decompose into trend + seasonal + residual
    trend, seasonal, residual = stl_decompose(df, period=12, seasonal=7)

    print("STL Decomposition Example:")
    print(f"  Input shape: {df.shape}")
    print(f"  Trend: {trend.notna().sum()} valid")
    print(f"  Seasonal: {seasonal.notna().sum()} valid")
    print(f"  Residual: {residual.notna().sum()} valid")
    print()


def example_cycle_extraction():
    """Example: Business cycle extraction."""
    # Create quarterly data with business cycle
    n = 200  # 50 years
    t = np.arange(n)
    # Business cycle: 6 year period = 24 quarters
    business_cycle = 10 * np.sin(2 * np.pi * t / 24)
    noise = np.random.randn(n) * 2

    df = pd.DataFrame({
        "asset_id": ["GDP"] * n,
        "date": pd.date_range("2000-01-01", periods=n, freq='QS'),
        "value": business_cycle + noise,
    })

    # Extract 2-8 year cycles (8-32 quarters)
    cycle = extract_cycle(df, low_period=8, high_period=32)

    print("Cycle Extraction Example:")
    print(f"  Input shape: {df.shape}")
    print(f"  Business cycle extracted: {cycle.notna().sum()} valid observations")
    print()


def example_wavelet_decomposition():
    """Example: Wavelet decomposition and smoothing."""
    n = 128
    t = np.arange(n)
    signal = 10 * np.sin(2 * np.pi * t / 32)
    noise = np.random.randn(n) * 2

    df = pd.DataFrame({
        "asset_id": ["TSLA"] * n,
        "date": pd.date_range("2020-01-01", periods=n),
        "value": signal + noise,
    })

    # Multi-level decomposition
    components = wavelet_decompose(df, wavelet='db4', level=3)

    # Smooth using approximation
    smoothed = wavelet_smooth(df, wavelet='db4', level=3)

    print("Wavelet Decomposition Example:")
    print(f"  Input shape: {df.shape}")
    print(f"  Components extracted: {list(components.keys())}")
    print(f"  Approximation (a3): {components['a3'].notna().sum()} valid")
    print(f"  Smoothed signal: {smoothed.notna().sum()} valid")
    print()


def example_multi_asset():
    """Example: Processing multiple assets."""
    n = 100
    assets = ["AAPL", "GOOGL", "MSFT"]

    dfs = []
    for asset in assets:
        t = np.arange(n)
        value = np.random.randn(n).cumsum()
        dfs.append(pd.DataFrame({
            "asset_id": asset,
            "date": pd.date_range("2020-01-01", periods=n),
            "value": value,
        }))

    df = pd.concat(dfs, ignore_index=True)

    # HP filter processes each asset independently
    trend = hp_filter(df, lambda_param=1600)

    print("Multi-Asset Example:")
    print(f"  Total shape: {df.shape}")
    print(f"  Assets: {df['asset_id'].unique().tolist()}")
    print(f"  Trend extracted per asset:")
    for asset in assets:
        asset_mask = df['asset_id'] == asset
        valid = trend[asset_mask].notna().sum()
        print(f"    {asset}: {valid} valid observations")
    print()


if __name__ == "__main__":
    print("Time Series Decomposition Examples\n")
    print("=" * 50)
    print()

    example_hp_filter()
    example_stl_decomposition()
    example_cycle_extraction()
    example_wavelet_decomposition()
    example_multi_asset()

    print("=" * 50)
    print("\nAll decomposition methods are:")
    print("  ✓ Causal (use only past data via shift(1))")
    print("  ✓ Multi-asset (process assets independently)")
    print("  ✓ NaN-safe (propagate NaN appropriately)")
    print("  ✓ Validated (check sorting, reject invalid params)")
