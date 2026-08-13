"""
Benchmark script comparing polars backend vs pandas/numpy reference.

Validates 3-5x speedup target for large panels.
"""
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Reference implementations
from factor_preprocess.transforms.cross_sectional import (
    cs_rank,
    cs_zscore,
    cs_demean,
    cs_winsor,
)
from factor_preprocess.neutralization.ols import ols_neutralize

# Polars backend
try:
    from factor_preprocess.backends.polars_backend import (
        cs_rank_polars,
        cs_zscore_polars,
        cs_demean_polars,
        cs_winsor_polars,
        ols_neutralize_polars,
        POLARS_AVAILABLE,
    )
except ImportError:
    POLARS_AVAILABLE = False


def generate_panel_data(n_dates: int, n_assets: int, seed: int = 42):
    """Generate synthetic panel data."""
    np.random.seed(seed)

    dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    asset_ids = [f"ASSET_{i:04d}" for i in range(n_assets)]

    df = pd.DataFrame({
        "date": [d for d in dates for _ in asset_ids],
        "asset_id": asset_ids * n_dates,
        "value": np.random.randn(n_dates * n_assets) * 10 + 100,
    })

    # Add 10% missing data
    missing_mask = np.random.rand(len(df)) < 0.1
    df.loc[missing_mask, "value"] = np.nan

    return df


def benchmark_rank(df):
    """Benchmark cs_rank."""
    print(f"\n{'='*70}")
    print(f"Benchmarking cs_rank on {len(df):,} rows ({df['date'].nunique()} dates × {df['asset_id'].nunique()} assets)")
    print(f"{'='*70}")

    # Reference implementation
    start = time.perf_counter()
    ref_result = []
    for date, group in df.groupby("date"):
        ranks = cs_rank(group["value"].values)
        ref_result.extend(ranks)
    ref_time = time.perf_counter() - start
    print(f"Reference (numpy):     {ref_time:.4f}s")

    # Polars backend
    if POLARS_AVAILABLE:
        start = time.perf_counter()
        polars_result = cs_rank_polars(df, value_col="value", group_col="date")
        polars_time = time.perf_counter() - start
        print(f"Polars backend:        {polars_time:.4f}s")

        speedup = ref_time / polars_time
        print(f"Speedup:               {speedup:.2f}x")

        # Verify correctness
        np.testing.assert_allclose(
            ref_result,
            polars_result.values,
            rtol=1e-10,
            equal_nan=True,
        )
        print("✓ Results match")
    else:
        print("Polars backend not available")


def benchmark_zscore(df):
    """Benchmark cs_zscore."""
    print(f"\n{'='*70}")
    print(f"Benchmarking cs_zscore on {len(df):,} rows")
    print(f"{'='*70}")

    # Reference implementation
    start = time.perf_counter()
    ref_result = []
    for date, group in df.groupby("date"):
        zscores = cs_zscore(group["value"].values)
        ref_result.extend(zscores)
    ref_time = time.perf_counter() - start
    print(f"Reference (numpy):     {ref_time:.4f}s")

    # Polars backend
    if POLARS_AVAILABLE:
        start = time.perf_counter()
        polars_result = cs_zscore_polars(df, value_col="value", group_col="date")
        polars_time = time.perf_counter() - start
        print(f"Polars backend:        {polars_time:.4f}s")

        speedup = ref_time / polars_time
        print(f"Speedup:               {speedup:.2f}x")

        # Verify correctness
        np.testing.assert_allclose(
            ref_result,
            polars_result.values,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
        print("✓ Results match")
    else:
        print("Polars backend not available")


def benchmark_demean(df):
    """Benchmark cs_demean."""
    print(f"\n{'='*70}")
    print(f"Benchmarking cs_demean on {len(df):,} rows")
    print(f"{'='*70}")

    # Reference implementation
    start = time.perf_counter()
    ref_result = []
    for date, group in df.groupby("date"):
        demeaned = cs_demean(group["value"].values)
        ref_result.extend(demeaned)
    ref_time = time.perf_counter() - start
    print(f"Reference (numpy):     {ref_time:.4f}s")

    # Polars backend
    if POLARS_AVAILABLE:
        start = time.perf_counter()
        polars_result = cs_demean_polars(df, value_col="value", group_col="date")
        polars_time = time.perf_counter() - start
        print(f"Polars backend:        {polars_time:.4f}s")

        speedup = ref_time / polars_time
        print(f"Speedup:               {speedup:.2f}x")

        # Verify correctness
        np.testing.assert_allclose(
            ref_result,
            polars_result.values,
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
        print("✓ Results match")
    else:
        print("Polars backend not available")


def benchmark_winsor(df):
    """Benchmark cs_winsor."""
    print(f"\n{'='*70}")
    print(f"Benchmarking cs_winsor on {len(df):,} rows")
    print(f"{'='*70}")

    # Reference implementation
    start = time.perf_counter()
    ref_result = []
    for date, group in df.groupby("date"):
        winsorized = cs_winsor(group["value"].values, lower=0.05, upper=0.95)
        ref_result.extend(winsorized)
    ref_time = time.perf_counter() - start
    print(f"Reference (numpy):     {ref_time:.4f}s")

    # Polars backend
    if POLARS_AVAILABLE:
        start = time.perf_counter()
        polars_result = cs_winsor_polars(
            df, value_col="value", group_col="date", lower=0.05, upper=0.95
        )
        polars_time = time.perf_counter() - start
        print(f"Polars backend:        {polars_time:.4f}s")

        speedup = ref_time / polars_time
        print(f"Speedup:               {speedup:.2f}x")

        # Verify correctness
        np.testing.assert_allclose(
            ref_result,
            polars_result.values,
            rtol=1e-10,
            equal_nan=True,
        )
        print("✓ Results match")
    else:
        print("Polars backend not available")


def benchmark_neutralization(df_values, df_exposures):
    """Benchmark ols_neutralize."""
    print(f"\n{'='*70}")
    print(f"Benchmarking ols_neutralize on {len(df_values):,} rows")
    print(f"{'='*70}")

    # Reference implementation
    start = time.perf_counter()
    ref_result = ols_neutralize(
        df_values,
        df_exposures,
        date_col="date",
        asset_col="asset_id",
        value_col="value",
    )
    ref_time = time.perf_counter() - start
    print(f"Reference (pandas):    {ref_time:.4f}s")

    # Polars backend
    if POLARS_AVAILABLE:
        start = time.perf_counter()
        polars_result = ols_neutralize_polars(
            df_values,
            df_exposures,
            date_col="date",
            asset_col="asset_id",
            value_col="value",
        )
        polars_time = time.perf_counter() - start
        print(f"Polars backend:        {polars_time:.4f}s")

        speedup = ref_time / polars_time
        print(f"Speedup:               {speedup:.2f}x")

        # Verify correctness
        np.testing.assert_allclose(
            ref_result.values,
            polars_result.values,
            rtol=1e-8,
            atol=1e-8,
            equal_nan=True,
        )
        print("✓ Results match")
    else:
        print("Polars backend not available")


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("Polars Backend Performance Benchmarks")
    print("=" * 70)
    print(f"Target: 3-5x speedup over pandas/numpy reference")
    print()

    if not POLARS_AVAILABLE:
        print("ERROR: Polars not available. Install with: pip install polars")
        return

    # Small panel (warm-up)
    print("\nWarm-up run (small panel)...")
    df_small = generate_panel_data(n_dates=10, n_assets=100)
    benchmark_rank(df_small)

    # Medium panel (252 trading days × 500 assets)
    print("\n" + "=" * 70)
    print("MEDIUM PANEL: 252 dates × 500 assets = 126,000 rows")
    print("=" * 70)
    df_medium = generate_panel_data(n_dates=252, n_assets=500)

    benchmark_rank(df_medium)
    benchmark_zscore(df_medium)
    benchmark_demean(df_medium)
    benchmark_winsor(df_medium)

    # Neutralization benchmark
    df_exposures = df_medium[["date", "asset_id"]].copy()
    df_exposures["market_cap"] = np.random.randn(len(df_exposures)) * 1e9 + 5e9
    df_exposures["sector"] = np.random.choice([0, 1, 2], size=len(df_exposures))
    benchmark_neutralization(df_medium, df_exposures)

    # Large panel (252 trading days × 2000 assets)
    print("\n" + "=" * 70)
    print("LARGE PANEL: 252 dates × 2000 assets = 504,000 rows")
    print("=" * 70)
    df_large = generate_panel_data(n_dates=252, n_assets=2000)

    benchmark_rank(df_large)
    benchmark_zscore(df_large)
    benchmark_demean(df_large)

    print("\n" + "=" * 70)
    print("Benchmark complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
