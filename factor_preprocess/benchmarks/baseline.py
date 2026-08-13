"""
Baseline performance benchmarks for factor preprocessing.

Run with: python benchmarks/baseline.py
"""
import time
import numpy as np
import pandas as pd
from factor_preprocess.transforms.cross_sectional import cs_rank, cs_zscore, cs_winsor
from factor_preprocess.transforms.rolling import rolling_mean, rolling_std


def benchmark_cs_rank():
    """Benchmark cross-sectional rank."""
    values = np.random.randn(10000)
    start = time.perf_counter()
    for _ in range(100):
        cs_rank(values)
    elapsed = time.perf_counter() - start
    print(f"cs_rank (10k values, 100 iterations): {elapsed:.3f}s")


def benchmark_cs_zscore():
    """Benchmark cross-sectional z-score."""
    values = np.random.randn(10000)
    start = time.perf_counter()
    for _ in range(100):
        cs_zscore(values)
    elapsed = time.perf_counter() - start
    print(f"cs_zscore (10k values, 100 iterations): {elapsed:.3f}s")


def benchmark_rolling_mean():
    """Benchmark rolling mean."""
    n_assets = 100
    n_dates = 252
    df = pd.DataFrame({
        "asset_id": np.repeat(np.arange(n_assets), n_dates),
        "date": pd.date_range("2020-01-01", periods=n_dates).tolist() * n_assets,
        "value": np.random.randn(n_assets * n_dates),
    })

    start = time.perf_counter()
    rolling_mean(df, window=20)
    elapsed = time.perf_counter() - start
    print(f"rolling_mean ({n_assets} assets × {n_dates} dates, window=20): {elapsed:.3f}s")


def benchmark_rolling_std():
    """Benchmark rolling std."""
    n_assets = 100
    n_dates = 252
    df = pd.DataFrame({
        "asset_id": np.repeat(np.arange(n_assets), n_dates),
        "date": pd.date_range("2020-01-01", periods=n_dates).tolist() * n_assets,
        "value": np.random.randn(n_assets * n_dates),
    })

    start = time.perf_counter()
    rolling_std(df, window=20)
    elapsed = time.perf_counter() - start
    print(f"rolling_std ({n_assets} assets × {n_dates} dates, window=20): {elapsed:.3f}s")


if __name__ == "__main__":
    print("Factor Preprocess Baseline Benchmarks")
    print("=" * 60)
    print()

    print("Cross-sectional transforms:")
    benchmark_cs_rank()
    benchmark_cs_zscore()
    print()

    print("Rolling transforms:")
    benchmark_rolling_mean()
    benchmark_rolling_std()
    print()

    print("Baseline benchmarks complete.")
    print("These establish reference performance for regression detection.")
