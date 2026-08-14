#!/usr/bin/env python3
"""
Factor Preprocess Transform Benchmarks

Tests preprocessing transform performance:
- Neutralization (industry, market)
- Standardization (zscore, rank)
- Winsorization
- Missing value handling
- Backend comparison (pandas vs polars)

Measures:
- Throughput (cells/sec)
- Latency per transform
- Memory efficiency
- Backend speedup
"""

import gc
import time
import traceback
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    from factor_preprocess.neutralization.ols import neutralize_ols
    from factor_preprocess.standardization import standardize_zscore, standardize_rank
    FP_AVAILABLE = True
except ImportError:
    FP_AVAILABLE = False


def generate_panel_data(
    n_assets: int,
    n_dates: int,
    n_factors: int = 1,
    seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic panel data."""
    np.random.seed(seed)

    dates = pd.date_range('2020-01-01', periods=n_dates, freq='D')
    assets = [f"asset_{i:05d}" for i in range(n_assets)]

    # Generate factor values
    factor_values = np.random.randn(n_dates, n_assets, n_factors).astype(np.float32)

    # Add some structure
    for f in range(n_factors):
        # Time trend
        trend = np.linspace(-0.5, 0.5, n_dates).reshape(-1, 1)
        factor_values[:, :, f] += trend * 0.2

        # Asset effects
        asset_effects = np.random.randn(n_assets) * 0.3
        factor_values[:, :, f] += asset_effects

    # Add missing values (5%)
    mask = np.random.random((n_dates, n_assets, n_factors)) < 0.05
    factor_values[mask] = np.nan

    # Create DataFrame
    data = []
    for t_idx, date in enumerate(dates):
        for a_idx, asset in enumerate(assets):
            row = {'date': date, 'asset': asset}
            for f in range(n_factors):
                row[f'factor_{f}'] = factor_values[t_idx, a_idx, f]
            # Add industry
            row['industry'] = f"ind_{a_idx % 10}"
            data.append(row)

    df = pd.DataFrame(data)

    return df, factor_values


def benchmark_zscore_pandas(df: pd.DataFrame, factor_cols: list) -> Dict[str, Any]:
    """Benchmark zscore standardization (pandas)."""
    try:
        gc.collect()
        start = time.perf_counter()

        result = df.copy()
        for col in factor_cols:
            grouped = result.groupby('date')[col]
            mean = grouped.transform('mean')
            std = grouped.transform('std')
            result[f'{col}_zscore'] = (result[col] - mean) / (std + 1e-8)

        elapsed = time.perf_counter() - start

        n_cells = len(df) * len(factor_cols)
        throughput = n_cells / elapsed

        return {
            "backend": "pandas",
            "operation": "zscore",
            "n_cells": n_cells,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_cells_per_sec": round(throughput, 0),
        }
    except Exception as e:
        return {"backend": "pandas", "operation": "zscore", "error": str(e)}


def benchmark_zscore_polars(df: pd.DataFrame, factor_cols: list) -> Dict[str, Any]:
    """Benchmark zscore standardization (polars)."""
    if not POLARS_AVAILABLE:
        return {"backend": "polars", "operation": "zscore", "error": "polars not available"}

    try:
        # Convert to polars
        pl_df = pl.from_pandas(df)

        gc.collect()
        start = time.perf_counter()

        for col in factor_cols:
            pl_df = pl_df.with_columns([
                ((pl.col(col) - pl.col(col).mean().over('date')) /
                 (pl.col(col).std().over('date') + 1e-8)).alias(f'{col}_zscore')
            ])

        result = pl_df.to_pandas()
        elapsed = time.perf_counter() - start

        n_cells = len(df) * len(factor_cols)
        throughput = n_cells / elapsed

        return {
            "backend": "polars",
            "operation": "zscore",
            "n_cells": n_cells,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_cells_per_sec": round(throughput, 0),
        }
    except Exception as e:
        return {"backend": "polars", "operation": "zscore", "error": str(e)}


def benchmark_rank_pandas(df: pd.DataFrame, factor_cols: list) -> Dict[str, Any]:
    """Benchmark rank standardization (pandas)."""
    try:
        gc.collect()
        start = time.perf_counter()

        result = df.copy()
        for col in factor_cols:
            result[f'{col}_rank'] = result.groupby('date')[col].rank(pct=True)

        elapsed = time.perf_counter() - start

        n_cells = len(df) * len(factor_cols)
        throughput = n_cells / elapsed

        return {
            "backend": "pandas",
            "operation": "rank",
            "n_cells": n_cells,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_cells_per_sec": round(throughput, 0),
        }
    except Exception as e:
        return {"backend": "pandas", "operation": "rank", "error": str(e)}


def benchmark_neutralization_pandas(df: pd.DataFrame, factor_col: str) -> Dict[str, Any]:
    """Benchmark industry neutralization (pandas)."""
    try:
        gc.collect()
        start = time.perf_counter()

        result = df.copy()

        # Simple industry neutralization
        for date in df['date'].unique():
            date_mask = result['date'] == date
            date_df = result[date_mask].copy()

            # Compute industry means
            industry_means = date_df.groupby('industry')[factor_col].transform('mean')

            # Subtract industry mean
            result.loc[date_mask, f'{factor_col}_neutral'] = (
                date_df[factor_col] - industry_means
            )

        elapsed = time.perf_counter() - start

        n_cells = len(df)
        throughput = n_cells / elapsed

        return {
            "backend": "pandas",
            "operation": "neutralization",
            "n_cells": n_cells,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_cells_per_sec": round(throughput, 0),
        }
    except Exception as e:
        return {"backend": "pandas", "operation": "neutralization", "error": str(e)}


def benchmark_winsorization_pandas(df: pd.DataFrame, factor_cols: list, limits: Tuple[float, float] = (0.01, 0.99)) -> Dict[str, Any]:
    """Benchmark winsorization (pandas)."""
    try:
        gc.collect()
        start = time.perf_counter()

        result = df.copy()
        for col in factor_cols:
            grouped = result.groupby('date')[col]
            lower = grouped.transform(lambda x: x.quantile(limits[0]))
            upper = grouped.transform(lambda x: x.quantile(limits[1]))
            result[f'{col}_winsor'] = result[col].clip(lower=lower, upper=upper)

        elapsed = time.perf_counter() - start

        n_cells = len(df) * len(factor_cols)
        throughput = n_cells / elapsed

        return {
            "backend": "pandas",
            "operation": "winsorization",
            "n_cells": n_cells,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_cells_per_sec": round(throughput, 0),
        }
    except Exception as e:
        return {"backend": "pandas", "operation": "winsorization", "error": str(e)}


def benchmark_scale(scale_name: str, n_assets: int, n_dates: int, n_factors: int = 3) -> Dict[str, Any]:
    """Run all benchmarks at given scale."""

    print(f"\n  Generating data: {n_assets} assets × {n_dates} dates × {n_factors} factors")

    df, _ = generate_panel_data(n_assets, n_dates, n_factors)
    factor_cols = [f'factor_{i}' for i in range(n_factors)]

    total_cells = n_assets * n_dates * n_factors
    print(f"    Total cells: {total_cells:,}")

    results = {}

    # Zscore (pandas)
    print(f"    Running zscore (pandas)...")
    results['zscore_pandas'] = benchmark_zscore_pandas(df, factor_cols)

    # Zscore (polars)
    if POLARS_AVAILABLE:
        print(f"    Running zscore (polars)...")
        results['zscore_polars'] = benchmark_zscore_polars(df, factor_cols)

    # Rank
    print(f"    Running rank (pandas)...")
    results['rank_pandas'] = benchmark_rank_pandas(df, factor_cols)

    # Neutralization (single factor)
    print(f"    Running neutralization (pandas)...")
    results['neutralization_pandas'] = benchmark_neutralization_pandas(df, factor_cols[0])

    # Winsorization
    print(f"    Running winsorization (pandas)...")
    results['winsorization_pandas'] = benchmark_winsorization_pandas(df, factor_cols)

    return {
        "scale": scale_name,
        "n_assets": n_assets,
        "n_dates": n_dates,
        "n_factors": n_factors,
        "total_cells": total_cells,
        "operations": results
    }


def main():
    """Run all factor preprocess benchmarks."""

    print("=" * 70)
    print("FACTOR PREPROCESS TRANSFORM BENCHMARKS")
    print("=" * 70)

    scales = [
        ("small", 100, 252, 3),
        ("medium", 1000, 252, 3),
        ("large", 3000, 504, 5),
    ]

    results = {}
    total_start = time.perf_counter()

    for scale_name, n_assets, n_dates, n_factors in scales:
        print(f"\n{'='*70}")
        print(f"Scale: {scale_name.upper()}")
        print(f"{'='*70}")

        result = benchmark_scale(scale_name, n_assets, n_dates, n_factors)
        results[scale_name] = result

        # Print summary
        print(f"\n  Summary:")
        for op_name, op_result in result['operations'].items():
            if 'error' not in op_result:
                throughput_mcells = op_result['throughput_cells_per_sec'] / 1e6
                print(f"    {op_name}: {throughput_mcells:.2f} Mcells/s")
            else:
                print(f"    {op_name}: ERROR")

    total_elapsed = time.perf_counter() - total_start

    return {
        "benchmark": "factor_preprocess",
        "description": "Preprocessing transform performance",
        "results": results,
        "total_time_s": round(total_elapsed, 2)
    }


if __name__ == "__main__":
    import json
    result = main()
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(json.dumps(result, indent=2))
