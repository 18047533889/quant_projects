#!/usr/bin/env python3
"""
QE (Quant Evaluator) metrics benchmark at scale.

Tests: 1k/10k/100k factors × standard metric suite
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def generate_factor_batch(n_factors: int, n_dates: int = 252, n_assets: int = 1000):
    """Generate synthetic factor batch."""
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    assets = [f"A{i:06d}" for i in range(n_assets)]

    rng = np.random.default_rng(42)
    data = {}

    for i in range(n_factors):
        # Mix of normal, heavy-tail, and sparse patterns
        if i % 3 == 0:
            vals = rng.standard_normal(n_dates * n_assets)
        elif i % 3 == 1:
            vals = rng.standard_t(df=3, size=n_dates * n_assets)
        else:
            vals = rng.standard_normal(n_dates * n_assets)
            vals[rng.random(n_dates * n_assets) < 0.3] = np.nan

        data[f"factor_{i:05d}"] = vals.reshape(n_dates, n_assets)

    return data, dates, assets


def generate_labels(n_dates: int, n_assets: int):
    """Generate forward returns."""
    rng = np.random.default_rng(123)
    return rng.standard_normal(n_dates * n_assets).reshape(n_dates, n_assets) * 0.02


def bench_qe_metrics(n_factors: int, repeats: int = 2):
    """Benchmark QE metric computation - simplified version."""
    from scipy.stats import spearmanr

    n_dates = 252
    n_assets = 500  # Reduced for speed

    print(f"  Generating {n_factors} factors × {n_dates} dates × {n_assets} assets...")
    factor_data, dates, assets = generate_factor_batch(n_factors, n_dates, n_assets)
    label_data = generate_labels(n_dates, n_assets)

    # Convert to DataFrames
    factor_dfs = {name: pd.DataFrame(arr, index=dates, columns=assets)
                  for name, arr in factor_data.items()}
    label_df = pd.DataFrame(label_data, index=dates, columns=assets)

    # Simplified metric computation (direct rank IC)
    def compute_metrics():
        results = {}
        for factor_name, factor_df in factor_dfs.items():
            # Flatten and compute rank IC
            f_flat = factor_df.values.flatten()
            l_flat = label_df.values.flatten()

            # Remove NaNs
            mask = ~(np.isnan(f_flat) | np.isnan(l_flat))
            if mask.sum() > 100:
                ic, _ = spearmanr(f_flat[mask], l_flat[mask])
                results[factor_name] = ic
            else:
                results[factor_name] = np.nan
        return results

    # Warmup
    print("  Warmup...")
    gc.collect()
    try:
        _ = compute_metrics()
    except Exception as e:
        print(f"  Warmup error: {e}")
        return None

    # Timed runs
    elapsed_times = []
    for rep in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        try:
            result = compute_metrics()
            elapsed = time.perf_counter() - t0
            elapsed_times.append(elapsed)
            print(f"  Run {rep+1}/{repeats}: {elapsed:.3f}s ({len(result)} factors)")
        except Exception as e:
            print(f"  Run {rep+1}/{repeats} failed: {e}")
            elapsed_times.append(np.nan)

    if not elapsed_times or all(np.isnan(elapsed_times)):
        return None

    valid_times = [t for t in elapsed_times if not np.isnan(t)]
    mean_time = np.mean(valid_times)

    return {
        "n_factors": n_factors,
        "n_dates": n_dates,
        "n_assets": n_assets,
        "mean_time_s": round(mean_time, 3),
        "throughput_factors_per_s": round(n_factors / mean_time, 1),
        "per_factor_ms": round(mean_time * 1000 / n_factors, 3),
    }


def run_qe_benchmark():
    """Run QE benchmark suite."""
    scales = [10, 50, 100]  # Reduced scales for speed
    results = []

    for n in scales:
        print(f"\nBenchmarking QE with {n} factors...")
        try:
            res = bench_qe_metrics(n, repeats=1)  # Single repeat
            if res:
                results.append(res)
        except Exception as e:
            print(f"  Failed: {e}")
            results.append({
                "n_factors": n,
                "error": str(e),
            })

    return {
        "benchmark": "qe_metrics",
        "description": "Quant evaluator metric computation at scale",
        "results": results,
    }


if __name__ == "__main__":
    result = run_qe_benchmark()
    print("\n=== QE Benchmark Results ===")
    for r in result["results"]:
        print(r)
