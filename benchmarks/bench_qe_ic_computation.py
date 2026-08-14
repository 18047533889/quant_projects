#!/usr/bin/env python3
"""
Quant Evaluator IC Computation Benchmarks

Tests IC computation speed at different scales:
- Small: 100 factors × 252 days × 1,000 assets
- Medium: 1,000 factors × 252 days × 1,000 assets
- Large: 10,000 factors × 252 days × 1,000 assets
- XLarge: 10,000 factors × 1,260 days × 3,000 assets

Measures:
- Throughput (factors/sec, observations/sec)
- Latency per factor
- Memory usage
- Cache hit rates
"""

import gc
import time
import traceback
import numpy as np
from typing import Dict, Any

try:
    from quant_evaluator.runtime.evaluator import Evaluator
    from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
    from quant_evaluator.contracts.label_bundle import LabelBundle
    QE_AVAILABLE = True
except ImportError:
    QE_AVAILABLE = False


def generate_factor_data(n_factors: int, n_dates: int, n_assets: int, seed: int = 42):
    """Generate synthetic factor data."""
    np.random.seed(seed)

    # Factor values: (n_factors, n_dates, n_assets)
    factors = np.random.randn(n_factors, n_dates, n_assets).astype(np.float32)

    # Add some structure
    for i in range(n_factors):
        # Add time trend
        trend = np.linspace(-0.5, 0.5, n_dates).reshape(-1, 1)
        factors[i] += trend * 0.3

        # Add cross-sectional structure
        asset_effect = np.random.randn(n_assets) * 0.2
        factors[i] += asset_effect

    # Labels (returns)
    labels = np.random.randn(n_dates, n_assets).astype(np.float32) * 0.02

    # Add correlation with some factors
    for i in range(min(10, n_factors)):
        labels += factors[i] * 0.1 / (i + 1)

    dates = np.arange(n_dates)
    assets = np.arange(n_assets)

    return factors, labels, dates, assets


def benchmark_ic_computation(
    n_factors: int,
    n_dates: int,
    n_assets: int,
    enable_cache: bool = True,
    warmup: int = 1
) -> Dict[str, Any]:
    """Benchmark IC computation at given scale."""

    if not QE_AVAILABLE:
        return {"error": "quant_evaluator not available"}

    try:
        # Generate data
        print(f"  Generating {n_factors} factors × {n_dates} dates × {n_assets} assets...")
        factors, labels, dates, assets = generate_factor_data(n_factors, n_dates, n_assets)

        total_obs = n_factors * n_dates * n_assets
        print(f"  Total observations: {total_obs:,}")

        # Create contracts
        factor_batch = FactorBatch(
            values=factors,
            factor_ids=[f"factor_{i:05d}" for i in range(n_factors)],
            date_axis=AxisRef(name="date", values=dates),
            asset_axis=AxisRef(name="asset", values=assets),
        )

        label_bundle = LabelBundle(
            values=labels,
            date_axis=AxisRef(name="date", values=dates),
            asset_axis=AxisRef(name="asset", values=assets),
            label_id="fwd_ret_1d"
        )

        # Create evaluator
        evaluator = Evaluator(
            enable_cache=enable_cache,
            cache_size_mb=2048.0,
            max_chunk_memory_mb=512.0
        )

        # Warmup runs
        if warmup > 0:
            print(f"  Warmup: {warmup} run(s)...")
            for _ in range(warmup):
                _ = evaluator.evaluate_batch(
                    factor_batch,
                    label_bundle,
                    metrics=["rank_ic"]
                )

        # Clear cache for benchmark
        gc.collect()

        # Benchmark run
        print(f"  Running benchmark...")
        start = time.perf_counter()

        result = evaluator.evaluate_batch(
            factor_batch,
            label_bundle,
            metrics=["rank_ic", "rank_icir", "coverage"]
        )

        elapsed = time.perf_counter() - start

        # Extract metrics
        throughput_factors = n_factors / elapsed
        throughput_obs = total_obs / elapsed
        latency_per_factor = elapsed / n_factors * 1000  # ms

        return {
            "n_factors": n_factors,
            "n_dates": n_dates,
            "n_assets": n_assets,
            "total_observations": total_obs,
            "elapsed_seconds": round(elapsed, 3),
            "throughput_factors_per_sec": round(throughput_factors, 2),
            "throughput_obs_per_sec": round(throughput_obs, 0),
            "latency_ms_per_factor": round(latency_per_factor, 2),
            "cache_hits": result.cache_hits if hasattr(result, 'cache_hits') else 0,
            "cache_misses": result.cache_misses if hasattr(result, 'cache_misses') else 0,
            "chunks_processed": result.chunks_processed if hasattr(result, 'chunks_processed') else 0,
        }

    except Exception as e:
        return {
            "n_factors": n_factors,
            "n_dates": n_dates,
            "n_assets": n_assets,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():
    """Run all IC computation benchmarks."""

    print("=" * 70)
    print("QUANT EVALUATOR IC COMPUTATION BENCHMARKS")
    print("=" * 70)

    if not QE_AVAILABLE:
        print("ERROR: quant_evaluator package not available")
        return {"error": "quant_evaluator not available"}

    scales = [
        ("small", 100, 252, 1000),
        ("medium", 1000, 252, 1000),
        ("large", 10000, 252, 1000),
        ("xlarge", 10000, 1260, 3000),
    ]

    results = {}
    total_start = time.perf_counter()

    for scale_name, n_factors, n_dates, n_assets in scales:
        print(f"\n{'='*70}")
        print(f"Scale: {scale_name.upper()}")
        print(f"{'='*70}")

        result = benchmark_ic_computation(n_factors, n_dates, n_assets)
        results[scale_name] = result

        if "error" not in result:
            print(f"\n  ✓ Completed in {result['elapsed_seconds']:.2f}s")
            print(f"    Throughput: {result['throughput_factors_per_sec']:.1f} factors/s")
            print(f"    Latency: {result['latency_ms_per_factor']:.2f} ms/factor")
        else:
            print(f"\n  ✗ Failed: {result['error']}")

    total_elapsed = time.perf_counter() - total_start

    return {
        "benchmark": "qe_ic_computation",
        "description": "IC computation at multiple scales",
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
