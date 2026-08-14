#!/usr/bin/env python3
"""
Comprehensive standalone benchmark suite.

Tests core computational patterns without external dependencies:
- QE: Metric computation (IC, ICIR, etc.)
- FP: Panel transforms (rolling, ranking, etc.)
- FA: Financial operations (ratios, growth, PIT joins)
- FO: Search operations (mutation, dedup, validation)
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


# ============================================================================
# QE: Quant Evaluator Metrics
# ============================================================================

def compute_rank_ic(factor: np.ndarray, labels: np.ndarray) -> float:
    """Compute rank IC between factor and forward returns."""
    # Rank both arrays
    factor_rank = pd.Series(factor).rank(pct=True)
    label_rank = pd.Series(labels).rank(pct=True)

    # Pearson correlation of ranks
    valid = ~(np.isnan(factor_rank) | np.isnan(label_rank))
    if valid.sum() < 10:
        return np.nan

    return np.corrcoef(factor_rank[valid], label_rank[valid])[0, 1]


def bench_qe_metrics(n_factors: int, n_dates: int = 252, n_assets: int = 1000):
    """Benchmark QE metric computation."""
    print(f"  Generating {n_factors} factors × {n_dates} dates × {n_assets} assets...")

    rng = np.random.default_rng(42)

    # Generate factors
    factors = rng.standard_normal((n_factors, n_dates, n_assets))
    labels = rng.standard_normal((n_dates, n_assets)) * 0.02

    # Warmup
    _ = compute_rank_ic(factors[0, 0, :], labels[0, :])

    # Benchmark
    gc.collect()
    t0 = time.perf_counter()

    ics = []
    for i in range(n_factors):
        date_ics = []
        for d in range(n_dates):
            ic = compute_rank_ic(factors[i, d, :], labels[d, :])
            date_ics.append(ic)

        # Compute ICIR
        ic_mean = np.nanmean(date_ics)
        ic_std = np.nanstd(date_ics)
        icir = ic_mean / ic_std if ic_std > 0 else 0
        ics.append((ic_mean, icir))

    elapsed = time.perf_counter() - t0

    return {
        "n_factors": n_factors,
        "n_dates": n_dates,
        "n_assets": n_assets,
        "elapsed_s": round(elapsed, 3),
        "throughput_factors_per_s": round(n_factors / elapsed, 1),
        "per_factor_ms": round(elapsed * 1000 / n_factors, 3),
    }


def run_qe_benchmark():
    """Run QE benchmark suite."""
    print("\n" + "=" * 80)
    print("QE: Quant Evaluator Metrics")
    print("=" * 80)

    scales = [100, 1000, 10000]
    results = []

    for n in scales:
        print(f"\n{n} factors:")
        try:
            res = bench_qe_metrics(n, n_dates=252, n_assets=1000)
            results.append(res)
            print(f"  Time: {res['elapsed_s']}s")
            print(f"  Throughput: {res['throughput_factors_per_s']} factors/s")
            print(f"  Latency: {res['per_factor_ms']}ms/factor")
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({"n_factors": n, "error": str(e)})

    return {
        "benchmark": "qe_metrics",
        "description": "Rank IC and ICIR computation at scale",
        "results": results,
    }


# ============================================================================
# FP: Factor Processing Transforms
# ============================================================================

def ts_mean(series: pd.Series, window: int) -> pd.Series:
    """Time-series rolling mean."""
    return series.rolling(window=window, min_periods=1).mean()


def ts_std(series: pd.Series, window: int) -> pd.Series:
    """Time-series rolling std."""
    return series.rolling(window=window, min_periods=1).std()


def ts_zscore(series: pd.Series, window: int) -> pd.Series:
    """Time-series rolling z-score."""
    mean = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    return (series - mean) / (std + 1e-8)


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank."""
    return df.rank(axis=1, pct=True)


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score."""
    mean = df.mean(axis=1, keepdims=True)
    std = df.std(axis=1, keepdims=True)
    return (df - mean) / (std + 1e-8)


def bench_fp_transforms(n_assets: int, n_dates: int):
    """Benchmark panel transforms."""
    rng = np.random.default_rng(42)

    # Generate panel
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    assets = [f"A{i:06d}" for i in range(n_assets)]
    data = rng.uniform(10, 100, (n_dates, n_assets))
    df = pd.DataFrame(data, index=dates, columns=assets)

    results = {}

    # Time-series transforms
    for name, func, window in [
        ("ts_mean_20", ts_mean, 20),
        ("ts_std_20", ts_std, 20),
        ("ts_zscore_20", ts_zscore, 20),
    ]:
        gc.collect()
        t0 = time.perf_counter()

        result = df.apply(lambda col: func(col, window))

        elapsed = time.perf_counter() - t0
        throughput = (n_assets * n_dates) / elapsed / 1e6

        results[name] = {
            "elapsed_s": round(elapsed, 4),
            "throughput_mcells_per_s": round(throughput, 2),
        }

    # Cross-sectional transforms
    for name, func in [
        ("cs_rank", cs_rank),
        ("cs_zscore", cs_zscore),
    ]:
        gc.collect()
        t0 = time.perf_counter()

        result = func(df)

        elapsed = time.perf_counter() - t0
        throughput = (n_assets * n_dates) / elapsed / 1e6

        results[name] = {
            "elapsed_s": round(elapsed, 4),
            "throughput_mcells_per_s": round(throughput, 2),
        }

    return results


def run_fp_benchmark():
    """Run FP transform benchmark suite."""
    print("\n" + "=" * 80)
    print("FP: Factor Processing Transforms")
    print("=" * 80)

    scales = [
        ("small", 100, 252),
        ("medium", 1000, 756),
        ("large", 10000, 2520),
    ]

    results = {}

    for scale_name, n_assets, n_dates in scales:
        print(f"\n{scale_name.upper()}: {n_assets} assets × {n_dates} dates = {n_assets*n_dates:,} cells")

        try:
            transform_results = bench_fp_transforms(n_assets, n_dates)

            results[scale_name] = {
                "n_assets": n_assets,
                "n_dates": n_dates,
                "total_cells": n_assets * n_dates,
                "transforms": transform_results,
            }

            for name, res in transform_results.items():
                print(f"  {name:20s}: {res['elapsed_s']:6.3f}s  ({res['throughput_mcells_per_s']:6.2f} Mcells/s)")

        except Exception as e:
            print(f"  FAILED: {e}")
            results[scale_name] = {"error": str(e)}

    return {
        "benchmark": "fp_transforms",
        "description": "Panel transforms (rolling, ranking, etc.)",
        "results": results,
    }


# ============================================================================
# FA: Fundamental Analysis Operations
# ============================================================================

def bench_fa_ratios(n_assets: int, n_quarters: int = 20):
    """Benchmark financial ratio computations."""
    rng = np.random.default_rng(42)

    # Generate fundamental data
    assets = [f"A{i:06d}" for i in range(n_assets)]
    quarters = pd.date_range("2020-03-31", periods=n_quarters, freq="QE")

    data = {
        "revenue": rng.uniform(1e8, 1e10, n_assets * n_quarters),
        "cogs": rng.uniform(1e7, 5e9, n_assets * n_quarters),
        "total_assets": rng.uniform(1e9, 1e11, n_assets * n_quarters),
        "equity": rng.uniform(5e8, 5e10, n_assets * n_quarters),
        "net_income": rng.uniform(1e7, 1e9, n_assets * n_quarters),
    }

    df = pd.DataFrame(data)

    # Benchmark ratio computation
    gc.collect()
    t0 = time.perf_counter()

    df["gross_margin"] = (df["revenue"] - df["cogs"]) / df["revenue"]
    df["roa"] = df["net_income"] / df["total_assets"]
    df["roe"] = df["net_income"] / df["equity"]
    df["asset_turnover"] = df["revenue"] / df["total_assets"]

    elapsed = time.perf_counter() - t0
    throughput = len(df) / elapsed / 1000

    return {
        "elapsed_s": round(elapsed, 4),
        "throughput_krows_per_s": round(throughput, 1),
    }


def bench_fa_pit_join(n_assets: int, n_dates: int = 252):
    """Benchmark point-in-time join."""
    rng = np.random.default_rng(123)

    # Daily prices
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    assets = [f"A{i:06d}" for i in range(n_assets)]

    price_data = []
    for date in dates:
        for asset in assets:
            price_data.append({
                "asset": asset,
                "date": date,
                "price": rng.uniform(10, 100),
            })

    prices = pd.DataFrame(price_data)

    # Quarterly fundamentals
    quarters = pd.date_range("2023-03-31", periods=4, freq="QE")
    fund_data = []
    for asset in assets:
        for quarter in quarters:
            fund_data.append({
                "asset": asset,
                "report_date": quarter,
                "eps": rng.uniform(0.5, 5.0),
            })

    fundamentals = pd.DataFrame(fund_data)

    # PIT merge
    gc.collect()
    t0 = time.perf_counter()

    prices = prices.sort_values(["asset", "date"])
    fundamentals = fundamentals.sort_values(["asset", "report_date"])

    result = pd.merge_asof(
        prices,
        fundamentals,
        left_on="date",
        right_on="report_date",
        by="asset",
        direction="backward"
    )

    result["pe_ratio"] = result["price"] / result["eps"]

    elapsed = time.perf_counter() - t0
    throughput = len(result) / elapsed / 1000

    return {
        "elapsed_s": round(elapsed, 4),
        "throughput_krows_per_s": round(throughput, 1),
    }


def run_fa_benchmark():
    """Run FA operations benchmark suite."""
    print("\n" + "=" * 80)
    print("FA: Fundamental Analysis Operations")
    print("=" * 80)

    scales = [
        ("small", 1000),
        ("medium", 10000),
        ("large", 100000),
    ]

    results = {}

    for scale_name, n_assets in scales:
        print(f"\n{scale_name.upper()}: {n_assets} assets")

        scale_results = {
            "n_assets": n_assets,
            "operations": {},
        }

        # Financial ratios
        try:
            res = bench_fa_ratios(n_assets, n_quarters=20)
            scale_results["operations"]["financial_ratios"] = res
            print(f"  Financial ratios:     {res['elapsed_s']:6.3f}s  ({res['throughput_krows_per_s']:8.1f} Krows/s)")
        except Exception as e:
            print(f"  Financial ratios:     FAILED - {e}")
            scale_results["operations"]["financial_ratios"] = {"error": str(e)}

        # PIT join (skip for large due to memory)
        if n_assets <= 10000:
            try:
                res = bench_fa_pit_join(n_assets, n_dates=252)
                scale_results["operations"]["pit_join"] = res
                print(f"  PIT join (252 days):  {res['elapsed_s']:6.3f}s  ({res['throughput_krows_per_s']:8.1f} Krows/s)")
            except Exception as e:
                print(f"  PIT join:             FAILED - {e}")
                scale_results["operations"]["pit_join"] = {"error": str(e)}

        results[scale_name] = scale_results

    return {
        "benchmark": "fa_operations",
        "description": "Financial ratios and point-in-time joins",
        "results": results,
    }


# ============================================================================
# FO: Factor Optimization Search
# ============================================================================

class SeenCache:
    """Simple seen-set for deduplication."""

    def __init__(self):
        self.seen = set()
        self.collision_count = 0

    def mark_seen(self, candidate: dict) -> bool:
        """Return True if already seen."""
        key = self._hash(candidate)
        if key in self.seen:
            self.collision_count += 1
            return True
        self.seen.add(key)
        return False

    def _hash(self, candidate: dict) -> str:
        s = f"{candidate['op']}_{candidate.get('window', 0)}"
        return hashlib.md5(s.encode()).hexdigest()[:16]


def generate_candidate():
    """Generate random factor candidate."""
    rng = np.random.default_rng()
    ops = ["ts_mean", "ts_std", "ts_zscore", "rank", "zscore"]
    op = rng.choice(ops)

    candidate = {"op": op}
    if op.startswith("ts_"):
        candidate["window"] = int(rng.choice([5, 10, 20, 60]))

    return candidate


def bench_fo_search(n_trials: int):
    """Benchmark search operations."""
    results = {}

    # Mutation generation
    gc.collect()
    t0 = time.perf_counter()
    candidates = [generate_candidate() for _ in range(n_trials)]
    elapsed = time.perf_counter() - t0

    results["generation"] = {
        "elapsed_s": round(elapsed, 4),
        "throughput_per_s": round(n_trials / elapsed, 1),
    }

    # Deduplication
    gc.collect()
    t0 = time.perf_counter()

    cache = SeenCache()
    unique = sum(1 for c in candidates if not cache.mark_seen(c))

    elapsed = time.perf_counter() - t0
    dedup_rate = cache.collision_count / n_trials * 100

    results["deduplication"] = {
        "elapsed_s": round(elapsed, 4),
        "throughput_per_s": round(n_trials / elapsed, 1),
        "unique_count": unique,
        "collision_count": cache.collision_count,
        "dedup_rate_pct": round(dedup_rate, 1),
    }

    return results


def run_fo_benchmark():
    """Run FO search benchmark suite."""
    print("\n" + "=" * 80)
    print("FO: Factor Optimization Search")
    print("=" * 80)

    scales = [
        ("small", 100),
        ("medium", 1000),
        ("large", 10000),
    ]

    results = {}

    for scale_name, n_trials in scales:
        print(f"\n{scale_name.upper()}: {n_trials} trials")

        try:
            search_results = bench_fo_search(n_trials)

            results[scale_name] = {
                "n_trials": n_trials,
                "operations": search_results,
            }

            gen = search_results["generation"]
            print(f"  Generation:      {gen['elapsed_s']:6.3f}s  ({gen['throughput_per_s']:10.1f} trials/s)")

            dedup = search_results["deduplication"]
            print(f"  Deduplication:   {dedup['elapsed_s']:6.3f}s  ({dedup['throughput_per_s']:10.1f} trials/s)")
            print(f"    Unique: {dedup['unique_count']}/{n_trials}, Dedup rate: {dedup['dedup_rate_pct']:.1f}%")

        except Exception as e:
            print(f"  FAILED: {e}")
            results[scale_name] = {"error": str(e)}

    return {
        "benchmark": "fo_search",
        "description": "Mutation generation and deduplication",
        "results": results,
    }


# ============================================================================
# Main Runner
# ============================================================================

def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("QUANTITATIVE PLATFORM COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 80)

    all_results = {
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "platform": "quantitative_analysis",
        "benchmarks": {},
    }

    # Run all benchmarks
    for bench_func, name in [
        (run_fo_benchmark, "fo"),
        (run_fa_benchmark, "fa"),
        (run_fp_benchmark, "fp"),
        (run_qe_benchmark, "qe"),
    ]:
        try:
            t0 = time.perf_counter()
            result = bench_func()
            result["total_time_s"] = round(time.perf_counter() - t0, 2)
            all_results["benchmarks"][name] = result
            print(f"\n✓ {name.upper()} completed in {result['total_time_s']}s")
        except Exception as e:
            print(f"\n✗ {name.upper()} failed: {e}")
            all_results["benchmarks"][name] = {"error": str(e)}

    # Save results
    output_path = ROOT / "baseline.json"
    output_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    print(f"\n✓ Results written to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for name, data in all_results["benchmarks"].items():
        if "error" in data:
            print(f"{name.upper()}: ERROR - {data['error']}")
        else:
            print(f"{name.upper()}: {data['total_time_s']}s - {data['description']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
