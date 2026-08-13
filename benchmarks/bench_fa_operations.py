#!/usr/bin/env python3
"""
FA (Fundamental Analysis) operations benchmark.

Tests: Point-in-time operations, financial computations at scale
- Small: 1k assets
- Medium: 10k assets
- Large: 100k assets
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DA_ROOT = ROOT / "dataaccess"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DA_ROOT))


def generate_fundamental_data(n_assets: int, n_quarters: int = 20):
    """Generate synthetic fundamental data."""
    rng = np.random.default_rng(42)

    assets = [f"A{i:06d}" for i in range(n_assets)]
    quarters = pd.date_range("2020-03-31", periods=n_quarters, freq="QE")

    records = []
    for asset in assets:
        for quarter in quarters:
            # Simulate financial statement items
            revenue = rng.uniform(1e8, 1e10)
            cogs = revenue * rng.uniform(0.4, 0.7)
            opex = revenue * rng.uniform(0.1, 0.3)
            net_income = (revenue - cogs - opex) * rng.uniform(0.7, 1.0)

            total_assets = revenue * rng.uniform(2, 5)
            total_liabilities = total_assets * rng.uniform(0.3, 0.6)
            equity = total_assets - total_liabilities

            records.append({
                "asset": asset,
                "quarter": quarter,
                "revenue": revenue,
                "cogs": cogs,
                "opex": opex,
                "net_income": net_income,
                "total_assets": total_assets,
                "total_liabilities": total_liabilities,
                "equity": equity,
            })

    return pd.DataFrame(records)


def bench_financial_ratios(df: pd.DataFrame):
    """Compute standard financial ratios."""
    gc.collect()
    t0 = time.perf_counter()

    # Profit margins
    df["gross_margin"] = (df["revenue"] - df["cogs"]) / df["revenue"]
    df["operating_margin"] = (df["revenue"] - df["cogs"] - df["opex"]) / df["revenue"]
    df["net_margin"] = df["net_income"] / df["revenue"]

    # Returns
    df["roa"] = df["net_income"] / df["total_assets"]
    df["roe"] = df["net_income"] / df["equity"]

    # Leverage
    df["debt_to_equity"] = df["total_liabilities"] / df["equity"]
    df["debt_to_assets"] = df["total_liabilities"] / df["total_assets"]

    # Asset turnover
    df["asset_turnover"] = df["revenue"] / df["total_assets"]

    elapsed = time.perf_counter() - t0
    return elapsed, len(df)


def bench_growth_rates(df: pd.DataFrame):
    """Compute YoY and QoQ growth rates."""
    gc.collect()
    t0 = time.perf_counter()

    df = df.sort_values(["asset", "quarter"])

    # YoY growth (4 quarters back)
    for col in ["revenue", "net_income", "equity"]:
        df[f"{col}_yoy"] = df.groupby("asset")[col].pct_change(periods=4)

    # QoQ growth
    for col in ["revenue", "net_income"]:
        df[f"{col}_qoq"] = df.groupby("asset")[col].pct_change(periods=1)

    elapsed = time.perf_counter() - t0
    return elapsed, len(df)


def bench_rolling_aggregations(df: pd.DataFrame):
    """Compute trailing metrics (TTM)."""
    gc.collect()
    t0 = time.perf_counter()

    df = df.sort_values(["asset", "quarter"])

    # Trailing 4 quarters (TTM)
    for col in ["revenue", "net_income", "cogs"]:
        df[f"{col}_ttm"] = df.groupby("asset")[col].rolling(window=4, min_periods=1).sum().values

    # Average metrics
    for col in ["total_assets", "equity"]:
        df[f"{col}_avg4q"] = df.groupby("asset")[col].rolling(window=4, min_periods=1).mean().values

    elapsed = time.perf_counter() - t0
    return elapsed, len(df)


def bench_pit_join_simulation(n_assets: int, n_dates: int = 252):
    """Simulate point-in-time join operation."""
    rng = np.random.default_rng(123)

    # Daily price data
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    assets = [f"A{i:06d}" for i in range(n_assets)]

    price_records = []
    for date in dates:
        for asset in assets:
            price_records.append({
                "asset": asset,
                "date": date,
                "price": rng.uniform(10, 100),
            })

    prices = pd.DataFrame(price_records)

    # Quarterly fundamental data (sparse)
    quarters = pd.date_range("2023-03-31", periods=4, freq="QE")
    fundamental_records = []
    for asset in assets:
        for quarter in quarters:
            fundamental_records.append({
                "asset": asset,
                "report_date": quarter,
                "eps": rng.uniform(0.5, 5.0),
                "book_value": rng.uniform(20, 80),
            })

    fundamentals = pd.DataFrame(fundamental_records)

    # PIT merge
    gc.collect()
    t0 = time.perf_counter()

    # Sort both dataframes by merge keys
    prices = prices.sort_values(["date", "asset"])
    fundamentals = fundamentals.sort_values(["report_date", "asset"])

    result = pd.merge_asof(
        prices,
        fundamentals,
        left_on="date",
        right_on="report_date",
        by="asset",
        direction="backward"
    )

    # Compute PE ratio
    result["pe_ratio"] = result["price"] / result["eps"]
    result["pb_ratio"] = result["price"] / result["book_value"]

    elapsed = time.perf_counter() - t0
    return elapsed, len(result)


def run_fa_benchmark():
    """Run FA operations benchmark suite."""
    scales = [
        ("small", 1000),
        ("medium", 10000),
        ("large", 100000),
    ]

    results = {}

    for scale_name, n_assets in scales:
        print(f"\n=== {scale_name.upper()}: {n_assets} assets ===")

        df = generate_fundamental_data(n_assets, n_quarters=20)
        n_records = len(df)

        scale_results = {
            "n_assets": n_assets,
            "n_quarters": 20,
            "n_records": n_records,
            "operations": {},
        }

        # Test 1: Financial ratios
        try:
            elapsed, rows = bench_financial_ratios(df.copy())
            throughput = rows / elapsed / 1000  # K rows/s
            scale_results["operations"]["financial_ratios"] = {
                "elapsed_s": round(elapsed, 4),
                "throughput_krows_per_s": round(throughput, 1),
            }
            print(f"  Financial ratios:     {elapsed:6.3f}s  ({throughput:8.1f} Krows/s)")
        except Exception as e:
            scale_results["operations"]["financial_ratios"] = {"error": str(e)}
            print(f"  Financial ratios:     FAILED - {e}")

        # Test 2: Growth rates
        try:
            elapsed, rows = bench_growth_rates(df.copy())
            throughput = rows / elapsed / 1000
            scale_results["operations"]["growth_rates"] = {
                "elapsed_s": round(elapsed, 4),
                "throughput_krows_per_s": round(throughput, 1),
            }
            print(f"  Growth rates:         {elapsed:6.3f}s  ({throughput:8.1f} Krows/s)")
        except Exception as e:
            scale_results["operations"]["growth_rates"] = {"error": str(e)}
            print(f"  Growth rates:         FAILED - {e}")

        # Test 3: Rolling aggregations
        try:
            elapsed, rows = bench_rolling_aggregations(df.copy())
            throughput = rows / elapsed / 1000
            scale_results["operations"]["rolling_agg"] = {
                "elapsed_s": round(elapsed, 4),
                "throughput_krows_per_s": round(throughput, 1),
            }
            print(f"  Rolling aggregations: {elapsed:6.3f}s  ({throughput:8.1f} Krows/s)")
        except Exception as e:
            scale_results["operations"]["rolling_agg"] = {"error": str(e)}
            print(f"  Rolling aggregations: FAILED - {e}")

        # Test 4: PIT join (only for small/medium due to memory)
        if n_assets <= 10000:
            try:
                elapsed, rows = bench_pit_join_simulation(n_assets, n_dates=252)
                throughput = rows / elapsed / 1000
                scale_results["operations"]["pit_join"] = {
                    "elapsed_s": round(elapsed, 4),
                    "throughput_krows_per_s": round(throughput, 1),
                }
                print(f"  PIT join (252 days):  {elapsed:6.3f}s  ({throughput:8.1f} Krows/s)")
            except Exception as e:
                scale_results["operations"]["pit_join"] = {"error": str(e)}
                print(f"  PIT join:             FAILED - {e}")

        results[scale_name] = scale_results

    return {
        "benchmark": "fa_operations",
        "description": "Fundamental analysis operations at scale",
        "results": results,
    }


if __name__ == "__main__":
    result = run_fa_benchmark()

    print("\n=== FA Benchmark Summary ===")
    for scale, data in result["results"].items():
        print(f"\n{scale}: {data['n_assets']} assets × {data['n_quarters']} quarters = {data['n_records']:,} records")
        if data["operations"]:
            times = [v["elapsed_s"] for v in data["operations"].values() if "elapsed_s" in v]
            if times:
                print(f"  Mean time: {np.mean(times):.3f}s")
                print(f"  Total time: {sum(times):.3f}s")
