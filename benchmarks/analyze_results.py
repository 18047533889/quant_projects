#!/usr/bin/env python3
"""
Benchmark results analyzer.

Reads baseline.json and provides performance analysis and comparisons.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def analyze_qe(results: list):
    """Analyze QE benchmark results."""
    print("\n" + "=" * 80)
    print("QE (Quant Evaluator) Analysis")
    print("=" * 80)

    if not results:
        print("  No results found")
        return

    for r in results:
        if "error" in r:
            print(f"  {r['n_factors']:5d} factors: ERROR - {r['error']}")
            continue

        print(f"\n  {r['n_factors']:5d} factors:")
        print(f"    Elapsed:    {r['elapsed_s']:8.2f}s")
        print(f"    Throughput: {r['throughput_factors_per_s']:8.2f} factors/s")
        print(f"    Latency:    {r['per_factor_ms']:8.3f} ms/factor")

        # Calculate total operations
        n_ops = r['n_factors'] * r['n_dates'] * r['n_assets']
        print(f"    Total ops:  {n_ops:12,d} ({r['n_factors']} × {r['n_dates']} × {r['n_assets']})")


def analyze_fp(results: dict):
    """Analyze FP benchmark results."""
    print("\n" + "=" * 80)
    print("FP (Factor Processing) Analysis")
    print("=" * 80)

    if not results:
        print("  No results found")
        return

    for scale, data in results.items():
        if "error" in data:
            print(f"\n  {scale.upper()}: ERROR - {data['error']}")
            continue

        print(f"\n  {scale.upper()} ({data['total_cells']:,} cells):")

        transforms = data.get("transforms", {})
        if not transforms:
            continue

        # Calculate averages
        valid_results = [t for t in transforms.values() if "elapsed_s" in t]
        if not valid_results:
            print("    No successful transforms")
            continue

        avg_time = sum(t["elapsed_s"] for t in valid_results) / len(valid_results)
        avg_throughput = sum(t["throughput_mcells_per_s"] for t in valid_results) / len(valid_results)

        print(f"    Transforms:     {len(valid_results)}/{len(transforms)} successful")
        print(f"    Avg time:       {avg_time:8.3f}s")
        print(f"    Avg throughput: {avg_throughput:8.2f} Mcells/s")

        # Show individual transforms
        for name, res in transforms.items():
            if "error" in res:
                print(f"      {name:20s}: ERROR")
            else:
                print(f"      {name:20s}: {res['elapsed_s']:6.3f}s  ({res['throughput_mcells_per_s']:6.2f} Mcells/s)")


def analyze_fa(results: dict):
    """Analyze FA benchmark results."""
    print("\n" + "=" * 80)
    print("FA (Fundamental Analysis) Analysis")
    print("=" * 80)

    if not results:
        print("  No results found")
        return

    for scale, data in results.items():
        if "error" in data:
            print(f"\n  {scale.upper()}: ERROR - {data['error']}")
            continue

        print(f"\n  {scale.upper()} ({data['n_assets']:,} assets):")

        ops = data.get("operations", {})
        if not ops:
            continue

        for op_name, res in ops.items():
            if "error" in res:
                print(f"    {op_name:25s}: ERROR - {res['error']}")
            else:
                print(f"    {op_name:25s}: {res['elapsed_s']:6.3f}s  ({res['throughput_krows_per_s']:8.1f} Krows/s)")


def analyze_fo(results: dict):
    """Analyze FO benchmark results."""
    print("\n" + "=" * 80)
    print("FO (Factor Optimization) Analysis")
    print("=" * 80)

    if not results:
        print("  No results found")
        return

    for scale, data in results.items():
        if "error" in data:
            print(f"\n  {scale.upper()}: ERROR - {data['error']}")
            continue

        print(f"\n  {scale.upper()} ({data['n_trials']:,} trials):")

        ops = data.get("operations", {})
        if not ops:
            continue

        for op_name, res in ops.items():
            if "error" in res:
                print(f"    {op_name:20s}: ERROR")
            else:
                print(f"    {op_name:20s}: {res['elapsed_s']:6.3f}s  ({res['throughput_per_s']:10.1f} trials/s)")

                # Additional details for deduplication
                if "dedup_rate_pct" in res:
                    print(f"      Unique: {res['unique_count']}, Dedup rate: {res['dedup_rate_pct']:.1f}%")


def print_summary(data: dict):
    """Print overall summary."""
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Generated: {data.get('generated_at', 'Unknown')}")
    print(f"Platform:  {data.get('platform', 'Unknown')}")
    print()

    benchmarks = data.get("benchmarks", {})

    for name in ["qe", "fp", "fa", "fo"]:
        bench = benchmarks.get(name, {})

        if "error" in bench:
            status = f"ERROR - {bench['error']}"
        elif "total_time_s" in bench:
            status = f"SUCCESS ({bench['total_time_s']:.2f}s)"
        else:
            status = "INCOMPLETE"

        print(f"  {name.upper():3s}: {status}")


def main():
    """Main analyzer."""
    if len(sys.argv) > 1:
        baseline_path = Path(sys.argv[1])
    else:
        baseline_path = Path(__file__).parent / "baseline.json"

    if not baseline_path.exists():
        print(f"Error: {baseline_path} not found")
        print("Usage: python3 analyze_results.py [path/to/baseline.json]")
        return 1

    print(f"Reading results from: {baseline_path}")

    with open(baseline_path) as f:
        data = json.load(f)

    # Print summary first
    print_summary(data)

    # Analyze each benchmark
    benchmarks = data.get("benchmarks", {})

    if "qe" in benchmarks and "results" in benchmarks["qe"]:
        analyze_qe(benchmarks["qe"]["results"])

    if "fp" in benchmarks and "results" in benchmarks["fp"]:
        analyze_fp(benchmarks["fp"]["results"])

    if "fa" in benchmarks and "results" in benchmarks["fa"]:
        analyze_fa(benchmarks["fa"]["results"])

    if "fo" in benchmarks and "results" in benchmarks["fo"]:
        analyze_fo(benchmarks["fo"]["results"])

    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
