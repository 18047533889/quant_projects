#!/usr/bin/env python3
"""
CI/CD Integration Script for Performance Benchmarks

Designed to run in CI pipelines:
- Runs benchmarks
- Compares against baseline
- Fails if regressions detected
- Generates artifacts
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple


def run_command(cmd: List[str], timeout: int = 1800) -> Tuple[int, str, str]:
    """Run command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout"
    except Exception as e:
        return -1, "", str(e)


def main():
    """CI/CD benchmark runner."""

    print("=" * 80)
    print("CI/CD PERFORMANCE BENCHMARK RUNNER")
    print("=" * 80)
    print(f"\nStarted: {datetime.now().isoformat()}")

    benchmark_dir = Path(__file__).parent
    results_dir = benchmark_dir / "results"
    results_dir.mkdir(exist_ok=True)

    baseline_path = results_dir / "baseline.json"
    latest_path = results_dir / "latest_results.json"

    # Step 1: Run benchmarks
    print("\n" + "=" * 80)
    print("STEP 1: RUNNING BENCHMARKS")
    print("=" * 80)

    exit_code, stdout, stderr = run_command(
        [sys.executable, str(benchmark_dir / "run_all_benchmarks_v2.py")]
    )

    print(stdout)
    if stderr:
        print("STDERR:", file=sys.stderr)
        print(stderr, file=sys.stderr)

    if exit_code != 0:
        print(f"\n✗ Benchmarks failed with exit code {exit_code}")
        return 1

    print("\n✓ Benchmarks completed")

    # Step 2: Check if results generated
    if not latest_path.exists():
        print(f"\n✗ Results file not found: {latest_path}")
        return 1

    # Step 3: Compare against baseline (if exists)
    if baseline_path.exists():
        print("\n" + "=" * 80)
        print("STEP 2: CHECKING FOR REGRESSIONS")
        print("=" * 80)

        exit_code, stdout, stderr = run_command(
            [sys.executable, str(benchmark_dir / "detect_regression.py")]
        )

        print(stdout)
        if stderr:
            print("STDERR:", file=sys.stderr)
            print(stderr, file=sys.stderr)

        if exit_code != 0:
            print("\n✗ Performance regressions detected")
            print("\nTo update baseline (if intentional):")
            print(f"  cp {latest_path} {baseline_path}")
            return 1

        print("\n✓ No regressions detected")
    else:
        print("\n⚠️  No baseline found, skipping regression check")
        print(f"To establish baseline:\n  cp {latest_path} {baseline_path}")

    # Step 4: Generate visualizations
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING VISUALIZATIONS")
    print("=" * 80)

    exit_code, stdout, stderr = run_command(
        [sys.executable, str(benchmark_dir / "visualize_performance.py")]
    )

    print(stdout)
    if stderr:
        print("STDERR:", file=sys.stderr)
        print(stderr, file=sys.stderr)

    if exit_code != 0:
        print("\n⚠️  Visualization generation failed (non-fatal)")
    else:
        print("\n✓ Visualizations generated")

    # Step 5: Generate summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    with open(latest_path, 'r') as f:
        results = json.load(f)

    total_benchmarks = len(results.get('benchmarks', {}))
    successful_benchmarks = sum(
        1 for r in results.get('benchmarks', {}).values()
        if r.get('status') == 'SUCCESS'
    )

    total_stress = len(results.get('stress_tests', {}))
    successful_stress = sum(
        1 for r in results.get('stress_tests', {}).values()
        if r.get('status') == 'SUCCESS'
    )

    print(f"\nBenchmarks: {successful_benchmarks}/{total_benchmarks} successful")
    print(f"Stress Tests: {successful_stress}/{total_stress} successful")

    if 'total_wall_time_s' in results:
        print(f"Total Time: {results['total_wall_time_s']:.1f}s")

    print(f"\nResults: {latest_path}")
    print(f"Charts: {results_dir / 'charts'}")

    # Exit code
    if successful_benchmarks < total_benchmarks:
        print("\n⚠️  Some benchmarks failed")
        return 1

    print("\n✓ All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
