#!/usr/bin/env python3
"""
Master Benchmark Runner

Runs all benchmarks and stress tests, generates comprehensive report.
"""

import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

BENCHMARK_DIR = Path(__file__).parent

BENCHMARKS = [
    ("bench_qe_ic_computation.py", "Quant Evaluator IC Computation"),
    ("bench_rc_ledger.py", "Research Control Ledger"),
    ("bench_fo_search_optimized.py", "Factor Optimizer Search"),
    ("bench_fa_similarity.py", "Factor Assets Similarity"),
    ("bench_fp_preprocess.py", "Factor Preprocess Transforms"),
]

STRESS_TESTS = [
    ("stress_tests.py", "Comprehensive Stress Tests"),
]


def run_benchmark(script_name: str, description: str) -> Dict[str, Any]:
    """Run a single benchmark script."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*70}")

    script_path = BENCHMARK_DIR / script_name

    if not script_path.exists():
        return {
            "benchmark": script_name,
            "description": description,
            "error": f"Script not found: {script_path}",
            "status": "FAILED"
        }

    try:
        start = time.perf_counter()

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )

        elapsed = time.perf_counter() - start

        # Print stdout
        if result.stdout:
            print(result.stdout)

        # Print stderr if any
        if result.stderr:
            print("STDERR:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)

        # Try to parse JSON output from stdout
        output_lines = result.stdout.strip().split('\n')
        json_output = None

        for line in reversed(output_lines):
            line = line.strip()
            if line.startswith('{'):
                try:
                    json_output = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

        if json_output:
            json_output["script"] = script_name
            json_output["wall_time_s"] = round(elapsed, 2)
            json_output["status"] = "SUCCESS" if result.returncode == 0 else "FAILED"
            json_output["return_code"] = result.returncode
            return json_output
        else:
            return {
                "benchmark": script_name,
                "description": description,
                "wall_time_s": round(elapsed, 2),
                "status": "PARTIAL" if result.returncode == 0 else "FAILED",
                "return_code": result.returncode,
                "output_length": len(result.stdout),
                "stderr_length": len(result.stderr)
            }

    except subprocess.TimeoutExpired:
        return {
            "benchmark": script_name,
            "description": description,
            "error": "Timeout (30 minutes)",
            "status": "TIMEOUT"
        }
    except Exception as e:
        return {
            "benchmark": script_name,
            "description": description,
            "error": str(e),
            "status": "ERROR"
        }


def generate_summary_report(all_results: Dict[str, Any]) -> str:
    """Generate human-readable summary report."""

    lines = []
    lines.append("=" * 80)
    lines.append("COMPREHENSIVE BENCHMARK AND STRESS TEST REPORT")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {all_results['generated_at']}")
    lines.append(f"Total Time: {all_results['total_wall_time_s']:.1f}s")
    lines.append(f"\nPlatform: {all_results['platform']}")

    # Benchmarks summary
    lines.append("\n" + "=" * 80)
    lines.append("BENCHMARKS")
    lines.append("=" * 80)

    for bench_name, bench_result in all_results['benchmarks'].items():
        status = bench_result.get('status', 'UNKNOWN')
        desc = bench_result.get('description', bench_name)

        lines.append(f"\n{desc}")
        lines.append(f"  Status: {status}")

        if 'wall_time_s' in bench_result:
            lines.append(f"  Time: {bench_result['wall_time_s']:.1f}s")

        if 'error' in bench_result:
            lines.append(f"  Error: {bench_result['error']}")

        # Extract key metrics
        if 'results' in bench_result and isinstance(bench_result['results'], dict):
            lines.append(f"  Results:")
            for key, value in bench_result['results'].items():
                if isinstance(value, dict):
                    if 'throughput_per_sec' in value:
                        lines.append(f"    {key}: {value['throughput_per_sec']:.1f} ops/s")
                    elif 'throughput_queries_per_sec' in value:
                        lines.append(f"    {key}: {value['throughput_queries_per_sec']:.1f} queries/s")
                    elif 'throughput_factors_per_sec' in value:
                        lines.append(f"    {key}: {value['throughput_factors_per_sec']:.1f} factors/s")

    # Stress tests summary
    lines.append("\n" + "=" * 80)
    lines.append("STRESS TESTS")
    lines.append("=" * 80)

    for test_name, test_result in all_results['stress_tests'].items():
        status = test_result.get('status', 'UNKNOWN')
        desc = test_result.get('description', test_name)

        lines.append(f"\n{desc}")
        lines.append(f"  Status: {status}")

        if 'wall_time_s' in test_result:
            lines.append(f"  Time: {test_result['wall_time_s']:.1f}s")

        if 'error' in test_result:
            lines.append(f"  Error: {test_result['error']}")

    # Overall summary
    lines.append("\n" + "=" * 80)
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 80)

    total_benchmarks = len(all_results['benchmarks'])
    successful_benchmarks = sum(1 for r in all_results['benchmarks'].values() if r.get('status') == 'SUCCESS')

    total_stress = len(all_results['stress_tests'])
    successful_stress = sum(1 for r in all_results['stress_tests'].values() if r.get('status') == 'SUCCESS')

    lines.append(f"\nBenchmarks: {successful_benchmarks}/{total_benchmarks} successful")
    lines.append(f"Stress Tests: {successful_stress}/{total_stress} successful")
    lines.append(f"\nTotal Execution Time: {all_results['total_wall_time_s']:.1f}s")

    return '\n'.join(lines)


def main():
    """Run all benchmarks and generate report."""

    print("=" * 80)
    print("MASTER BENCHMARK RUNNER")
    print("=" * 80)
    print(f"\nStarting at {datetime.now().isoformat()}")
    print(f"Benchmark directory: {BENCHMARK_DIR}")

    all_results = {
        "schema_version": "2.0",
        "generated_at": datetime.now().isoformat(),
        "platform": "quant_projects",
        "benchmarks": {},
        "stress_tests": {}
    }

    total_start = time.perf_counter()

    # Run benchmarks
    print(f"\n{'='*80}")
    print("RUNNING BENCHMARKS")
    print(f"{'='*80}")

    for script_name, description in BENCHMARKS:
        result = run_benchmark(script_name, description)
        benchmark_id = script_name.replace('.py', '')
        all_results['benchmarks'][benchmark_id] = result

    # Run stress tests
    print(f"\n{'='*80}")
    print("RUNNING STRESS TESTS")
    print(f"{'='*80}")

    for script_name, description in STRESS_TESTS:
        result = run_benchmark(script_name, description)
        test_id = script_name.replace('.py', '')
        all_results['stress_tests'][test_id] = result

    total_elapsed = time.perf_counter() - total_start
    all_results['total_wall_time_s'] = round(total_elapsed, 2)

    # Save results
    output_json = BENCHMARK_DIR / 'results' / f'benchmark_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    output_json.parent.mkdir(exist_ok=True)

    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {output_json}")

    # Generate summary report
    summary = generate_summary_report(all_results)

    output_txt = BENCHMARK_DIR / 'results' / f'benchmark_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    with open(output_txt, 'w') as f:
        f.write(summary)

    print(f"Summary saved to: {output_txt}")

    # Print summary
    print("\n" + summary)

    # Save latest as baseline
    latest_json = BENCHMARK_DIR / 'results' / 'latest_results.json'
    with open(latest_json, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nLatest results: {latest_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
