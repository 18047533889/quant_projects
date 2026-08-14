#!/usr/bin/env python3
"""
Performance Regression Detection

Compares current benchmark results against baseline to detect regressions.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def extract_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    """Extract numeric metrics from benchmark result."""
    metrics = {}

    # Direct metrics
    for key in ['elapsed_seconds', 'throughput_per_sec', 'throughput_factors_per_sec',
                'throughput_queries_per_sec', 'throughput_cells_per_sec', 'latency_ms_per_factor']:
        if key in result:
            metrics[key] = float(result[key])

    # Nested results
    if 'results' in result and isinstance(result['results'], dict):
        for scale_name, scale_result in result['results'].items():
            if isinstance(scale_result, dict):
                for key, value in scale_result.items():
                    if isinstance(value, (int, float)):
                        metrics[f"{scale_name}_{key}"] = float(value)

    return metrics


def compare_metrics(baseline: Dict[str, float], current: Dict[str, float], threshold: float = 0.1) -> List[Dict[str, Any]]:
    """Compare metrics and detect regressions."""
    regressions = []

    for metric_name in baseline:
        if metric_name not in current:
            continue

        baseline_value = baseline[metric_name]
        current_value = current[metric_name]

        if baseline_value == 0:
            continue

        # Calculate relative change
        relative_change = (current_value - baseline_value) / baseline_value

        # For throughput metrics, negative change is bad
        # For latency/time metrics, positive change is bad
        is_throughput = any(x in metric_name.lower() for x in ['throughput', 'per_sec'])
        is_latency = any(x in metric_name.lower() for x in ['latency', 'elapsed', 'time'])

        is_regression = False
        if is_throughput and relative_change < -threshold:
            is_regression = True
        elif is_latency and relative_change > threshold:
            is_regression = True

        if is_regression:
            regressions.append({
                'metric': metric_name,
                'baseline': baseline_value,
                'current': current_value,
                'relative_change': relative_change,
                'absolute_change': current_value - baseline_value,
                'threshold': threshold,
                'type': 'throughput' if is_throughput else 'latency'
            })

    return regressions


def analyze_benchmark(benchmark_name: str, baseline_result: Dict[str, Any], current_result: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single benchmark for regressions."""

    baseline_metrics = extract_metrics(baseline_result)
    current_metrics = extract_metrics(current_result)

    regressions = compare_metrics(baseline_metrics, current_metrics)

    return {
        'benchmark': benchmark_name,
        'status': current_result.get('status', 'UNKNOWN'),
        'baseline_metrics_count': len(baseline_metrics),
        'current_metrics_count': len(current_metrics),
        'regressions_detected': len(regressions),
        'regressions': regressions
    }


def main():
    """Run regression detection."""

    benchmark_dir = Path(__file__).parent
    baseline_path = benchmark_dir / 'results' / 'baseline.json'
    latest_path = benchmark_dir / 'results' / 'latest_results.json'

    print("=" * 80)
    print("PERFORMANCE REGRESSION DETECTION")
    print("=" * 80)

    # Check if files exist
    if not baseline_path.exists():
        print(f"\nERROR: Baseline file not found: {baseline_path}")
        print("Run benchmarks first to establish baseline.")
        return 1

    if not latest_path.exists():
        print(f"\nERROR: Latest results not found: {latest_path}")
        print("Run benchmarks to generate results.")
        return 1

    print(f"\nBaseline: {baseline_path}")
    print(f"Current:  {latest_path}")

    # Load results
    baseline = load_json(baseline_path)
    current = load_json(latest_path)

    # Analyze each benchmark
    all_analyses = []

    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    # Analyze benchmarks
    if 'benchmarks' in current:
        for bench_name, current_result in current['benchmarks'].items():
            # Try to find in baseline
            baseline_result = None

            # Check in benchmarks
            if 'benchmarks' in baseline and bench_name in baseline['benchmarks']:
                baseline_result = baseline['benchmarks'][bench_name]
            # Check at top level
            elif bench_name in baseline:
                baseline_result = baseline[bench_name]

            if baseline_result is None:
                print(f"\n{bench_name}: NO BASELINE (new benchmark)")
                continue

            analysis = analyze_benchmark(bench_name, baseline_result, current_result)
            all_analyses.append(analysis)

            print(f"\n{bench_name}:")
            print(f"  Status: {analysis['status']}")
            print(f"  Metrics compared: {analysis['current_metrics_count']}")
            print(f"  Regressions: {analysis['regressions_detected']}")

            if analysis['regressions_detected'] > 0:
                print(f"\n  REGRESSIONS DETECTED:")
                for reg in analysis['regressions']:
                    pct_change = reg['relative_change'] * 100
                    print(f"    - {reg['metric']}")
                    print(f"      Baseline: {reg['baseline']:.2f}")
                    print(f"      Current:  {reg['current']:.2f}")
                    print(f"      Change:   {pct_change:+.1f}%")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    total_regressions = sum(a['regressions_detected'] for a in all_analyses)
    total_benchmarks = len(all_analyses)

    print(f"\nBenchmarks analyzed: {total_benchmarks}")
    print(f"Total regressions: {total_regressions}")

    if total_regressions > 0:
        print(f"\n⚠️  PERFORMANCE REGRESSIONS DETECTED")
        print(f"\nReview the regressions above before committing changes.")
        return 1
    else:
        print(f"\n✓ No significant performance regressions detected")
        return 0


if __name__ == "__main__":
    sys.exit(main())
