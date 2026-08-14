#!/usr/bin/env python3
"""
Performance Visualization

Generate performance charts and graphs from benchmark results.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path: Path) -> Dict[str, Any]:
    """Load benchmark results."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_throughput_comparison(results: Dict[str, Any], output_dir: Path):
    """Plot throughput comparison across benchmarks."""

    fig, ax = plt.subplots(figsize=(12, 6))

    benchmarks = []
    throughputs = []

    if 'benchmarks' in results:
        for bench_name, bench_result in results['benchmarks'].items():
            if 'results' in bench_result:
                for scale_name, scale_result in bench_result['results'].items():
                    if isinstance(scale_result, dict):
                        # Extract throughput
                        throughput = None
                        if 'throughput_per_sec' in scale_result:
                            throughput = scale_result['throughput_per_sec']
                        elif 'throughput_factors_per_sec' in scale_result:
                            throughput = scale_result['throughput_factors_per_sec']
                        elif 'throughput_queries_per_sec' in scale_result:
                            throughput = scale_result['throughput_queries_per_sec']

                        if throughput:
                            benchmarks.append(f"{bench_name}\n{scale_name}")
                            throughputs.append(throughput)

    if benchmarks:
        x = np.arange(len(benchmarks))
        ax.bar(x, throughputs, color='#38BDF8', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=45, ha='right')
        ax.set_ylabel('Throughput (ops/s)')
        ax.set_title('Benchmark Throughput Comparison')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / 'throughput_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Generated: {output_path}")


def plot_scaling_analysis(results: Dict[str, Any], output_dir: Path):
    """Plot scaling behavior for each benchmark."""

    if 'benchmarks' not in results:
        return

    for bench_name, bench_result in results['benchmarks'].items():
        if 'results' not in bench_result:
            continue

        # Extract scale data
        scales = []
        sizes = []
        times = []

        for scale_name, scale_result in bench_result['results'].items():
            if not isinstance(scale_result, dict):
                continue

            scales.append(scale_name)

            # Get size metric
            size = None
            if 'n_factors' in scale_result:
                size = scale_result['n_factors']
            elif 'n_candidates' in scale_result:
                size = scale_result['n_candidates']
            elif 'total_cells' in scale_result:
                size = scale_result['total_cells']
            elif 'n_database' in scale_result:
                size = scale_result['n_database']

            # Get time metric
            time_val = scale_result.get('elapsed_seconds', 0)

            if size and time_val:
                sizes.append(size)
                times.append(time_val)

        if len(sizes) >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(sizes, times, 'o-', linewidth=2, markersize=8, color='#38BDF8')
            ax.set_xlabel('Problem Size')
            ax.set_ylabel('Time (seconds)')
            ax.set_title(f'{bench_name.replace("_", " ").title()} - Scaling Behavior')
            ax.grid(alpha=0.3)

            # Log scale if range is large
            if max(sizes) / min(sizes) > 100:
                ax.set_xscale('log')
                ax.set_yscale('log')

            plt.tight_layout()
            output_path = output_dir / f'{bench_name}_scaling.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Generated: {output_path}")


def plot_memory_usage(results: Dict[str, Any], output_dir: Path):
    """Plot memory usage from stress tests."""

    if 'stress_tests' not in results:
        return

    stress_results = results['stress_tests'].get('stress_tests', {})
    if not stress_results or 'results' not in stress_results:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Memory Usage Analysis', fontsize=16)

    # Large arrays test
    if 'memory_large_arrays' in stress_results['results']:
        test_result = stress_results['results']['memory_large_arrays']
        if 'allocations' in test_result:
            ax = axes[0, 0]
            allocs = test_result['allocations']
            sizes = [a['target_size_mb'] for a in allocs if a.get('success')]
            allocated = [a['allocated_mb'] for a in allocs if a.get('success')]

            ax.plot(sizes, allocated, 'o-', linewidth=2, markersize=8, color='#6EE7B7')
            ax.plot(sizes, sizes, '--', color='gray', alpha=0.5, label='Ideal')
            ax.set_xlabel('Target Size (MB)')
            ax.set_ylabel('Allocated (MB)')
            ax.set_title('Large Array Allocations')
            ax.legend()
            ax.grid(alpha=0.3)

    # Repeated allocations
    if 'repeated_allocations' in stress_results['results']:
        test_result = stress_results['results']['repeated_allocations']
        ax = axes[0, 1]

        initial = test_result.get('initial_memory_mb', 0)
        final = test_result.get('final_memory_mb', 0)

        ax.bar(['Initial', 'Final'], [initial, final], color=['#38BDF8', '#E9A568'])
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Repeated Allocations Memory')
        ax.grid(axis='y', alpha=0.3)

    # Long running
    if 'long_running_computation' in stress_results['results']:
        test_result = stress_results['results']['long_running_computation']
        ax = axes[1, 0]

        initial = test_result.get('initial_memory_mb', 0)
        peak = test_result.get('peak_memory_mb', 0)
        final = test_result.get('final_memory_mb', 0)

        ax.bar(['Initial', 'Peak', 'Final'], [initial, peak, final],
               color=['#38BDF8', '#E9A568', '#6EE7B7'])
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Long-Running Computation Memory')
        ax.grid(axis='y', alpha=0.3)

    # Large dataset
    if 'large_dataset' in stress_results['results']:
        test_result = stress_results['results']['large_dataset']
        ax = axes[1, 1]

        stages = ['Initial', 'After Gen', 'After Proc', 'Final']
        values = [
            test_result.get('initial_memory_mb', 0),
            test_result.get('after_generation_mb', 0),
            test_result.get('after_processing_mb', 0),
            test_result.get('final_memory_mb', 0)
        ]

        ax.plot(stages, values, 'o-', linewidth=2, markersize=8, color='#38BDF8')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Large Dataset Processing Memory')
        ax.grid(alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    output_path = output_dir / 'memory_usage.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Generated: {output_path}")


def plot_latency_distribution(results: Dict[str, Any], output_dir: Path):
    """Plot latency distribution for operations."""

    if 'benchmarks' not in results:
        return

    latency_data = {}

    for bench_name, bench_result in results['benchmarks'].items():
        if 'results' not in bench_result:
            continue

        for scale_name, scale_result in bench_result['results'].items():
            if not isinstance(scale_result, dict):
                continue

            # Extract latency metrics
            for key in ['latency_ms_per_factor', 'latency_ms_per_write',
                       'latency_ms_per_query', 'latency_ms_per_mutation']:
                if key in scale_result:
                    label = f"{bench_name}\n{scale_name}"
                    latency_data[label] = scale_result[key]

    if latency_data:
        fig, ax = plt.subplots(figsize=(12, 6))

        labels = list(latency_data.keys())
        values = list(latency_data.values())

        x = np.arange(len(labels))
        bars = ax.bar(x, values, color='#E9A568', alpha=0.8)

        # Color code by latency
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val < 1:
                bar.set_color('#6EE7B7')  # Fast
            elif val < 10:
                bar.set_color('#38BDF8')  # Medium
            else:
                bar.set_color('#E9A568')  # Slow

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Operation Latency Distribution')
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / 'latency_distribution.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Generated: {output_path}")


def main():
    """Generate all performance visualizations."""

    print("=" * 80)
    print("PERFORMANCE VISUALIZATION")
    print("=" * 80)

    benchmark_dir = Path(__file__).parent
    results_path = benchmark_dir / 'results' / 'latest_results.json'
    output_dir = benchmark_dir / 'results' / 'charts'

    if not results_path.exists():
        print(f"\nERROR: Results not found: {results_path}")
        print("Run benchmarks first.")
        return 1

    output_dir.mkdir(exist_ok=True)

    print(f"\nInput: {results_path}")
    print(f"Output: {output_dir}")

    # Load results
    results = load_results(results_path)

    print("\nGenerating charts...")

    # Generate plots
    plot_throughput_comparison(results, output_dir)
    plot_scaling_analysis(results, output_dir)
    plot_memory_usage(results, output_dir)
    plot_latency_distribution(results, output_dir)

    print(f"\n✓ All charts generated in: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
