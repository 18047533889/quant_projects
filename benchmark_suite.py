"""
Performance benchmark suite for optimization validation.
Run before and after optimizations to measure impact.
"""

import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, asdict
import tracemalloc


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    category: str
    median_time_ms: float
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    peak_memory_mb: float
    iterations: int
    timestamp: str


class PerformanceBenchmark:
    """Run performance benchmarks and track results."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("./benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def benchmark(
        self,
        name: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        n_iterations: int = 10,
        warmup: int = 2,
        category: str = "general"
    ) -> BenchmarkResult:
        """Benchmark a function with multiple iterations."""
        kwargs = kwargs or {}

        # Warmup runs
        for _ in range(warmup):
            func(*args, **kwargs)

        # Timed runs
        times = []
        peak_memory = 0

        for _ in range(n_iterations):
            tracemalloc.start()
            gc_collect()

            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed = time.perf_counter() - start

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            times.append(elapsed * 1000)  # Convert to ms
            peak_memory = max(peak_memory, peak / 1024 / 1024)  # Convert to MB

        # Calculate statistics
        result = BenchmarkResult(
            name=name,
            category=category,
            median_time_ms=float(np.median(times)),
            mean_time_ms=float(np.mean(times)),
            std_time_ms=float(np.std(times)),
            min_time_ms=float(np.min(times)),
            max_time_ms=float(np.max(times)),
            peak_memory_mb=peak_memory,
            iterations=n_iterations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

        self.results.append(result)
        return result

    def save_results(self, filename: str = None):
        """Save benchmark results to JSON."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        output_path = self.output_dir / filename

        results_dict = [asdict(r) for r in self.results]

        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'results': results_dict
            }, f, indent=2)

        print(f"Results saved to {output_path}")
        return output_path

    def print_summary(self):
        """Print summary of benchmark results."""
        if not self.results:
            print("No benchmark results to display")
            return

        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)

        # Group by category
        by_category = {}
        for result in self.results:
            by_category.setdefault(result.category, []).append(result)

        for category, results in sorted(by_category.items()):
            print(f"\n{category.upper()}")
            print("-" * 80)
            print(f"{'Benchmark':<40} {'Median':>12} {'Mean±Std':>20} {'Memory':>10}")
            print("-" * 80)

            for result in sorted(results, key=lambda r: r.median_time_ms):
                print(f"{result.name:<40} "
                      f"{result.median_time_ms:>10.2f}ms  "
                      f"{result.mean_time_ms:>8.2f}±{result.std_time_ms:>6.2f}ms  "
                      f"{result.peak_memory_mb:>8.2f}MB")

        print("\n" + "="*80 + "\n")


def compare_benchmarks(before_file: Path, after_file: Path):
    """Compare two benchmark runs and show improvements."""
    with open(before_file) as f:
        before = json.load(f)

    with open(after_file) as f:
        after = json.load(f)

    before_results = {r['name']: r for r in before['results']}
    after_results = {r['name']: r for r in after['results']}

    print("\n" + "="*80)
    print("BENCHMARK COMPARISON: BEFORE vs AFTER")
    print("="*80)
    print(f"Before: {before['timestamp']}")
    print(f"After:  {after['timestamp']}")
    print("="*80)

    improvements = []

    print(f"\n{'Benchmark':<40} {'Before':>12} {'After':>12} {'Speedup':>12}")
    print("-" * 80)

    for name in sorted(before_results.keys()):
        if name not in after_results:
            continue

        before_time = before_results[name]['median_time_ms']
        after_time = after_results[name]['median_time_ms']
        speedup = before_time / after_time

        improvements.append({
            'name': name,
            'speedup': speedup,
            'before': before_time,
            'after': after_time
        })

        # Color code based on speedup
        if speedup >= 1.1:
            status = "✅"
        elif speedup >= 0.9:
            status = "➖"
        else:
            status = "❌"

        print(f"{name:<40} "
              f"{before_time:>10.2f}ms  "
              f"{after_time:>10.2f}ms  "
              f"{status} {speedup:>8.2f}x")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    speedups = [i['speedup'] for i in improvements]
    print(f"Average speedup:  {np.mean(speedups):.2f}x")
    print(f"Median speedup:   {np.median(speedups):.2f}x")
    print(f"Best speedup:     {np.max(speedups):.2f}x")
    print(f"Worst speedup:    {np.min(speedups):.2f}x")

    regressions = [i for i in improvements if i['speedup'] < 0.9]
    if regressions:
        print(f"\n⚠️  {len(regressions)} regressions detected:")
        for reg in regressions:
            print(f"  - {reg['name']}: {reg['speedup']:.2f}x slower")

    significant_improvements = [i for i in improvements if i['speedup'] >= 2.0]
    if significant_improvements:
        print(f"\n🎉 {len(significant_improvements)} significant improvements (≥2x):")
        for imp in sorted(significant_improvements, key=lambda x: -x['speedup'])[:5]:
            print(f"  - {imp['name']}: {imp['speedup']:.2f}x faster")

    print("\n" + "="*80 + "\n")


def gc_collect():
    """Force garbage collection."""
    import gc
    gc.collect()


# ============================================================================
# BENCHMARK DEFINITIONS
# ============================================================================

def benchmark_quant_evaluator(benchmark: PerformanceBenchmark):
    """Benchmark quant_evaluator critical paths."""
    print("\n📊 Benchmarking quant_evaluator...")

    try:
        from quant_evaluator.metrics import ic, quantile
        from quant_evaluator.contracts import FactorBatch, AxisRef, LabelBundle
    except ImportError:
        print("⚠️  quant_evaluator not available, skipping")
        return

    # Test data
    T, N, F = 252, 1000, 100  # 252 days, 1000 assets, 100 factors
    factor_values = np.random.randn(T, N, F)
    label_values = np.random.randn(T, N)

    # Create factor batch
    factor_batch = FactorBatch(
        values=factor_values,
        time_axis=AxisRef(index=None, values=None, label="time"),
        asset_axis=AxisRef(index=None, values=None, label="asset"),
        factor_axis=AxisRef(index=None, values=None, label="factor"),
        validity=None
    )

    label_bundle = LabelBundle(
        values=label_values,
        time_axis=AxisRef(index=None, values=None, label="time"),
        asset_axis=AxisRef(index=None, values=None, label="asset"),
        validity=None
    )

    # Benchmark IC computation
    try:
        benchmark.benchmark(
            name="IC computation (252×1000×100)",
            func=ic.compute_daily_ic,
            kwargs={
                'factor_batch': factor_batch,
                'label_bundle': label_bundle,
                'method': 'pearson',
                'min_obs': 20
            },
            n_iterations=5,
            category="quant_evaluator"
        )
    except Exception as e:
        print(f"  ⚠️  IC benchmark failed: {e}")

    # Benchmark quantile computation
    try:
        benchmark.benchmark(
            name="Quantile returns (252×1000×100)",
            func=quantile.compute_quantile_returns,
            kwargs={
                'factor_batch': factor_batch,
                'forward_returns': label_values,
                'n_quantiles': 5,
                'min_samples': 20
            },
            n_iterations=5,
            category="quant_evaluator"
        )
    except Exception as e:
        print(f"  ⚠️  Quantile benchmark failed: {e}")

    print("✅ quant_evaluator benchmarks complete")


def benchmark_factor_optimizer(benchmark: PerformanceBenchmark):
    """Benchmark factor_optimizer critical paths."""
    print("\n🔍 Benchmarking factor_optimizer...")

    try:
        from factor_optimizer.search import pareto, lineage
    except ImportError:
        print("⚠️  factor_optimizer not available, skipping")
        return

    # Benchmark Pareto frontier operations
    try:
        from factor_optimizer.search.pareto import ParetoFrontier, ParetoPoint

        frontier = ParetoFrontier(maximize=[True, True])

        def add_points():
            for i in range(100):
                point = ParetoPoint(
                    trial_id=f"trial_{i}",
                    objectives=(np.random.rand(), np.random.rand())
                )
                frontier.add_point(point)

        benchmark.benchmark(
            name="Pareto frontier (100 points)",
            func=add_points,
            n_iterations=10,
            category="factor_optimizer"
        )
    except Exception as e:
        print(f"  ⚠️  Pareto benchmark failed: {e}")

    # Benchmark lineage tree operations
    try:
        from factor_optimizer.search.lineage import LineageTree

        tree = LineageTree()

        # Build tree
        for i in range(100):
            parent_ids = [f"trial_{i-1}"] if i > 0 else []
            tree.add_node(f"trial_{i}", parent_ids)

        def query_ancestors():
            for i in range(50, 100):
                tree.get_ancestors(f"trial_{i}")

        benchmark.benchmark(
            name="Lineage tree queries (50 queries)",
            func=query_ancestors,
            n_iterations=10,
            category="factor_optimizer"
        )
    except Exception as e:
        print(f"  ⚠️  Lineage benchmark failed: {e}")

    print("✅ factor_optimizer benchmarks complete")


def benchmark_research_control(benchmark: PerformanceBenchmark):
    """Benchmark research_control critical paths."""
    print("\n📝 Benchmarking research_control...")

    try:
        from research_control.sync import idempotency
        from research_control.ledger import trial, campaign
    except ImportError:
        print("⚠️  research_control not available, skipping")
        return

    # Benchmark event processing
    try:
        # Create test events
        events = [
            {
                'event_id': f'event_{i}',
                'campaign_id': 'test_campaign',
                'state': 'open',
                'timestamp': time.time(),
                'metadata': {'trial_num': i}
            }
            for i in range(100)
        ]

        # Note: This would need actual ledger setup
        print("  ⚠️  Event processing benchmark needs setup")
    except Exception as e:
        print(f"  ⚠️  Event benchmark failed: {e}")

    print("✅ research_control benchmarks complete")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run full benchmark suite."""
    print("="*80)
    print("PERFORMANCE BENCHMARK SUITE")
    print("="*80)
    print("This will benchmark critical paths across all packages")
    print("Results will be saved for before/after comparison")
    print("="*80)

    benchmark = PerformanceBenchmark()

    # Run benchmarks for each package
    benchmark_quant_evaluator(benchmark)
    benchmark_factor_optimizer(benchmark)
    benchmark_research_control(benchmark)

    # Print summary
    benchmark.print_summary()

    # Save results
    output_file = benchmark.save_results()

    print(f"\n✅ Benchmark complete!")
    print(f"Results saved to: {output_file}")
    print("\nTo compare with future runs:")
    print(f"  python benchmark_suite.py --compare {output_file} <new_results.json>")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        if len(sys.argv) != 4:
            print("Usage: python benchmark_suite.py --compare <before.json> <after.json>")
            sys.exit(1)

        before_file = Path(sys.argv[2])
        after_file = Path(sys.argv[3])

        if not before_file.exists():
            print(f"Error: {before_file} not found")
            sys.exit(1)

        if not after_file.exists():
            print(f"Error: {after_file} not found")
            sys.exit(1)

        compare_benchmarks(before_file, after_file)
    else:
        main()
