"""
Performance optimization utilities and profiling helpers.
Provides tools for benchmarking, profiling, and optimizing critical paths.
"""

import functools
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, TypeVar
import cProfile
import pstats
import io
import numpy as np
import pandas as pd

F = TypeVar('F', bound=Callable[..., Any])


class PerformanceTimer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        if self.verbose:
            print(f"[{self.name}] {self.elapsed*1000:.2f}ms")

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000 if self.elapsed else 0.0


def timeit(func: F) -> F:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[TIMING] {func.__name__}: {elapsed*1000:.2f}ms")
        return result
    return wrapper


class MemoryProfiler:
    """Context manager for memory profiling."""

    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.snapshot_start = None
        self.snapshot_end = None

    def __enter__(self):
        tracemalloc.start()
        self.snapshot_start = tracemalloc.take_snapshot()
        return self

    def __exit__(self, *args):
        self.snapshot_end = tracemalloc.take_snapshot()
        tracemalloc.stop()

        if self.verbose:
            top_stats = self.snapshot_end.compare_to(
                self.snapshot_start, 'lineno'
            )

            print(f"\n[MEMORY] {self.name}")
            print(f"Total memory increase: {sum(stat.size_diff for stat in top_stats) / 1024 / 1024:.2f} MB")
            print("\nTop 5 memory allocations:")
            for stat in top_stats[:5]:
                print(f"  {stat}")


def profile_cpu(func: F) -> F:
    """Decorator to profile CPU usage."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        # Print stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        print(f"\n[CPU PROFILE] {func.__name__}")
        print(s.getvalue())

        return result
    return wrapper


class PerformanceComparison:
    """Compare performance of different implementations."""

    def __init__(self, name: str, n_iterations: int = 10):
        self.name = name
        self.n_iterations = n_iterations
        self.results: Dict[str, List[float]] = {}

    def benchmark(self, impl_name: str, func: Callable, *args, **kwargs) -> float:
        """Benchmark a single implementation."""
        times = []
        for _ in range(self.n_iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            times.append(time.perf_counter() - start)

        self.results[impl_name] = times
        return np.median(times)

    def report(self) -> None:
        """Print comparison report."""
        print(f"\n{'='*60}")
        print(f"Performance Comparison: {self.name}")
        print(f"{'='*60}")

        if not self.results:
            print("No results to compare")
            return

        # Calculate statistics
        stats = {}
        for name, times in self.results.items():
            stats[name] = {
                'median': np.median(times) * 1000,
                'mean': np.mean(times) * 1000,
                'std': np.std(times) * 1000,
                'min': np.min(times) * 1000,
                'max': np.max(times) * 1000,
            }

        # Find baseline (slowest)
        baseline_name = max(stats.keys(), key=lambda k: stats[k]['median'])
        baseline_time = stats[baseline_name]['median']

        # Print results
        print(f"\n{'Implementation':<30} {'Median':<12} {'Mean±Std':<20} {'Speedup':>10}")
        print("-" * 75)

        for name in sorted(stats.keys(), key=lambda k: stats[k]['median']):
            s = stats[name]
            speedup = baseline_time / s['median']
            speedup_str = f"{speedup:.2f}x" if name != baseline_name else "baseline"

            print(f"{name:<30} {s['median']:>8.2f}ms  "
                  f"{s['mean']:>8.2f}±{s['std']:>6.2f}ms  "
                  f"{speedup_str:>10}")

        print(f"\n{'='*60}\n")


def compare_vectorized_vs_loop(
    data: pd.Series,
    window: int = 20,
    operation: str = "mean"
) -> None:
    """Compare vectorized vs loop-based operations."""

    comp = PerformanceComparison(f"Rolling {operation} (window={window})", n_iterations=5)

    # Method 1: Vectorized
    if operation == "mean":
        comp.benchmark("Vectorized (pandas)", lambda: data.rolling(window).mean())
        comp.benchmark("Apply lambda (slow)", lambda: data.rolling(window).apply(lambda x: x.mean()))
    elif operation == "std":
        comp.benchmark("Vectorized (pandas)", lambda: data.rolling(window).std())
        comp.benchmark("Apply lambda (slow)", lambda: data.rolling(window).apply(lambda x: x.std()))

    comp.report()


def check_unnecessary_copies(func: F) -> F:
    """Decorator to detect unnecessary DataFrame copies."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        copy_count = [0]
        original_copy = pd.DataFrame.copy

        def tracked_copy(self, *args, **kwargs):
            copy_count[0] += 1
            return original_copy(self, *args, **kwargs)

        pd.DataFrame.copy = tracked_copy

        try:
            result = func(*args, **kwargs)
            if copy_count[0] > 0:
                print(f"[COPY WARNING] {func.__name__} made {copy_count[0]} DataFrame copies")
            return result
        finally:
            pd.DataFrame.copy = original_copy

    return wrapper


class CacheAnalyzer:
    """Analyze cache hit rates and effectiveness."""

    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.cache: Dict[str, Any] = {}

    def get(self, key: str, compute_func: Callable) -> Any:
        """Get from cache or compute."""
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            value = compute_func()
            self.cache[key] = value
            return value

    def report(self) -> None:
        """Print cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0

        print(f"\n[CACHE STATS]")
        print(f"  Hits: {self.hits}")
        print(f"  Misses: {self.misses}")
        print(f"  Hit Rate: {hit_rate:.1f}%")
        print(f"  Cache Size: {len(self.cache)} items")


def detect_nested_loops(source_code: str) -> List[Dict]:
    """Detect nested loops in source code that could be vectorized."""
    import ast

    class LoopDetector(ast.NodeVisitor):
        def __init__(self):
            self.nested_loops = []
            self.loop_depth = 0

        def visit_For(self, node):
            self.loop_depth += 1
            if self.loop_depth >= 2:
                self.nested_loops.append({
                    'line': node.lineno,
                    'depth': self.loop_depth,
                    'type': 'for'
                })
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_While(self, node):
            self.loop_depth += 1
            if self.loop_depth >= 2:
                self.nested_loops.append({
                    'line': node.lineno,
                    'depth': self.loop_depth,
                    'type': 'while'
                })
            self.generic_visit(node)
            self.loop_depth -= 1

    try:
        tree = ast.parse(source_code)
        detector = LoopDetector()
        detector.visit(tree)
        return detector.nested_loops
    except SyntaxError:
        return []


# Example usage patterns
if __name__ == "__main__":
    # Test data
    data = pd.Series(np.random.randn(10000))

    # Compare vectorized vs loop
    compare_vectorized_vs_loop(data, window=20, operation="mean")

    # Memory profiling example
    with MemoryProfiler("Large DataFrame Creation"):
        df = pd.DataFrame(np.random.randn(10000, 100))

    # Cache analyzer example
    cache = CacheAnalyzer()

    def expensive_computation():
        time.sleep(0.1)
        return np.random.randn(1000)

    # First call - cache miss
    result1 = cache.get("key1", expensive_computation)
    # Second call - cache hit
    result2 = cache.get("key1", expensive_computation)

    cache.report()
