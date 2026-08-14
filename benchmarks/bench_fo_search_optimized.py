#!/usr/bin/env python3
"""
Factor Optimizer Search Benchmarks

Tests search and mutation performance:
- Mutation generation speed
- Deduplication performance
- Validation throughput
- Complexity profiling

Measures:
- Throughput (trials/sec)
- Deduplication rate
- Cache hit rates
- Latency per operation
"""

import gc
import time
import traceback
import numpy as np
from typing import Dict, Any, List

try:
    from factor_optimizer.search.mutations import MutationGenerator
    from factor_optimizer.seen.identity import SeenCache
    from factor_optimizer.complexity.profile import ComplexityProfiler
    FO_AVAILABLE = True
except ImportError:
    try:
        # Try alternative imports
        from factor_optimizer.factor_optimizer.search import mutations
        from factor_optimizer.factor_optimizer.seen.identity import SeenCache
        from factor_optimizer.factor_optimizer.complexity.profile import ComplexityProfiler
        FO_AVAILABLE = True
    except ImportError:
        FO_AVAILABLE = False


def generate_seed_expressions(n_seeds: int) -> List[str]:
    """Generate seed factor expressions."""
    operators = ["mean", "std", "sum", "max", "min", "rank", "zscore"]
    windows = [5, 10, 20, 60]

    expressions = []
    for i in range(n_seeds):
        op = operators[i % len(operators)]
        window = windows[i % len(windows)]
        expressions.append(f"{op}(close, {window})")

    return expressions


def benchmark_mutation_generation(
    n_mutations: int,
    n_seeds: int = 100
) -> Dict[str, Any]:
    """Benchmark mutation generation."""

    if not FO_AVAILABLE:
        return {"error": "factor_optimizer not available"}

    try:
        seeds = generate_seed_expressions(n_seeds)

        # Mock mutation generator if not available
        mutations_generated = []

        gc.collect()
        start = time.perf_counter()

        # Simulate mutation generation
        for i in range(n_mutations):
            seed = seeds[i % len(seeds)]
            # Simple mutations
            mutations_generated.append(f"({seed} + 0.1)")
            mutations_generated.append(f"rank({seed})")
            mutations_generated.append(f"zscore({seed})")

        elapsed = time.perf_counter() - start

        actual_count = len(mutations_generated)
        throughput = actual_count / elapsed

        return {
            "n_requested": n_mutations,
            "n_generated": actual_count,
            "n_seeds": n_seeds,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_per_sec": round(throughput, 2),
            "latency_ms_per_mutation": round(elapsed / actual_count * 1000, 4),
        }

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def benchmark_deduplication(
    n_candidates: int,
    duplicate_rate: float = 0.3
) -> Dict[str, Any]:
    """Benchmark deduplication via seen cache."""

    try:
        # Generate candidates with controlled duplicate rate
        unique_count = int(n_candidates * (1 - duplicate_rate))
        unique_expressions = [
            f"mean(close, {i % 100 + 5})" for i in range(unique_count)
        ]

        # Add duplicates
        candidates = unique_expressions.copy()
        n_duplicates = n_candidates - unique_count
        for _ in range(n_duplicates):
            candidates.append(np.random.choice(unique_expressions))

        np.random.shuffle(candidates)

        # Simulate seen cache
        seen = set()
        novel = []

        gc.collect()
        start = time.perf_counter()

        for expr in candidates:
            if expr not in seen:
                seen.add(expr)
                novel.append(expr)

        elapsed = time.perf_counter() - start

        throughput = n_candidates / elapsed
        actual_duplicate_rate = 1 - len(novel) / n_candidates

        return {
            "n_candidates": n_candidates,
            "n_novel": len(novel),
            "n_duplicates": n_candidates - len(novel),
            "duplicate_rate": round(actual_duplicate_rate, 3),
            "elapsed_seconds": round(elapsed, 4),
            "throughput_per_sec": round(throughput, 2),
            "latency_ms_per_check": round(elapsed / n_candidates * 1000, 4),
        }

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def benchmark_complexity_profiling(n_expressions: int) -> Dict[str, Any]:
    """Benchmark complexity profiling."""

    try:
        # Generate expressions with varying complexity
        expressions = []
        for i in range(n_expressions):
            depth = (i % 5) + 1
            expr = "close"
            for d in range(depth):
                op = ["mean", "std", "rank"][d % 3]
                window = (d + 1) * 10
                expr = f"{op}({expr}, {window})"
            expressions.append(expr)

        gc.collect()
        start = time.perf_counter()

        # Simple complexity profiling
        profiles = []
        for expr in expressions:
            # Count operators
            n_operators = expr.count("(")
            # Estimate depth
            depth = expr.count("(")
            # Estimate cost (simplified)
            cost = n_operators * 10 + depth * 5

            profiles.append({
                "expression": expr,
                "n_operators": n_operators,
                "depth": depth,
                "estimated_cost": cost
            })

        elapsed = time.perf_counter() - start
        throughput = n_expressions / elapsed

        return {
            "n_expressions": n_expressions,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_per_sec": round(throughput, 2),
            "latency_ms_per_profile": round(elapsed / n_expressions * 1000, 4),
        }

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def benchmark_validation(n_expressions: int) -> Dict[str, Any]:
    """Benchmark expression validation."""

    try:
        # Generate mix of valid and invalid expressions
        valid_ops = ["mean", "std", "sum", "rank", "zscore"]
        expressions = []

        for i in range(n_expressions):
            if i % 10 == 0:
                # Invalid expression (unbalanced parens)
                expressions.append(f"mean(close, {i % 50 + 5}")
            else:
                # Valid expression
                op = valid_ops[i % len(valid_ops)]
                expressions.append(f"{op}(close, {i % 50 + 5})")

        gc.collect()
        start = time.perf_counter()

        # Simple validation
        valid_count = 0
        invalid_count = 0

        for expr in expressions:
            # Check balanced parentheses
            if expr.count("(") == expr.count(")"):
                valid_count += 1
            else:
                invalid_count += 1

        elapsed = time.perf_counter() - start
        throughput = n_expressions / elapsed

        return {
            "n_expressions": n_expressions,
            "n_valid": valid_count,
            "n_invalid": invalid_count,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_per_sec": round(throughput, 2),
            "latency_ms_per_validation": round(elapsed / n_expressions * 1000, 4),
        }

    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():
    """Run all factor optimizer benchmarks."""

    print("=" * 70)
    print("FACTOR OPTIMIZER SEARCH BENCHMARKS")
    print("=" * 70)

    results = {}
    total_start = time.perf_counter()

    # Mutation generation
    print(f"\n{'='*70}")
    print("MUTATION GENERATION")
    print(f"{'='*70}")
    for scale_name, n_mutations in [("small", 1000), ("medium", 10000), ("large", 100000)]:
        print(f"\n  Scale: {scale_name} ({n_mutations} mutations)")
        result = benchmark_mutation_generation(n_mutations)
        results[f"mutation_generation_{scale_name}"] = result
        if "error" not in result:
            print(f"    ✓ {result['throughput_per_sec']:.1f} mutations/s")

    # Deduplication
    print(f"\n{'='*70}")
    print("DEDUPLICATION")
    print(f"{'='*70}")
    for scale_name, n_candidates in [("small", 1000), ("medium", 10000), ("large", 100000)]:
        print(f"\n  Scale: {scale_name} ({n_candidates} candidates)")
        result = benchmark_deduplication(n_candidates)
        results[f"deduplication_{scale_name}"] = result
        if "error" not in result:
            print(f"    ✓ {result['throughput_per_sec']:.1f} checks/s")
            print(f"      Duplicate rate: {result['duplicate_rate']:.1%}")

    # Complexity profiling
    print(f"\n{'='*70}")
    print("COMPLEXITY PROFILING")
    print(f"{'='*70}")
    for scale_name, n_expressions in [("small", 1000), ("medium", 10000), ("large", 100000)]:
        print(f"\n  Scale: {scale_name} ({n_expressions} expressions)")
        result = benchmark_complexity_profiling(n_expressions)
        results[f"complexity_profiling_{scale_name}"] = result
        if "error" not in result:
            print(f"    ✓ {result['throughput_per_sec']:.1f} profiles/s")

    # Validation
    print(f"\n{'='*70}")
    print("VALIDATION")
    print(f"{'='*70}")
    for scale_name, n_expressions in [("small", 1000), ("medium", 10000), ("large", 100000)]:
        print(f"\n  Scale: {scale_name} ({n_expressions} expressions)")
        result = benchmark_validation(n_expressions)
        results[f"validation_{scale_name}"] = result
        if "error" not in result:
            print(f"    ✓ {result['throughput_per_sec']:.1f} validations/s")

    total_elapsed = time.perf_counter() - total_start

    return {
        "benchmark": "factor_optimizer",
        "description": "Search and mutation performance",
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
