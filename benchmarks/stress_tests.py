#!/usr/bin/env python3
"""
Comprehensive Stress Tests

Large-scale, long-running, and high-concurrency stress tests:
1. Memory pressure tests
2. Long-running stability tests
3. High-concurrency simulations
4. Large dataset handling
5. Resource exhaustion scenarios

Tests system behavior under extreme conditions.
"""

import gc
import time
import traceback
import tempfile
import shutil
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import json

# Memory tracking
def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def stress_test_memory_large_arrays(max_size_gb: float = 2.0) -> Dict[str, Any]:
    """Stress test with large array allocations."""
    print(f"\n  Testing memory pressure up to {max_size_gb:.1f} GB...")

    try:
        initial_mem = get_memory_usage_mb()
        results = []

        # Gradually increase array size
        sizes_mb = [100, 500, 1000, 2000, 5000]  # MB

        for size_mb in sizes_mb:
            if size_mb > max_size_gb * 1000:
                break

            gc.collect()
            start_mem = get_memory_usage_mb()

            try:
                # Allocate large array
                n_elements = int(size_mb * 1024 * 1024 / 8)  # float64
                arr = np.random.randn(n_elements)

                # Perform operation
                result = arr.sum()

                end_mem = get_memory_usage_mb()
                allocated = end_mem - start_mem

                results.append({
                    "target_size_mb": size_mb,
                    "allocated_mb": round(allocated, 2),
                    "peak_memory_mb": round(end_mem, 2),
                    "success": True
                })

                # Cleanup
                del arr
                gc.collect()

            except MemoryError as e:
                results.append({
                    "target_size_mb": size_mb,
                    "error": "MemoryError",
                    "success": False
                })
                break

        final_mem = get_memory_usage_mb()

        return {
            "test": "memory_large_arrays",
            "initial_memory_mb": round(initial_mem, 2),
            "final_memory_mb": round(final_mem, 2),
            "memory_leaked_mb": round(final_mem - initial_mem, 2),
            "allocations": results,
            "success": True
        }

    except Exception as e:
        return {
            "test": "memory_large_arrays",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def stress_test_repeated_allocations(n_iterations: int = 1000, size_mb: float = 100) -> Dict[str, Any]:
    """Stress test with repeated allocations to detect memory leaks."""
    print(f"\n  Testing {n_iterations} repeated allocations ({size_mb} MB each)...")

    try:
        initial_mem = get_memory_usage_mb()
        memory_samples = []

        n_elements = int(size_mb * 1024 * 1024 / 8)

        start = time.perf_counter()

        for i in range(n_iterations):
            # Allocate
            arr = np.random.randn(n_elements)

            # Use it
            _ = arr.sum()

            # Deallocate
            del arr

            # Sample memory every 100 iterations
            if i % 100 == 0:
                gc.collect()
                mem = get_memory_usage_mb()
                memory_samples.append(mem)

        gc.collect()
        elapsed = time.perf_counter() - start
        final_mem = get_memory_usage_mb()

        # Check for memory leak
        if len(memory_samples) >= 2:
            mem_growth = memory_samples[-1] - memory_samples[0]
            mem_growth_rate = mem_growth / len(memory_samples)
        else:
            mem_growth = 0
            mem_growth_rate = 0

        return {
            "test": "repeated_allocations",
            "n_iterations": n_iterations,
            "size_mb_per_allocation": size_mb,
            "elapsed_seconds": round(elapsed, 2),
            "throughput_alloc_per_sec": round(n_iterations / elapsed, 2),
            "initial_memory_mb": round(initial_mem, 2),
            "final_memory_mb": round(final_mem, 2),
            "memory_growth_mb": round(mem_growth, 2),
            "memory_growth_rate_mb_per_sample": round(mem_growth_rate, 4),
            "leak_detected": abs(mem_growth) > 50,  # > 50 MB growth
            "success": True
        }

    except Exception as e:
        return {
            "test": "repeated_allocations",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def stress_test_long_running_computation(duration_seconds: float = 60) -> Dict[str, Any]:
    """Long-running computation stress test."""
    print(f"\n  Running long computation ({duration_seconds}s)...")

    try:
        initial_mem = get_memory_usage_mb()
        start = time.perf_counter()

        iterations = 0
        memory_samples = []
        computation_results = []

        while time.perf_counter() - start < duration_seconds:
            # Simulate complex computation
            size = np.random.randint(1000, 10000)
            arr1 = np.random.randn(size, 100)
            arr2 = np.random.randn(100, size)

            # Matrix multiplication
            result = arr1 @ arr2

            # Store result summary
            computation_results.append(result.sum())

            iterations += 1

            # Sample memory every 10 iterations
            if iterations % 10 == 0:
                mem = get_memory_usage_mb()
                memory_samples.append(mem)

            # Cleanup
            del arr1, arr2, result

        gc.collect()
        elapsed = time.perf_counter() - start
        final_mem = get_memory_usage_mb()

        return {
            "test": "long_running_computation",
            "target_duration_s": duration_seconds,
            "actual_duration_s": round(elapsed, 2),
            "iterations_completed": iterations,
            "throughput_iter_per_sec": round(iterations / elapsed, 2),
            "initial_memory_mb": round(initial_mem, 2),
            "final_memory_mb": round(final_mem, 2),
            "peak_memory_mb": round(max(memory_samples) if memory_samples else final_mem, 2),
            "memory_stable": abs(final_mem - initial_mem) < 100,  # < 100 MB growth
            "success": True
        }

    except Exception as e:
        return {
            "test": "long_running_computation",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def stress_test_high_concurrency_simulation(n_tasks: int = 100, task_duration_ms: float = 100) -> Dict[str, Any]:
    """Simulate high concurrency (sequential execution)."""
    print(f"\n  Simulating {n_tasks} concurrent tasks...")

    try:
        initial_mem = get_memory_usage_mb()

        # Simulate concurrent tasks
        task_results = []
        start = time.perf_counter()

        for task_id in range(n_tasks):
            task_start = time.perf_counter()

            # Simulate task work
            size = 1000 + task_id % 100
            arr = np.random.randn(size, size)
            result = arr.sum()

            task_elapsed = (time.perf_counter() - task_start) * 1000  # ms

            task_results.append({
                "task_id": task_id,
                "elapsed_ms": round(task_elapsed, 2),
                "result": float(result)
            })

            del arr

        elapsed = time.perf_counter() - start
        final_mem = get_memory_usage_mb()

        # Compute statistics
        task_times = [t['elapsed_ms'] for t in task_results]
        mean_time = np.mean(task_times)
        p95_time = np.percentile(task_times, 95)
        p99_time = np.percentile(task_times, 99)

        return {
            "test": "high_concurrency_simulation",
            "n_tasks": n_tasks,
            "total_elapsed_s": round(elapsed, 2),
            "throughput_tasks_per_sec": round(n_tasks / elapsed, 2),
            "mean_task_latency_ms": round(mean_time, 2),
            "p95_task_latency_ms": round(p95_time, 2),
            "p99_task_latency_ms": round(p99_time, 2),
            "initial_memory_mb": round(initial_mem, 2),
            "final_memory_mb": round(final_mem, 2),
            "success": True
        }

    except Exception as e:
        return {
            "test": "high_concurrency_simulation",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def stress_test_large_dataset_processing(n_factors: int = 50000, n_observations: int = 1000) -> Dict[str, Any]:
    """Test processing of very large datasets."""
    print(f"\n  Processing large dataset: {n_factors} factors × {n_observations} observations...")

    try:
        initial_mem = get_memory_usage_mb()

        # Generate large dataset
        print(f"    Generating data...")
        gen_start = time.perf_counter()
        data = np.random.randn(n_factors, n_observations).astype(np.float32)
        gen_elapsed = time.perf_counter() - gen_start

        after_gen_mem = get_memory_usage_mb()

        # Process: compute statistics
        print(f"    Computing statistics...")
        proc_start = time.perf_counter()

        means = data.mean(axis=1)
        stds = data.std(axis=1)
        ranks = np.argsort(np.argsort(data, axis=1), axis=1)

        proc_elapsed = time.perf_counter() - proc_start

        after_proc_mem = get_memory_usage_mb()

        # Cleanup
        del data, means, stds, ranks
        gc.collect()

        final_mem = get_memory_usage_mb()

        total_elements = n_factors * n_observations
        data_size_mb = total_elements * 4 / 1024 / 1024  # float32

        return {
            "test": "large_dataset_processing",
            "n_factors": n_factors,
            "n_observations": n_observations,
            "total_elements": total_elements,
            "data_size_mb": round(data_size_mb, 2),
            "generation_time_s": round(gen_elapsed, 3),
            "processing_time_s": round(proc_elapsed, 3),
            "throughput_elements_per_sec": round(total_elements / proc_elapsed, 0),
            "initial_memory_mb": round(initial_mem, 2),
            "after_generation_mb": round(after_gen_mem, 2),
            "after_processing_mb": round(after_proc_mem, 2),
            "final_memory_mb": round(final_mem, 2),
            "memory_cleaned_up": (after_proc_mem - final_mem) > data_size_mb * 0.5,
            "success": True
        }

    except Exception as e:
        return {
            "test": "large_dataset_processing",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def main():
    """Run all stress tests."""

    print("=" * 70)
    print("COMPREHENSIVE STRESS TESTS")
    print("=" * 70)
    print("\nWARNING: These tests may consume significant system resources.")
    print("Initial memory usage: {:.1f} MB".format(get_memory_usage_mb()))

    results = {}
    total_start = time.perf_counter()

    # Memory pressure tests
    print(f"\n{'='*70}")
    print("MEMORY PRESSURE TESTS")
    print(f"{'='*70}")

    results['memory_large_arrays'] = stress_test_memory_large_arrays(max_size_gb=1.0)
    results['repeated_allocations'] = stress_test_repeated_allocations(n_iterations=500)

    # Long-running tests
    print(f"\n{'='*70}")
    print("LONG-RUNNING STABILITY TESTS")
    print(f"{'='*70}")

    results['long_running_computation'] = stress_test_long_running_computation(duration_seconds=30)

    # Concurrency tests
    print(f"\n{'='*70}")
    print("HIGH CONCURRENCY SIMULATION")
    print(f"{'='*70}")

    results['high_concurrency'] = stress_test_high_concurrency_simulation(n_tasks=200)

    # Large dataset tests
    print(f"\n{'='*70}")
    print("LARGE DATASET PROCESSING")
    print(f"{'='*70}")

    results['large_dataset'] = stress_test_large_dataset_processing(n_factors=10000, n_observations=5000)

    total_elapsed = time.perf_counter() - total_start
    final_mem = get_memory_usage_mb()

    return {
        "test_suite": "stress_tests",
        "description": "Large-scale stress and stability tests",
        "results": results,
        "total_time_s": round(total_elapsed, 2),
        "final_memory_mb": round(final_mem, 2)
    }


if __name__ == "__main__":
    result = main()
    print("\n" + "="*70)
    print("STRESS TEST SUMMARY")
    print("="*70)
    print(json.dumps(result, indent=2))
