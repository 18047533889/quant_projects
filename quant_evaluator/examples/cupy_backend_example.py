"""
Example usage of CuPy GPU backend for quant_evaluator.

Demonstrates GPU-accelerated IC computation, correlation matrices, and ranking
with automatic fallback to CPU when GPU unavailable.
"""

import numpy as np
import time

from quant_evaluator.backends import (
    is_gpu_available,
    get_available_backends,
    OptionalDependencyMissing,
)
from quant_evaluator.backends.selector import BackendSelector, benchmark_backends
from quant_evaluator.kernels.fast import fast_ic_batch


def example_basic_gpu_usage():
    """Basic GPU backend usage example."""
    print("=" * 70)
    print("Example 1: Basic GPU Backend Usage")
    print("=" * 70)

    print(f"\nAvailable backends: {get_available_backends()}")
    print(f"GPU available: {is_gpu_available()}")

    if not is_gpu_available():
        print("\nGPU not available. Install cupy with: pip install cupy-cuda12x")
        return

    from quant_evaluator.backends.cupy_backend import create_gpu_backend

    # Create GPU backend
    gpu_backend = create_gpu_backend()
    if gpu_backend is None:
        print("Failed to create GPU backend")
        return

    # Print device info
    info = gpu_backend.get_device_info()
    print(f"\nGPU Device: {info['name']}")
    print(f"Compute Capability: {info['compute_capability']}")
    print(f"Total Memory: {info['total_memory_gb']:.2f} GB")
    print(f"Free Memory: {info['free_memory_gb']:.2f} GB")

    # Generate test data
    T, N, F = 252, 3000, 1000
    print(f"\nTest data shape: T={T}, N={N}, F={F}")
    print(f"Total elements: {T * N * F:,}")

    factors = np.random.randn(T, N, F)
    labels = np.random.randn(T, N)

    # CPU computation
    print("\nRunning CPU computation...")
    start = time.perf_counter()
    ic_cpu, counts_cpu = fast_ic_batch(factors, labels, method="pearson")
    cpu_time = time.perf_counter() - start
    print(f"CPU time: {cpu_time:.3f}s")

    # GPU computation
    print("\nRunning GPU computation...")
    start = time.perf_counter()
    ic_gpu, counts_gpu = gpu_backend.fast_ic_batch_gpu(
        factors, labels, method="pearson"
    )
    gpu_time = time.perf_counter() - start
    print(f"GPU time: {gpu_time:.3f}s")

    speedup = cpu_time / gpu_time
    print(f"\n🚀 GPU Speedup: {speedup:.1f}x")

    # Verify correctness
    max_diff = np.nanmax(np.abs(ic_cpu - ic_gpu))
    print(f"Max difference: {max_diff:.2e}")

    # Memory usage
    used_gb, total_gb = gpu_backend.get_memory_usage()
    print(f"\nGPU Memory Used: {used_gb:.2f} GB / {total_gb:.2f} GB")

    # Clean up
    gpu_backend.clear_memory_pool()


def example_automatic_backend_selection():
    """Automatic backend selection based on data size."""
    print("\n" + "=" * 70)
    print("Example 2: Automatic Backend Selection")
    print("=" * 70)

    selector = BackendSelector(prefer_gpu=True)

    # Test different data sizes
    test_cases = [
        (50, 500, 100, "Small batch"),
        (100, 1000, 500, "Medium batch"),
        (252, 3000, 1000, "Large batch"),
        (500, 5000, 2000, "Very large batch"),
    ]

    print("\nBackend selection based on data size:\n")
    for T, N, F, description in test_cases:
        backend = selector.select_backend_for_ic_batch(
            factor_shape=(T, N, F),
            label_shape=(T, N),
            method="pearson",
        )
        total_elements = T * N * F
        print(f"{description:20s} ({T:3d}×{N:4d}×{F:4d}) -> {backend:10s} ({total_elements:,} elements)")


def example_correlation_matrix():
    """GPU-accelerated correlation matrix computation."""
    print("\n" + "=" * 70)
    print("Example 3: Correlation Matrix on GPU")
    print("=" * 70)

    if not is_gpu_available():
        print("\nGPU not available.")
        return

    from quant_evaluator.backends.cupy_backend import create_gpu_backend

    gpu_backend = create_gpu_backend()
    if gpu_backend is None:
        return

    # Generate factor data
    T, F = 252, 500
    print(f"\nComputing correlation matrix for {F} factors over {T} periods")

    data = np.random.randn(T, F)

    # GPU computation
    start = time.perf_counter()
    corr_matrix = gpu_backend.fast_correlation_matrix_gpu(
        data, min_obs=100, method="pearson"
    )
    gpu_time = time.perf_counter() - start

    print(f"GPU time: {gpu_time:.3f}s")
    print(f"Correlation matrix shape: {corr_matrix.shape}")
    print(f"Diagonal values (should be 1.0): {np.diag(corr_matrix)[:5]}")
    print(f"Matrix is symmetric: {np.allclose(corr_matrix, corr_matrix.T)}")

    gpu_backend.clear_memory_pool()


def example_quantile_ranking():
    """GPU-accelerated quantile ranking."""
    print("\n" + "=" * 70)
    print("Example 4: Quantile Ranking on GPU")
    print("=" * 70)

    if not is_gpu_available():
        print("\nGPU not available.")
        return

    from quant_evaluator.backends.cupy_backend import create_gpu_backend

    gpu_backend = create_gpu_backend()
    if gpu_backend is None:
        return

    # Generate factor data
    T, N, F = 252, 3000, 500
    n_quantiles = 5

    print(f"\nRanking {F} factors into {n_quantiles} quantiles")
    print(f"Data shape: ({T}, {N}, {F})")

    factors = np.random.randn(T, N, F)

    # GPU computation
    start = time.perf_counter()
    quantiles = gpu_backend.fast_quantile_ranking_gpu(
        factors, n_quantiles=n_quantiles
    )
    gpu_time = time.perf_counter() - start

    print(f"GPU time: {gpu_time:.3f}s")
    print(f"Quantile assignments shape: {quantiles.shape}")

    # Check quantile distribution
    for q in range(n_quantiles):
        count = np.sum(quantiles == q)
        pct = 100 * count / np.sum(quantiles >= 0)
        print(f"Quantile {q}: {count:,} assets ({pct:.1f}%)")

    gpu_backend.clear_memory_pool()


def example_benchmark_comparison():
    """Benchmark CPU vs GPU performance."""
    print("\n" + "=" * 70)
    print("Example 5: Comprehensive Benchmark")
    print("=" * 70)

    if not is_gpu_available():
        print("\nGPU not available. Showing CPU-only results.")

    print("\nRunning benchmarks (this may take a minute)...\n")

    # Benchmark different data sizes
    test_configs = [
        {"T": 100, "N": 1000, "F": 500, "label": "Small"},
        {"T": 252, "N": 3000, "F": 1000, "label": "Medium"},
        {"T": 500, "N": 5000, "F": 2000, "label": "Large"},
    ]

    for config in test_configs:
        T, N, F = config["T"], config["N"], config["F"]
        label = config["label"]

        print(f"{label} batch ({T}×{N}×{F}):")

        results = benchmark_backends(
            operation="ic",
            T=T,
            N=N,
            F=F,
            n_runs=3,
        )

        for backend, stats in results.items():
            if stats.get("available", False) and stats.get("mean_time") is not None:
                time_ms = stats["mean_time"] * 1000
                speedup = stats.get("speedup", 1.0)
                print(f"  {backend:8s}: {time_ms:8.2f}ms (speedup: {speedup:6.1f}x)")

        print()


def example_error_handling():
    """Demonstrate error handling and fallback."""
    print("\n" + "=" * 70)
    print("Example 6: Error Handling and Fallback")
    print("=" * 70)

    # Try to create GPU backend
    try:
        from quant_evaluator.backends.cupy_backend import create_gpu_backend

        gpu_backend = create_gpu_backend()

        if gpu_backend is None:
            print("\nGPU backend not available, falling back to CPU")
            print("Reasons could be:")
            print("  - CuPy not installed")
            print("  - No CUDA GPU detected")
            print("  - GPU driver issues")
        else:
            print("\nGPU backend created successfully!")
            info = gpu_backend.get_device_info()
            print(f"Using: {info['name']}")

    except OptionalDependencyMissing as e:
        print(f"\nOptional dependency missing: {e.package}")
        print(f"Required for: {e.feature}")
        print(f"Install with: pip install {e.package}")

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Falling back to CPU computation")


if __name__ == "__main__":
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║         CuPy GPU Backend Examples for quant_evaluator             ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    example_basic_gpu_usage()
    example_automatic_backend_selection()
    example_correlation_matrix()
    example_quantile_ranking()
    example_benchmark_comparison()
    example_error_handling()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
