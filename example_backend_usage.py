#!/usr/bin/env python3
"""
Example demonstrating automatic backend selection in quant_evaluator and factor_preprocess.

Shows:
1. Capability detection
2. Automatic backend selection based on data size
3. Backend benchmarking
4. Backend registry usage
"""

import numpy as np

print("=" * 70)
print("Automatic Backend Selection Demo")
print("=" * 70)

# ============================================================================
# Part 1: quant_evaluator - IC computation backends
# ============================================================================
print("\n[1] quant_evaluator - IC Computation Backends")
print("-" * 70)

from quant_evaluator.backends import (
    BackendSelector,
    BackendType,
    get_backend_capabilities,
    benchmark_backends,
)

# Detect available backends
caps = get_backend_capabilities()
print("\nAvailable backends:")
for backend_type, capability in caps.items():
    status = "✓" if capability.available else "✗"
    print(f"  {status} {backend_type.value:8s} - ", end="")
    if capability.available:
        print(f"v{capability.version}", end="")
        if capability.supports_gpu:
            print(f" (GPU: {capability.device_info})", end="")
        if capability.supports_jit:
            print(" (JIT)", end="")
        print()
    else:
        print("Not available")

# Automatic selection for different data sizes
selector = BackendSelector(
    numba_threshold=50_000,
    cupy_threshold=500_000,
    polars_threshold=100_000,
)

print("\nAutomatic backend selection for IC computation:")
test_cases = [
    (10, 100, 50, "Small"),      # 50K elements
    (100, 1000, 100, "Medium"),  # 10M elements
    (252, 5000, 200, "Large"),   # 252M elements
]

for T, N, F, size in test_cases:
    backend = selector.select_for_ic(T, N, F)
    total = T * N * F
    print(f"  {size:8s} ({T:4d}×{N:5d}×{F:4d} = {total:12,d} elements) → {backend.value}")

# Benchmark if numba available
if caps[BackendType.NUMBA].available:
    print("\nRunning quick benchmarks (T=100, N=500, F=50)...")
    results = benchmark_backends(
        operation="ic",
        T=100, N=500, F=50,
        n_runs=3,
    )
    print("Benchmark results:")
    for backend_name, metrics in results.items():
        if isinstance(metrics, dict) and 'mean_time' in metrics:
            print(f"  {backend_name:8s}: {metrics['mean_time']*1000:8.2f} ms")
        elif isinstance(metrics, dict) and 'error' in metrics:
            print(f"  {backend_name:8s}: {metrics['error']}")

# ============================================================================
# Part 2: factor_preprocess - Factor transformation backends
# ============================================================================
print("\n[2] factor_preprocess - Factor Transformation Backends")
print("-" * 70)

from factor_preprocess.backends import (
    BackendSelector as FPSelector,
    BackendType as FPBackendType,
    get_backend_capabilities as fp_get_caps,
)

# Detect capabilities
fp_caps = fp_get_caps()
print("\nAvailable backends:")
for backend_type, capability in fp_caps.items():
    status = "✓" if capability.available else "✗"
    print(f"  {status} {backend_type.value:8s}", end="")
    if capability.available:
        print(f" - v{capability.version}")
    else:
        print(" - Not available")

# Automatic selection
fp_selector = FPSelector(
    numba_threshold=50_000,
    polars_threshold=100_000,
)

print("\nAutomatic backend selection for cross-sectional operations:")
cs_cases = [
    (10, 1000, "Small"),     # 10K elements
    (252, 5000, "Medium"),   # 1.26M elements
    (1000, 10000, "Large"),  # 10M elements
]

for T, N, size in cs_cases:
    backend = fp_selector.select_for_cs_rank(T, N)
    print(f"  {size:8s} ({T:4d}×{N:5d} = {T*N:10,d} elements) → {backend.value}")

# ============================================================================
# Part 3: Backend Registry - Manual registration
# ============================================================================
print("\n[3] Backend Registry - Custom Operation Registration")
print("-" * 70)

from quant_evaluator.backends import BackendRegistry, register_backend, get_backend

# Register custom implementations
def numpy_custom(x):
    return np.mean(x) * 2

def numba_custom(x):
    return np.mean(x) * 2  # In real case, this would be @njit decorated

registry = BackendRegistry()
registry.register("custom_op", BackendType.NUMPY, numpy_custom)

if caps[BackendType.NUMBA].available:
    registry.register("custom_op", BackendType.NUMBA, numba_custom)

print("\nRegistered 'custom_op' implementations:")
available = registry.list_backends("custom_op")
for backend in available:
    print(f"  ✓ {backend.value}")

# Get implementation with fallback
impl = registry.get("custom_op", preferred=BackendType.CUPY)
if impl:
    test_data = np.array([1, 2, 3, 4, 5])
    result = impl(test_data)
    print(f"\nExecuted custom_op (fell back to available backend):")
    print(f"  Input: {test_data}")
    print(f"  Result: {result}")

print("\n" + "=" * 70)
print("Demo complete!")
print("=" * 70)
