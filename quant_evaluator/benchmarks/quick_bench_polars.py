"""
Quick benchmark for polars backend vs numpy.
"""

import numpy as np
import time

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    print("ERROR: Polars not installed")
    exit(1)

from quant_evaluator.backends.polars_backend import polars_ic_batch
from quant_evaluator.kernels.fast import fast_ic_batch

print("Quick Polars Backend Benchmark")
print("=" * 50)

# Small test
T, N, F = 100, 200, 100
print(f"\nTest size: T={T}, N={N}, F={F}")

np.random.seed(42)
factors = np.random.randn(T, N, F)
labels = np.random.randn(T, N)
factor_ids = tuple(f"f{i:03d}" for i in range(F))

# NumPy
start = time.perf_counter()
ic_np, _ = fast_ic_batch(factors, labels)
np_time = time.perf_counter() - start
print(f"NumPy:  {np_time:.3f}s")

# Polars
start = time.perf_counter()
ic_pl, _ = polars_ic_batch(factors, labels, factor_ids)
pl_time = time.perf_counter() - start
print(f"Polars: {pl_time:.3f}s")

speedup = np_time / pl_time
print(f"Speedup: {speedup:.2f}x")

# Verify correctness
max_diff = np.nanmax(np.abs(ic_np - ic_pl))
print(f"Max difference: {max_diff:.2e}")

print("\n" + "=" * 50)
print("Benchmark complete!")
