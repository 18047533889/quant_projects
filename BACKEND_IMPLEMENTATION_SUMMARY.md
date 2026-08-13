# Automatic Backend Selection Implementation Summary

## Overview
Implemented automatic backend selection system for both `quant_evaluator` and `factor_preprocess` projects. The system automatically detects available acceleration libraries (numba, cupy, polars) and selects the optimal backend based on data size and operation type.

## Implementation Details

### Core Components

1. **BackendSelector** (`selector.py`)
   - Auto-detects available backends (numpy, numba, cupy, polars)
   - Selects optimal backend based on configurable thresholds
   - Operation-specific selection logic (IC, quantile, rolling, etc.)
   - Optional auto-benchmarking for threshold calibration

2. **BackendRegistry** (`registry.py`)
   - Register multiple implementations of operations across backends
   - Automatic fallback chain: CuPy → Numba → Polars → NumPy
   - Metadata support for versioning and requirements

3. **Benchmarking Utility** (`selector.py:benchmark_backends`)
   - Compare backend performance for specific data sizes
   - Supports multiple operations (IC, quantile, rolling, etc.)
   - Returns detailed timing statistics

### Selection Logic

#### quant_evaluator
- **IC Computation**: CuPy (>500K) → Polars (>100K, F≥100) → Numba (>50K) → NumPy
- **Quantile Binning**: CuPy (>500K) → Polars (>100K) → Numba (>50K) → NumPy
- **Rolling Operations**: Polars (>50K) → Numba (>50K) → NumPy

#### factor_preprocess
- **Cross-sectional Rank**: Polars (>100K) → Numba (>50K) → NumPy
- **Cross-sectional Z-score**: Polars (>100K) → Numba (>50K) → NumPy
- **Rolling Operations**: Polars (>50K) → Numba (>50K) → NumPy
- **Neutralization**: Polars (>100K) → NumPy

## Key Features

1. **Graceful Degradation**: Always falls back to NumPy if accelerators unavailable
2. **GPU Detection**: Checks actual GPU availability, not just CuPy installation
3. **Configurable Thresholds**: Adjust selection thresholds per use case
4. **Capability Introspection**: Query available backends and their versions
5. **Backend Registry**: Register custom implementations with automatic fallback

## Files Modified/Created

### quant_evaluator
- `/home/shw/quant_projects/quant_evaluator/backends/__init__.py` - Enhanced exports
- `/home/shw/quant_projects/quant_evaluator/backends/selector.py` - Already existed
- `/home/shw/quant_projects/quant_evaluator/backends/registry.py` - Fixed BackendType import
- `/home/shw/quant_projects/quant_evaluator/tests/test_backend_selector.py` - Test adjustments

### factor_preprocess
- `/home/shw/quant_projects/factor_preprocess/factor_preprocess/backends/__init__.py` - Enhanced exports
- `/home/shw/quant_projects/factor_preprocess/factor_preprocess/backends/selector.py` - Already existed
- `/home/shw/quant_projects/factor_preprocess/factor_preprocess/backends/registry.py` - Fixed BackendType import

### Documentation
- `/home/shw/quant_projects/example_backend_usage.py` - Complete usage example

## Critical Fix
Fixed BackendType enum duplication issue where `registry.py` defined its own BackendType instead of importing from `selector.py`, causing fallback logic to fail. Now both modules share a single source of truth.

## Test Results

### quant_evaluator
- 17 tests passed
- Coverage: capability detection, selection logic, benchmarking, registry operations

### factor_preprocess
- 20 tests passed
- Coverage: capability detection, operation-specific selection, benchmarking, integration

## Usage Examples

### Basic Usage
```python
from quant_evaluator.backends import BackendSelector, get_backend_capabilities

# Auto-detect capabilities
caps = get_backend_capabilities()

# Create selector with default thresholds
selector = BackendSelector()

# Get optimal backend for data size
backend = selector.select_for_ic(T=252, N=5000, F=100)
print(f"Selected: {backend.value}")
```

### Custom Thresholds
```python
selector = BackendSelector(
    numba_threshold=100_000,
    cupy_threshold=1_000_000,
    polars_threshold=200_000,
)
```

### Backend Registry
```python
from quant_evaluator.backends import BackendRegistry, BackendType

registry = BackendRegistry()
registry.register("my_op", BackendType.NUMPY, numpy_impl)
registry.register("my_op", BackendType.NUMBA, numba_impl)

# Get implementation with automatic fallback
impl = registry.get("my_op", preferred=BackendType.CUPY)
```

### Benchmarking
```python
from quant_evaluator.backends import benchmark_backends

results = benchmark_backends(
    operation="ic",
    T=252, N=3000, F=100,
    n_runs=5,
)

for backend, metrics in results.items():
    print(f"{backend}: {metrics['mean_time']*1000:.2f} ms")
```

## Performance Characteristics

- **NumPy**: Reference implementation, always available
- **Numba**: 5-50x speedup for numeric operations via JIT compilation
- **CuPy**: 10-100x speedup for large arrays on GPU
- **Polars**: 3-20x speedup for dataframe operations, especially groupby/rolling

## Future Enhancements

1. Persistent benchmark cache for threshold calibration
2. Runtime profiling for adaptive threshold adjustment
3. Multi-GPU support for CuPy backend
4. JAX backend integration
5. Automatic parallelization strategy selection
