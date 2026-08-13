"""
Test backend selection and registry functionality for quant_evaluator.

Tests automatic backend selection based on data size and operation type.
"""

import numpy as np
import pytest
from quant_evaluator.backends import (
    BackendSelector,
    BackendType,
    get_backend_capabilities,
    benchmark_backends,
    BackendRegistry,
    register_backend,
    get_backend,
)


class TestBackendSelector:
    """Test BackendSelector data-aware selection."""

    def test_capability_detection(self):
        """Test that backend capabilities are detected correctly."""
        caps = get_backend_capabilities()

        # NumPy should always be available
        assert caps[BackendType.NUMPY].available
        assert caps[BackendType.NUMPY].version is not None

        # Check structure
        for backend_type, cap in caps.items():
            assert hasattr(cap, 'available')
            assert hasattr(cap, 'backend')
            if cap.available:
                assert cap.version is not None

    def test_ic_selection_small(self):
        """Test IC backend selection for small data."""
        selector = BackendSelector(
            numba_threshold=100_000,
            cupy_threshold=1_000_000,
        )

        # Small data should use NumPy
        backend = selector.select_for_ic(T=10, N=100, F=10)
        assert backend == BackendType.NUMPY

    def test_ic_selection_medium(self):
        """Test IC backend selection for medium data."""
        selector = BackendSelector(
            numba_threshold=50_000,
            cupy_threshold=1_000_000,
            polars_threshold=20_000_000,  # High threshold to avoid Polars
        )

        # Medium data should prefer Numba (if available)
        backend = selector.select_for_ic(T=100, N=1000, F=100)  # 10M elements
        caps = get_backend_capabilities()

        if caps[BackendType.NUMBA].available:
            assert backend == BackendType.NUMBA
        else:
            assert backend == BackendType.NUMPY

    def test_ic_selection_large(self):
        """Test IC backend selection for large data."""
        selector = BackendSelector(
            numba_threshold=50_000,
            cupy_threshold=500_000,
            polars_threshold=50_000_000,  # Very high to test numba/cupy
        )

        # Large data with few factors to avoid Polars
        backend = selector.select_for_ic(T=1000, N=10000, F=50)  # 500M elements, F < 100
        caps = get_backend_capabilities()

        if caps[BackendType.CUPY].available:
            assert backend == BackendType.CUPY
        elif caps[BackendType.NUMBA].available:
            assert backend == BackendType.NUMBA
        else:
            assert backend == BackendType.NUMPY

    def test_rolling_selection(self):
        """Test rolling operation backend selection."""
        selector = BackendSelector(numba_threshold=50_000)

        # Small rolling
        backend = selector.select_for_rolling(T=100, N=100, window=20)
        assert backend == BackendType.NUMPY

        # Large rolling should prefer Numba
        backend = selector.select_for_rolling(T=1000, N=5000, window=20)
        caps = get_backend_capabilities()

        if caps[BackendType.NUMBA].available:
            assert backend == BackendType.NUMBA
        else:
            assert backend == BackendType.NUMPY

    def test_quantile_selection(self):
        """Test quantile binning backend selection."""
        selector = BackendSelector(
            numba_threshold=50_000,
            cupy_threshold=500_000,
            polars_threshold=50_000_000,
        )

        backend = selector.select_for_quantile(T=252, N=3000, F=100)
        assert backend in [BackendType.NUMPY, BackendType.NUMBA, BackendType.CUPY, BackendType.POLARS]

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        selector = BackendSelector(
            numba_threshold=100_000_000,  # Extremely high threshold
            cupy_threshold=1_000_000_000,
            polars_threshold=1_000_000_000,
        )

        # Even medium data should use NumPy with very high threshold
        backend = selector.select_for_ic(T=100, N=1000, F=100)  # 10M elements
        assert backend == BackendType.NUMPY


class TestBackendRegistry:
    """Test BackendRegistry for managing implementations."""

    def test_register_and_get(self):
        """Test registering and retrieving implementations."""
        registry = BackendRegistry()

        def numpy_impl(x):
            return x * 2

        def numba_impl(x):
            return x * 3

        registry.register("test_op", BackendType.NUMPY, numpy_impl)
        registry.register("test_op", BackendType.NUMBA, numba_impl)

        # Get NumPy implementation
        impl = registry.get("test_op", preferred=BackendType.NUMPY)
        assert impl is not None
        assert impl(5) == 10

        # Get Numba implementation
        impl = registry.get("test_op", preferred=BackendType.NUMBA)
        assert impl is not None
        assert impl(5) == 15

    def test_fallback(self):
        """Test fallback to available implementations."""
        registry = BackendRegistry()

        def numpy_impl(x):
            return x * 2

        registry.register("test_op", BackendType.NUMPY, numpy_impl)

        # Request CuPy but should fallback to NumPy
        impl = registry.get("test_op", preferred=BackendType.CUPY)
        assert impl is not None
        assert impl(5) == 10

    def test_list_operations(self):
        """Test listing registered operations."""
        registry = BackendRegistry()

        registry.register("op1", BackendType.NUMPY, lambda x: x)
        registry.register("op2", BackendType.NUMPY, lambda x: x)

        ops = registry.list_operations()
        assert "op1" in ops
        assert "op2" in ops

    def test_list_backends(self):
        """Test listing backends for an operation."""
        registry = BackendRegistry()

        registry.register("test_op", BackendType.NUMPY, lambda x: x)
        registry.register("test_op", BackendType.NUMBA, lambda x: x)

        backends = registry.list_backends("test_op")
        assert BackendType.NUMPY in backends
        assert BackendType.NUMBA in backends

    def test_metadata(self):
        """Test storing and retrieving metadata."""
        registry = BackendRegistry()

        metadata = {"version": "1.0", "requires": "numba>=0.50"}
        registry.register(
            "test_op",
            BackendType.NUMBA,
            lambda x: x,
            metadata=metadata,
        )

        retrieved = registry.get_metadata("test_op", BackendType.NUMBA)
        assert retrieved is not None
        assert retrieved["version"] == "1.0"
        assert retrieved["requires"] == "numba>=0.50"


class TestBenchmarking:
    """Test benchmarking utilities."""

    def test_benchmark_ic_runs(self):
        """Test that IC benchmarking completes without error."""
        # Small benchmark
        results = benchmark_backends(
            operation="ic",
            T=50,
            N=100,
            F=10,
            n_runs=1,
        )

        # Should have at least NumPy results
        assert "numpy" in results
        assert results["numpy"]["available"]
        assert results["numpy"]["mean_time"] > 0

    def test_benchmark_rolling_runs(self):
        """Test that rolling benchmarking completes without error."""
        results = benchmark_backends(
            operation="rolling",
            T=100,
            N=200,
            F=10,
            n_runs=1,
        )

        assert "numpy" in results
        assert results["numpy"]["available"]

    def test_benchmark_invalid_operation(self):
        """Test that invalid operation raises error."""
        with pytest.raises(ValueError, match="Unknown operation"):
            benchmark_backends(operation="invalid_op", n_runs=1)


def test_integration_selector_with_real_data():
    """Integration test with realistic data sizes."""
    selector = BackendSelector()

    # Simulate typical quant workflow dimensions
    scenarios = [
        (252, 3000, 1000, "yearly, many factors"),
        (21, 5000, 100, "monthly, fewer factors"),
        (1260, 500, 50, "multi-year, small universe"),
    ]

    for T, N, F, desc in scenarios:
        backend = selector.select_for_ic(T, N, F)
        print(f"IC {desc} ({T}×{N}×{F}): {backend.value}")
        assert backend in [BackendType.NUMPY, BackendType.NUMBA, BackendType.CUPY, BackendType.POLARS]


def test_integration_global_registry():
    """Test global registry decorator and retrieval."""

    @register_backend("custom_op", BackendType.NUMPY)
    def custom_numpy(x):
        return x + 1

    # Retrieve from global registry
    impl = get_backend("custom_op", preferred=BackendType.NUMPY)
    assert impl is not None
    assert impl(5) == 6
