"""
Test backend selection and registry functionality for factor_preprocess.

Tests automatic backend selection based on data size and operation type.
"""

import numpy as np
import pytest
from factor_preprocess.backends import (
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

    def test_cs_rank_selection_small(self):
        """Test cross-sectional rank backend selection for small data."""
        selector = BackendSelector(
            numba_threshold=100_000,
            polars_threshold=200_000,
        )

        # Small data should use NumPy
        backend = selector.select_for_cs_rank(T=100, N=100)
        assert backend == BackendType.NUMPY

    def test_cs_rank_selection_large(self):
        """Test cross-sectional rank backend selection for large data."""
        selector = BackendSelector(
            polars_threshold=100_000,
        )

        # Large data should prefer Polars (if available)
        backend = selector.select_for_cs_rank(T=500, N=5000)  # 2.5M elements
        caps = get_backend_capabilities()

        if caps[BackendType.POLARS].available:
            assert backend == BackendType.POLARS
        else:
            assert backend == BackendType.NUMPY

    def test_rolling_selection_small(self):
        """Test rolling operation backend selection for small data."""
        selector = BackendSelector(numba_threshold=100_000)

        # Small rolling should use NumPy
        backend = selector.select_for_rolling(T=100, N=100, window=20)
        assert backend == BackendType.NUMPY

    def test_rolling_selection_large(self):
        """Test rolling operation backend selection for large data."""
        selector = BackendSelector(numba_threshold=50_000)

        # Large rolling should prefer Numba
        backend = selector.select_for_rolling(T=1000, N=5000, window=20)
        caps = get_backend_capabilities()

        if caps[BackendType.NUMBA].available:
            assert backend == BackendType.NUMBA
        else:
            assert backend == BackendType.NUMPY

    def test_cs_zscore_selection(self):
        """Test z-score backend selection."""
        selector = BackendSelector(
            cupy_threshold=500_000,
            polars_threshold=100_000,
        )

        # Medium data
        backend = selector.select_for_cs_zscore(T=252, N=3000)
        caps = get_backend_capabilities()

        if caps[BackendType.POLARS].available:
            assert backend == BackendType.POLARS
        else:
            assert backend == BackendType.NUMPY

    def test_neutralization_selection(self):
        """Test neutralization backend selection."""
        selector = BackendSelector(
            numba_threshold=50_000,
            polars_threshold=100_000,
        )

        # Large neutralization task
        backend = selector.select_for_neutralization(T=252, N=3000, n_features=5)
        caps = get_backend_capabilities()

        if caps[BackendType.POLARS].available:
            assert backend == BackendType.POLARS
        elif caps[BackendType.NUMBA].available:
            assert backend == BackendType.NUMBA
        else:
            assert backend == BackendType.NUMPY

    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        selector = BackendSelector(
            numba_threshold=10_000_000,  # Very high threshold
            polars_threshold=20_000_000,
        )

        # Even large data should use NumPy with high threshold
        backend = selector.select_for_cs_rank(T=1000, N=5000)
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

        registry.register("cs_rank", BackendType.NUMPY, numpy_impl)
        registry.register("cs_rank", BackendType.NUMBA, numba_impl)

        # Get NumPy implementation
        impl = registry.get("cs_rank", preferred=BackendType.NUMPY)
        assert impl is not None
        assert impl(5) == 10

        # Get Numba implementation
        impl = registry.get("cs_rank", preferred=BackendType.NUMBA)
        assert impl is not None
        assert impl(5) == 15

    def test_fallback(self):
        """Test fallback to available implementations."""
        registry = BackendRegistry()

        def numpy_impl(x):
            return x * 2

        registry.register("cs_rank", BackendType.NUMPY, numpy_impl)

        # Request Polars but should fallback to NumPy
        impl = registry.get("cs_rank", preferred=BackendType.POLARS)
        assert impl is not None
        assert impl(5) == 10

    def test_list_operations(self):
        """Test listing registered operations."""
        registry = BackendRegistry()

        registry.register("cs_rank", BackendType.NUMPY, lambda x: x)
        registry.register("rolling_mean", BackendType.NUMPY, lambda x: x)

        ops = registry.list_operations()
        assert "cs_rank" in ops
        assert "rolling_mean" in ops

    def test_list_backends(self):
        """Test listing backends for an operation."""
        registry = BackendRegistry()

        registry.register("cs_rank", BackendType.NUMPY, lambda x: x)
        registry.register("cs_rank", BackendType.POLARS, lambda x: x)

        backends = registry.list_backends("cs_rank")
        assert BackendType.NUMPY in backends
        assert BackendType.POLARS in backends

    def test_metadata(self):
        """Test storing and retrieving metadata."""
        registry = BackendRegistry()

        metadata = {"version": "1.0", "requires": "polars>=0.18"}
        registry.register(
            "cs_rank",
            BackendType.POLARS,
            lambda x: x,
            metadata=metadata,
        )

        retrieved = registry.get_metadata("cs_rank", BackendType.POLARS)
        assert retrieved is not None
        assert retrieved["version"] == "1.0"
        assert retrieved["requires"] == "polars>=0.18"


class TestBenchmarking:
    """Test benchmarking utilities."""

    def test_benchmark_rolling_runs(self):
        """Test that rolling benchmarking completes without error."""
        results = benchmark_backends(
            operation="rolling",
            T=100,
            N=200,
            window=20,
            n_runs=1,
        )

        # Should have at least NumPy results
        assert "numpy" in results
        assert results["numpy"]["available"]
        assert results["numpy"]["mean_time"] > 0

    def test_benchmark_cs_rank_runs(self):
        """Test that cs_rank benchmarking completes without error."""
        results = benchmark_backends(
            operation="cs_rank",
            T=100,
            N=500,
            n_runs=1,
        )

        assert "numpy" in results
        assert results["numpy"]["available"]

    def test_benchmark_cs_zscore_runs(self):
        """Test that cs_zscore benchmarking completes without error."""
        results = benchmark_backends(
            operation="cs_zscore",
            T=100,
            N=500,
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

    # Simulate typical preprocessing workflow dimensions
    scenarios = [
        (252, 3000, "yearly, many assets"),
        (1260, 5000, "multi-year, large universe"),
        (21, 500, "monthly, small universe"),
    ]

    for T, N, desc in scenarios:
        backend = selector.select_for_cs_rank(T, N)
        print(f"CS Rank {desc} ({T}×{N}): {backend.value}")
        assert backend in [BackendType.NUMPY, BackendType.NUMBA, BackendType.POLARS]

        backend = selector.select_for_rolling(T, N, window=20)
        print(f"Rolling {desc} ({T}×{N}): {backend.value}")
        assert backend in [BackendType.NUMPY, BackendType.NUMBA, BackendType.CUPY]


def test_integration_global_registry():
    """Test global registry decorator and retrieval."""

    @register_backend("custom_transform", BackendType.NUMPY)
    def custom_numpy(x):
        return x + 1

    # Retrieve from global registry
    impl = get_backend("custom_transform", preferred=BackendType.NUMPY)
    assert impl is not None
    assert impl(5) == 6


def test_integration_with_existing_kernels():
    """Test that selector works with existing kernel implementations."""
    from factor_preprocess.kernels.fast import fast_cs_rank, fast_rolling_mean

    selector = BackendSelector()

    # Test with real data
    data = np.random.randn(100, 200)

    # Should select appropriate backend
    backend = selector.select_for_cs_rank(T=100, N=200)
    assert backend in [BackendType.NUMPY, BackendType.POLARS]

    # Verify existing kernels still work
    result = fast_cs_rank(data, axis=-1)
    assert result.shape == data.shape

    result = fast_rolling_mean(data, window=20)
    assert result.shape == data.shape
