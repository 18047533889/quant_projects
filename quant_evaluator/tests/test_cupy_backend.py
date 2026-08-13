"""
Parity tests for CuPy GPU backend.

Verifies that GPU implementations produce identical results to CPU reference
implementations within numerical tolerance.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from quant_evaluator.backends import is_gpu_available
from quant_evaluator.kernels.fast import fast_ic_batch, fast_quantile_binning


# Skip all tests if GPU not available
pytestmark = pytest.mark.skipif(
    not is_gpu_available(),
    reason="GPU not available or CuPy not installed"
)


@pytest.fixture
def gpu_backend():
    """Create GPU backend instance."""
    from quant_evaluator.backends.cupy_backend import create_gpu_backend
    backend = create_gpu_backend()
    yield backend
    if backend:
        backend.clear_memory_pool()


class TestICBatchParity:
    """Test IC batch computation parity between CPU and GPU."""

    def test_pearson_ic_small_batch(self, gpu_backend):
        """Test Pearson IC on small batch."""
        np.random.seed(42)
        T, N, F = 50, 100, 10

        factors = np.random.randn(T, N, F)
        labels = np.random.randn(T, N)

        # CPU reference
        ic_cpu, counts_cpu = fast_ic_batch(factors, labels, method="pearson")

        # GPU implementation
        ic_gpu, counts_gpu = gpu_backend.fast_ic_batch_gpu(
            factors, labels, method="pearson"
        )

        # Check parity
        assert_allclose(ic_gpu, ic_cpu, rtol=1e-6, atol=1e-8)
        assert np.array_equal(counts_gpu, counts_cpu)

    def test_pearson_ic_large_batch(self, gpu_backend):
        """Test Pearson IC on large batch (GPU should shine here)."""
        np.random.seed(123)
        T, N, F = 252, 3000, 1000

        factors = np.random.randn(T, N, F)
        labels = np.random.randn(T, N)

        # CPU reference
        ic_cpu, counts_cpu = fast_ic_batch(factors, labels, method="pearson")

        # GPU implementation
        ic_gpu, counts_gpu = gpu_backend.fast_ic_batch_gpu(
            factors, labels, method="pearson"
        )

        # Check parity (slightly relaxed tolerance for large batch)
        assert_allclose(ic_gpu, ic_cpu, rtol=1e-5, atol=1e-7)
        assert np.array_equal(counts_gpu, counts_cpu)

    def test_pearson_ic_with_nans(self, gpu_backend):
        """Test Pearson IC with NaN values."""
        np.random.seed(456)
        T, N, F = 100, 500, 50

        factors = np.random.randn(T, N, F)
        labels = np.random.randn(T, N)

        # Inject NaNs (20% missing)
        factors[np.random.rand(T, N, F) < 0.2] = np.nan
        labels[np.random.rand(T, N) < 0.1] = np.nan

        # CPU reference
        ic_cpu, counts_cpu = fast_ic_batch(factors, labels, method="pearson")

        # GPU implementation
        ic_gpu, counts_gpu = gpu_backend.fast_ic_batch_gpu(
            factors, labels, method="pearson"
        )

        # Check parity
        assert_allclose(ic_gpu, ic_cpu, rtol=1e-6, atol=1e-8)
        assert np.array_equal(counts_gpu, counts_cpu)

    def test_pearson_ic_with_validity_masks(self, gpu_backend):
        """Test Pearson IC with explicit validity masks."""
        np.random.seed(789)
        T, N, F = 80, 400, 20

        factors = np.random.randn(T, N, F)
        labels = np.random.randn(T, N)

        factor_validity = np.random.rand(T, N, F) > 0.15
        label_validity = np.random.rand(T, N) > 0.05

        # CPU reference
        ic_cpu, counts_cpu = fast_ic_batch(
            factors, labels, method="pearson",
            factor_validity=factor_validity,
            label_validity=label_validity,
        )

        # GPU implementation
        ic_gpu, counts_gpu = gpu_backend.fast_ic_batch_gpu(
            factors, labels, method="pearson",
            factor_validity=factor_validity,
            label_validity=label_validity,
        )

        # Check parity
        assert_allclose(ic_gpu, ic_cpu, rtol=1e-6, atol=1e-8)
        assert np.array_equal(counts_gpu, counts_cpu)

    def test_spearman_ic_small_batch(self, gpu_backend):
        """Test Spearman IC on small batch."""
        np.random.seed(321)
        T, N, F = 60, 200, 15

        factors = np.random.randn(T, N, F)
        labels = np.random.randn(T, N)

        # CPU reference
        ic_cpu, counts_cpu = fast_ic_batch(factors, labels, method="spearman")

        # GPU implementation
        ic_gpu, counts_gpu = gpu_backend.fast_ic_batch_gpu(
            factors, labels, method="spearman"
        )

        # Check parity (slightly relaxed for ranking operations)
        assert_allclose(ic_gpu, ic_cpu, rtol=1e-5, atol=1e-7)
        assert np.array_equal(counts_gpu, counts_cpu)

    def test_ic_min_obs_threshold(self, gpu_backend):
        """Test IC computation respects min_obs threshold."""
        np.random.seed(111)
        T, N, F = 50, 100, 10

        factors = np.random.randn(T, N, F)
        labels = np.random.randn(T, N)

        # Create sparse data
        factors[np.random.rand(T, N, F) < 0.9] = np.nan

        min_obs = 20

        # CPU reference
        ic_cpu, counts_cpu = fast_ic_batch(
            factors, labels, method="pearson", min_obs=min_obs
        )

        # GPU implementation
        ic_gpu, counts_gpu = gpu_backend.fast_ic_batch_gpu(
            factors, labels, method="pearson", min_obs=min_obs
        )

        # Check parity
        assert_allclose(ic_gpu, ic_cpu, rtol=1e-6, atol=1e-8)
        assert np.array_equal(counts_gpu, counts_cpu)

        # Verify min_obs is respected
        assert np.all(np.isnan(ic_gpu) | (counts_gpu >= min_obs))


class TestCorrelationMatrixParity:
    """Test correlation matrix computation parity."""

    def test_pearson_correlation_matrix(self, gpu_backend):
        """Test Pearson correlation matrix."""
        np.random.seed(42)
        T, F = 252, 100

        data = np.random.randn(T, F)

        # CPU reference (using numpy corrcoef)
        # Need to compute pairwise with min_obs handling
        corr_cpu = np.corrcoef(data.T)

        # GPU implementation
        corr_gpu = gpu_backend.fast_correlation_matrix_gpu(
            data, min_obs=10, method="pearson"
        )

        # Check parity (ignore NaN positions)
        finite_mask = np.isfinite(corr_cpu) & np.isfinite(corr_gpu)
        assert_allclose(corr_gpu[finite_mask], corr_cpu[finite_mask], rtol=1e-5)

    def test_correlation_matrix_with_nans(self, gpu_backend):
        """Test correlation matrix with missing data."""
        np.random.seed(123)
        T, F = 200, 50

        data = np.random.randn(T, F)
        data[np.random.rand(T, F) < 0.1] = np.nan

        # GPU implementation
        corr_gpu = gpu_backend.fast_correlation_matrix_gpu(
            data, min_obs=30, method="pearson"
        )

        # Basic checks
        assert corr_gpu.shape == (F, F)
        assert np.allclose(np.diag(corr_gpu), 1.0, rtol=1e-6)
        assert np.allclose(corr_gpu, corr_gpu.T, rtol=1e-6)  # Symmetric


class TestQuantileRankingParity:
    """Test quantile ranking parity."""

    def test_quantile_ranking_basic(self, gpu_backend):
        """Test basic quantile ranking."""
        np.random.seed(42)
        T, N, F = 50, 200, 10

        factors = np.random.randn(T, N, F)
        n_quantiles = 5

        # CPU reference
        quantiles_cpu = fast_quantile_binning(factors, n_quantiles=n_quantiles)

        # GPU implementation
        quantiles_gpu = gpu_backend.fast_quantile_ranking_gpu(
            factors, n_quantiles=n_quantiles
        )

        # Check parity
        assert np.array_equal(quantiles_gpu, quantiles_cpu)

    def test_quantile_ranking_with_nans(self, gpu_backend):
        """Test quantile ranking with NaN values."""
        np.random.seed(456)
        T, N, F = 100, 500, 20

        factors = np.random.randn(T, N, F)
        factors[np.random.rand(T, N, F) < 0.15] = np.nan

        n_quantiles = 10

        # CPU reference
        quantiles_cpu = fast_quantile_binning(factors, n_quantiles=n_quantiles)

        # GPU implementation
        quantiles_gpu = gpu_backend.fast_quantile_ranking_gpu(
            factors, n_quantiles=n_quantiles
        )

        # Check parity
        assert np.array_equal(quantiles_gpu, quantiles_cpu)

        # Verify -1 for invalid values
        nan_mask = np.isnan(factors)
        assert np.all(quantiles_gpu[nan_mask] == -1)


class TestRollingCorrelationParity:
    """Test rolling correlation parity."""

    def test_rolling_correlation_basic(self, gpu_backend):
        """Test basic rolling correlation."""
        np.random.seed(42)
        T, F = 300, 50
        window = 20

        x = np.random.randn(T, F)
        y = np.random.randn(T, F)

        # GPU implementation
        corr_gpu = gpu_backend.fast_rolling_correlation_gpu(
            x, y, window=window, min_obs=15
        )

        # Basic checks
        assert corr_gpu.shape == (T, F)
        assert np.all(np.isnan(corr_gpu[:window-1]))  # First window-1 should be NaN

        # Check a few values manually
        for t in [window, window + 10, T - 1]:
            for f in [0, F // 2, F - 1]:
                x_window = x[t - window + 1:t + 1, f]
                y_window = y[t - window + 1:t + 1, f]

                mask = np.isfinite(x_window) & np.isfinite(y_window)
                if np.sum(mask) >= 15:
                    expected_corr = np.corrcoef(
                        x_window[mask], y_window[mask]
                    )[0, 1]

                    if np.isfinite(corr_gpu[t, f]):
                        assert_allclose(corr_gpu[t, f], expected_corr, rtol=1e-5)


class TestGPUBackendInfo:
    """Test GPU backend metadata and info methods."""

    def test_device_info(self, gpu_backend):
        """Test device info retrieval."""
        info = gpu_backend.get_device_info()

        assert "device_id" in info
        assert "name" in info
        assert "compute_capability" in info
        assert "total_memory_gb" in info
        assert "free_memory_gb" in info
        assert "multiprocessor_count" in info

        assert info["total_memory_gb"] > 0
        assert info["free_memory_gb"] > 0

    def test_memory_tracking(self, gpu_backend):
        """Test memory usage tracking."""
        # Allocate some data
        T, N, F = 100, 1000, 500
        factors = np.random.randn(T, N, F)
        labels = np.random.randn(T, N)

        initial_used, total = gpu_backend.get_memory_usage()

        # Run computation
        gpu_backend.fast_ic_batch_gpu(factors, labels, method="pearson")

        used_after, _ = gpu_backend.get_memory_usage()

        # Memory should have been used
        assert used_after >= initial_used

        # Clear memory pool
        gpu_backend.clear_memory_pool()

        used_after_clear, _ = gpu_backend.get_memory_usage()

        # Memory should be freed (or at least not increase)
        assert used_after_clear <= used_after


class TestSpeedupBenchmark:
    """Test that GPU provides meaningful speedup on large datasets."""

    @pytest.mark.slow
    def test_ic_speedup_large_batch(self, gpu_backend):
        """Verify GPU speedup on large IC computation."""
        import time

        np.random.seed(42)
        T, N, F = 252, 5000, 2000

        factors = np.random.randn(T, N, F)
        labels = np.random.randn(T, N)

        # CPU timing
        start = time.perf_counter()
        ic_cpu, _ = fast_ic_batch(factors, labels, method="pearson")
        cpu_time = time.perf_counter() - start

        # GPU timing (with warmup)
        gpu_backend.fast_ic_batch_gpu(factors[:10], labels[:10], method="pearson")

        start = time.perf_counter()
        ic_gpu, _ = gpu_backend.fast_ic_batch_gpu(
            factors, labels, method="pearson"
        )
        gpu_time = time.perf_counter() - start

        speedup = cpu_time / gpu_time

        # Clean up
        gpu_backend.clear_memory_pool()

        # Verify results match
        assert_allclose(ic_gpu, ic_cpu, rtol=1e-5, atol=1e-7)

        # Should see at least 10x speedup (conservative estimate)
        # Target is 50-200x, but account for overhead and small batches
        assert speedup > 10.0, f"GPU speedup only {speedup:.1f}x (expected >10x)"

        print(f"\nGPU speedup: {speedup:.1f}x (CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s)")
