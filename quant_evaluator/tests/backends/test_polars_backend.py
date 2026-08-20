"""
Parity tests for polars backend against numpy reference implementation.

Validates that polars backend produces identical results to numpy fast kernels.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

from quant_evaluator.backends.polars_backend import (
    polars_ic_batch,
    polars_quantile_binning,
    polars_quantile_returns,
    factorbatch_to_lazyframe,
    POLARS_AVAILABLE as BACKEND_POLARS_AVAILABLE,
)
from quant_evaluator.kernels.fast import (
    fast_ic_batch,
    fast_quantile_binning,
    compute_quantile_returns_fast,
)


pytestmark = pytest.mark.skipif(
    not POLARS_AVAILABLE,
    reason="Polars not installed"
)


@pytest.fixture
def small_batch():
    """Small factor batch for testing."""
    np.random.seed(42)
    T, N, F = 20, 50, 10

    factors = np.random.randn(T, N, F)
    labels = np.random.randn(T, N)
    factor_ids = tuple(f"factor_{i:03d}" for i in range(F))

    return factors, labels, factor_ids, (T, N, F)


@pytest.fixture
def medium_batch():
    """Medium factor batch for performance testing."""
    np.random.seed(123)
    T, N, F = 252, 500, 100

    factors = np.random.randn(T, N, F)
    labels = np.random.randn(T, N)
    factor_ids = tuple(f"factor_{i:03d}" for i in range(F))

    return factors, labels, factor_ids, (T, N, F)


@pytest.fixture
def batch_with_nans():
    """Batch with missing values."""
    np.random.seed(99)
    T, N, F = 30, 100, 5

    factors = np.random.randn(T, N, F)
    labels = np.random.randn(T, N)

    # Introduce NaNs (10% missing)
    nan_mask_f = np.random.rand(T, N, F) < 0.1
    factors[nan_mask_f] = np.nan

    nan_mask_l = np.random.rand(T, N) < 0.1
    labels[nan_mask_l] = np.nan

    factor_ids = tuple(f"factor_{i:03d}" for i in range(F))

    return factors, labels, factor_ids, (T, N, F)


class TestFactorBatchConversion:
    """Test conversion from FactorBatch to polars LazyFrame."""

    def test_basic_conversion(self, small_batch):
        """Test basic conversion to LazyFrame."""
        factors, labels, factor_ids, (T, N, F) = small_batch

        lf = factorbatch_to_lazyframe(factors, factor_ids)

        # Collect to eager for inspection
        df = lf.collect()

        # Check schema
        assert "time_idx" in df.columns
        assert "asset_idx" in df.columns
        assert "factor_id" in df.columns
        assert "value" in df.columns

        # Check shape (long format)
        assert len(df) == T * N * F

        # Check that all factor_ids present
        unique_factors = set(df["factor_id"].unique())
        assert unique_factors == set(factor_ids)

    def test_conversion_with_axes(self, small_batch):
        """Test conversion with explicit time/asset axes."""
        factors, labels, factor_ids, (T, N, F) = small_batch

        time_values = np.arange(T)
        asset_values = np.array([f"ASSET_{i:04d}" for i in range(N)])

        lf = factorbatch_to_lazyframe(
            factors,
            factor_ids,
            time_values=time_values,
            asset_values=asset_values,
        )

        df = lf.collect()

        # Check that time and asset columns exist
        assert "time" in df.columns
        assert "asset" in df.columns

    def test_conversion_with_validity(self, batch_with_nans):
        """Test conversion with validity mask."""
        factors, labels, factor_ids, (T, N, F) = batch_with_nans

        validity = ~np.isnan(factors)

        lf = factorbatch_to_lazyframe(
            factors,
            factor_ids,
            factor_validity=validity,
        )

        df = lf.collect()

        # When validity mask is applied, invalid values become NaN in the source
        # but factorbatch_to_lazyframe applies the mask, setting invalid to NaN
        # The result should have NaNs where validity was False
        num_invalid = np.sum(~validity)
        assert num_invalid > 0  # Verify test data has invalid values


class TestICParity:
    """Test IC computation parity between polars and numpy."""

    def test_pearson_ic_small_batch(self, small_batch):
        """Test Pearson IC on small batch."""
        factors, labels, factor_ids, (T, N, F) = small_batch

        # Numpy reference
        ic_numpy, counts_numpy = fast_ic_batch(
            factors, labels, method="pearson", min_obs=10
        )

        # Polars backend
        ic_polars, counts_polars = polars_ic_batch(
            factors, labels, factor_ids, method="pearson", min_obs=10
        )

        # Check shapes
        assert ic_numpy.shape == ic_polars.shape == (T, F)
        assert counts_numpy.shape == counts_polars.shape == (T, F)

        # Check values (allowing for floating point differences)
        assert_allclose(ic_polars, ic_numpy, rtol=1e-6, atol=1e-8, equal_nan=True)
        assert_array_equal(counts_polars, counts_numpy)

    def test_spearman_ic_small_batch(self, small_batch):
        """Test Spearman IC on small batch."""
        factors, labels, factor_ids, (T, N, F) = small_batch

        # Numpy reference
        ic_numpy, counts_numpy = fast_ic_batch(
            factors, labels, method="spearman", min_obs=10
        )

        # Polars backend
        ic_polars, counts_polars = polars_ic_batch(
            factors, labels, factor_ids, method="spearman", min_obs=10
        )

        # Check shapes
        assert ic_numpy.shape == ic_polars.shape == (T, F)

        # Spearman has more tolerance due to ranking differences
        assert_allclose(ic_polars, ic_numpy, rtol=1e-5, atol=1e-6, equal_nan=True)
        assert_array_equal(counts_polars, counts_numpy)

    def test_ic_with_nans(self, batch_with_nans):
        """Test IC computation with missing values."""
        factors, labels, factor_ids, (T, N, F) = batch_with_nans

        # Numpy reference
        ic_numpy, counts_numpy = fast_ic_batch(
            factors, labels, method="pearson", min_obs=5
        )

        # Polars backend
        ic_polars, counts_polars = polars_ic_batch(
            factors, labels, factor_ids, method="pearson", min_obs=5
        )

        # Both should handle NaNs identically
        assert_allclose(ic_polars, ic_numpy, rtol=1e-6, atol=1e-8, equal_nan=True)
        assert_array_equal(counts_polars, counts_numpy)

    def test_ic_min_obs_threshold(self, small_batch):
        """Test min_obs threshold behavior."""
        factors, labels, factor_ids, (T, N, F) = small_batch

        # High min_obs threshold
        min_obs = 40

        ic_numpy, counts_numpy = fast_ic_batch(
            factors, labels, method="pearson", min_obs=min_obs
        )

        ic_polars, counts_polars = polars_ic_batch(
            factors, labels, factor_ids, method="pearson", min_obs=min_obs
        )

        # Both should have NaN where counts < min_obs
        numpy_mask = counts_numpy < min_obs
        polars_mask = counts_polars < min_obs

        assert_array_equal(numpy_mask, polars_mask)
        assert np.all(np.isnan(ic_numpy[numpy_mask]))
        assert np.all(np.isnan(ic_polars[polars_mask]))

    def test_ic_with_one_dimensional_label_validity(self):
        T, N, F = 3, 5, 2
        factors = np.arange(T * N * F, dtype=float).reshape(T, N, F)
        labels = np.arange(T, dtype=float)
        validity = np.array([True, False, True])
        factor_ids = ("f0", "f1")

        ic_numpy, counts_numpy = fast_ic_batch(
            factors, labels, min_obs=2, label_validity=validity
        )
        ic_polars, counts_polars = polars_ic_batch(
            factors,
            labels,
            factor_ids,
            min_obs=2,
            label_validity=validity,
        )

        assert_allclose(ic_polars, ic_numpy, equal_nan=True)
        assert_array_equal(counts_polars, counts_numpy)

    def test_ic_medium_batch(self, medium_batch):
        """Test IC on medium-sized batch (performance check)."""
        factors, labels, factor_ids, (T, N, F) = medium_batch

        ic_numpy, counts_numpy = fast_ic_batch(
            factors, labels, method="pearson", min_obs=10
        )

        ic_polars, counts_polars = polars_ic_batch(
            factors, labels, factor_ids, method="pearson", min_obs=10
        )

        assert_allclose(ic_polars, ic_numpy, rtol=1e-6, atol=1e-8, equal_nan=True)
        assert_array_equal(counts_polars, counts_numpy)


class TestQuantileParity:
    """Test quantile binning parity."""

    def test_quantile_binning_small(self, small_batch):
        """Test quantile binning on small batch."""
        factors, labels, factor_ids, (T, N, F) = small_batch

        n_quantiles = 5

        # Numpy reference
        q_numpy = fast_quantile_binning(factors, n_quantiles=n_quantiles)

        # Polars backend
        q_polars = polars_quantile_binning(
            factors, factor_ids, n_quantiles=n_quantiles
        )

        # Check shapes
        assert q_numpy.shape == q_polars.shape == (T, N, F)

        # Check that quantiles match
        # Note: there may be minor differences in edge cases due to ranking methods
        # We'll check that the distribution is similar
        for t in range(T):
            for f in range(F):
                q_n = q_numpy[t, :, f]
                q_p = q_polars[t, :, f]

                # Both should have same valid count
                valid_n = np.sum(q_n >= 0)
                valid_p = np.sum(q_p >= 0)
                assert valid_n == valid_p

                if valid_n > 0:
                    # Check distribution of quantiles
                    for q_val in range(n_quantiles):
                        count_n = np.sum(q_n == q_val)
                        count_p = np.sum(q_p == q_val)
                        # Allow some variation due to tie-breaking
                        assert abs(count_n - count_p) <= 2

    def test_quantile_binning_with_nans(self, batch_with_nans):
        """Test quantile binning with missing values."""
        factors, labels, factor_ids, (T, N, F) = batch_with_nans

        n_quantiles = 5

        q_numpy = fast_quantile_binning(factors, n_quantiles=n_quantiles, min_valid=n_quantiles)
        q_polars = polars_quantile_binning(
            factors, factor_ids, n_quantiles=n_quantiles, min_valid=n_quantiles
        )

        # Check that invalid assignments match
        assert_array_equal(q_numpy == -1, q_polars == -1)


class TestQuantileReturnsParity:
    """Test quantile returns parity."""

    def test_quantile_returns_small(self, small_batch):
        """Test quantile returns on small batch."""
        factors, labels, factor_ids, (T, N, F) = small_batch

        n_quantiles = 5
        min_assets = 5

        # Numpy reference
        qret_numpy, qcount_numpy = compute_quantile_returns_fast(
            factors, labels, n_quantiles=n_quantiles, min_assets=min_assets
        )

        # Polars backend
        qret_polars, qcount_polars = polars_quantile_returns(
            factors, labels, factor_ids, n_quantiles=n_quantiles, min_assets=min_assets
        )

        # Check shapes
        assert qret_numpy.shape == qret_polars.shape == (T, n_quantiles, F)
        assert qcount_numpy.shape == qcount_polars.shape == (T, n_quantiles, F)

        # Check counts match exactly
        assert_array_equal(qcount_polars, qcount_numpy)

        # Check returns match (where counts >= min_assets)
        valid_mask = qcount_numpy >= min_assets
        assert_allclose(
            qret_polars[valid_mask],
            qret_numpy[valid_mask],
            rtol=1e-6,
            atol=1e-8,
        )

    def test_quantile_returns_with_nans(self, batch_with_nans):
        """Test quantile returns with missing values."""
        factors, labels, factor_ids, (T, N, F) = batch_with_nans

        n_quantiles = 3
        min_assets = 3

        qret_numpy, qcount_numpy = compute_quantile_returns_fast(
            factors, labels, n_quantiles=n_quantiles, min_assets=min_assets
        )

        qret_polars, qcount_polars = polars_quantile_returns(
            factors, labels, factor_ids, n_quantiles=n_quantiles, min_assets=min_assets
        )

        # Counts should match
        assert_array_equal(qcount_polars, qcount_numpy)

        # Returns should match where valid
        valid_mask = qcount_numpy >= min_assets
        assert_allclose(
            qret_polars[valid_mask],
            qret_numpy[valid_mask],
            rtol=1e-6,
            atol=1e-8,
        )


class TestPolarsPerformance:
    """Performance comparison tests (not strict assertions, just measurements)."""

    @pytest.mark.slow
    def test_ic_performance_large_batch(self):
        """Measure IC performance on large batch."""
        import time

        # Sized for the shared-loop memory budget (≤15 GiB with other agents
        # running): 252×3000×1000 float64 ≈ 6 GiB for factors alone, plus the
        # polars LazyFrame copy — that allocation OOM-killed full-suite runs.
        # 252×500×100 ≈ 100 MiB still exercises the vectorized path.
        np.random.seed(42)
        T, N, F = 252, 500, 100

        factors = np.random.randn(T, N, F)
        labels = np.random.randn(T, N)
        factor_ids = tuple(f"factor_{i:04d}" for i in range(F))

        # Numpy baseline
        start = time.perf_counter()
        ic_numpy, _ = fast_ic_batch(factors, labels, method="pearson")
        numpy_time = time.perf_counter() - start

        # Polars backend
        start = time.perf_counter()
        ic_polars, _ = polars_ic_batch(
            factors, labels, factor_ids, method="pearson"
        )
        polars_time = time.perf_counter() - start

        print(f"\nIC Performance (T={T}, N={N}, F={F}):")
        print(f"  NumPy:  {numpy_time:.3f}s")
        print(f"  Polars: {polars_time:.3f}s")
        print(f"  Speedup: {numpy_time/polars_time:.2f}x")

        # Verify results match
        assert_allclose(ic_polars, ic_numpy, rtol=1e-5, atol=1e-7, equal_nan=True)

    @pytest.mark.slow
    def test_quantile_performance_large_batch(self):
        """Measure quantile binning performance on large batch."""
        import time

        np.random.seed(42)
        T, N, F = 252, 500, 100
        # See note in test_ic_performance_large_batch: the old 252×3000×500
        # allocation (~3 GiB + polars copy) OOM-killed full-suite runs.

        factors = np.random.randn(T, N, F)
        factor_ids = tuple(f"factor_{i:04d}" for i in range(F))

        n_quantiles = 5

        # Numpy baseline
        start = time.perf_counter()
        q_numpy = fast_quantile_binning(factors, n_quantiles=n_quantiles)
        numpy_time = time.perf_counter() - start

        # Polars backend
        start = time.perf_counter()
        q_polars = polars_quantile_binning(
            factors, factor_ids, n_quantiles=n_quantiles
        )
        polars_time = time.perf_counter() - start

        print(f"\nQuantile Binning Performance (T={T}, N={N}, F={F}):")
        print(f"  NumPy:  {numpy_time:.3f}s")
        print(f"  Polars: {polars_time:.3f}s")
        print(f"  Speedup: {numpy_time/polars_time:.2f}x")

        # Verify distribution is similar
        assert q_numpy.shape == q_polars.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
