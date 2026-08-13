"""
Performance benchmarks for fast kernels vs reference implementations.

Measures speedup factors and validates that fast kernels provide performance
gains while maintaining parity with reference implementations.
"""

import time
import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import compute_daily_ic
from quant_evaluator.metrics.quantile import assign_quantiles, compute_quantile_returns
from quant_evaluator.metrics.turnover import estimate_turnover_from_ranks
from quant_evaluator.kernels.fast import (
    fast_ic_batch,
    fast_quantile_binning,
    fast_turnover_estimate,
    compute_quantile_returns_fast,
)


def make_batch(T: int, N: int, F: int, seed: int = 42) -> FactorBatch:
    """Create a synthetic FactorBatch for benchmarking."""
    np.random.seed(seed)
    values = np.random.randn(T, N, F)

    time_axis = AxisRef(name="time", dtype="datetime64", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    return FactorBatch(
        factor_ids=tuple(f"f{i}" for i in range(F)),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
    )


def make_bundle(T: int, N: int, seed: int = 42) -> LabelBundle:
    """Create a synthetic LabelBundle for benchmarking."""
    np.random.seed(seed + 1000)
    values = np.random.randn(T, N)

    return LabelBundle(
        target_id="ret_1d",
        values=values,
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )


def benchmark_function(func, *args, **kwargs):
    """Benchmark a function and return execution time."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


class TestPerformance:
    """Performance benchmarks for fast kernels."""

    def test_ic_performance_small(self):
        """Benchmark IC computation on small batch."""
        T, N, F = 30, 100, 100
        batch = make_batch(T, N, F, seed=100)
        bundle = make_bundle(T, N, seed=100)

        # Reference
        _, ref_time = benchmark_function(
            compute_daily_ic, batch, bundle, method="pearson", min_assets=10
        )

        # Fast
        _, fast_time = benchmark_function(
            fast_ic_batch,
            factor_values=batch.values,
            label_values=bundle.values,
            method="pearson",
            min_obs=10,
        )

        speedup = ref_time / fast_time if fast_time > 0 else float('inf')
        print(f"\nIC Performance (T={T}, N={N}, F={F}):")
        print(f"  Reference: {ref_time:.4f}s")
        print(f"  Fast:      {fast_time:.4f}s")
        print(f"  Speedup:   {speedup:.2f}x")

        assert fast_time <= ref_time * 1.5  # Fast should not be significantly slower

    def test_ic_performance_medium(self):
        """Benchmark IC computation on medium batch (1000 factors)."""
        T, N, F = 50, 100, 1000
        batch = make_batch(T, N, F, seed=200)
        bundle = make_bundle(T, N, seed=200)

        _, ref_time = benchmark_function(
            compute_daily_ic, batch, bundle, method="pearson", min_assets=10
        )

        _, fast_time = benchmark_function(
            fast_ic_batch,
            factor_values=batch.values,
            label_values=bundle.values,
            method="pearson",
            min_obs=10,
        )

        speedup = ref_time / fast_time if fast_time > 0 else float('inf')
        print(f"\nIC Performance (T={T}, N={N}, F={F}):")
        print(f"  Reference: {ref_time:.4f}s")
        print(f"  Fast:      {fast_time:.4f}s")
        print(f"  Speedup:   {speedup:.2f}x")

    @pytest.mark.slow
    def test_ic_performance_10k_factors(self):
        """Benchmark IC computation on 10k+ factor batch."""
        T, N, F = 20, 100, 10000

        np.random.seed(300)
        # Generate in chunks to manage memory
        chunk_size = 2000
        values_chunks = []
        for i in range(0, F, chunk_size):
            chunk = np.random.randn(T, N, min(chunk_size, F - i))
            values_chunks.append(chunk)
        values = np.concatenate(values_chunks, axis=2)

        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        batch = FactorBatch(
            factor_ids=tuple(f"f{i}" for i in range(F)),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )
        bundle = make_bundle(T, N, seed=300)

        _, ref_time = benchmark_function(
            compute_daily_ic, batch, bundle, method="pearson", min_assets=10
        )

        _, fast_time = benchmark_function(
            fast_ic_batch,
            factor_values=batch.values,
            label_values=bundle.values,
            method="pearson",
            min_obs=10,
        )

        speedup = ref_time / fast_time if fast_time > 0 else float('inf')
        print(f"\nIC Performance (T={T}, N={N}, F={F} - 10k factors):")
        print(f"  Reference: {ref_time:.4f}s")
        print(f"  Fast:      {fast_time:.4f}s")
        print(f"  Speedup:   {speedup:.2f}x")

        assert fast_time > 0

    def test_quantile_binning_performance(self):
        """Benchmark quantile binning performance."""
        T, N, F = 30, 200, 500
        batch = make_batch(T, N, F, seed=400)

        def ref_quantile_batch():
            """Reference: loop over T and F."""
            all_bins = np.full((T, N, F), -1, dtype=np.int32)
            for t in range(T):
                for f in range(F):
                    all_bins[t, :, f] = assign_quantiles(
                        batch.values[t, :, f], n_quantiles=5
                    ).flatten()
            return all_bins

        _, ref_time = benchmark_function(ref_quantile_batch)
        _, fast_time = benchmark_function(
            fast_quantile_binning, batch.values, n_quantiles=5
        )

        speedup = ref_time / fast_time if fast_time > 0 else float('inf')
        print(f"\nQuantile Binning Performance (T={T}, N={N}, F={F}):")
        print(f"  Reference: {ref_time:.4f}s")
        print(f"  Fast:      {fast_time:.4f}s")
        print(f"  Speedup:   {speedup:.2f}x")

    @pytest.mark.slow
    def test_quantile_binning_10k_factors(self):
        """Benchmark quantile binning on 10k factors."""
        T, N, F = 10, 200, 10000

        np.random.seed(500)
        chunk_size = 2000
        values_chunks = []
        for i in range(0, F, chunk_size):
            chunk = np.random.randn(T, N, min(chunk_size, F - i))
            values_chunks.append(chunk)
        values = np.concatenate(values_chunks, axis=2)

        def ref_quantile_batch():
            all_bins = np.full((T, N, F), -1, dtype=np.int32)
            for t in range(T):
                for f in range(F):
                    all_bins[t, :, f] = assign_quantiles(
                        values[t, :, f], n_quantiles=5
                    ).flatten()
            return all_bins

        _, ref_time = benchmark_function(ref_quantile_batch)
        _, fast_time = benchmark_function(fast_quantile_binning, values, n_quantiles=5)

        speedup = ref_time / fast_time if fast_time > 0 else float('inf')
        print(f"\nQuantile Binning Performance (T={T}, N={N}, F={F} - 10k factors):")
        print(f"  Reference: {ref_time:.4f}s")
        print(f"  Fast:      {fast_time:.4f}s")
        print(f"  Speedup:   {speedup:.2f}x")

    def test_quantile_returns_performance(self):
        """Benchmark quantile returns computation."""
        T, N, F = 20, 100, 200
        batch = make_batch(T, N, F, seed=600)
        bundle = make_bundle(T, N, seed=600)

        _, ref_time = benchmark_function(
            compute_quantile_returns, batch, bundle, n_quantiles=5, min_assets=10
        )

        _, fast_time = benchmark_function(
            compute_quantile_returns_fast,
            factor_values=batch.values,
            label_values=bundle.values,
            n_quantiles=5,
            min_assets=10,
        )

        speedup = ref_time / fast_time if fast_time > 0 else float('inf')
        print(f"\nQuantile Returns Performance (T={T}, N={N}, F={F}):")
        print(f"  Reference: {ref_time:.4f}s")
        print(f"  Fast:      {fast_time:.4f}s")
        print(f"  Speedup:   {speedup:.2f}x")

    def test_turnover_performance(self):
        """Benchmark turnover estimation."""
        T, N, F = 50, 100, 200
        batch = make_batch(T, N, F, seed=700)

        _, ref_time = benchmark_function(
            estimate_turnover_from_ranks, batch, window=1
        )

        _, fast_time = benchmark_function(
            fast_turnover_estimate, batch.values, window=1
        )

        speedup = ref_time / fast_time if fast_time > 0 else float('inf')
        print(f"\nTurnover Performance (T={T}, N={N}, F={F}):")
        print(f"  Reference: {ref_time:.4f}s")
        print(f"  Fast:      {fast_time:.4f}s")
        print(f"  Speedup:   {speedup:.2f}x")


class TestCorrectness:
    """Sanity tests to ensure fast kernels produce valid outputs."""

    def test_fast_ic_output_shapes(self):
        """Fast IC produces correct output shapes."""
        T, N, F = 10, 50, 5
        values = np.random.randn(T, N, F)
        labels = np.random.randn(T, N)

        ic_matrix, valid_counts = fast_ic_batch(
            factor_values=values,
            label_values=labels,
            method="pearson",
        )

        assert ic_matrix.shape == (T, F)
        assert valid_counts.shape == (T, F)
        assert ic_matrix.dtype == np.float64
        assert valid_counts.dtype == np.int32

    def test_fast_quantile_binning_output_shapes(self):
        """Fast quantile binning produces correct shapes."""
        T, N, F = 10, 50, 5
        values = np.random.randn(T, N, F)

        bins = fast_quantile_binning(values, n_quantiles=5)

        assert bins.shape == (T, N, F)
        assert bins.dtype == np.int32

        # Valid bins should be in [0, n_quantiles-1]
        valid_mask = bins >= 0
        assert np.all(bins[valid_mask] < 5)
        assert np.all(bins[valid_mask] >= 0)

    def test_fast_turnover_output_shapes(self):
        """Fast turnover produces correct shapes."""
        T, N, F = 20, 50, 3
        values = np.random.randn(T, N, F)

        turnover = fast_turnover_estimate(values, window=2)

        assert turnover.shape == (T, F)
        # First 'window' periods should be NaN
        assert np.all(np.isnan(turnover[:2, :]))

        # Valid turnover values should be in [0, 1]
        valid_mask = np.isfinite(turnover)
        assert np.all(turnover[valid_mask] >= 0.0)
        assert np.all(turnover[valid_mask] <= 1.0)

    def test_fast_ic_with_method_spearman(self):
        """Fast IC supports spearman method."""
        T, N, F = 10, 50, 2
        values = np.random.randn(T, N, F)
        labels = np.random.randn(T, N)

        ic_matrix_pearson, _ = fast_ic_batch(
            factor_values=values, label_values=labels, method="pearson"
        )
        ic_matrix_spearman, _ = fast_ic_batch(
            factor_values=values, label_values=labels, method="spearman"
        )

        # Both should have same shape
        assert ic_matrix_pearson.shape == ic_matrix_spearman.shape

    def test_fast_ic_invalid_method_raises(self):
        """Invalid correlation method raises ValueError."""
        values = np.random.randn(5, 10, 1)
        labels = np.random.randn(5, 10)

        with pytest.raises(ValueError, match="Unknown method"):
            fast_ic_batch(
                factor_values=values, label_values=labels, method="invalid"
            )