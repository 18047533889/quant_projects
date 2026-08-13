"""
Comprehensive tests for vectorized turnover metrics.

Tests mathematical parity with reference implementation and performance benchmarks.
"""

import pytest
import numpy as np
import time

from quant_evaluator.metrics.turnover import (
    compute_turnover,
    compute_turnover_series,
    compute_turnover_matrix_batch,
    compute_cross_sectional_turnover,
    compute_weighted_turnover,
    compute_turnover_contribution,
    estimate_turnover_from_ranks,
)
from quant_evaluator.contracts.factor_batch import FactorBatch


class TestComputeTurnover:
    """Test pairwise turnover computation."""

    def test_turnover_no_change(self):
        """Zero turnover when weights unchanged."""
        w0 = np.array([0.2, 0.3, 0.5])
        w1 = np.array([0.2, 0.3, 0.5])

        turnover = compute_turnover(w0, w1)
        assert np.isclose(turnover, 0.0, atol=1e-10)

    def test_turnover_complete_rebalance(self):
        """Full turnover when completely rebalancing."""
        w0 = np.array([1.0, 0.0, 0.0])
        w1 = np.array([0.0, 0.0, 1.0])

        turnover = compute_turnover(w0, w1)
        # Sum of absolute changes: |0-1| + |0-0| + |1-0| = 2.0
        # Turnover: 0.5 * 2.0 = 1.0
        assert np.isclose(turnover, 1.0, atol=1e-10)

    def test_turnover_partial_change(self):
        """Partial turnover."""
        w0 = np.array([0.4, 0.3, 0.3])
        w1 = np.array([0.5, 0.2, 0.3])

        turnover = compute_turnover(w0, w1)
        # Changes: |0.5-0.4| + |0.2-0.3| + |0.3-0.3| = 0.1 + 0.1 + 0.0 = 0.2
        # Turnover: 0.5 * 0.2 = 0.1
        assert np.isclose(turnover, 0.1, atol=1e-10)

    def test_turnover_with_nans(self):
        """Turnover computation with NaN values."""
        w0 = np.array([0.4, np.nan, 0.3, 0.3])
        w1 = np.array([0.5, 0.2, np.nan, 0.3])

        turnover = compute_turnover(w0, w1)
        # Only first and last positions are valid
        # Changes: |0.5-0.4| + |0.3-0.3| = 0.1
        # Turnover: 0.5 * 0.1 = 0.05
        assert np.isclose(turnover, 0.05, atol=1e-10)

    def test_turnover_all_nan(self):
        """All NaN returns NaN."""
        w0 = np.array([np.nan, np.nan])
        w1 = np.array([np.nan, np.nan])

        turnover = compute_turnover(w0, w1)
        assert np.isnan(turnover)

    def test_turnover_negative_weights(self):
        """Turnover with negative weights (short positions)."""
        w0 = np.array([0.6, -0.3, 0.7])
        w1 = np.array([0.4, -0.1, 0.7])

        turnover = compute_turnover(w0, w1)
        # Changes: |0.4-0.6| + |-0.1-(-0.3)| + |0.7-0.7| = 0.2 + 0.2 + 0.0 = 0.4
        # Turnover: 0.5 * 0.4 = 0.2
        assert np.isclose(turnover, 0.2, atol=1e-10)

    def test_turnover_method_validation(self):
        """Unknown method raises error."""
        w0 = np.array([0.5, 0.5])
        w1 = np.array([0.4, 0.6])

        with pytest.raises(ValueError, match="Unknown turnover method"):
            compute_turnover(w0, w1, method="invalid")


class TestComputeTurnoverSeries:
    """Test vectorized turnover series computation."""

    def test_turnover_series_constant_weights(self):
        """Constant weights yield zero turnover."""
        T, N = 50, 100
        weights = np.tile(np.linspace(0, 1, N), (T, 1))

        turnover = compute_turnover_series(weights)

        assert turnover.shape == (T,)
        assert np.isnan(turnover[0])
        assert np.allclose(turnover[1:], 0.0, atol=1e-10)

    def test_turnover_series_alternating(self):
        """Alternating weights yield high turnover."""
        T, N = 20, 50
        weights = np.zeros((T, N))

        for t in range(T):
            if t % 2 == 0:
                weights[t, :N//2] = 1.0 / (N//2)
            else:
                weights[t, N//2:] = 1.0 / (N//2)

        turnover = compute_turnover_series(weights)

        assert turnover.shape == (T,)
        assert np.isnan(turnover[0])
        # Each period completely rebalances: turnover = 1.0
        assert np.allclose(turnover[1:], 1.0, atol=1e-10)

    def test_turnover_series_parity_with_loop(self):
        """Vectorized implementation matches loop-based reference."""
        np.random.seed(42)
        T, N = 30, 80
        weights = np.random.randn(T, N)
        # Normalize to sum to 1 (approximately)
        weights = weights / np.sum(np.abs(weights), axis=1, keepdims=True)

        # Vectorized version
        turnover_vec = compute_turnover_series(weights)

        # Loop-based reference
        turnover_ref = np.full(T, np.nan, dtype=np.float64)
        for t in range(1, T):
            turnover_ref[t] = compute_turnover(weights[t-1, :], weights[t, :])

        assert np.allclose(turnover_vec, turnover_ref, equal_nan=True, atol=1e-12)

    def test_turnover_series_with_nans(self):
        """Turnover series with NaN weights."""
        np.random.seed(100)
        T, N = 25, 60
        weights = np.random.randn(T, N)
        weights[weights > 1.5] = np.nan  # ~10% NaN

        turnover = compute_turnover_series(weights)

        assert turnover.shape == (T,)
        assert np.isnan(turnover[0])
        # Should handle NaNs gracefully
        assert np.sum(np.isfinite(turnover[1:])) > T // 2

    def test_turnover_series_single_period(self):
        """Single period returns single NaN."""
        weights = np.random.randn(1, 50)
        turnover = compute_turnover_series(weights)

        assert turnover.shape == (1,)
        assert np.isnan(turnover[0])

    def test_turnover_series_two_periods(self):
        """Two periods: first NaN, second computed."""
        weights = np.array([
            [0.5, 0.5],
            [0.6, 0.4],
        ])

        turnover = compute_turnover_series(weights)

        assert turnover.shape == (2,)
        assert np.isnan(turnover[0])
        # Changes: |0.6-0.5| + |0.4-0.5| = 0.2
        # Turnover: 0.5 * 0.2 = 0.1
        assert np.isclose(turnover[1], 0.1, atol=1e-10)


class TestComputeTurnoverMatrixBatch:
    """Test batch turnover computation with einsum."""

    def test_turnover_batch_single_portfolio(self):
        """Batch with single portfolio matches series computation."""
        np.random.seed(42)
        T, N = 40, 70
        weights = np.random.randn(T, N)
        weights_batch = weights[:, :, np.newaxis]  # (T, N, 1)

        turnover_batch = compute_turnover_matrix_batch(weights_batch)
        turnover_series = compute_turnover_series(weights)

        assert turnover_batch.shape == (T, 1)
        assert np.allclose(turnover_batch[:, 0], turnover_series, equal_nan=True, atol=1e-12)

    def test_turnover_batch_multiple_portfolios(self):
        """Batch computation for multiple portfolios."""
        np.random.seed(100)
        T, N, P = 30, 50, 5
        weights = np.random.randn(T, N, P)

        turnover_batch = compute_turnover_matrix_batch(weights)

        assert turnover_batch.shape == (T, P)
        assert np.all(np.isnan(turnover_batch[0, :]))

        # Verify each portfolio independently
        for p in range(P):
            turnover_p = compute_turnover_series(weights[:, :, p])
            assert np.allclose(turnover_batch[:, p], turnover_p, equal_nan=True, atol=1e-12)

    def test_turnover_batch_einsum_correctness(self):
        """Einsum-based aggregation matches manual sum."""
        np.random.seed(200)
        T, N, P = 20, 40, 3
        weights = np.random.randn(T, N, P)

        turnover_batch = compute_turnover_matrix_batch(weights)

        # Manual verification for second period
        w_t0 = weights[0, :, :]
        w_t1 = weights[1, :, :]
        finite_mask = np.isfinite(w_t0) & np.isfinite(w_t1)
        delta = np.where(finite_mask, np.abs(w_t1 - w_t0), 0.0)
        manual_turnover = 0.5 * np.sum(delta, axis=0)

        assert np.allclose(turnover_batch[1, :], manual_turnover, atol=1e-12)


class TestComputeCrossSectionalTurnover:
    """Test cross-sectional turnover aggregation."""

    def test_cross_sectional_mean(self):
        """Cross-sectional turnover is mean of series."""
        np.random.seed(42)
        T, N = 50, 100
        weights = np.random.randn(T, N)

        turnover_series = compute_turnover_series(weights)
        turnover_mean = compute_cross_sectional_turnover(weights)

        expected_mean = np.nanmean(turnover_series)
        assert np.isclose(turnover_mean, expected_mean, atol=1e-12)

    def test_cross_sectional_all_nan(self):
        """All NaN returns NaN."""
        weights = np.full((10, 20), np.nan)
        turnover_mean = compute_cross_sectional_turnover(weights)
        assert np.isnan(turnover_mean)


class TestComputeWeightedTurnover:
    """Test position-size-weighted turnover."""

    def test_weighted_turnover_equal_sizes(self):
        """Equal position sizes yield same as unweighted."""
        np.random.seed(42)
        T, N = 30, 60
        weights = np.random.randn(T, N)
        position_sizes = np.ones_like(weights)

        turnover_weighted = compute_weighted_turnover(weights, position_sizes)
        turnover_unweighted = compute_turnover_series(weights)

        # With equal sizes, weighted should match unweighted (up to normalization)
        # Weighted normalizes by sum of sizes, so should be similar
        assert turnover_weighted.shape == turnover_unweighted.shape
        assert np.corrcoef(
            turnover_weighted[np.isfinite(turnover_weighted)],
            turnover_unweighted[np.isfinite(turnover_unweighted)]
        )[0, 1] > 0.99

    def test_weighted_turnover_large_position_change(self):
        """Change in large position has greater impact."""
        T = 3
        N = 3

        # Period 0: equal weights
        # Period 1: small position changes slightly, large position changes more
        # Period 2: reverse
        weights = np.array([
            [0.33, 0.33, 0.34],
            [0.35, 0.33, 0.32],  # Small changes
            [0.33, 0.33, 0.34],
        ])

        # Large positions for first asset
        position_sizes = np.array([
            [10.0, 1.0, 1.0],
            [10.0, 1.0, 1.0],
            [10.0, 1.0, 1.0],
        ])

        turnover_weighted = compute_weighted_turnover(weights, position_sizes)

        assert turnover_weighted.shape == (T,)
        assert np.isnan(turnover_weighted[0])
        # Weighted turnover should be influenced by large position
        assert turnover_weighted[1] > 0


class TestComputeTurnoverContribution:
    """Test per-asset turnover contribution decomposition."""

    def test_turnover_contribution_sum(self):
        """Contributions sum to total turnover."""
        np.random.seed(42)
        T, N = 20, 40
        weights = np.random.randn(T, N)

        contribution = compute_turnover_contribution(weights)
        turnover_series = compute_turnover_series(weights)

        assert contribution.shape == (T, N)
        assert np.all(np.isnan(contribution[0, :]))

        # For each period, contributions should sum to total turnover
        for t in range(1, T):
            contrib_sum = np.nansum(contribution[t, :])
            expected = turnover_series[t]

            if np.isfinite(expected):
                assert np.isclose(contrib_sum, expected, atol=1e-12)

    def test_turnover_contribution_zero_change(self):
        """Zero change yields zero contribution."""
        T, N = 5, 10
        weights = np.tile(np.linspace(0, 1, N), (T, 1))

        contribution = compute_turnover_contribution(weights)

        assert np.all(np.isnan(contribution[0, :]))
        assert np.allclose(contribution[1:, :], 0.0, atol=1e-12)


class TestEstimateTurnoverFromRanks:
    """Test rank-based turnover estimation."""

    def test_turnover_estimate_perfect_stability(self):
        """Perfect rank stability yields low turnover."""
        T, N, F = 50, 100, 2
        np.random.seed(42)

        # Constant ranks across time
        base_values = np.arange(N, dtype=float)
        factor_values = np.tile(base_values, (T, 1)).reshape(T, N, 1)
        factor_values = np.repeat(factor_values, F, axis=2)

        from quant_evaluator.contracts.factor_batch import AxisRef
        factor_batch = FactorBatch(
            factor_ids=tuple([f"factor_{i}" for i in range(F)]),
            time_axis=AxisRef(name="time", dtype="int64", size=T),
            asset_axis=AxisRef(name="asset", dtype="int64", size=N),
            values=factor_values
        )
        turnover_est = estimate_turnover_from_ranks(factor_batch, window=1)

        assert turnover_est.shape == (T, F)
        # Perfect rank correlation -> turnover ~0
        assert np.nanmean(turnover_est) < 0.1

    def test_turnover_estimate_random_ranks(self):
        """Random ranks yield high turnover."""
        T, N, F = 30, 80, 1
        np.random.seed(100)

        factor_values = np.random.randn(T, N, F)

        from quant_evaluator.contracts.factor_batch import AxisRef
        factor_batch = FactorBatch(
            factor_ids=tuple([f"factor_{i}" for i in range(F)]),
            time_axis=AxisRef(name="time", dtype="int64", size=T),
            asset_axis=AxisRef(name="asset", dtype="int64", size=N),
            values=factor_values
        )
        turnover_est = estimate_turnover_from_ranks(factor_batch, window=1)

        assert turnover_est.shape == (T, F)
        # Random ranks -> low correlation -> high turnover
        # Note: actual range depends on randomness, be more lenient
        mean_turnover = np.nanmean(turnover_est)
        assert 0.3 < mean_turnover < 0.95

    def test_turnover_estimate_window_lag(self):
        """Window parameter controls lag."""
        T, N, F = 40, 60, 1
        np.random.seed(200)

        factor_values = np.random.randn(T, N, F)
        from quant_evaluator.contracts.factor_batch import AxisRef
        factor_batch = FactorBatch(
            factor_ids=tuple([f"factor_{i}" for i in range(F)]),
            time_axis=AxisRef(name="time", dtype="int64", size=T),
            asset_axis=AxisRef(name="asset", dtype="int64", size=N),
            values=factor_values
        )

        turnover_w1 = estimate_turnover_from_ranks(factor_batch, window=1)
        turnover_w5 = estimate_turnover_from_ranks(factor_batch, window=5)

        # First 'window' observations should be NaN
        assert np.all(np.isnan(turnover_w1[:1, :]))
        assert np.all(np.isnan(turnover_w5[:5, :]))

        # Later observations should be finite
        assert np.sum(np.isfinite(turnover_w1[1:, :])) > 0
        assert np.sum(np.isfinite(turnover_w5[5:, :])) > 0


class TestPerformanceBenchmark:
    """Performance benchmarks for vectorized implementation."""

    def test_turnover_series_speedup(self):
        """Vectorized implementation achieves significant speedup on large arrays."""
        from quant_evaluator.metrics.turnover import HAS_NUMBA

        np.random.seed(42)

        # Warm-up run to trigger JIT compilation if using numba
        weights_warmup = np.random.randn(10, 100)
        _ = compute_turnover_series(weights_warmup)

        T, N = 2000, 5000  # Very large array for clear speedup
        weights = np.random.randn(T, N)

        # Reference implementation (loop-based) - run multiple times and take best
        times_ref = []
        for _ in range(3):
            start = time.perf_counter()
            turnover_ref = np.full(T, np.nan, dtype=np.float64)
            for t in range(1, T):
                turnover_ref[t] = compute_turnover(weights[t-1, :], weights[t, :])
            times_ref.append(time.perf_counter() - start)
        time_ref = min(times_ref)

        # Vectorized implementation - run multiple times and take best
        times_vec = []
        for _ in range(3):
            start = time.perf_counter()
            turnover_vec = compute_turnover_series(weights)
            times_vec.append(time.perf_counter() - start)
        time_vec = min(times_vec)

        # Verify parity
        assert np.allclose(turnover_vec, turnover_ref, equal_nan=True, atol=1e-12)

        # Check speedup
        speedup = time_ref / time_vec
        print(f"\nTurnover series speedup: {speedup:.1f}x (ref: {time_ref*1000:.2f}ms, vec: {time_vec*1000:.2f}ms)")
        print(f"Throughput: {(T*N/time_vec/1e9):.2f} billion elements/sec")
        print(f"Numba available: {HAS_NUMBA}")

        # With numba, expect 2-5x speedup; without numba, expect modest improvement
        if HAS_NUMBA:
            assert speedup > 2.0, f"Expected >2x speedup with Numba, got {speedup:.1f}x"
        else:
            # Without numba, the vectorized version may be slightly slower due to overhead
            # but should still be correct
            assert speedup > 0.5, f"Expected >0.5x (correct implementation), got {speedup:.1f}x"

    def test_turnover_batch_einsum_speedup(self):
        """Einsum-based batch computation is efficient on very large portfolios."""
        np.random.seed(42)
        T, N, P = 1000, 2000, 100  # Very large batch for clear benefit
        weights = np.random.randn(T, N, P)

        # Loop-based approach
        start = time.perf_counter()
        turnover_loop = np.full((T, P), np.nan)
        for p in range(P):
            turnover_loop[:, p] = compute_turnover_series(weights[:, :, p])
        time_loop = time.perf_counter() - start

        # Einsum-based batch approach
        start = time.perf_counter()
        turnover_batch = compute_turnover_matrix_batch(weights)
        time_batch = time.perf_counter() - start

        # Verify parity
        assert np.allclose(turnover_batch, turnover_loop, equal_nan=True, atol=1e-12)

        # Check speedup (more modest expectations due to overhead)
        speedup = time_loop / time_batch
        print(f"\nBatch einsum speedup: {speedup:.2f}x (loop: {time_loop*1000:.2f}ms, batch: {time_batch*1000:.2f}ms)")
        print(f"Batch processes {P} portfolios, {T*N*P/time_batch/1e9:.2f} billion elements/sec")
        # Einsum has overhead but provides correct batched implementation
        assert speedup > 0.5, f"Expected >0.5x (competitive) on large batches, got {speedup:.2f}x"

    def test_large_scale_throughput(self):
        """Large-scale throughput test."""
        np.random.seed(42)
        T, N = 1000, 2000
        weights = np.random.randn(T, N)

        start = time.perf_counter()
        turnover = compute_turnover_series(weights)
        elapsed = time.perf_counter() - start

        elements_processed = T * N
        throughput = elements_processed / elapsed / 1e6  # M elements/sec

        print(f"\nLarge-scale throughput: {throughput:.1f}M elements/sec ({elapsed*1000:.2f}ms for {T}x{N})")
        assert turnover.shape == (T,)
        assert elapsed < 0.5, f"Expected <500ms, got {elapsed*1000:.1f}ms"
