"""
Parity tests for Numba backend against reference implementations.

Validates that Numba-accelerated kernels produce mathematically identical
results to reference implementations across various scenarios.
"""

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.kernels.fast import (
    fast_ic_batch,
    fast_quantile_binning,
    compute_quantile_returns_fast,
)
from quant_evaluator.kernels.numba_backend import (
    numba_ic_batch,
    numba_quantile_binning,
    numba_quantile_returns,
    numba_rolling_mean,
    numba_rolling_std,
    numba_rolling_corr,
    numba_corrcoef_matrix,
)


def make_test_batch(T: int, N: int, F: int, seed: int = 42, sparsity: float = 0.0) -> np.ndarray:
    """Create test factor batch with optional sparsity."""
    np.random.seed(seed)
    values = np.random.randn(T, N, F)

    if sparsity > 0:
        mask = np.random.rand(T, N, F) < sparsity
        values[mask] = np.nan

    return values


def make_test_labels(T: int, N: int, seed: int = 42, sparsity: float = 0.0) -> np.ndarray:
    """Create test labels with optional sparsity."""
    np.random.seed(seed + 1000)
    values = np.random.randn(T, N)

    if sparsity > 0:
        mask = np.random.rand(T, N) < sparsity
        values[mask] = np.nan

    return values


class TestICParity:
    """Test IC computation parity."""

    def test_pearson_ic_basic(self):
        """Test basic Pearson IC parity."""
        T, N, F = 30, 100, 50
        factor_values = make_test_batch(T, N, F, seed=42)
        label_values = make_test_labels(T, N, seed=42)

        ic_fast, counts_fast = fast_ic_batch(
            factor_values, label_values, method="pearson", min_obs=10
        )
        ic_numba, counts_numba = numba_ic_batch(
            factor_values, label_values, method="pearson", min_obs=10
        )

        # Check counts match exactly
        assert np.array_equal(counts_fast, counts_numba), "Counts mismatch"

        # Check IC values match (NaN-aware)
        mask = np.isfinite(ic_fast) & np.isfinite(ic_numba)
        assert np.allclose(ic_fast[mask], ic_numba[mask], rtol=1e-9, atol=1e-12), \
            f"IC mismatch: max diff = {np.max(np.abs(ic_fast[mask] - ic_numba[mask]))}"

        # Check NaN positions match
        assert np.array_equal(np.isnan(ic_fast), np.isnan(ic_numba)), "NaN positions mismatch"

    def test_pearson_ic_sparse(self):
        """Test Pearson IC with sparse data."""
        T, N, F = 50, 100, 100
        factor_values = make_test_batch(T, N, F, seed=123, sparsity=0.3)
        label_values = make_test_labels(T, N, seed=123, sparsity=0.2)

        # Fast implementation doesn't need validity masks if NaNs are already in data
        ic_fast, counts_fast = fast_ic_batch(
            factor_values, label_values, method="pearson", min_obs=20,
            factor_validity=None, label_validity=None
        )
        ic_numba, counts_numba = numba_ic_batch(
            factor_values, label_values, method="pearson", min_obs=20
        )

        assert np.array_equal(counts_fast, counts_numba), "Counts mismatch with sparse data"

        mask = np.isfinite(ic_fast) & np.isfinite(ic_numba)
        assert np.allclose(ic_fast[mask], ic_numba[mask], rtol=1e-9, atol=1e-12), \
            "IC mismatch with sparse data"

    def test_spearman_ic_basic(self):
        """Test basic Spearman IC parity."""
        T, N, F = 30, 100, 50
        factor_values = make_test_batch(T, N, F, seed=42)
        label_values = make_test_labels(T, N, seed=42)

        ic_fast, counts_fast = fast_ic_batch(
            factor_values, label_values, method="spearman", min_obs=10
        )
        ic_numba, counts_numba = numba_ic_batch(
            factor_values, label_values, method="spearman", min_obs=10
        )

        # Check counts match exactly
        assert np.array_equal(counts_fast, counts_numba), "Counts mismatch"

        # Check IC values match (NaN-aware)
        mask = np.isfinite(ic_fast) & np.isfinite(ic_numba)
        assert np.allclose(ic_fast[mask], ic_numba[mask], rtol=1e-8, atol=1e-10), \
            f"Spearman IC mismatch: max diff = {np.max(np.abs(ic_fast[mask] - ic_numba[mask]))}"

    def test_spearman_ic_sparse(self):
        """Test Spearman IC with sparse data."""
        T, N, F = 40, 100, 80
        factor_values = make_test_batch(T, N, F, seed=456, sparsity=0.25)
        label_values = make_test_labels(T, N, seed=456, sparsity=0.15)

        ic_fast, counts_fast = fast_ic_batch(
            factor_values, label_values, method="spearman", min_obs=15,
            factor_validity=None, label_validity=None
        )
        ic_numba, counts_numba = numba_ic_batch(
            factor_values, label_values, method="spearman", min_obs=15
        )

        assert np.array_equal(counts_fast, counts_numba), "Counts mismatch with sparse data"

        mask = np.isfinite(ic_fast) & np.isfinite(ic_numba)
        if np.any(mask):
            assert np.allclose(ic_fast[mask], ic_numba[mask], rtol=1e-8, atol=1e-10), \
                "Spearman IC mismatch with sparse data"

    def test_ic_edge_cases(self):
        """Test edge cases: constant values, all NaN, etc."""
        T, N, F = 20, 50, 10

        # Case 1: Some constant factors
        factor_values = make_test_batch(T, N, F, seed=99)
        factor_values[:, :, 0] = 5.0  # Constant factor
        label_values = make_test_labels(T, N, seed=99)

        ic_fast, _ = fast_ic_batch(factor_values, label_values, method="pearson", min_obs=10)
        ic_numba, _ = numba_ic_batch(factor_values, label_values, method="pearson", min_obs=10)

        # Constant factor should have NaN IC
        assert np.all(np.isnan(ic_fast[:, 0])), "Fast: Constant factor should be NaN"
        assert np.all(np.isnan(ic_numba[:, 0])), "Numba: Constant factor should be NaN"

        # Other factors should match
        mask = np.isfinite(ic_fast[:, 1:]) & np.isfinite(ic_numba[:, 1:])
        assert np.allclose(ic_fast[:, 1:][mask], ic_numba[:, 1:][mask], rtol=1e-9, atol=1e-12)

        # Case 2: All NaN slice
        factor_values[:, :, 1] = np.nan
        ic_fast, _ = fast_ic_batch(factor_values, label_values, method="pearson", min_obs=10)
        ic_numba, _ = numba_ic_batch(factor_values, label_values, method="pearson", min_obs=10)

        assert np.all(np.isnan(ic_fast[:, 1])), "Fast: All-NaN should be NaN"
        assert np.all(np.isnan(ic_numba[:, 1])), "Numba: All-NaN should be NaN"


class TestQuantileParity:
    """Test quantile binning and returns parity."""

    def test_quantile_binning_basic(self):
        """Test basic quantile binning parity."""
        T, N, F = 30, 100, 50
        factor_values = make_test_batch(T, N, F, seed=42)

        bins_fast = fast_quantile_binning(factor_values, n_quantiles=5, min_valid=5)
        bins_numba = numba_quantile_binning(factor_values, n_quantiles=5, min_valid=5)

        assert np.array_equal(bins_fast, bins_numba), "Quantile binning mismatch"

    def test_quantile_binning_sparse(self):
        """Test quantile binning with sparse data."""
        T, N, F = 40, 100, 80
        factor_values = make_test_batch(T, N, F, seed=123, sparsity=0.3)

        bins_fast = fast_quantile_binning(factor_values, n_quantiles=10, min_valid=10)
        bins_numba = numba_quantile_binning(factor_values, n_quantiles=10, min_valid=10)

        assert np.array_equal(bins_fast, bins_numba), "Quantile binning mismatch with sparse data"

    def test_quantile_binning_edge_cases(self):
        """Test quantile binning edge cases."""
        T, N, F = 20, 50, 10
        factor_values = make_test_batch(T, N, F, seed=99)

        # Not enough valid observations for some slices
        factor_values[0, :, 0] = np.nan  # All NaN for one slice

        bins_fast = fast_quantile_binning(factor_values, n_quantiles=5, min_valid=30)
        bins_numba = numba_quantile_binning(factor_values, n_quantiles=5, min_valid=30)

        assert np.array_equal(bins_fast, bins_numba), "Quantile binning edge case mismatch"

        # Check that all-NaN slice has all -1 bins
        assert np.all(bins_fast[0, :, 0] == -1), "All-NaN slice should have -1 bins"
        assert np.all(bins_numba[0, :, 0] == -1), "All-NaN slice should have -1 bins"

    def test_quantile_returns_basic(self):
        """Test quantile returns computation parity."""
        T, N, F = 30, 100, 50
        factor_values = make_test_batch(T, N, F, seed=42)
        label_values = make_test_labels(T, N, seed=42)

        qret_fast, qcounts_fast = compute_quantile_returns_fast(
            factor_values, label_values, n_quantiles=5, min_assets=10
        )
        qret_numba, qcounts_numba = numba_quantile_returns(
            factor_values, label_values, n_quantiles=5, min_assets=10
        )

        # Counts should match exactly
        assert np.array_equal(qcounts_fast, qcounts_numba), "Quantile counts mismatch"

        # Returns should match (NaN-aware)
        mask = np.isfinite(qret_fast) & np.isfinite(qret_numba)
        assert np.allclose(qret_fast[mask], qret_numba[mask], rtol=1e-9, atol=1e-12), \
            f"Quantile returns mismatch: max diff = {np.max(np.abs(qret_fast[mask] - qret_numba[mask]))}"

    def test_quantile_returns_sparse(self):
        """Test quantile returns with sparse data."""
        T, N, F = 40, 100, 80
        factor_values = make_test_batch(T, N, F, seed=456, sparsity=0.25)
        label_values = make_test_labels(T, N, seed=456, sparsity=0.15)

        qret_fast, qcounts_fast = compute_quantile_returns_fast(
            factor_values, label_values, n_quantiles=10, min_assets=5
        )
        qret_numba, qcounts_numba = numba_quantile_returns(
            factor_values, label_values, n_quantiles=10, min_assets=5
        )

        assert np.array_equal(qcounts_fast, qcounts_numba), "Quantile counts mismatch with sparse data"

        mask = np.isfinite(qret_fast) & np.isfinite(qret_numba)
        if np.any(mask):
            assert np.allclose(qret_fast[mask], qret_numba[mask], rtol=1e-9, atol=1e-12), \
                "Quantile returns mismatch with sparse data"


class TestRollingStatsParity:
    """Test rolling statistics parity against NumPy/Pandas."""

    def test_rolling_mean_basic(self):
        """Test rolling mean parity."""
        T, F = 100, 50
        np.random.seed(42)
        values = np.random.randn(T, F)

        result_numba = numba_rolling_mean(values, window=20, min_periods=10)

        # Reference: compute manually
        result_ref = np.full((T, F), np.nan)
        for t in range(T):
            start = max(0, t - 20 + 1)
            for f in range(F):
                window_vals = values[start:t+1, f]
                finite_vals = window_vals[np.isfinite(window_vals)]
                if len(finite_vals) >= 10:
                    result_ref[t, f] = np.mean(finite_vals)

        mask = np.isfinite(result_ref) & np.isfinite(result_numba)
        assert np.allclose(result_ref[mask], result_numba[mask], rtol=1e-9, atol=1e-12), \
            "Rolling mean mismatch"

    def test_rolling_std_basic(self):
        """Test rolling std parity."""
        T, F = 100, 50
        np.random.seed(42)
        values = np.random.randn(T, F)

        result_numba = numba_rolling_std(values, window=20, min_periods=10)

        # Reference: compute manually
        result_ref = np.full((T, F), np.nan)
        for t in range(T):
            start = max(0, t - 20 + 1)
            for f in range(F):
                window_vals = values[start:t+1, f]
                finite_vals = window_vals[np.isfinite(window_vals)]
                if len(finite_vals) >= 10:
                    result_ref[t, f] = np.std(finite_vals, ddof=0)

        mask = np.isfinite(result_ref) & np.isfinite(result_numba)
        assert np.allclose(result_ref[mask], result_numba[mask], rtol=1e-9, atol=1e-12), \
            "Rolling std mismatch"

    def test_rolling_corr_basic(self):
        """Test rolling correlation parity."""
        T, F = 100, 30
        np.random.seed(42)
        x_values = np.random.randn(T, F)
        y_values = np.random.randn(T, F) + 0.5 * x_values  # Add correlation

        result_numba = numba_rolling_corr(x_values, y_values, window=20, min_periods=10)

        # Reference: compute manually
        result_ref = np.full((T, F), np.nan)
        for t in range(T):
            start = max(0, t - 20 + 1)
            for f in range(F):
                x_win = x_values[start:t+1, f]
                y_win = y_values[start:t+1, f]

                mask = np.isfinite(x_win) & np.isfinite(y_win)
                if np.sum(mask) >= 10:
                    x_finite = x_win[mask]
                    y_finite = y_win[mask]

                    # Pearson correlation
                    n = len(x_finite)
                    sum_x = np.sum(x_finite)
                    sum_y = np.sum(y_finite)
                    sum_xx = np.sum(x_finite ** 2)
                    sum_yy = np.sum(y_finite ** 2)
                    sum_xy = np.sum(x_finite * y_finite)

                    num = n * sum_xy - sum_x * sum_y
                    den_x = n * sum_xx - sum_x ** 2
                    den_y = n * sum_yy - sum_y ** 2

                    if den_x > 0 and den_y > 0:
                        result_ref[t, f] = num / np.sqrt(den_x * den_y)

        mask = np.isfinite(result_ref) & np.isfinite(result_numba)
        assert np.allclose(result_ref[mask], result_numba[mask], rtol=1e-9, atol=1e-12), \
            "Rolling correlation mismatch"

    def test_rolling_with_nans(self):
        """Test rolling stats with NaN values."""
        T, F = 100, 20
        np.random.seed(123)
        values = np.random.randn(T, F)

        # Insert some NaNs
        mask = np.random.rand(T, F) < 0.2
        values[mask] = np.nan

        result_mean = numba_rolling_mean(values, window=30, min_periods=15)
        result_std = numba_rolling_std(values, window=30, min_periods=15)

        # Just check that we get reasonable results (not all NaN)
        assert np.any(np.isfinite(result_mean)), "Rolling mean should have some finite values"
        assert np.any(np.isfinite(result_std)), "Rolling std should have some finite values"

        # Check no negative std
        finite_std = result_std[np.isfinite(result_std)]
        assert np.all(finite_std >= 0), "Rolling std should be non-negative"


class TestCorrMatrixParity:
    """Test correlation matrix parity."""

    def test_corrcoef_basic(self):
        """Test correlation matrix parity against NumPy."""
        T, F = 100, 50
        np.random.seed(42)
        values = np.random.randn(T, F)

        corr_numba = numba_corrcoef_matrix(values, min_obs=10)

        # Reference: NumPy corrcoef
        corr_ref = np.corrcoef(values, rowvar=False)

        # Check finite values match
        mask = np.isfinite(corr_ref) & np.isfinite(corr_numba)
        assert np.allclose(corr_ref[mask], corr_numba[mask], rtol=1e-9, atol=1e-12), \
            f"Corrcoef mismatch: max diff = {np.max(np.abs(corr_ref[mask] - corr_numba[mask]))}"

        # Check diagonal is 1.0
        assert np.allclose(np.diag(corr_numba), 1.0, rtol=1e-12), "Diagonal should be 1.0"

        # Check symmetry
        assert np.allclose(corr_numba, corr_numba.T, rtol=1e-12), "Matrix should be symmetric"

    def test_corrcoef_with_nans(self):
        """Test correlation matrix with NaN values."""
        T, F = 100, 30
        np.random.seed(123)
        values = np.random.randn(T, F)

        # Insert some NaNs
        mask = np.random.rand(T, F) < 0.15
        values[mask] = np.nan

        corr_numba = numba_corrcoef_matrix(values, min_obs=50)

        # Check diagonal is 1.0
        diag = np.diag(corr_numba)
        assert np.allclose(diag[np.isfinite(diag)], 1.0, rtol=1e-12), "Diagonal should be 1.0"

        # Check symmetry
        finite_mask = np.isfinite(corr_numba)
        assert np.allclose(corr_numba[finite_mask], corr_numba.T[finite_mask], rtol=1e-12), \
            "Matrix should be symmetric"

        # Check values in valid range
        finite_vals = corr_numba[finite_mask]
        assert np.all(finite_vals >= -1.0) and np.all(finite_vals <= 1.0), \
            "Correlation values should be in [-1, 1]"


class TestNumericalStability:
    """Test numerical stability and accuracy."""

    def test_high_correlation(self):
        """Test with highly correlated data."""
        T, N, F = 50, 100, 20
        np.random.seed(42)

        # Create highly correlated factors and labels
        base = np.random.randn(T, N)
        factor_values = np.repeat(base[:, :, np.newaxis], F, axis=2)
        factor_values += 0.01 * np.random.randn(T, N, F)  # Small noise
        label_values = base + 0.01 * np.random.randn(T, N)

        ic_fast, _ = fast_ic_batch(factor_values, label_values, method="pearson", min_obs=10)
        ic_numba, _ = numba_ic_batch(factor_values, label_values, method="pearson", min_obs=10)

        # Should have very high IC
        assert np.nanmean(ic_fast) > 0.95, "Should have high IC"

        # Should match closely
        mask = np.isfinite(ic_fast) & np.isfinite(ic_numba)
        assert np.allclose(ic_fast[mask], ic_numba[mask], rtol=1e-9, atol=1e-12)

    def test_near_zero_variance(self):
        """Test with near-zero variance (but not exactly zero)."""
        T, N, F = 30, 100, 10
        np.random.seed(42)

        factor_values = np.ones((T, N, F))
        factor_values += 1e-10 * np.random.randn(T, N, F)  # Tiny noise
        label_values = make_test_labels(T, N, seed=42)

        ic_fast, _ = fast_ic_batch(factor_values, label_values, method="pearson", min_obs=10)
        ic_numba, _ = numba_ic_batch(factor_values, label_values, method="pearson", min_obs=10)

        # Should produce NaN or near-zero IC (not crash)
        assert np.all(np.isnan(ic_fast) | (np.abs(ic_fast) < 0.1))
        assert np.all(np.isnan(ic_numba) | (np.abs(ic_numba) < 0.1))

    def test_large_values(self):
        """Test with large magnitude values."""
        T, N, F = 30, 100, 20
        np.random.seed(42)

        factor_values = 1e6 * np.random.randn(T, N, F)
        label_values = 1e6 * make_test_labels(T, N, seed=42)

        ic_fast, _ = fast_ic_batch(factor_values, label_values, method="pearson", min_obs=10)
        ic_numba, _ = numba_ic_batch(factor_values, label_values, method="pearson", min_obs=10)

        # Should match
        mask = np.isfinite(ic_fast) & np.isfinite(ic_numba)
        assert np.allclose(ic_fast[mask], ic_numba[mask], rtol=1e-8, atol=1e-10)

        # IC should be in valid range
        finite_ic = ic_numba[np.isfinite(ic_numba)]
        assert np.all(finite_ic >= -1.0) and np.all(finite_ic <= 1.0), \
            "IC should be in [-1, 1]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
