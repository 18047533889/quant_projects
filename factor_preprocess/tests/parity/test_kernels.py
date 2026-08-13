"""
Parity tests: verify fast kernels match reference implementations exactly.

These tests ensure mathematical correctness across all fast paths.
"""
import numpy as np
import pytest

from factor_preprocess.kernels.reference_bridge import (
    reference_cs_rank,
    reference_cs_zscore,
    reference_cs_demean,
    reference_rolling_mean,
    reference_rolling_std,
)

from factor_preprocess.kernels.fast import (
    fast_cs_rank,
    fast_cs_zscore,
    fast_cs_demean,
    fast_rolling_mean,
    fast_rolling_std,
    numba_rolling_mean,
    numba_rolling_std,
    get_capabilities,
    HAS_BOTTLENECK,
    HAS_NUMBA,
)


@pytest.fixture
def rng():
    """Fixed random state for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def sample_cs_data(rng):
    """Cross-sectional data: (100 dates, 50 assets)."""
    data = rng.randn(100, 50)
    # Inject some NaNs
    nan_mask = rng.rand(100, 50) < 0.05
    data[nan_mask] = np.nan
    return data


@pytest.fixture
def sample_ts_data(rng):
    """Time-series data: (500 dates, 30 assets)."""
    data = rng.randn(500, 30) * 10 + 100
    # Inject some NaNs
    nan_mask = rng.rand(500, 30) < 0.03
    data[nan_mask] = np.nan
    return data


# ============================================================================
# Cross-sectional parity tests
# ============================================================================

@pytest.mark.parity
class TestCrossSecionalParity:
    """Test fast vs reference for cross-sectional operations."""

    def test_rank_basic(self, sample_cs_data):
        """Test rank parity on basic case."""
        ref = reference_cs_rank(sample_cs_data, axis=-1)
        fast = fast_cs_rank(sample_cs_data, axis=-1)

        # Must match exactly (rank is integer-valued)
        np.testing.assert_array_equal(ref, fast)

    def test_rank_percentile(self, sample_cs_data):
        """Test rank parity with percentile mode."""
        ref = reference_cs_rank(sample_cs_data, axis=-1, pct=True)
        fast = fast_cs_rank(sample_cs_data, axis=-1, pct=True)

        np.testing.assert_allclose(ref, fast, rtol=1e-10, atol=1e-14)

    def test_rank_1d(self, rng):
        """Test rank on 1D array."""
        data = rng.randn(100)
        data[rng.rand(100) < 0.1] = np.nan

        ref = reference_cs_rank(data)
        fast = fast_cs_rank(data)

        np.testing.assert_array_equal(ref, fast)

    def test_rank_all_nan(self):
        """Test rank with all-NaN slice."""
        data = np.full((10, 20), np.nan)

        ref = reference_cs_rank(data)
        fast = fast_cs_rank(data)

        assert np.all(np.isnan(ref))
        assert np.all(np.isnan(fast))

    def test_rank_constants(self):
        """Test rank with constant slices."""
        data = np.ones((10, 20))
        data[0, :] = np.arange(20)  # First row varies

        ref = reference_cs_rank(data, axis=-1)
        fast = fast_cs_rank(data, axis=-1)

        np.testing.assert_array_equal(ref, fast)

    def test_zscore_basic(self, sample_cs_data):
        """Test zscore parity."""
        ref = reference_cs_zscore(sample_cs_data, axis=-1)
        fast = fast_cs_zscore(sample_cs_data, axis=-1)

        np.testing.assert_allclose(ref, fast, rtol=1e-10, atol=1e-14)

    def test_zscore_constant_slice(self):
        """Test zscore on constant slice."""
        data = np.ones((10, 20))
        data[0, :] = np.arange(20)

        ref = reference_cs_zscore(data, axis=-1, constant_value=0.0)
        fast = fast_cs_zscore(data, axis=-1, constant_value=0.0)

        np.testing.assert_allclose(ref, fast, rtol=1e-10, atol=1e-14)

    def test_zscore_with_nans(self, sample_cs_data):
        """Test zscore preserves NaN positions."""
        ref = reference_cs_zscore(sample_cs_data, axis=-1)
        fast = fast_cs_zscore(sample_cs_data, axis=-1)

        # NaN positions must match exactly
        np.testing.assert_array_equal(np.isnan(ref), np.isnan(fast))

        # Finite values must match
        finite_mask = np.isfinite(ref)
        np.testing.assert_allclose(
            ref[finite_mask],
            fast[finite_mask],
            rtol=1e-10,
            atol=1e-14
        )

    def test_demean_basic(self, sample_cs_data):
        """Test demean parity."""
        ref = reference_cs_demean(sample_cs_data, axis=-1)
        fast = fast_cs_demean(sample_cs_data, axis=-1)

        np.testing.assert_allclose(ref, fast, rtol=1e-10, atol=1e-14)

    def test_demean_mean_is_zero(self, sample_cs_data):
        """Test that demeaned data has zero mean."""
        result = fast_cs_demean(sample_cs_data, axis=-1)

        # Mean should be ~0 (within floating point error)
        mean = np.nanmean(result, axis=-1)
        np.testing.assert_allclose(mean, 0.0, atol=1e-12)


# ============================================================================
# Rolling window parity tests
# ============================================================================

@pytest.mark.parity
class TestRollingParity:
    """Test fast vs reference for rolling operations."""

    def test_rolling_mean_basic(self, sample_ts_data):
        """Test rolling mean parity."""
        window = 20

        ref = reference_rolling_mean(sample_ts_data, window, axis=0)
        fast = fast_rolling_mean(sample_ts_data, window, axis=0)

        np.testing.assert_allclose(ref, fast, rtol=1e-10, atol=1e-12)

    def test_rolling_mean_small_window(self, sample_ts_data):
        """Test rolling mean with small window."""
        window = 5

        ref = reference_rolling_mean(sample_ts_data, window, axis=0)
        fast = fast_rolling_mean(sample_ts_data, window, axis=0)

        np.testing.assert_allclose(ref, fast, rtol=1e-10, atol=1e-12)

    def test_rolling_mean_window_1(self, sample_ts_data):
        """Test rolling mean with window=1 (should equal input)."""
        window = 1

        ref = reference_rolling_mean(sample_ts_data, window, axis=0)
        fast = fast_rolling_mean(sample_ts_data, window, axis=0)

        np.testing.assert_allclose(ref, fast, rtol=1e-10, atol=1e-12)

    def test_rolling_mean_warmup_nans(self, sample_ts_data):
        """Test that first window-1 rows are NaN."""
        window = 20
        result = fast_rolling_mean(sample_ts_data, window, axis=0)

        # First 19 rows should be all NaN
        assert np.all(np.isnan(result[:window - 1]))

        # Row 19 onward should have some finite values
        assert np.any(np.isfinite(result[window - 1:]))

    def test_rolling_std_basic(self, sample_ts_data):
        """Test rolling std parity."""
        window = 20

        ref = reference_rolling_std(sample_ts_data, window, axis=0)
        fast = fast_rolling_std(sample_ts_data, window, axis=0)

        np.testing.assert_allclose(ref, fast, rtol=1e-10, atol=1e-12)

    def test_rolling_std_ddof(self, sample_ts_data):
        """Test rolling std with different ddof."""
        window = 20

        ref_ddof0 = reference_rolling_std(sample_ts_data, window, axis=0, ddof=0)
        fast_ddof0 = fast_rolling_std(sample_ts_data, window, axis=0, ddof=0)

        ref_ddof1 = reference_rolling_std(sample_ts_data, window, axis=0, ddof=1)
        fast_ddof1 = fast_rolling_std(sample_ts_data, window, axis=0, ddof=1)

        np.testing.assert_allclose(ref_ddof0, fast_ddof0, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(ref_ddof1, fast_ddof1, rtol=1e-10, atol=1e-12)

        # ddof=0 should give smaller or equal std than ddof=1 (divides by n vs n-1)
        # Actually ddof=1 gives LARGER std because denominator is smaller
        finite_mask = np.isfinite(fast_ddof0) & np.isfinite(fast_ddof1)
        assert np.all(fast_ddof1[finite_mask] >= fast_ddof0[finite_mask])

    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not available")
    def test_numba_rolling_mean_parity(self, sample_ts_data):
        """Test numba rolling mean matches reference."""
        window = 20

        ref = reference_rolling_mean(sample_ts_data, window, axis=0)
        numba_result = numba_rolling_mean(sample_ts_data, window, axis=0)

        np.testing.assert_allclose(ref, numba_result, rtol=1e-10, atol=1e-12)

    @pytest.mark.skipif(not HAS_NUMBA, reason="Numba not available")
    def test_numba_rolling_std_parity(self, sample_ts_data):
        """Test numba rolling std matches reference."""
        window = 20

        ref = reference_rolling_std(sample_ts_data, window, axis=0)
        numba_result = numba_rolling_std(sample_ts_data, window, axis=0)

        np.testing.assert_allclose(ref, numba_result, rtol=1e-10, atol=1e-12)

    def test_rolling_mean_1d(self, rng):
        """Test rolling mean on 1D array."""
        data = rng.randn(100)
        window = 10

        ref = reference_rolling_mean(data.reshape(-1, 1), window, axis=0).ravel()
        fast = fast_rolling_mean(data.reshape(-1, 1), window, axis=0).ravel()

        np.testing.assert_allclose(ref, fast, rtol=1e-10, atol=1e-12)


# ============================================================================
# Edge case tests
# ============================================================================

@pytest.mark.parity
class TestEdgeCases:
    """Test edge cases for parity."""

    def test_empty_array(self):
        """Test empty arrays."""
        data = np.array([]).reshape(0, 10)

        ref_rank = reference_cs_rank(data)
        fast_rank = fast_cs_rank(data)
        assert ref_rank.shape == fast_rank.shape == (0, 10)

        ref_zscore = reference_cs_zscore(data)
        fast_zscore = fast_cs_zscore(data)
        assert ref_zscore.shape == fast_zscore.shape == (0, 10)

    def test_single_value(self):
        """Test single-value arrays."""
        data = np.array([[5.0]])

        ref_rank = reference_cs_rank(data, pct=True)
        fast_rank = fast_cs_rank(data, pct=True)
        # Single value should get percentile 0.5
        assert ref_rank[0, 0] == fast_rank[0, 0] == 0.5

        ref_zscore = reference_cs_zscore(data)
        fast_zscore = fast_cs_zscore(data)
        # Single value has zero std, should return constant_value
        assert ref_zscore[0, 0] == fast_zscore[0, 0] == 0.0

    def test_all_zeros(self):
        """Test arrays with all zeros."""
        data = np.zeros((10, 20))

        ref_zscore = reference_cs_zscore(data)
        fast_zscore = fast_cs_zscore(data)

        # All zeros -> zero std -> should return constant_value
        np.testing.assert_array_equal(ref_zscore, fast_zscore)

    def test_inf_handling(self):
        """Test that Inf is treated as non-finite."""
        data = np.array([[1.0, 2.0, np.inf, 4.0, -np.inf]])

        ref_rank = reference_cs_rank(data)
        fast_rank = fast_cs_rank(data)

        # Inf should produce NaN in output
        assert np.isnan(ref_rank[0, 2])
        assert np.isnan(fast_rank[0, 2])
        assert np.isnan(ref_rank[0, 4])
        assert np.isnan(fast_rank[0, 4])

    def test_very_small_values(self):
        """Test numerical stability with very small values."""
        data = np.array([[1e-15, 2e-15, 3e-15, 4e-15, 5e-15]])

        ref_zscore = reference_cs_zscore(data)
        fast_zscore = fast_cs_zscore(data)

        # Should still compute correctly despite small magnitudes
        np.testing.assert_allclose(ref_zscore, fast_zscore, rtol=1e-10, atol=1e-14)

    def test_very_large_values(self):
        """Test numerical stability with very large values."""
        data = np.array([[1e10, 2e10, 3e10, 4e10, 5e10]])

        ref_zscore = reference_cs_zscore(data)
        fast_zscore = fast_cs_zscore(data)

        # Should still compute correctly despite large magnitudes
        np.testing.assert_allclose(ref_zscore, fast_zscore, rtol=1e-10, atol=1e-12)


# ============================================================================
# Capability reporting
# ============================================================================

def test_capabilities():
    """Test that get_capabilities returns expected keys."""
    caps = get_capabilities()

    assert isinstance(caps, dict)
    assert "bottleneck" in caps
    assert "numba" in caps
    assert "stride_tricks" in caps

    # stride_tricks should always be available
    assert caps["stride_tricks"] is True

    # bottleneck and numba availability should match imports
    assert caps["bottleneck"] == HAS_BOTTLENECK
    assert caps["numba"] == HAS_NUMBA
