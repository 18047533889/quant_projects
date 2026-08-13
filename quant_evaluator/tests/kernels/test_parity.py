"""
Parity tests between reference and fast kernel implementations.

Verifies that fast kernels produce mathematically equivalent results to reference
implementations across various input scenarios.
"""

import pytest
import numpy as np

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.kernels.reference_bridge import (
    check_ic_parity,
    check_quantile_parity,
    check_quantile_returns_parity,
    check_turnover_parity,
)


def make_batch(T: int, N: int, F: int, seed: int = 42) -> FactorBatch:
    """Create a synthetic FactorBatch for testing."""
    np.random.seed(seed)
    values = np.random.randn(T, N, F)

    # Inject ~5% NaN values
    nan_mask = np.random.random((T, N, F)) < 0.05
    values[nan_mask] = np.nan

    time_axis = AxisRef(name="time", dtype="datetime64", size=T)
    asset_axis = AxisRef(name="asset", dtype="int64", size=N)

    return FactorBatch(
        factor_ids=tuple(f"f{i}" for i in range(F)),
        time_axis=time_axis,
        asset_axis=asset_axis,
        values=values,
    )


def make_bundle(T: int, N: int, seed: int = 42, correlated_with: np.ndarray = None) -> LabelBundle:
    """Create a synthetic LabelBundle for testing."""
    np.random.seed(seed + 1000)

    if correlated_with is not None:
        # Correlated labels
        values = correlated_with + np.random.randn(T, N) * 0.3
    else:
        values = np.random.randn(T, N)

    return LabelBundle(
        target_id="ret_1d",
        values=values,
        horizon=1,
        decision_time=tuple(range(T)),
        label_start_time=tuple(range(T)),
        label_end_time=tuple(range(1, T + 1)),
    )


class TestICParity:
    """Parity tests for IC computation (reference vs fast)."""

    def test_pearson_ic_parity_small(self):
        """Pearson IC parity on small batch."""
        batch = make_batch(T=10, N=50, F=3)
        bundle = make_bundle(T=10, N=50, correlated_with=batch.values[:, :, 0])

        result = check_ic_parity(batch, bundle, method="pearson", min_assets=10)

        assert result["passed"], (
            f"IC parity failed: max_diff={result['max_ic_diff']}, "
            f"count_diff={result['max_count_diff']}"
        )
        assert result["ic_match"]
        assert result["counts_match"]
        assert result["max_ic_diff"] < 1e-9

    def test_pearson_ic_parity_medium(self):
        """Pearson IC parity on medium batch (1000 factors)."""
        batch = make_batch(T=50, N=100, F=10, seed=123)
        bundle = make_bundle(T=50, N=100, seed=123, correlated_with=batch.values[:, :, 0])

        result = check_ic_parity(batch, bundle, method="pearson", min_assets=20)

        assert result["passed"], f"IC parity failed: {result['max_ic_diff']}"
        assert result["max_ic_diff"] < 1e-9

    def test_spearman_ic_parity_small(self):
        """Spearman RankIC parity on small batch."""
        batch = make_batch(T=10, N=50, F=2, seed=456)
        bundle = make_bundle(T=10, N=50, seed=456, correlated_with=batch.values[:, :, 0])

        result = check_ic_parity(batch, bundle, method="spearman", min_assets=10)

        assert result["passed"], f"RankIC parity failed: {result['max_ic_diff']}"
        assert result["max_ic_diff"] < 1e-9

    def test_spearman_ic_parity_medium(self):
        """Spearman RankIC parity on medium batch."""
        batch = make_batch(T=30, N=80, F=5, seed=789)
        bundle = make_bundle(T=30, N=80, seed=789, correlated_with=batch.values[:, :, 0])

        result = check_ic_parity(batch, bundle, method="spearman", min_assets=15)

        assert result["passed"]
        assert result["max_ic_diff"] < 1e-9

    def test_ic_parity_with_constant_factors(self):
        """Parity holds when some factors are constant (should produce NaN)."""
        T, N, F = 10, 50, 3
        np.random.seed(42)
        values = np.random.randn(T, N, F)
        # Make factor 1 constant across all periods
        values[:, :, 1] = 5.0

        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        batch = FactorBatch(
            factor_ids=("f0", "f1", "f2"),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )
        bundle = make_bundle(T=T, N=N, seed=42)

        result = check_ic_parity(batch, bundle, method="pearson", min_assets=10)

        assert result["passed"]
        # Factor 1 should be all NaN
        assert np.all(np.isnan(result["reference_ic"][:, 1]))
        assert np.all(np.isnan(result["fast_ic"][:, 1]))

    def test_ic_parity_with_validity_masks(self):
        """Parity holds with validity masks applied."""
        batch = make_batch(T=15, N=60, F=2, seed=999)

        # Add validity mask
        T, N, F = batch.values.shape
        validity = np.ones((T, N, F), dtype=bool)
        # Mask out 10% of observations
        mask = np.random.random((T, N, F)) < 0.1
        validity[mask] = False

        # Create new batch with validity
        from dataclasses import replace
        batch_with_validity = replace(batch, validity=validity)

        bundle = make_bundle(T=T, N=N, seed=999, correlated_with=batch.values[:, :, 0])

        result = check_ic_parity(batch_with_validity, bundle, method="pearson", min_assets=10)

        assert result["passed"]
        assert result["max_ic_diff"] < 1e-9

    def test_ic_parity_min_assets_threshold(self):
        """Parity holds with min_assets threshold filtering."""
        batch = make_batch(T=20, N=30, F=3, seed=111)
        bundle = make_bundle(T=20, N=30, seed=111, correlated_with=batch.values[:, :, 0])

        result = check_ic_parity(batch, bundle, method="pearson", min_assets=25)

        assert result["passed"]
        # Most periods should be NaN due to min_assets=25
        assert np.sum(np.isnan(result["reference_ic"])) > 0
        assert np.array_equal(
            np.isnan(result["reference_ic"]),
            np.isnan(result["fast_ic"]),
        )


class TestQuantileParity:
    """Parity tests for quantile binning (reference vs fast)."""

    def test_quantile_binning_parity_small(self):
        """Quantile binning parity on small batch."""
        batch = make_batch(T=10, N=50, F=2)

        result = check_quantile_parity(batch, n_quantiles=5)

        assert result["passed"], (
            f"Binning mismatch: max_diff={result['max_bin_diff']}, "
            f"mismatch_rate={result['mismatch_rate']}"
        )
        assert result["mismatch_rate"] == 0.0

    def test_quantile_binning_parity_medium(self):
        """Quantile binning parity on medium batch."""
        batch = make_batch(T=30, N=100, F=5, seed=222)

        result = check_quantile_parity(batch, n_quantiles=10)

        assert result["passed"]
        assert result["mismatch_rate"] == 0.0

    def test_quantile_binning_with_nans(self):
        """Binning handles NaN values consistently."""
        T, N, F = 10, 50, 2
        np.random.seed(333)
        values = np.random.randn(T, N, F)
        # Inject NaN
        values[0, 0:5, :] = np.nan
        values[5, 10:15, :] = np.nan

        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        batch = FactorBatch(
            factor_ids=("f0", "f1"),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        result = check_quantile_parity(batch, n_quantiles=5)

        assert result["passed"]
        # NaN positions should be -1 in both
        # Check t=0, assets 0:5
        assert np.all(result["reference_bins"][0, 0:5, :] == -1)
        assert np.all(result["fast_bins"][0, 0:5, :] == -1)
        # Check t=5, assets 10:15
        assert np.all(result["reference_bins"][5, 10:15, :] == -1)
        assert np.all(result["fast_bins"][5, 10:15, :] == -1)

    def test_quantile_returns_parity(self):
        """Quantile returns computation parity."""
        batch = make_batch(T=20, N=100, F=3, seed=444)
        bundle = make_bundle(T=20, N=100, seed=444, correlated_with=batch.values[:, :, 0])

        result = check_quantile_returns_parity(
            batch, bundle, n_quantiles=5, min_assets=10
        )

        assert result["passed"], (
            f"Returns mismatch: max_diff={result['max_return_diff']}, "
            f"count_diff={result['max_count_diff']}"
        )
        assert result["max_return_diff"] < 1e-9
        assert result["counts_match"]

    def test_quantile_returns_parity_10_quantiles(self):
        """Quantile returns with 10 quantiles."""
        batch = make_batch(T=15, N=200, F=2, seed=555)
        bundle = make_bundle(T=15, N=200, seed=555, correlated_with=batch.values[:, :, 0])

        result = check_quantile_returns_parity(
            batch, bundle, n_quantiles=10, min_assets=15
        )

        assert result["passed"]


class TestTurnoverParity:
    """Parity tests for turnover estimation (reference vs fast)."""

    def test_turnover_parity_small(self):
        """Turnover parity on small batch."""
        batch = make_batch(T=20, N=50, F=2, seed=666)

        result = check_turnover_parity(batch, window=1)

        assert result["passed"], f"Turnover mismatch: max_diff={result['max_turnover_diff']}"
        assert result["max_turnover_diff"] < 1e-9

    def test_turnover_parity_medium(self):
        """Turnover parity on medium batch with larger window."""
        batch = make_batch(T=50, N=100, F=5, seed=777)

        result = check_turnover_parity(batch, window=5)

        assert result["passed"]
        assert result["max_turnover_diff"] < 1e-9

    def test_turnover_window_consistency(self):
        """Window parameter produces consistent parity."""
        batch = make_batch(T=30, N=80, F=3, seed=888)

        for window in [1, 2, 3, 5]:
            result = check_turnover_parity(batch, window=window)
            assert result["passed"], f"Window={window} failed"

    def test_turnover_first_window_nans(self):
        """First 'window' periods are NaN in both implementations."""
        batch = make_batch(T=20, N=50, F=2, seed=999)

        window = 3
        result = check_turnover_parity(batch, window=window)

        assert result["passed"]
        # First 3 periods should be NaN
        assert np.all(np.isnan(result["reference_turnover"][:window, :]))
        assert np.all(np.isnan(result["fast_turnover"][:window, :]))


class TestParityLargeScale:
    """Parity tests on large-scale factor batches (10k+ factors)."""

    def test_ic_parity_10k_factors(self):
        """IC parity on batch with 10k+ factors."""
        # 10,000 factors: T=20, N=50, F=10000
        # To keep memory manageable, use smaller T/N with many factors
        T, N, F = 20, 50, 10000

        np.random.seed(1234)
        # Generate factor values in chunks to manage memory
        chunk_size = 1000
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
        bundle = make_bundle(T=T, N=N, seed=1234)

        result = check_ic_parity(batch, bundle, method="pearson", min_assets=10)

        assert result["passed"], f"10k factor IC parity failed: {result['max_ic_diff']}"
        assert result["max_ic_diff"] < 1e-9
        assert result["ic_match"]
        assert result["counts_match"]

    def test_quantile_parity_10k_factors(self):
        """Quantile binning parity on 10k factors."""
        T, N, F = 10, 100, 10000

        np.random.seed(5678)
        # Generate in chunks
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

        result = check_quantile_parity(batch, n_quantiles=5)

        assert result["passed"], f"10k factor quantile parity failed"
        assert result["mismatch_rate"] == 0.0