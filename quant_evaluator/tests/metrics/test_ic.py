"""
Tests for IC metrics with golden reference values.
"""

import pytest
import numpy as np
from scipy import stats

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import (
    compute_daily_ic,
    compute_mean_ic,
    _pearson_correlation,
    _spearman_rank_correlation,
)
from quant_evaluator.contracts.errors import InvalidContractError


class TestPearsonCorrelation:
    """Test Pearson correlation reference implementation."""

    def test_one_dimensional_label_validity_is_broadcast(self):
        batch = FactorBatch(
            factor_ids=("f1",),
            time_axis=AxisRef(name="time", dtype="int64", size=3),
            asset_axis=AxisRef(name="asset", dtype="int64", size=2),
            values=np.array([[[1.0], [2.0]], [[2.0], [4.0]], [[3.0], [6.0]]]),
        )
        bundle = LabelBundle(
            target_id="ret",
            values=np.array([1.0, 2.0, 3.0]),
            validity=np.array([True, False, True]),
            horizon=1,
            decision_time=(1, 2, 3),
            label_start_time=(1, 2, 3),
            label_end_time=(2, 3, 4),
        )

        series, counts = compute_daily_ic(batch, bundle, min_assets=2)

        assert np.isnan(series[0, 0])
        assert np.isnan(series[1, 0])
        assert np.isnan(series[2, 0])
        assert counts[:, 0].tolist() == [2, 0, 2]


    def test_perfect_positive_correlation(self):
        """Perfect positive correlation."""
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([2, 4, 6, 8, 10], dtype=float)
        corr = _pearson_correlation(x, y, min_obs=3)
        assert np.isclose(corr, 1.0, atol=1e-10)

    def test_perfect_negative_correlation(self):
        """Perfect negative correlation."""
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([10, 8, 6, 4, 2], dtype=float)
        corr = _pearson_correlation(x, y, min_obs=3)
        assert np.isclose(corr, -1.0, atol=1e-10)

    def test_zero_correlation(self):
        """Uncorrelated data."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        corr = _pearson_correlation(x, y, min_obs=10)
        # Should be close to 0 for independent random
        assert abs(corr) < 0.2

    def test_constant_x_returns_nan(self):
        """Constant factor returns NaN."""
        x = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        y = np.array([1, 2, 3, 4, 5], dtype=float)
        corr = _pearson_correlation(x, y, min_obs=3)
        assert np.isnan(corr)

    def test_constant_y_returns_nan(self):
        """Constant label returns NaN."""
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        corr = _pearson_correlation(x, y, min_obs=3)
        assert np.isnan(corr)

    def test_insufficient_observations(self):
        """Insufficient observations returns NaN."""
        x = np.array([1, 2, 3], dtype=float)
        y = np.array([4, 5, 6], dtype=float)
        corr = _pearson_correlation(x, y, min_obs=10)
        assert np.isnan(corr)

    def test_pairwise_finite_filtering(self):
        """NaN/Inf values are filtered pairwise."""
        x = np.array([1, 2, np.nan, 4, 5, np.inf], dtype=float)
        y = np.array([2, 4, 6, np.nan, 10, 12], dtype=float)
        # Valid pairs: (1,2), (2,4), (5,10) -> 3 pairs
        corr = _pearson_correlation(x, y, min_obs=3)
        expected_corr = np.corrcoef([1, 2, 5], [2, 4, 10])[0, 1]
        assert np.isclose(corr, expected_corr, atol=1e-10)

    def test_all_nan_returns_nan(self):
        """All NaN data returns NaN."""
        x = np.array([np.nan, np.nan, np.nan])
        y = np.array([1, 2, 3], dtype=float)
        corr = _pearson_correlation(x, y, min_obs=1)
        assert np.isnan(corr)


class TestSpearmanRankCorrelation:
    """Test Spearman rank correlation with average ties."""

    def test_perfect_rank_correlation(self):
        """Perfect rank correlation."""
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([10, 20, 30, 40, 50], dtype=float)
        corr = _spearman_rank_correlation(x, y, min_obs=3)
        assert np.isclose(corr, 1.0, atol=1e-10)

    def test_reverse_rank_correlation(self):
        """Reverse rank correlation."""
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([50, 40, 30, 20, 10], dtype=float)
        corr = _spearman_rank_correlation(x, y, min_obs=3)
        assert np.isclose(corr, -1.0, atol=1e-10)

    def test_average_ties_handling(self):
        """Ties are handled with average ranks."""
        x = np.array([1, 2, 2, 3], dtype=float)
        y = np.array([1, 2, 3, 4], dtype=float)
        corr = _spearman_rank_correlation(x, y, min_obs=3)
        # Scipy uses average ties by default
        expected, _ = stats.spearmanr(x, y)
        assert np.isclose(corr, expected, atol=1e-10)

    def test_nonlinear_monotonic(self):
        """Captures monotonic but nonlinear relationship."""
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([1, 4, 9, 16, 25], dtype=float)  # x^2
        corr = _spearman_rank_correlation(x, y, min_obs=3)
        assert np.isclose(corr, 1.0, atol=1e-10)  # Perfect monotonic

    def test_constant_returns_nan(self):
        """All-same values return NaN."""
        x = np.array([7, 7, 7, 7], dtype=float)
        y = np.array([1, 2, 3, 4], dtype=float)
        corr = _spearman_rank_correlation(x, y, min_obs=3)
        assert np.isnan(corr)


class TestDailyIC:
    """Test daily IC computation."""

    def test_single_factor_single_period(self):
        """Single factor, single time period."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=1)
        asset_axis = AxisRef(name="asset", dtype="int64", size=50)

        np.random.seed(100)
        factor_values = np.random.randn(50)
        label_values = factor_values + np.random.randn(50) * 0.1  # Correlated

        batch = FactorBatch(
            factor_ids=("factor_1",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values.reshape(1, 50, 1),
        )

        bundle = LabelBundle(
            target_id="ret_1d",
            values=label_values.reshape(1, 50),  # Shape (T, N)
            horizon=1,
            decision_time=(0,),
            label_start_time=(0,),
            label_end_time=(1,),
        )

        ic_series, valid_counts = compute_daily_ic(batch, bundle, method="pearson", min_assets=10)

        assert ic_series.shape == (1, 1)
        assert valid_counts[0, 0] == 50
        assert ic_series[0, 0] > 0.8  # Should be highly correlated

    def test_multiple_periods(self):
        """Multiple time periods."""
        T = 10
        N = 30
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(42)
        factor_values = np.random.randn(T, N, 1)
        label_values = factor_values[:, :, 0] + np.random.randn(T, N) * 0.5

        batch = FactorBatch(
            factor_ids=("factor_1",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values,
        )

        bundle = LabelBundle(
            target_id="ret_1d",
            values=label_values,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T+1)),
        )

        ic_series, valid_counts = compute_daily_ic(batch, bundle, method="pearson", min_assets=10)

        assert ic_series.shape == (T, 1)
        assert np.all(valid_counts[:, 0] == N)
        assert np.all(ic_series[:, 0] > 0)  # Should be positive

    def test_spearman_method(self):
        """Spearman RankIC computation."""
        T = 5
        N = 40
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(123)
        factor_values = np.random.randn(T, N, 1)
        # Nonlinear monotonic relationship
        label_values = np.sign(factor_values[:, :, 0]) * (factor_values[:, :, 0] ** 2)

        batch = FactorBatch(
            factor_ids=("factor_1",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values,
        )

        bundle = LabelBundle(
            target_id="ret_1d",
            values=label_values,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T+1)),
        )

        ic_series, _ = compute_daily_ic(batch, bundle, method="spearman", min_assets=10)

        # Spearman should detect monotonic relationship
        assert np.all(ic_series[:, 0] > 0.5)

    def test_shape_mismatch_raises(self):
        """Mismatched time axes raise error."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=10)
        asset_axis = AxisRef(name="asset", dtype="int64", size=50)

        batch = FactorBatch(
            factor_ids=("f1",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=np.random.randn(10, 50, 1),
        )

        bundle = LabelBundle(
            target_id="ret",
            values=np.random.randn(5, 50),  # Wrong time length
            horizon=1,
            decision_time=tuple(range(5)),
            label_start_time=tuple(range(5)),
            label_end_time=tuple(range(1, 6)),
        )

        with pytest.raises(InvalidContractError, match="does not match"):
            compute_daily_ic(batch, bundle, method="pearson")

    def test_min_assets_threshold(self):
        """Insufficient assets per period returns NaN."""
        time_axis = AxisRef(name="time", dtype="datetime64", size=2)
        asset_axis = AxisRef(name="asset", dtype="int64", size=5)

        batch = FactorBatch(
            factor_ids=("f1",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=np.random.randn(2, 5, 1),
        )

        bundle = LabelBundle(
            target_id="ret",
            values=np.random.randn(2, 5),
            horizon=1,
            decision_time=(0, 1),
            label_start_time=(0, 1),
            label_end_time=(1, 2),
        )

        ic_series, valid_counts = compute_daily_ic(batch, bundle, method="pearson", min_assets=10)

        # Both periods have < 10 assets
        assert np.all(np.isnan(ic_series))


class TestMeanIC:
    """Test mean IC aggregation."""

    def test_mean_ic_basic(self):
        """Basic mean IC computation."""
        ic_series = np.array([[0.5], [0.6], [0.4], [0.5], [0.55]])
        mean_ic, ic_std = compute_mean_ic(ic_series, min_periods=3)

        assert np.isclose(mean_ic[0], 0.51, atol=1e-10)
        assert ic_std[0] > 0

    def test_insufficient_periods_returns_nan(self):
        """Insufficient periods returns NaN."""
        ic_series = np.array([[0.5], [0.6]])
        mean_ic, ic_std = compute_mean_ic(ic_series, min_periods=5)

        assert np.isnan(mean_ic[0])
        assert np.isnan(ic_std[0])

    def test_nan_handling(self):
        """NaN values are excluded from mean."""
        ic_series = np.array([[0.5], [np.nan], [0.6], [np.nan], [0.4]])
        mean_ic, ic_std = compute_mean_ic(ic_series, min_periods=3)

        # Only 3 valid periods: 0.5, 0.6, 0.4
        expected_mean = (0.5 + 0.6 + 0.4) / 3
        assert np.isclose(mean_ic[0], expected_mean, atol=1e-10)
