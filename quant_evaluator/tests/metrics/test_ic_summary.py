"""
Tests for ic_summary metrics with golden reference values.
"""

import pytest
import numpy as np
from scipy import stats

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic_summary import (
    compute_icir,
    compute_ic_tstat,
    compute_ic_decay,
    compute_ic_stability,
)


class TestICIR:
    """Test Information Coefficient Information Ratio."""

    def test_icir_basic(self):
        """Basic ICIR computation."""
        ic_series = np.array([
            [0.05, 0.10],
            [0.06, 0.12],
            [0.04, 0.08],
            [0.05, 0.11],
            [0.07, 0.09],
        ] * 5)  # 25 periods, 2 factors

        icir = compute_icir(ic_series, min_periods=20)

        assert icir.shape == (2,)
        # ICIR = mean / std
        expected_icir_f0 = np.mean(ic_series[:, 0]) / np.std(ic_series[:, 0], ddof=1)
        assert np.isclose(icir[0], expected_icir_f0, atol=1e-10)

    def test_icir_high_consistency(self):
        """High IC consistency yields high ICIR."""
        ic_series = np.array([[0.05], [0.051], [0.049], [0.05], [0.0505]] * 10)  # 50 periods
        icir = compute_icir(ic_series, min_periods=20)

        # Very low variance -> high ICIR
        assert icir[0] > 10.0

    def test_icir_low_consistency(self):
        """High IC volatility yields low ICIR."""
        np.random.seed(42)
        ic_series = np.random.randn(100, 1) * 0.2  # High variance, mean ~0
        icir = compute_icir(ic_series, min_periods=20)

        # Close to zero mean, high variance -> low ICIR
        assert abs(icir[0]) < 1.0

    def test_icir_insufficient_periods(self):
        """Insufficient periods returns NaN."""
        ic_series = np.array([[0.05], [0.06], [0.04]])
        icir = compute_icir(ic_series, min_periods=10)

        assert np.isnan(icir[0])

    def test_icir_zero_variance(self):
        """Zero variance IC returns NaN."""
        ic_series = np.array([[0.05], [0.05], [0.05]] * 10)
        icir = compute_icir(ic_series, min_periods=10)

        assert np.isnan(icir[0])

    def test_icir_with_nans(self):
        """NaN values are filtered."""
        ic_series = np.array([[0.05], [np.nan], [0.06], [0.04], [np.nan]] * 10)
        icir = compute_icir(ic_series, min_periods=20)

        # 30 valid periods
        valid_ic = ic_series[~np.isnan(ic_series[:, 0]), 0]
        expected_icir = np.mean(valid_ic) / np.std(valid_ic, ddof=1)
        assert np.isclose(icir[0], expected_icir, atol=1e-10)


class TestICTStat:
    """Test IC t-statistic."""

    def test_tstat_positive_significant(self):
        """Positive mean IC with low variance yields significant t-stat."""
        ic_series = np.array([[0.05]] * 30)  # Constant positive IC
        t_stat, p_value = compute_ic_tstat(ic_series, min_periods=20)

        # Should be highly significant
        assert t_stat[0] > 0
        assert p_value[0] < 0.01

    def test_tstat_zero_mean(self):
        """Zero mean IC yields t-stat near zero."""
        np.random.seed(200)
        ic_series = np.random.randn(200, 1)  # Larger sample, mean ~0
        t_stat, p_value = compute_ic_tstat(ic_series, min_periods=20)

        # t-stat should be close to zero for independent random data
        assert abs(t_stat[0]) < 1.5
        # p-value should be high (not significant)
        assert p_value[0] > 0.05

    def test_tstat_negative_significant(self):
        """Negative mean IC yields negative t-stat."""
        ic_series = np.array([[-0.08]] * 25)
        t_stat, p_value = compute_ic_tstat(ic_series, min_periods=20)

        assert t_stat[0] < 0
        assert p_value[0] < 0.01

    def test_tstat_scipy_parity(self):
        """Verify parity with scipy.stats.ttest_1samp."""
        np.random.seed(42)
        ic_series = np.random.randn(50, 1) + 0.1
        t_stat, p_value = compute_ic_tstat(ic_series, min_periods=20)

        expected_t, expected_p = stats.ttest_1samp(ic_series[:, 0], 0.0)

        assert np.isclose(t_stat[0], expected_t, atol=1e-10)
        assert np.isclose(p_value[0], expected_p, atol=1e-10)

    def test_tstat_insufficient_periods(self):
        """Insufficient periods returns NaN."""
        ic_series = np.array([[0.05], [0.06], [0.04]])
        t_stat, p_value = compute_ic_tstat(ic_series, min_periods=10)

        assert np.isnan(t_stat[0])
        assert np.isnan(p_value[0])


class TestICDecay:
    """Test IC decay analysis."""

    def test_decay_single_factor(self):
        """IC decay for single factor across horizons."""
        T, N = 50, 100
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(123)
        factor_values = np.random.randn(T, N, 1)

        batch = FactorBatch(
            factor_ids=("factor_1",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values,
        )

        # Create labels at different horizons with decaying correlation
        label_1d = factor_values[:, :, 0] + np.random.randn(T, N) * 0.5
        label_5d = factor_values[:, :, 0] + np.random.randn(T, N) * 1.0
        label_20d = factor_values[:, :, 0] + np.random.randn(T, N) * 2.0

        bundles = (
            LabelBundle(
                target_id="ret_1d", values=label_1d, horizon=1,
                decision_time=tuple(range(T)), label_start_time=tuple(range(T)),
                label_end_time=tuple(range(1, T+1)),
            ),
            LabelBundle(
                target_id="ret_5d", values=label_5d, horizon=5,
                decision_time=tuple(range(T)), label_start_time=tuple(range(T)),
                label_end_time=tuple(range(5, T+5)),
            ),
            LabelBundle(
                target_id="ret_20d", values=label_20d, horizon=20,
                decision_time=tuple(range(T)), label_start_time=tuple(range(T)),
                label_end_time=tuple(range(20, T+20)),
            ),
        )

        decay_matrix = compute_ic_decay(batch, bundles, method="pearson", min_assets=10)

        assert decay_matrix.shape == (3, 1)
        # IC should decay: 1d > 5d > 20d
        assert decay_matrix[0, 0] > decay_matrix[1, 0]
        assert decay_matrix[1, 0] > decay_matrix[2, 0]

    def test_decay_multiple_factors(self):
        """IC decay for multiple factors."""
        T, N, F = 40, 80, 2
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(200)
        factor_values = np.random.randn(T, N, F)

        batch = FactorBatch(
            factor_ids=("f1", "f2"),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values,
        )

        label_1d = factor_values[:, :, 0] + np.random.randn(T, N) * 0.3
        label_10d = factor_values[:, :, 0] + np.random.randn(T, N) * 1.5

        bundles = (
            LabelBundle(
                target_id="ret_1d", values=label_1d, horizon=1,
                decision_time=tuple(range(T)), label_start_time=tuple(range(T)),
                label_end_time=tuple(range(1, T+1)),
            ),
            LabelBundle(
                target_id="ret_10d", values=label_10d, horizon=10,
                decision_time=tuple(range(T)), label_start_time=tuple(range(T)),
                label_end_time=tuple(range(10, T+10)),
            ),
        )

        decay_matrix = compute_ic_decay(batch, bundles, method="spearman", min_assets=10)

        assert decay_matrix.shape == (2, 2)
        # Both factors should show decay
        assert np.all(~np.isnan(decay_matrix))


class TestICStability:
    """Test IC stability measurement."""

    def test_stability_perfect(self):
        """Perfect stability with consistent IC pattern."""
        # Create IC series that is perfectly stable (constant within each window)
        # Use repeating constant blocks of size 30 (half the window size)
        ic_series = np.zeros((120, 1))
        for i in range(4):
            ic_series[i*30:(i+1)*30, 0] = 0.05 + i * 0.01

        stability_series, valid_windows = compute_ic_stability(
            ic_series, window_size=60, min_periods=20
        )

        # First and second halves of each 60-period window share the same blocks
        # → expect high correlation on average
        mean_stability = np.nanmean(stability_series[:, 0])
        assert mean_stability > 0.8
        assert valid_windows[0] > 0

    def test_stability_low(self):
        """Low stability with random IC."""
        np.random.seed(42)
        ic_series = np.random.randn(100, 1)

        stability_series, valid_windows = compute_ic_stability(
            ic_series, window_size=60, min_periods=20
        )

        # Random data -> low correlation between halves
        mean_stability = np.nanmean(stability_series[:, 0])
        assert abs(mean_stability) < 0.3

    def test_stability_insufficient_data(self):
        """Insufficient data returns empty result."""
        ic_series = np.random.randn(30, 1)

        stability_series, valid_windows = compute_ic_stability(
            ic_series, window_size=60, min_periods=20
        )

        assert stability_series.shape[0] == 0
        assert valid_windows[0] == 0

    def test_stability_window_size_validation(self):
        """Invalid window size raises error."""
        ic_series = np.random.randn(100, 1)

        with pytest.raises(ValueError, match="window_size must be even"):
            compute_ic_stability(ic_series, window_size=61, min_periods=20)

        with pytest.raises(ValueError, match="window_size must be even and >= 40"):
            compute_ic_stability(ic_series, window_size=30, min_periods=20)

    def test_stability_multiple_factors(self):
        """Stability for multiple factors."""
        np.random.seed(100)
        # Factor 1: stable pattern, Factor 2: unstable
        ic_f1 = np.tile([0.05, 0.06, 0.04], 40).reshape(-1, 1)  # 120 periods
        ic_f2 = np.random.randn(120, 1)
        ic_series = np.hstack([ic_f1, ic_f2])

        stability_series, valid_windows = compute_ic_stability(
            ic_series, window_size=60, min_periods=20
        )

        assert stability_series.shape[1] == 2
        # Factor 1 should have higher stability than Factor 2
        mean_stab_f1 = np.nanmean(stability_series[:, 0])
        mean_stab_f2 = np.nanmean(stability_series[:, 1])
        assert mean_stab_f1 > mean_stab_f2
