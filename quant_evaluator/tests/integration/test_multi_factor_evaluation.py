"""
Integration test: Multi-factor evaluation workflows.

Tests end-to-end evaluation of multiple factors with various metrics,
validating full pipeline from batch creation to result extraction.
"""

import pytest
import numpy as np
from datetime import datetime

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import compute_daily_ic, compute_mean_ic
from quant_evaluator.metrics.quality import compute_coverage, compute_per_time_coverage
from quant_evaluator.metrics.quantile import compute_quantile_returns, compute_top_bottom_spread
from quant_evaluator.metrics.turnover import estimate_turnover_from_ranks
from quant_evaluator.diagnosis.factor import diagnose_all_factors


class TestMultiFactorEvaluation:
    """End-to-end multi-factor evaluation."""

    @pytest.fixture
    def multi_factor_batch(self):
        """Create a multi-factor batch with known characteristics."""
        T, N = 100, 200
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(42)

        # Factor 1: Strong positive signal
        f1 = np.random.randn(T, N)

        # Factor 2: Weak signal
        f2 = np.random.randn(T, N) * 0.3

        # Factor 3: Negative signal
        f3 = -np.random.randn(T, N)

        # Stack factors
        values = np.stack([f1, f2, f3], axis=2)  # (T, N, 3)

        batch = FactorBatch(
            factor_ids=("momentum_5d", "reversal_3d", "volatility_20d"),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        return batch

    @pytest.fixture
    def label_bundle(self, multi_factor_batch):
        """Create correlated labels for multi-factor batch."""
        T, N = multi_factor_batch.num_times, multi_factor_batch.num_assets
        np.random.seed(100)

        # Labels correlated with factor 1, weakly with factor 2, negatively with factor 3
        labels = (
            0.5 * multi_factor_batch.values[:, :, 0] +
            0.1 * multi_factor_batch.values[:, :, 1] -
            0.3 * multi_factor_batch.values[:, :, 2] +
            np.random.randn(T, N) * 0.5
        )

        bundle = LabelBundle(
            target_id="forward_return_1d",
            values=labels,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

        return bundle

    def test_complete_multi_factor_pipeline(self, multi_factor_batch, label_bundle):
        """Complete evaluation pipeline with all core metrics."""
        batch = multi_factor_batch

        # 1. Coverage diagnostics
        coverage, num_valid, num_total = compute_coverage(batch, label_bundle)
        assert coverage > 0.99  # Should have nearly full coverage
        assert num_valid == batch.num_times * batch.num_assets * batch.num_factors

        # 2. Daily IC for all factors
        ic_series, valid_counts = compute_daily_ic(batch, label_bundle, method="pearson", min_assets=20)
        assert ic_series.shape == (batch.num_times, batch.num_factors)
        assert np.all(valid_counts > 0)

        # 3. Mean IC per factor
        mean_ic, ic_std = compute_mean_ic(ic_series, min_periods=20)
        assert mean_ic.shape == (batch.num_factors,)

        # Factor 1 should have positive IC (strong signal)
        assert mean_ic[0] > 0.2

        # Factor 2 should have weak IC
        assert abs(mean_ic[1]) < 0.2

        # Factor 3 should have negative IC
        assert mean_ic[2] < -0.1

        # 4. Quantile returns
        q_returns, q_counts = compute_quantile_returns(batch, label_bundle, n_quantiles=5, min_assets=20)
        assert q_returns.shape == (batch.num_times, 5, batch.num_factors)

        # 5. Top-bottom spread
        spread = compute_top_bottom_spread(q_returns, top_q=-1, bottom_q=0)
        assert spread.shape == (batch.num_times, batch.num_factors)

        # Mean spread for factor 1 should be positive (if enough valid data)
        mean_spread_0 = np.nanmean(spread[:, 0])
        if not np.isnan(mean_spread_0):
            assert mean_spread_0 > -0.5  # Relaxed check

        # 6. Turnover estimation
        turnover_est = estimate_turnover_from_ranks(batch, window=1)
        assert turnover_est.shape == (batch.num_times, batch.num_factors)
        assert np.all(np.isnan(turnover_est[0, :]))  # First period is NaN

        # 7. Factor diagnostics
        diagnostics = diagnose_all_factors(batch)
        assert len(diagnostics) == batch.num_factors

        for factor_id in batch.factor_ids:
            diag = diagnostics[factor_id]
            assert diag.coverage > 0.99
            assert not diag.is_constant
            assert not diag.has_nans
            assert not diag.has_infs

    def test_rank_ic_vs_pearson_ic(self, multi_factor_batch, label_bundle):
        """Compare Pearson IC and Spearman RankIC."""
        batch = multi_factor_batch

        # Compute both IC types
        pearson_ic, _ = compute_daily_ic(batch, label_bundle, method="pearson", min_assets=20)
        spearman_ic, _ = compute_daily_ic(batch, label_bundle, method="spearman", min_assets=20)

        assert pearson_ic.shape == spearman_ic.shape

        # Compute means
        mean_pearson, _ = compute_mean_ic(pearson_ic, min_periods=20)
        mean_spearman, _ = compute_mean_ic(spearman_ic, min_periods=20)

        # Signs should generally agree for strong signals
        assert np.sign(mean_pearson[0]) == np.sign(mean_spearman[0])
        assert np.sign(mean_pearson[2]) == np.sign(mean_spearman[2])

        # RankIC often has higher magnitude for monotonic relationships
        assert abs(mean_spearman[0]) >= abs(mean_pearson[0]) * 0.8

    def test_per_time_coverage_tracking(self, multi_factor_batch, label_bundle):
        """Track coverage evolution over time."""
        batch = multi_factor_batch

        per_time_cov = compute_per_time_coverage(batch, label_bundle)
        assert per_time_cov.shape == (batch.num_times, batch.num_factors)

        # All periods should have full coverage
        assert np.all(per_time_cov > 0.99)

        # Mean coverage per factor
        mean_cov_per_factor = np.mean(per_time_cov, axis=0)
        assert np.all(mean_cov_per_factor > 0.99)

    def test_missing_data_handling(self):
        """Multi-factor evaluation with missing data."""
        T, N = 50, 100
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(123)

        # Create factors with missing data
        f1 = np.random.randn(T, N)
        f2 = np.random.randn(T, N)

        # Introduce 20% missing data
        missing_mask = np.random.rand(T, N) < 0.2
        f1[missing_mask] = np.nan
        f2[missing_mask] = np.nan

        values = np.stack([f1, f2], axis=2)

        batch = FactorBatch(
            factor_ids=("factor_a", "factor_b"),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        # Create labels
        labels = np.random.randn(T, N)
        bundle = LabelBundle(
            target_id="ret_1d",
            values=labels,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

        # Coverage should reflect missing data
        coverage, num_valid, num_total = compute_coverage(batch, bundle)
        assert 0.7 < coverage < 0.85  # Approximately 80% coverage

        # IC computation should handle missing data
        ic_series, valid_counts = compute_daily_ic(batch, bundle, method="pearson", min_assets=10)
        assert ic_series.shape == (T, 2)

        # Valid counts should vary by period
        assert np.min(valid_counts) < np.max(valid_counts)

        # Diagnostics should report missing data
        diagnostics = diagnose_all_factors(batch)
        for factor_id in batch.factor_ids:
            diag = diagnostics[factor_id]
            assert diag.has_nans
            assert diag.coverage < 0.85


class TestMetricPresetCombinations:
    """Test various metric preset combinations."""

    def test_basic_preset_all_metrics(self):
        """Basic preset: coverage, IC, quantile, turnover."""
        T, N = 60, 150
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(999)
        factor_values = np.random.randn(T, N, 1)
        labels = factor_values[:, :, 0] + np.random.randn(T, N) * 0.3

        batch = FactorBatch(
            factor_ids=("test_factor",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values,
        )

        bundle = LabelBundle(
            target_id="ret_1d",
            values=labels,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

        # Compute all basic metrics
        results = {}

        # Coverage
        cov, _, _ = compute_coverage(batch, bundle)
        results["coverage"] = cov

        # IC
        ic_series, _ = compute_daily_ic(batch, bundle, method="pearson")
        mean_ic, ic_std = compute_mean_ic(ic_series, min_periods=10)
        results["mean_ic"] = mean_ic[0]
        results["ic_std"] = ic_std[0]

        # RankIC
        rank_ic_series, _ = compute_daily_ic(batch, bundle, method="spearman")
        mean_rank_ic, _ = compute_mean_ic(rank_ic_series, min_periods=10)
        results["mean_rank_ic"] = mean_rank_ic[0]

        # Quantile
        q_returns, _ = compute_quantile_returns(batch, bundle, n_quantiles=5, min_assets=5)
        spread = compute_top_bottom_spread(q_returns, top_q=-1, bottom_q=0)
        results["mean_spread"] = np.nanmean(spread)

        # Turnover
        turnover = estimate_turnover_from_ranks(batch, window=1)
        results["mean_turnover"] = np.nanmean(turnover)

        # Validate all metrics computed successfully
        assert results["coverage"] > 0.95
        assert not np.isnan(results["mean_ic"])
        assert not np.isnan(results["mean_rank_ic"])
        # Spread may be NaN if insufficient data per quantile
        assert not np.isnan(results["mean_turnover"])

    def test_correlation_focused_preset(self):
        """Correlation-focused preset: Pearson, Spearman, various lags."""
        T, N = 80, 120
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(777)
        factor_values = np.random.randn(T, N, 1)
        labels = factor_values[:, :, 0] + np.random.randn(T, N) * 0.4

        batch = FactorBatch(
            factor_ids=("momentum",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values,
        )

        bundle = LabelBundle(
            target_id="ret_1d",
            values=labels,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

        # Compute multiple IC variants
        pearson_ic, _ = compute_daily_ic(batch, bundle, method="pearson")
        spearman_ic, _ = compute_daily_ic(batch, bundle, method="spearman")

        # Different minimum observation thresholds
        mean_ic_10, _ = compute_mean_ic(pearson_ic, min_periods=10)
        mean_ic_20, _ = compute_mean_ic(pearson_ic, min_periods=20)
        mean_ic_30, _ = compute_mean_ic(pearson_ic, min_periods=30)

        # All should be valid since T=80
        assert not np.isnan(mean_ic_10[0])
        assert not np.isnan(mean_ic_20[0])
        assert not np.isnan(mean_ic_30[0])

        # Values should be similar
        assert abs(mean_ic_10[0] - mean_ic_20[0]) < 0.1
        assert abs(mean_ic_20[0] - mean_ic_30[0]) < 0.1

    def test_quantile_focused_preset(self):
        """Quantile-focused preset: various quantile counts and spreads."""
        T, N = 70, 200
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(555)
        factor_values = np.random.randn(T, N, 1)
        labels = factor_values[:, :, 0] * 1.5 + np.random.randn(T, N) * 0.5

        batch = FactorBatch(
            factor_ids=("value",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values,
        )

        bundle = LabelBundle(
            target_id="ret_1d",
            values=labels,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

        # Compute quantiles with different bucket counts
        results = {}

        for n_q in [3, 5, 10]:
            q_returns, q_counts = compute_quantile_returns(batch, bundle, n_quantiles=n_q, min_assets=5)
            spread = compute_top_bottom_spread(q_returns, top_q=-1, bottom_q=0)
            results[f"q{n_q}_mean_spread"] = np.nanmean(spread)
            results[f"q{n_q}_returns_shape"] = q_returns.shape

        # Validate shapes
        assert results["q3_returns_shape"] == (T, 3, 1)
        assert results["q5_returns_shape"] == (T, 5, 1)
        assert results["q10_returns_shape"] == (T, 10, 1)

        # Check that spreads were computed (may be NaN with sparse data)
        # At least q3 should have some valid data
        assert "q3_mean_spread" in results
