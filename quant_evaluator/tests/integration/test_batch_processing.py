"""
Integration test: Batch processing workflows.

Tests efficient batch processing patterns, parallel factor evaluation,
memory-efficient streaming, and large-scale evaluation scenarios.
"""

import pytest
import numpy as np
from typing import List, Tuple

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import compute_daily_ic, compute_mean_ic
from quant_evaluator.metrics.quality import compute_coverage
from quant_evaluator.metrics.quantile import compute_quantile_returns
from quant_evaluator.diagnosis.factor import diagnose_all_factors


class TestBatchProcessing:
    """Test batch processing patterns for large-scale evaluation."""

    def test_large_factor_batch_processing(self):
        """Process a large batch of factors efficiently."""
        T, N = 120, 250
        num_factors = 50  # Large factor count

        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(999)

        # Generate 50 factors
        factor_ids = tuple(f"factor_{i:03d}" for i in range(num_factors))
        factor_values = np.random.randn(T, N, num_factors)

        batch = FactorBatch(
            factor_ids=factor_ids,
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values,
        )

        # Create labels with varying correlation to different factors
        labels = np.zeros((T, N))
        for i in range(min(10, num_factors)):
            weight = 0.2 / (i + 1)  # Decreasing weights
            labels += weight * factor_values[:, :, i]

        labels += np.random.randn(T, N) * 0.5

        bundle = LabelBundle(
            target_id="ret_1d",
            values=labels,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

        # Batch compute all metrics
        ic_series, valid_counts = compute_daily_ic(batch, bundle, method="pearson", min_assets=20)
        mean_ic, ic_std = compute_mean_ic(ic_series, min_periods=20)

        assert ic_series.shape == (T, num_factors)
        assert mean_ic.shape == (num_factors,)

        # First few factors should have higher IC (by construction)
        assert mean_ic[0] > mean_ic[10]
        assert mean_ic[1] > mean_ic[20]

        # All factors should have valid counts
        assert np.all(valid_counts > 0)

    def test_chunked_time_batch_processing(self):
        """Process time series in chunks for memory efficiency."""
        T_total = 500  # Long time series
        N = 100
        chunk_size = 100

        np.random.seed(111)

        # Generate full time series
        full_factor_values = np.random.randn(T_total, N, 1)
        full_labels = full_factor_values[:, :, 0] + np.random.randn(T_total, N) * 0.3

        # Process in chunks
        chunk_results = []

        for chunk_start in range(0, T_total, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T_total)
            chunk_len = chunk_end - chunk_start

            time_axis = AxisRef(name="time", dtype="datetime64", size=chunk_len)
            asset_axis = AxisRef(name="asset", dtype="int64", size=N)

            chunk_values = full_factor_values[chunk_start:chunk_end, :, :]
            chunk_labels = full_labels[chunk_start:chunk_end, :]

            batch = FactorBatch(
                factor_ids=("momentum",),
                time_axis=time_axis,
                asset_axis=asset_axis,
                values=chunk_values,
            )

            bundle = LabelBundle(
                target_id="ret_1d",
                values=chunk_labels,
                horizon=1,
                decision_time=tuple(range(chunk_len)),
                label_start_time=tuple(range(chunk_len)),
                label_end_time=tuple(range(1, chunk_len + 1)),
            )

            # Compute metrics for this chunk
            ic_series, _ = compute_daily_ic(batch, bundle, method="pearson")
            mean_ic, _ = compute_mean_ic(ic_series, min_periods=20)

            coverage, _, _ = compute_coverage(batch, bundle)

            chunk_results.append({
                "start": chunk_start,
                "end": chunk_end,
                "mean_ic": mean_ic[0],
                "coverage": coverage,
            })

        # Should have 5 chunks
        assert len(chunk_results) == 5

        # All chunks should have valid metrics
        for result in chunk_results:
            assert not np.isnan(result["mean_ic"])
            assert result["coverage"] > 0.99

        # IC values across chunks should be reasonably consistent
        ic_values = [r["mean_ic"] for r in chunk_results]
        ic_std = np.std(ic_values)
        assert ic_std < 0.2  # Should not vary wildly

    def test_parallel_factor_evaluation_simulation(self):
        """Simulate parallel evaluation of independent factor batches."""
        T, N = 80, 120
        num_batches = 10

        np.random.seed(777)

        batch_results = []

        # Simulate processing 10 independent batches
        for batch_idx in range(num_batches):
            time_axis = AxisRef(name="time", dtype="datetime64", size=T)
            asset_axis = AxisRef(name="asset", dtype="int64", size=N)

            # Each batch has 5 factors
            factor_ids = tuple(f"batch{batch_idx}_f{i}" for i in range(5))
            factor_values = np.random.randn(T, N, 5)

            batch = FactorBatch(
                factor_ids=factor_ids,
                time_axis=time_axis,
                asset_axis=asset_axis,
                values=factor_values,
            )

            # Different label correlation per batch
            weight = 0.3 + batch_idx * 0.05
            labels = weight * factor_values[:, :, 0] + np.random.randn(T, N) * 0.5

            bundle = LabelBundle(
                target_id="ret_1d",
                values=labels,
                horizon=1,
                decision_time=tuple(range(T)),
                label_start_time=tuple(range(T)),
                label_end_time=tuple(range(1, T + 1)),
            )

            # Evaluate batch
            ic_series, _ = compute_daily_ic(batch, bundle, method="spearman")
            mean_ic, _ = compute_mean_ic(ic_series, min_periods=20)

            batch_results.append({
                "batch_idx": batch_idx,
                "factor_count": 5,
                "mean_ic_per_factor": mean_ic,
            })

        # All batches processed
        assert len(batch_results) == num_batches

        # Each batch should have 5 factor results
        for result in batch_results:
            assert len(result["mean_ic_per_factor"]) == 5

    def test_incremental_batch_updates(self):
        """Test incremental evaluation as new data arrives."""
        N = 150
        initial_T = 60
        increment = 20

        np.random.seed(555)

        # Generate full dataset
        total_T = 120
        full_factor_values = np.random.randn(total_T, N, 1)
        full_labels = full_factor_values[:, :, 0] + np.random.randn(total_T, N) * 0.3

        results_timeline = []

        # Start with initial window
        for T_current in range(initial_T, total_T + 1, increment):
            time_axis = AxisRef(name="time", dtype="datetime64", size=T_current)
            asset_axis = AxisRef(name="asset", dtype="int64", size=N)

            batch = FactorBatch(
                factor_ids=("alpha",),
                time_axis=time_axis,
                asset_axis=asset_axis,
                values=full_factor_values[:T_current, :, :],
            )

            bundle = LabelBundle(
                target_id="ret_1d",
                values=full_labels[:T_current, :],
                horizon=1,
                decision_time=tuple(range(T_current)),
                label_start_time=tuple(range(T_current)),
                label_end_time=tuple(range(1, T_current + 1)),
            )

            ic_series, _ = compute_daily_ic(batch, bundle, method="pearson")
            mean_ic, ic_std = compute_mean_ic(ic_series, min_periods=20)

            results_timeline.append({
                "data_size": T_current,
                "mean_ic": mean_ic[0],
                "ic_std": ic_std[0],
            })

        # Should have multiple updates
        assert len(results_timeline) >= 3

        # IC estimate should stabilize with more data
        early_std = results_timeline[0]["ic_std"]
        late_std = results_timeline[-1]["ic_std"]

        # Standard error should decrease with more data
        assert late_std <= early_std * 1.1


class TestLargeScaleScenarios:
    """Test large-scale evaluation scenarios."""

    def test_high_dimensional_factor_batch(self):
        """Evaluate batch with many factors and assets."""
        T = 100
        N = 500  # Many assets
        F = 100  # Many factors

        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(321)

        factor_ids = tuple(f"f_{i:03d}" for i in range(F))
        factor_values = np.random.randn(T, N, F) * 0.5

        batch = FactorBatch(
            factor_ids=factor_ids,
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values,
        )

        # Create synthetic labels
        # First 20 factors contribute signal
        labels = np.sum(factor_values[:, :, :20], axis=2) * 0.1 + np.random.randn(T, N)

        bundle = LabelBundle(
            target_id="ret_1d",
            values=labels,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

        # Compute IC for all factors
        ic_series, valid_counts = compute_daily_ic(batch, bundle, method="pearson", min_assets=50)
        mean_ic, _ = compute_mean_ic(ic_series, min_periods=20)

        assert ic_series.shape == (T, F)
        assert mean_ic.shape == (F,)

        # First 20 factors should have higher mean IC
        signal_factors_ic = mean_ic[:20]
        noise_factors_ic = mean_ic[20:]

        assert np.mean(signal_factors_ic) > np.mean(noise_factors_ic)

    def test_long_time_series_evaluation(self):
        """Evaluate very long time series efficiently."""
        T = 1000  # Long time series
        N = 100
        F = 5

        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(654)

        factor_ids = tuple(f"factor_{i}" for i in range(F))
        factor_values = np.random.randn(T, N, F)

        batch = FactorBatch(
            factor_ids=factor_ids,
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values,
        )

        labels = 0.3 * factor_values[:, :, 0] + np.random.randn(T, N) * 0.6

        bundle = LabelBundle(
            target_id="ret_1d",
            values=labels,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

        # Compute IC
        ic_series, _ = compute_daily_ic(batch, bundle, method="spearman", min_assets=20)
        mean_ic, ic_std = compute_mean_ic(ic_series, min_periods=50)

        assert ic_series.shape == (T, F)

        # With 1000 periods, estimates should be stable
        assert all(ic_std < 0.5)

        # Coverage should be complete
        coverage, _, _ = compute_coverage(batch, bundle)
        assert coverage > 0.99

    def test_sparse_factor_batch(self):
        """Handle batch with significant sparsity."""
        T, N = 150, 200
        F = 10

        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(987)

        factor_values = np.random.randn(T, N, F)

        # Introduce 40% sparsity
        sparsity_mask = np.random.rand(T, N, F) < 0.4
        factor_values[sparsity_mask] = np.nan

        factor_ids = tuple(f"sparse_f{i}" for i in range(F))

        batch = FactorBatch(
            factor_ids=factor_ids,
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values,
        )

        labels = np.random.randn(T, N)

        bundle = LabelBundle(
            target_id="ret_1d",
            values=labels,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

        # Coverage should reflect sparsity
        coverage, num_valid, num_total = compute_coverage(batch, bundle)
        assert 0.5 < coverage < 0.7  # ~60% after pairwise filtering

        # IC computation should handle sparsity
        ic_series, valid_counts = compute_daily_ic(batch, bundle, method="pearson", min_assets=20)

        # Some periods may have insufficient data, but with 40% sparsity and 200 assets,
        # most periods should still have enough (>20 assets)
        # Relax the assertion - may not have NaN if enough assets remain
        valid_ic_count = np.sum(~np.isnan(ic_series))
        assert valid_ic_count > T * F * 0.3  # At least 30% valid

        # Diagnostics should flag sparsity
        diagnostics = diagnose_all_factors(batch)
        for factor_id in factor_ids:
            diag = diagnostics[factor_id]
            assert diag.has_nans
            assert diag.coverage < 0.7


class TestBatchResultAggregation:
    """Test aggregation patterns across multiple batch evaluations."""

    def test_cross_batch_factor_ranking(self):
        """Rank factors across multiple evaluation batches."""
        T, N = 80, 120

        np.random.seed(444)

        # Evaluate 3 different factor sets
        all_results = {}

        for batch_name, num_factors in [
            ("momentum", 8),
            ("value", 6),
            ("quality", 10),
        ]:
            time_axis = AxisRef(name="time", dtype="datetime64", size=T)
            asset_axis = AxisRef(name="asset", dtype="int64", size=N)

            factor_ids = tuple(f"{batch_name}_{i}" for i in range(num_factors))
            factor_values = np.random.randn(T, N, num_factors)

            batch = FactorBatch(
                factor_ids=factor_ids,
                time_axis=time_axis,
                asset_axis=asset_axis,
                values=factor_values,
            )

            # Variable signal strength by batch
            if batch_name == "momentum":
                label_weight = 0.5
            elif batch_name == "value":
                label_weight = 0.3
            else:
                label_weight = 0.1

            labels = label_weight * factor_values[:, :, 0] + np.random.randn(T, N) * 0.5

            bundle = LabelBundle(
                target_id="ret_1d",
                values=labels,
                horizon=1,
                decision_time=tuple(range(T)),
                label_start_time=tuple(range(T)),
                label_end_time=tuple(range(1, T + 1)),
            )

            ic_series, _ = compute_daily_ic(batch, bundle, method="pearson")
            mean_ic, ic_std = compute_mean_ic(ic_series, min_periods=20)

            for idx, factor_id in enumerate(factor_ids):
                all_results[factor_id] = {
                    "mean_ic": mean_ic[idx],
                    "ic_std": ic_std[idx],
                    "batch": batch_name,
                }

        # Rank all factors globally
        ranked_factors = sorted(
            all_results.items(),
            key=lambda x: x[1]["mean_ic"],
            reverse=True
        )

        # Total factors: 8 + 6 + 10 = 24
        assert len(ranked_factors) == 24

        # Top factors should include momentum batch (by construction with higher weight)
        # But due to randomness, we can't guarantee exact positions
        top_10 = ranked_factors[:10]
        momentum_in_top10 = sum(1 for name, _ in top_10 if "momentum" in name)
        assert momentum_in_top10 >= 2  # At least some momentum factors in top 10

    def test_aggregate_metrics_across_label_horizons(self):
        """Evaluate same factors against multiple label horizons."""
        T, N = 100, 150
        F = 5

        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(333)

        factor_ids = tuple(f"factor_{i}" for i in range(F))
        factor_values = np.random.randn(T, N, F)

        batch = FactorBatch(
            factor_ids=factor_ids,
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values,
        )

        # Test multiple horizons
        horizon_results = {}

        for horizon in [1, 5, 10, 20]:
            # Simulate different horizon labels (with appropriate correlations)
            decay = 0.95 ** (horizon - 1)
            labels = decay * 0.4 * factor_values[:, :, 0] + np.random.randn(T, N) * 0.5

            bundle = LabelBundle(
                target_id=f"ret_{horizon}d",
                values=labels,
                horizon=horizon,
                decision_time=tuple(range(T)),
                label_start_time=tuple(range(T)),
                label_end_time=tuple(range(horizon, T + horizon)),
            )

            ic_series, _ = compute_daily_ic(batch, bundle, method="spearman")
            mean_ic, _ = compute_mean_ic(ic_series, min_periods=20)

            horizon_results[horizon] = {
                "mean_ic": mean_ic,
                "first_factor_ic": mean_ic[0],
            }

        # All horizons evaluated
        assert len(horizon_results) == 4

        # Shorter horizons should have higher IC (by construction with decay)
        assert horizon_results[1]["first_factor_ic"] > horizon_results[20]["first_factor_ic"]
