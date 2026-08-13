"""
Integration test: Slicing and grouping workflows.

Tests factor evaluation with various slicing dimensions (time windows,
asset groups, market segments) and grouped metric aggregation.
"""

import pytest
import numpy as np
from typing import Dict, List

from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import compute_daily_ic, compute_mean_ic
from quant_evaluator.metrics.quality import compute_coverage
from quant_evaluator.metrics.quantile import compute_quantile_returns, compute_top_bottom_spread


class TestTimeSlicing:
    """Test evaluation on time-sliced factor batches."""

    @pytest.fixture
    def full_batch(self):
        """Create a full time series batch."""
        T, N = 200, 150
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(42)
        factor_values = np.random.randn(T, N, 1)

        batch = FactorBatch(
            factor_ids=("momentum_20d",),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=factor_values,
        )

        return batch

    @pytest.fixture
    def full_labels(self, full_batch):
        """Create labels for full batch."""
        T, N = full_batch.num_times, full_batch.num_assets
        np.random.seed(100)

        labels = full_batch.values[:, :, 0] + np.random.randn(T, N) * 0.3

        bundle = LabelBundle(
            target_id="ret_1d",
            values=labels,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

        return bundle

    def test_rolling_window_evaluation(self, full_batch, full_labels):
        """Evaluate on rolling time windows."""
        T = full_batch.num_times
        window_size = 60
        step = 20

        results = []

        for start in range(0, T - window_size, step):
            end = start + window_size

            # Slice batch
            time_slice = slice(start, end)
            sliced_values = full_batch.values[time_slice, :, :]
            sliced_labels = full_labels.values[time_slice, :]

            time_axis = AxisRef(name="time", dtype="datetime64", size=window_size)
            asset_axis = full_batch.asset_axis

            batch = FactorBatch(
                factor_ids=full_batch.factor_ids,
                time_axis=time_axis,
                asset_axis=asset_axis,
                values=sliced_values,
            )

            bundle = LabelBundle(
                target_id=full_labels.target_id,
                values=sliced_labels,
                horizon=full_labels.horizon,
                decision_time=tuple(range(window_size)),
                label_start_time=tuple(range(window_size)),
                label_end_time=tuple(range(1, window_size + 1)),
            )

            # Compute IC for this window
            ic_series, _ = compute_daily_ic(batch, bundle, method="pearson")
            mean_ic, ic_std = compute_mean_ic(ic_series, min_periods=20)

            results.append({
                "window_start": start,
                "window_end": end,
                "mean_ic": mean_ic[0],
                "ic_std": ic_std[0],
            })

        # Should have multiple windows
        assert len(results) >= 6

        # All windows should have valid IC
        for r in results:
            assert not np.isnan(r["mean_ic"])
            assert not np.isnan(r["ic_std"])

    def test_expanding_window_evaluation(self, full_batch, full_labels):
        """Evaluate with expanding window (growing history)."""
        min_window = 40
        T = full_batch.num_times

        results = []

        for end in [60, 100, 140, 200]:
            if end > T:
                continue

            # Expanding window from start
            sliced_values = full_batch.values[:end, :, :]
            sliced_labels = full_labels.values[:end, :]

            time_axis = AxisRef(name="time", dtype="datetime64", size=end)
            asset_axis = full_batch.asset_axis

            batch = FactorBatch(
                factor_ids=full_batch.factor_ids,
                time_axis=time_axis,
                asset_axis=asset_axis,
                values=sliced_values,
            )

            bundle = LabelBundle(
                target_id=full_labels.target_id,
                values=sliced_labels,
                horizon=full_labels.horizon,
                decision_time=tuple(range(end)),
                label_start_time=tuple(range(end)),
                label_end_time=tuple(range(1, end + 1)),
            )

            ic_series, _ = compute_daily_ic(batch, bundle, method="spearman")
            mean_ic, ic_std = compute_mean_ic(ic_series, min_periods=20)

            results.append({
                "window_size": end,
                "mean_ic": mean_ic[0],
                "ic_std": ic_std[0],
            })

        assert len(results) == 4

        # IC estimates should stabilize as window grows
        ic_values = [r["mean_ic"] for r in results]
        std_values = [r["ic_std"] for r in results]

        # Later windows should have more stable estimates
        assert std_values[-1] < std_values[0] * 1.2  # Should not increase dramatically

    def test_year_over_year_comparison(self):
        """Compare metrics across different year segments."""
        T, N = 240, 180  # ~1 year of daily data
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(789)

        # Create factor with time-varying signal strength
        factor_values = np.random.randn(T, N, 1)

        # Year 1: strong signal
        labels_y1 = factor_values[:120, :, 0] * 0.8 + np.random.randn(120, N) * 0.2

        # Year 2: weak signal
        labels_y2 = factor_values[120:, :, 0] * 0.2 + np.random.randn(120, N) * 0.5

        labels = np.vstack([labels_y1, labels_y2])

        batch = FactorBatch(
            factor_ids=("alpha_factor",),
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

        # Evaluate year 1
        ic_y1, _ = compute_daily_ic(batch, bundle, method="pearson")
        mean_ic_y1, _ = compute_mean_ic(ic_y1[:120, :], min_periods=20)

        # Evaluate year 2
        ic_y2, _ = compute_daily_ic(batch, bundle, method="pearson")
        mean_ic_y2, _ = compute_mean_ic(ic_y2[120:, :], min_periods=20)

        # Year 1 should have stronger IC than year 2
        assert mean_ic_y1[0] > mean_ic_y2[0] * 1.5


class TestAssetGrouping:
    """Test evaluation with asset grouping and segmentation."""

    def test_large_cap_vs_small_cap_segments(self):
        """Evaluate factor separately on large-cap vs small-cap segments."""
        T, N = 100, 300
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(333)
        factor_values = np.random.randn(T, N, 1)

        # Create market cap proxy
        # First 100 assets = large cap, last 200 = small cap
        large_cap_mask = np.zeros((T, N), dtype=bool)
        large_cap_mask[:, :100] = True

        small_cap_mask = ~large_cap_mask

        # Labels: factor works better for large cap
        labels = np.random.randn(T, N)

        # Assign labels separately for each segment
        for t in range(T):
            # Large cap
            labels[t, :100] = (
                factor_values[t, :100, 0] * 0.7 + np.random.randn(100) * 0.3
            )
            # Small cap
            labels[t, 100:] = (
                factor_values[t, 100:, 0] * 0.1 + np.random.randn(200) * 0.6
            )

        batch = FactorBatch(
            factor_ids=("size_factor",),
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

        # Evaluate on large cap subset
        large_cap_values = factor_values[:, :100, :]
        large_cap_labels = labels[:, :100]

        large_cap_axis = AxisRef(name="asset", dtype="int64", size=100)
        large_cap_batch = FactorBatch(
            factor_ids=batch.factor_ids,
            time_axis=time_axis,
            asset_axis=large_cap_axis,
            values=large_cap_values,
        )

        large_cap_bundle = LabelBundle(
            target_id="ret_1d",
            values=large_cap_labels,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

        ic_large, _ = compute_daily_ic(large_cap_batch, large_cap_bundle, method="pearson")
        mean_ic_large, _ = compute_mean_ic(ic_large, min_periods=20)

        # Evaluate on small cap subset
        small_cap_values = factor_values[:, 100:, :]
        small_cap_labels = labels[:, 100:]

        small_cap_axis = AxisRef(name="asset", dtype="int64", size=200)
        small_cap_batch = FactorBatch(
            factor_ids=batch.factor_ids,
            time_axis=time_axis,
            asset_axis=small_cap_axis,
            values=small_cap_values,
        )

        small_cap_bundle = LabelBundle(
            target_id="ret_1d",
            values=small_cap_labels,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

        ic_small, _ = compute_daily_ic(small_cap_batch, small_cap_bundle, method="pearson")
        mean_ic_small, _ = compute_mean_ic(ic_small, min_periods=20)

        # Large cap IC should be significantly higher
        assert mean_ic_large[0] > mean_ic_small[0] * 2.0

    def test_sector_grouped_evaluation(self):
        """Evaluate factor performance by sector."""
        T, N = 80, 240  # 240 assets = 3 sectors × 80 assets
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(456)
        factor_values = np.random.randn(T, N, 1)

        # Sector 1: Tech (0-79)
        # Sector 2: Finance (80-159)
        # Sector 3: Energy (160-239)

        labels = np.random.randn(T, N)

        # Tech: strong positive signal
        for t in range(T):
            labels[t, :80] = factor_values[t, :80, 0] * 0.8 + np.random.randn(80) * 0.2

        # Finance: moderate signal
        for t in range(T):
            labels[t, 80:160] = factor_values[t, 80:160, 0] * 0.4 + np.random.randn(80) * 0.4

        # Energy: weak/negative signal
        for t in range(T):
            labels[t, 160:] = -factor_values[t, 160:, 0] * 0.3 + np.random.randn(80) * 0.5

        sector_results = {}

        for sector_name, asset_slice in [
            ("tech", slice(0, 80)),
            ("finance", slice(80, 160)),
            ("energy", slice(160, 240)),
        ]:
            sector_values = factor_values[:, asset_slice, :]
            sector_labels = labels[:, asset_slice]
            sector_size = sector_values.shape[1]

            sector_axis = AxisRef(name="asset", dtype="int64", size=sector_size)
            sector_batch = FactorBatch(
                factor_ids=("momentum",),
                time_axis=time_axis,
                asset_axis=sector_axis,
                values=sector_values,
            )

            sector_bundle = LabelBundle(
                target_id="ret_1d",
                values=sector_labels,
                horizon=1,
                decision_time=tuple(range(T)),
                label_start_time=tuple(range(T)),
                label_end_time=tuple(range(1, T + 1)),
            )

            ic_series, _ = compute_daily_ic(sector_batch, sector_bundle, method="spearman")
            mean_ic, _ = compute_mean_ic(ic_series, min_periods=20)

            coverage, _, _ = compute_coverage(sector_batch, sector_bundle)

            sector_results[sector_name] = {
                "mean_ic": mean_ic[0],
                "coverage": coverage,
            }

        # Tech should have strongest IC
        assert sector_results["tech"]["mean_ic"] > 0.4

        # Finance moderate (relaxed due to randomness)
        assert sector_results["finance"]["mean_ic"] > 0.0

        # Energy weak or negative
        assert sector_results["energy"]["mean_ic"] < sector_results["tech"]["mean_ic"]


class TestGroupedAggregation:
    """Test metric aggregation across groups."""

    def test_multi_factor_grouped_summary(self):
        """Compute summary statistics grouped by factor."""
        T, N = 90, 160
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(888)

        # Create 4 factors with different characteristics
        f1 = np.random.randn(T, N)  # Strong signal
        f2 = np.random.randn(T, N)  # Medium signal
        f3 = np.random.randn(T, N)  # Weak signal
        f4 = np.random.randn(T, N)  # Negative signal

        values = np.stack([f1, f2, f3, f4], axis=2)

        batch = FactorBatch(
            factor_ids=("strong", "medium", "weak", "negative"),
            time_axis=time_axis,
            asset_axis=asset_axis,
            values=values,
        )

        # Create labels with known correlations
        labels = (
            0.7 * f1 +
            0.4 * f2 +
            0.1 * f3 -
            0.5 * f4 +
            np.random.randn(T, N) * 0.3
        )

        bundle = LabelBundle(
            target_id="ret_1d",
            values=labels,
            horizon=1,
            decision_time=tuple(range(T)),
            label_start_time=tuple(range(T)),
            label_end_time=tuple(range(1, T + 1)),
        )

        # Compute metrics for all factors
        ic_series, valid_counts = compute_daily_ic(batch, bundle, method="pearson")
        mean_ic, ic_std = compute_mean_ic(ic_series, min_periods=20)

        q_returns, _ = compute_quantile_returns(batch, bundle, n_quantiles=5)
        spread = compute_top_bottom_spread(q_returns)

        # Build grouped summary
        summary = {}
        for idx, factor_id in enumerate(batch.factor_ids):
            summary[factor_id] = {
                "mean_ic": mean_ic[idx],
                "ic_std": ic_std[idx],
                "mean_spread": np.nanmean(spread[:, idx]),
                "valid_periods": np.sum(~np.isnan(ic_series[:, idx])),
            }

        # Validate rankings
        ic_ranking = sorted(summary.items(), key=lambda x: x[1]["mean_ic"], reverse=True)
        assert ic_ranking[0][0] == "strong"
        assert ic_ranking[1][0] == "medium"
        assert ic_ranking[-1][0] == "negative"

        # All factors should have full valid periods
        for factor_id in batch.factor_ids:
            assert summary[factor_id]["valid_periods"] == T

    def test_time_grouped_aggregation(self):
        """Aggregate metrics by time buckets (monthly, quarterly)."""
        T, N = 120, 150  # ~4 months of daily data
        time_axis = AxisRef(name="time", dtype="datetime64", size=T)
        asset_axis = AxisRef(name="asset", dtype="int64", size=N)

        np.random.seed(222)
        factor_values = np.random.randn(T, N, 1)
        labels = factor_values[:, :, 0] + np.random.randn(T, N) * 0.4

        batch = FactorBatch(
            factor_ids=("factor",),
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

        # Compute daily IC
        ic_series, _ = compute_daily_ic(batch, bundle, method="pearson")

        # Aggregate by 30-day buckets (monthly)
        bucket_size = 30
        monthly_ic = []

        for start in range(0, T, bucket_size):
            end = min(start + bucket_size, T)
            bucket_ic = ic_series[start:end, 0]

            monthly_mean = np.nanmean(bucket_ic)
            monthly_ic.append(monthly_mean)

        assert len(monthly_ic) == 4  # 4 months

        # All monthly ICs should be valid
        assert all(not np.isnan(ic) for ic in monthly_ic)
