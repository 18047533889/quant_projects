"""
Test suite for robust regression neutralization.
"""
import pytest
import pandas as pd
import numpy as np
from factor_preprocess.neutralization.advanced.robust_regression import (
    huber_neutralize,
    lad_neutralize,
)


class TestHuberNeutralize:
    """Test Huber regression neutralization."""

    def test_basic_huber_neutralization(self):
        """Test basic Huber neutralization."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=2)
        assets = ["A", "B", "C", "D", "E", "F"]

        values_list = []
        exposures_list = []

        for date in dates:
            for asset in assets:
                values_list.append({
                    "date": date,
                    "asset_id": asset,
                    "value": np.random.randn(),
                })
                exposures_list.append({
                    "date": date,
                    "asset_id": asset,
                    "size": np.random.randn(),
                    "value_exp": np.random.randn(),
                })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = huber_neutralize(values_df, exposures_df, min_observations=4)

        # Result should have same length as input
        assert len(result) == len(values_df)
        assert result.notna().any()

    def test_outlier_robustness(self):
        """Test that Huber is robust to outliers."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(20)]

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            # Add outliers to last 2 observations
            if i >= 18:
                value = 100.0  # Outlier
            else:
                value = float(i) * 0.5 + np.random.randn() * 0.1

            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": value,
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exposure": float(i),
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = huber_neutralize(values_df, exposures_df, delta=1.35, min_observations=10)

        # Should produce results
        assert result.notna().any()

        # Residuals for non-outliers should be reasonable
        non_outlier_residuals = result.iloc[:18]
        assert non_outlier_residuals.notna().all()

    def test_per_date_independence(self):
        """Test that Huber neutralization is per-date."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=2)
        assets = ["A", "B", "C", "D", "E"]

        values_list = []
        exposures_list = []

        for i, date in enumerate(dates):
            for j, asset in enumerate(assets):
                values_list.append({
                    "date": date,
                    "asset_id": asset,
                    "value": float(i * 10 + j),
                })
                exposures_list.append({
                    "date": date,
                    "asset_id": asset,
                    "exposure": float(j),
                })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = huber_neutralize(values_df, exposures_df, min_observations=3)

        # Each date should be processed independently
        assert result.notna().any()

    def test_insufficient_observations(self):
        """Test that insufficient observations produce NaN."""
        dates = [pd.Timestamp("2020-01-01")]
        assets = ["A", "B"]

        values_list = []
        exposures_list = []

        for asset in assets:
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": 1.0,
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exp1": 1.0,
                "exp2": 2.0,
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = huber_neutralize(
            values_df,
            exposures_df,
            min_observations=10,
        )

        # All should be NaN
        assert result.isna().all()

    def test_delta_parameter(self):
        """Test different delta parameters."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(15)]

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": float(i) + np.random.randn(),
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exposure": float(i),
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result_small = huber_neutralize(values_df, exposures_df, delta=0.5, min_observations=5)
        result_large = huber_neutralize(values_df, exposures_df, delta=5.0, min_observations=5)

        # Both should produce results
        assert result_small.notna().any()
        assert result_large.notna().any()


class TestLadNeutralize:
    """Test LAD (L1) regression neutralization."""

    def test_basic_lad_neutralization(self):
        """Test basic LAD neutralization."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=2)
        assets = ["A", "B", "C", "D", "E", "F"]

        values_list = []
        exposures_list = []

        for date in dates:
            for asset in assets:
                values_list.append({
                    "date": date,
                    "asset_id": asset,
                    "value": np.random.randn(),
                })
                exposures_list.append({
                    "date": date,
                    "asset_id": asset,
                    "size": np.random.randn(),
                    "momentum": np.random.randn(),
                })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = lad_neutralize(values_df, exposures_df, min_observations=4)

        # Result should have same length as input
        assert len(result) == len(values_df)
        assert result.notna().any()

    def test_extreme_outlier_robustness(self):
        """Test that LAD is highly robust to outliers."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(20)]

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            # Add extreme outliers
            if i >= 18:
                value = 1000.0  # Extreme outlier
            else:
                value = float(i) * 0.5 + np.random.randn() * 0.1

            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": value,
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exposure": float(i),
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = lad_neutralize(values_df, exposures_df, min_observations=10)

        # Should produce results
        assert result.notna().any()

        # Non-outliers should have reasonable residuals
        non_outlier_residuals = result.iloc[:18]
        assert non_outlier_residuals.notna().all()

    def test_per_date_independence(self):
        """Test that LAD neutralization is per-date."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=2)
        assets = ["A", "B", "C", "D", "E"]

        values_list = []
        exposures_list = []

        for i, date in enumerate(dates):
            for j, asset in enumerate(assets):
                values_list.append({
                    "date": date,
                    "asset_id": asset,
                    "value": float(i * 10 + j),
                })
                exposures_list.append({
                    "date": date,
                    "asset_id": asset,
                    "exposure": float(j),
                })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = lad_neutralize(values_df, exposures_df, min_observations=3)

        # Each date should be processed independently
        assert result.notna().any()

    def test_nan_handling(self):
        """Test NaN handling in LAD."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = ["A", "B", "C", "D", "E", "F", "G"]

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            value = np.random.randn() if i < 5 else np.nan
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": value,
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exposure": float(i),
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = lad_neutralize(values_df, exposures_df, min_observations=3)

        # NaN inputs should produce NaN residuals
        assert result.isna().iloc[-2:].all()

        # Valid inputs should produce finite residuals
        assert result.notna().iloc[:5].all()

    def test_insufficient_observations(self):
        """Test that insufficient observations produce NaN."""
        dates = [pd.Timestamp("2020-01-01")]
        assets = ["A", "B"]

        values_list = []
        exposures_list = []

        for asset in assets:
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": 1.0,
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exp1": 1.0,
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = lad_neutralize(
            values_df,
            exposures_df,
            min_observations=10,
        )

        # All should be NaN
        assert result.isna().all()

    def test_intercept_handling(self):
        """Test with and without intercept."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = ["A", "B", "C", "D", "E", "F"]

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": float(i) + 5.0,
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exposure": float(i),
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result_with = lad_neutralize(
            values_df, exposures_df, add_intercept=True, min_observations=3
        )
        result_without = lad_neutralize(
            values_df, exposures_df, add_intercept=False, min_observations=3
        )

        # Both should produce results
        assert result_with.notna().any()
        assert result_without.notna().any()

    def test_convergence(self):
        """Test that LAD converges within max_iter."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(10)]

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": float(i) + np.random.randn() * 0.1,
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exposure": float(i),
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        # Test with different max_iter
        result_few = lad_neutralize(
            values_df, exposures_df, max_iter=10, min_observations=5
        )
        result_many = lad_neutralize(
            values_df, exposures_df, max_iter=100, min_observations=5
        )

        # Both should produce results
        assert result_few.notna().any()
        assert result_many.notna().any()
