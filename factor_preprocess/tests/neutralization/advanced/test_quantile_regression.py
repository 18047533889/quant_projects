"""
Test suite for quantile regression neutralization.
"""
import pytest
import pandas as pd
import numpy as np
from factor_preprocess.neutralization.advanced.quantile_regression import (
    quantile_neutralize,
)


class TestQuantileNeutralize:
    """Test quantile regression neutralization."""

    def test_basic_quantile_neutralization(self):
        """Test basic quantile neutralization at median."""
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

        result = quantile_neutralize(
            values_df, exposures_df, quantile=0.5, min_observations=4
        )

        # Result should have same length as input
        assert len(result) == len(values_df)
        assert result.notna().any()

    def test_different_quantiles(self):
        """Test neutralization at different quantiles."""
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

        # Test different quantiles
        result_lower = quantile_neutralize(
            values_df, exposures_df, quantile=0.25, min_observations=5
        )
        result_median = quantile_neutralize(
            values_df, exposures_df, quantile=0.5, min_observations=5
        )
        result_upper = quantile_neutralize(
            values_df, exposures_df, quantile=0.75, min_observations=5
        )

        # All should produce results
        assert result_lower.notna().any()
        assert result_median.notna().any()
        assert result_upper.notna().any()

    def test_extreme_quantiles(self):
        """Test neutralization at extreme quantiles."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(20)]

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": float(i) + np.random.randn() * 0.5,
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exposure": float(i),
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        # Test extreme quantiles
        result_low = quantile_neutralize(
            values_df, exposures_df, quantile=0.05, min_observations=10
        )
        result_high = quantile_neutralize(
            values_df, exposures_df, quantile=0.95, min_observations=10
        )

        # Both should produce results
        assert result_low.notna().any()
        assert result_high.notna().any()

    def test_per_date_independence(self):
        """Test that quantile neutralization is per-date."""
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

        result = quantile_neutralize(
            values_df, exposures_df, quantile=0.5, min_observations=3
        )

        # Each date should be processed independently
        assert result.notna().any()

    def test_invalid_quantile(self):
        """Test that invalid quantile raises error."""
        dates = [pd.Timestamp("2020-01-01")]
        assets = ["A", "B", "C"]

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
                "exposure": 1.0,
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        with pytest.raises(ValueError, match="Quantile must be in"):
            quantile_neutralize(values_df, exposures_df, quantile=1.5)

        with pytest.raises(ValueError, match="Quantile must be in"):
            quantile_neutralize(values_df, exposures_df, quantile=-0.1)

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

        result = quantile_neutralize(
            values_df,
            exposures_df,
            quantile=0.5,
            min_observations=10,
        )

        # All should be NaN
        assert result.isna().all()

    def test_nan_handling(self):
        """Test NaN handling in quantile regression."""
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

        result = quantile_neutralize(
            values_df, exposures_df, quantile=0.5, min_observations=3
        )

        # NaN inputs should produce NaN residuals
        assert result.isna().iloc[-2:].all()

        # Valid inputs should produce finite residuals
        assert result.notna().iloc[:5].all()

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

        result_with = quantile_neutralize(
            values_df, exposures_df, quantile=0.5, add_intercept=True, min_observations=3
        )
        result_without = quantile_neutralize(
            values_df, exposures_df, quantile=0.5, add_intercept=False, min_observations=3
        )

        # Both should produce results
        assert result_with.notna().any()
        assert result_without.notna().any()

    def test_outlier_robustness(self):
        """Test that quantile regression is robust to outliers."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(20)]

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            # Add outliers
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

        result = quantile_neutralize(
            values_df, exposures_df, quantile=0.5, min_observations=10
        )

        # Should produce results
        assert result.notna().any()

        # Non-outliers should have reasonable residuals
        non_outlier_residuals = result.iloc[:18]
        assert non_outlier_residuals.notna().all()

    def test_convergence_parameters(self):
        """Test different convergence parameters."""
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

        # Test with different max_iter and tol
        result_few = quantile_neutralize(
            values_df, exposures_df, quantile=0.5, max_iter=10, min_observations=5
        )
        result_many = quantile_neutralize(
            values_df, exposures_df, quantile=0.5, max_iter=100, min_observations=5
        )
        result_tight = quantile_neutralize(
            values_df, exposures_df, quantile=0.5, tol=1e-6, min_observations=5
        )

        # All should produce results
        assert result_few.notna().any()
        assert result_many.notna().any()
        assert result_tight.notna().any()
