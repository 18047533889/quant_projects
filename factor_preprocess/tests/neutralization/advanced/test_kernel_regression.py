"""
Test suite for kernel regression neutralization.
"""
import pytest
import pandas as pd
import numpy as np
from factor_preprocess.neutralization.advanced.kernel_regression import (
    kernel_neutralize,
)


class TestKernelNeutralize:
    """Test kernel regression neutralization."""

    def test_basic_kernel_neutralization_gaussian(self):
        """Test basic kernel neutralization with Gaussian kernel."""
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

        result = kernel_neutralize(
            values_df, exposures_df, kernel="gaussian", min_observations=4
        )

        # Result should have same length as input
        assert len(result) == len(values_df)
        assert result.notna().any()

    def test_different_kernels(self):
        """Test different kernel functions."""
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

        # Test all kernel types
        result_gaussian = kernel_neutralize(
            values_df, exposures_df, kernel="gaussian", min_observations=5
        )
        result_epanechnikov = kernel_neutralize(
            values_df, exposures_df, kernel="epanechnikov", min_observations=5
        )
        result_tricube = kernel_neutralize(
            values_df, exposures_df, kernel="tricube", min_observations=5
        )

        # All should produce results
        assert result_gaussian.notna().any()
        assert result_epanechnikov.notna().any()
        assert result_tricube.notna().any()

    def test_invalid_kernel(self):
        """Test that invalid kernel raises error."""
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

        with pytest.raises(ValueError, match="Unknown kernel"):
            kernel_neutralize(values_df, exposures_df, kernel="invalid")

    def test_bandwidth_parameter(self):
        """Test different bandwidth parameters."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(12)]

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": float(i) + np.random.randn() * 0.2,
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exposure": float(i),
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        # Test different bandwidths
        result_small = kernel_neutralize(
            values_df, exposures_df, bandwidth=0.5, min_observations=5
        )
        result_large = kernel_neutralize(
            values_df, exposures_df, bandwidth=5.0, min_observations=5
        )
        result_auto = kernel_neutralize(
            values_df, exposures_df, bandwidth=None, min_observations=5
        )

        # All should produce results
        assert result_small.notna().any()
        assert result_large.notna().any()
        assert result_auto.notna().any()

    def test_local_constant_vs_local_linear(self):
        """Test local constant vs local linear regression."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(12)]

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": float(i) + np.random.randn() * 0.2,
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exposure": float(i),
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result_constant = kernel_neutralize(
            values_df, exposures_df, local_constant=True, min_observations=5
        )
        result_linear = kernel_neutralize(
            values_df, exposures_df, local_constant=False, min_observations=5
        )

        # Both should produce results
        assert result_constant.notna().any()
        assert result_linear.notna().any()

    def test_per_date_independence(self):
        """Test that kernel neutralization is per-date."""
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

        result = kernel_neutralize(
            values_df, exposures_df, kernel="gaussian", min_observations=3
        )

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
                "exposure": 1.0,
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = kernel_neutralize(
            values_df,
            exposures_df,
            kernel="gaussian",
            min_observations=10,
        )

        # All should be NaN
        assert result.isna().all()

    def test_nan_handling(self):
        """Test NaN handling in kernel regression."""
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

        result = kernel_neutralize(
            values_df, exposures_df, kernel="gaussian", min_observations=3
        )

        # NaN inputs should produce NaN residuals
        assert result.isna().iloc[-2:].all()

        # Valid inputs should produce finite residuals
        assert result.notna().iloc[:5].all()

    def test_multidimensional_exposures(self):
        """Test kernel regression with multiple exposures."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(15)]

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
                "exp1": float(i),
                "exp2": float(i) * 0.5 + np.random.randn(),
                "exp3": np.random.randn(),
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = kernel_neutralize(
            values_df, exposures_df, kernel="gaussian", min_observations=8
        )

        # Should handle multiple dimensions
        assert result.notna().any()

    def test_normalize_option(self):
        """Test normalization option."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(10)]

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": float(i),
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exp1": float(i) * 100,  # Different scale
                "exp2": float(i) * 0.01,
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result_normalized = kernel_neutralize(
            values_df, exposures_df, normalize=True, min_observations=5
        )
        result_not_normalized = kernel_neutralize(
            values_df, exposures_df, normalize=False, min_observations=5
        )

        # Both should produce results
        assert result_normalized.notna().any()
        assert result_not_normalized.notna().any()

    def test_no_exposure_columns_fails(self):
        """Test that missing exposure columns fail gracefully."""
        values_df = pd.DataFrame({
            "date": [pd.Timestamp("2020-01-01")] * 3,
            "asset_id": ["A", "B", "C"],
            "value": [1.0, 2.0, 3.0],
        })
        exposures_df = pd.DataFrame({
            "date": [pd.Timestamp("2020-01-01")] * 3,
            "asset_id": ["A", "B", "C"],
        })

        with pytest.raises(ValueError, match="No exposure columns"):
            kernel_neutralize(values_df, exposures_df)

    def test_nonlinear_relationship(self):
        """Test kernel regression handles nonlinear relationships."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(20)]

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            x = float(i) / 5.0
            # Nonlinear relationship: y = x^2 + noise
            value = x ** 2 + np.random.randn() * 0.1
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": value,
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exposure": x,
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = kernel_neutralize(
            values_df, exposures_df, kernel="gaussian", bandwidth=1.0, min_observations=10
        )

        # Should handle nonlinear relationship
        assert result.notna().any()

        # Residuals should be smaller than original values (on average)
        # since we're removing the systematic nonlinear component
        assert result.abs().mean() < values_df["value"].abs().mean()
