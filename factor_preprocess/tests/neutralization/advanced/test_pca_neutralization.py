"""
Test suite for PCA neutralization.
"""
import pytest
import pandas as pd
import numpy as np
from factor_preprocess.neutralization.advanced.pca_neutralization import (
    pca_neutralize,
)


class TestPcaNeutralize:
    """Test PCA neutralization."""

    def test_basic_pca_neutralization(self):
        """Test basic PCA neutralization with fixed components."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=3)
        assets = ["A", "B", "C", "D", "E", "F", "G", "H"]

        values_list = []
        exposures_list = []

        for date in dates:
            for asset in assets:
                values_list.append({
                    "date": date,
                    "asset_id": asset,
                    "value": np.random.randn(),
                })
                # Create correlated exposures
                base = np.random.randn()
                exposures_list.append({
                    "date": date,
                    "asset_id": asset,
                    "size": base + np.random.randn() * 0.1,
                    "value": base * 0.8 + np.random.randn() * 0.1,
                    "momentum": np.random.randn(),
                })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        # Neutralize against top 2 PCs
        result = pca_neutralize(
            values_df, exposures_df, n_components=2, min_observations=5
        )

        # Result should have same length as input
        assert len(result) == len(values_df)
        assert result.notna().any()

    def test_variance_threshold(self):
        """Test PCA with variance threshold instead of fixed components."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

        values_list = []
        exposures_list = []

        for asset in assets:
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": np.random.randn(),
            })
            # Create exposures with strong first PC
            base = np.random.randn()
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exp1": base + np.random.randn() * 0.1,
                "exp2": base * 0.9 + np.random.randn() * 0.1,
                "exp3": base * 0.8 + np.random.randn() * 0.1,
                "exp4": np.random.randn() * 0.1,  # Weak component
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        # Use variance threshold
        result = pca_neutralize(
            values_df,
            exposures_df,
            variance_threshold=0.9,
            min_observations=5,
        )

        assert result.notna().any()
        assert len(result) == len(values_df)

    def test_per_date_independence(self):
        """Test that PCA neutralization is per-date."""
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
                    "exp1": float(j),
                    "exp2": float(j * 2),
                })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = pca_neutralize(
            values_df, exposures_df, n_components=1, min_observations=3
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
                "exp1": 1.0,
                "exp2": 2.0,
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result = pca_neutralize(
            values_df,
            exposures_df,
            n_components=1,
            min_observations=10,  # More than available
        )

        # All should be NaN
        assert result.isna().all()

    def test_nan_handling(self):
        """Test NaN handling in PCA."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = ["A", "B", "C", "D", "E", "F"]

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            value = np.random.randn() if i < 4 else np.nan
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

        result = pca_neutralize(
            values_df, exposures_df, n_components=1, min_observations=3
        )

        # NaN inputs should produce NaN residuals
        assert result.isna().iloc[-2:].all()

        # Valid inputs should produce finite residuals
        assert result.notna().iloc[:4].all()

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
            pca_neutralize(values_df, exposures_df)

    def test_center_and_scale_options(self):
        """Test centering and scaling options."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = ["A", "B", "C", "D", "E", "F"]

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
                "exp1": float(i) * 10,  # Different scales
                "exp2": float(i) * 0.1,
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        # Test all combinations
        result_cs = pca_neutralize(
            values_df, exposures_df, n_components=1, center=True, scale=True, min_observations=3
        )
        result_c = pca_neutralize(
            values_df, exposures_df, n_components=1, center=True, scale=False, min_observations=3
        )
        result_s = pca_neutralize(
            values_df, exposures_df, n_components=1, center=False, scale=True, min_observations=3
        )
        result_none = pca_neutralize(
            values_df, exposures_df, n_components=1, center=False, scale=False, min_observations=3
        )

        # All should produce results
        assert result_cs.notna().any()
        assert result_c.notna().any()
        assert result_s.notna().any()
        assert result_none.notna().any()

    def test_intercept_handling(self):
        """Test with and without intercept."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = ["A", "B", "C", "D", "E"]

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": float(i) + 10.0,  # Non-zero mean
            })
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exposure": float(i),
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        result_with = pca_neutralize(
            values_df, exposures_df, n_components=1, add_intercept=True, min_observations=3
        )
        result_without = pca_neutralize(
            values_df, exposures_df, n_components=1, add_intercept=False, min_observations=3
        )

        # Both should produce results
        assert result_with.notna().any()
        assert result_without.notna().any()

    def test_collinear_exposures(self):
        """Test PCA handles highly collinear exposures."""
        np.random.seed(42)
        dates = [pd.Timestamp("2020-01-01")]
        assets = [f"A{i}" for i in range(10)]

        values_list = []
        exposures_list = []

        for i, asset in enumerate(assets):
            base = float(i)
            values_list.append({
                "date": dates[0],
                "asset_id": asset,
                "value": np.random.randn(),
            })
            # Highly collinear exposures
            exposures_list.append({
                "date": dates[0],
                "asset_id": asset,
                "exp1": base,
                "exp2": base * 1.01,  # Almost identical
                "exp3": base * 0.99,
                "exp4": base * 1.005,
            })

        values_df = pd.DataFrame(values_list)
        exposures_df = pd.DataFrame(exposures_list)

        # PCA should handle collinearity gracefully
        result = pca_neutralize(
            values_df, exposures_df, n_components=2, min_observations=5
        )

        assert result.notna().any()
        assert len(result) == len(values_df)
